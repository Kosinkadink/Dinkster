"""Document, OTIO, wire, edit equivalence, and bounded CPU timeline contracts."""

from __future__ import annotations

import asyncio
import io
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np
import opentimelineio as otio
import pytest
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_nodes_image.drawing import DrawText
from dinkster_nodes_media_io import register_media_types
from dinkster_nodes_media_io.video_document import VIDEO_DOCUMENT_NODES
from dinkster_nodes_media_io.video_ops import CropVideo, TrimVideo
from dinkster_values import (
    TypeRegistry,
    annotate_image,
    decode_video,
    edit_video,
    encode_video,
    video_from_source,
)
from dinkster_values.timeline_video import TIMELINE_MAGIC, TimelineVideo
from dinkster_values.video_codec import validate_video_encoded, video_fingerprint, video_meta
from dinkster_values.video_document import (
    DOCUMENT_TYPE,
    TimelineError,
    decode_document,
    document,
    effective_document,
    encode_document,
    extension,
    item_duration,
    json_bytes,
    range_seconds,
    set_extension,
    time_range,
    video_reference,
)
from dinkster_video import assemble_video, disassemble_video, save_video_stream
from dinkster_video.document import COMMAND_NODES, clip, export_otio, import_otio, make, mutate
from dinkster_video.image_math import masked_transition
from dinkster_video.timeline import (
    compile_timeline,
    iter_timeline_audio,
    iter_timeline_frames,
    single_clip_video,
)
from dinkster_video.timeline_runtime import CPUKernels, SourceMedia

from tests.test_video_runtime import _packets, _source

FIXTURES = Path(__file__).parent / "fixtures" / "otio"


@pytest.fixture
def media_assets(tmp_path):
    vault = AssetVault(tmp_path / "vault")

    def publish(data, name="source.mp4"):
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        return AssetRef(digest, name, len(data), resolver=vault)

    return publish, lambda wire: AssetRef.from_wire(wire, vault)


@pytest.fixture
def bound_video(media_assets):
    publish, factory = media_assets
    data = _source()
    return video_from_source(publish(data)), factory, data


def timeline_for(value, **kwargs):
    result = make(width=64, height=32, rate=10, clips=[clip("video", duration=2, **kwargs)])
    return mutate(result, "bind_source", {"source": "video", "reference": video_reference(value)})


@pytest.mark.parametrize("name", ["effects", "clip_example", "nested_example", "multiple_track"])
def test_real_otio_structural_equivalence(name):
    text = (FIXTURES / f"{name}.otio").read_text()
    original = otio.adapters.read_from_string(text, "otio_json")
    imported = import_otio(text)
    assert imported["sources"] == {}
    exported = export_otio(decode_document(encode_document(imported)))
    actual = otio.adapters.read_from_string(exported, "otio_json")
    assert original.is_equivalent_to(actual)


def test_document_rejects_inline_media_paths_and_nonfinite():
    base = make()
    for reference in (
        {"type": "comfy.VIDEO", "bytes": "pixels"},
        {"type": "comfy.VIDEO", "asset": {"path": "/etc/passwd"}},
    ):
        with pytest.raises(ValueError):
            mutate(base, "bind_source", {"source": "a", "reference": reference})
    with pytest.raises(ValueError):
        decode_document(b'{"version":1,"version":1}')
    with pytest.raises(ValueError):
        document({**base, "settings": {"width": 64, "height": 32, "rate": float("nan")}})
    with pytest.raises(ValueError, match="1 MiB"):
        document({**base, "timeline": {**base["timeline"], "metadata": {"huge": "x" * 1048576}}})
    with pytest.raises(TimelineError, match="Timeline.1 root"):
        document({**base, "timeline": base["timeline"]["tracks"]})


def test_opaque_otio_url_never_reaches_resolver():
    tree = make(clips=[clip("absent", duration=2)])
    child = tree["timeline"]["tracks"]["children"][0]["children"][0]
    child["media_references"]["DEFAULT_MEDIA"] = {
        "OTIO_SCHEMA": "ExternalReference.1",
        "target_url": "file:///etc/passwd",
        "metadata": {},
        "available_range": None,
        "available_image_bounds": None,
        "name": "",
    }
    imported = import_otio(export_otio(tree))
    with pytest.raises(TimelineError, match="unbound_source"):
        compile_timeline(imported)
    assert "file:///etc/passwd" in export_otio(imported)


@pytest.mark.parametrize(
    "widget",
    [
        {},
        {"trim": None, "crop": None, "foreign": "preserved"},
        {"trim": {"start_time": 0.5, "duration": 0}},
        {"crop": {"x": -7, "y": 3, "width": 31, "height": 23}},
        {
            "trim": {"start_time": -0.5, "duration": 99},
            "crop": {"width": 0, "height": -1},
            "strict_duration": "ignored",
            "foreign": [1, 2],
        },
    ],
)
def test_widget_one_authority_and_verbatim_roundtrip(bound_video, widget):
    value, factory, _ = bound_video
    base = edit_video(value, {"trim": {"start_time": 0.5, "duration": 1.5}})
    doc = timeline_for(base, video_edit=widget)
    child = doc["timeline"]["tracks"]["children"][0]["children"][0]
    assert extension(child)["video_edit"] == widget
    child["source_range"]["start_time"]["value"] = 999
    normalized = document(doc)
    actual = single_clip_video(normalized, SourceMedia(factory))
    expected = CropVideo.execute(
        video=TrimVideo.execute(video=base, video_edit=widget)["video"], video_edit=widget
    )["video"]
    assert encode_video(actual) == encode_video(expected)
    roundtrip = import_otio(export_otio(normalized))
    assert (
        extension(roundtrip["timeline"]["tracks"]["children"][0]["children"][0])["video_edit"]
        == widget
    )
    assert encode_document(roundtrip) == encode_document(effective_document(normalized))


def test_strict_widget_sibling_and_safe_packet_copy(bound_video):
    value, factory, source = bound_video
    invalid = timeline_for(
        value, video_edit={"trim": {"start_time": 1.5, "duration": 2}}, strict_duration=True
    )
    assert decode_document(encode_document(invalid)) == invalid
    with pytest.raises(ValueError, match="exceeds"):
        single_clip_video(invalid, SourceMedia(factory))
    doc = timeline_for(value, video_edit={"trim": {"start_time": 0.5, "duration": 0.5}})
    video = single_clip_video(doc, SourceMedia(factory))
    assert video is not None and "source" in video
    output = io.BytesIO()
    save_video_stream(video, output)
    assert _packets(output.getvalue()) == _packets(source)[5:10]


def test_v3_wire_is_source_only_and_v2_stable(bound_video):
    value, factory, _ = bound_video
    before, identity = encode_video(value), video_fingerprint("comfy.VIDEO")(value)
    doc = timeline_for(value)
    doc = mutate(doc, "add_track", {"name": "overlay"})
    video = TimelineVideo(doc, factory)
    encoded = encode_video(video)
    assert encoded.startswith(TIMELINE_MAGIC) and len(encoded) < 1048576
    assert b"pixels" not in encoded and b"waveform" not in encoded
    decoded = decode_video(encoded)
    assert encode_video(decoded) == encoded
    assert isinstance(decoded, TimelineVideo)
    assert decoded.factory is None
    validate_video_encoded(encoded, video_meta(video))
    bad_meta = dict(video_meta(video), asset_refs=[])
    with pytest.raises(ValueError, match="asset_refs"):
        validate_video_encoded(encoded, bad_meta)
    assert encode_video(value) == before
    assert video_fingerprint("comfy.VIDEO")(value) == identity


def test_node_command_enumeration_and_codec_registration():
    assert {node.define_schema().node_type for node in VIDEO_DOCUMENT_NODES} == set(
        COMMAND_NODES.values()
    )
    registry = TypeRegistry()
    register_media_types(registry)
    assert DOCUMENT_TYPE in registry


def test_split_move_ripple_roll_and_retime():
    doc = make(clips=[clip("a", start=2, duration=10)])
    split = mutate(doc, "split", {"position": 4})
    children = split["timeline"]["tracks"]["children"][0]["children"]
    assert [range_seconds(c["source_range"]) for c in children] == [(2, 4), (6, 6)]
    roll = mutate(split, "roll", {"delta": 1})
    assert item_duration(roll["timeline"]["tracks"]) == 10
    ripple = mutate(roll, "ripple", {"duration": 3})
    assert item_duration(ripple["timeline"]["tracks"]) == 8
    moved = mutate(ripple, "move", {"index": 1})
    assert (
        range_seconds(moved["timeline"]["tracks"]["children"][0]["children"][0]["source_range"])[0]
        == 7
    )
    retimed = mutate(doc, "retime", {"scalar": 2})
    assert item_duration(retimed["timeline"]["tracks"]) == 10
    assert doc == make(clips=[clip("a", start=2, duration=10)])


def test_crossfade_title_render_uses_shared_kernels(bound_video):
    value, factory, _ = bound_video
    doc = timeline_for(value)
    track = doc["timeline"]["tracks"]["children"][0]
    track["children"] = [
        clip("video", start=0.5, duration=0.5),
        clip("video", start=1, duration=0.5),
    ]
    doc = mutate(doc, "transition", {"in_offset": 0.2, "out_offset": 0.2})
    doc = mutate(
        doc,
        "set_effect",
        {
            "effect": {
                "node_type": "dinkster.image.draw_text",
                "parameters": {
                    "text": "test",
                    "font_size": 10,
                    "opacity": {
                        "type": "dinkster.curve",
                        "value": {
                            "points": [{"position": 0, "value": 0}, {"position": 1, "value": 1}]
                        },
                    },
                },
            }
        },
    )
    frames = list(iter_timeline_frames(doc, SourceMedia(factory), CPUKernels()))
    assert len(frames) == 10
    assert [t for t, _ in frames] == [Fraction(i, 10) for i in range(10)]
    out = io.BytesIO()
    save_video_stream(TimelineVideo(doc, factory), out)
    assert len(_packets(out.getvalue())) == 10
    frame = np.zeros((32, 64, 3), np.float32)
    expected = cast(
        np.ndarray,
        DrawText.execute(image=frame[None], text="test", font_size=10, opacity=0.5)["image"],
    )[0]
    actual = CPUKernels().effect(
        "dinkster.image.draw_text", frame, {"text": "test", "font_size": 10, "opacity": 0.5}
    )
    np.testing.assert_array_equal(actual, expected)


def test_widget_projection_source_origin_crop_and_split(bound_video):
    value, factory, _ = bound_video
    widget = {
        "trim": {"start_time": 0.5, "duration": 0},
        "crop": {"x": 1, "y": 3, "width": 32, "height": 16},
        "unknown": "kept",
    }
    doc = timeline_for(value, video_edit=widget)
    child = doc["timeline"]["tracks"]["children"][0]["children"][0]
    child["media_references"]["DEFAULT_MEDIA"]["available_range"] = time_range(86400, 2)
    selected = single_clip_video(doc, SourceMedia(factory))
    projected = effective_document(doc)
    assert range_seconds(
        projected["timeline"]["tracks"]["children"][0]["children"][0]["source_range"]
    ) == (Fraction(172801, 2), Fraction(3, 2))
    doc = mutate(doc, "add_track", {})
    doc["settings"].update(width=32, height=16)
    frames = [frame for _, frame in iter_timeline_frames(doc, SourceMedia(factory), CPUKernels())]
    np.testing.assert_array_equal(np.stack(frames), disassemble_video(selected)["images"])
    split = mutate(doc, "split", {"position": 0.5})
    children = split["timeline"]["tracks"]["children"][0]["children"]
    assert [extension(c)["video_edit"]["trim"]["duration"] for c in children] == [0.5, 1.0]
    assert all(extension(c)["video_edit"]["unknown"] == "kept" for c in children)
    assert extension(child)["video_edit"] == widget


def test_imported_source_range_converts_once_to_widget(bound_video):
    value, factory, _ = bound_video
    doc = timeline_for(value)
    child = doc["timeline"]["tracks"]["children"][0]["children"][0]
    child["media_references"]["DEFAULT_MEDIA"]["available_range"] = time_range(100, 2)
    child["source_range"] = time_range(100.5, 1)
    expected = single_clip_video(doc, SourceMedia(factory))
    edited = mutate(doc, "trim", {"video_edit": {"foreign": 5}})
    actual = single_clip_video(edited, SourceMedia(factory))
    np.testing.assert_array_equal(
        disassemble_video(actual)["images"], disassemble_video(expected)["images"]
    )
    assert encode_video(actual) == encode_video(
        edit_video(value, {"trim": {"start_time": 0.5, "duration": 1}, "strict_duration": False})
    )
    ext = extension(edited["timeline"]["tracks"]["children"][0]["children"][0])
    assert ext["video_edit"] == {"trim": {"start_time": 0.5, "duration": 1}, "foreign": 5}


def test_nested_track_opacity_applies_once_to_composited_group(bound_video):
    value, factory, _ = bound_video
    doc = timeline_for(value)
    doc = mutate(doc, "add_track", {"opacity": 0.5})
    doc = mutate(doc, "add_clip", {"track": 1, "item": clip("video", duration=2)})
    nested = doc["timeline"]["tracks"]
    set_extension(nested, opacity=0.5)
    frames = list(iter_timeline_frames(doc, SourceMedia(factory), CPUKernels()))
    baseline = list(SourceMedia(factory).frames(doc["sources"]["video"], Fraction(0), Fraction(2)))
    for (_, actual), (_, expected) in zip(frames, baseline, strict=True):
        np.testing.assert_array_equal(actual, expected * 0.5)


def test_layer_source_uses_declared_resources_and_shared_renderer(media_assets):
    from PIL import Image

    from tests.test_image_document_rendering import (
        _canonical,
        _document,
        _layer,
        _png,
        _render,
        _resource,
    )

    publish, factory = media_assets
    png = _png([[(255, 32, 64, 255), (16, 64, 128, 128)]])
    raster = publish(png, "raster.png")
    layer = _document(
        2, 1, {"l1": _layer("l1", "r2", 2, 1)}, {"r2": _resource("r2", png, 2, 1)}, ["l1"]
    )
    asset = publish(_canonical(layer), "layers.json")
    reference = {
        "type": "dinkster.layers",
        "asset": asset.to_wire(),
        "resources": [raster.to_wire()],
    }
    doc = make(width=2, height=1, clips=[clip("layer", duration=1)])
    doc = mutate(doc, "bind_source", {"source": "layer", "reference": reference})
    result = next(SourceMedia(factory).frames(reference, Fraction(0), Fraction(1)))[1]
    expected = _render(layer, {raster.digest: png})
    with Image.open(io.BytesIO(expected.png)) as image:
        np.testing.assert_array_equal(result, np.asarray(image, dtype=np.float32) / 255)
    assert len(cast(list, video_meta(TimelineVideo(doc))["asset_refs"])) == 2
    with pytest.raises(TimelineError, match="explicitly declared"):
        next(SourceMedia(factory).frames({**reference, "resources": []}, Fraction(0), Fraction(1)))


def test_audio_windows_gain_crossfade_and_silence(bound_video):
    value, factory, _ = bound_video
    doc = timeline_for(value)
    doc = mutate(doc, "add_track", {"kind": "Audio"})
    for _ in range(2):
        doc = mutate(doc, "add_clip", {"track": 1, "item": clip("video", start=0.5, duration=0.5)})
    doc = mutate(doc, "transition", {"track": 1, "in_offset": 0.1, "out_offset": 0.1})
    doc = mutate(doc, "mix_audio", {"track": 1, "audio_mix": {"gain": 0.5}})

    class Windows(SourceMedia):
        def audio_window(self, reference, start, count, rate):
            assert 0 <= start < 2 and count <= 7
            return np.ones((2, count), np.float32)

    windows = list(iter_timeline_audio(doc, Windows(factory), CPUKernels(), rate=100, window=7))
    assert len(windows) == 29
    samples = np.concatenate([w for _, w in windows], axis=1)
    assert samples.shape == (2, 200)
    np.testing.assert_array_equal(samples[:, :100], np.full((2, 100), 0.5, np.float32))
    np.testing.assert_array_equal(samples[:, 100:], np.zeros((2, 100), np.float32))


def test_document_and_v3_render_match_in_process_and_shared_memory(bound_video):
    from dinkster_nodes_media_io import MEDIA_IO_NODES
    from dinkster_protocol import Invocation
    from dinkster_schema import build_node_types
    from dinkster_values import register_core_types
    from dinkster_workers import InProcessWorker, IsolatedWorker

    value, factory, _ = bound_video
    doc = mutate(timeline_for(value), "add_track", {})

    async def scenario():
        registry = TypeRegistry()
        register_core_types(registry)
        register_media_types(registry)

        async def run(worker):
            result = await worker.invoke(
                Invocation(
                    invocation_id="render",
                    node_id="render",
                    node_type="dinkster.video_document.render",
                    inputs={"document": registry.wrap(DOCUMENT_TYPE, doc)},
                    effective_schema=worker.schemas["dinkster.video_document.render"],
                )
            )
            assert result.error is None and result.outputs is not None
            return encode_video(result.outputs["video"].resolve())

        expected = await run(InProcessWorker(build_node_types(MEDIA_IO_NODES), registry))
        diagnostics = []
        worker = IsolatedWorker(
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-media-io/dinkster-pack.toml",
            registry,
            shm_threshold=64,
            on_diagnostic=diagnostics.append,
        )
        await worker.start()
        try:
            actual = await run(worker)
            assert actual == expected and actual.startswith(TIMELINE_MAGIC)
            assert len(actual) < 10000
            assert any(c.transport == "shm" for d in diagnostics for c in d.inputs)
            output = io.BytesIO()
            from dinkster_values import bind_video_sources

            save_video_stream(bind_video_sources(decode_video(actual), factory), output)
            assert len(_packets(output.getvalue())) == 20
        finally:
            await worker.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "scalar,start,length,indices",
    [
        (2, 0, 1, list(range(0, 20, 2))),
        (0, 1, 0.5, [10] * 5),
        (-1, 1.9, 2, list(reversed(range(20)))),
    ],
)
def test_time_effects_sample_without_changing_duration(bound_video, scalar, start, length, indices):
    value, factory, _ = bound_video
    doc = timeline_for(value)
    doc["timeline"]["tracks"]["children"][0]["children"] = [
        clip("video", start=start, duration=length)
    ]
    doc = mutate(doc, "retime", {"scalar": scalar})
    obj, _, duration = compile_timeline(doc)
    assert duration == Fraction(str(length))
    frames = [frame for _, frame in iter_timeline_frames(obj, SourceMedia(factory), CPUKernels())]
    original = cast(np.ndarray, disassemble_video(value)["images"])
    np.testing.assert_array_equal(np.stack(frames), original[indices])


def test_cancel_closes_all_active_media_readers(bound_video):
    value, factory, _ = bound_video
    doc = timeline_for(value)
    closed = []

    class Tracked(SourceMedia):
        def frames(self, reference, start, end, *, crop=None):
            try:
                yield from super().frames(reference, start, end, crop=crop)
            finally:
                closed.append(reference)

    iterator = iter_timeline_frames(doc, Tracked(factory), CPUKernels())
    next(iterator)
    assert closed == []
    iterator.close()
    assert len(closed) == 1


def test_disabled_clip_and_unsupported_composition_effect(bound_video):
    value, factory, _ = bound_video
    doc = timeline_for(value)
    child = doc["timeline"]["tracks"]["children"][0]["children"][0]
    child["enabled"] = False
    media = SourceMedia(factory)
    assert single_clip_video(doc, media) is None
    frames = [frame for _, frame in iter_timeline_frames(doc, media, CPUKernels())]
    assert len(frames) == 20
    assert not np.any(frames)
    set_extension(doc["timeline"]["tracks"], effects=[{"node_type": "unknown", "parameters": {}}])
    with pytest.raises(TimelineError, match="unsupported_effect"):
        compile_timeline(doc)


def test_loopback_remote_sink_stages_sources_and_returns_only_encoded_asset(
    bound_video, tmp_path, monkeypatch
):
    import av
    from aiohttp import web
    from dinkster_assets import SAVE_TARGET_TYPE
    from dinkster_protocol import Invocation
    from dinkster_values import register_core_types
    from dinkster_workers import RemoteWorker
    from test_remote import TOKEN, start_service, stop_service
    from test_remote_asset_staging import ENDPOINT_TOKEN, AssetHost
    from test_video_io import _mount

    value, _, data = bound_video
    doc = mutate(timeline_for(value), "add_track", {})
    lazy = TimelineVideo(doc)
    assert len(encode_video(lazy)) < 10000
    digest = digest_bytes(data)
    output_root, _ = _mount(tmp_path, monkeypatch)

    class VideoHost(AssetHost):
        async def _handle(self, request: web.Request) -> web.StreamResponse:
            self.requests += 1
            assert request.headers.get("Authorization") == f"Bearer {ENDPOINT_TOKEN}"
            assert request.match_info["digest"] == digest
            return web.Response(body=data)

    async def scenario():
        registry = TypeRegistry()
        register_core_types(registry)
        register_media_types(registry)
        vault = tmp_path / "daemon-vault"
        manifest = (
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-media-io/dinkster-pack.toml"
        )
        proc, host, port = await start_service(manifest, tmp_path, "--asset-vault", str(vault))
        try:
            async with VideoHost() as endpoint:
                worker = RemoteWorker(
                    host,
                    port,
                    TOKEN,
                    registry,
                    name="timeline-sink",
                    connect_timeout=10.0,
                    asset_endpoint=endpoint.endpoint,
                    asset_endpoint_token=ENDPOINT_TOKEN,
                )
                await worker.start()
                try:
                    with monkeypatch.context() as producer:
                        producer.setattr(
                            av, "open", lambda *a, **kw: pytest.fail("producer decoded media")
                        )
                        result = await worker.invoke(
                            Invocation(
                                invocation_id="timeline-save",
                                node_id="save",
                                node_type="dinkster.save_video_value",
                                inputs={
                                    "video": registry.wrap("comfy.VIDEO", lazy),
                                    "target": registry.wrap(
                                        SAVE_TARGET_TYPE,
                                        {"mount": "comfy-output", "prefix": "timeline"},
                                    ),
                                },
                                effective_schema=worker.schemas["dinkster.save_video_value"],
                            )
                        )
                    assert result.error is None and result.outputs is not None
                    assert set(result.outputs) == {"video"}
                    ref = result.outputs["video"].resolve()
                    assert isinstance(ref, AssetRef)
                    assert endpoint.requests == 1
                    assert AssetVault(vault).has(digest)
                    outputs = list(output_root.glob("*.mp4"))
                    assert len(outputs) == 1
                    assert ref.digest == digest_bytes(outputs[0].read_bytes())
                    assert len(_packets(outputs[0].read_bytes())) == 20
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_track_blend_keeps_source_color_over_transparent_destination():
    destination = np.zeros((1, 1, 4), np.float32)
    source = np.array([[[0, 0, 1, 0.5]]], np.float32)
    actual = CPUKernels().composite(destination, source, "multiply", 1)
    np.testing.assert_array_equal(actual, source)
    destination[0, 0] = [1, 0, 0, 0.5]
    actual = CPUKernels().composite(destination, source, "multiply", 1)
    expected = np.array([[[1 / 3, 0, 1 / 3, 0.75]]], np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_alpha_dissolve_interpolates_associated_color_before_composition():
    first = np.array([[[1, 0, 0, 1]]], np.float32)
    second = np.array([[[0, 0, 1, 0]]], np.float32)
    expected = np.array([[[1, 0, 0, 0.5]]], np.float32)
    np.testing.assert_array_equal(
        masked_transition(first, second, np.array([[0.5]], np.float32)), expected
    )
    transition = CPUKernels().transition(first, second, 0.5)
    np.testing.assert_array_equal(transition, expected)
    flattened = CPUKernels().composite(np.zeros((1, 1, 3), np.float32), transition, "normal", 1)
    np.testing.assert_array_equal(flattened, np.array([[[0.5, 0, 0]]], np.float32))


def test_timeline_compositor_converts_premultiplied_video_pixels_once(bound_video):
    source, _, _ = bound_video
    pixels = np.zeros((20, 32, 64, 4), np.float32)
    pixels[..., 0] = pixels[..., 3] = 0.5
    video = assemble_video(annotate_image(pixels, alpha="premultiplied"), fps=10)
    reference = {
        "type": "comfy.VIDEO",
        "asset": cast(AssetRef, source["source"]).to_wire(),
    }
    doc = make(width=64, height=32, rate=10, clips=[clip("video", duration=2)])
    doc = mutate(doc, "bind_source", {"source": "video", "reference": reference})
    media = SourceMedia(None)
    media.values[json_bytes(reference).decode()] = video
    frame = next(iter_timeline_frames(doc, media, CPUKernels()))[1]
    np.testing.assert_array_equal(frame, np.full((32, 64, 3), [0.5, 0, 0], np.float32))


def test_split_preserves_negative_otio_source_coordinate_origin():
    item = clip("source", start=-98, duration=5)
    item["media_references"]["DEFAULT_MEDIA"]["available_range"] = time_range(-100, 20)
    split = mutate(make(clips=[item]), "split", {"position": 2})
    children = split["timeline"]["tracks"]["children"][0]["children"]
    assert [range_seconds(child["source_range"]) for child in children] == [(-98, 2), (-96, 3)]


def test_asset_only_video_binding_cannot_hold_last_frame_past_eof(bound_video):
    value, factory, _ = bound_video
    reference = {"type": "comfy.VIDEO", "asset": cast(AssetRef, value["source"]).to_wire()}
    obj = make(width=64, height=32, rate=10, clips=[clip("video", duration=3)])
    obj = mutate(obj, "bind_source", {"source": "video", "reference": reference})
    with pytest.raises(TimelineError, match="source_range_unavailable"):
        list(iter_timeline_frames(obj, SourceMedia(factory), CPUKernels()))
