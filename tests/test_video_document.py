"""Document, OTIO, wire, edit equivalence, and bounded CPU timeline contracts."""

from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np
import opentimelineio as otio
import pytest
from dinkster_api.v1 import VIDEO_TYPE, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_values import (
    TypeRegistry,
    annotate_image,
    decode_video,
    encode_video,
    video_from_source,
)
from dinkster_values.timeline_video import TIMELINE_MAGIC, TimelineVideo
from dinkster_values.video_document import (
    DOCUMENT_TYPE,
    TimelineError,
    decode_document,
    document,
    document_fingerprint,
    document_meta,
    encode_document,
    item_duration,
    json_bytes,
    range_seconds,
    set_extension,
    video_reference,
)
from dinkster_video_runtime import assemble_video, save_video_stream
from dinkster_video_runtime.document import clip, export_otio, import_otio, make, mutate
from dinkster_video_runtime.image_math import masked_transition
from dinkster_video_runtime.timeline import (
    compile_timeline,
    iter_timeline_audio,
    iter_timeline_frames,
)
from dinkster_video_runtime.timeline_runtime import CPUKernels, SourceMedia

from tests.async_scenarios import run_scenario
from tests.test_video_diagnostics import FormatSave
from tests.test_video_diagnostics import register_types as register_format_types
from tests.test_video_runtime import _packets, _source

FIXTURES = Path(__file__).parent / "fixtures" / "otio"


class DocumentVideo(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="test.document-video",
            inputs=(InputSpec("document", TypeExpr.concrete(DOCUMENT_TYPE)),),
            outputs=(OutputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),),
        )

    @classmethod
    def execute(cls, document):
        return cls.outputs(video=TimelineVideo(document))


DOCUMENT_NODES = (DocumentVideo, FormatSave)


def register_document_types(registry):
    register_format_types(registry)
    registry.register(
        DOCUMENT_TYPE,
        encode=encode_document,
        decode=decode_document,
        coerce=document,
        fingerprint=document_fingerprint,
        meta=document_meta,
    )


@pytest.fixture
def document_manifest(tmp_path):
    manifest = tmp_path / "document-pack.toml"
    manifest.write_text(
        '[pack]\nname="video-document-fixture"\npublisher="dinkster"\n'
        '[pack.entry]\nnodes="tests.test_video_document:DOCUMENT_NODES"\n'
        'types="tests.test_video_document:register_document_types"\n'
    )
    return manifest


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


def test_crossfade_title_render_preserves_timing_and_frame_count(bound_video):
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
                        "type": "CURVE",
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


def test_document_and_v3_render_match_in_process_and_shared_memory(bound_video, document_manifest):
    from dinkster_protocol import Invocation
    from dinkster_schema import build_node_types
    from dinkster_values import register_core_types
    from dinkster_workers import InProcessWorker, IsolatedWorker

    value, factory, _ = bound_video
    doc = mutate(timeline_for(value), "add_track", {})

    async def scenario():
        registry = TypeRegistry()
        register_core_types(registry)
        register_document_types(registry)

        async def run(worker):
            result = await worker.invoke(
                Invocation(
                    invocation_id="render",
                    node_id="render",
                    node_type="test.document-video",
                    inputs={"document": registry.wrap(DOCUMENT_TYPE, doc)},
                    effective_schema=worker.schemas["test.document-video"],
                )
            )
            assert result.error is None and result.outputs is not None
            return encode_video(result.outputs["video"].resolve())

        expected = await run(InProcessWorker(build_node_types(DOCUMENT_NODES), registry))
        diagnostics = []
        worker = IsolatedWorker(
            document_manifest,
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

    run_scenario(scenario())


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


def test_loopback_remote_sink_stages_sources_and_returns_only_encoded_asset(
    bound_video, tmp_path, monkeypatch, document_manifest
):
    import av
    from aiohttp import web
    from dinkster_protocol import Invocation
    from dinkster_values import register_core_types
    from dinkster_workers import RemoteWorker
    from test_remote import TOKEN, start_service, stop_service
    from test_remote_asset_staging import ENDPOINT_TOKEN, AssetHost
    from test_video_diagnostics import _mount

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
        register_document_types(registry)
        vault = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            document_manifest, tmp_path, "--asset-vault", str(vault)
        )
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
                                node_type="test.format-save",
                                inputs={
                                    "video": registry.wrap(VIDEO_TYPE, lazy),
                                },
                                effective_schema=worker.schemas["test.format-save"],
                            )
                        )
                    assert result.error is None and result.outputs is not None
                    assert set(result.outputs) == {"asset"}
                    ref = result.outputs["asset"].resolve()
                    assert isinstance(ref, AssetRef)
                    assert endpoint.requests == 1
                    assert AssetVault(vault).has(digest)
                    outputs = list(output_root.glob("video/*.mp4"))
                    assert len(outputs) == 1
                    assert ref.digest == digest_bytes(outputs[0].read_bytes())
                    assert len(_packets(outputs[0].read_bytes())) == 20
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    run_scenario(scenario())


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
        "type": VIDEO_TYPE,
        "asset": cast(AssetRef, source["source"]).to_wire(),
    }
    doc = make(width=64, height=32, rate=10, clips=[clip("video", duration=2)])
    doc = mutate(doc, "bind_source", {"source": "video", "reference": reference})
    media = SourceMedia(None)
    media.values[json_bytes(reference).decode()] = video
    frame = next(iter_timeline_frames(doc, media, CPUKernels()))[1]
    np.testing.assert_array_equal(frame, np.full((32, 64, 3), [0.5, 0, 0], np.float32))


def test_asset_only_video_binding_cannot_hold_last_frame_past_eof(bound_video):
    value, factory, _ = bound_video
    reference = {"type": VIDEO_TYPE, "asset": cast(AssetRef, value["source"]).to_wire()}
    obj = make(width=64, height=32, rate=10, clips=[clip("video", duration=3)])
    obj = mutate(obj, "bind_source", {"source": "video", "reference": reference})
    with pytest.raises(TimelineError, match="source_range_unavailable"):
        list(iter_timeline_frames(obj, SourceMedia(factory), CPUKernels()))
