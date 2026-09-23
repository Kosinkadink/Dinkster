"""Native asset-backed video I/O contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import av
import dinkster_nodes_media_io.video as video_module
import numpy as np
import pytest
from av.codec import Codec as AvCodec
from av.error import InvalidDataError
from dinkster_api.v1 import image_input
from dinkster_assets import AssetRef, AssetVault, MountSnapshotResolver, digest_bytes
from dinkster_nodes_media_io.video import LoadVideo, LoadVideoValue, SaveVideo, SaveVideoValue
from dinkster_schema import (
    schema_to_wire,
    validate_replacement_references,
)
from dinkster_video import assemble_video


def _frames(count: int = 4, *, alpha: bool = False) -> np.ndarray:
    channels = 4 if alpha else 3
    result = np.zeros((count, 64, 64, channels), dtype=np.float32)
    for index in range(count):
        result[index, ..., 0] = index / max(1, count - 1)
        result[index, ..., 1] = 0.25
        result[index, ..., 2] = 0.75
        if alpha:
            result[index, ..., 3] = (index + 1) / count
    return result


def _audio(samples: int = 16_000, sample_rate: int = 16_000) -> dict[str, object]:
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    waveform = np.sin(2 * np.pi * 440 * timeline, dtype=np.float32)[None, None, :]
    return {"waveform": waveform, "sample_rate": sample_rate}


def _mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "out"
    root.mkdir()
    index = root / ".dinkster-asset-index.json"
    index.write_text("{}", "utf-8")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "comfy-output",
                        "root": str(root),
                        "index": str(index),
                        "mode": "readwrite",
                    }
                ]
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root, snapshot


def _bound(ref: AssetRef, snapshot: Path) -> AssetRef:
    return AssetRef(
        ref.digest,
        ref.name,
        ref.size,
        ref.media_type,
        ref.virtual_path,
        MountSnapshotResolver(snapshot),
    )


def _save_frames(
    *,
    images: object,
    fps: float = 24,
    format: str = "mp4_h264",
    crf: int = 23,
    bit_depth: str = "8",
    audio: object = None,
    target: object = None,
) -> dict[str, object]:
    video = assemble_video(images, fps=fps, bit_depth=bit_depth, audio=audio)
    result = SaveVideo.execute(video=video, format=format, crf=crf, target=target)
    assert result["video"] is video
    return {"video": result["asset"]}


@pytest.mark.parametrize(
    ("format_name", "container_name", "codec_name", "media_type"),
    [
        ("mp4_h264", "mov,mp4,m4a,3gp,3g2,mj2", "h264", "video/mp4"),
        ("webm_vp9", "matroska,webm", "vp9", "video/webm"),
        ("webm_av1", "matroska,webm", "av1", "video/webm"),
    ],
)
def test_save_advertised_8_bit_formats_with_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    container_name: str,
    codec_name: str,
    media_type: str,
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    result = _save_frames(
        images=_frames(),
        target={"mount": "comfy-output", "prefix": f"video/{format_name}"},
        fps=4.0,
        format=format_name,
        crf=28,
        bit_depth="8",
        audio=_audio(samples=44_100, sample_rate=44_100),
    )
    ref = cast(AssetRef, result["video"])
    assert ref.media_type == media_type
    assert ref.digest == digest_bytes((root / "video" / ref.name).read_bytes())
    with av.open(root / "video" / ref.name) as container:
        assert container.format.name == container_name
        assert container.streams.video[0].codec_context.codec.canonical_name == codec_name
        assert container.streams.audio[0].codec_context.name == (
            "aac" if format_name == "mp4_h264" else "opus"
        )
        assert sum(1 for _ in container.decode(video=0)) == 4
    decoded_audio: list[av.AudioFrame] = []
    with av.open(root / "video" / ref.name) as container:
        decoded_audio = list(container.decode(audio=0))
    assert decoded_audio
    expected_rate = 44_100 if format_name == "mp4_h264" else 48_000
    assert decoded_audio[0].sample_rate == expected_rate
    assert sum(frame.samples for frame in decoded_audio) == pytest.approx(expected_rate, abs=1024)


def test_save_video_schema_migrates_frame_inputs_and_asset_output_links() -> None:
    schema = SaveVideo.schema()
    assert schema.version == 4
    assert [spec.id for spec in schema.inputs] == [
        "video",
        "target",
        "container",
        "codec",
        "profile",
        "audio_layout",
        "trim_to_audio",
        "crf",
        "metadata",
        "format",
    ]
    assert schema.inputs[0].type.types == ("comfy.VIDEO",)
    assert [output.id for output in schema.outputs] == ["video", "asset"]
    wire = schema_to_wire(schema)
    interface = {
        spec["id"]: spec
        for spec in cast(list[dict[str, Any]], wire["interface"])
        if spec["role"] == "input"
    }
    assert interface["container"]["default"] == "auto"
    assert interface["codec"]["default"] == "auto"
    assert interface["metadata"]["default"] == "{}"
    (migration,) = schema.replacements
    assert migration.from_type == "dinkster.save_video"
    assert migration.migration is not None
    assert migration.migration.historical_inputs == (
        "images",
        "fps",
        "audio",
        "bit_depth",
        "format.crf",
        "format.bit_depth",
    )
    assert migration.cases[-1].unconditional
    for case, prefix in zip(migration.cases, ("format.", ""), strict=True):
        mappings = dict(case.inputs)
        for key in ("images", "fps", "audio"):
            assert mappings["assemble:" + key].input == key
            assert mappings["assemble:" + key].kind == "copy"
        assert mappings["crf"].input == prefix + "crf"
        assert mappings["assemble:bit_depth"].input == prefix + "bit_depth"
        assert mappings["format"].input == "format"
        assert mappings["format"].kind == "copy"
        assert "metadata" not in mappings
        assert case.nodes is not None
        assemble = next(node for node_id, node in case.nodes if node_id == "assemble")
        assert assemble.values == (("color_space", "sRGB"),)
        assert dict(case.outputs) == {"asset": "video"}
        assert [(link.from_address, link.to) for link in case.links] == [
            ("assemble:video", "video")
        ]
    from dinkster_nodes_media_io import AssembleVideo

    assert (
        validate_replacement_references(
            {schema.node_type: schema, "dinkster.video.assemble": AssembleVideo.schema()}
        )
        == ()
    )
    replacement_wire = cast(list[dict[str, Any]], wire["replacements"])[0]
    assert replacement_wire["cases"][0]["when"] == {
        "kind": "any",
        "of": [
            {"kind": "valuePresent", "input": "format.crf"},
            {"kind": "inputConnected", "input": "format.crf"},
            {"kind": "valuePresent", "input": "format.bit_depth"},
            {"kind": "inputConnected", "input": "format.bit_depth"},
        ],
    }
    assert replacement_wire["migration"] == {
        "historicalInputs": list(migration.migration.historical_inputs)
    }


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("format_name", ["mp4_h264", "webm_vp9", "webm_av1"])
def test_migrated_frame_saver_encodes_selected_codec_and_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, format_name: str
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    from dinkster_nodes_media_io import AssembleVideo

    depth = "8" if format_name == "webm_vp9" else "10"
    prefix = "format." if nested else ""
    source = {
        "images": _frames(2),
        "fps": 2.0,
        "format": format_name,
        prefix + "crf": 30,
        prefix + "bit_depth": depth,
    }
    case = SaveVideo.schema().replacements[0].cases[0 if nested else 1]
    mapped = {
        target: source[mapping.input] for target, mapping in case.inputs if mapping.input in source
    }
    assembled = AssembleVideo.execute(
        **{
            target.removeprefix("assemble:"): value
            for target, value in mapped.items()
            if target.startswith("assemble:")
        }
    )["video"]
    result = SaveVideo.execute(
        video=assembled,
        **{target: value for target, value in mapped.items() if not target.startswith("assemble:")},
    )
    assert result["video"] is assembled
    ref = cast(AssetRef, result["asset"])
    with av.open(root / "video" / ref.name) as container:
        assert (
            container.streams.video[0].codec_context.codec.canonical_name
            == format_name.split("_")[1]
        )
        assert next(container.decode(video=0)).format.name == (
            "yuv420p10le" if depth == "10" else "yuv420p"
        )


def test_save_video_passthrough_and_json_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    video = assemble_video(_frames(2), fps=2, bit_depth="10", color_space="HDR PQ")
    workflow = {"nodes": [{"id": "clip"}], "enabled": True}
    result = SaveVideo.execute(
        video=video, codec="h264", metadata=json.dumps({"workflow": workflow})
    )
    assert result["video"] is video
    ref = cast(AssetRef, result["asset"])
    with av.open(root / "video" / ref.name) as container:
        assert json.loads(container.metadata["workflow"]) == workflow
        assert next(container.decode(video=0)).format.name == "yuv420p10le"


def test_video_value_load_and_save_preserve_container_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, snapshot = _mount(tmp_path, monkeypatch)
    saved = _save_frames(
        images=_frames(),
        target={"mount": "comfy-output", "prefix": "video/source"},
        fps=4.0,
        format="mp4_h264",
    )
    source_ref = _bound(cast(AssetRef, saved["video"]), snapshot)
    source_bytes = (root / "video" / source_ref.name).read_bytes()

    value = LoadVideoValue.execute(video=source_ref)["video"]
    assert cast(dict[str, object], value)["source"] is source_ref
    assert cast(dict[str, object], value)["edits"] == []
    copied = SaveVideoValue.execute(
        video=value,
        target={"mount": "comfy-output", "prefix": "video/copied"},
    )
    copied_ref = cast(AssetRef, copied["video"])
    assert copied_ref.media_type == "video/mp4"
    assert (root / "video" / copied_ref.name).read_bytes() == source_bytes


@pytest.mark.parametrize("format_name", ["mp4_h264", "webm_av1"])
def test_save_advertised_10_bit_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    images = _frames(2)
    images[:, :, :32, :] = 0.5
    images[:, :, 32:, :] = 0.502
    result = _save_frames(
        images=images,
        target={"mount": "comfy-output", "prefix": "video/ten"},
        fps=2.0,
        format=format_name,
        crf=0,
        bit_depth="10",
    )
    ref = cast(AssetRef, result["video"])
    with av.open(root / "video" / ref.name) as container:
        frame = next(container.decode(video=0))
        assert frame.format.name == "yuv420p10le"
        rgb = frame.to_ndarray(format="rgb48le")
        assert float(rgb[:, 32:, :].mean() - rgb[:, :32, :].mean()) > 30


def test_vp9_10_bit_uses_preserving_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    result = _save_frames(images=_frames(2), format="webm_vp9", bit_depth="10", fps=2.0)
    ref = cast(AssetRef, result["video"])
    assert ref.name.endswith(".mkv")
    with av.open(root / "video" / ref.name) as container:
        assert container.streams.video[0].codec_context.codec.canonical_name == "ffv1"
        assert next(container.decode(video=0)).format.name == "gbrp16le"


def test_selected_codec_unavailable_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)

    def codec(name: str, mode: Literal["r", "w"]) -> object:
        if name == "libsvtav1":
            raise RuntimeError("selected encoder libsvtav1 is unavailable")
        return AvCodec(name, mode)

    monkeypatch.setattr(av, "Codec", codec)
    with pytest.raises(RuntimeError, match="selected encoder libsvtav1 is unavailable"):
        _save_frames(images=_frames(2), format="webm_av1", fps=2.0)
    assert not list(root.rglob("*.webm"))


def test_vp9_alpha_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, snapshot = _mount(tmp_path, monkeypatch)
    saved = _save_frames(
        images=_frames(3, alpha=True),
        target={"mount": "comfy-output", "prefix": "video/alpha"},
        fps=3.0,
        format="webm_vp9",
        bit_depth="8",
    )
    ref = cast(AssetRef, saved["video"])
    path = root / "video" / ref.name
    with av.open(path) as container:
        assert container.streams.video[0].metadata["alpha_mode"] == "1"
    loaded = LoadVideo.execute(video=_bound(ref, snapshot))
    images = cast(np.ndarray, image_input(loaded["images"]))
    assert images.shape == (3, 64, 64, 4)
    assert images[..., 3].min() == pytest.approx(1 / 3, abs=2 / 255)
    assert images[..., 3].max() == pytest.approx(1.0, abs=2 / 255)


@pytest.mark.parametrize(
    ("alpha", "format_name", "fps"),
    [(False, "mp4_h264", 23.976), (True, "webm_vp9", 3.0)],
)
def test_rgb_and_rgba_round_trips_preserve_exact_frame_count_at_awkward_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alpha: bool,
    format_name: str,
    fps: float,
) -> None:
    _root, snapshot = _mount(tmp_path, monkeypatch)
    saved = _save_frames(images=_frames(3, alpha=alpha), fps=fps, format=format_name)
    ref = _bound(cast(AssetRef, saved["video"]), snapshot)
    loaded = LoadVideo.execute(video=ref)
    images = cast(np.ndarray, loaded["images"])
    assert loaded["frame_count"] == 3
    assert images.shape == (3, 64, 64, 4 if alpha else 3)


def test_load_selection_resize_timing_and_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, snapshot = _mount(tmp_path, monkeypatch)
    saved = _save_frames(
        images=_frames(8),
        fps=8.0,
        format="mp4_h264",
        audio=_audio(samples=32_000),
    )
    ref = _bound(cast(AssetRef, saved["video"]), snapshot)
    loaded = LoadVideo.execute(
        video=ref,
        force_rate=8.0,
        custom_width=20,
        custom_height=0,
        start_time=0.25,
        frame_load_cap=2,
        select_every_nth=2,
    )
    images = cast(np.ndarray, loaded["images"])
    assert images.shape == (2, 20, 20, 3)
    assert loaded["frame_count"] == 2
    assert loaded["fps"] == pytest.approx(4.0)
    assert loaded["duration"] == pytest.approx(0.5)
    audio = cast(dict[str, object], loaded["audio"])
    waveform = cast(np.ndarray, audio["waveform"])
    assert waveform.shape[:2] == (1, 1)
    assert waveform.shape[2] == pytest.approx(8_000, abs=1024)


def test_late_start_seeks_video_and_audio_and_bounds_decoded_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, snapshot = _mount(tmp_path, monkeypatch)
    saved = _save_frames(
        images=_frames(120),
        fps=30.0,
        format="mp4_h264",
        audio=_audio(samples=220_500, sample_rate=44_100),
    )
    ref = _bound(cast(AssetRef, saved["video"]), snapshot)
    real_open = cast(Any, av.open)
    seek_calls: list[int] = []

    class TrackingContainer:
        def __init__(self, container: object) -> None:
            self._container = container

        def __enter__(self):  # noqa: ANN204
            self._container.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *args: object) -> object:
            return self._container.__exit__(*args)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self._container, name)

        def seek(self, offset: int, **kwargs: object) -> None:
            seek_calls.append(offset)
            self._container.seek(offset, **kwargs)  # type: ignore[attr-defined]

    def tracking_open(*args: object, **kwargs: object) -> TrackingContainer:
        return TrackingContainer(real_open(*args, **kwargs))

    decoded = 0
    real_video_frames = video_module._video_frames

    def counting_frames(container: object, stream: object):  # noqa: ANN202
        nonlocal decoded
        for frame in real_video_frames(container, stream):
            decoded += 1
            yield frame

    monkeypatch.setattr(av, "open", tracking_open)
    monkeypatch.setattr(video_module, "_video_frames", counting_frames)
    loaded = LoadVideo.execute(video=ref, start_time=3.0, frame_load_cap=1)
    assert loaded["frame_count"] == 1
    assert len(seek_calls) == 2
    assert decoded < 20


def test_input_budget_preflight_runs_before_global_pixel_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = np.broadcast_to(np.zeros((1, 2, 2, 3), dtype=np.float32), (2**24, 2, 2, 3))

    def unexpected_scan(_array: object) -> object:
        raise AssertionError("isfinite ran before the byte budget check")

    monkeypatch.setattr(np, "isfinite", unexpected_scan)
    with pytest.raises(ValueError, match="512 MiB"):
        assemble_video(images)


def test_encoded_output_spools_to_disk_and_enforces_limit() -> None:
    spool = video_module._BoundedSpool(10 * 1024 * 1024)
    try:
        chunk = b"x" * (1024 * 1024)
        for _ in range(9):
            spool.write(chunk)
        assert cast(Any, spool)._rolled is True
        with pytest.raises(ValueError, match="output limit"):
            spool.write(chunk * 2)
    finally:
        spool.close()


def test_bool_crf_is_rejected() -> None:
    with pytest.raises(ValueError, match="crf.*integer"):
        _save_frames(images=_frames(1), crf=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"custom_width": 512.5}, "custom_width"),
        ({"custom_width": "512"}, "custom_width"),
        ({"custom_width": True}, "custom_width"),
        ({"custom_height": 512.0}, "custom_height"),
        ({"frame_load_cap": 1.5}, "frame_load_cap"),
        ({"frame_load_cap": "1"}, "frame_load_cap"),
        ({"frame_load_cap": True}, "frame_load_cap"),
        ({"select_every_nth": 2.0}, "select_every_nth"),
        ({"select_every_nth": "2"}, "select_every_nth"),
    ],
)
def test_load_rejects_non_integer_frame_controls(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    # Validation fires before any decode work, so the asset bytes are never read.
    payload = b"never decoded"
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    ref = AssetRef(digest, "clip.mp4", len(payload), "video/mp4", resolver=vault)
    with pytest.raises(ValueError, match=match):
        LoadVideo.execute(video=ref, **kwargs)


def test_load_refuses_malformed_asset(tmp_path: Path) -> None:
    payload = b"not a video"
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    ref = AssetRef(digest, "bad.mp4", len(payload), "video/mp4", resolver=vault)
    with pytest.raises(InvalidDataError):
        LoadVideo.execute(video=ref)


@pytest.mark.parametrize(
    ("images", "match"),
    [
        (np.empty((0, 8, 8, 3), dtype=np.float32), "layout"),
        (np.full((1, 8, 8, 3), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_save_refuses_invalid_frame_batches(images: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _save_frames(images=images, format="mp4_h264")


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((1, 7, 8, 3), dtype=np.float32),
        np.zeros((1, 8, 8, 4), dtype=np.float32),
    ],
)
def test_save_uses_compatible_format_for_non_h264_frame_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, images: np.ndarray
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    result = _save_frames(images=images, format="mp4_h264")
    ref = cast(AssetRef, result["video"])
    with av.open(root / "video" / ref.name) as container:
        assert len(list(container.decode(video=0))) == 1


def test_video_schemas_preserve_preview_and_asset_contracts() -> None:
    load = LoadVideo.schema()
    load_value = LoadVideoValue.schema()
    save = SaveVideo.schema()
    save_value = SaveVideoValue.schema()
    assert load.node_type == "dinkster.load_video"
    assert load_value.node_type == "dinkster.load_video_value"
    assert save.node_type == "dinkster.save_video"
    assert save_value.node_type == "dinkster.save_video_value"
    assert load.aliases == ()
    assert save.aliases == ()
    assert load.inputs[0].type.element is not None
    assert load.inputs[0].type.element.types == ("comfy.VIDEO",)
    assert load.inputs[0].type.kind == "asset"
    assert [(output.id, output.preview, output.optional) for output in load.outputs] == [
        ("images", True, False),
        ("frame_count", False, False),
        ("audio", True, True),
        ("fps", False, False),
        ("duration", False, False),
    ]
    assert save.outputs[0].type.types == ("comfy.VIDEO",)
    assert save.outputs[1].id == "asset"
    assert save.outputs[1].type.kind == "asset"
    assert save.outputs[1].preview is True
    assert save.output_node is True
    assert save.idempotent is False
    assert load_value.outputs[0].type.types == ("comfy.VIDEO",)
    assert save_value.inputs[0].type.types == ("comfy.VIDEO",)
    assert save_value.outputs[0].type.kind == "asset"
    assert save_value.output_node is True
    assert save_value.idempotent is False
