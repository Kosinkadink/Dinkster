"""Native asset-backed video I/O contracts."""

from __future__ import annotations

import io
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import av
import dinkster_video_runtime.runtime as video_module
import numpy as np
import pytest
from av.codec import Codec as AvCodec
from av.error import InvalidDataError
from dinkster_api.v1 import image_input
from dinkster_assets import (
    AssetRef,
    AssetVault,
    AssetWriter,
    MountSnapshotResolver,
    MountSnapshotWriter,
    digest_bytes,
)
from dinkster_values import edit_video
from dinkster_video.video_load import LoadVideo
from dinkster_video.webm_save import EncodedVideoSpool
from dinkster_video_runtime import assemble_video, disassemble_video, save_video_stream

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


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
    container, codec = format.split("_", 1)
    with io.BytesIO() as output:
        suffix, mime = save_video_stream(video, output, container=container, codec=codec, crf=crf)
        writer = AssetWriter(MountSnapshotWriter(os.environ["DINKSTER_MOUNTS_SNAPSHOT"]))
        ref = writer.save_bytes(
            target or {"mount": "comfy-output", "prefix": "video/clip"},
            output.getvalue(),
            suffix=suffix,
            media_type=mime,
        )
    return {"video": ref}


@pytest.mark.parametrize("indexed", (False, True))
def test_public_video_admission_respects_bound_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indexed: bool
) -> None:
    from dinkster_api.v1 import admit_video_source
    from dinkster_assets import AssetError, LocalAssetRef, MountDef, MountTable
    from dinkster_values import video_source
    from dinkster_video_runtime import save_video_stream

    root = tmp_path / "input"
    root.mkdir()
    path = root / "clip.mp4"
    with path.open("w+b") as output:
        save_video_stream(assemble_video(_frames(), fps=7), output)
    data = path.read_bytes()
    table = MountTable(tmp_path / "mounts.json")
    table.add(MountDef(id="input", path=root), kind="media/video")
    bound = table.bind_name(path.name, "media/video")
    assert isinstance(bound, LocalAssetRef)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(tmp_path / "scratch"))
    if indexed:
        table.scan("input")
        admitted = admit_video_source(bound)
        source = video_source(admitted)
        assert isinstance(source, AssetRef)
        assert source.digest == digest_bytes(data)
        assert source.read_bytes() == data
        assert video_source(admit_video_source(source)) is source
    else:
        assert "digest" not in bound.to_wire()
        with pytest.raises(AssetError, match="must be indexed"):
            admit_video_source(bound)
    assert not (tmp_path / "vault").exists()
    assert not (tmp_path / "scratch").exists()
    assert bound.read_bytes() == data


def test_video_admission_enforces_size_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_api.v1 import admit_video_source
    from dinkster_assets import AssetError
    from dinkster_assets import value as asset_value

    data = b"too large to admit"
    monkeypatch.setattr(asset_value, "GIBIBYTE", len(data) - 1)
    with pytest.raises(AssetError, match="exceeds 1 GiB"):
        admit_video_source(data)


def test_indexed_video_over_one_gib_is_admitted_without_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_api.v1 import admit_video_source
    from dinkster_assets import AssetError, ChainResolver, MountDef, MountTable
    from dinkster_assets.value import bind_video_value
    from dinkster_values import GIBIBYTE, decode_video, encode_video, video_source
    from dinkster_values.video_document import video_reference
    from dinkster_video_runtime import save_video_stream
    from dinkster_video_runtime.document import clip, make, mutate
    from dinkster_video_runtime.timeline import single_clip_video
    from dinkster_video_runtime.timeline_runtime import SourceMedia

    root = tmp_path / "input"
    root.mkdir()
    path = root / "large.mp4"
    with path.open("w+b") as output:
        save_video_stream(assemble_video(_frames(), fps=7), output)
        output.truncate(GIBIBYTE + 1)
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot)
    table.add(MountDef(id="input", path=root), kind="media/video")
    table.scan("input")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    selected = MountSnapshotResolver(snapshot).bind_relative_file("media/video", root, path.name)
    assert isinstance(selected, AssetRef)
    assert selected.size == GIBIBYTE + 1
    value = admit_video_source(selected)
    source = video_source(value)
    assert isinstance(source, AssetRef)
    assert source.local_path() == path
    path_source = video_source(admit_video_source(path))
    assert isinstance(path_source, AssetRef)
    assert path_source.digest == selected.digest
    loaded = LoadVideo.execute(file=selected)["VIDEO"]
    loaded_source = video_source(loaded)
    assert isinstance(loaded_source, AssetRef)
    assert loaded_source.digest == selected.digest
    received = bind_video_value(
        decode_video(encode_video(value)), ChainResolver(MountSnapshotResolver(snapshot))
    )
    received_source = video_source(received)
    assert isinstance(received_source, AssetRef)
    assert received_source.local_path() == path
    assert not (tmp_path / "vault").exists()
    assert not (tmp_path / "scratch").exists()
    document = make(width=64, height=64, rate=7, clips=[clip("selected", duration=0.25)])
    document = mutate(
        document, "bind_source", {"source": "selected", "reference": video_reference(received)}
    )
    rendered = single_clip_video(
        document,
        SourceMedia(lambda wire: AssetRef.from_wire(wire, MountSnapshotResolver(snapshot))),
    )
    assert rendered is not None
    rendered_source = video_source(rendered)
    assert isinstance(rendered_source, AssetRef)
    assert rendered_source.digest == selected.digest
    assert rendered_source.local_path() == path
    for store in (tmp_path / "vault", tmp_path / "scratch"):
        assert not any(entry.is_file() for entry in store.rglob("*"))
    table.remove("input")
    with pytest.raises(AssetError):
        received_source.open()
    with pytest.raises(AssetError):
        rendered_source.open()


def test_ungranted_video_source_over_copy_limit_is_rejected(tmp_path: Path) -> None:
    from dinkster_api.v1 import admit_video_source
    from dinkster_assets import AssetError
    from dinkster_values import GIBIBYTE

    path = tmp_path / "ungranted.mp4"
    with path.open("wb") as handle:
        handle.truncate(GIBIBYTE + 1)
    with pytest.raises(AssetError, match="requires an indexed asset reference"):
        admit_video_source(path)


@pytest.mark.parametrize("change", ("replace", "revoke", "regrant"))
def test_video_admission_rejects_changed_binding_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from dinkster_api.v1 import admit_video_source
    from dinkster_assets import AssetError, LocalAssetRef, MountDef, MountTable
    from dinkster_assets import value as asset_value

    root = tmp_path / "input"
    root.mkdir()
    path = root / "clip.mp4"
    path.write_bytes(b"not decoded")
    table = MountTable(tmp_path / "mounts.json")
    mount = MountDef(id="input", path=root)
    table.add(mount, kind="media/video")
    bound = table.bind_name(path.name, "media/video")
    assert isinstance(bound, LocalAssetRef)
    if change == "replace":
        path.unlink()
        path.write_bytes(b"replacement")
    else:
        table.remove("input")
        if change == "regrant":
            table.add(mount, kind="media/video")

    def unexpected_probe(*args: object, **kwargs: object) -> object:
        pytest.fail("revoked or changed source reached video probe")

    monkeypatch.setattr(asset_value, "video_from_source", unexpected_probe)
    monkeypatch.setattr(asset_value, "probe_video", unexpected_probe)
    with pytest.raises(AssetError, match="unavailable"):
        admit_video_source(bound)


@pytest.mark.parametrize(
    ("format_name", "container_name", "codec_name", "media_type"),
    [
        ("mp4_h264", "mov,mp4,m4a,3gp,3g2,mj2", "h264", "video/mp4"),
        ("webm_vp9", "matroska,webm", "vp9", "video/webm"),
        ("webm_av1", "matroska,webm", "av1", "video/webm"),
    ],
)
def test_runtime_saves_8_bit_formats_with_audio(
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


def test_runtime_save_preserves_depth_and_json_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    video = assemble_video(_frames(2), fps=2, bit_depth="10", color_space="HDR PQ")
    workflow = {"nodes": [{"id": "clip"}], "enabled": True}
    path = root / "metadata.mp4"
    with path.open("w+b") as output:
        save_video_stream(video, output, codec="h264", metadata={"workflow": workflow})
    with av.open(path) as container:
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

    value = LoadVideo.execute(file=source_ref)["VIDEO"]
    assert cast(dict[str, object], value)["source"] is source_ref
    assert cast(dict[str, object], value)["edits"] == []
    with io.BytesIO() as output:
        _suffix, mime = save_video_stream(value, output)
        assert mime == "video/mp4"
        assert output.getvalue() == source_bytes


@pytest.mark.parametrize("format_name", ["mp4_h264", "webm_av1"])
def test_runtime_saves_10_bit_formats(
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
    loaded = disassemble_video(LoadVideo.execute(file=_bound(ref, snapshot))["VIDEO"])
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
    loaded = disassemble_video(LoadVideo.execute(file=ref)["VIDEO"])
    images = cast(np.ndarray, loaded["images"])
    assert loaded["frame_count"] == 3
    assert images.shape == (3, 64, 64, 4 if alpha else 3)


@pytest.mark.parametrize("format_name", ["mp4_h264", "mkv_ffv1"])
def test_late_start_bounds_video_decode_and_preserves_audio_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format_name: str
) -> None:
    from dinkster_values.audio_codec import audio_window

    _root, snapshot = _mount(tmp_path, monkeypatch)
    soundtrack = _audio(samples=220_500, sample_rate=44_100)
    waveform = cast(np.ndarray, soundtrack["waveform"])
    waveform *= np.linspace(0.25, 1, waveform.shape[-1], dtype=np.float32)
    saved = _save_frames(
        images=_frames(120),
        fps=30.0,
        format=format_name,
        audio=soundtrack,
    )
    ref = _bound(cast(AssetRef, saved["video"]), snapshot)
    value = LoadVideo.execute(file=ref)["VIDEO"]
    full_audio = audio_window(disassemble_video(value)["audio"], 0, 176_400)["waveform"]
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
    real_video_frames = video_module._decoded_video

    def counting_frames(container: object, stream: object):  # noqa: ANN202
        nonlocal decoded
        for frame in real_video_frames(container, stream):
            decoded += 1
            yield frame

    monkeypatch.setattr(av, "open", tracking_open)
    monkeypatch.setattr(video_module, "_decoded_video", counting_frames)
    loaded = disassemble_video(
        edit_video(value, {"trim": {"start_time": 3, "duration": Fraction(1, 30)}})
    )
    assert loaded["frame_count"] == 1
    audio = audio_window(loaded["audio"], 0, 1470)
    assert audio["waveform"].shape == (1, 1, 1470)
    np.testing.assert_array_equal(audio["waveform"], full_audio[..., 132_300:133_770])
    # AAC requires origin decode for its stateful PNS; FLAC can seek independently.
    assert len(seek_calls) == (1 if format_name == "mp4_h264" else 2)
    assert decoded < 20


def test_encoded_output_spools_to_disk_and_enforces_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_video import webm_save

    monkeypatch.setattr(webm_save, "ENCODED_MEDIA_LIMIT_BYTES", 10 * 1024 * 1024)
    spool = EncodedVideoSpool()
    try:
        chunk = b"x" * (1024 * 1024)
        for _ in range(9):
            spool.write(chunk)
        assert cast(Any, spool)._rolled is True
        with pytest.raises(ValueError, match="output byte limit"):
            spool.write(chunk * 2)
    finally:
        spool.close()


def test_bool_crf_is_rejected() -> None:
    with pytest.raises(ValueError, match="crf must be a finite number in 0..51"):
        _save_frames(images=_frames(1), crf=True)


def test_load_refuses_malformed_asset(tmp_path: Path) -> None:
    payload = b"not a video"
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    ref = AssetRef(digest, "bad.mp4", len(payload), "video/mp4", resolver=vault)
    with pytest.raises(InvalidDataError):
        LoadVideo.execute(file=ref)


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
