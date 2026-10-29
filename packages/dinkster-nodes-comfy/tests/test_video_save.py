"""Mounted output behavior of the native SaveVideo body.

SaveVideo provenance: ComfyUI comfy_extras/nodes_video.py at immutable
revision bc5f880dbdd43e426077f42ccec53bb577ffee9f. Inputs are the exact
three 8x12 RGB fixture frames below at 3 FPS, with auto container/codec or
MKV/H.264 re-encode at CRF18 (and fractional18.25). Auto resolves to MP4
and H.264. Frame count, dimensions, codec and unchanged source/output are
hand-derived contract checks. The red-left/green-top/blue-right ordering
comes directly from the fixture; the 100-byte contrast margin distinguishes
the painted region from black after lossy encoding, not an F measurement.
Mount paths/adjacency and opaque submitted metadata are Dinkster contracts.
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest
from dinkster_api.v1 import ExportSnapshot, MountWidget, read_video_metadata
from dinkster_nodes_comfy.video_components import CreateVideo
from dinkster_nodes_comfy.video_save import SaveVideo, SaveWEBM
from dinkster_schema.elaborate import elaborate


def _mounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    default, alternate = tmp_path / "default", tmp_path / "alternate"
    default.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "outputMount": "default",
                "mounts": [
                    {"id": "default", "root": str(default), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                ],
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return default, alternate


def _video() -> object:
    images = np.zeros((3, 8, 12, 3), dtype=np.float32)
    images[0, :, :4, 0] = 1
    images[1, :4, :, 1] = 1
    images[2, :, 4:, 2] = 1
    return CreateVideo.execute(images=images, fps=3.0)["VIDEO"]


@pytest.mark.parametrize("mount", (None, "alternate"))
def test_save_video_uses_default_or_explicit_mount_and_preserves_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount: str | None
) -> None:
    default, alternate = _mounts(tmp_path, monkeypatch)
    video = _video()
    images = video["components"]["images"].copy()
    result = SaveVideo.execute(
        video=video, filename_prefix="video/%width%x%height%/clip", format="auto", mount=mount
    )
    assert result["video"] is video
    np.testing.assert_array_equal(video["components"]["images"], images)
    root, untouched = (default, alternate) if mount is None else (alternate, default)
    paths = list(root.glob("video/12x8/clip_*.mp4"))
    assert len(paths) == 1
    assert not list(untouched.rglob("*"))
    with av.open(paths[0]) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (12, 8)
        assert stream.codec_context.name == "h264"
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    assert len(frames) == 3
    assert frames[0][:, :4, 0].mean() > frames[0][:, 4:, 0].mean() + 100
    assert frames[1][:4, :, 1].mean() > frames[1][4:, :, 1].mean() + 100
    assert frames[2][:, 4:, 2].mean() > frames[2][:, :4, 2].mean() + 100


@pytest.mark.parametrize("flattened", (False, True))
@pytest.mark.parametrize("crf", (18.0, 18.25))
def test_save_video_nested_and_elaborated_codec_controls_encode_mkv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flattened: bool, crf: float
) -> None:
    default, _ = _mounts(tmp_path, monkeypatch)
    controls = (
        {"format": "mkv", "format.codec": "h264", "format.codec.encoding.crf": crf}
        if flattened
        else {
            "format": {
                "format": "mkv",
                "codec": {"codec": "h264", "encoding": {"encoding": "re-encode", "crf": crf}},
            }
        }
    )
    SaveVideo.execute(video=_video(), filename_prefix="clip", **controls)
    paths = list(default.glob("clip_*.mkv"))
    assert len(paths) == 1
    with av.open(paths[0]) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert len(list(container.decode(video=0))) == 3


def test_save_video_preserves_opaque_submitted_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_nodes_comfy import video_save

    default, _ = _mounts(tmp_path, monkeypatch)
    prompt = {"17": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "clip"}}}
    workflow = {"nodes": [{"id": 17, "type": "SaveVideo"}], "extra": {"value": [2, "x"]}}
    monkeypatch.setattr(
        video_save,
        "current_export_snapshot",
        lambda: ExportSnapshot(prompt=prompt, extra_pnginfo={"workflow": workflow, "flag": False}),
    )
    SaveVideo.execute(video=_video(), filename_prefix="clip", format="mkv")
    metadata = read_video_metadata(next(default.glob("clip_*.mkv")))
    assert metadata == {"prompt": prompt, "workflow": workflow, "flag": False}


def test_save_video_schema_keeps_order_and_appends_adjacent_mount() -> None:
    schema = SaveVideo.schema()
    assert schema.input_order == ("video", "filename_prefix", "format", "codec", "mount")
    mount = schema.inputs[-1]
    assert mount.id == "mount"
    assert not mount.required and mount.default is None
    assert mount.adjacent_to == "filename_prefix"
    assert isinstance(mount.widget, MountWidget)
    assert schema.outputs[0].preview and schema.outputs[0].id == "video"
    effective = elaborate(
        schema,
        ["video", "filename_prefix", "format.codec.encoding.crf", "mount"],
        slot_variants={
            "format": "mkv",
            "format.codec": "h264",
            "format.codec.encoding": "re-encode",
        },
    )
    assert [item.id for item in effective.inputs] == [
        "video",
        "filename_prefix",
        "format.codec.encoding.crf",
        "mount",
    ]


def test_save_video_refuses_unknown_mount_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default, alternate = _mounts(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="mount"):
        SaveVideo.execute(video=_video(), filename_prefix="clip", format="auto", mount="unknown")
    assert not list(default.rglob("*"))
    assert not list(alternate.rglob("*"))


@pytest.mark.parametrize("channels", (3, 4))
def test_save_webm_preserves_frames_metadata_source_and_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: int
) -> None:
    # Hand-derived from SaveWEBM in ComfyUI comfy_extras/nodes_video.py,
    # bc5f880dbdd43e426077f42ccec53bb577ffee9f: three 16x24 frames at
    # 3.125 FPS, VP9/CRF24; RGB/RGBA red-left and green-right fixtures.
    # These geometry/color expectations are not captured reference outputs.
    from dinkster_nodes_comfy import video_save

    default, alternate = _mounts(tmp_path, monkeypatch)
    pixels = np.zeros((3, 16, 24, channels), dtype=np.float32)
    pixels[:, :, :12, 0] = 1
    pixels[:, :, 12:, 1] = 1
    if channels == 4:
        pixels[..., 3] = 0.5
    original = pixels.copy()
    prompt = {"4": {"class_type": "SaveWEBM", "inputs": {"fps": 3.125}}}
    workflow = {"nodes": [{"id": 4, "type": "SaveWEBM"}]}
    monkeypatch.setattr(
        video_save,
        "current_export_snapshot",
        lambda: ExportSnapshot(prompt=prompt, extra_pnginfo={"workflow": workflow}),
    )
    result = SaveWEBM.execute(
        images=pixels,
        filename_prefix="web/clip",
        codec="vp9",
        fps=3.125,
        crf=24.0,
        mount="alternate",
    )
    assert result["images"] is pixels
    np.testing.assert_array_equal(pixels, original)
    assert not list(default.rglob("*"))
    paths = list(alternate.glob("web/clip_*.webm"))
    assert len(paths) == 1
    with av.open(paths[0]) as container:
        tags = {name.lower(): value for name, value in container.metadata.items()}
        assert json.loads(tags["prompt"]) == prompt
        assert json.loads(tags["workflow"]) == workflow
        stream = container.streams.video[0]
        assert stream.codec_context.name == "vp9"
        assert (stream.width, stream.height) == (24, 16)
        assert float(stream.average_rate) == 3.125
        if channels == 4:
            assert stream.metadata["alpha_mode"] == "1"
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    assert len(frames) == 3
    assert frames[0][:, :10, 0].mean() > frames[0][:, 14:, 0].mean() + 100
    assert frames[0][:, 14:, 1].mean() > frames[0][:, :10, 1].mean() + 100
