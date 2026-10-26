"""Mounted output behavior of the native SaveVideo body."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest
from dinkster_api.v1 import MountWidget
from dinkster_nodes_comfy.video_components import CreateVideo
from dinkster_nodes_comfy.video_save import SaveVideo
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
def test_save_video_nested_and_elaborated_codec_controls_encode_mkv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flattened: bool
) -> None:
    default, _ = _mounts(tmp_path, monkeypatch)
    controls = (
        {"format": "mkv", "format.codec": "h264", "format.codec.encoding.crf": 18.0}
        if flattened
        else {
            "format": {
                "format": "mkv",
                "codec": {"codec": "h264", "encoding": {"encoding": "re-encode", "crf": 18.0}},
            }
        }
    )
    SaveVideo.execute(video=_video(), filename_prefix="clip", **controls)
    paths = list(default.glob("clip_*.mkv"))
    assert len(paths) == 1
    with av.open(paths[0]) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert len(list(container.decode(video=0))) == 3


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
