"""CPU SaveImage output behavior."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from dinkster_nodes_comfy.image_file_tools import PreviewImage, SaveImage


def _mount(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    default = tmp_path / "default"
    alternate = tmp_path / "alternate"
    default.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "default", "root": str(default), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                ],
                "outputMount": "default",
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return default, alternate


def test_save_image_writes_each_cpu_batch_item_and_preserves_source(
    tmp_path: Path, monkeypatch
) -> None:
    default, alternate = _mount(tmp_path, monkeypatch)
    images = torch.tensor(
        [
            [[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    default_result = SaveImage.execute(images=images, filename_prefix="default/%width%x%height%")
    assert default_result["images"] is images
    assert (default / "default" / "2x1_00001.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (default / "default" / "2x1_00002.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    chosen = images[:1]
    chosen_result = SaveImage.execute(images=chosen, filename_prefix="chosen", mount="alternate")
    assert chosen_result["images"] is chosen
    assert (alternate / "chosen_00001.png").is_file()
    assert torch.equal(images[0, 0, 0], torch.tensor([0.0, 0.5, 1.0]))


def test_preview_image_preserves_cpu_source() -> None:
    images = torch.zeros((1, 2, 3, 3), dtype=torch.float32)

    assert PreviewImage.execute(images=images)["images"] is images
