"""Encoder export writes actual safetensors through mounted output authority."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from dinkster_nodes_comfy.model_file_savers import CLIPSave, VAESave
from safetensors import safe_open


class _Clip:
    def load_model(self) -> None:
        pass

    def state_dict_for_saving(self) -> dict[str, torch.Tensor]:
        return {
            "clip_l.transformer.left": torch.tensor([1.25, -3.5]),
            "clip_g.right": torch.tensor([[8, -2]], dtype=torch.int32),
            "transformer.other": torch.tensor([9.0], dtype=torch.float64),
        }


class _Vae:
    def get_sd(self) -> dict[str, torch.Tensor]:
        return {"decoder.weight": torch.tensor([[1.25, -3.5]], dtype=torch.float64)}


def test_encoder_exports_partition_clip_keys_and_preserve_vae_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "output", "root": str(output), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                ],
                "outputMount": "output",
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    assert CLIPSave.execute(clip=_Clip(), mount="alternate") == {}
    assert VAESave.execute(vae=_Vae()) == {}
    expected = (
        (alternate / "clip/ComfyUI_clip_l_00001.safetensors", "left", torch.tensor([1.25, -3.5])),
        (
            alternate / "clip/ComfyUI_clip_g_00001.safetensors",
            "right",
            torch.tensor([[8, -2]], dtype=torch.int32),
        ),
        (
            alternate / "clip/ComfyUI_00001.safetensors",
            "other",
            torch.tensor([9.0], dtype=torch.float64),
        ),
        (
            output / "vae/ComfyUI_vae_00001.safetensors",
            "decoder.weight",
            torch.tensor([[1.25, -3.5]], dtype=torch.float64),
        ),
    )
    for path, key, tensor in expected:
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert list(handle.keys()) == [key]
            assert handle.metadata() == (
                {"prompt": ""} if key == "decoder.weight" else {"format": "pt", "prompt": ""}
            )
            saved = handle.get_tensor(key)
            assert saved.dtype == tensor.dtype
            torch.testing.assert_close(saved, tensor, rtol=0, atol=0)
