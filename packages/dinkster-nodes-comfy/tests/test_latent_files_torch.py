"""Actual safetensors latent files across the public asset boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import AssetRef
from dinkster_assets import digest_file
from dinkster_nodes_comfy.latent_file_tools import LoadLatent, SaveLatent
from safetensors import safe_open
from safetensors.torch import save_file


@pytest.mark.parametrize("versioned", (False, True))
def test_load_latent_converts_to_float_and_scales_only_legacy_files(
    tmp_path: Path, versioned: bool
) -> None:
    path = tmp_path / "selected.latent"
    tensor = torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]], dtype=torch.float64)
    state = {"latent_tensor": tensor}
    if versioned:
        state["latent_format_version_0"] = torch.tensor([])
    save_file(state, str(path))
    asset = AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=SimpleNamespace(resolve=lambda _digest: path),
    )
    actual = LoadLatent.execute(latent=asset)["LATENT"]
    expected = torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]])
    if not versioned:
        expected *= 1.0 / 0.18215
    assert set(actual) == {"samples"}
    assert actual["samples"].dtype == torch.float32
    assert actual["samples"].device.type == "cpu"
    torch.testing.assert_close(actual["samples"], expected, rtol=0, atol=0)


def test_save_latent_preserves_source_and_writes_versioned_tensors_to_selected_mount(
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
    tensor = torch.tensor([[[[1.125, -3.5, 0.25], [8.0, -0.75, 2.0]]]], dtype=torch.float64)
    samples = {"samples": tensor.transpose(-1, -2), "batch_index": [7], "origin": "source"}
    assert not samples["samples"].is_contiguous()
    assert SaveLatent.execute(samples=samples)["samples"] is samples
    assert (
        SaveLatent.execute(samples=samples, filename_prefix="chosen/latent", mount="alternate")[
            "samples"
        ]
        is samples
    )
    for path in (
        output / "latents/ComfyUI_00001.latent",
        alternate / "chosen/latent_00001.latent",
    ):
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert set(handle.keys()) == {"latent_tensor", "latent_format_version_0"}
            assert handle.metadata() == {"prompt": ""}
            saved = handle.get_tensor("latent_tensor")
            assert saved.dtype == torch.float64
            assert saved.is_contiguous()
            torch.testing.assert_close(saved, tensor.transpose(-1, -2), rtol=0, atol=0)
            assert handle.get_tensor("latent_format_version_0").numel() == 0
    assert samples["batch_index"] == [7] and samples["origin"] == "source"
    assert not samples["samples"].is_contiguous()
