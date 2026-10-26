"""CPU coverage for real LoRA file-loader tensor application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DINKSTER_ACCELERATOR"] = "cpu"

import torch
from dinkster_api.v1 import (
    AssetRef,
    LocalAssetRef,
    initialize_inference_process,
)
from dinkster_assets import MountDef, MountTable
from dinkster_nodes_comfy.lora_file_loaders import (
    LoadLoraModel,
    LoadLoraTextEncoder,
    LoraLoader,
    LoraLoaderBypass,
    LoraLoaderBypassModelOnly,
    LoraLoaderModelOnly,
)


class _RawModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.linear = torch.nn.Linear(2, 2, bias=False)
        self.diffusion_model.linear.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.model_config = type("Config", (), {"unet_config": {}})()


class _RawClip(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = torch.nn.Linear(2, 2, bias=False)
        self.text_encoder.weight.data.copy_(torch.tensor([[2.0, -3.0], [4.0, 5.0]]))


def _model() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(_RawModel(), torch.device("cpu"), torch.device("cpu"))


def _clip() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    raw = _RawClip()
    clip = CLIP(no_init=True)
    clip.patcher = ModelPatcher(raw, torch.device("cpu"), torch.device("cpu"))
    clip.cond_stage_model = raw
    clip.tokenizer = object()
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = True
    clip.apply_hooks_to_conds = None
    return clip


def _asset(tmp_path: Path, *, verified: bool) -> AssetRef | LocalAssetRef:
    from safetensors.torch import save_file

    checkpoint = tmp_path / "tiny.safetensors"
    save_file(
        {
            "lora_unet_linear.lora_down.weight": torch.tensor([[2.0, -1.0]]),
            "lora_unet_linear.lora_up.weight": torch.tensor([[3.0], [5.0]]),
            "text_encoders.text_encoder.lora_down.weight": torch.tensor([[7.0, -2.0]]),
            "text_encoders.text_encoder.lora_up.weight": torch.tensor([[11.0], [13.0]]),
        },
        str(checkpoint),
        metadata={"purpose": "tiny-real-lora", "rank": "1"},
    )
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="loras", path=tmp_path), kind="model/lora")
    if verified:
        table.scan("loras")
    asset = table.bind_name(checkpoint.name, "model/lora")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    return asset


def _assert_normal_patches(model: Any, clip: Any) -> None:
    from dinkster_inference.lora import calculate_weight

    model_weight = calculate_weight(
        model.patches["diffusion_model.linear.weight"],
        model.model.diffusion_model.linear.weight.detach(),
        "diffusion_model.linear.weight",
    )
    clip_weight = calculate_weight(
        clip.patcher.patches["text_encoder.weight"],
        clip.cond_stage_model.text_encoder.weight.detach(),
        "text_encoder.weight",
    )
    torch.testing.assert_close(
        model_weight, torch.tensor([[-2.0, 3.5], [-2.0, 6.5]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        clip_weight, torch.tensor([[98.25, -30.5], [117.75, -27.5]]), rtol=0, atol=0
    )
    assert model.get_attachment("lora_metadata") == {"purpose": "tiny-real-lora", "rank": "1"}
    assert clip.patcher.get_attachment("lora_metadata") == {
        "purpose": "tiny-real-lora",
        "rank": "1",
    }


@pytest.mark.parametrize("verified", (False, True))
def test_real_safetensors_apply_through_all_lora_file_loaders(
    tmp_path: Path, verified: bool
) -> None:
    asset = _asset(tmp_path, verified=verified)
    full = LoraLoader.execute(
        model=_model(), clip=_clip(), lora_name=asset, strength_model=-0.5, strength_clip=1.25
    )
    _assert_normal_patches(full["MODEL"], full["CLIP"])

    model_only = LoraLoaderModelOnly.execute(model=_model(), lora_name=asset, strength_model=-0.5)
    assert set(model_only["MODEL"].patches) == {"diffusion_model.linear.weight"}

    model_stack = LoadLoraModel.execute(
        model=_model(),
        loras={
            "first.lora_name": asset,
            "first.strength": -0.5,
            "first.enabled": True,
            "disabled.lora_name": asset,
            "disabled.strength": 1.25,
            "disabled.enabled": False,
            "zero.lora_name": asset,
            "zero.strength": 0.0,
            "zero.enabled": True,
            "last.lora_name": asset,
            "last.strength": 1.25,
            "last.enabled": True,
        },
    )["MODEL"]
    assert [patch[0] for patch in model_stack.patches["diffusion_model.linear.weight"]] == [
        -0.5,
        1.25,
    ]

    clip_stack = LoadLoraTextEncoder.execute(
        clip=_clip(),
        loras={
            "first.lora_name": asset,
            "first.strength": -0.5,
            "first.enabled": True,
            "disabled.lora_name": asset,
            "disabled.strength": 1.25,
            "disabled.enabled": False,
            "zero.lora_name": asset,
            "zero.strength": 0.0,
            "zero.enabled": True,
            "last.lora_name": asset,
            "last.strength": 1.25,
            "last.enabled": True,
        },
    )["CLIP"]
    assert [patch[0] for patch in clip_stack.patcher.patches["text_encoder.weight"]] == [-0.5, 1.25]

    bypass = LoraLoaderBypass.execute(
        model=_model(), clip=_clip(), lora_name=asset, strength_model=-0.5, strength_clip=1.25
    )
    assert set(bypass["MODEL"].injections) == {"bypass_lora"}
    assert set(bypass["CLIP"].patcher.injections) == {"bypass_lora"}
    bypass["MODEL"].inject_model()
    torch.testing.assert_close(
        bypass["MODEL"].model.diffusion_model.linear(torch.tensor([[1.0, 0.0]])),
        torch.tensor([[-2.0, -2.0]]),
        rtol=0,
        atol=0,
    )
    bypass["MODEL"].eject_model()

    bypass_model_only = LoraLoaderBypassModelOnly.execute(
        model=_model(), lora_name=asset, strength_model=-0.5
    )["MODEL"]
    assert set(bypass_model_only.injections) == {"bypass_lora"}
