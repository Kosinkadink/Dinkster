"""Controlled adapters for LoRA file-loader node contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_models import lora_file_loaders as loaders


class _VerifiedAsset:
    def __init__(self) -> None:
        self.digest = "digest"
        self.local_path_calls = 0

    def local_path(self) -> Path:
        self.local_path_calls += 1
        return Path("/verified/lora.safetensors")


class _LocalAsset:
    def __init__(self) -> None:
        self.binding = "mount:lora.safetensors"
        self.local_path_calls = 0

    def local_path(self) -> Path:
        self.local_path_calls += 1
        return Path("/local/lora.safetensors")


def _patch_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loaders, "AssetRef", _VerifiedAsset)
    monkeypatch.setattr(loaders, "LocalAssetRef", _LocalAsset)


def test_zero_strength_returns_before_asset_resolution_or_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized: list[bool] = []
    monkeypatch.setattr(loaders, "initialize_inference_process", lambda: initialized.append(True))
    asset = object()

    assert loaders.LoraLoader.execute(
        model="model", clip="clip", lora_name=asset, strength_model=0.0, strength_clip=0.0
    ) == {"MODEL": "model", "CLIP": "clip"}
    assert loaders.LoraLoaderBypassModelOnly.execute(
        model="model", lora_name=asset, strength_model=0.0
    ) == {"MODEL": "model"}
    assert initialized == []


def test_stack_rows_preserve_order_and_skip_disabled_or_zero_strength(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def apply(*args: object, **kwargs: object) -> tuple[str, str]:
        calls.append(args)
        return f"{args[0]}+{args[2]}", f"{args[1]}+{args[2]}"

    monkeypatch.setattr(loaders, "_apply_lora", apply)
    rows = {
        "second.lora_name": "second",
        "second.strength": -0.5,
        "second.enabled": True,
        "disabled.lora_name": "disabled",
        "disabled.strength": 1.0,
        "disabled.enabled": False,
        "zero.lora_name": "zero",
        "zero.strength": 0.0,
        "zero.enabled": True,
        "first.lora_name": "first",
        "first.strength": 1.25,
        "first.enabled": True,
    }

    assert loaders.LoadLoraModel.execute(model="model", loras=rows) == {
        "MODEL": "model+second+first"
    }
    assert calls == [
        ("model", None, "second", -0.5, 0.0),
        ("model+second", None, "first", 1.25, 0.0),
    ]

    calls.clear()
    assert loaders.LoadLoraTextEncoder.execute(clip="clip", loras=rows) == {
        "CLIP": "clip+second+first"
    }
    assert calls == [
        (None, "clip", "second", 0.0, -0.5),
        (None, "clip+second", "first", 0.0, 1.25),
    ]
