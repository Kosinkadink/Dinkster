"""Controlled adapters for LoRA file-loader node contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dinkster_api.v1 import ASSET_TYPE, CLIP_TYPE, CORE_BOOLEAN, CORE_FLOAT, MODEL_TYPE, TypeExpr
from dinkster_nodes_comfy import lora_file_loaders as loaders


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


def test_lora_file_loader_schemas_are_torch_free() -> None:
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from dinkster_nodes_comfy.lora_file_loaders import LORA_FILE_LOADER_NODES

assert tuple(node.schema().node_type for node in LORA_FILE_LOADER_NODES) == (
    "LoraLoader", "LoraLoaderModelOnly", "LoadLoraModel", "LoadLoraTextEncoder",
    "LoraLoaderBypass", "LoraLoaderBypassModelOnly",
)
assert "torch" not in sys.modules
assert not any(name.startswith("dinkster_inference.") for name in sys.modules)
""",
        ],
        check=True,
        cwd=root,
    )


def test_loader_schemas_preserve_file_picker_strength_and_stack_contracts() -> None:
    expected = {
        "LoraLoader": (
            "Load LoRA (Model and CLIP)",
            ("model", "clip", "lora_name", "strength_model", "strength_clip"),
        ),
        "LoraLoaderModelOnly": ("Load LoRA", ("model", "lora_name", "strength_model")),
        "LoadLoraModel": ("Load LoRA (Model)", ("model",)),
        "LoadLoraTextEncoder": ("Load LoRA (Text Encoder)", ("clip",)),
        "LoraLoaderBypass": (
            "Load LoRA (Bypass) (For debugging)",
            ("model", "clip", "lora_name", "strength_model", "strength_clip"),
        ),
        "LoraLoaderBypassModelOnly": (
            "Load LoRA (Bypass, Model Only) (for debugging)",
            ("model", "lora_name", "strength_model"),
        ),
    }
    for node in loaders.LORA_FILE_LOADER_NODES:
        schema = node.schema()
        display, inputs = expected[schema.node_type]
        assert schema.display_name == display
        assert schema.category == "model/loaders"
        assert tuple(item.id for item in schema.inputs) == inputs
        for item in schema.inputs:
            if item.id == "lora_name":
                assert item.type == TypeExpr.concrete(ASSET_TYPE)
                assert item.widget is not None and item.widget.kind == "model/lora"
            elif item.id.startswith("strength"):
                assert item.type == TypeExpr.concrete(CORE_FLOAT)
                assert item.default == 1.0
                assert (item.widget.min, item.widget.max, item.widget.step) == (-100.0, 100.0, 0.01)

    for node, expected_type in (
        (loaders.LoadLoraModel, MODEL_TYPE),
        (loaders.LoadLoraTextEncoder, CLIP_TYPE),
    ):
        schema = node.schema()
        family = schema.input_families[0]
        assert schema.outputs[0].type == TypeExpr.concrete(expected_type)
        assert (family.id, family.min_members, family.max_members, family.display_name) == (
            "loras",
            1,
            20,
            "LoRA",
        )
        fields = {item.id: item for item in family.template}
        assert fields["lora_name"].type == TypeExpr.concrete(ASSET_TYPE)
        assert fields["lora_name"].widget.kind == "model/lora"
        assert fields["strength"].default == 1.0
        assert fields["enabled"].type == TypeExpr.concrete(CORE_BOOLEAN)
        assert fields["enabled"].default is True


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


def test_normal_loader_forwards_metadata_and_revalidates_before_resident_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_assets(monkeypatch)
    asset = _VerifiedAsset()
    events: list[object] = []

    def obtain(key: str, load: Any, *, ram_bytes: Any) -> tuple[dict[str, str], dict[str, str]]:
        events.append(("resident", key, ram_bytes(({}, {}))))
        return load()

    def load(path: str, **options: object) -> tuple[dict[str, str], dict[str, str]]:
        events.append(("load", path, options))
        return {"weights": "normal"}, {"metadata": "kept"}

    def apply(*args: Any, **kwargs: Any) -> tuple[str, str]:
        events.append(("apply", args, kwargs))
        return "patched-model", "patched-clip"

    monkeypatch.setattr(
        loaders, "initialize_inference_process", lambda: events.append("initialize")
    )
    monkeypatch.setattr(loaders, "obtain_resident", obtain)
    monkeypatch.setattr(
        loaders.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(load_torch_file=load)
            if name == "dinkster_inference.utils"
            else SimpleNamespace(load_lora_for_models=apply)
        ),
    )

    assert loaders.LoraLoader.execute(
        model="model", clip="clip", lora_name=asset, strength_model=-0.5, strength_clip=1.25
    ) == {"MODEL": "patched-model", "CLIP": "patched-clip"}
    assert asset.local_path_calls == 1
    assert events[0] == "initialize"
    assert events[1][0] == "resident"
    assert events[1][1] == '{"purpose":"lora-weights","asset":["verified","digest"]}'
    assert events[2] == (
        "load",
        "/verified/lora.safetensors",
        {"safe_load": True, "return_metadata": True},
    )
    assert events[3][2] == {"lora_metadata": {"metadata": "kept"}}


def test_bypass_loader_uses_bypass_api_without_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_assets(monkeypatch)
    asset = _LocalAsset()
    calls: list[object] = []

    monkeypatch.setattr(loaders, "initialize_inference_process", lambda: calls.append("initialize"))
    monkeypatch.setattr(loaders, "obtain_resident", lambda _key, load, **_kwargs: load())
    monkeypatch.setattr(
        loaders.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(
                load_torch_file=lambda path, **options: (
                    calls.append(("load", path, options)) or {"weights": 1}
                )
            )
            if name == "dinkster_inference.utils"
            else SimpleNamespace(
                load_bypass_lora_for_models=lambda *args: (
                    calls.append(("bypass", args)) or ("bypass-model", "bypass-clip")
                )
            )
        ),
    )

    assert loaders.LoraLoaderBypass.execute(
        model="model", clip="clip", lora_name=asset, strength_model=-1.0, strength_clip=0.25
    ) == {"MODEL": "bypass-model", "CLIP": "bypass-clip"}
    assert asset.local_path_calls == 1
    assert calls[1] == (
        "load",
        "/local/lora.safetensors",
        {"safe_load": True, "return_metadata": False},
    )
    assert calls[2] == ("bypass", ("model", "clip", {"weights": 1}, -1.0, 0.25))


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
