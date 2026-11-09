"""LoRA weights as native runtime hooks."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_FLOAT,
    HOOKS_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

from .hook_scheduling import retain_hook_weights


def _schema(node_type: str, *, model_only: bool) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name="Create Hook LoRA" + (" (MO)" if model_only else ""),
        category="advanced/hooks/create",
        inputs=(
            InputSpec(
                "lora_name",
                TypeExpr.concrete(ASSET_TYPE),
                widget=AssetWidget(kind="model/lora", accept=("application/octet-stream",)),
            ),
            *(
                InputSpec(
                    name,
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=-20.0, max=20.0, step=0.01),
                )
                for name in (
                    ("strength_model",) if model_only else ("strength_model", "strength_clip")
                )
            ),
            InputSpec("prev_hooks", TypeExpr.concrete(HOOKS_TYPE), required=False),
        ),
        outputs=(OutputSpec("HOOKS", TypeExpr.concrete(HOOKS_TYPE)),),
    )


def _weights(path: str) -> dict[str, Any]:
    return importlib.import_module("dinkster_inference.utils").load_torch_file(path, safe_load=True)


def _ram_bytes(weights: dict[str, Any]) -> int:
    storages: dict[int, int] = {}
    for tensor in weights.values():
        if tensor.device.type == "cpu":
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


class CreateHookLora(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("CreateHookLora", model_only=False)

    @classmethod
    def execute(
        cls,
        *,
        lora_name: object,
        strength_model: float,
        strength_clip: float,
        prev_hooks: Any = None,
    ) -> Mapping[str, object]:
        if not isinstance(lora_name, (AssetRef, LocalAssetRef)):
            raise TypeError("lora_name must be an AssetRef or LocalAssetRef")
        initialize_inference_process()
        hooks = importlib.import_module("dinkster_inference.hooks")
        previous = hooks.HookGroup() if prev_hooks is None else prev_hooks
        if strength_model == 0 and strength_clip == 0:
            return cls.outputs(HOOKS=previous)
        path = str(lora_name.local_path())
        api = importlib.import_module("dinkster_api.v1")
        identity = (
            ("verified", lora_name.digest)
            if isinstance(lora_name, AssetRef)
            else ("local", lora_name.binding)
        )
        key = json.dumps({"purpose": "lora-hook-weights", "asset": identity}, separators=(",", ":"))
        weights = api.obtain_resident(key, lambda: _weights(path), ram_bytes=_ram_bytes)
        added = hooks.create_hook_lora(
            lora=weights, strength_model=strength_model, strength_clip=strength_clip
        )
        result = previous.clone_and_combine(added)
        added._dinkster_resident_refs = (weights,)
        return cls.outputs(HOOKS=retain_hook_weights(result, previous, added))


class CreateHookLoraModelOnly(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("CreateHookLoraModelOnly", model_only=True)

    @classmethod
    def execute(
        cls, *, lora_name: object, strength_model: float, prev_hooks: Any = None
    ) -> Mapping[str, object]:
        return CreateHookLora.execute(
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=0.0,
            prev_hooks=prev_hooks,
        )


HOOK_LORA_NODES = (CreateHookLora, CreateHookLoraModelOnly)
