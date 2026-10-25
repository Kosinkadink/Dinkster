"""Checkpoint weights as native runtime hooks."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import lru_cache
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


def _schema(node_type: str, *, model_only: bool) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name="Create Hook Model as LoRA" + (" (MO)" if model_only else ""),
        category="advanced/hooks/create",
        inputs=(
            InputSpec(
                "ckpt_name",
                TypeExpr.concrete(ASSET_TYPE),
                widget=AssetWidget(kind="model/checkpoint", accept=("application/octet-stream",)),
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


@lru_cache(maxsize=1)
def _weights(
    identity: tuple[str, str], path: str, embedding_roots: tuple[str, ...]
) -> tuple[Any, Any]:
    # Strength changes reuse extracted weights, not a loaded checkpoint or resolver.
    del identity
    sd = importlib.import_module("dinkster_inference.sd")
    hooks = importlib.import_module("dinkster_inference.hooks")
    loaded = sd.load_checkpoint_guess_config(
        path,
        output_vae=True,
        output_clip=True,
        embedding_directory=list(embedding_roots),
    )
    return (
        hooks.get_patch_weights_from_model(loaded[0]),
        hooks.get_patch_weights_from_model(loaded[1].patcher if loaded[1] else loaded[1]),
    )


class CreateHookModelAsLora(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("CreateHookModelAsLora", model_only=False)

    @classmethod
    def execute(
        cls,
        *,
        ckpt_name: object,
        strength_model: float,
        strength_clip: float,
        prev_hooks: Any = None,
    ) -> Mapping[str, object]:
        if not isinstance(ckpt_name, (AssetRef, LocalAssetRef)):
            raise TypeError("ckpt_name must be an AssetRef or LocalAssetRef")
        # Revalidate authority even when extracted weights can be reused.
        path = str(ckpt_name.local_path())
        api = importlib.import_module("dinkster_api.v1")
        embedding_roots = tuple(str(root) for root in api.mount_roots_for_kind("model/embedding"))
        initialize_inference_process()
        hooks = importlib.import_module("dinkster_inference.hooks")
        identity = (
            ("verified", ckpt_name.digest)
            if isinstance(ckpt_name, AssetRef)
            else ("local", ckpt_name.binding)
        )
        weights_model, weights_clip = _weights(identity, path, embedding_roots)
        added = hooks.create_hook_model_as_lora(
            weights_model=weights_model,
            weights_clip=weights_clip,
            strength_model=strength_model,
            strength_clip=strength_clip,
        )
        previous = hooks.HookGroup() if prev_hooks is None else prev_hooks
        return cls.outputs(HOOKS=previous.clone_and_combine(added))


class CreateHookModelAsLoraModelOnly(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("CreateHookModelAsLoraModelOnly", model_only=True)

    @classmethod
    def execute(
        cls,
        *,
        ckpt_name: object,
        strength_model: float,
        prev_hooks: Any = None,
    ) -> Mapping[str, object]:
        return CreateHookModelAsLora.execute(
            ckpt_name=ckpt_name,
            strength_model=strength_model,
            strength_clip=0.0,
            prev_hooks=prev_hooks,
        )


HOOK_CHECKPOINT_NODES = (CreateHookModelAsLora, CreateHookModelAsLoraModelOnly)
