"""Native model-family node adapters."""

from __future__ import annotations

from typing import Any

from dinkster_inference.component_registry import execution_symbol

_MODEL_PATCH_LOADERS = ("dinkster_native.families.minimax_h3:load_model_patch",)


def load_registered_model_patch(
    asset: object,
    source: object,
    context: object,
) -> tuple[Any, str] | None:
    for reference in _MODEL_PATCH_LOADERS:
        loaded = execution_symbol(reference)(asset, source, context)
        if loaded is not None:
            return loaded
    return None
