"""Official Qwen Image per-component artifact planning and identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .assembly import (
    AssemblyError,
    ComponentPlan,
    QwenImageAssemblyPlan,
    QwenImageComponentRole,
    plan_qwen_image_component,
)
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .qwen_image import QWEN_IMAGE_CONFIG
from .weights import AssetIdentifiedSource, WeightSource

if TYPE_CHECKING:
    from .component_checkpoint import ComponentCheckpointPlan

QWEN_IMAGE_PROVIDER_REVISION = (
    "Comfy-Org/Qwen-Image_ComfyUI@46839d338df81ce625d5fae27d7e370314c0fbc9"
)


class QwenImageComponentAssemblyError(ValueError):
    """A source is not the exact official Qwen Image component requested."""


def qwen_image_checkpoint_assembly(plan: ComponentCheckpointPlan) -> QwenImageAssemblyPlan:
    """Validate the declared DiT, text and causal codec composition."""
    components = cast("Mapping[str, ComponentPlan[Any]]", plan.components)
    try:
        return QwenImageAssemblyPlan(
            plan.family,
            components["diffusion"],
            components["qwen2_5_vl_7b"],
            components["vae"],
            plan.unclaimed,
        )
    except (KeyError, ValueError) as error:
        raise AssemblyError(f"incomplete or incompatible Qwen Image checkpoint: {error}") from error


def plan_qwen_image_official_component(
    source: WeightSource,
    *,
    role: QwenImageComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[object]:
    """Plan one official split component and bind its immutable asset identity."""

    source_path = getattr(source, "path", None)
    if source_path != path:
        raise QwenImageComponentAssemblyError(
            f"Qwen Image {role} source path differs from artifact selection"
        )
    if not bind_asset_identity:
        return plan_qwen_image_component(source, role)
    if not isinstance(source, AssetIdentifiedSource):
        raise QwenImageComponentAssemblyError(f"Qwen Image {role} source must carry asset identity")
    if not source.asset_digest or type(source.asset_size) is not int or source.asset_size < 0:
        raise QwenImageComponentAssemblyError(f"Qwen Image {role} source must carry asset identity")
    plan = plan_qwen_image_component(source, role)
    return replace(
        plan,
        identity_facts=(
            *plan.identity_facts,
            f"provider_revision={QWEN_IMAGE_PROVIDER_REVISION}",
            f"asset_digest={source.asset_digest}",
            f"asset_size={source.asset_size}",
        ),
    )


def qwen_image_component_runtime_identity(
    plan: ComponentPlan[object],
    role: QwenImageComponentRole,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded Qwen Image component."""

    if plan.component != role:
        raise ValueError(f"Qwen Image {role} identity requires the {role} plan")
    return build_runtime_identity_from_facts(
        QWEN_IMAGE_CONFIG.family_id,
        runtime_component_identity(QWEN_IMAGE_CONFIG.family_id, (plan,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "qwen2_5_vl_7b" else "unloaded",
        vae_dtype=compute_dtype.name if role == "vae" else "unloaded",
        fp8_matmul=False,
        runtime_facts=plan.runtime_facts,
    )


__all__ = [
    "QWEN_IMAGE_PROVIDER_REVISION",
    "QwenImageComponentAssemblyError",
    "plan_qwen_image_official_component",
    "qwen_image_component_runtime_identity",
]
