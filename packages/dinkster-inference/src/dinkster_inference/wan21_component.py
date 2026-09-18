"""Wan 2.1 per-component artifact planning and identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from .assembly import (
    WAN21_UMT5_PREFIX,
    AssemblyError,
    ComponentPlan,
    Wan21AssemblyPlan,
    Wan21StandaloneComponentPlan,
    Wan21StandaloneComponentRole,
    plan_wan21_standalone_component,
    plan_wan_diffusion_component,
    plan_wan_text_component,
    plan_wan_vae_component,
    plan_wan_vision_component,
)
from .catalog import WAN21
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .t5_text import T5Config
from .wan21_vae import WAN21_FLOW_RVS_VAE_CONFIG, WAN21_VAE_CONFIG
from .weights import AssetIdentifiedSource, WeightSource

if TYPE_CHECKING:
    from .component_checkpoint import ComponentCheckpointPlan


class Wan21ComponentAssemblyError(ValueError):
    """A source is not the exact Wan 2.1 component requested."""


@dataclass(frozen=True)
class WanCheckpointText:
    component: ComponentPlan[T5Config]
    tokenizer_source_key: str
    tokenizer_vendored: bool
    legacy_quantization_marker: str = ""

    @property
    def role(self) -> str:
        return "umt5xxl"

    @property
    def identity_components(self) -> tuple[ComponentPlan[T5Config], ...]:
        return (self.component,)

    @property
    def auxiliary_source_claims(self) -> tuple[tuple[Path, str], ...]:
        return tuple(
            (self.component.path, key)
            for key in (self.tokenizer_source_key, self.legacy_quantization_marker)
            if key
        )


def wan_checkpoint_assembly(plan: ComponentCheckpointPlan) -> Wan21AssemblyPlan:
    """Retain the selected tokenizer origin and image-conditioning requirement."""
    components = cast("Mapping[str, ComponentPlan[Any]]", plan.components)
    if plan.unclaimed:
        raise AssemblyError(
            "checkpoint: unsupported or duplicate Wan tensors: " + ", ".join(plan.unclaimed)
        )
    try:
        text = dict(plan.role_plans)["umt5xxl"]
        if not isinstance(text, WanCheckpointText):
            raise AssemblyError("umt5xxl: the selected text plan has no tokenizer origin")
        return Wan21AssemblyPlan(
            plan.family,
            components["diffusion"],
            components["umt5xxl"],
            components["vae"],
            text.tokenizer_source_key,
            clip_vision=components.get("clip_vision"),
            tokenizer_vendored=text.tokenizer_vendored,
        )
    except (KeyError, ValueError) as error:
        raise AssemblyError(f"incomplete or incompatible Wan checkpoint: {error}") from error


def _checkpoint_component(
    source: WeightSource, role: str
) -> ComponentPlan[Any] | WanCheckpointText:
    if role == "diffusion":
        return plan_wan_diffusion_component(family=WAN21, diffusion=source)
    if role == "umt5xxl":
        component, tokenizer_source_key, tokenizer_vendored = plan_wan_text_component(
            umt5xxl=source
        )
        legacy_markers = tuple(
            key for key in ("scaled_fp8", WAN21_UMT5_PREFIX + "scaled_fp8") if key in source.keys()
        )
        legacy_quantization_marker = (
            legacy_markers[0] if component.quant and len(legacy_markers) == 1 else ""
        )
        return WanCheckpointText(
            component,
            tokenizer_source_key,
            tokenizer_vendored,
            legacy_quantization_marker,
        )
    if role == "clip_vision":
        return plan_wan_vision_component(source)
    if role != "vae":
        raise AssemblyError(f"unknown Wan component role {role!r}")
    errors: list[AssemblyError] = []
    for config in (WAN21_VAE_CONFIG, WAN21_FLOW_RVS_VAE_CONFIG):
        try:
            return plan_wan_vae_component(config=config, vae=source)
        except AssemblyError as error:
            errors.append(error)
    raise AssemblyError("vae: " + "; ".join(map(str, errors))) from errors[-1]


@overload
def plan_wan21_split_component(
    source: WeightSource,
    *,
    role: Wan21StandaloneComponentRole,
    path: Path,
    bind_asset_identity: Literal[True] = True,
) -> Wan21StandaloneComponentPlan: ...


@overload
def plan_wan21_split_component(
    source: WeightSource,
    *,
    role: Wan21StandaloneComponentRole | Literal["clip_vision"],
    path: Path,
    bind_asset_identity: Literal[False],
) -> ComponentPlan[Any] | WanCheckpointText: ...


def plan_wan21_split_component(
    source: WeightSource,
    *,
    role: Wan21StandaloneComponentRole | Literal["clip_vision"],
    path: Path,
    bind_asset_identity: bool = True,
) -> Wan21StandaloneComponentPlan | ComponentPlan[Any] | WanCheckpointText:
    """Plan one split Wan component and bind its immutable asset identity."""

    source_path = getattr(source, "path", None)
    if source_path != path:
        raise Wan21ComponentAssemblyError(f"Wan 2.1 {role} source path differs from selection")
    if not bind_asset_identity:
        return _checkpoint_component(source, role)
    if role == "clip_vision":
        raise Wan21ComponentAssemblyError(
            "CLIP vision loads through its vision component constructor"
        )
    if not isinstance(source, AssetIdentifiedSource):
        raise Wan21ComponentAssemblyError(f"Wan 2.1 {role} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise Wan21ComponentAssemblyError(f"Wan 2.1 {role} source must carry asset identity")
    try:
        planned = plan_wan21_standalone_component(source, role)
    except ValueError as error:
        raise Wan21ComponentAssemblyError(f"Wan 2.1 {role}: {error}") from error
    component = replace(
        planned.component,
        identity_facts=(
            *planned.component.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )
    return replace(planned, component=component)


def wan21_component_runtime_identity(
    planned: Wan21StandaloneComponentPlan,
    compute_dtype: DType,
) -> str:
    """Build the native identity for one independently loaded Wan component."""

    role = planned.role
    component = planned.component
    return build_runtime_identity_from_facts(
        "dinkster.wan21",
        runtime_component_identity("dinkster.wan21", (component,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "umt5xxl" else "unloaded",
        vae_dtype=compute_dtype.name if role == "vae" else "unloaded",
        fp8_matmul=False,
        runtime_facts=component.runtime_facts,
    )


__all__ = [
    "Wan21ComponentAssemblyError",
    "plan_wan21_split_component",
    "wan21_component_runtime_identity",
]
