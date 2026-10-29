"""LTX-2 per-component artifact planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from dinkster_protocol import AttentionPolicy, AttentionRouteToken

from .assembly import (
    ComponentPlan,
    LTXAVAudioCodecPlan,
    LTXAVStandaloneComponentPlan,
    LTXAVStandaloneComponentRole,
    plan_ltxav_audio_codec,
    plan_ltxav_standalone_component,
)
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource


class LTXAVAudioCodecAssemblyError(ValueError):
    """A source is not an exact supported LTX-2 audio codec."""


class LTXAVComponentAssemblyError(ValueError):
    """A source is not the exact LTX-2 component requested."""


def _require_asset_source(
    source: WeightSource,
    path: Path,
    component: str,
) -> tuple[str, int]:
    if getattr(source, "path", None) != path:
        raise LTXAVComponentAssemblyError(f"LTX-2 {component} source path differs from selection")
    if not isinstance(source, AssetIdentifiedSource):
        raise LTXAVComponentAssemblyError(f"LTX-2 {component} source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise LTXAVComponentAssemblyError(f"LTX-2 {component} source must carry asset identity")
    return digest, size


def plan_ltxav_split_component(
    source: WeightSource,
    *,
    role: LTXAVStandaloneComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> LTXAVStandaloneComponentPlan:
    """Plan one component and bind its immutable asset identity."""
    if not bind_asset_identity:
        if getattr(source, "path", None) != path:
            raise LTXAVComponentAssemblyError(f"LTX-2 {role} source path differs from selection")
        return plan_ltxav_standalone_component(source, role)
    digest, size = _require_asset_source(source, path, role)
    try:
        planned = plan_ltxav_standalone_component(source, role)
    except ValueError as error:
        raise LTXAVComponentAssemblyError(f"LTX-2 {role}: {error}") from error
    component = replace(
        planned.component,
        identity_facts=(
            *planned.component.identity_facts,
            f"asset_digest={digest}",
            f"asset_size={size}",
        ),
    )
    return replace(planned, component=component)


def ltxav_component_uses_fp8_matmul(
    role: LTXAVStandaloneComponentRole,
    component: ComponentPlan[Any],
) -> bool:
    """Whether this component's checkpoint selects native FP8 matmul."""
    return role == "diffusion" and any(
        quant.format == "float8_e4m3fn" and not quant.full_precision_matmul
        for quant in component.quant.values()
    )


def ltxav_component_runtime_identity(
    planned: LTXAVStandaloneComponentPlan,
    compute_dtype: DType,
    *,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> str:
    """Build the native identity for one independently loaded component."""
    role = planned.role
    component = planned.component
    has_attention = role in ("diffusion", "gemma3_12b", "gemma4_12b", "connectors")
    return build_runtime_identity_from_facts(
        "dinkster.ltxav",
        runtime_component_identity("dinkster.ltxav", (component,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=(
            compute_dtype.name
            if role in ("gemma3_12b", "gemma4_12b", "text_projection", "connectors")
            else "unloaded"
        ),
        vae_dtype=compute_dtype.name if role in ("latent_upscaler", "vae") else "unloaded",
        fp8_matmul=ltxav_component_uses_fp8_matmul(role, component),
        attention_policy=attention_policy if has_attention else "auto",
        attention_route_token=attention_route_token if has_attention else None,
        runtime_facts=component.runtime_facts,
    )


def plan_ltxav_split_audio_codec(
    source: WeightSource,
    *,
    path: Path,
) -> LTXAVAudioCodecPlan:
    """Plan one audio codec and bind its immutable asset identity."""
    source_path = getattr(source, "path", None)
    if source_path != path:
        raise LTXAVAudioCodecAssemblyError("LTX-2 audio codec source path differs from selection")
    if not isinstance(source, AssetIdentifiedSource):
        raise LTXAVAudioCodecAssemblyError("LTX-2 audio codec source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise LTXAVAudioCodecAssemblyError("LTX-2 audio codec source must carry asset identity")
    try:
        planned = plan_ltxav_audio_codec(source)
    except ValueError as error:
        raise LTXAVAudioCodecAssemblyError(f"LTX-2 audio codec: {error}") from error
    identity_facts = (f"asset_digest={digest}", f"asset_size={size}")
    return replace(
        planned,
        audio_vae=replace(
            planned.audio_vae,
            identity_facts=(*planned.audio_vae.identity_facts, *identity_facts),
        ),
        vocoder=replace(
            planned.vocoder,
            identity_facts=(*planned.vocoder.identity_facts, *identity_facts),
        ),
    )


def ltxav_audio_codec_runtime_identity(planned: LTXAVAudioCodecPlan) -> str:
    """Build the native identity for one independently loaded audio codec."""
    runtime_facts = tuple(
        sorted(
            {fact for component in planned.identity_components for fact in component.runtime_facts}
        )
    )
    return build_runtime_identity_from_facts(
        "dinkster.ltxav",
        runtime_component_identity("dinkster.ltxav", planned.identity_components),
        diffusion_dtype="unloaded",
        text_dtype="unloaded",
        vae_dtype="float32",
        fp8_matmul=False,
        runtime_facts=runtime_facts,
    )


__all__ = [
    "LTXAVAudioCodecAssemblyError",
    "LTXAVComponentAssemblyError",
    "ltxav_component_runtime_identity",
    "ltxav_audio_codec_runtime_identity",
    "plan_ltxav_split_component",
    "plan_ltxav_split_audio_codec",
]
