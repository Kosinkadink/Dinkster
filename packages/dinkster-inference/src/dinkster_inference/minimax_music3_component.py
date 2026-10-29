"""MiniMax Music 3 split-component planning and identity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from dinkster_protocol import AttentionPolicy, AttentionRouteToken

from .assembly import (
    ComponentPlan,
    MiniMaxMusic3ComponentRole,
    plan_minimax_music3_component,
)
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource


class MiniMaxMusic3ComponentAssemblyError(ValueError):
    """A source is not the exact MiniMax Music 3 component requested."""


def plan_minimax_music3_split_component(
    source: WeightSource,
    *,
    role: MiniMaxMusic3ComponentRole,
    path: Path,
    bind_asset_identity: bool = True,
) -> ComponentPlan[object]:
    source_path = getattr(source, "path", None)
    if source_path != path:
        raise MiniMaxMusic3ComponentAssemblyError(f"MiniMax Music 3 {role} path differs")
    if not bind_asset_identity:
        return cast("ComponentPlan[object]", plan_minimax_music3_component(source, role))
    if not isinstance(source, AssetIdentifiedSource):
        raise MiniMaxMusic3ComponentAssemblyError(
            f"MiniMax Music 3 {role} source must carry asset identity"
        )
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise MiniMaxMusic3ComponentAssemblyError(
            f"MiniMax Music 3 {role} source must carry asset identity"
        )
    try:
        planned = plan_minimax_music3_component(source, role)
    except ValueError as error:
        raise MiniMaxMusic3ComponentAssemblyError(f"MiniMax Music 3 {role}: {error}") from error
    return cast(
        "ComponentPlan[object]",
        replace(
            planned,
            identity_facts=(
                *planned.identity_facts,
                f"asset_digest={digest}",
                f"asset_size={size}",
            ),
        ),
    )


def minimax_music3_component_runtime_identity(
    planned: ComponentPlan[object],
    role: MiniMaxMusic3ComponentRole,
    compute_dtype: DType,
    *,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> str:
    if planned.component != role:
        raise ValueError(f"MiniMax Music 3 {role} identity requires the {role} plan")
    return build_runtime_identity_from_facts(
        "dinkster.minimax_music3",
        runtime_component_identity("dinkster.minimax_music3", (planned,)),
        diffusion_dtype=compute_dtype.name if role == "diffusion" else "unloaded",
        text_dtype=compute_dtype.name if role == "text" else "unloaded",
        vae_dtype=compute_dtype.name if role == "vae" else "unloaded",
        fp8_matmul=False,
        attention_policy=attention_policy if role != "vae" else "auto",
        attention_route_token=attention_route_token if role != "vae" else None,
        runtime_facts=planned.runtime_facts,
    )


__all__ = [
    "MiniMaxMusic3ComponentAssemblyError",
    "minimax_music3_component_runtime_identity",
    "plan_minimax_music3_split_component",
]
