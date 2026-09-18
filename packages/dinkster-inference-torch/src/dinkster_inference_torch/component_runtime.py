"""Architecture-specific construction behind generic component handles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from dinkster_assets import AssetRef
from dinkster_inference import AttentionPolicy, AttentionRouteToken
from dinkster_inference.catalog import WAN21
from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
from dinkster_inference.component_registry import execution_symbol
from dinkster_inference.minimax_h3_assembly import MiniMaxH3CommonComponentRole

from .chroma_runtime import ChromaDiffusionRuntime
from .minimax_h3_assembly import (
    MiniMaxH3Model,
    load_minimax_h3_component,
    load_minimax_h3_model,
)
from .trellis2_assembly import AssembledTrellis2
from .trellis2_runtime import Trellis2DiffusionRuntime
from .wan21_causal import Wan21CausalModel
from .wan21_runtime import Wan21CausalDiffusionRuntime, Wan21DiffusionRuntime


def load_component_checkpoint(plan: ComponentCheckpointPlan, **options: Any) -> Any:
    """Pass the selected role plans and resolved runtime policy to their declared constructor."""
    reference = plan.descriptor.checkpoint_loader
    if reference is None:
        raise ValueError(f"{plan.descriptor.id} has no checkpoint composition loader")
    return execution_symbol(reference)(plan, **options)


def load_h3_component(
    path: Path,
    *,
    asset: AssetRef,
    expected_role: str,
    expected_identity: str,
    compute_dtype: torch.dtype,
    artifact_role: str | None = None,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> Any:
    if expected_role != "diffusion":
        return load_minimax_h3_component(
            path,
            asset=asset,
            expected_role=cast("MiniMaxH3CommonComponentRole", expected_role),
            expected_identity=expected_identity,
            compute_dtype=compute_dtype,
        )
    if artifact_role not in ("fl2va-dit", "ref2va-dit"):
        raise ValueError("MiniMax H3 diffusion loading requires an explicit DiT role")
    runtime = load_minimax_h3_model(
        path,
        asset=asset,
        role=artifact_role,
        expected_identity=expected_identity,
        diffusion_dtype=compute_dtype,
        attention_policy=attention_policy if attention_route_token is not None else "auto",
        attention_route_token=attention_route_token,
    )
    return SimpleNamespace(role="diffusion", module=runtime.assembled.diffusion, runtime=runtime)


def h3_runtime(loaded: Any, identity: str, dtype: torch.dtype) -> MiniMaxH3Model:
    del dtype
    return replace(loaded.runtime, runtime_identity=identity)


def chroma_runtime(
    loaded: Any, identity: str, dtype: torch.dtype, **options: Any
) -> ChromaDiffusionRuntime:
    status = loaded.attention_status
    return ChromaDiffusionRuntime(
        loaded.module,
        runtime_identity=identity,
        compute_dtype=dtype,
        attention_status=status if isinstance(status, Mapping) else {"flux": status},
        **options,
    )


def wan_runtime(
    loaded: Any, identity: str, dtype: torch.dtype
) -> Wan21DiffusionRuntime | Wan21CausalDiffusionRuntime:
    runtime_type = (
        Wan21CausalDiffusionRuntime
        if isinstance(loaded.module, Wan21CausalModel)
        else Wan21DiffusionRuntime
    )
    return runtime_type(
        loaded.module,
        WAN21,
        runtime_identity=identity,
        compute_dtype=dtype,
    )


def trellis_runtime(loaded: Any, identity: str, dtype: torch.dtype) -> Trellis2DiffusionRuntime:
    return Trellis2DiffusionRuntime(
        AssembledTrellis2(loaded.module, loaded.plan),
        runtime_identity=identity,
        compute_dtype=dtype,
    )
