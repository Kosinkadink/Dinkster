"""Load registered control components from their unchanged header plans."""

from __future__ import annotations

from typing import cast

import torch
from dinkster_inference import AttentionPolicy, AttentionRouteToken
from dinkster_inference.assembly import (
    ControlNetAssemblyPlan,
    SDXLControlLoRAAssemblyPlan,
    SDXLControlNetAssemblyPlan,
    SDXLControlNetUnionAssemblyPlan,
    T2IAdapterAssemblyPlan,
)
from dinkster_inference.control_components import ControlComponentPlan
from dinkster_inference.controlnet import ControlNetSourceLayout, sd15_controlnet_layout

from .assemble import (
    AssembledSD,
    assemble_sd15_controlnet,
    assemble_sd15_t2i_adapter,
    assemble_sdxl_control_lora,
    assemble_sdxl_controlnet,
    assemble_sdxl_controlnet_union,
)
from .attention import AttentionRole
from .controlnet import SD15ControlNet, SDXLControlLoRA, SDXLControlNet, SDXLControlNetUnion
from .t2i_adapter import SD15T2IAdapter


def _asset_digest(plan: ControlComponentPlan) -> str:
    if plan.asset_digest is None:
        raise ValueError("control loading requires an asset-identified plan")
    return plan.asset_digest


def load_sd15_controlnet(plan: ControlComponentPlan, dtype: torch.dtype) -> SD15ControlNet:
    assembled = assemble_sd15_controlnet(
        ControlNetAssemblyPlan(
            plan.component,
            sd15_controlnet_layout(plan.component.config),
            cast("ControlNetSourceLayout", plan.source_layout),
            _asset_digest(plan),
            tuple(sorted(plan.component.keys.values())),
        ),
        controlnet_dtype=dtype,
    )
    return assembled.controlnet


def load_sd15_t2i_adapter(plan: ControlComponentPlan, dtype: torch.dtype) -> SD15T2IAdapter:
    return assemble_sd15_t2i_adapter(
        T2IAdapterAssemblyPlan(
            plan.component, _asset_digest(plan), tuple(sorted(plan.component.keys.values()))
        ),
        adapter_dtype=dtype,
    ).adapter


def load_sdxl_controlnet(plan: ControlComponentPlan, dtype: torch.dtype) -> SDXLControlNet:
    return assemble_sdxl_controlnet(
        SDXLControlNetAssemblyPlan(
            plan.component, _asset_digest(plan), tuple(sorted(plan.component.keys.values()))
        ),
        controlnet_dtype=dtype,
    ).controlnet


def load_sdxl_controlnet_union(
    plan: ControlComponentPlan,
    dtype: torch.dtype,
    *,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    attention_backend: AttentionRole,
) -> SDXLControlNetUnion:
    return assemble_sdxl_controlnet_union(
        SDXLControlNetUnionAssemblyPlan(
            plan.component, _asset_digest(plan), tuple(sorted(plan.component.keys.values()))
        ),
        controlnet_dtype=dtype,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
        attention_backend=attention_backend,
    ).controlnet_union


def load_sdxl_control_lora(
    plan: ControlComponentPlan,
    dtype: torch.dtype,
    *,
    base: AssembledSD,
    base_asset_digest: str,
) -> SDXLControlLoRA:
    return assemble_sdxl_control_lora(
        SDXLControlLoRAAssemblyPlan(
            plan.component,
            _asset_digest(plan),
            base_asset_digest,
            tuple(sorted(plan.component.keys.values())),
        ),
        base,
        control_lora_dtype=dtype,
    ).control_lora
