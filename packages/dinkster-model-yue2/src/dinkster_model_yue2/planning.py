"""Header-only YuE2 combined-checkpoint planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dinkster_inference import AssemblyError, ComponentPlan
from dinkster_inference.assembly import (
    _component_source,  # pyright: ignore[reportPrivateUsage]
    _plan,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference.component_checkpoint import ComponentCheckpointPlan

from .declarations import YUE2_COMPONENT, YuE2Detector


@dataclass(frozen=True)
class YuE2ComponentConfig:
    role: str


def _component(
    source: Any,
    role: str,
    prefix: str,
    *,
    drop: frozenset[str] = frozenset(),
) -> ComponentPlan[YuE2ComponentConfig]:
    extracted = _component_source(role, None, source, prefix)
    if not extracted.geometries:
        raise AssemblyError(f"YuE2 checkpoint has no {role} tensors")
    renames = None
    if role == "vae":
        renames = {
            key: (
                key.removesuffix(".weight_g") + ".parametrizations.weight.original0"
                if key.endswith(".weight_g")
                else key.removesuffix(".weight_v") + ".parametrizations.weight.original1"
            )
            for key in extracted.geometries
            if key.endswith((".weight_g", ".weight_v"))
        }
    return _plan(role, extracted, YuE2ComponentConfig(role), drop=drop, renames=renames)


def _plans(source: Any) -> tuple[tuple[str, ComponentPlan[YuE2ComponentConfig]], ...]:
    if YuE2Detector().detect(source) is None:
        return ()
    return (
        ("diffusion", _component(source, "diffusion", "model.diffusion_model.")),
        (
            "text",
            _component(
                source,
                "text",
                "text_encoders.",
                drop=frozenset({"yue2_tokenizer_json"}),
            ),
        ),
        ("vae", _component(source, "vae", "vae.")),
    )


def detect_components(
    source: Any,
    _path: object,
    **_options: object,
) -> tuple[tuple[str, object], ...]:
    return tuple((role, plan) for role, plan in _plans(source))


def plan_assembly(
    *,
    checkpoint: Any | None,
    diffusion: Any | None,
    **sources: object,
) -> ComponentCheckpointPlan:
    if diffusion is not None or any(value is not None for value in sources.values()):
        raise AssemblyError("YuE2 is distributed as one combined checkpoint")
    if checkpoint is None:
        raise AssemblyError("YuE2 requires a combined checkpoint")
    plans = _plans(checkpoint)
    if not plans:
        raise AssemblyError("checkpoint is not the exact YuE2 aggregate layout")
    return ComponentCheckpointPlan(YUE2_COMPONENT, plans)


__all__ = ["YuE2ComponentConfig", "detect_components", "plan_assembly"]
