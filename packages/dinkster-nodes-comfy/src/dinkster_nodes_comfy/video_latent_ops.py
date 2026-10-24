"""Video latent transform nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)


class TrimVideoLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TrimVideoLatent",
            display_name="Trim Video Latent",
            category="model/latent",
            inputs=(
                InputSpec("samples", _LATENT),
                InputSpec(
                    "trim_amount",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=99999),
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], trim_amount: int) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        result = samples.copy()
        result["samples"] = samples["samples"][:, :, trim_amount:]
        return cls.outputs(LATENT=result)


class LTXVFreezeLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVFreezeLatent",
            display_name="LTXV Freeze Latent",
            category="model/latent/ltxv",
            inputs=(InputSpec("latent", _LATENT),),
            outputs=(OutputSpec("latent", _LATENT),),
        )

    @classmethod
    def execute(cls, *, latent: dict[str, Any]) -> Mapping[str, object]:
        latent = materialize_latent(latent)
        samples = latent["samples"]
        torch = importlib.import_module("torch")
        if not torch.is_tensor(samples):
            raise ValueError(
                "Freeze Latent expects a plain tensor, not a concatenated AV latent. "
                "Split with Separate AV Latent first."
            )
        result = latent.copy()
        if samples.ndim == 5:
            batch, _, frames, _, _ = samples.shape
            shape = (batch, 1, frames, 1, 1)
        elif samples.ndim == 4:
            batch, _, frames, _ = samples.shape
            shape = (batch, 1, frames, 1)
        else:
            raise ValueError(
                f"Expected a 4D audio or 5D video latent, got shape {list(samples.shape)}."
            )
        result["noise_mask"] = torch.zeros(shape, dtype=torch.float32, device=samples.device)
        return cls.outputs(latent=result)


VIDEO_LATENT_OPERATION_NODES = (TrimVideoLatent, LTXVFreezeLatent)
