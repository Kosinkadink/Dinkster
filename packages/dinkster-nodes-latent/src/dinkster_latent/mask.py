"""Native implementations of ComfyUI's core mask and compositing nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_INT,
    LATENT_TYPE,
    MASK_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
    with_identity_replacement,
)

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


_INT = TypeExpr.concrete(CORE_INT)


_LATENT = TypeExpr.concrete(LATENT_TYPE)


_MASK = TypeExpr.concrete(MASK_TYPE)


_MAX_RESOLUTION = 16384


def _integer(name: str, default: int, minimum: int, maximum: int, step: int) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _repeat_to_batch_size(tensor: Any, batch_size: int) -> Any:
    if tensor.shape[0] > batch_size:
        return tensor.narrow(0, 0, batch_size)
    if tensor.shape[0] < batch_size:
        return tensor.repeat(
            [int(-(-batch_size // tensor.shape[0]))] + [1] * (len(tensor.shape) - 1)
        ).narrow(0, 0, batch_size)
    return tensor


def composite(
    destination: Any,
    source: Any,
    x: int,
    y: int,
    mask: Any | None = None,
    multiplier: int = 8,
    resize_source: bool = False,
) -> Any:
    torch = importlib.import_module("torch")
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear"
        )
    source = _repeat_to_batch_size(source, destination.shape[0])
    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))
    left, top = x // multiplier, y // multiplier
    right, bottom = left + source.shape[-1], top + source.shape[-2]
    if mask is None:
        resolved_mask: Any = torch.ones_like(source)
    else:
        resolved_mask = materialize_image(mask).to(destination.device, copy=True)
        resolved_mask = torch.nn.functional.interpolate(
            resolved_mask.reshape((-1, 1, resolved_mask.shape[-2], resolved_mask.shape[-1])),
            size=(source.shape[-2], source.shape[-1]),
            mode="bilinear",
        )
        resolved_mask = _repeat_to_batch_size(resolved_mask, source.shape[0])
    visible_width = destination.shape[-1] - left + min(0, x)
    visible_height = destination.shape[-2] - top + min(0, y)
    resolved_mask = resolved_mask[:, :, :visible_height, :visible_width]
    if resolved_mask.ndim < source.ndim:
        resolved_mask = resolved_mask.unsqueeze(1)
    source_portion: Any = resolved_mask * source[..., :visible_height, :visible_width]
    destination_portion: Any = (torch.ones_like(resolved_mask) - resolved_mask) * destination[
        ..., top:bottom, left:right
    ]
    destination[..., top:bottom, left:right] = source_portion + destination_portion
    return destination


class LatentCompositeMasked(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.latent.latent_composite_masked",
                display_name="Latent Composite Masked",
                category="model/latent",
                inputs=(
                    InputSpec("destination", _LATENT),
                    InputSpec("source", _LATENT),
                    _integer("x", 0, 0, _MAX_RESOLUTION, 8),
                    _integer("y", 0, 0, _MAX_RESOLUTION, 8),
                    InputSpec("resize_source", _BOOLEAN, default=False),
                    InputSpec("mask", _MASK, required=False),
                ),
                outputs=(OutputSpec("LATENT", _LATENT),),
            ),
            "LatentCompositeMasked",
        )

    @classmethod
    def execute(
        cls,
        *,
        destination: dict[str, Any],
        source: dict[str, Any],
        x: int,
        y: int,
        resize_source: bool,
        mask: Any | None = None,
    ) -> Mapping[str, object]:
        destination = materialize_latent(destination)
        source = materialize_latent(source)
        output = destination.copy()
        output["samples"] = composite(
            destination["samples"].clone(), source["samples"], x, y, mask, 8, resize_source
        )
        return cls.outputs(LATENT=output)
