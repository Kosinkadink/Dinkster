"""Chroma Radiance model options applied over a sigma range."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)


class ChromaRadianceOptions(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ChromaRadianceOptions",
            category="model/patch/chroma radiance",
            description="Allows setting advanced options for the Chroma Radiance model.",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "preserve_wrapper",
                    _BOOLEAN,
                    default=True,
                    doc=(
                        "When enabled, delegates to an existing model function wrapper "
                        "if it exists. "
                        "Generally should be left enabled."
                    ),
                ),
                InputSpec(
                    "start_sigma",
                    _FLOAT,
                    default=1.0,
                    advanced=True,
                    doc="First sigma that these options will be in effect.",
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "end_sigma",
                    _FLOAT,
                    default=0.0,
                    advanced=True,
                    doc="Last sigma that these options will be in effect.",
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "nerf_tile_size",
                    _INT,
                    default=-1,
                    advanced=True,
                    doc=(
                        "Allows overriding the default NeRF tile size. -1 means use the "
                        "default (32). "
                        "0 means use non-tiling mode (may require a lot of VRAM)."
                    ),
                    widget=NumberWidget(min=-1, step=1),
                ),
                InputSpec(
                    "force_sequential_txt_ids",
                    _BOOLEAN,
                    default=False,
                    advanced=True,
                    doc=(
                        "Force usage of sequential text token IDs instead of zeroes. Should be "
                        "used for checkpoints from 2026-05-22 to 2026-06-01 that are trained "
                        "in this way but do "
                        "not contain the __sequential__ key in the state dict."
                    ),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        preserve_wrapper: bool,
        start_sigma: float,
        end_sigma: float,
        nerf_tile_size: int,
        force_sequential_txt_ids: bool,
    ) -> Mapping[str, object]:
        radiance_options: dict[str, object] = {}
        if nerf_tile_size >= 0:
            radiance_options["nerf_tile_size"] = nerf_tile_size
        if force_sequential_txt_ids:
            radiance_options["use_sequential_txt_ids"] = True
        if not radiance_options:
            return cls.outputs(MODEL=model)

        old_wrapper = model.model_options.get("model_function_wrapper")

        def model_function_wrapper(apply_model: Any, args: dict[str, Any]) -> Any:
            c = args["c"].copy()
            sigma = args["timestep"].max().detach().cpu().item()
            if end_sigma <= sigma <= start_sigma:
                transformer_options = c.get("transformer_options", {}).copy()
                transformer_options["chroma_radiance_options"] = radiance_options.copy()
                c["transformer_options"] = transformer_options
            if preserve_wrapper and old_wrapper:
                return old_wrapper(apply_model, args | {"c": c})
            return apply_model(args["input"], args["timestep"], **c)

        patched = model.clone()
        patched.set_model_unet_function_wrapper(model_function_wrapper)
        return cls.outputs(MODEL=patched)
