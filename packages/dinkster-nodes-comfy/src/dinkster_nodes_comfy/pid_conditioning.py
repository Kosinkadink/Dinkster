"""Native PiD latent conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    LATENT_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

from .conditioning import conditioning_set_values

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


class PiDConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PiDConditioning",
            display_name="PiD Conditioning",
            category="model/conditioning",
            description=(
                "Attaches a latent and a degrade_sigma scalar to a CONDITIONING for PiD "
                "decoding/upscaling"
            ),
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec(
                    "latent",
                    _LATENT,
                    doc="latent (from VAEEncode or a KSampler).",
                ),
                InputSpec(
                    "latent_format",
                    _COMBO,
                    default="flux",
                    widget=ComboWidget(("flux", "sd3", "sdxl", "qwenimage")),
                    doc=(
                        "Flux1 (16-ch) and Flux2 (128-ch) latents are auto-detected from "
                        "channel dim under 'flux'. For SD3 (16-ch), SDXL (4-ch), or QwenImage "
                        "(16-ch), select manually."
                    ),
                ),
                InputSpec(
                    "degrade_sigma",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    doc="0 = clean latent. Increase to denoise corrupted latent outputs.",
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        latent: object,
        latent_format: str,
        degrade_sigma: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        samples = materialize_latent(latent)["samples"]
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")
        if latent_format == "flux":
            format_class = latent_formats.Flux2 if samples.shape[1] == 128 else latent_formats.Flux
        elif latent_format == "sd3":
            format_class = latent_formats.SD3
        elif latent_format == "sdxl":
            format_class = latent_formats.SDXL
        elif latent_format == "qwenimage":
            format_class = latent_formats.Wan21
        else:
            raise ValueError(f"Unknown latent_format: {latent_format}")
        lq_latent = format_class().process_in(samples)
        if lq_latent.ndim == 5:
            lq_latent = lq_latent[:, :, 0]
        torch = importlib.import_module("torch")
        return cls.outputs(
            CONDITIONING=conditioning_set_values(
                positive,
                {
                    "lq_latent": lq_latent,
                    "degrade_sigma": torch.tensor([float(degrade_sigma)], dtype=torch.float32),
                },
            )
        )
