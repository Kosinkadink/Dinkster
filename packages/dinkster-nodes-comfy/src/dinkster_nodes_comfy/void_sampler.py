"""VOID DDIM sampler node."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    SAMPLER_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class VOIDSampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VOIDSampler",
            category="model/sampling/samplers",
            inputs=(),
            outputs=(OutputSpec("SAMPLER", TypeExpr.concrete(SAMPLER_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        samplers = importlib.import_module("dinkster_inference.samplers")
        trange = importlib.import_module("dinkster_inference.utils").model_trange

        class VOID_DDIM(samplers.Sampler):
            def sample(
                self,
                model_wrap: Any,
                sigmas: Any,
                extra_args: dict[str, Any],
                callback: Any,
                noise: Any,
                latent_image: Any = None,
                denoise_mask: Any = None,
                disable_pbar: bool = False,
            ) -> Any:
                del latent_image, denoise_mask
                x = noise.to(torch.float32)
                model_options = extra_args.get("model_options", {})
                seed = extra_args.get("seed")
                s_in = x.new_ones([x.shape[0]])

                for index in trange(len(sigmas) - 1, disable=disable_pbar):
                    sigma = sigmas[index]
                    sigma_next = sigmas[index + 1]
                    denoised = model_wrap(x, sigma * s_in, model_options=model_options, seed=seed)

                    if callback is not None:
                        callback(index, denoised, x, len(sigmas) - 1)

                    if sigma_next == 0:
                        x = denoised
                    else:
                        alpha_t = 1.0 / (1.0 + sigma**2)
                        alpha_prev = 1.0 / (1.0 + sigma_next**2)
                        pred_eps = (x - alpha_t**0.5 * denoised) / (1.0 - alpha_t) ** 0.5
                        x = alpha_prev**0.5 * denoised + (1.0 - alpha_prev) ** 0.5 * pred_eps

                return x

        return cls.outputs(SAMPLER=VOID_DDIM())
