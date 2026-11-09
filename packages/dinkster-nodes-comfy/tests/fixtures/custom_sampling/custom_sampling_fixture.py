"""Independent custom-sampling values and tensor expectations."""

import importlib
from typing import Any

import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    GUIDER_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
    register_inference_types,
)
from dinkster_nodes_comfy.noise import RandomNoise
from test_standard_sampling_torch import _TinyModel


class Model(_TinyModel):
    def extra_conds(self, **kwargs: Any) -> dict[str, Any]:
        conds = importlib.import_module("dinkster_inference.conds")
        return {"c_crossattn": conds.CONDCrossAttn(kwargs["cross_attn"])}

    def apply_model(
        self, samples: torch.Tensor, _sigma: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        condition = kwargs["c_crossattn"].mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        return samples * 0.25 + condition

    def process_latent_out(self, samples: torch.Tensor) -> torch.Tensor:
        return samples * 3


def latent() -> dict[str, Any]:
    return {
        "samples": torch.tensor([[[[2.0, -4.0]]], [[[8.0, 1.0]]]]),
        "batch_index": [6, 2],
        "noise_mask": torch.tensor([[[[1.0, 0.0]]], [[[0.25, 0.75]]]]),
        "origin": "fixture",
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 1,
    }


class CustomSamplingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="custom-sampling-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("model", MODEL_TYPE),
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("sampler", SAMPLER_TYPE),
                    ("sigmas", SIGMAS_TYPE),
                    ("latent", LATENT_TYPE),
                    ("noise", NOISE_TYPE),
                    ("guider", GUIDER_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        samplers = importlib.import_module("dinkster_inference.samplers")
        patcher_type = importlib.import_module("dinkster_inference.model_patcher").ModelPatcher
        patcher = patcher_type(Model(), torch.device("cpu"), torch.device("cpu"))
        positive = [[torch.full((1, 1, 1), 2.0), {}]]
        negative = [[torch.full((1, 1, 1), -1.0), {}]]
        guider = samplers.CFGGuider(patcher)
        guider.set_conds(positive, negative)
        guider.set_cfg(2.0)
        return cls.outputs(
            model=patcher,
            positive=positive,
            negative=negative,
            sampler=samplers.ksampler("euler"),
            sigmas=torch.tensor([1.0, 0.5]),
            latent=latent(),
            noise=RandomNoise.execute(noise_seed=41)["NOISE"],
            guider=guider,
        )


class CustomSamplingConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="custom-sampling-fixture.consumer",
            inputs=(
                InputSpec("output", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("denoised_output", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("source", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("add_noise", TypeExpr.concrete(CORE_BOOLEAN)),
            ),
            outputs=(OutputSpec("sampled", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(
        cls, *, output: Any, denoised_output: Any, source: Any, add_noise: bool
    ) -> dict[str, object]:
        output, denoised_output, source = map(materialize_latent, (output, denoised_output, source))
        original = latent()
        samples, mask = original["samples"], original["noise_mask"]
        generator = torch.Generator(device="cpu").manual_seed(41)
        draws = [torch.randn((1, 1, 1, 2), generator=generator) for _ in range(7)]
        noise = torch.cat((draws[6], draws[2])) if add_noise else torch.zeros_like(samples)
        initial = samples + noise
        denoiser_input = initial * mask + samples * (1 - mask)
        # CFG 2 combines positive 2 and negative -1 into a bias of 5.
        denoised = (denoiser_input * 0.25 + 5) * mask + samples * (1 - mask)
        expected = (initial * 0.5 + denoised * 0.5) * 3
        torch.testing.assert_close(output["samples"], expected)
        torch.testing.assert_close(denoised_output["samples"], denoised * 3)
        assert not torch.equal(output["samples"], denoised_output["samples"])
        for result in (output, denoised_output, source):
            assert result["batch_index"] == [6, 2]
            assert result["origin"] == "fixture"
            assert torch.equal(result["noise_mask"], mask)
        assert "downscale_ratio_spacial" not in output
        assert "downscale_ratio_temporal" not in output
        assert denoised_output["downscale_ratio_spacial"] == source["downscale_ratio_spacial"] == 8
        assert (
            denoised_output["downscale_ratio_temporal"] == source["downscale_ratio_temporal"] == 1
        )
        assert torch.equal(source["samples"], samples)
        return cls.outputs(sampled=True)


NODES = (CustomSamplingSource, CustomSamplingConsumer)
__all__ = ["NODES", "register_inference_types"]
