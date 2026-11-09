"""Independent public inference types and tensor checks for ordinary sampling hosts."""

from typing import Any

import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
    register_inference_types,
)
from test_standard_sampling_torch import _batch_noise, _conditioning, _latent, _model


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="standard-sampling-fixture.source",
            inputs=(),
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        model, _ = _model()
        return cls.outputs(model=model, conditioning=_conditioning(), latent=_latent())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="standard-sampling-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(LATENT_TYPE))
                for name in ("source", "standard", "advanced")
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, source: Any, standard: Any, advanced: Any) -> dict[str, object]:
        source, standard, advanced = map(materialize_latent, (source, standard, advanced))
        original = _latent()
        torch.testing.assert_close(source["samples"], original["samples"])
        torch.testing.assert_close(source["noise_mask"], original["noise_mask"])
        samples = original["samples"].repeat(2, 1, 1, 1)
        mask = original["noise_mask"]
        noisy = samples + _batch_noise(samples, 739) * 0.5 * mask
        expected = noisy * 0.25 * mask + samples * (1 - mask)
        torch.testing.assert_close(standard["samples"], expected)
        torch.testing.assert_close(advanced["samples"], original["samples"] * (1 - 0.375 * mask))
        for result in (standard, advanced):
            assert result["preserved"] == "metadata"
            assert tuple(result["batch_index"]) == (2, 0)
            torch.testing.assert_close(result["noise_mask"], mask)
            assert "downscale_ratio_spacial" not in result
            assert "downscale_ratio_temporal" not in result
        assert source["downscale_ratio_spacial"] == 8
        assert source["downscale_ratio_temporal"] == 1
        return cls.outputs(summary="sampling tensors and source metadata preserved")


FIXTURE_NODES = (Source, Inspect)

__all__ = ["FIXTURE_NODES", "register_inference_types"]
