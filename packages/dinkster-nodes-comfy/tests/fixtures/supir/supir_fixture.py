"""SUPIR ordinary-host tensor and lifetime checks with independent public types."""

from typing import Any

import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    register_inference_types,
)
from test_supir_torch import Base, Control, Vae, patcher


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="supir-fixture.source",
            inputs=(),
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("model", MODEL_TYPE),
                    ("weights", MODEL_PATCH_TYPE),
                    ("vae", VAE_TYPE),
                    ("image", IMAGE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            model=patcher(Base()),
            weights=patcher(Control(True)),
            vae=Vae(),
            image=torch.tensor([0.25, 0.5, 1.0, 0.9]).expand(1, 2, 3, 4).clone(),
        )


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="supir-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("source", MODEL_TYPE),
                    ("patched", MODEL_TYPE),
                    ("weights", MODEL_PATCH_TYPE),
                    ("vae", VAE_TYPE),
                    ("image", IMAGE_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, source: Any, patched: Any, weights: Any, vae: Any, image: Any
    ) -> dict[str, object]:
        image = materialize_image(image)
        assert patched is not source
        assert source.model_options == {"transformer_options": {}}
        assert not vae.patcher.patches
        patches = patched.model_options["transformer_options"]["patches"]
        middle = patches["middle_block_after_patch"][0]
        assert middle.__self__.models() == [weights]
        for sigma, expected in ((0.0, 11.5), (14.6146, 34.0)):
            value = torch.full((1, 1, 2, 3), 4.0)
            result = middle(
                {
                    "h": value,
                    "x": value,
                    "timesteps": torch.tensor([0.0]),
                    "context": None,
                    "y": None,
                    "transformer_options": {"sigmas": torch.tensor([sigma], dtype=torch.float64)},
                }
            )
            torch.testing.assert_close(result["h"], torch.full_like(value, expected))
        restore = patched.model_options["sampler_post_cfg_function"][0]
        denoised = torch.tensor([10.0, 20.0, 30.0]).reshape(3, 1, 1, 1).expand(3, 1, 2, 3)
        assert restore({"denoised": denoised, "sigma": torch.tensor(0.25)}) is denoised
        result = restore({"denoised": denoised, "sigma": torch.tensor([7.3073])})
        torch.testing.assert_close(result, (denoised + 103.0) / 2)
        assert torch.equal(denoised[:, 0, 0, 0], torch.tensor([10.0, 20.0, 30.0]))
        assert torch.equal(image, torch.tensor([0.25, 0.5, 1.0, 0.9]).expand(1, 2, 3, 4))
        return cls.outputs(summary="SUPIR control and restoration tensors preserved")


FIXTURE_NODES = (Source, Inspect)

__all__ = ["FIXTURE_NODES", "register_inference_types"]
