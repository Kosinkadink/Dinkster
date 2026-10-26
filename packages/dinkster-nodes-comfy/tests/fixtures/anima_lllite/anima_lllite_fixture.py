"""Small model seams and independent inference types for ordinary hosts."""

from typing import Any

import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
    register_inference_types,
)
from test_anima_lllite_torch import _Base, _Conditioning, _patcher


class AnimaSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="anima-lllite-fixture.source",
            inputs=(),
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            model=_patcher(_Base()),
            patch=_patcher(_Conditioning(4)),
            image=torch.tensor([0.125, 0.5, 1.0, 0.875]).expand(1, 8, 16, 4).clone(),
            mask=torch.ones((1, 8, 16)),
        )


class AnimaInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="anima-lllite-fixture.inspect",
            inputs=(
                InputSpec("source", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("patched", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("weights", TypeExpr.concrete(MODEL_PATCH_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, source: Any, patched: Any, weights: Any, image: Any, mask: Any
    ) -> dict[str, object]:
        image = materialize_image(image)
        mask = materialize_image(mask)
        assert patched is not source
        assert source.model_options.get("transformer_options", {}).get("patches", {}) == {}
        patches = patched.model_options["transformer_options"]["patches"]
        post = patches["post_input"][0]
        assert post.models() == [weights]
        q, k, v = (torch.tensor([[[value]]]) for value in (1.0, 10.0, 100.0))
        for sigma in (10 * (1 - 0.2), 10 * (1 - 0.7)):
            options = {
                "sigmas": torch.tensor([sigma], dtype=torch.float64),
                "model_patch_data": {},
                "block_index": 4,
            }
            post({"x": torch.zeros((1, 4, 1, 1, 2)), "transformer_options": options})
            expected = torch.tensor([-0.75, 0.0, 1.0, 1.0]).view(1, 4, 1, 1)
            assert torch.equal(options["model_patch_data"][post], expected.expand(1, 4, 8, 16))
            attention = patches["attn1_patch"][0](q, k, v, extra_options=options)
            assert [attention[key].item() for key in ("q", "k", "v")] == [4.0, 14.5, 107.5]
            cross = patches["attn2_patch"][0](q, k, v, extra_options=options)
            assert [cross[key].item() for key in ("q", "k", "v")] == [11.5, 10.0, 100.0]
            assert (
                patches["mlp_patch"][0]({"x": q, "transformer_options": options})["x"].item()
                == 17.5
            )
        assert torch.equal(q, torch.tensor([[[1.0]]]))
        assert torch.equal(image, torch.tensor([0.125, 0.5, 1.0, 0.875]).expand(1, 8, 16, 4))
        assert torch.equal(mask, torch.ones((1, 8, 16)))
        return cls.outputs(summary="anima patch values and sources preserved")


ANIMA_FIXTURE_NODES = (AnimaSource, AnimaInspect)

__all__ = ["ANIMA_FIXTURE_NODES", "register_inference_types"]
