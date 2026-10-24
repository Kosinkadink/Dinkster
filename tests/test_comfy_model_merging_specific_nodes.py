from __future__ import annotations

from typing import cast

import pytest
from dinkster_nodes_comfy.model_merging_specific import (
    MODEL_SPECIFIC_MERGE_NODES,
    ModelMergeSD1,
    ModelMergeSD2,
    ModelMergeSDXL,
)


class _Model:
    def __init__(self, patches: dict[str, float], weights: dict[str, float] | None = None) -> None:
        self.patches = patches
        self.weights = {} if weights is None else weights

    def clone(self) -> _Model:
        return _Model({}, self.weights.copy())

    def get_key_patches(self, prefix: str) -> dict[str, float]:
        assert prefix == "diffusion_model."
        return self.patches

    def add_patches(self, patches: dict[str, float], first: float, second: float) -> None:
        for key, patch in patches.items():
            self.weights[key] = self.weights[key] * first + patch * second


def test_specific_model_merge_schemas_have_stable_public_contracts() -> None:
    expected = (
        ("ModelMergeSD1", 30),
        ("ModelMergeSD2", 30),
        ("ModelMergeSDXL", 24),
        ("ModelMergeSD3_2B", 30),
        ("ModelMergeAuraflow", 43),
        ("ModelMergeFlux1", 63),
        ("ModelMergeSD35_Large", 44),
        ("ModelMergeMochiPreview", 53),
        ("ModelMergeLTXV", 33),
        ("ModelMergeCosmos7B", 34),
        ("ModelMergeCosmos14B", 42),
        ("ModelMergeWAN2_1", 46),
        ("ModelMergeCosmosPredict2_2B", 33),
        ("ModelMergeCosmosPredict2_14B", 41),
        ("ModelMergeQwenImage", 66),
        ("ModelMergeKrea2", 38),
    )
    assert (
        tuple((node.__name__, len(node.schema().inputs) - 2) for node in MODEL_SPECIFIC_MERGE_NODES)
        == expected
    )
    for node in MODEL_SPECIFIC_MERGE_NODES:
        schema = node.schema()
        assert schema.node_type == node.__name__
        assert schema.category == "model/merging/model specific"
        assert tuple(input_spec.id for input_spec in schema.inputs[:2]) == ("model1", "model2")

    assert MODEL_SPECIFIC_MERGE_NODES[11].schema().description == (
        "1.3B model has 30 blocks, 14B model has 40 blocks. "
        "Image to video model has the extra img_emb."
    )


def test_sd2_has_its_own_node_identity_with_sd1_coefficients() -> None:
    assert ModelMergeSD2 is not ModelMergeSD1
    assert ModelMergeSD2.schema().node_type == "ModelMergeSD2"
    assert ModelMergeSD2.schema().inputs == ModelMergeSD1.schema().inputs


def test_specific_merge_uses_longest_prefix_and_the_first_ratio_by_default() -> None:
    source = _Model(
        {
            "diffusion_model.input_blocks.1.weight": 100.0,
            "diffusion_model.input_blocks.10.weight": 200.0,
            "diffusion_model.elsewhere.weight": 300.0,
        }
    )
    merged = cast(
        _Model,
        ModelMergeSDXL.execute(
            model1=_Model(
                {},
                {
                    "diffusion_model.input_blocks.1.weight": 10.0,
                    "diffusion_model.input_blocks.10.weight": 20.0,
                    "diffusion_model.elsewhere.weight": 30.0,
                },
            ),
            model2=source,
            **{"time_embed.": 0.2, "input_blocks.1": 0.4, "input_blocks.10": 0.8},
        )["MODEL"],
    )
    assert merged.weights == pytest.approx(
        {
            "diffusion_model.input_blocks.1.weight": 46.0,
            "diffusion_model.input_blocks.10.weight": 164.0,
            "diffusion_model.elsewhere.weight": 84.0,
        }
    )


@pytest.mark.parametrize("node", MODEL_SPECIFIC_MERGE_NODES)
def test_all_specific_merge_nodes_use_model_specific_category(node: type[ModelMergeSD1]) -> None:
    assert node.schema().category == "model/merging/model specific"
