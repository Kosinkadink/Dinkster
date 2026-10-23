from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from dinkster_nodes_comfy.model_merging_specific import (
    MODEL_SPECIFIC_MERGE_NODES,
    ModelMergeSD1,
    ModelMergeSD2,
    ModelMergeSDXL,
)


class _Model:
    def __init__(self, patches: dict[str, object]) -> None:
        self.patches = patches
        self.added: list[tuple[str, float, float]] = []

    def clone(self) -> _Model:
        return _Model({})

    def get_key_patches(self, prefix: str) -> dict[str, object]:
        assert prefix == "diffusion_model."
        return self.patches

    def add_patches(self, patches: dict[str, object], first: float, second: float) -> None:
        self.added.extend((key, first, second) for key in patches)


def test_specific_model_merge_schemas_match_the_fixed_reference() -> None:
    reference = json.loads(
        (Path(__file__).parents[4] / ".amp/in/comfy-nodes-511/contract-reference.json").read_text()
    )
    expected = {
        item["class_name"]: item
        for item in reference["nodes"]
        if item["file"] == "comfy_extras/nodes_model_merging_model_specific.py"
    }
    assert tuple(node.__name__ for node in MODEL_SPECIFIC_MERGE_NODES) == tuple(expected)
    for node in MODEL_SPECIFIC_MERGE_NODES:
        schema = node.schema()
        inputs = expected[node.__name__]["input_types"]["required"]
        assert schema.node_type == node.__name__
        assert schema.category == expected[node.__name__]["category"]
        assert tuple(input_spec.id for input_spec in schema.inputs) == tuple(inputs)

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
            "diffusion_model.input_blocks.1.weight": object(),
            "diffusion_model.input_blocks.10.weight": object(),
            "diffusion_model.elsewhere.weight": object(),
        }
    )
    merged = cast(
        _Model,
        ModelMergeSDXL.execute(
            model1=_Model({}),
            model2=source,
            **{"time_embed.": 0.2, "input_blocks.1": 0.4, "input_blocks.10": 0.8},
        )["MODEL"],
    )
    assert [item[0] for item in merged.added] == [
        "diffusion_model.input_blocks.1.weight",
        "diffusion_model.input_blocks.10.weight",
        "diffusion_model.elsewhere.weight",
    ]
    for actual, expected in zip(
        [item[1:] for item in merged.added],
        ((0.6, 0.4), (0.2, 0.8), (0.8, 0.2)),
        strict=True,
    ):
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("node", MODEL_SPECIFIC_MERGE_NODES)
def test_all_specific_merge_nodes_use_model_specific_category(node: type[ModelMergeSD1]) -> None:
    assert node.schema().category == "model/merging/model specific"
