"""Exact NewBie component detection and ordered recipe binding."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_inference import FLOAT32
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.gemma_text import (
    GEMMA3_NEWBIE_4B_CONFIG,
    GemmaTextDetectError,
    detect_gemma_text_config,
    gemma_text_layout,
)
from dinkster_inference.jina_clip_text import (
    JINA_CLIP_V2_CONFIG,
    detect_jina_clip_text_config,
    jina_clip_text_layout,
)
from dinkster_inference.text_recipes import UnresolvedTextRecipe, resolve_text_recipe
from dinkster_inference.weights import TensorGeometry, WeightEntry


class Header:
    asset_digest = "blake3:" + "7" * 64
    asset_size = 71

    def __init__(self, layout: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = {key: TensorGeometry(shape, FLOAT32) for key, shape in layout.items()}

    def keys(self) -> tuple[str, ...]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.shapes[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return {}

    def read_float_scalar(self, key: str) -> float:
        raise AssertionError(f"planner must not read model data: {key}")


def gemma_layout() -> dict[str, tuple[int, ...]]:
    return {
        f"model.{key}": shape for key, shape in gemma_text_layout(GEMMA3_NEWBIE_4B_CONFIG).items()
    }


def test_exact_newbie_layouts_reject_missing_and_wrong_variants() -> None:
    gemma = {
        key: TensorGeometry(shape, FLOAT32)
        for key, shape in gemma_text_layout(GEMMA3_NEWBIE_4B_CONFIG).items()
    }
    jina = {key: TensorGeometry(shape, FLOAT32) for key, shape in jina_clip_text_layout().items()}
    assert detect_gemma_text_config(gemma) == GEMMA3_NEWBIE_4B_CONFIG
    assert detect_jina_clip_text_config(jina) == JINA_CLIP_V2_CONFIG
    del gemma["layers.33.mlp.down_proj.weight"]
    with pytest.raises(GemmaTextDetectError):
        detect_gemma_text_config(gemma)
    del jina["model.encoder.layers.23.mlp.fc2.bias"]
    with pytest.raises(ValueError, match="not exact Jina"):
        detect_jina_clip_text_config(jina)


def test_production_registry_binds_both_orders_and_every_source() -> None:
    registry = default_component_registry()
    detected = tuple(
        registry.detect(source, Path(name))
        for source, name in (
            (Header(gemma_layout()), "gemma.safetensors"),
            (Header(jina_clip_text_layout()), "jina.safetensors"),
        )
    )
    binding = resolve_text_recipe(detected, "newbie")
    reverse = resolve_text_recipe(tuple(reversed(detected)), "dinkster.text_newbie")
    assert binding.family_id == "dinkster.newbie"
    assert binding.composition_roles == ("gemma", "jina")
    assert [(part.source_index, part.role) for part in binding.components] == [
        (0, "gemma"),
        (1, "jina"),
    ]
    assert [(part.source_index, part.role) for part in reverse.components] == [
        (1, "gemma"),
        (0, "jina"),
    ]
    assert all(
        "tokenizer_key=spiece_model" in part.plan.runtime_facts[-1] for part in binding.components
    )
    with pytest.raises(UnresolvedTextRecipe):
        resolve_text_recipe((detected[0], detected[0]), "newbie")
    with pytest.raises(UnresolvedTextRecipe):
        resolve_text_recipe((detected[0],), "newbie")
    with pytest.raises(UnresolvedTextRecipe):
        resolve_text_recipe((*detected, detected[1]), "newbie")
