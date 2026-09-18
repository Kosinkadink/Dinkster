"""Synthetic full headers exercise production LTX detection, not artifact parity."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from dinkster_inference import FLOAT16, UINT8
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.gemma_text import (
    GEMMA3_LTX_12B_CONFIG,
    GEMMA4_LTX_12B_CONFIG,
    LTX_TEXT_CONNECTOR_CONFIG,
    GemmaTextConfig,
    gemma_text_layout,
    ltx_connector_layout,
    ltx_text_projection_layout,
)
from dinkster_inference.ltx_text_recipe import bind_ltx_text_recipe
from dinkster_inference.recipe import WeightSourceRef
from dinkster_inference.text_recipes import (
    DetectedTextSources,
    UnresolvedTextRecipe,
    resolve_text_recipe,
)
from dinkster_inference.weights import TensorGeometry, WeightEntry


@dataclass
class Header:
    layout: dict[str, tuple[int, ...]]
    path: Path
    asset_digest: str
    asset_size: int = 1234
    meta: dict[str, str] = field(default_factory=dict)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.layout)

    def entry(self, key: str) -> WeightEntry:
        dtype = UINT8 if key in ("spiece_model", "tokenizer_json") else FLOAT16
        geometry = TensorGeometry(self.layout[key], dtype)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return self.meta


def headers(kind: str, *, connectors: bool = False) -> tuple[Header, Header]:
    gemma4 = kind == "dual_linear_gemma4"
    config = GEMMA4_LTX_12B_CONFIG if gemma4 else GEMMA3_LTX_12B_CONFIG
    gemma = Header(
        {
            **{f"model.{key}": shape for key, shape in gemma_text_layout(config).items()},
            "tokenizer_json" if gemma4 else "spiece_model": (10,),
        },
        Path("/gemma.safetensors"),
        "blake3:" + "1" * 64,
    )
    projection = Header(
        {
            f"text_embedding_projection.{key}": shape
            for key, shape in ltx_text_projection_layout(
                "single_linear" if kind == "single_linear" else "dual_linear"
            ).items()
        },
        Path("/projection.safetensors"),
        "blake3:" + "2" * 64,
    )
    if gemma4:
        projection.meta = {
            "model_version": "2.4.0",
            "gemma_source_checkpoint": json.dumps(
                {"ltx_version": "2.4.0", "gemma_version": "gemma4-12b-ltx-v1"}
            ),
        }
    if connectors:
        projection.layout.update(
            {
                f"model.diffusion_model.{stream}_embeddings_connector.{key}": shape
                for stream in ("video", "audio")
                for key, shape in ltx_connector_layout(LTX_TEXT_CONNECTOR_CONFIG).items()
            }
        )
    return gemma, projection


def detect(inputs: tuple[Header, ...]) -> DetectedTextSources:
    registry = default_component_registry()
    return tuple(registry.detect(source, source.path) for source in inputs)


@pytest.mark.parametrize("kind", ("single_linear", "dual_linear", "dual_linear_gemma4"))
@pytest.mark.parametrize("reverse", (False, True))
def test_production_detection_retains_order_contract_and_prompt_identity(
    kind: str, reverse: bool
) -> None:
    inputs = headers(kind, connectors=kind == "single_linear")
    if reverse:
        inputs = tuple(reversed(inputs))
    binding = resolve_text_recipe(detect(inputs), "ltxv")
    assert [part.source_index for part in binding.components] == sorted(
        part.source_index for part in binding.components
    )
    assert next(p.plan.config for p in binding.components if p.role == "text_projection") == kind
    refs = tuple(WeightSourceRef(h.asset_digest, h.path.name, h.asset_size) for h in inputs)
    recipe = binding.recipe(refs, "float32")
    assert tuple(item.source for item in recipe.sources) == refs
    assert kind in repr(recipe.component_identity)
    assert all(part.profile is None for part in binding.components)
    assert "ltx_prompt_template=false" in recipe.knobs.runtime_facts
    assert "ltx_hidden_states=pre_layers_and_final_normalized" in recipe.knobs.runtime_facts
    assert resolve_text_recipe(detect(inputs), binding.id).recipe(refs, "float32") == recipe
    changed = replace(
        binding,
        components=tuple(
            replace(
                part, plan=replace(part.plan, config=replace(part.plan.config, min_tokens=2048))
            )
            if isinstance(part.plan.config, GemmaTextConfig)
            else part
            for part in binding.components
        ),
    )
    assert changed.recipe(refs, "float32").runtime_identity != recipe.runtime_identity
    reverse_binding = bind_ltx_text_recipe(detect(tuple(reversed(inputs))))
    assert reverse_binding.recipe(tuple(reversed(refs)), "float32").runtime_identity != (
        recipe.runtime_identity
    )


def test_rejects_missing_ambiguous_unused_and_mismatched_components() -> None:
    gemma, projection = headers("dual_linear")
    for inputs, message in (
        ((gemma,), "needs a text projection"),
        ((projection,), "needs one Gemma"),
        ((gemma, gemma, projection), "unambiguous"),
        ((*headers("dual_linear", connectors=True),), "discard detected components"),
        (headers("single_linear"), "needs its video/audio connectors"),
        ((headers("dual_linear_gemma4")[0], projection), "versions do not match"),
    ):
        with pytest.raises(UnresolvedTextRecipe, match=message):
            bind_ltx_text_recipe(detect(inputs))
    with pytest.raises(UnresolvedTextRecipe, match="discard an input source"):
        bind_ltx_text_recipe((*detect((gemma, projection)), ()))
    detected = detect((gemma, projection))
    projection_match = detected[1][0]
    unsupported = (
        detected[0],
        (
            replace(
                projection_match,
                components=tuple(
                    (role, replace(plan.component, config="unsupported"))
                    if role == "text_projection"
                    else (role, plan)
                    for role, plan in projection_match.components
                ),
            ),
        ),
    )
    with pytest.raises(UnresolvedTextRecipe, match="detected LTX projection contract"):
        bind_ltx_text_recipe(unsupported)
