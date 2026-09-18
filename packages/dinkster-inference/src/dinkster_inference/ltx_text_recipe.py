"""Ordered Gemma and projection bindings for LTX audio/video text encoding."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from .component_registry import component_plans
from .gemma_text import GemmaTextConfig, LtxConnectorConfig
from .text_recipes import (
    DetectedTextSources,
    TextRecipeBinding,
    TextRecipeComponent,
    TextRecipeDescriptor,
    UnresolvedTextRecipe,
)

LTX_GEMMA_TOKENIZER_KEYS = MappingProxyType(
    {"gemma3_12b": "spiece_model", "gemma4_12b": "tokenizer_json"}
)

# Packing ids, minimum length and model normalization are retained in GemmaTextConfig.
_ENCODING_FACTS = (
    "ltx_prompt_template=false",
    "ltx_prompt_weights=disabled",
    "ltx_prompt_padding=left",
    "ltx_hidden_states=pre_layers_and_final_normalized",
    "ltx_projection_tokens=attended_suffix",
    "ltx_gemma_quantized_matmul=float32",
    "ltx_output_dtype=float32",
)


def bind_ltx_text_recipe(sources: DetectedTextSources) -> TextRecipeBinding:
    """Consume every detected component without reordering the submitted sources."""
    parts: list[TextRecipeComponent] = []
    for source_index, matches in enumerate(sources):
        for match in matches:
            if match.descriptor.family.id != "dinkster.ltxav":
                raise UnresolvedTextRecipe(
                    "LTX text recipe needs LTX audio-video Gemma and projection components"
                )
            for role, planned in match.components:
                for plan in component_plans(planned):
                    part = TextRecipeComponent(source_index, role, plan, None)
                    if part not in parts:
                        parts.append(part)
    roles = [part.role for part in parts]
    if len(set(roles)) != len(roles):
        raise UnresolvedTextRecipe("LTX text recipe requires unambiguous component roles")
    gemmas = [part for part in parts if isinstance(part.plan.config, GemmaTextConfig)]
    if len(gemmas) != 1 or gemmas[0].role not in LTX_GEMMA_TOKENIZER_KEYS:
        raise UnresolvedTextRecipe("LTX text recipe needs one Gemma 3 or Gemma 4 text tower")
    gemma = gemmas[0]
    projection = next((part for part in parts if part.role == "text_projection"), None)
    if projection is None:
        raise UnresolvedTextRecipe("LTX text recipe needs a text projection")
    kind = projection.plan.config
    if kind not in ("single_linear", "dual_linear", "dual_linear_gemma4"):
        raise UnresolvedTextRecipe("LTX text recipe needs a detected LTX projection contract")
    if (gemma.role == "gemma4_12b") != (kind == "dual_linear_gemma4"):
        raise UnresolvedTextRecipe("LTX projection and Gemma versions do not match")
    composition_roles = (gemma.role, "text_projection")
    if kind == "single_linear":
        connectors = next((part for part in parts if part.role == "connectors"), None)
        if connectors is None or not isinstance(connectors.plan.config, LtxConnectorConfig):
            raise UnresolvedTextRecipe("LTX single projection needs its video/audio connectors")
        composition_roles += ("connectors",)
    if set(roles) != set(composition_roles):
        raise UnresolvedTextRecipe("LTX text recipe would discard detected components")
    if {part.source_index for part in parts} != set(range(len(sources))):
        raise UnresolvedTextRecipe("LTX text recipe would discard an input source")
    tokenizer_key = LTX_GEMMA_TOKENIZER_KEYS[gemma.role]
    return TextRecipeBinding(
        "dinkster.text_ltxav",
        "dinkster.ltxav",
        tuple(
            replace(
                part,
                plan=replace(
                    part.plan,
                    runtime_facts=(
                        *part.plan.runtime_facts,
                        *_ENCODING_FACTS,
                        f"ltx_tokenizer_source_key={tokenizer_key}",
                    ),
                ),
            )
            if part is gemma
            else part
            for part in parts
        ),
        "dinkster_inference_torch.ltx_text_recipe:assemble_ltx_text_recipe",
        "dinkster_inference_torch.ltx_text_recipe:LtxTextRecipeRuntime",
        "dinkster_inference_torch.gemma_text:LtxGemmaTextEncoder",
        composition_roles,
    )


LTX_TEXT_RECIPE = TextRecipeDescriptor("dinkster.text_ltxav", bind_ltx_text_recipe, ("ltxv",))
