"""Ordered NewBie Gemma 3 and Jina CLIP v2 text recipe."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .assembly import ComponentPlan, _extract, _plan  # pyright: ignore[reportPrivateUsage]
from .component_registry import component_plans
from .gemma_text import GEMMA3_NEWBIE_4B_CONFIG, detect_gemma_text_config
from .jina_clip_text import JINA_CLIP_V2_CONFIG, detect_jina_clip_text_config
from .text_recipes import (
    DetectedTextSources,
    TextRecipeBinding,
    TextRecipeComponent,
    TextRecipeDescriptor,
    UnresolvedTextRecipe,
)
from .weights import AssetIdentifiedSource, WeightSource

NEWBIE_TOKENIZER_KEYS = {"gemma": "spiece_model", "jina": "spiece_model"}


def plan_newbie_component(
    source: WeightSource, path: Path, *, bind_asset_identity: bool = True
) -> tuple[tuple[str, ComponentPlan[Any]], ...]:
    keys = source.keys()
    if {
        "model.embed_tokens.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
    }.issubset(keys):
        role = "gemma"
        extracted = _extract(source, path, role, "model.")
        config = detect_gemma_text_config(extracted.geometries)
        if config != GEMMA3_NEWBIE_4B_CONFIG:
            return ()
    elif "model.encoder.layers.0.mixer.Wqkv.weight" in keys:
        role = "jina"
        extracted = _extract(source, path, role, "")
        config = detect_jina_clip_text_config(extracted.geometries)
    else:
        return ()
    planned = _plan(role, extracted, config)
    facts = planned.identity_facts
    if bind_asset_identity and isinstance(source, AssetIdentifiedSource):
        facts += (f"asset_digest={source.asset_digest}", f"asset_size={source.asset_size}")
    return ((role, replace(planned, identity_facts=facts)),)


def bind_newbie_text(sources: DetectedTextSources) -> TextRecipeBinding:
    components: list[TextRecipeComponent] = []
    for role, expected in (("gemma", GEMMA3_NEWBIE_4B_CONFIG), ("jina", JINA_CLIP_V2_CONFIG)):
        candidates: list[TextRecipeComponent] = []
        for index, matches in enumerate(sources):
            for match in matches:
                for detected_role, planned in match.components:
                    if detected_role == role:
                        for plan in component_plans(planned):
                            candidate = TextRecipeComponent(index, role, plan, None)
                            if candidate not in candidates:
                                candidates.append(candidate)
        if len(candidates) != 1 or candidates[0].plan.config != expected:
            raise UnresolvedTextRecipe(f"NewBie needs one exact {role} component")
        part = candidates[0]
        facts = (
            f"newbie_{role}_config=" + json.dumps(asdict(expected), sort_keys=True),
            f"newbie_{role}_tokenizer_key={NEWBIE_TOKENIZER_KEYS[role]}",
        )
        components.append(
            replace(part, plan=replace(part.plan, runtime_facts=(*part.plan.runtime_facts, *facts)))
        )
    if {part.source_index for part in components} != set(range(len(sources))):
        raise UnresolvedTextRecipe("NewBie would discard an input source")
    return TextRecipeBinding(
        "dinkster.text_newbie",
        "dinkster.newbie",
        tuple(components),
        "dinkster_inference_torch.newbie_text:assemble_newbie_text",
        "dinkster_inference_torch.newbie_text:NewBieTextRuntime",
        "dinkster_inference_torch.newbie_text:compose_newbie_conditioning",
        ("gemma", "jina"),
    )


NEWBIE_TEXT_RECIPE = TextRecipeDescriptor("dinkster.text_newbie", bind_newbie_text, ("newbie",))
