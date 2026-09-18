"""ACE-Step 1.5 text components and their ordered recipe contract."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from .assembly import (
    AssemblyError,
    ComponentPlan,
    _extract,  # pyright: ignore[reportPrivateUsage]
    _plan,  # pyright: ignore[reportPrivateUsage]
)
from .component_registry import ComponentDescriptor
from .families import ModelFamily
from .prompt_tokens import TokenizerProfile
from .qwen_bpe import QWEN_MERGES_SHA256, QWEN_VOCAB_SHA256
from .qwen_text import (
    ACE15_QWEN3_06B_CONFIG,
    ACE15_QWEN3_2B_CONFIG,
    ACE15_QWEN3_4B_CONFIG,
    QwenTextConfig,
    QwenTextDetectError,
    detect_qwen_text_config,
)
from .text_recipes import (
    DetectedTextSources,
    TextEncodingProfile,
    TextRecipeBinding,
    TextRecipeDescriptor,
    UnresolvedTextRecipe,
    _component,  # pyright: ignore[reportPrivateUsage]
)
from .weights import AssetIdentifiedSource, WeightSource

ACE15_TEXT_ROLES = ("qwen3_06b_ace15", "qwen3_ace15_lm")
ACE15_SOURCE_COMMIT = "25dfc16f9ac0a87991d34fbf5f02d6c25c844639"
ACE15_TOKENIZER_PROFILE = TokenizerProfile(
    max_length=99999999,
    start_token=None,
    end_token=None,
    pad_token=151643,
    pad_to_max_length=False,
    min_length=1,
)


def detect_ace15_components(
    source: WeightSource, path: Path, *, bind_asset_identity: bool = True
) -> tuple[tuple[str, ComponentPlan[QwenTextConfig]], ...]:
    """Detect exact ACE towers after component-scoped quantization extraction."""
    components: list[tuple[str, ComponentPlan[QwenTextConfig]]] = []
    keys = source.keys()
    prefixes = ("", "model.", "transformer.model.") + tuple(
        f"{root}{name}.transformer.model."
        for root in ("", "text_encoders.")
        for name in ("qwen3_06b", "qwen3_2b", "qwen3_4b")
    )
    for prefix in prefixes:
        if prefix + "embed_tokens.weight" not in keys:
            continue
        extracted = _extract(source, path, "ace15_text", prefix, root="")
        try:
            config = detect_qwen_text_config(extracted.geometries)
        except QwenTextDetectError:
            continue
        if config == ACE15_QWEN3_06B_CONFIG:
            role = ACE15_TEXT_ROLES[0]
        elif config in (ACE15_QWEN3_2B_CONFIG, ACE15_QWEN3_4B_CONFIG):
            role = ACE15_TEXT_ROLES[1]
        else:
            continue
        plan = _plan(role, extracted, config)
        if bind_asset_identity:
            if not isinstance(source, AssetIdentifiedSource) or not source.asset_digest:
                raise AssemblyError("ACE text source must carry immutable asset identity")
            plan = replace(
                plan,
                identity_facts=(
                    *plan.identity_facts,
                    f"asset_digest={source.asset_digest}",
                    f"asset_size={source.asset_size}",
                ),
            )
        components.append((role, plan))
    return tuple(components)


def ace15_component_descriptor(family: ModelFamily) -> ComponentDescriptor:
    return ComponentDescriptor(
        family,
        detect_ace15_components,
        ACE15_TEXT_ROLES,
        ACE15_TEXT_ROLES,
        (),
        "dinkster_inference_torch.ace15_text:assemble_ace15_text_recipe",
        "dinkster_inference_torch.ace15_text:ACE15TextRuntime",
        requires_text_recipe=True,
        attention_roles=ACE15_TEXT_ROLES,
    )


def bind_ace15_text_recipe(sources: DetectedTextSources) -> TextRecipeBinding:
    detected = tuple(
        _component(
            sources,
            role,
            TextEncodingProfile(
                ACE15_TOKENIZER_PROFILE,
                layer_norm_hidden_state=False,
                projected_pooled=False,
                attention_masked=True,
            ),
        )
        for role in ACE15_TEXT_ROLES
    )
    components = tuple(
        replace(
            part,
            plan=replace(
                part.plan,
                runtime_facts=(
                    *part.plan.runtime_facts,
                    "ace15_qwen_config=" + json.dumps(asdict(part.plan.config), sort_keys=True),
                    f"ace15_tokenizer_vocab_sha256={QWEN_VOCAB_SHA256}",
                    f"ace15_tokenizer_merges_sha256={QWEN_MERGES_SHA256}",
                    f"ace15_source_commit={ACE15_SOURCE_COMMIT}",
                    "ace15_structured_prompts=v1",
                    "ace15_composer_generation_bound=min_as_max",
                ),
            ),
        )
        for part in detected
    )
    if {part.source_index for part in components} != set(range(len(sources))):
        raise UnresolvedTextRecipe("ACE text recipe would discard an input source")
    return TextRecipeBinding(
        "dinkster.text_ace15",
        "dinkster.ace_step_1_5",
        tuple(sorted(components, key=lambda part: part.source_index)),
        "dinkster_inference_torch.ace15_text:assemble_ace15_text_recipe",
        "dinkster_inference_torch.ace15_text:ACE15TextRuntime",
        "dinkster_inference_torch.ace15_text:compose_ace15_conditioning",
        ACE15_TEXT_ROLES,
    )


ACE15_TEXT_RECIPE = TextRecipeDescriptor(
    "dinkster.text_ace15", bind_ace15_text_recipe, ("ace", "ace15")
)
