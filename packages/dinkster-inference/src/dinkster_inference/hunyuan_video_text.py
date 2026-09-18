"""Ordered original Hunyuan Video text recipe."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from .component_registry import component_plans
from .llama3_text import LLAMA3_TOKENIZER_SHA256
from .prompt_tokens import CLIP_L_PROFILE
from .qwen_text import QwenTextConfig
from .text_recipes import (
    DetectedTextSources,
    TextEncodingProfile,
    TextRecipeBinding,
    TextRecipeComponent,
    TextRecipeDescriptor,
    UnresolvedTextRecipe,
)


def bind_hunyuan_video_text(sources: DetectedTextSources) -> TextRecipeBinding:
    components: list[TextRecipeComponent] = []
    for role in ("llama", "clip_l"):
        candidates: list[TextRecipeComponent] = []
        for index, matches in enumerate(sources):
            for match in matches:
                if role not in match.descriptor.text_encoder_roles:
                    continue
                for detected_role, planned in match.components:
                    if detected_role != role:
                        continue
                    for plan in component_plans(planned):
                        part = TextRecipeComponent(
                            index,
                            role,
                            plan,
                            None
                            if role == "llama"
                            else TextEncodingProfile(CLIP_L_PROFILE, projected_pooled=False),
                        )
                        if part not in candidates:
                            candidates.append(part)
        if len(candidates) != 1:
            raise UnresolvedTextRecipe(f"Hunyuan Video needs one unambiguous {role} component")
        part = candidates[0]
        if role == "llama":
            if not isinstance(part.plan.config, QwenTextConfig):
                raise UnresolvedTextRecipe("Hunyuan Video requires a Llama decoder config")
            part = replace(
                part,
                plan=replace(
                    part.plan,
                    runtime_facts=(
                        *part.plan.runtime_facts,
                        "llama_encoding=" + json.dumps(asdict(part.plan.config), sort_keys=True),
                        f"llama_tokenizer_sha256={LLAMA3_TOKENIZER_SHA256}",
                        "llama_crop=user_header_through_eot.v1",
                    ),
                ),
            )
        components.append(part)
    if {part.source_index for part in components} != set(range(len(sources))):
        raise UnresolvedTextRecipe("Hunyuan Video would discard an input source")
    return TextRecipeBinding(
        "dinkster.text_hunyuan_video",
        "dinkster.hunyuan_video",
        tuple(components),
        "dinkster_inference_torch.hunyuan_video_text:assemble_hunyuan_video_text",
        "dinkster_inference_torch.hunyuan_video_text:HunyuanVideoTextRuntime",
        "dinkster_inference_torch.hunyuan_video_text:compose_hunyuan_video_conditioning",
        ("llama", "clip_l"),
    )


HUNYUAN_VIDEO_TEXT_RECIPE = TextRecipeDescriptor(
    "dinkster.text_hunyuan_video", bind_hunyuan_video_text, ("hunyuan_video",)
)
