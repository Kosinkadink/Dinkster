"""Ordered Hunyuan Image Qwen and ByT5 text recipe."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .assembly import ComponentPlan, _extract, _plan  # pyright: ignore[reportPrivateUsage]
from .component_registry import component_plans
from .prompt_tokens import TokenizerProfile
from .qwen_bpe import QWEN_MERGES_SHA256, QWEN_VOCAB_SHA256
from .qwen_image_text import QwenImageTextConfig, detect_qwen_image_text_config
from .t5_text import BYT5_SMALL_GLYPH_CONFIG
from .text_recipes import (
    DetectedTextSources,
    TextEncodingProfile,
    TextRecipeBinding,
    TextRecipeComponent,
    TextRecipeDescriptor,
    UnresolvedTextRecipe,
)
from .weights import AssetIdentifiedSource, WeightSource

HUNYUAN_IMAGE_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
)

QWEN_TOKENIZER_CONFIG_SHA256 = "49292bd4a58a43382bf01311bc3ed7a151a7f16fe07d155e900ea1af281555db"
QWEN_TOKENIZER_SOURCE_URL = (
    "https://github.com/Comfy-Org/ComfyUI/tree/"
    "947c2749d25b8bb765608d29f051df1d58eb6680/comfy/text_encoders/qwen25_tokenizer"
)
# Uncompressed vocab, merges, and config sizes are 2,776,833, 1,671,853, and 8,018 bytes.
BYT5_TOKENIZER_SOURCE_URL = (
    "https://github.com/Comfy-Org/ComfyUI/tree/"
    "947c2749d25b8bb765608d29f051df1d58eb6680/comfy/text_encoders/byt5_tokenizer"
)
# Uncompressed sizes are 3,018, 3,090, and 25,602 bytes respectively.
BYT5_ADDED_TOKENS_SHA256 = "0004781309423057d33b9edf50d2669253d4347873ea33d4e7755cb866f23883"
BYT5_SPECIAL_TOKENS_SHA256 = "3c2572df5f4ec60a476e5d90a1f52a989f9eefbbf381d358924a44a648bf0edb"
BYT5_TOKENIZER_CONFIG_SHA256 = "3be33c710b17ac8686cd2b82ec044b0cd1122ff92c651d7170bfe1af8256ebfb"
BYT5_TOKENIZER_HASHES = (
    BYT5_ADDED_TOKENS_SHA256,
    BYT5_SPECIAL_TOKENS_SHA256,
    BYT5_TOKENIZER_CONFIG_SHA256,
)


def plan_hunyuan_image_qwen(
    source: WeightSource,
    path: Path,
    *,
    bind_asset_identity: bool = True,
) -> tuple[tuple[str, ComponentPlan[Any]], ...]:
    if not {
        "model.embed_tokens.weight",
        "visual.patch_embed.proj.weight",
    }.issubset(source.keys()):
        return ()
    extracted = _extract(source, path, "qwen25_vl", "")
    config = detect_qwen_image_text_config(extracted.geometries)
    planned = _plan("qwen25_vl", extracted, config)
    facts = planned.identity_facts
    if bind_asset_identity and isinstance(source, AssetIdentifiedSource):
        facts += (f"asset_digest={source.asset_digest}", f"asset_size={source.asset_size}")
    return (("qwen25_vl", replace(planned, identity_facts=facts)),)


def bind_hunyuan_image_text(sources: DetectedTextSources) -> TextRecipeBinding:
    components: list[TextRecipeComponent] = []
    for role in ("qwen25_vl", "byt5_small"):
        candidates: list[TextRecipeComponent] = []
        for index, matches in enumerate(sources):
            for match in matches:
                for detected_role, planned in match.components:
                    if detected_role != role:
                        continue
                    for plan in component_plans(planned):
                        profile = (
                            None
                            if role == "qwen25_vl"
                            else TextEncodingProfile(
                                TokenizerProfile(
                                    max_length=99999999,
                                    start_token=None,
                                    end_token=1,
                                    pad_token=0,
                                    pad_to_max_length=False,
                                    min_length=1,
                                ),
                                attention_masked=True,
                                zero_out_masked=True,
                            )
                        )
                        candidate = TextRecipeComponent(index, role, plan, profile)
                        if candidate not in candidates:
                            candidates.append(candidate)
        if len(candidates) != 1:
            raise UnresolvedTextRecipe(f"Hunyuan Image needs one unambiguous {role} component")
        part = candidates[0]
        if role == "qwen25_vl":
            if not isinstance(part.plan.config, QwenImageTextConfig):
                raise UnresolvedTextRecipe("Hunyuan Image requires Qwen2.5-VL-7B")
            facts = (
                "qwen_encoding=" + json.dumps(asdict(part.plan.config), sort_keys=True),
                f"qwen_tokenizer_source={QWEN_TOKENIZER_SOURCE_URL}",
                f"qwen_tokenizer_vocab_sha256={QWEN_VOCAB_SHA256}",
                f"qwen_tokenizer_merges_sha256={QWEN_MERGES_SHA256}",
                f"qwen_tokenizer_config_sha256={QWEN_TOKENIZER_CONFIG_SHA256}",
                "qwen_hidden_layer=-3",
                "qwen_hidden_normalization=false",
                "qwen_crop=second_user_header.v1",
            )
        else:
            if part.plan.config != BYT5_SMALL_GLYPH_CONFIG:
                raise UnresolvedTextRecipe("Hunyuan Image requires exact ByT5-small")
            facts = (
                f"byt5_tokenizer_source={BYT5_TOKENIZER_SOURCE_URL}",
                *(f"byt5_tokenizer_sha256={value}" for value in BYT5_TOKENIZER_HASHES),
            )
        components.append(
            replace(
                part,
                plan=replace(part.plan, runtime_facts=(*part.plan.runtime_facts, *facts)),
            )
        )
    if {part.source_index for part in components} != set(range(len(sources))):
        raise UnresolvedTextRecipe("Hunyuan Image would discard an input source")
    return TextRecipeBinding(
        "dinkster.text_hunyuan_image",
        "dinkster.hunyuan_image",
        tuple(components),
        "dinkster_inference_torch.hunyuan_image_text:assemble_hunyuan_image_text",
        "dinkster_inference_torch.hunyuan_image_text:HunyuanImageTextRuntime",
        "dinkster_inference_torch.hunyuan_image_text:compose_hunyuan_image_conditioning",
        ("qwen25_vl", "byt5_small"),
    )


HUNYUAN_IMAGE_TEXT_RECIPE = TextRecipeDescriptor(
    "dinkster.text_hunyuan_image", bind_hunyuan_image_text, ("hunyuan_image",)
)
