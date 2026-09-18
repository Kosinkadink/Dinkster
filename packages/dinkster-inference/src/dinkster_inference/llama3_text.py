"""Original Hunyuan Video Llama3 layout and component planning.

The configuration and model keys match ComfyUI comfy/text_encoders/llama.py
at 25dfc16f9ac0a87991d34fbf5f02d6c25c844639.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .assembly import (
    AssemblyError,
    ComponentPlan,
    _extract,  # pyright: ignore[reportPrivateUsage]
    _plan,  # pyright: ignore[reportPrivateUsage]
)
from .qwen_text import QwenTextConfig, qwen_text_layout
from .weights import AssetIdentifiedSource, TensorGeometry, WeightSource

HUNYUAN_VIDEO_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "Describe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, "
    "physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
LLAMA3_TOKENIZER_SHA256 = "036b3f4da1a2ad75e829debed3e96393998c97b386c0a288f6f7431d5908432b"
HUNYUAN_LLAMA3_CONFIG = QwenTextConfig(
    architecture="hunyuan_video_llama3",
    vocab_size=128320,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=8192,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    qkv_bias=False,
    qk_norm=False,
    prompt_template=HUNYUAN_VIDEO_TEMPLATE,
    min_tokens=1,
    pad_token_id=128258,
    output_hidden_layer=-3,
    layer_norm_hidden_state=False,
)


def detect_llama3_config(geometries: Mapping[str, TensorGeometry]) -> QwenTextConfig:
    """Require the complete Llama3 layout, never a generic Qwen fallthrough."""
    expected = qwen_text_layout(HUNYUAN_LLAMA3_CONFIG)
    if {key: geometry.shape for key, geometry in geometries.items()} != expected:
        raise AssemblyError("llama: header does not match the exact Hunyuan Llama3 layout")
    return HUNYUAN_LLAMA3_CONFIG


def plan_llama3_component(
    source: WeightSource, path: Path, *, bind_asset_identity: bool = True
) -> ComponentPlan[QwenTextConfig]:
    """Strip one model prefix and retain quantization and immutable source identity."""
    suffix = "embed_tokens.weight"
    prefixes = [key[: -len(suffix)] for key in source.keys() if key.endswith(suffix)]
    if len(prefixes) != 1:
        raise AssemblyError("llama: source must contain one token embedding subtree")
    extracted = _extract(source, path, "llama", prefixes[0], root="")
    config = detect_llama3_config(extracted.geometries)
    plan = _plan("llama", extracted, config)
    if not bind_asset_identity:
        return plan
    if not isinstance(source, AssetIdentifiedSource) or not source.asset_digest:
        raise AssemblyError("llama: source must carry immutable asset identity")
    return replace(
        plan,
        identity_facts=(
            f"asset_digest={source.asset_digest}",
            f"asset_size={source.asset_size}",
        ),
    )


def detect_llama3_component(
    source: WeightSource, path: Path, *, bind_asset_identity: bool = True
) -> tuple[tuple[str, ComponentPlan[QwenTextConfig]], ...]:
    keys = source.keys()
    if not any(key.endswith("embed_tokens.weight") for key in keys):
        return ()
    try:
        return (
            ("llama", plan_llama3_component(source, path, bind_asset_identity=bind_asset_identity)),
        )
    except AssemblyError:
        return ()
