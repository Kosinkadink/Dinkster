"""Exact Jina CLIP v2 text tower layout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .weights import TensorGeometry


@dataclass(frozen=True)
class JinaClipTextConfig:
    vocab_size: int = 250002
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    rotary_base: float = 20000.0
    layer_norm_eps: float = 1e-5
    bos_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 1


JINA_CLIP_V2_CONFIG = JinaClipTextConfig()


def jina_clip_text_layout(
    config: JinaClipTextConfig = JINA_CLIP_V2_CONFIG,
) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    layout = {
        "model.embeddings.word_embeddings.weight": (config.vocab_size, hidden),
        "model.embeddings.token_type_embeddings.weight": (1, hidden),
        "model.emb_ln.weight": (hidden,),
        "model.emb_ln.bias": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"model.encoder.layers.{index}."
        layout[f"{prefix}mixer.Wqkv.weight"] = (hidden * 3, hidden)
        layout[f"{prefix}mixer.Wqkv.bias"] = (hidden * 3,)
        layout[f"{prefix}mixer.out_proj.weight"] = (hidden, hidden)
        layout[f"{prefix}mixer.out_proj.bias"] = (hidden,)
        layout[f"{prefix}norm1.weight"] = (hidden,)
        layout[f"{prefix}norm1.bias"] = (hidden,)
        layout[f"{prefix}mlp.fc1.weight"] = (config.intermediate_size, hidden)
        layout[f"{prefix}mlp.fc1.bias"] = (config.intermediate_size,)
        layout[f"{prefix}mlp.fc2.weight"] = (hidden, config.intermediate_size)
        layout[f"{prefix}mlp.fc2.bias"] = (hidden,)
        layout[f"{prefix}norm2.weight"] = (hidden,)
        layout[f"{prefix}norm2.bias"] = (hidden,)
    return layout


def detect_jina_clip_text_config(
    geometries: Mapping[str, TensorGeometry],
) -> JinaClipTextConfig:
    layout = jina_clip_text_layout()
    problems: list[str] = []
    for key, shape in layout.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(layout)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise ValueError(f"not exact Jina CLIP v2: {shown}")
    return JINA_CLIP_V2_CONFIG
