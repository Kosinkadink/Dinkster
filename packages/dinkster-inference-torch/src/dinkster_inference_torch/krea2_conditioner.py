"""Krea 2 Qwen3-VL-4B language-tower conditioner.

Text-only conditioning path matching ComfyUI b78cec87's
``comfy/text_encoders/krea2.py``: the fixed template, Qwen BPE,
twelve residual-stream taps stacked as ``(batch, 12, seq, 2560)``,
the post-template strip, and the ``(batch, seq, 30720)`` flatten the
DiT's TextFusion module unpacks. The checkpoint's vision tower is
never executed here; image-conditioned workflows are out of scope for
this family. State keys match the ``model.language_model.*`` subtree
after the assembly planner strips that source prefix.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
from dinkster_inference import Conditioning
from dinkster_inference.krea2_text import KREA2_TEXT_CONFIG, select_krea2_output
from dinkster_inference.qwen_bpe import tokenize_krea2_prompt

from .attention import AttentionKernel, select_attention
from .operations import INITLESS, Operations, bound_compute_device
from .qwen_image_text import QwenImageLanguageModel

_DEFAULT_QWEN_ATTENTION = select_attention("qwen").kernel


def krea2_language_model(
    *,
    operations: Operations = INITLESS,
    attention_kernel: AttentionKernel = _DEFAULT_QWEN_ATTENTION,
) -> QwenImageLanguageModel:
    """Build the exact Krea 2 Qwen3-VL-4B language tower."""

    config = KREA2_TEXT_CONFIG
    return QwenImageLanguageModel.reduced(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        rope_dims=config.rope_dims,
        head_dim=config.head_dim,
        rope_theta=config.rope_theta,
        qkv_bias=config.qkv_bias,
        qk_norm=config.qk_norm,
        final_norm=config.final_norm,
        interleaved_mrope=config.interleaved_mrope,
        max_position_embeddings=config.max_position_embeddings,
        architecture="Krea 2",
        operations=operations,
        attention_kernel=attention_kernel,
    )


class Krea2TextEncoder:
    """Raw prompt to Krea 2's stacked twelve-tap sequence conditioning."""

    def __init__(self, model: QwenImageLanguageModel) -> None:
        config = KREA2_TEXT_CONFIG
        shape = model.shape
        expected = {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_layers": config.num_hidden_layers,
            "num_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "rope_dims": config.rope_dims,
            "qkv_bias": config.qkv_bias,
            "qk_norm": config.qk_norm,
            "final_norm": config.final_norm,
            "interleaved_mrope": config.interleaved_mrope,
            "max_position_embeddings": config.max_position_embeddings,
            "architecture": "Krea 2",
        }
        if asdict(shape) != expected:
            raise ValueError("Krea2TextEncoder requires the exact Krea 2 language profile")
        self.model = model

    def encode(self, text: str) -> Conditioning[torch.Tensor]:
        tokens = tokenize_krea2_prompt(text)
        self.model.validate_sequence_length(len(tokens.ids))
        selection = select_krea2_output([list(tokens.ids)], [list(tokens.attention_mask)])
        device = (
            bound_compute_device(self.model.embed_tokens) or self.model.embed_tokens.weight.device
        )
        ids = torch.tensor(tokens.ids, dtype=torch.long, device=device).unsqueeze(0)
        attention = torch.tensor(tokens.attention_mask, dtype=torch.long, device=device).unsqueeze(
            0
        )
        stacked = self.model.tapped_states(ids, attention, tap_layers=KREA2_TEXT_CONFIG.tap_layers)
        stripped = stacked[:, :, selection.slice_start :, :]
        batch, taps, length, hidden = stripped.shape
        flattened = stripped.permute(0, 2, 1, 3).reshape(batch, length, taps * hidden)
        return Conditioning(flattened.float(), None)


__all__ = ["Krea2TextEncoder", "krea2_language_model"]
