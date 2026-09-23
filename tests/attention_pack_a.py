"""Attention contract proof pack A: one descriptor per surface, unet backend."""

from __future__ import annotations

from typing import Any

from dinkster_api.v1 import (
    AttentionBackendDescriptor,
    AttentionContribution,
    AttentionOutputDescriptor,
    AttentionQKVDescriptor,
    AttentionSelector,
    AttentionWrapperDescriptor,
    BlockInjectionDescriptor,
    InferenceContribution,
)

SELECTOR = AttentionSelector(
    family="unet",
    block="middle_block.1.transformer_blocks.0",
    kind="self",
)

ATTENTION: AttentionContribution[Any] = AttentionContribution(
    qkv=(
        AttentionQKVDescriptor(
            "attention_a.qkv",
            SELECTOR,
            lambda q, k, v, context: (q, k, v),
            order=1,
            behavior_metadata=(("config.signed", "yes"),),
        ),
    ),
    wrappers=(
        AttentionWrapperDescriptor(
            "attention_a.wrapper",
            SELECTOR,
            lambda q, k, v, context, next: next(q, k, v),
            order=-2,
            terminal=True,
        ),
    ),
    outputs=(
        AttentionOutputDescriptor("attention_a.output", SELECTOR, lambda output, context: output),
    ),
    backends=(
        AttentionBackendDescriptor("attention_a.backend.unet", "unet", lambda q, k, v, context: q),
    ),
    blocks=(
        BlockInjectionDescriptor(
            "attention_a.inject",
            SELECTOR,
            lambda value, context: value,
            phase="before",
        ),
    ),
    torch_version="2.13.0+cpu",
    aimdo_version="0.5.5",
)


def register() -> InferenceContribution:
    return InferenceContribution(attention=ATTENTION)
