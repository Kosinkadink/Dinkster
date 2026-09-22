"""Attention contract proof pack B: flux backend plus collision/mismatch variants."""

from __future__ import annotations

from typing import Any

from dinkster_api.v1 import (
    AttentionBackendDescriptor,
    AttentionContribution,
    AttentionQKVDescriptor,
    AttentionSelector,
    AttentionWrapperDescriptor,
    InferenceContribution,
)

SELECTOR = AttentionSelector(family="flux", block="double_blocks.0", kind="cross")


def _kernel(q: Any, k: Any, v: Any, context: Any) -> Any:
    return v


def _coexisting() -> AttentionContribution[Any]:
    return AttentionContribution(
        qkv=(
            AttentionQKVDescriptor("attention_b.qkv", SELECTOR, lambda q, k, v, context: (v, k, q)),
        ),
        backends=(AttentionBackendDescriptor("attention_b.backend.flux", "flux", _kernel),),
        torch_version="2.13.0+cpu",
        aimdo_version="0.5.5",
    )


def _colliding() -> AttentionContribution[Any]:
    return AttentionContribution(
        backends=(AttentionBackendDescriptor("attention_b.backend.unet", "unet", _kernel),),
        torch_version="2.13.0+cpu",
        aimdo_version="0.5.5",
    )


def _mismatched_pin() -> AttentionContribution[Any]:
    return AttentionContribution(
        backends=(AttentionBackendDescriptor("attention_b.backend.flux", "flux", _kernel),),
        torch_version="999.0.0",
        aimdo_version="0.5.5",
    )


def register() -> InferenceContribution:
    return InferenceContribution(attention=_coexisting())


def register_colliding() -> InferenceContribution:
    return InferenceContribution(attention=_colliding())


def register_mismatched_pin() -> InferenceContribution:
    return InferenceContribution(attention=_mismatched_pin())


def register_duplicate_id() -> InferenceContribution:
    return InferenceContribution(
        attention=AttentionContribution(
            qkv=(
                AttentionQKVDescriptor(
                    "attention_a.qkv",
                    AttentionSelector(family="unet"),
                    lambda q, k, v, context: (q, k, v),
                ),
            ),
            torch_version="2.13.0+cpu",
            aimdo_version="0.5.5",
        )
    )


def register_terminal_wrapper() -> InferenceContribution:
    return InferenceContribution(
        attention=AttentionContribution(
            wrappers=(
                AttentionWrapperDescriptor(
                    "attention_b.terminal",
                    SELECTOR,
                    lambda q, k, v, context, next: v,
                    terminal=True,
                ),
            ),
            torch_version="2.13.0+cpu",
            aimdo_version="0.5.5",
        )
    )
