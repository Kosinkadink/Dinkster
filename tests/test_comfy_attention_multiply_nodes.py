"""Attention multiplier schemas remain available without importing torch."""

from __future__ import annotations

from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.attention_multiply import ATTENTION_MULTIPLY_NODES, CLIPAttentionMultiply


def test_attention_multiplier_nodes_are_exported_by_the_complete_pack() -> None:
    names = (
        "UNetSelfAttentionMultiply",
        "UNetCrossAttentionMultiply",
        "CLIPAttentionMultiply",
        "UNetTemporalAttentionMultiply",
    )
    assert tuple(node.__name__ for node in ATTENTION_MULTIPLY_NODES) == names
    assert set(ATTENTION_MULTIPLY_NODES).issubset(COMFY_NODES)
    assert tuple(input_spec.id for input_spec in CLIPAttentionMultiply.schema().inputs) == (
        "clip",
        "q",
        "k",
        "v",
        "out",
    )
