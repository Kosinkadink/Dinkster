"""Attention multiplier schemas remain available without importing torch."""

from __future__ import annotations

from dinkster_models import NODES
from dinkster_models.attention_multiply import (
    CLIPAttentionMultiply,
    UNetCrossAttentionMultiply,
    UNetSelfAttentionMultiply,
    UNetTemporalAttentionMultiply,
)


def test_attention_multiplier_nodes_are_exported_by_the_models_pack() -> None:
    nodes = (
        UNetSelfAttentionMultiply,
        UNetCrossAttentionMultiply,
        CLIPAttentionMultiply,
        UNetTemporalAttentionMultiply,
    )
    assert tuple(node.schema().node_type for node in nodes) == (
        "dinkster.models.u_net_self_attention_multiply",
        "dinkster.models.u_net_cross_attention_multiply",
        "dinkster.models.clip_attention_multiply",
        "dinkster.models.u_net_temporal_attention_multiply",
    )
    assert set(nodes).issubset(NODES)
    assert tuple(input_spec.id for input_spec in CLIPAttentionMultiply.schema().inputs) == (
        "clip",
        "q",
        "k",
        "v",
        "out",
    )
