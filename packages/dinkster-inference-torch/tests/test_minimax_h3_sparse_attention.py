# pyright: reportPrivateUsage=false

"""CPU structural tests for MiniMax H3 sparse-attention binding."""

from __future__ import annotations

import torch
from dinkster_inference import MiniMaxH3SparseAttentionConfig
from dinkster_inference_torch.minimax_h3_attention import MiniMaxH3PackedSequenceFacts
from dinkster_inference_torch.minimax_h3_sparse_attention import (
    MiniMaxH3SparseAttention,
    _vsa_plan,
)


def _facts() -> MiniMaxH3PackedSequenceFacts:
    return MiniMaxH3PackedSequenceFacts(
        80,
        (
            (0, 70, "text"),
            (70, 72, "audio"),
            (72, 80, "video"),
        ),
        (2, 2, 2),
    )


def test_vsa_plan_pads_each_prefix_segment_and_video_cube_independently() -> None:
    facts = _facts()
    plan = _vsa_plan(facts, torch.device("cpu"))

    assert plan.prefix_blocks == 3
    assert plan.padded_rows == 4 * 64
    assert plan.block_lengths.tolist() == [64, 6, 2, 8]
    assert torch.equal(
        plan.source_rows[plan.inverse_rows],
        torch.arange(facts.sequence_length),
    )


def test_sparse_binding_derives_conditioning_sinks_and_guidance_lane_identity() -> None:
    sparse = MiniMaxH3SparseAttention(MiniMaxH3SparseAttentionConfig(selection="vsa", min_tokens=0))
    conditional = sparse.bind(sigma=0.5, lane="conditional", facts=_facts())
    unconditional = sparse.bind(sigma=0.5, lane="unconditional", facts=_facts())

    assert conditional._sinks() == ((0, 2), (1, 2))
    assert conditional.lane != unconditional.lane


def test_sparse_binding_honors_sigma_token_and_dense_block_gates_before_backend() -> None:
    sparse = MiniMaxH3SparseAttention(
        MiniMaxH3SparseAttentionConfig(
            selection="vsa",
            start_percent=0.2,
            end_percent=0.8,
            dense_blocks=frozenset((4,)),
            min_tokens=100,
        )
    )
    hidden = torch.empty((1, 80, 128), dtype=torch.bfloat16)

    assert sparse.bind(sigma=1.0, lane="conditional", facts=_facts())._eligible(hidden, 3) is False
    assert sparse.bind(sigma=0.5, lane="conditional", facts=_facts())._eligible(hidden, 4) is False
    assert sparse.bind(sigma=0.5, lane="conditional", facts=_facts())._eligible(hidden, 3) is False
