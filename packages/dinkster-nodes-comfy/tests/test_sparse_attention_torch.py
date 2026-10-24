"""CPU seams for BlockSparseAttention model patching."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.sparse_attention import (
    BlockSparseAttention,
    SparseAttnPatch,
    install_override,
    make_attention_override,
)


class _Sampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 1.0 - percent


class _Model:
    def __init__(self) -> None:
        self.model_options = {"transformer_options": {}}
        self.callbacks: dict[str, Any] = {}

    def clone(self) -> _Model:
        result = _Model()
        result.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        return result

    def get_model_object(self, name: str) -> Any:
        return _Sampling() if name == "model_sampling" else object()

    def add_callback_with_key(self, _kind: Any, key: str, callback: Any) -> None:
        self.callbacks[key] = callback


def _patch(*, minimum: int = 0) -> SparseAttnPatch:
    return SparseAttnPatch(1.3, 0.0, False, 1.0, 0.0, minimum, {2}, "off", 0, False)


def test_schema_has_source_identity_required_inputs_and_stable_methods() -> None:
    schema = BlockSparseAttention.define_schema()
    assert BlockSparseAttention in COMFY_NODES
    assert schema.node_type == "BlockSparseAttention"
    assert not schema.aliases
    assert [item.id for item in schema.inputs] == [
        "model",
        "start_percent",
        "end_percent",
        "dense_blocks",
        "min_tokens",
        "extra_tokens",
        "sink_conditioning",
        "verbose",
    ]
    assert all(item.required for item in schema.inputs)
    assert [option.key for option in schema.combos[0].options] == ["sol-attn", "sla", "vsa"]


def test_dense_fallback_preserves_previous_override_and_all_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_nodes_comfy.sparse_attention._runtime", lambda: (torch, object()))
    patch = _patch(minimum=8)
    calls: list[str] = []

    def base(*_args: Any, **_kwargs: Any) -> str:
        calls.append("base")
        return "base"

    def previous(func: Any, *args: Any, **kwargs: Any) -> str:
        calls.append("previous")
        return func(*args, **kwargs)

    override = make_attention_override(patch, previous)
    q = torch.zeros((1, 4, 128))
    for options in ({}, {"block_index": 2}, {"sigmas": torch.tensor([2.0])}):
        assert override(base, q, q, q, 1, transformer_options=options) == "base"
    assert override(base, q, q, q, 1, mask=torch.zeros((1, 1))) == "base"
    assert calls == ["previous", "base"] * 4
    assert patch.dense_reason({"sigmas": torch.tensor([1.0])}, 8, 0) is None
    assert patch.dense_reason({"sigmas": torch.tensor([0.0])}, 8, 0) is None


def test_vsa_plan_inverse_rope_cache_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dinkster_nodes_comfy.sparse_attention._runtime", lambda: (torch, object()))
    patch = _patch()
    layout = SimpleNamespace(
        signature=(0, 2, 4, 4, 0),
        segments=((0, 3, "text"), (3, 11, "video")),
        seq_len=11,
    )
    plan = patch.vsa_plan(layout, torch.device("cpu"))
    assert plan["src"][plan["inv"]].tolist() == list(range(layout.seq_len))
    freqs = torch.randn((1, layout.seq_len, 2))
    assert patch.vsa_rope_freqs(freqs, plan) is patch.vsa_rope_freqs(freqs, plan)
    patch.reset()
    assert not patch.vsa_plans and patch.vsa_rope is None


def test_override_installation_is_idempotent_and_chains_reinstall() -> None:
    patch = _patch()
    options: dict[str, Any] = {"optimized_attention_override": lambda *_args, **_kwargs: None}
    original = options["optimized_attention_override"]
    install_override(patch, options)
    installed = options["optimized_attention_override"]
    install_override(patch, options)
    assert options["optimized_attention_override"] is installed
    options["optimized_attention_override"] = original
    install_override(patch, options)
    assert options["optimized_attention_override"] is not original
