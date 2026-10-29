"""Qwen3 sparse router, causal model, and expert-residency tests."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, cast
from unittest.mock import patch

import dinkster_inference_torch.qwen_generation as qwen_generation_mod
import pytest
import torch
import torch.nn.functional as F
from dinkster_inference import (
    QWEN3_30B_A3B_CONFIG,
    GenerationRequest,
    GenerationSessionHandle,
    GenerationStopConditions,
    GenerationTerminalEvent,
    GenerationTokenEvent,
    Qwen3MoeConfig,
    qwen3_moe_layout,
)
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference_torch import (
    Qwen3MoeBlock,
    Qwen3MoeCheckpointError,
    Qwen3MoeExpert,
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
    QwenContinuousGenerationProvider,
    QwenGenerationProvider,
    ResidencyRouted,
    enroll_component,
    load_qwen3_moe_checkpoint,
)
from dinkster_inference_torch import qwen3_moe as qwen3_moe_mod
from dinkster_inference_torch import qwen3_moe_checkpoint as qwen3_moe_checkpoint_mod
from dinkster_inference_torch import qwen_text as qwen_text_mod
from test_autoencoder_kl import write_safetensors


def _tiny_config() -> Qwen3MoeConfig:
    return replace(
        QWEN3_30B_A3B_CONFIG,
        architecture="qwen3_moe_test",
        vocab_size=19,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        num_experts=4,
        num_experts_per_tok=2,
        eos_token_id=18,
    )


def _fill(module: torch.nn.Module, *, dtype: torch.dtype = torch.float32) -> None:
    generator = torch.Generator().manual_seed(527)
    state = {
        key: (torch.randn(value.shape, generator=generator, dtype=torch.float32) * 0.125).to(dtype)
        for key, value in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True, assign=True)


def _model() -> Qwen3MoeForCausalLM:
    model = Qwen3MoeForCausalLM(_tiny_config())
    _fill(model)
    return model


class _GenerationTokenizer:
    def encode(self, text: str) -> list[int]:
        if text != "prompt":
            return []
        return [1, 4, 7]

    def decode_bytes(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> bytes:
        del skip_special_tokens
        return bytes(65 + token_id % 26 for token_id in token_ids)


def _generation_provider(model: Qwen3MoeForCausalLM) -> QwenGenerationProvider:
    with patch.object(qwen_generation_mod, "QWEN3_30B_A3B_CONFIG", model.config):
        return QwenGenerationProvider(
            model,
            _GenerationTokenizer(),
            "native:qwen3-moe:test",
            eos_token_ids=(),
            block_tokens=4,
            max_device_blocks=16,
        )


def _generation_request(
    provider: QwenGenerationProvider | QwenContinuousGenerationProvider,
    *,
    maximum: int,
    open_session: bool = False,
    session: GenerationSessionHandle | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        provider.id,
        "native:qwen3-moe:test",
        prompt="prompt",
        stop=GenerationStopConditions(maximum),
        open_session=open_session,
        session=session,
    )


def test_original_checkpoint_key_layout_and_strict_load() -> None:
    config = _tiny_config()
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    expected = qwen3_moe_layout(config)
    assert sorted((key, tuple(value.shape)) for key, value in model.state_dict().items()) == sorted(
        expected.items()
    )
    state = {key: torch.empty(shape, device="meta") for key, shape in expected.items()}
    model.load_state_dict(state, strict=True, assign=True)
    del state["model.layers.1.mlp.experts.3.down_proj.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True, assign=True)


def test_sharded_checkpoint_loader_streams_exact_state_to_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _model().state_dict()
    keys = tuple(expected)
    first = write_safetensors(
        tmp_path / "model-00001-of-00002.safetensors",
        {key: expected[key] for key in keys[::2]},
    )
    second = write_safetensors(
        tmp_path / "model-00002-of-00002.safetensors",
        {key: expected[key] for key in keys[1::2]},
    )
    seen: set[str] = set()

    def detect(geometries: object) -> Qwen3MoeConfig:
        seen.update(cast("dict[str, object]", geometries))
        return _tiny_config()

    payload_orders: list[tuple[str, ...]] = []
    expected_payload_orders: list[tuple[str, ...]] = []
    real_load = qwen3_moe_checkpoint_mod.load_tensors_from_file
    real_header = qwen3_moe_checkpoint_mod.load_safetensors_header

    def reverse_header(path: Path) -> SafetensorsSource:
        source = real_header(path)
        return replace(source, entries=dict(reversed(tuple(source.entries.items()))))

    def load_in_order(
        handle: BinaryIO,
        source: SafetensorsSource,
        selected: Iterable[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        ordered = tuple(source.keys()) if selected is None else tuple(selected)
        payload_orders.append(ordered)
        expected_payload_orders.append(
            tuple(sorted(source.keys(), key=lambda key: source.entry(key).offset))
        )
        return real_load(handle, source, ordered)

    reservations: list[tuple[torch.device, int]] = []

    def reserve(device: torch.device, size: int) -> None:
        reservations.append((device, size))

    monkeypatch.setattr(qwen3_moe_checkpoint_mod, "detect_qwen3_moe_config", detect)
    monkeypatch.setattr(qwen3_moe_checkpoint_mod, "load_safetensors_header", reverse_header)
    monkeypatch.setattr(qwen3_moe_checkpoint_mod, "load_tensors_from_file", load_in_order)
    monkeypatch.setattr(qwen3_moe_checkpoint_mod, "_reserve_cuda_model_bytes", reserve)
    loaded = load_qwen3_moe_checkpoint((first, second), device="cpu")

    assert seen == set(expected)
    assert reservations == [
        (
            torch.device("cpu"),
            sum(value.numel() * value.element_size() for value in expected.values()),
        )
    ]
    assert payload_orders == expected_payload_orders
    assert not loaded.training
    assert all(tensor.device == torch.device("cpu") for tensor in loaded.state_dict().values())
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_checkpoint_loader_prereserves_exact_model_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations: list[tuple[int, torch.dtype, torch.device]] = []

    def empty(
        size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object:
        allocations.append((size, dtype, device))
        return object()

    monkeypatch.setattr(qwen3_moe_checkpoint_mod.torch, "empty", empty)
    qwen3_moe_checkpoint_mod._reserve_cuda_model_bytes(  # pyright: ignore[reportPrivateUsage]
        torch.device("cuda:1"),
        123_456,
    )
    qwen3_moe_checkpoint_mod._reserve_cuda_model_bytes(  # pyright: ignore[reportPrivateUsage]
        torch.device("cpu"),
        654_321,
    )

    assert allocations == [(123_456, torch.uint8, torch.device("cuda:1"))]


def test_sharded_checkpoint_loader_rejects_duplicate_keys_before_construction(
    tmp_path: Path,
) -> None:
    first = write_safetensors(
        tmp_path / "model-00001-of-00002.safetensors",
        {"model.embed_tokens.weight": torch.ones((2, 3))},
    )
    second = write_safetensors(
        tmp_path / "model-00002-of-00002.safetensors",
        {"model.embed_tokens.weight": torch.zeros((2, 3))},
    )
    with pytest.raises(Qwen3MoeCheckpointError, match="appears in multiple shards"):
        load_qwen3_moe_checkpoint((first, second), device="cpu")


def test_router_matches_float32_softmax_topk_normalization_and_expert_order() -> None:
    block = Qwen3MoeSparseMoeBlock(_tiny_config())
    _fill(block)
    hidden = torch.linspace(-1.5, 1.25, 48, dtype=torch.float32).reshape(2, 3, 8)

    flat = hidden.reshape(-1, hidden.shape[-1])
    probabilities = F.softmax(F.linear(flat, block.gate.weight), dim=-1, dtype=torch.float32)
    weights, selected = torch.topk(probabilities, block.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = torch.zeros_like(flat)
    expert_mask = F.one_hot(selected, num_classes=block.num_experts).permute(2, 1, 0)
    for expert_index, untyped_expert in enumerate(block.experts):
        expert = cast(Qwen3MoeExpert, untyped_expert)
        top_index, token_index = torch.where(expert_mask[expert_index])
        if token_index.numel() == 0:
            continue
        current = F.linear(flat[token_index], expert.gate_proj.weight)
        current = F.silu(current) * F.linear(flat[token_index], expert.up_proj.weight)
        current = F.linear(current, expert.down_proj.weight)
        expected.index_add_(0, token_index, current * weights[token_index, top_index, None])

    with torch.no_grad():
        actual = block(hidden)
    torch.testing.assert_close(actual, expected.reshape_as(hidden), rtol=0, atol=0)


def test_bfloat16_norm_and_rope_match_authoritative_operation_order() -> None:
    block = Qwen3MoeBlock(_tiny_config())
    _fill(block, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(528)

    hidden = torch.randn((2, 3, 8), generator=generator).to(torch.bfloat16)
    normalized = hidden.float()
    normalized = normalized * torch.rsqrt(
        normalized.pow(2).mean(-1, keepdim=True) + cast(float, block.input_layernorm.eps)
    )
    expected_norm = block.input_layernorm.weight * normalized.to(hidden.dtype)
    actual_norm = qwen3_moe_mod._qwen3_moe_rms_norm(  # pyright: ignore[reportPrivateUsage]
        block.input_layernorm,
        hidden,
    )
    torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
    assert not torch.equal(
        actual_norm,
        F.rms_norm(
            hidden,
            block.input_layernorm.normalized_shape,
            block.input_layernorm.weight,
            block.input_layernorm.eps,
        ),
    )

    query = torch.randn((2, 2, 3, 4), generator=generator).to(torch.bfloat16)
    key = torch.randn((2, 1, 3, 4), generator=generator).to(torch.bfloat16)
    frequencies = qwen3_moe_mod._rope(  # pyright: ignore[reportPrivateUsage]
        4,
        3,
        _tiny_config().rope_theta,
        device=torch.device("cpu"),
    )
    cosine, sine, negative_sine = frequencies
    cosine = cosine.to(query.dtype)
    sine = torch.cat((sine, -negative_sine), dim=-1).to(query.dtype)

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    expected_query = (query * cosine) + (rotate_half(query) * sine)
    expected_key = (key * cosine) + (rotate_half(key) * sine)
    actual_query, actual_key = block.self_attn._apply_rotary(  # pyright: ignore[reportPrivateUsage]
        query,
        key,
        frequencies,
    )
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
    old_query, old_key = qwen_text_mod._apply_rope(  # pyright: ignore[reportPrivateUsage]
        query,
        key,
        frequencies,
    )
    assert not torch.equal(actual_query, old_query)
    assert not torch.equal(actual_key, old_key)


def test_initial_prefill_uses_authoritative_causal_attention_contract() -> None:
    calls: list[tuple[torch.Tensor | None, bool]] = []

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        del k, v, scale, enable_gqa
        calls.append((mask, causal))
        return torch.zeros_like(q)

    config = _tiny_config()
    model = Qwen3MoeForCausalLM(config, attention_kernel=attention)
    _fill(model)
    ids = torch.tensor(((1, 4, 7, 3, 8),), dtype=torch.long)
    cache = tuple(
        (
            torch.empty(1, config.num_key_value_heads, ids.shape[1], config.head_dim),
            torch.empty(1, config.num_key_value_heads, ids.shape[1], config.head_dim),
        )
        for _ in range(config.num_hidden_layers)
    )

    with torch.no_grad():
        model.forward_causal(ids[:, :3], cache)
    assert calls == [(None, True), (None, True)]

    calls.clear()
    with torch.no_grad():
        model.forward_causal(ids[:, 3:], cache, cache_position=3)
    assert len(calls) == config.num_hidden_layers
    assert all(mask is not None and causal is False for mask, causal in calls)


def test_dispatch_executes_only_selected_experts_in_index_order() -> None:
    block = Qwen3MoeSparseMoeBlock(_tiny_config())
    _fill(block)
    hidden = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(1, 3, 8)
    with torch.no_grad():
        selected = torch.topk(
            F.softmax(block.gate(hidden.reshape(-1, 8)).float(), dim=-1),
            block.top_k,
            dim=-1,
        ).indices
    expected = sorted(set(selected.flatten().tolist()))
    called: list[int] = []
    handles = [
        expert.register_forward_hook(
            lambda _module, _args, _output, index=index: called.append(index)
        )
        for index, expert in enumerate(block.experts)
    ]
    try:
        with torch.no_grad():
            block(hidden)
    finally:
        for handle in handles:
            handle.remove()
    assert called == expected


def test_model_prefetch_walk_contains_only_router_selected_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    calls: list[tuple[torch.nn.Module, ...]] = []
    original = qwen3_moe_mod.make_prefetch_queue

    def capture(blocks: Iterable[torch.nn.Module]) -> object:
        selected = tuple(blocks)
        calls.append(selected)
        return original(selected)

    monkeypatch.setattr(qwen3_moe_mod, "make_prefetch_queue", capture)
    ids = torch.tensor(((1, 4, 7),), dtype=torch.long)
    with torch.no_grad():
        model(ids)
    assert len(calls) == model.config.num_hidden_layers
    assert all(call for call in calls)
    assert all(isinstance(module, Qwen3MoeExpert) for call in calls for module in call)

    before = len(calls)
    with torch.no_grad():
        model.forward_causal(ids, prefetch=False)
    assert len(calls) == before


def test_causal_chunks_match_one_shot_logits() -> None:
    model = _model()
    ids = torch.tensor(((1, 4, 7, 3, 8),), dtype=torch.long)
    with torch.no_grad():
        expected = model(ids)

    config = model.config
    cache = tuple(
        (
            torch.empty(1, config.num_key_value_heads, ids.shape[1], config.head_dim),
            torch.empty(1, config.num_key_value_heads, ids.shape[1], config.head_dim),
        )
        for _ in range(config.num_hidden_layers)
    )
    with torch.no_grad():
        first, _ = model.forward_causal(ids[:, :3], cache)
        second, _ = model.forward_causal(ids[:, 3:], cache, cache_position=3)
    torch.testing.assert_close(torch.cat((first, second), dim=1), expected, rtol=1e-5, atol=1e-6)


def test_each_expert_is_one_residency_unit_and_survives_loaded_offloaded_churn() -> None:
    model = _model()
    ids = torch.tensor(((2, 5, 1),), dtype=torch.long)
    with torch.no_grad():
        expected = model(ids)
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)

    expected_experts = {
        f"model.layers.{layer}.mlp.experts.{expert}"
        for layer in range(model.config.num_hidden_layers)
        for expert in range(model.config.num_experts)
    }
    units = mechanism.residency_units()
    expert_units = {unit.name for unit in units if unit.expert}
    assert expert_units == expected_experts
    assert {unit.name for unit in units if not unit.expert}.isdisjoint(expected_experts)
    for layer_index, untyped_layer in enumerate(model.model.layers):
        layer = cast(Qwen3MoeBlock, untyped_layer)
        for expert_index, untyped_expert in enumerate(layer.mlp.experts):
            expert = cast(Qwen3MoeExpert, untyped_expert)
            expected_unit = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
            bindings = [
                cast(ResidencyRouted, projection).residency_binding()
                for projection in (expert.gate_proj, expert.up_proj, expert.down_proj)
            ]
            assert all(binding is not None for binding in bindings)
            assert {binding.unit for binding in bindings if binding is not None} == {expected_unit}

    mechanism.partially_unload(mechanism.loaded_bytes())
    assert not mechanism.loaded_unit_names()
    with torch.no_grad():
        offloaded = model(ids)
    torch.testing.assert_close(offloaded, expected, rtol=0, atol=0)


def test_position_limit_is_refused_before_execution() -> None:
    model = _model()
    ids = torch.zeros((1, model.config.max_position_embeddings + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="maximum is 16"):
        model(ids)


def test_generation_provider_requires_the_exact_qwen3_moe_profile() -> None:
    with pytest.raises(ValueError, match="exact Anima Qwen3-0.6B or Qwen3-30B-A3B profile"):
        QwenGenerationProvider(
            _model(),
            _GenerationTokenizer(),
            "native:qwen3-moe:test",
        )


@pytest.mark.parametrize(
    ("module_name", "mismatch", "message"),
    (
        ("model.layers.0.self_attn.q_proj", "device", "homogeneous model placement"),
        ("model.layers.0.post_attention_layernorm", "dtype", "homogeneous compute dtype"),
        ("model.layers.0.mlp.gate", "device", "homogeneous model placement"),
        ("model.layers.0.mlp.experts.0.down_proj", "dtype", "homogeneous compute dtype"),
        ("lm_head", "device", "homogeneous model placement"),
    ),
)
def test_generation_provider_requires_homogeneous_qwen3_moe_placement(
    module_name: str,
    mismatch: str,
    message: str,
) -> None:
    model = _model()
    module = dict(model.named_modules())[module_name]
    weight = cast("torch.nn.Linear | torch.nn.RMSNorm", module).weight
    replacement = (
        torch.empty_like(weight, device="meta")
        if mismatch == "device"
        else weight.detach().to(torch.float64)
    )
    module.register_parameter("weight", torch.nn.Parameter(replacement))
    with (
        patch.object(qwen_generation_mod, "QWEN3_30B_A3B_CONFIG", model.config),
        pytest.raises(ValueError, match=message),
    ):
        QwenGenerationProvider(
            model,
            _GenerationTokenizer(),
            "native:qwen3-moe:test",
            eos_token_ids=(),
        )


def test_generation_provider_uses_moe_logits_and_commits_sessions() -> None:
    model = _model()
    provider = _generation_provider(model)
    prompt = torch.tensor(((1, 4, 7),), dtype=torch.long)
    with torch.inference_mode():
        expected_first = int(torch.argmax(model(prompt)[0, -1]).item())

    events = tuple(
        provider.generate(
            _generation_request(provider, maximum=3, open_session=True),
            cancelled=lambda: False,
        )
    )
    assert isinstance(events[0], GenerationTokenEvent)
    assert events[0].token_id == expected_first
    terminal = cast(GenerationTerminalEvent, events[-1])
    handle = terminal.result.continuation
    assert handle is not None
    assert len(terminal.result.token_ids or ()) == 3
    assert [state.token_count for state in provider.cache.session_states()] == [6]

    continuation = tuple(
        provider.generate(
            _generation_request(provider, maximum=1, session=handle),
            cancelled=lambda: False,
        )
    )
    continued_terminal = cast(GenerationTerminalEvent, continuation[-1])
    assert continued_terminal.result.continuation == handle
    assert [state.token_count for state in provider.cache.session_states()] == [10]
    provider.close_session(handle)


def test_continuous_generation_batches_qwen3_moe_decode() -> None:
    model = _model()
    provider = _generation_provider(model)
    baseline = tuple(
        provider.generate(
            _generation_request(provider, maximum=3),
            cancelled=lambda: False,
        )
    )
    expected_tokens = cast(GenerationTerminalEvent, baseline[-1]).result.token_ids
    batch_sizes: list[int] = []
    hook = model.model.embed_tokens.register_forward_pre_hook(
        lambda _module, args: batch_sizes.append(cast(torch.Tensor, args[0]).shape[0])
    )
    try:
        with QwenContinuousGenerationProvider(
            provider,
            max_batch_size=2,
            slot_capacity=12,
            prefill_chunk_tokens=3,
            batch_wait_s=0.01,
        ) as scheduled:
            streams = tuple(
                scheduled.generate(
                    _generation_request(scheduled, maximum=3),
                    cancelled=lambda: False,
                )
                for _ in range(2)
            )
            barrier = threading.Barrier(2)

            def drain(index: int) -> tuple[object, ...]:
                barrier.wait()
                return tuple(streams[index])

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(executor.map(drain, range(2)))
    finally:
        hook.remove()

    assert all(isinstance(events[-1], GenerationTerminalEvent) for events in results)
    assert all(
        cast(GenerationTerminalEvent, events[-1]).result.token_ids == expected_tokens
        for events in results
    )
    assert 2 in batch_sizes


def test_generation_prefetches_selected_offloaded_moe_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    provider = _generation_provider(model)
    baseline = tuple(
        provider.generate(
            _generation_request(provider, maximum=1),
            cancelled=lambda: False,
        )
    )
    expected_token = cast(GenerationTokenEvent, baseline[0]).token_id
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    mechanism.partially_unload(mechanism.loaded_bytes())
    calls: list[tuple[torch.nn.Module, ...]] = []
    original = qwen3_moe_mod.make_prefetch_queue

    def capture(blocks: Iterable[torch.nn.Module]) -> object:
        selected = tuple(blocks)
        calls.append(selected)
        return original(selected)

    monkeypatch.setattr(qwen3_moe_mod, "make_prefetch_queue", capture)
    actual = tuple(
        provider.generate(
            _generation_request(provider, maximum=1),
            cancelled=lambda: False,
        )
    )

    assert cast(GenerationTokenEvent, actual[0]).token_id == expected_token
    assert len(calls) == model.config.num_hidden_layers
    assert all(call for call in calls)
    assert all(isinstance(expert, Qwen3MoeExpert) for call in calls for expert in call)
    assert not mechanism.loaded_unit_names()
