"""Explicit Qwen layer placement and device-local KV proofs."""

from __future__ import annotations

import logging
import threading
from collections.abc import MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import dinkster_inference_torch.anima_runtime as anima_runtime_module
import dinkster_inference_torch.qwen_generation as qwen_generation_module
import dinkster_inference_torch.qwen_layer_placement as qwen_layer_placement_module
import dinkster_inference_torch.qwen_text as qwen_text_module
import pytest
import torch
from dinkster_inference import (
    GenerationEvent,
    GenerationRequest,
    GenerationSessionBusyError,
    GenerationSessionClosedError,
    GenerationSessionHandle,
    GenerationStopConditions,
    GenerationTerminalEvent,
    GenerationTokenEvent,
    OvisPromptTokens,
    QwenTextConfig,
)
from dinkster_inference.patches import PatchSet
from dinkster_inference_torch import (
    INITLESS,
    DeviceMemory,
    MemoryPolicy,
    OvisTextEncoder,
    PagedKVSessionBusyError,
    PagedKVShapeError,
    QwenContinuousGenerationProvider,
    QwenGenerationProvider,
    QwenLayerPlacement,
    QwenLayerRange,
    QwenPagedKVCache,
    QwenTextModel,
    ResidencyManager,
    ResidencyUnit,
    ResidentWeights,
    StoredWeight,
    enroll_component_placement,
    enroll_qwen_layer_placement,
)
from dinkster_inference_torch.operations import bound_compute_device

CPU = torch.device("cpu")
CPU0 = torch.device("cpu", 0)


def _config() -> QwenTextConfig:
    return QwenTextConfig(
        architecture="anima_qwen3_06b",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        qkv_bias=False,
        qk_norm=True,
        prompt_template="{}",
        min_tokens=1,
        pad_token_id=0,
    )


def _model() -> QwenTextModel:
    torch.manual_seed(0x525)
    model = QwenTextModel(_config(), operations=INITLESS)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


def test_layer_ranges_require_complete_explicit_contiguous_devices() -> None:
    placement = QwenLayerPlacement(
        (
            QwenLayerRange(0, 2, torch.device("cuda", 0)),
            QwenLayerRange(2, 4, torch.device("cuda", 1)),
        )
    )
    assert placement.layer_count == 4
    assert placement.device_for_layer(0) == torch.device("cuda", 0)
    assert placement.device_for_layer(3) == torch.device("cuda", 1)
    assert QwenLayerRange(0, 1, torch.device("mps")).device == torch.device("mps")

    with pytest.raises(ValueError, match="explicit device index"):
        QwenLayerRange(0, 1, torch.device("cuda"))
    with pytest.raises(ValueError, match="CPU.*device index"):
        QwenLayerRange(0, 1, CPU0)
    with pytest.raises(ValueError, match="ordered and contiguous"):
        QwenLayerPlacement(
            (
                QwenLayerRange(0, 1, torch.device("cuda", 0)),
                QwenLayerRange(2, 3, torch.device("cuda", 1)),
            )
        )
    with pytest.raises(ValueError, match="one contiguous range"):
        QwenLayerPlacement(
            (
                QwenLayerRange(0, 1, torch.device("cuda", 0)),
                QwenLayerRange(1, 2, torch.device("cuda", 0)),
            )
        )
    with pytest.raises(ValueError, match="multi-range.*indexed CUDA"):
        QwenLayerPlacement(
            (
                QwenLayerRange(0, 1, CPU),
                QwenLayerRange(1, 2, torch.device("mps")),
            )
        )


def test_implicit_single_device_placement_retains_mps_support() -> None:
    model = _model()
    with patch.object(
        qwen_layer_placement_module,
        "_module_device",
        return_value=torch.device("mps"),
    ):
        placement = qwen_layer_placement_module.resolve_qwen_layer_placement(model)
    assert placement.ranges == (QwenLayerRange(0, 2, torch.device("mps")),)


def test_generic_component_placement_owns_disjoint_state_per_device() -> None:
    module = torch.nn.Sequential(
        INITLESS.linear(3, 4),
        INITLESS.linear(4, 2),
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(0.25)
    placement = enroll_component_placement(
        module,
        {"0": CPU, "1": CPU0},
        offload_device=CPU,
    )
    assert set(placement) == {CPU, CPU0}
    assert len(placement.mechanisms) == 2
    manager = ResidencyManager()
    manager.load(placement.mechanisms, force_full_load=True)
    assert bound_compute_device(module[0]) == CPU
    assert bound_compute_device(module[1]) == CPU0
    assert all(
        mechanism.loaded_bytes() == mechanism.total_bytes() for mechanism in placement.values()
    )


@pytest.mark.parametrize("device", ("mps", "xpu:0", "privateuseone:0"))
def test_generic_component_placement_passes_device_through_with_diagnostic(
    device: str, caplog: pytest.LogCaptureFixture
) -> None:
    module = INITLESS.linear(3, 4)
    with caplog.at_level(logging.WARNING):
        placement = enroll_component_placement(module, {"": device}, offload_device=CPU)
    target = torch.device(device)
    assert set(placement) == {target}
    assert placement[target].load_device == target
    assert bound_compute_device(module) == target
    assert f"placement on {device} uses the portable torch path" in caplog.text


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_generic_component_placement_executes_on_mps() -> None:
    module = INITLESS.linear(3, 4)
    with torch.no_grad():
        module.weight.fill_(0.25)
        module.bias.fill_(0.5)
    inputs = torch.ones(2, 3, device="mps")
    expected = torch.nn.functional.linear(inputs, module.weight.to("mps"), module.bias.to("mps"))
    placement = enroll_component_placement(module, {"": "mps"}, offload_device=CPU)
    ResidencyManager().load(placement.mechanisms, force_full_load=True)
    assert torch.equal(module(inputs), expected)
    assert module(inputs).device.type == "mps"


def test_generic_component_placement_refuses_tied_state_split_before_factory() -> None:
    module = torch.nn.Sequential(
        INITLESS.linear(3, 3),
        INITLESS.linear(3, 3),
    )
    module[1].weight = module[0].weight
    calls = 0

    def factory(*_args: object, **_kwargs: object) -> ResidentWeights:
        nonlocal calls
        calls += 1
        raise AssertionError("factory must not run")

    with pytest.raises(ValueError, match="tied state assigned to multiple devices"):
        enroll_component_placement(
            module,
            {"0": CPU, "1": CPU0},
            offload_device=CPU,
            mechanism_factory=factory,
        )
    assert calls == 0
    assert not hasattr(module, "_dinkster_resident_weights")
    assert module[0].residency_binding() is None  # type: ignore[attr-defined]
    assert module[1].residency_binding() is None  # type: ignore[attr-defined]


def test_generic_component_placement_factory_failure_unloads_and_restores() -> None:
    module = torch.nn.Sequential(
        INITLESS.linear(3, 4),
        INITLESS.linear(4, 2),
    )
    original = tuple(module.parameters())
    constructed: list[ResidentWeights] = []
    calls = 0

    class TrackingResidentWeights(ResidentWeights):
        unload_calls = 0

        def unload(self) -> None:
            self.unload_calls += 1
            super().unload()

    def factory(
        weights: MutableMapping[str, StoredWeight],
        *,
        load_device: torch.device | str,
        offload_device: torch.device | str,
        patch_set: PatchSet[torch.Tensor] | None = None,
        units: Sequence[ResidencyUnit] | None = None,
        intermediate_dtype: torch.dtype = torch.float32,
    ) -> ResidentWeights:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected factory failure")
        mechanism = TrackingResidentWeights(
            weights,
            load_device=load_device,
            offload_device=offload_device,
            patch_set=patch_set,
            units=units,
            intermediate_dtype=intermediate_dtype,
        )
        constructed.append(mechanism)
        return mechanism

    with pytest.raises(RuntimeError, match="injected factory failure"):
        enroll_component_placement(
            module,
            {"0": CPU, "1": CPU0},
            offload_device=CPU,
            mechanism_factory=factory,
        )
    assert all(
        actual is expected for actual, expected in zip(module.parameters(), original, strict=True)
    )
    assert not hasattr(module, "_dinkster_resident_weights")
    assert module[0].residency_binding() is None  # type: ignore[attr-defined]
    assert module[1].residency_binding() is None  # type: ignore[attr-defined]
    assert len(constructed) == 1
    assert cast(TrackingResidentWeights, constructed[0]).unload_calls == 1


def test_generic_component_placement_bind_failure_restores_prior_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = torch.nn.Sequential(
        INITLESS.linear(3, 4),
        INITLESS.linear(4, 2),
    )
    original = tuple(module.parameters())
    previous = cast(Any, object())
    module[1].__dict__["_residency"] = previous

    def fail_bind(binding: object) -> None:
        module[1].__dict__["_residency"] = binding
        raise RuntimeError("injected bind failure")

    monkeypatch.setattr(module[1], "bind_residency", fail_bind)
    with pytest.raises(RuntimeError, match="injected bind failure"):
        enroll_component_placement(
            module,
            {"0": CPU, "1": CPU0},
            offload_device=CPU,
        )
    assert all(
        actual is expected for actual, expected in zip(module.parameters(), original, strict=True)
    )
    assert not hasattr(module, "_dinkster_resident_weights")
    assert "_residency" not in module[0].__dict__
    assert module[1].__dict__.get("_residency") is previous


def test_one_device_qwen_placement_preserves_exact_outputs() -> None:
    model = _model()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention = torch.tensor([[1, 1, 0]], dtype=torch.long)
    expected_encoded = model(ids, attention)
    expected_hidden, expected_kv = model.forward_causal(ids)
    placement = QwenLayerPlacement((QwenLayerRange(0, 2, CPU),))
    enrolled = enroll_qwen_layer_placement(model, placement, offload_device=CPU)
    ResidencyManager().load(enrolled.mechanisms, force_full_load=True)

    assert torch.equal(model(ids, attention), expected_encoded)
    actual_hidden, actual_kv = model.forward_causal(ids)
    assert torch.equal(actual_hidden, expected_hidden)
    assert all(
        torch.equal(actual, expected)
        for actual_pair, expected_pair in zip(actual_kv, expected_kv, strict=True)
        for actual, expected in zip(actual_pair, expected_pair, strict=True)
    )


def test_causal_frequencies_refuse_empty_or_malformed_sequences() -> None:
    model = _model()
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="cannot be empty"):
        model.forward_causal(ids, frequencies=cast(Any, ()))
    with pytest.raises(ValueError, match="one span or one span per model layer"):
        model.forward_causal(ids, frequencies=cast(Any, (torch.ones(1),)))


def test_partitioned_kv_exposes_one_manager_mechanism_per_range() -> None:
    first = torch.device("cuda", 0)
    second = torch.device("cuda", 1)
    placement = QwenLayerPlacement(
        (
            QwenLayerRange(0, 1, first),
            QwenLayerRange(1, 2, second),
        )
    )
    cache = QwenPagedKVCache(
        "qwen-test",
        "model-test",
        placement,
        block_tokens=4,
        kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        max_device_blocks=4,
    )
    assert tuple(mechanism.load_device for mechanism in cache.residency_mechanisms) == (
        first,
        second,
    )
    cache.create_session("session")
    with cache.pin("session") as lease:
        assert lease.token_count == 0
    assert cache.session_states()[0].token_count == 0
    cache.close_session("session")
    assert cache.session_states() == ()


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [1, 2 + sum(text.encode("utf-8")) % 8]

    def decode_bytes(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> bytes:
        del skip_special_tokens
        return bytes(65 + token_id for token_id in token_ids)


def _two_gpu_provider(
    *,
    max_device_blocks: int = 8,
    max_host_blocks: int = 0,
) -> QwenGenerationProvider:
    first = torch.device("cuda", 0)
    second = torch.device("cuda", 1)
    model = _model()
    placement = QwenLayerPlacement(
        (
            QwenLayerRange(0, 1, first),
            QwenLayerRange(1, 2, second),
        )
    )
    enrolled = enroll_qwen_layer_placement(model, placement, offload_device=CPU)
    manager = ResidencyManager()
    manager.load(enrolled.mechanisms, force_full_load=True)
    with patch.object(qwen_generation_module, "ANIMA_QWEN3_06B_CONFIG", model.config):
        provider = QwenGenerationProvider(
            model,
            _Tokenizer(),
            "qwen-test",
            eos_token_ids=(),
            block_tokens=4,
            max_device_blocks=max_device_blocks,
            max_host_blocks=max_host_blocks,
        )
    manager.load(provider.cache_residency_mechanisms)
    return provider


def _request(
    provider: QwenGenerationProvider | QwenContinuousGenerationProvider,
    prompt: str,
    maximum: int,
    *,
    open_session: bool = False,
    session: GenerationSessionHandle | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        provider.id,
        "qwen-test",
        prompt=prompt,
        stop=GenerationStopConditions(maximum),
        open_session=open_session,
        session=session,
    )


def _token_ids(events: Sequence[GenerationEvent]) -> tuple[int, ...]:
    return cast(tuple[int, ...], cast(GenerationTerminalEvent, events[-1]).result.token_ids)


def _drain_concurrently(
    streams: Sequence[object],
) -> list[tuple[GenerationEvent, ...]]:
    barrier = threading.Barrier(len(streams))

    def drain(stream: object) -> tuple[GenerationEvent, ...]:
        barrier.wait()
        return tuple(cast(Any, stream))

    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        futures = [executor.submit(drain, stream) for stream in streams]
        return [future.result(timeout=10.0) for future in futures]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_direct_and_continuous_generation_are_bit_exact() -> None:
    first = torch.device("cuda", 0)
    second = torch.device("cuda", 1)
    if torch.cuda.get_device_name(first) != torch.cuda.get_device_name(second):
        pytest.skip("requires homogeneous CUDA devices")
    baseline = _model().to(first)
    placed = _model()
    placed.load_state_dict(baseline.state_dict())
    placement = QwenLayerPlacement(
        (
            QwenLayerRange(0, 1, first),
            QwenLayerRange(1, 2, second),
        )
    )
    enrolled = enroll_qwen_layer_placement(placed, placement, offload_device=CPU)
    manager = ResidencyManager()
    manager.load(enrolled.mechanisms, force_full_load=True)
    assert anima_runtime_module._module_compute_device(placed) == first  # pyright: ignore[reportPrivateUsage]
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=first)
    attention = torch.tensor([[1, 1, 0]], dtype=torch.long, device=first)
    expected_encoded = baseline(ids, attention)
    actual_encoded = placed(ids, attention)
    assert torch.equal(actual_encoded.to(first), expected_encoded)
    expected, _ = baseline.forward_causal(ids)
    actual, key_values = placed.forward_causal(ids)
    assert torch.equal(actual.to(first), expected)
    assert tuple(key.device for key, _value in key_values) == (first, second)

    with patch.object(qwen_generation_module, "ANIMA_QWEN3_06B_CONFIG", placed.config):
        direct = QwenGenerationProvider(
            placed,
            _Tokenizer(),
            "qwen-test",
            eos_token_ids=(),
            block_tokens=4,
            max_device_blocks=8,
        )
    manager.load(direct.cache_residency_mechanisms)
    request = GenerationRequest(
        direct.id,
        "qwen-test",
        prompt="prompt",
        stop=GenerationStopConditions(2),
        open_session=True,
    )
    direct_events = tuple(direct.generate(request, cancelled=lambda: False))
    with QwenContinuousGenerationProvider(
        direct,
        max_batch_size=2,
        slot_capacity=8,
        batch_wait_s=0.0,
    ) as continuous:
        scheduled_events = tuple(continuous.generate(request, cancelled=lambda: False))
    direct_result = cast(GenerationTerminalEvent, direct_events[-1]).result
    scheduled_result = cast(GenerationTerminalEvent, scheduled_events[-1]).result
    assert direct_result.text == scheduled_result.text
    assert direct_result.token_ids == scheduled_result.token_ids
    assert direct_result.finish_reason is scheduled_result.finish_reason
    assert direct_result.continuation is not None
    assert scheduled_result.continuation is not None
    direct.close_session(direct_result.continuation)
    direct.close_session(scheduled_result.continuation)
    assert len(direct.cache_residency_mechanisms) == 2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_ovis_encoding_routes_zero_mask_to_output_device() -> None:
    first = torch.device("cuda", 0)
    second = torch.device("cuda", 1)
    config = replace(_config(), architecture="ovis_qwen3_2b", zero_masked=True)
    torch.manual_seed(0x525)
    baseline = QwenTextModel(config, operations=INITLESS)
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.uniform_(-0.1, 0.1)
    baseline.to(first)
    placed = QwenTextModel(config, operations=INITLESS)
    placed.load_state_dict(baseline.state_dict())
    enrolled = enroll_qwen_layer_placement(
        placed,
        QwenLayerPlacement(
            (
                QwenLayerRange(0, 1, first),
                QwenLayerRange(1, 2, second),
            )
        ),
        offload_device=CPU,
    )
    ResidencyManager().load(enrolled.mechanisms, force_full_load=True)
    tokens = OvisPromptTokens((1, 2, 3), (1, 1, 0), 0)
    with patch.object(qwen_text_module, "tokenize_ovis_prompt", return_value=tokens):
        expected = OvisTextEncoder(baseline).encode("prompt").embeddings
        actual = OvisTextEncoder(placed).encode("prompt").embeddings
    assert actual.device == second
    assert torch.equal(actual.to(first), expected)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_kv_append_failure_rolls_back_every_partition() -> None:
    first = torch.device("cuda", 0)
    second = torch.device("cuda", 1)
    placement = QwenLayerPlacement(
        (
            QwenLayerRange(0, 1, first),
            QwenLayerRange(1, 2, second),
        )
    )
    cache = QwenPagedKVCache(
        "qwen-test",
        "model-test",
        placement,
        block_tokens=4,
        kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        max_device_blocks=4,
    )
    cache.create_session("session")
    good = torch.ones((1, 1, 2, 4), device=first)
    wrong_device = torch.ones((1, 1, 2, 4), device=first)
    with pytest.raises(PagedKVShapeError, match="device"):
        cache.pin("session").append_layers(
            (
                (good, good.clone()),
                (wrong_device, wrong_device.clone()),
            )
        )
    state = cache.session_states()[0]
    assert state.token_count == 0
    assert state.active is False
    assert cache.block_states() == ()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_manager_full_offload_preserves_and_remove_releases_session() -> None:
    first = torch.device("cuda", 0)
    second = torch.device("cuda", 1)
    cache = QwenPagedKVCache(
        "qwen-test",
        "model-test",
        QwenLayerPlacement(
            (
                QwenLayerRange(0, 1, first),
                QwenLayerRange(1, 2, second),
            )
        ),
        block_tokens=4,
        kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        max_device_blocks=1,
        max_host_blocks=1,
    )
    cache.create_session("session")
    first_kv = torch.arange(8, dtype=torch.float32, device=first).reshape(1, 1, 2, 4)
    second_kv = (first_kv + 100).to(second)
    with cache.pin("session") as lease:
        lease.append_layers(
            (
                (first_kv, first_kv + 10),
                (second_kv, second_kv + 10),
            )
        )
        lease.commit()
    mechanisms = cache.residency_mechanisms
    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            min_weight_memory_ratio=0.0,
            load_inflation=1.0,
        ),
        free_memory=lambda _device: DeviceMemory(free_total=0, free_torch=0),
        total_memory=lambda device: next(
            mechanism.loaded_bytes() for mechanism in mechanisms if mechanism.load_device == device
        ),
        empty_cache=lambda _device: None,
    )
    for mechanism in mechanisms:
        manager._touch(mechanism)  # pyright: ignore[reportPrivateUsage]

    for mechanism in mechanisms:
        manager.free(2 * mechanism.loaded_bytes(), mechanism.load_device)

    assert manager.registered() == tuple(reversed(mechanisms))
    assert cache.loaded_bytes() == 0
    assert cache.offloaded_bytes() == cache.total_bytes()
    assert cache.session_states()[0].host_block_count == 2
    with cache.pin("session") as lease:
        assert torch.equal(lease.block_views(0)[0].key, first_kv[0].transpose(0, 1))
        assert torch.equal(lease.block_views(1)[0].key, second_kv[0].transpose(0, 1))
    manager.remove(mechanisms)
    assert manager.registered() == ()
    assert cache.session_states() == ()
    assert cache.total_bytes() == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_manager_refuses_mixed_active_pressure_atomically() -> None:
    first = torch.device("cuda", 0)
    second = torch.device("cuda", 1)
    cache = QwenPagedKVCache(
        "qwen-test",
        "model-test",
        QwenLayerPlacement(
            (
                QwenLayerRange(0, 1, first),
                QwenLayerRange(1, 2, second),
            )
        ),
        block_tokens=4,
        kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        max_device_blocks=2,
        max_host_blocks=2,
    )
    first_kv = torch.arange(8, dtype=torch.float32, device=first).reshape(1, 1, 2, 4)
    second_kv = (first_kv + 100).to(second)
    for session_id, offset in (("active", 0), ("idle", 200)):
        cache.create_session(session_id)
        with cache.pin(session_id) as lease:
            lease.append_layers(
                (
                    (first_kv + offset, first_kv + offset + 10),
                    (second_kv + offset, second_kv + offset + 10),
                )
            )
            lease.commit()
    mechanisms = cache.residency_mechanisms
    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            min_weight_memory_ratio=0.0,
            load_inflation=1.0,
        ),
        free_memory=lambda _device: DeviceMemory(free_total=0, free_torch=0),
        total_memory=lambda device: next(
            mechanism.loaded_bytes() for mechanism in mechanisms if mechanism.load_device == device
        ),
        empty_cache=lambda _device: None,
    )
    for mechanism in mechanisms:
        manager._touch(mechanism)  # pyright: ignore[reportPrivateUsage]
    active = cache.pin("active")
    before_sessions = cache.session_states()
    before_blocks = cache.block_states()
    mechanism = mechanisms[0]
    assert mechanism.partial_unload_capacity() == 0

    with pytest.raises(PagedKVSessionBusyError, match="active sessions"):
        manager.free(mechanism.loaded_bytes() // 4, mechanism.load_device)

    assert cache.session_states() == before_sessions
    assert cache.block_states() == before_blocks
    assert manager.registered() == tuple(reversed(mechanisms))
    active.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_sessions_fork_cancel_offload_restore_and_close_atomically() -> None:
    provider = _two_gpu_provider(max_host_blocks=8)
    opened = tuple(
        provider.generate(
            _request(provider, "open", 2, open_session=True),
            cancelled=lambda: False,
        )
    )
    handle = cast(GenerationTerminalEvent, opened[-1]).result.continuation
    assert handle is not None
    committed = provider.cache.session_states()[0].token_count
    assert committed == 4
    fork = provider.fork_session(handle, token_count=3)
    assert sorted(state.token_count for state in provider.cache.session_states()) == [3, 4]

    active = provider.generate(
        _request(provider, "active", 1, session=handle),
        cancelled=lambda: False,
    )
    with pytest.raises(GenerationSessionBusyError, match="active"):
        provider.close_session(handle)
    with pytest.raises(PagedKVSessionBusyError, match="active"):
        provider.cache_residency_mechanisms[0].unload()
    active.close()
    assert sorted(state.token_count for state in provider.cache.session_states()) == [3, 4]

    cancelled = tuple(
        provider.generate(
            _request(provider, "cancel", 1, session=handle),
            cancelled=lambda: True,
        )
    )
    terminal = cast(GenerationTerminalEvent, cancelled[-1])
    assert terminal.result.continuation == handle
    assert sorted(state.token_count for state in provider.cache.session_states()) == [3, committed]

    for mechanism in provider.cache_residency_mechanisms:
        loaded = mechanism.loaded_bytes()
        assert loaded > 0
        assert mechanism.partially_unload(loaded) == loaded
    assert provider.cache.loaded_bytes() == 0
    assert provider.cache.offloaded_bytes() == provider.cache.total_bytes()
    assert all(state.host_block_count == 2 for state in provider.cache.session_states())

    restored = tuple(
        provider.generate(
            _request(provider, "restore", 1, session=handle),
            cancelled=lambda: True,
        )
    )
    assert cast(GenerationTerminalEvent, restored[-1]).result.continuation == handle
    assert provider.cache.offloaded_bytes() == 0
    assert provider.cache.loaded_bytes() == provider.cache.total_bytes()

    provider.close_session(fork)
    provider.close_session(handle)
    assert provider.cache.session_states() == ()
    assert provider.cache.block_states() == ()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_manager_eviction_removes_all_session_partitions() -> None:
    provider = _two_gpu_provider()
    opened = tuple(
        provider.generate(
            _request(provider, "evict", 1, open_session=True),
            cancelled=lambda: False,
        )
    )
    handle = cast(GenerationTerminalEvent, opened[-1]).result.continuation
    assert handle is not None
    mechanism = provider.cache_residency_mechanisms[0]
    loaded = mechanism.loaded_bytes()
    assert loaded > 0
    assert mechanism.partially_load(-loaded) == -loaded
    assert provider.cache.session_states() == ()
    assert provider.cache.block_states() == ()
    with pytest.raises(GenerationSessionClosedError, match="closed or unknown"):
        provider.generate(
            _request(provider, "after-eviction", 1, session=handle),
            cancelled=lambda: False,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_capacity_eviction_removes_the_same_session_from_every_partition() -> None:
    provider = _two_gpu_provider(max_device_blocks=1)
    first = tuple(
        provider.generate(
            _request(provider, "first", 1, open_session=True),
            cancelled=lambda: False,
        )
    )
    first_handle = cast(GenerationTerminalEvent, first[-1]).result.continuation
    assert first_handle is not None

    second = tuple(
        provider.generate(
            _request(provider, "second", 1, open_session=True),
            cancelled=lambda: False,
        )
    )
    second_handle = cast(GenerationTerminalEvent, second[-1]).result.continuation
    assert second_handle is not None
    assert len(provider.cache.session_states()) == 1
    assert len(provider.cache.block_states()) == 2
    with pytest.raises(GenerationSessionClosedError, match="closed or unknown"):
        provider.generate(
            _request(provider, "evicted", 1, session=first_handle),
            cancelled=lambda: False,
        )
    provider.close_session(second_handle)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_two_gpu_continuous_slot_compaction_preserves_stream_outputs() -> None:
    provider = _two_gpu_provider()
    expected = {
        prompt: _token_ids(
            tuple(
                provider.generate(
                    _request(provider, prompt, 4),
                    cancelled=lambda: False,
                )
            )
        )
        for prompt in ("one", "three")
    }
    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=3,
        slot_capacity=12,
        batch_wait_s=0.01,
    ) as continuous:
        streams = [
            continuous.generate(
                _request(continuous, prompt, 4),
                cancelled=lambda: False,
            )
            for prompt in ("one", "two", "three")
        ]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(next, stream) for stream in streams]
            first_events = [future.result(timeout=10.0) for future in futures]
        streams[1].close()
        tails = _drain_concurrently((streams[0], streams[2]))

    actual = {
        prompt: (
            cast(GenerationTokenEvent, first_event).token_id,
            *_token_ids(tail)[1:],
        )
        for prompt, first_event, tail in zip(
            ("one", "three"),
            (first_events[0], first_events[2]),
            tails,
            strict=True,
        )
    }
    assert actual == expected
