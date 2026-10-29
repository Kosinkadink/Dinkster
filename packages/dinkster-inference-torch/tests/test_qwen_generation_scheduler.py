"""Continuous native Qwen generation scheduling proofs."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import patch

import dinkster_inference_torch.qwen_generation as qwen_generation_module
import dinkster_inference_torch.qwen_generation_scheduler as qwen_generation_scheduler_module
import pytest
import torch
from dinkster_inference import (
    GenerationEvent,
    GenerationFinishReason,
    GenerationRequest,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSamplerStage,
    GenerationStopConditions,
    GenerationStream,
    GenerationTerminalEvent,
    GenerationTokenEvent,
    QwenTextConfig,
)
from dinkster_inference_torch import (
    QwenContinuousGenerationProvider,
    QwenGenerationProvider,
)

MODEL_ID = "native:qwen:continuous-test"


class _Tokenizer:
    _prompts = {
        "p": [1, 2],
        "long-a": [1, 1, 1, 1, 1, 1],
        "long-b": [2, 2, 2, 2, 2, 2],
        "one": [1],
        "two": [2],
        "three": [3],
        "fifty": [2] * 50,
    }

    def encode(self, text: str) -> list[int]:
        return list(self._prompts.get(text, ()))

    def decode_bytes(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> bytes:
        del skip_special_tokens
        return bytes(65 + token_id % 26 for token_id in token_ids)


class _BatchModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = QwenTextConfig(
            architecture="anima_qwen3_06b",
            vocab_size=32,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            qkv_bias=False,
            qk_norm=True,
            prompt_template="{}",
            min_tokens=1,
            pad_token_id=0,
        )
        self.embed_tokens = torch.nn.Embedding(32, 4)
        self.layers = torch.nn.ModuleList((torch.nn.Identity(),))
        self.calls: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
        self.first_call_started = threading.Event()
        self.release_first_call = threading.Event()
        self.block_first_call = False
        self._call_lock = threading.Lock()

    def forward_causal(
        self,
        ids: torch.Tensor,
        cache_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        cache_position: int,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prefetch: bool,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        del prefetch
        assert frequencies[0].shape[2] == ids.shape[1]
        rows = tuple(tuple(int(value) for value in row) for row in ids.tolist())
        with self._call_lock:
            first = not self.calls
            self.calls.append((cache_position, rows))
        if first and self.block_first_call:
            self.first_call_started.set()
            if not self.release_first_call.wait(5.0):
                raise TimeoutError("test did not release the first model call")

        batch, length = ids.shape
        cache_key, cache_value = cache_key_values[0]
        prefix = cache_key[:, 0, :cache_position, 0].sum(dim=1, keepdim=True)
        totals = prefix + ids.to(torch.float32).cumsum(dim=1)
        hidden = totals.unsqueeze(-1).expand(batch, length, self.config.hidden_size)
        key = (
            ids.to(torch.float32)
            .reshape(batch, 1, length, 1)
            .expand(batch, 1, length, self.config.head_dim)
        )
        value = key + 0.5
        end = cache_position + length
        cache_key[:, :, cache_position:end].copy_(key)
        cache_value[:, :, cache_position:end].copy_(value)
        return hidden, ((key, value),)

    def causal_frequencies(
        self,
        length: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((1, 1, length, 4), dtype=torch.float32, device=device),
            torch.zeros((1, 1, length, 2), dtype=torch.float32, device=device),
            torch.zeros((1, 1, length, 2), dtype=torch.float32, device=device),
        )

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        centers = hidden[..., :1].to(torch.long).remainder(20).add(4)
        vocabulary = torch.arange(32, dtype=torch.float32, device=hidden.device)
        return -(vocabulary - centers).abs().mul(0.7)


def _provider() -> tuple[QwenGenerationProvider, _BatchModel]:
    model = _BatchModel()
    with patch.object(qwen_generation_module, "ANIMA_QWEN3_06B_CONFIG", model.config):
        provider = QwenGenerationProvider(
            cast("Any", model),
            _Tokenizer(),
            MODEL_ID,
            eos_token_ids=(),
            block_tokens=4,
            max_device_blocks=32,
        )
    return provider, model


def _request(
    provider: QwenGenerationProvider | QwenContinuousGenerationProvider,
    *,
    prompt: str = "p",
    maximum: int = 3,
    sampler: GenerationSamplerChain | None = None,
    seed: int | None = None,
    open_session: bool = False,
    session: object = None,
) -> GenerationRequest:
    kwargs: dict[str, object] = {}
    if sampler is not None:
        kwargs["sampler"] = sampler
    return GenerationRequest(
        provider.id,
        MODEL_ID,
        prompt=prompt,
        stop=GenerationStopConditions(maximum),
        seed=seed,
        open_session=open_session,
        session=cast("Any", session),
        **cast("Any", kwargs),
    )


def _drain_concurrently(streams: Sequence[GenerationStream]) -> list[tuple[GenerationEvent, ...]]:
    barrier = threading.Barrier(len(streams))

    def drain(stream: GenerationStream) -> tuple[GenerationEvent, ...]:
        barrier.wait()
        return tuple(stream)

    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        futures = [executor.submit(drain, stream) for stream in streams]
        return [future.result(timeout=10.0) for future in futures]


def _token_ids(events: Sequence[GenerationEvent]) -> tuple[int, ...]:
    terminal = cast(GenerationTerminalEvent, events[-1])
    return cast(tuple[int, ...], terminal.result.token_ids)


def _wait_for_condition_acquirer(provider: QwenContinuousGenerationProvider) -> None:
    deadline = qwen_generation_scheduler_module.time.perf_counter() + 2.0
    while True:
        with provider._condition_queue_lock:  # pyright: ignore[reportPrivateUsage]
            queued = provider._queued_condition_acquirers  # pyright: ignore[reportPrivateUsage]
        if queued:
            return
        assert qwen_generation_scheduler_module.time.perf_counter() < deadline
        threading.Event().wait(0.001)


@pytest.mark.parametrize(
    ("sampler", "seed"),
    (
        (None, None),
        (
            GenerationSamplerChain((GenerationSamplerStage(GenerationSamplerKind.MULTINOMIAL),)),
            1234,
        ),
    ),
)
def test_four_stream_decode_batches_preserve_solo_tokens(
    sampler: GenerationSamplerChain | None,
    seed: int | None,
) -> None:
    solo_provider, _ = _provider()
    expected = [
        _token_ids(
            tuple(
                solo_provider.generate(
                    _request(solo_provider, maximum=5, sampler=sampler, seed=seed),
                    cancelled=lambda: False,
                )
            )
        )
        for _ in range(4)
    ]

    provider, model = _provider()
    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=4,
        slot_capacity=16,
        prefill_chunk_tokens=1,
        max_decode_batches_before_prefill=1,
        batch_wait_s=0.01,
    ) as scheduled:
        streams = [
            scheduled.generate(
                _request(scheduled, maximum=5, sampler=sampler, seed=seed),
                cancelled=lambda: False,
            )
            for _ in range(4)
        ]
        actual = [_token_ids(events) for events in _drain_concurrently(streams)]

    assert actual == expected
    assert any(len(rows) == 4 for _, rows in model.calls), model.calls


@pytest.mark.parametrize(("open_session", "expected_forwards"), ((False, 3), (True, 4)))
def test_continuous_timer_stops_after_one_synchronization_per_model_batch(
    monkeypatch: pytest.MonkeyPatch,
    open_session: bool,
    expected_forwards: int,
) -> None:
    provider, _model = _provider()
    original_forward = provider._forward_logits  # pyright: ignore[reportPrivateUsage]
    original_timer = qwen_generation_scheduler_module.time.perf_counter
    state = {"pending": False, "forwards": 0, "synchronizations": 0}

    def forward_logits(*args: Any, **kwargs: Any) -> Any:
        assert not state["pending"]
        result = original_forward(*args, **kwargs)
        state["pending"] = True
        state["forwards"] += 1
        return result

    def synchronize(_self: Any) -> None:
        state["synchronizations"] += 1
        if state["pending"]:
            state["pending"] = False

    def perf_counter() -> float:
        assert not state["pending"]
        return original_timer()

    monkeypatch.setattr(provider, "_forward_logits", forward_logits)
    monkeypatch.setattr(
        qwen_generation_module._QwenGenerationStream,  # pyright: ignore[reportPrivateUsage]
        "_synchronize",
        synchronize,
    )
    monkeypatch.setattr(qwen_generation_scheduler_module.time, "perf_counter", perf_counter)

    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=2,
        slot_capacity=8,
        prefill_chunk_tokens=2,
        batch_wait_s=0.01,
    ) as scheduled:
        streams = [
            scheduled.generate(
                _request(scheduled, maximum=2, open_session=open_session),
                cancelled=lambda: False,
            )
            for _ in range(2)
        ]
        results = _drain_concurrently(streams)

    assert all(
        cast(GenerationTerminalEvent, events[-1]).result.stats.decode_time_s is not None
        for events in results
    )
    assert state["forwards"] == expected_forwards
    assert state["synchronizations"] == state["forwards"]
    assert not state["pending"]
    if open_session:
        for events in results:
            handle = cast(GenerationTerminalEvent, events[-1]).result.continuation
            assert handle is not None
            provider.close_session(handle)


def test_chunked_prefill_yields_to_a_waiting_stream() -> None:
    provider, model = _provider()
    model.block_first_call = True
    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=2,
        slot_capacity=16,
        prefill_chunk_tokens=2,
        batch_wait_s=0.0,
    ) as scheduled:
        first = scheduled.generate(
            _request(scheduled, prompt="long-a", maximum=1),
            cancelled=lambda: False,
        )
        second = scheduled.generate(
            _request(scheduled, prompt="long-b", maximum=1),
            cancelled=lambda: False,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(tuple, first)
            assert model.first_call_started.wait(2.0)
            second_future = executor.submit(tuple, second)
            _wait_for_condition_acquirer(scheduled)
            model.release_first_call.set()
            first_events = first_future.result(timeout=10.0)
            second_events = second_future.result(timeout=10.0)

    assert _token_ids(first_events)
    assert _token_ids(second_events)
    assert model.calls[:2] == [
        (0, ((1, 1),)),
        (0, ((2, 2),)),
    ]


def test_close_interrupts_chunked_prefill_at_a_chunk_boundary() -> None:
    provider, model = _provider()
    model.block_first_call = True
    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=2,
        slot_capacity=16,
        prefill_chunk_tokens=2,
        batch_wait_s=0.0,
    ) as scheduled:
        stream = scheduled.generate(
            _request(scheduled, prompt="long-a", maximum=1),
            cancelled=lambda: False,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            stream_future = executor.submit(tuple, stream)
            assert model.first_call_started.wait(2.0)
            close_future = executor.submit(scheduled.close)
            _wait_for_condition_acquirer(scheduled)
            model.release_first_call.set()
            close_future.result(timeout=10.0)
            with pytest.raises(RuntimeError, match="provider closed"):
                stream_future.result(timeout=10.0)

    assert model.calls == [(0, ((1, 1),))]


def test_unequal_decode_cohorts_take_bounded_round_robin_turns() -> None:
    provider, model = _provider()
    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=2,
        slot_capacity=64,
        prefill_chunk_tokens=64,
        batch_wait_s=0.01,
    ) as scheduled:
        short = scheduled.generate(
            _request(scheduled, prompt="one", maximum=8),
            cancelled=lambda: False,
        )
        long = scheduled.generate(
            _request(scheduled, prompt="fifty", maximum=2),
            cancelled=lambda: False,
        )
        assert isinstance(next(short), GenerationTokenEvent)
        assert isinstance(next(long), GenerationTokenEvent)

        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting_long = executor.submit(next, long)
            while not cast("Any", long)._waiting:
                threading.Event().wait(0.001)
            short_events = tuple(short)
            assert isinstance(waiting_long.result(timeout=2.0), GenerationTokenEvent)
        long.close()

    assert isinstance(short_events[-1], GenerationTerminalEvent)
    positions = [position for position, _rows in model.calls]
    assert positions[:4] == [0, 0, 1, 50]


def test_cancellation_rolls_back_and_releases_a_session_for_continuation() -> None:
    provider, _ = _provider()
    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=2,
        slot_capacity=24,
        batch_wait_s=0.0,
    ) as scheduled:
        opened = tuple(
            scheduled.generate(
                _request(scheduled, maximum=2, open_session=True),
                cancelled=lambda: False,
            )
        )
        handle = cast(GenerationTerminalEvent, opened[-1]).result.continuation
        assert handle is not None
        committed = provider.cache.session_states()[0].token_count

        cancel = threading.Event()
        stream = scheduled.generate(
            _request(scheduled, prompt="one", maximum=3, session=handle),
            cancelled=cancel.is_set,
        )
        assert isinstance(next(stream), GenerationTokenEvent)
        cancel.set()
        terminal = cast(GenerationTerminalEvent, next(stream))
        assert terminal.result.finish_reason is GenerationFinishReason.CANCELLED
        assert terminal.result.continuation == handle
        assert provider.cache.session_states()[0].token_count == committed

        continued = tuple(
            scheduled.generate(
                _request(scheduled, prompt="two", maximum=1, session=handle),
                cancelled=lambda: False,
            )
        )
        assert cast(GenerationTerminalEvent, continued[-1]).result.continuation == handle


def test_independent_sessions_continue_in_one_decode_batch() -> None:
    provider, model = _provider()
    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=2,
        slot_capacity=24,
        batch_wait_s=0.01,
    ) as scheduled:
        handles = []
        for prompt in ("one", "two"):
            events = tuple(
                scheduled.generate(
                    _request(scheduled, prompt=prompt, maximum=1, open_session=True),
                    cancelled=lambda: False,
                )
            )
            handle = cast(GenerationTerminalEvent, events[-1]).result.continuation
            assert handle is not None
            handles.append(handle)
        before = sorted(state.token_count for state in provider.cache.session_states())
        streams = [
            scheduled.generate(
                _request(scheduled, prompt="p", maximum=2, session=handle),
                cancelled=lambda: False,
            )
            for handle in handles
        ]
        results = _drain_concurrently(streams)

    terminals = [cast(GenerationTerminalEvent, events[-1]) for events in results]
    assert [terminal.result.continuation for terminal in terminals] == handles
    assert sorted(state.token_count for state in provider.cache.session_states()) == [
        count + 4 for count in before
    ]
    assert any(len(rows) == 2 for position, rows in model.calls if position > 1)
    for handle in handles:
        provider.close_session(handle)


def test_capacity_slot_admission_and_idempotent_abandonment() -> None:
    provider, _ = _provider()
    scheduled = QwenContinuousGenerationProvider(
        provider,
        max_batch_size=2,
        slot_capacity=4,
        batch_wait_s=0.0,
    )
    with pytest.raises(ValueError, match="requires 5 cache positions"):
        scheduled.generate(
            _request(scheduled, maximum=3),
            cancelled=lambda: False,
        )
    first = scheduled.generate(
        _request(scheduled, maximum=2),
        cancelled=lambda: False,
    )
    second = scheduled.generate(
        _request(scheduled, maximum=2),
        cancelled=lambda: False,
    )
    with pytest.raises(RuntimeError, match="no available request slot"):
        scheduled.generate(
            _request(scheduled, maximum=2),
            cancelled=lambda: False,
        )
    first.close()
    first.close()
    replacement = scheduled.generate(
        _request(scheduled, maximum=2),
        cancelled=lambda: False,
    )
    second.close()
    replacement.close()
    assert scheduled.active_stream_count == 0
    assert provider.cache.session_states() == ()
    scheduled.close()


def test_slot_compaction_preserves_remaining_stream_cache_isolation() -> None:
    solo_provider, _ = _provider()
    expected = {
        prompt: _token_ids(
            tuple(
                solo_provider.generate(
                    _request(solo_provider, prompt=prompt, maximum=4),
                    cancelled=lambda: False,
                )
            )
        )
        for prompt in ("one", "three")
    }

    provider, _ = _provider()
    with QwenContinuousGenerationProvider(
        provider,
        max_batch_size=3,
        slot_capacity=12,
        batch_wait_s=0.01,
    ) as scheduled:
        streams = [
            scheduled.generate(
                _request(scheduled, prompt=prompt, maximum=4),
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


def test_close_wakes_a_blocked_consumer_and_releases_fixed_cache() -> None:
    provider, _ = _provider()
    scheduled = QwenContinuousGenerationProvider(
        provider,
        max_batch_size=2,
        slot_capacity=16,
        batch_wait_s=1.0,
    )
    expected_bytes = 2 * 2 * 1 * 16 * 4 * torch.empty((), dtype=torch.float32).itemsize
    assert scheduled.working_cache_bytes == expected_bytes
    stream = scheduled.generate(
        _request(scheduled, maximum=3),
        cancelled=lambda: False,
    )
    assert isinstance(next(stream), GenerationTokenEvent)
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocked = executor.submit(next, stream)
        while not cast("Any", stream)._waiting:
            threading.Event().wait(0.001)
        scheduled.close()
        with pytest.raises(RuntimeError, match="provider closed"):
            blocked.result(timeout=2.0)

    assert scheduled.active_stream_count == 0
    assert scheduled.working_cache_bytes == 0
    scheduled.close()
    with pytest.raises(RuntimeError, match="provider is closed"):
        scheduled.generate(
            _request(scheduled, maximum=1),
            cancelled=lambda: False,
        )
