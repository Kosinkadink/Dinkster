"""Continuous fixed-slot scheduling for native Qwen generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import torch
from dinkster_inference import (
    GenerationEvent,
    GenerationProviderCapabilities,
    GenerationRequest,
    GenerationSessionHandle,
    GenerationStream,
    GenerationTerminalEvent,
)

from .paged_kv import PagedKVCache
from .qwen_generation import (
    QwenGenerationProvider,
    _allocate_qwen_frequency_cache,  # pyright: ignore[reportPrivateUsage]
    _QwenFrequencyCache,  # pyright: ignore[reportPrivateUsage]
    _QwenGenerationStream,  # pyright: ignore[reportPrivateUsage]
    _slice_qwen_frequency_cache,  # pyright: ignore[reportPrivateUsage]
)
from .qwen_paged_kv import QwenPagedKVCache
from .residency import ResidencyMechanism


class QwenContinuousGenerationProvider:
    """Schedule pull-driven Qwen streams over shared fixed-capacity KV slots.

    Decode rows are batched only when their cache positions match. This keeps
    the existing unmasked causal-attention route while amortizing model-weight
    reads across concurrently advancing sessions.
    """

    def __init__(
        self,
        provider: QwenGenerationProvider,
        *,
        max_batch_size: int = 4,
        slot_capacity: int = 4096,
        prefill_chunk_tokens: int = 256,
        max_decode_batches_before_prefill: int = 8,
        batch_wait_s: float = 0.001,
    ) -> None:
        if type(provider) is not QwenGenerationProvider:
            raise TypeError("continuous Qwen scheduling requires a QwenGenerationProvider")
        for name, value in (
            ("max_batch_size", max_batch_size),
            ("slot_capacity", slot_capacity),
            ("prefill_chunk_tokens", prefill_chunk_tokens),
            ("max_decode_batches_before_prefill", max_decode_batches_before_prefill),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if slot_capacity > provider._model.config.max_position_embeddings:  # pyright: ignore[reportPrivateUsage]
            raise ValueError("Qwen scheduler slot capacity exceeds the model position limit")
        if type(batch_wait_s) is not float:
            raise TypeError("batch_wait_s must be an exact float")
        if not 0.0 <= batch_wait_s <= 1.0:
            raise ValueError("batch_wait_s must be between zero and one second")

        self._provider = provider
        self._max_batch_size = max_batch_size
        self._slot_capacity = slot_capacity
        self._prefill_chunk_tokens = prefill_chunk_tokens
        self._max_decode_batches_before_prefill = max_decode_batches_before_prefill
        self._batch_wait_s = batch_wait_s
        config = provider._model.config  # pyright: ignore[reportPrivateUsage]
        shape = (
            max_batch_size,
            config.num_key_value_heads,
            slot_capacity,
            config.head_dim,
        )
        self._cache_layers = tuple(
            (
                torch.empty(
                    shape,
                    dtype=provider._dtype,  # pyright: ignore[reportPrivateUsage]
                    device=device,
                ),
                torch.empty(
                    shape,
                    dtype=provider._dtype,  # pyright: ignore[reportPrivateUsage]
                    device=device,
                ),
            )
            for device in provider._layer_devices  # pyright: ignore[reportPrivateUsage]
        )
        self._frequency_cache: _QwenFrequencyCache | None = _allocate_qwen_frequency_cache(
            provider,
            slot_capacity,
        )
        self._condition = threading.Condition(threading.RLock())
        self._condition_queue_lock = threading.Lock()
        self._queued_condition_acquirers = 0
        self._pending_handoffs = 0
        self._states: list[_ScheduledState] = []
        self._closed = False
        self._prefill_clock = 0
        self._decode_clock = 0
        self._decode_batches_since_prefill = 0
        self._prefill_deadline: float | None = None
        self._decode_deadline: float | None = None
        self._worker = threading.Thread(
            target=self._run,
            name=f"{provider.id}-continuous-generation",
            daemon=True,
        )
        self._worker.start()

    @property
    def id(self) -> str:
        return self._provider.id

    @property
    def capabilities(self) -> GenerationProviderCapabilities:
        return self._provider.capabilities

    @property
    def cache(self) -> PagedKVCache | QwenPagedKVCache:
        return self._provider.cache

    @property
    def cache_residency_mechanisms(self) -> tuple[ResidencyMechanism, ...]:
        return self._provider.cache_residency_mechanisms

    @property
    def working_cache_bytes(self) -> int:
        return sum(key.nbytes + value.nbytes for key, value in self._cache_layers)

    @property
    def active_stream_count(self) -> int:
        with self._acquire_condition():
            return len(self._states)

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> GenerationStream:
        inner = self._provider.generate(request, cancelled=cancelled)
        required = inner._base_tokens + len(inner._prompt_ids) + inner._limit  # pyright: ignore[reportPrivateUsage]
        if required > self._slot_capacity:
            inner.close()
            raise ValueError(
                f"Qwen generation requires {required} cache positions; "
                f"continuous slot capacity is {self._slot_capacity}"
            )
        with self._acquire_condition():
            if self._closed:
                inner.close()
                raise RuntimeError("continuous Qwen generation provider is closed")
            if len(self._states) >= self._max_batch_size:
                inner.close()
                raise RuntimeError("continuous Qwen generation has no available request slot")
            stream = _ScheduledGenerationStream(self)
            state = _ScheduledState(
                inner,
                stream,
                len(self._states),
                self._prefill_clock,
                self._decode_clock,
            )
            stream._state = state  # pyright: ignore[reportPrivateUsage]
            frequencies = self._frequency_cache
            if frequencies is None:
                inner.close()
                raise RuntimeError("continuous Qwen generation provider is closed")
            try:
                inner._bind_working_cache(  # pyright: ignore[reportPrivateUsage]
                    self._slot_views(state.slot),
                    frequencies,
                )
            except BaseException:
                inner.close()
                raise
            self._states.append(state)
            self._condition.notify_all()
            return stream

    def close_session(self, session: GenerationSessionHandle) -> None:
        self._provider.close_session(session)

    def fork_session(
        self,
        session: GenerationSessionHandle,
        *,
        token_count: int | None = None,
    ) -> GenerationSessionHandle:
        return self._provider.fork_session(session, token_count=token_count)

    def close(self) -> None:
        with self._acquire_condition():
            if self._closed:
                return
            self._closed = True
            for state in tuple(self._states):
                self._fail_state(state, RuntimeError("continuous Qwen generation provider closed"))
            self._condition.notify_all()
        self._worker.join()
        self._cache_layers = ()
        self._frequency_cache = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def _acquire_condition(self) -> Generator[None]:
        queued = False
        try:
            with self._condition_queue_lock:
                self._queued_condition_acquirers += 1
                queued = True
            with self._condition:
                with self._condition_queue_lock:
                    self._queued_condition_acquirers -= 1
                    queued = False
                self._condition.notify_all()
                yield
        finally:
            if queued:
                with self._condition_queue_lock:
                    self._queued_condition_acquirers -= 1
                with self._condition:
                    self._condition.notify_all()

    def _request_event(self, state: _ScheduledState) -> GenerationEvent:
        with self._acquire_condition():
            stream = state.stream
            if stream._waiting:  # pyright: ignore[reportPrivateUsage]
                raise RuntimeError("generation stream already has a waiting consumer")
            if stream._done:  # pyright: ignore[reportPrivateUsage]
                raise StopIteration
            stream._waiting = True  # pyright: ignore[reportPrivateUsage]
            state.demand = True
            self._condition.notify_all()
            try:
                while stream._event is None and stream._error is None and not stream._done:  # pyright: ignore[reportPrivateUsage]
                    self._condition.wait()
                if stream._error is not None:  # pyright: ignore[reportPrivateUsage]
                    error = stream._error  # pyright: ignore[reportPrivateUsage]
                    stream._error = None  # pyright: ignore[reportPrivateUsage]
                    stream._done = True  # pyright: ignore[reportPrivateUsage]
                    raise error
                event = stream._event  # pyright: ignore[reportPrivateUsage]
                if event is None:
                    raise StopIteration
                stream._event = None  # pyright: ignore[reportPrivateUsage]
                if isinstance(event, GenerationTerminalEvent):
                    stream._done = True  # pyright: ignore[reportPrivateUsage]
                return event
            finally:
                stream._waiting = False  # pyright: ignore[reportPrivateUsage]
                if stream._handoff_pending:  # pyright: ignore[reportPrivateUsage]
                    stream._handoff_pending = False  # pyright: ignore[reportPrivateUsage]
                    self._pending_handoffs -= 1
                    self._condition.notify_all()

    def _close_stream(self, state: _ScheduledState) -> None:
        with self._acquire_condition():
            if state not in self._states:
                state.stream._done = True  # pyright: ignore[reportPrivateUsage]
                return
            state.inner.close()
            self._remove_state(state)
            state.stream._done = True  # pyright: ignore[reportPrivateUsage]
            self._notify_consumer(state)

    def _run(self) -> None:
        with self._condition:
            while not self._closed:
                if not any(state.demand for state in self._states):
                    self._condition.wait()
                    continue
                try:
                    self._step()
                except BaseException as error:
                    for state in tuple(self._states):
                        self._fail_state(state, error)

    def _step(self) -> None:
        demanded = tuple(state for state in self._states if state.demand)
        for state in demanded:
            if state.inner._cancelled():  # pyright: ignore[reportPrivateUsage]
                try:
                    event = state.inner._cancelled_terminal()  # pyright: ignore[reportPrivateUsage]
                except BaseException as error:
                    self._fail_state(state, error)
                else:
                    self._deliver(state, event, terminal=True)
                return

        prefill = tuple(
            state
            for state in demanded
            if not state.inner._prefilled  # pyright: ignore[reportPrivateUsage]
        )
        decode = tuple(
            state
            for state in demanded
            if state.inner._prefilled  # pyright: ignore[reportPrivateUsage]
        )
        if prefill and not decode and len(prefill) < self._max_batch_size and self._batch_wait_s:
            now = time.perf_counter()
            if self._prefill_deadline is None:
                self._prefill_deadline = now + self._batch_wait_s
            if now < self._prefill_deadline:
                self._condition.wait(self._prefill_deadline - now)
                return
        self._prefill_deadline = None
        initial_decode = (
            decode
            and all(
                len(state.inner._generated) == 1  # pyright: ignore[reportPrivateUsage]
                for state in decode
            )
            and all(
                len(state.inner._prompt_ids) - state.inner._prefill_offset  # pyright: ignore[reportPrivateUsage]
                <= self._prefill_chunk_tokens
                for state in prefill
            )
        )
        if prefill and (
            not decode
            or initial_decode
            or self._decode_batches_since_prefill >= self._max_decode_batches_before_prefill
        ):
            self._run_prefill(min(prefill, key=lambda state: state.prefill_turn))
            return

        stateless_terminal = next(
            (
                state
                for state in decode
                if state.inner._pending_finish is not None  # pyright: ignore[reportPrivateUsage]
                and state.inner._handle is None  # pyright: ignore[reportPrivateUsage]
            ),
            None,
        )
        if stateless_terminal is not None:
            try:
                event = stateless_terminal.inner._successful_terminal()  # pyright: ignore[reportPrivateUsage]
            except BaseException as error:
                self._fail_state(stateless_terminal, error)
            else:
                self._deliver(stateless_terminal, event, terminal=True)
            return

        if decode:
            now = time.perf_counter()
            if len(decode) >= self._max_batch_size:
                self._decode_deadline = None
            elif self._decode_deadline is None and self._batch_wait_s:
                self._decode_deadline = now + self._batch_wait_s
            if self._decode_deadline is not None and now < self._decode_deadline:
                self._condition.wait(self._decode_deadline - now)
                return
            self._decode_deadline = None
            first = min(decode, key=lambda state: (state.decode_turn, state.slot))
            position = first.inner._cache_position  # pyright: ignore[reportPrivateUsage]
            prefetch_enabled = first.inner._resolve_prefetch()  # pyright: ignore[reportPrivateUsage]
            cohort = tuple(
                state
                for state in decode
                if state.inner._cache_position == position  # pyright: ignore[reportPrivateUsage]
                and state.inner._resolve_prefetch() == prefetch_enabled  # pyright: ignore[reportPrivateUsage]
            )
            self._decode_clock += 1
            for state in cohort:
                state.decode_turn = self._decode_clock
            self._run_decode(cohort, position, prefetch_enabled)
            return

        if prefill:
            self._run_prefill(min(prefill, key=lambda state: state.prefill_turn))

    def _run_prefill(self, state: _ScheduledState) -> None:
        state.demand = False
        self._prefill_clock += 1
        state.prefill_turn = self._prefill_clock
        self._decode_batches_since_prefill = 0
        try:
            event = state.inner._advance_prefill(  # pyright: ignore[reportPrivateUsage]
                self._prefill_chunk_tokens
            )
        except BaseException as error:
            self._fail_state(state, error)
            return
        if event is None:
            state.demand = True
            self._condition.notify_all()
            while True:
                with self._condition_queue_lock:
                    queued = self._queued_condition_acquirers
                if not queued and not self._pending_handoffs:
                    break
                self._condition.wait()
        else:
            self._deliver(state, event, terminal=False)

    def _run_decode(
        self,
        states: tuple[_ScheduledState, ...],
        position: int,
        prefetch_enabled: bool,
    ) -> None:
        self._compact(states)
        batch = len(states)
        for state in states:
            state.demand = False
        tokens: list[torch.Tensor] = []
        flushing: list[bool] = []
        for state in states:
            token = state.inner._pending_token  # pyright: ignore[reportPrivateUsage]
            if token is None:
                raise RuntimeError("scheduled Qwen decode has no pending device token")
            tokens.append(token.reshape(1))
            state.inner._pending_token = None  # pyright: ignore[reportPrivateUsage]
            flushing.append(state.inner._pending_finish is not None)  # pyright: ignore[reportPrivateUsage]
        ids = torch.stack(tokens)
        caches = tuple((key[:batch], value[:batch]) for key, value in self._cache_layers)
        frequency_cache = self._frequency_cache
        if frequency_cache is None:
            raise RuntimeError("continuous Qwen generation frequencies were released")
        frequencies = _slice_qwen_frequency_cache(
            frequency_cache,
            position,
            position + 1,
        )
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                logits, new_key_values = self._provider._forward_logits(  # pyright: ignore[reportPrivateUsage]
                    ids,
                    caches,
                    cache_position=position,
                    frequencies=frequencies,
                    prefetch=prefetch_enabled,
                )
                states[0].inner._synchronize()  # pyright: ignore[reportPrivateUsage]
        except BaseException as error:
            for state in states:
                self._fail_state(state, error)
            return
        elapsed = time.perf_counter() - started
        self._decode_batches_since_prefill += 1
        outcomes: list[tuple[_ScheduledState, GenerationEvent, bool]] = []
        failures: list[tuple[_ScheduledState, BaseException]] = []
        for row, state in enumerate(states):
            try:
                state.inner._accept_model_output(  # pyright: ignore[reportPrivateUsage]
                    tuple(
                        (key[row : row + 1], value[row : row + 1]) for key, value in new_key_values
                    ),
                    1,
                )
                if flushing[row]:
                    event = state.inner._successful_terminal(  # pyright: ignore[reportPrivateUsage]
                        model_flushed=True,
                        flush_elapsed=elapsed,
                    )
                    outcomes.append((state, event, True))
                else:
                    state.inner._decode_time += elapsed  # pyright: ignore[reportPrivateUsage]
                    event = state.inner._accept_logits(logits[row])  # pyright: ignore[reportPrivateUsage]
                    outcomes.append((state, event, False))
            except BaseException as error:
                failures.append((state, error))
        for state, event, terminal in outcomes:
            self._deliver(state, event, terminal=terminal)
        for state, error in failures:
            self._fail_state(state, error)

    def _deliver(
        self,
        state: _ScheduledState,
        event: GenerationEvent,
        *,
        terminal: bool,
    ) -> None:
        state.stream._event = event  # pyright: ignore[reportPrivateUsage]
        if terminal:
            self._remove_state(state)
        self._notify_consumer(state)

    def _fail_state(self, state: _ScheduledState, error: BaseException) -> None:
        try:
            state.inner.close()
        except BaseException as close_error:
            error = close_error
        if state in self._states:
            self._remove_state(state)
        state.stream._error = error  # pyright: ignore[reportPrivateUsage]
        self._notify_consumer(state)

    def _notify_consumer(self, state: _ScheduledState) -> None:
        stream = state.stream
        if stream._waiting and not stream._handoff_pending:  # pyright: ignore[reportPrivateUsage]
            stream._handoff_pending = True  # pyright: ignore[reportPrivateUsage]
            self._pending_handoffs += 1
        self._condition.notify_all()

    def _compact(self, selected: tuple[_ScheduledState, ...]) -> None:
        for target, state in enumerate(selected):
            if state.slot != target:
                self._swap_slots(target, state.slot)

    def _swap_slots(self, first: int, second: int) -> None:
        first_state = self._states[first]
        second_state = self._states[second]
        span = max(
            first_state.inner._cache_position,  # pyright: ignore[reportPrivateUsage]
            second_state.inner._cache_position,  # pyright: ignore[reportPrivateUsage]
        )
        with torch.no_grad():
            for key, value in self._cache_layers:
                for cache in (key, value):
                    temporary = cache[first, :, :span].clone()
                    cache[first, :, :span].copy_(cache[second, :, :span])
                    cache[second, :, :span].copy_(temporary)
        self._states[first], self._states[second] = second_state, first_state
        first_state.slot = second
        second_state.slot = first
        self._rebind(first_state)
        self._rebind(second_state)

    def _remove_state(self, state: _ScheduledState) -> None:
        slot = state.slot
        last = self._states[-1]
        if last is not state:
            span = last.inner._cache_position  # pyright: ignore[reportPrivateUsage]
            with torch.no_grad():
                for key, value in self._cache_layers:
                    key[slot, :, :span].copy_(key[last.slot, :, :span])
                    value[slot, :, :span].copy_(value[last.slot, :, :span])
            self._states[slot] = last
            last.slot = slot
            self._rebind(last)
        self._states.pop()

    def _rebind(self, state: _ScheduledState) -> None:
        if self._frequency_cache is None:
            raise RuntimeError("continuous Qwen generation frequencies were released")
        state.inner._cache_layers = self._slot_views(state.slot)  # pyright: ignore[reportPrivateUsage]
        state.inner._frequency_cache = self._frequency_cache  # pyright: ignore[reportPrivateUsage]

    def _slot_views(self, slot: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (key[slot : slot + 1], value[slot : slot + 1]) for key, value in self._cache_layers
        )


@dataclass(eq=False, slots=True)
class _ScheduledState:
    inner: _QwenGenerationStream
    stream: _ScheduledGenerationStream
    slot: int
    prefill_turn: int
    decode_turn: int
    demand: bool = False


class _ScheduledGenerationStream:
    def __init__(self, provider: QwenContinuousGenerationProvider) -> None:
        self._provider = provider
        self._state: _ScheduledState
        self._event: GenerationEvent | None = None
        self._error: BaseException | None = None
        self._waiting = False
        self._handoff_pending = False
        self._done = False

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> GenerationEvent:
        return self._provider._request_event(self._state)  # pyright: ignore[reportPrivateUsage]

    def close(self) -> None:
        if self._done:
            return
        self._provider._close_stream(self._state)  # pyright: ignore[reportPrivateUsage]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["QwenContinuousGenerationProvider"]
