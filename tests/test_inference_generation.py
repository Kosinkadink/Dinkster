"""Torch-free autoregressive generation contract proofs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from types import TracebackType
from typing import Any, Self, cast

import pytest
from dinkster_inference import (
    GenerationEvent,
    GenerationFinishReason,
    GenerationMessage,
    GenerationMessageRole,
    GenerationProvider,
    GenerationProviderCapabilities,
    GenerationRequest,
    GenerationResult,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSamplerStage,
    GenerationSessionBusyError,
    GenerationSessionClosedError,
    GenerationSessionHandle,
    GenerationStats,
    GenerationStopConditions,
    GenerationStream,
    GenerationTerminalEvent,
    GenerationTokenEvent,
)

PROVIDER_ID = "dinkster.native-text"
MODEL_IDENTITY = "native:dinkster.qwen3:blake3:" + "1" * 64
SESSION_ID = "2" * 32


def _session(
    *, provider_id: str = PROVIDER_ID, model_identity: str = MODEL_IDENTITY
) -> GenerationSessionHandle:
    return GenerationSessionHandle(provider_id, model_identity, SESSION_ID)


def _sampler() -> GenerationSamplerChain:
    return GenerationSamplerChain(
        (
            GenerationSamplerStage(GenerationSamplerKind.REPETITION_PENALTY, 1.05),
            GenerationSamplerStage(GenerationSamplerKind.TEMPERATURE, 0.7),
            GenerationSamplerStage(GenerationSamplerKind.TOP_K, 64),
            GenerationSamplerStage(GenerationSamplerKind.TOP_P, 0.95),
            GenerationSamplerStage(GenerationSamplerKind.MIN_P, 0.05),
            GenerationSamplerStage(GenerationSamplerKind.MULTINOMIAL),
        )
    )


def test_request_values_are_frozen_equal_and_preserve_sampler_order() -> None:
    messages = (
        GenerationMessage(GenerationMessageRole.SYSTEM, "Answer briefly."),
        GenerationMessage(GenerationMessageRole.USER, "Hello"),
    )
    request = GenerationRequest(
        provider_id=PROVIDER_ID,
        model_identity=MODEL_IDENTITY,
        messages=messages,
        sampler=_sampler(),
        stop=GenerationStopConditions(32, ("END",), (2,)),
        seed=7,
        open_session=True,
    )
    equal = GenerationRequest(
        provider_id=PROVIDER_ID,
        model_identity=MODEL_IDENTITY,
        messages=messages,
        sampler=_sampler(),
        stop=GenerationStopConditions(32, ("END",), (2,)),
        seed=7,
        open_session=True,
    )

    assert request == equal
    assert tuple(stage.kind for stage in request.sampler.stages) == (
        GenerationSamplerKind.REPETITION_PENALTY,
        GenerationSamplerKind.TEMPERATURE,
        GenerationSamplerKind.TOP_K,
        GenerationSamplerKind.TOP_P,
        GenerationSamplerKind.MIN_P,
        GenerationSamplerKind.MULTINOMIAL,
    )
    with pytest.raises(FrozenInstanceError):
        cast("Any", request).seed = 8
    with pytest.raises(FrozenInstanceError):
        cast("Any", messages[0]).content = "changed"


def test_request_requires_one_input_and_validates_session_binding() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        GenerationRequest(PROVIDER_ID, MODEL_IDENTITY)
    with pytest.raises(ValueError, match="exactly one"):
        GenerationRequest(
            PROVIDER_ID,
            MODEL_IDENTITY,
            prompt="prompt",
            messages=(GenerationMessage(GenerationMessageRole.USER, "message"),),
        )
    with pytest.raises(TypeError, match="tuple"):
        GenerationRequest(
            PROVIDER_ID,
            MODEL_IDENTITY,
            messages=cast("Any", [GenerationMessage(GenerationMessageRole.USER, "message")]),
        )
    with pytest.raises(ValueError, match="provider"):
        GenerationRequest(
            PROVIDER_ID,
            MODEL_IDENTITY,
            prompt="continue",
            session=_session(provider_id="other.provider"),
        )
    with pytest.raises(ValueError, match="model identity"):
        GenerationRequest(
            PROVIDER_ID,
            MODEL_IDENTITY,
            prompt="continue",
            session=_session(model_identity="another-model"),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        GenerationRequest(
            PROVIDER_ID,
            MODEL_IDENTITY,
            prompt="continue",
            session=_session(),
            open_session=True,
        )


def test_sampler_chain_requires_typed_stages() -> None:
    greedy = GenerationSamplerStage(GenerationSamplerKind.GREEDY)
    assert GenerationSamplerChain((greedy,)).stages == (greedy,)
    with pytest.raises(TypeError, match="entries"):
        GenerationSamplerChain(cast("Any", ((GenerationSamplerKind.TOP_K, 1),)))


def test_sampler_chain_rejects_invalid_values_and_order() -> None:
    with pytest.raises(TypeError, match="exact integer"):
        GenerationSamplerStage(GenerationSamplerKind.TOP_K, 4.0)
    with pytest.raises(ValueError, match="finite"):
        GenerationSamplerStage(GenerationSamplerKind.TEMPERATURE, float("nan"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        GenerationSamplerStage(GenerationSamplerKind.TOP_P, 1.1)
    with pytest.raises(ValueError, match="positive"):
        GenerationSamplerStage(GenerationSamplerKind.REPETITION_PENALTY, 0.0)
    with pytest.raises(ValueError, match="unique"):
        GenerationSamplerChain(
            (
                GenerationSamplerStage(GenerationSamplerKind.TOP_P, 0.9),
                GenerationSamplerStage(GenerationSamplerKind.TOP_P, 0.8),
                GenerationSamplerStage(GenerationSamplerKind.MULTINOMIAL),
            )
        )
    with pytest.raises(ValueError, match="end in exactly one"):
        GenerationSamplerChain(
            (
                GenerationSamplerStage(GenerationSamplerKind.GREEDY),
                GenerationSamplerStage(GenerationSamplerKind.TOP_K, 10),
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        GenerationSamplerChain(
            (
                GenerationSamplerStage(GenerationSamplerKind.GREEDY),
                GenerationSamplerStage(GenerationSamplerKind.MULTINOMIAL),
            )
        )


def test_stop_conditions_are_strict_and_unique() -> None:
    assert GenerationStopConditions(1, ("stop",), (0,)).max_new_tokens == 1
    with pytest.raises(TypeError, match="exact integer"):
        GenerationStopConditions(cast("Any", True))
    with pytest.raises(ValueError, match="unique"):
        GenerationStopConditions(1, ("stop", "stop"))
    with pytest.raises(ValueError, match="non-empty"):
        GenerationStopConditions(1, ("",))
    with pytest.raises(ValueError, match="non-negative exact integers"):
        GenerationStopConditions(1, stop_token_ids=cast("Any", (True,)))


def test_stats_results_events_and_capabilities_validate_exact_values() -> None:
    stats = GenerationStats(
        total_time_s=1.0,
        prompt_tokens=4,
        generated_tokens=2,
        time_to_first_token_s=0.2,
        prefill_time_s=0.15,
        decode_time_s=0.8,
    )
    result = GenerationResult(
        "hello",
        GenerationFinishReason.EOS,
        stats,
        token_ids=(10, 11),
        continuation=_session(),
    )
    assert GenerationTokenEvent(0, "hel", 10).token_id == 10
    assert GenerationTerminalEvent(result).result == result
    capabilities = GenerationProviderCapabilities(
        frozenset({GenerationSamplerKind.GREEDY, GenerationSamplerKind.MULTINOMIAL}),
        chat=True,
        sessions=True,
        token_ids=True,
        ordered_sampler_chain=True,
    )
    assert capabilities.sessions

    with pytest.raises(TypeError, match="exact float"):
        GenerationStats(cast("Any", 1))
    with pytest.raises(ValueError, match="must not exceed"):
        GenerationStats(1.0, time_to_first_token_s=1.1)
    with pytest.raises(ValueError, match="non-negative"):
        GenerationTokenEvent(-1, "")
    with pytest.raises(TypeError, match="frozenset"):
        GenerationProviderCapabilities(cast("Any", {GenerationSamplerKind.GREEDY}))
    with pytest.raises(ValueError, match="token selector"):
        GenerationProviderCapabilities(frozenset({GenerationSamplerKind.TOP_P}))


class _FakeStream:
    def __init__(
        self,
        provider: _FakeProvider,
        request: GenerationRequest,
        cancelled: Callable[[], bool],
        session: GenerationSessionHandle | None,
        *,
        provisional: bool,
    ) -> None:
        self._provider = provider
        self._request = request
        self._cancelled = cancelled
        self._session = session
        self._provisional = provisional
        self._sequence = 0
        self._finished = False

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> GenerationEvent:
        if self._finished:
            raise StopIteration
        if self._cancelled():
            generated = self._sequence
            continuation = self._request.session
            self._finish(commit=False)
            return GenerationTerminalEvent(
                GenerationResult(
                    "ok" if generated else "",
                    GenerationFinishReason.CANCELLED,
                    GenerationStats(0.0, generated_tokens=generated),
                    token_ids=(1,) if generated else (),
                    continuation=continuation,
                )
            )
        if self._sequence == 0:
            self._sequence = 1
            return GenerationTokenEvent(0, "ok", 1)
        self._finish(commit=True)
        return GenerationTerminalEvent(
            GenerationResult(
                "ok",
                GenerationFinishReason.EOS,
                GenerationStats(0.1, prompt_tokens=1, generated_tokens=1),
                token_ids=(1,),
                continuation=self._session,
            )
        )

    def close(self) -> None:
        self._finish(commit=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _finish(self, *, commit: bool) -> None:
        if self._finished:
            return
        self._provider.finish(self, commit=commit)
        self._finished = True


class _FakeProvider:
    id = PROVIDER_ID
    capabilities = GenerationProviderCapabilities(
        frozenset(GenerationSamplerKind),
        chat=True,
        sessions=True,
        token_ids=True,
        ordered_sampler_chain=True,
    )

    def __init__(self, existing: tuple[GenerationSessionHandle, ...] = ()) -> None:
        self.sessions = {session: 0 for session in existing}
        self.active: dict[GenerationSessionHandle, _FakeStream] = {}
        self.closed: set[GenerationSessionHandle] = set()
        self.provisional_cleanups = 0

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> GenerationStream:
        session = request.session
        provisional = request.open_session
        if session is not None:
            if session in self.active:
                raise GenerationSessionBusyError("generation session already has an active request")
            if session not in self.sessions:
                raise GenerationSessionClosedError("generation session is closed or unknown")
        elif provisional:
            session = self._new_session()
        stream = _FakeStream(
            self,
            request,
            cancelled,
            session,
            provisional=provisional,
        )
        if session is not None:
            self.active[session] = stream
        return stream

    def close_session(self, session: GenerationSessionHandle) -> None:
        if session in self.active:
            raise GenerationSessionBusyError("active generation session cannot be closed")
        self.sessions.pop(session, None)
        self.closed.add(session)

    def finish(self, stream: _FakeStream, *, commit: bool) -> None:
        session = stream._session
        if session is None:
            return
        assert self.active.pop(session) is stream
        if stream._provisional:
            if commit:
                self.sessions[session] = 1
            else:
                self.provisional_cleanups += 1
        elif commit:
            self.sessions[session] += 1

    def _new_session(self) -> GenerationSessionHandle:
        for digit in "23456789abcdef":
            session = GenerationSessionHandle(PROVIDER_ID, MODEL_IDENTITY, digit * 32)
            if (
                session not in self.sessions
                and session not in self.active
                and session not in self.closed
            ):
                return session
        raise AssertionError("fake provider exhausted its session IDs")


def _collect(
    provider: GenerationProvider,
    request: GenerationRequest,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[GenerationEvent, ...]:
    with provider.generate(request, cancelled=cancelled) as stream:
        return tuple(stream)


def test_provider_protocol_streams_terminal_continuation_and_closes_session() -> None:
    provider = _FakeProvider()
    first = _collect(
        provider,
        GenerationRequest(PROVIDER_ID, MODEL_IDENTITY, prompt="first", open_session=True),
    )
    assert first[0] == GenerationTokenEvent(0, "ok", 1)
    assert isinstance(first[-1], GenerationTerminalEvent)
    continuation = first[-1].result.continuation
    assert continuation == _session()

    second = _collect(
        provider,
        GenerationRequest(PROVIDER_ID, MODEL_IDENTITY, prompt="second", session=continuation),
    )
    assert isinstance(second[-1], GenerationTerminalEvent)
    assert second[-1].result.continuation == continuation

    assert continuation is not None
    provider.close_session(continuation)
    provider.close_session(continuation)
    assert provider.closed == {continuation}
    with pytest.raises(GenerationSessionClosedError):
        provider.generate(
            GenerationRequest(
                PROVIDER_ID,
                MODEL_IDENTITY,
                prompt="closed",
                session=continuation,
            ),
            cancelled=lambda: False,
        )


def test_provider_protocol_reports_cancellation_as_a_terminal_result() -> None:
    events = _collect(
        _FakeProvider(),
        GenerationRequest(PROVIDER_ID, MODEL_IDENTITY, prompt="cancel"),
        cancelled=lambda: True,
    )
    assert events == (
        GenerationTerminalEvent(
            GenerationResult(
                "",
                GenerationFinishReason.CANCELLED,
                GenerationStats(0.0, generated_tokens=0),
                token_ids=(),
            )
        ),
    )


def test_existing_session_cancellation_after_a_token_rolls_back() -> None:
    session = _session()
    provider = _FakeProvider((session,))
    cancel = False
    with provider.generate(
        GenerationRequest(PROVIDER_ID, MODEL_IDENTITY, prompt="partial", session=session),
        cancelled=lambda: cancel,
    ) as stream:
        assert next(stream) == GenerationTokenEvent(0, "ok", 1)
        cancel = True
        terminal = next(stream)
    assert terminal == GenerationTerminalEvent(
        GenerationResult(
            "ok",
            GenerationFinishReason.CANCELLED,
            GenerationStats(0.0, generated_tokens=1),
            token_ids=(1,),
            continuation=session,
        )
    )
    assert provider.sessions[session] == 0
    assert not provider.active


def test_open_session_cancellation_after_a_token_discards_provisional_state() -> None:
    provider = _FakeProvider()
    cancel = False
    with provider.generate(
        GenerationRequest(PROVIDER_ID, MODEL_IDENTITY, prompt="partial", open_session=True),
        cancelled=lambda: cancel,
    ) as stream:
        assert next(stream) == GenerationTokenEvent(0, "ok", 1)
        cancel = True
        terminal = next(stream)
    assert isinstance(terminal, GenerationTerminalEvent)
    assert terminal.result.finish_reason is GenerationFinishReason.CANCELLED
    assert terminal.result.continuation is None
    assert provider.sessions == {}
    assert provider.provisional_cleanups == 1
    assert not provider.active


@pytest.mark.parametrize("open_session", [False, True])
def test_provider_context_closes_partial_stream_once_and_rolls_back(
    open_session: bool,
) -> None:
    session = None if open_session else _session()
    provider = _FakeProvider(() if session is None else (session,))
    with provider.generate(
        GenerationRequest(
            PROVIDER_ID,
            MODEL_IDENTITY,
            prompt="partial",
            session=session,
            open_session=open_session,
        ),
        cancelled=lambda: False,
    ) as stream:
        assert next(stream) == GenerationTokenEvent(0, "ok", 1)
    stream.close()
    stream.close()
    assert not provider.active
    if session is None:
        assert provider.sessions == {}
        assert provider.provisional_cleanups == 1
    else:
        assert provider.sessions[session] == 0
        assert provider.provisional_cleanups == 0


def test_session_rejects_concurrent_reuse_and_active_close() -> None:
    session = _session()
    provider = _FakeProvider((session,))
    stream = provider.generate(
        GenerationRequest(PROVIDER_ID, MODEL_IDENTITY, prompt="active", session=session),
        cancelled=lambda: False,
    )
    assert next(stream) == GenerationTokenEvent(0, "ok", 1)
    with pytest.raises(GenerationSessionBusyError):
        provider.generate(
            GenerationRequest(PROVIDER_ID, MODEL_IDENTITY, prompt="second", session=session),
            cancelled=lambda: False,
        )
    with pytest.raises(GenerationSessionBusyError):
        provider.close_session(session)
    stream.close()
    provider.close_session(session)
    provider.close_session(session)
    assert provider.closed == {session}
