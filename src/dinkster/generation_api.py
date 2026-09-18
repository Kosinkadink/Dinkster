"""HTTP adapters and lifecycle ownership for neutral text generation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast

from aiohttp import web
from dinkster_inference import (
    GenerationEvent,
    GenerationFinishReason,
    GenerationMessage,
    GenerationMessageRole,
    GenerationProvider,
    GenerationRequest,
    GenerationResult,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSamplerStage,
    GenerationSessionBusyError,
    GenerationSessionClosedError,
    GenerationSessionHandle,
    GenerationStopConditions,
    GenerationStream,
    GenerationTerminalEvent,
    GenerationTokenEvent,
)
from dinkster_server import principal_for

_MAX_REQUEST_BYTES = 1 << 20
_END = object()


class ProviderFactory(Protocol):
    def __call__(self) -> GenerationProvider: ...


class ProviderCloser(Protocol):
    def __call__(self, provider: GenerationProvider) -> None: ...


@dataclass(frozen=True, slots=True)
class GenerationModel:
    """One public model name and its lazily constructed provider."""

    name: str
    provider_id: str
    factory: ProviderFactory
    model_identity: str | None = None
    close_provider: ProviderCloser | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "provider_id"):
            value = cast("object", getattr(self, field_name))
            if type(value) is not str or not value or len(value) > 1024 or not value.isprintable():
                raise ValueError(
                    f"generation model {field_name} must be a bounded printable string"
                )
        if not callable(self.factory):
            raise TypeError("generation model factory must be callable")
        if self.model_identity is not None and (
            type(cast("object", self.model_identity)) is not str or not self.model_identity
        ):
            raise ValueError("generation model identity must be a non-empty string or None")
        if self.close_provider is not None and not callable(self.close_provider):
            raise TypeError("generation model provider closer must be callable or None")


@dataclass(slots=True)
class _ModelState:
    registration: GenerationModel
    created: int
    provider: GenerationProvider | None = None
    model_identity: str | None = None
    active: int = 0


@dataclass(frozen=True, slots=True)
class _OwnedSession:
    owner: str
    model: str
    handle: GenerationSessionHandle


class GenerationModelBusyError(RuntimeError):
    pass


class GenerationServiceClosedError(RuntimeError):
    pass


class _ManagedGenerationStream:
    def __init__(
        self,
        service: GenerationService,
        state: _ModelState,
        stream: GenerationStream,
        *,
        owner: str,
        input_session_id: str | None,
        input_session_handle: GenerationSessionHandle | None,
        open_session: bool,
        store_session: bool,
        cancel: Callable[[], None],
    ) -> None:
        self._service = service
        self._state = state
        self._stream = stream
        self._owner = owner
        self._input_session_id = input_session_id
        self._input_session_handle = input_session_handle
        self._open_session = open_session
        self._store_session = store_session
        self._cancel = cancel
        self._closed = False
        self._terminal = False
        self._pull_lock = threading.Lock()
        self.public_session_id: str | None = None
        self._reserved_session_id: str | None = None

    def __iter__(self) -> _ManagedGenerationStream:
        return self

    def __next__(self) -> GenerationEvent:
        with self._pull_lock:
            if self._closed:
                raise StopIteration
            event = next(self._stream)
            if self._terminal:
                raise RuntimeError("generation stream produced an event after its terminal event")
            if isinstance(event, GenerationTerminalEvent):
                try:
                    next(self._stream)
                except StopIteration:
                    pass
                else:
                    raise RuntimeError(
                        "generation stream produced an event after its terminal event"
                    )
                self._validate_terminal(event.result)
                result = self._service._record_terminal(self, event.result)
                self._terminal = True
                return GenerationTerminalEvent(result)
            return event

    def _validate_terminal(self, result: GenerationResult) -> None:
        continuation = result.continuation
        if self._input_session_handle is not None:
            if continuation != self._input_session_handle:
                raise RuntimeError("generation provider replaced or dropped an existing session")
        elif self._open_session:
            if result.finish_reason is GenerationFinishReason.CANCELLED:
                if continuation is not None:
                    raise RuntimeError("cancelled provisional generation returned a session")
            elif continuation is None:
                raise RuntimeError("successful provisional generation did not return a session")
        elif continuation is not None:
            raise RuntimeError("stateless generation unexpectedly returned a session")

    def close(self) -> None:
        self._cancel()
        with self._pull_lock:
            if self._closed:
                return
            self._stream.close()
            self._closed = True
            self._service._release(self._state, self)

    def discard_session(self) -> None:
        if not self._terminal:
            raise RuntimeError("generation stream has no completed session")
        session_id = self.public_session_id
        if session_id is None:
            return
        try:
            self._service.close_session(session_id, owner=self._owner)
        finally:
            self.public_session_id = None


class GenerationService:
    """Own lazy providers, active streams, sessions, unload, and shutdown."""

    def __init__(self, models: Sequence[GenerationModel]) -> None:
        states: dict[str, _ModelState] = {}
        created = int(time.time())
        for model in models:
            if type(cast("object", model)) is not GenerationModel:
                raise TypeError("generation models must contain GenerationModel values")
            if model.name in states:
                raise ValueError(f"duplicate generation model name: {model.name!r}")
            states[model.name] = _ModelState(model, created)
        if not states:
            raise ValueError("generation service requires at least one model")
        self._models = states
        self._sessions: dict[str, _OwnedSession] = {}
        self._claimed_sessions: dict[str, _OwnedSession] = {}
        self._pending_sessions: list[_OwnedSession] = []
        self._reserved_session_ids: set[str] = set()
        self._active: set[_ManagedGenerationStream] = set()
        self._lock = threading.RLock()
        self._closed = False
        self._cleanup_complete = False

    def model_rows(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(
                {
                    "id": state.registration.name,
                    "object": "model",
                    "created": state.created,
                    "owned_by": "dinkster",
                    "loaded": state.model_identity is not None,
                }
                for state in self._models.values()
            )

    def open_stream(
        self,
        model: str,
        *,
        owner: str,
        prompt: str | None,
        messages: tuple[GenerationMessage, ...],
        sampler: GenerationSamplerChain,
        stop: GenerationStopConditions,
        seed: int | None,
        session_id: str | None,
        open_session: bool,
        store_session: bool = True,
        cancelled: Callable[[], bool],
        cancel: Callable[[], None],
    ) -> _ManagedGenerationStream:
        with self._lock:
            if self._closed:
                raise GenerationServiceClosedError("generation service is closed")
            state = self._state(model)
            handle = None
            owned = None
            if session_id is not None:
                claimed = self._claimed_sessions.get(session_id)
                if claimed is not None:
                    if claimed.owner != owner or claimed.model != model:
                        raise GenerationSessionClosedError("unknown generation session")
                    raise GenerationSessionBusyError("generation session is active")
                owned = self._sessions.get(session_id)
                if owned is None or owned.owner != owner or owned.model != model:
                    raise GenerationSessionClosedError("unknown generation session")
                handle = owned.handle
            provider, model_identity = self._load_locked(state)
            request = GenerationRequest(
                provider_id=state.registration.provider_id,
                model_identity=model_identity,
                prompt=prompt,
                messages=messages,
                sampler=sampler,
                stop=stop,
                seed=seed,
                session=handle,
                open_session=open_session,
            )
            stream = provider.generate(request, cancelled=cancelled)
            state.active += 1
            managed = _ManagedGenerationStream(
                self,
                state,
                stream,
                owner=owner,
                input_session_id=session_id,
                input_session_handle=handle,
                open_session=open_session,
                store_session=store_session,
                cancel=cancel,
            )
            if session_id is not None:
                assert owned is not None
                self._sessions.pop(session_id)
                self._claimed_sessions[session_id] = owned
            if session_id is not None:
                managed.public_session_id = session_id
            elif open_session:
                public_session_id = self._reserve_session_id_locked()
                managed.public_session_id = public_session_id
                managed._reserved_session_id = public_session_id
            self._active.add(managed)
            return managed

    def load_model(self, model: str) -> None:
        with self._lock:
            if self._closed:
                raise GenerationServiceClosedError("generation service is closed")
            self._load_locked(self._state(model))

    def unload_model(self, model: str) -> None:
        with self._lock:
            state = self._state(model)
            if state.active:
                raise GenerationModelBusyError("generation model has active requests")
            provider = state.provider
            if provider is None:
                return
            for owned in tuple(self._pending_sessions):
                if owned.model != model:
                    continue
                provider.close_session(owned.handle)
                self._pending_sessions.remove(owned)
            sessions = tuple(
                (session_id, owned)
                for session_id, owned in self._sessions.items()
                if owned.model == model
            )
            for session_id, owned in sessions:
                self._sessions.pop(session_id, None)
                try:
                    provider.close_session(owned.handle)
                except Exception:
                    self._remember_pending_locked(owned)
                    raise
            self._close_provider(state, provider)
            state.provider = None
            state.model_identity = None

    def close_session(self, session_id: str, *, owner: str) -> None:
        with self._lock:
            claimed = self._claimed_sessions.get(session_id)
            if claimed is not None:
                if claimed.owner != owner:
                    raise GenerationSessionClosedError("unknown generation session")
                raise GenerationSessionBusyError("generation session is active")
            owned = self._sessions.pop(session_id, None)
            if owned is None or owned.owner != owner:
                if owned is not None:
                    self._sessions[session_id] = owned
                raise GenerationSessionClosedError("unknown generation session")
            state = self._state(owned.model)
            provider = state.provider
            if provider is None:
                raise GenerationSessionClosedError("unknown generation session")
            try:
                provider.close_session(owned.handle)
            except Exception:
                self._remember_pending_locked(owned)
                raise

    def public_session_id(
        self,
        stream: _ManagedGenerationStream,
        *,
        replace: bool = False,
    ) -> str:
        """Return the opaque server identity reserved for one active stream."""
        with self._lock:
            if stream not in self._active:
                raise GenerationSessionClosedError("generation stream is not active")
            if replace and stream._input_session_id is not None:
                if stream._reserved_session_id is not None:
                    self._reserved_session_ids.discard(stream._reserved_session_id)
                session_id = self._reserve_session_id_locked()
                stream.public_session_id = session_id
                stream._reserved_session_id = session_id
            elif stream.public_session_id is None:
                session_id = self._reserve_session_id_locked()
                stream.public_session_id = session_id
                stream._reserved_session_id = session_id
            return stream.public_session_id

    def close(self) -> None:
        with self._lock:
            if self._cleanup_complete:
                return
            self._closed = True
            active = tuple(self._active)
        errors: list[Exception] = []
        for stream in active:
            try:
                stream.close()
            except Exception as error:
                errors.append(error)
        with self._lock:
            pending_before = tuple(self._pending_sessions)
            for session_id, owned in tuple(self._sessions.items()):
                state = self._state(owned.model)
                if state.active:
                    continue
                if state.provider is None:
                    self._sessions.pop(session_id, None)
                    continue
                self._sessions.pop(session_id, None)
                try:
                    state.provider.close_session(owned.handle)
                except Exception as error:
                    self._remember_pending_locked(owned)
                    errors.append(error)
            for owned in pending_before:
                if owned not in self._pending_sessions:
                    continue
                state = self._state(owned.model)
                if state.active or state.provider is None:
                    continue
                try:
                    state.provider.close_session(owned.handle)
                except Exception as error:
                    errors.append(error)
                else:
                    self._pending_sessions.remove(owned)
            for state in self._models.values():
                if state.provider is not None and not state.active:
                    try:
                        provider_closed = self._close_provider(state, state.provider)
                    except Exception as error:
                        errors.append(error)
                    else:
                        has_sessions = any(
                            owned.model == state.registration.name
                            for owned in (
                                *self._sessions.values(),
                                *self._claimed_sessions.values(),
                                *self._pending_sessions,
                            )
                        )
                        if has_sessions and not provider_closed:
                            continue
                        state.provider = None
                        state.model_identity = None
                        for session_id, owned in tuple(self._sessions.items()):
                            if owned.model == state.registration.name:
                                self._sessions.pop(session_id, None)
                        self._pending_sessions[:] = [
                            owned
                            for owned in self._pending_sessions
                            if owned.model != state.registration.name
                        ]
            self._cleanup_complete = (
                not self._active
                and not self._sessions
                and not self._claimed_sessions
                and not self._pending_sessions
                and all(state.provider is None for state in self._models.values())
            )
        if errors:
            raise ExceptionGroup("generation service cleanup failed", errors)

    def _state(self, model: str) -> _ModelState:
        try:
            return self._models[model]
        except KeyError as error:
            raise KeyError(f"unknown generation model: {model!r}") from error

    def _load_locked(self, state: _ModelState) -> tuple[GenerationProvider, str]:
        provider = state.provider
        if provider is None:
            provider = state.registration.factory()
            state.provider = provider
            try:
                if provider.id != state.registration.provider_id:
                    raise RuntimeError("generation provider factory returned the wrong provider")
                identity = state.registration.model_identity
                if identity is None:
                    candidate = getattr(provider, "model_identity", None)
                    if type(candidate) is not str or not candidate:
                        raise RuntimeError("generation provider did not publish its model identity")
                    identity = candidate
            except BaseException:
                self._close_provider(state, provider)
                state.provider = None
                raise
            state.model_identity = identity
        elif state.model_identity is None:
            raise RuntimeError("generation provider cleanup is pending")
        assert state.model_identity is not None
        return provider, state.model_identity

    @staticmethod
    def _close_provider(state: _ModelState, provider: GenerationProvider) -> bool:
        closer = state.registration.close_provider
        if closer is not None:
            closer(provider)
            return True
        close = getattr(provider, "close", None)
        if callable(close):
            close()
            return True
        return False

    def _reserve_session_id_locked(self) -> str:
        while True:
            session_id = uuid.uuid4().hex
            if (
                session_id not in self._sessions
                and session_id not in self._claimed_sessions
                and session_id not in self._reserved_session_ids
            ):
                self._reserved_session_ids.add(session_id)
                return session_id

    def _remember_pending_locked(self, owned: _OwnedSession) -> None:
        if owned not in self._pending_sessions:
            self._pending_sessions.append(owned)

    def _record_terminal(
        self,
        stream: _ManagedGenerationStream,
        result: GenerationResult,
    ) -> GenerationResult:
        with self._lock:
            state = stream._state
            input_session_id = stream._input_session_id
            continuation = result.continuation
            if continuation is None:
                if stream._reserved_session_id is not None:
                    self._reserved_session_ids.discard(stream._reserved_session_id)
                    stream._reserved_session_id = None
                stream.public_session_id = None
                return result
            if continuation.provider_id != state.registration.provider_id:
                raise RuntimeError("generation provider returned a foreign session")
            if continuation.model_identity != state.model_identity:
                raise RuntimeError("generation provider returned a session for another model")
            if input_session_id is not None:
                if self._claimed_sessions.pop(input_session_id, None) is None:
                    raise RuntimeError("generation stream lost ownership of its input session")
            owned = _OwnedSession(
                stream._owner,
                state.registration.name,
                continuation,
            )
            if not stream._store_session:
                if stream._reserved_session_id is not None:
                    self._reserved_session_ids.discard(stream._reserved_session_id)
                    stream._reserved_session_id = None
                stream.public_session_id = None
                provider = state.provider
                if provider is None:
                    raise RuntimeError("generation provider disappeared before session cleanup")
                try:
                    provider.close_session(continuation)
                except Exception:
                    self._remember_pending_locked(owned)
                    raise
                return replace(result, continuation=None)
            public_session_id = stream.public_session_id
            if public_session_id is None:
                public_session_id = self._reserve_session_id_locked()
                stream.public_session_id = public_session_id
                stream._reserved_session_id = public_session_id
            if input_session_id is not None and input_session_id != public_session_id:
                self._sessions.pop(input_session_id, None)
            self._sessions[public_session_id] = owned
            self._reserved_session_ids.discard(public_session_id)
            stream._reserved_session_id = None
            public_handle = GenerationSessionHandle(
                continuation.provider_id,
                continuation.model_identity,
                public_session_id,
            )
            return replace(result, continuation=public_handle)

    def _release(self, state: _ModelState, stream: _ManagedGenerationStream) -> None:
        with self._lock:
            if state.active <= 0:
                raise RuntimeError("generation stream released more than once")
            state.active -= 1
            self._active.discard(stream)
            input_session_id = stream._input_session_id
            if input_session_id is not None:
                claimed = self._claimed_sessions.pop(input_session_id, None)
                if claimed is not None:
                    self._sessions[input_session_id] = claimed
            if stream._reserved_session_id is not None:
                self._reserved_session_ids.discard(stream._reserved_session_id)
                stream._reserved_session_id = None


@dataclass(frozen=True, slots=True)
class _ParsedRequest:
    model: str
    prompt: str | None
    messages: tuple[GenerationMessage, ...]
    sampler: GenerationSamplerChain
    stop: GenerationStopConditions
    seed: int | None
    session_id: str | None
    open_session: bool
    stream: bool
    include_usage: bool
    store_session: bool = True


class _RequestError(ValueError):
    pass


def _object(value: object, subject: str) -> dict[str, object]:
    if type(value) is not dict:
        raise _RequestError(f"{subject} must be a JSON object")
    return cast("dict[str, object]", value)


def _only(values: Mapping[str, object], allowed: frozenset[str], subject: str) -> None:
    extras = sorted(set(values) - allowed)
    if extras:
        raise _RequestError(f"{subject} has unsupported fields: {', '.join(extras)}")


async def _read_json(request: web.Request) -> dict[str, object]:
    length = request.content_length
    if length is not None and length > _MAX_REQUEST_BYTES:
        raise _RequestError("generation request body exceeds 1 MiB")
    body = await request.read()
    if len(body) > _MAX_REQUEST_BYTES:
        raise _RequestError("generation request body exceeds 1 MiB")
    try:
        value: object = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _RequestError("generation request body must be valid UTF-8 JSON") from error
    return _object(value, "generation request")


def _string(value: object, name: str, *, allow_empty: bool = True) -> str:
    if type(value) is not str:
        raise _RequestError(f"{name} must be a string")
    if not allow_empty and not value:
        raise _RequestError(f"{name} must not be empty")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise _RequestError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _RequestError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float):
        raise _RequestError(f"{name} must be a number")
    result = float(cast("int | float", value))
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise _RequestError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _messages(value: object, name: str = "messages") -> tuple[GenerationMessage, ...]:
    if type(value) is not list or not value:
        raise _RequestError(f"{name} must be a non-empty array")
    result: list[GenerationMessage] = []
    for index, raw in enumerate(cast("list[object]", value)):
        item = _object(raw, f"{name}[{index}]")
        _only(item, frozenset({"role", "content"}), f"{name}[{index}]")
        try:
            role = GenerationMessageRole(_string(item.get("role"), f"{name}[{index}].role"))
        except ValueError as error:
            raise _RequestError(f"{name}[{index}].role is unsupported") from error
        result.append(
            GenerationMessage(role, _string(item.get("content"), f"{name}[{index}].content"))
        )
    return tuple(result)


def _stop_texts(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if type(value) is str:
        return (_string(value, "stop", allow_empty=False),)
    if type(value) is not list:
        raise _RequestError("stop must be a string or an array of strings")
    result = tuple(
        _string(item, f"stop[{index}]", allow_empty=False)
        for index, item in enumerate(cast("list[object]", value))
    )
    if len(result) != len(set(result)):
        raise _RequestError("stop strings must be unique")
    return result


def _native_sampler(value: object) -> GenerationSamplerChain:
    if value is None:
        return GenerationSamplerChain((GenerationSamplerStage(GenerationSamplerKind.GREEDY),))
    if type(value) is not list or not value:
        raise _RequestError("sampler must be a non-empty array")
    stages: list[GenerationSamplerStage] = []
    for index, raw in enumerate(cast("list[object]", value)):
        item = _object(raw, f"sampler[{index}]")
        _only(item, frozenset({"kind", "value"}), f"sampler[{index}]")
        try:
            kind = GenerationSamplerKind(_string(item.get("kind"), f"sampler[{index}].kind"))
            stages.append(
                GenerationSamplerStage(
                    kind,
                    cast("int | float | None", item.get("value")),
                )
            )
        except (TypeError, ValueError) as error:
            raise _RequestError(str(error)) from error
    try:
        return GenerationSamplerChain(tuple(stages))
    except (TypeError, ValueError) as error:
        raise _RequestError(str(error)) from error


def _openai_sampler(values: Mapping[str, object]) -> GenerationSamplerChain:
    temperature = _number(values.get("temperature", 1.0), "temperature", 0.0, 2.0)
    top_p = _number(values.get("top_p", 1.0), "top_p", 0.0, 1.0)
    presence = _number(values.get("presence_penalty", 0.0), "presence_penalty", -2.0, 2.0)
    frequency = _number(values.get("frequency_penalty", 0.0), "frequency_penalty", -2.0, 2.0)
    stages: list[GenerationSamplerStage] = []
    if presence:
        stages.append(GenerationSamplerStage(GenerationSamplerKind.PRESENCE_PENALTY, presence))
    if frequency:
        stages.append(GenerationSamplerStage(GenerationSamplerKind.FREQUENCY_PENALTY, frequency))
    if temperature not in (0.0, 1.0):
        stages.append(GenerationSamplerStage(GenerationSamplerKind.TEMPERATURE, temperature))
    if top_p != 1.0:
        stages.append(GenerationSamplerStage(GenerationSamplerKind.TOP_P, top_p))
    stages.append(
        GenerationSamplerStage(
            GenerationSamplerKind.GREEDY
            if temperature == 0.0
            else GenerationSamplerKind.MULTINOMIAL
        )
    )
    return GenerationSamplerChain(tuple(stages))


def _parse_native(values: Mapping[str, object]) -> _ParsedRequest:
    _only(
        values,
        frozenset(
            {
                "model",
                "prompt",
                "messages",
                "sampler",
                "maxNewTokens",
                "stopTexts",
                "stopTokenIds",
                "seed",
                "sessionId",
                "openSession",
                "stream",
            }
        ),
        "generation request",
    )
    prompt_value = values.get("prompt")
    messages_value = values.get("messages")
    if (prompt_value is None) == (messages_value is None):
        raise _RequestError("generation request requires exactly one of prompt or messages")
    prompt = None if prompt_value is None else _string(prompt_value, "prompt")
    messages = () if messages_value is None else _messages(messages_value)
    stop_ids_value = values.get("stopTokenIds", [])
    if type(stop_ids_value) is not list:
        raise _RequestError("stopTokenIds must be an array")
    stop_ids = tuple(
        _integer(item, f"stopTokenIds[{index}]", 0, 2**63 - 1)
        for index, item in enumerate(cast("list[object]", stop_ids_value))
    )
    if len(stop_ids) != len(set(stop_ids)):
        raise _RequestError("stopTokenIds entries must be unique")
    seed_value = values.get("seed")
    session_value = values.get("sessionId")
    return _ParsedRequest(
        model=_string(values.get("model"), "model", allow_empty=False),
        prompt=prompt,
        messages=messages,
        sampler=_native_sampler(values.get("sampler")),
        stop=GenerationStopConditions(
            _integer(values.get("maxNewTokens", 128), "maxNewTokens", 1, 1_000_000),
            _stop_texts(values.get("stopTexts")),
            stop_ids,
        ),
        seed=(None if seed_value is None else _integer(seed_value, "seed", 0, 2**64 - 1)),
        session_id=(
            None
            if session_value is None
            else _string(session_value, "sessionId", allow_empty=False)
        ),
        open_session=_bool(values.get("openSession", False), "openSession"),
        stream=_bool(values.get("stream", False), "stream"),
        include_usage=False,
    )


_OPENAI_COMMON = frozenset(
    {
        "model",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "seed",
        "n",
        "user",
    }
)


def _parse_openai(values: Mapping[str, object], *, chat: bool) -> _ParsedRequest:
    payload_field = "messages" if chat else "prompt"
    _only(values, _OPENAI_COMMON | frozenset({payload_field}), "OpenAI generation request")
    _integer(values.get("n", 1), "n", 1, 1)
    if "user" in values:
        _string(values["user"], "user")
    max_tokens = values.get("max_completion_tokens", values.get("max_tokens", 128))
    if "max_completion_tokens" in values and "max_tokens" in values:
        raise _RequestError("max_tokens and max_completion_tokens are mutually exclusive")
    stream = _bool(values.get("stream", False), "stream")
    stream_options = values.get("stream_options")
    include_usage = False
    if stream_options is not None:
        if not stream:
            raise _RequestError("stream_options requires stream=true")
        options = _object(stream_options, "stream_options")
        _only(options, frozenset({"include_usage"}), "stream_options")
        include_usage = _bool(options.get("include_usage", False), "stream_options.include_usage")
    seed_value = values.get("seed")
    return _ParsedRequest(
        model=_string(values.get("model"), "model", allow_empty=False),
        prompt=None if chat else _string(values.get("prompt"), "prompt"),
        messages=_messages(values.get("messages")) if chat else (),
        sampler=_openai_sampler(values),
        stop=GenerationStopConditions(
            _integer(max_tokens, "max_tokens", 1, 1_000_000),
            _stop_texts(values.get("stop")),
        ),
        seed=(None if seed_value is None else _integer(seed_value, "seed", 0, 2**64 - 1)),
        session_id=None,
        open_session=False,
        stream=stream,
        include_usage=include_usage,
    )


def _parse_responses(values: Mapping[str, object]) -> _ParsedRequest:
    _only(
        values,
        frozenset(
            {
                "model",
                "input",
                "instructions",
                "stream",
                "max_output_tokens",
                "temperature",
                "top_p",
                "previous_response_id",
                "store",
                "user",
            }
        ),
        "OpenAI responses request",
    )
    input_value = values.get("input")
    instructions = values.get("instructions")
    prompt: str | None = None
    messages: tuple[GenerationMessage, ...] = ()
    if type(input_value) is str:
        if instructions is None:
            prompt = input_value
        else:
            messages = (
                GenerationMessage(
                    GenerationMessageRole.SYSTEM,
                    _string(instructions, "instructions"),
                ),
                GenerationMessage(GenerationMessageRole.USER, input_value),
            )
    else:
        messages = _messages(input_value, "input")
        if instructions is not None:
            messages = (
                GenerationMessage(
                    GenerationMessageRole.SYSTEM,
                    _string(instructions, "instructions"),
                ),
                *messages,
            )
    previous = values.get("previous_response_id")
    if previous is not None:
        previous = _string(previous, "previous_response_id", allow_empty=False)
        if not previous.startswith("resp_"):
            raise _RequestError("previous_response_id is not a Dinkster response id")
        previous = previous.removeprefix("resp_")
    if "user" in values:
        _string(values["user"], "user")
    store = _bool(values.get("store", True), "store")
    return _ParsedRequest(
        model=_string(values.get("model"), "model", allow_empty=False),
        prompt=prompt,
        messages=messages,
        sampler=_openai_sampler(values),
        stop=GenerationStopConditions(
            _integer(values.get("max_output_tokens", 128), "max_output_tokens", 1, 1_000_000)
        ),
        seed=None,
        session_id=cast("str | None", previous),
        open_session=store and previous is None,
        stream=_bool(values.get("stream", False), "stream"),
        include_usage=False,
        store_session=store,
    )


def _stats(result: GenerationResult) -> dict[str, object]:
    stats = result.stats
    return {
        "totalTimeSeconds": stats.total_time_s,
        "promptTokens": stats.prompt_tokens,
        "generatedTokens": stats.generated_tokens,
        "timeToFirstTokenSeconds": stats.time_to_first_token_s,
        "prefillTimeSeconds": stats.prefill_time_s,
        "decodeTimeSeconds": stats.decode_time_s,
    }


def _native_result(result: GenerationResult) -> dict[str, object]:
    return {
        "text": result.text,
        "finishReason": result.finish_reason.value,
        "stats": _stats(result),
        "tokenIds": None if result.token_ids is None else list(result.token_ids),
        "sessionId": None if result.continuation is None else result.continuation.session_id,
    }


def _usage(result: GenerationResult) -> dict[str, int]:
    prompt = result.stats.prompt_tokens or 0
    completion = result.stats.generated_tokens or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _finish_reason(reason: GenerationFinishReason) -> str:
    return {
        GenerationFinishReason.EOS: "stop",
        GenerationFinishReason.STOP: "stop",
        GenerationFinishReason.LENGTH: "length",
        GenerationFinishReason.CANCELLED: "cancelled",
    }[reason]


async def _next_event(
    stream: _ManagedGenerationStream,
    cancelled: threading.Event,
) -> GenerationEvent | object:
    def pull() -> GenerationEvent | object:
        try:
            return next(stream)
        except StopIteration:
            return _END

    task = asyncio.create_task(asyncio.to_thread(pull))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        loop = asyncio.get_running_loop()

        def close_after_pull(_: asyncio.Task[GenerationEvent | object]) -> None:
            loop.create_task(asyncio.to_thread(stream.close))

        task.add_done_callback(close_after_pull)
        raise


async def _collect(
    stream: _ManagedGenerationStream,
    cancelled: threading.Event,
) -> tuple[tuple[GenerationTokenEvent, ...], GenerationResult]:
    tokens: list[GenerationTokenEvent] = []
    terminal: GenerationResult | None = None
    try:
        while True:
            event = await _next_event(stream, cancelled)
            if event is _END:
                break
            if terminal is not None:
                raise RuntimeError("generation stream produced an event after its terminal event")
            if isinstance(event, GenerationTokenEvent):
                if event.sequence != len(tokens):
                    raise RuntimeError("generation token sequence is not contiguous")
                tokens.append(event)
            elif isinstance(event, GenerationTerminalEvent):
                terminal = event.result
            else:
                raise RuntimeError("generation stream produced an unknown event")
        if terminal is None:
            raise RuntimeError("generation stream ended without a terminal event")
        return tuple(tokens), terminal
    finally:
        await asyncio.to_thread(stream.close)


async def _open_stream(
    service: GenerationService,
    parsed: _ParsedRequest,
    owner: str,
    cancelled: threading.Event,
) -> _ManagedGenerationStream:
    return await asyncio.to_thread(
        service.open_stream,
        parsed.model,
        owner=owner,
        prompt=parsed.prompt,
        messages=parsed.messages,
        sampler=parsed.sampler,
        stop=parsed.stop,
        seed=parsed.seed,
        session_id=parsed.session_id,
        open_session=parsed.open_session,
        store_session=parsed.store_session,
        cancelled=cancelled.is_set,
        cancel=cancelled.set,
    )


def _native_error(message: str, status: int) -> web.Response:
    return web.json_response(
        {"error": "generation-request-failed", "message": message},
        status=status,
        headers={"Cache-Control": "no-store"},
    )


def _openai_error_payload(message: str, status: int) -> dict[str, object]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error" if status < 500 else "server_error",
            "param": None,
            "code": "generation_request_failed",
        }
    }


def _openai_error(message: str, status: int) -> web.Response:
    return web.json_response(
        _openai_error_payload(message, status),
        status=status,
        headers={"Cache-Control": "no-store"},
    )


def _status(error: BaseException) -> int:
    if isinstance(error, KeyError):
        return 404
    if isinstance(error, (GenerationSessionBusyError, GenerationModelBusyError)):
        return 409
    if isinstance(error, GenerationSessionClosedError):
        return 404
    if isinstance(error, (TypeError, ValueError)):
        return 400
    return 502


def _message(error: BaseException) -> str:
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    if isinstance(error, (TypeError, ValueError, RuntimeError)):
        return str(error)
    return "generation provider failed"


async def _write_sse(response: web.StreamResponse, *, event: str | None, data: object) -> None:
    prefix = b"" if event is None else f"event: {event}\n".encode()
    if data == "[DONE]":
        payload = b"[DONE]"
    else:
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode()
    await response.write(prefix + b"data: " + payload + b"\n\n")


async def _stream_native(
    request: web.Request,
    stream: _ManagedGenerationStream,
    cancelled: threading.Event,
) -> web.StreamResponse:
    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        }
    )
    terminal = False
    token_sequence = 0
    try:
        await response.prepare(request)
        while True:
            event = await _next_event(stream, cancelled)
            if event is _END:
                if not terminal:
                    raise RuntimeError("generation stream ended without a terminal event")
                break
            if terminal:
                raise RuntimeError("generation stream produced an event after its terminal event")
            if isinstance(event, GenerationTokenEvent):
                if event.sequence != token_sequence:
                    raise RuntimeError("generation token sequence is not contiguous")
                token_sequence += 1
                await _write_sse(
                    response,
                    event="token",
                    data={
                        "sequence": event.sequence,
                        "text": event.text,
                        "tokenId": event.token_id,
                    },
                )
            elif isinstance(event, GenerationTerminalEvent):
                terminal = True
                try:
                    await _write_sse(response, event="result", data=_native_result(event.result))
                except BaseException:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(stream.discard_session)
                    raise
            else:
                raise RuntimeError("generation stream produced an unknown event")
        await response.write_eof()
        return response
    except (ConnectionError, asyncio.CancelledError):
        cancelled.set()
        raise
    except Exception as error:
        if not response.prepared:
            raise
        with contextlib.suppress(ConnectionError, RuntimeError):
            await _write_sse(
                response,
                event="error",
                data={"error": "generation-stream-failed", "message": _message(error)},
            )
            await response.write_eof()
        return response
    finally:
        cancelled.set()
        await asyncio.to_thread(stream.close)


async def _handle_native(request: web.Request, service: GenerationService) -> web.StreamResponse:
    try:
        parsed = _parse_native(await _read_json(request))
        cancelled = threading.Event()
        stream = await _open_stream(
            service,
            parsed,
            principal_for(request).principal_id,
            cancelled,
        )
        if parsed.stream:
            return await _stream_native(request, stream, cancelled)
        _, result = await _collect(stream, cancelled)
        return web.json_response(_native_result(result), headers={"Cache-Control": "no-store"})
    except Exception as error:
        return _native_error(_message(error), _status(error))


def _completion_response(
    request_id: str,
    created: int,
    model: str,
    result: GenerationResult,
    *,
    chat: bool,
) -> dict[str, object]:
    choice: dict[str, object] = {
        "index": 0,
        "finish_reason": _finish_reason(result.finish_reason),
    }
    if chat:
        choice["message"] = {"role": "assistant", "content": result.text}
    else:
        choice["text"] = result.text
    return {
        "id": request_id,
        "object": "chat.completion" if chat else "text_completion",
        "created": created,
        "model": model,
        "choices": [choice],
        "usage": _usage(result),
    }


async def _stream_completion(
    request: web.Request,
    stream: _ManagedGenerationStream,
    cancelled: threading.Event,
    *,
    request_id: str,
    created: int,
    model: str,
    chat: bool,
    include_usage: bool,
) -> web.StreamResponse:
    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        }
    )
    terminal = False
    token_sequence = 0
    try:
        await response.prepare(request)
        while True:
            event = await _next_event(stream, cancelled)
            if event is _END:
                if not terminal:
                    raise RuntimeError("generation stream ended without a terminal event")
                break
            if terminal:
                raise RuntimeError("generation stream produced an event after its terminal event")
            if isinstance(event, GenerationTokenEvent):
                if event.sequence != token_sequence:
                    raise RuntimeError("generation token sequence is not contiguous")
                token_sequence += 1
                choice: dict[str, object] = {"index": 0, "finish_reason": None}
                if chat:
                    choice["delta"] = {"content": event.text}
                else:
                    choice["text"] = event.text
                chunk: dict[str, object] = {
                    "id": request_id,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [choice],
                }
                if include_usage:
                    chunk["usage"] = None
                await _write_sse(
                    response,
                    event=None,
                    data=chunk,
                )
            elif isinstance(event, GenerationTerminalEvent):
                terminal = True
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            **({"delta": {}} if chat else {"text": ""}),
                            "finish_reason": _finish_reason(event.result.finish_reason),
                        }
                    ],
                }
                if include_usage:
                    chunk["usage"] = None
                await _write_sse(
                    response,
                    event=None,
                    data=chunk,
                )
                if include_usage:
                    await _write_sse(
                        response,
                        event=None,
                        data={
                            "id": request_id,
                            "object": "chat.completion.chunk" if chat else "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": _usage(event.result),
                        },
                    )
            else:
                raise RuntimeError("generation stream produced an unknown event")
        await _write_sse(response, event=None, data="[DONE]")
        await response.write_eof()
        return response
    except (ConnectionError, asyncio.CancelledError):
        cancelled.set()
        raise
    except Exception as error:
        if not response.prepared:
            raise
        with contextlib.suppress(ConnectionError, RuntimeError):
            await _write_sse(
                response,
                event=None,
                data=_openai_error_payload(_message(error), _status(error)),
            )
            await _write_sse(response, event=None, data="[DONE]")
            await response.write_eof()
        return response
    finally:
        cancelled.set()
        await asyncio.to_thread(stream.close)


async def _handle_completion(
    request: web.Request,
    service: GenerationService,
    *,
    chat: bool,
) -> web.StreamResponse:
    try:
        parsed = _parse_openai(await _read_json(request), chat=chat)
        cancelled = threading.Event()
        stream = await _open_stream(
            service,
            parsed,
            principal_for(request).principal_id,
            cancelled,
        )
        request_id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        if parsed.stream:
            return await _stream_completion(
                request,
                stream,
                cancelled,
                request_id=request_id,
                created=created,
                model=parsed.model,
                chat=chat,
                include_usage=parsed.include_usage,
            )
        _, result = await _collect(stream, cancelled)
        return web.json_response(
            _completion_response(request_id, created, parsed.model, result, chat=chat),
            headers={"Cache-Control": "no-store"},
        )
    except Exception as error:
        return _openai_error(_message(error), _status(error))


def _response_payload(
    response_id: str,
    created: int,
    model: str,
    result: GenerationResult,
    *,
    item_id: str | None = None,
    previous_response_id: str | None = None,
) -> dict[str, object]:
    status = (
        "cancelled"
        if result.finish_reason is GenerationFinishReason.CANCELLED
        else "incomplete"
        if result.finish_reason is GenerationFinishReason.LENGTH
        else "completed"
    )
    message = _response_message(item_id or "msg_" + uuid.uuid4().hex, result.text, status)
    prompt_tokens = result.stats.prompt_tokens or 0
    output_tokens = result.stats.generated_tokens or 0
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "completed_at": created if status == "completed" else None,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"}
            if result.finish_reason is GenerationFinishReason.LENGTH
            else None
        ),
        "instructions": None,
        "max_output_tokens": None,
        "metadata": {},
        "model": model,
        "output": [message],
        "parallel_tool_calls": False,
        "previous_response_id": previous_response_id,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "none",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt_tokens + output_tokens,
        },
    }


def _response_message(item_id: str, text: str, status: str) -> dict[str, object]:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed" if status == "completed" else "incomplete",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


async def _stream_response(
    request: web.Request,
    stream: _ManagedGenerationStream,
    cancelled: threading.Event,
    *,
    response_id: str,
    created: int,
    model: str,
    previous_response_id: str | None,
) -> web.StreamResponse:
    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        }
    )
    item_id = "msg_" + uuid.uuid4().hex
    sequence = 0
    token_sequence = 0
    terminal = False
    initial = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "metadata": {},
        "model": model,
        "output": [],
        "parallel_tool_calls": False,
        "previous_response_id": previous_response_id,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "none",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": None,
    }

    async def emit(event_type: str, **values: object) -> None:
        nonlocal sequence
        await _write_sse(
            response,
            event=event_type,
            data={"type": event_type, "sequence_number": sequence, **values},
        )
        sequence += 1

    try:
        await response.prepare(request)
        await emit("response.created", response=initial)
        await emit("response.in_progress", response=initial)
        await emit(
            "response.output_item.added",
            output_index=0,
            item={
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        )
        await emit(
            "response.content_part.added",
            item_id=item_id,
            output_index=0,
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        )
        while True:
            event = await _next_event(stream, cancelled)
            if event is _END:
                if not terminal:
                    raise RuntimeError("generation stream ended without a terminal event")
                break
            if terminal:
                raise RuntimeError("generation stream produced an event after its terminal event")
            if isinstance(event, GenerationTokenEvent):
                if event.sequence != token_sequence:
                    raise RuntimeError("generation token sequence is not contiguous")
                token_sequence += 1
                await emit(
                    "response.output_text.delta",
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    delta=event.text,
                    logprobs=[],
                )
            elif isinstance(event, GenerationTerminalEvent):
                terminal = True
                completed = _response_payload(
                    response_id,
                    created,
                    model,
                    event.result,
                    item_id=item_id,
                    previous_response_id=previous_response_id,
                )
                status = cast("str", completed["status"])
                part = {
                    "type": "output_text",
                    "text": event.result.text,
                    "annotations": [],
                    "logprobs": [],
                }
                await emit(
                    "response.output_text.done",
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    text=event.result.text,
                    logprobs=[],
                )
                await emit(
                    "response.content_part.done",
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part=part,
                )
                await emit(
                    "response.output_item.done",
                    output_index=0,
                    item=_response_message(item_id, event.result.text, status),
                )
                await emit(
                    "response.completed" if status == "completed" else "response.incomplete",
                    response=completed,
                )
            else:
                raise RuntimeError("generation stream produced an unknown event")
        await response.write_eof()
        return response
    except (ConnectionError, asyncio.CancelledError):
        cancelled.set()
        raise
    except Exception as error:
        if not response.prepared:
            raise
        failed = {
            **initial,
            "status": "failed",
            "error": {"code": "server_error", "message": _message(error)},
        }
        with contextlib.suppress(ConnectionError, RuntimeError):
            await emit("response.failed", response=failed)
            await response.write_eof()
        return response
    finally:
        cancelled.set()
        await asyncio.to_thread(stream.close)


async def _handle_responses(request: web.Request, service: GenerationService) -> web.StreamResponse:
    try:
        parsed = _parse_responses(await _read_json(request))
        cancelled = threading.Event()
        stream = await _open_stream(
            service,
            parsed,
            principal_for(request).principal_id,
            cancelled,
        )
        public_session_id = await asyncio.to_thread(
            service.public_session_id,
            stream,
            replace=parsed.session_id is not None,
        )
        response_id = "resp_" + public_session_id
        previous_response_id = None if parsed.session_id is None else "resp_" + parsed.session_id
        created = int(time.time())
        if parsed.stream:
            return await _stream_response(
                request,
                stream,
                cancelled,
                response_id=response_id,
                created=created,
                model=parsed.model,
                previous_response_id=previous_response_id,
            )
        _, result = await _collect(stream, cancelled)
        return web.json_response(
            _response_payload(
                response_id,
                created,
                parsed.model,
                result,
                previous_response_id=previous_response_id,
            ),
            headers={"Cache-Control": "no-store"},
        )
    except Exception as error:
        return _openai_error(_message(error), _status(error))


async def _model_action(
    request: web.Request,
    service: GenerationService,
    action: Callable[[str], None],
) -> web.Response:
    try:
        values = await _read_json(request)
        _only(values, frozenset({"model"}), "generation model request")
        model = _string(values.get("model"), "model", allow_empty=False)
        await asyncio.to_thread(action, model)
        return web.json_response(
            {"model": model, "status": "ok"},
            headers={"Cache-Control": "no-store"},
        )
    except Exception as error:
        return _native_error(_message(error), _status(error))


def add_generation_routes(
    app: web.Application,
    service: GenerationService,
    *,
    register_cleanup: bool = True,
) -> None:
    """Mount native and OpenAI-compatible adapters over one generation service."""
    if type(service) is not GenerationService:
        raise TypeError("generation routes require a GenerationService")

    async def native(request: web.Request) -> web.StreamResponse:
        return await _handle_native(request, service)

    async def completions(request: web.Request) -> web.StreamResponse:
        return await _handle_completion(request, service, chat=False)

    async def chat_completions(request: web.Request) -> web.StreamResponse:
        return await _handle_completion(request, service, chat=True)

    async def responses(request: web.Request) -> web.StreamResponse:
        return await _handle_responses(request, service)

    async def models(_: web.Request) -> web.Response:
        return web.json_response(
            {"object": "list", "data": await asyncio.to_thread(service.model_rows)},
            headers={"Cache-Control": "no-store"},
        )

    async def load(request: web.Request) -> web.Response:
        return await _model_action(request, service, service.load_model)

    async def unload(request: web.Request) -> web.Response:
        return await _model_action(request, service, service.unload_model)

    async def close_session(request: web.Request) -> web.Response:
        try:
            session_id = _string(
                request.match_info["session_id"],
                "session_id",
                allow_empty=False,
            )
            await asyncio.to_thread(
                service.close_session,
                session_id,
                owner=principal_for(request).principal_id,
            )
            return web.Response(status=204)
        except Exception as error:
            return _native_error(_message(error), _status(error))

    app.router.add_post("/api/generation", native)
    app.router.add_get("/api/generation/models", models)
    app.router.add_post("/api/generation/models/load", load)
    app.router.add_post("/api/generation/models/unload", unload)
    app.router.add_delete("/api/generation/sessions/{session_id}", close_session)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/responses", responses)

    async def close_generation(_: web.Application) -> None:
        await asyncio.to_thread(service.close)

    if register_cleanup:
        app.on_cleanup.append(close_generation)


__all__ = [
    "GenerationModel",
    "GenerationModelBusyError",
    "GenerationService",
    "GenerationServiceClosedError",
    "add_generation_routes",
]
