"""Torch-free contracts for autoregressive text generation.

ComfyUI's current TextGenerate path passes loose sampler arguments into one
model call and keeps its mutable KV state inside the text model
(comfy_extras/nodes_textgen.py and comfy/text_encoders/llama.py @ 0a33ed6c).
These contracts separate logical requests, provider-owned sessions, and a
pull-based event stream so native and forwarded providers expose one surface.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Protocol, Self, TypeAlias, cast

from .registry import validate_registry_id


class GenerationMessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class GenerationMessage:
    role: GenerationMessageRole
    content: str

    def __post_init__(self) -> None:
        if type(cast("object", self.role)) is not GenerationMessageRole:
            raise TypeError("generation message role must be a GenerationMessageRole")
        if type(cast("object", self.content)) is not str:
            raise TypeError("generation message content must be a string")


_SESSION_ID_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class GenerationSessionHandle:
    """Reference to mutable generation state owned by exactly one provider."""

    provider_id: str
    model_identity: str
    session_id: str

    def __post_init__(self) -> None:
        _validate_provider_id(self.provider_id)
        _validate_model_identity(self.model_identity)
        session_id = cast("object", self.session_id)
        if not (
            type(session_id) is str
            and 32 <= len(session_id) <= 64
            and all(char in _SESSION_ID_CHARS for char in session_id)
        ):
            raise ValueError("generation session id must be 32-64 lowercase hex characters")


class GenerationSessionError(RuntimeError):
    """Base error for generation-session lifecycle violations."""


class GenerationSessionBusyError(GenerationSessionError):
    """Raised when an operation targets a session with an active request."""


class GenerationSessionClosedError(GenerationSessionError):
    """Raised when generation targets a closed or unknown session."""


class GenerationSamplerKind(StrEnum):
    REPETITION_PENALTY = "repetition_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    FREQUENCY_PENALTY = "frequency_penalty"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"
    MIN_P = "min_p"
    TYPICAL_P = "typical_p"
    GREEDY = "greedy"
    MULTINOMIAL = "multinomial"


_SELECTION_SAMPLERS = frozenset({GenerationSamplerKind.GREEDY, GenerationSamplerKind.MULTINOMIAL})
_PROBABILITY_SAMPLERS = frozenset(
    {
        GenerationSamplerKind.TOP_P,
        GenerationSamplerKind.MIN_P,
        GenerationSamplerKind.TYPICAL_P,
    }
)


@dataclass(frozen=True, slots=True)
class GenerationSamplerStage:
    """One logits transform or final token-selection operation.

    Penalties consume the complete prompt and generated-token history. Providers
    whose penalty scope differs must refuse those stages.
    """

    kind: GenerationSamplerKind
    value: int | float | None = None

    def __post_init__(self) -> None:
        if type(cast("object", self.kind)) is not GenerationSamplerKind:
            raise TypeError("generation sampler kind must be a GenerationSamplerKind")
        value = cast("object", self.value)
        if self.kind in _SELECTION_SAMPLERS:
            if value is not None:
                raise ValueError(f"{self.kind.value} does not accept a value")
            return
        if self.kind is GenerationSamplerKind.TOP_K:
            if type(value) is not int:
                raise TypeError("top_k value must be an exact integer")
            if not 1 <= value <= 2**31 - 1:
                raise ValueError("top_k value must be in [1, 2^31 - 1]")
            return
        if type(value) is not float:
            raise TypeError(f"{self.kind.value} value must be an exact float")
        if not math.isfinite(value):
            raise ValueError(f"{self.kind.value} value must be finite")
        if self.kind in _PROBABILITY_SAMPLERS and not 0.0 <= value <= 1.0:
            raise ValueError(f"{self.kind.value} value must be in [0, 1]")
        if self.kind is GenerationSamplerKind.TEMPERATURE and value <= 0.0:
            raise ValueError("temperature value must be positive")
        if self.kind is GenerationSamplerKind.REPETITION_PENALTY and value <= 0.0:
            raise ValueError("repetition_penalty value must be positive")


@dataclass(frozen=True, slots=True)
class GenerationSamplerChain:
    """Ordered logits operations ending in exactly one token selector."""

    stages: tuple[GenerationSamplerStage, ...]

    def __post_init__(self) -> None:
        if type(cast("object", self.stages)) is not tuple:
            raise TypeError("generation sampler stages must be a tuple")
        if not self.stages:
            raise ValueError("generation sampler chain must not be empty")
        if any(type(cast("object", stage)) is not GenerationSamplerStage for stage in self.stages):
            raise TypeError(
                "generation sampler chain entries must be GenerationSamplerStage values"
            )
        kinds = tuple(stage.kind for stage in self.stages)
        if len(kinds) != len(set(kinds)):
            raise ValueError("generation sampler kinds must be unique within a chain")
        selectors = tuple(kind for kind in kinds if kind in _SELECTION_SAMPLERS)
        if len(selectors) != 1 or kinds[-1] not in _SELECTION_SAMPLERS:
            raise ValueError("generation sampler chain must end in exactly one token selector")


@dataclass(frozen=True, slots=True)
class GenerationStopConditions:
    max_new_tokens: int
    stop_texts: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(cast("object", self.max_new_tokens)) is not int:
            raise TypeError("max_new_tokens must be an exact integer")
        if not 1 <= self.max_new_tokens <= 1_000_000:
            raise ValueError("max_new_tokens must be in [1, 1000000]")
        if type(cast("object", self.stop_texts)) is not tuple:
            raise TypeError("stop_texts must be a tuple")
        if any(type(cast("object", text)) is not str or not text for text in self.stop_texts):
            raise ValueError("stop_texts entries must be non-empty strings")
        if len(self.stop_texts) != len(set(self.stop_texts)):
            raise ValueError("stop_texts entries must be unique")
        if type(cast("object", self.stop_token_ids)) is not tuple:
            raise TypeError("stop_token_ids must be a tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop_token_ids entries must be non-negative exact integers")
        if len(self.stop_token_ids) != len(set(self.stop_token_ids)):
            raise ValueError("stop_token_ids entries must be unique")


_GREEDY_SAMPLER_CHAIN = GenerationSamplerChain(
    (GenerationSamplerStage(GenerationSamplerKind.GREEDY),)
)
_DEFAULT_STOP_CONDITIONS = GenerationStopConditions(128)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Logical request semantics, independent of provider batching policy."""

    provider_id: str
    model_identity: str
    prompt: str | None = None
    messages: tuple[GenerationMessage, ...] = ()
    sampler: GenerationSamplerChain = _GREEDY_SAMPLER_CHAIN
    stop: GenerationStopConditions = _DEFAULT_STOP_CONDITIONS
    seed: int | None = None
    session: GenerationSessionHandle | None = None
    open_session: bool = False

    def __post_init__(self) -> None:
        _validate_provider_id(self.provider_id)
        _validate_model_identity(self.model_identity)
        prompt = cast("object", self.prompt)
        if prompt is not None and type(prompt) is not str:
            raise TypeError("generation prompt must be a string or None")
        if type(cast("object", self.messages)) is not tuple:
            raise TypeError("generation messages must be a tuple")
        if any(type(cast("object", message)) is not GenerationMessage for message in self.messages):
            raise TypeError("generation messages must contain GenerationMessage values")
        if (self.prompt is None) == (not self.messages):
            raise ValueError("generation request requires exactly one of prompt or messages")
        if type(cast("object", self.sampler)) is not GenerationSamplerChain:
            raise TypeError("generation sampler must be a GenerationSamplerChain")
        if type(cast("object", self.stop)) is not GenerationStopConditions:
            raise TypeError("generation stop must be GenerationStopConditions")
        if self.seed is not None and (
            type(cast("object", self.seed)) is not int or not 0 <= self.seed <= 2**64 - 1
        ):
            raise ValueError("generation seed must be a uint64 or None")
        if type(cast("object", self.open_session)) is not bool:
            raise TypeError("open_session must be an exact boolean")
        if self.session is not None:
            if type(cast("object", self.session)) is not GenerationSessionHandle:
                raise TypeError("generation session must be a GenerationSessionHandle or None")
            if self.open_session:
                raise ValueError("open_session and an existing session are mutually exclusive")
            if self.session.provider_id != self.provider_id:
                raise ValueError("generation session provider does not match the request")
            if self.session.model_identity != self.model_identity:
                raise ValueError("generation session model identity does not match the request")


class GenerationFinishReason(StrEnum):
    EOS = "eos"
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class GenerationStats:
    total_time_s: float
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    time_to_first_token_s: float | None = None
    prefill_time_s: float | None = None
    decode_time_s: float | None = None

    def __post_init__(self) -> None:
        _validate_duration("total_time_s", self.total_time_s)
        for name in ("prompt_tokens", "generated_tokens"):
            value = cast("object", getattr(self, name))
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative exact integer or None")
        for name in ("time_to_first_token_s", "prefill_time_s", "decode_time_s"):
            value = cast("object", getattr(self, name))
            if value is not None:
                duration = _validate_duration(name, value)
                if duration > self.total_time_s:
                    raise ValueError(f"{name} must not exceed total_time_s")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    finish_reason: GenerationFinishReason
    stats: GenerationStats
    token_ids: tuple[int, ...] | None = None
    continuation: GenerationSessionHandle | None = None

    def __post_init__(self) -> None:
        if type(cast("object", self.text)) is not str:
            raise TypeError("generation result text must be a string")
        if type(cast("object", self.finish_reason)) is not GenerationFinishReason:
            raise TypeError("generation finish reason must be a GenerationFinishReason")
        if type(cast("object", self.stats)) is not GenerationStats:
            raise TypeError("generation result stats must be GenerationStats")
        if self.token_ids is not None:
            if type(cast("object", self.token_ids)) is not tuple:
                raise TypeError("generation result token_ids must be a tuple or None")
            if any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids):
                raise ValueError("generation result token_ids must be non-negative exact integers")
        if (
            self.continuation is not None
            and type(cast("object", self.continuation)) is not GenerationSessionHandle
        ):
            raise TypeError("generation continuation must be a GenerationSessionHandle or None")


@dataclass(frozen=True, slots=True)
class GenerationTokenEvent:
    sequence: int
    text: str
    token_id: int | None = None

    def __post_init__(self) -> None:
        if type(cast("object", self.sequence)) is not int or self.sequence < 0:
            raise ValueError("generation token event sequence must be a non-negative exact integer")
        if type(cast("object", self.text)) is not str:
            raise TypeError("generation token event text must be a string")
        if self.token_id is not None and (
            type(cast("object", self.token_id)) is not int or self.token_id < 0
        ):
            raise ValueError("generation token event token_id must be non-negative or None")


@dataclass(frozen=True, slots=True)
class GenerationTerminalEvent:
    result: GenerationResult

    def __post_init__(self) -> None:
        if type(cast("object", self.result)) is not GenerationResult:
            raise TypeError("generation terminal event result must be a GenerationResult")


GenerationEvent: TypeAlias = GenerationTokenEvent | GenerationTerminalEvent


class GenerationStream(Iterator[GenerationEvent], Protocol):
    """Pull-based event iterator with explicit abandonment cleanup.

    ``close`` is idempotent and completes request rollback and provisional
    session cleanup before returning. The context manager closes the stream on
    every exit, including an early loop break or consumer exception.
    """

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class GenerationProviderCapabilities:
    """Request features a provider can preserve without semantic downgrade."""

    sampler_kinds: frozenset[GenerationSamplerKind]
    chat: bool = False
    sessions: bool = False
    token_ids: bool = False
    ordered_sampler_chain: bool = False

    def __post_init__(self) -> None:
        if type(cast("object", self.sampler_kinds)) is not frozenset or any(
            type(cast("object", kind)) is not GenerationSamplerKind for kind in self.sampler_kinds
        ):
            raise TypeError("provider sampler_kinds must be a frozenset of GenerationSamplerKind")
        if not self.sampler_kinds.intersection(_SELECTION_SAMPLERS):
            raise ValueError("provider must support at least one token selector")
        for name in ("chat", "sessions", "token_ids", "ordered_sampler_chain"):
            if type(cast("object", getattr(self, name))) is not bool:
                raise TypeError(f"provider capability {name} must be an exact boolean")


class GenerationProvider(Protocol):
    """Common pull-based stream for native and forwarded generation.

    ``generate`` yields zero or more contiguous token events followed by one
    terminal event and then ends. Pulling the next event grants backpressure;
    consumers own streams with a ``with`` statement, and providers also poll
    ``cancelled`` while work is in flight. Provider execution batching and
    transport chunking do not alter the request or event semantics.

    A request using an existing session is one transaction. An EOS, stop, or
    length terminal result atomically commits its generated tokens. A cancelled
    terminal result, stream close, or provider exception rolls back all request
    mutations before releasing the session; cancelled results return the input
    handle as their continuation. An opened session remains provisional until
    a successful terminal result publishes its handle. Every other exit
    disposes it, and a cancelled result has no continuation.

    Each session is single-flight. ``generate`` raises
    ``GenerationSessionBusyError`` before returning a second stream for an
    active handle, and raises ``GenerationSessionClosedError`` for a closed or
    unknown handle. ``close_session`` refuses an active session with
    ``GenerationSessionBusyError``; otherwise it is idempotent.
    """

    @property
    def id(self) -> str: ...

    @property
    def capabilities(self) -> GenerationProviderCapabilities: ...

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> GenerationStream: ...

    def close_session(self, session: GenerationSessionHandle) -> None: ...


def _validate_provider_id(provider_id: str) -> None:
    if type(cast("object", provider_id)) is not str:
        raise TypeError("generation provider id must be a string")
    validate_registry_id(provider_id)


def _validate_model_identity(model_identity: str) -> None:
    if type(cast("object", model_identity)) is not str:
        raise TypeError("generation model identity must be a string")
    if not model_identity:
        raise ValueError("generation model identity must be non-empty")


def _validate_duration(name: str, value: object) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


__all__ = [
    "GenerationEvent",
    "GenerationFinishReason",
    "GenerationMessage",
    "GenerationMessageRole",
    "GenerationProvider",
    "GenerationProviderCapabilities",
    "GenerationRequest",
    "GenerationResult",
    "GenerationSamplerChain",
    "GenerationSamplerKind",
    "GenerationSamplerStage",
    "GenerationSessionBusyError",
    "GenerationSessionClosedError",
    "GenerationSessionError",
    "GenerationSessionHandle",
    "GenerationStats",
    "GenerationStopConditions",
    "GenerationStream",
    "GenerationTerminalEvent",
    "GenerationTokenEvent",
]
