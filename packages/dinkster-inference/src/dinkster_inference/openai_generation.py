"""OpenAI-compatible HTTP provider for neutral generation contracts."""

from __future__ import annotations

import asyncio
import codecs
import concurrent.futures
import hashlib
import io
import json
import math
import queue
import socket
import threading
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Any, Self, cast
from urllib.parse import urlsplit

import aiohttp
from dinkster_values import MEBIBYTE

from .generation import (
    GenerationEvent,
    GenerationFinishReason,
    GenerationProviderCapabilities,
    GenerationRequest,
    GenerationResult,
    GenerationSamplerKind,
    GenerationSessionClosedError,
    GenerationSessionHandle,
    GenerationStats,
    GenerationTerminalEvent,
    GenerationTokenEvent,
)
from .registry import validate_registry_id

_MAX_ERROR_BYTES = 64 * 1024
_MAX_JSON_BYTES = 16 * MEBIBYTE
_MAX_SSE_EVENT_CHARS = MEBIBYTE
_MAX_COMPLETION_CHARS = 16 * MEBIBYTE
_CANCEL_POLL_SECONDS = 0.05
_REDACTED = "[REDACTED]"

_STANDARD_SAMPLERS = frozenset(
    {
        GenerationSamplerKind.TEMPERATURE,
        GenerationSamplerKind.TOP_P,
        GenerationSamplerKind.GREEDY,
        GenerationSamplerKind.MULTINOMIAL,
    }
)
_LLAMA_CPP_SAMPLERS = frozenset(
    kind
    for kind in GenerationSamplerKind
    if kind
    not in (
        GenerationSamplerKind.REPETITION_PENALTY,
        GenerationSamplerKind.PRESENCE_PENALTY,
        GenerationSamplerKind.FREQUENCY_PENALTY,
    )
)
_SELECTORS = frozenset(
    {
        GenerationSamplerKind.GREEDY,
        GenerationSamplerKind.MULTINOMIAL,
    }
)
_LLAMA_CPP_SAMPLER_NAMES = {
    GenerationSamplerKind.TEMPERATURE: "temperature",
    GenerationSamplerKind.TOP_K: "top_k",
    GenerationSamplerKind.TOP_P: "top_p",
    GenerationSamplerKind.MIN_P: "min_p",
    GenerationSamplerKind.TYPICAL_P: "typ_p",
}


class OpenAICompatibility(StrEnum):
    """Wire capabilities guaranteed by one configured endpoint."""

    STANDARD = "openai"
    LLAMA_CPP = "llama.cpp"


class OpenAIGenerationError(RuntimeError):
    """An OpenAI-compatible request or response violated its contract."""


@dataclass(frozen=True, slots=True)
class _Failure:
    error: OpenAIGenerationError


_QueueItem = GenerationEvent | _Failure


@dataclass(frozen=True, slots=True)
class _Usage:
    prompt_tokens: int
    completion_tokens: int


class OpenAIGenerationProvider:
    """Forward generation to one explicitly configured OpenAI-compatible model.

    The API key is transport authority only. It is excluded from provider and
    model identities, request bodies, representations, and diagnostics.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        provider_id: str = "dinkster.openai",
        compatibility: OpenAICompatibility = OpenAICompatibility.STANDARD,
        stream: bool = True,
        timeout_s: float = 300.0,
        proxy_socket: str | None = None,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        if type(cast("object", model)) is not str:
            raise TypeError("OpenAI model must be a string")
        if not model:
            raise ValueError("OpenAI model must be a non-empty string")
        if len(model) > 1024 or any(ord(char) < 32 for char in model):
            raise ValueError("OpenAI model must be a bounded printable string")
        if api_key is not None and type(cast("object", api_key)) is not str:
            raise TypeError("OpenAI API key must be a string or None")
        if api_key is not None and (
            not api_key or any(ord(char) < 32 or ord(char) == 127 for char in api_key)
        ):
            raise ValueError("OpenAI API key must be non-empty and header-safe")
        if type(cast("object", provider_id)) is not str:
            raise TypeError("OpenAI provider ID must be a string")
        validate_registry_id(provider_id)
        if type(cast("object", compatibility)) is not OpenAICompatibility:
            raise TypeError("OpenAI compatibility must be an OpenAICompatibility value")
        if type(cast("object", stream)) is not bool:
            raise TypeError("OpenAI stream must be an exact boolean")
        if type(cast("object", timeout_s)) is not float:
            raise TypeError("OpenAI timeout must be an exact float")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("OpenAI timeout must be finite and positive")
        if proxy_socket is not None:
            if type(cast("object", proxy_socket)) is not str:
                raise TypeError("OpenAI proxy socket must be an exact string or None")
            if not proxy_socket or not proxy_socket.isprintable():
                raise ValueError("OpenAI proxy socket must be a non-empty printable path")

        self._remote_model = model
        self._api_key = api_key
        self._id = provider_id
        self._compatibility = compatibility
        self._stream_responses = stream
        self._timeout_s = timeout_s
        identity_payload = json.dumps(
            {
                "base_url": self._base_url,
                "compatibility": compatibility.value,
                "model": model,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self._model_identity = f"openai:sha256:{hashlib.sha256(identity_payload).hexdigest()}"
        self._capabilities = GenerationProviderCapabilities(
            _LLAMA_CPP_SAMPLERS
            if compatibility is OpenAICompatibility.LLAMA_CPP
            else _STANDARD_SAMPLERS,
            chat=True,
            ordered_sampler_chain=compatibility is OpenAICompatibility.LLAMA_CPP,
        )
        self._lock = threading.RLock()
        self._runtime = _OpenAIRuntime(timeout_s, proxy_socket)
        self._streams: set[_OpenAIGenerationStream] = set()
        self._closed = False

    @property
    def id(self) -> str:
        return self._id

    @property
    def model_identity(self) -> str:
        return self._model_identity

    @property
    def remote_model(self) -> str:
        return self._remote_model

    @property
    def capabilities(self) -> GenerationProviderCapabilities:
        return self._capabilities

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self._base_url!r}, "
            f"model={self._remote_model!r}, provider_id={self._id!r}, "
            f"compatibility={self._compatibility.value!r}, "
            f"stream={self._stream_responses!r}, timeout_s={self._timeout_s!r})"
        )

    def __enter__(self) -> Self:
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenAI generation provider is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _OpenAIGenerationStream:
        if type(request) is not GenerationRequest:
            raise TypeError("OpenAI generation request must be a GenerationRequest")
        if not callable(cancelled):
            raise TypeError("OpenAI cancellation callback must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenAI generation provider is closed")
        self._validate_request(request)
        payload = self._payload(request)
        path = "/chat/completions" if request.messages else "/completions"
        stream = _OpenAIGenerationStream(
            self,
            cancelled,
            f"{self._base_url}{path}",
            payload,
            chat=bool(request.messages),
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenAI generation provider is closed")
            self._streams.add(stream)
        return stream

    def close_session(self, session: GenerationSessionHandle) -> None:
        if type(session) is not GenerationSessionHandle:
            raise TypeError("generation session must be a GenerationSessionHandle")
        raise GenerationSessionClosedError(
            "OpenAI-compatible stateless provider has no persistent generation sessions"
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            streams = tuple(self._streams)
        for stream in streams:
            stream.close()
        self._runtime.close()

    def _release(self, stream: _OpenAIGenerationStream) -> None:
        with self._lock:
            self._streams.discard(stream)

    def _validate_request(self, request: GenerationRequest) -> None:
        if request.provider_id != self._id:
            raise ValueError("generation request targets a different provider")
        if request.model_identity != self._model_identity:
            raise ValueError("generation request targets a different OpenAI model identity")
        if request.session is not None or request.open_session:
            raise ValueError(
                "OpenAI-compatible stateless provider does not advertise persistent sessions"
            )
        if request.stop.stop_token_ids:
            raise ValueError("OpenAI-compatible generation cannot preserve token-ID stops")
        penalties = tuple(
            stage.kind.value
            for stage in request.sampler.stages
            if stage.kind
            in (
                GenerationSamplerKind.REPETITION_PENALTY,
                GenerationSamplerKind.PRESENCE_PENALTY,
                GenerationSamplerKind.FREQUENCY_PENALTY,
            )
        )
        if penalties:
            names = ", ".join(penalties)
            raise ValueError(
                "OpenAI-compatible generation cannot guarantee complete-history "
                f"semantics for penalty stages: {names}"
            )
        unsupported = tuple(
            stage.kind
            for stage in request.sampler.stages
            if stage.kind not in self._capabilities.sampler_kinds
        )
        if unsupported:
            names = ", ".join(kind.value for kind in unsupported)
            raise ValueError(f"OpenAI endpoint does not support sampler stages: {names}")
        transforms = tuple(
            stage for stage in request.sampler.stages if stage.kind not in _SELECTORS
        )
        if self._compatibility is OpenAICompatibility.STANDARD and len(transforms) > 1:
            raise ValueError(
                "standard OpenAI endpoints do not preserve ordered multi-stage sampling"
            )

    def _payload(self, request: GenerationRequest) -> dict[str, object]:
        selector = request.sampler.stages[-1].kind
        payload: dict[str, object] = {
            "model": self._remote_model,
            "max_tokens": request.stop.max_new_tokens,
            "stream": self._stream_responses,
            "temperature": 0.0 if selector is GenerationSamplerKind.GREEDY else 1.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        if self._stream_responses:
            payload["stream_options"] = {"include_usage": True}
        if request.prompt is not None:
            payload["prompt"] = request.prompt
        else:
            payload["messages"] = [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ]
        if request.stop.stop_texts:
            payload["stop"] = list(request.stop.stop_texts)
        if request.seed is not None and selector is not GenerationSamplerKind.GREEDY:
            payload["seed"] = request.seed

        sampler_names: list[str] = []
        if self._compatibility is OpenAICompatibility.LLAMA_CPP:
            payload.update(
                {
                    "repeat_penalty": 1.0,
                    "top_k": 0,
                    "min_p": 0.0,
                    "typical_p": 1.0,
                }
            )
        for stage in request.sampler.stages[:-1]:
            kind = stage.kind
            value = cast("int | float", stage.value)
            if kind is GenerationSamplerKind.TEMPERATURE:
                payload["temperature"] = value
            elif kind is GenerationSamplerKind.TOP_K:
                payload["top_k"] = value
            elif kind is GenerationSamplerKind.TOP_P:
                payload["top_p"] = value
            elif kind is GenerationSamplerKind.MIN_P:
                payload["min_p"] = value
            elif kind is GenerationSamplerKind.TYPICAL_P:
                payload["typical_p"] = value
            if self._compatibility is OpenAICompatibility.LLAMA_CPP:
                sampler_names.append(_LLAMA_CPP_SAMPLER_NAMES[kind])
        if selector is GenerationSamplerKind.GREEDY:
            payload["temperature"] = 0.0
        if self._compatibility is OpenAICompatibility.LLAMA_CPP:
            if selector is GenerationSamplerKind.GREEDY:
                sampler_names = ["temperature"]
            payload["samplers"] = sampler_names
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream" if self._stream_responses else "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _safe_error(self, error: BaseException) -> OpenAIGenerationError:
        if isinstance(error, OpenAIGenerationError):
            message = str(error)
        elif isinstance(error, TimeoutError):
            message = f"OpenAI generation timed out after {self._timeout_s:g} seconds"
        elif isinstance(error, aiohttp.ClientError):
            message = f"OpenAI transport failed: {error}"
        else:
            message = f"OpenAI generation failed with {type(error).__name__}"
        if self._api_key is not None:
            message = message.replace(self._api_key, _REDACTED)
        return OpenAIGenerationError(message)


class _OpenAIRuntime:
    """One pooled aiohttp client running outside synchronous node execution."""

    def __init__(self, timeout_s: float, proxy_socket: str | None) -> None:
        self._timeout_s = timeout_s
        self._proxy_socket = proxy_socket
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: aiohttp.ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._proxy_url: str | None = None
        self._startup_error: BaseException | None = None
        self._futures: set[concurrent.futures.Future[None]] = set()
        self._closed = False

    def submit(self, coroutine: Coroutine[Any, Any, None]) -> concurrent.futures.Future[None]:
        self._start()
        with self._lock:
            loop = self._loop
            if self._closed or loop is None:
                coroutine.close()
                raise RuntimeError("OpenAI HTTP runtime is closed")
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            self._futures.add(future)
        future.add_done_callback(self._discard_future)
        return future

    def call_soon(self, callback: Callable[[], object]) -> None:
        with self._lock:
            loop = self._loop
            if not self._closed and loop is not None:
                loop.call_soon_threadsafe(callback)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            loop = self._loop
            stop = self._stop
            futures = tuple(self._futures)
        for future in futures:
            future.cancel()
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(stop.set)
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _discard_future(self, future: concurrent.futures.Future[None]) -> None:
        with self._lock:
            self._futures.discard(future)

    def session(self) -> aiohttp.ClientSession:
        session = self._session
        if session is None:
            raise RuntimeError("OpenAI HTTP runtime did not create its client session")
        return session

    def timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(total=self._timeout_s)

    def proxy(self) -> str | None:
        return self._proxy_url

    def _start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenAI HTTP runtime is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._thread_main,
                    name="dinkster-openai-http",
                    daemon=True,
                )
                self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("OpenAI HTTP runtime failed to start") from self._startup_error

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lock:
            self._loop = loop
        try:
            loop.run_until_complete(self._run())
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
        finally:
            loop.close()

    async def _run(self) -> None:
        stop = asyncio.Event()
        with self._lock:
            self._stop = stop
            closed = self._closed
        if closed:
            stop.set()
        connector = (
            None
            if self._proxy_socket is None
            else aiohttp.TCPConnector(socket_factory=self._proxy_socket_factory)
        )
        self._proxy_url = None if connector is None else "http://127.0.0.1:1"
        try:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=None),
                raise_for_status=False,
            ) as session:
                self._session = session
                self._ready.set()
                await stop.wait()
        finally:
            self._session = None
            self._proxy_url = None

    def _proxy_socket_factory(self, _address: object) -> socket.socket:
        path = self._proxy_socket
        if path is None:
            raise RuntimeError("OpenAI HTTP runtime has no Unix proxy socket")
        return _UnixProxySocket(path)


class _UnixProxySocket(socket.socket):
    """Route aiohttp's proxy TCP dial into Dinkster's Unix CONNECT proxy."""

    def __init__(self, path: str) -> None:
        family = getattr(socket, "AF_UNIX", None)
        if not isinstance(family, int):
            raise RuntimeError("OpenAI Unix proxy sockets are unavailable on this platform")
        super().__init__(family, socket.SOCK_STREAM)
        self._path = path

    def connect(self, _address: object) -> None:
        super().connect(self._path)


class _OpenAIGenerationStream:
    def __init__(
        self,
        provider: OpenAIGenerationProvider,
        cancelled: Callable[[], bool],
        url: str,
        payload: Mapping[str, object],
        *,
        chat: bool,
    ) -> None:
        self._provider = provider
        self._cancelled = cancelled
        self._url = url
        self._payload = payload
        self._chat = chat
        self._queue: queue.Queue[_QueueItem] = queue.Queue()
        self._lock = threading.Lock()
        self._future: concurrent.futures.Future[None] | None = None
        self._ack: asyncio.Event | None = None
        self._abandoned = False
        self._finished = False
        self._closed = False
        self._text = io.StringIO()
        self._text_chars = 0
        self._sequence = 0
        self._finish_reason: GenerationFinishReason | None = None
        self._usage: _Usage | None = None
        self._started: float | None = None
        self._first_token_at: float | None = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> GenerationEvent:
        if self._finished:
            raise StopIteration
        while True:
            try:
                cancelled = self._cancelled()
            except BaseException:
                self.close()
                raise
            if cancelled:
                return self._cancelled_terminal()
            self._ensure_started()
            try:
                item = self._queue.get(timeout=_CANCEL_POLL_SECONDS)
            except queue.Empty:
                future = self._future
                if future is not None and future.done():
                    self._wait_future()
                    self._finished = True
                    raise OpenAIGenerationError(
                        "OpenAI generation ended without a terminal event"
                    ) from None
                continue
            self._acknowledge()
            if isinstance(item, _Failure):
                self._finished = True
                raise item.error
            if isinstance(item, GenerationTerminalEvent):
                self._finished = True
            return item

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            abandon = not self._finished
            if abandon:
                self._abandoned = True
            future = self._future
            ack = self._ack
        if ack is not None:
            self._provider._runtime.call_soon(ack.set)  # pyright: ignore[reportPrivateUsage]
        if abandon and future is not None:
            future.cancel()
        self._wait_future()
        self._finished = True
        self._provider._release(self)  # pyright: ignore[reportPrivateUsage]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_started(self) -> None:
        if self._future is None:
            self._started = time.perf_counter()
            self._future = self._provider._runtime.submit(  # pyright: ignore[reportPrivateUsage]
                self._run()
            )

    async def _run(self) -> None:
        try:
            session = self._provider._runtime.session()  # pyright: ignore[reportPrivateUsage]
            timeout = self._provider._runtime.timeout()  # pyright: ignore[reportPrivateUsage]
            async with session.post(
                self._url,
                json=self._payload,
                headers=self._provider._headers(),  # pyright: ignore[reportPrivateUsage]
                timeout=timeout,
                allow_redirects=False,
                proxy=self._provider._runtime.proxy(),  # pyright: ignore[reportPrivateUsage]
            ) as response:
                if not 200 <= response.status < 300:
                    raise await _http_error(response)
                if self._provider._stream_responses:  # pyright: ignore[reportPrivateUsage]
                    await self._run_sse(response)
                else:
                    await self._run_json(response)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            await self._publish(_Failure(self._provider._safe_error(error)))  # pyright: ignore[reportPrivateUsage]
        finally:
            self._provider._release(self)  # pyright: ignore[reportPrivateUsage]

    async def _run_sse(self, response: aiohttp.ClientResponse) -> None:
        _require_content_type(response, "text/event-stream")
        parser = _SSEParser()
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        async for chunk in response.content.iter_chunked(16 * 1024):
            try:
                text = decoder.decode(chunk, final=False)
            except UnicodeDecodeError as error:
                raise OpenAIGenerationError("OpenAI SSE response is not valid UTF-8") from error
            for data in parser.feed(text):
                if data == "[DONE]":
                    await self._publish_terminal()
                    return
                await self._accept_packet(_decode_json(data), streaming=True)
        try:
            trailing = decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise OpenAIGenerationError("OpenAI SSE response is not valid UTF-8") from error
        for data in parser.feed(trailing, final=True):
            if data == "[DONE]":
                await self._publish_terminal()
                return
            await self._accept_packet(_decode_json(data), streaming=True)
        raise OpenAIGenerationError("OpenAI SSE response ended without data: [DONE]")

    async def _run_json(self, response: aiohttp.ClientResponse) -> None:
        _require_json_content_type(response)
        encoded = await _read_limited(response, _MAX_JSON_BYTES)
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise OpenAIGenerationError("OpenAI JSON response is not valid UTF-8") from error
        payload = _decode_json(text)
        await self._accept_packet(payload, streaming=False)
        await self._publish_terminal()

    async def _accept_packet(self, packet: object, *, streaming: bool) -> None:
        body = _object(packet, "OpenAI response")
        usage = body.get("usage")
        if usage is not None:
            if self._usage is not None:
                raise OpenAIGenerationError("OpenAI response contains duplicate usage")
            self._usage = _parse_usage(usage)
        choices = body.get("choices")
        if type(choices) is not list:
            raise OpenAIGenerationError("OpenAI response choices must be a list")
        choice_items = cast("list[object]", choices)
        if not choice_items:
            if usage is None or self._finish_reason is None or not streaming:
                raise OpenAIGenerationError("OpenAI response has no completion choice")
            return
        if len(choice_items) != 1:
            raise OpenAIGenerationError(
                "OpenAI response must contain exactly one completion choice"
            )
        if self._finish_reason is not None:
            raise OpenAIGenerationError(
                "OpenAI response contains an event after its terminal choice"
            )
        choice = _object(choice_items[0], "OpenAI completion choice")
        index = choice.get("index")
        if type(index) is not int or index != 0:
            raise OpenAIGenerationError("OpenAI completion choice index must be exact integer zero")
        text = _choice_text(choice, chat=self._chat, streaming=streaming)
        if text:
            text_chars = self._text_chars + len(text)
            if text_chars > _MAX_COMPLETION_CHARS:
                raise OpenAIGenerationError("OpenAI completion exceeds the size limit")
            if self._first_token_at is None:
                self._first_token_at = time.perf_counter()
            self._text.write(text)
            self._text_chars = text_chars
            event = GenerationTokenEvent(self._sequence, text)
            self._sequence += 1
            if not await self._publish(event):
                return
        finish = choice.get("finish_reason")
        if finish is not None:
            self._finish_reason = _finish_reason(finish)
        elif not streaming:
            raise OpenAIGenerationError("OpenAI non-stream response has no finish reason")

    async def _publish_terminal(self) -> None:
        if self._finish_reason is None:
            raise OpenAIGenerationError("OpenAI response ended without a finish reason")
        await self._publish(
            GenerationTerminalEvent(
                GenerationResult(
                    self._completion_text(),
                    self._finish_reason,
                    self._stats(),
                )
            )
        )

    async def _publish(self, item: _QueueItem) -> bool:
        with self._lock:
            if self._abandoned:
                return False
            ack = asyncio.Event()
            self._ack = ack
            self._queue.put_nowait(item)
        await ack.wait()
        with self._lock:
            if self._ack is ack:
                self._ack = None
            return not self._abandoned

    def _acknowledge(self) -> None:
        with self._lock:
            ack = self._ack
        if ack is not None:
            self._provider._runtime.call_soon(ack.set)  # pyright: ignore[reportPrivateUsage]

    def _cancelled_terminal(self) -> GenerationTerminalEvent:
        with self._lock:
            self._abandoned = True
            future = self._future
            ack = self._ack
        if ack is not None:
            self._provider._runtime.call_soon(ack.set)  # pyright: ignore[reportPrivateUsage]
        if future is not None:
            future.cancel()
        self._wait_future()
        self._finished = True
        self._closed = True
        self._provider._release(self)  # pyright: ignore[reportPrivateUsage]
        return GenerationTerminalEvent(
            GenerationResult(
                self._completion_text(),
                GenerationFinishReason.CANCELLED,
                self._stats(),
            )
        )

    def _completion_text(self) -> str:
        return self._text.getvalue()

    def _stats(self) -> GenerationStats:
        now = time.perf_counter()
        started = now if self._started is None else self._started
        usage = self._usage
        first = None if self._first_token_at is None else self._first_token_at - started
        return GenerationStats(
            now - started,
            prompt_tokens=None if usage is None else usage.prompt_tokens,
            generated_tokens=None if usage is None else usage.completion_tokens,
            time_to_first_token_s=first,
        )

    def _wait_future(self) -> None:
        future = self._future
        if future is None:
            return
        try:
            future.result()
        except (concurrent.futures.CancelledError, OpenAIGenerationError):
            pass


class _SSEParser:
    def __init__(self) -> None:
        self._buffer = ""
        self._data: list[str] = []
        self._event_chars = 0

    def feed(self, text: str, *, final: bool = False) -> tuple[str, ...]:
        self._buffer += text
        events: list[str] = []
        while (line := self._pop_line(final=final)) is not None:
            if line == "":
                if self._data:
                    events.append("\n".join(self._data))
                    self._data.clear()
                    self._event_chars = 0
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if field != "data":
                raise OpenAIGenerationError(
                    f"OpenAI SSE response contains unsupported field {field!r}"
                )
            if not separator:
                value = ""
            elif value.startswith(" "):
                value = value[1:]
            self._data.append(value)
            self._event_chars += len(line) + 1
            if self._event_chars > _MAX_SSE_EVENT_CHARS:
                raise OpenAIGenerationError("OpenAI SSE event exceeds the size limit")
        if final:
            if self._buffer:
                raise OpenAIGenerationError("OpenAI SSE response has an incomplete final line")
            if self._data:
                events.append("\n".join(self._data))
                self._data.clear()
                self._event_chars = 0
        elif len(self._buffer) > _MAX_SSE_EVENT_CHARS:
            raise OpenAIGenerationError("OpenAI SSE line exceeds the size limit")
        return tuple(events)

    def _pop_line(self, *, final: bool) -> str | None:
        for index, char in enumerate(self._buffer):
            if char not in "\r\n":
                continue
            if char == "\r" and index + 1 == len(self._buffer) and not final:
                return None
            width = 2 if char == "\r" and self._buffer[index + 1 : index + 2] == "\n" else 1
            line = self._buffer[:index]
            self._buffer = self._buffer[index + width :]
            return line
        if final and self._buffer:
            line = self._buffer
            self._buffer = ""
            return line
        return None


def _validate_base_url(base_url: str) -> str:
    if type(cast("object", base_url)) is not str:
        raise TypeError("OpenAI base URL must be a string")
    if len(base_url) > 4096 or any(ord(char) < 32 for char in base_url):
        raise ValueError("OpenAI base URL must be a bounded printable URL")
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("OpenAI base URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OpenAI base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("OpenAI base URL must not contain a query or fragment")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("OpenAI base URL contains an invalid port") from error
    if any(char.isspace() for char in parsed.hostname):
        raise ValueError("OpenAI base URL host must not contain whitespace")
    return base_url.rstrip("/")


async def _read_limited(response: aiohttp.ClientResponse, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        total += len(chunk)
        if total > limit:
            raise OpenAIGenerationError("OpenAI response exceeds the size limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def _http_error(response: aiohttp.ClientResponse) -> OpenAIGenerationError:
    try:
        body = await _read_limited(response, _MAX_ERROR_BYTES)
    except OpenAIGenerationError:
        return OpenAIGenerationError(f"OpenAI endpoint returned HTTP {response.status}")
    message: str | None = None
    try:
        payload = _decode_json(body.decode("utf-8"))
    except (UnicodeDecodeError, OpenAIGenerationError):
        payload = None
    if isinstance(payload, dict):
        error = cast("dict[object, object]", payload).get("error")
        if isinstance(error, dict):
            candidate = cast("dict[object, object]", error).get("message")
            if type(candidate) is str:
                message = candidate[:4096]
    suffix = "" if message is None else f": {message}"
    return OpenAIGenerationError(f"OpenAI endpoint returned HTTP {response.status}{suffix}")


def _require_content_type(response: aiohttp.ClientResponse, expected: str) -> None:
    actual = response.headers.get("Content-Type", "").partition(";")[0].strip().lower()
    if actual != expected:
        raise OpenAIGenerationError(
            f"OpenAI response Content-Type must be {expected}, got {actual or 'missing'}"
        )


def _require_json_content_type(response: aiohttp.ClientResponse) -> None:
    actual = response.headers.get("Content-Type", "").partition(";")[0].strip().lower()
    if actual != "application/json" and not actual.endswith("+json"):
        raise OpenAIGenerationError(
            f"OpenAI response Content-Type must be JSON, got {actual or 'missing'}"
        )


def _decode_json(text: str) -> object:
    def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OpenAIGenerationError(f"OpenAI JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise OpenAIGenerationError(f"OpenAI JSON contains invalid constant {value!r}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except OpenAIGenerationError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise OpenAIGenerationError("OpenAI response is not valid JSON") from error


def _object(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise OpenAIGenerationError(f"{name} must be an object")
    return cast("dict[str, object]", value)


def _choice_text(choice: Mapping[str, object], *, chat: bool, streaming: bool) -> str:
    if not chat:
        text = choice.get("text")
        if type(text) is not str:
            raise OpenAIGenerationError("OpenAI completion choice text must be a string")
        return text
    field = "delta" if streaming else "message"
    message = _object(choice.get(field), f"OpenAI chat choice {field}")
    role = message.get("role")
    if role is not None and role != "assistant":
        raise OpenAIGenerationError("OpenAI chat response role must be assistant")
    for unsupported in (
        "tool_calls",
        "function_call",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
    ):
        value = message.get(unsupported)
        if value not in (None, (), [], ""):
            raise OpenAIGenerationError(
                f"OpenAI chat response contains unsupported {unsupported} content"
            )
    content = message.get("content", "")
    if content is None:
        return ""
    if type(content) is not str:
        raise OpenAIGenerationError("OpenAI chat response content must be a string or null")
    return content


def _finish_reason(value: object) -> GenerationFinishReason:
    if value == "stop":
        return GenerationFinishReason.STOP
    if value == "length":
        return GenerationFinishReason.LENGTH
    raise OpenAIGenerationError(f"OpenAI response has unsupported finish reason {value!r}")


def _parse_usage(value: object) -> _Usage:
    usage = _object(value, "OpenAI usage")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(prompt) is not int or prompt < 0:
        raise OpenAIGenerationError("OpenAI prompt token usage must be a non-negative integer")
    if type(completion) is not int or completion < 0:
        raise OpenAIGenerationError("OpenAI completion token usage must be a non-negative integer")
    total = usage.get("total_tokens")
    if total is not None and (type(total) is not int or total != prompt + completion):
        raise OpenAIGenerationError(
            "OpenAI total token usage does not match prompt plus completion"
        )
    return _Usage(prompt, completion)


__all__ = [
    "OpenAICompatibility",
    "OpenAIGenerationError",
    "OpenAIGenerationProvider",
]
