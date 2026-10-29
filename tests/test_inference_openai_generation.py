"""OpenAI-compatible generation transport and conformance proofs."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import dinkster_inference.openai_generation as openai_generation
import pytest
from dinkster_inference import (
    GenerationFinishReason,
    GenerationMessage,
    GenerationMessageRole,
    GenerationRequest,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSamplerStage,
    GenerationSessionClosedError,
    GenerationSessionHandle,
    GenerationStopConditions,
    GenerationTerminalEvent,
    GenerationTokenEvent,
    OpenAICompatibility,
    OpenAIGenerationError,
    OpenAIGenerationProvider,
)

_API_KEY = "test-secret-api-key"


@dataclass(frozen=True, slots=True)
class _Reply:
    chunks: tuple[bytes, ...]
    content_type: str = "application/json"
    status: int = 200
    pre_delay_s: float = 0.0
    chunk_delay_s: float = 0.0
    hold_open: bool = False


@dataclass(frozen=True, slots=True)
class _RecordedRequest:
    path: str
    headers: Mapping[str, str]
    payload: object


class _OpenAITestServer:
    def __init__(self) -> None:
        self.reply = _json_reply(_completion("ok"))
        self.requests: list[_RecordedRequest] = []
        self.disconnected = threading.Event()
        self.release = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                owner._handle(self)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.01),
            daemon=True,
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length)
        try:
            payload: object = json.loads(body)
        except json.JSONDecodeError:
            payload = body
        self.requests.append(
            _RecordedRequest(
                handler.path,
                {key: value for key, value in handler.headers.items()},
                payload,
            )
        )
        reply = self.reply
        if reply.pre_delay_s:
            time.sleep(reply.pre_delay_s)
        try:
            handler.send_response(reply.status)
            handler.send_header("Content-Type", reply.content_type)
            handler.send_header("Connection", "close")
            if not reply.hold_open:
                handler.send_header("Content-Length", str(sum(map(len, reply.chunks))))
            handler.end_headers()
            for chunk in reply.chunks:
                handler.wfile.write(chunk)
                handler.wfile.flush()
                if reply.chunk_delay_s:
                    time.sleep(reply.chunk_delay_s)
            while reply.hold_open and not self.release.wait(0.01):
                handler.wfile.write(b": keepalive\n\n")
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.disconnected.set()
        finally:
            handler.close_connection = True


@pytest.fixture
def openai_server() -> Iterator[_OpenAITestServer]:
    server = _OpenAITestServer()
    try:
        yield server
    finally:
        server.close()


def _provider(
    server: _OpenAITestServer,
    *,
    compatibility: OpenAICompatibility = OpenAICompatibility.STANDARD,
    stream: bool = True,
    timeout_s: float = 2.0,
    api_key: str | None = _API_KEY,
) -> OpenAIGenerationProvider:
    return OpenAIGenerationProvider(
        server.base_url,
        "test-model",
        api_key=api_key,
        compatibility=compatibility,
        stream=stream,
        timeout_s=timeout_s,
    )


def _request(
    provider: OpenAIGenerationProvider,
    *,
    prompt: str | None = "hello",
    messages: tuple[GenerationMessage, ...] = (),
    sampler: GenerationSamplerChain | None = None,
    stop: GenerationStopConditions | None = None,
    seed: int | None = None,
    open_session: bool = False,
    session: GenerationSessionHandle | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        provider.id,
        provider.model_identity,
        prompt=prompt,
        messages=messages,
        sampler=(
            GenerationSamplerChain((GenerationSamplerStage(GenerationSamplerKind.GREEDY),))
            if sampler is None
            else sampler
        ),
        stop=GenerationStopConditions(8) if stop is None else stop,
        seed=seed,
        open_session=open_session,
        session=session,
    )


def _collect(
    provider: OpenAIGenerationProvider,
    request: GenerationRequest,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[GenerationTokenEvent | GenerationTerminalEvent, ...]:
    with provider.generate(request, cancelled=cancelled) as stream:
        return tuple(stream)


def _completion(
    text: str,
    *,
    finish_reason: str | None = "stop",
    usage: object | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}]
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _chat_chunk(
    content: str | None = None,
    *,
    finish_reason: str | None = None,
    role: str | None = None,
) -> dict[str, object]:
    delta: dict[str, object] = {}
    if content is not None:
        delta["content"] = content
    if role is not None:
        delta["role"] = role
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def _json_reply(payload: object, *, status: int = 200) -> _Reply:
    return _Reply((json.dumps(payload, separators=(",", ":")).encode(),), status=status)


def _sse_reply(
    *payloads: object,
    done: bool = True,
    split_bytes: bool = False,
    hold_open: bool = False,
) -> _Reply:
    body = b"".join(
        f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()
        for payload in payloads
    )
    if done:
        body += b"data: [DONE]\n\n"
    chunks = tuple(bytes((byte,)) for byte in body) if split_bytes else (body,)
    return _Reply(
        chunks,
        content_type="text/event-stream; charset=utf-8",
        chunk_delay_s=0.0001 if split_bytes else 0.0,
        hold_open=hold_open,
    )


def _multinomial(*stages: GenerationSamplerStage) -> GenerationSamplerChain:
    return GenerationSamplerChain(
        (*stages, GenerationSamplerStage(GenerationSamplerKind.MULTINOMIAL))
    )


def test_non_stream_completion_maps_request_and_normalizes_usage(
    openai_server: _OpenAITestServer,
) -> None:
    usage = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    openai_server.reply = _json_reply(_completion("answer", usage=usage))
    provider = _provider(openai_server, stream=False)
    request = _request(
        provider,
        sampler=_multinomial(GenerationSamplerStage(GenerationSamplerKind.TOP_P, 0.8)),
        stop=GenerationStopConditions(12, ("END",)),
        seed=17,
    )

    with provider:
        events = _collect(provider, request)

    assert events[0] == GenerationTokenEvent(0, "answer")
    terminal = cast("GenerationTerminalEvent", events[1])
    assert terminal.result.text == "answer"
    assert terminal.result.finish_reason is GenerationFinishReason.STOP
    assert terminal.result.stats.prompt_tokens == 2
    assert terminal.result.stats.generated_tokens == 1
    assert terminal.result.token_ids is None
    assert terminal.result.continuation is None

    recorded = openai_server.requests[0]
    assert recorded.path == "/v1/completions"
    assert recorded.headers["Authorization"] == f"Bearer {_API_KEY}"
    assert recorded.payload == {
        "model": "test-model",
        "max_tokens": 12,
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.8,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "prompt": "hello",
        "stop": ["END"],
        "seed": 17,
    }
    assert _API_KEY not in json.dumps(recorded.payload)
    assert _API_KEY not in repr(provider)
    assert _API_KEY not in provider.model_identity


def test_streaming_chat_handles_arbitrary_byte_and_utf8_boundaries(
    openai_server: _OpenAITestServer,
) -> None:
    usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    openai_server.reply = _sse_reply(
        _chat_chunk(role="assistant"),
        _chat_chunk("he\u0301"),
        _chat_chunk("llo", finish_reason="stop"),
        {"choices": [], "usage": usage},
        split_bytes=True,
    )
    provider = _provider(openai_server)
    messages = (
        GenerationMessage(GenerationMessageRole.SYSTEM, "Answer briefly."),
        GenerationMessage(GenerationMessageRole.USER, "Say hello."),
    )

    with provider:
        events = _collect(provider, _request(provider, prompt=None, messages=messages))

    assert events[:2] == (
        GenerationTokenEvent(0, "he\u0301"),
        GenerationTokenEvent(1, "llo"),
    )
    terminal = cast("GenerationTerminalEvent", events[2])
    assert terminal.result.text == "he\u0301llo"
    assert terminal.result.stats.prompt_tokens == 4
    assert terminal.result.stats.generated_tokens == 2
    recorded = openai_server.requests[0]
    assert recorded.path == "/v1/chat/completions"
    assert cast("dict[str, object]", recorded.payload)["messages"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Say hello."},
    ]
    assert cast("dict[str, object]", recorded.payload)["stream_options"] == {"include_usage": True}


def test_llama_cpp_preserves_supported_sampler_order_and_neutralizes_defaults(
    openai_server: _OpenAITestServer,
) -> None:
    openai_server.reply = _json_reply(_completion("sampled"))
    provider = _provider(
        openai_server,
        compatibility=OpenAICompatibility.LLAMA_CPP,
        stream=False,
    )
    sampler = _multinomial(
        GenerationSamplerStage(GenerationSamplerKind.TEMPERATURE, 0.6),
        GenerationSamplerStage(GenerationSamplerKind.TOP_K, 40),
        GenerationSamplerStage(GenerationSamplerKind.TOP_P, 0.9),
        GenerationSamplerStage(GenerationSamplerKind.MIN_P, 0.05),
        GenerationSamplerStage(GenerationSamplerKind.TYPICAL_P, 0.8),
    )

    with provider:
        _collect(provider, _request(provider, sampler=sampler, seed=9))
        _collect(provider, _request(provider))

    payload = cast("dict[str, object]", openai_server.requests[0].payload)
    assert payload["samplers"] == ["temperature", "top_k", "top_p", "min_p", "typ_p"]
    assert payload["temperature"] == 0.6
    assert payload["top_k"] == 40
    assert payload["top_p"] == 0.9
    assert payload["min_p"] == 0.05
    assert payload["typical_p"] == 0.8
    assert payload["repeat_penalty"] == 1.0
    assert payload["presence_penalty"] == 0.0
    assert payload["frequency_penalty"] == 0.0
    assert payload["seed"] == 9
    greedy = cast("dict[str, object]", openai_server.requests[1].payload)
    assert greedy["temperature"] == 0.0
    assert greedy["samplers"] == ["temperature"]
    assert "seed" not in greedy


def test_greedy_selector_cannot_be_overridden_by_preceding_transforms(
    openai_server: _OpenAITestServer,
) -> None:
    openai_server.reply = _json_reply(_completion("greedy"))
    provider = _provider(
        openai_server,
        compatibility=OpenAICompatibility.LLAMA_CPP,
        stream=False,
    )
    sampler = GenerationSamplerChain(
        (
            GenerationSamplerStage(GenerationSamplerKind.TEMPERATURE, 0.7),
            GenerationSamplerStage(GenerationSamplerKind.TOP_K, 40),
            GenerationSamplerStage(GenerationSamplerKind.GREEDY),
        )
    )

    with provider:
        _collect(provider, _request(provider, sampler=sampler))

    payload = cast("dict[str, object]", openai_server.requests[0].payload)
    assert payload["temperature"] == 0.0
    assert payload["samplers"] == ["temperature"]


@pytest.mark.parametrize(
    "kind,value",
    [
        (GenerationSamplerKind.REPETITION_PENALTY, 1.1),
        (GenerationSamplerKind.PRESENCE_PENALTY, 0.2),
        (GenerationSamplerKind.FREQUENCY_PENALTY, 0.2),
    ],
)
def test_llama_cpp_refuses_penalties_with_unbounded_contract_history(
    openai_server: _OpenAITestServer,
    kind: GenerationSamplerKind,
    value: float,
) -> None:
    provider = _provider(openai_server, compatibility=OpenAICompatibility.LLAMA_CPP)
    request = _request(
        provider,
        sampler=_multinomial(GenerationSamplerStage(kind, value)),
    )

    with pytest.raises(ValueError, match="complete-history"):
        provider.generate(request, cancelled=lambda: False)
    provider.close()
    assert not openai_server.requests


def test_standard_endpoint_refuses_unrepresentable_sampler_semantics(
    openai_server: _OpenAITestServer,
) -> None:
    provider = _provider(openai_server)
    top_k = _request(
        provider,
        sampler=_multinomial(GenerationSamplerStage(GenerationSamplerKind.TOP_K, 10)),
    )
    ordered = _request(
        provider,
        sampler=_multinomial(
            GenerationSamplerStage(GenerationSamplerKind.TEMPERATURE, 0.7),
            GenerationSamplerStage(GenerationSamplerKind.TOP_P, 0.9),
        ),
    )

    with pytest.raises(ValueError, match="does not support sampler"):
        provider.generate(top_k, cancelled=lambda: False)
    with pytest.raises(ValueError, match="ordered multi-stage"):
        provider.generate(ordered, cancelled=lambda: False)
    provider.close()
    assert not openai_server.requests


def test_prompt_cancellation_closes_live_http_stream(
    openai_server: _OpenAITestServer,
) -> None:
    openai_server.reply = _sse_reply(
        _completion("first", finish_reason=None),
        done=False,
        hold_open=True,
    )
    provider = _provider(openai_server)
    cancel = False

    with provider:
        with provider.generate(_request(provider), cancelled=lambda: cancel) as stream:
            assert next(stream) == GenerationTokenEvent(0, "first")
            cancel = True
            terminal = cast("GenerationTerminalEvent", next(stream))
            assert terminal.result.text == "first"
            assert terminal.result.finish_reason is GenerationFinishReason.CANCELLED
            with pytest.raises(StopIteration):
                next(stream)

    assert openai_server.disconnected.wait(1.0)


def test_cancellation_before_first_pull_performs_no_io(
    openai_server: _OpenAITestServer,
) -> None:
    provider = _provider(openai_server)

    with provider:
        events = _collect(provider, _request(provider), cancelled=lambda: True)

    assert len(events) == 1
    terminal = cast("GenerationTerminalEvent", events[0])
    assert terminal.result.finish_reason is GenerationFinishReason.CANCELLED
    assert terminal.result.text == ""
    assert not openai_server.requests
    assert cast("Any", provider)._runtime._thread is None


def test_timeout_is_bounded_and_provider_closes_without_leaking_thread(
    openai_server: _OpenAITestServer,
) -> None:
    openai_server.reply = _Reply(
        _json_reply(_completion("late")).chunks,
        pre_delay_s=0.3,
    )
    provider = _provider(openai_server, stream=False, timeout_s=0.05)

    with pytest.raises(OpenAIGenerationError, match="timed out after 0.05 seconds"):
        _collect(provider, _request(provider))
    provider.close()
    provider.close()

    thread = cast("Any", provider)._runtime._thread
    assert thread is not None
    assert not thread.is_alive()


def test_http_error_redacts_api_key_from_remote_diagnostic(
    openai_server: _OpenAITestServer,
) -> None:
    openai_server.reply = _json_reply(
        {"error": {"message": f"credential {_API_KEY} was refused"}},
        status=401,
    )
    provider = _provider(openai_server, stream=False)

    with pytest.raises(OpenAIGenerationError) as raised:
        _collect(provider, _request(provider))
    provider.close()

    assert "HTTP 401" in str(raised.value)
    assert _API_KEY not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)


@pytest.mark.parametrize(
    "reply,match",
    [
        (_Reply((b"not json",), content_type="application/json"), "not valid JSON"),
        (_Reply((b"\xff",), content_type="application/json"), "not valid UTF-8"),
        (
            _Reply(
                (b'{"choices":NaN}',),
                content_type="application/json",
            ),
            "invalid constant",
        ),
        (
            _Reply(
                (b'{"choices":[],"choices":[]} ',),
                content_type="application/json",
            ),
            "duplicate key",
        ),
        (_Reply((b"{}",), content_type="text/plain"), "Content-Type must be JSON"),
        (
            _json_reply(_completion("x", finish_reason="content_filter")),
            "unsupported finish reason",
        ),
        (
            _json_reply(
                _completion(
                    "x",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
                )
            ),
            "total token usage",
        ),
    ],
)
def test_non_stream_response_validation_is_strict(
    openai_server: _OpenAITestServer,
    reply: _Reply,
    match: str,
) -> None:
    openai_server.reply = reply
    provider = _provider(openai_server, stream=False)

    with pytest.raises(OpenAIGenerationError, match=match):
        _collect(provider, _request(provider))
    provider.close()


@pytest.mark.parametrize(
    "reply,match",
    [
        (_Reply((b"{}",), content_type="application/json"), "text/event-stream"),
        (_Reply((b"data: \xff\n\n",), content_type="text/event-stream"), "valid UTF-8"),
        (_sse_reply(_completion("x"), done=False), "without data: \\[DONE\\]"),
        (
            _sse_reply(
                _completion("x"),
                _completion("again"),
            ),
            "event after its terminal choice",
        ),
        (
            _sse_reply(_completion("x", finish_reason=None)),
            "without a finish reason",
        ),
        (
            _Reply((b"event: message\n\n",), content_type="text/event-stream"),
            "unsupported field",
        ),
        (
            _Reply((b"data:\n" * 180_000 + b"\n",), content_type="text/event-stream"),
            "event exceeds the size limit",
        ),
    ],
)
def test_sse_response_validation_is_strict(
    openai_server: _OpenAITestServer,
    reply: _Reply,
    match: str,
) -> None:
    openai_server.reply = reply
    provider = _provider(openai_server)

    with pytest.raises(OpenAIGenerationError, match=match):
        _collect(provider, _request(provider))
    provider.close()


def test_sse_completion_has_an_aggregate_size_bound(
    openai_server: _OpenAITestServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_generation, "_MAX_COMPLETION_CHARS", 5)
    openai_server.reply = _sse_reply(
        _completion("abc", finish_reason=None),
        _completion("def"),
    )
    provider = _provider(openai_server)

    with pytest.raises(OpenAIGenerationError, match="completion exceeds the size limit"):
        _collect(provider, _request(provider))
    provider.close()


@pytest.mark.parametrize(
    "delta,match",
    [
        ({"role": "user", "content": "wrong"}, "role must be assistant"),
        ({"reasoning_content": "private"}, "unsupported reasoning_content"),
        ({"tool_calls": [{"id": "call"}]}, "unsupported tool_calls"),
    ],
)
def test_chat_response_refuses_unsupported_semantics(
    openai_server: _OpenAITestServer,
    delta: dict[str, object],
    match: str,
) -> None:
    openai_server.reply = _sse_reply(
        {"choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}]}
    )
    provider = _provider(openai_server)
    messages = (GenerationMessage(GenerationMessageRole.USER, "hello"),)

    with pytest.raises(OpenAIGenerationError, match=match):
        _collect(provider, _request(provider, prompt=None, messages=messages))
    provider.close()


def test_provider_rejects_stateful_and_foreign_requests_before_io(
    openai_server: _OpenAITestServer,
) -> None:
    provider = _provider(openai_server)
    session = GenerationSessionHandle(provider.id, provider.model_identity, "1" * 32)
    cases = (
        _request(provider, open_session=True),
        _request(provider, session=session),
        _request(provider, stop=GenerationStopConditions(8, stop_token_ids=(2,))),
        GenerationRequest("other.provider", provider.model_identity, prompt="hello"),
        GenerationRequest(provider.id, "other-model", prompt="hello"),
    )

    for request in cases:
        with pytest.raises(ValueError):
            provider.generate(request, cancelled=lambda: False)
    with pytest.raises(GenerationSessionClosedError, match="stateless"):
        provider.close_session(session)
    with pytest.raises(TypeError, match="GenerationSessionHandle"):
        cast("Any", provider).close_session("not-a-session")
    provider.close()
    assert not openai_server.requests


def test_provider_close_abandons_partial_stream_and_is_idempotent(
    openai_server: _OpenAITestServer,
) -> None:
    openai_server.reply = _sse_reply(
        _completion("partial", finish_reason=None),
        done=False,
        hold_open=True,
    )
    provider = _provider(openai_server)
    stream = provider.generate(_request(provider), cancelled=lambda: False)
    assert next(stream) == GenerationTokenEvent(0, "partial")

    provider.close()
    provider.close()
    stream.close()
    assert openai_server.disconnected.wait(1.0)
    with pytest.raises(RuntimeError, match="closed"):
        provider.generate(_request(provider), cancelled=lambda: False)


def test_provider_close_during_stream_submission_does_not_deadlock(
    openai_server: _OpenAITestServer,
) -> None:
    provider = _provider(openai_server)
    stream = provider.generate(_request(provider), cancelled=lambda: False)
    runtime = cast("Any", provider)._runtime
    original_submit = runtime.submit
    submitted = threading.Event()
    release_submit = threading.Event()

    async def held_run() -> None:
        await asyncio.Event().wait()

    def delayed_submit(coroutine: object) -> object:
        future = original_submit(coroutine)
        submitted.set()
        release_submit.wait()
        return future

    cast("Any", stream)._run = held_run
    runtime.submit = delayed_submit
    pull_error: list[BaseException] = []

    def pull() -> None:
        try:
            next(stream)
        except BaseException as error:
            pull_error.append(error)

    pull_thread = threading.Thread(target=pull, daemon=True)
    pull_thread.start()
    assert submitted.wait(1.0)
    close_thread = threading.Thread(target=provider.close)
    close_thread.start()
    close_thread.join(1.0)
    release_submit.set()
    pull_thread.join(1.0)

    assert not close_thread.is_alive()
    assert not pull_thread.is_alive()
    assert len(pull_error) == 1
    assert isinstance(pull_error[0], RuntimeError)
    assert not openai_server.requests
    assert not runtime._futures


def test_provider_close_during_runtime_startup_does_not_deadlock(
    openai_server: _OpenAITestServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(openai_server)
    stream = provider.generate(_request(provider), cancelled=lambda: False)
    runtime = cast("Any", provider)._runtime
    original = runtime._thread_main
    entered = threading.Event()
    release = threading.Event()
    closing = threading.Event()

    def delayed_start() -> None:
        entered.set()
        release.wait()
        original()

    runtime._thread_main = delayed_start
    pull_error: list[BaseException] = []

    def pull() -> None:
        try:
            next(stream)
        except BaseException as error:
            pull_error.append(error)

    pull_thread = threading.Thread(target=pull, daemon=True)
    pull_thread.start()
    assert entered.wait(1.0)
    runtime_thread = runtime._thread
    original_join = runtime_thread.join

    def closing_join(timeout: float | None = None) -> None:
        closing.set()
        original_join(timeout)

    monkeypatch.setattr(runtime_thread, "join", closing_join)
    close_thread = threading.Thread(target=provider.close)
    close_thread.start()
    try:
        assert closing.wait(1.0)
    finally:
        release.set()
    close_thread.join(1.0)
    pull_thread.join(1.0)

    assert not close_thread.is_alive()
    assert not pull_thread.is_alive()
    assert len(pull_error) == 1
    assert isinstance(pull_error[0], RuntimeError)
    assert not openai_server.requests


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"base_url": "file:///tmp/api"}, ValueError),
        ({"base_url": "https://user:pass@example.test/v1"}, ValueError),
        ({"base_url": "https://example.test/v1?key=value"}, ValueError),
        ({"base_url": "https://example.test:invalid/v1"}, ValueError),
        ({"base_url": "https://exam ple.test/v1"}, ValueError),
        ({"model": 1}, TypeError),
        ({"api_key": "line\nbreak"}, ValueError),
        ({"api_key": "null\x00byte"}, ValueError),
        ({"api_key": "delete\x7fbyte"}, ValueError),
        ({"compatibility": "llama.cpp"}, TypeError),
        ({"stream": 1}, TypeError),
        ({"timeout_s": 1}, TypeError),
        ({"timeout_s": float("inf")}, ValueError),
    ],
)
def test_provider_configuration_is_strict(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    values: dict[str, object] = {
        "base_url": "https://example.test/v1",
        "model": "model",
    }
    values.update(kwargs)
    with pytest.raises(error):
        OpenAIGenerationProvider(**cast("Any", values))
