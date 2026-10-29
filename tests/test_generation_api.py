"""Stateful Dinkster and OpenAI-compatible generation HTTP adapters."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from types import TracebackType
from typing import Self, cast

import pytest
from aiohttp import ClientResponse, web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_inference import (
    GenerationEvent,
    GenerationFinishReason,
    GenerationMessageRole,
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
    GenerationTerminalEvent,
    GenerationTokenEvent,
)
from dinkster_server import create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from dinkster.generation_api import (
    GenerationModel,
    GenerationService,
    _stream_completion,
    _stream_native,
    add_generation_routes,
)

PROVIDER_ID = "dinkster.test-generation"
MODEL_IDENTITY = "test:qwen:" + "1" * 64
MODEL = "test-qwen"


class _Stream(Iterator[GenerationEvent]):
    def __init__(self, events: tuple[GenerationEvent, ...], provider: _Provider) -> None:
        self._events = iter(events)
        self._provider = provider
        self.closed = False

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> GenerationEvent:
        return next(self._events)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._provider.closed_streams += 1

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class _Provider:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.sessions: set[str] = set()
        self.closed_sessions: list[str] = []
        self.closed_streams = 0
        self.closed = False
        self.model_identity = MODEL_IDENTITY

    @property
    def id(self) -> str:
        return PROVIDER_ID

    @property
    def capabilities(self) -> GenerationProviderCapabilities:
        return GenerationProviderCapabilities(
            frozenset(GenerationSamplerKind),
            chat=True,
            sessions=True,
            token_ids=True,
            ordered_sampler_chain=True,
        )

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _Stream:
        assert request.provider_id == self.id
        assert request.model_identity == self.model_identity
        assert not cancelled()
        self.requests.append(request)
        continuation = request.session
        if request.open_session:
            session_id = f"{len(self.sessions) + 1:032x}"
            self.sessions.add(session_id)
            continuation = GenerationSessionHandle(self.id, self.model_identity, session_id)
        text = "Hello world"
        return _Stream(
            (
                GenerationTokenEvent(0, "Hello", 10),
                GenerationTokenEvent(1, " world", 11),
                GenerationTerminalEvent(
                    GenerationResult(
                        text,
                        GenerationFinishReason.LENGTH,
                        GenerationStats(
                            total_time_s=0.1,
                            prompt_tokens=3,
                            generated_tokens=2,
                            time_to_first_token_s=0.02,
                            prefill_time_s=0.01,
                            decode_time_s=0.08,
                        ),
                        token_ids=(10, 11),
                        continuation=continuation,
                    )
                ),
            ),
            self,
        )

    def close_session(self, session: GenerationSessionHandle) -> None:
        self.sessions.discard(session.session_id)
        self.closed_sessions.append(session.session_id)

    def close(self) -> None:
        self.closed = True


class _FailingStream(_Stream):
    def __next__(self) -> GenerationEvent:
        raise RuntimeError("provider stream failed")


class _FailingProvider(_Provider):
    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _FailingStream:
        assert not cancelled()
        self.requests.append(request)
        return _FailingStream((), self)


class _MalformedProvider(_Provider):
    mode = "normal"

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _Stream:
        base = super().generate(request, cancelled=cancelled)
        events = tuple(base)
        base.close()
        terminal = cast("GenerationTerminalEvent", events[-1])
        if self.mode == "duplicate":
            events = (*events, terminal)
        elif self.mode == "drop":
            events = (
                *events[:-1],
                GenerationTerminalEvent(replace(terminal.result, continuation=None)),
            )
        elif self.mode in {"replace", "spontaneous"}:
            events = (
                *events[:-1],
                GenerationTerminalEvent(
                    replace(
                        terminal.result,
                        continuation=GenerationSessionHandle(
                            self.id,
                            self.model_identity,
                            "f" * 32,
                        ),
                    )
                ),
            )
        return _Stream(events, self)


class _CleanupProvider(_Provider):
    def __init__(self) -> None:
        super().__init__()
        self.close_session_calls: list[str] = []
        self.close_calls = 0

    def close_session(self, session: GenerationSessionHandle) -> None:
        self.close_session_calls.append(session.session_id)
        if len(self.close_session_calls) == 1:
            raise RuntimeError("session cleanup failed")
        super().close_session(session)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("provider cleanup failed")
        super().close()


class _WrongIdCleanupProvider(_CleanupProvider):
    @property
    def id(self) -> str:
        return "dinkster.wrong-provider"


class _SessionOnlyCleanupProvider(_Provider):
    def __init__(self) -> None:
        super().__init__()
        self.close_session_calls = 0

    def __getattribute__(self, name: str):
        if name == "close":
            return None
        return super().__getattribute__(name)

    def close_session(self, session: GenerationSessionHandle) -> None:
        self.close_session_calls += 1
        if self.close_session_calls == 1:
            raise RuntimeError("session cleanup failed")
        super().close_session(session)


class _CloseFailingStream(_Stream):
    def __init__(self, events: tuple[GenerationEvent, ...], provider: _Provider) -> None:
        super().__init__(events, provider)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("stream cleanup failed")
        super().close()


class _CloseFailingProvider(_Provider):
    def __init__(self) -> None:
        super().__init__()
        self.stream: _CloseFailingStream | None = None

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _CloseFailingStream:
        assert not cancelled()
        self.requests.append(request)
        self.stream = _CloseFailingStream((), self)
        return self.stream


class _LookaheadBarrierStream(_Stream):
    def __init__(
        self,
        events: tuple[GenerationEvent, ...],
        provider: _Provider,
        started: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(events, provider)
        self._started = started
        self._release = release

    def __next__(self) -> GenerationEvent:
        try:
            return super().__next__()
        except StopIteration:
            self._started.set()
            if not self._release.wait(2.0):
                raise RuntimeError("lookahead release timed out") from None
            raise


class _LookaheadBarrierProvider(_Provider):
    def __init__(self) -> None:
        super().__init__()
        self.block_lookahead = False
        self.lookahead_started = threading.Event()
        self.release_lookahead = threading.Event()

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _Stream:
        stream = super().generate(request, cancelled=cancelled)
        if not self.block_lookahead:
            return stream
        events = tuple(stream)
        stream.close()
        return _LookaheadBarrierStream(
            events,
            self,
            self.lookahead_started,
            self.release_lookahead,
        )


def _make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas={},
        registry=registry,
        worker=InProcessWorker({}, registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


async def _client(
    *,
    provider: _Provider | None = None,
) -> tuple[TestClient, GenerationService, _Provider, list[_Provider]]:
    first = provider or _Provider()
    made: list[_Provider] = []

    def factory() -> _Provider:
        selected = first if not made else _Provider()
        made.append(selected)
        return selected

    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, factory),))
    app = create_app(_make_engine, {})
    add_generation_routes(app, service)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, service, first, made


def _sse_data(body: str) -> list[object]:
    result: list[object] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        value = line.removeprefix("data: ")
        result.append(value if value == "[DONE]" else json.loads(value))
    return result


def _sse_frames(body: str) -> list[tuple[str, dict[str, object]]]:
    frames: list[tuple[str, dict[str, object]]] = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        frames.append(
            (
                lines[0].removeprefix("event: "),
                json.loads(lines[1].removeprefix("data: ")),
            )
        )
    return frames


def _direct_stream(
    service: GenerationService,
    *,
    session_id: str | None = None,
    open_session: bool = False,
    store_session: bool = True,
):
    cancelled = threading.Event()
    return service.open_stream(
        MODEL,
        owner="owner",
        prompt="Hi",
        messages=(),
        sampler=GenerationSamplerChain((GenerationSamplerStage(GenerationSamplerKind.GREEDY),)),
        stop=GenerationStopConditions(2),
        seed=None,
        session_id=session_id,
        open_session=open_session,
        store_session=store_session,
        cancelled=cancelled.is_set,
        cancel=cancelled.set,
    )


def test_native_generation_sessions_are_principal_owned_and_lifecycle_managed() -> None:
    async def scenario() -> None:
        client, service, provider, made = await _client()
        try:
            rows = await (await client.get("/api/generation/models")).json()
            assert len(rows["data"]) == 1
            assert rows["data"][0] == {
                "id": MODEL,
                "object": "model",
                "created": rows["data"][0]["created"],
                "owned_by": "dinkster",
                "loaded": False,
            }
            assert type(rows["data"][0]["created"]) is int
            response = await client.post(
                "/api/generation",
                json={
                    "model": MODEL,
                    "prompt": "Hi",
                    "maxNewTokens": 2,
                    "openSession": True,
                },
            )
            assert response.status == 200
            result = await response.json()
            assert result["text"] == "Hello world"
            assert result["tokenIds"] == [10, 11]
            session_id = result["sessionId"]
            assert isinstance(session_id, str) and len(session_id) == 32
            assert session_id != "00000000000000000000000000000001"
            assert len(made) == 1
            assert provider.requests[0].prompt == "Hi"
            assert provider.requests[0].open_session is True

            continued = await client.post(
                "/api/generation",
                json={"model": MODEL, "prompt": "Again", "sessionId": session_id},
            )
            assert continued.status == 200
            assert provider.requests[-1].session is not None
            assert provider.requests[-1].session.session_id == "00000000000000000000000000000001"
            assert (await continued.json())["sessionId"] == session_id

            closed = await client.delete(f"/api/generation/sessions/{session_id}")
            assert closed.status == 204
            assert provider.closed_sessions == ["00000000000000000000000000000001"]
            missing = await client.post(
                "/api/generation",
                json={"model": MODEL, "prompt": "Again", "sessionId": session_id},
            )
            assert missing.status == 404

            unloaded = await client.post(
                "/api/generation/models/unload",
                json={"model": MODEL},
            )
            assert unloaded.status == 200
            assert provider.closed is True
            await asyncio.to_thread(service.load_model, MODEL)
            assert len(made) == 2
        finally:
            await client.close()
        assert made[-1].closed is True

    asyncio.run(scenario())


def test_public_session_ids_do_not_collide_across_provider_instances() -> None:
    providers = (_Provider(), _Provider())
    service = GenerationService(
        (
            GenerationModel("first", PROVIDER_ID, lambda: providers[0]),
            GenerationModel("second", PROVIDER_ID, lambda: providers[1]),
        )
    )
    public_ids: list[str] = []
    for model in ("first", "second"):
        cancelled = threading.Event()
        stream = service.open_stream(
            model,
            owner="owner",
            prompt="Hi",
            messages=(),
            sampler=GenerationSamplerChain((GenerationSamplerStage(GenerationSamplerKind.GREEDY),)),
            stop=GenerationStopConditions(2),
            seed=None,
            session_id=None,
            open_session=True,
            cancelled=cancelled.is_set,
            cancel=cancelled.set,
        )
        events = tuple(stream)
        stream.close()
        terminal = events[-1]
        assert isinstance(terminal, GenerationTerminalEvent)
        assert terminal.result.continuation is not None
        public_ids.append(terminal.result.continuation.session_id)
    assert len(set(public_ids)) == 2
    assert "00000000000000000000000000000001" not in public_ids
    for public_id in public_ids:
        service.close_session(public_id, owner="owner")
    assert providers[0].closed_sessions == ["00000000000000000000000000000001"]
    assert providers[1].closed_sessions == ["00000000000000000000000000000001"]
    service.close()


def test_native_generation_stream_uses_protocol_events() -> None:
    async def scenario() -> None:
        client, _, provider, _ = await _client()
        try:
            response = await client.post(
                "/api/generation",
                json={"model": MODEL, "prompt": "Hi", "stream": True},
            )
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/event-stream")
            body = await response.text()
            assert "event: token" in body
            assert "event: result" in body
            data = _sse_data(body)
            assert data[0] == {"sequence": 0, "text": "Hello", "tokenId": 10}
            assert data[-1]["finishReason"] == "length"  # type: ignore[index]
            async with asyncio.timeout(2.0):
                while provider.closed_streams != 1:
                    await asyncio.sleep(0.01)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_openai_completion_and_chat_adapters_map_neutral_requests() -> None:
    async def scenario() -> None:
        client, _, provider, _ = await _client()
        try:
            completion = await client.post(
                "/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": "Hi",
                    "max_tokens": 2,
                    "temperature": 0,
                },
            )
            assert completion.status == 200
            wire = await completion.json()
            assert wire["object"] == "text_completion"
            assert wire["choices"] == [
                {"index": 0, "finish_reason": "length", "text": "Hello world"}
            ]
            assert wire["usage"] == {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            }
            assert provider.requests[-1].sampler.stages[-1].kind is GenerationSamplerKind.GREEDY

            chat = await client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            assert chat.status == 200
            chunks = _sse_data(await chat.text())
            assert chunks[-1] == "[DONE]"
            assert chunks[0]["choices"][0]["delta"] == {"content": "Hello"}  # type: ignore[index]
            assert "usage" not in chunks[-2]  # type: ignore[operator]
            assert provider.requests[-1].messages[0].role is GenerationMessageRole.USER

            with_usage = await client.post(
                "/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": "Hi",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            assert with_usage.status == 200
            usage_chunks = _sse_data(await with_usage.text())
            assert usage_chunks[-1] == "[DONE]"
            for chunk in usage_chunks[:-2]:
                assert chunk["usage"] is None  # type: ignore[index]
            assert usage_chunks[-2] == {
                "id": usage_chunks[0]["id"],  # type: ignore[index]
                "object": "text_completion",
                "created": usage_chunks[0]["created"],  # type: ignore[index]
                "model": MODEL,
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_responses_adapter_stores_and_continues_sessions_by_default() -> None:
    async def scenario() -> None:
        client, _, provider, _ = await _client()
        try:
            first = await client.post(
                "/v1/responses",
                json={"model": MODEL, "input": "Hi"},
            )
            assert first.status == 200
            body = await first.json()
            assert body["object"] == "response"
            assert body["id"].startswith("resp_")
            assert body["id"] != "resp_00000000000000000000000000000001"
            assert body["output"][0]["content"][0]["text"] == "Hello world"
            assert body["status"] == "incomplete"
            assert body["incomplete_details"] == {"reason": "max_output_tokens"}

            second = await client.post(
                "/v1/responses",
                json={
                    "model": MODEL,
                    "input": "Again",
                    "previous_response_id": body["id"],
                },
            )
            assert second.status == 200
            assert provider.requests[-1].session is not None
            assert provider.requests[-1].session.session_id == ("00000000000000000000000000000001")
            second_body = await second.json()
            assert second_body["id"] != body["id"]
            assert second_body["previous_response_id"] == body["id"]

            third = await client.post(
                "/v1/responses",
                json={
                    "model": MODEL,
                    "input": "Again",
                    "previous_response_id": second_body["id"],
                },
            )
            assert third.status == 200
            assert (await third.json())["id"] not in {body["id"], second_body["id"]}
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_responses_store_false_does_not_preserve_initial_or_continued_session(
    stream: bool,
) -> None:
    async def response_id(response: ClientResponse) -> str:
        if not stream:
            return cast("str", (await response.json())["id"])
        frames = _sse_frames(await response.text())
        return cast("str", frames[0][1]["response"]["id"])  # type: ignore[index]

    async def scenario() -> None:
        client, _, provider, _ = await _client()
        try:
            stateless = await client.post(
                "/v1/responses",
                json={"model": MODEL, "input": "Hi", "store": False, "stream": stream},
            )
            assert stateless.status == 200
            stateless_id = await response_id(stateless)
            assert provider.requests[-1].open_session is False
            missing = await client.post(
                "/v1/responses",
                json={
                    "model": MODEL,
                    "input": "Again",
                    "previous_response_id": stateless_id,
                },
            )
            assert missing.status == 404

            stored = await client.post(
                "/v1/responses",
                json={"model": MODEL, "input": "Hi"},
            )
            stored_id = cast("str", (await stored.json())["id"])
            continued = await client.post(
                "/v1/responses",
                json={
                    "model": MODEL,
                    "input": "Again",
                    "previous_response_id": stored_id,
                    "store": False,
                    "stream": stream,
                },
            )
            assert continued.status == 200
            continued_id = await response_id(continued)
            assert provider.requests[-1].session is not None
            assert provider.closed_sessions[-1] == "00000000000000000000000000000001"
            missing = await client.post(
                "/v1/responses",
                json={
                    "model": MODEL,
                    "input": "Again",
                    "previous_response_id": continued_id,
                },
            )
            assert missing.status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_responses_stream_emits_complete_typed_event_lifecycle() -> None:
    async def scenario() -> None:
        client, _, _, _ = await _client()
        try:
            response = await client.post(
                "/v1/responses",
                json={"model": MODEL, "input": "Hi", "store": True, "stream": True},
            )
            assert response.status == 200
            body = await response.text()
            assert "data: [DONE]" not in body
            frames = _sse_frames(body)
            events = [event for event, _ in frames]
            assert events == [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.incomplete",
            ]
            assert [data["sequence_number"] for _, data in frames] == list(range(len(frames)))
            for event, data in frames:
                assert data["type"] == event
            response_id = frames[0][1]["response"]["id"]  # type: ignore[index]
            assert frames[-1][1]["response"]["id"] == response_id  # type: ignore[index]
            item_id = frames[2][1]["item"]["id"]  # type: ignore[index]
            assert frames[3][1]["item_id"] == item_id
            assert frames[4][1]["item_id"] == item_id
            assert frames[4][1]["logprobs"] == []
            assert frames[-1][1]["response"]["usage"] == {  # type: ignore[index]
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5,
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_generation_adapters_refuse_malformed_or_unsupported_requests() -> None:
    async def scenario() -> None:
        client, _, provider, _ = await _client()
        try:
            cases: tuple[tuple[str, Mapping[str, object]], ...] = (
                ("/api/generation", {"model": MODEL, "prompt": "x", "extra": True}),
                ("/api/generation", {"model": MODEL, "prompt": "x", "messages": []}),
                ("/v1/completions", {"model": MODEL, "prompt": "x", "n": 2}),
                ("/v1/completions", {"model": MODEL, "prompt": "x", "n": True}),
                ("/v1/completions", {"model": MODEL, "prompt": "x", "n": 1.0}),
                (
                    "/v1/completions",
                    {
                        "model": MODEL,
                        "prompt": "x",
                        "stream_options": {"include_usage": True},
                    },
                ),
                ("/v1/chat/completions", {"model": MODEL, "messages": []}),
                ("/v1/responses", {"model": MODEL, "input": 3}),
            )
            for path, payload in cases:
                response = await client.post(path, json=payload)
                assert response.status == 400, (path, await response.text())
            unknown = await client.post(
                "/v1/completions",
                json={"model": "missing", "prompt": "x"},
            )
            assert unknown.status == 404
            assert provider.requests == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_openai_streams_emit_typed_failures_after_headers() -> None:
    async def scenario() -> None:
        client, _, provider, _ = await _client(provider=_FailingProvider())
        try:
            completion = await client.post(
                "/v1/completions",
                json={"model": MODEL, "prompt": "Hi", "stream": True},
            )
            assert completion.status == 200
            completion_data = _sse_data(await completion.text())
            assert completion_data[-1] == "[DONE]"
            assert completion_data[-2] == {
                "error": {
                    "message": "provider stream failed",
                    "type": "server_error",
                    "param": None,
                    "code": "generation_request_failed",
                }
            }

            responses = await client.post(
                "/v1/responses",
                json={"model": MODEL, "input": "Hi", "stream": True},
            )
            assert responses.status == 200
            frames = _sse_frames(await responses.text())
            assert frames[-1][0] == "response.failed"
            assert frames[-1][1]["sequence_number"] == 4
            failed = frames[-1][1]["response"]
            assert failed["status"] == "failed"  # type: ignore[index]
            assert failed["error"]["message"] == "provider stream failed"  # type: ignore[index]
            async with asyncio.timeout(2.0):
                while provider.closed_streams != 2:
                    await asyncio.sleep(0.01)
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("stream_kind", ["native", "completion"])
def test_stream_prepare_failure_closes_provider_stream(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
) -> None:
    async def fail_prepare(*_: object, **__: object) -> None:
        raise RuntimeError("prepare failed")

    async def scenario() -> None:
        provider = _Provider()
        service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
        stream = _direct_stream(service)
        cancelled = threading.Event()
        request = cast("web.Request", None)
        with pytest.raises(RuntimeError, match="prepare failed"):
            if stream_kind == "native":
                await _stream_native(request, stream, cancelled)
            else:
                await _stream_completion(
                    request,
                    stream,
                    cancelled,
                    request_id="cmpl-test",
                    created=1,
                    model=MODEL,
                    chat=False,
                    include_usage=False,
                )
        assert provider.closed_streams == 1
        service.close()

    monkeypatch.setattr("aiohttp.web.StreamResponse.prepare", fail_prepare)
    asyncio.run(scenario())


def test_native_terminal_write_failure_discards_unpublished_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def prepare(*_: object, **__: object) -> None:
        return None

    async def fail_result(
        _: web.StreamResponse,
        *,
        event: str | None,
        data: object,
    ) -> None:
        del data
        if event == "result":
            raise ConnectionError("result write failed")

    async def scenario() -> None:
        provider = _Provider()
        service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
        stream = _direct_stream(service, open_session=True)
        public_session_id = service.public_session_id(stream)
        with pytest.raises(ConnectionError, match="result write failed"):
            await _stream_native(cast("web.Request", None), stream, threading.Event())
        assert provider.closed_sessions == ["00000000000000000000000000000001"]
        with pytest.raises(GenerationSessionClosedError):
            service.close_session(public_session_id, owner="owner")
        service.close()

    monkeypatch.setattr("aiohttp.web.StreamResponse.prepare", prepare)
    monkeypatch.setattr("dinkster.generation_api._write_sse", fail_result)
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["drop", "replace"])
def test_malformed_existing_session_terminal_preserves_mapping(mode: str) -> None:
    provider = _MalformedProvider()
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
    first = _direct_stream(service, open_session=True)
    first_events = tuple(first)
    first.close()
    first_terminal = cast("GenerationTerminalEvent", first_events[-1])
    continuation = first_terminal.result.continuation
    assert continuation is not None

    provider.mode = mode
    malformed = _direct_stream(service, session_id=continuation.session_id)
    with pytest.raises(RuntimeError, match="replaced or dropped"):
        tuple(malformed)
    malformed.close()

    provider.mode = "normal"
    continued = _direct_stream(service, session_id=continuation.session_id)
    assert isinstance(tuple(continued)[-1], GenerationTerminalEvent)
    continued.close()
    service.close_session(continuation.session_id, owner="owner")
    service.close()


def test_active_session_cannot_be_deleted_during_terminal_lookahead() -> None:
    provider = _LookaheadBarrierProvider()
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
    first = _direct_stream(service, open_session=True)
    first_terminal = cast("GenerationTerminalEvent", tuple(first)[-1])
    first.close()
    continuation = first_terminal.result.continuation
    assert continuation is not None

    provider.block_lookahead = True
    continued = _direct_stream(service, session_id=continuation.session_id)
    events: list[GenerationEvent] = []
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            events.extend(continued)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=consume)
    thread.start()
    assert provider.lookahead_started.wait(2.0)
    with pytest.raises(GenerationSessionBusyError, match="active"):
        service.close_session(continuation.session_id, owner="owner")
    with pytest.raises(GenerationSessionBusyError, match="active"):
        _direct_stream(service, session_id=continuation.session_id)
    provider.release_lookahead.set()
    thread.join(2.0)
    assert not thread.is_alive()
    assert errors == []
    assert isinstance(events[-1], GenerationTerminalEvent)
    continued.close()

    service.close_session(continuation.session_id, owner="owner")
    service.close()


def test_store_false_cleanup_failure_revokes_all_public_session_ids() -> None:
    provider = _SessionOnlyCleanupProvider()
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
    first = _direct_stream(service, open_session=True)
    first_terminal = cast("GenerationTerminalEvent", tuple(first)[-1])
    first.close()
    continuation = first_terminal.result.continuation
    assert continuation is not None

    continued = _direct_stream(
        service,
        session_id=continuation.session_id,
        store_session=False,
    )
    replacement_id = service.public_session_id(continued, replace=True)
    with pytest.raises(RuntimeError, match="session cleanup failed"):
        tuple(continued)
    continued.close()

    for session_id in (continuation.session_id, replacement_id):
        with pytest.raises(GenerationSessionClosedError):
            _direct_stream(service, session_id=session_id)
    assert provider.close_session_calls == 1
    service.close()
    assert provider.close_session_calls == 2


def test_duplicate_terminal_does_not_publish_provisional_session() -> None:
    provider = _MalformedProvider()
    provider.mode = "duplicate"
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
    stream = _direct_stream(service, open_session=True)
    reserved = service.public_session_id(stream)
    with pytest.raises(RuntimeError, match="after its terminal"):
        tuple(stream)
    stream.close()
    with pytest.raises(GenerationSessionClosedError):
        service.close_session(reserved, owner="owner")
    service.close()


def test_stateless_terminal_cannot_publish_spontaneous_session() -> None:
    provider = _MalformedProvider()
    provider.mode = "spontaneous"
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
    stream = _direct_stream(service)
    with pytest.raises(RuntimeError, match="stateless generation"):
        tuple(stream)
    stream.close()
    service.close()


def test_service_close_continues_after_failures_and_retries_remaining_cleanup() -> None:
    provider = _CleanupProvider()
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
    for _ in range(2):
        stream = _direct_stream(service, open_session=True)
        assert isinstance(tuple(stream)[-1], GenerationTerminalEvent)
        stream.close()

    with pytest.raises(ExceptionGroup) as caught:
        service.close()
    assert len(caught.value.exceptions) == 2
    assert provider.close_session_calls == [
        "00000000000000000000000000000001",
        "00000000000000000000000000000002",
    ]
    assert provider.close_calls == 1

    service.close()
    assert provider.close_session_calls[-1] == "00000000000000000000000000000001"
    assert provider.close_calls == 2
    assert provider.closed is True


def test_service_close_retries_invalid_provider_cleanup() -> None:
    provider = _WrongIdCleanupProvider()
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))

    with pytest.raises(RuntimeError, match="provider cleanup failed"):
        service.load_model(MODEL)
    assert provider.close_calls == 1
    assert service.model_rows()[0]["loaded"] is False
    with pytest.raises(RuntimeError, match="cleanup is pending"):
        service.load_model(MODEL)

    service.close()
    assert provider.close_calls == 2
    assert provider.closed is True


def test_service_close_retries_failed_active_stream_cleanup() -> None:
    provider = _CloseFailingProvider()
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
    _direct_stream(service)

    with pytest.raises(ExceptionGroup, match="cleanup failed"):
        service.close()
    assert provider.stream is not None
    assert provider.stream.close_calls == 1
    assert provider.closed is False

    service.close()
    assert provider.stream.close_calls == 2
    assert provider.closed is True


def test_failed_session_cleanup_remains_retryable_without_provider_closer() -> None:
    provider = _SessionOnlyCleanupProvider()
    service = GenerationService((GenerationModel(MODEL, PROVIDER_ID, lambda: provider),))
    stream = _direct_stream(service, open_session=True)
    assert isinstance(tuple(stream)[-1], GenerationTerminalEvent)
    stream.close()

    with pytest.raises(ExceptionGroup) as caught:
        service.close()
    assert str(caught.value.exceptions[0]) == "session cleanup failed"
    assert provider.close_session_calls == 1

    service.close()
    assert provider.close_session_calls == 2


def test_models_endpoint_runs_service_lookup_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        client, service, _, _ = await _client()
        event_loop_thread = threading.get_ident()
        original = service.model_rows
        acquired = threading.Event()
        release = threading.Event()
        call_threads: list[int] = []

        def blocking_model_rows() -> tuple[dict[str, object], ...]:
            call_threads.append(threading.get_ident())
            acquired.set()
            release.wait(2.0)
            return original()

        monkeypatch.setattr(service, "model_rows", blocking_model_rows)
        watchdog = threading.Timer(1.0, release.set)
        watchdog.start()
        try:
            request = asyncio.create_task(client.get("/v1/models"))
            assert await asyncio.to_thread(acquired.wait, 1.5)
            assert len(call_threads) == 1
            assert call_threads[0] != event_loop_thread
            assert not request.done()
            release.set()
            response = await request
            assert response.status == 200
        finally:
            release.set()
            watchdog.cancel()
            await client.close()

    asyncio.run(scenario())


class _BlockingStream(_Stream):
    def __init__(
        self,
        provider: _Provider,
        cancelled: Callable[[], bool],
        cancellation_seen: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
        pulling: threading.Event,
    ) -> None:
        super().__init__((), provider)
        self._cancelled = cancelled
        self._cancellation_seen = cancellation_seen
        self._loop = loop
        self._pulling = pulling

    def __next__(self) -> GenerationEvent:
        self._pulling.set()
        deadline = time.monotonic() + 5.0
        try:
            while time.monotonic() < deadline:
                if self._cancelled():
                    self._loop.call_soon_threadsafe(self._cancellation_seen.set)
                    raise StopIteration
                time.sleep(0.005)
            raise AssertionError("generation disconnect was not cancelled")
        finally:
            self._pulling.clear()

    def close(self) -> None:
        if not self.closed:
            assert self._cancelled()
            assert not self._pulling.is_set()
            self._loop.call_soon_threadsafe(self._cancellation_seen.set)
        super().close()


class _BlockingProvider(_Provider):
    def __init__(
        self,
        cancellation_seen: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self.cancellation_seen = cancellation_seen
        self.loop = loop
        self.pulling = threading.Event()

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _BlockingStream:
        self.requests.append(request)
        return _BlockingStream(self, cancelled, self.cancellation_seen, self.loop, self.pulling)


def test_stream_disconnect_cancels_and_closes_provider_stream() -> None:
    async def scenario() -> None:
        cancellation_seen = asyncio.Event()
        provider = _BlockingProvider(cancellation_seen, asyncio.get_running_loop())
        client, _, _, _ = await _client(provider=provider)
        closed = False
        try:
            response = await client.post(
                "/v1/completions",
                json={"model": MODEL, "prompt": "Hi", "stream": True},
            )
            assert response.status == 200
            assert await asyncio.to_thread(provider.pulling.wait, 2.0)
            await client.close()
            closed = True
            await asyncio.wait_for(cancellation_seen.wait(), 2.0)
            async with asyncio.timeout(2.0):
                while provider.closed_streams != 1:
                    await asyncio.sleep(0.01)
        finally:
            if not closed:
                await client.close()

    asyncio.run(scenario())
