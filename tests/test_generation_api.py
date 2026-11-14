"""Stateful Dinkster and OpenAI-compatible generation HTTP adapters."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from types import TracebackType
from typing import Self, cast

import pytest
from aiohttp import ClientResponse
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_inference_wire import (
    GenerationEvent,
    GenerationFinishReason,
    GenerationMessageRole,
    GenerationProviderCapabilities,
    GenerationRequest,
    GenerationResult,
    GenerationSamplerKind,
    GenerationSessionHandle,
    GenerationStats,
    GenerationTerminalEvent,
    GenerationTokenEvent,
)
from dinkster_server import create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from dinkster.generation_api import (
    GenerationModel,
    GenerationService,
    add_generation_routes,
)
from tests.async_scenarios import run_scenario

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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())


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

    run_scenario(scenario())
