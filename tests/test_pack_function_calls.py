"""A button's Call step reaches a declared pack function in the pack's own worker."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_protocol.lookups import LOOKUP_CALLS_UNDER_WAY
from dinkster_protocol.pack_functions import (
    EXTRA_KEY_MESSAGE,
    PACK_FUNCTION_PATH,
    PAUSED_MESSAGE,
    TIMEOUT_MESSAGE,
    PackFunctionAnswer,
    PackFunctionRequest,
)
from dinkster_schema import NodeSchema, TypeExpr
from dinkster_schema.body import Button, Call, CallContext, pack_function
from dinkster_schema.model import Input
from dinkster_server import Principal, StaticBearerAuthenticator, create_app
from dinkster_server.auth import install_auth
from dinkster_server.call_budget import CallBudget
from dinkster_server.pack_functions import install_pack_functions
from dinkster_workers.pack_functions import run_pack_function

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario

DEV_PACK = Path(__file__).parents[1] / "packages/dinkster-nodes-dev/dinkster-pack.toml"
TRANSLATE = "dinkster_nodes_dev.buttons.translate_now"
PROBE = "dinkster_nodes_dev.buttons.call_probe"


async def _call(
    client: TestClient[Any, Any], wire: dict[str, object]
) -> tuple[int, list[Any], bytes]:
    """POST one call; the progress and final lines, and the file bytes after a file line."""
    response = await client.post(PACK_FUNCTION_PATH, json=wire)
    if response.status != 200:
        return response.status, [await response.json()], b""
    lines: list[Any] = []
    while True:
        raw = await response.content.readline()
        if not raw:
            return response.status, lines, b""
        line = json.loads(raw)
        lines.append(line)
        if "file" in line:
            return response.status, lines, await response.content.readexactly(line["file"])


def _probe(mode: str, delay: float = 0.0, *, download: bool = False) -> dict[str, object]:
    return PackFunctionRequest(
        "dev.buttons.call_probe",
        PROBE,
        (mode, delay),
        () if download else ("first", "second"),
        "probe.txt" if download else None,
    ).to_wire()


def test_a_call_runs_the_declared_function_and_answers_each_way() -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(DEV_PACK, trust_reserved=True))
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                schema_owners=composition.schema_owners,
                choice_owners=composition.choice_owners,
                pack_function_dispatch=composer.call_pack_function,
            )
            async with TestClient(TestServer(app)) as client:
                translate = PackFunctionRequest(
                    "dev.buttons.translate",
                    TRANSLATE,
                    ("hola, un gato en la playa", "auto", "en", 0.2),
                    ("text",),
                ).to_wire()
                status, lines, _ = await _call(client, translate)
                assert status == 200
                assert lines[-1] == {"answer": {"text": "hello, a cat on the beach"}, "notes": []}
                assert any("progress" in line for line in lines[:-1])

                translate["args"] = ["hola perro", "es", "en", 0.0]
                _, lines, _ = await _call(client, translate)
                assert lines[-1] == {
                    "answer": {"text": "hello perro"},
                    "notes": ["Kept as written: perro."],
                }

                _, lines, _ = await _call(client, _probe("answer"))
                assert lines[-1] == {"answer": {"first": "one", "second": 2}, "notes": []}
                _, lines, _ = await _call(client, _probe("partial"))
                assert lines[-1] == {
                    "answer": {"first": "only"},
                    "notes": ["The second field was left out."],
                }
                _, lines, _ = await _call(client, _probe("raise"))
                assert lines[-1] == {
                    "error": "raised",
                    "message": "No sheet named video in this file.",
                }
                _, lines, _ = await _call(client, _probe("extra"))
                assert lines[-1] == {"error": "extra-key", "message": EXTRA_KEY_MESSAGE}

                started = time.monotonic()
                _, lines, _ = await _call(client, _probe("slow"))
                assert lines[-1] == {"error": "timeout", "message": TIMEOUT_MESSAGE}
                assert 1.5 < time.monotonic() - started < 2.9

                status, lines, data = await _call(client, _probe("file", download=True))
                assert lines[-1] == {"file": 10, "notes": []} and data == b"pack file\n"
                _, lines, _ = await _call(client, _probe("large_file", download=True))
                assert lines[-1]["error"] == "too-large"

                # A function, writes or node type the declaration does not name is refused,
                # and no module is imported for it.
                for wire in (
                    {**_probe("answer"), "writes": ["first"]},
                    {**_probe("answer"), "nodeType": "dev.buttons.translate"},
                    {**_probe("answer"), "function": "os.system"},
                    {**_probe("answer"), "nodeType": "dev.image.gradient"},
                ):
                    _, lines, _ = await _call(client, wire)
                    assert lines[-1]["error"] == "not-declared", wire
                _, lines, _ = await _call(client, {**_probe("answer"), "args": [3, 0.0]})
                assert lines[-1]["error"] == "wrong-arguments"
                _, lines, _ = await _call(
                    client, {**_probe("answer"), "args": ["answer", {"$float": 0}]}
                )
                assert lines[-1]["answer"] == {"first": "one", "second": 2}
                assert (await client.post(PACK_FUNCTION_PATH, json={"nodeType": 1})).status == 400
        finally:
            await composer.close()

    run_scenario(scenario())


def test_closing_the_request_cancels_the_call_and_the_worker_takes_the_next() -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(DEV_PACK, trust_reserved=True))
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                schema_owners=composition.schema_owners,
                choice_owners=composition.choice_owners,
                pack_function_dispatch=composer.call_pack_function,
            )
            async with TestClient(TestServer(app)) as client:
                response = await client.post(PACK_FUNCTION_PATH, json=_probe("answer", 1.8))
                assert "progress" in json.loads(await response.content.readline())
                response.close()
                started = time.monotonic()
                _, lines, _ = await _call(client, _probe("answer"))
                assert lines[-1]["answer"] == {"first": "one", "second": 2}
                # The cancelled function stopped waiting instead of holding its thread for 1.8 s.
                assert time.monotonic() - started < 1.0
        finally:
            await composer.close()

    run_scenario(scenario())


def test_the_route_refuses_agents_and_over_budget_presses() -> None:
    calls: list[PackFunctionRequest] = []
    gate = asyncio.Event()

    async def dispatch(request: PackFunctionRequest, on_progress: Any) -> PackFunctionAnswer:
        calls.append(request)
        await gate.wait()
        return PackFunctionAnswer({"first": "one"}, None)

    async def scenario() -> None:
        app = web.Application()
        budget = CallBudget()
        capabilities = install_pack_functions(app, dispatch, budget)
        auth = StaticBearerAuthenticator(
            {
                "person": Principal("person", grants={"local": frozenset({"jobs:submit"})}),
                "agent": Principal(
                    "agent", grants={"local": frozenset({"jobs:submit"})}, kind="agent"
                ),
                "reader": Principal("reader", grants={"local": frozenset({"jobs:read"})}),
            }
        )
        install_auth(app, auth, route_capabilities=capabilities)
        async with TestClient(TestServer(app)) as client:
            wire = _probe("answer")
            reader = await client.post(
                PACK_FUNCTION_PATH, json=wire, headers={"Authorization": "Bearer reader"}
            )
            assert reader.status == 403
            agent = await client.post(
                PACK_FUNCTION_PATH, json=wire, headers={"Authorization": "Bearer agent"}
            )
            assert agent.status == 403 and (await agent.json())["error"] == "headless"
            client.session.headers["Authorization"] = "Bearer person"
            held = [budget.admit("person") for _ in range(LOOKUP_CALLS_UNDER_WAY - 1)]
            running = asyncio.ensure_future(client.post(PACK_FUNCTION_PATH, json=wire))
            while not calls:
                await asyncio.sleep(0.01)
            paused = await client.post(PACK_FUNCTION_PATH, json=wire)
            assert paused.status == 429
            assert await paused.json() == {"error": "paused", "message": PAUSED_MESSAGE}
            gate.set()
            response = await running
            assert json.loads(await response.content.readline()) == {
                "answer": {"first": "one"},
                "notes": [],
            }
            assert len(calls) == 1
            for end in held:
                end()
            assert budget.admit("bob") and budget.admit("person")

    run_scenario(scenario())


saw_cancel: list[bool] = []


@pack_function(time_limit=5.0)
def waits_for_cancel(context: CallContext, seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if context.cancelled:
            saw_cancel.append(True)
            return {"note": "late"}
        time.sleep(0.01)
    saw_cancel.append(False)
    return {"note": "done"}


def test_a_cancelled_call_tells_the_function_and_sends_no_answer() -> None:
    schema = NodeSchema(
        "test.cancel",
        body=(
            Input("seconds", TypeExpr.concrete("FLOAT"), default=2.0),
            Input("note", TypeExpr.concrete("STRING"), default=""),
            Button("go", "Go", (Call(waits_for_cancel, args=("seconds",), writes=("note",)),)),
        ),
    )
    request = PackFunctionRequest("test.cancel", f"{__name__}.waits_for_cancel", (2.0,), ("note",))
    sent: list[dict[str, object]] = []

    async def send(header: dict[str, object], blobs: Any) -> None:
        sent.append(header)

    async def scenario() -> None:
        cancelled = threading.Event()
        task = asyncio.ensure_future(
            run_pack_function(
                {"requestId": "r", "call": request.to_wire()},
                {"test.cancel": schema},
                send,
                cancelled,
            )
        )
        await asyncio.sleep(0.2)
        cancelled.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        started = time.monotonic()
        while not saw_cancel and time.monotonic() - started < 1.5:
            await asyncio.sleep(0.01)
        assert saw_cancel == [True]
        assert [header for header in sent if header["type"] == "packFunctionResult"] == []

    run_scenario(scenario())
