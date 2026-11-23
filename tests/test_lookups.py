"""Calling declared lookups: the route, sharing and budget, and the real dev pack worker."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import ComboOption, lookup
from dinkster_protocol.lookups import (
    LOOKUP_ANSWER_MAX_BYTES,
    LOOKUP_CALLS_PER_MINUTE,
    LOOKUP_CALLS_UNDER_WAY,
    LookupCallError,
    LookupRequest,
    lookup_answer_to_wire,
    lookup_arguments,
)
from dinkster_schema.body import LookupMark
from dinkster_server import create_app
from dinkster_server.lookups import LookupCalls, lookups_changed

import dinkster.compose
from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario

DEV_PACK = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-dev/dinkster-pack.toml"
NOTE = "dev.lookups.note"
SUMMARY = "dinkster_nodes_dev.lookups.lora_summary"
QUICK = "dinkster_nodes_dev.lookups.quick_note"


@lookup
def scaled(width: float, count: int, names: list[str]) -> list[ComboOption]:
    return [ComboOption(name, label=f"{name} x{count}") for name in names][: int(width)]


@lookup
def limit(value: int) -> int | None:
    return None if value < 0 else value


def _no_assets(wire: object) -> object:
    raise AssertionError("no asset argument expected")


def test_arguments_meet_parameters_by_position_and_type() -> None:
    assert lookup_arguments(scaled, [2, 3, ["a", "b"]], asset=_no_assets) == [2.0, 3, ["a", "b"]]
    refused = (
        [2.5, 2.5, ["a"]],  # a decimal never narrows to a whole number
        [2, True, ["a"]],  # a bool is not a whole number
        [2, 3, ["a", 1]],  # every list item is text
        [2, 3],  # too few
        [2, 3, ["a"], "extra"],  # too many
    )
    for args in refused:
        with pytest.raises(LookupCallError) as error:
            lookup_arguments(scaled, args, asset=_no_assets)
        assert error.value.code == "lookup-invalid-arguments", args


def test_answers_must_fit_their_annotation_and_the_size_bound() -> None:
    assert lookup_answer_to_wire(scaled, scaled(1, 2, ["a", "b"])) == [
        {"value": "a", "label": "a x2"}
    ]
    assert lookup_answer_to_wire(limit, None) is None
    assert lookup_answer_to_wire(limit, 7) == 7
    for function, answer in (
        (limit, 2.5),
        (limit, True),
        (limit, 2**60),
        (scaled, ["plain text where options are declared"]),
        (scaled, [ComboOption("x" * 1024, label="y" * 1024)] * (LOOKUP_ANSWER_MAX_BYTES // 2048)),
    ):
        with pytest.raises(LookupCallError) as error:
            lookup_answer_to_wire(function, answer)
        assert error.value.code == "lookup-wrong-shape"


def test_equal_requests_share_one_call_and_leaving_cancels_only_when_last() -> None:
    async def scenario() -> None:
        started: list[tuple[object, ...]] = []
        release = asyncio.Event()
        cancelled = asyncio.Event()

        async def dispatch(node_type: str, function: str, args: object) -> object:
            started.append((function, args))
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return f"answer {args}"

        calls = LookupCalls(dispatch)
        same = LookupRequest(NOTE, SUMMARY, ("sd", 0.0, False))
        waiters = [asyncio.create_task(calls.call("user", 1, "pack", same)) for _ in range(50)]
        other_args = asyncio.create_task(
            calls.call("user", 1, "pack", LookupRequest(NOTE, SUMMARY, ("sdxl", 0.0, False)))
        )
        other_user = asyncio.create_task(calls.call("someone", 1, "pack", same))
        other_catalog = asyncio.create_task(calls.call("user", 2, "pack", same))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(started) == 4

        waiters[0].cancel()
        await asyncio.sleep(0)
        assert not cancelled.is_set()
        release.set()
        answers = await asyncio.gather(*waiters[1:])
        assert set(answers) == {"answer ('sd', 0.0, False)"}
        assert await other_args == "answer ('sdxl', 0.0, False)"
        await asyncio.gather(other_user, other_catalog)

        release.clear()
        lonely = asyncio.create_task(calls.call("user", 1, "pack", same))
        await asyncio.sleep(0)
        lonely.cancel()
        await asyncio.wait_for(cancelled.wait(), 1)
        assert len(started) == 5

    run_scenario(scenario())


def test_each_user_has_eight_calls_under_way_and_sixty_a_minute() -> None:
    async def scenario() -> None:
        now = [0.0]
        release = asyncio.Event()

        async def dispatch(node_type: str, function: str, args: object) -> object:
            await release.wait()
            return args

        calls = LookupCalls(dispatch, clock=lambda: now[0])

        def request(index: int) -> LookupRequest:
            return LookupRequest(NOTE, SUMMARY, (f"c{index}", 0.0, False))

        busy = [
            asyncio.create_task(calls.call("user", 1, "pack", request(index)))
            for index in range(LOOKUP_CALLS_UNDER_WAY)
        ]
        joined = asyncio.create_task(calls.call("user", 1, "pack", request(0)))
        await asyncio.sleep(0)
        with pytest.raises(LookupCallError) as error:
            await calls.call("user", 1, "pack", request(99))
        assert error.value.code == "lookup-paused"
        other = asyncio.create_task(calls.call("other", 1, "pack", request(99)))
        await asyncio.sleep(0)
        assert not other.done()
        release.set()
        await asyncio.gather(*busy, joined, other)

        for index in range(LOOKUP_CALLS_UNDER_WAY, LOOKUP_CALLS_PER_MINUTE):
            await calls.call("user", 1, "pack", request(index))
        with pytest.raises(LookupCallError) as error:
            await calls.call("user", 1, "pack", request(1000))
        assert error.value.code == "lookup-paused"
        now[0] = 60.0
        assert await calls.call("user", 1, "pack", request(1000)) == ("c1000", 0.0, False)

    run_scenario(scenario())


def test_run_and_asset_events_name_the_lookups_they_make_old() -> None:
    marks = {
        "a": {"pack.f": LookupMark(watches=("model/lora",)), "pack.g": LookupMark(after_run=True)},
        "b": {"pack.h": LookupMark()},
    }

    def of(node_type: str) -> dict[str, LookupMark]:
        return marks.get(node_type, {})

    assert lookups_changed(["a", "b"], of, after_run=True) == ["pack.g"]
    assert lookups_changed(["a", "b"], of, asset_kinds=["model/lora"]) == ["pack.f"]
    assert lookups_changed(["a", "b"], of, asset_kinds=["model/checkpoint"]) == []
    assert lookups_changed(["b"], of, after_run=True) == []


def test_dev_pack_lookups_through_the_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(dinkster.compose, "LOOKUP_TIME_LIMIT", 1.5)

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(DEV_PACK))
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
                lookup_dispatch=composer.call_lookup,
                lookup_marks=composer.lookup_marks,
            )
            async with TestClient(TestServer(app)) as client:

                async def ask(function: str, *args: object, **headers: str) -> tuple[int, object]:
                    response = await client.post(
                        "/api/lookups",
                        json={"nodeType": NOTE, "function": function, "args": list(args)},
                        headers=headers,
                    )
                    return response.status, await response.json()

                assert await ask(SUMMARY, "sdxl", 0, False) == (200, {"answer": "2 LoRAs in sdxl"})
                assert await ask(SUMMARY, "sd", 0, True) == (502, {"error": "lookup-failed"})
                assert await ask(SUMMARY, "sd", "soon", False) == (
                    400,
                    {"error": "lookup-invalid-arguments"},
                )
                assert await ask("os.system", "true") == (404, {"error": "lookup-unknown"})
                assert await ask(SUMMARY, "sd", 4, False) == (504, {"error": "lookup-timeout"})
                stale = await client.post(
                    "/api/lookups",
                    json={"nodeType": NOTE, "function": QUICK, "args": []},
                    headers={"If-Match": "0"},
                )
                assert stale.status == 412

                slow = asyncio.create_task(ask(SUMMARY, "All", 1, False))
                await asyncio.sleep(0.1)
                began = time.monotonic()
                assert await ask(QUICK) == (200, {"answer": "Answered at once."})
                assert time.monotonic() - began < 0.5
                assert not slow.done()
                assert await slow == (200, {"answer": "5 LoRAs in All"})
        finally:
            await composer.close()

    run_scenario(scenario())
