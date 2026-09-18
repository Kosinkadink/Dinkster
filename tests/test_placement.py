"""PlacementWorker: placement consumes residency (DESIGN 3.10).

RoutingWorker routes by node type; PlacementWorker picks *which instance*
of homogeneous workers runs an invocation, from the device residency its
input values declare. DeviceMap has already translated residency into the
parent namespace, so device strings here name silicon unambiguously.
"""

from __future__ import annotations

import asyncio

import pytest
from dinkster_engine import Invocation, InvocationResult, OnInvocationEvent
from dinkster_schema import InputSpec, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import (
    CORE_STRING,
    RESOURCES_META_KEY,
    TypeRegistry,
    Value,
    register_core_types,
)
from dinkster_workers import PlacementWorker

STRING = TypeExpr.concrete(CORE_STRING)
SCHEMA = NodeSchema(
    node_type="test.sample",
    display_name="Sample",
    category="test",
    inputs=(InputSpec("model", STRING), InputSpec("prompt", STRING)),
    outputs=(OutputSpec("image", STRING),),
)


def resident_value(resources: object) -> Value:
    registry = TypeRegistry()
    registry.register("t.model", meta=lambda obj: {RESOURCES_META_KEY: resources})
    return registry.wrap("t.model", object())


def plain_value(text: str = "hi") -> Value:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry.wrap(CORE_STRING, text)


def make_invocation(inputs: dict[str, Value]) -> Invocation:
    return Invocation(
        invocation_id="i1",
        node_id="n1",
        node_type="test.sample",
        inputs=inputs,
        effective_schema=SCHEMA,
    )


class FakeWorker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.prepared: list[list[str]] = []
        self.invoked: list[Invocation] = []

    async def prepare(self, node_types) -> None:  # noqa: ANN001
        self.prepared.append(list(node_types))

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        self.invoked.append(invocation)
        return InvocationResult(outputs={"worker": plain_value(self.name)})


def two_workers() -> tuple[FakeWorker, FakeWorker, PlacementWorker]:
    a, b = FakeWorker("a"), FakeWorker("b")
    placement = PlacementWorker(
        {"a": a, "b": b},
        devices={"cuda:0": "a", "cuda:1": "b"},
        default="a",
    )
    return a, b, placement


def run_placed(placement: PlacementWorker, inputs: dict[str, Value]) -> InvocationResult:
    return asyncio.run(placement.invoke(make_invocation(inputs)))


def placed_on(result: InvocationResult) -> str:
    assert result.error is None, result.error
    assert result.outputs is not None
    name = result.outputs["worker"].resolve()
    assert isinstance(name, str)
    return name


# -- construction ---------------------------------------------------------


def test_unknown_device_owner_fails_at_construction() -> None:
    with pytest.raises(KeyError, match="unknown worker"):
        PlacementWorker({"a": FakeWorker("a")}, devices={"cuda:0": "nope"})


def test_unknown_default_fails_at_construction() -> None:
    with pytest.raises(KeyError, match="unknown worker"):
        PlacementWorker({"a": FakeWorker("a")}, devices={}, default="nope")


# -- pinned by residency --------------------------------------------------


def test_resident_input_pins_to_owner() -> None:
    a, b, placement = two_workers()
    result = run_placed(
        placement,
        {"model": resident_value({"gpu": "cuda:1"}), "prompt": plain_value()},
    )
    assert placed_on(result) == "b"
    assert not a.invoked and len(b.invoked) == 1


def test_two_inputs_same_owner_agree() -> None:
    _, b, placement = two_workers()
    result = run_placed(
        placement,
        {
            "model": resident_value({"gpu": "cuda:1"}),
            "prompt": resident_value({"gpu": "cuda:1"}),
        },
    )
    assert placed_on(result) == "b"


def test_multi_device_residency_same_owner_pins() -> None:
    # A value spanning devices both owned by one worker (the multigpu case).
    a, b, placement_multi = (
        FakeWorker("a"),
        FakeWorker("b"),
        None,
    )
    placement_multi = PlacementWorker(
        {"a": a, "b": b},
        devices={"cuda:0": "a", "cuda:1": "a", "cuda:2": "b"},
    )
    result = run_placed(
        placement_multi,
        {"model": resident_value({"gpu": ("cuda:0", "cuda:1")})},
    )
    assert placed_on(result) == "a"


# -- impossible placements are error results, not guesses -----------------


def test_inputs_on_different_workers_is_error() -> None:
    a, b, placement = two_workers()
    result = run_placed(
        placement,
        {
            "model": resident_value({"gpu": "cuda:0"}),
            "prompt": resident_value({"gpu": "cuda:1"}),
        },
    )
    assert result.error is not None
    assert "different workers" in result.error.message
    assert not a.invoked and not b.invoked


def test_multi_device_value_spanning_workers_is_error() -> None:
    _, _, placement = two_workers()
    result = run_placed(placement, {"model": resident_value({"gpu": ("cuda:0", "cuda:1")})})
    assert result.error is not None
    assert "different workers" in result.error.message


def test_unowned_device_is_error() -> None:
    _, _, placement = two_workers()
    result = run_placed(placement, {"model": resident_value({"gpu": "cuda:9"})})
    assert result.error is not None
    assert "cuda:9" in result.error.message


# -- unpinned: policy, then default ---------------------------------------


def test_unpinned_uses_policy() -> None:
    a, b = FakeWorker("a"), FakeWorker("b")
    placement = PlacementWorker(
        {"a": a, "b": b},
        devices={"cuda:0": "a", "cuda:1": "b"},
        place=lambda invocation: "b",
        default="a",
    )
    result = run_placed(placement, {"prompt": plain_value()})
    assert placed_on(result) == "b"


def test_policy_naming_unknown_worker_is_error() -> None:
    a = FakeWorker("a")
    placement = PlacementWorker({"a": a}, devices={}, place=lambda invocation: "ghost")
    result = run_placed(placement, {"prompt": plain_value()})
    assert result.error is not None
    assert "ghost" in result.error.message
    assert not a.invoked


def test_unpinned_without_policy_uses_default() -> None:
    a, _, placement = two_workers()
    result = run_placed(placement, {"prompt": plain_value()})
    assert placed_on(result) == "a"


def test_unpinned_without_policy_or_default_is_error() -> None:
    placement = PlacementWorker({"a": FakeWorker("a")}, devices={"cuda:0": "a"})
    result = run_placed(placement, {"prompt": plain_value()})
    assert result.error is not None
    assert "no placement policy" in result.error.message


def test_residency_beats_policy() -> None:
    # Owners win: policy is only consulted for unpinned invocations.
    a, b = FakeWorker("a"), FakeWorker("b")
    placement = PlacementWorker(
        {"a": a, "b": b},
        devices={"cuda:0": "a", "cuda:1": "b"},
        place=lambda invocation: "a",
    )
    result = run_placed(placement, {"model": resident_value({"gpu": "cuda:1"})})
    assert placed_on(result) == "b"


# -- prepare fans out (homogeneous by contract) ---------------------------


def test_prepare_reaches_every_worker() -> None:
    a, b, placement = two_workers()
    asyncio.run(placement.prepare(["test.sample"]))
    assert a.prepared == [["test.sample"]]
    assert b.prepared == [["test.sample"]]
