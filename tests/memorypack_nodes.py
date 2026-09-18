"""Test pack for the worker memory relay. Imported by the worker HOST
process (via a manifest entry), never by the test process's engine side.

``mem.load`` admits a fake model into a real ResidentPool (the relay's
concrete target), declaring vram and ram costs. The pack exposes the pool
through the ``consumers`` manifest entry, so the parent's governor sees it
as a relayed consumer - footprints and details from pushed snapshots,
vram shedding over the live round trip, ram pressure gated parent-side.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from pathlib import Path

from dinkster_compat_comfy import ResidentPool, register_resident_type
from dinkster_memory import FullReleaseResult, MeasuredMemory, PressureSignal
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import CORE_INT, CORE_STRING, COST_META_KEY, TypeRegistry

MODEL = TypeExpr.concrete("mem.model")
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)


class FakeModel:
    """Stands in for a loaded checkpoint: costs are declared, not measured."""

    def __init__(self, name: str, vram: int, ram: int, device: str) -> None:
        self.name = name
        self.cost = {f"vram:{device}": vram, "ram": ram}


def _cost_of(obj: object) -> Mapping[str, object]:
    assert isinstance(obj, FakeModel)
    return {COST_META_KEY: dict(obj.cost)}


POOL = ResidentPool(cost_of=_cost_of)


class ZeroCostConsumer:
    def __init__(self) -> None:
        self.released = False

    def footprint(self, device: str) -> int:
        del device
        return 0

    async def shed(self, pressure: PressureSignal) -> int:
        del pressure
        return 0

    async def full_release(self) -> FullReleaseResult:
        self.released = True
        return FullReleaseResult("complete")


ZERO_COST_CONSUMERS: list[ZeroCostConsumer] = []


def register_types(registry: TypeRegistry) -> None:
    register_resident_type(registry, "mem.model", table=POOL, meta=_cost_of)


def memory_consumers() -> dict[str, object]:
    consumer = ZeroCostConsumer()
    ZERO_COST_CONSUMERS.append(consumer)
    return {"models": POOL, "zero-cost": consumer}


class SerialFullReleaseConsumer:
    def __init__(self) -> None:
        self.active = False

    def footprint(self, device: str) -> int:
        del device
        return 0

    async def shed(self, pressure: PressureSignal) -> int:
        del pressure
        return 0

    async def full_release(self) -> FullReleaseResult:
        if self.active:
            return FullReleaseResult("busy")
        self.active = True
        try:
            await asyncio.sleep(0.05)
            return FullReleaseResult("complete")
        finally:
            self.active = False


SERIAL_FULL_RELEASE = SerialFullReleaseConsumer()


def serial_memory_consumers() -> dict[str, object]:
    return {"serial": SERIAL_FULL_RELEASE}


# Fake device telemetry, worker-side: mem.load "consumes" measured free on
# the device it loads to, so a telemetry refresh is observable from the
# parent (initial values via hello, decremented values via the
# post-invocation memoryReport).
TELEMETRY: dict[str, MeasuredMemory] = {
    "vram:cuda:0": MeasuredMemory(free_bytes=8_000, total_bytes=10_000),
}


def memory_telemetry() -> dict[str, MeasuredMemory]:
    return dict(TELEMETRY)


def structured_memory_telemetry() -> dict[str, MeasuredMemory]:
    return {
        "vram:cuda:0": MeasuredMemory(
            free_bytes=8_000,
            total_bytes=10_000,
            driver_free_bytes=6_000,
            allocator_reclaimable_bytes=1_000,
            dynamic_evictable_bytes=1_000,
            dynamic_pinned_bytes=2_000,
        )
    }


def malformed_telemetry() -> dict[object, object]:
    """One valid entry among garbage: the worker host must keep the valid
    sibling, drop everything else, and stay alive."""
    return {
        "vram:cuda:0": MeasuredMemory(free_bytes=8_000, total_bytes=10_000),
        "": MeasuredMemory(free_bytes=1, total_bytes=2),  # empty key
        123: MeasuredMemory(free_bytes=1, total_bytes=2),  # non-str key
        "vram:cuda:7": "not-measured",  # wrong value type
        "vram:cuda:8": MeasuredMemory(free_bytes=-1, total_bytes=2),  # negative free
        "vram:cuda:9": MeasuredMemory(free_bytes=1, total_bytes=0),  # zero total
        "vram:cuda:10": MeasuredMemory(free_bytes=5, total_bytes=2),  # free > total
    }


def raising_telemetry() -> dict[str, MeasuredMemory]:
    raise RuntimeError("probe exploded")


class MemLoad(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="mem.load",
            display_name="Fake Model Load",
            category="test",
            inputs=(
                InputSpec("name", STRING, default="fake.safetensors"),
                InputSpec("vram", INT, default=400),
                InputSpec("ram", INT, default=300),
                InputSpec("device", STRING, default="cuda:0"),
            ),
            outputs=(OutputSpec("model", MODEL),),
        )

    @classmethod
    def execute(cls, *, name: str, vram: int, ram: int, device: str) -> Mapping[str, object]:
        model = FakeModel(name, vram, ram, device)
        POOL.label(model, name)
        residency = f"vram:{device}"
        measured = TELEMETRY.get(residency)
        if measured is not None:
            # The load "consumed" device memory: the next telemetry
            # snapshot must show less free, like a real allocator would.
            TELEMETRY[residency] = MeasuredMemory(
                free_bytes=max(measured.free_bytes - vram, 0),
                total_bytes=measured.total_bytes,
            )
        return cls.outputs(model=model)


class MemExit(Node):
    """Kills the worker process mid-conversation (crash-path fixture)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="mem.exit",
            display_name="Worker Exit",
            category="test",
            inputs=(InputSpec("code", INT, default=3),),
            outputs=(OutputSpec("never", STRING),),
        )

    @classmethod
    def execute(cls, *, code: int) -> Mapping[str, object]:
        os._exit(code)


class MemUse(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="mem.use",
            display_name="Fake Model Use",
            category="test",
            inputs=(InputSpec("model", MODEL),),
            outputs=(OutputSpec("name", STRING),),
        )

    @classmethod
    def execute(cls, *, model: FakeModel) -> Mapping[str, object]:
        return cls.outputs(name=model.name)


class MemHold(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="mem.hold",
            display_name="Hold Worker",
            category="test",
            inputs=(InputSpec("milliseconds", INT, default=100),),
            outputs=(OutputSpec("done", STRING),),
        )

    @classmethod
    async def execute(cls, *, milliseconds: int) -> Mapping[str, object]:
        await asyncio.sleep(milliseconds / 1000)
        return cls.outputs(done="done")


class MemConsumerState(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="mem.consumer_state",
            outputs=(OutputSpec("created", INT), OutputSpec("live", INT)),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            created=len(ZERO_COST_CONSUMERS),
            live=sum(not consumer.released for consumer in ZERO_COST_CONSUMERS),
        )


class MemThreadHold(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="mem.thread_hold",
            inputs=(InputSpec("entered", STRING), InputSpec("finish", STRING)),
            outputs=(OutputSpec("done", STRING),),
        )

    @classmethod
    def execute(cls, *, entered: str, finish: str) -> Mapping[str, object]:
        Path(entered).touch()
        while not Path(finish).exists():
            time.sleep(0.005)
        return cls.outputs(done="done")


NODES = [MemLoad, MemExit, MemUse, MemHold, MemConsumerState, MemThreadHold]
