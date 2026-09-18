"""Test pack for isolated-worker tests. Imported by the worker HOST process
(via a manifest entry), never by the test process's engine side - keep it
dependency-light and side-effect-free at import (hazard H5)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Mapping

from dinkster_memory import InvocationView, ReservationRequest
from dinkster_protocol import (
    ExportSnapshot,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaReady,
    WorkGroupMessage,
    WorkGroupReleased,
)
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    pack_logger,
    report_event,
    report_log,
    report_preview,
    report_progress,
)
from dinkster_values import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    COST_META_KEY,
    RESOURCES_META_KEY,
    TypeRegistry,
)

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
BLOB = TypeExpr.concrete("iso.blob")
GPU_BLOB = TypeExpr.concrete("iso.gpu_blob")


def register_types(registry: TypeRegistry) -> None:
    # Deliberately bare: no codec, no fingerprint - exercises the
    # correct-everywhere default path and the fallback-codec diagnostic.
    registry.register("iso.blob")
    # A value that publishes device facts in this process's namespace -
    # the device-namespacing test subject (a stand-in for a loaded model).
    registry.register(
        "iso.gpu_blob",
        meta=lambda obj: {
            RESOURCES_META_KEY: {"gpu": "cuda:0"},
            COST_META_KEY: {"vram:cuda:0": 128, "ram": 16},
        },
    )


class Sleepy(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.sleepy",
            display_name="Sleepy",
            category="test",
            inputs=(
                InputSpec("value", STRING),
                InputSpec("seconds", FLOAT, default=0.05),
            ),
            outputs=(OutputSpec("value", STRING),),
            io_bound=True,
        )

    @classmethod
    async def execute(cls, *, value: str, seconds: float) -> Mapping[str, object]:
        deadline = time.perf_counter() + seconds
        while (remaining := deadline - time.perf_counter()) > 0:
            await asyncio.sleep(remaining)
        return cls.outputs(value=value)


class Chatty(Node):
    """Reports progress, a binary preview, and a pack event while running -
    the cross-boundary streaming test subject. ``text`` carries the node's
    input value so concurrent invocations' events are distinguishable."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.chatty",
            display_name="Chatty",
            category="test",
            inputs=(
                InputSpec("value", STRING),
                InputSpec("seconds", FLOAT, default=0.0),
            ),
            outputs=(OutputSpec("value", STRING),),
            io_bound=True,
        )

    @classmethod
    async def execute(cls, *, value: str, seconds: float) -> Mapping[str, object]:
        report_progress(1, 2, text=value)
        if seconds:
            await asyncio.sleep(seconds)
        report_progress(2, 2, text=value)
        report_preview(b"\x89fakepng:" + value.encode(), mime="image/png", width=4, height=2)
        report_event("isopack.stage", {"stage": "done", "value": value})
        return cls.outputs(value=value.upper())


class Talky(Node):
    """Emits execution log records through every capture path: an explicit
    report_log, a print to stdout, a raw stderr write, and a pack logging
    record - the cross-boundary execution-log test subject."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.talky",
            display_name="Talky",
            category="test",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        report_log("info", f"explicit:{value}", data={"detail": "d"})
        print(f"printed:{value}")
        sys.stderr.write(f"errline:{value}\n")
        pack_logger("isopack").warning("logged:%s", value)
        return cls.outputs(value=value.upper())


class Boom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.boom",
            display_name="Boom",
            category="test",
            inputs=(InputSpec("tag", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, *, tag: str) -> Mapping[str, object]:
        raise RuntimeError(f"boom: {tag}")


class LazyProbe(Node):
    """Lazy-hook subject for the isolated worker boundary."""

    _cancel_active = False
    _cancel_cleanups = 0

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.lazy_probe",
            inputs=(
                InputSpec("mode", STRING),
                InputSpec("value", STRING, lazy=True),
            ),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def check_lazy_status(cls, *, mode: str, value: str | None) -> object:
        if mode == "error":
            raise RuntimeError("lazy hook exploded")
        if mode == "awaitable":
            return cls._await_status(value)
        if mode == "awaitable-error":
            return cls._await_error()
        if mode == "awaitable-malformed":
            return cls._await_malformed()
        if mode == "awaitable-cancel":
            return cls._await_cancel()
        if mode == "cleanup-retry":
            if cls._cancel_active or cls._cancel_cleanups != 1:
                raise RuntimeError("cancelled lazy hook did not clean up exactly once")
            return ["value"] if value is None else []
        if mode == "malformed":
            return [17]
        if mode == "nonserializable":
            return [object()]
        return ["value"] if value is None else []

    @classmethod
    async def _await_status(cls, value: str | None) -> object:
        await asyncio.sleep(0)
        return ["value"] if value is None else []

    @classmethod
    async def _await_error(cls) -> object:
        await asyncio.sleep(0)
        raise RuntimeError("async lazy hook exploded")

    @classmethod
    async def _await_malformed(cls) -> object:
        await asyncio.sleep(0)
        return [17]

    @classmethod
    async def _await_cancel(cls) -> object:
        if cls._cancel_active:
            raise RuntimeError("lazy cancellation probe is already active")
        cls._cancel_active = True
        try:
            await asyncio.sleep(30)
        finally:
            cls._cancel_active = False
            cls._cancel_cleanups += 1

    @classmethod
    def execute(cls, *, mode: str, value: str | None) -> Mapping[str, object]:
        del mode
        return cls.outputs(value=value)


class Exiter(Node):
    """Kills the worker process mid-invocation - the crash test."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.exit",
            display_name="Exiter",
            category="test",
            inputs=(InputSpec("code", INT, default=3),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, *, code: int) -> Mapping[str, object]:
        os._exit(code)


class BlobOut(Node):
    """Produces a value of a type the engine process has not registered."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.blob_out",
            display_name="Blob Out",
            category="test",
            inputs=(InputSpec("size", INT, default=8),),
            outputs=(OutputSpec("blob", BLOB),),
        )

    @classmethod
    def execute(cls, *, size: int) -> Mapping[str, object]:
        return cls.outputs(blob={"n": size, "data": "x" * size})


class BlobLen(Node):
    """Consumes iso.blob - proving an opaque value relays back intact."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.blob_len",
            display_name="Blob Len",
            category="test",
            inputs=(InputSpec("blob", BLOB),),
            outputs=(OutputSpec("length", INT),),
        )

    @classmethod
    def execute(cls, *, blob: Mapping[str, object]) -> Mapping[str, object]:
        data = blob["data"]
        assert isinstance(data, str)
        return cls.outputs(length=len(data))


class Hog(Node):
    """Pretends to materialize nbytes of host memory: the reservation test
    subject. Holds the (planned) reservation across a sleep so overlap
    between two hogs is observable from the parent's governor."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.hog",
            display_name="Hog",
            category="test",
            inputs=(
                InputSpec("nbytes", INT),
                InputSpec("seconds", FLOAT, default=0.0),
                InputSpec("residency", STRING, default="ram"),
            ),
            outputs=(OutputSpec("nbytes", INT),),
            io_bound=True,
        )

    @classmethod
    async def execute(cls, *, nbytes: int, seconds: float, residency: str) -> Mapping[str, object]:
        await asyncio.sleep(seconds)
        return cls.outputs(nbytes=nbytes)


class GpuBlobOut(Node):
    """Produces a value whose meta claims cuda:0 residency - in *this*
    process's device namespace."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.gpu_blob_out",
            display_name="GPU Blob Out",
            category="test",
            inputs=(),
            outputs=(OutputSpec("blob", GPU_BLOB),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(blob={"pretend": "model"})


class GpuBlobIdentity(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.gpu_blob_identity",
            inputs=(InputSpec("blob", GPU_BLOB),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, *, blob: object) -> Mapping[str, object]:
        del blob
        return cls.outputs(value="resident")


class ExecutionContextProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.execution_context",
            display_name="Execution Context",
            category="test",
            outputs=(OutputSpec("snapshot", STRING),),
            output_node=True,
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        from dinkster_workers import current_execution_context

        context = current_execution_context()
        assert context is not None
        assert context.node_id is not None
        snapshot = context.export_snapshot
        assert isinstance(snapshot, ExportSnapshot)
        return cls.outputs(snapshot=f"{context.node_id}|{snapshot.prompt}|{snapshot.extra_pnginfo}")


class IdentityEcho(Node):
    """Echoes the invocation's expected execution identity - the dispatch
    cache tag a policy selection carried across the worker boundary."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="iso.identity_echo",
            display_name="Identity Echo",
            category="test",
            outputs=(OutputSpec("identity", STRING),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        from dinkster_workers import current_execution_context

        context = current_execution_context()
        assert context is not None
        return cls.outputs(identity=context.expected_execution_identity or "none")


def plan_reservations(invocation: InvocationView) -> tuple[ReservationRequest, ...]:
    """Pack reservation policy: iso.hog reserves what it claims to need, on
    the residency class it names (in this process's device namespace).
    iso.boom also plans a small reservation so release-on-node-error is
    exercised (the node then fails after the grant)."""
    if invocation.node_type == "iso.hog":
        nbytes = invocation.inputs["nbytes"].resolve()
        residency_value = invocation.inputs.get("residency")
        residency = residency_value.resolve() if residency_value is not None else "ram"
        assert isinstance(nbytes, int)
        assert isinstance(residency, str)
        return (ReservationRequest(residency=residency, nbytes=nbytes),)
    if invocation.node_type == "iso.boom":
        return (ReservationRequest(residency="ram", nbytes=10),)
    return ()


def workgroup_handler_factory():
    async def handle(message: WorkGroupMessage) -> tuple[WorkGroupMessage, ...]:
        fields = {
            "worker": message.worker,
            "replica": message.replica,
            "group": message.group,
            "attempt": message.attempt,
            "device": message.device,
        }
        if type(message) is PrepareReplica:
            return (ReplicaReady(**fields),)
        if type(message) is ReleaseWorkGroup:
            return (WorkGroupReleased(**fields),)
        raise AssertionError(type(message))

    return handle


def invalid_workgroup_handler_factory() -> object:
    return object()


NODES = [
    Sleepy,
    Chatty,
    Talky,
    Boom,
    LazyProbe,
    Exiter,
    BlobOut,
    BlobLen,
    Hog,
    GpuBlobOut,
    GpuBlobIdentity,
    ExecutionContextProbe,
    IdentityEcho,
]
