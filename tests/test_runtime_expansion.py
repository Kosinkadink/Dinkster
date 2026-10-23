"""Runtime expansion uses the same invocation and value ownership on every worker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from dinkster_memory import (
    GovernorReservationService,
    MemoryGovernor,
    ReservationRequest,
)
from dinkster_protocol import (
    CurrentOutputRef,
    ExpandedReturn,
    Invocation,
    LocalExpansion,
    LocalNode,
    LocalOutputRef,
    PresentOutput,
    Worker,
)
from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr, build_node_types
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import GovernedWorker, InProcessWorker, IsolatedWorker, RemoteWorker


class WireExpansionNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "expansion.test",
            outputs=(
                OutputSpec("result", TypeExpr.concrete("core.int")),
                OutputSpec("seed", TypeExpr.concrete("core.int")),
            ),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls) -> Any:
        values = TypeRegistry()
        register_core_types(values)
        return ExpandedReturn(
            LocalExpansion(
                (LocalNode("child", "expansion.child", (("value", CurrentOutputRef("seed")),)),)
            ),
            (
                ("result", LocalOutputRef("child", "value")),
                ("seed", PresentOutput(values.wrap("core.int", 17))),
            ),
        )


PACK_NODES = [WireExpansionNode]


@pytest.mark.parametrize(
    ("worker_kind", "use_shm"),
    [
        ("in-process", False),
        ("governed", False),
        ("isolated", False),
        ("isolated", True),
        ("remote", False),
    ],
)
def test_expansion_result_survives_worker_round_trip(
    tmp_path: Path, worker_kind: str, use_shm: bool
) -> None:
    from test_remote import TOKEN, start_service, stop_service

    async def scenario() -> None:
        values = TypeRegistry()
        register_core_types(values)
        boundary_worker: IsolatedWorker | RemoteWorker | None = None
        service: asyncio.subprocess.Process | None = None
        worker: Worker
        if worker_kind in ("isolated", "remote"):
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "expansion"\nnamespaces = ["expansion"]\n'
                '[pack.entry]\nnodes = "tests.test_runtime_expansion:PACK_NODES"\n',
                encoding="ascii",
            )
            root = Path(__file__).resolve().parent.parent
            if worker_kind == "remote":
                service, host, port = await start_service(manifest, tmp_path, pythonpath=(root,))
                boundary_worker = RemoteWorker(host, port, TOKEN, values, name="expansion")
            else:
                boundary_worker = IsolatedWorker(
                    manifest,
                    values,
                    use_shm=use_shm,
                    shm_threshold=1,
                    extra_env={"PYTHONPATH": str(root)},
                )
            worker = boundary_worker
        else:
            worker = InProcessWorker(build_node_types(PACK_NODES), values)
            if worker_kind == "governed":
                worker = GovernedWorker(
                    worker,
                    planner=lambda _: (ReservationRequest("ram", 1),),
                    reservations=GovernorReservationService(MemoryGovernor({"ram": 16})),
                )
        try:
            if boundary_worker is not None:
                await boundary_worker.start()
            for index in range(2):
                result = await worker.invoke(
                    Invocation(
                        f"invocation-{index}",
                        "parent",
                        "expansion.test",
                        {},
                        WireExpansionNode.schema(),
                        job_ref="run",
                    )
                )
                assert result.error is None
                assert result.outputs is not None
                assert result.outputs["seed"].resolve() == 17
                assert result.expansion is not None
                bindings = dict(result.expansion.bindings)
                assert bindings["result"] == LocalOutputRef("child", "value")
                assert isinstance(bindings["seed"], PresentOutput)
                assert bindings["seed"].value is result.outputs["seed"]
                assert result.expansion.expansion.nodes == (
                    LocalNode("child", "expansion.child", (("value", CurrentOutputRef("seed")),)),
                )
        finally:
            try:
                if boundary_worker is not None:
                    await boundary_worker.close()
            finally:
                if service is not None:
                    await stop_service(service)

    asyncio.run(scenario())
