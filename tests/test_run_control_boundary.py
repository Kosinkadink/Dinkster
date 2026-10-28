"""Live run control uses the same graph and admission across worker conversations."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_assets.declared import produced_output_scratch
from dinkster_caches import BudgetedDiskCAS, MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import GovernorReservationService, MemoryGovernor, ReservationRequest
from dinkster_protocol import Invocation
from dinkster_schema import RunExecutionContext, build_schemas, report_progress
from dinkster_values import TypeRegistry, Value, ValueMeta, is_absent, register_core_types
from dinkster_workers import IsolatedWorker, RemoteWorker

from tests.test_run_control import NODES, Publisher, Waiter


class SourcePublisher(Publisher):
    @classmethod
    def define_schema(cls):
        return replace(super().define_schema(), node_type="control.source_publish")

    @classmethod
    async def execute(cls, value: int, execution_context: RunExecutionContext) -> dict[str, object]:
        vault = produced_output_scratch()
        assert vault is not None
        data = f"published-source-{value}".encode("ascii")
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        envelope = replace(
            execution_context.inputs["value"],
            meta=ValueMeta({"asset_refs": [AssetRef(digest, "source.bin", len(data)).to_wire()]}),
        )
        await execution_context.publish_run_value("waiter", {"value": envelope})
        return {}


class SignallingWaiter(Waiter):
    @classmethod
    def define_schema(cls):
        return replace(super().define_schema(), node_type="control.signal_wait")

    @classmethod
    async def execute(cls, execution_context: RunExecutionContext) -> Mapping[str, Value]:
        report_progress(0, 1)
        return await super().execute(execution_context)


PACK_NODES = [*NODES, SourcePublisher, SignallingWaiter]


def reserve(_invocation: Invocation) -> Sequence[ReservationRequest]:
    return (ReservationRequest("ram", 1),)


@pytest.mark.parametrize(
    ("worker_kind", "use_shm"), [("isolated", False), ("isolated", True), ("remote", False)]
)
def test_control_round_trip_releases_admission_and_updates_only_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_kind: str, use_shm: bool
) -> None:
    from test_remote import TOKEN, start_service, stop_service

    vault = AssetVault(tmp_path / "engine-vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))

    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "control"\nnamespaces = ["control"]\n'
            '[pack.entry]\nnodes = "tests.test_run_control_boundary:PACK_NODES"\n'
            'reservations = "tests.test_run_control_boundary:reserve"\n',
            encoding="ascii",
        )
        root = Path(__file__).resolve().parent.parent
        registry = TypeRegistry()
        register_core_types(registry)
        reservations = GovernorReservationService(MemoryGovernor({"ram": 1}), timeout=2)
        service: asyncio.subprocess.Process | None = None
        worker: IsolatedWorker | RemoteWorker
        if worker_kind == "remote":
            service, host, port = await start_service(
                manifest,
                tmp_path,
                "--asset-vault",
                str(tmp_path / "producer"),
                "--value-store",
                str(tmp_path / "daemon-store"),
                pythonpath=(root,),
            )
            worker = RemoteWorker(
                host,
                port,
                TOKEN,
                registry,
                name="control",
                reservations=reservations,
                value_store=BudgetedDiskCAS(tmp_path / "transfer"),
            )
        else:
            worker = IsolatedWorker(
                manifest,
                registry,
                use_shm=use_shm,
                shm_threshold=1,
                reservations=reservations,
                extra_env={
                    "PYTHONPATH": str(root),
                    "DINKSTER_PACK_SCRATCH": str(tmp_path / "producer"),
                },
            )
        try:
            await worker.start()
            started = asyncio.Event()
            engine = Engine(
                schemas=build_schemas(PACK_NODES),
                worker=worker,
                cache=MemoryLRUCache(),
                registry=registry,
                max_concurrency=1,
                on_event=lambda event: started.set() if event.kind == "node_event" else None,
            )
            waiting = asyncio.create_task(
                engine.run(
                    Graph({"waiter": GraphNode("control.signal_wait")}),
                    ["waiter"],
                )
            )
            await asyncio.wait_for(started.wait(), 10)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(waiting, 10)
            for seed in (19, -43):
                result = await asyncio.wait_for(
                    engine.run(
                        Graph(
                            {
                                "waiter": GraphNode("control.wait"),
                                "publisher": GraphNode("control.publish", {"value": seed}),
                            }
                        ),
                        ["waiter", "publisher"],
                    ),
                    10,
                )
                assert result.outputs["waiter"]["value"].resolve() == seed
            result = await asyncio.wait_for(
                engine.run(
                    Graph(
                        {
                            "waiter": GraphNode("control.wait"),
                            "publisher": GraphNode("control.source_publish", {"value": 37}),
                        }
                    ),
                    ["waiter", "publisher"],
                ),
                10,
            )
            assert result.outputs["waiter"]["value"].resolve() == 37
            source_data = b"published-source-37"
            assert vault.has(digest_bytes(source_data))
            graph = Graph(
                {
                    "controller": GraphNode("control.edit"),
                    "source": GraphNode("control.pass", {"value": 45}),
                    "victim": GraphNode("control.pass", {"value": Link("controller", "value")}),
                    "muted": GraphNode("control.pass", {"value": Link("controller", "value")}),
                }
            )
            result = await asyncio.wait_for(engine.run(graph, ["victim", "muted"]), 10)
            assert result.outputs["victim"]["value"].resolve() == 45
            assert is_absent(result.outputs["muted"]["value"])
            assert graph.nodes["victim"].inputs["value"] == Link("controller", "value")
        finally:
            try:
                await worker.close()
            finally:
                if service is not None:
                    await stop_service(service)
        retained_source = vault.resolve(digest_bytes(source_data))
        assert retained_source is not None
        assert retained_source.read_bytes() == source_data

    asyncio.run(scenario())
