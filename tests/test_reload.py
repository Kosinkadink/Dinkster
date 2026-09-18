"""Pack hot reload (DESIGN 3.9): restart one isolated worker, swap its
slice of the live surface.

What this proves: a reload is a fresh import in a fresh process (code
changes on disk actually land - the thing in-process module surgery could
never guarantee), the swap is atomic at every layer (routes, engine
surface, served state - one epoch bump, one schema_changed), a FAILED
reload leaves the old worker serving and reaps the new process, and runs
in flight finish against the schema mapping they entered with.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError, GraphValidationError, Worker
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import STATE_KEY, create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import GroupIsolatedWorker, InProcessWorker, IsolatedWorker, RoutingWorker

from dinkster.compose import CompositionError, PackSpec, ServingComposer, UnknownPackError
from dinkster.reload_api import add_reload_routes, apply_reload, apply_remove

STRING = TypeExpr.concrete("core.string")

MODULE_TEMPLATE = '''\
"""Reload test pack, generation {gen!r}: rewritten on disk mid-test."""

from collections.abc import Mapping

from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import TypeRegistry

STRING = TypeExpr.concrete("core.string")


def register_types(registry: TypeRegistry) -> None:
    pass


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="rl.echo",
            display_name="Reload Echo",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text + "-" + {gen!r})


class Extra(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type={extra_type!r},
            display_name="Reload Extra",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text)


NODES = [Echo, Extra]
'''


def write_pack(
    directory: Path,
    gen: str,
    *,
    name: str = "rlpack",
    extra_type: str = "rl.legacy",
    display_name: str | None = None,
    schema_only: bool = False,
) -> Path:
    """Write (or rewrite) the reload test pack: rl.echo whose output names
    the generation, plus one extra node whose type the generation controls
    (v1 ships rl.legacy, v2 drops it for rl.extra)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rlpack_nodes.py").write_text(
        MODULE_TEMPLATE.format(gen=gen, extra_type=extra_type)
    )
    manifest = directory / "dinkster-pack.toml"
    schema_only_entry = f'schema-only = ["rl.echo", "{extra_type}"]\n' if schema_only else ""
    manifest.write_text(
        f'[pack]\nname = "{name}"\nnamespaces = ["rl"]\n{schema_only_entry}\n'
        '[pack.entry]\nnodes = "rlpack_nodes:NODES"\n'
        'types = "rlpack_nodes:register_types"\n\n'
        "[pack.presentation]\n"
        f'display_name = "{display_name or f"Reload Pack {gen}"}"\n'
    )
    return manifest


def write_group_pack(
    directory: Path,
    name: str,
    generation: str,
    *,
    schema_only: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    module = f"{name}_nodes"
    (directory / f"{module}.py").write_text(
        "import os\n"
        "from collections.abc import Mapping\n"
        "from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr\n"
        "STRING = TypeExpr.concrete('core.string')\n"
        "class Echo(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls) -> NodeSchema:\n"
        f"        return NodeSchema(node_type='{name}.echo', display_name='Echo', "
        "inputs=(InputSpec('text', STRING),), outputs=(OutputSpec('out', STRING),))\n"
        "    @classmethod\n"
        "    async def execute(cls, *, text: str) -> Mapping[str, object]:\n"
        f"        return cls.outputs(out=text + '-{generation}')\n"
        "class Exit(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls) -> NodeSchema:\n"
        f"        return NodeSchema(node_type='{name}.exit', display_name='Exit', "
        "inputs=(), outputs=())\n"
        "    @classmethod\n"
        "    async def execute(cls) -> Mapping[str, object]:\n"
        "        os._exit(7)\n"
        "NODES = [Echo, Exit]\n"
    )
    manifest = directory / "dinkster-pack.toml"
    schema_only_entry = f'schema-only = ["{name}.echo", "{name}.exit"]\n' if schema_only else ""
    manifest.write_text(
        f'[pack]\nname = "{name}"\nnamespaces = ["{name}"]\n{schema_only_entry}\n'
        f'[pack.entry]\nnodes = "{module}:NODES"\n'
    )
    return manifest


def worker_env(directory: Path) -> dict[str, str]:
    return {"PYTHONPATH": str(directory)}


def echo_graph(text: str = "hi") -> Graph:
    return Graph(nodes={"e": GraphNode("rl.echo", {"text": text})})


async def run_echo(engine: Engine, text: str = "hi") -> str:
    result = await engine.run(echo_graph(text), ["e"])
    out = result.outputs["e"]["out"].resolve()
    assert isinstance(out, str)
    return out


# -- composer level ------------------------------------------------------------


def test_reload_swaps_code_types_and_table(tmp_path: Path) -> None:
    """The core promise: rewrite the pack on disk, reload, and the LIVE
    engine executes the new code - new worker process, redefined type,
    dropped type gone, added type routed, pack-table entry updated. The
    exact handler sequence: reload_pack, then engine.replace_schemas with
    the result's arguments."""

    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        try:
            delta = await composer.add_pack(manifest)
            # An engine made after add_pack snapshots the composed surface.
            engine = composer.composition.make_engine(lambda event: None)
            assert await run_echo(engine) == "hi-v1"
            assert "rl.legacy" in delta.schemas

            write_pack(tmp_path, "v2", extra_type="rl.extra")
            result = await composer.reload_pack("rlpack")
            # Uniform remove-then-add: every old type removed, survivors
            # re-announced, drops derivable as the difference.
            assert set(result.removed_types) == {"rl.echo", "rl.legacy"}
            assert set(result.delta.schemas) == {"rl.echo", "rl.extra"}
            assert result.removed_packs == ("rlpack",)
            assert result.delta.packs["rlpack"].display_name == "Reload Pack v2"

            engine.replace_schemas(result.removed_types, result.delta.schemas)
            # Fresh input (the v1 result is still cached under its old
            # key - hazard H4; the endpoint clears the cache, the raw
            # composer does not): fresh process, new code.
            assert await run_echo(engine, "two") == "two-v2"
            extra = await engine.run(
                Graph(nodes={"x": GraphNode("rl.extra", {"text": "ok"})}), ["x"]
            )
            assert extra.outputs["x"]["out"].resolve() == "ok"

            # The composer's registries swapped consistently: the dropped
            # type is gone, the composition serves the new surface, and a
            # second reload of the same pack works (records re-keyed).
            composition = composer.composition
            assert "rl.legacy" not in composition.schemas
            assert "rl.extra" in composition.schemas
            assert composition.node_packs["rl.extra"] == "rlpack"
            result = await composer.reload_pack("rlpack")
            assert set(result.delta.schemas) == {"rl.echo", "rl.extra"}
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_failed_reload_keeps_old_worker_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start-new-first: a reload whose new worker fails validation (node
    type outside the declared namespaces) raises, the OLD worker keeps
    executing old code, every registry is untouched, and the new process
    is reaped immediately - a dev session may retry reload many times and
    must not leak one live process per attempt."""

    started: list[object] = []
    closed: list[object] = []

    class Recording(IsolatedWorker):
        async def start(self) -> None:
            await super().start()
            started.append(self)

        async def close(self) -> None:
            closed.append(self)
            await super().close()

    monkeypatch.setattr("dinkster.compose.IsolatedWorker", Recording)

    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        try:
            await composer.add_pack(manifest)
            engine = composer.composition.make_engine(lambda event: None)

            write_pack(tmp_path, "v2", extra_type="outside.claims")
            before_schemas = dict(composer.composition.schemas)
            with pytest.raises(CompositionError, match="outside the pack"):
                await composer.reload_pack("rlpack")
            # The new worker started (its hello is how schemas arrive) and
            # was closed on the failure path; the old worker was NOT.
            assert len(started) == 2
            assert closed == [started[1]]
            assert composer.composition.schemas == before_schemas
            # Fresh inputs per probe so every run truly executes (same
            # input would cache-hit): the old process still serves.
            assert await run_echo(engine, "still") == "still-v1"

            # And the pack is still reloadable once the code is fixed.
            write_pack(tmp_path, "v3", extra_type="rl.extra")
            result = await composer.reload_pack("rlpack")
            engine.replace_schemas(result.removed_types, result.delta.schemas)
            assert await run_echo(engine, "fixed") == "fixed-v3"
        finally:
            await composer.close()
        assert set(closed) == set(started)  # close() reaps the survivor too

    asyncio.run(scenario())


def test_group_member_reload_swaps_every_route_transactionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def invoke(engine: Engine, node_type: str, text: str) -> str:
        result = await engine.run(Graph(nodes={"n": GraphNode(node_type, {"text": text})}), ["n"])
        value = result.outputs["n"]["out"].resolve()
        assert isinstance(value, str)
        return value

    async def scenario() -> None:
        alpha = write_group_pack(tmp_path / "alpha", "alpha", "v1")
        beta = write_group_pack(tmp_path / "beta", "beta", "v1")
        manifests = (alpha, beta)
        env = {"PYTHONPATH": os.pathsep.join((str(alpha.parent), str(beta.parent)))}
        specs = (
            PackSpec(alpha, env=env, worker_group="models", group_manifests=manifests),
            PackSpec(beta, env=env, worker_group="models", group_manifests=manifests),
        )
        scratch_root = tmp_path / "library" / "scratch"
        expected_scratch = str(scratch_root / "groups" / "models")
        composer = ServingComposer(pack_scratch_root=scratch_root)
        try:
            for item in specs:
                await composer.add_pack(item)
            engine = composer.composition.make_engine(lambda event: None)
            assert await invoke(engine, "alpha.echo", "a") == "a-v1"
            assert await invoke(engine, "beta.echo", "b") == "b-v1"
            old_owner = composer._group_owners["models"]
            assert old_owner._extra_env["DINKSTER_PACK_SCRATCH"] == expected_scratch

            write_group_pack(tmp_path / "alpha", "alpha", "version-two")
            result = await composer.reload_pack("alpha")
            new_owner = composer._group_owners["models"]
            assert result.reloaded_packs == ("alpha", "beta")
            assert set(result.delta.schemas) == {
                "alpha.echo",
                "alpha.exit",
                "beta.echo",
                "beta.exit",
            }
            assert new_owner is not old_owner
            assert new_owner._extra_env["DINKSTER_PACK_SCRATCH"] == expected_scratch
            assert composer._records["alpha"].worker._group is new_owner
            assert composer._records["beta"].worker._group is new_owner
            assert await invoke(engine, "alpha.echo", "new-a") == "new-a-version-two"
            assert await invoke(engine, "beta.echo", "new-b") == "new-b-v1"

            class FailingGroup(GroupIsolatedWorker):
                async def start(self) -> None:
                    raise RuntimeError("replacement refused")

            monkeypatch.setattr("dinkster.compose.GroupIsolatedWorker", FailingGroup)
            with pytest.raises(RuntimeError, match="replacement refused"):
                await composer.reload_pack("beta")
            assert composer._group_owners["models"] is new_owner
            assert await invoke(engine, "alpha.echo", "still-a") == "still-a-version-two"
            assert await invoke(engine, "beta.echo", "still-b") == "still-b-v1"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_group_reload_preserves_unrouted_schema_only_members(tmp_path: Path) -> None:
    async def scenario() -> None:
        alpha = write_group_pack(tmp_path / "alpha", "alpha", "v1", schema_only=True)
        beta = write_group_pack(tmp_path / "beta", "beta", "v1")
        manifests = (alpha, beta)
        env = {"PYTHONPATH": os.pathsep.join((str(alpha.parent), str(beta.parent)))}
        specs = (
            PackSpec(alpha, env=env, worker_group="models", group_manifests=manifests),
            PackSpec(beta, env=env, worker_group="models", group_manifests=manifests),
        )
        composer = ServingComposer()
        try:
            for spec in specs:
                await composer.add_pack(spec)
            assert not composer._routing.has_route("alpha.echo")
            assert composer._routing.has_route("beta.echo")

            write_group_pack(tmp_path / "beta", "beta", "v2")
            result = await composer.reload_pack("beta")
            assert set(result.removed_types) == {"beta.echo", "beta.exit"}
            assert set(result.delta.schemas) == {"beta.echo", "beta.exit"}
            assert not composer._routing.has_route("alpha.echo")
            assert composer._routing.has_route("beta.echo")
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_group_process_death_fails_every_member_with_group_attribution(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        alpha = write_group_pack(tmp_path / "alpha", "alpha", "v1")
        beta = write_group_pack(tmp_path / "beta", "beta", "v1")
        manifests = (alpha, beta)
        env = {"PYTHONPATH": os.pathsep.join((str(alpha.parent), str(beta.parent)))}
        composer = ServingComposer()
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest,
                        env=env,
                        worker_group="models",
                        group_manifests=manifests,
                    )
                )
            engine = composer.composition.make_engine(lambda event: None)
            with pytest.raises(ExecutionError) as crashed:
                await engine.run(Graph(nodes={"n": GraphNode("alpha.exit", {})}), ["n"])
            assert crashed.value.error.message == (
                "pack 'alpha' is not running (worker group 'models' exited with code 7)"
            )
            with pytest.raises(ExecutionError) as sibling:
                await engine.run(Graph(nodes={"n": GraphNode("beta.echo", {"text": "x"})}), ["n"])
            assert sibling.value.error.message == (
                "pack 'beta' is not running (worker group 'models' exited with code 7)"
            )
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_group_worker_start_is_all_or_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        alpha = write_group_pack(tmp_path / "alpha", "alpha", "v1")
        beta = write_group_pack(tmp_path / "beta", "beta", "v1")
        env = {"PYTHONPATH": os.pathsep.join((str(alpha.parent), str(beta.parent)))}
        registry = TypeRegistry()
        register_core_types(registry)
        owner = GroupIsolatedWorker(
            "models", (alpha, beta), registry, extra_env=env, start_timeout=5.0
        )

        begin = owner._sessions[1].begin

        async def refuse_begin(*args: object, **kwargs: object) -> None:
            await begin(*args, **kwargs)  # type: ignore[arg-type]
            raise RuntimeError("hello refused")

        monkeypatch.setattr(owner._sessions[1], "begin", refuse_begin)
        with pytest.raises(RuntimeError, match="hello refused"):
            await owner.start()
        assert owner._proc is not None and owner._proc.returncode is not None
        assert not any(member.alive for member in owner.members.values())

    asyncio.run(scenario())


def test_group_worker_authenticates_every_tcp_endpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        alpha = write_group_pack(tmp_path / "alpha", "alpha", "v1")
        beta = write_group_pack(tmp_path / "beta", "beta", "v1")
        env = {"PYTHONPATH": os.pathsep.join((str(alpha.parent), str(beta.parent)))}
        registry = TypeRegistry()
        register_core_types(registry)
        owner = GroupIsolatedWorker(
            "models",
            (alpha, beta),
            registry,
            extra_env=env,
            transport="tcp",
            start_timeout=5.0,
        )
        try:
            await owner.start()
            assert all(member.alive for member in owner.members.values())
        finally:
            await owner.close()

    asyncio.run(scenario())


def test_grouped_placement_preserves_surface_and_invocation_identity(
    tmp_path: Path,
) -> None:
    async def capture(composer: ServingComposer) -> tuple[object, ...]:
        composition = composer.composition
        engine = composition.make_engine(lambda event: None)
        outputs: list[str] = []
        for node_type in ("alpha.echo", "beta.echo"):
            result = await engine.run(
                Graph(nodes={"n": GraphNode(node_type, {"text": "same"})}), ["n"]
            )
            value = result.outputs["n"]["out"].resolve()
            assert isinstance(value, str)
            outputs.append(value)
        return (
            dict(composition.schemas),
            dict(composition.packs),
            dict(composition.node_packs),
            tuple(outputs),
        )

    async def scenario() -> None:
        alpha = write_group_pack(tmp_path / "alpha", "alpha", "v1")
        beta = write_group_pack(tmp_path / "beta", "beta", "v1")
        manifests = (alpha, beta)
        env = {"PYTHONPATH": os.pathsep.join((str(alpha.parent), str(beta.parent)))}
        solo = ServingComposer()
        try:
            await solo.add_pack(PackSpec(alpha, env=env))
            await solo.add_pack(PackSpec(beta, env=env))
            solo_surface = await capture(solo)
        finally:
            await solo.close()
        grouped = ServingComposer()
        try:
            for manifest in manifests:
                await grouped.add_pack(
                    PackSpec(
                        manifest,
                        env=env,
                        worker_group="models",
                        group_manifests=manifests,
                    )
                )
            assert grouped._records["alpha"].domain is grouped._records["beta"].domain
            assert len(grouped._live_token_owners(grouped._topology)) == 1
            assert await capture(grouped) == solo_surface
        finally:
            await grouped.close()

    asyncio.run(scenario())


def test_pack_spec_aimdo_reaches_add_and_reload_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster.compose import PackSpec

    modes: list[object] = []

    class Recording(IsolatedWorker):
        def __init__(self, *args: object, **kwargs: object) -> None:
            modes.append(kwargs.get("aimdo_arm"))
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("dinkster.compose.IsolatedWorker", Recording)

    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        try:
            await composer.add_pack(PackSpec(manifest, aimdo="on"))
            await composer.reload_pack("rlpack")
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert modes == ["on", "on"]


@pytest.mark.parametrize("single_job", [False, True])
def test_reload_preserves_multi_gpu_worker_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_job: bool,
) -> None:
    from dinkster.compose import PackSpec, _ReplicaWorkerPool, _SingleJobWorkerPool

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        spec = PackSpec(
            manifest,
            replica_cuda_indices=() if single_job else (0, 1),
            single_job_cuda_indices=(0, 1) if single_job else (),
        )
        expected = _SingleJobWorkerPool if single_job else _ReplicaWorkerPool
        scratch_root = tmp_path / "library" / "scratch"
        expected_scratch = str(scratch_root / "packs" / "rlpack")
        composer = ServingComposer(
            worker_env=worker_env(tmp_path),
            pack_scratch_root=scratch_root,
        )
        try:
            await composer.add_pack(spec)
            original = composer._records["rlpack"].worker
            assert isinstance(original, expected)
            assert len(original.lanes) == 2
            assert {lane.worker._extra_env["DINKSTER_PACK_SCRATCH"] for lane in original.lanes} == {
                expected_scratch
            }
            await composer.reload_pack("rlpack")
            replacement = composer._records["rlpack"].worker
            assert isinstance(replacement, expected)
            assert replacement is not original
            assert len(replacement.lanes) == 2
            assert {
                lane.worker._extra_env["DINKSTER_PACK_SCRATCH"] for lane in replacement.lanes
            } == {expected_scratch}
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_reload_refreshes_restart_scoped_runtime_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster.compose import PackSpec

    launches: list[tuple[object, object, object, object]] = []

    class Recording(IsolatedWorker):
        def __init__(self, *args: object, **kwargs: object) -> None:
            launches.append(
                (
                    kwargs.get("aimdo_arm"),
                    kwargs.get("reserve_vram"),
                    kwargs.get("vram_budgets"),
                    kwargs.get("comfy_args"),
                )
            )
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("dinkster.compose.IsolatedWorker", Recording)

    async def scenario() -> None:
        effective_aimdo = "auto"
        effective_reserve = 256 * 1024**2
        effective_budgets = {"vram:cuda:0": 20_000}
        effective_args: tuple[str, ...] = ()
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(
            worker_env=worker_env(tmp_path),
            runtime_worker_settings=lambda: (
                effective_aimdo,
                effective_reserve,
                effective_budgets,
                effective_args,
            ),
        )
        try:
            await composer.add_pack(
                PackSpec(
                    manifest,
                    aimdo="auto",
                    reserve_vram=256 * 1024**2,
                    vram_budgets={"vram:cuda:0": 20_000},
                    runtime_settings=True,
                )
            )
            assert composer.residency_memory_budgets() == {"rlpack": {"vram:cuda:0": 20_000}}
            effective_aimdo = "off"
            effective_reserve = 128 * 1024**2
            effective_budgets = {
                "ram": 100_000,
                "vram:cuda:0": 15_000,
                "vram:cuda:0@remote": 10_000,
            }
            effective_args = ("--preview-size", "256")
            await composer.reload_pack("rlpack")
            assert composer.residency_memory_budgets() == {"rlpack": {"vram:cuda:0": 15_000}}
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert launches == [
        ("auto", 256 * 1024**2, {"vram:cuda:0": 20_000}, ()),
        (
            "off",
            128 * 1024**2,
            {"vram:cuda:0": 15_000},
            ("--preview-size", "256"),
        ),
    ]


def test_reload_refuses_rename_and_unknown_pack(tmp_path: Path) -> None:
    """Reload swaps a known identity: a manifest rename is remove-and-add,
    not a reload, and an unknown pack name is its own error (the endpoint
    maps it to 404 vs the 409 every other failure gets)."""

    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        try:
            await composer.add_pack(manifest)
            with pytest.raises(UnknownPackError, match="no composed pack"):
                await composer.reload_pack("nope")
            write_pack(tmp_path, "v2", name="renamed")
            with pytest.raises(CompositionError, match="rename"):
                await composer.reload_pack("rlpack")
        finally:
            await composer.close()

    asyncio.run(scenario())


# -- engine level --------------------------------------------------------------


class Gate(Node):
    """Holds mid-run on a class-level gate so the test can reload under it."""

    gate: asyncio.Event
    entered: asyncio.Event

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.gate",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
            idempotent=False,
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        Gate.entered.set()
        await Gate.gate.wait()
        return cls.outputs(out=text)


class Tail(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.tail",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text.upper())


def test_engine_replace_schemas_validation() -> None:
    """replace_schemas mirrors announce_schemas: unknown removals refused,
    redefinition allowed exactly for types removed in the same swap, and a
    refused swap changes nothing."""
    registry = TypeRegistry()
    register_core_types(registry)
    schemas = build_schemas([Gate, Tail])
    engine = Engine(
        schemas=schemas,
        registry=registry,
        worker=InProcessWorker(build_node_types([Gate, Tail]), registry),
        cache=MemoryLRUCache(),
    )
    with pytest.raises(ValueError, match="unknown node type"):
        engine.replace_schemas(["test.nope"], {})
    with pytest.raises(ValueError, match="not being removed"):
        engine.replace_schemas([], build_schemas([Tail]))
    # Swap: redefine test.gate, drop test.tail.
    engine.replace_schemas(["test.gate", "test.tail"], build_schemas([Gate]))
    with pytest.raises(ValueError, match="unknown node type"):
        engine.replace_schemas(["test.tail"], {})


def test_in_flight_run_finishes_against_pinned_schemas() -> None:
    """H10 under reload: a run pins the schema mapping it entered with, so
    a replace_schemas that drops a type mid-run cannot break the run's own
    lookups - it completes against the old surface while NEW runs already
    see (and are validated against) the new one."""

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker(build_node_types([Gate, Tail]), registry)
        engine = Engine(
            schemas=build_schemas([Gate, Tail]),
            registry=registry,
            worker=worker,
            cache=MemoryLRUCache(),
        )
        Gate.gate = asyncio.Event()
        Gate.entered = asyncio.Event()
        graph = Graph(
            nodes={
                "g": GraphNode("test.gate", {"text": "hi"}),
                "t": GraphNode("test.tail", {"text": Link("g", "out")}),
            }
        )
        task = asyncio.create_task(engine.run(graph, ["t"]))
        await Gate.entered.wait()
        # Mid-run: drop BOTH types from the live surface.
        engine.replace_schemas(["test.gate", "test.tail"], {})
        # New runs fail validation against the new surface...
        with pytest.raises(GraphValidationError):
            await engine.run(Graph(nodes={"t": GraphNode("test.tail", {"text": "x"})}), ["t"])
        # ...while the in-flight run finishes against its pinned mapping
        # (test.tail's post-gate lookups included).
        Gate.gate.set()
        result = await task
        assert result.outputs["t"]["out"].resolve() == "HI"
        assert engine._run_schemas == {}  # pins released

    asyncio.run(scenario())


def test_routing_swap_routes() -> None:
    """swap_routes is one reference swap with add_routes' refusals: removed
    types must be routed, added types may collide only with the removed."""
    registry = TypeRegistry()
    register_core_types(registry)
    a: Worker = InProcessWorker(build_node_types([Gate]), registry)
    b: Worker = InProcessWorker(build_node_types([Gate, Tail]), registry)
    routing = RoutingWorker({"test.gate": a})
    with pytest.raises(ValueError, match="unrouted"):
        routing.swap_routes(["test.tail"], {})
    with pytest.raises(ValueError, match="not being removed"):
        routing.swap_routes([], {"test.gate": b})
    routing.swap_routes(["test.gate"], {"test.gate": b, "test.tail": b})
    assert routing._worker_for("test.gate") is b
    assert routing._worker_for("test.tail") is b
    routing.swap_routes(["test.tail"], {})
    with pytest.raises(KeyError):
        routing._worker_for("test.tail")


# -- server level (the full endpoint sequence) ---------------------------------


def submit_body(graph: Graph, targets: list[str], **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "clientId": "c1",
        "jobId": "j1",
        "graph": graph_to_wire(graph),
        "targets": targets,
    }
    body.update(overrides)
    return body


async def poll_job(client: TestClient, job_id: str) -> dict[str, object]:
    async with asyncio.timeout(10):
        while True:
            status = await (await client.get(f"/api/jobs/c1/{job_id}")).json()
            if status["state"] in ("completed", "failed"):
                return status
            await asyncio.sleep(0.01)


def test_concurrent_reload_and_removal_publish_in_composer_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        try:
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
            )
            state = app[STATE_KEY]
            delta = await composer.add_pack(manifest)
            assert state.announce(delta.schemas, delta.packs, delta.node_packs) == 2
            write_pack(tmp_path, "v2", extra_type="rl.extra")

            original_prepare = state.prepare_replace
            validation_started = asyncio.Event()
            release_validation = asyncio.Event()
            calls = 0

            async def pause_first_validation(*args: Any, **kwargs: Any) -> Any:
                nonlocal calls
                calls += 1
                if calls == 1:
                    validation_started.set()
                    await release_validation.wait()
                return await original_prepare(*args, **kwargs)

            monkeypatch.setattr(state, "prepare_replace", pause_first_validation)
            reload_task = asyncio.create_task(apply_reload(state, composer, "rlpack"))
            await validation_started.wait()
            remove_task = asyncio.create_task(apply_remove(state, composer, "rlpack"))
            await asyncio.sleep(0)
            assert not remove_task.done()
            assert "rl.extra" in composer.composition.schemas
            assert "rl.extra" not in state.schemas

            release_validation.set()
            reload_result, remove_result = await asyncio.gather(reload_task, remove_task)
            assert reload_result["epoch"] == 3
            assert remove_result["epoch"] == 4
            assert "rlpack" not in composer.composition.packs
            assert "rlpack" not in state.packs
            assert "rl.extra" not in composer.composition.schemas
            assert "rl.extra" not in state.schemas
            assert state.schemas == composer.composition.schemas
            assert set(state.packs) == {*composer.composition.packs, "core"}
            assert state.node_packs == composer.composition.node_packs
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_reload_cancellation_finishes_committed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        try:
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
            )
            state = app[STATE_KEY]
            delta = await composer.add_pack(manifest)
            assert state.announce(delta.schemas, delta.packs, delta.node_packs) == 2
            write_pack(tmp_path, "v2", extra_type="rl.extra")

            original_prepare = state.prepare_replace
            validation_started = asyncio.Event()
            release_validation = asyncio.Event()

            async def pause_validation(*args: Any, **kwargs: Any) -> Any:
                validation_started.set()
                await release_validation.wait()
                return await original_prepare(*args, **kwargs)

            monkeypatch.setattr(state, "prepare_replace", pause_validation)
            reload_task = asyncio.create_task(apply_reload(state, composer, "rlpack"))
            await validation_started.wait()
            reload_task.cancel()
            await asyncio.sleep(0)
            assert not reload_task.done()
            reload_task.cancel()
            await asyncio.sleep(0)
            assert not reload_task.done()
            release_validation.set()
            with pytest.raises(asyncio.CancelledError):
                await reload_task

            assert state.schema_epoch == 3
            assert "rl.extra" in composer.composition.schemas
            assert "rl.extra" in state.schemas
            assert state.composition_packs["rlpack"] == {"state": "announced", "epoch": 3}
            assert state.schemas == composer.composition.schemas
            assert set(state.packs) == {*composer.composition.packs, "core"}
            assert state.node_packs == composer.composition.node_packs
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_reload_endpoint_swaps_live_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/packs/{packId}/reload end to end: one epoch bump, one
    schema_changed emitted only after /api/nodes serves the swapped
    surface, jobs execute the NEW code through the live queue, the result
    cache is cleared (unchanged-signature code changes must not
    stale-hit), the composition report tracks the new epoch, and an
    unknown pack is 404."""

    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        composition = composer.composition
        app = create_app(
            composition.make_engine,
            composition.schemas,
            packs=composition.packs,
            node_packs=composition.node_packs,
        )
        add_reload_routes(app, composer)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            delta = await composer.add_pack(manifest)
            assert state.announce(delta.schemas, delta.packs, delta.node_packs) == 2
            prepare_replace = AsyncMock(wraps=state.prepare_replace)
            monkeypatch.setattr(state, "prepare_replace", prepare_replace)

            # A completed job seeds the result cache with old-code outputs.
            body = submit_body(echo_graph(), ["e"], jobId="before")
            assert (await client.post("/api/jobs", json=body)).status == 202
            assert (await poll_job(client, "before"))["state"] == "completed"

            assert (await client.post("/api/packs/nope/reload")).status == 404

            write_pack(tmp_path, "v2", extra_type="rl.extra")
            ws = await client.ws_connect("/api/events?clientId=c1")
            response = await client.post("/api/packs/rlpack/reload")
            assert response.status == 200
            prepare_replace.assert_awaited_once()
            payload = await response.json()
            assert payload["pack"] == "rlpack"
            assert payload["epoch"] == 3
            assert payload["nodes"] == ["rl.echo", "rl.extra"]
            assert payload["removedNodes"] == ["rl.legacy"]
            assert payload["cacheCleared"] >= 1  # the seeded entry went

            async with asyncio.timeout(5):
                while True:
                    event = await ws.receive_json()
                    if event["type"] == "schema_changed":
                        break
            assert event == {"type": "schema_changed", "epoch": 3}
            data = await (await client.get("/api/nodes")).json()
            assert data["epoch"] == 3
            assert "rl.legacy" not in data["nodes"]
            assert data["nodes"]["rl.extra"]["pack"] == "rlpack"
            assert data["packs"]["rlpack"]["displayName"] == "Reload Pack v2"

            report = await (await client.get("/api/composition")).json()
            assert report["packs"]["rlpack"] == {"state": "announced", "epoch": 3}

            # The same graph now executes the NEW code (and the old cached
            # result cannot answer for it - the cache was cleared).
            body = submit_body(echo_graph(), ["e"], jobId="after")
            assert (await client.post("/api/jobs", json=body)).status == 202
            status = await poll_job(client, "after")
            assert status["state"] == "completed", status
            resp = await client.get("/api/values?clientId=c1&jobId=after&nodeId=e&outputId=out")
            assert resp.status == 200
            data = await resp.json()
            assert data["descriptor"]["value"] == "hi-v2"
            await ws.close()
        finally:
            await client.close()
            await composer.close()

    asyncio.run(scenario())


def test_reload_endpoint_publishes_only_routable_schema_owner_nodes(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1", schema_only=True)
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        composition = composer.composition
        app = create_app(
            composition.make_engine,
            composition.schemas,
            packs=composition.packs,
            node_packs=composition.node_packs,
        )
        add_reload_routes(app, composer)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            delta = await composer.add_pack(manifest)
            state.announce(
                delta.schemas,
                delta.packs,
                delta.node_packs,
                execution_arms=delta.execution_arms,
            )
            assert not state.schemas

            write_pack(tmp_path, "v2")
            response = await client.post("/api/packs/rlpack/reload")
            assert response.status == 200
            assert set(state.schemas) == {"rl.echo", "rl.legacy"}

            write_pack(tmp_path, "v3", schema_only=True)
            response = await client.post("/api/packs/rlpack/reload")
            assert response.status == 200
            assert not state.schemas
        finally:
            await client.close()
            await composer.close()

    asyncio.run(scenario())


def test_reload_endpoint_failure_is_409_and_surface_intact(tmp_path: Path) -> None:
    """A failed reload is 409 with the composition error, the epoch does
    not bump, no schema_changed fires, and the OLD pack keeps serving jobs
    - strictly better than a dead pack."""

    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        composition = composer.composition
        app = create_app(
            composition.make_engine,
            composition.schemas,
            packs=composition.packs,
            node_packs=composition.node_packs,
        )
        add_reload_routes(app, composer)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            delta = await composer.add_pack(manifest)
            state.announce(delta.schemas, delta.packs, delta.node_packs)

            write_pack(tmp_path, "v2", extra_type="outside.claims")
            response = await client.post("/api/packs/rlpack/reload")
            assert response.status == 409
            payload = await response.json()
            assert payload["error"] == "reload-failed"
            assert "outside the pack" in payload["detail"]
            # Failure bodies carry the traceback (redacted); absolute
            # filesystem paths never reach the wire.
            assert "traceback" in payload
            assert str(tmp_path) not in payload["detail"]
            assert str(tmp_path) not in payload["traceback"]

            data = await (await client.get("/api/nodes")).json()
            assert data["epoch"] == 2
            assert "rl.legacy" in data["nodes"]
            body = submit_body(echo_graph(), ["e"])
            assert (await client.post("/api/jobs", json=body)).status == 202
            assert (await poll_job(client, "j1"))["state"] == "completed"
        finally:
            await client.close()
            await composer.close()

    asyncio.run(scenario())


# -- removal (the other half of the seam) --------------------------------------


def test_remove_pack_retracts_and_allows_readd(tmp_path: Path) -> None:
    """Composer-level removal: every node type retracts, the worker
    stops, the registries rebuild from the survivors - so the SAME pack
    name can compose again afterwards (remove-and-add is the rename/
    update story reload refuses). Removal needs no manifest on disk."""

    async def scenario() -> None:
        write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        try:
            await composer.add_pack(tmp_path)
            composition = composer.composition

            with pytest.raises(UnknownPackError):
                await composer.remove_pack("nope")

            # Sources gone from disk: removal must still work.
            (tmp_path / "dinkster-pack.toml").unlink()
            result = await composer.remove_pack("rlpack")
            assert set(result.removed_types) == {"rl.echo", "rl.legacy"}
            assert result.removed_packs == ("rlpack",)
            assert "rl.echo" not in composition.schemas
            assert "rlpack" not in composition.packs
            assert "rl.echo" not in composition.node_packs
            assert composer.watch_targets() == {}

            with pytest.raises(UnknownPackError):
                await composer.remove_pack("rlpack")

            # The name and namespace claims are free again.
            write_pack(tmp_path, "v2", extra_type="rl.extra")
            delta = await composer.add_pack(tmp_path)
            assert set(delta.schemas) == {"rl.echo", "rl.extra"}
            engine = composition.make_engine(lambda event: None)
            assert await run_echo(engine) == "hi-v2"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_remove_endpoint_retracts_live_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DELETE /api/packs/{packId} end to end: one epoch bump, one
    schema_changed after /api/nodes serves the retracted surface, the
    pack row flips to "removed" on /api/composition, the result cache
    clears (remove-edit-recompose must not stale-hit), and an unknown
    pack is 404."""

    async def scenario() -> None:
        manifest = write_pack(tmp_path, "v1")
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        composition = composer.composition
        app = create_app(
            composition.make_engine,
            composition.schemas,
            packs=composition.packs,
            node_packs=composition.node_packs,
        )
        add_reload_routes(app, composer)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            delta = await composer.add_pack(manifest)
            assert state.announce(delta.schemas, delta.packs, delta.node_packs) == 2
            prepare_replace = AsyncMock(wraps=state.prepare_replace)
            monkeypatch.setattr(state, "prepare_replace", prepare_replace)

            # A completed job seeds the result cache.
            body = submit_body(echo_graph(), ["e"], jobId="before")
            assert (await client.post("/api/jobs", json=body)).status == 202
            assert (await poll_job(client, "before"))["state"] == "completed"

            assert (await client.delete("/api/packs/nope")).status == 404

            ws = await client.ws_connect("/api/events?clientId=c1")
            response = await client.delete("/api/packs/rlpack")
            assert response.status == 200
            prepare_replace.assert_awaited_once()
            payload = await response.json()
            assert payload == {
                "pack": "rlpack",
                "epoch": 3,
                "removedNodes": ["rl.echo", "rl.legacy"],
                "removedPacks": ["rlpack"],
                "cacheCleared": payload["cacheCleared"],
            }
            assert payload["cacheCleared"] >= 1

            async with asyncio.timeout(5):
                while True:
                    event = await ws.receive_json()
                    if event["type"] == "schema_changed":
                        break
            assert event == {"type": "schema_changed", "epoch": 3}
            data = await (await client.get("/api/nodes")).json()
            assert data["epoch"] == 3
            assert "rl.echo" not in data["nodes"]
            assert "rl.legacy" not in data["nodes"]
            assert "rlpack" not in data["packs"]

            report = await (await client.get("/api/composition")).json()
            assert report["packs"]["rlpack"] == {"state": "removed", "epoch": 3}

            # New submissions naming a retracted type fail loudly.
            body = submit_body(echo_graph(), ["e"], jobId="after")
            response = await client.post("/api/jobs", json=body)
            if response.status == 202:
                assert (await poll_job(client, "after"))["state"] == "failed"
            else:
                assert response.status == 400
            await ws.close()
        finally:
            await client.close()
            await composer.close()

    asyncio.run(scenario())
