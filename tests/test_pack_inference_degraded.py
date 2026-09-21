"""Packs with sampler entries compose when no native sampling worker is live.

The acceptance surface for issue Kosinkadink/comfy-vibe-station#149: a pack that
ships samplers keeps its nodes, routes and events on a host without the native
inference pack, reports its inference surface unavailable with a reason, and
still refuses to run a graph that selects one of its samplers.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_graph import Graph, GraphNode
from dinkster_server import PackInfo, create_app

from dinkster.compose import PackSpec, ServingComposer

FIXTURES = Path(__file__).parent
ATTENTION_PROVIDER = FIXTURES / "fixtures/attention_provider"
DEGRADED_PACK = FIXTURES / "fixtures/degraded-sampler-pack"
DEGRADED_PACK_ID = "dinkster-degraded-sampler-fixture"
SOLVER_ID = "degraded.fast_solver"
SCHEDULE_ID = "degraded.stepped_schedule"
PROBE_TYPE = "degraded.solver_probe"


def _worker_env() -> dict[str, str]:
    return {
        "PYTHONPATH": os.pathsep.join((str(ATTENTION_PROVIDER), str(FIXTURES), str(DEGRADED_PACK)))
    }


def _host_manifest(root: Path) -> Path:
    """Native sampling host: the arm that materializes every inference surface."""
    root.mkdir(parents=True)
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "sampling-host"\n'
        'namespaces = ["dinkster", "comfy"]\n\n'
        '[pack.arms]\nnative = ["dinkster.ksampler"]\n\n'
        '[pack.entry]\nnodes = "s1_sampler_host:NODES"\n'
        'arm_nodes = "s1_sampler_host:ARM_NODES"\n'
        'choices = "s1_sampler_host:choices"\n',
        encoding="utf-8",
    )
    return manifest


def _degraded_spec() -> PackSpec:
    return PackSpec(
        DEGRADED_PACK / "dinkster-pack.toml",
        packs={DEGRADED_PACK_ID: PackInfo(display_name="Degraded Sampler", version="1.0.0")},
    )


async def _wire(app: object, path: str) -> dict[str, object]:
    async with TestClient(TestServer(app)) as client:  # type: ignore[arg-type]
        response = await client.get(path)
        assert response.status == 200
        return await response.json()


def test_samplers_compose_without_worker_and_keep_nodes_routes_and_events() -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env())
        try:
            delta = await composer.add_pack(_degraded_spec())
            assert PROBE_TYPE in delta.schemas
            pack = composer.composition.packs[DEGRADED_PACK_ID]
            assert pack.inference_unavailable is not None
            detail = pack.inference_unavailable
            assert "native dinkster.ksampler worker" in detail.reason
            assert detail.entry == "degraded_sampler_pack:register_inference"
            assert detail.worker == "dinkster.ksampler"
            assert {(item.registry, item.id) for item in detail.providers} == {
                ("dinkster.samplers", SOLVER_ID),
                ("dinkster.schedulers", SCHEDULE_ID),
            }
            assert all(item.to_wire()["available"] is False for item in detail.providers)

            # Nodes and routes answer while the inference surface is degraded.
            sampled = await composer.composition.make_engine(lambda _event: None).run(
                Graph(nodes={"probe": GraphNode(PROBE_TYPE, {"solver": "plain"})}), ["probe"]
            )
            assert sampled.outputs["probe"]["solver"].resolve() == "plain"

            runtime = composer._runtime_seat.pin()
            extension = next(
                item
                for item in runtime.extension_snapshot.extensions
                if item.id == DEGRADED_PACK_ID
            )
            answered = await composer.call_pack_route(
                DEGRADED_PACK_ID,
                extension.routes[0],
                {},
                runtime.extension_snapshot_digest,
            )
            assert answered == {"message": "Degraded sampler pack route ready"}
            assert [event.name for event in extension.events] == ["degraded.sampler.executed"]
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_degraded_pack_reports_on_pack_table_and_diagnostics() -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env())
        try:
            delta = await composer.add_pack(_degraded_spec())
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
            )
            assert delta.schemas  # the pack's node type composed on the live surface

            nodes = await _wire(app, "/api/nodes")
            packs = nodes["packs"]
            assert isinstance(packs, dict)
            row = packs[DEGRADED_PACK_ID]
            assert isinstance(row, dict)
            assert PROBE_TYPE in nodes["nodes"]
            wire = row["inferenceUnavailable"]
            assert isinstance(wire, dict)
            assert "native dinkster.ksampler worker" in str(wire["reason"])
            assert wire["worker"] == "dinkster.ksampler"
            assert {(item["registry"], item["id"]) for item in wire["providers"]} == {
                ("dinkster.samplers", SOLVER_ID),
                ("dinkster.schedulers", SCHEDULE_ID),
            }
            assert all(item["available"] is False for item in wire["providers"])

            diagnostics = await _wire(app, "/api/diagnostics")
            degraded = diagnostics["packInferenceUnavailable"]
            assert isinstance(degraded, list) and len(degraded) == 1
            entry = degraded[0]
            assert isinstance(entry, dict)
            assert entry["packId"] == DEGRADED_PACK_ID
            assert "native dinkster.ksampler worker" in str(entry["reason"])
            # One record per pack, never one per declared sampler.
            healthy = packs["core"]
            assert isinstance(healthy, dict)
            assert "inferenceUnavailable" not in healthy
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_degraded_samplers_stay_out_of_registries_and_choices() -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env())
        try:
            await composer.add_pack(_degraded_spec())
            runtime = composer._runtime_seat.pin()
            ids = {sampler.id for sampler in runtime.sampler_registry_snapshot.samplers}
            assert SOLVER_ID not in ids
            for choice_id in ("dinkster.samplers", "comfy.samplers"):
                assert SOLVER_ID not in composer.composition.choices.get(choice_id, ())
            for choice_id in ("dinkster.schedulers", "comfy.schedulers"):
                assert SCHEDULE_ID not in composer.composition.choices.get(choice_id, ())
            extension = next(
                item
                for item in runtime.extension_snapshot.extensions
                if item.id == DEGRADED_PACK_ID
            )
            assert not [
                item for item in extension.contribution_ids if item.startswith("inference:")
            ]
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_graph_selecting_a_degraded_sampler_fails_with_the_recorded_reason() -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env())
        try:
            await composer.add_pack(_degraded_spec())
            detail = composer.composition.packs[DEGRADED_PACK_ID].inference_unavailable
            assert detail is not None
            # The engine wraps a planning refusal in its node error; the reason
            # the user is told is the one recorded when the surface degraded.
            with pytest.raises(Exception, match="inference surface is unavailable") as raised:
                await composer.composition.make_engine(lambda _event: None).run(
                    Graph(nodes={"probe": GraphNode(PROBE_TYPE, {"solver": SOLVER_ID})}),
                    ["probe"],
                )
            assert detail.reason in str(raised.value)
            # The same pack still plans a graph that names no degraded sampler.
            planned = await composer.composition.make_engine(lambda _event: None).run(
                Graph(nodes={"probe": GraphNode(PROBE_TYPE, {"solver": "other"})}), ["probe"]
            )
            assert planned.outputs["probe"]["solver"].resolve() == "other"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_with_native_worker_the_same_pack_composes_its_samplers(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env())
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            await composer.add_pack(_degraded_spec())
            pack = composer.composition.packs[DEGRADED_PACK_ID]
            assert pack.inference_unavailable is None
            runtime = composer._runtime_seat.pin()
            ids = {sampler.id for sampler in runtime.sampler_registry_snapshot.samplers}
            assert SOLVER_ID in ids
            extension = next(
                item
                for item in runtime.extension_snapshot.extensions
                if item.id == DEGRADED_PACK_ID
            )
            assert [item for item in extension.contribution_ids if item.startswith("inference:")]
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_adding_the_native_worker_later_clears_the_degradation(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env())
        try:
            await composer.add_pack(_degraded_spec())
            assert composer.composition.packs[DEGRADED_PACK_ID].inference_unavailable is not None

            host_delta = await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            # The healed row rides the host's own announcement: the degraded pack
            # is not the pack being added, so nothing else would re-announce it.
            assert host_delta.packs[DEGRADED_PACK_ID].inference_unavailable is None
            assert composer.composition.packs[DEGRADED_PACK_ID].inference_unavailable is None
            ids = {
                sampler.id
                for sampler in composer._runtime_seat.pin().sampler_registry_snapshot.samplers
            }
            assert SOLVER_ID in ids
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_removing_the_native_worker_degrades_the_remaining_pack(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env())
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            await composer.add_pack(_degraded_spec())
            removed = await composer.remove_pack("sampling-host")
            detail = removed.packs[DEGRADED_PACK_ID].inference_unavailable
            assert detail is not None
            assert "native dinkster.ksampler worker" in detail.reason
            assert {(item.registry, item.id) for item in detail.providers} == {
                ("dinkster.samplers", SOLVER_ID),
                ("dinkster.schedulers", SCHEDULE_ID),
            }
            assert composer.composition.packs[DEGRADED_PACK_ID].inference_unavailable is not None
        finally:
            await composer.close()

    asyncio.run(scenario())
