"""Live install activation: reconcile the served surface with the install
root's current generation, no restart (DESIGN 3.9's production follow-up).

What this proves: the GET plan classifies managed packs (add / remove /
reload / unchanged) by diffing served specs against the current
generation and never counts dev --pack additions; POST stages the complete
generation and publishes it in one epoch; a generation pin from a stale plan
is refused; activation during startup composition is refused; any pack failure
rolls the complete staged generation back with ordered diagnostics.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import canonical_extension_snapshot, extension_behavior_hash
from dinkster_registry import Lockfile
from dinkster_server import STATE_KEY, create_app

import dinkster.installer as installer_module
from dinkster.activation import add_activation_routes, compute_plan
from dinkster.compose import ServingComposer, default_pack_specs
from dinkster.installer import Installer, lock_local_pack

MODULE_TEMPLATE = '''\
"""Activation test pack {name!r}, marker {marker!r}."""

from collections.abc import Mapping

from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr

STRING = TypeExpr.concrete("core.string")


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="{name}.echo",
            display_name="Echo {name}",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text + "-" + {marker!r})


NODES = [Echo]


def extension_registration():
    from dinkster_api.v1 import CompositionMode, ContributionSurfaceDescriptor

    return [
        ContributionSurfaceDescriptor(
            {surface!r}, CompositionMode.EXCLUSIVE
        )
    ]
'''


@pytest.fixture(autouse=True)
def _no_host_runtime_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic: the default runtime probe shells out to nvidia-smi."""
    monkeypatch.setattr(installer_module, "detect_runtime", lambda _accelerator: ())


def write_pack(
    directory: Path,
    name: str,
    marker: str = "v1",
    *,
    extension: bool = False,
    surface: str = "proof-exclusive",
) -> Path:
    """Write (or rewrite) one single-node pack; ``marker`` changes the
    module bytes, so a marker bump is a new artifact digest - which is
    exactly what makes activation classify the pack as a reload."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_nodes.py").write_text(
        MODULE_TEMPLATE.format(name=name, marker=marker, surface=surface)
    )
    extension_table = (
        "\n[pack.extension]\n"
        f'schema = "{name}_nodes:extension_registration"\n'
        'privileges = ["schema"]\n'
        'capabilities = ["routes"]\n'
        if extension
        else ""
    )
    (directory / "dinkster-pack.toml").write_text(
        f'[pack]\nname = "{name}"\nnamespaces = ["{name}"]\n\n'
        f'[pack.entry]\nnodes = "{name}_nodes:NODES"\n' + extension_table
    )
    return directory


def make_installer(tmp_path: Path) -> Installer:
    return Installer(tmp_path / "root")


def apply_generation(
    installer: Installer, *packs: Path, allow_doctor_findings: bool = False
) -> int:
    """Lock the given pack directories and make them the current
    generation - dinkster-pack's confirmed plan/apply, compressed. No venvs:
    activation tests run every worker on the host interpreter."""
    target = Lockfile.of([lock_local_pack(pack, installer.artifacts_dir)[0] for pack in packs])
    number, _ = installer.apply(
        target,
        venvs=False,
        allow_doctor_findings=allow_doctor_findings,
    )
    return number


def test_activation_plan_and_apply_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole story against one live server: plan classification,
    composing/stale-pin/bad-body refusals, apply-through-the-coordinators
    (epochs, composition rows, /api/nodes), dev-pack invisibility,
    idempotence, and whole-generation failure rollback."""

    async def scenario() -> None:
        installer = make_installer(tmp_path)
        alpha = write_pack(tmp_path / "alpha", "alpha")
        beta = write_pack(tmp_path / "beta", "beta")
        assert apply_generation(installer, alpha, beta) == 1

        # Workers here run on the host interpreter (no venvs staged), so
        # module imports resolve via PYTHONPATH over the stable source
        # dirs - whose bytes are exactly what each generation's store
        # holds at the moment it locks. Production packs instead import
        # through their per-pack venv (editable-installed at staging).
        pack_dirs = [tmp_path / name for name in ("alpha", "beta", "gamma", "dev", "broken")]
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join(str(d) for d in pack_dirs)}
        )
        mutation_lock = composer._mutate
        client: TestClient | None = None
        try:
            for spec in installer.packs_for_serving():
                await composer.add_pack(spec)
            # A dev --pack addition, outside the store: activation must
            # never classify or touch it.
            dev_dir = write_pack(tmp_path / "dev", "devpack")
            await composer.add_pack(dev_dir / "dinkster-pack.toml")

            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
            )
            add_activation_routes(app, composer, installer)
            client = TestClient(TestServer(app))
            await client.start_server()
            state = app[STATE_KEY]

            # Settled surface = empty plan; the dev pack appears nowhere.
            plan = await (await client.get("/api/install/activation")).json()
            assert plan == {
                "generation": 1,
                "add": [],
                "remove": [],
                "reload": [],
                "unchanged": ["alpha", "beta"],
            }

            # New generation: alpha's bytes change (new digest = reload),
            # beta leaves, gamma arrives.
            write_pack(tmp_path / "alpha", "alpha", marker="v2")
            gamma = write_pack(tmp_path / "gamma", "gamma")
            assert apply_generation(installer, tmp_path / "alpha", gamma) == 2
            plan = await (await client.get("/api/install/activation")).json()
            assert plan == {
                "generation": 2,
                "add": ["gamma"],
                "remove": ["beta"],
                "reload": ["alpha"],
                "unchanged": [],
            }
            # The GET is read-only: nothing moved.
            assert "beta.echo" in composer.composition.schemas

            # Refusals, each with its machine-readable key: mid-startup
            # composition, a stale generation pin, a malformed body.
            state.narrate_composition(0, 1)
            response = await client.post("/api/install/activation")
            assert response.status == 409
            assert (await response.json())["error"] == "composition-in-progress"
            state.complete_composition()
            response = await client.post("/api/install/activation", json={"generation": 1})
            assert response.status == 409
            assert (await response.json())["error"] == "generation-changed"
            response = await client.post("/api/install/activation", data=b"[1]")
            assert response.status == 400

            # Apply, pinned to the planned generation. The full surface lands
            # in one epoch, never as three independently visible changes.
            before = state.schema_epoch
            prepare_generation = AsyncMock(wraps=state.prepare_generation)
            monkeypatch.setattr(state, "prepare_generation", prepare_generation)
            response = await client.post("/api/install/activation", json={"generation": 2})
            assert response.status == 200
            prepare_generation.assert_awaited_once()
            payload = await response.json()
            assert payload["generation"] == 2
            assert payload["added"] == ["gamma"]
            assert payload["removed"] == ["beta"]
            assert payload["reloaded"] == ["alpha"]
            assert payload["unchanged"] == []
            assert payload["failed"] == {}
            assert payload["epoch"] == before + 1
            assert composer._mutate is mutation_lock

            data = await (await client.get("/api/nodes")).json()
            assert data["epoch"] == before + 1
            assert "beta.echo" not in data["nodes"]
            assert data["nodes"]["gamma.echo"]["pack"] == "gamma"
            assert data["nodes"]["devpack.echo"]["pack"] == "devpack"
            report = await (await client.get("/api/composition")).json()
            assert report["packs"]["beta"] == {
                "state": "removed",
                "epoch": before + 1,
            }
            assert report["packs"]["alpha"] == {
                "state": "announced",
                "epoch": before + 1,
            }
            assert report["packs"]["gamma"] == {
                "state": "announced",
                "epoch": before + 1,
            }

            # Idempotent: a second apply finds nothing to do.
            response = await client.post("/api/install/activation")
            payload = await response.json()
            assert payload["added"] == []
            assert payload["removed"] == []
            assert payload["reloaded"] == []
            assert payload["unchanged"] == ["alpha", "gamma"]
            assert state.schema_epoch == before + 1

            # Generation 3 adds a broken pack next to the two good ones. The
            # complete candidate rolls back; previous workers and surface stay.
            broken = tmp_path / "broken"
            broken.mkdir()
            (broken / "broken_nodes.py").write_text("raise RuntimeError('boom')\n")
            (broken / "dinkster-pack.toml").write_text(
                '[pack]\nname = "broken"\nnamespaces = ["broken"]\n\n'
                '[pack.entry]\nnodes = "broken_nodes:NODES"\n'
            )
            assert (
                apply_generation(
                    installer,
                    tmp_path / "alpha",
                    gamma,
                    broken,
                    allow_doctor_findings=True,
                )
                == 3
            )
            # The live dev worker survives even if its source disappears;
            # rebuilding the complete next generation reports it as missing.
            (dev_dir / "dinkster-pack.toml").unlink()
            response = await client.post("/api/install/activation")
            assert response.status == 200
            payload = await response.json()
            assert payload["added"] == []
            assert list(payload["failed"]) == ["broken", "devpack"]
            assert payload["unchanged"] == ["alpha", "gamma"]
            assert payload["rolledBack"] is True
            assert payload["diagnostics"][0]["pack"] == "broken"
            assert payload["diagnostics"][0]["kind"] == "incompatible-pack"
            assert "schema catalog is missing or stale" in payload["diagnostics"][0]["error"]
            assert payload["diagnostics"][1]["pack"] == "devpack"
            assert payload["diagnostics"][1]["kind"] == "missing-pack"
            assert payload["rollbackEvidence"]["epoch"] == before + 1
            assert state.schema_epoch == before + 1  # no published change
            assert composer._mutate is mutation_lock
            report = await (await client.get("/api/composition")).json()
            assert "broken" not in report["packs"]
            assert "alpha.echo" in composer.composition.schemas
            assert "gamma.echo" in composer.composition.schemas
            # The broken pack stays in the plan: fixing the install and
            # re-activating is the retry path.
            plan = await (await client.get("/api/install/activation")).json()
            assert plan["add"] == ["broken"]
        finally:
            if client is not None:
                await client.close()
            await composer.close()

    asyncio.run(scenario())


def test_activation_reuses_unchanged_in_process_default_packs(tmp_path: Path) -> None:
    async def scenario() -> None:
        installer = make_installer(tmp_path)
        alpha = write_pack(tmp_path / "alpha", "alpha")
        composer = ServingComposer(worker_env={"PYTHONPATH": str(alpha)})
        client: TestClient | None = None
        try:
            for standard in default_pack_specs():
                await composer.add_pack(standard)
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
            )
            add_activation_routes(app, composer, installer)
            client = TestClient(TestServer(app))
            await client.start_server()

            assert apply_generation(installer, alpha) == 1
            response = await client.post("/api/install/activation")
            assert response.status == 200
            payload = await response.json()
            assert payload["added"] == ["alpha"]
            assert payload["rolledBack"] is False

            nodes = await (await client.get("/api/nodes")).json()
            assert nodes["nodes"]["std.math.add_ints"]["pack"] == "dinkster-nodes-foundation"
            assert nodes["nodes"]["alpha.echo"]["pack"] == "alpha"
            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(
                Graph(
                    nodes={
                        "add": GraphNode("std.math.add_ints", {"a": 2, "b": 3}),
                        "echo": GraphNode("alpha.echo", {"text": "live"}),
                    }
                ),
                ["add", "echo"],
            )
            assert result.outputs["add"]["sum"].resolve() == 5
            assert result.outputs["echo"]["out"].resolve() == "live-v1"
        finally:
            if client is not None:
                await client.close()
            await composer.close()

    asyncio.run(scenario())


def test_extension_snapshot_activation_transport_and_deactivation(tmp_path: Path) -> None:
    """The proof pack activates through disk, publishes one canonical snapshot,
    and deactivation returns to the exact empty snapshot and digest."""

    async def scenario() -> None:
        installer = make_installer(tmp_path)
        proof = write_pack(tmp_path / "proof", "proof", extension=True)
        assert apply_generation(installer, proof) == 1
        composer = ServingComposer(worker_env={"PYTHONPATH": str(proof)})
        client: TestClient | None = None
        try:
            spec = replace(
                installer.packs_for_serving()[0],
                extension_config={"ratio": 1.25},
            )
            await composer.add_pack(spec)
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
            )
            add_activation_routes(app, composer, installer)
            client = TestClient(TestServer(app))
            await client.start_server()

            nodes = await (await client.get("/api/nodes")).json()
            snapshot = await (await client.get("/api/extensions/snapshot")).json()
            active = app[STATE_KEY].engine.extension_snapshot
            assert snapshot == json.loads(canonical_extension_snapshot(active))
            assert nodes["extensionSnapshotDigest"] == ("sha256:" + extension_behavior_hash(active))
            assert [item["id"] for item in snapshot["extensions"]] == ["proof"]
            assert snapshot["extensions"][0]["capabilities"] == ["routes"]
            assert snapshot["extensions"][0]["contributionIds"] == ["schema:proof-exclusive"]
            assert snapshot["extensions"][0]["behaviorConfiguration"] == [
                {"key": "ratio", "value": "1.25"}
            ]

            assert apply_generation(installer) == 2
            response = await client.post("/api/install/activation")
            payload = await response.json()
            assert payload["removed"] == ["proof"]
            empty = app[STATE_KEY].engine.extension_snapshot
            assert empty.extensions == ()
            empty_wire = await (await client.get("/api/extensions/snapshot")).json()
            assert empty_wire == json.loads(canonical_extension_snapshot(empty))
            empty_nodes = await (await client.get("/api/nodes")).json()
            assert empty_nodes["extensionSnapshotDigest"] == (
                "sha256:" + extension_behavior_hash(empty)
            )
        finally:
            if client is not None:
                await client.close()
            await composer.close()

    asyncio.run(scenario())


def test_exclusive_extension_collision_rolls_back_deterministically(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        installer = make_installer(tmp_path)
        alpha = write_pack(tmp_path / "alpha", "alpha", extension=True)
        beta = write_pack(tmp_path / "beta", "beta", extension=True)
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(alpha), str(beta)))}
        )
        client: TestClient | None = None
        try:
            composition = composer.composition
            app = create_app(composition.make_engine, composition.schemas)
            add_activation_routes(app, composer, installer)
            client = TestClient(TestServer(app))
            await client.start_server()
            before_digest = app[STATE_KEY].engine.extension_snapshot_digest

            assert apply_generation(installer, alpha, beta) == 1
            payload = await (await client.post("/api/install/activation")).json()
            assert payload["rolledBack"] is True
            assert payload["diagnostics"] == [
                {
                    "pack": "beta",
                    "kind": "incompatible-pack",
                    "error": (
                        "exclusive extension surface schema:proof-exclusive has "
                        "multiple contributors: alpha, beta"
                    ),
                }
            ]
            assert app[STATE_KEY].engine.extension_snapshot_digest == before_digest
            assert composer.pack_specs() == {}
        finally:
            if client is not None:
                await client.close()
            await composer.close()

    asyncio.run(scenario())


def test_compute_plan_with_no_generation_is_empty(tmp_path: Path) -> None:
    """An install root that never applied a generation plans nothing -
    and in particular does not classify dev packs as removals."""

    async def scenario() -> None:
        installer = make_installer(tmp_path)
        dev_dir = write_pack(tmp_path / "dev", "devpack")
        composer = ServingComposer(worker_env={"PYTHONPATH": str(dev_dir)})
        try:
            await composer.add_pack(dev_dir / "dinkster-pack.toml")
            plan, specs = compute_plan(composer, installer)
            assert plan.generation is None
            assert plan.add == ()
            assert plan.remove == ()
            assert plan.reload == ()
            assert plan.unchanged == ()
            assert specs == {}
        finally:
            await composer.close()

    asyncio.run(scenario())
