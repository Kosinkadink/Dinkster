"""S1 worker-local sampler registry and minimal execution-context proofs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_graph import Graph, GraphNode
from dinkster_inference import (
    Parameterization,
    SamplerExtensionEntry,
    SamplerInfo,
    SamplingCancelled,
    StepEvent,
    builtin_sampler_registry,
    builtin_sampler_snapshot,
    materialize_sampler_registry,
    sampling_execution_context,
    use_additional_sampling_cancellation,
    use_sampling_environment,
    write_sampler_catalog,
)
from dinkster_protocol import SamplerRegistrySnapshot, extension_behavior_hash
from dinkster_server import STATE_KEY, create_app

from dinkster.compose import CompositionError, PackSpec, ServingComposer
from dinkster.reload_api import apply_reload, apply_remove

FIXTURES = Path(__file__).parent
ATTENTION_PROVIDER = FIXTURES / "fixtures/attention_provider"
PROOF_MODULES = ("s1_sampler_pack_a", "s1_sampler_pack_b")


def _host_manifest(root: Path) -> Path:
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


def _extension_manifest(
    root: Path,
    name: str,
    module: str,
    *,
    namespace: str | None = None,
    contract: str = "",
) -> Path:
    root.mkdir(parents=True)
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{name}"\n'
        f'namespaces = ["{namespace or name}"]\n\n'
        '[pack.entry]\nnodes = "s1_sampler_empty:NODES"\n\n'
        "[pack.extension]\n"
        f'inference = "{module}:register"\n'
        'privileges = ["inference"]\n'
        f"{contract}",
        encoding="utf-8",
    )
    return manifest


def _write_collision_module(root: Path, module: str, descriptor: str) -> None:
    (root / f"{module}.py").write_text(
        "from dinkster_api.v1 import (NoiseKind, SamplerContribution, "
        "SamplerDescriptor)\n"
        "def make(_options):\n"
        "    def solve(_denoiser, x, _sigmas, _info, *, noise=None, "
        "on_step=None):\n"
        "        return x\n"
        "    return solve\n"
        f"SAMPLER = SamplerDescriptor({descriptor}, display_name='collision', "
        "make=make, noise=NoiseKind.NONE)\n"
        "def register():\n"
        "    return SamplerContribution((SAMPLER,))\n",
        encoding="utf-8",
    )


def _write_scheduler_module(root: Path, module: str, descriptor: str) -> None:
    (root / f"{module}.py").write_text(
        "from dinkster_api.v1 import InferenceContribution, SchedulerDescriptor\n"
        "def make_sigmas(steps, _space):\n"
        "    return (float(steps), 0.0)\n"
        f"SCHEDULER = SchedulerDescriptor({descriptor}, display_name='collision', "
        "make_sigmas=make_sigmas)\n"
        "def register():\n"
        "    return InferenceContribution(schedulers=(SCHEDULER,))\n",
        encoding="utf-8",
    )


def _write_reloaded_module(root: Path) -> None:
    (root / "s1_sampler_reloaded.py").write_text(
        "from dinkster_api.v1 import NoiseKind, SamplerContribution, SamplerDescriptor\n"
        "def make(_options):\n"
        "    def solve(_denoiser, x, _sigmas, _info, *, noise=None, on_step=None):\n"
        "        return x\n"
        "    return solve\n"
        "SAMPLER = SamplerDescriptor(id='proof_a.reloaded', "
        "display_name='reloaded', make=make, noise=NoiseKind.NONE, "
        "aliases=('s1_reloaded',))\n"
        "def register():\n"
        "    return SamplerContribution((SAMPLER,))\n",
        encoding="utf-8",
    )


def _worker_env(tmp_path: Path) -> dict[str, str]:
    return {"PYTHONPATH": os.pathsep.join((str(ATTENTION_PROVIDER), str(FIXTURES), str(tmp_path)))}


def test_initial_sampler_host_publication_replaces_its_new_derived_choice(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        composition = composer.composition
        app = create_app(
            composition.make_engine,
            composition.schemas,
            packs=composition.packs,
            node_packs=composition.node_packs,
            choices=composition.choices,
        )
        try:
            state = app[STATE_KEY]
            assert "comfy.samplers" not in state.choices
            delta = await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            assert delta.choices["comfy.samplers"] == ("euler",)
            assert "ar_video" not in delta.choices["comfy.samplers"]
            assert "ar_video" in delta.derived_choices["comfy.samplers"]

            state.replace(
                (),
                (),
                delta.schemas,
                delta.packs,
                delta.node_packs,
                execution_arms=delta.execution_arms,
                remove_choices=tuple(delta.derived_choices),
                choices={**delta.choices, **delta.derived_choices},
            )

            assert state.choices["comfy.samplers"] == composition.choices["comfy.samplers"]
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_two_out_of_tree_packs_compose_in_sampling_worker_and_unload_exactly(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        for module in PROOF_MODULES:
            sys.modules.pop(module, None)
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            baseline = composer._runtime_seat.pin()
            baseline_ids = tuple(
                sampler.id for sampler in baseline.sampler_registry_snapshot.samplers
            )
            baseline_choices = dict(composer.composition.choices)

            # Deliberately add in reverse order. Composition order is extension id,
            # not activation timing, and pack code stays out of the parent.
            await composer.add_pack(
                _extension_manifest(tmp_path / "proof_b", "proof_b", "s1_sampler_pack_b")
            )
            await composer.add_pack(
                _extension_manifest(tmp_path / "proof_a", "proof_a", "s1_sampler_pack_a")
            )
            assert all(module not in sys.modules for module in PROOF_MODULES)

            runtime = composer._runtime_seat.pin()
            ids = tuple(sampler.id for sampler in runtime.sampler_registry_snapshot.samplers)
            assert ids == (
                *baseline_ids,
                "proof_a.scaled_euler",
                "proof_b.context_probe",
            )
            assert composer.composition.choices["dinkster.samplers"] == ids
            assert composer.composition.choices["comfy.samplers"][-2:] == (
                "s1_scaled_euler",
                "s1_context_probe",
            )
            assert composer.composition.choices["dinkster.schedulers"][-1] == ("proof_a.scheduler")
            assert composer.composition.choices["comfy.schedulers"][-1] == "s1_scheduler"
            assert [extension.id for extension in runtime.extension_snapshot.extensions] == [
                "proof_a",
                "proof_b",
            ]
            assert tuple(
                contribution.id
                for extension in runtime.extension_snapshot.extensions
                for contribution in extension.keyed_contributions
            ) == (
                "proof_a.scaled_euler",
                "proof_a.scheduler",
                "proof_b.context_probe",
            )
            catalog = json.loads(composer._sampler_catalog_path.read_text(encoding="utf-8"))
            assert all(key.startswith("sha256:") for key in catalog["records"])

            engine = composer.composition.make_engine(lambda _event: None)
            sampled = await engine.run(
                Graph(nodes={"probe": GraphNode("dinkster.ksampler", {})}),
                ["probe"],
            )
            assert sampled.outputs["probe"]["value"].resolve() == 3.0

            # The synchronous sampling loop runs in asyncio.to_thread. Parent
            # cancellation must still set the worker-local cooperative token,
            # stop that thread, and leave the sampling worker usable.
            marker = tmp_path / "cancelled.txt"
            cancellation = asyncio.create_task(
                engine.run(
                    Graph(
                        nodes={"probe": GraphNode("dinkster.cancel_probe", {"marker": str(marker)})}
                    ),
                    ["probe"],
                )
            )
            await asyncio.sleep(0.1)
            cancellation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancellation
            for _ in range(100):
                try:
                    marker_content = marker.read_text(encoding="ascii")
                except FileNotFoundError:
                    pass
                else:
                    if marker_content == "cancelled":
                        break
                await asyncio.sleep(0.02)
            assert marker.read_text(encoding="ascii") == "cancelled"
            assert all(worker.alive for worker in composer.composition._isolated)

            nonempty_hash = extension_behavior_hash(runtime.extension_snapshot)
            assert nonempty_hash != extension_behavior_hash(baseline.extension_snapshot)

            await composer.remove_pack("proof_b")
            await composer.remove_pack("proof_a")
            restored = composer._runtime_seat.pin()
            assert restored.extension_snapshot.extensions == ()
            assert restored.sampler_registry_snapshot == builtin_sampler_snapshot()
            assert composer.composition.choices == baseline_choices
        finally:
            await composer.close()
            for module in PROOF_MODULES:
                sys.modules.pop(module, None)

    asyncio.run(scenario())


def test_pack_registry_provider_orders_consumer_and_executes_declared_sampler(
    tmp_path: Path,
) -> None:
    provider = _extension_manifest(
        tmp_path / "proof_a",
        "proof_a",
        "s1_sampler_pack_a",
        contract=(
            '\n[pack.provides.registry]\n"dinkster.samplers" = ["proof_a.scaled_euler"]\n'
            '"dinkster.schedulers" = ["proof_a.scheduler"]\n'
        ),
    )
    consumer = _extension_manifest(
        tmp_path / "proof_b",
        "proof_b",
        "s1_sampler_pack_b",
        contract=(
            '\n[pack.requirements.registry]\n"dinkster.samplers" = ["proof_a.scaled_euler"]\n'
        ),
    )

    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            ordered = composer.order_pack_entries((consumer, provider))
            for spec in ordered:
                await composer.add_pack(spec)
            composition = composer.composition
            registry_receipts = [
                item for item in composition.generation.resolutions if item.kind == "registry"
            ]
            assert len(registry_receipts) == 1
            assert registry_receipts[0].pack == "proof-b"
            assert registry_receipts[0].requirement == ("dinkster.samplers:proof_a.scaled_euler")
            assert registry_receipts[0].provider == "proof-a"
            sampled = await composition.make_engine(lambda _event: None).run(
                Graph(nodes={"probe": GraphNode("dinkster.ksampler", {})}),
                ["probe"],
            )
            assert sampled.outputs["probe"]["value"].resolve() == 3.0
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_pack_registry_provider_must_match_materialized_contribution(tmp_path: Path) -> None:
    provider = _extension_manifest(
        tmp_path / "proof_a",
        "proof_a",
        "s1_sampler_pack_a",
        contract=('\n[pack.provides.registry]\n"dinkster.samplers" = ["proof_a.missing"]\n'),
    )

    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            with pytest.raises(
                CompositionError,
                match=(
                    "declares registry provider "
                    "dinkster.samplers:proof_a.missing, but its inference contribution "
                    "does not register it"
                ),
            ):
                await composer.add_pack(provider)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_live_sampler_choices_publish_add_reload_remove_and_rollback(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        composition = composer.composition
        app = create_app(
            composition.make_engine,
            composition.schemas,
            packs=composition.packs,
            node_packs=composition.node_packs,
            choices=composition.choices,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            state = app[STATE_KEY]
            host = await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            state.replace(
                (),
                (),
                host.schemas,
                host.packs,
                host.node_packs,
                choices=host.choices,
            )

            added = await composer.add_pack(
                _extension_manifest(tmp_path / "proof_a", "proof_a", "s1_sampler_pack_a")
            )
            assert set(added.derived_choices) == {
                "dinkster.samplers",
                "dinkster.schedulers",
                "comfy.samplers",
                "comfy.schedulers",
            }
            state.replace(
                (),
                (),
                added.schemas,
                added.packs,
                added.node_packs,
                remove_choices=tuple(added.derived_choices),
                choices={**added.choices, **added.derived_choices},
            )
            assert await (await client.get("/api/choices/dinkster.samplers")).json() == list(
                composition.choices["dinkster.samplers"]
            )
            assert "proof_a.scaled_euler" in state.choices["dinkster.samplers"]

            before_failure = dict(state.choices)
            with pytest.raises(CompositionError, match="No module named"):
                await composer.add_pack(
                    _extension_manifest(tmp_path / "missing", "missing", "module_does_not_exist")
                )
            assert state.choices == before_failure

            _write_reloaded_module(tmp_path)
            reloaded_manifest = _extension_manifest(
                tmp_path / "proof_a_reloaded",
                "proof_a",
                "s1_sampler_reloaded",
            )
            result = await apply_reload(
                state,
                composer,
                "proof_a",
                PackSpec(reloaded_manifest),
            )
            assert result["pack"] == "proof_a"
            assert "proof_a.reloaded" in state.choices["dinkster.samplers"]
            assert "proof_a.scaled_euler" not in state.choices["dinkster.samplers"]
            assert await (await client.get("/api/choices/comfy.samplers")).json() == list(
                composition.choices["comfy.samplers"]
            )

            await apply_remove(state, composer, "proof_a")
            assert state.choices["dinkster.samplers"] == tuple(
                sampler.id for sampler in builtin_sampler_snapshot().samplers
            )
            assert "s1_reloaded" not in state.choices["comfy.samplers"]
        finally:
            await client.close()
            await composer.close()

    asyncio.run(scenario())


def test_sampler_collision_and_activation_failure_roll_back_staged_generation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        builtin_module = "s1_collision_builtin"
        alias_module = "s1_collision_alias"
        broken_module = "s1_activation_broken"
        _write_collision_module(
            tmp_path,
            builtin_module,
            "id='dinkster.euler'",
        )
        _write_collision_module(
            tmp_path,
            alias_module,
            "id='collision.other', aliases=('s1_scaled_euler',)",
        )
        (tmp_path / f"{broken_module}.py").write_text(
            "def register():\n    raise RuntimeError('activation boom')\n",
            encoding="utf-8",
        )
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            await composer.add_pack(
                _extension_manifest(tmp_path / "proof_a", "proof_a", "s1_sampler_pack_a")
            )
            before = composer._runtime_seat.pin()
            before_choices = dict(composer.composition.choices)
            before_specs = composer.pack_specs()

            with pytest.raises(CompositionError, match="dinkster.euler.*already registered"):
                await composer.add_pack(
                    PackSpec(
                        _extension_manifest(
                            tmp_path / "builtin_collision",
                            "builtin_collision",
                            builtin_module,
                            namespace="dinkster",
                        ),
                        trust_reserved=True,
                    )
                )
            with pytest.raises(CompositionError, match="s1_scaled_euler.*already registered"):
                await composer.add_pack(
                    _extension_manifest(
                        tmp_path / "alias_collision",
                        "collision",
                        alias_module,
                    )
                )
            with pytest.raises(CompositionError, match="activation boom"):
                await composer.add_pack(
                    _extension_manifest(tmp_path / "broken", "broken", broken_module)
                )
            with pytest.raises(CompositionError, match="No module named"):
                await composer.add_pack(
                    _extension_manifest(tmp_path / "missing", "missing", "module_does_not_exist")
                )

            assert composer._runtime_seat.pin() is before
            assert composer.composition.choices == before_choices
            assert composer.pack_specs() == before_specs
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_scheduler_collision_from_two_extensions_fails_host_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        _write_scheduler_module(
            tmp_path,
            "s1_scheduler_collision_a",
            "id='scheduler_a.value', aliases=('shared_scheduler',)",
        )
        _write_scheduler_module(
            tmp_path,
            "s1_scheduler_collision_b",
            "id='scheduler_b.value', aliases=('second_scheduler',)",
        )
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            await composer.add_pack(
                _extension_manifest(
                    tmp_path / "scheduler_a",
                    "scheduler_a",
                    "s1_scheduler_collision_a",
                )
            )
            await composer.add_pack(
                _extension_manifest(
                    tmp_path / "scheduler_b",
                    "scheduler_b",
                    "s1_scheduler_collision_b",
                )
            )
            entries, contributions = await composer._materialize_inference_contributions(
                composer._records, composer._topology
            )
            scheduler_b = contributions["scheduler_b"][0]
            contributions["scheduler_b"] = (
                replace(
                    scheduler_b,
                    aliases=("shared_scheduler",),
                ),
            )

            async def colliding_contributions(_records, _topology):
                return entries, contributions

            monkeypatch.setattr(
                composer,
                "_materialize_inference_contributions",
                colliding_contributions,
            )
            with pytest.raises(
                CompositionError,
                match="extension 'scheduler_b' scheduler registry collision: "
                "'shared_scheduler' already registered",
            ):
                await composer._build_extension_snapshot(composer._records, composer._topology)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_worker_materialization_validates_declarations_bidirectionally(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(FIXTURES))
    try:
        catalog = tmp_path / "catalog.json"
        key = f"candidate:{uuid.uuid4().hex}"
        entries = (SamplerExtensionEntry("proof_a", "s1_sampler_pack_a:register"),)
        write_sampler_catalog(catalog, key, entries)
        produced = materialize_sampler_registry(key, catalog_path=catalog)
        assert produced.extension_ids == ("proof_a",)
        assert produced.snapshot.samplers[-1].id == "proof_a.scaled_euler"

        mismatch_key = f"candidate:{uuid.uuid4().hex}"
        write_sampler_catalog(
            catalog,
            mismatch_key,
            entries,
            SamplerRegistrySnapshot(produced.snapshot.samplers[:-1]),
        )
        with pytest.raises(RuntimeError, match="declaration mismatch"):
            materialize_sampler_registry(mismatch_key, catalog_path=catalog)
    finally:
        sys.path.remove(str(FIXTURES))
        sys.modules.pop("s1_sampler_pack_a", None)


def test_context_solver_threads_ordinals_rng_progress_state_and_cancellation() -> None:
    sys.path.insert(0, str(FIXTURES))
    try:
        from s1_sampler_pack_a import SCALED_EULER
        from s1_sampler_pack_b import CONTEXT_PROBE

        info = SamplerInfo(Parameterization.EPS, seed=42)
        schedule = (2.0, 1.0, 0.0)

        def denoiser(x: float, sigma: float) -> float:
            del x, sigma
            return 0.0

        euler = builtin_sampler_registry().get("euler")
        assert euler is not None
        builtin_output = euler.build()(denoiser, 8.0, schedule, info)
        scaled_output = SCALED_EULER.build()(denoiser, 8.0, schedule, info)
        assert scaled_output != builtin_output

        progress: list[StepEvent] = []
        with use_sampling_environment(("proof_a", "proof_b"), lambda: False):
            first = CONTEXT_PROBE.build()(denoiser, 8.0, schedule, info, on_step=progress.append)
        with use_sampling_environment(("proof_b", "proof_a"), lambda: False):
            second = CONTEXT_PROBE.build()(denoiser, 8.0, schedule, info)
        assert first == second
        assert [event.step for event in progress] == [0, 1]

        cancelled = False

        def is_cancelled() -> bool:
            return cancelled

        def cancel_after_first(_event: StepEvent) -> None:
            nonlocal cancelled
            cancelled = True

        with use_sampling_environment(("proof_b",), is_cancelled):
            with pytest.raises(SamplingCancelled, match="sampling cancelled"):
                CONTEXT_PROBE.build()(
                    denoiser,
                    8.0,
                    schedule,
                    info,
                    on_step=cancel_after_first,
                )
        assert cancelled is True
    finally:
        sys.path.remove(str(FIXTURES))
        for module in PROOF_MODULES:
            sys.modules.pop(module, None)


def test_additional_sampling_cancellation_preserves_environment() -> None:
    cancelled = False
    peer_failed = False

    with use_sampling_environment(("proof_b",), lambda: cancelled):
        execution = sampling_execution_context((1.0, 0.0), 7)
        with use_additional_sampling_cancellation(lambda: peer_failed):
            assert tuple(execution.extension_state) == ("proof_b",)
            execution.cancellation.check()
            peer_failed = True
            with pytest.raises(SamplingCancelled, match="sampling cancelled"):
                execution.cancellation.check()
            peer_failed = False
            cancelled = True
            with pytest.raises(SamplingCancelled, match="sampling cancelled"):
                execution.cancellation.check()
