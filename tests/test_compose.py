"""Multi-pack composition: --pack manifests become one serving surface.

What this proves: compose_serving merges default + configured pack schemas behind one
RoutingWorker (the engine cannot tell where a node runs), pack provenance
flows from the manifest's loading record to /api/nodes (attribution is
host-side, never a schema's own claim), and misconfiguration - reserved
pack ids, duplicate names, node-type collisions - fails loudly at startup
with no orphaned pack processes.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import shutil
import weakref
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetError, AssetRef
from dinkster_assets.scratch import claim_inactive_produced_job
from dinkster_caches import LayeredCache, MemoryLRUCache
from dinkster_engine import EngineEvent
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, graph_to_wire
from dinkster_memory import (
    GovernorReservationService,
    MeasuredMemory,
    MemoryGovernor,
    ReportedTelemetry,
)
from dinkster_memory.resident import ResidentPool
from dinkster_registry import LockedPack, Lockfile
from dinkster_schema import build_schemas
from dinkster_server import PackInfo, create_app
from dinkster_values import EncodedPayload, TypeRegistry, Value, ValueMeta, default_encode
from dinkster_workers.doctor import prepare_catalog

from dinkster.compose import (
    CompositionError,
    PackSpec,
    ServingComposer,
    compose_serving,
    default_pack_spec,
    order_pack_entries_by_requirements,
    resolve_manifest_path,
)
from tests.async_scenarios import run_scenario

TESTS_DIR = Path(__file__).parent
DEV_PACK_MANIFEST = TESTS_DIR.parent / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"
ATTENTION_PROVIDER = TESTS_DIR / "fixtures/attention_provider"
WORKER_ENV = {"PYTHONPATH": os.pathsep.join((str(ATTENTION_PROVIDER), str(TESTS_DIR)))}


@pytest.fixture(autouse=True)
def synthetic_pack_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster import compose

    monkeypatch.setattr(compose, "_FIRST_PARTY_PACK_MODULES", {"dinkster-text": "dinkster_text"})
    monkeypatch.setattr(
        compose, "_FIRST_PARTY_PACK_DISTRIBUTIONS", {"dinkster-text": "dinkster-text"}
    )


@pytest.fixture
def text_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from dinkster import compose

    root = tmp_path / "text-pack"
    module = root / "src/dinkster_text"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("NODES = ()\n", encoding="utf-8")
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "dinkster-text"\nversion = "0.0.4"\n'
        'engine = ">=0.0.2.dev0,<2"\npublisher = "dinkster"\n'
        'namespaces = ["dinkster.text"]\n'
        '[pack.entry]\nnodes = "dinkster_text:NODES"\n',
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "dinkster-text"\nversion = "0.0.4"\n', encoding="utf-8"
    )
    lock = Lockfile.of(
        [
            LockedPack(
                "dinkster-text",
                "0.0.4",
                compose._installed_pack_digest(manifest, module),
                "dinkster",
                ("dinkster-text",),
                "python:dinkster-text==0.0.4",
            )
        ]
    )
    monkeypatch.setattr(compose, "_default_pack_lock", lambda: lock)
    return root


def editable_distribution(root: Path, version: str) -> SimpleNamespace:
    return SimpleNamespace(
        version=version,
        files=(),
        locate_file=lambda path: root / path,
        read_text=lambda name: (
            json.dumps({"url": root.as_uri(), "dir_info": {"editable": True}})
            if name == "direct_url.json"
            else None
        ),
    )


def test_default_pack_specs_verify_independently_versioned_checkout_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster.compose as compose

    checkout = tmp_path / "checkout"
    package_data = checkout / "src/dinkster"
    package_data.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("")
    pack = tmp_path / "dinkster_text"
    pack.mkdir()
    (pack / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-text"\npublisher = "dinkster"\n'
        'version = "1.2.3"\nengine = ">=0.0.2.dev0,<2"\n'
        'namespaces = ["official"]\n'
        '[pack.entry]\nnodes = "dinkster_text:NODES"\n',
        encoding="utf-8",
    )
    module = pack / "dinkster_text"
    module.mkdir()
    origin = module / "__init__.py"
    origin.write_text("NODES = ()\n", encoding="utf-8")
    digest = compose._installed_pack_digest(pack / "dinkster-pack.toml", module)
    lock = Lockfile.of(
        [
            LockedPack(
                "dinkster-text",
                "1.2.3",
                digest,
                "dinkster",
                ("official",),
                "registry:https://registry.invalid",
            ),
        ]
    )
    catalog = package_data / "checkout-packs.lock"
    catalog.write_text(lock.record_json(), encoding="utf-8")
    monkeypatch.setattr(compose, "__file__", str(package_data / "compose.py"))
    monkeypatch.setattr(compose, "_CHECKOUT_PACKS_LOCK", catalog)
    monkeypatch.setattr(
        compose.importlib.metadata, "distribution", lambda _: editable_distribution(pack, "1.2.3")
    )
    compose._default_pack_lock.cache_clear()
    compose.default_pack_specs.cache_clear()
    try:
        assert compose.default_pack_ids() == ("dinkster-text",)
        spec = compose.default_pack_spec("dinkster-text")
        assert spec.require_catalog
        assert not spec.in_process
        assert spec.packs is not None
        assert spec.packs["dinkster-text"].version == "1.2.3"
        assert spec.packs["dinkster-text"].artifact_digest == digest
        assert spec.packs["dinkster-text"].source == "registry:https://registry.invalid"
        assert Path(spec.manifest).parent == pack

        async def catalog_is_required() -> None:
            isolated = replace(spec, in_process=False, env={"PYTHONPATH": str(pack)})
            composer = ServingComposer()
            try:
                with pytest.raises(CompositionError, match="catalog is missing or stale"):
                    await composer.add_pack(isolated)
                assert prepare_catalog(isolated.manifest, environment=isolated.env).ok
                await composer.add_pack(isolated)
            finally:
                await composer.close()

        run_scenario(catalog_is_required())
        origin.write_text("NODES = ()\nVALUE = 'tampered'\n", encoding="utf-8")
        with pytest.raises(CompositionError, match="installed artifact is"):
            compose.default_pack_spec("dinkster-text")
    finally:
        compose.default_pack_specs.cache_clear()
        compose._default_pack_lock.cache_clear()


@pytest.fixture
def isolated_full_free_pool(monkeypatch: pytest.MonkeyPatch) -> ResidentPool:
    pool = ResidentPool()
    monkeypatch.setattr("dinkster.compose.default_pool", lambda: pool)
    return pool


def write_iso_manifest(
    directory: Path,
    name: str = "isopack",
    *,
    presentation: bool = True,
    namespaces: tuple[str, ...] | None = ("iso",),
) -> Path:
    """The iso test pack's nodes are iso.*, so its claim is "iso" - pack id
    and node namespace need not coincide (the std/comfy shape). None omits
    the field, leaving the default claim (the pack name)."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "dinkster-pack.toml"
    claims = (
        ""
        if namespaces is None
        else "namespaces = [" + ", ".join(f'"{c}"' for c in namespaces) + "]\n"
    )
    body = (
        f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
        f'type-namespaces = ["iso"]\n{claims}\n[pack.entry]\n'
        'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
    )
    if presentation:
        body += (
            '\n[pack.presentation]\ndisplay_name = "Iso Pack"\n'
            'abbr = "ISO"\nmark = "\\U0001F9EA"\ncolor = "#336699"\n'
        )
    manifest.write_text(body)
    return manifest


def write_contract_manifest(
    directory: Path,
    name: str,
    declarations: str = "",
    *,
    namespace: str = "iso",
    nodes_entry: str = "empty_pack_nodes:NODES",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
        f'namespaces = ["{namespace}"]\n'
        '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
        f"{declarations}"
        f'[pack.entry]\nnodes = "{nodes_entry}"\n',
        encoding="utf-8",
    )
    return manifest


def test_resolve_manifest_path(tmp_path: Path) -> None:
    manifest = write_iso_manifest(tmp_path / "pack")
    assert resolve_manifest_path(manifest) == manifest
    assert resolve_manifest_path(manifest.parent) == manifest
    with pytest.raises(CompositionError, match="not found"):
        resolve_manifest_path(tmp_path / "nowhere")


def test_pack_contract_resolver_orders_dependencies_and_capability_providers(
    tmp_path: Path,
) -> None:
    provider = write_contract_manifest(
        tmp_path / "provider",
        "provider",
        '[pack.capabilities]\n"provider.video-generation" = "2.1.0"\n',
    )
    consumer = write_contract_manifest(
        tmp_path / "consumer",
        "consumer",
        '[pack.dependencies]\nprovider = ">=2,<3"\n'
        "[pack.requirements.capabilities]\n"
        '"provider.video-generation" = ">=2,<3"\n',
    )
    digest = "blake3:" + "a" * 64
    provider_spec = PackSpec(
        provider,
        packs={
            "provider": PackInfo(display_name="Provider", version="2.0.0", artifact_digest=digest)
        },
    )
    consumer_spec = PackSpec(
        consumer,
        packs={
            "consumer": PackInfo(display_name="Consumer", version="1.0.0", artifact_digest=digest)
        },
    )
    composer = ServingComposer()
    try:
        ordered = composer.order_pack_entries((consumer_spec, provider_spec))
        assert [Path(spec.manifest) for spec in ordered] == [provider, consumer]
    finally:
        run_scenario(composer.close())


def test_requirement_ordering_survives_unresolvable_contracts(tmp_path: Path) -> None:
    """A pack whose contracts cannot resolve must not undo the ordering of the
    healthy packs: the consumer still composes after its declared provider."""
    provider = write_contract_manifest(tmp_path / "provider", "provider")
    consumer = write_contract_manifest(
        tmp_path / "consumer", "consumer", '[pack.dependencies]\nprovider = ">=2,<3"\n'
    )
    broken = write_contract_manifest(
        tmp_path / "broken",
        "broken",
        '[pack.requirements.capabilities]\n"missing.capability" = ">=1,<2"\n',
    )
    ordered = order_pack_entries_by_requirements((consumer, provider, broken))
    assert [Path(spec.manifest).parent.name for spec in ordered] == [
        "provider",
        "consumer",
        "broken",
    ]


def test_requirement_ordering_tolerates_cycles_unknowns_and_broken_manifests(
    tmp_path: Path,
) -> None:
    first = write_contract_manifest(
        tmp_path / "first", "first-cycle", '[pack.dependencies]\n"second-cycle" = ">=1,<2"\n'
    )
    second = write_contract_manifest(
        tmp_path / "second", "second-cycle", '[pack.dependencies]\n"first-cycle" = ">=1,<2"\n'
    )
    ghost = write_contract_manifest(
        tmp_path / "ghost", "ghost", '[pack.dependencies]\nunlisted-pack = ">=1,<2"\n'
    )
    entries = (first, second, ghost, tmp_path / "nowhere" / "dinkster-pack.toml")
    ordered = order_pack_entries_by_requirements(entries)
    # Unknown dependencies and unreadable manifests never refuse the set; the
    # cycle members follow once nothing else can move, in given order.
    assert [Path(spec.manifest).parent.name for spec in ordered] == [
        "ghost",
        "nowhere",
        "first",
        "second",
    ]


def test_pack_contract_resolver_refuses_cycles_collisions_and_missing_capabilities(
    tmp_path: Path,
) -> None:
    digest = "blake3:" + "b" * 64

    def spec(path: Path, name: str) -> PackSpec:
        return PackSpec(
            path,
            packs={name: PackInfo(display_name=name, version="1.0.0", artifact_digest=digest)},
        )

    alpha = write_contract_manifest(
        tmp_path / "alpha", "alpha", '[pack.dependencies]\nbeta = ">=1"\n'
    )
    beta = write_contract_manifest(
        tmp_path / "beta", "beta", '[pack.dependencies]\nalpha = ">=1"\n'
    )
    composer = ServingComposer()
    try:
        with pytest.raises(CompositionError, match="dependency cycle"):
            composer.order_pack_entries((spec(alpha, "alpha"), spec(beta, "beta")))

        first = write_contract_manifest(
            tmp_path / "first",
            "first",
            '[pack.capabilities]\n"shared.provider-id" = "1.0.0"\n',
        )
        second = write_contract_manifest(
            tmp_path / "second",
            "second",
            '[pack.capabilities]\n"shared.provider_id" = "2.0.0"\n',
        )
        with pytest.raises(CompositionError, match="provided by both"):
            composer.order_pack_entries((spec(first, "first"), spec(second, "second")))

        missing = write_contract_manifest(
            tmp_path / "missing",
            "missing",
            '[pack.requirements.capabilities]\n"missing.family" = ">=1"\n',
        )
        with pytest.raises(CompositionError, match="no pack provides it"):
            composer.order_pack_entries((spec(missing, "missing"),))

        invalid_provenance = PackSpec(
            write_contract_manifest(tmp_path / "invalid-provenance", "invalid-provenance"),
            packs={
                "invalid-provenance": PackInfo(
                    display_name="Invalid", artifact_digest="not-a-digest"
                )
            },
        )
        with pytest.raises(CompositionError, match="invalid artifact digest"):
            composer.order_pack_entries((invalid_provenance,))

        invalid_version = PackSpec(
            write_contract_manifest(tmp_path / "invalid-version", "invalid-version"),
            packs={"invalid-version": PackInfo(display_name="Invalid", version="latest")},
        )
        with pytest.raises(CompositionError, match="invalid version"):
            composer.order_pack_entries((invalid_version,))

        contract_mismatch = write_contract_manifest(tmp_path / "mismatch", "mismatch")
        contract_mismatch.write_text(
            contract_mismatch.read_text().replace("dinkster-pack-host/1", "dinkster-pack-host/2")
        )
        with pytest.raises(CompositionError, match="requires host contract"):
            composer.order_pack_entries((spec(contract_mismatch, "mismatch"),))
    finally:
        run_scenario(composer.close())


def test_compose_defaults_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Implicit selection uses the configured defaults without adding other packs."""
    spec = PackSpec(write_iso_manifest(tmp_path))
    monkeypatch.setattr("dinkster.compose.default_pack_specs", lambda: (spec,))

    async def scenario() -> None:
        composition = await compose_serving(worker_env=WORKER_ENV)
        try:
            from tests.isopack_nodes import NODES

            expected = set(build_schemas(NODES))
            assert set(composition.schemas) == expected
            assert set(composition.packs) == {"isopack"}
            assert composition.node_packs == dict.fromkeys(expected, "isopack")
            engine = composition.make_engine(lambda event: None)
            result = await engine.run(
                Graph(nodes={"g": GraphNode("iso.chatty", {"value": "left:right"})}), ["g"]
            )
            assert result.outputs["g"]["value"].resolve() == "LEFT:RIGHT"
            app = create_app(composition.make_engine, composition.schemas)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                response = await client.get("/api/nodes")
                payload = await response.json()
                assert response.status == 200, payload
                assert set(payload["nodes"]) == expected
            finally:
                await client.close()
        finally:
            await composition.close()

    run_scenario(scenario())


def test_default_pack_publishes_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text_pack: Path
) -> None:
    from dinkster import compose

    source = text_pack
    copied = tmp_path / "pack"
    shutil.copytree(source, copied)
    manifest = copied / "dinkster-pack.toml"
    manifest.write_text(manifest.read_text() + '\n[pack.docs]\ndir = "docs"\n')
    node = copied / "docs/nodes/dinkster.text.string_concatenate"
    node.mkdir(parents=True)
    body = b"# Concatenate Text\n\nJoins two strings.\n"
    (node / "en.md").write_bytes(
        b'+++\ntitle = "Concatenate Text"\nsummary = "Joins strings."\nschema_version = 1\n+++\n'
        + body
    )
    monkeypatch.setattr(compose, "_default_pack_lock", lambda: Lockfile.of([]))
    monkeypatch.setattr(
        compose.importlib.metadata,
        "distribution",
        lambda _name: editable_distribution(copied, "0.0.2"),
    )
    spec = default_pack_spec("dinkster-text")
    assert spec.packs is not None
    docs = spec.packs["dinkster-text"].docs
    assert docs is not None
    assert [(page.kind, page.id, page.locale) for page in docs.pages] == [
        ("node", "dinkster.text.string_concatenate", "en"),
    ]
    page = next(page for page in docs.pages if page.kind == "node")
    assert page.title == "Concatenate Text"
    assert page.summary == "Joins strings."
    assert page.schema_version == 1
    assert page.data == body
    assert page.digest == "sha256:" + hashlib.sha256(page.data).hexdigest()


def test_default_pack_uses_embedded_wheel_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text_pack: Path
) -> None:
    from dinkster_registry import artifact_digest, unpack_artifact
    from dinkster_workers.pack_wheel import build_pack_archive

    from dinkster import compose

    lock = compose._default_pack_lock()
    source = text_pack
    output = tmp_path / "wheel"
    build_pack_archive(source, output)
    archive = next(output.glob("*.whl"))
    unpack_artifact(archive, tmp_path / "site", artifact_digest(archive.read_bytes()))
    bundled = tmp_path / "site/dinkster_text/dinkster-pack.toml"
    distribution = next(
        iter(compose.importlib.metadata.distributions(path=[str(tmp_path / "site")]))
    )

    monkeypatch.setattr(
        "dinkster.compose.importlib.metadata.distribution", lambda _name: distribution
    )
    monkeypatch.setattr("dinkster.compose._default_pack_lock", lambda: lock)

    spec = default_pack_spec("dinkster-text")
    assert spec.manifest == bundled
    assert spec.packs is not None
    info = spec.packs["dinkster-text"]
    assert info.source == "python:dinkster-text==0.0.4"
    locked = lock.get("dinkster-text")
    assert locked is not None
    assert info.artifact_digest == locked.artifact_digest


def test_default_pack_refuses_version_outside_suite_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text_pack: Path,
) -> None:
    from dinkster import compose

    lock = compose._default_pack_lock()
    source = text_pack
    copied = tmp_path / "source"
    copied.mkdir()
    manifest = copied / "dinkster-pack.toml"
    manifest.write_text(
        (source / "dinkster-pack.toml")
        .read_text()
        .replace('version = "0.0.4"', 'version = "9.0.0"')
    )
    shutil.copytree(source / "src/dinkster_text", copied / "src/dinkster_text")
    monkeypatch.setattr(
        compose.importlib.metadata,
        "distribution",
        lambda _name: editable_distribution(copied, "0.0.4"),
    )
    monkeypatch.setattr("dinkster.compose._default_pack_lock", lambda: lock)

    with pytest.raises(CompositionError, match="locks dinkster-text version 0.0.4"):
        default_pack_spec("dinkster-text")


@pytest.mark.parametrize("checkout", [False, True])
def test_engine_has_no_implicit_pack_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkout: bool
) -> None:
    from dinkster import compose

    package = tmp_path / ("src" if checkout else "site-packages") / "dinkster"
    package.mkdir(parents=True)
    if checkout:
        (tmp_path / "pyproject.toml").write_text("")
    monkeypatch.setattr(compose, "__file__", str(package / "compose.py"))
    monkeypatch.setattr(compose, "_CHECKOUT_PACKS_LOCK", package / "checkout-packs.lock")
    compose._default_pack_lock.cache_clear()
    try:
        assert compose.default_pack_ids() == ()
    finally:
        compose._default_pack_lock.cache_clear()


def test_composed_layered_cache_reuses_disk_then_promotes_to_memory(tmp_path: Path) -> None:
    async def scenario() -> None:
        graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": 8, "height": 4})})
        cache_dir = tmp_path / "execution-cache"

        async def compose():
            return await compose_serving(
                [PackSpec(DEV_PACK_MANIFEST, trust_reserved=True)],
                include_default_packs=False,
                dev=True,
                cache_mode="layered",
                cache_memory_entries=1,
                cache_dir=cache_dir,
                cache_disk_budget=1024**2,
            )

        first = await compose()
        try:
            first_result = await first.make_engine(lambda _event: None).run(graph, ["g"])
            assert first_result.executed == ("g",)
        finally:
            await first.close()

        restarted = await compose()
        try:
            events: list[EngineEvent] = []
            engine = restarted.make_engine(events.append)
            assert isinstance(engine.cache, LayeredCache)

            disk_result = await engine.run(graph, ["g"])
            disk_hit = next(event for event in events if event.kind == "node_cached")
            assert disk_result.cached == ("g",)
            assert disk_hit.detail["cacheLayer"] == "disk"

            events.clear()
            memory_result = await engine.run(graph, ["g"])
            memory_hit = next(event for event in events if event.kind == "node_cached")
            assert memory_result.cached == ("g",)
            assert memory_hit.detail["cacheLayer"] == "memory"
        finally:
            await restarted.close()

    run_scenario(scenario())


def test_full_free_clears_memory_execution_cache_but_preserves_disk(
    tmp_path: Path, isolated_full_free_pool: ResidentPool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            dev=True,
            cache_mode="layered",
            cache_memory_entries=1,
            cache_dir=tmp_path / "execution-cache",
            cache_disk_budget=1024**2,
        )
        try:
            await composer.add_pack(PackSpec(DEV_PACK_MANIFEST, trust_reserved=True))
            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            graph = Graph(nodes={"g": GraphNode("dev.image.gradient", {"width": 8, "height": 4})})
            first = await engine.run(graph, ["g"])
            assert first.executed == ("g",)
            events.clear()

            staged = composer.spawn_empty()
            await staged.add_pack(PackSpec(DEV_PACK_MANIFEST, trust_reserved=True))
            old = composer.adopt(staged)
            await old.close()
            assert isinstance(engine.cache, LayeredCache)
            memory = engine.cache.layers[0]
            assert isinstance(memory, MemoryLRUCache)
            assert len(memory) == 1

            local = (await composer.full_free("maintenance-cache"))[0]
            assert local["status"] == "complete"
            consumers = cast("list[dict[str, object]]", local["consumers"])
            assert consumers[0] == {
                "consumer": "execution-cache",
                "status": "complete",
            }

            second = await engine.run(graph, ["g"])
            assert second.cached == ("g",)
            disk_hit = next(event for event in events if event.kind == "node_cached")
            assert disk_hit.detail["cacheLayer"] == "disk"
        finally:
            await composer.close()

    run_scenario(scenario())


def test_full_free_releases_tensors_held_by_memory_execution_cache(
    isolated_full_free_pool: ResidentPool,
) -> None:
    torch = pytest.importorskip("torch")

    async def scenario() -> None:
        composer = ServingComposer(dev=True, cache_mode="memory", cache_memory_entries=1)
        try:
            engine = composer.composition.make_engine(lambda _event: None)
            registry = TypeRegistry()
            registry.register("test.tensor", fingerprint=lambda _tensor: "schedule-output")

            async def cache_tensor() -> weakref.ReferenceType[object]:
                tensor = torch.ones(4)
                reference = weakref.ref(tensor)
                await engine.cache.put(
                    "schedule-output",
                    {"sigmas": registry.wrap("test.tensor", tensor)},
                )
                return reference

            tensor_reference = await cache_tensor()
            gc.collect()
            assert tensor_reference() is not None

            (local,) = await composer.full_free("maintenance-tensor-cache")
            assert local["status"] == "complete"
            gc.collect()
            assert tensor_reference() is None
        finally:
            await composer.close()

    run_scenario(scenario())


def test_compose_rejects_reserved_pack_name(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path, name="core", presentation=False)
        with pytest.raises(CompositionError, match="reserved"):
            await compose_serving([manifest], include_default_packs=False, worker_env=WORKER_ENV)

    run_scenario(scenario())


def test_compose_rejects_duplicate_pack_name(tmp_path: Path) -> None:
    async def scenario() -> None:
        first = write_iso_manifest(tmp_path / "a", presentation=False)
        second = write_iso_manifest(tmp_path / "b", presentation=False)
        with pytest.raises(CompositionError, match="duplicate pack name"):
            await compose_serving(
                [first, second], include_default_packs=False, worker_env=WORKER_ENV
            )

    run_scenario(scenario())


def test_compose_rejects_namespace_collision(tmp_path: Path) -> None:
    """Two packs claiming one namespace is a hard startup error naming
    both owners - never a silent precedence pick. This subsumes the old
    pack-vs-pack node-type collision: two isolated packs cannot announce
    the same node type without their claims overlapping first, so the
    error now fires before the second worker even starts."""

    async def scenario() -> None:
        first = write_iso_manifest(tmp_path / "a", name="isopack", presentation=False)
        second = write_iso_manifest(tmp_path / "b", name="isopack2", presentation=False)
        second.write_text(
            second.read_text()
            .replace('type-namespaces = ["iso"]\n', "")
            .replace('types = "isopack_nodes:register_types"\n', "")
        )
        with pytest.raises(
            CompositionError,
            match="'iso' claimed by 'isopack2' overlaps 'iso' claimed by 'isopack'",
        ):
            await compose_serving(
                [first, second], include_default_packs=False, worker_env=WORKER_ENV
            )

    run_scenario(scenario())


@pytest.mark.parametrize("in_process", (True, False))
def test_authored_inputs_execute_dependencies_and_invalidate_cache(
    tmp_path: Path, in_process: bool
) -> None:
    module = f"authored_inputs_{in_process}"
    (tmp_path / f"{module}.py").write_text(
        "from dinkster_schema import "
        "ComboWidget, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr\n"
        "INT = TypeExpr.concrete('INT')\n"
        "STRING = TypeExpr.concrete('STRING')\n"
        "class Index(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='authored.index', inputs=(InputSpec('value', INT),), "
        "outputs=(OutputSpec('value', INT),))\n"
        "    @classmethod\n"
        "    def execute(cls, value):\n"
        "        return cls.outputs(value=value)\n"
        "class Choice(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='authored.choice', inputs=(InputSpec('choice', "
        "TypeExpr.concrete('COMBO'), widget=ComboWidget(options=(), editable_options=True)),), "
        "outputs=(OutputSpec('result', STRING),), additional_inputs=TypeExpr.wildcard())\n"
        "    @classmethod\n"
        "    def execute(cls, choice, **kwargs):\n"
        "        selected = kwargs[str(kwargs['index'])]\n"
        "        return cls.outputs(result=f\"{choice}|{selected}|{kwargs['count']}\")\n"
        "NODES = (Index, Choice)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "authored-inputs"\npublisher = "dinkster"\nnamespaces = ["authored"]\n'
        f'[pack.entry]\nnodes = "{module}:NODES"\n',
        encoding="ascii",
    )
    env = {
        **WORKER_ENV,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), WORKER_ENV["PYTHONPATH"])),
    }

    async def scenario() -> None:
        composer = ServingComposer(worker_env=env)
        try:
            await composer.add_pack(PackSpec(manifest, in_process=in_process, env=env))
            engine = composer.composition.make_engine(lambda _event: None)
            inputs = {
                "choice": "outside-menu",
                "index": Link("index", "value"),
                "2": TypedLiteral("STRING", "left"),
                "9": TypedLiteral("STRING", "right"),
                "count": TypedLiteral("INT", 13),
            }
            graph = Graph(
                nodes={
                    "index": GraphNode("authored.index", {"value": 9}),
                    "choice": GraphNode("authored.choice", inputs),
                }
            )
            first = await engine.run(graph, ["choice"])
            assert first.outputs["choice"]["result"].resolve() == "outside-menu|right|13"
            assert first.executed == ("index", "choice")
            repeated = await engine.run(graph, ["choice"])
            assert repeated.outputs["choice"]["result"].resolve() == "outside-menu|right|13"
            assert repeated.cached == ("index", "choice")
            assert repeated.executed == ()
            graph = Graph(nodes={**graph.nodes, "index": GraphNode("authored.index", {"value": 2})})
            changed_index = await engine.run(graph, ["choice"])
            assert changed_index.outputs["choice"]["result"].resolve() == "outside-menu|left|13"
            assert changed_index.executed == ("index", "choice")
            inputs["2"] = TypedLiteral("STRING", "changed")
            changed_option = await engine.run(graph, ["choice"])
            assert changed_option.outputs["choice"]["result"].resolve() == "outside-menu|changed|13"
            assert changed_option.executed == ("choice",)
            assert changed_option.cached == ("index",)
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize(
    ("in_process", "supplied_registry"), ((True, False), (False, False), (False, True))
)
@pytest.mark.parametrize("verified", (False, True))
def test_asset_literals_use_host_and_worker_resolvers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_process: bool,
    supplied_registry: bool,
    verified: bool,
) -> None:
    from dinkster_assets import (
        LocalAssetRef,
        MountDef,
        MountSnapshotResolver,
        MountTable,
        register_asset_type,
    )
    from dinkster_values import register_core_types

    root = tmp_path / "models"
    root.mkdir()
    payload = b"model bytes for the ordinary asset input"
    (root / "weights.bin").write_bytes(payload)
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "indexes")
    table.add(MountDef(id="models", path=root), kind="model/checkpoint")
    if verified:
        table.scan("models")
    ref = table.bind_name("weights.bin", "model/checkpoint")
    assert isinstance(ref, AssetRef if verified else LocalAssetRef)
    module = f"asset_input_{in_process}_{supplied_registry}_{verified}"
    (tmp_path / f"{module}.py").write_text(
        "from assetpack_nodes import NODES, register_types\n", encoding="ascii"
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "apack"\npublisher = "dinkster"\nnamespaces = ["apack"]\n'
        f'[pack.entry]\nnodes = "{module}:NODES"\ntypes = "{module}:register_types"\n',
        encoding="ascii",
    )
    env = {
        **WORKER_ENV,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), WORKER_ENV["PYTHONPATH"])),
        "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
    }
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    registry = None
    if supplied_registry:
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, MountSnapshotResolver(snapshot))

    async def scenario() -> None:
        composer = ServingComposer(worker_env=env, registry=registry)
        try:
            await composer.add_pack(PackSpec(manifest, in_process=in_process, env=env))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(nodes={"read": GraphNode("apack.read", {"data": ref.to_wire()})})
            for _ in range(2):
                result = await engine.run(graph, ["read"])
                assert result.outputs["read"]["text"].resolve() == payload.decode()
                assert result.outputs["read"]["size"].resolve() == len(payload)
        finally:
            await composer.close()

    run_scenario(scenario())


def test_compose_rejects_nested_namespace_claims(tmp_path: Path) -> None:
    """Overlap is atom-bounded across separator spellings: iso-extra IS
    iso.extra, which nests inside iso - two owners could otherwise both
    cover an iso.extra.* node type."""

    async def scenario() -> None:
        first = write_iso_manifest(tmp_path / "a", presentation=False)
        second = write_iso_manifest(
            tmp_path / "b",
            name="isopack2",
            presentation=False,
            namespaces=("iso-extra",),
        )
        with pytest.raises(CompositionError, match="overlaps 'iso'"):
            await compose_serving(
                [first, second], include_default_packs=False, worker_env=WORKER_ENV
            )

    run_scenario(scenario())


def test_compose_rejects_separator_equivalent_pack_names(tmp_path: Path) -> None:
    """iso_pack and iso-pack are ONE pack identity (separators are an
    equivalence class) - the drift that made one ComfyUI pack look like
    two packages must fail loudly here."""

    async def scenario() -> None:
        first = write_iso_manifest(tmp_path / "a", name="iso-pack", presentation=False)
        second = write_iso_manifest(tmp_path / "b", name="iso_pack", presentation=False)
        with pytest.raises(CompositionError, match="one identity"):
            await compose_serving(
                [first, second], include_default_packs=False, worker_env=WORKER_ENV
            )

    run_scenario(scenario())


def test_compose_rejects_uncovered_node_type(tmp_path: Path) -> None:
    """A worker announcing a node type outside its manifest's claims is a
    startup error: the loading record, not the schema, is the authority
    for what a pack may provide. Omitting namespaces claims the pack name,
    which does not cover iso.*."""

    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path, presentation=False, namespaces=None)
        with pytest.raises(CompositionError, match="outside the pack's declared namespaces"):
            await compose_serving([manifest], include_default_packs=False, worker_env=WORKER_ENV)

    run_scenario(scenario())


def test_compose_gates_reserved_claims_on_host_trust(tmp_path: Path) -> None:
    """A reserved-root claim composes only when the spec vouches for the
    pack (the local analog of the registry's grant table): untrusted is a
    startup error before the worker even launches, trusted composes."""
    from dinkster.compose import PackSpec

    async def scenario() -> None:
        manifest = write_iso_manifest(
            tmp_path, presentation=False, namespaces=("iso", "comfy.extras")
        )
        with pytest.raises(CompositionError, match="explicit host trust"):
            await compose_serving([manifest], include_default_packs=False, worker_env=WORKER_ENV)

        composition = await compose_serving(
            [PackSpec(manifest=manifest, env=WORKER_ENV, trust_reserved=True)],
            include_default_packs=False,
        )
        try:
            assert composition.node_packs["iso.chatty"] == "isopack"
        finally:
            await composition.close()

    run_scenario(scenario())


def test_composed_server_end_to_end(tmp_path: Path) -> None:
    """One server: development nodes in-process, the pack isolated, both executing
    through the same queue; /api/nodes attributes each side truthfully and
    carries the manifest's declared presentation."""

    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path)
        composition = await compose_serving(
            [
                PackSpec(DEV_PACK_MANIFEST, in_process=True),
                manifest,
            ],
            include_default_packs=False,
            worker_env=WORKER_ENV,
        )
        published_schemas = {
            node_type: schema
            for node_type, schema in composition.schemas.items()
            if node_type.startswith("iso.") or node_type == "dev.util.delay"
        }
        published_packs = {
            pack_id: info
            for pack_id, info in composition.packs.items()
            if pack_id in {"isopack", "dinkster-nodes-dev"}
        }
        app = create_app(
            composition.make_engine,
            published_schemas,
            packs=published_packs,
            node_packs={
                node_type: pack_id
                for node_type, pack_id in composition.node_packs.items()
                if node_type in published_schemas
            },
            choices=composition.choices,
            lazy_choices=composition.lazy_choices,
        )

        async def close_composition(_: object) -> None:
            await composition.close()

        app.on_cleanup.append(close_composition)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # Provenance: manifest presentation reaches the packs table,
            # isolated and in-process nodes both attribute to their packs. A
            # plain dev pack exposes where its bytes live but no pin -
            # version/artifactDigest stay omitted (unpinned), never null.
            data = await (await client.get("/api/nodes")).json()
            assert data["packs"]["isopack"] == {
                "displayName": "Iso Pack",
                "abbr": "ISO",
                "mark": "\U0001f9ea",
                "color": "#336699",
                "source": f"local:{Path(manifest).parent.resolve()}",
            }
            assert data["nodes"]["iso.chatty"]["pack"] == "isopack"
            assert data["nodes"]["dev.util.delay"]["pack"] == "dinkster-nodes-dev"
            blob_size = next(
                item
                for item in data["nodes"]["iso.blob_out"]["interface"]
                if item["role"] == "input" and item["id"] == "size"
            )
            assert blob_size["widget"] == {
                "type": "isopack.size",
                "min": 1,
                "max": 12,
                "unit": "bytes",
            }

            samplers = await client.get("/api/choices/dev.gallery.samplers")
            assert samplers.status == 200, await samplers.text()
            assert "euler" in await samplers.json()

            # One graph spanning both workers: the development delay runs
            # in-process, iso.chatty in the pack process, one job.
            graph = Graph(
                nodes={
                    "g": GraphNode(
                        "dev.util.delay",
                        {"value": "left:right", "seconds": 0},
                    ),
                    "c": GraphNode("iso.chatty", {"value": "hi"}),
                    "b": GraphNode("iso.blob_out", {"size": 3}),
                }
            )
            body = {
                "clientId": "c1",
                "jobId": "j1",
                "graph": graph_to_wire(graph),
                "targets": ["g", "c", "b"],
            }
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 202, await resp.text()
            status: dict[str, object] = {}
            for _ in range(200):
                status = await (await client.get("/api/jobs/c1/j1")).json()
                if status["state"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)
            assert status.get("state") == "completed", status

            for node_id, output_id, expected in (
                ("g", "value", "left:right"),
                ("c", "value", "HI"),
            ):
                response = await client.get(
                    f"/api/values?clientId=c1&jobId=j1&nodeId={node_id}&outputId={output_id}"
                )
                assert response.status == 200, await response.text()
                assert (await response.json())["descriptor"]["value"] == expected

            value_url = "/api/values?clientId=c1&jobId=j1&nodeId=b&outputId=blob"
            discovery = await client.get(value_url)
            assert discovery.status == 200, await discovery.text()
            value_data = await discovery.json()
            assert value_data["descriptor"]["typeId"] == "iso.blob"
            assert value_data["renditions"] == [
                {
                    "kind": "summary",
                    "mime": "text/plain",
                    "default": True,
                    "cacheKey": "summary/1",
                    "version": "1",
                    "parameters": ["prefix"],
                    "defaults": {"prefix": "blob"},
                    "limits": {"prefixLength": 32},
                }
            ]
            rendered = await client.get(value_url + "&rendition=summary/1&prefix=pack")
            assert rendered.status == 200, await rendered.text()
            assert rendered.content_type == "text/plain"
            assert await rendered.read() == b"pack:3:xxx"
        finally:
            await client.close()

    run_scenario(scenario())


@pytest.mark.parametrize(
    "in_process, cold", [(False, False), (False, True), (True, False), (True, True)]
)
def test_pack_type_owners_are_discovered_without_activation_and_retracted(
    tmp_path: Path, in_process: bool, cold: bool
) -> None:
    module = f"type_owner_pack_{int(in_process)}_{int(cold)}"
    source = tmp_path / f"{module}.py"
    source.write_text(
        'NODES = []\ndef register_types(registry):\n    registry.register("mypack.thing")\n'
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "mypack"\npublisher = "someone"\n'
        "[pack.entry]\n"
        f'nodes = "{module}:NODES"\ntypes = "{module}:register_types"\n'
    )
    if cold:
        assert prepare_catalog(manifest, environment=os.environ).ok

    async def scenario() -> None:
        composer = ServingComposer()
        client: TestClient | None = None
        try:
            await composer.add_pack(PackSpec(manifest, in_process=in_process, require_catalog=cold))
            registry = composer.composition._registry
            assert registry.type_owners()["mypack.thing"] == "mypack"
            if cold:
                assert composer._records["mypack"].worker.cold
                assert "mypack.thing" not in registry
            client = TestClient(
                TestServer(
                    create_app(composer.composition.make_engine, composer.composition.schemas)
                )
            )
            await client.start_server()
            data = await (await client.get("/api/nodes")).json()
            assert data["typeOwners"]["mypack.thing"] == "mypack"
            assert data["typeOwners"]["INT"] == "core"
            if cold:
                assert composer._records["mypack"].worker.cold
            if not in_process:
                original_manifest = manifest.read_text()
                manifest.write_text(
                    original_manifest.replace(
                        "[pack.entry]", 'type-namespaces = ["FOREIGN"]\n[pack.entry]'
                    )
                )
                with pytest.raises(CompositionError, match="must be prefixed"):
                    await composer.reload_pack("mypack")
                data = await (await client.get("/api/nodes")).json()
                assert data["typeOwners"]["mypack.thing"] == "mypack"
                manifest.write_text(original_manifest)
                source.write_text(source.read_text().replace("mypack.thing", "mypack.replacement"))
                if cold:
                    assert prepare_catalog(manifest, environment=os.environ).ok
                await composer.reload_pack("mypack")
                data = await (await client.get("/api/nodes")).json()
                assert "mypack.thing" not in data["typeOwners"]
                assert data["typeOwners"]["mypack.replacement"] == "mypack"
            await composer.remove_pack("mypack")
            data = await (await client.get("/api/nodes")).json()
            assert "mypack.thing" not in data["typeOwners"]
            assert "mypack.replacement" not in data["typeOwners"]
            assert data["typeOwners"]["INT"] == "core"
            assert registry.owner_of("mypack.thing") is None
            assert "mypack.thing" not in registry
        finally:
            if client is not None:
                await client.close()
            await composer.close()

    run_scenario(scenario())


def test_dependent_pack_references_shared_type_without_claiming_it(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    consumer = tmp_path / "consumer"
    owner.mkdir()
    consumer.mkdir()
    (owner / "shared_type_owner.py").write_text(
        "from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr\n"
        "def register_types(registry):\n"
        "    registry.register('owner.thing')\n"
        "class Produce(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='owner.produce', outputs=(\n"
        "            OutputSpec('thing', TypeExpr.concrete('owner.thing')),))\n"
        "    @classmethod\n"
        "    def execute(cls):\n"
        "        return cls.outputs(thing={'width': 11, 'height': 29})\n"
        "NODES = [Produce]\n"
    )
    (consumer / "shared_type_consumer.py").write_text(
        "from dinkster_api.v1 import Node, NodeSchema, InputSpec, OutputSpec, TypeExpr\n"
        "from dinkster_values import CORE_INT\n"
        "class Consume(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='consumer.consume', inputs=(\n"
        "            InputSpec('thing', TypeExpr.concrete('owner.thing')),), outputs=(\n"
        "            OutputSpec('difference', TypeExpr.concrete(CORE_INT)),))\n"
        "    @classmethod\n"
        "    def execute(cls, *, thing):\n"
        "        return cls.outputs(difference=thing['height'] - thing['width'])\n"
        "NODES = [Consume]\n"
    )
    owner_manifest = owner / "dinkster-pack.toml"
    owner_manifest.write_text(
        '[pack]\nname = "owner"\npublisher = "owner-author"\n'
        '[pack.entry]\nnodes = "shared_type_owner:NODES"\n'
        'types = "shared_type_owner:register_types"\n'
    )
    consumer_manifest = consumer / "dinkster-pack.toml"
    consumer_manifest.write_text(
        '[pack]\nname = "consumer"\npublisher = "consumer-author"\n'
        '[pack.dependencies]\nowner = ">=1,<2"\n'
        '[pack.entry]\nnodes = "shared_type_consumer:NODES"\n'
    )

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            with pytest.raises(CompositionError, match="requires pack 'owner'.*not composed"):
                await composer.add_pack(PackSpec(consumer_manifest, in_process=True))
            await composer.add_pack(
                PackSpec(
                    owner_manifest,
                    in_process=True,
                    packs={"owner": PackInfo(display_name="Owner", version="1.0.0")},
                )
            )
            await composer.add_pack(PackSpec(consumer_manifest, in_process=True))
            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(
                Graph(
                    nodes={
                        "produce": GraphNode("owner.produce"),
                        "consume": GraphNode(
                            "consumer.consume", {"thing": Link("produce", "thing")}
                        ),
                    }
                ),
                ["consume"],
            )
            assert result.outputs["consume"]["difference"].resolve() == 18
            assert composer.composition._registry.type_owners()["owner.thing"] == "owner"
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.usefixtures("unrestricted_cuda_devices")
@pytest.mark.parametrize("replica_cuda_indices", [(), (0, 1)], ids=["lazy", "replica-pool"])
def test_cold_catalog_pack_renditions_relay_through_composed_registry(
    tmp_path: Path, replica_cuda_indices: tuple[int, ...]
) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path)
        environment = {**os.environ, **WORKER_ENV}
        assert prepare_catalog(manifest, environment=environment).ok
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(
                PackSpec(
                    manifest,
                    require_catalog=True,
                    replica_cuda_indices=replica_cuda_indices,
                )
            )
            worker = composer._records["isopack"].worker
            assert worker.cold

            registry = composer.composition._registry
            spec = registry.renditions_of("iso.blob")[0]
            metadata = {"size": 3}
            assert await registry.rendition_mime(spec, metadata) == "text/plain"
            mime, parameters = await registry.resolve_rendition(spec, metadata, {"prefix": "pack"})
            assert (mime, parameters) == ("text/plain", {"prefix": "pack"})
            value = Value(
                type_id="iso.blob",
                fingerprint="iso-blob-test",
                meta=ValueMeta(metadata),
                payload=EncodedPayload(
                    "iso.blob",
                    default_encode({"n": 3, "data": "xxx"}),
                    None,
                ),
            )
            rendition = await registry.render_async(
                value,
                "summary",
                parameters,
            )
            assert (rendition.mime, rendition.data) == ("text/plain", b"pack:3:xxx")
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize(
    ("argument", "flag", "owner"),
    [
        ("--port", "--port", "Dinkster server"),
        ("--enable-cors-header=*", "--enable-cors-header", "Dinkster server"),
        ("--front-end-root=/tmp", "--front-end-root", "Dinkster server"),
        ("--reserve-vram", "--reserve-vram", "Dinkster memory/aimdo policy"),
        ("--gpu-only=true", "--gpu-only", "Dinkster memory/aimdo policy"),
        ("--novram", "--novram", "Dinkster memory/aimdo policy"),
        ("--vram-headroom=1", "--vram-headroom", "Dinkster memory/aimdo policy"),
        ("--bf16-unet", "--bf16-unet", "Dinkster dtype policy"),
        ("--por=8188", "--por", "Dinkster server"),
        ("--gpu-o", "--gpu-o", "Dinkster memory/aimdo policy"),
    ],
)
def test_pack_spec_comfy_args_deny_dinkster_owned_flags(
    tmp_path: Path, argument: str, flag: str, owner: str
) -> None:
    from dinkster_server import ComfyArgumentError

    from dinkster.compose import PackSpec

    with pytest.raises(ComfyArgumentError) as raised:
        PackSpec(write_iso_manifest(tmp_path), comfy_args=(argument,))
    assert raised.value.flag == flag
    assert raised.value.owner == owner
    assert flag in str(raised.value)
    assert owner in str(raised.value)


def test_serving_composer_incremental(tmp_path: Path) -> None:
    """The diagnostic host starts empty and grows through ordinary pack deltas."""

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            composition = composer.composition
            assert composition.schemas == {}
            assert composition.packs == {}
            engine = composition.make_engine(lambda event: None)

            app = create_app(composition.make_engine, composition.schemas)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                assert await (await client.get("/api/health")).json() == {"ok": True}
                nodes = await (await client.get("/api/nodes")).json()
                assert nodes["nodes"] == {}
            finally:
                await client.close()

            delta = await composer.add_pack(write_iso_manifest(tmp_path))
            assert delta.pack == "isopack"
            assert "iso.chatty" in delta.schemas
            assert delta.node_packs["iso.chatty"] == "isopack"
            assert delta.packs["isopack"].display_name == "Iso Pack"
            # Merged into the shared composition too (the all-at-once path).
            assert composition.schemas["iso.chatty"] is delta.schemas["iso.chatty"]
            assert composition.node_packs["iso.chatty"] == "isopack"

            # The engine made BEFORE the pack composed executes it after
            # announcement - no rebuild, no restart.
            engine.announce_schemas(delta.schemas)
            result = await engine.run(
                Graph(nodes={"c": GraphNode("iso.chatty", {"value": "hi"})}),
                ["c"],
            )
            assert result.outputs["c"]["value"].resolve() == "HI"
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("ending", ["success", "failure", "cancel", "native failure"])
@pytest.mark.parametrize("other_job", ["other job", "../main job"])
def test_composed_job_scratch_stays_live_between_nodes_until_run_settles(
    tmp_path: Path, ending: str, other_job: str
) -> None:
    async def scenario() -> None:
        scratch = tmp_path / "scratch"
        entered = {job: asyncio.Event() for job in ("main", "other")}
        resume = {job: asyncio.Event() for job in entered}
        released: list[str] = []

        class Policy:
            async def select(
                self,
                node_type: str,
                inputs: Mapping[str, Value],
                candidates: tuple[str, Mapping[str, str]],
                **context: object,
            ) -> None:
                value = inputs["value"].resolve()
                if value in ("FIRST-SECOND", "other"):
                    label = "other" if value == "other" else "main"
                    entered[label].set()
                    await resume[label].wait()
                    if label == "main" and ending == "failure":
                        raise RuntimeError("dispatch failed")

            def release_run(self, run_id: str) -> None:
                with claim_inactive_produced_job(scratch, run_id) as inactive:
                    assert not inactive, "scratch released before native run settlement"
                released.append(run_id)
                if len(released) == 1 and ending == "native failure":
                    raise RuntimeError("native settlement failed")

        composer = ServingComposer(
            pack_scratch_root=scratch, native_policy=Policy(), worker_env=WORKER_ENV
        )
        tasks: list[asyncio.Task[object]] = []
        try:
            await composer.add_pack(write_iso_manifest(tmp_path))
            engine = composer.composition.make_engine(lambda event: None)
            with pytest.raises(AssetError, match="ownership state is unavailable"):
                with claim_inactive_produced_job(scratch, "unknown job"):
                    pytest.fail("missing ownership state was treated as an idle job")
            main = asyncio.create_task(
                engine.run(
                    Graph(
                        nodes={
                            "first": GraphNode("iso.chatty", {"value": "first-second"}),
                            "second": GraphNode("iso.chatty", {"value": Link("first", "value")}),
                        }
                    ),
                    ["second"],
                    run_id="../main job",
                    attempt_id=2,
                )
            )
            other = asyncio.create_task(
                composer.composition.make_engine(lambda event: None).run(
                    Graph(nodes={"other": GraphNode("iso.chatty", {"value": "other"})}),
                    ["other"],
                    run_id=other_job,
                )
            )
            tasks.extend((main, other))
            async with asyncio.timeout(20):
                await asyncio.gather(*(event.wait() for event in entered.values()))
                for job in ("../main job", other_job):
                    with claim_inactive_produced_job(scratch, job) as inactive:
                        assert not inactive
                if ending == "cancel":
                    main.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await main
                else:
                    resume["main"].set()
                    if ending in ("failure", "native failure"):
                        with pytest.raises(
                            Exception, match="dispatch failed|native settlement failed"
                        ):
                            await main
                    else:
                        assert (await main).outputs["second"]["value"].resolve() == "FIRST-SECOND"
                with claim_inactive_produced_job(scratch, "../main job") as inactive:
                    assert inactive == (other_job != "../main job")
                with claim_inactive_produced_job(scratch, other_job) as inactive:
                    assert not inactive
                assert released == ["../main job"]
                resume["other"].set()
                assert (await other).outputs["other"]["value"].resolve() == "OTHER"
                with claim_inactive_produced_job(scratch, other_job) as inactive:
                    assert inactive
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await composer.close()

    run_scenario(scenario())


def test_broken_default_pack_leaves_zero_node_host_and_registry_healthy(tmp_path: Path) -> None:
    module = tmp_path / "broken_media.py"
    module.write_text(
        "def register_types(registry):\n"
        "    registry.register('broken.partial')\n"
        "    raise RuntimeError('broken media types')\n"
        "NODES = []\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "dinkster-nodes-media-io"\npublisher = "dinkster"\n'
        'type-namespaces = ["dinkster", "broken"]\nnamespaces = ["dinkster"]\n'
        '[pack.entry]\nnodes = "broken_media:NODES"\n'
        'types = "broken_media:register_types"\n',
        encoding="utf-8",
    )

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            with pytest.raises(RuntimeError, match="broken media types"):
                await composer.add_pack(PackSpec(manifest, in_process=True, trust_reserved=True))
            assert composer.composition.schemas == {}
            assert composer.composition.packs == {}
            assert "broken.partial" not in composer.composition._registry

            app = create_app(composer.composition.make_engine, composer.composition.schemas)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                assert await (await client.get("/api/health")).json() == {"ok": True}
                nodes = await (await client.get("/api/nodes")).json()
                assert nodes["nodes"] == {}
            finally:
                await client.close()
        finally:
            await composer.close()

    run_scenario(scenario())


def test_serving_composer_maintains_asset_catalog(tmp_path: Path) -> None:
    """[[pack.assets]] declarations ride composition onto the live asset
    catalog: add_pack registers the pack's artifact root and needs (and
    the packs-table wire carries the descriptors), reload swaps them for
    the new manifest's declarations, remove retracts everything - the
    SAME catalog object throughout, since ServerLibrary holds the
    reference for the process lifetime."""
    from dinkster_assets import PackagedSource, digest_bytes

    from dinkster.compose import ServingComposer

    model_bytes = b"iso aux model" * 64
    model_digest = digest_bytes(model_bytes)

    def write_manifest_with_assets(directory: Path) -> Path:
        manifest = write_iso_manifest(directory)
        (directory / "assets").mkdir(exist_ok=True)
        (directory / "assets" / "model.bin").write_bytes(model_bytes)
        manifest.write_text(
            manifest.read_text()
            + "\n[[pack.assets]]\n"
            + f'id = "aux-model"\nname = "Aux Model"\ndigest = "{model_digest}"\n'
            + 'kind = "model/auxiliary"\nfile = "assets/model.bin"\n'
            + 'urls = ["https://hub.example/aux.bin"]\nnodes = ["iso.chatty"]\n'
        )
        return manifest

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            catalog = composer.composition.asset_catalog
            pack_dir = tmp_path / "isopack"
            delta = await composer.add_pack(write_manifest_with_assets(pack_dir))

            # Descriptors ride the packs table; bytes never do.
            wire = delta.packs["isopack"].to_wire()
            assets = wire["assets"]
            assert isinstance(assets, list) and len(assets) == 1
            assert assets[0]["id"] == "aux-model"
            assert assets[0]["digest"] == model_digest
            assert assets[0]["nodes"] == ["iso.chatty"]

            # The catalog knows the root and the need, by digest and by
            # requiring node type.
            assert catalog.pack_roots() == {"isopack": pack_dir}
            need = catalog.need_for(model_digest)
            assert need is not None
            assert PackagedSource(pack="isopack", path="assets/model.bin") in (need.sources)
            assert set(catalog.needs_for_nodes({"iso.chatty"})) == {model_digest}

            # Reload against a manifest WITHOUT the declaration: the
            # catalog swaps to the new surface - no stale need survives.
            write_iso_manifest(pack_dir)
            await composer.reload_pack("isopack")
            assert catalog.need_for(model_digest) is None
            assert catalog.needs_for_nodes({"iso.chatty"}) == {}

            # And back: reload with the declaration restores it.
            write_manifest_with_assets(pack_dir)
            await composer.reload_pack("isopack")
            assert catalog.need_for(model_digest) is not None

            # Removal retracts declarations and root alike.
            await composer.remove_pack("isopack")
            assert catalog.need_for(model_digest) is None
            assert catalog.pack_roots() == {}
        finally:
            await composer.close()

    run_scenario(scenario())


def test_serving_composer_failed_add_pack_reaps_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An add_pack that fails AFTER its worker started (node type outside
    the declared namespaces) leaves the worker registered for cleanup:
    close() reaps every started worker, the earlier pack's included."""
    from dinkster_workers import IsolatedWorker

    from dinkster.compose import ServingComposer

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
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(write_iso_manifest(tmp_path / "a"))
            bad = write_iso_manifest(tmp_path / "b", name="isopack2", namespaces=None)
            bad.write_text(
                bad.read_text()
                .replace('type-namespaces = ["iso"]\n', "")
                .replace('types = "isopack_nodes:register_types"\n', "")
            )
            with pytest.raises(CompositionError, match="outside the pack's declared namespaces"):
                await composer.add_pack(bad)
        finally:
            await composer.close()
        assert len(started) == 2  # the bad pack failed only after its hello
        assert set(closed) == set(started)

    run_scenario(scenario())


def test_serving_composer_failed_add_pack_merges_nothing(tmp_path: Path) -> None:
    """add_pack is atomic: a pack failing composition AFTER its pack-table
    entry would have merged (node type outside its declared namespaces -
    the last validation) leaves the composition untouched - no schemas, no
    pack entry, no name or claim held. A continue-on-failure host serves
    survivors from an unpolluted table, and a LATER pack may reuse the
    failed pack's name and namespace."""
    from dinkster.compose import ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            composition = composer.composition
            bad = write_iso_manifest(tmp_path / "a", presentation=False, namespaces=None)
            with pytest.raises(CompositionError, match="outside the pack's declared namespaces"):
                await composer.add_pack(bad)
            assert "isopack" not in composition.packs
            assert not any(t.startswith("iso.") for t in composition.schemas)

            delta = await composer.add_pack(write_iso_manifest(tmp_path / "b"))
            assert delta.pack == "isopack"
            assert composition.packs["isopack"].display_name == "Iso Pack"
            assert "iso.chatty" in composition.schemas
        finally:
            await composer.close()

    run_scenario(scenario())


def test_failed_start_reaps_started_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A composition error after some packs started must not orphan their
    processes: every started worker is closed before the error escapes."""
    from dinkster_workers import IsolatedWorker

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
        good = write_iso_manifest(tmp_path / "a", presentation=False)
        bad = write_iso_manifest(
            tmp_path / "b",
            name="zzbad",
            presentation=False,
            namespaces=("zzbad",),
        )
        bad.write_text(
            bad.read_text()
            .replace("isopack_nodes:NODES", "missing_nodes:NODES")
            .replace('type-namespaces = ["iso"]\n', "")
            .replace('types = "isopack_nodes:register_types"\n', "")
        )
        with pytest.raises(RuntimeError, match="failed to start"):
            await compose_serving([good, bad], include_default_packs=False, worker_env=WORKER_ENV)
        assert started
        assert all(worker in closed for worker in started)

    run_scenario(scenario())


def test_group_contract_conflict_refuses_before_launch(tmp_path: Path) -> None:
    from dinkster.compose import PackSpec

    async def scenario() -> None:
        alpha = write_iso_manifest(tmp_path / "a", name="alpha", presentation=False)
        beta = write_iso_manifest(tmp_path / "b", name="beta", presentation=False)
        manifests = (alpha, beta)
        specs = (
            PackSpec(alpha, env=WORKER_ENV, worker_group="models", group_manifests=manifests),
            PackSpec(
                beta,
                env={**WORKER_ENV, "EXTRA": "different"},
                worker_group="models",
                group_manifests=manifests,
            ),
        )
        with pytest.raises(CompositionError, match="incompatible launch contracts"):
            await compose_serving(specs, include_default_packs=False)

    run_scenario(scenario())


# -- memory governance (DESIGN 3.10) ---------------------------------------
#
# The relay, lease, and shedding protocols are proven at the session level
# (test_relay.py, test_reservations.py). These tests prove the COMPOSITION
# wiring: a host-supplied governor and reservation service reach every
# worker the composer starts, so a pack's manifest-declared consumers and
# reservation planner are live on the production path, not just when a test
# constructs IsolatedWorker by hand.


def test_compose_serving_relays_pack_consumers_to_governor(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor()
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\npublisher = "dinkster"\n'
            'type-namespaces = ["mem"]\nnamespaces = ["mem"]\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:memory_consumers"\n'
        )
        composition = await compose_serving(
            [manifest],
            include_default_packs=False,
            worker_env=WORKER_ENV,
            governor=governor,
            reservations=GovernorReservationService(governor),
        )
        try:
            engine = composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "mem.load",
                        {"name": "sd15.safetensors", "vram": 400, "ram": 300},
                    )
                }
            )
            result = await engine.run(graph, ["load"])
            assert "load" in result.outputs
            # The child's snapshot push races the result frame: poll until
            # the relayed consumer's footprint lands in the parent governor.
            async with asyncio.timeout(8):
                while governor.footprint("vram:cuda:0") != 400:
                    await asyncio.sleep(0.02)
            assert governor.footprint("ram") == 300
        finally:
            await composition.close()

    run_scenario(scenario())


def test_serving_composer_full_free_releases_live_worker_without_startup_guard(
    tmp_path: Path,
    isolated_full_free_pool: ResidentPool,
) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 1000})
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\npublisher = "dinkster"\n'
            'type-namespaces = ["mem"]\nnamespaces = ["mem"]\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:memory_consumers"\n'
        )
        composer = ServingComposer(
            worker_env=WORKER_ENV,
            governor=governor,
            dev=True,
            cache_mode="layered",
            cache_dir=tmp_path / "execution-cache",
        )
        try:
            await composer.add_pack(PackSpec(DEV_PACK_MANIFEST, trust_reserved=True))
            await composer.add_pack(manifest)
            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            gradient = Graph(
                nodes={"g": GraphNode("dev.image.gradient", {"width": 8, "height": 4})}
            )
            assert (await engine.run(gradient, ["g"])).executed == ("g",)
            graph = Graph(
                nodes={
                    "load": GraphNode(
                        "mem.load",
                        {"name": "sd15.safetensors", "vram": 0, "ram": 0},
                    )
                }
            )
            await engine.run(graph, ["load"])

            async with asyncio.timeout(5):
                while True:
                    results = await composer.full_free("maintenance-composed")
                    worker = next(row for row in results[1:] if row["worker"] == "memorypack")
                    if worker["status"] != "busy":
                        break
                    await asyncio.sleep(0)
            local = results[0]
            assert local["status"] == "complete"
            assert worker["worker"] == "memorypack"
            assert worker["status"] == "complete"
            consumers = cast("list[dict[str, object]]", worker["consumers"])
            assert {item["consumer"] for item in consumers} == {
                "models",
                "zero-cost",
            }
            events.clear()
            assert (await engine.run(gradient, ["g"])).cached == ("g",)
            disk_hit = next(event for event in events if event.kind == "node_cached")
            assert disk_hit.detail["cacheLayer"] == "disk"

            use = Graph(
                nodes={
                    "load": GraphNode("mem.load", {"name": "sd15.safetensors"}),
                    "use": GraphNode("mem.use", {"model": Link("load", "model")}),
                }
            )
            rerun = await engine.run(use, ["use"])
            assert "load" in rerun.executed
            assert rerun.outputs["use"]["name"].resolve() == "sd15.safetensors"

            live_worker = composer._records["memorypack"].worker
            await live_worker.close()
            results = await composer.full_free("maintenance-missing")
            local = results[0]
            missing = next(row for row in results[1:] if row["worker"] == "memorypack")
            assert local["status"] == "complete"
            assert missing == {
                "worker": "memorypack",
                "workerInstance": None,
                "deviceMap": {"mapping": {}, "qualifier": None},
                "status": "error",
                "error": "worker is unavailable at the maintenance snapshot",
                "consumers": [],
            }
        finally:
            await composer.close()

    run_scenario(scenario())


def test_serving_composer_full_free_serializes_group_members_by_process(
    tmp_path: Path,
    isolated_full_free_pool: ResidentPool,
) -> None:
    async def scenario() -> None:
        manifests: list[Path] = []
        for name in ("alpha", "beta"):
            directory = tmp_path / name
            directory.mkdir()
            module = directory / f"{name}_nodes.py"
            module.write_text(
                "from dinkster_schema import Node,NodeSchema\n"
                "class Empty(Node):\n"
                "    @classmethod\n"
                "    def define_schema(cls):\n"
                f"        return NodeSchema(node_type='{name}.empty')\n"
                "    @classmethod\n"
                "    async def execute(cls):\n"
                "        return cls.outputs()\n"
                "NODES=[Empty]\n"
            )
            manifest = directory / "dinkster-pack.toml"
            manifest.write_text(
                f'[pack]\nname = "{name}"\nnamespaces = ["{name}"]\n\n'
                f'[pack.entry]\nnodes = "{name}_nodes:NODES"\n'
                'consumers = "memorypack_nodes:serial_memory_consumers"\n'
            )
            manifests.append(manifest)
        env = {
            "PYTHONPATH": os.pathsep.join(
                (str(TESTS_DIR), *(str(manifest.parent) for manifest in manifests))
            )
        }
        composer = ServingComposer()
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest,
                        env=env,
                        worker_group="models",
                        group_manifests=tuple(manifests),
                    )
                )
            workers = await composer.full_free("group-members")
            grouped = [worker for worker in workers if worker["worker"] in {"alpha", "beta"}]
            assert len(grouped) == 2
            assert len({worker["workerInstance"] for worker in grouped}) == 1
            assert all(worker["status"] == "complete" for worker in grouped)
            assert all(
                worker["consumers"] == [{"consumer": "serial", "status": "complete"}]
                for worker in grouped
            )
        finally:
            await composer.close()

    run_scenario(scenario())


def test_compose_serving_delivers_pack_telemetry(tmp_path: Path) -> None:
    async def scenario() -> None:
        reported = ReportedTelemetry()
        governor = MemoryGovernor(telemetry=reported.probe, telemetry_devices=reported.devices)
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\npublisher = "dinkster"\n'
            'type-namespaces = ["mem"]\nnamespaces = ["mem"]\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:memory_consumers"\n'
            'telemetry = "memorypack_nodes:memory_telemetry"\n'
        )
        composition = await compose_serving(
            [manifest],
            include_default_packs=False,
            worker_env=WORKER_ENV,
            governor=governor,
            telemetry=reported,
        )
        try:
            # Measurements rode the worker's hello: no invocation needed.
            assert reported.probe("vram:cuda:0") == MeasuredMemory(
                free_bytes=8_000, total_bytes=10_000
            )
            # The governor reports the measured, unbudgeted device.
            status = governor.status()
            assert status["vram:cuda:0"]["budgetBytes"] is None
            assert status["vram:cuda:0"]["measured"] == {
                "freeBytes": 8_000,
                "totalBytes": 10_000,
            }
        finally:
            await composition.close()
        # Composition close closes every worker: measurements die with it.
        assert reported.devices() == frozenset()

    run_scenario(scenario())


@pytest.mark.parametrize(
    "drift",
    (
        "valid",
        "presentation",
        "type",
        "required",
        "default",
        "lazy",
        "missing",
        "undeclared",
        "closed",
        "declared",
    ),
)
def test_authored_input_worker_validates_its_own_declaration(drift: str) -> None:
    from dinkster_protocol import Invocation
    from dinkster_schema import (
        InputSpec,
        Node,
        NodeSchema,
        OutputSpec,
        TypeExpr,
        elaborate,
    )
    from dinkster_values import register_core_types
    from dinkster_workers import InProcessWorker

    calls: list[dict[str, object]] = []
    declaration = NodeSchema(
        node_type="authored.inputs",
        inputs=(InputSpec("choice", TypeExpr.concrete("STRING")),),
        outputs=(OutputSpec("result", TypeExpr.concrete("STRING")),),
        additional_inputs=TypeExpr.wildcard(),
    )

    class AuthoredInputs(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return (
                replace(declaration, additional_inputs=None) if drift == "closed" else declaration
            )

        @classmethod
        def execute(cls, **kwargs: object) -> Mapping[str, object]:
            calls.append(kwargs)
            return cls.outputs(result=f"{kwargs['choice']}|{kwargs['index']}")

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        effective = elaborate(declaration, ("choice", "index"))
        choice, index = effective.inputs
        if drift == "type":
            index = replace(index, type=TypeExpr.concrete("INT"))
        elif drift == "required":
            index = replace(index, required=True)
        elif drift == "default":
            index = replace(index, default=91)
        elif drift == "lazy":
            index = replace(index, lazy=True)
        elif drift == "declared":
            choice = replace(choice, required=False)
        elif drift == "presentation":
            choice = replace(
                choice,
                display_name="Authored choice",
                doc="Choose the authored value.",
            )
        effective = replace(effective, inputs=(choice, index))
        if drift == "missing":
            effective = replace(effective, inputs=(choice,))
        inputs = {
            "choice": registry.wrap("STRING", "outside-menu"),
            "index": registry.wrap("INT", 7),
        }
        if drift == "undeclared":
            inputs["unlisted"] = registry.wrap("INT", 13)
        worker = InProcessWorker({declaration.node_type: AuthoredInputs}, registry)
        try:
            result = await worker.invoke(
                Invocation(
                    invocation_id="authored-inputs",
                    node_id="node",
                    node_type=declaration.node_type,
                    inputs=inputs,
                    effective_schema=effective,
                )
            )
            if drift in {"valid", "presentation"}:
                assert result.error is None
                assert result.outputs is not None
                assert result.outputs["result"].resolve() == "outside-menu|7"
                assert calls == [{"choice": "outside-menu", "index": 7}]
            else:
                assert result.error is not None
                assert "schema skew" in result.error.message
                assert result.outputs is None
                assert calls == []
        finally:
            await worker.close()

    run_scenario(scenario())


def test_compose_serving_wires_worker_reservations(tmp_path: Path) -> None:
    async def scenario() -> None:
        governor = MemoryGovernor({"ram": 100})
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "isopack"\npublisher = "dinkster"\n'
            'type-namespaces = ["iso"]\nnamespaces = ["iso"]\n\n[pack.entry]\n'
            'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
            'reservations = "isopack_nodes:plan_reservations"\n'
        )
        composition = await compose_serving(
            [manifest],
            include_default_packs=False,
            worker_env=WORKER_ENV,
            governor=governor,
            reservations=GovernorReservationService(governor),
        )
        try:
            engine = composition.make_engine(lambda event: None)
            graph = Graph(nodes={"hog": GraphNode("iso.hog", {"nbytes": 60, "seconds": 0.3})})
            task = asyncio.create_task(engine.run(graph, ["hog"]))
            # The planned lease is visible in the parent's governor while
            # the child executes...
            async with asyncio.timeout(8):
                while governor.reserved("ram") != 60:
                    await asyncio.sleep(0.005)
            result = await task
            assert "hog" in result.outputs
            # ...and released once the invocation settles.
            async with asyncio.timeout(8):
                while governor.reserved("ram") != 0:
                    await asyncio.sleep(0.005)
        finally:
            await composition.close()

    run_scenario(scenario())
