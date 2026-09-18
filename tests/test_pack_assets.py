"""Pack-declared assets ([[pack.assets]] - templates/asset distribution).

A pack that needs model files declares them as DATA in its manifest -
digest-pinned, packaged in the artifact and/or behind remote URLs, with
optional node-type associations - instead of shipping downloader code.
What this proves: the manifest parser validates and warns-and-drops per
entry; the composed surface's catalog indexes declarations by digest and
by requiring node type and swaps atomically on add/reload/remove; job
preflight turns "this graph instantiates that node" into the existing
409 plan -> digest-exact consent -> verified acquisition flow; and
nothing ever downloads at composition time or without consent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    KIND_MODEL_CHECKPOINT,
    AssetComponent,
    AssetComponentManifest,
    AssetNeed,
    AssetVault,
    DeclaredAsset,
    LibraryStore,
    PackagedSource,
    PackAssetCatalog,
    ProvenanceRecord,
    ProvenanceStore,
    RemoteSource,
    digest_bytes,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
from dinkster_graph import Graph, GraphNode, Link, RegionNode, TypedLiteral, graph_to_wire
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import ServerLibrary, create_app
from dinkster_server.preflight import graph_node_types, graph_provider_selections
from dinkster_values import CORE_INT, TypeRegistry, register_core_types
from dinkster_workers import (
    InProcessWorker,
    load_manifest,
    load_pack_assets,
    validate_pack_asset,
)
from dinkster_workers.staging import declared_assets_from_wire

from tests.platform_support import symlink_or_skip

MODEL_BYTES = b"aux model bytes for pack asset tests" * 32
MODEL_DIGEST = digest_bytes(MODEL_BYTES)
OTHER_BYTES = b"a different auxiliary model"
OTHER_DIGEST = digest_bytes(OTHER_BYTES)


# --- manifest validation --------------------------------------------------------


def write_pack_dir(tmp_path: Path) -> Path:
    pack = tmp_path / "pack"
    (pack / "assets").mkdir(parents=True)
    (pack / "assets" / "model.bin").write_bytes(MODEL_BYTES)
    return pack


def test_validate_pack_asset_shapes(tmp_path: Path) -> None:
    """Packaged-only, remote-only, and both-sourced declarations validate;
    the need carries the descriptive fields and node types verbatim
    (legacy node types are not grammar-valid names, so no grammar check)."""
    pack = write_pack_dir(tmp_path)
    manifest_path = pack / "dinkster-pack.toml"

    both, problem = validate_pack_asset(
        manifest_path,
        {
            "id": "lineart",
            "name": "Anime Lineart v2",
            "digest": MODEL_DIGEST,
            "kind": "model/auxiliary",
            "size": len(MODEL_BYTES),
            "media_type": "application/octet-stream",
            "file": "assets/model.bin",
            "urls": ["https://hub.example/lineart.bin"],
            "nodes": ["comfy.controlnet_aux.LineartPreprocessor"],
        },
        "aux",
    )
    assert problem is None and both is not None
    assert both.id == "lineart"
    assert both.need.name == "Anime Lineart v2"
    assert both.need.digest == MODEL_DIGEST
    assert both.need.kind == "model/auxiliary"
    assert both.need.size == len(MODEL_BYTES)
    assert both.need.media_type == "application/octet-stream"
    assert both.nodes == ("comfy.controlnet_aux.LineartPreprocessor",)
    assert both.need.sources == (
        PackagedSource(pack="aux", path="assets/model.bin"),
        RemoteSource("https://hub.example/lineart.bin"),
    )

    packaged_only, problem = validate_pack_asset(
        manifest_path,
        {"id": "p", "name": "P", "digest": MODEL_DIGEST, "file": "assets/model.bin"},
        "aux",
    )
    assert problem is None and packaged_only is not None
    assert packaged_only.need.sources == (PackagedSource(pack="aux", path="assets/model.bin"),)
    assert packaged_only.nodes == ()

    remote_only, problem = validate_pack_asset(
        manifest_path,
        {
            "id": "r",
            "name": "R",
            "digest": MODEL_DIGEST,
            "urls": ["https://hub.example/r.bin"],
        },
        "aux",
    )
    assert problem is None and remote_only is not None
    assert remote_only.need.sources == (RemoteSource("https://hub.example/r.bin"),)


def test_validate_pack_asset_rejects(tmp_path: Path) -> None:
    pack = write_pack_dir(tmp_path)
    manifest_path = pack / "dinkster-pack.toml"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(MODEL_BYTES)
    symlink_or_skip(pack / "leak.bin", outside)
    good = {
        "id": "m",
        "name": "M",
        "digest": MODEL_DIGEST,
        "urls": ["https://hub.example/m.bin"],
    }
    cases: list[tuple[object, str]] = [
        ("not a table", "must be a table"),
        ({**good, "id": ""}, "non-empty string 'id'"),
        ({**good, "id": "Bad ID"}, "lowercase"),
        ({**good, "name": ""}, "non-empty string 'name'"),
        ({k: v for k, v in good.items() if k != "digest"}, "must declare a 'digest'"),
        ({**good, "digest": "sha256:abc"}, "digest"),
        ({**good, "kind": "no-slash"}, "not a valid asset kind"),
        ({**good, "kind": 7}, "'kind' must be a string"),
        ({**good, "size": -2}, "'size' must be a non-negative integer"),
        ({**good, "size": True}, "'size' must be a non-negative integer"),
        ({**good, "media_type": 5}, "'media_type' must be a string"),
        ({**good, "components": "model/vae"}, "component manifest must be a list"),
        ({**good, "components": [{"kind": "vae"}]}, "not a valid asset kind"),
        ({**good, "nodes": "x"}, "'nodes' must be a list"),
        ({**good, "nodes": [""]}, "'nodes' must be a list"),
        ({**good, "file": str(outside)}, "relative path"),
        ({**good, "file": "../outside.bin"}, "relative path"),
        ({**good, "file": "leak.bin"}, "escape the pack"),
        ({**good, "file": "assets/missing.bin"}, "does not exist"),
        ({**good, "urls": ["ftp://nope"]}, "http(s)"),
        ({**good, "urls": "https://hub.example/m.bin"}, "'urls' must be a list"),
        ({k: v for k, v in good.items() if k != "urls"}, "at least one source"),
    ]
    for entry, expected in cases:
        asset, problem = validate_pack_asset(manifest_path, entry, "aux")
        assert asset is None and problem is not None, entry
        assert expected in problem, (entry, problem)


def test_manifest_assets_warn_and_drop(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A malformed [[pack.assets]] entry warns and drops WITHOUT touching
    its siblings or stopping the pack; duplicate ids keep the first."""
    pack = write_pack_dir(tmp_path)
    declared = pack / "dinkster-pack.toml"
    declared.write_text(
        "[pack]\n"
        'name = "aux"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[[pack.assets]]\n"
        f'id = "good"\nname = "Good"\ndigest = "{MODEL_DIGEST}"\n'
        'file = "assets/model.bin"\n'
        "[[pack.assets]]\n"
        f'id = "broken"\nname = "Broken"\ndigest = "{MODEL_DIGEST}"\n'
        'file = "assets/missing.bin"\n'
        "[[pack.assets]]\n"
        f'id = "good"\nname = "Duplicate"\ndigest = "{OTHER_DIGEST}"\n'
        'urls = ["https://hub.example/dup.bin"]\n'
        "[[pack.assets]]\n"
        f'id = "other"\nname = "Other"\ndigest = "{OTHER_DIGEST}"\n'
        'urls = ["https://hub.example/other.bin"]\n'
    )
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        manifest = load_manifest(declared)
    assert [asset.id for asset in manifest.assets] == ["good", "other"]
    assert manifest.assets[0].need.name == "Good"  # first declaration wins
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "does not exist" in messages and "duplicates" in messages
    # Packaged sources resolve under the manifest's pack id.
    assert manifest.assets[0].need.sources == (PackagedSource(pack="aux", path="assets/model.bin"),)


def test_load_pack_assets(tmp_path: Path) -> None:
    """The assets-only loader: reads [[pack.assets]] from a file that need
    not be a loadable pack manifest - the on-ramp for unported legacy
    packs, riding the same presentation-only dinkster-pack.toml as icons and
    blueprints. Advisory throughout: missing file or no declaration is ()."""
    pack = write_pack_dir(tmp_path)
    assets_only = pack / "dinkster-pack.toml"
    assets_only.write_text(
        "[pack]\n"
        "[[pack.assets]]\n"
        f'id = "m"\nname = "M"\ndigest = "{MODEL_DIGEST}"\n'
        'file = "assets/model.bin"\n'
    )
    assets = load_pack_assets(assets_only, pack="comfy.legacy_aux")
    assert [asset.id for asset in assets] == ["m"]
    # Packaged sources resolve under the id the HOST hands in - for
    # legacy packs, the synthetic comfy.<dir> id the spec's asset_roots
    # maps to the pack directory.
    assert assets[0].need.sources == (
        PackagedSource(pack="comfy.legacy_aux", path="assets/model.bin"),
    )

    assert load_pack_assets(tmp_path / "missing.toml", pack="x") == ()
    no_declaration = tmp_path / "plain.toml"
    no_declaration.write_text('[pack]\nname = "p"\n')
    assert load_pack_assets(no_declaration, pack="p") == ()


def test_pack_info_from_manifest_assets(tmp_path: Path) -> None:
    """Manifest assets cross the umbrella bridge verbatim and ride the
    packs table as descriptors: identity + leads, never bytes."""
    from dinkster.packs import pack_info_from_manifest

    pack = write_pack_dir(tmp_path)
    declared = pack / "dinkster-pack.toml"
    declared.write_text(
        "[pack]\n"
        'name = "aux"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[[pack.assets]]\n"
        f'id = "lineart"\nname = "Lineart"\ndigest = "{MODEL_DIGEST}"\n'
        'kind = "model/auxiliary"\n'
        f"size = {len(MODEL_BYTES)}\n"
        'file = "assets/model.bin"\nurls = ["https://hub.example/l.bin"]\n'
        'nodes = ["aux.lineart"]\n'
    )
    info = pack_info_from_manifest(load_manifest(declared))
    assert len(info.assets) == 1
    wire = info.to_wire()
    assert wire["assets"] == [
        {
            "id": "lineart",
            "name": "Lineart",
            "digest": MODEL_DIGEST,
            "kind": "model/auxiliary",
            "size": len(MODEL_BYTES),
            "sources": [
                {"type": "packaged", "pack": "aux", "path": "assets/model.bin"},
                {"type": "remote", "url": "https://hub.example/l.bin"},
            ],
            "nodes": ["aux.lineart"],
        }
    ]


def test_checkpoint_components_cross_declared_asset_transport(tmp_path: Path) -> None:
    manifest = AssetComponentManifest(
        (AssetComponent("model/diffusion", architecture="sd15"), AssetComponent("model/vae"))
    )
    declared, problem = validate_pack_asset(
        write_pack_dir(tmp_path) / "dinkster-pack.toml",
        {
            "id": "checkpoint",
            "name": "SD 1.5",
            "digest": MODEL_DIGEST,
            "kind": KIND_MODEL_CHECKPOINT,
            "components": manifest.to_wire(),
            "urls": ["https://models.example/sd15.safetensors"],
        },
        "models",
    )
    assert problem is None and declared is not None
    assert declared.need.component_manifest == manifest

    decoded = declared_assets_from_wire([declared.descriptor()])
    assert decoded == (declared,)
    assert decoded[0].need.matches_kind("model/vae")

    catalog = PackAssetCatalog()
    catalog.replace_all(
        [
            (
                "legacy",
                None,
                [
                    DeclaredAsset(
                        id="legacy-checkpoint",
                        need=AssetNeed(name="SD 1.5", digest=MODEL_DIGEST),
                    )
                ],
            ),
            ("typed", None, [declared]),
        ]
    )
    merged = catalog.need_for(MODEL_DIGEST)
    assert merged is not None
    assert merged.kind == KIND_MODEL_CHECKPOINT
    assert merged.component_manifest == manifest
    assert merged.matches_kind("model/vae")


# --- the catalog ----------------------------------------------------------------


def declared(
    asset_id: str,
    digest: str,
    *,
    pack: str,
    nodes: tuple[str, ...] = (),
    url: str = "",
    path: str = "",
    kind: str = "",
    metadata: Mapping[str, object] | None = None,
    component_manifest: AssetComponentManifest | None = None,
) -> DeclaredAsset:
    sources: list[PackagedSource | RemoteSource] = []
    if path:
        sources.append(PackagedSource(pack=pack, path=path))
    if url:
        sources.append(RemoteSource(url))
    return DeclaredAsset(
        id=asset_id,
        need=AssetNeed(
            name=asset_id,
            digest=digest,
            kind=kind,
            metadata=metadata or {},
            sources=tuple(sources),
            component_manifest=component_manifest,
        ),
        nodes=nodes,
    )


def test_declared_asset_requires_digest() -> None:
    with pytest.raises(Exception, match="digest"):
        DeclaredAsset(id="m", need=AssetNeed(name="m"))


def test_catalog_indexes_and_swaps(tmp_path: Path) -> None:
    catalog = PackAssetCatalog()
    catalog.replace_all(
        [
            (
                "aux",
                tmp_path / "aux",
                [
                    declared(
                        "lineart",
                        MODEL_DIGEST,
                        pack="aux",
                        nodes=("aux.lineart",),
                        path="assets/m.bin",
                        kind="model/auxiliary",
                    ),
                    declared("spare", OTHER_DIGEST, pack="aux"),
                ],
            ),
            (
                "mirror",
                None,
                [
                    declared(
                        "lineart-mirror",
                        MODEL_DIGEST,
                        pack="mirror",
                        nodes=("mirror.lineart",),
                        url="https://mirror.example/m.bin",
                    )
                ],
            ),
        ]
    )
    assert catalog.pack_roots() == {"aux": tmp_path / "aux"}

    # One digest declared by two packs is ONE need with merged mirrors;
    # descriptive fields come from the first declaration that filled them.
    merged = catalog.need_for(MODEL_DIGEST)
    assert merged is not None
    assert merged.sources == (
        PackagedSource(pack="aux", path="assets/m.bin"),
        RemoteSource("https://mirror.example/m.bin"),
    )
    assert merged.kind == "model/auxiliary"
    assert catalog.need_for(digest_bytes(b"nobody declared this")) is None

    # Node-type index: only the asked-for associations contribute (the
    # mirror pack's declaration requires mirror.lineart, not asked for -
    # preflight still gets its mirror via need_for), and an unused
    # declaration ("spare", no nodes) never becomes a requirement.
    needs = catalog.needs_for_nodes({"aux.lineart", "std.math.add_ints"})
    assert set(needs) == {MODEL_DIGEST}
    assert needs[MODEL_DIGEST].sources == (PackagedSource(pack="aux", path="assets/m.bin"),)
    assert catalog.needs_for_nodes({"std.math.add_ints"}) == {}

    # replace_all swaps wholesale: a removed pack's declarations and root
    # vanish; the same catalog OBJECT serves the new surface (the library
    # holds the reference for the process lifetime).
    catalog.replace_all(
        [
            (
                "mirror",
                None,
                [declared("m", MODEL_DIGEST, pack="mirror", url="https://mirror.example/m.bin")],
            )
        ]
    )
    assert catalog.pack_roots() == {}
    survivor = catalog.need_for(MODEL_DIGEST)
    assert survivor is not None
    assert survivor.sources == (RemoteSource("https://mirror.example/m.bin"),)
    assert catalog.needs_for_nodes({"aux.lineart"}) == {}


def test_graph_node_types_scans_regions() -> None:
    inner = Graph(nodes={"i": GraphNode("test.inner", {"x": 1})})
    graph = Graph(
        nodes={
            "a": GraphNode("test.outer", {"x": 2}),
            "r": RegionNode(
                kind="map",
                body=inner,
                ports={"xs": TypeExpr.concrete(CORE_INT)},
                inputs={"xs": [1, 2]},
                element_ports=("xs",),
            ),
        }
    )
    # Region pseudo-nodes are structure, not node types; bodies scan fully.
    assert graph_node_types(graph) == {"test.outer", "test.inner"}


def test_provider_assets_follow_literal_and_linked_selections() -> None:
    first = declared("first", MODEL_DIGEST, pack="first")
    second = declared("second", OTHER_DIGEST, pack="second")
    catalog = PackAssetCatalog()
    catalog.replace_all(
        [("first", None, [first]), ("second", None, [second])],
        provider_entries=[
            ("test.depth", "first", [first]),
            ("test.depth", "second", [second]),
        ],
    )
    literal_graph = Graph(
        nodes={
            "a": GraphNode("test.depth", {"provider": "first"}),
            "b": GraphNode(
                "test.depth",
                {"provider": TypedLiteral("core.combo", "second")},
            ),
        }
    )
    selections, linked = graph_provider_selections(literal_graph)
    assert selections == {("test.depth", "first"), ("test.depth", "second")}
    assert linked == set()
    assert set(catalog.needs_for_nodes((), provider_selections=selections)) == {
        MODEL_DIGEST,
        OTHER_DIGEST,
    }

    linked_graph = Graph(
        nodes={
            "choice": GraphNode("test.choice", {}),
            "depth": GraphNode("test.depth", {"provider": Link("choice", "value")}),
        }
    )
    selections, linked = graph_provider_selections(linked_graph)
    assert selections == set()
    assert linked == {"test.depth"}
    assert set(catalog.needs_for_nodes((), linked_provider_nodes=linked)) == {
        MODEL_DIGEST,
        OTHER_DIGEST,
    }

    linked_model_graph = Graph(
        nodes={
            "choice": GraphNode("test.choice", {}),
            "depth": GraphNode(
                "test.depth",
                {"model": Link("choice", "value")},
            ),
        }
    )
    selections, linked = graph_provider_selections(linked_model_graph)
    assert selections == set()
    assert linked == {"test.depth"}
    assert set(catalog.needs_for_nodes((), linked_provider_nodes=linked)) == {
        MODEL_DIGEST,
        OTHER_DIGEST,
    }


# --- job preflight through the server -------------------------------------------

INT = TypeExpr.concrete(CORE_INT)


class Lineart(Node):
    """A node with a FIXED internal model: no asset input, no dropdown -
    the manifest association is the only thing that makes the model a
    preflight requirement."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.lineart",
            inputs=(InputSpec("x", INT),),
            outputs=(OutputSpec("y", INT),),
        )

    @classmethod
    def execute(cls, *, x: int) -> Mapping[str, object]:
        return cls.outputs(y=x + 1)


class Plain(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.plain",
            inputs=(InputSpec("x", INT),),
            outputs=(OutputSpec("y", INT),),
        )

    @classmethod
    def execute(cls, *, x: int) -> Mapping[str, object]:
        return cls.outputs(y=x * 2)


class ProviderVision(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.provider_vision",
            inputs=(
                InputSpec("x", INT),
                InputSpec("provider", TypeExpr.concrete("core.combo")),
            ),
            outputs=(OutputSpec("y", INT),),
        )

    @classmethod
    def execute(cls, *, x: int, provider: str) -> Mapping[str, object]:
        return cls.outputs(y=x + len(provider))


NODES = (Lineart, Plain, ProviderVision)
SCHEMAS = build_schemas(NODES)


async def make_client(
    tmp_path: Path, catalog: PackAssetCatalog
) -> tuple[TestClient, ServerLibrary]:
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        provenance=ProvenanceStore(tmp_path / "provenance.json"),
        pack_assets=catalog,
    )

    def make_engine(on_event: EventListener | None = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas=SCHEMAS,
            registry=registry,
            worker=InProcessWorker(build_node_types(NODES), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    app = create_app(make_engine, SCHEMAS, library=library)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, library


def submit_body(graph: Graph, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "clientId": "c1",
        "jobId": "j1",
        "graph": graph_to_wire(graph),
        "targets": ["n"],
    }
    body.update(overrides)
    return body


async def wait_terminal(client: TestClient, job_id: str = "j1") -> dict[str, object]:
    for _ in range(500):
        status = await (await client.get(f"/api/jobs/c1/{job_id}")).json()
        if status["state"] in ("completed", "failed", "cancelled"):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("job never reached a terminal state")


def lineart_catalog(pack_root: Path, *, bytes_ok: bool = True) -> PackAssetCatalog:
    (pack_root / "assets").mkdir(parents=True, exist_ok=True)
    (pack_root / "assets" / "model.bin").write_bytes(
        MODEL_BYTES if bytes_ok else b"wrong bytes entirely"
    )
    catalog = PackAssetCatalog()
    catalog.replace_all(
        [
            (
                "aux",
                pack_root,
                [
                    declared(
                        "lineart",
                        MODEL_DIGEST,
                        pack="aux",
                        nodes=("test.lineart",),
                        path="assets/model.bin",
                        kind="model/auxiliary",
                    )
                ],
            )
        ]
    )
    return catalog


def test_node_declared_asset_gates_submit_then_consent_runs(tmp_path: Path) -> None:
    """The controlnet-aux story end to end: instantiating the node makes
    its fixed model a requirement (409 with a plan naming the packaged
    source), consenting to that exact digest acquires it out of the
    installed pack - verified, no network - and the job runs."""

    async def scenario() -> None:
        catalog = lineart_catalog(tmp_path / "packs" / "aux")
        client, library = await make_client(tmp_path, catalog)
        try:
            graph = Graph(nodes={"n": GraphNode("test.lineart", {"x": 1})})
            resp = await client.post("/api/jobs", json=submit_body(graph))
            assert resp.status == 409
            plan = await resp.json()
            assert plan["error"] == "assets-missing"
            assert plan["assets"] == [
                {
                    "digest": MODEL_DIGEST,
                    "name": "lineart",
                    "status": "missing",
                    "sources": [],
                    "fetchable": True,  # packaged: no URL needed
                    "kind": "model/auxiliary",
                    "packagedFrom": ["aux"],
                }
            ]
            # Nothing queued, nothing landed without consent.
            assert (await client.get("/api/jobs/c1/j1")).status == 404
            assert library.vault.resolve(MODEL_DIGEST) is None

            resp = await client.post(
                "/api/jobs",
                json=submit_body(graph, acquireAssets=[MODEL_DIGEST]),
            )
            assert resp.status == 202
            assert library.vault.resolve(MODEL_DIGEST) is not None
            status = await wait_terminal(client)
            assert status["state"] == "completed"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_checkpoint_manifest_is_exposed_in_missing_asset_plan(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = AssetComponentManifest(
            (
                AssetComponent("model/diffusion", architecture="sdxl", dtype="fp16"),
                AssetComponent("model/vae", metadata={"scale": 0.13025}),
            )
        )
        checkpoint = declared(
            "checkpoint",
            MODEL_DIGEST,
            pack="models",
            nodes=("test.lineart",),
            url="https://models.example/sdxl.safetensors",
            kind=KIND_MODEL_CHECKPOINT,
            metadata={"family": "sdxl"},
            component_manifest=manifest,
        )
        catalog = PackAssetCatalog()
        catalog.replace_all([("models", None, [checkpoint])])
        client, _library = await make_client(tmp_path, catalog)
        try:
            graph = Graph(nodes={"n": GraphNode("test.lineart", {"x": 1})})
            response = await client.post("/api/jobs", json=submit_body(graph))
            assert response.status == 409
            entry = (await response.json())["assets"][0]
            assert entry["kind"] == KIND_MODEL_CHECKPOINT
            assert entry["metadata"] == {"family": "sdxl"}
            assert entry["components"] == manifest.to_wire()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_provider_selection_preflights_only_its_artifacts(tmp_path: Path) -> None:
    async def scenario() -> None:
        first = declared("first", MODEL_DIGEST, pack="first")
        second = declared("second", OTHER_DIGEST, pack="second")
        catalog = PackAssetCatalog()
        catalog.replace_all(
            [("first", None, [first]), ("second", None, [second])],
            provider_entries=[
                ("test.provider_vision", "first", [first]),
                ("test.provider_vision", "second", [second]),
            ],
        )
        client, _library = await make_client(tmp_path, catalog)
        try:
            graph = Graph(
                nodes={
                    "n": GraphNode(
                        "test.provider_vision",
                        {"x": 1, "provider": "first"},
                    )
                }
            )
            response = await client.post("/api/jobs", json=submit_body(graph))
            assert response.status == 409
            plan = await response.json()
            assert [asset["digest"] for asset in plan["assets"]] == [MODEL_DIGEST]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_unused_declaration_never_gates(tmp_path: Path) -> None:
    """A declared asset whose node types the graph does not instantiate is
    dormant: submission proceeds, nothing acquires."""

    async def scenario() -> None:
        catalog = lineart_catalog(tmp_path / "packs" / "aux")
        client, library = await make_client(tmp_path, catalog)
        try:
            graph = Graph(nodes={"n": GraphNode("test.plain", {"x": 3})})
            resp = await client.post("/api/jobs", json=submit_body(graph))
            assert resp.status == 202
            status = await wait_terminal(client)
            assert status["state"] == "completed"
            assert library.vault.resolve(MODEL_DIGEST) is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_wrong_packaged_bytes_fail_and_land_nothing(tmp_path: Path) -> None:
    """A pack shipping the wrong file for its declared digest is a failed
    lead reported in the plan - never a poisoned vault, never a queued job."""

    async def scenario() -> None:
        catalog = lineart_catalog(tmp_path / "packs" / "aux", bytes_ok=False)
        client, library = await make_client(tmp_path, catalog)
        try:
            graph = Graph(nodes={"n": GraphNode("test.lineart", {"x": 1})})
            resp = await client.post(
                "/api/jobs",
                json=submit_body(graph, acquireAssets=[MODEL_DIGEST]),
            )
            assert resp.status == 409
            plan = await resp.json()
            entry = plan["assets"][0]
            assert entry["status"] == "failed"
            assert "did not verify" in entry["detail"]
            assert library.vault.resolve(MODEL_DIGEST) is None
            assert (await client.get("/api/jobs/c1/j1")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_region_bodies_reach_node_declared_assets(tmp_path: Path) -> None:
    async def scenario() -> None:
        catalog = lineart_catalog(tmp_path / "packs" / "aux")
        client, _library = await make_client(tmp_path, catalog)
        try:
            body_graph = Graph(nodes={"i": GraphNode("test.lineart", {"x": 1})})
            graph = Graph(
                nodes={
                    "n": RegionNode(
                        kind="map",
                        body=body_graph,
                        ports={"xs": INT},
                        inputs={"xs": [1, 2]},
                        element_ports=("xs",),
                    )
                }
            )
            resp = await client.post("/api/jobs", json=submit_body(graph))
            assert resp.status == 409
            plan = await resp.json()
            assert plan["assets"][0]["digest"] == MODEL_DIGEST
        finally:
            await client.close()

    asyncio.run(scenario())


def test_declared_leads_merge_with_submission_hints(tmp_path: Path) -> None:
    """Declared remote leads and submission-supplied assetSources merge in
    the plan without duplication; a graph LITERAL naming the same digest
    keeps its display name over the declaration's."""

    async def scenario() -> None:
        catalog = PackAssetCatalog()
        catalog.replace_all(
            [
                (
                    "aux",
                    None,
                    [
                        declared(
                            "lineart",
                            MODEL_DIGEST,
                            pack="aux",
                            nodes=("test.lineart",),
                            url="https://hub.example/lineart.bin",
                        )
                    ],
                )
            ]
        )
        client, _library = await make_client(tmp_path, catalog)
        try:
            graph = Graph(nodes={"n": GraphNode("test.lineart", {"x": 1})})
            resp = await client.post(
                "/api/jobs",
                json=submit_body(
                    graph,
                    assetSources={
                        MODEL_DIGEST: [
                            "https://hub.example/lineart.bin",  # duplicate
                            "https://mirror.example/lineart.bin",
                        ]
                    },
                ),
            )
            assert resp.status == 409
            entry = (await resp.json())["assets"][0]
            assert entry["sources"] == [
                "https://hub.example/lineart.bin",
                "https://mirror.example/lineart.bin",
            ]
            assert entry["fetchable"] is True
        finally:
            await client.close()

    asyncio.run(scenario())


def test_sources_endpoint_merges_declared_leads(tmp_path: Path) -> None:
    """GET /api/assets/{digest}/sources merges catalog-declared URLs at
    READ time on top of the provenance record - so a removed pack's leads
    disappear with it instead of fossilizing in provenance.json."""

    async def scenario() -> None:
        catalog = PackAssetCatalog()
        entries = [
            (
                "aux",
                None,
                [
                    declared(
                        "lineart",
                        MODEL_DIGEST,
                        pack="aux",
                        url="https://hub.example/lineart.bin",
                    )
                ],
            )
        ]
        catalog.replace_all(entries)
        client, library = await make_client(tmp_path, catalog)
        try:
            assert library.provenance is not None
            library.provenance.add(
                ProvenanceRecord(
                    digest=MODEL_DIGEST,
                    sources=("https://elsewhere.example/l.bin",),
                )
            )
            record = await (await client.get(f"/api/assets/{MODEL_DIGEST}/sources")).json()
            # Provenance first, declared leads appended (no duplicates).
            assert record["sources"] == [
                "https://elsewhere.example/l.bin",
                "https://hub.example/lineart.bin",
            ]

            # The pack goes away -> its lead goes with it; provenance stays.
            catalog.replace_all([])
            record = await (await client.get(f"/api/assets/{MODEL_DIGEST}/sources")).json()
            assert record["sources"] == ["https://elsewhere.example/l.bin"]

            # A live pack's lead shows even with NO provenance record, and
            # an undeclared, unrecorded digest stays an empty record.
            catalog.replace_all(entries)
            fresh = digest_bytes(b"nothing recorded for this")
            record = await (await client.get(f"/api/assets/{fresh}/sources")).json()
            assert record["sources"] == []
        finally:
            await client.close()

    asyncio.run(scenario())
