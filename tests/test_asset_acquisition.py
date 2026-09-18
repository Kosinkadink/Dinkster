"""Template/asset distribution foundations (DESIGN 3.12 + roadmap).

Needs declare what a template or workflow requires - display name always,
digest as the sole authority when present, kind from an open namespaced
vocabulary, source leads (packaged or remote). Acquisition materializes a
need verified or not at all; consent lives above it and is digest-exact.
Job submission preflights every referenced asset identity: missing ones
answer 409 with a machine-readable acquisition plan, and nothing ever
downloads without the caller consenting to that exact digest.
"""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    KIND_MODEL_CHECKPOINT,
    AssetComponent,
    AssetComponentManifest,
    AssetError,
    AssetNeed,
    AssetRef,
    AssetVault,
    DeclaredAsset,
    LibraryStore,
    PackagedSource,
    PackAssetCatalog,
    ProvenanceRecord,
    ProvenanceStore,
    RemoteSource,
    acquire_need,
    asset_kind_matches,
    digest_bytes,
    is_asset_kind,
    probe_file,
    register_asset_type,
    require_asset_kind,
    source_from_wire,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import CompiledGraph, Engine, EventListener, ExecutionRuntime
from dinkster_graph import Graph, GraphNode, RegionNode, graph_to_wire
from dinkster_protocol import (
    GRAPH_COMPILERS_SURFACE,
    GraphCompilerRegistrySnapshot,
    KeyedContribution,
)
from dinkster_schema import (
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_server import STATE_KEY, ServerLibrary, create_app
from dinkster_values import CORE_INT, TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from tests.platform_support import symlink_or_skip

MODEL_BYTES = b"tiny model bytes for acquisition tests" * 32
MODEL_DIGEST = digest_bytes(MODEL_BYTES)
OTHER_BYTES = b"a second, different asset"
OTHER_DIGEST = digest_bytes(OTHER_BYTES)


# --- kinds: open namespaced vocabulary ----------------------------------------


def test_asset_kind_grammar() -> None:
    for kind in ("media/image", "model/lora", "model/text-encoder", "a/b/c-2"):
        assert is_asset_kind(kind)
        assert require_asset_kind(kind) == kind
    for bad in ("model", "Model/Lora", "model/", "/lora", "model//lora", "m odel/x"):
        assert not is_asset_kind(bad)
        with pytest.raises(AssetError, match="asset kind"):
            require_asset_kind(bad)


def test_unknown_kinds_are_first_class() -> None:
    # The vocabulary is open: nobody pre-registered this kind and it is
    # still valid everywhere a kind goes.
    need = AssetNeed(name="quantized thing", kind="model/gguf-quant")
    assert need.kind == "model/gguf-quant"


def test_checkpoint_kind_and_component_manifest_wire_round_trip() -> None:
    assert KIND_MODEL_CHECKPOINT == "model/checkpoint"
    metadata = {"roles": ["positive", "negative"], "guidance": 7.5}
    manifest = AssetComponentManifest(
        (
            AssetComponent(
                "model/diffusion",
                architecture="sdxl",
                dtype="fp16",
                metadata=metadata,
            ),
            AssetComponent("model/text-encoder", architecture="clip-l"),
            AssetComponent("model/text-encoder", architecture="clip-g"),
            AssetComponent("model/vae"),
        )
    )
    need = AssetNeed(
        name="SDXL checkpoint",
        digest=MODEL_DIGEST,
        kind=KIND_MODEL_CHECKPOINT,
        component_manifest=manifest,
    )

    wire = json.loads(json.dumps(need.to_wire()))
    assert wire["components"] == [
        {
            "kind": "model/diffusion",
            "architecture": "sdxl",
            "dtype": "fp16",
            "metadata": {"roles": ["positive", "negative"], "guidance": 7.5},
        },
        {"kind": "model/text-encoder", "architecture": "clip-l"},
        {"kind": "model/text-encoder", "architecture": "clip-g"},
        {"kind": "model/vae"},
    ]
    assert AssetNeed.from_wire(wire) == need
    metadata["roles"].append("mutated")
    assert manifest.to_wire()[0]["metadata"] == {
        "roles": ["positive", "negative"],
        "guidance": 7.5,
    }


def test_component_manifest_validation_and_kind_matching() -> None:
    manifest = AssetComponentManifest(
        (AssetComponent("model/diffusion"), AssetComponent("model/vae"))
    )
    checkpoint = AssetNeed(
        name="checkpoint",
        kind=KIND_MODEL_CHECKPOINT,
        component_manifest=manifest,
    )

    assert checkpoint.matches_kind(KIND_MODEL_CHECKPOINT)
    assert checkpoint.matches_kind("model/diffusion")
    assert checkpoint.matches_kind("model/vae")
    assert not checkpoint.matches_kind("model/lora")
    assert not AssetNeed(name="untyped").matches_kind("model/vae")
    assert not asset_kind_matches("model/lora", "model/vae", manifest)
    assert not asset_kind_matches(KIND_MODEL_CHECKPOINT, "model/vae")

    with pytest.raises(AssetError, match="asset kind"):
        AssetComponent("vae")
    assert AssetComponentManifest(()).to_wire() == []
    with pytest.raises(AssetError, match="metadata must be an object"):
        AssetComponent.from_wire({"kind": "model/vae", "metadata": []})
    with pytest.raises(AssetError, match="finite JSON"):
        AssetComponent("model/vae", metadata={"scale": float("nan")})
    with pytest.raises(AssetError, match="component manifest must be a list"):
        AssetNeed.from_wire({"name": "bad", "components": {}})


def test_need_without_component_manifest_keeps_legacy_wire_shape() -> None:
    assert AssetNeed(name="plain", kind="model/vae").to_wire() == {
        "name": "plain",
        "kind": "model/vae",
    }
    source = RemoteSource("https://models.example/plain.safetensors")
    positional = AssetNeed("plain", "", "model/vae", -1, "", {}, (source,))
    assert positional.sources == (source,)
    assert positional.component_manifest is None


# --- needs: declaration model --------------------------------------------------


def test_need_wire_round_trip() -> None:
    need = AssetNeed(
        name="SD 1.5 Pruned",
        digest=MODEL_DIGEST,
        kind="model/diffusion",
        size=len(MODEL_BYTES),
        media_type="application/octet-stream",
        metadata={"family": "sd15"},
        sources=(
            PackagedSource(pack="starter", path="assets/sd15.safetensors"),
            RemoteSource(url="https://hub.example/sd15.safetensors"),
        ),
    )
    wire = need.to_wire()
    assert wire["name"] == "SD 1.5 Pruned"
    assert wire["sources"] == [
        {"type": "packaged", "pack": "starter", "path": "assets/sd15.safetensors"},
        {"type": "remote", "url": "https://hub.example/sd15.safetensors"},
    ]
    assert AssetNeed.from_wire(json.loads(json.dumps(wire))) == need


def test_need_digest_is_optional_but_name_is_not() -> None:
    # Digestless needs are representable on purpose (imported ComfyUI
    # workflows know names before anyone hashed bytes).
    assert AssetNeed(name="unhashed.safetensors").digest == ""
    with pytest.raises(AssetError, match="display name"):
        AssetNeed(name="   ")


def test_need_rejects_malformed_pieces() -> None:
    with pytest.raises(AssetError):
        AssetNeed(name="x", digest="sha256:abc")  # wrong digest scheme
    with pytest.raises(AssetError):
        AssetNeed(name="x", kind="NotAKind")
    with pytest.raises(AssetError):
        PackagedSource(pack="p", path="../escape.bin")  # catalog path grammar
    with pytest.raises(AssetError):
        RemoteSource(url="file:///etc/passwd")
    with pytest.raises(AssetError, match="unknown asset source"):
        source_from_wire({"type": "carrier-pigeon"})
    with pytest.raises(AssetError):
        AssetNeed.from_wire({"name": "x", "sources": "not-a-list"})


# --- acquisition: verified or nothing ------------------------------------------


def test_acquire_already_held_is_free(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(MODEL_DIGEST) as writer:
        writer.write(MODEL_BYTES)
        writer.commit()
    result = acquire_need(AssetNeed(name="m", digest=MODEL_DIGEST), vault)
    assert result.status == "held" and result.ok
    assert result.path is not None and result.path.read_bytes() == MODEL_BYTES


def test_acquire_packaged_source_verified(tmp_path: Path) -> None:
    pack_root = tmp_path / "packs" / "starter"
    (pack_root / "assets").mkdir(parents=True)
    (pack_root / "assets" / "model.bin").write_bytes(MODEL_BYTES)
    vault = AssetVault(tmp_path / "vault")
    need = AssetNeed(
        name="starter model",
        digest=MODEL_DIGEST,
        sources=(PackagedSource(pack="starter", path="assets/model.bin"),),
    )
    result = acquire_need(need, vault, pack_roots={"starter": pack_root})
    assert result.status == "acquired" and result.ok
    assert vault.resolve(MODEL_DIGEST) is not None


def test_acquire_uses_packaged_before_lan_and_notifies_materialization(tmp_path: Path) -> None:
    pack_root = tmp_path / "packs" / "starter"
    pack_root.mkdir(parents=True)
    (pack_root / "model.bin").write_bytes(MODEL_BYTES)
    notified: list[None] = []

    def unexpected_lan(_digest: str) -> Path | None:
        raise AssertionError("LAN must not run before a packaged source")

    result = acquire_need(
        AssetNeed(
            name="model",
            digest=MODEL_DIGEST,
            sources=(PackagedSource("starter", "model.bin"),),
        ),
        AssetVault(tmp_path / "vault"),
        pack_roots={"starter": pack_root},
        lan_resolve=unexpected_lan,
        materialized=lambda: notified.append(None),
    )
    assert result.status == "acquired"
    assert notified == [None]


def test_acquire_packaged_wrong_bytes_lands_nothing(tmp_path: Path) -> None:
    """A pack shipping the wrong file for a declared digest is a failed
    lead, never a poisoned vault - the digest is the only authority."""
    pack_root = tmp_path / "packs" / "starter"
    (pack_root / "assets").mkdir(parents=True)
    (pack_root / "assets" / "model.bin").write_bytes(b"not those bytes")
    vault = AssetVault(tmp_path / "vault")
    need = AssetNeed(
        name="m",
        digest=MODEL_DIGEST,
        sources=(PackagedSource(pack="starter", path="assets/model.bin"),),
    )
    result = acquire_need(need, vault, pack_roots={"starter": pack_root})
    assert result.status == "failed"
    assert "did not verify" in result.detail
    assert vault.resolve(MODEL_DIGEST) is None
    assert not [p for p in vault.root.rglob("*") if p.is_file()]


def test_acquire_packaged_cannot_escape_pack_root(tmp_path: Path) -> None:
    secret = tmp_path / "secret.bin"
    secret.write_bytes(MODEL_BYTES)
    pack_root = tmp_path / "packs" / "starter"
    pack_root.mkdir(parents=True)
    symlink_or_skip(pack_root / "leak", secret)
    vault = AssetVault(tmp_path / "vault")
    need = AssetNeed(
        name="m",
        digest=MODEL_DIGEST,
        sources=(PackagedSource(pack="starter", path="leak"),),
    )
    # The symlink resolves outside the pack root: not a usable source.
    result = acquire_need(need, vault, pack_roots={"starter": pack_root})
    assert result.status == "failed"
    assert vault.resolve(MODEL_DIGEST) is None


def test_acquire_unknown_pack_is_failed_not_error(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    need = AssetNeed(
        name="m",
        digest=MODEL_DIGEST,
        sources=(PackagedSource(pack="ghost", path="a.bin"),),
    )
    result = acquire_need(need, vault, pack_roots={})
    assert result.status == "failed" and "not present" in result.detail


def make_asset_host(assets: dict[str, bytes], hits: list[str]) -> web.Application:
    async def serve(request: web.Request) -> web.Response:
        hits.append(request.match_info["name"])
        data = assets.get(request.match_info["name"])
        if data is None:
            return web.Response(status=404)
        return web.Response(body=data)

    app = web.Application()
    app.router.add_get("/files/{name}", serve)
    return app


def test_acquire_remote_verified_and_provenance_recorded(tmp_path: Path) -> None:
    async def scenario() -> None:
        hits: list[str] = []
        server = TestServer(make_asset_host({"good": MODEL_BYTES, "lying": b"tampered"}, hits))
        await server.start_server()
        try:
            vault = AssetVault(tmp_path / "vault")
            provenance = ProvenanceStore(tmp_path / "provenance.json")
            need = AssetNeed(
                name="m",
                digest=MODEL_DIGEST,
                sources=(
                    RemoteSource(url=str(server.make_url("/files/lying"))),
                    RemoteSource(url=str(server.make_url("/files/good"))),
                ),
            )
            result = await asyncio.to_thread(acquire_need, need, vault, provenance=provenance)
            assert result.status == "acquired"
            assert vault.resolve(MODEL_DIGEST) is not None
            assert hits == ["lying", "good"]  # tampered mirror was tried, rejected
            # Success recorded the need's leads for the next machine.
            assert str(server.make_url("/files/good")) in provenance.sources(MODEL_DIGEST)
        finally:
            await server.close()

    asyncio.run(scenario())


def test_acquire_uses_lan_before_http_and_falls_back_when_peer_disappears(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        hits: list[str] = []
        server = TestServer(make_asset_host({"model": MODEL_BYTES}, hits))
        await server.start_server()
        try:
            need = AssetNeed(
                name="model",
                digest=MODEL_DIGEST,
                sources=(RemoteSource(str(server.make_url("/files/model"))),),
            )
            lan_vault = AssetVault(tmp_path / "lan-vault")
            calls: list[str] = []

            def lan_resolve(digest: str) -> Path | None:
                calls.append(digest)
                with lan_vault.writer(digest) as writer:
                    writer.write(MODEL_BYTES)
                    return writer.commit()

            resolved = await asyncio.to_thread(
                acquire_need,
                need,
                lan_vault,
                lan_resolve=lan_resolve,
            )
            assert resolved.status == "acquired"
            assert calls == [MODEL_DIGEST]
            assert hits == []

            fallback_vault = AssetVault(tmp_path / "fallback-vault")
            notified: list[None] = []
            fallback = await asyncio.to_thread(
                acquire_need,
                need,
                fallback_vault,
                lan_resolve=lambda digest: calls.append(digest),
                materialized=lambda: notified.append(None),
            )
            assert fallback.status == "acquired"
            assert hits == ["model"]
            assert notified == [None]
        finally:
            await server.close()

    asyncio.run(scenario())


def test_acquire_consults_provenance_leads(tmp_path: Path) -> None:
    async def scenario() -> None:
        hits: list[str] = []
        server = TestServer(make_asset_host({"m": MODEL_BYTES}, hits))
        await server.start_server()
        try:
            vault = AssetVault(tmp_path / "vault")
            provenance = ProvenanceStore(tmp_path / "provenance.json")
            provenance.add(
                ProvenanceRecord(
                    digest=MODEL_DIGEST,
                    sources=(str(server.make_url("/files/m")),),
                )
            )
            # The need itself declares no sources; the store's lead wins.
            need = AssetNeed(name="m", digest=MODEL_DIGEST)
            result = await asyncio.to_thread(acquire_need, need, vault, provenance=provenance)
            assert result.status == "acquired"
        finally:
            await server.close()

    asyncio.run(scenario())


def test_acquire_digestless_is_unverifiable_and_touches_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        hits: list[str] = []
        server = TestServer(make_asset_host({"m": MODEL_BYTES}, hits))
        await server.start_server()
        try:
            vault = AssetVault(tmp_path / "vault")
            need = AssetNeed(
                name="named-but-unhashed.safetensors",
                sources=(RemoteSource(url=str(server.make_url("/files/m"))),),
            )
            result = await asyncio.to_thread(acquire_need, need, vault)
            assert result.status == "unverifiable" and not result.ok
            assert hits == []  # no digest, no fetch - never a name-match guess
            assert vault.digests() == []
        finally:
            await server.close()

    asyncio.run(scenario())


def test_acquire_no_sources_is_failed(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    result = acquire_need(AssetNeed(name="m", digest=MODEL_DIGEST), vault)
    assert result.status == "failed"
    assert "no usable sources" in result.detail


# --- probe: digest-keyed metadata ----------------------------------------------


def make_safetensors(tensors: dict[str, tuple[str, list[int]]], **meta: str) -> bytes:
    """Minimal valid safetensors bytes: dtype/shape per tensor, contiguous
    zero-filled data section."""
    dtype_bytes = {"F16": 2, "BF16": 2, "F32": 4, "I64": 8, "U8": 1}
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        elements = 1
        for dim in shape:
            elements *= dim
        size = elements * dtype_bytes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if meta:
        header["__metadata__"] = dict(meta)
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset


def test_probe_safetensors_mixed_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(
        make_safetensors(
            {
                "weight": ("F16", [4, 8]),
                "bias": ("F16", [8]),
                "norm": ("F32", [8]),
            },
            format="pt",
            architecture="test-arch",
        )
    )
    meta = probe_file(path)
    assert meta["format"] == "safetensors"
    assert meta["tensorCount"] == 3
    assert meta["parameterCount"] == 4 * 8 + 8 + 8
    assert meta["dtypes"] == {"F16": 2, "F32": 1}  # mixed dtypes visible
    assert meta["extra"] == {"format": "pt", "architecture": "test-arch"}


def test_probe_unrecognized_bytes_report_unknown(tmp_path: Path) -> None:
    for name, data in {
        "image.png": b"\x89PNG\r\n\x1a\nnot really",
        "truncated.safetensors": struct.pack("<Q", 10_000) + b"{}",
        "empty.bin": b"",
        "hostile.safetensors": struct.pack("<Q", 2**63) + b"{}",
    }.items():
        path = tmp_path / name
        path.write_bytes(data)
        assert probe_file(path) == {"format": "unknown"}  # never an exception


# --- widget kind: presentation, open vocabulary --------------------------------


def test_asset_widget_kind_round_trips_on_the_wire() -> None:
    schema = NodeSchema(
        node_type="test.load",
        inputs=(
            InputSpec(
                "model",
                TypeExpr.concrete("dinkster.asset"),
                widget=AssetWidget(accept=("application/octet-stream",), kind="model/lora"),
            ),
        ),
        outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
    )
    wire = schema_to_wire(schema)
    entry = next(e for e in wire["interface"] if e["id"] == "model")  # type: ignore[index, union-attr]
    assert entry["widget"] == {
        "type": "ASSET",
        "accept": ["application/octet-stream"],
        "kind": "model/lora",
    }
    decoded = schema_from_wire(json.loads(json.dumps(wire)))
    assert decoded.inputs[0].widget == AssetWidget(
        accept=("application/octet-stream",), kind="model/lora"
    )
    # Kindless widgets omit the field entirely (no null noise).
    bare = schema_to_wire(
        NodeSchema(
            node_type="test.bare",
            inputs=(InputSpec("a", TypeExpr.concrete("dinkster.asset"), widget=AssetWidget()),),
            outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
        )
    )
    bare_entry = next(e for e in bare["interface"] if e["id"] == "a")  # type: ignore[index, union-attr]
    assert "kind" not in bare_entry["widget"]  # type: ignore[operator]


def test_asset_widget_rejects_ungrammatical_kind() -> None:
    with pytest.raises(ValueError, match="namespaced asset kind"):
        AssetWidget(kind="just-a-word")


# --- job preflight: plan, consent, acquire, run --------------------------------

ASSET = TypeExpr.concrete("dinkster.asset")
INT = TypeExpr.concrete(CORE_INT)


class AssetSize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.asset.size",
            inputs=(InputSpec("asset", ASSET),),
            outputs=(OutputSpec("bytes", INT),),
        )

    @classmethod
    def execute(cls, *, asset: AssetRef) -> Mapping[str, object]:
        return cls.outputs(bytes=len(asset.read_bytes()))


class AssetCompileSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.asset.compile_source",
            inputs=(),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(value=1)


NODES = (AssetSize, AssetCompileSource)
SCHEMAS = build_schemas(NODES)


def asset_literal(digest: str, name: str, size: int) -> dict[str, object]:
    return AssetRef(digest=digest, name=name, size=size).to_wire()


def asset_graph(digest: str = MODEL_DIGEST, name: str = "model.bin") -> Graph:
    return Graph(
        nodes={
            "s": GraphNode(
                "test.asset.size",
                {"asset": asset_literal(digest, name, len(MODEL_BYTES))},
            )
        }
    )


def asset_compile_source_graph() -> Graph:
    return Graph({"source": GraphNode("test.asset.compile_source", {})})


def asset_compiler_runtime(engine: Engine) -> ExecutionRuntime:
    async def unused_transport(*_args: object) -> Mapping[str, object]:
        raise AssertionError("asset tests replace compile_for_execution")

    contribution = KeyedContribution(
        surface_id=GRAPH_COMPILERS_SURFACE,
        id="test.asset.compiler",
        behavior_metadata=(("contractVersion", 1), ("order", 0)),
    )
    return replace(
        engine.pin_execution(),
        graph_compiler_registry=GraphCompilerRegistrySnapshot((contribution,)),
        graph_compile_transport=unused_transport,
    )


def test_preflight_sees_through_typed_literals() -> None:
    """A typed literal is a stamped literal: an asset descriptor inside one
    (scalar or list element) must reach preflight exactly like a plain
    literal, or consent could be bypassed by stamping."""
    from dinkster_graph import TypedLiteral
    from dinkster_server.preflight import graph_asset_names

    scalar = Graph(
        nodes={
            "s": GraphNode(
                "test.asset.size",
                {
                    "asset": TypedLiteral(
                        "dinkster.asset",
                        asset_literal(MODEL_DIGEST, "stamped.bin", len(MODEL_BYTES)),
                    )
                },
            )
        }
    )
    assert graph_asset_names(scalar) == {MODEL_DIGEST: "stamped.bin"}
    listy = Graph(
        nodes={
            "s": GraphNode(
                "test.asset.size",
                {
                    "asset": TypedLiteral(
                        "list<dinkster.asset>",
                        [asset_literal(MODEL_DIGEST, "in-list.bin", len(MODEL_BYTES))],
                    )
                },
            )
        }
    )
    assert graph_asset_names(listy) == {MODEL_DIGEST: "in-list.bin"}


async def make_asset_client(tmp_path: Path) -> tuple[TestClient, ServerLibrary]:
    vault = AssetVault(tmp_path / "vault")
    library = ServerLibrary(
        vault=vault,
        store=LibraryStore(tmp_path / "library.sqlite"),
        provenance=ProvenanceStore(tmp_path / "provenance.json"),
    )

    def make_engine(on_event: EventListener | None = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, vault)
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


async def make_compiled_asset_client(
    tmp_path: Path,
    *,
    pack_assets: PackAssetCatalog | None = None,
) -> tuple[TestClient, web.Application, Engine, ExecutionRuntime]:
    vault = AssetVault(tmp_path / "vault")
    library = ServerLibrary(
        vault=vault,
        store=LibraryStore(tmp_path / "library.sqlite"),
        provenance=ProvenanceStore(tmp_path / "provenance.json"),
        pack_assets=pack_assets,
    )
    engine_box: list[Engine] = []
    runtime_box: list[ExecutionRuntime] = []

    def make_engine(on_event: EventListener | None = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry, vault)
        engine = Engine(
            schemas=SCHEMAS,
            registry=registry,
            worker=InProcessWorker(build_node_types(NODES), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )
        runtime = asset_compiler_runtime(engine)
        engine._pin_execution = lambda: runtime
        engine_box.append(engine)
        runtime_box.append(runtime)
        return engine

    app = create_app(make_engine, SCHEMAS, library=library)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, app, engine_box[0], runtime_box[0]


def submit_body(graph: Graph, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "clientId": "c1",
        "jobId": "j1",
        "graph": graph_to_wire(graph),
        "targets": ["s"],
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


def test_submit_missing_asset_returns_plan(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await make_asset_client(tmp_path)
        try:
            assert library.provenance is not None
            library.provenance.add(
                ProvenanceRecord(digest=MODEL_DIGEST, sources=("https://hub.example/model.bin",))
            )
            resp = await client.post("/api/jobs", json=submit_body(asset_graph()))
            assert resp.status == 409
            plan = await resp.json()
            assert plan["error"] == "assets-missing"
            assert plan["assets"] == [
                {
                    "digest": MODEL_DIGEST,
                    "name": "model.bin",
                    "status": "missing",
                    "sources": ["https://hub.example/model.bin"],
                    "fetchable": True,
                }
            ]
            # Nothing queued, nothing downloaded.
            assert (await client.get("/api/jobs/c1/j1")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_unselected_missing_asset_branch_still_refuses_graph_wide_preflight(
    tmp_path: Path,
) -> None:
    """Targets prune execution, never document-wide asset admission."""

    async def scenario() -> None:
        client, _ = await make_asset_client(tmp_path)
        try:
            graph = Graph(
                {
                    "selected": GraphNode("test.asset.compile_source", {}),
                    "unselected": GraphNode(
                        "test.asset.size",
                        {"asset": asset_literal(MODEL_DIGEST, "unselected.bin", len(MODEL_BYTES))},
                    ),
                }
            )
            response = await client.post(
                "/api/jobs",
                json=submit_body(graph, targets=["selected"]),
            )
            assert response.status == 409
            body = await response.json()
            assert body["error"] == "assets-missing"
            assert body["assets"][0]["digest"] == MODEL_DIGEST
            assert body["assets"][0]["name"] == "unselected.bin"
            assert (await client.get("/api/jobs/c1/j1")).status == 404
        finally:
            await client.close()

    asyncio.run(scenario())


def test_compiled_literal_asset_is_preflighted_before_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        client, app, engine, runtime = await make_compiled_asset_client(tmp_path)
        calls = 0

        async def compile_graph(*_args: object, **_kwargs: object) -> CompiledGraph:
            nonlocal calls
            calls += 1
            graph = Graph(
                {
                    "compiled": GraphNode(
                        "test.asset.size",
                        {
                            "asset": asset_literal(
                                MODEL_DIGEST, "compiled-only.bin", len(MODEL_BYTES)
                            )
                        },
                    )
                }
            )
            return CompiledGraph(graph, ("compiled",), runtime.extension_snapshot_digest, {})

        monkeypatch.setattr(engine, "compile_for_execution", compile_graph)
        try:
            response = await client.post(
                "/api/jobs",
                json=submit_body(asset_compile_source_graph(), targets=["source"]),
            )
            assert response.status == 409
            plan = await response.json()
            assert plan["error"] == "assets-missing"
            assert plan["assets"][0]["digest"] == MODEL_DIGEST
            assert plan["assets"][0]["name"] == "compiled-only.bin"
            assert calls == 1
            assert app[STATE_KEY].queue.get("c1", "j1") is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_compiled_node_pack_asset_is_preflighted_before_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        catalog = PackAssetCatalog()
        catalog.replace_all(
            [
                (
                    "asset-pack",
                    None,
                    [
                        DeclaredAsset(
                            id="compiled-model",
                            need=AssetNeed(name="compiled model", digest=MODEL_DIGEST),
                            nodes=("test.asset.size",),
                        )
                    ],
                )
            ]
        )
        client, app, engine, runtime = await make_compiled_asset_client(
            tmp_path,
            pack_assets=catalog,
        )

        async def compile_graph(*_args: object, **_kwargs: object) -> CompiledGraph:
            graph = Graph(
                {
                    "compiled": GraphNode(
                        "test.asset.size",
                        {"asset": asset_literal(OTHER_DIGEST, "held-elsewhere.bin", 1)},
                    )
                }
            )
            return CompiledGraph(graph, ("compiled",), runtime.extension_snapshot_digest, {})

        monkeypatch.setattr(engine, "compile_for_execution", compile_graph)
        try:
            response = await client.post(
                "/api/jobs",
                json=submit_body(asset_compile_source_graph(), targets=["source"]),
            )
            assert response.status == 409
            plan = await response.json()
            by_digest = {entry["digest"]: entry for entry in plan["assets"]}
            assert by_digest[MODEL_DIGEST]["name"] == "compiled model"
            assert by_digest[OTHER_DIGEST]["name"] == "held-elsewhere.bin"
            assert app[STATE_KEY].queue.get("c1", "j1") is None
        finally:
            await client.close()

    asyncio.run(scenario())


def test_submit_never_downloads_without_consent(tmp_path: Path) -> None:
    async def scenario() -> None:
        hits: list[str] = []
        host = TestServer(make_asset_host({"m": MODEL_BYTES}, hits))
        await host.start_server()
        client, library = await make_asset_client(tmp_path)
        try:
            url = str(host.make_url("/files/m"))
            body = submit_body(asset_graph(), assetSources={MODEL_DIGEST: [url]})
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 409
            plan = await resp.json()
            assert plan["assets"][0]["status"] == "missing"
            assert plan["assets"][0]["sources"] == [url]
            assert hits == []  # a known source is a lead, never permission
        finally:
            await client.close()
            await host.close()

    asyncio.run(scenario())


def test_submit_with_consent_acquires_and_runs(tmp_path: Path) -> None:
    async def scenario() -> None:
        hits: list[str] = []
        host = TestServer(make_asset_host({"m": MODEL_BYTES}, hits))
        await host.start_server()
        client, library = await make_asset_client(tmp_path)
        try:
            url = str(host.make_url("/files/m"))
            body = submit_body(
                asset_graph(),
                assetSources={MODEL_DIGEST: [url]},
                acquireAssets=[MODEL_DIGEST],
            )
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 202
            assert hits == ["m"]
            status = await wait_terminal(client)
            assert status["state"] == "completed"
            # The verified bytes are now held; a resubmission is preflight-
            # silent and network-free.
            assert library.vault.resolve(MODEL_DIGEST) is not None
            resp = await client.post("/api/jobs", json=submit_body(asset_graph(), jobId="j2"))
            assert resp.status == 202
            assert hits == ["m"]
            # Success recorded the lead for future machines.
            assert library.provenance is not None
            assert url in library.provenance.sources(MODEL_DIGEST)
        finally:
            await client.close()
            await host.close()

    asyncio.run(scenario())


def test_consent_is_digest_exact(tmp_path: Path) -> None:
    """Consenting to one digest authorizes exactly that identity: another
    missing asset in the same graph still blocks, and a lying source
    cannot land different bytes under the consented name."""

    async def scenario() -> None:
        hits: list[str] = []
        host = TestServer(make_asset_host({"m": MODEL_BYTES, "lie": b"wrong bytes"}, hits))
        await host.start_server()
        client, library = await make_asset_client(tmp_path)
        try:
            graph = Graph(
                nodes={
                    "s": GraphNode(
                        "test.asset.size",
                        {"asset": asset_literal(MODEL_DIGEST, "model.bin", len(MODEL_BYTES))},
                    ),
                    "t": GraphNode(
                        "test.asset.size",
                        {"asset": asset_literal(OTHER_DIGEST, "other.bin", len(OTHER_BYTES))},
                    ),
                }
            )
            body = submit_body(
                graph,
                targets=["s", "t"],
                assetSources={
                    MODEL_DIGEST: [str(host.make_url("/files/m"))],
                    OTHER_DIGEST: [str(host.make_url("/files/lie"))],
                },
                acquireAssets=[MODEL_DIGEST],
            )
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 409
            plan = await resp.json()
            by_digest = {entry["digest"]: entry for entry in plan["assets"]}
            # The consented identity acquired; the unconsented one is the
            # remaining plan row - and its source was never contacted.
            assert MODEL_DIGEST not in by_digest
            assert by_digest[OTHER_DIGEST]["status"] == "missing"
            assert hits == ["m"]

            # Consenting to the second digest cannot launder wrong bytes:
            # the lying source fails verification, lands nothing.
            body["acquireAssets"] = [OTHER_DIGEST]
            resp = await client.post("/api/jobs", json=body)
            assert resp.status == 409
            plan = await resp.json()
            (entry,) = plan["assets"]
            assert entry["digest"] == OTHER_DIGEST
            assert entry["status"] == "failed"
            assert library.vault.resolve(OTHER_DIGEST) is None
        finally:
            await client.close()
            await host.close()

    asyncio.run(scenario())


def test_preflight_scans_region_bodies(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await make_asset_client(tmp_path)
        try:
            body_graph = Graph(
                nodes={
                    "inner": GraphNode(
                        "test.asset.size",
                        {"asset": asset_literal(MODEL_DIGEST, "nested.bin", len(MODEL_BYTES))},
                    )
                }
            )
            graph = Graph(
                nodes={
                    "r": RegionNode(
                        kind="map",
                        body=body_graph,
                        ports={"xs": INT},
                        inputs={"xs": [1, 2]},
                        element_ports=("xs",),
                    )
                }
            )
            resp = await client.post("/api/jobs", json=submit_body(graph, targets=["r"]))
            assert resp.status == 409
            plan = await resp.json()
            assert plan["assets"][0]["digest"] == MODEL_DIGEST
            assert plan["assets"][0]["name"] == "nested.bin"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_submit_consent_fields_validated(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await make_asset_client(tmp_path)
        try:
            for bad in (
                {"acquireAssets": "yes-please"},
                {"acquireAssets": [True]},
                {"acquireAssets": ["sha256:" + "0" * 64]},
                {"assetSources": [1, 2]},
                {"assetSources": {"not-a-digest": []}},
                {"assetSources": {MODEL_DIGEST: ["ftp://x"]}},
            ):
                resp = await client.post("/api/jobs", json=submit_body(asset_graph(), **bad))
                assert resp.status == 400, bad
        finally:
            await client.close()

    asyncio.run(scenario())


# --- asset metadata + sources endpoints -----------------------------------------


def test_asset_metadata_endpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, _library = await make_asset_client(tmp_path)
        try:
            data = make_safetensors({"w": ("F16", [2, 3])}, family="test")
            digest = (await (await client.post("/api/assets", data=data)).json())["digest"]
            resp = await client.get(f"/api/assets/{digest}/metadata")
            assert resp.status == 200
            payload = await resp.json()
            assert payload["digest"] == digest
            assert payload["metadata"]["format"] == "safetensors"
            assert payload["metadata"]["parameterCount"] == 6
            assert payload["metadata"]["extra"] == {"family": "test"}
            # Digest-immutable contract: forever cache + conditional GET.
            assert resp.headers["ETag"] == f'"{digest}"'
            resp304 = await client.get(
                f"/api/assets/{digest}/metadata",
                headers={"If-None-Match": f'"{digest}"'},
            )
            assert resp304.status == 304
            missing = await client.get(f"/api/assets/blake3:{'0' * 64}/metadata")
            assert missing.status == 404
            assert (await client.get("/api/assets/nope/metadata")).status == 400
        finally:
            await client.close()

    asyncio.run(scenario())


def test_asset_sources_endpoints(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await make_asset_client(tmp_path)
        try:
            # Unknown identity: an empty record, not a 404 - "no leads"
            # is a useful planning answer.
            resp = await client.get(f"/api/assets/{MODEL_DIGEST}/sources")
            assert resp.status == 200
            assert (await resp.json())["sources"] == []

            resp = await client.post(
                f"/api/assets/{MODEL_DIGEST}/sources",
                json={
                    "urls": ["https://hub.example/m", "https://mirror.example/m"],
                    "license": "apache-2.0",
                },
            )
            assert resp.status == 200
            # Additive merge: re-posting unions, never replaces.
            resp = await client.post(
                f"/api/assets/{MODEL_DIGEST}/sources",
                json={"urls": ["https://third.example/m"]},
            )
            record = await resp.json()
            assert record["sources"] == [
                "https://hub.example/m",
                "https://mirror.example/m",
                "https://third.example/m",
            ]
            assert record["license"] == "apache-2.0"
            # Persisted: the store on disk agrees.
            assert library.provenance is not None
            assert len(library.provenance.sources(MODEL_DIGEST)) == 3

            bad = await client.post(
                f"/api/assets/{MODEL_DIGEST}/sources",
                json={"urls": ["file:///etc/passwd"]},
            )
            assert bad.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())
