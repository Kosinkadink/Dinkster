"""Name-based best-guess asset matching: the lookup half of the
compatibility tier for imported ComfyUI workflows.

Legacy names are digestless and can never be verified; this surface only
ever SUGGESTS real identities (digest + confidence tier) and the client
accepts explicitly. Covered here: the pure matcher's tiers and
determinism, and POST /api/assets/guess over both corpora - mount
catalogs and pack-declared assets."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetEntry,
    AssetError,
    AssetNeed,
    AssetVault,
    DeclaredAsset,
    LibraryStore,
    MountDef,
    MountTable,
    PackagedSource,
    PackAssetCatalog,
    digest_bytes,
    guess_matches,
    match_confidence,
    normalize_guess_query,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener
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
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

from dinkster.guess_api import add_guess_routes
from dinkster.mounts_api import MountService, add_mount_routes

STRING = TypeExpr.concrete("core.string")


class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.echo",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text)


NODES = (Echo,)
SCHEMAS = build_schemas(NODES)


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


def seed(root: Path, files: Mapping[str, bytes]) -> None:
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def entry(path: str, data: bytes) -> AssetEntry:
    return AssetEntry(virtual_path=path, digest=digest_bytes(data), size=len(data))


# --- normalization ------------------------------------------------------------


def test_normalize_guess_query() -> None:
    assert normalize_guess_query("SD15\\model.safetensors") == "SD15/model.safetensors"
    assert normalize_guess_query("./a/./b.pt") == "a/b.pt"
    assert normalize_guess_query("a//b.pt") == "a/b.pt"
    # Traversal segments make a name that is not a plain subpath: no match.
    assert normalize_guess_query("../model.pt") == ""
    assert normalize_guess_query("") == ""
    assert normalize_guess_query("//") == ""


# --- confidence tiers ---------------------------------------------------------


def test_match_confidence_tiers() -> None:
    path = "mounts/models/checkpoints/SD15/model.safetensors"
    # Folder part present and matching (case-insensitively): strongest.
    assert match_confidence("sd15/model.safetensors", path) == "path"
    assert match_confidence("SD15/Model.SAFETENSORS", path) == "path"
    # Bare name, identical.
    assert match_confidence("model.safetensors", path) == "name"
    # Bare name, differing only by case.
    assert match_confidence("MODEL.safetensors", path) == "name-insensitive"
    # Extension swapped: a suggestion at best.
    assert match_confidence("model.ckpt", path) == "stem"
    assert match_confidence("model", path) == "stem"
    # Unrelated.
    assert match_confidence("other.safetensors", path) is None
    # Folder part that does NOT match falls through to the name tiers.
    assert match_confidence("sdxl/model.safetensors", path) == "name"
    # A query deeper than the candidate cannot path-match it.
    assert match_confidence("a/b/c/model.pt", "c/model.pt") == "name"


def test_guess_matches_ranks_and_dedupes() -> None:
    exact = entry("mounts/models/SD15/model.safetensors", b"exact")
    same_name = entry("mounts/models/other/model.safetensors", b"same-name")
    stem_only = entry("mounts/models/other/model.ckpt", b"stem")
    alias = entry("mounts/models/zz-alias/model.safetensors", b"exact")
    unrelated = entry("mounts/models/other/thing.pt", b"unrelated")
    matches = guess_matches(
        "SD15\\model.safetensors", [stem_only, alias, unrelated, same_name, exact]
    )
    assert [m.confidence for m in matches] == ["path", "name", "stem"]
    # The alias shares the exact match's digest: one candidate, best path.
    assert [m.virtual_path for m in matches] == [
        "mounts/models/SD15/model.safetensors",
        "mounts/models/other/model.safetensors",
        "mounts/models/other/model.ckpt",
    ]
    assert guess_matches("nothing.pt", [exact, same_name]) == ()
    assert guess_matches("", [exact]) == ()


# --- the API surface ----------------------------------------------------------


async def make_client(
    *,
    mounts: MountService | None = None,
    library: ServerLibrary | None = None,
) -> TestClient:
    app = create_app(make_engine, SCHEMAS, library=library)
    if mounts is not None:
        add_mount_routes(app, mounts)
    add_guess_routes(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_guess_endpoint_over_mounts(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(
        root,
        {
            "checkpoints/SD15/model.safetensors": b"exact",
            "checkpoints/other/model.safetensors": b"same-name",
            "loras/detail.safetensors": b"lora",
        },
    )
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="models", path=root))
    table.scan("models")

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        resp = await client.post(
            "/api/assets/guess",
            json={"names": ["SD15\\model.safetensors", "missing.pt"]},
        )
        assert resp.status == 200
        matches = (await resp.json())["matches"]
        assert [m["query"] for m in matches] == [
            "SD15\\model.safetensors",
            "missing.pt",
        ]
        first, second = matches[0]["candidates"][:2]
        assert first["confidence"] == "path"
        assert first["digest"] == digest_bytes(b"exact")
        assert first["virtualPath"] == ("mounts/models/checkpoints/SD15/model.safetensors")
        assert first["mountId"] == "models"
        assert first["held"] is True
        assert second["confidence"] == "name"
        assert second["digest"] == digest_bytes(b"same-name")
        assert matches[1]["candidates"] == []
        await client.close()

    asyncio.run(scenario())


def test_guess_endpoint_over_declared_assets(tmp_path: Path) -> None:
    digest = digest_bytes(b"declared-lora")
    catalog = PackAssetCatalog()
    catalog.replace_all(
        [
            (
                "packA",
                None,
                [
                    DeclaredAsset(
                        id="detail-lora",
                        need=AssetNeed(
                            name="Detail Tweaker",
                            digest=digest,
                            kind="model/lora",
                            sources=(
                                PackagedSource(
                                    pack="packA",
                                    path="assets/add_detail.safetensors",
                                ),
                            ),
                        ),
                    )
                ],
            )
        ]
    )
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        pack_assets=catalog,
    )

    async def scenario() -> None:
        client = await make_client(library=library)
        # The packaged file path matches even though the display name does
        # not - it is a real filename, the string a legacy workflow used.
        resp = await client.post("/api/assets/guess", json={"names": ["add_detail.safetensors"]})
        (match,) = (await resp.json())["matches"]
        (candidate,) = match["candidates"]
        assert candidate["digest"] == digest
        assert candidate["confidence"] == "name"
        assert candidate["name"] == "Detail Tweaker"
        assert candidate["declaredBy"] == ["packA"]
        assert candidate["kind"] == "model/lora"
        assert candidate["held"] is False  # digest known, bytes not local yet
        await client.close()

    asyncio.run(scenario())


def test_guess_endpoint_prefers_held_bytes_over_declarations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models"
    seed(root, {"loras/add_detail.safetensors": b"lora-bytes"})
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="models", path=root))
    table.scan("models")
    catalog = PackAssetCatalog()
    catalog.replace_all(
        [
            (
                "packA",
                None,
                [
                    DeclaredAsset(
                        id="detail-lora",
                        need=AssetNeed(
                            name="add_detail.safetensors",
                            digest=digest_bytes(b"lora-bytes"),
                        ),
                    )
                ],
            )
        ]
    )
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        pack_assets=catalog,
    )

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table), library=library)
        resp = await client.post("/api/assets/guess", json={"names": ["add_detail.safetensors"]})
        (match,) = (await resp.json())["matches"]
        # One digest, one candidate: the cataloged file represents it.
        (candidate,) = match["candidates"]
        assert candidate["held"] is True
        assert candidate["virtualPath"] == "mounts/models/loras/add_detail.safetensors"
        await client.close()

    asyncio.run(scenario())


def test_digest_hints_mix_with_unchanged_name_guessing(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(
        root,
        {
            "renamed/local.bin": b"known-by-digest",
            "checkpoints/model.safetensors": b"known-by-name",
        },
    )
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="models", path=root))
    table.scan("models")
    hinted_digest = digest_bytes(b"known-by-digest")

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        baseline = await client.post("/api/assets/guess", json={"names": ["model.safetensors"]})
        baseline_match = (await baseline.json())["matches"][0]
        resp = await client.post(
            "/api/assets/guess",
            json={
                "names": ["old-name.bin", "model.safetensors"],
                "digestHints": {"old-name.bin": hinted_digest},
            },
        )
        assert resp.status == 200
        hinted, unhinted = (await resp.json())["matches"]
        assert [hinted["query"], unhinted["query"]] == [
            "old-name.bin",
            "model.safetensors",
        ]
        (candidate,) = hinted["candidates"]
        assert candidate == {
            "digest": hinted_digest,
            "name": "local.bin",
            "confidence": "digest",
            "virtualPath": "mounts/models/renamed/local.bin",
            "held": True,
            "mountId": "models",
            "size": len(b"known-by-digest"),
            "mediaType": "application/octet-stream",
        }
        assert unhinted == baseline_match
        await client.close()

    asyncio.run(scenario())


def test_digest_hint_matches_renamed_local_entry(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"new/location/completely-renamed.bin": b"same identity"})
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="models", path=root))
    table.scan("models")
    digest = digest_bytes(b"same identity")

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        resp = await client.post(
            "/api/assets/guess",
            json={"names": ["legacy.ckpt"], "digestHints": {"legacy.ckpt": digest}},
        )
        (candidate,) = (await resp.json())["matches"][0]["candidates"]
        assert candidate["digest"] == digest
        assert candidate["confidence"] == "digest"
        assert candidate["virtualPath"] == ("mounts/models/new/location/completely-renamed.bin")
        await client.close()

    asyncio.run(scenario())


def test_digest_hint_orders_every_local_reference_and_caps_at_ten(
    tmp_path: Path,
) -> None:
    data = b"shared identity"
    table = MountTable(tmp_path / "snap.json")
    specs = (("zeta", -2, 4), ("alpha", -2, 4), ("middle", 7, 4))
    for mount_id, priority, count in specs:
        root = tmp_path / mount_id
        seed(root, {f"copies/{index:02}.bin": data for index in range(count)})
        table.add(MountDef(id=mount_id, path=root, priority=priority))
        table.scan(mount_id)
    digest = digest_bytes(data)

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        resp = await client.post(
            "/api/assets/guess",
            json={"names": ["anything"], "digestHints": {"anything": digest}},
        )
        candidates = (await resp.json())["matches"][0]["candidates"]
        assert len(candidates) == 10
        assert [candidate["virtualPath"] for candidate in candidates] == [
            *(f"mounts/alpha/copies/{index:02}.bin" for index in range(4)),
            *(f"mounts/zeta/copies/{index:02}.bin" for index in range(4)),
            *(f"mounts/middle/copies/{index:02}.bin" for index in range(2)),
        ]
        assert all(candidate["digest"] == digest for candidate in candidates)
        assert all(candidate["confidence"] == "digest" for candidate in candidates)
        await client.close()

    asyncio.run(scenario())


def test_digest_hint_uses_ready_local_mounts_only(tmp_path: Path) -> None:
    data = b"local identity"
    digest = digest_bytes(data)
    table = MountTable(tmp_path / "snap.json")
    ready_root = tmp_path / "ready"
    pending_root = tmp_path / "pending"
    seed(ready_root, {"ready.bin": data})
    seed(pending_root, {"pending.bin": data})
    table.add(MountDef(id="ready", path=ready_root))
    table.add(MountDef(id="pending", path=pending_root))
    table.add(MountDef(id="failed", path=tmp_path / "missing"))
    table.scan("ready")
    with pytest.raises((AssetError, OSError)):
        table.scan("failed")

    catalog = PackAssetCatalog()
    catalog.replace_all(
        [
            (
                "packA",
                None,
                [DeclaredAsset(id="remote", need=AssetNeed(name="remote.bin", digest=digest))],
            )
        ]
    )
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
        pack_assets=catalog,
    )

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table), library=library)
        resp = await client.post(
            "/api/assets/guess",
            json={"names": ["remote.bin"], "digestHints": {"remote.bin": digest}},
        )
        (candidate,) = (await resp.json())["matches"][0]["candidates"]
        assert candidate["virtualPath"] == "mounts/ready/ready.bin"
        assert candidate["mountId"] == "ready"
        assert candidate["confidence"] == "digest"
        assert "declaredBy" not in candidate
        await client.close()

    asyncio.run(scenario())


def test_digest_hint_zero_matches_without_name_fallback(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"model.safetensors": b"different bytes"})
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="models", path=root))
    table.scan("models")

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        resp = await client.post(
            "/api/assets/guess",
            json={
                "names": ["model.safetensors"],
                "digestHints": {"model.safetensors": digest_bytes(b"absent bytes")},
            },
        )
        assert (await resp.json())["matches"] == [{"query": "model.safetensors", "candidates": []}]
        await client.close()

    asyncio.run(scenario())


def test_digest_hint_validation_is_closed(tmp_path: Path) -> None:
    valid = digest_bytes(b"valid")
    names = [f"name-{index}" for index in range(64)]
    invalid_bodies: tuple[object, ...] = (
        {"names": ["name"], "extra": True},
        {"names": ["name"], "digestHints": []},
        {"names": names, "digestHints": {**dict.fromkeys(names, valid), "extra": valid}},
        {"names": ["name"], "digestHints": {"unknown": valid}},
        {"names": ["name"], "digestHints": {"name": None}},
        {"names": ["name"], "digestHints": {"name": "abc123"}},
        {"names": ["name"], "digestHints": {"name": "blake3:abc123"}},
        {"names": ["name"], "digestHints": {"name": valid.upper()}},
        {"names": ["name"], "digestHints": {"name": "sha256:" + "a" * 64}},
    )

    async def scenario() -> None:
        client = await make_client()
        for body in invalid_bodies:
            resp = await client.post("/api/assets/guess", json=body)
            assert resp.status == 400, (body, await resp.text())
        await client.close()

    asyncio.run(scenario())


def test_digest_hint_preserves_repeated_name_occurrences(tmp_path: Path) -> None:
    root = tmp_path / "models"
    seed(root, {"renamed.bin": b"identity"})
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="models", path=root))
    table.scan("models")
    digest = digest_bytes(b"identity")

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        resp = await client.post(
            "/api/assets/guess",
            json={"names": ["old.bin", "old.bin"], "digestHints": {"old.bin": digest}},
        )
        first, second = (await resp.json())["matches"]
        assert first == second
        assert first["query"] == "old.bin"
        assert first["candidates"][0]["confidence"] == "digest"
        await client.close()

    asyncio.run(scenario())


def test_guess_endpoint_without_corpus_answers_empty(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = await make_client()
        resp = await client.post("/api/assets/guess", json={"names": ["model.pt"]})
        assert resp.status == 200
        (match,) = (await resp.json())["matches"]
        assert match["candidates"] == []
        await client.close()

    asyncio.run(scenario())


def test_guess_endpoint_validates(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = await make_client()
        for body in (
            [],
            {"names": []},
            {"names": "model.pt"},
            {"names": [42]},
            {"names": ["  "]},
            {"names": ["x"] * 65},
            {"names": ["y" * 1025]},
        ):
            resp = await client.post("/api/assets/guess", json=body)
            assert resp.status == 400, await resp.text()
        resp = await client.post(
            "/api/assets/guess",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400
        await client.close()

    asyncio.run(scenario())
