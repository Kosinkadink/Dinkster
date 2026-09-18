"""Worker-side declared-asset resolution (the [[pack.assets]] read end).

What this proves: pack code names a declaration's stable pack-local id
through the v1 door and gets readable bytes - or a loud AssetError naming
exactly which contract was unmet (no table, unknown id, no store, not
acquired). Never a download: execution reads what consented acquisition
already landed. The isolated E2E runs a real worker process whose node
calls ``dinkster_api.v1.declared_asset`` against a real vault.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from dinkster_assets import (
    AssetError,
    AssetNeed,
    AssetVault,
    ChainResolver,
    DeclaredAsset,
    RemoteSource,
    clear_declared_assets,
    declared_asset,
    digest_bytes,
    install_declared_assets,
    resolver_from_env,
    use_declared_asset_pack,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import Graph, GraphNode
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker, load_manifest
from dinkster_workers.host import load_pack

TESTS_DIR = Path(__file__).parent

MODEL_BYTES = b"tiny auxiliary model weights" * 8
MODEL_DIGEST = digest_bytes(MODEL_BYTES)


@pytest.fixture(autouse=True)
def _clean_table() -> Iterator[None]:
    """Every test starts and ends without a process-global table."""
    clear_declared_assets()
    yield
    clear_declared_assets()


def vault_with_model(tmp_path: Path) -> Path:
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(MODEL_DIGEST) as writer:
        writer.write(MODEL_BYTES)
        writer.commit()
    return vault.root


def model_declaration(
    asset_id: str = "aux-model",
    *,
    size: int = -1,
    media_type: str = "",
) -> DeclaredAsset:
    return DeclaredAsset(
        id=asset_id,
        need=AssetNeed(
            name="Aux Model",
            digest=MODEL_DIGEST,
            size=size,
            media_type=media_type,
            sources=(RemoteSource("https://hub.example/aux.bin"),),
        ),
    )


# --- resolver_from_env -----------------------------------------------------------


def test_resolver_from_env_empty_is_none() -> None:
    assert resolver_from_env({}) is None


def test_resolver_from_env_vault_only(tmp_path: Path) -> None:
    root = vault_with_model(tmp_path)
    resolver = resolver_from_env({"DINKSTER_ASSET_VAULT": str(root)})
    assert isinstance(resolver, AssetVault)
    path = resolver.resolve(MODEL_DIGEST)
    assert path is not None and path.read_bytes() == MODEL_BYTES


def test_resolver_from_env_chains_vault_and_mounts(tmp_path: Path) -> None:
    root = vault_with_model(tmp_path)
    resolver = resolver_from_env(
        {
            "DINKSTER_ASSET_VAULT": str(root),
            "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts-snapshot.json"),
        }
    )
    assert isinstance(resolver, ChainResolver)
    path = resolver.resolve(MODEL_DIGEST)
    assert path is not None and path.read_bytes() == MODEL_BYTES


def test_resolver_from_env_skips_unreadable_root(tmp_path: Path) -> None:
    """An unscanned/stale library index is advisory: the chain assembles
    without it (warning logged) instead of failing worker startup - reads
    that needed it still fail loudly at the read site."""
    assert resolver_from_env({"DINKSTER_ASSET_ROOT": str(tmp_path / "library")}) is None
    vault_root = vault_with_model(tmp_path)
    resolver = resolver_from_env(
        {
            "DINKSTER_ASSET_VAULT": str(vault_root),
            "DINKSTER_ASSET_ROOT": str(tmp_path / "library"),
        }
    )
    assert isinstance(resolver, AssetVault)


# --- declared_asset --------------------------------------------------------------


def test_no_table_is_a_worker_context_error() -> None:
    with pytest.raises(AssetError, match="no declaration table"):
        declared_asset("aux-model")


def test_unknown_id_lists_declared_ids(tmp_path: Path) -> None:
    install_declared_assets(
        "auxpack", [model_declaration("lineart"), model_declaration("depth")], None
    )
    with (
        use_declared_asset_pack("auxpack"),
        pytest.raises(AssetError, match=r"declares no asset 'typo'") as excinfo,
    ):
        declared_asset("typo")
    assert "depth, lineart" in str(excinfo.value)
    assert "auxpack" in str(excinfo.value)


def test_no_store_configured_is_loud() -> None:
    install_declared_assets("auxpack", [model_declaration()], None)
    with (
        use_declared_asset_pack("auxpack"),
        pytest.raises(AssetError, match="no asset store configured"),
    ):
        declared_asset("aux-model")


def test_unacquired_correlates_with_declaration_and_never_downloads(
    tmp_path: Path,
) -> None:
    """The digest is declared and remotely sourced, but the vault is empty:
    the error names id, digest, and the consent path - and nothing fetched
    (the remote URL is not even resolvable)."""
    empty_vault = AssetVault(tmp_path / "vault")
    install_declared_assets("auxpack", [model_declaration()], empty_vault)
    with (
        use_declared_asset_pack("auxpack"),
        pytest.raises(AssetError, match="not acquired") as excinfo,
    ):
        declared_asset("aux-model")
    message = str(excinfo.value)
    assert "aux-model" in message
    assert MODEL_DIGEST in message
    assert "acquireAssets" in message


def test_acquired_resolves_to_readable_ref(tmp_path: Path) -> None:
    vault = AssetVault(vault_with_model(tmp_path))
    install_declared_assets(
        "auxpack",
        [model_declaration(size=len(MODEL_BYTES), media_type="application/x-model")],
        vault,
    )
    with use_declared_asset_pack("auxpack"):
        ref = declared_asset("aux-model")
    assert ref.digest == MODEL_DIGEST
    assert ref.name == "Aux Model"
    assert ref.size == len(MODEL_BYTES)
    assert ref.media_type == "application/x-model"
    assert ref.read_bytes() == MODEL_BYTES


def test_undeclared_size_and_media_fall_back(tmp_path: Path) -> None:
    """A declaration without size stats the landed file; without a media
    type the ref carries the octet-stream default (AssetRef's contract)."""
    vault = AssetVault(vault_with_model(tmp_path))
    install_declared_assets("auxpack", [model_declaration()], vault)
    with use_declared_asset_pack("auxpack"):
        ref = declared_asset("aux-model")
    assert ref.size == len(MODEL_BYTES)
    assert ref.media_type == "application/octet-stream"


def test_reinstall_replaces_the_table(tmp_path: Path) -> None:
    vault = AssetVault(vault_with_model(tmp_path))
    install_declared_assets("first", [model_declaration("a")], vault)
    install_declared_assets("second", [model_declaration("b")], vault)
    with use_declared_asset_pack("second"):
        with pytest.raises(AssetError, match="'second' declares no asset 'a'"):
            declared_asset("a")
        assert declared_asset("b").read_bytes() == MODEL_BYTES


def test_declared_asset_context_is_pack_local(tmp_path: Path) -> None:
    vault = AssetVault(vault_with_model(tmp_path))
    install_declared_assets("first", [model_declaration("a")], vault)
    install_declared_assets("second", [model_declaration("b")], vault)
    with pytest.raises(AssetError, match="no declaration table"):
        declared_asset("a")
    with use_declared_asset_pack("first"):
        assert declared_asset("a").read_bytes() == MODEL_BYTES
        with pytest.raises(AssetError, match="declares no asset 'b'"):
            declared_asset("b")
    with use_declared_asset_pack("second"):
        assert declared_asset("b").read_bytes() == MODEL_BYTES
    with pytest.raises(AssetError, match="unknown declared-asset pack context"):
        with use_declared_asset_pack("absent"):
            pass


# --- the worker host installs the manifest's table --------------------------------


def write_decl_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        "[pack]\n"
        'name = "declpack"\n'
        'namespaces = ["decl"]\n'
        "[pack.entry]\n"
        'nodes = "declpack_nodes:NODES"\n'
        "[[pack.assets]]\n"
        f'id = "aux-model"\nname = "Aux Model"\ndigest = "{MODEL_DIGEST}"\n'
        'urls = ["https://hub.example/aux.bin"]\n'
    )
    return manifest


def test_load_pack_installs_declared_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The host-side seam: loading a manifest makes its declarations
    reachable through declared_asset, resolver bound from the worker's
    environment contract."""
    root = vault_with_model(tmp_path)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(root))
    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    monkeypatch.delenv("DINKSTER_ASSET_ROOT", raising=False)
    load_pack(load_manifest(write_decl_manifest(tmp_path)))
    with use_declared_asset_pack("declpack"):
        assert declared_asset("aux-model").read_bytes() == MODEL_BYTES


# --- E2E: a real isolated worker reads its own declaration -------------------------


def core_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def read_graph(asset_id: str = "aux-model") -> Graph:
    return Graph(nodes={"r": GraphNode("decl.read", {"asset_id": asset_id})})


def test_isolated_worker_reads_declared_asset(tmp_path: Path) -> None:
    """The whole story end to end: manifest declares, acquisition landed
    the bytes in the vault, and the node - a real worker process going
    through dinkster_api.v1 only - reads them by pack-local id."""

    async def scenario() -> None:
        vault_root = vault_with_model(tmp_path)
        registry = core_registry()
        worker = IsolatedWorker(
            write_decl_manifest(tmp_path),
            registry,
            extra_env={
                "PYTHONPATH": str(TESTS_DIR),
                "DINKSTER_ASSET_VAULT": str(vault_root),
            },
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            result = await engine.run(read_graph(), ["r"])
            assert result.outputs["r"]["text"].resolve() == MODEL_BYTES.decode()
            assert result.outputs["r"]["size"].resolve() == len(MODEL_BYTES)
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_isolated_worker_unacquired_fails_with_consent_pointer(tmp_path: Path) -> None:
    """Same worker, empty vault: execution fails with the declaration-
    correlated message instead of downloading or guessing."""

    async def scenario() -> None:
        empty_vault = tmp_path / "empty-vault"
        empty_vault.mkdir()
        registry = core_registry()
        worker = IsolatedWorker(
            write_decl_manifest(tmp_path),
            registry,
            extra_env={
                "PYTHONPATH": str(TESTS_DIR),
                "DINKSTER_ASSET_VAULT": str(empty_vault),
            },
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            with pytest.raises(ExecutionError) as excinfo:
                await engine.run(read_graph(), ["r"])
            message = excinfo.value.error.message
            assert "not acquired" in message
            assert MODEL_DIGEST in message

            # Unknown id fails as a typo, same worker keeps serving.
            with pytest.raises(ExecutionError) as excinfo:
                await engine.run(read_graph("nope"), ["r"])
            assert "declares no asset 'nope'" in excinfo.value.error.message
        finally:
            await worker.close()

    asyncio.run(scenario())
