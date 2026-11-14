"""Worker-side declared-asset resolution (the [[pack.assets]] read end).

What this proves: pack code names a declaration's stable pack-local id
through the v1 door and gets readable bytes - or a loud AssetError naming
exactly which contract was unmet (no table, unknown id, no store, not
acquired). Never a download: execution reads what consented acquisition
already landed. The isolated E2E runs a real worker process whose node
calls ``dinkster_api.v1.declared_asset`` against a real vault.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from dinkster_assets import (
    AssetError,
    AssetNeed,
    AssetVault,
    DeclaredAsset,
    RemoteSource,
    clear_declared_assets,
    declared_asset,
    digest_bytes,
    install_declared_assets,
    use_declared_asset_pack,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import Graph, GraphNode
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker

from tests.async_scenarios import run_scenario

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


# --- declared_asset --------------------------------------------------------------


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
        'publisher = "dinkster"\n'
        'namespaces = ["decl"]\n'
        "[pack.entry]\n"
        'nodes = "declpack_nodes:NODES"\n'
        "[[pack.assets]]\n"
        f'id = "aux-model"\nname = "Aux Model"\ndigest = "{MODEL_DIGEST}"\n'
        'urls = ["https://hub.example/aux.bin"]\n'
    )
    return manifest


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

    run_scenario(scenario())


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

    run_scenario(scenario())
