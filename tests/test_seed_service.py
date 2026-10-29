from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, BinaryIO, cast
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    P2P_FORMAT_POLICY_VERSION,
    AssetVault,
    ProvenanceStore,
    PublicAcquisitionReceiptStore,
    ResolverSubscriptionStore,
    identity,
    p2p_descriptor,
    p2p_storage,
    p2p_store,
)
from dinkster_assets.p2p_global import provider_declarations
from dinkster_assets.p2p_store import ExistingSeedStore
from dinkster_p2p import (
    LanNetworkPolicy,
    P2PSidecarManager,
    SeedLease,
    default_p2p_settings,
    normalize_p2p_settings,
)
from dinkster_p2p import runtime as runtime_module
from dinkster_p2p.runtime import P2PSessionPlan, SidecarError, SidecarRuntime, session_settings

from dinkster.lan_p2p import LanP2PController
from dinkster.seed import SeedService, parser
from tests.p2p_global_fixtures import build_provider_fixture
from tests.platform_support import symlink_or_skip
from tests.test_p2p_storage import _safetensors


def test_mapper_size_filter_restart_and_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_provider_fixture(tmp_path / "models", payload_size=128 * 1024)
    vault = AssetVault(tmp_path / "vault")
    (fixture.source.parent / "different-size").write_bytes(b"x")
    store = ExistingSeedStore(vault, [fixture.source.parent])
    calls: list[Path] = []
    reads: list[int] = []
    derive = p2p_descriptor.derive_p2p_descriptor
    open_regular = p2p_storage._open_regular

    def no_hash(*_args: object, **_kwargs: object) -> Any:
        pytest.fail("redundant full-file scan")

    class CountedReader:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle

        def __getattr__(self, name: str) -> Any:
            return getattr(self.handle, name)

        def read(self, size: int = -1) -> bytes:
            data = self.handle.read(size)
            reads.append(len(data))
            return data

    @contextmanager
    def counted_open(path: Path, *, writable: bool) -> Iterator[BinaryIO]:
        with open_regular(path, writable=writable) as handle:
            yield cast(BinaryIO, CountedReader(handle))

    def scanned(path: Path, *, handle: BinaryIO | None = None) -> Any:
        calls.append(path)
        assert handle is not None
        return derive(path, handle=handle)

    assert not hasattr(p2p_store, "digest_file")
    monkeypatch.setattr(identity, "digest_file", no_hash)
    monkeypatch.setattr(p2p_storage, "_hash_handle", no_hash)
    monkeypatch.setattr(p2p_storage, "_open_regular", counted_open)
    monkeypatch.setattr(p2p_storage, "derive_p2p_descriptor", scanned)
    assert store.refresh((fixture.snapshot,)) == (fixture.descriptor.asset_digest,)
    assert calls == [fixture.source]
    header_size = int.from_bytes(fixture.source.read_bytes()[:8], "little")
    assert sum(reads) == fixture.descriptor.size + 4 + 8 + header_size
    assert store.local_path_for(fixture.descriptor.asset_digest) == fixture.source
    assert not vault.has(fixture.descriptor.asset_digest)
    assert fixture.source.stat().st_nlink == 1

    reads.clear()
    with monkeypatch.context() as patch:
        patch.setattr(p2p_storage, "derive_p2p_descriptor", no_hash)
        restarted = ExistingSeedStore(AssetVault(vault.root), [fixture.source.parent])
        assert restarted.refresh((fixture.snapshot,)) == (fixture.descriptor.asset_digest,)
        mapping = vault.verify_p2p_local_file(
            fixture.descriptor.asset_digest,
            fixture.descriptor.size,
            fixture.source,
            P2P_FORMAT_POLICY_VERSION,
        )
        assert mapping.is_current()
        assert (
            p2p_storage.verified_p2p_seed_descriptor(
                vault.root, mapping.digest, mapping.size, mapping.path
            )
            == fixture.descriptor
        )
    assert reads == []

    before = fixture.source.stat()
    changed = bytearray(fixture.source.read_bytes())
    changed[-1] ^= 1
    fixture.source.write_bytes(changed)
    os.utime(fixture.source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert store.local_path_for(fixture.descriptor.asset_digest) is None
    assert store.refresh((fixture.snapshot,)) == ()
    assert not mapping.is_current()


@pytest.mark.parametrize("change", ["mutate", "replace", "symlink"])
def test_single_scan_rejects_changed_open_file_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    fixture = build_provider_fixture(tmp_path / "models", payload_size=32)
    vault = AssetVault(tmp_path / "vault")
    derive = p2p_descriptor.derive_p2p_descriptor

    def changed(path: Path, *, handle: BinaryIO | None = None) -> Any:
        result = derive(path, handle=handle)
        if change == "mutate":
            before = path.stat()
            with path.open("r+b") as writer:
                writer.seek(-1, 2)
                writer.write(b"X")
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        else:
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            if change == "replace":
                if sys.platform == "win32":
                    path.unlink()
                replacement.replace(path)
            else:
                path.unlink()
                symlink_or_skip(path, replacement)
        return result

    monkeypatch.setattr(p2p_storage, "derive_p2p_descriptor", changed)
    with pytest.raises(p2p_storage.P2PStorageError):
        vault.verify_p2p_local_file(
            fixture.descriptor.asset_digest,
            fixture.descriptor.size,
            fixture.source,
            P2P_FORMAT_POLICY_VERSION,
        )
    assert p2p_storage.cached_p2p_local_file(vault.root, fixture.source) is None


def test_descriptor_borrows_open_handle_without_reopening_or_closing(tmp_path: Path) -> None:
    fixture = build_provider_fixture(tmp_path / "models", payload_size=32)
    with fixture.source.open("rb") as handle:
        handle.seek(17)
        result = p2p_descriptor.derive_p2p_descriptor(tmp_path / "not-opened", handle=handle)
        assert result == fixture.descriptor
        assert not handle.closed


def test_mapper_omission_expiry_and_symlinks(tmp_path: Path) -> None:
    fixture = build_provider_fixture(tmp_path / "models", payload_size=32)
    vault = AssetVault(tmp_path / "vault")
    store = ExistingSeedStore(vault, [fixture.source.parent])
    assert store.refresh((fixture.snapshot,))
    assert store.refresh((fixture.tombstoned(),)) == ()
    assert (
        store.refresh((fixture.snapshot,), now=fixture.snapshot.p2p_artifacts[0].expires_at) == ()
    )
    assert store.refresh(()) == ()
    outside = tmp_path / "outside"
    fixture.source.rename(outside)
    symlink_or_skip(fixture.source, outside)
    assert store.refresh((fixture.snapshot,)) == ()
    linked_root = tmp_path / "linked"
    symlink_or_skip(linked_root, fixture.source.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        ExistingSeedStore(vault, [linked_root])


@pytest.mark.parametrize("lan_first", [False, True])
def test_warm_mapping_reused_by_controller_manager_and_native_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lan_first: bool
) -> None:
    fixture = build_provider_fixture(tmp_path / "models", payload_size=32)
    vault = AssetVault(tmp_path / "vault")
    store = ExistingSeedStore(vault, [fixture.source.parent])
    assert store.refresh((fixture.snapshot,))
    monkeypatch.setattr(runtime_module, "current_lan_policy", lambda: LanNetworkPolicy(()))

    def no_hash(*_args: object) -> Any:
        pytest.fail("verified mapping was rehashed downstream")

    monkeypatch.setattr(p2p_storage, "_hash_handle", no_hash)
    monkeypatch.setattr(p2p_storage, "derive_p2p_descriptor", no_hash)
    monkeypatch.setattr(runtime_module, "derive_p2p_descriptor", no_hash)
    lease = cast(SeedLease, fixture.authorizations()[0].lease)
    manager = P2PSidecarManager(vault_root=vault.root)
    cast(Any, manager)._verify_global_seed(lease)
    controller = LanP2PController(
        vault=vault,
        resolver_indexes=ResolverSubscriptionStore(
            tmp_path / "resolver.json", ProvenanceStore(tmp_path / "provenance.json")
        ),
        receipts=PublicAcquisitionReceiptStore(tmp_path / "receipts.json"),
        local_path_for=store.local_path_for,
    )
    internal = cast(Any, controller)
    internal._settings = {"seedingEnabled": True}
    internal._snapshot = fixture.grants
    internal._discovery = object()
    internal._manager.grant_seed = AsyncMock()
    declarations = tuple(
        decision.declaration
        for decision in provider_declarations(
            fixture.snapshot,
            trusted_provider_ids=frozenset({fixture.snapshot.provider_id}),
            now=fixture.observed_at,
        )
        if decision.declaration is not None
    )
    asyncio.run(internal._reconcile_seed_leases(declarations))
    assert internal._manager.grant_seed.await_count == 1
    runtime = SidecarRuntime(
        state_root=vault.root / ".p2p",
        vault_root=vault.root,
        installation_root=None,
        settings={
            **default_p2p_settings(),
            "scope": "lan-only",
            "seedingEnabled": True,
        },
    )
    try:
        assert cast(Any, runtime)._verify_seed(lease) == fixture.descriptor
        if lan_first:
            runtime.grant(replace(lease, scope="lan-only").to_wire(), "seed")
        # Exercise global activation's real native add/borrow path without enabling networking.
        handle = cast(Any, runtime)._global._add(lease, ())
        assert handle.is_valid()
        flags = cast(Any, runtime)._lt.torrent_flags
        assert handle.status().flags & flags.upload_mode
        assert not handle.status().flags & flags.auto_managed
        handle.force_recheck()
        handle.pause()
        handle.resume()
        runtime.apply_session_plan(P2PSessionPlan())
        assert handle.status().flags & flags.upload_mode
        assert not handle.status().flags & flags.auto_managed
        assert runtime.status()["listenPort"] is None
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("path", "../escape"),
        ("fingerprint", [0]),
        ("size", True),
        ("policy", 9),
        ("digest", "invalid"),
    ],
)
def test_bad_cache_never_restores_mapping(tmp_path: Path, field: str, value: object) -> None:
    fixture = build_provider_fixture(tmp_path / "models", payload_size=32)
    vault = AssetVault(tmp_path / "vault")
    store = ExistingSeedStore(vault, [fixture.source.parent])
    assert store.refresh((fixture.snapshot,))
    cache = next((vault.root / ".p2p" / "verified-local").glob("*.json"))
    row = json.loads(cache.read_text())
    row[field] = value
    cache.write_text(json.dumps(row))
    assert p2p_storage.cached_p2p_local_file(vault.root, fixture.source) is None
    assert store.refresh((fixture.snapshot,))


def test_descriptor_cache_validates_material(tmp_path: Path) -> None:
    fixture = build_provider_fixture(tmp_path / "models", payload_size=32)
    vault = AssetVault(tmp_path / "vault")
    store = ExistingSeedStore(vault, [fixture.source.parent])
    assert store.refresh((fixture.snapshot,))
    cache = next((vault.root / ".p2p" / "verified-local").glob("*.json"))
    row = json.loads(cache.read_text())
    row["descriptor"]["info"] = "00"
    cache.write_text(json.dumps(row))
    assert (
        p2p_storage.verified_p2p_seed_descriptor(
            vault.root, fixture.descriptor.asset_digest, fixture.descriptor.size, fixture.source
        )
        == fixture.descriptor
    )


def test_unsafe_same_digest_not_mapped(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.safetensors"
    source.write_bytes(b"unsafe bytes")
    fixture = build_provider_fixture(tmp_path, source=source)
    store = ExistingSeedStore(AssetVault(tmp_path / "vault"), [tmp_path])
    assert store.refresh((fixture.snapshot,)) == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("maxActiveSeeds", 0),
        ("maxActiveSeeds", 4097),
        ("maxActiveSeeds", True),
        ("listenPort", -1),
        ("listenPort", 65536),
        ("listenPort", True),
    ],
)
def test_seed_setting_bounds(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        normalize_p2p_settings({**default_p2p_settings(), field: value})


def test_seed_settings_legacy_and_listen_port() -> None:
    legacy = default_p2p_settings()
    del legacy["maxActiveSeeds"], legacy["listenPort"]
    assert normalize_p2p_settings(legacy) == default_p2p_settings()
    settings = {**default_p2p_settings(), "listenPort": 24680}
    assert (
        session_settings(settings, LanNetworkPolicy(()), P2PSessionPlan(global_tcp=True))[
            "listen_interfaces"
        ]
        == "0.0.0.0:24680"
    )


def test_218_native_seeds_and_bounded_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No interfaces, DHT, trackers, NAT mapping, or discovery in this native fixture.
    monkeypatch.setattr(runtime_module, "current_lan_policy", lambda: LanNetworkPolicy(()))
    vault = AssetVault(tmp_path / "vault")
    fixture_rows = []
    for index in range(219):
        source = tmp_path / "models" / f"{index}.safetensors"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(_safetensors(index.to_bytes(4, "little")))
        fixture_rows.append(build_provider_fixture(source.parent, source=source))
    settings = {
        **default_p2p_settings(),
        "scope": "lan-only",
        "downloadsEnabled": False,
        "seedingEnabled": True,
        "maxActiveSeeds": 218,
    }
    runtime = SidecarRuntime(
        state_root=vault.root / ".p2p",
        vault_root=vault.root,
        installation_root=None,
        settings=settings,
    )
    try:
        for index, fixture in enumerate(fixture_rows):
            lease = replace(
                cast(SeedLease, fixture.authorizations()[0].lease),
                scope="lan-only",
                lease_id=f"seed-{index}",
            )
            if index == 218:
                with pytest.raises(SidecarError, match="limit reached"):
                    runtime.grant(lease.to_wire(), "seed")
            else:
                runtime.grant(lease.to_wire(), "seed")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            handles = cast(Any, runtime)._session.get_torrents()
            if len(handles) == 218 and all(handle.status().is_seeding for handle in handles):
                break
            time.sleep(0.025)
        else:
            pytest.fail("218 native handles did not become seeds")
        assert len(runtime.status()["leases"]) == 218  # type: ignore[arg-type]
        assert runtime.status()["listenPort"] is None
        assert not any(vault.has(row.descriptor.asset_digest) for row in fixture_rows)
    finally:
        runtime.close()


def test_service_refresh_health_disable_and_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_provider_fixture(tmp_path / "models", payload_size=32)

    async def scenario() -> None:
        mode = "full"

        async def export(_request: web.Request) -> web.Response:
            if mode == "failure":
                raise web.HTTPServiceUnavailable()
            return web.json_response(
                {
                    "dinksterResolver": 1,
                    "name": "seed-fixture",
                    "updated": "2026-09-01T00:00:00Z",
                    "entries": []
                    if mode == "empty"
                    else [
                        {
                            "name": "model.safetensors",
                            "digest": fixture.descriptor.asset_digest,
                            "size": fixture.descriptor.size,
                            "urls": ["https://models.example/model.safetensors"],
                            "p2p": fixture.descriptor.descriptor.to_wire(),
                            "license": "unknown",
                        }
                    ],
                }
            )

        provider = web.Application()
        provider.router.add_get("/export", export)
        async with TestServer(provider) as server:
            args = parser().parse_args(
                [
                    "--store",
                    str(fixture.source.parent),
                    "--state-dir",
                    str(tmp_path / "state"),
                    "--provider-url",
                    str(server.make_url("/export")),
                    "--provider-id",
                    "seed-fixture",
                    "--refresh-seconds",
                    "37",
                    "--backoff-seconds",
                    "2",
                    "--backoff-max-seconds",
                    "5",
                ]
            )
            service = SeedService(args)
            assert service.settings["downloadsEnabled"] is True
            assert service.settings["seedingEnabled"] is True
            assert service.settings["scope"] == "lan-and-internet"
            monkeypatch.setattr(service.activity, "start", AsyncMock())
            monkeypatch.setattr(service.activity, "update", AsyncMock())
            monkeypatch.setattr(service.activity, "reconcile_network_policy", AsyncMock())
            reconcile = AsyncMock()
            monkeypatch.setattr(service.controller, "reconcile", reconcile)
            monkeypatch.setattr(
                service.controller,
                "status",
                AsyncMock(return_value={"state": "running", "sidecar": {"leases": []}}),
            )
            assert await service.refresh() == 37, service.error
            assert service.mapped == (fixture.descriptor.asset_digest,)
            assert reconcile.call_args.kwargs == {"local_files_changed": True}
            assert (await service.status())["ready"] is False
            monkeypatch.setattr(
                service.controller,
                "status",
                AsyncMock(
                    return_value={
                        "sidecar": {
                            "leases": [
                                {
                                    "kind": "seed",
                                    "state": "ready",
                                    "scope": "lan-and-internet",
                                    "digest": fixture.descriptor.asset_digest,
                                }
                            ]
                        },
                    }
                ),
            )
            assert (await service.status())["ready"] is True
            refreshed_at = service.last_refresh_at
            mode = "failure"
            assert await service.refresh() == 2
            assert await service.refresh() == 4
            assert await service.refresh() == 5
            assert service.last_refresh_at == refreshed_at
            assert service.mapped
            mode = "empty"
            assert await service.refresh() == 37
            assert service.mapped == ()
            assert (
                service.resolver.provider_p2p_snapshots()[0].tombstones[0].digest
                == fixture.descriptor.asset_digest
            )
            async with TestClient(TestServer(service.application())) as client:
                assert (await client.get("/health")).status == 200
                assert (await client.get("/ready")).status == 503
                response = await client.post("/enabled", json=False)
                assert response.status == 200
                assert (await response.json())["enabled"] is False
                assert (
                    await client.post(
                        "/enabled", json=True, headers={"Origin": "https://evil.example"}
                    )
                ).status == 403
                assert (await client.post("/enabled", json={})).status == 400
                disabled_service = SeedService(args)
                assert disabled_service.settings["downloadsEnabled"] is True
                assert disabled_service.settings["seedingEnabled"] is False
                assert (await client.post("/enabled", json=True)).status == 200
            assert service._refresh_task is not None and service._refresh_task.done()

    asyncio.run(scenario())


def test_no_store_mutation_or_execution_import(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    args = parser().parse_args(["--store", str(root), "--state-dir", str(root / "state")])
    with pytest.raises(ValueError, match="outside"):
        SeedService(args)
    assert list(root.iterdir()) == []
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster.seed; assert 'torch' not in sys.modules; "
            "assert 'dinkster.serve' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_environment_configuration_and_flag_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster import seed

    configured = AsyncMock()
    monkeypatch.setattr(seed, "run", configured)
    for name, value in {
        "STORES": str(tmp_path / "models"),
        "PROVIDER_URL": "https://provider.example/index",
        "PROVIDER_ID": "models",
        "UPLOAD_BYTES_PER_SECOND": "4096",
        "LISTEN_PORT": "24681",
        "MAX_ACTIVE_SEEDS": "1024",
        "REFRESH_SECONDS": "19",
        "NETWORK_COST": "unmetered",
    }.items():
        monkeypatch.setenv("DINKSTER_SEED_" + name, value)
    monkeypatch.setattr(sys, "argv", ["dinkster-seed", "--listen-port", "0"])
    seed.main()
    args = configured.call_args.args[0]
    assert args.store == [tmp_path / "models"]
    assert args.provider_id == "models"
    assert args.provider_url == "https://provider.example/index"
    assert args.listen_port == 0 and args.upload_bytes_per_second == 4096
    assert args.max_active_seeds == 1024 and args.refresh_seconds == 19
    assert args.network_cost == "unmetered"
