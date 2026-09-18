from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from ipaddress import IPv4Address, IPv4Network, ip_address
from pathlib import Path
from typing import Any, cast

import psutil
import pytest
from dinkster_assets import (
    P2P_PIECE_LENGTH,
    AssetVault,
    ProvenanceStore,
    PublicAcquisitionReceiptStore,
    PublicAcquisitionReceiptV1,
    ResolverSubscriptionStore,
    TransportCandidate,
    derive_p2p_descriptor,
)
from dinkster_p2p import (
    AuthorizedGlobalLease,
    DownloadLease,
    LanInterface,
    LanNetworkPolicy,
    P2PManagerError,
    P2PSessionPlan,
    P2PSidecarManager,
    SeedLease,
    current_lan_policy,
    default_p2p_settings,
    session_settings,
    torrent_flags_for_plan,
)
from dinkster_p2p import runtime as p2p_runtime
from dinkster_p2p.diagnostics import NativeDiagnostics
from dinkster_p2p.listeners import ListenerBindings
from dinkster_p2p.runtime import SidecarError, SidecarRuntime, safe_session_settings
from dinkster_server import LanMapping, LanPeerEndpoint

from dinkster import lan_p2p as lan_p2p_module
from dinkster.lan_p2p import LanP2PBackend, LanP2PController


def _settings(*, downloads: bool = False, seeding: bool = False) -> dict[str, object]:
    return {
        **default_p2p_settings(),
        "downloadsEnabled": downloads,
        "seedingEnabled": seeding,
        "scope": "lan-only",
    }


def _safetensors(size: int = P2P_PIECE_LENGTH + 257) -> bytes:
    header = json.dumps(
        {
            "weight": {
                "dtype": "U8",
                "shape": [size],
                "data_offsets": [0, size],
            }
        },
        separators=(",", ":"),
    ).encode()
    return struct.pack("<Q", len(header)) + header + b"s" * size


def _gguf(size: int = P2P_PIECE_LENGTH + 256) -> bytes:
    name = b"weight"
    header = b"".join(
        (
            b"GGUF",
            struct.pack("<IQQ", 3, 1, 0),
            struct.pack("<Q", len(name)),
            name,
            struct.pack("<I", 1),
            struct.pack("<Q", size // 4),
            struct.pack("<IQ", 0, 0),
        )
    )
    header += b"\0" * (-len(header) % 32)
    payload = b"g" * size
    payload += b"\0" * (-len(payload) % 32)
    return header + payload


async def _wait_for(
    sample: Callable[[], Awaitable[Any]],
    predicate: Callable[[Any], bool],
    *,
    timeout: float = 15.0,
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await sample()
        if predicate(value):
            return value
        await asyncio.sleep(0.05)
    raise AssertionError("condition did not become true before timeout")


def _leases(source: Path, suffix: str) -> tuple[SeedLease, DownloadLease]:
    derived = derive_p2p_descriptor(source)
    expires_at = time.time() + 120
    seed = SeedLease(
        version=1,
        kind="seed",
        lease_id=f"seed-{suffix}",
        digest=derived.asset_digest,
        size_bytes=derived.size,
        descriptor=derived.descriptor,
        grant_ids=(derived.descriptor.info_hash,),
        local_path=source.resolve(),
        scope="lan-only",
        expires_at=expires_at,
    )
    download = DownloadLease(
        version=1,
        kind="download",
        lease_id=f"download-{suffix}",
        digest=derived.asset_digest,
        size_bytes=derived.size,
        descriptor=derived.descriptor,
        staging_path=(
            f"{derived.descriptor.info_hash}/{derived.asset_digest.removeprefix('blake3:')}"
        ),
        scope="lan-only",
        expires_at=expires_at,
    )
    return seed, download


def _assert_lan_only_status(status: dict[str, object]) -> None:
    sidecar = status["sidecar"]
    assert isinstance(sidecar, dict)
    assert sidecar["listenInterfaces"] == list(current_lan_policy().addresses)
    assert sidecar["networkFeatures"] == {
        "dht": False,
        "trackers": False,
        "pex": False,
        "lsd": True,
        "upnp": False,
        "natMappings": False,
        "tcp": False,
        "utp": False,
        "natPmp": False,
        "pcp": False,
    }
    process = psutil.Process(int(sidecar["pid"]))
    listen_port = int(sidecar["listenPort"])
    matching = [
        connection
        for connection in process.net_connections(kind="tcp")
        if connection.laddr and connection.laddr.port == listen_port
    ]
    assert matching
    policy = current_lan_policy()
    assert all(connection.laddr.ip in policy.addresses for connection in matching)
    assert all(
        not connection.raddr or policy.allows_peer(connection.raddr.ip) for connection in matching
    )
    for connection in process.net_connections(kind="inet"):
        if connection.raddr:
            remote = ip_address(connection.raddr.ip)
            assert remote.is_loopback or policy.allows_peer(connection.raddr.ip)


def test_shared_session_plan_closes_global_features_without_closing_lan() -> None:
    policy = LanNetworkPolicy(
        (
            LanInterface(
                "ethernet",
                IPv4Address("192.168.10.20"),
                IPv4Network("192.168.10.0/24"),
            ),
        )
    )
    global_plan = P2PSessionPlan(
        lan_active=True,
        global_dht=True,
        global_trackers=True,
        global_pex=True,
        global_tcp=True,
        global_utp=True,
        global_nat_mapping=True,
        dht_bootstrap_nodes="router.example:6881",
    )
    active = session_settings(default_p2p_settings(), policy, global_plan)
    assert active["listen_interfaces"] == "0.0.0.0:0"
    assert active["enable_lsd"] is True
    assert active["enable_dht"] is True
    assert active["apply_filter_to_dht"] is False
    assert active["enable_upnp"] is True
    assert active["enable_natpmp"] is True

    lan_plan = global_plan.close_global()
    closed = session_settings(default_p2p_settings(), policy, lan_plan)
    assert lan_plan == P2PSessionPlan(lan_active=True)
    assert closed["listen_interfaces"] == "192.168.10.20:0l"
    assert closed["outgoing_interfaces"] == "192.168.10.20"
    assert closed["enable_lsd"] is True
    assert closed["enable_incoming_tcp"] is True
    assert closed["enable_outgoing_tcp"] is True
    assert closed["enable_dht"] is False
    assert closed["apply_filter_to_dht"] is True
    assert closed["enable_upnp"] is False
    assert closed["enable_natpmp"] is False
    assert closed["enable_incoming_utp"] is False
    assert closed["enable_outgoing_utp"] is False


def test_session_plan_rejects_global_features_without_an_available_peer_transport() -> None:
    with pytest.raises(ValueError, match="require TCP or uTP"):
        P2PSessionPlan(global_dht=True)
    with pytest.raises(ValueError, match="require TCP while LAN is active"):
        P2PSessionPlan(lan_active=True, global_utp=True)


def test_session_plan_keeps_global_features_off_lan_torrents() -> None:
    class TorrentFlags:
        apply_ip_filter = 1
        override_web_seeds = 2
        disable_dht = 4
        disable_pex = 8
        override_trackers = 16

    libtorrent = type("Libtorrent", (), {"torrent_flags": TorrentFlags})()
    plan = P2PSessionPlan(
        lan_active=True,
        global_dht=True,
        global_trackers=True,
        global_pex=True,
        global_tcp=True,
    )
    global_flags = torrent_flags_for_plan(libtorrent, plan, scope="lan-and-internet")
    lan_flags = torrent_flags_for_plan(libtorrent, plan, scope="lan-only")
    closed_flags = torrent_flags_for_plan(libtorrent, plan.close_global(), scope="lan-and-internet")
    disabled = TorrentFlags.disable_dht | TorrentFlags.disable_pex | TorrentFlags.override_trackers
    assert global_flags & disabled == 0
    assert global_flags & TorrentFlags.apply_ip_filter == 0
    assert lan_flags & disabled == disabled
    assert lan_flags & TorrentFlags.apply_ip_filter == TorrentFlags.apply_ip_filter
    assert closed_flags & disabled == disabled
    assert closed_flags & TorrentFlags.apply_ip_filter == TorrentFlags.apply_ip_filter


def test_runtime_applies_global_plan_and_closes_it_on_the_same_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_p2p.listeners.select_listen_port", lambda *_args: 49152)

    class IpFilter:
        def __init__(self) -> None:
            self.rules: list[tuple[str, str, int]] = []

        def add_rule(self, first: str, last: str, value: int) -> None:
            self.rules.append((first, last, value))

    class TorrentFlags:
        apply_ip_filter = 1
        override_web_seeds = 2
        disable_dht = 4
        disable_pex = 8
        override_trackers = 16

    class Libtorrent:
        torrent_flags = TorrentFlags
        ip_filter = IpFilter

    class Session:
        def __init__(self) -> None:
            self.settings: dict[str, object] = {}
            self.ip_filter: IpFilter | None = None
            self.paused = False
            self.apply_count = 0

        def apply_settings(self, settings: dict[str, object]) -> None:
            self.apply_count += 1
            self.settings.update(settings)

        def set_ip_filter(self, value: IpFilter) -> None:
            self.ip_filter = value

        def add_extension(self, _name: str) -> None:
            pass

        def pause(self) -> None:
            self.paused = True

        def resume(self) -> None:
            self.paused = False

    policy = LanNetworkPolicy(
        (
            LanInterface(
                "ethernet",
                IPv4Address("192.168.10.20"),
                IPv4Network("192.168.10.0/24"),
            ),
        )
    )
    session = Session()
    runtime = object.__new__(SidecarRuntime)
    runtime._diagnostics = NativeDiagnostics()
    runtime.settings = {
        **default_p2p_settings(),
        "downloadsEnabled": True,
        "scope": "lan-only",
    }
    runtime._paused = False
    runtime._network_paused = False
    runtime._session_plan = P2PSessionPlan(lan_active=True)
    runtime._listeners = ListenerBindings()
    runtime._session = session
    runtime._lt = Libtorrent()
    runtime._network_policy = policy
    runtime._torrents = {}
    runtime._leases = {}
    runtime._global = None

    global_plan = P2PSessionPlan(lan_active=True, global_dht=True, global_tcp=True)
    with pytest.raises(SidecarError, match="require lan-and-internet scope"):
        runtime.apply_session_plan(global_plan)
    assert session.settings == {}

    runtime.settings["scope"] = "lan-and-internet"
    runtime.apply_session_plan(global_plan)
    assert runtime._session is session
    assert session.settings["enable_lsd"] is True
    assert session.settings["enable_dht"] is True
    assert session.ip_filter is not None
    assert session.ip_filter.rules == [
        ("0.0.0.0", "255.255.255.255", 1),
        ("192.168.10.0", "192.168.10.255", 0),
    ]

    runtime._global_pex_loaded = False
    runtime._set_global_network_plan(True, False)
    global_plan = runtime._session_plan
    apply_count = session.apply_count
    runtime._set_global_network_plan(True, False)
    assert session.apply_count == apply_count

    runtime._network_paused = True
    runtime._apply_settings()
    assert runtime._session_plan == global_plan
    assert session.paused is True
    assert session.settings["enable_lsd"] is False
    assert session.settings["enable_dht"] is False
    assert session.settings["enable_incoming_tcp"] is False

    runtime._network_paused = False
    runtime.apply_session_plan(global_plan.close_global())
    assert runtime._session is session
    assert session.paused is False
    assert session.settings["enable_lsd"] is True
    assert session.settings["enable_dht"] is False
    assert session.ip_filter is not None
    assert session.ip_filter.rules == [
        ("0.0.0.0", "255.255.255.255", 1),
        ("192.168.10.0", "192.168.10.255", 0),
    ]


def test_session_with_no_eligible_interface_opens_no_peer_listener(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = LanNetworkPolicy(())
    settings = safe_session_settings(default_p2p_settings(), policy)
    assert settings["listen_interfaces"] == ""
    assert settings["outgoing_interfaces"] == ""
    assert settings["enable_dht"] is False
    assert settings["enable_lsd"] is False
    assert settings["enable_upnp"] is False
    assert settings["enable_natpmp"] is False

    monkeypatch.setattr(p2p_runtime, "current_lan_policy", lambda: policy)
    runtime = SidecarRuntime(
        state_root=tmp_path / "vault" / ".p2p",
        vault_root=tmp_path / "vault",
        installation_root=tmp_path / "install",
        settings=_settings(downloads=True),
    )
    try:
        status = runtime.status()
        assert status["listenPort"] is None
        assert status["listenInterfaces"] == []
        assert status["networkFeatures"] == {
            "dht": False,
            "trackers": False,
            "pex": False,
            "lsd": False,
            "upnp": False,
            "natMappings": False,
            "tcp": False,
            "utp": False,
            "natPmp": False,
            "pcp": False,
        }
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("seeding", "failed_bytes", "recorded"),
    [
        (False, 1, "none"),
        (True, 0, "none"),
        (False, 0, "durable_pieces"),
        (False, 0, "pending_flush"),
        (False, 0, "pending_reads"),
    ],
)
def test_download_rejects_wrong_metadata_and_hash_failed_piece(
    tmp_path: Path, seeding: bool, failed_bytes: int, recorded: str
) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(_safetensors(256))
    seed, download = _leases(source, "untrusted")

    class Handle:
        def torrent_file(self) -> Any:
            return type("TorrentFile", (), {"info_section": lambda self: b"wrong"})()

        def status(self) -> Any:
            return type("Status", (), {"total_failed_bytes": failed_bytes})()

        def trackers(self) -> list[object]:
            return []

        def url_seeds(self) -> list[object]:
            return []

        def http_seeds(self) -> list[object]:
            return []

        def is_valid(self) -> bool:
            return True

    handle = Handle()
    partial = AssetVault(tmp_path / "vault").open_p2p_partial(
        download.descriptor, download.digest, download.size_bytes
    )
    torrent = p2p_runtime._TorrentRuntime(download, handle, partial)
    runtime = object.__new__(SidecarRuntime)
    runtime._diagnostics = NativeDiagnostics()
    with pytest.raises(SidecarError, match="canonical descriptor info"):
        runtime._validate_download_metadata(torrent)
    if seeding:
        torrent.lease = seed
    if recorded != "none":
        getattr(torrent, recorded).add(0)

    class IgnoredAlert:
        pass

    class HashFailedAlert:
        def __init__(self) -> None:
            self.handle = handle
            self.piece_index = 0

        def message(self) -> str:
            return "piece hash failed"

    class FakeLibtorrent:
        listen_succeeded_alert = IgnoredAlert
        listen_failed_alert = IgnoredAlert
        metadata_failed_alert = IgnoredAlert
        torrent_error_alert = IgnoredAlert
        file_error_alert = IgnoredAlert
        hash_failed_alert = HashFailedAlert
        metadata_received_alert = IgnoredAlert
        torrent_checked_alert = IgnoredAlert
        piece_finished_alert = IgnoredAlert
        cache_flushed_alert = IgnoredAlert
        read_piece_alert = IgnoredAlert

    removed: list[object] = []
    runtime._lt = FakeLibtorrent()
    runtime._session = type(
        "Session",
        (),
        {"remove_torrent": lambda self, value: removed.append(value)},
    )()
    runtime._torrents = {download.lease_id: torrent}
    runtime._handle_alert(HashFailedAlert())
    assert torrent.state == "failed"
    assert torrent.error == "piece hash failed"
    assert torrent.stopped
    assert removed == [handle]


def test_seed_lease_waits_for_native_tcp_listener_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = LanNetworkPolicy(
        (LanInterface("test", IPv4Address("127.0.0.1"), IPv4Network("127.0.0.0/8")),)
    )
    monkeypatch.setattr(p2p_runtime, "current_lan_policy", lambda: policy)
    source = tmp_path / "model.safetensors"
    source.write_bytes(_safetensors(256))
    seed, _download = _leases(source, "listener")
    runtime = SidecarRuntime(
        state_root=tmp_path / "state",
        vault_root=tmp_path / "vault",
        installation_root=None,
        settings=_settings(seeding=True),
    )
    handle_alert = runtime._handle_alert
    tcp_bindings: list[tuple[str, int]] = []
    udp_count = 0

    def delay_tcp_confirmation(alert: Any) -> None:
        nonlocal udp_count
        if isinstance(alert, runtime._lt.listen_succeeded_alert):
            if alert.socket_type == runtime._lt.socket_type_t.tcp:
                tcp_bindings.append((alert.address, alert.port))
                return
            if alert.socket_type == runtime._lt.socket_type_t.udp:
                udp_count += 1
        handle_alert(alert)

    monkeypatch.setattr(runtime, "_handle_alert", delay_tcp_confirmation)
    try:
        runtime.grant(seed.to_wire(), "seed")
        torrent = runtime._torrents[seed.lease_id]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            runtime.poll_alerts()
            if torrent.state == "ready" and tcp_bindings and udp_count:
                break
            time.sleep(0.01)
        assert torrent.state == "ready" and tcp_bindings and udp_count
        assert runtime._lease_status(seed)["state"] == "checking"
        for address, port in tcp_bindings:
            runtime._listeners.tcp_succeeded(address, port)
        assert runtime._lease_status(seed)["state"] == "ready"
        runtime.configure(_settings(seeding=True))
        assert runtime._lease_status(seed)["state"] == "ready"
        runtime.set_network_paused({"paused": True})
        assert runtime._lease_status(seed)["state"] == "paused"
        runtime.set_network_paused({"paused": False})
        assert runtime._lease_status(seed)["state"] == "checking"
        runtime.resume_transfer({"digest": seed.digest})
        monkeypatch.setattr(runtime, "_handle_alert", handle_alert)
        for address, port in tcp_bindings:
            runtime._listeners.tcp_succeeded(address, port)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            runtime.poll_alerts()
            if runtime._lease_status(seed)["state"] == "ready":
                break
            time.sleep(0.01)
        assert runtime._lease_status(seed)["state"] == "ready"
        assert runtime._leases[seed.lease_id] == seed
    finally:
        runtime.close()


def test_zero_durable_bytes_does_not_reset_transfer_stall(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(_safetensors(256))
    _seed, download = _leases(source, "stalled")

    class Manager:
        revoked: list[str] = []
        granted: list[DownloadLease] = []

        async def grant_download(self, lease: DownloadLease) -> dict[str, object]:
            self.granted.append(lease)
            return {}

        async def lease_status(self, _lease_id: str) -> dict[str, object]:
            await asyncio.sleep(0)
            return {"state": "downloading", "durableBytes": 0}

        async def revoke(self, lease_id: str) -> dict[str, object]:
            self.revoked.append(lease_id)
            return {}

    manager = Manager()
    backend = LanP2PBackend(cast("P2PSidecarManager", manager))
    candidate = TransportCandidate(
        "lan-p2p",
        "http://192.168.1.2:1234",
        size_bytes=download.size_bytes,
        descriptor=download.descriptor,
        expires_at=download.expires_at,
        peer_address="192.168.1.2",
        peer_port=51413,
    )

    async def scenario() -> None:
        events = backend.transfer(download.digest, candidate, stall_timeout=0.01)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(events), 0.05)
        assert len(manager.granted) == 1
        assert manager.granted[0].version == 2
        assert manager.granted[0].peer_address == "192.168.1.2"
        assert manager.granted[0].peer_port == 51413
        assert len(manager.revoked) == 1
        assert manager.granted[0].lease_id == manager.revoked[0]

    asyncio.run(scenario())


def test_version_1_mapping_remains_a_usable_lsd_only_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(_safetensors(256))
    derived = derive_p2p_descriptor(source)
    expires_at = time.time() + 120
    mapping = LanMapping(
        LanPeerEndpoint("legacy", IPv4Address("192.168.70.44"), 42069),
        derived.asset_digest,
        derived.size,
        derived.descriptor,
        None,
    )

    async def discover(*_args: object, **_kwargs: object) -> tuple[LanMapping, ...]:
        return (mapping,)

    monkeypatch.setattr(lan_p2p_module, "discover_lan_mappings", discover)
    controller = LanP2PController(
        vault=AssetVault(tmp_path / "vault"),
        resolver_indexes=ResolverSubscriptionStore(
            tmp_path / "subscriptions.json",
            ProvenanceStore(tmp_path / "provenance.json"),
        ),
        receipts=PublicAcquisitionReceiptStore(tmp_path / "receipts.json"),
        local_path_for=lambda _digest: None,
    )
    controller._settings = {"downloadsEnabled": True}
    controller._discovery = cast("Any", object())
    controller._authorities = {
        derived.asset_digest: ((derived.size, derived.descriptor, expires_at),)
    }

    async def scenario() -> None:
        (candidate,) = await controller._discover(derived.asset_digest)
        assert candidate.peer_address is None
        assert candidate.peer_port is None

    asyncio.run(scenario())


def test_restored_version_2_lease_connects_once_on_exact_grant_retry(tmp_path: Path) -> None:
    policy = current_lan_policy()
    if not policy.interfaces:
        pytest.skip("host has no eligible LAN interface")
    source = tmp_path / "model.safetensors"
    source.write_bytes(_safetensors(256))
    _seed, download = _leases(source, "restored-hint")
    hinted = replace(
        download,
        version=2,
        peer_address=str(policy.interfaces[0].address),
        peer_port=51413,
    )
    arguments = {
        "state_root": tmp_path / "vault" / ".p2p",
        "vault_root": tmp_path / "vault",
        "installation_root": None,
        "settings": _settings(downloads=True),
    }
    original = SidecarRuntime(**arguments)
    try:
        original.grant(hinted.to_wire(), "download")
    finally:
        original.close()

    restored = SidecarRuntime(**arguments)
    real_activate = restored._activate_lease
    real_save_state = restored.save_state
    real_stop_torrent = restored._stop_torrent
    connected: list[tuple[str, int]] = []
    fail_connection = [True]

    class HandleProxy:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def connect_peer(self, endpoint: tuple[str, int]) -> None:
            connected.append(endpoint)
            if fail_connection[0]:
                raise RuntimeError("injected peer connection failure")
            self._handle.connect_peer(endpoint)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    def activate(*args: Any, **kwargs: Any) -> Any:
        runtime = real_activate(*args, **kwargs)
        runtime.handle = HandleProxy(runtime.handle)
        return runtime

    def stop_torrent(torrent: Any) -> None:
        if isinstance(torrent.handle, HandleProxy):
            torrent.handle = torrent.handle._handle
        real_stop_torrent(torrent)

    restored._activate_lease = activate
    restored._stop_torrent = stop_torrent
    try:
        assert restored._leases[hinted.lease_id] == hinted
        assert hinted.lease_id not in restored._torrents
        restored._network_policy = LanNetworkPolicy(())
        with pytest.raises(SidecarError, match="outside the active LAN"):
            restored.grant(hinted.to_wire(), "download")
        restored._network_policy = policy
        with pytest.raises(RuntimeError, match="injected peer connection failure"):
            restored.grant(hinted.to_wire(), "download")
        assert hinted.lease_id not in restored._torrents

        fail_connection[0] = False

        def fail_save_state() -> None:
            raise OSError("injected persistence failure")

        restored.save_state = fail_save_state
        with pytest.raises(OSError, match="injected persistence failure"):
            restored.grant(hinted.to_wire(), "download")
        assert hinted.lease_id not in restored._torrents

        restored.save_state = real_save_state
        restored.grant(hinted.to_wire(), "download")
        restored.grant(hinted.to_wire(), "download")
        assert connected == [(hinted.peer_address, hinted.peer_port)] * 3
    finally:
        restored.close()


def test_restored_inactive_lease_does_not_consume_active_seed_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_source = tmp_path / "first.safetensors"
    first_source.write_bytes(_safetensors(256))
    second_source = tmp_path / "second.safetensors"
    second_source.write_bytes(_safetensors(257))
    first_seed, _download = _leases(first_source, "first")
    second_seed, _download = _leases(second_source, "second")
    policy = LanNetworkPolicy(())
    monkeypatch.setattr(p2p_runtime, "current_lan_policy", lambda: policy)
    monkeypatch.setattr(p2p_runtime, "MAX_ACTIVE_SEEDS", 1)
    arguments = {
        "state_root": tmp_path / "vault" / ".p2p",
        "vault_root": tmp_path / "vault",
        "installation_root": tmp_path / "install",
        "settings": _settings(seeding=True),
    }

    first = SidecarRuntime(**arguments)
    try:
        first.grant(first_seed.to_wire(), "seed")
    finally:
        first.close()

    restored = SidecarRuntime(**arguments)
    try:
        assert restored._lease_status(first_seed)["state"] == "inactive"
        assert restored.grant(second_seed.to_wire(), "seed")["leaseId"] == second_seed.lease_id
    finally:
        restored.close()


@pytest.mark.parametrize("format_name", ["safetensors", "gguf"])
@pytest.mark.parametrize("corrupt_resumed_piece", [False, True])
def test_download_distinguishes_sparse_bytes_from_corrupt_resumed_piece(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    corrupt_resumed_piece: bool,
) -> None:
    policy = LanNetworkPolicy(
        (LanInterface("test", IPv4Address("127.0.0.1"), IPv4Network("127.0.0.0/8")),)
    )
    monkeypatch.setattr(p2p_runtime, "current_lan_policy", lambda: policy)
    source = tmp_path / f"model.{format_name}"
    data = _safetensors() if format_name == "safetensors" else _gguf()
    source.write_bytes(data)
    seed, download = _leases(source, "preallocated")
    if corrupt_resumed_piece:
        partial = AssetVault(tmp_path / "download-vault").open_p2p_partial(
            download.descriptor, download.digest, download.size_bytes
        )
        partial.write_piece(0, data[:P2P_PIECE_LENGTH])
        with partial.path.open("r+b") as stream:
            stream.write(b"\0" * P2P_PIECE_LENGTH)
        assert partial.completed_ranges == ((0, P2P_PIECE_LENGTH),)
    downloader = SidecarRuntime(
        state_root=tmp_path / "download-vault" / ".p2p",
        vault_root=tmp_path / "download-vault",
        installation_root=None,
        settings=_settings(downloads=True),
    )
    seeder = None
    try:
        downloader.grant(download.to_wire(), "download")
        torrent = downloader._torrents[download.lease_id]
        torrent.handle.set_metadata(derive_p2p_descriptor(source).info)
        rejected = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not rejected:
            for alert in downloader._session.pop_alerts():
                if isinstance(alert, downloader._lt.hash_failed_alert):
                    rejected.append(alert.message())
                downloader._handle_alert(alert)
            time.sleep(0.01)
        assert rejected, "initial sparse-file scan did not reject the placeholder bytes"
        if corrupt_resumed_piece:
            assert torrent.state == "failed"
            assert torrent.stopped
            assert torrent.error in rejected
            assert torrent.published_path is None
            return
        assert torrent.state == "downloading", torrent.error
        assert torrent.handle.status().total_failed_bytes == 0
        assert torrent.handle.status().total_payload_download == 0
        assert not torrent.stopped
        assert not torrent.durable_pieces
        assert torrent.partial is not None and torrent.partial.completed_ranges == ()

        seeder = SidecarRuntime(
            state_root=tmp_path / "seed-vault" / ".p2p",
            vault_root=tmp_path / "seed-vault",
            installation_root=None,
            settings=_settings(seeding=True),
        )
        seeder.grant(seed.to_wire(), "seed")
        deadline = time.monotonic() + 5
        seed_port = 0
        while time.monotonic() < deadline:
            seeder.poll_alerts()
            seed_port = seeder._session.listen_port()
            if seeder._torrents[seed.lease_id].state == "ready" and seed_port > 0:
                break
            time.sleep(0.01)
        assert seeder._torrents[seed.lease_id].state == "ready"
        assert seed_port > 0, seeder._listener_errors

        class LoopbackPolicy:
            interfaces = policy.interfaces
            addresses = policy.addresses

            def allows_peer(self, value: str | IPv4Address) -> bool:
                return str(value) == "127.0.0.1"

        downloader._network_policy = cast("LanNetworkPolicy", LoopbackPolicy())
        with pytest.raises(SidecarError, match="outside the active LAN"):
            downloader.grant(
                replace(
                    download,
                    version=2,
                    peer_address="198.51.100.10",
                    peer_port=seed_port,
                ).to_wire(),
                "download",
            )
        downloader.revoke({"leaseId": download.lease_id})
        hinted_download = replace(
            download,
            version=2,
            peer_address="127.0.0.1",
            peer_port=seed_port,
        )
        downloader.grant(hinted_download.to_wire(), "download")
        torrent = downloader._torrents[download.lease_id]
        torrent.handle.set_metadata(derive_p2p_descriptor(source).info)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and torrent.state not in {"complete", "failed"}:
            seeder.poll_alerts()
            downloader.poll_alerts()
            time.sleep(0.01)
        if torrent.state != "complete":
            native = torrent.handle.status() if torrent.handle.is_valid() else None
            pytest.fail(
                str(
                    {
                        "state": torrent.state,
                        "error": torrent.error,
                        "seed_port": seed_port,
                        "current_seed_port": seeder._session.listen_port(),
                        "listener_errors": seeder._listener_errors,
                        "native_state": str(native.state) if native is not None else None,
                        "native_pieces": list(native.pieces) if native is not None else None,
                        "peers": native.num_peers if native is not None else None,
                        "payload_bytes": native.total_payload_download
                        if native is not None
                        else None,
                        "durable_pieces": sorted(torrent.durable_pieces),
                        "pending_flush": sorted(torrent.pending_flush),
                        "pending_reads": sorted(torrent.pending_reads),
                    }
                )
            )
        assert torrent.published_path is not None
        assert Path(torrent.published_path).read_bytes() == data
        assert torrent.durable_pieces == {0, 1}
    finally:
        downloader.close()
        if seeder is not None:
            seeder.close()


def test_two_sidecars_transfer_safetensors_and_gguf_over_lsd_only(tmp_path: Path) -> None:
    if not current_lan_policy().interfaces:
        pytest.skip("host has no eligible LAN interface")

    async def scenario() -> None:
        sources = (
            (tmp_path / "model.safetensors", _safetensors(), "safetensors"),
            (tmp_path / "model.gguf", _gguf(), "gguf"),
        )
        for path, data, _kind in sources:
            path.write_bytes(data)
        leases = tuple(_leases(path, kind) for path, _data, kind in sources)
        seeder = P2PSidecarManager(vault_root=tmp_path / "seeder-vault")
        downloader = P2PSidecarManager(vault_root=tmp_path / "downloader-vault")
        await seeder.start(_settings(seeding=True))
        await downloader.start(_settings(downloads=True))
        try:
            for seed, download in leases:
                assert seed.descriptor.piece_length == 8 * 1024 * 1024
                await seeder.grant_seed(seed)
                await _wait_for(
                    lambda lease_id=seed.lease_id: seeder.lease_status(lease_id),
                    lambda value: value["state"] == "ready",
                )
                await downloader.grant_download(download)
                completed = await _wait_for(
                    lambda lease_id=download.lease_id: downloader.lease_status(lease_id),
                    lambda value: value["state"] in {"complete", "failed"},
                    timeout=30.0,
                )
                assert completed["state"] == "complete", completed.get("error", completed)
                assert completed["durableBytes"] == download.size_bytes
                published = Path(str(completed["path"]))
                assert published.read_bytes() == seed.local_path.read_bytes()
                assert published.with_name(published.name + ".verified.json").is_file()

            _assert_lan_only_status(await seeder.status())
            _assert_lan_only_status(await downloader.status())

            unsafe = tmp_path / "unsafe.bin"
            unsafe.write_bytes(b"not a supported model format")
            unsafe_seed, unsafe_download = _leases(unsafe, "unsafe")
            await seeder.grant_seed(unsafe_seed)
            await _wait_for(
                lambda: seeder.lease_status(unsafe_seed.lease_id),
                lambda value: value["state"] == "ready",
            )
            await downloader.grant_download(unsafe_download)
            rejected = await _wait_for(
                lambda: downloader.lease_status(unsafe_download.lease_id),
                lambda value: value["state"] == "failed",
            )
            assert "safetensors header size is invalid" in str(rejected["error"])
            assert (
                not (tmp_path / "downloader-vault")
                .joinpath(
                    unsafe_download.digest[7:9],
                    unsafe_download.digest[7:],
                )
                .exists()
            )
            await downloader.remove_partial(unsafe_download.lease_id)
            await seeder.revoke(unsafe_seed.lease_id)

            forged = tmp_path / "forged.safetensors"
            forged.write_bytes(b"x" * leases[0][0].size_bytes)
            with pytest.raises(P2PManagerError, match="verification failed"):
                await seeder.grant_seed(
                    replace(
                        leases[0][0],
                        lease_id="forged-seed",
                        local_path=forged.resolve(),
                    )
                )
        finally:
            await downloader.close()
            await seeder.close()

    asyncio.run(scenario())


def _controller_inputs(
    root: Path,
    source: Path,
) -> tuple[
    AssetVault,
    AssetVault,
    ResolverSubscriptionStore,
    PublicAcquisitionReceiptStore,
    PublicAcquisitionReceiptStore,
    str,
]:
    derived = derive_p2p_descriptor(source)
    now = time.time()
    index_path = root / "resolver.json"
    index_path.write_text(
        json.dumps(
            {
                "dinksterResolver": 1,
                "name": "fixture-provider",
                "updated": "2026-09-01T00:00:00Z",
                "entries": [
                    {
                        "digest": derived.asset_digest,
                        "name": source.name,
                        "urls": ["https://models.example/model.safetensors"],
                        "size": derived.size,
                        "license": "apache-2.0",
                        "p2p": derived.descriptor.to_wire(),
                    }
                ],
            }
        ),
        "utf-8",
    )
    indexes = ResolverSubscriptionStore(
        root / "subscriptions.json",
        ProvenanceStore(root / "provenance.json"),
        clock=lambda: now,
    )
    indexes.subscribe(str(index_path))
    public_source = indexes.public_sources()[0]
    receipt = PublicAcquisitionReceiptV1(
        version=1,
        receipt_id="1" * 32,
        digest=public_source.digest,
        size_bytes=public_source.size_bytes,
        source_type=public_source.source_type,
        source_id=public_source.source_id,
        source_revision=public_source.source_revision,
        listed_url=public_source.listed_urls[0],
        final_url=public_source.listed_urls[0],
        fetched_at=now,
    )
    receipt_path = root / "receipts.json"
    receipt_path.write_text(
        json.dumps(
            {
                "publicAcquisitionReceipts": 1,
                "receipts": [receipt.to_wire()],
            }
        ),
        "utf-8",
    )
    seeder_vault = AssetVault(root / "seeder-vault")
    with source.open("rb") as handle, seeder_vault.writer(derived.asset_digest) as writer:
        while chunk := handle.read(1024 * 1024):
            writer.write(chunk)
        writer.commit()
    return (
        seeder_vault,
        AssetVault(root / "downloader-vault"),
        indexes,
        PublicAcquisitionReceiptStore(receipt_path),
        PublicAcquisitionReceiptStore(root / "empty-receipts.json"),
        derived.asset_digest,
    )


@pytest.mark.parametrize("license_id", ["Apache-2.0", "", "LicenseRef-Custom"])
def test_controller_maps_trusted_resolver_snapshot_to_global_download_and_tombstone_revoke(
    tmp_path: Path,
    license_id: str,
) -> None:
    source = tmp_path / "source.safetensors"
    source.write_bytes(_safetensors(256))
    derived = derive_p2p_descriptor(source)
    now = [time.time()]
    index_path = tmp_path / "resolver.json"

    def write_index(*, include_p2p: bool) -> None:
        entry: dict[str, object] = {
            "digest": derived.asset_digest,
            "name": source.name,
            "urls": ["https://models.example/model.safetensors"],
            "size": derived.size,
            "license": license_id,
            "gated": True,
        }
        if include_p2p:
            entry["p2p"] = derived.descriptor.to_wire()
        index_path.write_text(
            json.dumps(
                {
                    "dinksterResolver": 1,
                    "name": "not-special-cased",
                    "updated": "2026-09-01T00:00:00Z",
                    "entries": [entry],
                }
            ),
            "utf-8",
        )

    write_index(include_p2p=True)
    indexes = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
        clock=lambda: now[0],
    )
    subscription = indexes.subscribe(str(index_path))
    indexes.set_p2p_trust(
        subscription.id,
        trusted_for_p2p=True,
        license_authoritative=False,
    )

    class Manager:
        global_network_policy = ("unmetered", False)

        def __init__(self) -> None:
            self.reconciliations: list[tuple[AuthorizedGlobalLease, ...]] = []

        async def reconcile_global(
            self, authorizations: tuple[AuthorizedGlobalLease, ...]
        ) -> dict[str, tuple[str, ...]]:
            self.reconciliations.append(authorizations)
            return {"granted": (), "revoked": ()}

    async def scenario() -> None:
        controller = LanP2PController(
            vault=AssetVault(tmp_path / "vault"),
            resolver_indexes=indexes,
            receipts=PublicAcquisitionReceiptStore(tmp_path / "receipts.json"),
            local_path_for=lambda _digest: None,
            clock=lambda: now[0],
        )
        manager = Manager()
        controller._manager = cast("P2PSidecarManager", manager)
        controller._settings = default_p2p_settings()

        await controller.reconcile()
        assert manager.reconciliations == [()]
        (candidate,) = controller._known_transports(derived.asset_digest)
        assert candidate.kind == "global-p2p"
        lease_id = await controller._request_global_download(derived.asset_digest, candidate.source)
        assert lease_id == f"global:download:{candidate.source}"
        (authorization,) = manager.reconciliations[-1]
        assert isinstance(authorization.lease, DownloadLease)
        assert authorization.lease.digest == derived.asset_digest
        assert authorization.trackers == ()

        now[0] += 1
        write_index(include_p2p=False)
        indexes.refresh(subscription.id)
        await controller.reconcile()
        assert manager.reconciliations[-1] == ()
        assert controller._known_transports(derived.asset_digest) == ()

    asyncio.run(scenario())


def test_controller_requires_receipt_then_maps_transfers_revokes_and_preserves_lan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not current_lan_policy().interfaces:
        pytest.skip("host has no eligible LAN interface")
    source = tmp_path / "source.safetensors"
    contents = bytearray(_safetensors())
    # Parallel test processes share mDNS, so each fixture needs a distinct digest.
    fixture_id = hashlib.sha256(str(tmp_path).encode()).digest()
    contents[-len(fixture_id) :] = fixture_id
    source.write_bytes(contents)
    seeder_vault, downloader_vault, indexes, receipts, empty_receipts, digest = _controller_inputs(
        tmp_path, source
    )

    async def scenario() -> None:
        unauthorized = LanP2PController(
            vault=seeder_vault,
            resolver_indexes=indexes,
            receipts=empty_receipts,
            local_path_for=seeder_vault.resolve,
        )
        await unauthorized.start(_settings(seeding=True))
        try:
            await unauthorized.reconcile()
            assert (await unauthorized.status())["lan"]["mappedDigests"] == []  # type: ignore[index]
        finally:
            await unauthorized.close()

        subscription = indexes.subscriptions()[0]
        indexes.set_p2p_trust(
            subscription.id,
            trusted_for_p2p=True,
            license_authoritative=True,
        )

        seeder = LanP2PController(
            vault=seeder_vault,
            resolver_indexes=indexes,
            receipts=receipts,
            local_path_for=seeder_vault.resolve,
        )
        downloader = LanP2PController(
            vault=downloader_vault,
            resolver_indexes=indexes,
            receipts=empty_receipts,
            local_path_for=downloader_vault.resolve,
        )
        await seeder.start(_settings(seeding=True))
        await downloader.start(_settings(downloads=True))
        try:
            seeded = await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is not None,
            )
            assert seeded["lan"]["mappedDigests"] == [digest]
            resolved = await asyncio.to_thread(downloader.resolve_sync, digest)
            assert resolved is not None
            assert resolved.read_bytes() == source.read_bytes()
            _assert_lan_only_status(await seeder.status())
            _assert_lan_only_status(await downloader.status())

            lease_status = seeder._manager.lease_status

            async def checking(lease_id: str) -> dict[str, Any]:
                return {**await lease_status(lease_id), "state": "checking"}

            authorized = dict(seeder._seed_leases)
            monkeypatch.setattr(seeder._manager, "lease_status", checking)
            checking_status = await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is None,
            )
            assert checking_status["lan"]["mappedDigests"] == []
            assert seeder._seed_leases == authorized
            monkeypatch.setattr(seeder._manager, "lease_status", lease_status)
            await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is not None,
            )

            async def unavailable(_lease_id: str) -> dict[str, Any]:
                raise P2PManagerError("sidecar unavailable")

            monkeypatch.setattr(seeder._manager, "lease_status", unavailable)
            unavailable_status = await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is None,
            )
            assert unavailable_status["lan"]["mappedDigests"] == []
            monkeypatch.setattr(seeder._manager, "lease_status", lease_status)
            await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is not None,
            )

            policy_provider = lan_p2p_module.current_lan_policy
            assert seeder.process is not None
            previous_pid = seeder.process.pid
            monkeypatch.setattr(
                lan_p2p_module,
                "current_lan_policy",
                lambda: LanNetworkPolicy(()),
            )
            await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is None,
            )
            assert seeder.process is not None and seeder.process.pid != previous_pid
            monkeypatch.setattr(lan_p2p_module, "current_lan_policy", policy_provider)
            await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is not None,
            )
            before_metered = await seeder.status()
            mapping_port = before_metered["lan"]["mappingPort"]  # type: ignore[index]
            assert seeder.process is not None
            sidecar_pid = seeder.process.pid

            metered = {
                **_settings(seeding=True),
                "networkCostOverride": "metered",
            }
            await seeder.update(metered)
            await seeder.set_global_network_policy("metered", True)
            metered_status = await seeder.status()
            assert metered_status["lan"] == {
                "networkAllowed": True,
                "mappingPort": mapping_port,
                "mappedDigests": [digest],
            }
            assert seeder.process is not None and seeder.process.pid == sidecar_pid
            assert metered_status["sidecar"]["networkPaused"] is False  # type: ignore[index]
            assert metered_status["sidecar"]["networkFeatures"]["lsd"] is True  # type: ignore[index]

            await seeder.set_network_paused(True)
            paused = await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is None,
            )
            assert paused["lan"] == {
                "networkAllowed": False,
                "mappingPort": None,
                "mappedDigests": [],
            }
            assert paused["sidecar"]["networkPaused"] is True  # type: ignore[index]
            assert paused["sidecar"]["networkFeatures"]["lsd"] is False  # type: ignore[index]

            await seeder.set_network_paused(False)
            await seeder.resume_transfer(digest)
            await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is not None,
            )

            await seeder.update({**_settings(seeding=True), "networkCostOverride": "unmetered"})
            await seeder.set_global_network_policy("unmetered", False)
            await _wait_for(
                seeder.status,
                lambda value: value["lan"]["mappingPort"] is not None,
            )
            subscription_id = indexes.subscriptions()[0].id
            assert indexes.unsubscribe(subscription_id)
            await seeder.reconcile()
            revoked = await _wait_for(
                seeder.status,
                lambda value: (
                    value["lan"]["mappedDigests"] == [] and value["lan"]["mappingPort"] is None
                ),
            )
            assert revoked["lan"]["mappingPort"] is None
            assert revoked["sidecar"]["leases"] == []  # type: ignore[index]
        finally:
            await downloader.close()
            await seeder.close()

    asyncio.run(scenario())
