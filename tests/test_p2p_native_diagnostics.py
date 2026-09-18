from __future__ import annotations

import asyncio
import json
import socket
import struct
import sys
import time
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from dinkster_p2p import LanInterface, LanNetworkPolicy
from dinkster_p2p import runtime as runtime_module
from dinkster_p2p.diagnostics import MAX_EVENTS, MAX_LISTENERS, MAX_TORRENTS, NativeDiagnostics
from dinkster_p2p.runtime import SidecarRuntime

from tests.test_lan_p2p_runtime import _leases, _safetensors, _settings
from tests.test_p2p_global_sidecar import (
    _accounting_controller,
    _AccountingHandle,
    _global_settings,
    _peer,
    _seed_lease,
)


class PeerError(SimpleNamespace):
    def message(self) -> str:
        raise AssertionError("raw native messages must not be read")


class Bound(PeerError):
    pass


class Failed(PeerError):
    pass


@pytest.fixture
def native() -> SimpleNamespace:
    return SimpleNamespace(
        peer_error_alert=PeerError,
        listen_succeeded_alert=Bound,
        listen_failed_alert=Failed,
    )


def snapshot(history: NativeDiagnostics) -> dict:
    return history.snapshot(lsd_peer_events_available=False)


def test_history_bounds_unread_events_and_excludes_raw_messages(native: SimpleNamespace) -> None:
    history = NativeDiagnostics()
    event = PeerError(ip=("127.0.0.1", 49152), error=SimpleNamespace(value=lambda: 111), op=1)
    for _ in range(MAX_EVENTS * 10):
        history.observe_alert(event, native, "blake3:" + "a" * 64)
    result = snapshot(history)
    assert len(result["events"]) == MAX_EVENTS
    assert result["overwrittenEvents"] == MAX_EVENTS * 9
    assert result["events"][-1]["errorCode"] == 111
    assert len(json.dumps(result).encode()) < 100_000
    result["events"][-1]["address"] = "reader mutation"
    assert snapshot(history)["events"][-1]["address"] == "127.0.0.1"


@pytest.mark.parametrize(
    "endpoint",
    [
        ("https://user:secret@host/?token=secret", 1),
        ("x" * 10000, 1),
        ("fe80::1%token=secret", 1),
        ("127.0.0.1", -1),
        ("127.0.0.1", True),
        ("127.0.0.1", 65536),
        (),
        None,
    ],
)
def test_malformed_peer_endpoints_are_omitted(native: SimpleNamespace, endpoint: object) -> None:
    history = NativeDiagnostics()
    history.observe_alert(PeerError(ip=endpoint), native, None)
    assert snapshot(history)["events"] == []
    assert snapshot(history)["malformedEvents"] == 1


def test_listener_observations_follow_each_configured_interface(native: SimpleNamespace) -> None:
    history = NativeDiagnostics()
    history.configure_listeners("127.0.0.1:49152l,127.0.0.2:49153l")
    for address, port in (("127.0.0.1", 49152), ("127.0.0.2", 49153)):
        history.observe_alert(Bound(address=address, port=port, socket_type=0), native, None)
    assert len(snapshot(history)["listeners"]) == 2
    history.observe_alert(Failed(address="127.0.0.1", port=49152, socket_type=0), native, None)
    assert [entry["address"] for entry in snapshot(history)["listeners"]] == ["127.0.0.2"]
    history.configure_listeners("127.0.0.1:49154l,127.0.0.2:49153l")
    history.observe_alert(Bound(address="127.0.0.1", port=49152, socket_type=0), native, None)
    assert [entry["address"] for entry in snapshot(history)["listeners"]] == ["127.0.0.2"]
    history.configure_listeners("")
    assert snapshot(history)["listeners"] == []


def test_listener_and_upload_samples_are_bounded(native: SimpleNamespace) -> None:
    history = NativeDiagnostics()
    history.configure_listeners("0.0.0.0:49152")
    for index in range(1, MAX_LISTENERS + 3):
        history.observe_alert(
            Bound(address=f"10.0.0.{index}", port=49152, socket_type=0), native, None
        )
    for index in range(MAX_TORRENTS + 2):
        history.sample_upload(
            f"{index:064x}",
            None,
            SimpleNamespace(all_time_upload=0, total_payload_upload=123, num_peers=1),
            None,
            None,
        )
    result = snapshot(history)
    assert len(result["listeners"]) == MAX_LISTENERS
    assert result["omittedListeners"] == 2
    assert len(result["uploadSamples"]) == MAX_TORRENTS
    assert result["omittedUploadSamples"] == 2
    assert result["uploadSamples"][-1]["nativePayloadUpload"] == 123
    assert result["uploadSamples"][-1]["digest"] is None
    assert len(json.dumps(result).encode()) < 100_000
    history.clear()
    assert snapshot(history)["events"] == []
    assert snapshot(history)["listeners"] == []
    assert snapshot(history)["uploadSamples"] == []


def test_native_loss_survives_event_eviction(native: SimpleNamespace) -> None:
    history = NativeDiagnostics()
    native.alerts_dropped_alert = SimpleNamespace
    history.observe_alert(SimpleNamespace(), native, None)
    for _ in range(MAX_EVENTS):
        history.selection_failed()
    result = snapshot(history)
    assert result["nativeAlertLossObserved"] is True
    assert all(event["kind"] != "native_alerts_dropped" for event in result["events"])
    history.clear()
    assert snapshot(history)["nativeAlertLossObserved"] is False


def test_upload_samples_preserve_disagreement_and_omit_invalid_fields() -> None:
    history = NativeDiagnostics()
    status = SimpleNamespace(all_time_upload=0, total_payload_upload=512120, num_peers=1)
    for _ in range(3):
        history.sample_upload("a" * 64, "blake3:" + "b" * 64, status, 0, 0)
    result = snapshot(history)
    assert len(result["events"]) == 1
    assert result["uploadSamples"][0]["nativePayloadUpload"] == 512120
    assert result["uploadSamples"][0]["lanApplicationUploadedBytes"] == 0
    assert result["uploadSamples"][0]["lanTransportUploadSample"] == 0
    status.all_time_upload = -1
    status.total_payload_upload = "https://user:secret@host"
    status.num_peers = 1 << 65
    history.sample_upload("a" * 64, "secret", status, None, None)
    latest = snapshot(history)["uploadSamples"][0]
    assert latest["digest"] is None
    assert latest["nativeAllTimeUpload"] is None
    assert latest["nativePayloadUpload"] is None
    assert latest["nativePeers"] is None
    assert "secret" not in json.dumps(snapshot(history))


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    policy = LanNetworkPolicy(
        (LanInterface("test", IPv4Address("127.0.0.1"), IPv4Network("127.0.0.0/8")),)
    )
    monkeypatch.setattr(runtime_module, "current_lan_policy", lambda: policy)
    instance = SidecarRuntime(
        state_root=tmp_path / "state",
        vault_root=tmp_path / "vault",
        installation_root=None,
        settings=_settings(downloads=True),
    )
    instance._session.apply_settings({"enable_lsd": False})
    try:
        yield instance
    finally:
        instance.close()


def test_native_session_restart_does_not_restore_diagnostics(runtime: SidecarRuntime) -> None:
    runtime._record_listener_error("discarded")
    runtime.save_state()
    runtime.close()
    assert snapshot(runtime._diagnostics)["events"] == []
    restored = SidecarRuntime(
        state_root=runtime.state_root,
        vault_root=runtime.vault_root,
        installation_root=None,
        settings=runtime.settings,
        network_paused=True,
    )
    try:
        result = snapshot(restored._diagnostics)
        assert result["events"] == []
        assert result["uploadSamples"] == []
        assert result["overwrittenEvents"] == 0
    finally:
        restored.close()


def test_global_owned_handle_is_sampled_without_lan_accounting(
    runtime: SidecarRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = SimpleNamespace(
        info_hashes=SimpleNamespace(has_v2=lambda: True, v2="a" * 64),
        all_time_upload=0,
        total_payload_upload=512120,
        num_peers=1,
    )
    handle = SimpleNamespace(status=lambda: status)
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_session", SimpleNamespace(get_torrents=lambda: [handle]))
        patch.setattr(
            runtime, "_global", SimpleNamespace(owns=lambda candidate: candidate is handle)
        )
        assert runtime._sync_activity() is False
    sample = snapshot(runtime._diagnostics)["uploadSamples"][0]
    assert sample["infoHash"] == "a" * 64
    assert sample["nativeAllTimeUpload"] == 0
    assert sample["nativePayloadUpload"] == 512120
    assert sample["lanApplicationUploadedBytes"] is None
    assert sample["lanTransportUploadSample"] is None
    assert runtime._activity == {}
    assert runtime._live_activity == {}


@pytest.mark.parametrize("lan_uploaded", [0, 37])
def test_known_global_digest_diagnostics_explicitly_scope_lan_counters(
    runtime: SidecarRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lan_uploaded: int,
) -> None:
    lease = _seed_lease(tmp_path)

    class Handle(_AccountingHandle):
        def status(self) -> SimpleNamespace:
            status = super().status()
            status.info_hashes = SimpleNamespace(has_v2=lambda: True, v2=lease.descriptor.info_hash)
            status.total_payload_upload = self.uploaded
            status.num_peers = len(self.peers)
            return status

    handle = Handle([_peer("198.51.100.20", uploaded=123)], uploaded=123)
    controller = _accounting_controller(tmp_path, lease, handle, allows_lan_peer=lambda _: False)
    record = runtime_module._ActivityRecord(
        lease.digest,
        lease.size_bytes,
        uploaded_bytes=lan_uploaded,
        transport_upload_sample=lan_uploaded,
    )
    try:
        controller.maintain()
        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_session", SimpleNamespace(get_torrents=lambda: [handle]))
            patch.setattr(runtime, "_global", controller)
            patch.setattr(runtime, "_leases", {lease.lease_id: lease})
            patch.setattr(runtime, "_activity", {lease.digest: record})
            patch.setattr(runtime, "settings", _global_settings())
            for _ in range(3):
                assert runtime._sync_activity() is False
                sample = snapshot(runtime._diagnostics)["uploadSamples"][0]
                assert sample["digest"] == lease.digest
                assert sample["nativeAllTimeUpload"] == sample["nativePayloadUpload"] == 123
                assert sample["lanApplicationUploadedBytes"] == lan_uploaded
                assert sample["lanTransportUploadSample"] == lan_uploaded
                assert "applicationUploadedBytes" not in sample
                assert "transportUploadSample" not in sample
                assert controller.counters(lease.digest).uploaded_bytes == 123
                assert runtime._transfer_status(record)["uploadedBytes"] == lan_uploaded + 123
                assert record.uploaded_bytes == record.transport_upload_sample == lan_uploaded
    finally:
        controller.close()


def test_stalled_native_transfer_reports_peer_disconnect(
    runtime: SidecarRuntime, tmp_path: Path
) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(_safetensors())
    _, download = _leases(source, "diagnostic")
    runtime.grant(download.to_wire(), "download")
    with socket.socket() as peer:
        peer.bind(("127.0.0.1", 0))
        peer.listen(1)
        peer.setblocking(False)
        port = peer.getsockname()[1]
        accepted = False
        runtime._torrents[download.lease_id].handle.connect_peer(("127.0.0.1", port))
        deadline = time.monotonic() + 5
        while True:
            try:
                connection, _ = peer.accept()
            except BlockingIOError:
                pass
            else:
                with connection:
                    # Reset a real connection without relying on OS-specific refusal timing.
                    connection.setsockopt(
                        socket.SOL_SOCKET,
                        socket.SO_LINGER,
                        struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0),
                    )
                accepted = True
            result = cast(dict[str, Any], runtime.status())
            events = result["diagnostics"]["events"]
            failures = [
                event
                for event in events
                if event["kind"] == "peer_disconnected" and event.get("port") == port
            ]
            if failures or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert accepted
        assert failures, result
        assert failures[-1]["errorCode"] != 0
        assert failures[-1]["digest"] == download.digest
        assert result["totals"]["downloadedBytes"] == 0
        assert runtime._torrents[download.lease_id].published_path is None
        assert result["diagnostics"]["lsdPeerEventsAvailable"] is False


def test_listener_recording_does_not_touch_slow_sinks(
    runtime: SidecarRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("diagnostics must not perform filesystem or stream I/O")

    monkeypatch.setattr(runtime_module, "_atomic_write", forbidden)
    monkeypatch.setattr(sys.stderr, "write", forbidden)
    runtime._record_listener_error("https://user:secret@host/?token=secret")
    rendered = json.dumps(snapshot(runtime._diagnostics))
    assert "listener_selection_failed" in rendered
    assert "secret" not in rendered


def test_slow_ipc_writer_does_not_block_alert_sampling(
    runtime: SidecarRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        blocked = asyncio.Event()
        release = asyncio.Event()
        sampled = asyncio.Event()

        async def slow_write(*args: object, **kwargs: object) -> None:
            blocked.set()
            await release.wait()

        writer = SimpleNamespace(close=lambda: None, wait_closed=lambda: asyncio.sleep(0))

        async def connect(endpoint: str):
            return object(), writer

        async def read(reader: object):
            return (
                {
                    "version": runtime_module.IPC_VERSION,
                    "id": 1,
                    "operation": "status",
                    "body": {},
                    "blobs": [],
                },
                [],
            )

        monkeypatch.setattr(runtime_module, "connect_endpoint", connect)
        monkeypatch.setattr(runtime_module, "read_frame", read)
        monkeypatch.setattr(runtime_module, "write_frame", slow_write)
        original = runtime.sample_activity

        def sample() -> None:
            original()
            if blocked.is_set():
                sampled.set()

        monkeypatch.setattr(runtime, "sample_activity", sample)
        serving = asyncio.create_task(runtime_module.serve(runtime, "unused"))
        sampling = asyncio.create_task(runtime.sample_activity_loop())
        try:
            await asyncio.wait_for(blocked.wait(), 2)
            await asyncio.wait_for(sampled.wait(), 2)
            assert not serving.done()
        finally:
            serving.cancel()
            sampling.cancel()
            await asyncio.gather(serving, sampling, return_exceptions=True)

    asyncio.run(scenario())
