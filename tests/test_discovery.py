"""Per-machine instance discovery (DESIGN 3.10): heartbeat files under a
runtime dir, atomic writes, stale entries read as absence. No daemon, no
shared state beyond the directory."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from dinkster_server import InstanceRegistry, default_runtime_dir


def test_announce_then_peers_sees_other_instances(tmp_path: Path) -> None:
    a = InstanceRegistry(tmp_path)
    b = InstanceRegistry(tmp_path)
    a.announce("inst-a", "http://127.0.0.1:8188")
    b.announce("inst-b", "http://127.0.0.1:8189")

    peers_of_a = a.peers()
    assert [p.instance_id for p in peers_of_a] == ["inst-b"]
    assert peers_of_a[0].endpoint == "http://127.0.0.1:8189"

    everyone = a.peers(include_self=True)
    assert {p.instance_id for p in everyone} == {"inst-a", "inst-b"}


def test_close_withdraws_the_entry(tmp_path: Path) -> None:
    a = InstanceRegistry(tmp_path)
    b = InstanceRegistry(tmp_path)
    a.announce("inst-a", "http://127.0.0.1:8188")
    b.announce("inst-b", "http://127.0.0.1:8189")
    b.close()
    assert a.peers() == []
    b.close()  # idempotent


def test_stale_heartbeats_read_as_absence(tmp_path: Path) -> None:
    a = InstanceRegistry(tmp_path, stale_after=10.0)
    b = InstanceRegistry(tmp_path, stale_after=10.0)
    b.announce("inst-b", "http://127.0.0.1:8189")
    # Age the entry by rewriting its heartbeat into the past - a crashed
    # instance that stopped writing.
    path = tmp_path / "inst-b.json"
    entry = json.loads(path.read_text())
    entry["heartbeatAt"] = time.time() - 60.0
    path.write_text(json.dumps(entry))
    assert a.peers() == []
    # A fresh announce (the heartbeat) resurrects it.
    b.announce("inst-b", "http://127.0.0.1:8189")
    assert [p.instance_id for p in a.peers()] == ["inst-b"]


def test_garbage_and_torn_files_are_ignored(tmp_path: Path) -> None:
    reader = InstanceRegistry(tmp_path)
    (tmp_path / "torn.json").write_text('{"instanceId": "x", "endpo')
    (tmp_path / "wrong-shape.json").write_text('["not", "an", "object"]')
    (tmp_path / "missing-fields.json").write_text('{"instanceId": "x"}')
    (tmp_path / "bad-types.json").write_text(
        json.dumps({"instanceId": "x", "endpoint": "e", "pid": True, "heartbeatAt": "now"})
    )
    assert reader.peers() == []


def test_one_registry_serves_one_instance(tmp_path: Path) -> None:
    a = InstanceRegistry(tmp_path)
    a.announce("inst-a", "http://127.0.0.1:8188")
    a.announce("inst-a", "http://127.0.0.1:9999")  # refresh is fine
    with pytest.raises(ValueError, match="one registry serves one instance"):
        a.announce("inst-other", "http://127.0.0.1:1")


def test_missing_directory_reads_as_no_peers(tmp_path: Path) -> None:
    reader = InstanceRegistry(tmp_path / "never-created")
    assert reader.peers() == []


def test_default_runtime_dir_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_RUNTIME_DIR", "/tmp/custom-dinkster")
    assert default_runtime_dir() == Path("/tmp/custom-dinkster")
    monkeypatch.delenv("DINKSTER_RUNTIME_DIR")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    assert default_runtime_dir() == Path("/run/user/1000/dinkster")
