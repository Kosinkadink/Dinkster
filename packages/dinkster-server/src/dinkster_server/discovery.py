"""Per-machine instance discovery (DESIGN 3.10).

Multiple Dinkster instances sharing a machine find each other through
heartbeat files under a well-known runtime directory - one JSON file per
instance naming its endpoint. There is no elected daemon and no shared
mutable state beyond this directory: an instance announces itself by
rewriting its own file on an interval, and readers treat a stale
heartbeat as absence. A crashed instance simply stops writing; its entry
ages out and its leases (over on the /memory endpoints) expire on their
own TTLs. Coordination degrades, correctness does not.

Writes are atomic (temp file + os.replace) so readers never observe a
torn entry; unreadable or malformed files are someone else's crash mid-
write or garbage in the directory, both ignored.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast


def default_runtime_dir() -> Path:
    """The well-known per-machine directory instances rendezvous in.

    DINKSTER_RUNTIME_DIR overrides; else XDG_RUNTIME_DIR/dinkster (per-user,
    tmpfs, cleared on logout - the right lifetime for liveness data);
    else the system temp dir. Never a config/data dir: heartbeats are
    runtime facts, not state worth persisting.
    """
    explicit = os.environ.get("DINKSTER_RUNTIME_DIR")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "dinkster"
    return Path(tempfile.gettempdir()) / "dinkster-instances"


@dataclass(frozen=True)
class PeerInfo:
    """One live instance as its heartbeat file describes it."""

    instance_id: str
    endpoint: str
    pid: int
    heartbeat_at: float  # unix time of the last write


class InstanceRegistry:
    """Announce this instance and read the others.

    Synchronous on purpose: one small file write per heartbeat and one
    directory scan per query. The caller owns the cadence - a server wires
    ``announce`` into a periodic task; ``close`` withdraws the entry.
    """

    def __init__(
        self,
        runtime_dir: Path | None = None,
        *,
        stale_after: float = 15.0,
    ) -> None:
        if stale_after <= 0:
            raise ValueError("stale_after must be > 0")
        self._dir = runtime_dir if runtime_dir is not None else default_runtime_dir()
        self._stale_after = stale_after
        self._own_path: Path | None = None
        self._instance_id: str | None = None

    @property
    def runtime_dir(self) -> Path:
        return self._dir

    def announce(self, instance_id: str, endpoint: str) -> None:
        """Write (or refresh) this instance's heartbeat entry atomically."""
        if not instance_id:
            raise ValueError("instance_id must be non-empty")
        if self._instance_id is not None and instance_id != self._instance_id:
            raise ValueError(
                f"registry already announced {self._instance_id!r}; "
                "one registry serves one instance"
            )
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{instance_id}.json"
        entry = {
            "instanceId": instance_id,
            "endpoint": endpoint,
            "pid": os.getpid(),
            "heartbeatAt": time.time(),
        }
        fd, tmp_name = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f)
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        self._own_path = path
        self._instance_id = instance_id

    def peers(self, *, include_self: bool = False) -> list[PeerInfo]:
        """Every live instance in the directory, stale entries excluded."""
        if not self._dir.is_dir():
            return []
        now = time.time()
        found: list[PeerInfo] = []
        for path in sorted(self._dir.glob("*.json")):
            entry = _read_entry(path)
            if entry is None:
                continue
            if now - entry.heartbeat_at > self._stale_after:
                continue
            if not include_self and entry.instance_id == self._instance_id:
                continue
            found.append(entry)
        return found

    def close(self) -> None:
        """Withdraw this instance's entry; idempotent."""
        if self._own_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(self._own_path)
            self._own_path = None


def _read_entry(path: Path) -> PeerInfo | None:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # torn write, permissions, or garbage: absence
    if not isinstance(raw, dict):
        return None
    entry = cast("dict[str, object]", raw)
    instance_id = entry.get("instanceId")
    endpoint = entry.get("endpoint")
    pid = entry.get("pid")
    heartbeat_at = entry.get("heartbeatAt")
    if not isinstance(instance_id, str) or not instance_id:
        return None
    if not isinstance(endpoint, str) or not endpoint:
        return None
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    if isinstance(heartbeat_at, bool) or not isinstance(heartbeat_at, (int, float)):
        return None
    return PeerInfo(
        instance_id=instance_id,
        endpoint=endpoint,
        pid=pid,
        heartbeat_at=float(heartbeat_at),
    )
