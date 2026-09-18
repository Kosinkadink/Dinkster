"""Dev-mode pack source watcher (DESIGN 3.9): sugar over hot reload.

Watches every composed pack's source directory and, when files change,
drives the SAME reload coordinator as ``POST /api/packs/{packId}/reload``
(:func:`dinkster.reload_api.apply_reload`) - one reload implementation, two
triggers. Registered only under ``dinkster-serve --dev --watch-packs``;
production installs change packs through the manager's plan/apply flow,
never a live file watcher.

Deliberately a POLLER, not a native filesystem-event subscriber: polling
is portable (Windows included - it matters for this project) with zero
new dependencies, and this is a dev affordance where the cost is one
directory scan per interval over dev-sized pack trees. Change detection
is a SETTLE cycle rather than fire-on-first-diff: a scan that finds a
diff marks the pack pending and re-baselines; the reload fires on the
first subsequent scan that shows NO further diff. Editor save bursts,
atomic-rename saves, and multi-file writes therefore coalesce into one
reload, and a file is never imported mid-write. The composer's own lock
already serializes reloads, so watcher- and endpoint-triggered reloads
cannot interleave.

Failure semantics are the endpoint's: a failed reload is logged, the OLD
worker keeps serving, and the pack does not re-fire until its files
change again (the baseline already advanced - no retry loop against a
broken source tree).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from dinkster_schema import core_logger

__all__ = ["PackWatcher", "snapshot_tree"]

_log = core_logger("watch")

# Pruned wholesale during the walk: build/VCS/venv trees that churn
# without meaning a source change (a reload itself writes __pycache__).
_IGNORED_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        "venv",
        ".venv",  # hidden dirs are pruned anyway; listed for clarity
    }
)

# Editor droppings and bytecode; a change here is never a source change.
_IGNORED_SUFFIXES = (".pyc", ".pyo", ".swp", ".swo", ".swx", ".tmp", "~")

# path -> (mtime_ns, size): cheap, portable change fingerprint.
Snapshot = dict[str, tuple[int, int]]


def _ignored_file(name: str) -> bool:
    return name.startswith(".") or name.endswith(_IGNORED_SUFFIXES)


def snapshot_tree(root: Path) -> Snapshot:
    """Fingerprint every watched file under ``root``.

    Hidden files/directories, VCS/venv/bytecode trees, and editor temp
    files are excluded; a file that vanishes mid-scan is skipped (the
    next scan sees the stable state - exactly the settle cycle's job).
    """
    snapshot: Snapshot = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS and not d.startswith(".")]
        for filename in filenames:
            if _ignored_file(filename):
                continue
            full = os.path.join(dirpath, filename)
            try:
                st = os.stat(full)
            except OSError:
                continue
            snapshot[full] = (st.st_mtime_ns, st.st_size)
    return snapshot


class PackWatcher:
    """Polls pack source roots; fires one reload per settled change burst.

    ``targets`` is called every scan (live composer records: packs
    composed after startup appear, removed packs drop out); a pack's
    FIRST sighting takes a baseline without firing - composition already
    imported that state. ``reload_pack`` is awaited once per settled
    change; its exceptions are logged, never propagated (one broken pack
    must not stop the watcher for the others - the isolation stance
    everywhere else).
    """

    def __init__(
        self,
        targets: Callable[[], Mapping[str, Path]],
        reload_pack: Callable[[str], Awaitable[object]],
        *,
        interval: float = 1.0,
    ) -> None:
        self._targets = targets
        self._reload = reload_pack
        self.interval = interval
        self._baselines: dict[str, Snapshot] = {}
        self._pending: set[str] = set()

    def scan(self) -> list[str]:
        """One scan pass: advance baselines, return the packs whose
        changes have settled (diff seen on an earlier scan, none on this
        one) in stable composition order."""
        targets = dict(self._targets())
        for name in list(self._baselines):
            if name not in targets:
                del self._baselines[name]
                self._pending.discard(name)
        due: list[str] = []
        for name, root in targets.items():
            snapshot = snapshot_tree(root)
            baseline = self._baselines.get(name)
            self._baselines[name] = snapshot
            if baseline is None:
                continue  # first sighting: baseline only, never fire
            if snapshot != baseline:
                self._pending.add(name)  # still churning: wait to settle
            elif name in self._pending:
                self._pending.discard(name)
                due.append(name)
        return due

    async def poll_once(self) -> list[str]:
        """Scan once and reload every settled pack; returns the packs
        whose reload SUCCEEDED (failures are logged, old worker serves)."""
        reloaded: list[str] = []
        for name in self.scan():
            _log.info("pack %s changed on disk, reloading", name)
            try:
                await self._reload(name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # The endpoint's own failure contract: old worker keeps
                # serving; no re-fire until the files change again.
                _log.error(
                    "pack %s reload failed (old worker keeps serving): %s",
                    name,
                    exc,
                )
            else:
                reloaded.append(name)
        return reloaded

    async def run(self) -> None:
        """Poll forever; the host cancels this task on shutdown."""
        while True:
            await asyncio.sleep(self.interval)
            await self.poll_once()
