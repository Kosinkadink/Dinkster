"""Read-only directory discovery backed by the P2P storage verification authority."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

from .identity import AssetError
from .p2p_global import ProviderP2PSnapshotV1
from .p2p_storage import P2PLocalFileMapping, cached_p2p_local_file, verified_p2p_seed_descriptor
from .vault import AssetVault


def _unlinked(path: Path) -> bool:
    return all(not part.is_symlink() and not part.is_junction() for part in (path, *path.parents))


class ExistingSeedStore:
    """Map current provider entries without creating any model copy or link."""

    def __init__(self, vault: AssetVault, roots: Sequence[Path]) -> None:
        self.vault = vault
        self.roots = tuple(Path(os.path.abspath(root)) for root in roots)
        if not self.roots or any(not root.is_dir() or not _unlinked(root) for root in self.roots):
            raise ValueError("seed stores must be existing non-symlink directories")
        self._mappings: dict[str, P2PLocalFileMapping] = {}

    def local_path_for(self, digest: str) -> Path | None:
        mapping = self._mappings.get(digest)
        if mapping is None or not _unlinked(mapping.path) or not mapping.is_current():
            return None
        return mapping.path

    def refresh(
        self, snapshots: Sequence[ProviderP2PSnapshotV1], *, now: float | None = None
    ) -> tuple[str, ...]:
        now = time.time() if now is None else now
        entries = {
            (row.digest, row.size_bytes): row
            for snapshot in snapshots
            for row in snapshot.p2p_artifacts
            if row.expires_at > now
            and row.digest not in {item.digest for item in snapshot.tombstones}
        }
        sizes = {size for _, size in entries}
        mapped: dict[str, P2PLocalFileMapping] = {}
        for root in self.roots:
            for directory, dirs, files in os.walk(root, followlinks=False):
                parent = Path(directory)
                if not _unlinked(parent):
                    dirs[:] = []
                    continue
                dirs[:] = sorted(name for name in dirs if _unlinked(parent / name))
                for name in sorted(files):
                    path = parent / name
                    try:
                        if (
                            not _unlinked(path)
                            or not path.is_file()
                            or path.stat().st_size not in sizes
                        ):
                            continue
                        result = verified_p2p_seed_descriptor(
                            self.vault.root, None, path.stat().st_size, path
                        )
                        digest = result.asset_digest
                        row = entries.get((digest, result.size))
                        if row is None or digest in mapped or result.descriptor != row.descriptor:
                            continue
                        mapping = cached_p2p_local_file(self.vault.root, path)
                        if mapping is not None and _unlinked(path):
                            mapped[digest] = mapping
                    except (AssetError, OSError):
                        continue
        self._mappings = mapped
        return tuple(sorted(mapped))
