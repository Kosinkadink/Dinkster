"""The asset catalog: file-like semantics without filesystem access.

The catalog separates what legacy Comfy conflates (DESIGN 3.12): *content*
(one digest per unique blob, however many names it has) and *references*
(virtual paths in a namespace pointing at content). Virtual folders are
namespace prefixes: listing, globbing, and prefix queries behave like a
filesystem - the part ComfyUI's asset work is missing - while nodes and
clients never touch real paths.

Virtual paths are ``/``-separated, relative, and validated: no empty
segments, no ``.``/``..``, no backslashes. A catalog is a queryable index;
where bytes live is a store concern (see AssetResolver).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .identity import AssetError, require_digest
from .model import AssetRef, AssetResolver


def glob_regex(pattern: str) -> re.Pattern[str]:
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if pattern[index : index + 2] == "**":
                parts.append(".*")
                index += 2
            else:
                parts.append("[^/]*")
                index += 1
        elif char == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(char))
            index += 1
    return re.compile("".join(parts) + r"\Z")


def validate_virtual_path(path: str) -> str:
    if "\\" in path:
        raise AssetError(f"virtual paths use '/' separators only: {path!r}")
    segments = path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise AssetError(f"invalid virtual path (empty/./.. segment): {path!r}")
    return path


@dataclass(frozen=True)
class AssetEntry:
    """One reference: a virtual path bound to a content identity. Many
    entries may share one digest (duplicate files, aliases)."""

    virtual_path: str
    digest: str
    size: int
    media_type: str = "application/octet-stream"
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        validate_virtual_path(self.virtual_path)
        require_digest(self.digest)

    @property
    def name(self) -> str:
        return self.virtual_path.rsplit("/", 1)[-1]

    def ref(self, resolver: AssetResolver | None = None) -> AssetRef:
        return AssetRef(
            digest=self.digest,
            name=self.name,
            size=self.size,
            media_type=self.media_type,
            virtual_path=self.virtual_path,
            resolver=resolver,
        )


@dataclass(frozen=True)
class FolderListing:
    """One level of a virtual folder: child folder names and the entries
    directly inside - the shape a file picker renders."""

    folders: tuple[str, ...]
    entries: tuple[AssetEntry, ...]


class AssetCatalog:
    """In-memory reference index. Entries are keyed by virtual path (unique);
    digests are secondary keys (one-to-many)."""

    def __init__(self, entries: Iterable[AssetEntry] = ()) -> None:
        self._by_path: dict[str, AssetEntry] = {}
        self._by_digest: dict[str, list[AssetEntry]] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: AssetEntry) -> None:
        existing = self._by_path.get(entry.virtual_path)
        if existing is not None:
            self._by_digest[existing.digest].remove(existing)
        self._by_path[entry.virtual_path] = entry
        self._by_digest.setdefault(entry.digest, []).append(entry)

    def remove(self, virtual_path: str) -> None:
        entry = self._by_path.pop(virtual_path, None)
        if entry is not None:
            self._by_digest[entry.digest].remove(entry)

    def get(self, virtual_path: str) -> AssetEntry | None:
        return self._by_path.get(virtual_path)

    def by_digest(self, digest: str) -> tuple[AssetEntry, ...]:
        """All virtual paths carrying this content - renames and duplicate
        copies are the same asset."""
        return tuple(self._by_digest.get(require_digest(digest), ()))

    def entries(self) -> tuple[AssetEntry, ...]:
        return tuple(self._by_path[path] for path in sorted(self._by_path))

    def list_folder(self, prefix: str = "") -> FolderListing:
        """Immediate children of a virtual folder, files and subfolders,
        like a directory listing."""
        if prefix:
            validate_virtual_path(prefix)
            prefix += "/"
        folders: set[str] = set()
        entries: list[AssetEntry] = []
        for path in sorted(self._by_path):
            if not path.startswith(prefix):
                continue
            remainder = path[len(prefix) :]
            head, separator, _ = remainder.partition("/")
            if separator:
                folders.add(head)
            else:
                entries.append(self._by_path[path])
        return FolderListing(folders=tuple(sorted(folders)), entries=tuple(entries))

    def glob(self, pattern: str) -> tuple[AssetEntry, ...]:
        """Match virtual paths against a glob pattern with filesystem
        semantics: ``*`` and ``?`` stay within one folder segment,
        ``**`` spans folders (``models/loras/*.safetensors``,
        ``models/**`` for a whole subtree)."""
        regex = glob_regex(pattern)
        return tuple(self._by_path[path] for path in sorted(self._by_path) if regex.match(path))

    def __len__(self) -> int:
        return len(self._by_path)
