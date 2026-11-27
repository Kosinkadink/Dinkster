"""Shared rules for the existing worker index format; no filesystem or database access."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import cast

INDEX_NAME = ".dinkster-asset-index.json"
WRITES_SIDECAR_NAME = ".dinkster-asset-writes.jsonl"
"""Writers append landed files here for resolution before the next full scan."""
PARTIAL_DOWNLOAD_EXTENSIONS = frozenset(
    {".part", ".partial", ".crdownload", ".download", ".tmp", ".aria2", ".!qb", ".opdownload"}
)
DEFAULT_IGNORE: tuple[str, ...] = (
    "put_*_here",
    "Thumbs.db",
    "desktop.ini",
    "*:Zone.Identifier",
    "*.tmp",
    "*.part",
    "*.crdownload",
    "*.aria2",
)


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


class IgnoreRules:
    """Patterns with '/' match relative paths; other patterns match file names."""

    def __init__(self, patterns: Iterable[str]) -> None:
        self._name_rules: list[re.Pattern[str]] = []
        self._path_rules: list[re.Pattern[str]] = []
        for pattern in patterns:
            if "/" in pattern:
                self._path_rules.append(glob_regex(pattern))
            else:
                self._name_rules.append(glob_regex(pattern))

    def matches(self, relative_path: str, name: str) -> bool:
        return any(rule.match(name) for rule in self._name_rules) or any(
            rule.match(relative_path) for rule in self._path_rules
        )


def retain_missing_versions(current: dict[str, object], previous: object) -> dict[str, object]:
    if not isinstance(previous, Mapping):
        return current
    old = cast("Mapping[str, object]", previous)
    versions = old.get("missingVersions", [])
    retained = (
        [
            dict(cast("Mapping[str, object]", row))
            for row in cast("list[object]", versions)
            if isinstance(row, Mapping)
            and cast("Mapping[str, object]", row).get("digest") != current["digest"]
        ]
        if isinstance(versions, list)
        else []
    )
    old_digest = old.get("digest")
    if (
        old_digest == current["digest"]
        and type(old.get("publishedAtNs")) is int
        and isinstance(old.get("verification"), Mapping)
        and old.get("verification") == current.get("verification")
        and old.get("size") == current.get("size")
        and old.get("mtimeNs") == current.get("mtimeNs")
    ):
        current["publishedAtNs"] = old["publishedAtNs"]
    if (
        isinstance(old_digest, str)
        and old_digest != current["digest"]
        and re.match(r"^blake3:[0-9a-f]{64}$", old_digest) is not None
    ):
        retained.append({key: value for key, value in old.items() if key != "missingVersions"})
    if retained:
        current["missingVersions"] = retained
    return current
