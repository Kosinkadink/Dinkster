"""Library compatibility disclosure for software-only environment plans."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from dinkster_store.library_compatibility import (
    engine_readers,
    library_requirements,
    minimum_release,
)
from packaging.version import InvalidVersion, Version

from .environment_history import history
from .environment_state import interpreter


def library_plan(
    root: Path, target_version: str | None = None, *, source: Path | None = None
) -> list[str]:
    library = os.environ.get("DINKSTER_LIBRARY_ROOT", str(root / "dinkster-library"))
    if not library:
        return ["library persistence is disabled"]
    revisions, hidden = library_requirements(Path(library))
    if source is not None:
        prospective, _ = engine_readers(source)
        revisions = tuple(sorted(set(revisions) | set(prospective)))
    release = minimum_release(revisions, hidden)
    if release is None:
        return [
            "library compatibility: earliest release is unknown for schema "
            + ", ".join(sorted(revisions))
            + "; retained snapshot compatibility cannot be established"
        ]
    minimum = Version(release)
    unknown_target = False
    if target_version is not None and source is None:
        try:
            target = Version(target_version)
        except InvalidVersion:
            target = None
        unknown_target = target is None or not Version("0.0.1") <= target <= Version("0.0.15")
        if not unknown_target and target is not None:
            # Released engines from 0.0.4 through 0.0.15 migrate to revision 0011.
            if target >= Version("0.0.4"):
                minimum = max(minimum, Version("0.0.4"))
                revisions = (*revisions, "0011_retention_clients")
            elif target >= Version("0.0.3"):
                minimum = max(minimum, Version("0.0.3"))
                revisions = (*revisions, "0010_result_renditions")
            elif target >= Version("0.0.2"):
                minimum = max(minimum, Version("0.0.2"))
                revisions = (*revisions, "0002_asset_observations")
    plan = [f"library after apply requires Dinkster {minimum} or newer"]
    if unknown_target:
        plan = [
            f"current library requires Dinkster {minimum} or newer; "
            f"earliest release after applying engine {target_version} is unknown"
        ]
    excluded: list[str] = []
    uncertain: list[str] = []
    for entry in history(root):
        snapshot = cast("dict[str, object]", entry["snapshot"])
        engine = cast("dict[str, object]", snapshot["engine"])
        version = str(engine["version"])
        python = interpreter(root / "generations" / str(entry.get("generation", "")))
        if python.is_file():
            supported, supports_hidden = engine_readers(python=python)
            older = bool(set(revisions) - set(supported)) or (hidden and not supports_hidden)
        else:
            metadata = cast("dict[str, object]", snapshot.get("metadata", {}))
            try:
                parsed = Version(version)
            except InvalidVersion:
                parsed = None
            if (
                parsed is None
                or parsed.is_prerelease
                or parsed.local is not None
                or "development-sources.json" in metadata
                or snapshot.get("source") == "source"
            ):
                uncertain.append(f"history entry {entry['id']} (Dinkster {version})")
                continue
            older = parsed < minimum
        if older:
            excluded.append(f"history entry {entry['id']} (Dinkster {version})")
    plan.append(
        "library excludes retained snapshots: " + "; ".join(excluded)
        if excluded
        else "library excludes no known retained release snapshots"
    )
    if uncertain:
        plan.append(
            "library reader must be checked for retained snapshots: " + "; ".join(uncertain)
        )
    return plan
