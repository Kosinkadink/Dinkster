"""Read standalone software inputs without opening an installer or user data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from blake3 import blake3

from .environment_artifacts import parse_metadata
from .environment_state import EnvironmentError, content_digest
from .environment_stores import registry_path
from .limits import ENVIRONMENT_STREAM_CHUNK_BYTES


def standalone_inputs(source: Path, target: Path, store: Path) -> dict[str, object]:
    """Describe only recorded sources and placement, never the ambient Python."""
    if not source.is_absolute() or source.is_symlink():
        raise EnvironmentError("--adopt-from must be an absolute, unlinked standalone root")
    source = source.resolve(strict=True)
    for path in (target, store, registry_path().resolve()):
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise EnvironmentError("adoption destination and store must be disjoint from old root")

    def software(path: Path) -> bytes:
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise EnvironmentError(f"adoption requires an intact software file: {path}")
        return path.read_bytes()

    current = software(source / "current").decode("utf-8").strip()
    if not current.isdecimal() or int(current) < 1:
        raise EnvironmentError("standalone root has an invalid current generation")
    lock_path = source / "generations" / f"{int(current)}.json"
    lock_text = software(lock_path).decode("utf-8")
    lock = parse_metadata(lock_text.encode("utf-8"))
    if lock.get("format") != "dinkster.lock/1" or not isinstance(lock.get("packs"), list):
        raise EnvironmentError("adoption requires a standalone pack source lock")
    packs: list[dict[str, object]] = []
    names: set[str] = set()
    for item in cast("list[object]", lock["packs"]):
        if not isinstance(item, dict):
            raise EnvironmentError("standalone source lock requires pack objects")
        row = cast("dict[str, object]", item)
        name, digest = row.get("pack"), row.get("artifactDigest")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
            or name in names
            or not isinstance(digest, str)
            or re.fullmatch(r"blake3:[0-9a-f]{64}", digest) is None
        ):
            raise EnvironmentError(
                "standalone source lock requires unique canonical pack identities"
            )
        names.add(name)
        packs.append(row)
    hosting_path = lock_path.with_suffix(".hosting.json")
    hosting_text = (
        software(hosting_path).decode("utf-8")
        if hosting_path.exists() or hosting_path.is_symlink()
        else None
    )
    hosting = parse_metadata(hosting_text.encode("utf-8")) if hosting_text is not None else {}
    if hosting.get("format") == "dinkster.hosting/1":
        groups, in_process = hosting.get("venvGroups", {}), hosting.get("inProcess", [])
    elif "format" not in hosting:
        groups, in_process = hosting, []
    else:
        raise EnvironmentError("unsupported standalone hosting format")
    if not isinstance(groups, dict) or not isinstance(in_process, list):
        raise EnvironmentError("invalid standalone hosting topology")
    in_process = cast("list[object]", in_process)
    if any(not isinstance(name, str) or name not in names for name in in_process) or len(
        set(cast("list[str]", in_process))
    ) != len(in_process):
        raise EnvironmentError("invalid standalone in-process membership")
    pointer = source / "shared-store"
    pointer_text = (
        software(pointer).decode("utf-8") if pointer.exists() or pointer.is_symlink() else None
    )
    content = Path(pointer_text.strip()) if pointer_text is not None else source
    if not content.is_absolute() or content.is_symlink():
        raise EnvironmentError("standalone shared store must be absolute and unlinked")
    content = content.resolve(strict=True)
    for path in (target, store, registry_path().resolve()):
        if path.is_relative_to(content) or content.is_relative_to(path):
            raise EnvironmentError("adoption must not write into the standalone content store")
    archives = content / "artifacts"
    if archives.is_symlink():
        raise EnvironmentError("standalone archives must not be linked")
    payloads: dict[str, str] = {}
    for row in packs:
        identity = cast("str", row["artifactDigest"])
        path = archives / f"{identity[7:]}.zip"
        if path.is_symlink() or not path.is_file():
            raise EnvironmentError(
                f"standalone pack archive is missing: {row['pack']} ({identity})"
            )
        actual = blake3()
        with path.open("rb") as stream:
            while chunk := stream.read(ENVIRONMENT_STREAM_CHUNK_BYTES):
                actual.update(chunk)
        if "blake3:" + actual.hexdigest() != identity:
            raise EnvironmentError(f"standalone pack archive digest differs: {row['pack']}")
        payloads[path.name] = content_digest(path)
    return {
        "source": str(source),
        "generation": int(current),
        "sourceLock": lock_text,
        "packs": packs,
        "hosting": hosting_text,
        "inProcessPacks": in_process,
        "venvGroups": groups,
        "storePointer": pointer_text,
        "artifacts": str(archives),
        "archiveDigests": payloads,
        "previousEngine": "not recorded by standalone installer",
        "previousBase": "not recorded by standalone installer",
    }
