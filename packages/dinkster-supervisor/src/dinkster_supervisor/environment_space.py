"""Read-only physical allocation report for managed software, excluding user libraries."""

from __future__ import annotations

import os
import stat
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinkster_store import object_entries

from .environment_collection import collection_plan, collection_preview, store_collection_plan
from .environment_state import EnvironmentError


@dataclass(frozen=True)
class _Owner:
    group: str
    automatic: bool
    explicit: bool


@dataclass
class _File:
    owners: list[_Owner]
    extents: list[tuple[int, int, bool]]
    links: int


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        getattr(info, "st_blocks", -1),
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _linux_extents(fd: int) -> list[tuple[int, int, bool]]:
    if sys.platform != "linux":
        raise EnvironmentError("Linux physical extent query requires Linux")
    import fcntl

    result: list[tuple[int, int, bool]] = []
    start = 0
    while True:
        buffer = bytearray(32 + 56 * 128)
        # No SYNC or CACHE flag: observing allocation must not flush or cache file data.
        struct.pack_into("=QQIIII", buffer, 0, start, (1 << 64) - 1 - start, 0, 0, 128, 0)
        fcntl.ioctl(fd, 0xC020660B, buffer, True)
        count = struct.unpack_from("=I", buffer, 20)[0]
        if count > 128:
            raise EnvironmentError("invalid physical extent count")
        if count == 0:
            return result
        end = start
        flags = 0
        for index in range(count):
            logical, physical, length, _, _, flags = struct.unpack_from(
                "=QQQQQI", buffer, 32 + index * 56
            )
            # LAST, UNWRITTEN, MERGED and SHARED preserve physical byte interpretation.
            if flags & ~(0x1 | 0x800 | 0x1000 | 0x2000) or not length or not physical:
                raise EnvironmentError("physical allocation is delayed, encoded or unavailable")
            result.append((physical, physical + length, bool(flags & 0x2000)))
            end = logical + length
        if end <= start:
            raise EnvironmentError("physical extent query made no progress")
        if flags & 1:
            return result
        start = end


def _allocation(
    trees: list[tuple[Path, _Owner]],
    explicit_paths: set[Path],
    group_paths: dict[Path, str],
    explicit_known: bool,
    automatic_enabled: bool,
) -> dict[str, object]:
    files: dict[tuple[int, int], _File] = {}
    observed: dict[Path, tuple[int, ...]] = {}
    warnings: list[str] = []
    exclusive: dict[str, int] = defaultdict(int)
    shared: dict[str, int] = defaultdict(int)
    events: dict[int, list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)
    total = shared_total = automatic = explicit = 0
    reclaim_known = True

    def visit(path: Path, owner: _Owner, parent_fd: int | None = None) -> None:
        nonlocal total, automatic, explicit
        if path in group_paths:
            owner = _Owner(group_paths[path], owner.automatic, owner.explicit)
        if path in explicit_paths:
            owner = _Owner(owner.group, owner.automatic, True)
        name = path if parent_fd is None else path.name
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        signature = _signature(info)
        observed[path] = signature
        if sys.platform != "linux":
            raise EnvironmentError("physical extent measurement is unavailable on this platform")
        if stat.S_ISLNK(info.st_mode):
            if info.st_blocks:
                raise EnvironmentError(f"physical allocation of a long symbolic link: {path}")
            return
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise EnvironmentError(f"unsupported managed software file: {path}")
        fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NOATIME | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        try:
            if _signature(os.fstat(fd)) != signature:
                raise EnvironmentError(f"managed software changed during observation: {path}")
            if stat.S_ISDIR(info.st_mode):
                size = info.st_blocks * 512
                exclusive[owner.group] += size
                total += size
                if owner.automatic:
                    automatic += size
                if owner.explicit or owner.automatic:
                    explicit += size
                for child in os.listdir(fd):
                    visit(path / child, owner, fd)
                return
            identity = (info.st_dev, info.st_ino)
            if identity in files:
                files[identity].owners.append(owner)
                return
            item = _File([owner], _linux_extents(fd), info.st_nlink)
            if sum(end - begin for begin, end, _ in item.extents) != info.st_blocks * 512:
                raise EnvironmentError(f"unmapped allocation or concurrent change: {path}")
            files[identity] = item
            for begin, end, _ in item.extents:
                events[info.st_dev].extend(((begin, 1, identity), (end, -1, identity)))
        finally:
            os.close(fd)

    for path, owner in trees:
        try:
            visit(path, owner)
        except (OSError, ValueError) as exc:
            warnings.append(str(exc))
    for device_events in events.values():
        active: dict[tuple[int, int], int] = defaultdict(int)
        previous = 0
        for position, delta, identity in sorted(device_events):
            length = position - previous
            if active and length:
                items = [files[key] for key in active]
                owners = [owner for item in items for owner in item.owners]
                groups = {owner.group for owner in owners}
                clone_shared = any(
                    is_shared and begin < position and end > previous
                    for item in items
                    for begin, end, is_shared in item.extents
                )
                outside_link = any(item.links > len(item.owners) for item in items)
                is_shared = len(groups) > 1 or outside_link or clone_shared
                total += length
                if is_shared:
                    shared_total += length
                    for group in groups:
                        shared[group] += length
                else:
                    exclusive[next(iter(groups))] += length
                auto = all(owner.automatic for owner in owners)
                gc = all(owner.explicit or owner.automatic for owner in owners)
                if not outside_link and (auto or gc) and clone_shared:
                    # FIEMAP says shared, but not how many references exist outside this scan.
                    reclaim_known = False
                elif not outside_link and not clone_shared:
                    if auto:
                        automatic += length
                    if gc:
                        explicit += length
            active[identity] += delta
            if not active[identity]:
                del active[identity]
            previous = position
    for path, signature in observed.items():
        try:
            if _signature(path.lstat()) != signature:
                warnings.append(f"managed software changed during observation: {path}")
        except OSError as exc:
            warnings.append(str(exc))
    complete = not warnings
    automatic_now = automatic if automatic_enabled else 0
    return {
        "allocatedBytes": total if complete else None,
        "observedAllocatedBytes": total,
        "sharedAllocatedBytes": shared_total if complete else None,
        "groups": {
            group: {
                "exclusiveAllocatedBytes": exclusive[group] if complete else None,
                "sharedAllocatedBytes": shared[group] if complete else None,
            }
            for group in sorted({owner.group for _, owner in trees} | set(exclusive) | set(shared))
        },
        "eligibleGenerationReclaimableBytes": automatic if complete and reclaim_known else None,
        "automaticReclaimableBytes": automatic_now
        if not automatic_enabled or (complete and reclaim_known)
        else None,
        "additionalExplicitReclaimableBytes": explicit - automatic_now
        if complete and reclaim_known and explicit_known
        else None,
        "complete": complete,
        "warnings": warnings,
        "reclaimabilityWarnings": []
        if reclaim_known
        else ["Shared clone reference counts outside this report are unavailable."],
    }


def space_report(root: Path) -> dict[str, object]:
    """Observe the same safety policy as cleanup; never recover, collect or change settings."""
    if any(p.is_symlink() or p.is_junction() for p in (root, *root.parents)):
        raise EnvironmentError("space report requires an unlinked project path")
    preview = collection_preview(root)
    policy = cast("dict[str, object]", preview["policy"])
    generations = [dict(row) for row in cast("list[dict[str, object]]", preview["generations"])]
    generation_directory = _signature((root / "generations").stat())
    trees: list[tuple[Path, _Owner]] = []
    explicit_generations: tuple[int, ...] = ()
    explicit_paths: tuple[str, ...] = ()
    explicit_known = True
    warnings: list[str] = []
    group_paths: dict[Path, str] = {}
    pointer = root / "store"
    store: Path | None = None
    if pointer.exists():
        if pointer.is_symlink() or pointer.is_junction():
            raise EnvironmentError("space report refuses a linked store pointer")
        store = Path(pointer.read_text(encoding="utf-8").strip())
        if (
            not store.is_absolute()
            or store.is_symlink()
            or store.is_junction()
            or store.is_relative_to(root)
            or root.is_relative_to(store)
            or any(p.is_symlink() or p.is_junction() for p in store.parents)
            or ".." in store.parts
        ):
            raise EnvironmentError("space report requires separate unlinked project and store")
    try:
        _, _, explicit_generations = collection_plan(root)
        if store is not None:
            explicit_paths = store_collection_plan(root, store, explicit_generations)
    except (OSError, ValueError) as exc:
        explicit_known = False
        warnings.append(f"Explicit cleanup eligibility is unavailable: {exc}")
    for row in generations:
        number = cast("int", row["generation"])
        group = f"generation/{number}"
        row["allocationGroup"] = group
        trees.append(
            (
                root / "generations" / str(number),
                _Owner(
                    group,
                    row["status"] == "candidate",
                    number in explicit_generations,
                ),
            )
        )
    if store is not None:
        for _, engine in object_entries(store / "engines"):
            group_paths[engine / "artifacts"] = "store/pack-archives"
        for category in ("bases", "wheels", "downloads", "engines", "uv-cache", ".garbage"):
            path = store / category
            group = f"store/{category}"
            if not path.exists():
                continue
            if path.is_symlink() or path.is_junction():
                raise EnvironmentError("space report refuses linked store directories")
            trees.append((path, _Owner(group, False, False)))
    if (root / ".garbage").exists():
        trees.append((root / ".garbage", _Owner("generation-quarantine", False, False)))
    allocation = _allocation(
        trees,
        {store / name for name in explicit_paths} if store is not None else set(),
        group_paths,
        explicit_known,
        bool(policy["automatic"]),
    )
    if (
        collection_preview(root) != preview
        or _signature((root / "generations").stat()) != generation_directory
        or (store is not None and pointer.read_text(encoding="utf-8").strip() != str(store))
    ):
        allocation.update(
            complete=False,
            allocatedBytes=None,
            automaticReclaimableBytes=None if policy["automatic"] else 0,
            eligibleGenerationReclaimableBytes=None,
            additionalExplicitReclaimableBytes=None,
        )
        cast("list[str]", allocation["warnings"]).append(
            "Cleanup state or generation membership changed during observation."
        )
    return {
        "format": "dinkster.space/1",
        "observedAt": time.time_ns(),
        "project": str(root),
        "store": str(store) if store is not None else None,
        "scope": "Managed software only. Models, outputs, workflows, mounts and other user data "
        "are excluded. Shared-store files may serve other projects.",
        "sharing": "Exclusive rows plus sharedAllocatedBytes give the total. Per-row shared "
        "bytes overlap and must not be added together.",
        "policy": policy,
        "settingsCommand": f"dinkster-env --root {root} cleanup --automatic on|off "
        "--keep-previous 1..100 --yes",
        "generations": generations,
        "cleanupWarnings": preview["warnings"],
        **allocation,
        "eligibilityWarnings": warnings,
    }
