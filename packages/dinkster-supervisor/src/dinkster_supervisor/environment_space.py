"""Read-only physical allocation report for managed software, excluding user libraries."""

from __future__ import annotations

import os
import shlex
import stat
import struct
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    allocated: int = 0


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


def _windows_allocation(path: Path) -> tuple[tuple[int, int], _File, list[str]]:
    import ctypes
    from ctypes import wintypes

    win_dll = getattr(ctypes, "WinDLL")  # noqa: B009 - Linux stubs omit it
    kernel: Any = win_dll("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel.GetVolumeInformationByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    kernel.DeviceIoControl.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetDiskFreeSpaceW.argtypes = [wintypes.LPCWSTR] + [ctypes.POINTER(wintypes.DWORD)] * 4
    get_error = getattr(ctypes, "get_last_error")  # noqa: B009 - Linux stubs omit it
    format_error = getattr(ctypes, "FormatError")  # noqa: B009 - Linux stubs omit it

    def fail() -> None:
        error = get_error()
        raise OSError(error, format_error(error), str(path))

    handles: list[Any] = []
    handle: Any = None

    def information(kind: int, size: int) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        if not kernel.GetFileInformationByHandleEx(handle, kind, buffer, size):
            if kind == 7 and get_error() == 38:  # No streams on this file or directory.
                return buffer.raw
            fail()
        return buffer.raw

    try:
        # Pin ancestors from the volume down so a rename cannot redirect a later open.
        for part in (*reversed(path.parents), path):
            handle = kernel.CreateFileW(
                str(part), 0x81 if part == path else 0x80, 3, None, 3, 0x02000000 | 0x00200000, None
            )
            if handle == ctypes.c_void_p(-1).value:
                fail()
            handles.append(handle)
            attributes = struct.unpack_from("=I", information(9, 8))[0]
            if attributes & 0x400:
                raise EnvironmentError(
                    f"physical allocation of a reparse point is unavailable: {part}"
                )
            if part != path and not attributes & 0x10:
                raise EnvironmentError(f"managed software ancestor is not a directory: {part}")
        before = path.lstat()
        directory = stat.S_ISDIR(before.st_mode)
        basic = information(0, 40)
        standard = information(1, 24)
        allocated, _, links = struct.unpack_from("=qqI", standard)
        identity_data = information(18, 24)
        identity = (
            struct.unpack_from("=Q", identity_data)[0],
            int.from_bytes(identity_data[8:24], "little"),
        )
        if identity[1] != before.st_ino:
            raise EnvironmentError(f"managed software changed during observation: {path}")
        if allocated < 0 or not links or bool(standard[21]) != directory:
            raise EnvironmentError(f"inconsistent Windows allocation metadata: {path}")
        # Named streams are not represented by the unnamed stream's allocation or extents.
        streams = information(7, 65536)
        stream_offset = 0
        while True:
            following, name_size = struct.unpack_from("=II", streams, stream_offset)
            if name_size % 2 or stream_offset + 24 + name_size > len(streams):
                raise EnvironmentError("invalid Windows stream entry")
            name = streams[stream_offset + 24 : stream_offset + 24 + name_size].decode("utf-16-le")
            if name_size and name != "::$DATA":
                raise EnvironmentError(f"named stream allocation is unavailable: {path}")
            if not following:
                break
            if following < 24 + name_size or stream_offset + following + 24 > len(streams):
                raise EnvironmentError("invalid Windows stream entry offset")
            stream_offset += following
        children: list[str] = []
        extents: list[tuple[int, int, bool]] = []
        if directory:
            restart = True
            while True:
                buffer = ctypes.create_string_buffer(65536)
                if not kernel.GetFileInformationByHandleEx(
                    handle, 15 if restart else 14, buffer, len(buffer)
                ):
                    if get_error() == 18:  # ERROR_NO_MORE_FILES
                        break
                    fail()
                restart = False
                offset = 0
                while True:
                    following = struct.unpack_from("=I", buffer.raw, offset)[0]
                    name_size = struct.unpack_from("=I", buffer.raw, offset + 60)[0]
                    if name_size % 2 or offset + 68 + name_size > len(buffer):
                        raise EnvironmentError("invalid Windows directory entry")
                    name = buffer.raw[offset + 68 : offset + 68 + name_size].decode("utf-16-le")
                    if name not in {".", ".."}:
                        if not name or "/" in name or "\\" in name:
                            raise EnvironmentError("invalid Windows directory entry name")
                        children.append(name)
                    if not following:
                        break
                    if following < 68 + name_size or offset + following + 68 > len(buffer):
                        raise EnvironmentError("invalid Windows directory entry offset")
                    offset += following
        elif allocated:
            flags = wintypes.DWORD()
            if not kernel.GetVolumeInformationByHandleW(
                handle, None, 0, None, None, ctypes.byref(flags), None, 0
            ):
                fail()
            if flags.value & 0x08000000:  # FILE_SUPPORTS_BLOCK_REFCOUNTING
                volume = ctypes.create_unicode_buffer(32768)
                if not kernel.GetVolumePathNameW(str(path), volume, len(volume)):
                    fail()
                sectors, sector_bytes, free, capacity = (wintypes.DWORD() for _ in range(4))
                if not kernel.GetDiskFreeSpaceW(
                    volume,
                    ctypes.byref(sectors),
                    ctypes.byref(sector_bytes),
                    ctypes.byref(free),
                    ctypes.byref(capacity),
                ):
                    fail()
                cluster = sectors.value * sector_bytes.value
                if not cluster:
                    raise EnvironmentError("Windows volume has no allocation unit size")
                vcn = 0
                resident = False
                while True:
                    start = ctypes.c_int64(vcn)
                    buffer = ctypes.create_string_buffer(65536)
                    returned = wintypes.DWORD()
                    ok = kernel.DeviceIoControl(
                        handle,
                        0x000903D3,
                        ctypes.byref(start),
                        8,
                        buffer,
                        len(buffer),
                        ctypes.byref(returned),
                        None,
                    )
                    error = 0 if ok else get_error()
                    if error == 38 and vcn == 0:  # Resident data has no retrieval pointers.
                        resident = True
                        break
                    if error not in {0, 234}:
                        fail()
                    if returned.value < 16:
                        raise EnvironmentError("truncated Windows physical extent reply")
                    count = struct.unpack_from("=I", buffer.raw)[0]
                    previous = struct.unpack_from("=q", buffer.raw, 8)[0]
                    if not count or previous != vcn or 16 + count * 24 > returned.value:
                        raise EnvironmentError("truncated Windows physical extents")
                    for index in range(count):
                        end, lcn, references = struct.unpack_from(
                            "=qqI", buffer.raw, 16 + index * 24
                        )
                        if end <= previous or lcn < -1 or (lcn >= 0 and not references):
                            raise EnvironmentError("invalid Windows physical extent")
                        if lcn >= 0:
                            extents.append(
                                (lcn * cluster, (lcn + end - previous) * cluster, references > 1)
                            )
                        previous = end
                    if ok:
                        break
                    if previous <= vcn:
                        raise EnvironmentError("Windows physical extent query made no progress")
                    vcn = previous
                if not resident:
                    if sum(end - begin for begin, end, _ in extents) != allocated:
                        raise EnvironmentError(f"unmapped allocation or concurrent change: {path}")
                    allocated = 0
        after_basic = information(0, 40)
        if (
            after_basic[:8] + after_basic[16:36] != basic[:8] + basic[16:36]
            or information(1, 24) != standard
            or information(18, 24) != identity_data
            or _signature(path.lstat()) != _signature(before)
        ):
            raise EnvironmentError(f"managed software changed during observation: {path}")
        return identity, _File([], extents, links, allocated), children
    finally:
        for opened in reversed(handles):
            kernel.CloseHandle(opened)


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

    def account(length: int, items: list[_File], clone_shared: bool) -> None:
        nonlocal total, shared_total, automatic, explicit, reclaim_known
        owners = [owner for item in items for owner in item.owners]
        groups = {owner.group for owner in owners}
        outside_link = any(item.links > len(item.owners) for item in items)
        total += length
        if len(groups) > 1 or outside_link or clone_shared:
            shared_total += length
            for group in groups:
                shared[group] += length
        else:
            exclusive[next(iter(groups))] += length
        auto = all(owner.automatic for owner in owners)
        gc = all(owner.explicit or owner.automatic for owner in owners)
        if not outside_link and (auto or gc) and clone_shared:
            reclaim_known = False
        elif not outside_link and not clone_shared:
            if auto:
                automatic += length
            if gc:
                explicit += length

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
        if sys.platform == "win32":
            identity, item, children = _windows_allocation(path)
            if stat.S_ISDIR(info.st_mode):
                item.owners.append(owner)
                account(item.allocated, [item], False)
                for child in children:
                    visit(path / child, owner)
            elif identity in files:
                files[identity].owners.append(owner)
            else:
                item.owners.append(owner)
                files[identity] = item
                for begin, end, _ in item.extents:
                    events[identity[0]].extend(((begin, 1, identity), (end, -1, identity)))
            return
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
            warnings.append(f"Allocation unavailable under {path}: {exc}")
    for item in files.values():
        if item.allocated:
            account(item.allocated, [item], False)
    for device_events in events.values():
        active: dict[tuple[int, int], int] = defaultdict(int)
        previous = 0
        for position, delta, identity in sorted(device_events):
            length = position - previous
            if active and length:
                items = [files[key] for key in active]
                clone_shared = any(
                    is_shared and begin < position and end > previous
                    for item in items
                    for begin, end, is_shared in item.extents
                )
                account(length, items, clone_shared)
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
    project_directory = _signature(root.stat())
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
    store_directory: tuple[int, ...] | None = None
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
        store_directory = _signature(store.stat())
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
        for category in (
            "bases",
            "wheels",
            "downloads",
            "engines",
            "uv-cache",
            ".garbage",
            "metadata",
        ):
            path = store / category
            group = f"store/{category}"
            if not path.exists():
                continue
            if path.is_symlink() or path.is_junction():
                raise EnvironmentError("space report refuses linked store directories")
            trees.append((path, _Owner(group, False, False)))
        for name in ("projects.json", ".lock"):
            path = store / name
            if path.exists():
                trees.append((path, _Owner("store/metadata", False, False)))
    if (root / ".garbage").exists():
        trees.append((root / ".garbage", _Owner("generation-quarantine", False, False)))
    for name in ("history", "journal", "current", "project.json", "store", "cleanup.json", ".lock"):
        path = root / name
        if path.exists():
            trees.append((path, _Owner("project/software-metadata", False, False)))
    allocation = _allocation(
        trees,
        {store / name for name in explicit_paths} if store is not None else set(),
        group_paths,
        explicit_known,
        bool(policy["automatic"]),
    )
    if explicit_known:
        try:
            _, _, final_generations = collection_plan(root)
            if final_generations != explicit_generations or (
                store is not None
                and store_collection_plan(root, store, final_generations) != explicit_paths
            ):
                raise EnvironmentError("Cleanup references changed during observation.")
        except (OSError, ValueError) as exc:
            allocation["additionalExplicitReclaimableBytes"] = None
            warnings.append(f"Explicit cleanup eligibility is unavailable: {exc}")
    if (
        collection_preview(root) != preview
        or _signature(root.stat()) != project_directory
        or _signature((root / "generations").stat()) != generation_directory
        or (store is not None and _signature(store.stat()) != store_directory)
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
    categories = {
        "bases": "Extracted bases",
        "wheels": "Dependency wheels",
        "downloads": "Base archives",
        "engines": "Engine archives",
        "pack-archives": "Pack archives",
        "uv-cache": "uv cache",
        ".garbage": "Store quarantine",
        "metadata": "Store metadata",
    }
    groups = cast("dict[str, dict[str, int | None]]", allocation["groups"])
    stores: list[dict[str, object]] = []
    for category, label in categories.items():
        group = f"store/{category}"
        groups.setdefault(
            group,
            {
                "exclusiveAllocatedBytes": 0 if allocation["complete"] else None,
                "sharedAllocatedBytes": 0 if allocation["complete"] else None,
            },
        )
        stores.append(
            {
                "allocationGroup": group,
                "label": label,
                "automaticCleanup": False,
                "reasons": [
                    "Shared stores are never removed by automatic cleanup.",
                    "Quarantined files are retained until operation recovery or cleanup."
                    if category == ".garbage"
                    else "Store metadata is protected from explicit cleanup."
                    if category == "metadata"
                    else "Explicit cleanup checks references from every registered project.",
                ],
            }
        )
    settings = [
        "dinkster-env",
        "--root",
        str(root),
        "cleanup",
        "--automatic",
        "on" if policy["automatic"] else "off",
        "--keep-previous",
        str(policy["previousSuccessfulActivations"]),
        "--yes",
    ]
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
        "settingsCommand": subprocess.list2cmdline(settings)
        if os.name == "nt"
        else shlex.join(settings),
        "settingsHelp": "Use --automatic on or off and --keep-previous 1 through 100. "
        "Changing settings does not collect software immediately.",
        "generations": generations,
        "stores": stores,
        "cleanupWarnings": preview["warnings"],
        **allocation,
        "eligibilityWarnings": warnings,
    }
