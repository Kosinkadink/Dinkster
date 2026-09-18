"""Process-usable system memory measurements."""

from __future__ import annotations

import mmap
import platform as platform_module
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path, PurePosixPath

HostMemoryQuery = Callable[[], tuple[int, int]]
HostSwapQuery = Callable[[], tuple[int, int]]
_POSIX_ROOT = PurePosixPath("/")
_64_BIT_KERNEL_MACHINES = frozenset(
    {
        "aarch64",
        "alpha",
        "amd64",
        "arm64",
        "ia64",
        "loongarch64",
        "mips64",
        "mips64el",
        "ppc64",
        "ppc64le",
        "riscv64",
        "s390x",
        "sparc64",
        "x86_64",
    }
)


@dataclass(frozen=True, slots=True)
class SystemMemorySnapshot:
    """One fresh host and effective memory and swap measurement in bytes."""

    host_total_bytes: int
    host_available_bytes: int
    effective_total_bytes: int
    effective_available_bytes: int
    provenance: tuple[str, ...]
    host_swap_total_bytes: int | None = None
    host_swap_available_bytes: int | None = None
    effective_swap_total_bytes: int | None = None
    effective_swap_available_bytes: int | None = None

    @property
    def effective_used_bytes(self) -> int:
        return self.effective_total_bytes - self.effective_available_bytes


@dataclass(frozen=True, slots=True)
class _CgroupMemoryStatus:
    version: int | None = None
    total_bytes: int | None = None
    available_bytes: int | None = None
    swap_total_bytes: int | None = None
    swap_available_bytes: int | None = None


LinuxCgroupQuery = Callable[[], tuple[int | None, int | None] | _CgroupMemoryStatus]


def _psutil_memory_status() -> tuple[int, int]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required to determine system memory") from exc
    status = psutil.virtual_memory()
    return int(status.total), int(status.available)


def _psutil_swap_status() -> tuple[int, int]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required to determine system memory") from exc
    status = psutil.swap_memory()
    return int(status.total), int(status.free)


def _read_cgroup_value(path: Path) -> int | None:
    value = path.read_bytes().strip()
    if value == b"max":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"negative cgroup memory value in {path}")
    return parsed


@cache
def _v1_unlimited_value(
    page_size: int | None = None,
    kernel_machine: str | None = None,
    userspace_maxsize: int | None = None,
) -> int:
    if page_size is None:
        page_size = mmap.PAGESIZE
    if page_size <= 0:
        raise ValueError("page size must be positive")
    if userspace_maxsize is None:
        userspace_maxsize = sys.maxsize
    if userspace_maxsize > (1 << 31) - 1:
        kernel_bits = 64
    else:
        if kernel_machine is None:
            kernel_machine = platform_module.machine()
        machine = kernel_machine.lower()
        kernel_bits = 64 if machine in _64_BIT_KERNEL_MACHINES else 32
    long_max = (1 << (kernel_bits - 1)) - 1
    # Linux page counters use different exact unlimited formulas by kernel word size.
    return (long_max // page_size) * page_size if kernel_bits == 64 else long_max * page_size


def _is_cgroup_v1_unlimited(
    value: int,
    page_size: int | None = None,
    kernel_machine: str | None = None,
    userspace_maxsize: int | None = None,
) -> bool:
    return value == -1 or value == _v1_unlimited_value(
        page_size,
        kernel_machine,
        userspace_maxsize,
    )


def _read_cgroup_v1_value(path: Path) -> int | None:
    parsed = int(path.read_bytes().strip())
    if _is_cgroup_v1_unlimited(parsed):
        return None
    if parsed < 0:
        raise ValueError(f"negative cgroup memory value in {path}")
    return parsed


def _cgroup_ancestors(cgroup: Path, root: Path) -> tuple[Path, ...]:
    if cgroup != root and root not in cgroup.parents:
        raise ValueError(f"cgroup path {cgroup} is outside root {root}")
    ancestors: list[Path] = []
    current = cgroup
    while True:
        ancestors.append(current)
        if current == root:
            return tuple(ancestors)
        current = current.parent


def _cgroup_v2_memory_status(
    cgroup: Path,
    root: Path,
    *,
    include_swap: bool,
) -> _CgroupMemoryStatus:
    limits: list[int] = []
    available: list[int] = []
    swap_limits: list[int] = []
    swap_available: list[int] = []
    for current in _cgroup_ancestors(cgroup, root):
        try:
            limit = _read_cgroup_value(current / "memory.max")
        except (OSError, ValueError):
            limit = None
        if limit is not None:
            try:
                usage = _read_cgroup_value(current / "memory.current")
            except (OSError, ValueError):
                usage = None
            if usage is not None:
                try:
                    inactive_file = _read_cgroup_stat(current / "memory.stat").get("inactive_file")
                except OSError:
                    inactive_file = None
                if inactive_file is not None:
                    # Inactive file cache is reclaimable, matching host available-RAM semantics.
                    usage = max(0, usage - inactive_file)
                limits.append(limit)
                available.append(max(0, limit - usage))
        if include_swap:
            try:
                swap_limit = _read_cgroup_value(current / "memory.swap.max")
            except (OSError, ValueError):
                swap_limit = None
            if swap_limit is not None:
                try:
                    swap_usage = _read_cgroup_value(current / "memory.swap.current")
                except (OSError, ValueError):
                    swap_usage = None
                if swap_usage is not None:
                    swap_limits.append(swap_limit)
                    swap_available.append(max(0, swap_limit - swap_usage))
    return _CgroupMemoryStatus(
        version=2,
        total_bytes=min(limits) if limits else None,
        available_bytes=min(available) if available else None,
        swap_total_bytes=min(swap_limits) if swap_limits else None,
        swap_available_bytes=min(swap_available) if swap_available else None,
    )


def _read_cgroup_stat(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            parsed = int(fields[1])
        except ValueError:
            continue
        if parsed >= 0:
            values[fields[0]] = parsed
    return values


def _finite_cgroup_v1_stat_value(values: dict[str, int], key: str) -> int | None:
    value = values.get(key)
    if value is None or _is_cgroup_v1_unlimited(value):
        return None
    return value


def _append_v1_memory_pair(
    limits: list[int],
    available: list[int],
    limit: int | None,
    usage: int | None,
) -> None:
    if limit is None:
        return
    limits.append(limit)
    if usage is not None:
        available.append(max(0, limit - usage))


def _append_v1_swap_pair(
    limits: list[int],
    available: list[int],
    *,
    memory_limit: int | None,
    memory_usage: int | None,
    memsw_limit: int | None,
    memsw_usage: int | None,
) -> None:
    if memory_limit is None or memsw_limit is None:
        return
    swap_limit = max(0, memsw_limit - memory_limit)
    limits.append(swap_limit)
    if memory_usage is None or memsw_usage is None:
        return
    swap_usage = max(0, memsw_usage - memory_usage)
    available.append(
        min(
            max(0, swap_limit - swap_usage),
            max(0, memsw_limit - memsw_usage),
        )
    )


def _cgroup_v1_memory_status(
    cgroup: Path,
    root: Path,
    *,
    include_swap: bool,
) -> _CgroupMemoryStatus:
    limits: list[int] = []
    available: list[int] = []
    swap_limits: list[int] = []
    swap_available: list[int] = []
    for current in _cgroup_ancestors(cgroup, root):
        try:
            memory_limit = _read_cgroup_v1_value(current / "memory.limit_in_bytes")
        except (OSError, ValueError):
            memory_limit = None
        try:
            memory_usage = _read_cgroup_v1_value(current / "memory.usage_in_bytes")
        except (OSError, ValueError):
            memory_usage = None
        _append_v1_memory_pair(limits, available, memory_limit, memory_usage)

        if include_swap:
            try:
                memsw_limit = _read_cgroup_v1_value(current / "memory.memsw.limit_in_bytes")
            except (OSError, ValueError):
                memsw_limit = None
            try:
                memsw_usage = _read_cgroup_v1_value(current / "memory.memsw.usage_in_bytes")
            except (OSError, ValueError):
                memsw_usage = None
            _append_v1_swap_pair(
                swap_limits,
                swap_available,
                memory_limit=memory_limit,
                memory_usage=memory_usage,
                memsw_limit=memsw_limit,
                memsw_usage=memsw_usage,
            )

    try:
        stat = _read_cgroup_stat(cgroup / "memory.stat")
    except OSError:
        stat = {}
    hierarchical_memory_limit = _finite_cgroup_v1_stat_value(stat, "hierarchical_memory_limit")
    if hierarchical_memory_limit is not None:
        limits.append(hierarchical_memory_limit)
    total = min(limits) if limits else None
    headroom = min(available) if available else None
    swap_total = min(swap_limits) if swap_limits else None
    swap_headroom = min(swap_available) if swap_available else None
    return _CgroupMemoryStatus(
        version=1,
        total_bytes=total,
        available_bytes=(
            min(headroom, total) if headroom is not None and total is not None else headroom
        ),
        swap_total_bytes=swap_total,
        swap_available_bytes=(
            min(swap_headroom, swap_total)
            if swap_headroom is not None and swap_total is not None
            else swap_headroom
        ),
    )


def _unescape_mountinfo_path(value: str) -> str:
    escapes = {"011": "\t", "012": "\n", "040": " ", "134": "\\"}
    decoded: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and value[index + 1 : index + 4] in escapes:
            decoded.append(escapes[value[index + 1 : index + 4]])
            index += 4
        else:
            decoded.append(value[index])
            index += 1
    return "".join(decoded)


def _absolute_posix_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid absolute path: {value}")
    return path


def _cgroup_mounts(
    proc_mountinfo: Path,
    *,
    filesystem: str,
    controller: str | None = None,
) -> tuple[tuple[PurePosixPath, Path], ...]:
    mounts: list[tuple[PurePosixPath, Path]] = []
    for line in proc_mountinfo.read_text().splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) <= separator + 3:
            continue
        if fields[separator + 1] != filesystem:
            continue
        super_options = fields[separator + 3].split(",")
        if controller is not None and controller not in super_options:
            continue
        try:
            root = _absolute_posix_path(_unescape_mountinfo_path(fields[3]))
            mountpoint = Path(_absolute_posix_path(_unescape_mountinfo_path(fields[4])))
        except ValueError:
            continue
        mounts.append((root, mountpoint))
    return tuple(mounts)


def _cgroup2_mounts(proc_mountinfo: Path) -> tuple[tuple[PurePosixPath, Path], ...]:
    return _cgroup_mounts(proc_mountinfo, filesystem="cgroup2")


def _cgroup1_memory_mounts(
    proc_mountinfo: Path,
) -> tuple[tuple[PurePosixPath, Path], ...]:
    return _cgroup_mounts(proc_mountinfo, filesystem="cgroup", controller="memory")


@lru_cache(maxsize=1)
def _default_cgroup2_mounts() -> tuple[tuple[PurePosixPath, Path], ...]:
    return _cgroup2_mounts(Path("/proc/self/mountinfo"))


@lru_cache(maxsize=1)
def _default_cgroup1_memory_mounts() -> tuple[tuple[PurePosixPath, Path], ...]:
    return _cgroup1_memory_mounts(Path("/proc/self/mountinfo"))


def _linux_cgroup_paths(
    proc_cgroup: Path,
) -> tuple[PurePosixPath | None, PurePosixPath | None]:
    v1_memory_path: PurePosixPath | None = None
    v2_path: PurePosixPath | None = None
    for line in proc_cgroup.read_text().splitlines():
        try:
            hierarchy, controllers, relative = line.split(":", 2)
            hierarchy_path = _absolute_posix_path(relative)
        except ValueError:
            continue
        if hierarchy == "0" and not controllers:
            v2_path = hierarchy_path
        elif "memory" in controllers.split(","):
            v1_memory_path = hierarchy_path
    return v1_memory_path, v2_path


def _linux_cgroup_location(
    hierarchy_path: PurePosixPath,
    mounts: tuple[tuple[PurePosixPath, Path], ...],
) -> tuple[Path, Path] | None:
    if len(mounts) == 1 and mounts[0][0] == _POSIX_ROOT:
        mountpoint = mounts[0][1]
        return mountpoint.joinpath(*hierarchy_path.parts[1:]), mountpoint

    candidates: list[tuple[int, Path, Path, bool]] = []
    namespace_mounts: list[tuple[PurePosixPath, Path]] = []
    for root, mountpoint in mounts:
        try:
            relative = hierarchy_path.relative_to(root)
        except ValueError:
            if root != _POSIX_ROOT:
                namespace_mounts.append((root, mountpoint))
            continue
        location = mountpoint.joinpath(*relative.parts)
        candidates.append((len(root.parts), location, mountpoint, False))
    if not candidates:
        relative = hierarchy_path.relative_to(_POSIX_ROOT)
        candidates = [
            (
                len(root.parts),
                mountpoint.joinpath(*relative.parts),
                mountpoint,
                True,
            )
            for root, mountpoint in namespace_mounts
        ]
    if not candidates:
        return None
    if len(candidates) == 1:
        _, location, mountpoint, namespace_relative = candidates[0]
        if namespace_relative and not (location / "cgroup.procs").is_file():
            return None
        return location, mountpoint
    for _, location, mountpoint, _ in sorted(
        candidates, key=lambda candidate: candidate[0], reverse=True
    ):
        if (location / "cgroup.procs").is_file():
            return location, mountpoint
    return None


def _linux_cgroup_status(
    *,
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    proc_mountinfo: Path | None = None,
    include_swap: bool = False,
) -> _CgroupMemoryStatus:
    v1_path, v2_path = _linux_cgroup_paths(proc_cgroup)
    if v1_path is not None:
        mounts = (
            _default_cgroup1_memory_mounts()
            if proc_mountinfo is None
            else _cgroup1_memory_mounts(proc_mountinfo)
        )
        location = _linux_cgroup_location(v1_path, mounts)
        if location is None:
            return _CgroupMemoryStatus()
        return _cgroup_v1_memory_status(*location, include_swap=include_swap)
    if v2_path is None:
        return _CgroupMemoryStatus()
    mounts = (
        _default_cgroup2_mounts() if proc_mountinfo is None else _cgroup2_mounts(proc_mountinfo)
    )
    location = _linux_cgroup_location(v2_path, mounts)
    if location is None:
        return _CgroupMemoryStatus()
    return _cgroup_v2_memory_status(*location, include_swap=include_swap)


def _linux_cgroup_memory_status(  # pyright: ignore[reportUnusedFunction]
    *,
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    proc_mountinfo: Path | None = None,
) -> tuple[int | None, int | None]:
    status = _linux_cgroup_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    )
    return status.total_bytes, status.available_bytes


def system_memory_snapshot(
    platform: str | None = None,
    *,
    host_query: HostMemoryQuery | None = None,
    host_swap_query: HostSwapQuery | None = None,
    linux_cgroup_query: LinuxCgroupQuery | None = None,
    include_swap: bool = False,
) -> SystemMemorySnapshot:
    """Measure host RAM and optional swap, then apply Linux cgroup headroom.

    The query seams keep platform and cgroup fixtures hermetic. Runtime callers
    omit them and receive a fresh psutil and cgroup measurement on every call.
    Swap is opt-in because RAM policy does not consume it and its files are
    independently mutable.
    """

    current = sys.platform if platform is None else platform
    query = _psutil_memory_status if host_query is None else host_query
    host_source = "psutil" if host_query is None else "injected"
    try:
        host_total, host_available = query()
        host_total = int(host_total)
        host_available = int(host_available)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("cannot determine system memory") from exc
    if host_total <= 0 or host_available < 0:
        raise RuntimeError("system memory query returned invalid values")
    host_available = min(host_available, host_total)

    if not include_swap:
        host_swap_total, host_swap_available = None, None
    else:
        swap_query = (
            _psutil_swap_status
            if host_swap_query is None and host_query is None
            else host_swap_query
        )
        if swap_query is None:
            host_swap_total, host_swap_available = None, None
        else:
            try:
                host_swap_total, host_swap_available = swap_query()
                host_swap_total = int(host_swap_total)
                host_swap_available = int(host_swap_available)
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError("cannot determine system swap") from exc
            if host_swap_total < 0 or host_swap_available < 0:
                raise RuntimeError("system swap query returned invalid values")
            host_swap_available = min(host_swap_available, host_swap_total)

    effective_total = host_total
    effective_available = host_available
    effective_swap_total = host_swap_total
    effective_swap_available = host_swap_available
    provenance = [host_source]
    if current.startswith("linux"):
        try:
            if linux_cgroup_query is None:
                cgroup = _linux_cgroup_status(include_swap=include_swap)
            else:
                injected_cgroup = linux_cgroup_query()
                if isinstance(injected_cgroup, _CgroupMemoryStatus):
                    cgroup = injected_cgroup
                else:
                    cgroup_total, cgroup_available = injected_cgroup
                    cgroup = _CgroupMemoryStatus(
                        version=2,
                        total_bytes=cgroup_total,
                        available_bytes=cgroup_available,
                    )
            values = (
                cgroup.total_bytes,
                cgroup.available_bytes,
                cgroup.swap_total_bytes,
                cgroup.swap_available_bytes,
            )
            if any(value is not None and value < 0 for value in values):
                raise ValueError("negative cgroup memory value")
        except (OSError, ValueError):
            cgroup = _CgroupMemoryStatus()
            values = (None, None, None, None)
        if cgroup.total_bytes is not None:
            effective_total = min(effective_total, cgroup.total_bytes)
        if cgroup.available_bytes is not None:
            effective_available = min(effective_available, cgroup.available_bytes)
        if effective_swap_total is not None and cgroup.swap_total_bytes is not None:
            effective_swap_total = min(effective_swap_total, cgroup.swap_total_bytes)
        if effective_swap_available is not None and cgroup.swap_available_bytes is not None:
            effective_swap_available = min(effective_swap_available, cgroup.swap_available_bytes)
        if any(value is not None for value in values):
            provenance.append(f"cgroup-v{cgroup.version}")

    effective_available = min(effective_available, effective_total)
    if effective_swap_total is not None and effective_swap_available is not None:
        effective_swap_available = min(effective_swap_available, effective_swap_total)
    return SystemMemorySnapshot(
        host_total_bytes=host_total,
        host_available_bytes=host_available,
        effective_total_bytes=effective_total,
        effective_available_bytes=effective_available,
        provenance=tuple(provenance),
        host_swap_total_bytes=host_swap_total,
        host_swap_available_bytes=host_swap_available,
        effective_swap_total_bytes=effective_swap_total,
        effective_swap_available_bytes=effective_swap_available,
    )


__all__ = ["SystemMemorySnapshot", "system_memory_snapshot"]
