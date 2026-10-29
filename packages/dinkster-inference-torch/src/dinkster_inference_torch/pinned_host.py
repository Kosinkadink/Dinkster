"""Torch-free process accounting for aimdo host pins.

Dinkster has no RAM_CACHE_HEADROOM subsystem, so the available-RAM arm
uses ComfyUI's standalone 2 GiB floor.

TOTAL_PINNED_STORAGE bounds physical host-buffer bytes. TOTAL_PINNED_MEMORY
tracks the subset currently registered with CUDA; registration pressure may
lower it without lowering physical storage. The physical cap is independent
of the lower Windows CUDA-registration ratio.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol

from dinkster_inference import GIBIBYTE, MEBIBYTE
from dinkster_memory import system_memory_snapshot

_logger = logging.getLogger(__name__)

PIN_PRESSURE_HYSTERESIS = 256 * MEBIBYTE
REGISTERABLE_PIN_HYSTERESIS = 2 * GIBIBYTE
AVAILABLE_RAM_FLOOR = 2 * GIBIBYTE
DEFAULT_STAGING_FRACTION = 0.30
MAXIMUM_STAGING_FRACTION = 0.90
STAGING_RESERVE_FRACTION = 0.20
MINIMUM_STAGING_RESERVE = 8 * GIBIBYTE
_STORAGE_RESERVE_CHECK_INTERVAL_NS = 100_000_000


class PinOwner(Protocol):
    pin_active: bool

    def free_pins(self, size: int) -> int: ...
    def free_registrations(self, size: int) -> int: ...


def _injected_host_query(
    platform: str,
    windows_query: Callable[[], tuple[int, int]] | None,
    sysconf_query: Callable[[str], int] | None,
) -> Callable[[], tuple[int, int]] | None:
    if platform.startswith("win") and windows_query is not None:
        return windows_query
    if sysconf_query is None:
        return None

    def query() -> tuple[int, int]:
        page_size = sysconf_query("SC_PAGE_SIZE")
        return (
            page_size * sysconf_query("SC_PHYS_PAGES"),
            page_size * sysconf_query("SC_AVPHYS_PAGES"),
        )

    return query


def memory_status(
    platform: str | None = None,
    *,
    windows_query: Callable[[], tuple[int, int]] | None = None,
    linux_cgroup_query: Callable[[], tuple[int | None, int | None]] | None = None,
    sysconf_query: Callable[[str], int] | None = None,
) -> tuple[int, int]:
    """Return effective physical RAM without importing torch."""
    current = sys.platform if platform is None else platform
    snapshot = system_memory_snapshot(
        current,
        host_query=_injected_host_query(current, windows_query, sysconf_query),
        linux_cgroup_query=linux_cgroup_query,
    )
    return snapshot.effective_total_bytes, snapshot.effective_available_bytes


def host_memory_status(
    platform: str | None = None,
    *,
    windows_query: Callable[[], tuple[int, int]] | None = None,
    sysconf_query: Callable[[str], int] | None = None,
) -> tuple[int, int]:
    """Return host RAM from the shared provider without applying its clamp."""
    current = sys.platform if platform is None else platform
    snapshot = system_memory_snapshot(
        current,
        host_query=_injected_host_query(current, windows_query, sysconf_query),
        linux_cgroup_query=lambda: (None, None),
    )
    return snapshot.host_total_bytes, snapshot.host_available_bytes


def platform_pin_ratio(platform: str | None = None) -> float:
    current = sys.platform if platform is None else platform
    return 0.40 if current.startswith("win") else 0.90


def _initial_maximum() -> int:
    try:
        workers = max(1, int(os.environ.get("DINKSTER_SINGLE_JOB_WORLD_SIZE", "1")))
        return int(memory_status()[0] * platform_pin_ratio() / workers)
    except (AttributeError, OSError, ValueError):
        return -1


def staging_fraction() -> float:
    try:
        value = float(os.environ.get("DINKSTER_PINNED_STAGING_FRACTION", DEFAULT_STAGING_FRACTION))
    except ValueError:
        return DEFAULT_STAGING_FRACTION
    return value if 0.0 <= value <= 1.0 else DEFAULT_STAGING_FRACTION


def _initial_storage_maximum() -> int:
    try:
        workers = max(1, int(os.environ.get("DINKSTER_SINGLE_JOB_WORLD_SIZE", "1")))
        total = memory_status()[0]
        ratio = min(MAXIMUM_STAGING_FRACTION, staging_fraction())
        return int(total * ratio / workers)
    except (AttributeError, OSError, ValueError):
        return -1


_lock = threading.RLock()
_storage_reserve_check_lock = threading.Lock()
_owners: list[PinOwner] = []
_storage_by_owner: dict[int, int] = {}
_storage_reserve_checked_at_ns = 0
_storage_reserve_checked_query: object | None = None
_warned_pin_refusals: set[str] = set()
DISABLED = False
TOTAL_PINNED_MEMORY = 0
TOTAL_PINNED_STORAGE = 0
MAX_PINNED_MEMORY = _initial_maximum()
MAX_PINNED_STORAGE = _initial_storage_maximum()


def _warn_pin_refusal(reason: str, message: str, *args: object) -> None:
    with _lock:
        if reason in _warned_pin_refusals:
            return
        _warned_pin_refusals.add(reason)
    _logger.warning("pinned host allocation refused: reason=%s " + message, reason, *args)


def configure(
    *,
    disabled: bool | None = None,
    maximum: int | None = None,
    storage_maximum: int | None = None,
) -> None:
    global DISABLED, MAX_PINNED_MEMORY, MAX_PINNED_STORAGE
    with _lock:
        if disabled is not None:
            DISABLED = disabled  # pyright: ignore[reportConstantRedefinition]
        if maximum is not None:
            MAX_PINNED_MEMORY = int(maximum)  # pyright: ignore[reportConstantRedefinition]
            if storage_maximum is None:
                MAX_PINNED_STORAGE = int(maximum)  # pyright: ignore[reportConstantRedefinition]
        if storage_maximum is not None:
            MAX_PINNED_STORAGE = int(storage_maximum)  # pyright: ignore[reportConstantRedefinition]


def pinned_hostbuf_size(size: int, *, high_ram: bool = False) -> int:
    maximum = int(size) if high_ram or MAX_PINNED_MEMORY < 0 else min(int(size), MAX_PINNED_MEMORY)
    return max(0, maximum * 2)


def available_ram() -> int:
    try:
        return memory_status()[1]
    except (AttributeError, OSError, ValueError):
        return 0


def register_owner(owner: PinOwner) -> None:
    """Strongly register or touch one physical-storage owner as most recent."""
    with _lock:
        for index, candidate in enumerate(_owners):
            if candidate is owner:
                _owners.append(_owners.pop(index))
                return
        _owners.append(owner)
        _storage_by_owner[id(owner)] = 0


def unregister_owner(owner: PinOwner) -> None:
    with _lock:
        if _storage_by_owner.get(id(owner), 0):
            raise RuntimeError("cannot unregister an owner with pinned host storage")
        _owners[:] = [candidate for candidate in _owners if candidate is not owner]
        _storage_by_owner.pop(id(owner), None)


def discard_owner_if_empty(owner: PinOwner) -> None:
    with _lock:
        if _storage_by_owner.get(id(owner), 0) == 0:
            _owners[:] = [candidate for candidate in _owners if candidate is not owner]
            _storage_by_owner.pop(id(owner), None)


def account_storage(owner: PinOwner, size: int) -> None:
    """Account physical host-buffer bytes independently of CUDA registration."""
    global TOTAL_PINNED_STORAGE
    delta = int(size)
    with _lock:
        key = id(owner)
        if key not in _storage_by_owner:
            raise RuntimeError("pinned host storage has no registered owner")
        owned = _storage_by_owner[key] + delta
        if owned < 0:
            raise RuntimeError("pinned host owner storage accounting became negative")
        total = TOTAL_PINNED_STORAGE + delta
        if total < 0:
            raise RuntimeError("pinned host storage accounting became negative")
        _storage_by_owner[key] = owned
        TOTAL_PINNED_STORAGE = total  # pyright: ignore[reportConstantRedefinition]


def _inactive_owners(exclude: PinOwner | None = None) -> tuple[PinOwner, ...]:
    with _lock:
        return tuple(owner for owner in _owners if owner is not exclude and not owner.pin_active)


def ensure_storage_reserve(
    *,
    status: Callable[[], tuple[int, int]] | None = None,
) -> bool:
    """Reclaim inactive staging when live host headroom falls below reserve."""
    query = memory_status if status is None else status
    if status is None:
        # Every routed layer enters here, so share successful pressure checks
        # briefly. New staging allocations still force a fresh check below.
        global _storage_reserve_checked_at_ns, _storage_reserve_checked_query
        with _storage_reserve_check_lock:
            now = time.monotonic_ns()
            if (
                query is _storage_reserve_checked_query
                and now - _storage_reserve_checked_at_ns < _STORAGE_RESERVE_CHECK_INTERVAL_NS
            ):
                return True
            result = ensure_storage_reserve(status=query)
            if result:
                _storage_reserve_checked_at_ns = time.monotonic_ns()
                _storage_reserve_checked_query = query
            return result
    try:
        total, available = query()
    except (AttributeError, OSError, ValueError):
        return True
    reserve = max(int(total * STAGING_RESERVE_FRACTION), MINIMUM_STAGING_RESERVE)
    if available >= reserve:
        return True
    for candidate in _inactive_owners():
        candidate.free_pins(reserve - available)
        try:
            _, available = query()
        except (AttributeError, OSError, ValueError):
            return True
        if available >= reserve:
            return True
    return False


def reserve_storage(owner: PinOwner, size: int) -> bool:
    """Reserve bounded physical staging after one finite inactive-LRU pass."""
    requested = max(0, int(size))
    ensure_storage_reserve(status=memory_status)
    register_owner(owner)
    with _lock:
        if MAX_PINNED_STORAGE < 0 or TOTAL_PINNED_STORAGE + requested <= MAX_PINNED_STORAGE:
            account_storage(owner, requested)
            return True
    for candidate in _inactive_owners(owner):
        with _lock:
            shortfall = TOTAL_PINNED_STORAGE + requested - MAX_PINNED_STORAGE
        if shortfall <= 0:
            break
        candidate.free_pins(shortfall)
    with _lock:
        if MAX_PINNED_STORAGE >= 0 and TOTAL_PINNED_STORAGE + requested > MAX_PINNED_STORAGE:
            stored = TOTAL_PINNED_STORAGE
            maximum = MAX_PINNED_STORAGE
        else:
            account_storage(owner, requested)
            return True
    if not DISABLED:
        _warn_pin_refusal(
            "physical-storage-cap",
            "requested_bytes=%d stored_bytes=%d maximum_bytes=%d",
            requested,
            stored,
            maximum,
        )
    return False


def log_storage_ledger(phase: str) -> None:
    """Log physical staging composition when explicitly enabled."""
    if os.environ.get("DINKSTER_PINNED_STORAGE_DEBUG") != "1":
        return
    import json

    with _lock:
        rows = []
        for owner in _owners:
            label = getattr(owner, "pin_debug_label", None)
            rows.append(
                {
                    "bytes": _storage_by_owner.get(id(owner), 0),
                    "owner": label() if callable(label) else type(owner).__name__,
                }
            )
        total = TOTAL_PINNED_STORAGE
        registered = TOTAL_PINNED_MEMORY
    print(
        "DINKSTER_PINNED_STORAGE "
        f"pid={os.getpid()} phase={phase} total={total} registered={registered} "
        f"owners={json.dumps(rows, separators=(',', ':'))}",
        file=sys.stderr,
        flush=True,
    )


def _visit(method: str, needed: int, evict_active: bool) -> int:
    freed = 0
    with _lock:
        live = tuple(_owners)
    for active in (False, True):
        if active and not evict_active:
            break
        owners = reversed(live) if active else live
        for owner in owners:
            if owner.pin_active != active:
                continue
            freed += int(getattr(owner, method)(needed - freed))
            if freed >= needed:
                return freed
    return freed


def free_pins(size: int, *, evict_active: bool = False) -> int:
    return _visit("free_pins", max(0, int(size)), evict_active)


def free_registrations(shortfall: int, *, evict_active: bool = True) -> bool:
    if MAX_PINNED_MEMORY == 0 or shortfall <= 0:
        return shortfall <= 0
    needed = int(shortfall) + REGISTERABLE_PIN_HYSTERESIS
    freed = _visit("free_registrations", needed, evict_active)
    return freed >= int(shortfall)


def ensure_pin_registerable(size: int, *, evict_active: bool = True) -> bool:
    if DISABLED:
        return False
    requested = int(size)
    if MAX_PINNED_MEMORY < 0:
        return True
    shortfall = TOTAL_PINNED_MEMORY + requested - MAX_PINNED_MEMORY
    if free_registrations(shortfall, evict_active=evict_active):
        return True
    _warn_pin_refusal(
        "registration-cap",
        "requested_bytes=%d registered_bytes=%d maximum_bytes=%d",
        requested,
        TOTAL_PINNED_MEMORY,
        MAX_PINNED_MEMORY,
    )
    return False


def ensure_pin_budget(
    size: int,
    *,
    available: Callable[[], int] = available_ram,
    evict_active: bool = False,
) -> bool:
    if DISABLED:
        return False
    requested = int(size)
    available_bytes = int(available())
    shortfall = requested + AVAILABLE_RAM_FLOOR - available_bytes
    if shortfall <= 0:
        return True
    reclaimed = free_pins(shortfall + PIN_PRESSURE_HYSTERESIS, evict_active=evict_active)
    if reclaimed >= shortfall:
        return True
    _warn_pin_refusal(
        "ram-budget",
        "requested_bytes=%d available_bytes=%d floor_bytes=%d reclaimed_bytes=%d",
        requested,
        available_bytes,
        AVAILABLE_RAM_FLOOR,
        reclaimed,
    )
    return False


def account(size: int) -> None:
    global TOTAL_PINNED_MEMORY
    with _lock:
        TOTAL_PINNED_MEMORY = max(  # pyright: ignore[reportConstantRedefinition]
            0, TOTAL_PINNED_MEMORY + int(size)
        )
