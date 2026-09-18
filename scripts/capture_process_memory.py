"""Capture sensitive Linux process anonymous-memory evidence.

Receipts retain raw mappings and pathnames plus pinned-storage owner labels.
Treat every receipt as sensitive evidence and restrict its storage and sharing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

SCHEMA = "dinkster-process-memory/1"
DEFAULT_EPSILON_BYTES = 8 * 1024 * 1024
MAX_SMAPS_BYTES = 16 * 1024 * 1024
MAX_ROLLUP_BYTES = 2 * 1024 * 1024
MAX_PROC_METADATA_BYTES = 2 * 1024 * 1024
MAX_CGROUP_FILE_BYTES = 2 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_CAPTURE_RAW_BYTES = 16 * 1024 * 1024
MAX_SMAPS_MAPPING_COUNT = 16384
MAX_SMAPS_FIELD_COUNT = 131072
MAX_PROCESS_COUNT = 256
MAX_TREE_TASK_COUNT = 8192
MAX_TREE_CHILDREN_BYTES = 64 * 1024 * 1024
MAX_LEDGER_LOG_BYTES = 64 * 1024 * 1024

# memory.stat gauges documented by cgroup v2 as byte amounts. All other
# fields are treated as counters and must remain exactly stable.
_MEMORY_STAT_BYTE_GAUGES = frozenset(
    {
        "active_anon",
        "active_file",
        "anon",
        "anon_thp",
        "file",
        "file_dirty",
        "file_mapped",
        "file_thp",
        "file_writeback",
        "hugetlb",
        "inactive_anon",
        "inactive_file",
        "kernel",
        "kernel_stack",
        "pagetables",
        "percpu",
        "sec_pagetables",
        "shmem",
        "shmem_thp",
        "slab",
        "slab_reclaimable",
        "slab_unreclaimable",
        "sock",
        "swapcached",
        "unevictable",
        "vmalloc",
        "zswap",
        "zswap_incomp",
        "zswapped",
    }
)

_HEADER = re.compile(
    r"^([0-9a-fA-F]+)-([0-9a-fA-F]+) "
    r"([rwxps-]{4}) ([0-9a-fA-F]+) ([0-9a-fA-F]+:[0-9a-fA-F]+) (\d+)\s*(.*)$"
)
_FIELD = re.compile(r"^([A-Za-z][A-Za-z0-9_()]*):\s*(.*?)\s*$")
_NUMBER = re.compile(r"^(\d+)(?:\s+([A-Za-z]+))?$")
_PINNED_STORAGE = re.compile(
    r"^DINKSTER_PINNED_STORAGE pid=(\d+) phase=(\S+) total=(\d+) "
    r"registered=(\d+) owners=(\[.*\])$"
)


class CaptureError(RuntimeError):
    """A stable, receipt-safe capture failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _interval_id(start: int, end: int) -> str:
    return f"{start:x}-{end:x}"


def _mapping_id(
    interval_id: str,
    permissions: str,
    offset: int,
    device: str,
    inode: int,
    pathname: str,
) -> str:
    return ":".join((interval_id, permissions, f"{offset:x}", device, str(inode), pathname))


def read_bounded(path: Path, maximum: int) -> str:
    """Read at most maximum bytes and reject truncated diagnostic evidence."""
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum + 1)
    except FileNotFoundError as error:
        raise CaptureError("not-found") from error
    except PermissionError as error:
        raise CaptureError("permission-denied") from error
    except OSError as error:
        raise CaptureError("read-failed") from error
    if len(payload) > maximum:
        raise CaptureError("read-limit-exceeded")
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise CaptureError("non-ascii-input") from error


def normalized_class(pathname: str) -> str:
    if not pathname:
        return "anonymous"
    if pathname.startswith("[") and pathname.endswith("]"):
        return f"kernel:{pathname}"
    return f"path:{pathname}"


def _field(raw_value: str) -> dict[str, object]:
    match = _NUMBER.fullmatch(raw_value)
    if match is None:
        return {"raw": raw_value}
    value = int(match.group(1))
    unit = match.group(2)
    parsed: dict[str, object] = {"raw": raw_value, "value": value}
    if unit is not None:
        parsed["unit"] = unit
        if unit == "kB":
            parsed["bytes"] = value * 1024
    return parsed


def parse_smaps(text: str) -> list[dict[str, object]]:
    """Parse smaps while preserving every field's source representation."""
    mappings: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    field_count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        header = _HEADER.fullmatch(line)
        if header is not None:
            start = int(header.group(1), 16)
            end = int(header.group(2), 16)
            if end <= start:
                raise ValueError(f"smaps line {line_number} has an invalid interval")
            pathname = header.group(7)
            interval_id = _interval_id(start, end)
            permissions = header.group(3)
            offset = int(header.group(4), 16)
            device = header.group(5).lower()
            inode = int(header.group(6))
            current = {
                "start": start,
                "end": end,
                "interval_id": interval_id,
                "permissions": permissions,
                "offset": offset,
                "device": device,
                "inode": inode,
                "pathname": pathname,
                "normalized_class": normalized_class(pathname),
                "fields": {},
            }
            if len(mappings) >= MAX_SMAPS_MAPPING_COUNT:
                raise ValueError("smaps mapping limit exceeded")
            current["mapping_id"] = _mapping_id(
                interval_id, permissions, offset, device, inode, pathname
            )
            mappings.append(current)
            continue
        if not line:
            continue
        if current is None:
            raise ValueError(f"smaps line {line_number} precedes a mapping header")
        field_match = _FIELD.fullmatch(line)
        if field_match is None:
            raise ValueError(f"smaps line {line_number} is malformed")
        fields = current["fields"]
        assert isinstance(fields, dict)
        name = field_match.group(1)
        if name in fields:
            raise ValueError(f"smaps line {line_number} repeats field {name}")
        field_count += 1
        if field_count > MAX_SMAPS_FIELD_COUNT:
            raise ValueError("smaps field limit exceeded")
        fields[name] = _field(field_match.group(2))
    if text.strip() and not mappings:
        raise ValueError("smaps has no mappings")
    return mappings


def parse_integer_table(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"integer table line {line_number} is malformed")
        name, raw_value = fields
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"integer table line {line_number} is not integral") from error
        if value < 0 or name in values:
            raise ValueError(f"integer table line {line_number} is invalid")
        values[name] = value
    return values


def parse_start_time(text: str) -> int:
    """Read field 22 from proc stat without treating spaces in comm as fields."""
    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("proc stat has no command terminator")
    suffix = text[closing_parenthesis + 1 :].split()
    if len(suffix) < 20:
        raise ValueError("proc stat is truncated")
    try:
        start_time = int(suffix[19])
    except ValueError as error:
        raise ValueError("proc stat start time is not integral") from error
    if start_time < 0:
        raise ValueError("proc stat start time is negative")
    return start_time


def parse_pinned_storage_log(text: str) -> list[dict[str, object]]:
    """Extract validated DINKSTER_PINNED_STORAGE rows from mixed stderr."""
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("DINKSTER_PINNED_STORAGE"):
            continue
        match = _PINNED_STORAGE.fullmatch(line)
        if match is None:
            raise ValueError(f"pinned-storage line {line_number} is malformed")
        try:
            owners = json.loads(match.group(5))
        except json.JSONDecodeError as error:
            raise ValueError(f"pinned-storage line {line_number} has malformed owners") from error
        if not isinstance(owners, list):
            raise ValueError(f"pinned-storage line {line_number} owners are not a list")
        validated_owners = []
        for owner in owners:
            if (
                not isinstance(owner, dict)
                or set(owner) != {"bytes", "owner"}
                or type(owner["bytes"]) is not int
                or owner["bytes"] < 0
                or not isinstance(owner["owner"], str)
            ):
                raise ValueError(f"pinned-storage line {line_number} has an invalid owner")
            validated_owners.append({"bytes": owner["bytes"], "owner": owner["owner"]})
        total = int(match.group(3))
        if sum(owner["bytes"] for owner in validated_owners) != total:
            raise ValueError(f"pinned-storage line {line_number} owner bytes do not sum to total")
        rows.append(
            {
                "pid": int(match.group(1)),
                "phase": match.group(2),
                "total": total,
                "registered": int(match.group(4)),
                "owners": validated_owners,
                "raw": line,
            }
        )
    return rows


def _field_bytes(mapping: Mapping[str, object], name: str) -> int | None:
    fields = mapping.get("fields")
    if not isinstance(fields, Mapping):
        return None
    field = fields.get(name)
    if not isinstance(field, Mapping):
        return None
    value = field.get("bytes")
    return value if isinstance(value, int) else None


def conservation_check(
    mappings: Sequence[Mapping[str, object]],
    rollup: Sequence[Mapping[str, object]],
    epsilon_bytes: int,
) -> dict[str, object]:
    vma_values = [_field_bytes(mapping, "Anonymous") for mapping in mappings]
    rollup_value = _field_bytes(rollup[0], "Anonymous") if len(rollup) == 1 else None
    if rollup_value is None or any(value is None for value in vma_values):
        return {"status": "unavailable", "epsilon_bytes": epsilon_bytes}
    vma_total = sum(value for value in vma_values if value is not None)
    difference = vma_total - rollup_value
    return {
        "status": "pass" if abs(difference) <= epsilon_bytes else "rejected",
        "epsilon_bytes": epsilon_bytes,
        "smaps_anonymous_bytes": vma_total,
        "rollup_anonymous_bytes": rollup_value,
        "difference_bytes": difference,
    }


def _mapping_summary(mapping: Mapping[str, object]) -> dict[str, object]:
    return {
        "interval_id": mapping["interval_id"],
        "mapping_id": mapping["mapping_id"],
        "normalized_class": mapping["normalized_class"],
    }


def _field_deltas(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, int]:
    before_fields = before.get("fields")
    after_fields = after.get("fields")
    if not isinstance(before_fields, Mapping) or not isinstance(after_fields, Mapping):
        return {}
    names = before_fields.keys() | after_fields.keys()
    deltas = {}
    for name in sorted(str(name) for name in names):
        old = before_fields.get(name)
        new = after_fields.get(name)
        old_bytes = old.get("bytes") if isinstance(old, Mapping) else None
        new_bytes = new.get("bytes") if isinstance(new, Mapping) else None
        if isinstance(old_bytes, int) or isinstance(new_bytes, int):
            deltas[name] = (new_bytes if isinstance(new_bytes, int) else 0) - (
                old_bytes if isinstance(old_bytes, int) else 0
            )
    return deltas


def _covering_mapping(
    mappings: Sequence[Mapping[str, object]], start: int, end: int
) -> Mapping[str, object] | None:
    for mapping in mappings:
        mapping_start = mapping.get("start")
        mapping_end = mapping.get("end")
        if isinstance(mapping_start, int) and isinstance(mapping_end, int):
            if mapping_start <= start and end <= mapping_end:
                return mapping
    return None


def _class_totals(mappings: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for mapping in mappings:
        mapping_class = mapping.get("normalized_class")
        fields = mapping.get("fields")
        if not isinstance(mapping_class, str) or not isinstance(fields, Mapping):
            continue
        class_values = totals.setdefault(mapping_class, {})
        for name, field in fields.items():
            if not isinstance(name, str) or not isinstance(field, Mapping):
                continue
            value = field.get("bytes")
            if isinstance(value, int):
                class_values[name] = class_values.get(name, 0) + value
    return totals


def diff_smaps(
    before: Sequence[Mapping[str, object]], after: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Compare exact intervals and partition address space across split/merge changes."""
    before_by_interval = {str(mapping["interval_id"]): mapping for mapping in before}
    after_by_interval = {str(mapping["interval_id"]): mapping for mapping in after}
    intervals = []
    for interval_id in sorted(before_by_interval.keys() | after_by_interval.keys()):
        old = before_by_interval.get(interval_id)
        new = after_by_interval.get(interval_id)
        state = "added" if old is None else "removed" if new is None else "stable"
        if old is not None and new is not None and old.get("mapping_id") != new.get("mapping_id"):
            state = "reused"
        interval: dict[str, object] = {
            "interval_id": interval_id,
            "state": state,
            "before": _mapping_summary(old) if old is not None else None,
            "after": _mapping_summary(new) if new is not None else None,
        }
        if old is not None and new is not None:
            interval["field_delta_bytes"] = _field_deltas(old, new)
        intervals.append(interval)

    boundaries = sorted(
        {
            value
            for mapping in (*before, *after)
            for key in ("start", "end")
            if isinstance((value := mapping.get(key)), int)
        }
    )
    segments = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        old = _covering_mapping(before, start, end)
        new = _covering_mapping(after, start, end)
        if old is None and new is None:
            continue
        segments.append(
            {
                "segment_id": f"{start:x}-{end:x}",
                "before_interval_id": old.get("interval_id") if old is not None else None,
                "after_interval_id": new.get("interval_id") if new is not None else None,
                "before_mapping_id": old.get("mapping_id") if old is not None else None,
                "after_mapping_id": new.get("mapping_id") if new is not None else None,
            }
        )

    before_totals = _class_totals(before)
    after_totals = _class_totals(after)
    classes: dict[str, dict[str, int]] = {}
    for mapping_class in sorted(before_totals.keys() | after_totals.keys()):
        old_fields = before_totals.get(mapping_class, {})
        new_fields = after_totals.get(mapping_class, {})
        classes[mapping_class] = {
            name: new_fields.get(name, 0) - old_fields.get(name, 0)
            for name in sorted(old_fields.keys() | new_fields.keys())
        }
    return {"intervals": intervals, "segments": segments, "class_field_delta_bytes": classes}


def diff_cgroup(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, object]:
    before_path = before.get("path")
    after_path = after.get("path")
    if not isinstance(before_path, str) or before_path != after_path:
        raise ValueError("cgroup paths do not match")
    difference: dict[str, object] = {
        "before_path": before_path,
        "after_path": after_path,
    }
    old_current = before.get("memory_current")
    new_current = after.get("memory_current")
    if type(old_current) is not int or type(new_current) is not int:
        raise ValueError("cgroup current values are malformed")
    difference["memory_current_delta"] = new_current - old_current
    for field in ("memory_stat", "memory_events"):
        old_values = before.get(field)
        new_values = after.get(field)
        if not isinstance(old_values, Mapping) or not isinstance(new_values, Mapping):
            raise ValueError(f"cgroup {field} values are malformed")
        names = old_values.keys() | new_values.keys()
        deltas = {}
        for name in sorted(str(name) for name in names):
            old_value = old_values.get(name, 0)
            new_value = new_values.get(name, 0)
            if type(old_value) is not int or type(new_value) is not int:
                raise ValueError(f"cgroup {field} value {name} is malformed")
            deltas[name] = new_value - old_value
        difference[f"{field}_delta"] = deltas
    return difference


def diff_pinned_storage(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    difference: dict[str, object] = {}
    for name in ("total", "registered"):
        old_value = before.get(name)
        new_value = after.get(name)
        if type(old_value) is not int or type(new_value) is not int:
            raise ValueError(f"pinned-storage {name} values are malformed")
        difference[f"{name}_delta"] = new_value - old_value
    owner_totals = []
    for row in (before, after):
        owners = row.get("owners")
        if not isinstance(owners, list):
            raise ValueError("pinned-storage owners are malformed")
        totals: dict[str, int] = {}
        for owner in owners:
            if not isinstance(owner, Mapping):
                raise ValueError("pinned-storage owner is malformed")
            label = owner.get("owner")
            size = owner.get("bytes")
            if not isinstance(label, str) or type(size) is not int:
                raise ValueError("pinned-storage owner is malformed")
            totals[label] = totals.get(label, 0) + size
        owner_totals.append(totals)
    difference["owner_delta_bytes"] = {
        owner: owner_totals[1].get(owner, 0) - owner_totals[0].get(owner, 0)
        for owner in sorted(owner_totals[0].keys() | owner_totals[1].keys())
    }
    return difference


def _parse_cgroup_path(text: str) -> str:
    paths = []
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            paths.append(fields[2])
    if len(paths) != 1:
        raise CaptureError("cgroup-v2-path-unavailable")
    parts = [part for part in paths[0].split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise CaptureError("invalid-cgroup-path")
    return "/" + "/".join(parts)


def _read_cgroup_current(path: Path) -> tuple[str, int]:
    raw = read_bounded(path, MAX_CGROUP_FILE_BYTES)
    try:
        current = int(raw.strip())
    except ValueError as error:
        raise CaptureError("invalid-cgroup-current") from error
    if current < 0:
        raise CaptureError("invalid-cgroup-current")
    return raw, current


def _read_cgroup_table(path: Path) -> tuple[str, dict[str, int]]:
    raw = read_bounded(path, MAX_CGROUP_FILE_BYTES)
    try:
        return raw, parse_integer_table(raw)
    except ValueError as error:
        raise CaptureError("invalid-cgroup-table") from error


def _integer_table_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {
        name: after.get(name, 0) - before.get(name, 0)
        for name in sorted(before.keys() | after.keys())
    }


def _memory_stat_stability(
    before: Mapping[str, int], after: Mapping[str, int], epsilon_bytes: int
) -> tuple[str, dict[str, dict[str, object]]]:
    evidence: dict[str, dict[str, object]] = {}
    stable = True
    for name in sorted(before.keys() | after.keys()):
        before_value = before.get(name)
        after_value = after.get(name)
        delta = None if before_value is None or after_value is None else after_value - before_value
        is_byte_gauge = name in _MEMORY_STAT_BYTE_GAUGES
        field_stable = delta is not None and (
            abs(delta) <= epsilon_bytes if is_byte_gauge else delta == 0
        )
        evidence[name] = {
            "after": after_value,
            "before": before_value,
            "delta": delta,
            "stability": "pass" if field_stable else "rejected",
            "stability_rule": "epsilon-bytes" if is_byte_gauge else "exact-count",
        }
        if is_byte_gauge:
            evidence[name]["epsilon_bytes"] = epsilon_bytes
        stable = stable and field_stable
    return ("pass" if stable else "rejected"), evidence


def _capture_cgroup(
    cgroup_membership: str, cgroup_root: Path, epsilon_bytes: int
) -> dict[str, object]:
    cgroup_path = _parse_cgroup_path(cgroup_membership)
    directory = cgroup_root.joinpath(*[part for part in cgroup_path.split("/") if part])
    raw_current_before, current_before = _read_cgroup_current(directory / "memory.current")
    raw_stat_before, stat_before = _read_cgroup_table(directory / "memory.stat")
    raw_events_before, events_before = _read_cgroup_table(directory / "memory.events")
    raw_current_after, current_after = _read_cgroup_current(directory / "memory.current")
    raw_stat_after, stat_after = _read_cgroup_table(directory / "memory.stat")
    raw_events_after, events_after = _read_cgroup_table(directory / "memory.events")
    current_drift = current_after - current_before
    stat_delta = _integer_table_delta(stat_before, stat_after)
    events_delta = _integer_table_delta(events_before, events_after)
    current_stability = "pass" if abs(current_drift) <= epsilon_bytes else "rejected"
    stat_stability, stat_stability_by_field = _memory_stat_stability(
        stat_before, stat_after, epsilon_bytes
    )
    events_stability = "pass" if all(delta == 0 for delta in events_delta.values()) else "rejected"
    stability = (
        "pass" if current_stability == stat_stability == events_stability == "pass" else "rejected"
    )
    return {
        "path": cgroup_path,
        "selected_snapshot": "before",
        "memory_current": current_before,
        "memory_current_before": current_before,
        "memory_current_after": current_after,
        "raw_memory_current_before": raw_current_before,
        "raw_memory_current_after": raw_current_after,
        "memory_current_drift_bytes": current_drift,
        "memory_current_epsilon_bytes": epsilon_bytes,
        "memory_current_stability": current_stability,
        "memory_stat": stat_before,
        "memory_stat_before": stat_before,
        "memory_stat_after": stat_after,
        "raw_memory_stat_before": raw_stat_before,
        "raw_memory_stat_after": raw_stat_after,
        "memory_stat_delta": stat_delta,
        "memory_stat_stability": stat_stability,
        "memory_stat_stability_by_field": stat_stability_by_field,
        "memory_events": events_before,
        "memory_events_before": events_before,
        "memory_events_after": events_after,
        "raw_memory_events_before": raw_events_before,
        "raw_memory_events_after": raw_events_after,
        "memory_events_delta": events_delta,
        "memory_events_stability": events_stability,
        "stability": stability,
    }


def capture_process(
    pid: int,
    *,
    phase: str,
    rank: str | None,
    parent_pid: int | None,
    discovered_start_time_ticks: int | None = None,
    proc_root: Path = Path("/proc"),
    epsilon_bytes: int = DEFAULT_EPSILON_BYTES,
    raw_budget_bytes: int | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "pid": pid,
        "parent_pid": parent_pid,
        "rank": rank,
        "phase": phase,
    }
    if discovered_start_time_ticks is not None:
        metadata["discovered_start_time_ticks"] = discovered_start_time_ticks
    pid_root = proc_root / str(pid)
    try:
        start_time = parse_start_time(read_bounded(pid_root / "stat", MAX_PROC_METADATA_BYTES))
        if discovered_start_time_ticks is not None and start_time != discovered_start_time_ticks:
            raise CaptureError("pid-reused")
        raw_budget = (
            MAX_SMAPS_BYTES + MAX_ROLLUP_BYTES if raw_budget_bytes is None else raw_budget_bytes
        )
        raw_smaps = read_bounded(pid_root / "smaps", min(MAX_SMAPS_BYTES, raw_budget))
        raw_budget -= len(raw_smaps)
        raw_rollup = read_bounded(pid_root / "smaps_rollup", min(MAX_ROLLUP_BYTES, raw_budget))
        mappings = parse_smaps(raw_smaps)
        rollup = parse_smaps(raw_rollup)
        if len(rollup) != 1:
            raise CaptureError("invalid-smaps-rollup")
        cgroup_membership = read_bounded(pid_root / "cgroup", MAX_PROC_METADATA_BYTES)
        cgroup_path = _parse_cgroup_path(cgroup_membership)
        if cgroup_path != _parse_cgroup_path(
            read_bounded(pid_root / "cgroup", MAX_PROC_METADATA_BYTES)
        ):
            raise CaptureError("cgroup-changed")
        if start_time != parse_start_time(read_bounded(pid_root / "stat", MAX_PROC_METADATA_BYTES)):
            raise CaptureError("pid-reused")
    except CaptureError as error:
        return {**metadata, "error": error.code}
    except ValueError:
        return {**metadata, "error": "malformed-proc-data"}
    return {
        **metadata,
        "start_time_ticks": start_time,
        "raw_smaps": raw_smaps,
        "raw_smaps_rollup": raw_rollup,
        "mappings": mappings,
        "rollup": rollup[0],
        "cgroup_path": cgroup_path,
        "conservation": conservation_check(mappings, rollup, epsilon_bytes),
    }


def _process_start_time(pid: int, proc_root: Path) -> int:
    try:
        return parse_start_time(
            read_bounded(proc_root / str(pid) / "stat", MAX_PROC_METADATA_BYTES)
        )
    except ValueError as error:
        raise CaptureError("malformed-proc-data") from error


def _numeric_task_ids(pid: int, proc_root: Path) -> list[int]:
    task_ids = []
    try:
        for task_path in (proc_root / str(pid) / "task").iterdir():
            if not task_path.name.isascii() or not task_path.name.isdecimal():
                continue
            task_ids.append(int(task_path.name))
            if len(task_ids) > MAX_TREE_TASK_COUNT:
                raise CaptureError("task-count-limit-exceeded")
    except CaptureError:
        raise
    except FileNotFoundError as error:
        raise CaptureError("not-found") from error
    except PermissionError as error:
        raise CaptureError("permission-denied") from error
    except OSError as error:
        raise CaptureError("read-failed") from error
    if not task_ids:
        raise CaptureError("process-has-no-tasks")
    return sorted(set(task_ids))


def discover_process_tree(
    root_pid: int, proc_root: Path = Path("/proc")
) -> tuple[list[tuple[int, int | None, int]], list[dict[str, object]]]:
    discovered: list[tuple[int, int | None, int]] = []
    errors: list[dict[str, object]] = []
    root_start_time = _process_start_time(root_pid, proc_root)
    queued: deque[tuple[int, int | None, int]] = deque([(root_pid, None, root_start_time)])
    queued_pids = {root_pid}
    seen: set[int] = set()
    task_count = 0
    children_bytes = 0
    while queued:
        pid, parent_pid, discovered_start_time = queued.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            if _process_start_time(pid, proc_root) != discovered_start_time:
                raise CaptureError("pid-reused")
        except CaptureError as error:
            errors.append({"pid": pid, "error": error.code})
            continue
        discovered.append((pid, parent_pid, discovered_start_time))
        if len(discovered) > MAX_PROCESS_COUNT:
            raise CaptureError("process-count-limit-exceeded")
        try:
            task_ids = _numeric_task_ids(pid, proc_root)
        except CaptureError as error:
            errors.append({"pid": pid, "error": error.code})
            continue
        task_count += len(task_ids)
        if task_count > MAX_TREE_TASK_COUNT:
            raise CaptureError("task-count-limit-exceeded")
        child_pids = set()
        for task_id in task_ids:
            try:
                children = read_bounded(
                    proc_root / str(pid) / "task" / str(task_id) / "children",
                    MAX_PROC_METADATA_BYTES,
                )
            except CaptureError as error:
                errors.append({"pid": pid, "task_id": task_id, "error": error.code})
                continue
            children_bytes += len(children)
            if children_bytes > MAX_TREE_CHILDREN_BYTES:
                raise CaptureError("process-tree-read-limit-exceeded")
            for raw_child in children.split():
                try:
                    child = int(raw_child)
                except ValueError as error:
                    raise CaptureError("invalid-process-tree") from error
                if child <= 0:
                    raise CaptureError("invalid-process-tree")
                child_pids.add(child)
        for child in sorted(child_pids):
            if child in queued_pids:
                continue
            try:
                child_start_time = _process_start_time(child, proc_root)
            except CaptureError as error:
                errors.append({"pid": child, "parent_pid": pid, "error": error.code})
                continue
            queued.append((child, pid, child_start_time))
            queued_pids.add(child)
    return discovered, errors


def capture_receipt(
    root_pid: int,
    *,
    phase: str,
    ranks: Mapping[int, str] | None = None,
    include_descendants: bool = False,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    epsilon_bytes: int = DEFAULT_EPSILON_BYTES,
    pinned_storage_rows: Sequence[Mapping[str, object]] = (),
    pinned_storage_phase: str | None = None,
) -> dict[str, object]:
    if include_descendants:
        try:
            process_ids, tree_errors = discover_process_tree(root_pid, proc_root)
        except CaptureError as error:
            process_ids, tree_errors = [], [{"pid": root_pid, "error": error.code}]
    else:
        process_ids, tree_errors = [(root_pid, None, None)], []
    rank_by_pid = {} if ranks is None else dict(ranks)
    captured_pids = {pid for pid, _parent_pid, _start_time in process_ids}
    tree_errors.extend(
        {"pid": pid, "error": "rank-pid-not-captured"}
        for pid in sorted(rank_by_pid.keys() - captured_pids)
    )
    processes = []
    raw_bytes = 0
    for pid, parent_pid, discovered_start_time in sorted(process_ids):
        process = capture_process(
            pid,
            phase=phase,
            rank=rank_by_pid.get(pid),
            parent_pid=parent_pid,
            discovered_start_time_ticks=discovered_start_time,
            proc_root=proc_root,
            epsilon_bytes=epsilon_bytes,
            raw_budget_bytes=MAX_CAPTURE_RAW_BYTES - raw_bytes,
        )
        raw_bytes += len(str(process.get("raw_smaps", ""))) + len(
            str(process.get("raw_smaps_rollup", ""))
        )
        processes.append(process)

    process_by_pid = {int(process["pid"]): process for process in processes}
    successful = [process for process in processes if "error" not in process]
    root_process = process_by_pid.get(root_pid)
    cgroup: dict[str, object] | None = None
    cgroup_error: str | None = None
    if root_process is None or "error" in root_process:
        cgroup_error = "root-process-unavailable"
    else:
        root_cgroup_path = root_process.get("cgroup_path")
        mismatched = sorted(
            int(process["pid"])
            for process in successful
            if process.get("cgroup_path") != root_cgroup_path
        )
        if mismatched:
            cgroup_error = "process-cgroup-mismatch"
        elif not isinstance(root_cgroup_path, str):
            cgroup_error = "root-cgroup-unavailable"
        else:
            try:
                cgroup = _capture_cgroup(f"0::{root_cgroup_path}\n", cgroup_root, epsilon_bytes)
                if cgroup.get("stability") != "pass":
                    cgroup_error = "cgroup-evidence-drift"
                pid_root = proc_root / str(root_pid)
                if root_process.get("start_time_ticks") != parse_start_time(
                    read_bounded(pid_root / "stat", MAX_PROC_METADATA_BYTES)
                ):
                    raise CaptureError("pid-reused")
                if root_cgroup_path != _parse_cgroup_path(
                    read_bounded(pid_root / "cgroup", MAX_PROC_METADATA_BYTES)
                ):
                    raise CaptureError("cgroup-changed")
            except (CaptureError, ValueError) as error:
                cgroup = None
                cgroup_error = (
                    error.code if isinstance(error, CaptureError) else "malformed-proc-data"
                )

    ledger_errors: list[dict[str, object]] = []
    ledger_by_pid: dict[int, Mapping[str, object]] = {}
    selected_ledger_pids: set[int] = set()
    ledger_phase = phase if pinned_storage_phase is None else pinned_storage_phase
    for row in pinned_storage_rows:
        row_pid = row.get("pid")
        row_phase = row.get("phase")
        if type(row_pid) is not int or not isinstance(row_phase, str):
            ledger_errors.append({"error": "malformed-pinned-storage-row"})
            continue
        if row_phase != ledger_phase:
            continue
        selected_ledger_pids.add(row_pid)
        if row_pid not in process_by_pid:
            ledger_errors.append({"pid": row_pid, "error": "ledger-pid-not-captured"})
        elif "error" in process_by_pid[row_pid]:
            ledger_errors.append({"pid": row_pid, "error": "ledger-pid-identity-unavailable"})
        elif row_pid in ledger_by_pid:
            ledger_errors.append({"pid": row_pid, "error": "duplicate-ledger-row"})
        else:
            ledger_by_pid[row_pid] = row
    for pid in sorted(rank_by_pid):
        if pid in process_by_pid and pid not in selected_ledger_pids:
            ledger_errors.append({"pid": pid, "error": "rank-ledger-row-missing"})
    for pid, row in sorted(ledger_by_pid.items()):
        process_by_pid[pid]["pinned_storage"] = {
            **row,
            "rank": process_by_pid[pid].get("rank"),
            "capture_phase": phase,
        }

    valid = (
        all(
            "error" not in process
            and isinstance(process.get("conservation"), Mapping)
            and process["conservation"].get("status") == "pass"  # type: ignore[union-attr]
            for process in processes
        )
        and not tree_errors
        and cgroup is not None
        and cgroup_error is None
        and not ledger_errors
    )
    return {
        "schema": SCHEMA,
        "root_pid": root_pid,
        "phase": phase,
        "include_descendants": include_descendants,
        "epsilon_bytes": epsilon_bytes,
        "valid": valid,
        "tree_errors": tree_errors,
        "ledger_errors": ledger_errors,
        "pinned_storage_phase": ledger_phase,
        "cgroup": cgroup,
        "cgroup_error": cgroup_error,
        "processes": processes,
    }


def compare_receipts(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    def by_pid(receipt: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
        processes = receipt.get("processes")
        if not isinstance(processes, list):
            raise ValueError("receipt processes are missing")
        result = {}
        for process in processes:
            if not isinstance(process, Mapping) or not isinstance(process.get("pid"), int):
                raise ValueError("receipt process is malformed")
            result[int(process["pid"])] = process
        return result

    old_processes = by_pid(before)
    new_processes = by_pid(after)
    process_differences = []
    for pid in sorted(old_processes.keys() | new_processes.keys()):
        old = old_processes.get(pid)
        new = new_processes.get(pid)
        difference: dict[str, object] = {
            "pid": pid,
            "before_rank": old.get("rank") if old is not None else None,
            "after_rank": new.get("rank") if new is not None else None,
        }
        if old is None or new is None:
            difference["state"] = "added" if old is None else "removed"
        elif "error" in old or "error" in new:
            difference.update(
                state="unavailable", before_error=old.get("error"), after_error=new.get("error")
            )
        else:
            if old.get("start_time_ticks") != new.get("start_time_ticks"):
                difference.update(state="pid-reused")
                process_differences.append(difference)
                continue
            old_mappings = old.get("mappings")
            new_mappings = new.get("mappings")
            old_rollup = old.get("rollup")
            new_rollup = new.get("rollup")
            if (
                not isinstance(old_mappings, list)
                or not isinstance(new_mappings, list)
                or not isinstance(old_rollup, Mapping)
                or not isinstance(new_rollup, Mapping)
            ):
                raise ValueError("receipt process evidence is malformed")
            difference.update(
                state="compared",
                smaps=diff_smaps(old_mappings, new_mappings),
                smaps_rollup_field_delta_bytes=_field_deltas(old_rollup, new_rollup),
            )
            old_pinned = old.get("pinned_storage")
            new_pinned = new.get("pinned_storage")
            if isinstance(old_pinned, Mapping) and isinstance(new_pinned, Mapping):
                difference["pinned_storage"] = diff_pinned_storage(old_pinned, new_pinned)
            elif old.get("rank") is not None or new.get("rank") is not None:
                difference["pinned_storage"] = {"status": "unavailable"}
        process_differences.append(difference)
    old_cgroup = before.get("cgroup")
    new_cgroup = after.get("cgroup")
    if not isinstance(old_cgroup, Mapping) or not isinstance(new_cgroup, Mapping):
        raise ValueError("receipt cgroup evidence is malformed")
    return {
        "schema": SCHEMA,
        "before_phase": before.get("phase"),
        "after_phase": after.get("phase"),
        "cgroup": diff_cgroup(old_cgroup, new_cgroup),
        "processes": process_differences,
    }


def _validate_integer_table(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"comparison receipt {name} is not an object")
    for key, item in value.items():
        if not isinstance(key, str) or type(item) is not int or item < 0:
            raise ValueError(f"comparison receipt {name} is malformed")


def _validate_mapping(value: object) -> tuple[int, int, int, str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("comparison receipt mapping is not an object")
    if (
        type(value.get("start")) is not int
        or type(value.get("end")) is not int
        or value["end"] <= value["start"]
        or not isinstance(value.get("interval_id"), str)
        or not isinstance(value.get("mapping_id"), str)
        or not isinstance(value.get("normalized_class"), str)
    ):
        raise ValueError("comparison receipt mapping identity is malformed")
    permissions = value.get("permissions")
    offset = value.get("offset")
    device = value.get("device")
    inode = value.get("inode")
    pathname = value.get("pathname")
    if (
        not isinstance(permissions, str)
        or re.fullmatch(r"[rwxps-]{4}", permissions) is None
        or type(offset) is not int
        or offset < 0
        or not isinstance(device, str)
        or re.fullmatch(r"[0-9a-f]+:[0-9a-f]+", device) is None
        or type(inode) is not int
        or inode < 0
        or not isinstance(pathname, str)
    ):
        raise ValueError("comparison receipt mapping identity is malformed")
    start = value["start"]
    end = value["end"]
    interval_id = _interval_id(start, end)
    mapping_id = _mapping_id(interval_id, permissions, offset, device, inode, pathname)
    if value["interval_id"] != interval_id or value["mapping_id"] != mapping_id:
        raise ValueError("comparison receipt mapping identity does not match its fields")
    if value["normalized_class"] != normalized_class(pathname):
        raise ValueError("comparison receipt normalized class does not match pathname")
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("comparison receipt mapping fields are not an object")
    for name, field in fields.items():
        if not isinstance(name, str) or not isinstance(field, Mapping):
            raise ValueError("comparison receipt mapping field is malformed")
        if not isinstance(field.get("raw"), str):
            raise ValueError("comparison receipt mapping field raw value is malformed")
        if "bytes" in field and (type(field["bytes"]) is not int or field["bytes"] < 0):
            raise ValueError("comparison receipt mapping field byte value is malformed")
    return len(fields), start, end, interval_id, mapping_id


def validate_comparison_receipt(value: object) -> None:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError("comparison receipt has an unsupported schema")
    if not isinstance(value.get("phase"), str):
        raise ValueError("comparison receipt phase is malformed")
    cgroup = value.get("cgroup")
    if not isinstance(cgroup, Mapping) or not isinstance(cgroup.get("path"), str):
        raise ValueError("comparison receipt cgroup is malformed")
    if type(cgroup.get("memory_current")) is not int or cgroup["memory_current"] < 0:
        raise ValueError("comparison receipt cgroup current is malformed")
    _validate_integer_table(cgroup.get("memory_stat"), "cgroup memory_stat")
    _validate_integer_table(cgroup.get("memory_events"), "cgroup memory_events")

    processes = value.get("processes")
    if not isinstance(processes, list):
        raise ValueError("comparison receipt processes are not an array")
    if len(processes) > MAX_PROCESS_COUNT:
        raise ValueError("comparison receipt process count limit exceeded")
    seen_pids: set[int] = set()
    mapping_count = 0
    field_count = 0
    for process in processes:
        if not isinstance(process, Mapping):
            raise ValueError("comparison receipt process is not an object")
        pid = process.get("pid")
        if type(pid) is not int or pid <= 0:
            raise ValueError("comparison receipt process PID is malformed")
        if pid in seen_pids:
            raise ValueError("comparison receipt contains duplicate PIDs")
        seen_pids.add(pid)
        if process.get("rank") is not None and not isinstance(process.get("rank"), str):
            raise ValueError("comparison receipt process rank is malformed")
        if not isinstance(process.get("phase"), str):
            raise ValueError("comparison receipt process phase is malformed")
        if "error" in process:
            if not isinstance(process["error"], str):
                raise ValueError("comparison receipt process error is malformed")
            continue
        if type(process.get("start_time_ticks")) is not int or process["start_time_ticks"] < 0:
            raise ValueError("comparison receipt process identity is malformed")
        mappings = process.get("mappings")
        rollup = process.get("rollup")
        if not isinstance(mappings, list):
            raise ValueError("comparison receipt process mappings are not an array")
        mapping_count += len(mappings) + 1
        if mapping_count > MAX_SMAPS_MAPPING_COUNT:
            raise ValueError("comparison receipt mapping count limit exceeded")
        seen_intervals: set[str] = set()
        seen_mapping_ids: set[str] = set()
        previous_end: int | None = None
        validated_mappings = sorted(
            (_validate_mapping(mapping) for mapping in mappings), key=lambda item: item[1]
        )
        for mapping_fields, start, end, interval_id, mapping_id in validated_mappings:
            if interval_id in seen_intervals:
                raise ValueError("comparison receipt contains duplicate mapping intervals")
            if mapping_id in seen_mapping_ids:
                raise ValueError("comparison receipt contains duplicate mapping IDs")
            if previous_end is not None and start < previous_end:
                raise ValueError("comparison receipt contains overlapping mapping intervals")
            seen_intervals.add(interval_id)
            seen_mapping_ids.add(mapping_id)
            previous_end = end
            field_count += mapping_fields
            if field_count > MAX_SMAPS_FIELD_COUNT:
                raise ValueError("comparison receipt field count limit exceeded")
        rollup_fields, _, _, _, _ = _validate_mapping(rollup)
        field_count += rollup_fields
        if field_count > MAX_SMAPS_FIELD_COUNT:
            raise ValueError("comparison receipt field count limit exceeded")


def deterministic_json(value: object, maximum_bytes: int | None = None) -> str:
    maximum = MAX_RECEIPT_BYTES if maximum_bytes is None else maximum_bytes
    encoder = json.JSONEncoder(ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    chunks = []
    size = 1
    for chunk in encoder.iterencode(value):
        size += len(chunk)
        if size > maximum:
            raise CaptureError("receipt-output-limit-exceeded")
        chunks.append(chunk)
    return "".join(chunks) + "\n"


def _parse_rank(value: str) -> tuple[int, str]:
    raw_pid, separator, rank = value.partition("=")
    if not separator or not rank:
        raise argparse.ArgumentTypeError("rank must be PID=RANK")
    try:
        pid = int(raw_pid)
    except ValueError as error:
        raise argparse.ArgumentTypeError("rank PID must be an integer") from error
    if pid <= 0:
        raise argparse.ArgumentTypeError("rank PID must be positive")
    return pid, rank


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "SENSITIVE EVIDENCE: receipts contain raw memory mappings and pathnames "
            "plus pinned-storage ledger owner labels. Restrict storage and sharing."
        ),
    )
    parser.add_argument("--pid", type=int, required=True, help="root process ID")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--rank", action="append", default=[], type=_parse_rank, help="PID=RANK")
    parser.add_argument("--include-descendants", action="store_true")
    parser.add_argument("--epsilon-bytes", type=int, default=DEFAULT_EPSILON_BYTES)
    parser.add_argument(
        "--pinned-storage-log",
        type=Path,
        help="bounded stderr log containing DINKSTER_PINNED_STORAGE rows",
    )
    parser.add_argument(
        "--pinned-storage-phase",
        help="ledger invocation phase to join (default: --phase)",
    )
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--output", type=Path, help="default: stdout")
    arguments = parser.parse_args(list(argv))
    if arguments.pid <= 0:
        parser.error("--pid must be positive")
    if arguments.epsilon_bytes < 0:
        parser.error("--epsilon-bytes must be nonnegative")
    if len(dict(arguments.rank)) != len(arguments.rank):
        parser.error("--rank repeats a PID")
    if arguments.pinned_storage_phase is not None and arguments.pinned_storage_log is None:
        parser.error("--pinned-storage-phase requires --pinned-storage-log")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    pinned_storage_rows = []
    if arguments.pinned_storage_log is not None:
        try:
            pinned_storage_rows = parse_pinned_storage_log(
                read_bounded(arguments.pinned_storage_log, MAX_LEDGER_LOG_BYTES)
            )
        except (CaptureError, ValueError) as error:
            raise SystemExit(f"cannot read pinned-storage log: {error}") from error
    receipt = capture_receipt(
        arguments.pid,
        phase=arguments.phase,
        ranks=dict(arguments.rank),
        include_descendants=arguments.include_descendants,
        epsilon_bytes=arguments.epsilon_bytes,
        pinned_storage_rows=pinned_storage_rows,
        pinned_storage_phase=arguments.pinned_storage_phase,
    )
    output: dict[str, object] = receipt
    if arguments.compare_to is not None:
        try:
            previous = json.loads(read_bounded(arguments.compare_to, MAX_RECEIPT_BYTES))
        except (CaptureError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read comparison receipt: {error}") from error
        try:
            validate_comparison_receipt(previous)
        except ValueError as error:
            raise SystemExit(f"comparison receipt is malformed: {error}") from error
        assert isinstance(previous, Mapping)
        output = {**receipt, "difference": compare_receipts(previous, receipt)}
    try:
        serialized = deterministic_json(output)
    except CaptureError as error:
        raise SystemExit(f"cannot serialize receipt: {error.code}") from error
    if arguments.output is None:
        sys.stdout.write(serialized)
    else:
        arguments.output.write_text(serialized, encoding="ascii")
    return 0 if receipt["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
