"""CPU-only contracts for the Linux process-memory diagnostic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "capture_process_memory.py"
_SPEC = importlib.util.spec_from_file_location("capture_process_memory", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
capture_process_memory = importlib.util.module_from_spec(_SPEC)
sys.modules["capture_process_memory"] = capture_process_memory
_SPEC.loader.exec_module(capture_process_memory)


def smaps_entry(
    interval: str,
    *,
    pathname: str = "",
    anonymous_kib: int = 4,
    size_kib: int = 8,
    permissions: str = "rw-p",
    offset: str = "00000000",
    inode: int = 0,
) -> str:
    suffix = f" {pathname}" if pathname else ""
    return (
        f"{interval} {permissions} {offset} 00:00 {inode}{suffix}\n"
        f"Size:                  {size_kib} kB\n"
        f"Rss:                   {anonymous_kib} kB\n"
        f"Pss:                   {anonymous_kib} kB\n"
        f"Anonymous:             {anonymous_kib} kB\n"
        f"Private_Dirty:         {anonymous_kib} kB\n"
        "AnonHugePages:         0 kB\n"
        "Swap:                  0 kB\n"
        "THPeligible:           0\n"
        "VmFlags: rd wr mr mw me ac sd\n"
    )


def write_process(
    root: Path,
    pid: int,
    smaps: str,
    rollup: str,
    *,
    children: str = "",
    cgroup_path: str = "/sealed",
) -> None:
    process = root / "proc" / str(pid)
    task = process / "task" / str(pid)
    task.mkdir(parents=True)
    (task / "children").write_bytes(children.encode("ascii"))
    stat_prefix = " ".join(str(value) for value in range(1, 19))
    (process / "stat").write_bytes(
        f"{pid} (worker with spaces) S {stat_prefix} {pid * 100} 0\n".encode("ascii")
    )
    (process / "smaps").write_bytes(smaps.encode("ascii"))
    (process / "smaps_rollup").write_bytes(rollup.encode("ascii"))
    (process / "cgroup").write_bytes(f"0::{cgroup_path}\n".encode("ascii"))
    cgroup = root / "cgroup" / cgroup_path.lstrip("/")
    cgroup.mkdir(parents=True, exist_ok=True)
    (cgroup / "memory.current").write_bytes(b"8192\n")
    (cgroup / "memory.stat").write_bytes(b"anon 4096\nfile 2048\n")
    (cgroup / "memory.events").write_bytes(b"low 1\noom 0\noom_kill 0\n")


def pinned_storage_log(pid: int, phase: str, first: int = 3072, second: int = 1024) -> str:
    return (
        "unrelated stderr\n"
        f"DINKSTER_PINNED_STORAGE pid={pid} phase={phase} total={first + second} "
        f'registered={first} owners=[{{"bytes":{first},"owner":"weights"}},'
        f'{{"bytes":{second},"owner":"arena"}}]\n'
    )


def comparison_receipt(processes: object) -> dict[str, object]:
    return {
        "schema": capture_process_memory.SCHEMA,
        "phase": "before",
        "cgroup": {
            "path": "/sealed",
            "memory_current": 8192,
            "memory_stat": {"anon": 4096},
            "memory_events": {"oom": 0},
        },
        "processes": processes,
    }


def test_parse_smaps_preserves_asymmetric_fields_and_interval_identity() -> None:
    text = smaps_entry(
        "1000-3000",
        pathname="/models/a file.safetensors (deleted)",
        anonymous_kib=3,
        size_kib=8,
        permissions="r--s",
        offset="00002000",
        inode=17,
    )
    (mapping,) = capture_process_memory.parse_smaps(text)

    assert mapping["start"] == 0x1000
    assert mapping["end"] == 0x3000
    assert mapping["interval_id"] == "1000-3000"
    assert mapping["permissions"] == "r--s"
    assert mapping["offset"] == 0x2000
    assert mapping["inode"] == 17
    assert mapping["pathname"] == "/models/a file.safetensors (deleted)"
    assert mapping["normalized_class"] == "path:/models/a file.safetensors (deleted)"
    assert mapping["fields"]["Anonymous"] == {
        "raw": "3 kB",
        "value": 3,
        "unit": "kB",
        "bytes": 3072,
    }
    assert mapping["fields"]["THPeligible"] == {"raw": "0", "value": 0}
    assert mapping["fields"]["VmFlags"] == {"raw": "rd wr mr mw me ac sd"}


@pytest.mark.parametrize(
    ("pathname", "expected"),
    [
        ("", "anonymous"),
        ("[heap]", "kernel:[heap]"),
        ("[anon:torch]", "kernel:[anon:torch]"),
        ("/usr/lib/libcuda.so", "path:/usr/lib/libcuda.so"),
        ("memfd:arena", "path:memfd:arena"),
    ],
)
def test_normalized_class_keeps_unnamed_kernel_and_path_mappings_separate(
    pathname: str, expected: str
) -> None:
    assert capture_process_memory.normalized_class(pathname) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Anonymous: 1 kB\n",
        "3000-1000 rw-p 00000000 00:00 0\nAnonymous: 1 kB\n",
        "1000-2000 rw-p 00000000 00:00 0\nnot a field\n",
        "1000-2000 rw-p 00000000 00:00 0\nAnonymous: 1 kB\nAnonymous: 2 kB\n",
    ],
)
def test_parse_smaps_rejects_malformed_evidence(text: str) -> None:
    with pytest.raises(ValueError):
        capture_process_memory.parse_smaps(text)


def test_parse_smaps_enforces_mapping_and_field_expansion_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capture_process_memory, "MAX_SMAPS_MAPPING_COUNT", 1)
    with pytest.raises(ValueError, match="mapping limit"):
        capture_process_memory.parse_smaps(smaps_entry("1000-2000") + smaps_entry("2000-3000"))

    monkeypatch.setattr(capture_process_memory, "MAX_SMAPS_MAPPING_COUNT", 10)
    monkeypatch.setattr(capture_process_memory, "MAX_SMAPS_FIELD_COUNT", 1)
    with pytest.raises(ValueError, match="field limit"):
        capture_process_memory.parse_smaps(smaps_entry("1000-2000"))


def test_diff_partitions_split_mapping_and_keeps_exact_class_totals() -> None:
    before = capture_process_memory.parse_smaps(
        smaps_entry("1000-5000", pathname="[heap]", anonymous_kib=12, size_kib=16)
    )
    after = capture_process_memory.parse_smaps(
        smaps_entry("1000-3000", pathname="[heap]", anonymous_kib=5)
        + smaps_entry("3000-5000", pathname="", anonymous_kib=9)
    )

    difference = capture_process_memory.diff_smaps(before, after)

    assert difference["segments"] == [
        {
            "segment_id": "1000-3000",
            "before_interval_id": "1000-5000",
            "after_interval_id": "1000-3000",
            "before_mapping_id": before[0]["mapping_id"],
            "after_mapping_id": after[0]["mapping_id"],
        },
        {
            "segment_id": "3000-5000",
            "before_interval_id": "1000-5000",
            "after_interval_id": "3000-5000",
            "before_mapping_id": before[0]["mapping_id"],
            "after_mapping_id": after[1]["mapping_id"],
        },
    ]
    assert difference["class_field_delta_bytes"]["kernel:[heap]"]["Anonymous"] == -7 * 1024
    assert difference["class_field_delta_bytes"]["anonymous"]["Anonymous"] == 9 * 1024


def test_diff_marks_same_interval_reused_by_a_different_mapping() -> None:
    before = capture_process_memory.parse_smaps(
        smaps_entry("1000-2000", pathname="/old", permissions="r--p", inode=1)
    )
    after = capture_process_memory.parse_smaps(
        smaps_entry("1000-2000", pathname="/new", permissions="rw-p", inode=2)
    )

    (interval,) = capture_process_memory.diff_smaps(before, after)["intervals"]

    assert interval["interval_id"] == "1000-2000"
    assert interval["state"] == "reused"
    assert interval["before"]["mapping_id"] != interval["after"]["mapping_id"]
    assert interval["field_delta_bytes"]["Anonymous"] == 0


def test_integer_table_parses_cgroup_values_and_rejects_ambiguous_rows() -> None:
    assert capture_process_memory.parse_integer_table("anon 12\nfile 3\n") == {
        "anon": 12,
        "file": 3,
    }
    for malformed in ("anon\n", "anon x\n", "anon -1\n", "anon 1\nanon 2\n"):
        with pytest.raises(ValueError):
            capture_process_memory.parse_integer_table(malformed)


def test_cgroup_difference_rejects_unrelated_cgroups() -> None:
    before = {
        "path": "/sealed-a",
        "memory_current": 10,
        "memory_stat": {"anon": 8},
        "memory_events": {"oom": 0},
    }
    after = {
        "path": "/sealed-b",
        "memory_current": 20,
        "memory_stat": {"anon": 16},
        "memory_events": {"oom": 0},
    }

    with pytest.raises(ValueError, match="paths do not match"):
        capture_process_memory.diff_cgroup(before, after)


def test_proc_stat_parser_uses_start_time_after_command_with_spaces() -> None:
    assert (
        capture_process_memory.parse_start_time(
            "42 (worker (rank 0)) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 98765 20\n"
        )
        == 98765
    )


def test_pinned_storage_parser_validates_owner_conservation_and_ignores_other_stderr() -> None:
    assert capture_process_memory.parse_pinned_storage_log(
        pinned_storage_log(30, "post-eviction")
    ) == [
        {
            "pid": 30,
            "phase": "post-eviction",
            "total": 4096,
            "registered": 3072,
            "owners": [
                {"bytes": 3072, "owner": "weights"},
                {"bytes": 1024, "owner": "arena"},
            ],
            "raw": pinned_storage_log(30, "post-eviction").splitlines()[1],
        }
    ]
    with pytest.raises(ValueError, match="do not sum"):
        capture_process_memory.parse_pinned_storage_log(
            "DINKSTER_PINNED_STORAGE pid=30 phase=x total=5 registered=0 "
            'owners=[{"bytes":4,"owner":"x"}]\n'
        )
    with pytest.raises(ValueError, match="malformed"):
        capture_process_memory.parse_pinned_storage_log("DINKSTER_PINNED_STORAGE broken\n")


def test_capture_process_tree_preserves_omitted_rank_and_captures_cgroup_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_smaps = smaps_entry("1000-3000", pathname="[heap]", anonymous_kib=4)
    root_rollup = smaps_entry("1000-3000", pathname="[rollup]", anonymous_kib=4)
    child_smaps = smaps_entry("4000-6000", anonymous_kib=2)
    child_rollup = smaps_entry("4000-6000", pathname="[rollup]", anonymous_kib=2)
    write_process(tmp_path, 20, root_smaps, root_rollup, children="31 30")
    write_process(tmp_path, 30, child_smaps, child_rollup)
    write_process(tmp_path, 31, child_smaps, child_rollup)
    cgroup_reads: list[str] = []
    read_bounded = capture_process_memory.read_bounded

    def track_cgroup_reads(path: Path, maximum: int) -> str:
        if path.name in {"memory.current", "memory.stat", "memory.events"}:
            cgroup_reads.append(path.name)
        return read_bounded(path, maximum)

    monkeypatch.setattr(capture_process_memory, "read_bounded", track_cgroup_reads)

    receipt = capture_process_memory.capture_receipt(
        20,
        phase="post-eviction",
        ranks={30: "rank-0"},
        include_descendants=True,
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
        pinned_storage_rows=capture_process_memory.parse_pinned_storage_log(
            pinned_storage_log(30, "post-eviction")
        ),
    )

    assert receipt["valid"] is True
    assert receipt["tree_errors"] == []
    assert [process["pid"] for process in receipt["processes"]] == [20, 30, 31]
    assert [process["rank"] for process in receipt["processes"]] == [None, "rank-0", None]
    assert [process["parent_pid"] for process in receipt["processes"]] == [None, 20, 20]
    assert receipt["cgroup"] == {
        "path": "/sealed",
        "selected_snapshot": "before",
        "memory_current": 8192,
        "memory_current_before": 8192,
        "memory_current_after": 8192,
        "raw_memory_current_before": "8192\n",
        "raw_memory_current_after": "8192\n",
        "memory_current_drift_bytes": 0,
        "memory_current_epsilon_bytes": 0,
        "memory_current_stability": "pass",
        "memory_stat": {"anon": 4096, "file": 2048},
        "memory_stat_before": {"anon": 4096, "file": 2048},
        "memory_stat_after": {"anon": 4096, "file": 2048},
        "raw_memory_stat_before": "anon 4096\nfile 2048\n",
        "raw_memory_stat_after": "anon 4096\nfile 2048\n",
        "memory_stat_delta": {"anon": 0, "file": 0},
        "memory_stat_stability": "pass",
        "memory_stat_stability_by_field": {
            "anon": {
                "after": 4096,
                "before": 4096,
                "delta": 0,
                "epsilon_bytes": 0,
                "stability": "pass",
                "stability_rule": "epsilon-bytes",
            },
            "file": {
                "after": 2048,
                "before": 2048,
                "delta": 0,
                "epsilon_bytes": 0,
                "stability": "pass",
                "stability_rule": "epsilon-bytes",
            },
        },
        "memory_events": {"low": 1, "oom": 0, "oom_kill": 0},
        "memory_events_before": {"low": 1, "oom": 0, "oom_kill": 0},
        "memory_events_after": {"low": 1, "oom": 0, "oom_kill": 0},
        "raw_memory_events_before": "low 1\noom 0\noom_kill 0\n",
        "raw_memory_events_after": "low 1\noom 0\noom_kill 0\n",
        "memory_events_delta": {"low": 0, "oom": 0, "oom_kill": 0},
        "memory_events_stability": "pass",
        "stability": "pass",
    }
    assert all("cgroup" not in process for process in receipt["processes"])
    assert receipt["processes"][1]["pinned_storage"]["rank"] == "rank-0"
    assert receipt["processes"][1]["pinned_storage"]["phase"] == "post-eviction"
    assert receipt["processes"][1]["pinned_storage"]["capture_phase"] == "post-eviction"
    assert receipt["processes"][0]["raw_smaps"] == root_smaps
    assert receipt["processes"][0]["raw_smaps_rollup"] == root_rollup
    assert sorted(cgroup_reads) == [
        "memory.current",
        "memory.current",
        "memory.events",
        "memory.events",
        "memory.stat",
        "memory.stat",
    ]


def test_process_tree_includes_child_owned_by_nonleader_task(tmp_path: Path) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 20, smaps, rollup, children="30")
    write_process(tmp_path, 30, smaps, rollup)
    write_process(tmp_path, 31, smaps, rollup)
    nonleader_task = tmp_path / "proc" / "20" / "task" / "21"
    nonleader_task.mkdir()
    (nonleader_task / "children").write_text("31 30", encoding="ascii")

    processes, errors = capture_process_memory.discover_process_tree(20, tmp_path / "proc")

    assert errors == []
    assert processes == [(20, None, 2000), (30, 20, 3000), (31, 20, 3100)]


def test_capture_rejects_replacement_child_before_identity_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 20, smaps, rollup, children="31")
    write_process(tmp_path, 31, smaps, rollup)
    child_stat = tmp_path / "proc" / "31" / "stat"
    child_stat_reads = 0
    read_bounded = capture_process_memory.read_bounded

    def replace_child_before_capture(path: Path, maximum: int) -> str:
        nonlocal child_stat_reads
        if path == child_stat:
            child_stat_reads += 1
            if child_stat_reads == 3:
                child_stat.write_text(
                    child_stat.read_text(encoding="ascii").replace("3100 0", "9900 0"),
                    encoding="ascii",
                )
        return read_bounded(path, maximum)

    monkeypatch.setattr(capture_process_memory, "read_bounded", replace_child_before_capture)

    receipt = capture_process_memory.capture_receipt(
        20,
        phase="replacement",
        ranks={31: "rank-0"},
        include_descendants=True,
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
        pinned_storage_rows=capture_process_memory.parse_pinned_storage_log(
            pinned_storage_log(31, "replacement")
        ),
    )

    child = next(process for process in receipt["processes"] if process["pid"] == 31)
    assert receipt["valid"] is False
    assert child["discovered_start_time_ticks"] == 3100
    assert child["error"] == "pid-reused"
    assert "pinned_storage" not in child
    assert receipt["ledger_errors"] == [{"pid": 31, "error": "ledger-pid-identity-unavailable"}]


def test_capture_records_vanished_and_malformed_process_without_os_error_text(
    tmp_path: Path,
) -> None:
    vanished = capture_process_memory.capture_receipt(
        404,
        phase="missing",
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
    )
    assert vanished["valid"] is False
    assert vanished["processes"] == [
        {"pid": 404, "parent_pid": None, "rank": None, "phase": "missing", "error": "not-found"}
    ]
    assert vanished["tree_errors"] == []

    write_process(
        tmp_path,
        12,
        "broken\n",
        smaps_entry("1000-2000", pathname="[rollup]"),
    )
    malformed = capture_process_memory.capture_receipt(
        12,
        phase="broken",
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
    )
    assert malformed["valid"] is False
    assert malformed["processes"][0]["error"] == "malformed-proc-data"


def test_capture_rejects_rank_for_process_outside_captured_tree(tmp_path: Path) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup)

    receipt = capture_process_memory.capture_receipt(
        12,
        phase="rank-check",
        ranks={13: "rank-1"},
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
    )

    assert receipt["valid"] is False
    assert receipt["tree_errors"] == [{"pid": 13, "error": "rank-pid-not-captured"}]


def test_capture_requires_one_matching_ledger_row_for_each_rank(tmp_path: Path) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup)

    missing = capture_process_memory.capture_receipt(
        12,
        phase="timed-2",
        ranks={12: "rank-0"},
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
        pinned_storage_rows=capture_process_memory.parse_pinned_storage_log(
            pinned_storage_log(12, "different-phase")
        ),
    )
    duplicate_rows = capture_process_memory.parse_pinned_storage_log(
        pinned_storage_log(12, "timed-2") + pinned_storage_log(12, "timed-2")
    )
    duplicate = capture_process_memory.capture_receipt(
        12,
        phase="timed-2",
        ranks={12: "rank-0"},
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
        pinned_storage_rows=duplicate_rows,
    )

    assert missing["valid"] is False
    assert missing["ledger_errors"] == [{"pid": 12, "error": "rank-ledger-row-missing"}]
    assert duplicate["valid"] is False
    assert duplicate["ledger_errors"] == [{"pid": 12, "error": "duplicate-ledger-row"}]


def test_capture_joins_explicit_invocation_phase_without_relabeling_capture(tmp_path: Path) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup)

    receipt = capture_process_memory.capture_receipt(
        12,
        phase="timed-2-post-eviction",
        ranks={12: "rank-0"},
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
        pinned_storage_rows=capture_process_memory.parse_pinned_storage_log(
            pinned_storage_log(12, "invocation-2")
        ),
        pinned_storage_phase="invocation-2",
    )

    assert receipt["valid"] is True
    assert receipt["pinned_storage_phase"] == "invocation-2"
    assert receipt["processes"][0]["pinned_storage"]["phase"] == "invocation-2"
    assert receipt["processes"][0]["pinned_storage"]["capture_phase"] == ("timed-2-post-eviction")


def test_capture_rejects_processes_in_different_cgroups(tmp_path: Path) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup, children="13")
    write_process(tmp_path, 13, smaps, rollup, cgroup_path="/other")

    receipt = capture_process_memory.capture_receipt(
        12,
        phase="mismatch",
        include_descendants=True,
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
    )

    assert receipt["valid"] is False
    assert receipt["cgroup"] is None
    assert receipt["cgroup_error"] == "process-cgroup-mismatch"


def test_capture_preserves_cgroup_current_bracket_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup)
    current_path = tmp_path / "cgroup" / "sealed" / "memory.current"
    current_reads = iter(("8192\n", "9217\n"))
    read_bounded = capture_process_memory.read_bounded

    def changing_current(path: Path, maximum: int) -> str:
        if path == current_path:
            return next(current_reads)
        return read_bounded(path, maximum)

    monkeypatch.setattr(capture_process_memory, "read_bounded", changing_current)

    receipt = capture_process_memory.capture_receipt(
        12,
        phase="drift",
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=1024,
    )

    assert receipt["valid"] is False
    assert receipt["cgroup_error"] == "cgroup-evidence-drift"
    assert receipt["cgroup"]["raw_memory_current_before"] == "8192\n"
    assert receipt["cgroup"]["raw_memory_current_after"] == "9217\n"
    assert receipt["cgroup"]["memory_current_before"] == 8192
    assert receipt["cgroup"]["memory_current_after"] == 9217
    assert receipt["cgroup"]["memory_current_drift_bytes"] == 1025
    assert receipt["cgroup"]["memory_current_stability"] == "rejected"
    assert receipt["cgroup"]["stability"] == "rejected"


def test_capture_rejects_intervening_cgroup_stat_and_event_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup)
    cgroup = tmp_path / "cgroup" / "sealed"
    stat_reads = iter(("anon 4096\nfile 2048\n", "anon 1073741824\nfile 2048\n"))
    event_reads = iter(("low 1\noom 0\noom_kill 0\n", "low 1\noom 7\noom_kill 0\n"))
    read_bounded = capture_process_memory.read_bounded

    def changing_tables(path: Path, maximum: int) -> str:
        if path == cgroup / "memory.stat":
            return next(stat_reads)
        if path == cgroup / "memory.events":
            return next(event_reads)
        return read_bounded(path, maximum)

    monkeypatch.setattr(capture_process_memory, "read_bounded", changing_tables)

    receipt = capture_process_memory.capture_receipt(
        12,
        phase="table-drift",
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=1024,
    )

    assert receipt["valid"] is False
    assert receipt["cgroup_error"] == "cgroup-evidence-drift"
    assert receipt["cgroup"]["selected_snapshot"] == "before"
    assert receipt["cgroup"]["memory_stat"] == {"anon": 4096, "file": 2048}
    assert receipt["cgroup"]["memory_stat_after"] == {"anon": 1073741824, "file": 2048}
    assert receipt["cgroup"]["raw_memory_stat_after"] == "anon 1073741824\nfile 2048\n"
    assert receipt["cgroup"]["memory_stat_delta"]["anon"] == 1073737728
    assert receipt["cgroup"]["memory_stat_stability"] == "rejected"
    assert receipt["cgroup"]["memory_events"] == {"low": 1, "oom": 0, "oom_kill": 0}
    assert receipt["cgroup"]["memory_events_after"]["oom"] == 7
    assert receipt["cgroup"]["raw_memory_events_after"] == "low 1\noom 7\noom_kill 0\n"
    assert receipt["cgroup"]["memory_events_delta"]["oom"] == 7
    assert receipt["cgroup"]["memory_events_stability"] == "rejected"
    assert receipt["cgroup"]["stability"] == "rejected"


def test_capture_requires_exact_stability_for_memory_stat_page_fault_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup)
    stat_path = tmp_path / "cgroup" / "sealed" / "memory.stat"
    stat_reads = iter(("anon 4096\npgfault 1\n", "anon 4097\npgfault 1000001\n"))
    read_bounded = capture_process_memory.read_bounded

    def changing_stat(path: Path, maximum: int) -> str:
        if path == stat_path:
            return next(stat_reads)
        return read_bounded(path, maximum)

    monkeypatch.setattr(capture_process_memory, "read_bounded", changing_stat)

    receipt = capture_process_memory.capture_receipt(
        12,
        phase="counter-drift",
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=2_000_000,
    )

    assert receipt["valid"] is False
    assert receipt["cgroup_error"] == "cgroup-evidence-drift"
    assert receipt["cgroup"]["memory_stat_stability"] == "rejected"
    assert receipt["cgroup"]["memory_stat_stability_by_field"] == {
        "anon": {
            "after": 4097,
            "before": 4096,
            "delta": 1,
            "epsilon_bytes": 2_000_000,
            "stability": "pass",
            "stability_rule": "epsilon-bytes",
        },
        "pgfault": {
            "after": 1_000_001,
            "before": 1,
            "delta": 1_000_000,
            "stability": "rejected",
            "stability_rule": "exact-count",
        },
    }


@pytest.mark.parametrize("field_name", ("zswap_incomp", "hugetlb"))
def test_memory_stat_current_byte_gauges_use_epsilon(field_name: str) -> None:
    stability, evidence = capture_process_memory._memory_stat_stability(
        {field_name: 1000}, {field_name: 1100}, 100
    )

    assert stability == "pass"
    assert evidence[field_name] == {
        "after": 1100,
        "before": 1000,
        "delta": 100,
        "epsilon_bytes": 100,
        "stability": "pass",
        "stability_rule": "epsilon-bytes",
    }


def test_memory_stat_unknown_field_remains_exact() -> None:
    stability, evidence = capture_process_memory._memory_stat_stability(
        {"future_counter": 1}, {"future_counter": 2}, 1024
    )

    assert stability == "rejected"
    assert evidence["future_counter"] == {
        "after": 2,
        "before": 1,
        "delta": 1,
        "stability": "rejected",
        "stability_rule": "exact-count",
    }


def test_conservation_accepts_boundary_and_rejects_noise_beyond_it() -> None:
    mappings = capture_process_memory.parse_smaps(
        smaps_entry("1000-2000", anonymous_kib=3) + smaps_entry("2000-3000", anonymous_kib=5)
    )
    close_rollup = capture_process_memory.parse_smaps(
        smaps_entry("1000-3000", pathname="[rollup]", anonymous_kib=7)
    )
    far_rollup = capture_process_memory.parse_smaps(
        smaps_entry("1000-3000", pathname="[rollup]", anonymous_kib=6)
    )

    accepted = capture_process_memory.conservation_check(mappings, close_rollup, 1024)
    rejected = capture_process_memory.conservation_check(mappings, far_rollup, 1024)

    assert accepted["status"] == "pass"
    assert accepted["difference_bytes"] == 1024
    assert rejected["status"] == "rejected"
    assert rejected["difference_bytes"] == 2048


def test_conservation_is_unavailable_when_anonymous_field_is_missing() -> None:
    mapping = capture_process_memory.parse_smaps("1000-2000 rw-p 00000000 00:00 0\nSize: 4 kB\n")
    rollup = capture_process_memory.parse_smaps(smaps_entry("1000-2000", pathname="[rollup]"))
    assert capture_process_memory.conservation_check(mapping, rollup, 0) == {
        "status": "unavailable",
        "epsilon_bytes": 0,
    }


def test_receipt_json_is_stable_ascii_and_does_not_capture_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smaps = smaps_entry("1000-2000", pathname="/tmp/model", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 42, smaps, rollup)
    monkeypatch.setenv("SECRET_DIAGNOSTIC_TOKEN", "must-not-appear")
    receipt = capture_process_memory.capture_receipt(
        42,
        phase="timed-2",
        ranks={42: "rank-\u03b1"},
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
        pinned_storage_rows=capture_process_memory.parse_pinned_storage_log(
            pinned_storage_log(42, "timed-2")
        ),
    )

    first = capture_process_memory.deterministic_json(receipt)
    second = capture_process_memory.deterministic_json(receipt)

    assert first == second
    assert first.endswith("\n")
    assert "must-not-appear" not in first
    assert "SECRET_DIAGNOSTIC_TOKEN" not in first
    assert "\\u03b1" in first
    assert json.loads(first) == receipt


def test_compare_receipts_reports_process_and_split_interval_changes(tmp_path: Path) -> None:
    before_smaps = smaps_entry("1000-3000", pathname="[heap]", anonymous_kib=4)
    after_smaps = smaps_entry("1000-2000", pathname="[heap]", anonymous_kib=2) + smaps_entry(
        "2000-3000", anonymous_kib=3
    )
    rollup = smaps_entry("1000-3000", pathname="[rollup]", anonymous_kib=5)
    write_process(
        tmp_path, 50, before_smaps, smaps_entry("1000-3000", pathname="[rollup]", anonymous_kib=4)
    )
    before = capture_process_memory.capture_receipt(
        50,
        phase="before",
        ranks={50: "rank-0"},
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
        pinned_storage_rows=capture_process_memory.parse_pinned_storage_log(
            pinned_storage_log(50, "before")
        ),
    )
    (tmp_path / "proc" / "50" / "smaps").write_text(after_smaps, encoding="ascii")
    (tmp_path / "proc" / "50" / "smaps_rollup").write_text(rollup, encoding="ascii")
    cgroup = tmp_path / "cgroup" / "sealed"
    (cgroup / "memory.current").write_text("12288\n", encoding="ascii")
    (cgroup / "memory.stat").write_text("anon 7168\nfile 1024\n", encoding="ascii")
    (cgroup / "memory.events").write_text("low 2\noom 0\noom_kill 0\n", encoding="ascii")
    after = capture_process_memory.capture_receipt(
        50,
        phase="after",
        ranks={50: "rank-0"},
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
        pinned_storage_rows=capture_process_memory.parse_pinned_storage_log(
            pinned_storage_log(50, "after", first=2048, second=0)
        ),
    )

    difference = capture_process_memory.compare_receipts(before, after)

    assert difference["before_phase"] == "before"
    assert difference["after_phase"] == "after"
    assert difference["processes"][0]["state"] == "compared"
    assert difference["processes"][0]["before_rank"] == "rank-0"
    assert difference["processes"][0]["after_rank"] == "rank-0"
    assert difference["processes"][0]["smaps_rollup_field_delta_bytes"]["Anonymous"] == 1024
    assert difference["processes"][0]["pinned_storage"] == {
        "total_delta": -2048,
        "registered_delta": -1024,
        "owner_delta_bytes": {"arena": -1024, "weights": -1024},
    }
    assert difference["cgroup"] == {
        "before_path": "/sealed",
        "after_path": "/sealed",
        "memory_current_delta": 4096,
        "memory_stat_delta": {"anon": 3072, "file": -1024},
        "memory_events_delta": {"low": 1, "oom": 0, "oom_kill": 0},
    }
    assert [
        segment["segment_id"] for segment in difference["processes"][0]["smaps"]["segments"]
    ] == [
        "1000-2000",
        "2000-3000",
    ]


def test_compare_receipts_refuses_reused_pid_identity(tmp_path: Path) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 50, smaps, rollup)
    before = capture_process_memory.capture_receipt(
        50,
        phase="before",
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
    )
    stat = tmp_path / "proc" / "50" / "stat"
    stat.write_text(stat.read_text(encoding="ascii").replace("5000 0", "6000 0"), encoding="ascii")
    after = capture_process_memory.capture_receipt(
        50,
        phase="after",
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
    )

    difference = capture_process_memory.compare_receipts(before, after)

    assert difference["processes"] == [
        {"pid": 50, "before_rank": None, "after_rank": None, "state": "pid-reused"}
    ]


def test_read_bounded_rejects_oversized_and_non_ascii_inputs(tmp_path: Path) -> None:
    path = tmp_path / "evidence"
    path.write_bytes(b"abcd")
    with pytest.raises(capture_process_memory.CaptureError, match="read-limit-exceeded"):
        capture_process_memory.read_bounded(path, 3)
    path.write_bytes(b"\xff")
    with pytest.raises(capture_process_memory.CaptureError, match="non-ascii-input"):
        capture_process_memory.read_bounded(path, 3)


def test_main_rejects_oversized_smaps_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup)
    capture_receipt = capture_process_memory.capture_receipt

    def capture_fixture(pid: int, **kwargs: object) -> dict[str, object]:
        return capture_receipt(
            pid,
            **kwargs,
            proc_root=tmp_path / "proc",
            cgroup_root=tmp_path / "cgroup",
        )

    monkeypatch.setattr(capture_process_memory, "capture_receipt", capture_fixture)
    monkeypatch.setattr(capture_process_memory, "MAX_SMAPS_BYTES", len(smaps) - 1)

    assert capture_process_memory.main(["--pid", "12", "--phase", "oversized"]) == 2
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["processes"][0]["error"] == "read-limit-exceeded"
    assert "raw_smaps" not in receipt["processes"][0]


def test_main_enforces_aggregate_raw_capture_budget_before_second_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup, children="13")
    write_process(tmp_path, 13, smaps, rollup)
    capture_receipt = capture_process_memory.capture_receipt

    def capture_fixture(pid: int, **kwargs: object) -> dict[str, object]:
        return capture_receipt(
            pid,
            **kwargs,
            proc_root=tmp_path / "proc",
            cgroup_root=tmp_path / "cgroup",
        )

    monkeypatch.setattr(capture_process_memory, "capture_receipt", capture_fixture)
    monkeypatch.setattr(
        capture_process_memory,
        "MAX_CAPTURE_RAW_BYTES",
        len(smaps) + len(rollup) + len(smaps) - 1,
    )

    assert (
        capture_process_memory.main(
            ["--pid", "12", "--phase", "aggregate", "--include-descendants"]
        )
        == 2
    )
    receipt = json.loads(capsys.readouterr().out)
    assert "error" not in receipt["processes"][0]
    assert receipt["processes"][1]["error"] == "read-limit-exceeded"
    assert "raw_smaps" not in receipt["processes"][1]


def test_main_rejects_oversized_receipt_input_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture_process_memory, "MAX_RECEIPT_BYTES", 128)
    monkeypatch.setattr(
        capture_process_memory,
        "capture_receipt",
        lambda *_args, **_kwargs: {
            "schema": capture_process_memory.SCHEMA,
            "valid": True,
            "x": "y" * 128,
        },
    )
    with pytest.raises(SystemExit, match="receipt-output-limit-exceeded"):
        capture_process_memory.main(["--pid", "12", "--phase", "output"])

    oversized = tmp_path / "oversized.json"
    oversized.write_text("{" + " " * 128 + "}", encoding="ascii")
    monkeypatch.setattr(
        capture_process_memory,
        "capture_receipt",
        lambda *_args, **_kwargs: comparison_receipt([]) | {"valid": True},
    )
    with pytest.raises(SystemExit, match="read-limit-exceeded"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "input", "--compare-to", str(oversized)]
        )


def test_main_rejects_257_process_comparison_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            comparison_receipt(
                [
                    {"pid": pid, "phase": "before", "rank": None, "error": "unavailable"}
                    for pid in range(1, 258)
                ]
            )
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(
        capture_process_memory,
        "capture_receipt",
        lambda *_args, **_kwargs: comparison_receipt([]) | {"valid": True},
    )

    with pytest.raises(SystemExit, match="process count limit exceeded"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


def test_main_rejects_duplicate_comparison_pids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            comparison_receipt(
                [
                    {"pid": 12, "phase": "before", "rank": None, "error": "first"},
                    {"pid": 12, "phase": "before", "rank": None, "error": "second"},
                ]
            )
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(
        capture_process_memory,
        "capture_receipt",
        lambda *_args, **_kwargs: comparison_receipt([]) | {"valid": True},
    )

    with pytest.raises(SystemExit, match="duplicate PIDs"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


def test_main_rejects_comparison_mapping_and_aggregate_field_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = capture_process_memory.parse_smaps(smaps_entry("1000-2000"))[0]
    process = {
        "pid": 12,
        "phase": "before",
        "rank": None,
        "start_time_ticks": 1200,
        "mappings": [mapping, mapping],
        "rollup": mapping,
    }
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(comparison_receipt([process])), encoding="ascii")
    monkeypatch.setattr(
        capture_process_memory,
        "capture_receipt",
        lambda *_args, **_kwargs: comparison_receipt([]) | {"valid": True},
    )
    monkeypatch.setattr(capture_process_memory, "MAX_SMAPS_MAPPING_COUNT", 2)

    with pytest.raises(SystemExit, match="mapping count limit exceeded"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )

    monkeypatch.setattr(capture_process_memory, "MAX_SMAPS_MAPPING_COUNT", 10)
    monkeypatch.setattr(capture_process_memory, "MAX_SMAPS_FIELD_COUNT", 1)
    with pytest.raises(SystemExit, match="field count limit exceeded"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


def _comparison_process(mappings: list[dict[str, object]]) -> dict[str, object]:
    rollup = capture_process_memory.parse_smaps(smaps_entry("1000-4000", pathname="[rollup]"))[0]
    return {
        "pid": 12,
        "phase": "before",
        "rank": None,
        "start_time_ticks": 1200,
        "mappings": mappings,
        "rollup": rollup,
    }


def _write_comparison_and_stub_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process: dict[str, object]
) -> Path:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(comparison_receipt([process])), encoding="ascii")
    monkeypatch.setattr(
        capture_process_memory,
        "capture_receipt",
        lambda *_args, **_kwargs: comparison_receipt([]) | {"valid": True},
    )
    return comparison


def test_main_rejects_duplicate_comparison_mapping_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = capture_process_memory.parse_smaps(smaps_entry("1000-2000"))[0]
    comparison = _write_comparison_and_stub_capture(
        tmp_path, monkeypatch, _comparison_process([mapping, mapping])
    )

    with pytest.raises(SystemExit, match="duplicate mapping intervals"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


@pytest.mark.parametrize("identity_name", ("interval_id", "mapping_id"))
def test_main_rejects_comparison_mapping_identity_not_derived_from_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity_name: str
) -> None:
    mapping = capture_process_memory.parse_smaps(smaps_entry("1000-2000"))[0]
    mapping[identity_name] = "bogus"
    comparison = _write_comparison_and_stub_capture(
        tmp_path, monkeypatch, _comparison_process([mapping])
    )

    with pytest.raises(SystemExit, match="mapping identity does not match its fields"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


def test_main_rejects_comparison_normalized_class_not_derived_from_pathname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = capture_process_memory.parse_smaps(
        smaps_entry("1000-2000", pathname="/models/a.safetensors")
    )[0]
    mapping["normalized_class"] = "anonymous"
    comparison = _write_comparison_and_stub_capture(
        tmp_path, monkeypatch, _comparison_process([mapping])
    )

    with pytest.raises(SystemExit, match="normalized class does not match pathname"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


@pytest.mark.parametrize(
    ("field_name", "malformed_value"),
    (
        ("permissions", 1),
        ("permissions", "bad!"),
        ("offset", "0"),
        ("device", 1),
        ("device", "00:GG"),
        ("inode", "0"),
        ("pathname", 1),
    ),
)
def test_main_rejects_malformed_comparison_mapping_constituent_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    malformed_value: object,
) -> None:
    mapping = capture_process_memory.parse_smaps(smaps_entry("1000-2000"))[0]
    mapping[field_name] = malformed_value
    comparison = _write_comparison_and_stub_capture(
        tmp_path, monkeypatch, _comparison_process([mapping])
    )

    with pytest.raises(SystemExit, match="mapping identity is malformed"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


def test_main_rejects_overlapping_comparison_mapping_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings = capture_process_memory.parse_smaps(
        smaps_entry("1000-3000") + smaps_entry("2000-4000")
    )
    comparison = _write_comparison_and_stub_capture(
        tmp_path, monkeypatch, _comparison_process(mappings)
    )

    with pytest.raises(SystemExit, match="overlapping mapping intervals"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


@pytest.mark.parametrize("processes", ({}, [None]))
def test_main_rejects_malformed_comparison_containers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, processes: object
) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(comparison_receipt(processes)), encoding="ascii")
    monkeypatch.setattr(
        capture_process_memory,
        "capture_receipt",
        lambda *_args, **_kwargs: comparison_receipt([]) | {"valid": True},
    )

    with pytest.raises(SystemExit, match="comparison receipt is malformed"):
        capture_process_memory.main(
            ["--pid", "12", "--phase", "after", "--compare-to", str(comparison)]
        )


def test_main_accepts_generated_comparison_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    smaps = smaps_entry("1000-2000", anonymous_kib=1)
    rollup = smaps_entry("1000-2000", pathname="[rollup]", anonymous_kib=1)
    write_process(tmp_path, 12, smaps, rollup)
    capture_receipt = capture_process_memory.capture_receipt
    before = capture_receipt(
        12,
        phase="before",
        proc_root=tmp_path / "proc",
        cgroup_root=tmp_path / "cgroup",
        epsilon_bytes=0,
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(capture_process_memory.deterministic_json(before), encoding="ascii")

    def capture_fixture(pid: int, **kwargs: object) -> dict[str, object]:
        return capture_receipt(
            pid,
            **kwargs,
            proc_root=tmp_path / "proc",
            cgroup_root=tmp_path / "cgroup",
        )

    monkeypatch.setattr(capture_process_memory, "capture_receipt", capture_fixture)

    assert (
        capture_process_memory.main(
            [
                "--pid",
                "12",
                "--phase",
                "after",
                "--epsilon-bytes",
                "0",
                "--compare-to",
                str(comparison),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["difference"]["before_phase"] == "before"
    assert output["difference"]["after_phase"] == "after"


def test_argument_parser_rejects_bad_pid_rank_and_epsilon() -> None:
    for arguments in (
        ["--pid", "0", "--phase", "x"],
        ["--pid", "1", "--phase", "x", "--rank", "bad"],
        ["--pid", "1", "--phase", "x", "--epsilon-bytes", "-1"],
        ["--pid", "1", "--phase", "x", "--rank", "1=a", "--rank", "1=b"],
    ):
        with pytest.raises(SystemExit):
            capture_process_memory.parse_arguments(arguments)


def test_argument_parser_accepts_explicit_pinned_storage_log(tmp_path: Path) -> None:
    log = tmp_path / "worker.stderr"
    arguments = capture_process_memory.parse_arguments(
        [
            "--pid",
            "12",
            "--phase",
            "timed-2-post-eviction",
            "--pinned-storage-log",
            str(log),
            "--pinned-storage-phase",
            "invocation-2",
        ]
    )
    assert arguments.pinned_storage_log == log
    assert arguments.pinned_storage_phase == "invocation-2"
    with pytest.raises(SystemExit):
        capture_process_memory.parse_arguments(
            ["--pid", "12", "--phase", "x", "--pinned-storage-phase", "invocation-2"]
        )


def test_cli_help_discloses_sensitive_receipt_fields(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        capture_process_memory.parse_arguments(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "SENSITIVE EVIDENCE" in help_text
    assert "raw memory mappings and pathnames" in help_text
    assert "pinned-storage ledger owner labels" in help_text
