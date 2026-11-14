from __future__ import annotations

import os
from pathlib import Path

import dinkster_memory.system as system_memory_module
import pytest
from dinkster_memory import system_memory_snapshot

linux_cgroup = pytest.mark.skipif(
    os.name != "posix",
    reason="cgroup filesystem fixtures require POSIX path semantics",
)


def _mountinfo_path(value: str | Path) -> str:
    return (
        str(value)
        .replace("\\", "\\134")
        .replace("\t", "\\011")
        .replace("\n", "\\012")
        .replace(" ", "\\040")
    )


def _write_cgroup2_mountinfo(
    tmp_path: Path,
    mountpoint: Path,
    *,
    root: str = "/",
) -> Path:
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 {_mountinfo_path(root)} {_mountinfo_path(mountpoint)} "
        "rw,nosuid - cgroup2 cgroup rw\n"
    )
    return proc_mountinfo


def _write_cgroup1_memory_mountinfo(
    tmp_path: Path,
    mountpoint: Path,
    *,
    root: str = "/",
    controllers: str = "rw,cpu,memory",
) -> Path:
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 {_mountinfo_path(root)} {_mountinfo_path(mountpoint)} "
        f"rw,nosuid - cgroup cgroup {controllers}\n"
    )
    return proc_mountinfo


@linux_cgroup
def test_system_snapshot_clamps_to_cgroup_v1_and_reports_swap(tmp_path: Path) -> None:
    root = tmp_path / "legacy-memory"
    child = root / "tenant" / "workload"
    child.mkdir(parents=True)
    (child / "memory.limit_in_bytes").write_text("700\n")
    (child / "memory.usage_in_bytes").write_text("200\n")
    (child / "memory.memsw.limit_in_bytes").write_text("1000\n")
    (child / "memory.memsw.usage_in_bytes").write_text("350\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("5:cpu,memory:/tenant/workload\n")
    proc_mountinfo = _write_cgroup1_memory_mountinfo(tmp_path, root)

    def cgroup_query() -> system_memory_module._CgroupMemoryStatus:
        return system_memory_module._linux_cgroup_status(
            proc_cgroup=proc_cgroup,
            proc_mountinfo=proc_mountinfo,
            include_swap=True,
        )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)
    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 900),
        host_swap_query=lambda: (500, 400),
        linux_cgroup_query=cgroup_query,
        include_swap=True,
    )
    assert snapshot.effective_total_bytes == 700
    assert snapshot.effective_available_bytes == 500
    assert snapshot.effective_swap_total_bytes == 300
    assert snapshot.effective_swap_available_bytes == 150
    assert snapshot.provenance == ("injected", "cgroup-v1")


@linux_cgroup
def test_cgroup_v1_uses_hierarchical_limits_and_ignores_huge_unlimited(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy-memory"
    child = root / "workload"
    child.mkdir(parents=True)
    unlimited = system_memory_module._v1_unlimited_value()
    (child / "memory.limit_in_bytes").write_text(f"{unlimited}\n")
    (child / "memory.usage_in_bytes").write_text("100\n")
    (child / "memory.memsw.limit_in_bytes").write_text(f"{unlimited}\n")
    (child / "memory.memsw.usage_in_bytes").write_text("150\n")
    (child / "memory.stat").write_text(
        "cache 10\nhierarchical_memory_limit 700\nhierarchical_memsw_limit 1000\n"
    )
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("4:memory:/workload\n")
    proc_mountinfo = _write_cgroup1_memory_mountinfo(tmp_path, root)

    status = system_memory_module._linux_cgroup_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
        include_swap=True,
    )
    assert status.total_bytes == 700
    assert status.available_bytes is None
    assert status.swap_total_bytes is None
    assert status.swap_available_bytes is None
    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 900),
        host_swap_query=lambda: (500, 400),
        linux_cgroup_query=lambda: status,
        include_swap=True,
    )
    assert snapshot.effective_total_bytes == 700
    assert snapshot.effective_available_bytes == 700
    assert snapshot.effective_swap_total_bytes == 500
    assert snapshot.effective_swap_available_bytes == 400


@linux_cgroup
def test_hybrid_cgroup_uses_v1_memory_controller_membership(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    unified = tmp_path / "unified"
    legacy_child = legacy / "legacy-workload"
    unified_child = unified / "unified-workload"
    legacy_child.mkdir(parents=True)
    unified_child.mkdir(parents=True)
    (legacy_child / "memory.limit_in_bytes").write_text("700\n")
    (legacy_child / "memory.usage_in_bytes").write_text("200\n")
    (unified_child / "memory.max").write_text("300\n")
    (unified_child / "memory.current").write_text("100\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("5:cpu,memory:/legacy-workload\n0::/unified-workload\n")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 / {_mountinfo_path(unified)} rw - cgroup2 cgroup rw\n"
        f"2 0 0:2 / {_mountinfo_path(legacy)} rw - cgroup cgroup rw,cpu,memory\n"
    )

    status = system_memory_module._linux_cgroup_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    )
    assert status.version == 1
    assert (status.total_bytes, status.available_bytes) == (700, 500)


@linux_cgroup
def test_system_snapshot_clamps_to_nested_cgroup_v2_limits(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    parent = root / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (root / "memory.max").write_text("max\n")
    (root / "memory.current").write_text("100\n")
    (parent / "memory.max").write_text("1000\n")
    (parent / "memory.current").write_text("400\n")
    (child / "memory.max").write_text("700\n")
    (child / "memory.current").write_text("200\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/parent/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    def cgroup_query() -> tuple[int | None, int | None]:
        return system_memory_module._linux_cgroup_memory_status(
            proc_cgroup=proc_cgroup,
            proc_mountinfo=proc_mountinfo,
        )

    assert cgroup_query() == (700, 500)
    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 900),
        linux_cgroup_query=cgroup_query,
    )
    assert snapshot.effective_total_bytes == 700
    assert snapshot.effective_available_bytes == 500
    assert snapshot.provenance == ("injected", "cgroup-v2")


@linux_cgroup
def test_system_snapshot_reads_mutable_cgroup_values_on_every_call(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    child = root / "child"
    child.mkdir(parents=True)
    limit = child / "memory.max"
    limit.write_text("1000\n")
    current = child / "memory.current"
    current.write_text("100\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    def cgroup_query() -> tuple[int | None, int | None]:
        return system_memory_module._linux_cgroup_memory_status(
            proc_cgroup=proc_cgroup,
            proc_mountinfo=proc_mountinfo,
        )

    first = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 1500),
        linux_cgroup_query=cgroup_query,
    )
    limit.write_text("600\n")
    current.write_text("800\n")
    second = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 1500),
        linux_cgroup_query=cgroup_query,
    )
    assert first.effective_available_bytes == 900
    assert second.effective_total_bytes == 600
    assert second.effective_available_bytes == 0


@linux_cgroup
def test_system_snapshot_degrades_to_host_when_mountinfo_is_unreadable(tmp_path: Path) -> None:
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/child\n")

    def cgroup_query() -> tuple[int | None, int | None]:
        return system_memory_module._linux_cgroup_memory_status(
            proc_cgroup=proc_cgroup,
            proc_mountinfo=tmp_path / "missing-mountinfo",
        )

    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 900),
        linux_cgroup_query=cgroup_query,
    )
    assert snapshot.effective_total_bytes == 2000
    assert snapshot.effective_available_bytes == 900
    assert snapshot.provenance == ("injected",)


@linux_cgroup
def test_system_snapshot_ignores_unlimited_cgroup_v2(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    child = root / "child"
    child.mkdir(parents=True)
    (root / "memory.max").write_text("max\n")
    (child / "memory.max").write_text("max\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    cgroup = system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    )
    assert cgroup == (None, None)
    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 900),
        linux_cgroup_query=lambda: cgroup,
    )
    assert snapshot.effective_total_bytes == 2000
    assert snapshot.effective_available_bytes == 900
    assert snapshot.provenance == ("injected",)
