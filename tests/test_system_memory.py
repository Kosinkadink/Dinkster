from __future__ import annotations

import os
from pathlib import Path

import dinkster_memory.system as system_memory_module
import pytest
from dinkster_memory import SystemMemorySnapshot, system_memory_snapshot

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


def test_system_snapshot_reports_psutil_source() -> None:
    snapshot = system_memory_snapshot(include_swap=True)
    assert snapshot.host_total_bytes > 0
    assert 0 <= snapshot.host_available_bytes <= snapshot.host_total_bytes
    assert snapshot.host_swap_available_bytes is not None
    assert snapshot.host_swap_total_bytes is not None
    assert 0 <= snapshot.host_swap_available_bytes <= snapshot.host_swap_total_bytes
    assert 0 < snapshot.effective_total_bytes <= snapshot.host_total_bytes
    assert 0 <= snapshot.effective_available_bytes <= snapshot.effective_total_bytes
    assert snapshot.effective_swap_available_bytes is not None
    assert snapshot.effective_swap_total_bytes is not None
    assert 0 <= snapshot.effective_swap_available_bytes <= snapshot.effective_swap_total_bytes
    assert snapshot.provenance[0] == "psutil"
    assert snapshot.effective_used_bytes == (
        snapshot.effective_total_bytes - snapshot.effective_available_bytes
    )


def test_system_snapshot_uses_injected_host_query_without_platform_reads() -> None:
    snapshot = system_memory_snapshot("win32", host_query=lambda: (2000, 900))
    assert snapshot == SystemMemorySnapshot(
        host_total_bytes=2000,
        host_available_bytes=900,
        effective_total_bytes=2000,
        effective_available_bytes=900,
        provenance=("injected",),
    )


def test_linux_non_container_uses_host_available_memory_semantics() -> None:
    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 1400),
        linux_cgroup_query=lambda: (None, None),
    )
    assert snapshot.effective_total_bytes == 2000
    assert snapshot.effective_available_bytes == 1400


def test_system_snapshot_reports_injected_host_swap_without_platform_reads() -> None:
    snapshot = system_memory_snapshot(
        "win32",
        host_query=lambda: (2000, 1400),
        host_swap_query=lambda: (800, 600),
        include_swap=True,
    )
    assert snapshot.host_swap_total_bytes == 800
    assert snapshot.host_swap_available_bytes == 600
    assert snapshot.effective_swap_total_bytes == 800
    assert snapshot.effective_swap_available_bytes == 600


def test_mountinfo_path_escape_decoder_is_single_pass() -> None:
    encoded = r"/tab\011newline\012space\040backslash\134end"
    assert system_memory_module._unescape_mountinfo_path(encoded) == (
        "/tab\tnewline\nspace backslash\\end"
    )
    assert system_memory_module._unescape_mountinfo_path(r"/literal\134040") == r"/literal\040"
    assert str(system_memory_module._absolute_posix_path(r"/literal\040")) == r"/literal\040"


@pytest.mark.parametrize("page_size", [4096, 16384, 65536])
def test_cgroup_v1_unlimited_values_match_kernel_word_sizes(page_size: int) -> None:
    assert (
        system_memory_module._v1_unlimited_value(
            page_size,
            "x86_64",
            (1 << 31) - 1,
        )
        == (((1 << 63) - 1) // page_size) * page_size
    )
    assert (
        system_memory_module._v1_unlimited_value(
            page_size,
            "i686",
            (1 << 31) - 1,
        )
        == ((1 << 31) - 1) * page_size
    )


def test_cgroup_v1_unlimited_detection_does_not_use_python_word_size() -> None:
    assert system_memory_module._is_cgroup_v1_unlimited(-1, 4096)
    assert system_memory_module._is_cgroup_v1_unlimited(9223372036854771712, 4096)
    assert not system_memory_module._is_cgroup_v1_unlimited(8796093018112, 4096)
    assert not system_memory_module._is_cgroup_v1_unlimited(4 * 1024**3, 4096)
    assert system_memory_module._is_cgroup_v1_unlimited(
        8796093018112,
        4096,
        "i686",
        (1 << 31) - 1,
    )


def test_cgroup_v1_32_bit_sentinel_is_finite_on_64_bit_kernel(tmp_path: Path) -> None:
    limit = tmp_path / "memory.limit_in_bytes"
    limit.write_text("8796093018112\n")
    assert system_memory_module._read_cgroup_v1_value(limit) == 8796093018112


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
def test_cgroup_v1_does_not_decompose_independent_hierarchical_memsw_minima(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy-memory"
    child = root / "workload"
    child.mkdir(parents=True)
    (child / "memory.stat").write_text(
        "hierarchical_memory_limit 100\nhierarchical_memsw_limit 950\n"
    )
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("4:memory:/workload\n")
    proc_mountinfo = _write_cgroup1_memory_mountinfo(tmp_path, root)

    status = system_memory_module._linux_cgroup_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
        include_swap=True,
    )
    assert status.total_bytes == 100
    assert status.swap_total_bytes is None
    assert status.swap_available_bytes is None


@linux_cgroup
def test_cgroup_v1_hierarchical_limit_survives_missing_leaf_usage(tmp_path: Path) -> None:
    root = tmp_path / "legacy-memory"
    child = root / "workload"
    child.mkdir(parents=True)
    (child / "memory.limit_in_bytes").write_text("700\n")
    (child / "memory.stat").write_text("hierarchical_memory_limit 600\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("4:memory:/workload\n")
    proc_mountinfo = _write_cgroup1_memory_mountinfo(tmp_path, root)

    status = system_memory_module._linux_cgroup_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    )
    assert status.total_bytes == 600
    assert status.available_bytes is None


@linux_cgroup
def test_cgroup_v1_finite_limits_survive_missing_usage(tmp_path: Path) -> None:
    root = tmp_path / "legacy-memory"
    child = root / "workload"
    child.mkdir(parents=True)
    (child / "memory.limit_in_bytes").write_text("700\n")
    (child / "memory.memsw.limit_in_bytes").write_text("1000\n")
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
    assert status.swap_total_bytes == 300
    assert status.swap_available_bytes is None


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
def test_cgroup_v1_missing_memsw_keeps_host_swap_view(tmp_path: Path) -> None:
    root = tmp_path / "legacy-memory"
    child = root / "workload"
    child.mkdir(parents=True)
    (child / "memory.limit_in_bytes").write_text("700\n")
    (child / "memory.usage_in_bytes").write_text("200\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("4:memory:/workload\n")
    proc_mountinfo = _write_cgroup1_memory_mountinfo(tmp_path, root)

    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 900),
        host_swap_query=lambda: (500, 400),
        linux_cgroup_query=lambda: system_memory_module._linux_cgroup_status(
            proc_cgroup=proc_cgroup,
            proc_mountinfo=proc_mountinfo,
            include_swap=True,
        ),
        include_swap=True,
    )
    assert snapshot.effective_swap_total_bytes == 500
    assert snapshot.effective_swap_available_bytes == 400


@linux_cgroup
def test_cgroup_v1_walks_visible_ancestors_from_remounted_root(tmp_path: Path) -> None:
    mountpoint = tmp_path / "delegated-memory"
    child = mountpoint / "workload"
    child.mkdir(parents=True)
    (mountpoint / "memory.limit_in_bytes").write_text("900\n")
    (mountpoint / "memory.usage_in_bytes").write_text("600\n")
    (child / "memory.limit_in_bytes").write_text("700\n")
    (child / "memory.usage_in_bytes").write_text("200\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("4:memory:/tenant/workload\n")
    proc_mountinfo = _write_cgroup1_memory_mountinfo(
        tmp_path,
        mountpoint,
        root="/tenant",
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 300)


@linux_cgroup
def test_cgroup_v1_prefers_hierarchy_relative_mount_over_namespace_fallback(
    tmp_path: Path,
) -> None:
    correct = tmp_path / "correct"
    correct_child = correct / "workload"
    unrelated = tmp_path / "unrelated"
    unrelated_child = unrelated / "tenant" / "workload"
    correct_child.mkdir(parents=True)
    unrelated_child.mkdir(parents=True)
    (correct_child / "cgroup.procs").write_text("")
    (correct_child / "memory.limit_in_bytes").write_text("700\n")
    (correct_child / "memory.usage_in_bytes").write_text("200\n")
    (unrelated_child / "cgroup.procs").write_text("")
    (unrelated_child / "memory.limit_in_bytes").write_text("2000\n")
    (unrelated_child / "memory.usage_in_bytes").write_text("100\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("4:memory:/tenant/workload\n")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 /tenant {_mountinfo_path(correct)} "
        "rw - cgroup cgroup rw,cpu,memory\n"
        f"2 0 0:2 /unrelated/deeper {_mountinfo_path(unrelated)} "
        "rw - cgroup cgroup rw,cpu,memory\n"
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


@linux_cgroup
def test_cgroup_v1_resolves_namespace_root_membership_from_remounted_root(
    tmp_path: Path,
) -> None:
    mountpoint = tmp_path / "delegated-memory"
    mountpoint.mkdir()
    (mountpoint / "cgroup.procs").write_text("")
    (mountpoint / "memory.limit_in_bytes").write_text("700\n")
    (mountpoint / "memory.usage_in_bytes").write_text("200\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("4:memory:/\n")
    proc_mountinfo = _write_cgroup1_memory_mountinfo(
        tmp_path,
        mountpoint,
        root="/tenant",
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


@linux_cgroup
def test_hybrid_cgroup_resolves_namespace_relative_v1_membership(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    unified = tmp_path / "unified"
    legacy_child = legacy / "workload"
    unified_child = unified / "unified-workload"
    legacy_child.mkdir(parents=True)
    unified_child.mkdir(parents=True)
    (legacy_child / "cgroup.procs").write_text("")
    (legacy_child / "memory.limit_in_bytes").write_text("700\n")
    (legacy_child / "memory.usage_in_bytes").write_text("200\n")
    (unified_child / "memory.max").write_text("300\n")
    (unified_child / "memory.current").write_text("100\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("5:cpu,memory:/workload\n0::/unified-workload\n")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 / {_mountinfo_path(unified)} rw - cgroup2 cgroup rw\n"
        f"2 0 0:2 /tenant {_mountinfo_path(legacy)} "
        "rw - cgroup cgroup rw,cpu,memory\n"
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
def test_cgroup_v2_available_uses_each_ancestors_inactive_file_working_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cgroup"
    parent = root / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (parent / "memory.max").write_text("1000\n")
    (parent / "memory.current").write_text("900\n")
    (parent / "memory.stat").write_text("anon 200\ninactive_file 700\n")
    (child / "memory.max").write_text("700\n")
    (child / "memory.current").write_text("200\n")
    (child / "memory.stat").write_text("inactive_file 50\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/parent/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 550)


@linux_cgroup
def test_cgroup_v2_inactive_file_cannot_make_available_exceed_limit(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("700\n")
    (child / "memory.current").write_text("200\n")
    (child / "memory.stat").write_text("inactive_file 300\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 700)


@pytest.mark.parametrize(
    "memory_stat",
    [None, "malformed\n", "inactive_file malformed\n", "inactive_file -1\n"],
)
@linux_cgroup
def test_cgroup_v2_invalid_inactive_file_falls_back_to_raw_current(
    tmp_path: Path,
    memory_stat: str | None,
) -> None:
    root = tmp_path / "cgroup"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("700\n")
    (child / "memory.current").write_text("200\n")
    if memory_stat is not None:
        (child / "memory.stat").write_text(memory_stat)
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


@linux_cgroup
def test_system_snapshot_keeps_cgroup_v2_swap_separate_from_ram(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("max\n")
    (child / "memory.swap.max").write_text("300\n")
    (child / "memory.swap.current").write_text("100\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    ram_only = system_memory_module._linux_cgroup_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    )
    assert ram_only.swap_total_bytes is None
    assert ram_only.swap_available_bytes is None

    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 900),
        host_swap_query=lambda: (500, 400),
        linux_cgroup_query=lambda: system_memory_module._linux_cgroup_status(
            proc_cgroup=proc_cgroup,
            proc_mountinfo=proc_mountinfo,
            include_swap=True,
        ),
        include_swap=True,
    )
    assert snapshot.effective_total_bytes == 2000
    assert snapshot.effective_available_bytes == 900
    assert snapshot.effective_swap_total_bytes == 300
    assert snapshot.effective_swap_available_bytes == 200
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
def test_system_snapshot_reads_cgroup_location_on_every_call(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    first = root / "first"
    second = root / "second"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "memory.max").write_text("1000\n")
    (first / "memory.current").write_text("100\n")
    (second / "memory.max").write_text("700\n")
    (second / "memory.current").write_text("200\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/first\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (1000, 900)
    proc_cgroup.write_text("0::/second\n")
    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


@linux_cgroup
def test_system_snapshot_resolves_rootless_remounted_cgroup_v2(tmp_path: Path) -> None:
    mountpoint = tmp_path / "delegated-cgroup"
    child = mountpoint / "workload"
    child.mkdir(parents=True)
    (mountpoint / "memory.max").write_text("1000\n")
    (mountpoint / "memory.current").write_text("400\n")
    (child / "memory.max").write_text("700\n")
    (child / "memory.current").write_text("200\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/user.slice/session.scope/workload\n")
    proc_mountinfo = _write_cgroup2_mountinfo(
        tmp_path,
        mountpoint,
        root="/user.slice/session.scope",
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


@linux_cgroup
def test_system_snapshot_prefers_most_specific_visible_cgroup_mount(tmp_path: Path) -> None:
    broad = tmp_path / "broad"
    specific = tmp_path / "specific"
    (broad / "tenant" / "workload").mkdir(parents=True)
    (specific / "workload").mkdir(parents=True)
    (broad / "tenant" / "workload" / "memory.max").write_text("1200\n")
    (broad / "tenant" / "workload" / "memory.current").write_text("100\n")
    (specific / "workload" / "memory.max").write_text("700\n")
    (specific / "workload" / "memory.current").write_text("200\n")
    (specific / "workload" / "cgroup.procs").write_text("")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/tenant/workload\n")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 / {_mountinfo_path(broad)} rw - cgroup2 cgroup rw\n"
        f"2 0 0:1 /tenant {_mountinfo_path(specific)} rw - cgroup2 cgroup rw\n"
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


@linux_cgroup
def test_system_snapshot_falls_back_from_stale_specific_mount(tmp_path: Path) -> None:
    broad = tmp_path / "broad"
    child = broad / "tenant" / "workload"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("1200\n")
    (child / "memory.current").write_text("100\n")
    (child / "cgroup.procs").write_text("")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/tenant/workload\n")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 / {_mountinfo_path(broad)} rw - cgroup2 cgroup rw\n"
        f"2 0 0:1 /tenant {_mountinfo_path(tmp_path / 'missing')} "
        "rw - cgroup2 cgroup rw\n"
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (1200, 1100)


@linux_cgroup
def test_system_snapshot_falls_back_from_unmounted_specific_root(tmp_path: Path) -> None:
    broad = tmp_path / "broad"
    child = broad / "tenant" / "workload"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("1200\n")
    (child / "memory.current").write_text("100\n")
    (child / "cgroup.procs").write_text("")
    unmounted = tmp_path / "unmounted"
    unmounted.mkdir()
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/tenant/workload\n")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 / {_mountinfo_path(broad)} rw - cgroup2 cgroup rw\n"
        f"2 0 0:1 /tenant/workload {_mountinfo_path(unmounted)} "
        "rw - cgroup2 cgroup rw\n"
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (1200, 1100)


@linux_cgroup
def test_system_snapshot_accepts_leaf_without_memory_controller_files(tmp_path: Path) -> None:
    broad = tmp_path / "broad"
    specific = tmp_path / "specific"
    (broad / "tenant" / "workload").mkdir(parents=True)
    leaf = specific / "workload"
    leaf.mkdir(parents=True)
    (specific / "memory.max").write_text("700\n")
    (specific / "memory.current").write_text("200\n")
    (leaf / "cgroup.procs").write_text("")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/tenant/workload\n")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        f"1 0 0:1 / {_mountinfo_path(broad)} rw - cgroup2 cgroup rw\n"
        f"2 0 0:1 /tenant {_mountinfo_path(specific)} rw - cgroup2 cgroup rw\n"
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


@linux_cgroup
def test_system_snapshot_decodes_mountinfo_path_escapes(tmp_path: Path) -> None:
    mountpoint = tmp_path / "cgroup with space"
    child = mountpoint / "workload"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("700\n")
    (child / "memory.current").write_text("200\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/delegated group/workload\n")
    proc_mountinfo = _write_cgroup2_mountinfo(
        tmp_path,
        mountpoint,
        root="/delegated group",
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


@linux_cgroup
def test_system_snapshot_does_not_match_cgroup_mount_root_by_string_prefix(
    tmp_path: Path,
) -> None:
    mountpoint = tmp_path / "tenant"
    mountpoint.mkdir()
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/tenant-other/workload\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, mountpoint, root="/tenant")

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (None, None)


@linux_cgroup
def test_system_snapshot_skips_malformed_cgroup_discovery_lines(tmp_path: Path) -> None:
    mountpoint = tmp_path / "cgroup"
    child = mountpoint / "child"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("700\n")
    (child / "memory.current").write_text("200\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("malformed\n2:cpu:/legacy\n0::/child\n")
    proc_mountinfo = tmp_path / "proc-self-mountinfo"
    proc_mountinfo.write_text(
        "malformed\n"
        f"1 0 0:1 relative {_mountinfo_path(mountpoint)} rw - cgroup2 cgroup rw\n"
        f"2 0 0:1 / {_mountinfo_path(mountpoint)} rw - cgroup2 cgroup rw\n"
    )

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (700, 500)


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


@pytest.mark.parametrize("current_value", [None, "malformed\n", "-1\n"])
@linux_cgroup
def test_system_snapshot_skips_limit_without_valid_current(
    tmp_path: Path,
    current_value: str | None,
) -> None:
    root = tmp_path / "cgroup"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("700\n")
    if current_value is not None:
        (child / "memory.current").write_text(current_value)
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    cgroup = system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    )
    assert cgroup == (None, None)


@linux_cgroup
def test_system_snapshot_uses_a_consistent_ancestor_pair(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    parent = root / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (parent / "memory.max").write_text("1000\n")
    (parent / "memory.current").write_text("400\n")
    (child / "memory.max").write_text("700\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/parent/child\n")
    proc_mountinfo = _write_cgroup2_mountinfo(tmp_path, root)

    assert system_memory_module._linux_cgroup_memory_status(
        proc_cgroup=proc_cgroup,
        proc_mountinfo=proc_mountinfo,
    ) == (1000, 600)


def test_system_snapshot_degrades_to_host_when_cgroup_query_fails() -> None:
    def unavailable() -> tuple[int | None, int | None]:
        raise OSError("cgroupfs unavailable")

    snapshot = system_memory_snapshot(
        "linux",
        host_query=lambda: (2000, 900),
        linux_cgroup_query=unavailable,
    )
    assert snapshot.effective_total_bytes == 2000
    assert snapshot.effective_available_bytes == 900
    assert snapshot.provenance == ("injected",)


def test_system_snapshot_fails_closed_when_host_query_fails() -> None:
    def unavailable() -> tuple[int, int]:
        raise OSError("host memory unavailable")

    with pytest.raises(RuntimeError, match="cannot determine system memory"):
        system_memory_snapshot("linux", host_query=unavailable)
