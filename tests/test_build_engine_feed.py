"""The engine feed builder: base identity, wheelhouse assembly and channels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from dinkster.engine_feed import parse_channel, parse_manifest

import scripts.build_engine_feed as builder
from scripts.build_engine_feed import (
    CONTROL_ENVIRONMENT,
    EXECUTION_ENVIRONMENT,
    BaseBuild,
    FeedError,
    WheelEntry,
    base_identity_hash,
    build_base,
    build_code_layer,
    build_manifest,
    cell_marker_environment,
    code_lock_closure,
    create_base_archive,
    load_cell_config,
    normalize_name,
    require_native_cell,
    select_cell_wheel,
    sha256_file,
    upload_feed,
    wheel_matches_cell,
)
from scripts.build_release import wheel_metadata

ROOT = Path(__file__).resolve().parents[1]
CELLS_PATH = ROOT / "scripts/engine_cells.json"
COMMIT = "a" * 40
WHEEL_SCRATCH = Path("/tmp/feed-test")


def make_wheel(
    path: Path, name: str, version: str = "1.0.0", requires_dist: list[str] | None = None
) -> Path:
    """A real (minimal) wheel archive for metadata and store tests."""
    dist = f"{name.replace('-', '_').replace('.', '_')}-{version}.dist-info"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    for requirement in requires_dist or []:
        metadata += f"Requires-Dist: {requirement}\n"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{dist}/METADATA", metadata)
        wheel.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\n")
    return path


def wheel_entry(name: str, version: str = "1.0.0", path: Path | None = None) -> WheelEntry:
    wheel = make_wheel(path or WHEEL_SCRATCH / f"{name}-{version}-py3-none-any.whl", name, version)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return WheelEntry(
        path=f"store/{digest}",
        sha256=digest,
        size=wheel.stat().st_size,
        filename=wheel.name,
        name=normalize_name(name),
        version=version,
        environments=(CONTROL_ENVIRONMENT, EXECUTION_ENVIRONMENT),
    )


def base_build(packages: dict[str, str] | None = None) -> BaseBuild:
    return BaseBuild(
        base_id=base_identity_hash(
            "linux-cu128",
            "linux",
            "x86_64",
            "cpython",
            "3.12.13",
            "20260623",
            packages or {"torch": "2.11.0+cu128"},
            "e" * 64,
        ),
        archive_path=f"base/linux-cu128/{COMMIT[:8]}.tar.gz",
        sha256="b" * 64,
        size=1234,
        packages=packages or {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"},
        python_path="bin/python3",
        reused=False,
    )


# ---------------------------------------------------------------------------
# Base identity


def test_base_id_ignores_kitchen_and_aimdo_changes() -> None:
    base = base_build()
    kitchen_old = make_wheel(WHEEL_SCRATCH / "kitchen-old.whl", "dinkster-kitchen", "0.2.35.post1")
    kitchen_new = make_wheel(WHEEL_SCRATCH / "kitchen-new.whl", "dinkster-kitchen", "0.2.36")
    manifest_old = build_manifest(
        COMMIT,
        "linux-cu128",
        base,
        [wheel_entry("dinkster-kitchen", "0.2.35.post1", kitchen_old)],
    )
    manifest_new = build_manifest(
        COMMIT,
        "linux-cu128",
        base,
        [wheel_entry("dinkster-kitchen", "0.2.36", kitchen_new)],
    )
    assert manifest_old["base"]["id"] == manifest_new["base"]["id"]
    assert manifest_old["wheels"] != manifest_new["wheels"]


def base_id(
    python_version: str = "3.12.13",
    python_build: str = "20260623",
    torch: str = "2.11.0+cu128",
    uv_sha256: str = "e" * 64,
) -> str:
    return base_identity_hash(
        "linux-cu128",
        "linux",
        "x86_64",
        "cpython",
        python_version,
        python_build,
        {"torch": torch},
        uv_sha256,
    )


def test_python_torch_build_and_uv_changes_change_base_id() -> None:
    assert base_id() != base_id(torch="2.12.0+cu128")
    assert base_id() != base_id(python_version="3.12.14")
    # A revised standalone build of the same CPython version ships different
    # interpreter bytes, and so does an updated host uv: neither may reuse a
    # base id.
    assert base_id() != base_id(python_build="20260721")
    assert base_id() != base_id(uv_sha256="f" * 64)


# ---------------------------------------------------------------------------
# Wheel selection and lock closure


def fake_lock() -> dict:
    return {
        "dinkster": {
            "source": {"editable": "."},
            "dependencies": [
                {"name": "dinkster-inference-torch"},
                {"name": "dinkster-p2p"},
                {"name": "attrs"},
            ],
        },
        "dinkster-inference-torch": {
            "source": {"editable": "packages/dinkster-inference-torch"},
            "dependencies": [{"name": "torch"}, {"name": "attrs"}],
        },
        "dinkster-p2p": {
            "source": {"git": "https://github.com/Kosinkadink/dinkster-p2p.git?rev=x#" + COMMIT},
            "dependencies": [{"name": "psutil"}],
        },
        "attrs": {
            "source": {"registry": "https://pypi.org/simple"},
            "wheels": [{"url": "https://files.pythonhosted.org/x/attrs-25.1.0-py3-none-any.whl"}],
        },
        "psutil": {
            "source": {"registry": "https://pypi.org/simple"},
            "wheels": [
                {
                    "url": "https://files.pythonhosted.org/x/psutil-7.0.0-cp312-cp312-manylinux_2_28_x86_64.whl"
                }
            ],
        },
        "torch": {
            "source": {"registry": "https://pypi.org/simple"},
            "dependencies": [{"name": "nvidia-cublas"}, {"name": "sympy"}],
            "wheels": [
                {
                    "url": "https://files.pythonhosted.org/x/torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"
                }
            ],
        },
        "nvidia-cublas": {"source": {"registry": "https://pypi.org/simple"}, "dependencies": []},
        "sympy": {"source": {"registry": "https://pypi.org/simple"}, "dependencies": []},
        "pytest": {
            "source": {"registry": "https://pypi.org/simple"},
            "dependencies": [],
        },
    }


def test_closure_follows_runtime_deps_and_excludes_dev() -> None:
    lock = fake_lock()
    config = load_cell_config(CELLS_PATH, "linux-cu128")
    closure = code_lock_closure(lock, {"torch": "2.11.0+cu128"}, config)
    expected = {"dinkster", "dinkster-inference-torch", "dinkster-p2p", "attrs", "psutil"}
    assert expected <= set(closure)
    assert "pytest" not in closure
    # The base supplies torch, so neither torch nor its lock-only
    # descendants appear in the code closure.
    assert "torch" not in closure
    assert "nvidia-cublas" not in closure
    assert "sympy" not in closure


def test_shared_dependency_absent_from_base_is_supplied() -> None:
    """A distribution the lock's torch and non-torch code both need, which
    the built base does not carry, must ship in the code layer. Excluding
    the lock's whole torch subtree would silently drop it."""
    lock = fake_lock()
    lock["shared-dep"] = {"source": {"registry": "https://pypi.org/simple"}, "dependencies": []}
    lock["torch"]["dependencies"] = [{"name": "shared-dep"}]
    lock["dinkster"]["dependencies"].append({"name": "shared-dep"})
    config = load_cell_config(CELLS_PATH, "linux-cu128")
    code = code_lock_closure(lock, {"torch": "2.11.0+cu128"}, config)
    assert "shared-dep" in code
    assert "torch" not in code


def test_lock_only_torch_descendant_absent_from_base_is_not_shipped() -> None:
    """A distribution only the lock's torch variant needs, absent from the
    built base, stays out of the code layer: no code wheel requires it, and
    shipping it would carry the lock's alternate torch closure (for the real
    lock, the PyPI torch's CUDA 13 stack, around 2.4 GB of wheels the cu128
    base torch never uses). Completeness against real wheel METADATA is what
    proves the layer stays sufficient."""
    lock = fake_lock()
    lock["torch"]["dependencies"] = [{"name": "nvidia-cublas"}]
    config = load_cell_config(CELLS_PATH, "linux-cu128")
    code = code_lock_closure(lock, {"torch": "2.11.0+cu128"}, config)
    assert "nvidia-cublas" not in code
    assert "torch" not in code


def test_lock_edges_with_false_markers_are_not_followed() -> None:
    lock = fake_lock()
    lock["linux-only"] = {"source": {"registry": "https://pypi.org/simple"}, "dependencies": []}
    lock["dinkster"]["dependencies"] = [
        {"name": "dinkster-inference-torch"},
        {"name": "psutil", "marker": "sys_platform == 'win32'"},
        {"name": "linux-only", "marker": "sys_platform == 'linux'"},
    ]
    linux = code_lock_closure(lock, {}, load_cell_config(CELLS_PATH, "linux-cu128"))
    assert "psutil" not in linux
    assert "linux-only" in linux
    windows = code_lock_closure(lock, {}, load_cell_config(CELLS_PATH, "win-cu128"))
    assert "psutil" in windows
    assert "linux-only" not in windows


def test_marker_environments_are_cell_specific() -> None:
    linux = cell_marker_environment(load_cell_config(CELLS_PATH, "linux-cu128"))
    windows = cell_marker_environment(load_cell_config(CELLS_PATH, "win-cu128"))
    mac = cell_marker_environment(load_cell_config(CELLS_PATH, "mac-arm64"))
    assert linux["sys_platform"] == "linux"
    assert linux["platform_machine"] == "x86_64"
    assert windows["sys_platform"] == "win32"
    assert windows["platform_machine"] == "AMD64"
    assert mac["sys_platform"] == "darwin"
    assert mac["platform_machine"] == "arm64"
    for environment in (linux, windows, mac):
        assert environment["python_full_version"] == "3.12.13"
        assert environment["python_version"] == "3.12"


@pytest.mark.parametrize(
    ("filename", "os_name", "arch", "expected"),
    [
        ("torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl", "linux", "x86_64", True),
        ("torch-2.11.0+cu128-cp312-cp312-win_amd64.whl", "linux", "x86_64", False),
        ("torch-2.11.0+cu128-cp312-cp312-win_amd64.whl", "windows", "amd64", True),
        ("torch-2.11.0-cp312-cp312-macosx_14_0_arm64.whl", "macos", "arm64", True),
        ("torch-2.13.0-cp312-cp312-manylinux_2_28_aarch64.whl", "macos", "arm64", False),
        ("psutil-7.0.0-cp39-abi3-manylinux_2_17_x86_64.whl", "linux", "x86_64", True),
        ("psutil-7.0.0-cp313-cp313-manylinux_2_28_x86_64.whl", "linux", "x86_64", False),
        ("attrs-25.1.0-py3-none-any.whl", "macos", "arm64", True),
        ("attrs-25.1.0-py3-none-any.whl", "linux", "x86_64", True),
    ],
)
def test_native_wheel_selection(filename: str, os_name: str, arch: str, expected: bool) -> None:
    assert wheel_matches_cell(filename, os_name, arch) is expected


def test_foreign_platform_only_wheel_is_refused() -> None:
    wheels = [{"url": "https://example.org/x/torch-2.13.0-cp312-cp312-win_amd64.whl"}]
    with pytest.raises(FeedError, match="no wheel for cell platform linux/x86_64"):
        select_cell_wheel("torch", wheels, "linux", "x86_64")


def test_ambiguous_wheel_selection_is_refused() -> None:
    wheels = [
        {"url": "https://example.org/x/pkg-1.0-cp312-cp312-manylinux_2_28_x86_64.whl"},
        {"url": "https://example.org/x/pkg-1.0-cp312-cp312-manylinux_2_28_x86_64.whl"},
    ]
    with pytest.raises(FeedError, match="ambiguous"):
        select_cell_wheel("pkg", wheels, "linux", "x86_64")


# ---------------------------------------------------------------------------
# Duplicate, missing and tampered artifacts


def test_duplicate_distribution_refused() -> None:
    entries: dict[str, WheelEntry] = {}
    builder._register_entry(entries, wheel_entry("dinkster-kitchen", "0.2.35.post1"))
    with pytest.raises(FeedError, match="duplicate distribution dinkster-kitchen"):
        builder._register_entry(entries, wheel_entry("dinkster-kitchen", "0.2.36"))


def test_base_and_code_overlap_refused() -> None:
    with pytest.raises(FeedError, match="both base and code layers"):
        builder._refuse_base_code_overlap({"torch": "2.11.0"}, [wheel_entry("torch", "2.13.0")])


def test_tampered_store_entry_detected(tmp_path: Path) -> None:
    good = make_wheel(tmp_path / "good.whl", "dinkster-kitchen", "0.2.35.post1")
    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    store_entry = tmp_path / "store" / digest
    store_entry.parent.mkdir(parents=True)
    store_entry.write_bytes(b"tampered")
    with pytest.raises(FeedError, match="tampered or corrupted store"):
        builder._place_wheel(good, tmp_path)


def test_wheel_records_carry_the_hash_lock_data(tmp_path: Path) -> None:
    entry = wheel_entry("dinkster-kitchen", "0.2.35.post1")
    record = build_manifest(COMMIT, "linux-cu128", base_build(), [entry])["wheels"][0]
    assert record["name"] == "dinkster-kitchen"
    assert record["version"] == "0.2.35.post1"
    assert record["sha256"] == entry.sha256
    assert record["path"] == "store/" + entry.sha256


def test_manifest_is_deterministic(tmp_path: Path) -> None:
    entry = wheel_entry("dinkster-kitchen", "0.2.35.post1")
    base = base_build()
    first = build_manifest(COMMIT, "linux-cu128", base, [entry])
    second = build_manifest(COMMIT, "linux-cu128", base, [entry])
    first_json = json.dumps(first, indent=2, sort_keys=True)
    second_json = json.dumps(second, indent=2, sort_keys=True)
    assert first_json == second_json
    assert first["format"] == "dinkster.engine/1"
    assert first["commit"] == COMMIT
    assert first["cell"] == "linux-cu128"
    assert set(first["base"]) == {"id", "archive", "python", "packages"}
    assert first["base"]["archive"]["path"].startswith("base/linux-cu128/")
    assert first["wheels"][0]["environments"] == ["control", "execution"]
    assert set(first["wheels"][0]) == {
        "path",
        "sha256",
        "size",
        "filename",
        "name",
        "version",
        "environments",
    }


def test_real_wheel_metadata_flows_into_manifest(tmp_path: Path) -> None:
    wheel = make_wheel(
        tmp_path / "store-src" / "dinkster_kitchen-0.2.35.post1-py3-none-any.whl",
        "dinkster-kitchen",
        "0.2.35.post1",
    )
    name, version = wheel_metadata(wheel)
    assert (name, version) == ("dinkster-kitchen", "0.2.35.post1")
    entry = builder._place_wheel(wheel, tmp_path)
    assert entry.path.startswith("store/")
    assert entry.version == "0.2.35.post1"
    assert (tmp_path / entry.path).is_file()
    assert (tmp_path / entry.path).read_bytes() == wheel.read_bytes()


def test_code_layer_completeness_refuses_missing_requirement(tmp_path: Path) -> None:
    wheel = make_wheel(
        tmp_path / "src" / "pkg_a-1.0.0-py3-none-any.whl",
        "pkg-a",
        requires_dist=["missing-dep>=1.0"],
    )
    entry = builder._place_wheel(wheel, tmp_path)
    environment = cell_marker_environment(load_cell_config(CELLS_PATH, "linux-cu128"))
    with pytest.raises(FeedError, match=r"missing-dep>=1\.0"):
        builder._verify_code_layer_completeness([entry], tmp_path, {}, environment)


def test_code_layer_completeness_satisfied_by_base_and_false_markers(tmp_path: Path) -> None:
    wheel = make_wheel(
        tmp_path / "src" / "pkg_a-1.0.0-py3-none-any.whl",
        "pkg-a",
        requires_dist=[
            "torch (>=2.10)",
            "win-dep; sys_platform == 'win32'",
            "pkg-a[fast]; extra == 'fast'",
        ],
    )
    entry = builder._place_wheel(wheel, tmp_path)
    environment = cell_marker_environment(load_cell_config(CELLS_PATH, "linux-cu128"))
    builder._verify_code_layer_completeness(
        [entry], tmp_path, {"torch": "2.11.0+cu128"}, environment
    )


# ---------------------------------------------------------------------------
# Base archive


def test_python_selection_rejects_active_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching active venv must never become the base interpreter."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], cwd: Path | None = None) -> str:
        commands.append(command)
        if "find" in command:
            return f"{tmp_path / '.venv' / 'bin' / 'python3'}\n"
        return ""

    monkeypatch.setattr(builder, "_run", fake_run)
    with pytest.raises(FeedError, match="not a managed interpreter install"):
        builder._uv_python_install_dir("uv", "3.12.13")
    find = next(command for command in commands if "find" in command)
    for flag in ("--managed-python", "--no-project", "--no-config", "--resolve-links"):
        assert flag in find


def test_managed_install_dir_accepts_standalone_shape(tmp_path: Path) -> None:
    root = tmp_path / "cpython-3.12.13-linux-x86_64-gnu"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python3.12").write_text("")
    (root / "bin" / "python3").write_text("")
    assert builder._managed_install_dir(root / "bin" / "python3.12") == root


def test_managed_install_dir_accepts_windows_shape(tmp_path: Path) -> None:
    root = tmp_path / "cpython-3.12.13-windows-amd64"
    root.mkdir(parents=True)
    (root / "python.exe").write_text("")
    assert builder._managed_install_dir(root / "python.exe") == root


def test_managed_install_dir_rejects_missing_executable(tmp_path: Path) -> None:
    root = tmp_path / "cpython-3.12.13-windows-amd64"
    root.mkdir(parents=True)
    with pytest.raises(FeedError, match="no executable"):
        builder._managed_install_dir(root / "python.exe")
    posix_root = tmp_path / "cpython-3.12.13-linux-x86_64-gnu"
    posix_root.mkdir(parents=True)
    with pytest.raises(FeedError, match="no executable"):
        builder._managed_install_dir(posix_root / "bin" / "python3.12")


def test_managed_install_dir_rejects_active_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install = tmp_path / "cpython-3.12.13-linux-x86_64-gnu"
    (install / "bin").mkdir(parents=True)
    (install / "bin" / "python3").write_text("")
    monkeypatch.setattr(sys, "prefix", str(install))
    with pytest.raises(FeedError, match="active environment"):
        builder._managed_install_dir(install / "bin" / "python3")


def test_installed_packages_reads_real_interpreter_site_packages() -> None:
    packages = builder._installed_packages(Path(sys.executable))
    assert packages, "an isolated interpreter must still report its own site-packages"
    assert all(name == normalize_name(name) for name in packages)


def test_archive_preserves_internal_links_and_rejects_external(tmp_path: Path) -> None:
    staging = tmp_path / "base-root"
    (staging / "lib").mkdir(parents=True)
    (staging / "lib" / "real.so").write_bytes(b"x")
    (staging / "lib" / "link.so").symlink_to("real.so")
    (staging / "bin").mkdir()
    (staging / "bin" / "escape").symlink_to("/etc/passwd")
    with pytest.raises(FeedError, match="absolute symlink"):
        create_base_archive(staging, tmp_path / "out.tar.gz")
    (staging / "bin" / "escape").unlink()
    (staging / "bin" / "up").symlink_to("../../outside")
    with pytest.raises(FeedError, match="escapes the base root"):
        create_base_archive(staging, tmp_path / "out.tar.gz")
    (staging / "bin" / "up").unlink()

    archive = tmp_path / "out.tar.gz"
    create_base_archive(staging, archive)
    create_base_archive(staging, tmp_path / "out2.tar.gz")
    assert archive.read_bytes() == (tmp_path / "out2.tar.gz").read_bytes()
    with tarfile.open(archive) as tar:
        members = {member.name: member for member in tar.getmembers()}
    assert members["lib/link.so"].issym() and members["lib/link.so"].linkname == "real.so"


def test_native_only_builds_are_enforced() -> None:
    config = load_cell_config(CELLS_PATH, "mac-arm64")
    with pytest.raises(FeedError, match="natively only"):
        require_native_cell(config)
    require_native_cell(load_cell_config(CELLS_PATH, "linux-cu128"))


def base_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fully stubbed native base build: interpreter, torch closure, uv."""
    interpreter = tmp_path / "cpython-3.12.13-linux-x86_64-gnu"
    (interpreter / "bin").mkdir(parents=True)
    (interpreter / "BUILD").write_text("20260623\n")
    (interpreter / "bin" / "python3").write_text("")
    uv_binary = tmp_path / "uv-host"
    uv_binary.write_bytes(b"uv-host-bytes")

    def fake_copy_uv(uv: str, staging_root: Path, os_name: str) -> tuple[str, str]:
        source = builder._uv_source(uv)
        (staging_root / "tools").mkdir(parents=True, exist_ok=True)
        (staging_root / "tools" / "uv").write_bytes(source.read_bytes())
        return "tools/uv", sha256_file(source)

    monkeypatch.setattr(builder, "_uv_source", lambda uv: uv_binary)
    monkeypatch.setattr(builder, "_uv_python_install_dir", lambda uv, version: interpreter)
    monkeypatch.setattr(
        builder,
        "_installed_packages",
        lambda python: {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"},
    )
    monkeypatch.setattr(builder, "_run", lambda command, cwd=None: "")
    monkeypatch.setattr(builder, "_copy_uv_binary", fake_copy_uv)
    return interpreter, uv_binary


def test_second_run_skips_unchanged_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_cell_config(CELLS_PATH, "linux-cu128")
    base_env(tmp_path, monkeypatch)

    first = build_base("uv", config, tmp_path)
    assert first.reused is False
    archive = tmp_path / first.archive_path
    assert archive.is_file()
    record = json.loads((tmp_path / "base/records.json").read_text())[
        builder._pin_identity_hash(config, "20260623", sha256_file(tmp_path / "uv-host"))
    ]
    assert record["python_path"] == "bin/python3"
    assert record["python_build"] == "20260623"
    assert record["uv_sha256"] == sha256_file(tmp_path / "uv-host")
    assert record["packages"] == {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"}

    second = build_base("uv", config, tmp_path)
    assert second.reused is True
    assert second.base_id == first.base_id

    archive.write_bytes(b"tampered")
    third = build_base("uv", config, tmp_path)
    assert third.reused is False
    assert third.base_id == first.base_id, "a tampered archive must be rebuilt at the same id"


def test_changed_uv_bytes_force_a_distinct_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_cell_config(CELLS_PATH, "linux-cu128")
    _, _ = base_env(tmp_path, monkeypatch)
    first = build_base("uv", config, tmp_path)

    uv_b = tmp_path / "uv-host-b"
    uv_b.write_bytes(b"different-uv-bytes")
    monkeypatch.setattr(builder, "_uv_source", lambda uv: uv_b)

    second = build_base("uv", config, tmp_path)
    assert second.reused is False, "updated uv bytes must never reuse the old base"
    assert second.base_id != first.base_id
    assert second.archive_path != first.archive_path


def test_changed_standalone_build_forces_a_distinct_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_cell_config(CELLS_PATH, "linux-cu128")
    interpreter, _ = base_env(tmp_path, monkeypatch)
    first = build_base("uv", config, tmp_path)

    (interpreter / "BUILD").write_text("20260721\n")

    second = build_base("uv", config, tmp_path)
    assert second.reused is False, "a revised standalone build must never reuse the old base"
    assert second.base_id != first.base_id
    assert second.archive_path != first.archive_path


# ---------------------------------------------------------------------------
# Channels


def channel_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "feed": Path("feed"),
        "cells": "linux-cu128",
        "uv": "uv",
        "frontend_wheel": None,
        "channel": "stable",
        "tag": "v0.0.1",
        "evidence_file": None,
        "upload_endpoint": None,
        "upload_bucket": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def prepared_feed(tmp_path: Path) -> Path:
    feed = tmp_path / "feed"
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = feed / f"engine/{commit}/linux-cu128.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"format": "dinkster.engine/1", "commit": commit}), encoding="utf-8"
    )
    return feed


def test_stable_channel_requires_matching_tag(tmp_path: Path) -> None:
    feed = prepared_feed(tmp_path)
    with pytest.raises(FeedError, match="vX.Y.Z"):
        builder._write_channel(ROOT, feed, builder.git_commit(ROOT), channel_args(tag="v1.2"))
    with pytest.raises(FeedError, match="requires --tag"):
        builder._write_channel(ROOT, feed, builder.git_commit(ROOT), channel_args(tag=None))
    with pytest.raises(FeedError, match="does not match workspace metadata"):
        builder._write_channel(ROOT, feed, builder.git_commit(ROOT), channel_args(tag="v9.9.9"))


def test_stable_channel_written_for_matching_tag(tmp_path: Path) -> None:
    feed = prepared_feed(tmp_path)
    commit = builder.git_commit(ROOT)
    assert builder._write_channel(ROOT, feed, commit, channel_args()) == 0
    channel = json.loads((feed / "channels/stable.json").read_text())
    assert channel["format"] == "dinkster.engine-channel/1"
    assert channel["channel"] == "stable"
    assert channel["commit"] == commit
    assert channel["minimumLauncherVersion"] == "0.0.1"
    entry = channel["cells"]["linux-cu128"]
    assert entry["path"] == f"engine/{commit}/linux-cu128.json"
    assert entry["sha256"] == builder.sha256_file(feed / entry["path"])


# ---------------------------------------------------------------------------
# Production parser compatibility


def test_builder_manifest_parses_with_production_parser(tmp_path: Path) -> None:
    wheel = make_wheel(
        tmp_path / "src" / "dinkster_kitchen-0.2.35.post1-py3-none-any.whl",
        "dinkster-kitchen",
        "0.2.35.post1",
    )
    entry = builder._place_wheel(wheel, tmp_path)
    base = BaseBuild(
        base_id="c" * 64,
        archive_path="base/linux-cu128/" + "c" * 64 + ".tar.gz",
        sha256="d" * 64,
        size=42,
        packages={"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"},
        python_path="bin/python3",
        reused=False,
    )
    manifest = build_manifest(COMMIT, "linux-cu128", base, [entry])
    parsed = parse_manifest(json.dumps(manifest).encode("utf-8"))
    assert parsed.format == "dinkster.engine/1"
    assert parsed.commit == COMMIT
    assert parsed.cell == "linux-cu128"
    assert parsed.base.id == base.base_id
    assert parsed.base.python == "bin/python3"
    assert parsed.base.packages["torch"] == "2.11.0+cu128"
    assert parsed.base.archive.size == 42
    (parsed_wheel,) = parsed.wheels
    assert parsed_wheel.name == "dinkster-kitchen"
    assert parsed_wheel.version == "0.2.35.post1"
    assert parsed_wheel.sha256 == entry.sha256
    assert parsed_wheel.environments == ("control", "execution")


def test_builder_channel_parses_with_production_parser(tmp_path: Path) -> None:
    feed = prepared_feed(tmp_path)
    commit = builder.git_commit(ROOT)
    assert builder._write_channel(ROOT, feed, commit, channel_args()) == 0
    parsed = parse_channel((feed / "channels/stable.json").read_bytes())
    assert parsed.channel == "stable"
    assert parsed.commit == commit
    assert parsed.minimum_launcher_version == "0.0.1"
    assert parsed.cells["linux-cu128"].path == f"engine/{commit}/linux-cu128.json"


def test_github_live_channel_requires_real_validation_evidence(tmp_path: Path) -> None:
    feed = prepared_feed(tmp_path)
    commit = builder.git_commit(ROOT)
    with pytest.raises(FeedError, match="evidence"):
        builder._write_channel(ROOT, feed, commit, channel_args(channel="github-live", tag=None))
    with pytest.raises(FeedError, match="takes --evidence-file, not --tag"):
        builder._write_channel(
            ROOT, feed, commit, channel_args(channel="github-live", tag="v0.0.1")
        )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"commit": "b" * 40, "validation": "run 123"}))
    with pytest.raises(FeedError, match="different commit"):
        builder._write_channel(
            ROOT,
            feed,
            commit,
            channel_args(channel="github-live", tag=None, evidence_file=evidence),
        )
    evidence.write_text(json.dumps({"commit": commit, "validation": "run 123"}))
    assert (
        builder._write_channel(
            ROOT,
            feed,
            commit,
            channel_args(channel="github-live", tag=None, evidence_file=evidence),
        )
        == 0
    )
    channel = json.loads((feed / "channels/github-live.json").read_text())
    assert channel["channel"] == "github-live"
    assert channel["commit"] == commit


def test_git_dependency_wheel_survives_its_build_directory(tmp_path: Path) -> None:
    source = tmp_path / "dep-src"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "dep-pack"\nversion = "0.1.0"\n'
        'requires-python = ">=3.12"\n\n[build-system]\n'
        'requires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/dep_pack"]\n',
        encoding="utf-8",
    )
    (source / "src/dep_pack").mkdir(parents=True)
    (source / "src/dep_pack/__init__.py").write_text("", encoding="utf-8")
    builder._run(["git", "init", "--quiet", str(source)])
    builder._run(["git", "-C", str(source), "add", "."])
    builder._run(["git", "-C", str(source), "commit", "--quiet", "-m", "dep"])
    head = builder._run(["git", "-C", str(source), "rev-parse", "HEAD"]).strip()
    output = tmp_path / "built"
    output.mkdir()
    wheel = builder._build_git_dependency_wheel("uv", f"{source}?rev={head}#{head}", output)
    assert wheel.is_file(), "returned wheel must outlive the temporary clone"
    assert wheel.parent == output
    name, version = wheel_metadata(wheel)
    assert (name, version) == ("dep-pack", "0.1.0")


# ---------------------------------------------------------------------------
# Upload helper


def test_upload_refuses_non_loopback_endpoints(tmp_path: Path) -> None:
    for endpoint in ("https://example.com", "http://192.168.1.10:9000", "not-a-url"):
        with pytest.raises(FeedError):
            upload_feed(tmp_path, endpoint, "bucket")


def test_upload_walks_feed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "store").mkdir()
    (tmp_path / "store" / "abc").write_text("wheel")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "linux-cu128.json").write_text("{}")
    uploaded: list[tuple[str, str]] = []

    class FakeS3:
        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            uploaded.append((bucket, key))

    class FakeBoto3:
        @staticmethod
        def client(service: str, endpoint_url: str) -> FakeS3:
            return FakeS3()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3)
    count = upload_feed(tmp_path, "http://127.0.0.1:9000", "test-bucket")
    assert count == 2
    assert sorted(key for _, key in uploaded) == ["engine/linux-cu128.json", "store/abc"]


def test_cli_refusal_exits_status_2(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.build_engine_feed", "--feed", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_channel_write_is_followed_by_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed = prepared_feed(tmp_path)
    uploads: list[tuple[str, str]] = []

    def fake_upload(feed_dir: Path, endpoint: str, bucket: str) -> int:
        uploads.append((endpoint, bucket))
        return 1

    monkeypatch.setattr(builder, "upload_feed", fake_upload)
    args = channel_args(
        feed=feed,
        upload_endpoint="http://127.0.0.1:9000",
        upload_bucket="test-bucket",
    )
    assert builder._main(args) == 0
    assert (feed / "channels/stable.json").is_file()
    assert uploads == [("http://127.0.0.1:9000", "test-bucket")]

    uploads.clear()
    with pytest.raises(FeedError, match="--upload-bucket is required"):
        builder._main(
            channel_args(feed=feed, upload_endpoint="http://127.0.0.1:9000", upload_bucket=None)
        )
    assert uploads == []


# ---------------------------------------------------------------------------
# Cell configuration


def test_cell_config_pins_are_exact(tmp_path: Path) -> None:
    config = load_cell_config(CELLS_PATH, "linux-cu128")
    assert config.recipe.torch_requirement == "torch==2.11.0+cu128"
    assert config.recipe.torchvision_requirement == "torchvision==0.26.0+cu128"
    assert config.recipe.index_url == "https://download.pytorch.org/whl/cu128"
    assert config.python_version == "3.12.13"
    mac = load_cell_config(CELLS_PATH, "mac-arm64")
    assert mac.recipe.torch_requirement == "torch==2.13.0"
    cpu = load_cell_config(CELLS_PATH, "linux-cpu")
    assert cpu.recipe.torch_requirement == "torch==2.13.0+cpu"
    with pytest.raises(FeedError, match="unknown cell"):
        load_cell_config(CELLS_PATH, "linux-cu129")


def test_code_layer_requires_frontend_wheel(tmp_path: Path) -> None:
    with pytest.raises(FeedError, match="frontend"):
        build_code_layer(
            "uv",
            ROOT,
            tmp_path,
            load_cell_config(CELLS_PATH, "linux-cu128"),
            {"torch": "2.11.0+cu128"},
            tmp_path / "missing.whl",
        )
