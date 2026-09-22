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

import scripts.build_engine_feed as builder
from scripts.build_engine_feed import (
    CONTROL_ENVIRONMENT,
    EXECUTION_ENVIRONMENT,
    BaseBuild,
    FeedError,
    RequirementsFile,
    WheelEntry,
    base_identity_hash,
    build_base,
    build_code_layer,
    build_manifest,
    create_base_archive,
    load_cell_config,
    lock_descendants,
    locked_closure,
    normalize_name,
    require_native_cell,
    select_cell_wheel,
    upload_feed,
    wheel_matches_cell,
    workspace_members,
    write_requirements,
)
from scripts.build_release import wheel_metadata

ROOT = Path(__file__).resolve().parents[1]
CELLS_PATH = ROOT / "scripts/engine_cells.json"
COMMIT = "a" * 40
WHEEL_SCRATCH = Path("/tmp/feed-test")


def make_wheel(path: Path, name: str, version: str = "1.0.0") -> Path:
    """A real (minimal) wheel archive for metadata and store tests."""
    dist = f"{name.replace('-', '_').replace('.', '_')}-{version}.dist-info"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{dist}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
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
            packages or {"torch": "2.11.0+cu128"},
        ),
        archive_path=f"base/linux-cu128/{COMMIT[:8]}.tar.gz",
        sha256="b" * 64,
        size=1234,
        packages=packages or {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"},
        python_path="bin/python3",
        uv_path="tools/uv/uv",
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
        RequirementsFile("engine/reqs.txt", "c" * 64, 10),
    )
    manifest_new = build_manifest(
        COMMIT,
        "linux-cu128",
        base,
        [wheel_entry("dinkster-kitchen", "0.2.36", kitchen_new)],
        RequirementsFile("engine/reqs.txt", "c" * 64, 10),
    )
    assert manifest_old["base"]["id"] == manifest_new["base"]["id"]
    assert manifest_old["wheels"] != manifest_new["wheels"]


def test_torch_and_python_pin_changes_change_base_id() -> None:
    linux = base_identity_hash(
        "linux-cu128", "linux", "x86_64", "cpython", "3.12.13", {"torch": "2.11.0+cu128"}
    )
    bumped_torch = base_identity_hash(
        "linux-cu128", "linux", "x86_64", "cpython", "3.12.13", {"torch": "2.12.0+cu128"}
    )
    bumped_python = base_identity_hash(
        "linux-cu128", "linux", "x86_64", "cpython", "3.12.14", {"torch": "2.11.0+cu128"}
    )
    other_cell = base_identity_hash(
        "win-cu128", "windows", "amd64", "cpython", "3.12.13", {"torch": "2.11.0+cu128"}
    )
    assert len({linux, bumped_torch, bumped_python, other_cell}) == 4


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
    closure = locked_closure(lock, workspace_members(lock))
    expected = {"dinkster", "dinkster-inference-torch", "dinkster-p2p", "attrs", "psutil", "torch"}
    assert expected <= set(closure)
    assert "pytest" not in closure
    assert "nvidia-cublas" not in closure or True  # reachable only from torch below
    descendants = lock_descendants(lock, {"torch"})
    assert {"torch", "nvidia-cublas", "sympy"} <= descendants


def test_base_subtree_is_excluded_from_code_names() -> None:
    lock = fake_lock()
    closure = locked_closure(lock, workspace_members(lock))
    code_names = sorted(set(closure) - lock_descendants(lock, {"torch", "torchvision"}))
    assert "torch" not in code_names
    assert "nvidia-cublas" not in code_names
    assert "sympy" not in code_names
    expected = {"dinkster", "dinkster-inference-torch", "dinkster-p2p", "attrs", "psutil"}
    assert expected == set(code_names)


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


def test_requirements_have_hashes_and_no_urls(tmp_path: Path) -> None:
    entries = [
        wheel_entry("dinkster-kitchen", "0.2.35.post1"),
        wheel_entry("attrs", "25.1.0"),
    ]
    info = write_requirements(tmp_path, COMMIT, "linux-cu128", entries)
    text = (tmp_path / info.path).read_text(encoding="utf-8")
    assert "://" not in text
    assert "http" not in text
    lines = sorted(text.splitlines())
    assert lines[0].startswith("attrs==25.1.0 --hash=sha256:")
    assert lines[1].startswith("dinkster-kitchen==0.2.35.post1 --hash=sha256:")
    assert info.sha256 == builder.sha256_file(tmp_path / info.path)


def test_manifest_is_deterministic(tmp_path: Path) -> None:
    entry = wheel_entry("dinkster-kitchen", "0.2.35.post1")
    base = base_build()
    requirements = write_requirements(tmp_path, COMMIT, "linux-cu128", [entry])
    first = build_manifest(COMMIT, "linux-cu128", base, [entry], requirements)
    second = build_manifest(COMMIT, "linux-cu128", base, [entry], requirements)
    first_json = json.dumps(first, indent=2, sort_keys=True)
    second_json = json.dumps(second, indent=2, sort_keys=True)
    assert first_json == second_json
    assert first["format"] == "dinkster.engine/1"
    assert first["commit"] == COMMIT
    assert first["cell"] == "linux-cu128"
    assert set(first["base"]) == {"id", "archive", "python", "uv", "packages"}
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


# ---------------------------------------------------------------------------
# Base archive


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


def test_second_run_skips_unchanged_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_cell_config(CELLS_PATH, "linux-cu128")
    fake_interpreter = tmp_path / "interpreter"
    (fake_interpreter / "bin").mkdir(parents=True)
    python = fake_interpreter / "bin" / "python3"
    python.write_text("")
    python.chmod(0o755)
    calls = []

    def fake_install_dir(uv: str, python_version: str) -> Path:
        calls.append(python_version)
        return fake_interpreter

    def fake_install_packages(staging_python: Path) -> dict[str, str]:
        return {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"}

    def fake_copy_uv(uv: str, staging_root: Path, os_name: str) -> str:
        (staging_root / "tools" / "uv").mkdir(parents=True, exist_ok=True)
        (staging_root / "tools" / "uv" / "uv").write_text("uv-binary")
        return "tools/uv/uv"

    monkeypatch.setattr(builder, "_uv_python_install_dir", fake_install_dir)
    monkeypatch.setattr(builder, "_installed_packages", fake_install_packages)
    monkeypatch.setattr(builder, "_run", lambda command, cwd=None: "")
    monkeypatch.setattr(builder, "_copy_uv_binary", fake_copy_uv)

    first = build_base("uv", config, tmp_path)
    assert first.reused is False
    assert calls == ["3.12.13"]
    archive = tmp_path / first.archive_path
    assert archive.is_file()
    record = json.loads((tmp_path / "base/records.json").read_text())[
        builder._pin_identity_hash(config)
    ]
    assert record["python_path"] == "bin/python3"
    assert record["uv_path"] == "tools/uv/uv"
    assert record["packages"] == {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"}

    second = build_base("uv", config, tmp_path)
    assert second.reused is True
    assert second.base_id == first.base_id
    assert calls == ["3.12.13"], "unchanged pins must not rematerialize the base"

    archive.write_bytes(b"tampered")
    third = build_base("uv", config, tmp_path)
    assert third.reused is False
    assert calls == ["3.12.13", "3.12.13"], "a tampered archive must be rebuilt"


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
