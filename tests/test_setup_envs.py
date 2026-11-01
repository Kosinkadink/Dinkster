"""Contributor selection uses producer-owned hardware bounds, never local torch pins."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts import setup_envs


@pytest.mark.parametrize("publication", ["legacy", "labelled", "wrong-manifest"])
def test_setup_uses_pinned_full_id_and_same_archive_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publication: str
) -> None:
    from test_environment_artifacts import _mirror

    from scripts import build_release

    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "uv.lock").write_text("package=[]\n")
    mirror = tmp_path / "mirror"
    base, _, _ = _mirror(mirror)
    identity = base["id"][7:]
    legacy = mirror / "base" / base["variant"] / identity
    label = f"2026-10-31-py3.14.10-torch1.0-{identity[:8]}"
    if publication != "legacy":
        base["label"] = label
        if publication == "wrong-manifest":
            base["python"] = "3.14.9"
        published = legacy.with_name(label)
        legacy.rename(published)
        (published / "manifest.json").write_text(json.dumps(base))
        index = [
            {"id": "sha256:" + "f" * 64, "path": "base/fixture-cpu/newer-unpinned"},
            {"id": base["id"], "path": f"base/{base['variant']}/{label}"},
        ]
        (published.parent / "index.json").write_text(json.dumps(index))
    else:
        # A stale or broken index must not override a valid legacy location.
        (legacy.parent / "index.json").write_text("not valid JSON")
    (root / "scripts/base_bundles.json").write_text(json.dumps({base["variant"]: identity}))
    monkeypatch.setenv("DINKSTER_EVIDENCE_ROOT", str(tmp_path / "no-evidence"))
    monkeypatch.setattr(build_release, "workspace_projects", lambda *args: {})

    def command(*args, **kwargs):
        if args[:2] == ("uv", "export"):
            (root / ".run/workspace-requirements.txt").write_text("")

    monkeypatch.setattr(setup_envs, "command", command)
    if publication == "wrong-manifest":
        with pytest.raises(ValueError, match="base manifest digest mismatch"):
            setup_envs.setup(root, mirror=str(mirror), variant=base["variant"])
        assert not (root / ".run").exists()
        return
    result = setup_envs.setup(root, mirror=str(mirror), variant=base["variant"])
    assert result == root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    receipt = json.loads((root / ".run/environment-base.json").read_text())
    assert receipt["id"] == base["id"]
    store = root / ".run/environment-store"
    assert (store / "bases" / identity / base["interpreter"]["executable"]).is_file()
    for wheel in base["wheels"]:
        assert (store / "wheels" / wheel["sha256"] / wheel["filename"]).read_bytes() == (
            store / "bases" / identity / "wheels" / wheel["filename"]
        ).read_bytes()
    assert {path.name for path in (store / "bases").iterdir()} == {identity}
    if publication == "labelled":
        assert all(label not in path.parts for path in store.rglob("*"))


@pytest.mark.parametrize(
    "failure", ["forbidden", "transport", "corrupt", "wrong-id", "path", "variant", "duplicate"]
)
def test_setup_index_cannot_bypass_source_errors_or_full_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from dinkster_supervisor import environment_downloads

    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    identity = "a" * 64
    (root / "scripts/base_bundles.json").write_text(json.dumps({"linux-cpu": identity}))
    label = "2026-10-31-py3.14.10-torch2.14.1-aaaaaaaa"
    row = {"id": "sha256:" + identity, "path": f"base/linux-cpu/{label}"}
    if failure == "wrong-id":
        row["id"] = "sha256:" + "b" * 64
    elif failure == "path":
        row["path"] = f"base/linux-cpu/../{label}"
    elif failure == "variant":
        row["path"] = f"base/win-cpu/{label}"
    index_reads = []

    def metadata(mirror, relative):
        if relative == f"base/linux-cpu/{identity}/manifest.json":
            if failure == "forbidden":
                raise HTTPError("https://example.invalid", 403, "forbidden", Message(), None)
            if failure == "transport":
                raise TimeoutError("metadata transport failed")
            if failure == "corrupt":
                return b"not valid JSON"
            raise HTTPError("https://example.invalid", 404, "missing", Message(), None)
        assert relative == "base/linux-cpu/index.json", "unsafe/unpinned fetch"
        index_reads.append(relative)
        return json.dumps([row, row] if failure == "duplicate" else [row]).encode()

    monkeypatch.setattr(environment_downloads, "read_metadata", metadata)
    with pytest.raises((ValueError, OSError)):
        setup_envs.setup(root, mirror="https://example.invalid")
    assert len(index_reads) == (0 if failure in ("forbidden", "transport", "corrupt") else 1)
    assert not (root / ".run").exists()


@pytest.fixture
def bases(monkeypatch):
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: "AMD64")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return [
        {"variant": "cpu", "platform": {"os": "windows", "arch": "x86_64"}, "nvidia": None},
        {
            "variant": "newer",
            "platform": {"os": "windows", "arch": "x86_64"},
            "nvidia": {
                "minimumDriver": "580",
                "minimumComputeCapability": "7.5",
                "maximumExclusiveComputeCapability": None,
            },
        },
        {
            "variant": "older",
            "platform": {"os": "windows", "arch": "x86_64"},
            "nvidia": {
                "minimumDriver": "525",
                "minimumComputeCapability": "5.0",
                "maximumExclusiveComputeCapability": "10.0",
            },
        },
    ]


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        ("580.1, 8.9", "newer"),
        ("579.9, 8.9", "older"),
        ("580.1, 6.1", "older"),
        ("524.9, 8.9", "cpu"),
        ("579.9, 10.0", "cpu"),
        ("580.1, 10.0", "newer"),
        ("580.1, 8.9\n530.1, 6.1", "older"),
    ],
)
def test_selects_compatible_bundle_from_manifest_bounds(bases, monkeypatch, hardware, expected):
    monkeypatch.setattr(setup_envs.subprocess, "check_output", lambda *args, **kwargs: hardware)
    assert setup_envs.select_base(bases)["variant"] == expected


@pytest.mark.parametrize("mask", ["", "-1"])
def test_disabled_gpu_selects_cpu_bundle(bases, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert setup_envs.select_base(bases)["variant"] == "cpu"
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases, "newer")


def test_unavailable_driver_selects_cpu_and_explicit_variant_cannot_bypass_bounds(
    bases, monkeypatch
):
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(setup_envs.subprocess, "check_output", unavailable)
    assert setup_envs.select_base(bases)["variant"] == "cpu"
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases, "older")


def test_no_bundle_for_another_platform_or_architecture(bases, monkeypatch):
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="No published base") as error:
        setup_envs.select_base(bases)
    assert "supported published variants: cpu, newer, older" in str(error.value)
    assert "--from-scratch" in str(error.value)
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="Intel macOS is unsupported"):
        setup_envs.select_base(bases)
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases)


def test_from_scratch_contributor_setup_names_missing_host_row(tmp_path: Path):
    root = Path(setup_envs.__file__).resolve().parents[1]
    variants = tmp_path / "variants.toml"
    variants.write_text('[base]\npython = "3.14.10"\npackages = ["torch"]\n[variants]\n')
    with pytest.raises(ValueError, match=r"missing variants.toml row: variants.absent-host"):
        setup_envs.setup(root, from_scratch=True, variant="absent-host", variants=variants)


def test_published_contributor_setup_substitutes_base_only_versions(tmp_path: Path, monkeypatch):
    from test_environment_artifacts import _mirror

    mirror, root = tmp_path / "mirror", tmp_path / "checkout"
    base, _, _ = _mirror(mirror)
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/base_bundles.json").write_text(json.dumps({base["variant"]: base["id"][7:]}))
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "1.0"\n')
    (root / "uv.lock").write_text("[[package]]\nname = 'fixture-base'\nversion = '9.0'\n")

    def command(*args, **kwargs):
        if args[:2] == ("uv", "export"):
            (root / ".run/workspace-requirements.txt").write_text(
                "fixture-base==9.0 --hash=sha256:" + "a" * 64 + "\n"
            )

    monkeypatch.setattr(setup_envs, "command", command)
    setup_envs.setup(root, mirror=str(mirror), variant=base["variant"])
    assert (root / ".run/workspace-requirements.txt").read_text() == (
        "fixture-base==1.0 --hash=sha256:" + base["wheels"][0]["sha256"] + "\n"
    )
    assert json.loads((root / ".run/environment-base.json").read_text())["id"] == base["id"]


@pytest.mark.parametrize("arch", ["arm64", "aarch64"])
def test_apple_silicon_selects_mac_bundle_without_cuda_probe(bases, monkeypatch, arch):
    monkeypatch.setattr(setup_envs.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: arch)
    mac = {"variant": "mac-arm64", "platform": {"os": "macos", "arch": "arm64"}, "nvidia": None}
    bases.append(mac)
    assert setup_envs.select_base(bases) == mac
    assert setup_envs.select_base(bases, "mac-arm64") == mac
    with pytest.raises(RuntimeError, match="No published base"):
        setup_envs.select_base(bases, "cpu")
    monkeypatch.setattr(setup_envs.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="Intel macOS is unsupported"):
        setup_envs.select_base(bases, "mac-arm64")


def test_setup_ignores_cwd_and_inherited_uv_project_target(tmp_path: Path):
    tools = tmp_path / "tools"
    tools.mkdir()
    probe = tmp_path / "probe.json"
    script = tools / "probe.py"
    script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SETUP_PROBE']).write_text(json.dumps({\n"
        "'project': os.environ.get('UV_PROJECT'),\n"
        "'environment': os.environ.get('UV_PROJECT_ENVIRONMENT'),\n"
        "'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
        "sys.exit(73)\n"
    )
    uv = tools / ("uv.cmd" if os.name == "nt" else "uv")
    uv.write_text(
        f'@"{sys.executable}" "{script}" %*\n'
        if os.name == "nt"
        else f'#!/bin/sh\n{shlex.quote(sys.executable)} {shlex.quote(str(script))} "$@"\n'
    )
    uv.chmod(0o755)
    root = Path(setup_envs.__file__).resolve().parents[1]
    environment = os.environ | {
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "UV_PROJECT": str(tmp_path / "wrong-project"),
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "wrong-environment"),
        "SETUP_PROBE": str(probe),
    }
    environment.pop("DINKSTER_SETUP_BOOTSTRAPPED", None)
    result = subprocess.run(
        [sys.executable, str(root / "scripts/setup_envs.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert probe.is_file(), f"stdout={result.stdout}\nstderr={result.stderr}"
    observed = json.loads(probe.read_text())
    assert Path(observed["project"]) == root
    assert Path(observed["environment"]) == root / ".run/bootstrap"
    assert Path(observed["cwd"]) == root
    assert observed["args"] == ["sync", "--locked", "--package", "dinkster-supervisor", "--no-dev"]
    assert not (tmp_path / "wrong-environment").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows runtime admission")
@pytest.mark.parametrize("matching_base", [False, True])
def test_setup_checks_reused_windows_environment_base(tmp_path, monkeypatch, matching_base):
    from dinkster_supervisor import environment_artifacts, environment_downloads

    from scripts import build_release

    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/base_bundles.json").write_text('{"win-cpu": "identity"}')
    (root / "uv.lock").write_text("package = []\n")
    gate = "packages/dinkster-supervisor/src/dinkster_supervisor/sqlite_runtime.py"
    (root / gate).parent.mkdir(parents=True)
    shutil.copyfile(Path(setup_envs.__file__).resolve().parents[1] / gate, root / gate)
    base_root = Path(sys.base_prefix).parent if matching_base else tmp_path / "other-base"
    base = {
        "id": "sha256:identity",
        "variant": "win-cpu",
        "platform": {"os": "windows", "arch": "x86_64"},
        "python": sys.version.split()[0],
        "interpreter": {"executable": Path(sys.base_prefix).name + "/python.exe"},
        "wheels": [
            {
                "name": "fixture",
                "version": "1.0",
                "filename": "fixture-1.0-py3-none-any.whl",
                "sha256": "a" * 64,
            }
        ],
        "replaces": ["fixture"],
    }
    environment = root / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    (root / ".run").mkdir()
    receipt = root / ".run/environment-base.json"
    receipt.write_text(json.dumps(base))
    monkeypatch.setattr(environment_downloads, "read_metadata", lambda *args: b"")
    monkeypatch.setattr(environment_artifacts, "validate_manifest", lambda *args: base)
    monkeypatch.setattr(environment_artifacts, "acquire_base", lambda *args: base_root)
    monkeypatch.setattr(setup_envs, "select_base", lambda *args: base)
    monkeypatch.setattr(build_release, "workspace_projects", lambda *args: {})
    run = setup_envs.command

    def fixture_command(*args, **kwargs):
        if args[:2] == ("uv", "export"):
            (root / ".run/workspace-requirements.txt").write_text("")
        elif args[0] != "uv":
            run(*args, **kwargs)

    monkeypatch.setattr(setup_envs, "command", fixture_command)
    if matching_base:
        assert setup_envs.setup(root) == environment / "Scripts/python.exe"
    else:
        before = receipt.read_bytes()
        with pytest.raises(subprocess.CalledProcessError):
            setup_envs.setup(root)
        assert receipt.read_bytes() == before
