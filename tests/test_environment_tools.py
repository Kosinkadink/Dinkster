"""Published tools run after extraction without a checkout or host package manager."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest
from dinkster_supervisor.environment_state import EnvironmentError
from dinkster_supervisor.environment_tools import tools_reference
from test_environment_artifacts import _mirror
from test_environment_generations import _standalone_tree

from scripts import build_tools_release
from scripts.build_environment_tools import build
from scripts.build_release import sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("damage", "layout"),
    [(None, "fresh"), (None, "collision"), (None, "legacy")]
    + [(damage, "fresh") for damage in ("manifest", "archive", "missing-pointer")],
)
def test_tools_cli_acquires_channel_publication_without_engine_payloads(
    tools_publication: tuple[Path, dict[str, bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    damage: str | None,
    layout: str,
) -> None:
    from dinkster_supervisor import environment, environment_tools

    output, payloads = tools_publication
    mirror = tmp_path / "mirror"
    _, _, channel = _mirror(mirror)
    data = (output / "tools-manifest.json").read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    label = f"{json.loads(data)['label']}-{sha[:8]}"
    release = mirror / "tools" / label
    shutil.copytree(output, release)
    channel["tools"] = {
        "manifest": str(release / "tools-manifest.json"),
        "sha256": sha,
        "release": str(release),
        "label": label,
    }
    (mirror / "channels/main.json").write_text(json.dumps(channel), encoding="ascii")
    shutil.rmtree(mirror / "engine")
    shutil.rmtree(mirror / "base")
    destination = tmp_path / "acquired"
    local_label = json.loads(data)["label"]
    name = "dinkster-tools-linux-x86_64.zip"
    if layout == "collision":
        from dinkster_store import object_path

        occupied = object_path(destination, "sha256:" + "1" * 64, label=local_label)
        occupied.mkdir()
        (occupied / name).write_bytes(b"previous tools")
        local_label += "--2"
    elif layout == "legacy":
        local_label = label
        (destination / label).mkdir(parents=True)
        (destination / label / name).write_bytes(payloads[name])
    before = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    monkeypatch.setattr(environment_tools, "machine", lambda: "x86_64")
    monkeypatch.setattr(environment_tools, "sys", SimpleNamespace(platform="linux"))
    arguments = [
        "dinkster-env",
        "--root",
        str(destination),
        "tools",
        "--mirror",
        str(mirror),
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", [*arguments, "--plan"])
    with pytest.raises(SystemExit) as planned:
        environment.main()
    assert planned.value.code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned"
    archive = destination / local_label / name
    assert plan["tools"]["archive"] == str(archive)
    assert {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()} == before
    if layout == "fresh":
        assert not destination.exists()
    if damage == "missing-pointer":
        del channel["tools"]
        (mirror / "channels/main.json").write_text(json.dumps(channel), encoding="ascii")
    elif damage:
        (release / ("tools-manifest.json" if damage == "manifest" else name)).write_bytes(b"wrong")
    monkeypatch.setattr(sys, "argv", [*arguments, "--yes"])
    with pytest.raises(SystemExit) as applied:
        environment.main()
    result = json.loads(capsys.readouterr().out)
    if damage:
        assert applied.value.code == 1
        assert result["status"] == "error"
        assert not archive.exists()
        if damage == "missing-pointer":
            assert "channel tools must be an object" in json.dumps(result)
    else:
        assert applied.value.code == 0
        assert result["tools"]["release"] == str(release)
        assert result["tools"]["label"] == local_label
        assert archive.read_bytes() == payloads[name]
        if layout == "collision":
            assert (
                destination / json.loads(data)["label"] / name
            ).read_bytes() == b"previous tools"
        assert not (destination / "project.json").exists()


@pytest.fixture
def tools_publication(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(tmp_path)],
        check=True,
    )
    project = tmp_path / "packages/dinkster-supervisor"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "dinkster-supervisor"\nversion = "2.7.4"\n', encoding="ascii"
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "dinkster"\nversion = "9.1.2"\n', encoding="ascii"
    )
    output = tmp_path / "publication"
    output.mkdir()
    payloads = {
        "dinkster-tools-linux-x86_64.zip": b"linux payload",
        "dinkster-tools-windows-x86_64.zip": b"windows payload with different length",
        "dinkster-tools-macos-arm64.zip": b"mac payload",
    }
    for name, data in payloads.items():
        (output / name).write_bytes(data)
    build_tools_release.build(tmp_path, output)
    return output, payloads


@pytest.mark.parametrize("release", ["https://releases.dinkster.org", "local", "windows"])
def test_channel_tools_pin_binds_locations_to_immutable_publication(
    release: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    label = "2026-11-05-2.7.4-12345678"
    if release == "windows":
        monkeypatch.setattr("dinkster_supervisor.environment_tools.Path", PureWindowsPath)
        directory = PureWindowsPath("C:/mirror") / "tools" / label
        release, manifest = str(directory), str(directory / "tools-manifest.json")
    elif release == "local":
        directory = tmp_path / "tools" / label
        release, manifest = str(directory), str(directory / "tools-manifest.json")
    else:
        release += "/tools/" + label
        manifest = release + "/tools-manifest.json"
    pin = {
        "label": label,
        "sha256": "12345678" + "a" * 56,
        "release": release,
        "manifest": manifest,
    }
    assert tools_reference(pin) == pin
    with pytest.raises(EnvironmentError, match="dated version and hash8"):
        tools_reference({**pin, "label": "2026-11-05-2.7.4-87654321"})
    with pytest.raises(EnvironmentError, match="locations differ"):
        tools_reference({**pin, "manifest": release + "/release-manifest.json"})
    with pytest.raises(EnvironmentError, match="credential-free HTTPS"):
        tools_reference({**pin, "release": "https://user:password@example.org/tools/" + label})


@pytest.mark.network("pypi.org", "files.pythonhosted.org")
@pytest.mark.parametrize("invalid_input", [None, "abi", "dependency"])
def test_tools_archive_runs_from_relocated_path_without_host_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_input: str | None,
) -> None:
    tree = tmp_path / "standalone"
    _standalone_tree(tree / "python")
    source = tmp_path / "python.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        archive.add(tree / "python", "python")
    wheels = tmp_path / "wheels"
    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run(
        [uv, "build", "--package", "dinkster-supervisor", "--wheel", "--out-dir", str(wheels)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    output = tmp_path / "tools"
    wheel = next(wheels.glob("*.whl"))
    if invalid_input == "abi":
        actual = sysconfig.get_config_var
        monkeypatch.setattr(
            sysconfig,
            "get_config_var",
            lambda name: "different-abi" if name == "SOABI" else actual(name),
        )
    elif invalid_input == "dependency":
        with zipfile.ZipFile(wheel) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        metadata = next(name for name in members if name.endswith(".dist-info/METADATA"))
        head, _, body = members[metadata].partition(b"\n\n")
        members[metadata] = (
            head.rstrip(b"\n") + b"\nRequires-Dist: missing-tools-fixture==1.0\n\n" + body
        )
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
    if invalid_input:
        match = "architecture and ABI" if invalid_input == "abi" else "missing-tools-fixture"
        with pytest.raises(ValueError, match=match):
            build(ROOT, output, str(source), source.stat().st_size, sha256(source), wheel, uv=uv)
        assert not output.exists()
        return
    row = build(
        ROOT,
        output,
        str(source),
        source.stat().st_size,
        sha256(source),
        wheel,
        uv=uv,
    )
    artifact = output / str(row["name"])
    assert row["size"] == artifact.stat().st_size
    assert row["sha256"] == sha256(artifact)
    shutil.rmtree(tree)
    source.unlink()
    wheels.rename(tmp_path / "not-on-path")
    relocated = tmp_path / "relocated tools with spaces"
    with zipfile.ZipFile(artifact) as archive:
        archive.extractall(relocated)
        if os.name != "nt":
            for member in archive.infolist():
                (relocated / member.filename).chmod((member.external_attr >> 16) & 0o777)
    python = relocated / ("python/python.exe" if os.name == "nt" else "python/bin/python3")
    scripts = python.parent / "Scripts" if os.name == "nt" else python.parent
    environment = {
        **os.environ,
        "PATH": str(scripts),
        "PYTHONPATH": "",
        "UV_INDEX_URL": "http://127.0.0.1:1/unreachable",
        "HTTP_PROXY": "http://127.0.0.1:1/unreachable",
        "HTTPS_PROXY": "http://127.0.0.1:1/unreachable",
        "XDG_DATA_HOME": str(tmp_path / "user-data"),
        "LOCALAPPDATA": str(tmp_path / "user-data"),
    }
    project = tmp_path / "absent project"
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "dinkster_supervisor.environment",
            "--root",
            str(project),
            "status",
            "--json",
        ],
        env=environment,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    document = json.loads(result.stdout)
    assert document["format"] == "dinkster.environment-result/1"
    assert document["current"] is None and document["engines"] == []
    assert not project.exists()
    bundled_uv = scripts / ("uv.exe" if os.name == "nt" else "uv")
    result = subprocess.run(
        [str(bundled_uv), "--version"], env=environment, text=True, capture_output=True, check=True
    )
    assert result.stdout.startswith("uv 0.11.29")
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('torch') is None; "
            "assert importlib.util.find_spec('dinkster') is None",
        ],
        env=environment,
        cwd=tmp_path,
        check=True,
    )
    fixture = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-I",
                str(ROOT / "tests/fixtures/environments/build.py"),
                "--output",
                str(tmp_path / "lifecycle"),
            ],
            env=environment,
            cwd=tmp_path,
            text=True,
        )
    )
    root = Path(fixture["root"])
    try:
        for operation, current, value in (("setup", 1, 41), ("update", 2, 59), ("rollback", 3, 41)):
            if operation == "update":
                subprocess.run(
                    fixture["commands"]["advance"],
                    env=environment,
                    cwd=tmp_path,
                    check=True,
                    capture_output=True,
                )
            result = subprocess.run(
                fixture["commands"][operation],
                env=environment,
                cwd=tmp_path,
                check=True,
                text=True,
                capture_output=True,
            )
            assert json.loads(result.stdout)["current"] == current
            assert (root / "current").read_text() == f"{current}\n"
            engine = root / f"generations/{current}/python"
            engine = engine / ("python.exe" if os.name == "nt" else "bin/python3")
            result = subprocess.check_output(
                [str(engine), "-I", "-c", "import fixture_engine; print(fixture_engine.VALUE)"],
                env=environment,
                cwd=tmp_path,
                text=True,
            )
            assert result.strip() == str(value)
        assert (root / "generations/2/generation.json").is_file()
        entries = [
            json.loads(path.read_bytes()) for path in sorted((root / "history").glob("*.json"))
        ]
        assert [entry["generation"] for entry in entries] == [1, 2, 3]
        assert entries[2]["restoredFrom"] == 1
        assert entries[2]["reason"] == "rollback"
        assert entries[2]["snapshot"] == entries[0]["snapshot"]
    finally:
        subprocess.run(
            fixture["commands"]["stop"],
            env=environment,
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
