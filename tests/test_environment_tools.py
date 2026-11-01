"""Published tools run after extraction without a checkout or host package manager."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import ssl
import subprocess
import sysconfig
import tarfile
import zipfile
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from test_environment_generations import _standalone_tree, _wheel

from scripts import build_environment_tools, build_native_tools
from scripts.build_environment_tools import acquire, build, locked_wheels
from scripts.build_release import sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("host,system", [("linux", "linux"), ("darwin", "macos")])
def test_native_tools_selects_the_producer_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, system: str
) -> None:
    (tmp_path / "variants.toml").write_text(
        '[base]\nrelease = "20261026"\n'
        '[interpreters.linux]\nfilename = "linux.tar.gz"\nsize = 17\n'
        f'sha256 = "{"a" * 64}"\nexecutable = "python/bin/python3"\n'
        '[interpreters.macos]\nfilename = "apple-arm64.tar.gz"\nsize = 29\n'
        f'sha256 = "{"b" * 64}"\nexecutable = "python/bin/python3"\n',
        encoding="ascii",
    )
    monkeypatch.setattr(build_native_tools, "sys", SimpleNamespace(platform=host))

    def download(source: str, destination: Path, size: int, digest: str) -> None:
        # Stop at the verified-download boundary; no foreign executable can run here.
        expected = {
            "linux": ("linux.tar.gz", 17, "a" * 64),
            "macos": ("apple-arm64.tar.gz", 29, "b" * 64),
        }[system]
        assert (source.rsplit("/", 1)[-1], size, digest) == expected
        assert "/20261026/" in source
        raise RuntimeError("selected approved interpreter")

    monkeypatch.setattr(build_native_tools, "acquire", download)
    with pytest.raises(RuntimeError, match="selected approved interpreter"):
        build_native_tools.build(tmp_path, tmp_path / "supervisor.whl", tmp_path / "tools")
    assert not (tmp_path / "tools").exists()


def test_native_tools_refuses_unknown_hosts_before_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "variants.toml").write_text("", encoding="ascii")
    monkeypatch.setattr(build_native_tools, "sys", SimpleNamespace(platform="freebsd"))
    with pytest.raises(ValueError, match="unsupported native tools platform: freebsd"):
        build_native_tools.build(tmp_path, tmp_path / "supervisor.whl", tmp_path / "tools")
    assert not (tmp_path / "tools").exists()


def test_tools_acquisition_refuses_corruption_before_extraction(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"corrupt")
    target = tmp_path / "archive"
    with pytest.raises(ValueError, match="size or SHA256 mismatch"):
        acquire(str(source), target, 7, "a" * 64)
    assert not target.exists()


def test_tools_acquisition_retries_transient_tls_failure_without_retaining_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"verified tools wheel"
    destination = tmp_path / "archive"
    attempts = []
    sleeps = []

    class InterruptedResponse(io.BytesIO):
        url = "https://files.pythonhosted.org/tools.whl"

        def read(self, size: int | None = -1) -> bytes:
            if self.tell() > 0:
                raise URLError(ssl.SSLEOFError(8, "fixture TLS EOF"))
            return super().read(4)

    class CompleteResponse(io.BytesIO):
        url = "https://files.pythonhosted.org/tools.whl"

    def open_remote(source: str, timeout: int):
        attempts.append((source, timeout, destination.exists()))
        return InterruptedResponse(data) if len(attempts) == 1 else CompleteResponse(data)

    monkeypatch.setattr(build_environment_tools, "urlopen", open_remote)
    monkeypatch.setattr(build_environment_tools.time, "sleep", sleeps.append)
    acquire(CompleteResponse.url, destination, len(data), hashlib.sha256(data).hexdigest())

    assert destination.read_bytes() == data
    assert attempts == [(CompleteResponse.url, 60, False), (CompleteResponse.url, 60, False)]
    assert sleeps == [1]


def test_tools_acquisition_does_not_retry_terminal_http_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "archive"
    calls = []
    sleeps = []

    def open_remote(source: str, timeout: int):
        calls.append((source, timeout))
        raise HTTPError(source, 404, "missing", Message(), None)

    monkeypatch.setattr(build_environment_tools, "urlopen", open_remote)
    monkeypatch.setattr(build_environment_tools.time, "sleep", sleeps.append)
    with pytest.raises(HTTPError, match="HTTP Error 404"):
        acquire("https://files.pythonhosted.org/missing.whl", destination, 1, "a" * 64)

    assert calls == [("https://files.pythonhosted.org/missing.whl", 60)]
    assert sleeps == []
    assert not destination.exists()


@pytest.mark.parametrize("corruption", [None, "size", "hash", "unadmitted"])
def test_locked_tools_acquisition_uses_verified_fixture_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str | None
) -> None:
    index = tmp_path / "index"
    index.mkdir()
    wheel, digest = _wheel(index, "fixture_tools", "1.0", 41)
    admitted = "a" * 64 if corruption == "unadmitted" else digest
    exported = f"fixture-tools==1.0 --hash=sha256:{admitted}\n"
    # Only the export is supplied by the fixture; selection and acquisition stay real.
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: exported)
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "fixture-tools"\nversion = "1.0"\n'
        "[[package.wheels]]\n"
        f"url = {json.dumps(wheel.as_posix())}\n"
        f'hash = "sha256:{digest}"\nsize = {wheel.stat().st_size}\n',
        encoding="ascii",
    )
    if corruption == "size":
        wheel.write_bytes(wheel.read_bytes() + b"corrupt")
    elif corruption == "hash":
        data = wheel.read_bytes()
        wheel.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    output = tmp_path / "output"
    output.mkdir()
    if corruption:
        match = "no locked native" if corruption == "unadmitted" else "size or SHA256"
        with pytest.raises(ValueError, match=match):
            locked_wheels(tmp_path, output, "uv")
        assert not (output / wheel.name).exists()
    else:
        constraints = locked_wheels(tmp_path, output, "uv")
        assert (output / wheel.name).read_bytes() == wheel.read_bytes()
        assert constraints.read_text() == exported


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
        for operation, current, value in (("setup", 1, 41), ("update", 2, 59), ("rollback", 1, 41)):
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
    finally:
        subprocess.run(
            fixture["commands"]["stop"],
            env=environment,
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
