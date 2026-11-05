"""Published tools run after extraction without a checkout or host package manager."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import ssl
import subprocess
import sys
import sysconfig
import tarfile
import time
import zipfile
from http.client import HTTPMessage
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from dinkster_supervisor.environment_state import EnvironmentError
from dinkster_supervisor.environment_tools import tools_reference, validate_tools_manifest
from test_environment_artifacts import _mirror
from test_environment_generations import _standalone_tree, _wheel

from scripts import build_environment_tools, build_native_tools, build_tools_release
from scripts.build_environment_tools import acquire, build, locked_wheels
from scripts.build_release import sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("changed", "expected"),
    [("engine-source", False), ("engine-lock", False), ("tools-lock", True), ("supervisor", True)],
)
def test_tools_build_selection_ignores_engine_only_changes(
    tmp_path: Path, changed: str, expected: bool
) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    lock = """
[[package]]
name = "dinkster-supervisor"
version = "1.0"
dependencies = [{ name = "transport" }]
[package.dev-dependencies]
environment-tools = [{ name = "uv" }]
[[package]]
name = "transport"
version = "2.0"
dependencies = [{ name = "codec", extra = ["native"] }]
[[package]]
name = "codec"
version = "3.0"
[package.optional-dependencies]
native = [{ name = "native-codec" }]
[[package]]
name = "native-codec"
version = "4.0"
[[package]]
name = "uv"
version = "5.0"
[[package]]
name = "dinkster-engine"
version = "6.0"
"""
    (tmp_path / "uv.lock").write_text(lock, encoding="ascii")

    def snapshot() -> str:
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        return subprocess.check_output(["git", "write-tree"], cwd=tmp_path, text=True).strip()

    before = snapshot()
    if changed.endswith("lock"):
        (tmp_path / "uv.lock").write_text(
            lock.replace('version = "6.0"', 'version = "6.1"')
            if changed == "engine-lock"
            else lock.replace('version = "4.0"', 'version = "4.1"'),
            encoding="ascii",
        )
    else:
        source = tmp_path / (
            "packages/dinkster-supervisor/client.py" if changed == "supervisor" else "src/engine.py"
        )
        source.parent.mkdir(parents=True)
        source.write_text("new_behavior = True\n", encoding="ascii")
    assert build_tools_release.inputs_changed(tmp_path, before, snapshot()) is expected


@pytest.mark.parametrize("damage", [None, "manifest", "archive"])
def test_tools_cli_acquires_channel_publication_without_engine_payloads(
    tools_publication: tuple[Path, dict[str, bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    damage: str | None,
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
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    assert not destination.exists()
    name = "dinkster-tools-linux-x86_64.zip"
    if damage:
        (release / ("tools-manifest.json" if damage == "manifest" else name)).write_bytes(b"wrong")
    monkeypatch.setattr(sys, "argv", [*arguments, "--yes"])
    with pytest.raises(SystemExit) as applied:
        environment.main()
    result = json.loads(capsys.readouterr().out)
    archive = destination / label / name
    if damage:
        assert applied.value.code == 1
        assert result["status"] == "error"
        assert not archive.exists()
    else:
        assert applied.value.code == 0
        assert result["tools"]["release"] == str(release)
        assert result["tools"]["label"] == label
        assert archive.read_bytes() == payloads[name]
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


def test_tools_publication_uses_supervisor_version_and_exact_native_payloads(
    tools_publication: tuple[Path, dict[str, bytes]],
) -> None:
    output, payloads = tools_publication
    data = (output / "tools-manifest.json").read_bytes()
    manifest = json.loads(data)
    digest = hashlib.sha256(data).hexdigest()
    assert validate_tools_manifest(data, digest) == manifest
    assert (
        manifest["commit"]
        == subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    )
    assert manifest["version"] == "2.7.4"
    assert str(manifest["label"]).endswith("-2.7.4")
    assert manifest["artifacts"] == [
        {
            "name": name,
            "size": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name in sorted(payloads)
    ]
    checksums = (output / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert set(checksums) == {
        f"{hashlib.sha256(payload).hexdigest()}  {name}" for name, payload in payloads.items()
    } | {f"{digest}  tools-manifest.json"}
    with pytest.raises(EnvironmentError, match="SHA256 mismatch"):
        validate_tools_manifest(data + b" ", digest)
    document = json.loads(data)
    document["artifacts"].pop()
    incomplete = json.dumps(document).encode("ascii")
    with pytest.raises(EnvironmentError, match="all three native archives"):
        validate_tools_manifest(incomplete, hashlib.sha256(incomplete).hexdigest())


@pytest.mark.parametrize(
    "corrupt",
    [None, "github", "mirror", "behind", "diverged", "ahead", "identical", "replay", "identity"],
)
def test_tools_publication_exposes_channel_only_after_both_sources_verify(
    tmp_path: Path,
    tools_publication: tuple[Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
    corrupt: str | None,
) -> None:
    from dinkster_supervisor.environment_downloads import DownloadError, fetch_artifact

    from scripts import publish_tools_release as publisher

    output, payloads = tools_publication
    data = (output / "tools-manifest.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    manifest = json.loads(data)
    label = manifest["label"] + "-" + digest[:8]
    public, github = tmp_path / "mirror", tmp_path / "github"
    (public / "channels").mkdir(parents=True)
    channel = {
        "format": "dinkster.channel/1",
        "engine": {
            "version": "9.1.2",
            "release": "sha256:" + "a" * 64,
            "locations": [
                {"type": "mirror", "path": "engine/" + "a" * 64 + "/release-manifest.json"}
            ],
        },
        "base": {
            "linux-cpu": {
                "id": "sha256:" + "b" * 64,
                "locations": [
                    {"type": "mirror", "path": "base/linux-cpu/" + "b" * 64 + "/manifest.json"}
                ],
            }
        },
        "minimumDesktopVersion": "0.2.0",
        "signature": None,
    }
    if corrupt in {"behind", "diverged", "ahead", "identical", "identity"}:
        previous = {**manifest, "commit": "c" * 40, "version": "2.7.3"}
        previous["label"] = str(manifest["label"]).removesuffix("2.7.4") + "2.7.3"
        previous_data = json.dumps(previous).encode("ascii")
        previous_sha = hashlib.sha256(previous_data).hexdigest()
        previous_label = str(previous["label"]) + "-" + previous_sha[:8]
        if corrupt == "identity":
            previous_label = previous_label.replace("2.7.3", "2.7.2")
        previous_mirror = tmp_path / "other-mirror" if corrupt == "identity" else public
        previous_release = previous_mirror / "tools" / previous_label
        previous_release.mkdir(parents=True)
        (previous_release / "tools-manifest.json").write_bytes(previous_data)
        channel["tools"] = {
            "label": previous_label,
            "sha256": previous_sha,
            "release": str(previous_release),
            "manifest": str(previous_release / "tools-manifest.json"),
        }
    before = json.dumps(channel).encode("ascii")
    (public / "channels/main.json").write_bytes(before)
    (public / "channels/stable.json").write_bytes(b"stable stays unchanged")
    for key, value in {
        "GITHUB_REPOSITORY": "Kosinkadink/Dinkster",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": manifest["commit"],
        "GITHUB_TOKEN": "fixture",
        "R2_PUBLIC_BASE_URL": str(public),
        "R2_BUCKET": "fixture",
    }.items():
        monkeypatch.setenv(key, value)

    class Bucket:
        def get_paginator(self, name):
            return self

        def paginate(self, *, Bucket, Prefix):
            return [
                {
                    "Contents": [
                        {"Key": path.relative_to(public).as_posix(), "Size": path.stat().st_size}
                        for path in public.rglob("*")
                        if path.is_file() and path.relative_to(public).as_posix().startswith(Prefix)
                    ]
                }
            ]

        def get_object(self, *, Bucket, Key):
            return {"Body": io.BytesIO((public / Key).read_bytes())}

        def upload_file(self, source, bucket, key, **kwargs):
            destination = public / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if corrupt == "mirror" and key.endswith(".zip"):
                destination.write_bytes(b"corruption")

    def release_command(command, **kwargs):
        if command[:3] == ["gh", "release", "view"]:
            if github.exists():
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps(
                        {"tagName": "tools-" + label, "targetCommitish": manifest["commit"]}
                    ),
                    "",
                )
            return subprocess.CompletedProcess(command, 1, "", "not found")
        assert command[:3] == ["gh", "release", "create"]
        assert command[3] == "tools-" + label
        assert command[command.index("--target") + 1] == manifest["commit"]
        github.mkdir()
        for path in output.iterdir():
            shutil.copyfile(path, github / path.name)
        if corrupt == "github":
            (github / "dinkster-tools-linux-x86_64.zip").write_bytes(b"corruption")
        return subprocess.CompletedProcess(command, 0, "", "")

    def download(sources, name, destination, sha, size):
        if sources[0].startswith("https://github.com/"):
            sources = (str(github),)
        return fetch_artifact(sources, name, destination, sha, size)

    monkeypatch.setattr(publisher, "object_store", Bucket)
    monkeypatch.setattr(publisher.subprocess, "run", release_command)
    monkeypatch.setattr(publisher, "fetch_artifact", download)

    def compare(request, **kwargs):
        assert request.full_url.endswith(f"/{'c' * 40}...{manifest['commit']}")
        status = "ahead" if corrupt == "identity" else corrupt
        return io.BytesIO(json.dumps({"status": status}).encode("ascii"))

    monkeypatch.setattr(publisher, "urlopen", compare)
    if corrupt in {"github", "mirror"}:
        with pytest.raises(DownloadError):
            publisher.publish(output)
        assert (public / "channels/main.json").read_bytes() == before
        return
    if corrupt in {"identical", "identity"}:
        with pytest.raises(ValueError, match="same source|identity differs"):
            publisher.publish(output)
        assert (public / "channels/main.json").read_bytes() == before
        return
    publisher.publish(output)
    if corrupt in {"behind", "diverged"}:
        assert (public / "channels/main.json").read_bytes() == before
        return
    if corrupt == "replay":
        first = (public / "channels/main.json").read_bytes()
        publisher.publish(output)
        assert (public / "channels/main.json").read_bytes() == first
    after = json.loads((public / "channels/main.json").read_bytes())
    reference = after.pop("tools")
    assert after == {key: value for key, value in channel.items() if key != "tools"}
    assert reference == {
        "label": label,
        "sha256": digest,
        "release": f"{public}/tools/{label}",
        "manifest": f"{public}/tools/{label}/tools-manifest.json",
    }
    assert (public / "channels/stable.json").read_bytes() == b"stable stays unchanged"
    for name, payload in payloads.items():
        assert (public / "tools" / label / name).read_bytes() == payload
        assert (github / name).read_bytes() == payload
    index = json.loads((public / "tools/index.json").read_bytes())
    assert len(index) == 1
    assert index[0]["label"] == label
    assert index[0]["path"] == "tools/" + label
    assert index[0]["sha256"] == digest
    assert index[0]["size"] == sum(path.stat().st_size for path in output.iterdir())


@pytest.mark.parametrize("release", ["https://releases.dinkster.org", "/absolute/mirror"])
def test_channel_tools_pin_binds_locations_to_immutable_publication(release: str) -> None:
    label = "2026-11-05-2.7.4-12345678"
    release += "/tools/" + label
    pin = {
        "label": label,
        "sha256": "12345678" + "a" * 56,
        "release": release,
        "manifest": release + "/tools-manifest.json",
    }
    assert tools_reference(pin) == pin
    with pytest.raises(EnvironmentError, match="dated version and hash8"):
        tools_reference({**pin, "label": "2026-11-05-2.7.4-87654321"})
    with pytest.raises(EnvironmentError, match="locations differ"):
        tools_reference({**pin, "manifest": release + "/release-manifest.json"})
    with pytest.raises(EnvironmentError, match="credential-free HTTPS"):
        tools_reference({**pin, "release": "https://user:password@example.org/tools/" + label})


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


@pytest.mark.parametrize("failure", ["handshake", "body", "timeout", "http"])
def test_tools_acquisition_recovers_verified_bytes_after_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"complete verified wheel")
    destination = tmp_path / "wheel"
    attempts = 0

    class Response(io.BytesIO):
        url = "https://files.pythonhosted.org/fixture.whl"

        def read(self, size: int | None = -1) -> bytes:
            if failure == "body" and attempts == 1:
                if self.tell() == 0:
                    return super().read(5)
                raise ssl.SSLEOFError("injected body EOF")
            return super().read(size)

    def open_url(*args: object, **kwargs: object) -> Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "handshake":
                raise URLError(ssl.SSLEOFError("injected handshake EOF"))
            if failure == "timeout":
                raise TimeoutError("injected timeout")
            if failure == "http":
                raise HTTPError(Response.url, 503, "unavailable", HTTPMessage(), None)
        return Response(source.read_bytes())

    monkeypatch.setattr(build_environment_tools, "urlopen", open_url)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    acquire(Response.url, destination, source.stat().st_size, sha256(source))
    assert destination.read_bytes() == source.read_bytes()
    assert attempts == 2


@pytest.mark.parametrize(
    "failure", ["exhausted", "body-exhausted", "not-found", "certificate", "corrupt"]
)
def test_tools_acquisition_refuses_terminal_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    destination = tmp_path / "wheel"
    attempts = 0
    url = "https://files.pythonhosted.org/fixture.whl"

    class Response(io.BytesIO):
        url = "https://files.pythonhosted.org/fixture.whl"

        def read(self, size: int | None = -1) -> bytes:
            if failure == "body-exhausted":
                if self.tell() == 0:
                    return super().read(5)
                raise ssl.SSLEOFError("injected body EOF")
            return super().read(size)

    def open_url(*args: object, **kwargs: object) -> Response:
        nonlocal attempts
        attempts += 1
        if failure == "exhausted":
            raise URLError(ssl.SSLEOFError("injected handshake EOF"))
        if failure == "not-found":
            raise HTTPError(url, 404, "missing", HTTPMessage(), None)
        if failure == "certificate":
            raise URLError(ssl.SSLCertVerificationError("untrusted certificate"))
        return Response(b"corrupt")

    monkeypatch.setattr(build_environment_tools, "urlopen", open_url)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises((URLError, ValueError, ssl.SSLEOFError)):
        acquire(url, destination, 7, "a" * 64)
    assert not destination.exists()
    assert attempts == (3 if failure.endswith("exhausted") else 1)


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
