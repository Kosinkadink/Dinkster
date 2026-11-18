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
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_environment_artifacts import _mirror
from test_environment_generations import _standalone_tree

from scripts import build_tools_release
from scripts.build_environment_tools import build
from scripts.build_release import sha256
from scripts.prepare_windows_python import require_uv_version

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("damage", "layout"),
    [(None, "fresh"), (None, "collision"), (None, "legacy")]
    + [
        (damage, "fresh")
        for damage in (
            "manifest",
            "archive",
            "missing-pointer",
            "unsigned-channel",
            "tampered-channel",
            "wrong-channel",
        )
    ],
)
def test_tools_cli_acquires_channel_publication_without_engine_payloads(
    tools_publication: tuple[Path, dict[str, bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    damage: str | None,
    layout: str,
) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from dinkster_supervisor import environment, environment_channel_signatures, environment_tools

    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes_raw()
    monkeypatch.setattr(
        environment_channel_signatures,
        "CHANNEL_KEYS",
        {"sha256:" + hashlib.sha256(public).hexdigest(): public},
    )
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
    (mirror / "channels/nightly.json").write_bytes(
        environment_channel_signatures.sign_channel(channel, "nightly", key)
    )
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
        (mirror / "channels/nightly.json").write_bytes(
            environment_channel_signatures.sign_channel(channel, "nightly", key)
        )
    elif damage in {"unsigned-channel", "tampered-channel", "wrong-channel"}:
        channel_path = mirror / "channels/nightly.json"
        if damage == "unsigned-channel":
            channel_path.write_text(json.dumps(channel), encoding="ascii")
        elif damage == "tampered-channel":
            channel_path.write_bytes(channel_path.read_bytes().replace(sha.encode(), b"0" * 64))
        else:
            channel_path.write_bytes(
                environment_channel_signatures.sign_channel(channel, "stable", key)
            )
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


@pytest.mark.parametrize("damage", [None, "unsigned", "tampered"])
def test_tools_publication_authenticates_and_preserves_channel_selections(
    tools_publication: tuple[Path, dict[str, bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str | None,
):
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
    from dinkster_supervisor import environment_channel_signatures
    from dinkster_supervisor.environment_state import EnvironmentError

    from scripts import publish_tools_release as publisher
    from tests.fixtures.environments.build import KEY, KEYS, channel_json, read_channel

    output, _ = tools_publication
    data = (output / "tools-manifest.json").read_bytes()
    manifest = json.loads(data)
    digest = hashlib.sha256(data).hexdigest()
    label = f"{manifest['label']}-{digest[:8]}"
    tag = f"tools-{label}"
    mirror = tmp_path / "mirror"
    _, _, selection = _mirror(mirror)
    selection["minimumDesktopVersion"] = "0.7.3"
    channel = mirror / "channels/nightly.json"
    channel.write_bytes(channel_json(selection))
    if damage == "unsigned":
        channel.write_text(json.dumps(selection), encoding="ascii")
    elif damage == "tampered":
        channel.write_bytes(channel.read_bytes().replace(b'"0.7.3"', b'"0.7.4"'))
    before = channel.read_bytes()
    monkeypatch.setattr(environment_channel_signatures, "CHANNEL_KEYS", KEYS)
    for name, value in {
        "GITHUB_REPOSITORY": "Kosinkadink/Dinkster",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": manifest["commit"],
        "R2_PUBLIC_BASE_URL": str(mirror),
        "DINKSTER_CHANNEL_SIGNING_KEY": KEY.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        ).decode("ascii"),
    }.items():
        monkeypatch.setenv(name, value)

    def github(command, **kwargs):
        if command[1:3] == ["release", "view"]:
            result = {
                "tagName": tag,
                "targetCommitish": manifest["commit"],
                "assets": [{"name": p.name, "size": p.stat().st_size} for p in output.iterdir()],
            }
        elif "/ref/tags/" in command[-1]:
            result = {"object": {"type": "tag", "sha": "a" * 40}}
        else:
            result = {
                "tag": tag,
                "message": json.dumps(
                    {
                        "version": manifest["version"],
                        "sourceCommit": manifest["commit"],
                        "manifest": digest,
                    }
                ),
                "object": {"type": "commit", "sha": manifest["commit"]},
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    fetch = publisher.fetch_artifact

    def fetch_artifact(sources, name, destination, digest, size):
        if sources[0].startswith("https://github.com/"):
            sources = (str(output),)
        return fetch(sources, name, destination, digest, size)

    def upload(client, source, key, **kwargs):
        destination = mirror / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(publisher.subprocess, "run", github)
    monkeypatch.setattr(publisher, "object_store", lambda: None)
    monkeypatch.setattr(publisher, "check_collision", lambda *args: None)
    monkeypatch.setattr(publisher, "write_index", lambda client, path, scope: path.write_text("[]"))
    monkeypatch.setattr(publisher, "fetch_artifact", fetch_artifact)
    monkeypatch.setattr(publisher, "upload", upload)
    if damage:
        with pytest.raises(EnvironmentError):
            publisher.publish(output)
        assert channel.read_bytes() == before
    else:
        publisher.publish(output)
        signed = read_channel(channel)
        assert signed.pop("tools") == {
            "label": label,
            "sha256": digest,
            "manifest": str(mirror / "tools" / label / "tools-manifest.json"),
            "release": str(mirror / "tools" / label),
        }
        selection.pop("signature", None)
        assert signed == dict(selection, channel="nightly")


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


def test_independent_tools_versions_keep_the_supervisor_release_line(
    tools_publication: tuple[Path, dict[str, bytes]], tmp_path: Path
):
    output, payloads = tools_publication
    development = json.loads((output / "tools-manifest.json").read_bytes())
    assert development["version"].startswith("2.7.5.dev")
    stable_output = tmp_path / "stable"
    stable_output.mkdir()
    for name, data in payloads.items():
        (stable_output / name).write_bytes(data)
    stable = build_tools_release.build(tmp_path, stable_output, stable=True)
    assert stable["version"] == "2.7.4"
    assert stable["commit"] == development["commit"]


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
    from dinkster_supervisor.environment_archives import extract_tools

    extract_tools(artifact, relocated)
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
    require_uv_version(str(bundled_uv))
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
            engine = root / f"builds/{current}/python"
            engine = engine / ("python.exe" if os.name == "nt" else "bin/python3")
            result = subprocess.check_output(
                [str(engine), "-I", "-c", "import fixture_engine; print(fixture_engine.VALUE)"],
                env=environment,
                cwd=tmp_path,
                text=True,
            )
            assert result.strip() == str(value)
        assert (root / "builds/2/build.json").is_file()
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

    from dinkster_library import LibraryClient
    from dinkster_supervisor import environment_channel_signatures
    from dinkster_supervisor.environment_state import EnvironmentError
    from dinkster_supervisor.environment_tools import TOOLS_ARCHIVES, library_command

    from tests.fixtures.environments.build import KEYS, channel_json

    monkeypatch.setattr(environment_channel_signatures, "CHANNEL_KEYS", KEYS)
    mirror = tmp_path / "tools mirror"
    _, _, channel = _mirror(mirror)
    publication = tmp_path / "new-tools"
    publication.mkdir()
    for name in TOOLS_ARCHIVES:
        target = publication / name
        if name != artifact.name:
            target.write_bytes(b"unused platform fixture")
            continue
        with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(target, "w") as archive:
            for member in source.infolist():
                data = source.read(member)
                if member.filename.endswith("/dinkster_library/__main__.py"):
                    # The fixture's API 2 aliases API 1; production offers only API 1.
                    data = data.replace(b'"apiMax": API_VERSION', b'"apiMax": API_VERSION + 1')
                    data = data.replace(
                        b"api != API_VERSION", b"api not in (API_VERSION, API_VERSION + 1)"
                    )
                archive.writestr(member, data)
    manifest = build_tools_release.build(ROOT, publication)
    identity = sha256(publication / "tools-manifest.json")
    label = f"{manifest['label']}-{identity[:8]}"
    release = mirror / "tools" / label
    shutil.move(str(publication), release)
    channel["tools"] = {
        "manifest": str(release / "tools-manifest.json"),
        "sha256": identity,
        "release": str(release),
        "label": label,
    }
    (mirror / "channels/nightly.json").write_bytes(channel_json(channel))
    project = tmp_path / "helper selection"
    sources = {"mirrors": [str(mirror)], "channel": "nightly"}
    project.mkdir()
    (project / "environment.json").write_text(json.dumps(sources))
    command = library_command(project, {"libraryApi": 2})
    assert Path(command[0]).is_relative_to(project / "tools")
    selection = (project / "library-tools.json").read_bytes()
    with LibraryClient(project / "library", command=command, api=2) as newer:
        created = newer.call("projects.register", id="film", name="Film")
    with pytest.raises(EnvironmentError, match="new helper offers 1 to 2"):
        library_command(project, {"libraryApi": 3}, sources=sources)
    assert (project / "library-tools.json").read_bytes() == selection
    shutil.rmtree(mirror)
    assert library_command(project, {"libraryApi": 1}) == command
    with LibraryClient(project / "library", command=command) as older:
        assert older.call("projects.get", id="film") == created
