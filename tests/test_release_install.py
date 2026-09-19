"""Release installs stay inside the extracted archive and preserve the lockfile."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import venv
import zipfile
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest
import yaml

from scripts.build_release import (
    build,
    desktop_windows_runtime,
    git_archive,
    vendor_identity,
    worker_protocol,
)
from scripts.install import install
from scripts.verify_release_install import stop


def test_artifact_install_does_not_activate_a_workspace_environment() -> None:
    root = Path(__file__).resolve().parent.parent
    workflow = yaml.safe_load((root / ".github/workflows/release.yml").read_text())
    assert workflow["jobs"]["install"]["env"]["UV_PYTHON"] == "3.12"
    steps = workflow["jobs"]["install"]["steps"]
    helper = yaml.safe_load(
        (root / ".github/actions/prepare-validation-inputs/action.yml").read_text()
    )
    (bootstrap,) = [step for step in steps if "GITHUB_PATH" in step.get("run", "")]
    (helper_bootstrap,) = [
        step for step in helper["runs"]["steps"] if "GITHUB_PATH" in step.get("run", "")
    ]
    assert bootstrap == helper_bootstrap
    assert all(
        steps.index(bootstrap) < index
        for index, step in enumerate(steps)
        if step.get("shell") == "bash"
    )
    (setup,) = [step for step in steps if step.get("uses") == "astral-sh/setup-uv@v5"]
    assert "python-version" not in setup["with"]
    assert setup["with"]["version"] == "latest"
    checkouts = [step for step in steps if step.get("uses", "").startswith("actions/checkout@")]
    assert [step["with"]["repository"] for step in checkouts] == ["Kosinkadink/dinkster-registry"]
    (extract,) = [step for step in steps if step.get("name", "").startswith("Extract the archive")]
    (install_step,) = [
        step for step in steps if step.get("name") == "Install from the archive without a checkout"
    ]
    for step in (extract, install_step):
        assert step["working-directory"] == "${{ runner.temp }}"
        assert step["run"].splitlines()[0].startswith("uv run --no-project --python 3.12 ")
    assert steps.index(extract) < steps.index(checkouts[0]) < steps.index(install_step)
    assert (
        'registry_command="$GITHUB_WORKSPACE/registry/.venv/bin/dinkster-registry-sqlite"'
        in (install_step["run"])
    )
    assert (
        'registry_command="$GITHUB_WORKSPACE/registry/.venv/Scripts/dinkster-registry-sqlite.exe"'
        in (install_step["run"])
    )
    assert '--registry-command "$registry_command"' in install_step["run"]


def test_release_registry_access_does_not_persist_credentials() -> None:
    root = Path(__file__).resolve().parent.parent
    workflow = yaml.safe_load((root / ".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["install"]["steps"]
    (access,) = [step for step in steps if step.get("uses") == "./.release-dependency-access"]
    assert access["with"] == {
        "repository": "Kosinkadink/dinkster-registry",
        "deploy-key": "${{ secrets.DINKSTER_REGISTRY_READ_KEY }}",
    }
    (extract,) = [step for step in steps if step.get("name", "").startswith("Extract the archive")]
    assert (
        "cp clean-install/dinkster-backend-${GITHUB_SHA}/.github/actions/"
        "configure-dinkster-identity/{action.yml,agent.cjs} "
        '"$GITHUB_WORKSPACE/.release-dependency-access/"'
    ) in extract["run"]
    (checkout,) = [step for step in steps if step.get("uses") == "actions/checkout@v4"]
    assert checkout["with"] == {
        "clean": True,
        "repository": "Kosinkadink/dinkster-registry",
        "ref": "772dd2a230e50dcc4f05c513c3f27279e0a6a652",
        "path": "registry",
        "persist-credentials": False,
    }
    (sync,) = [
        step for step in steps if step.get("name") == "Install the independent loopback registry"
    ]
    assert sync["run"].splitlines() == [
        "trap 'unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0 "
        "GIT_CONFIG_KEY_1 GIT_CONFIG_VALUE_1' EXIT",
        "uv sync --project registry --all-packages --frozen",
    ]
    assert sync["env"] == {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": (
            "url.https://x-access-token:${{ github.token }}"
            "@github.com/Kosinkadink/Dinkster.insteadOf"
        ),
        "GIT_CONFIG_VALUE_0": "https://github.com/Kosinkadink/Dinkster",
        "GIT_CONFIG_KEY_1": "credential.helper",
        "GIT_CONFIG_VALUE_1": "",
    }
    assert steps.index(extract) < steps.index(access) < steps.index(checkout) < steps.index(sync)


@pytest.fixture
def windows_runtime_source() -> dict[str, str]:
    profile = {
        "aimdo": {
            "repository": "Kosinkadink/dinkster-aimdo",
            "commit": "c" * 40,
            "releaseTag": "v0.6.0",
            "version": "0.6.0",
            "archive": "dinkster_aimdo-0.6.0-cp39-abi3-win_amd64.whl",
            "sha256": "d" * 64,
            "size": 123,
        },
        "cudaTorch": {
            "version": "2.14.0+cu130",
            "torchvisionVersion": "0.29.0",
            "cudaVersion": "13.0",
            "archive": "torch-2.14.0+cu130-cp312-cp312-win_amd64.whl",
            "url": "https://download-r2.pytorch.org/whl/cu130/fixture.whl",
            "sha256": "e" * 64,
            "size": 456,
        },
    }
    return {
        "scripts/desktop_windows_runtime.json": json.dumps(profile, indent=4) + "\n",
        "uv.lock": (
            '[[package]]\nname = "dinkster-aimdo"\nversion = "0.6.0"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
            'wheels = [\n  { url = "https://files.pythonhosted.org/packages/fixture/'
            'dinkster_aimdo-0.6.0-cp39-abi3-win_amd64.whl", '
            f'hash = "sha256:{"d" * 64}", size = 123 }}\n]\n\n'
            '[[package]]\nname = "torch"\nversion = "2.14.0"\n'
        ),
        "packages/dinkster-vision-birefnet/dinkster-pack.toml": (
            '[pack]\nrequires = ["torch==2.14.0", "torchvision==0.29.0"]\n'
        ),
    }


def test_repository_windows_runtime_matches_native_dependency_sources() -> None:
    root = Path(__file__).resolve().parent.parent
    profile = desktop_windows_runtime(root)
    assert profile == json.loads((root / "scripts/desktop_windows_runtime.json").read_text("utf-8"))


def test_archive_preserves_committed_bytes_with_windows_git_settings(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for key, value in (("core.autocrlf", "true"), ("core.eol", "crlf")):
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    (tmp_path / "guide.md").write_bytes(b"Install Dinkster\nRun the backend\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "guide.md"], check=True)
    tree = subprocess.check_output(["git", "-C", str(tmp_path), "write-tree"], text=True).strip()
    committed = subprocess.check_output(["git", "-C", str(tmp_path), "show", f"{tree}:guide.md"])
    archive = tmp_path / "source.zip"
    git_archive(tmp_path, tree, archive)
    with zipfile.ZipFile(archive) as source:
        assert source.read("guide.md") == committed == b"Install Dinkster\nRun the backend\n"


def test_stop_waits_for_venv_interpreter_file_handles(tmp_path: Path) -> None:
    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    for index in range(20):
        database = tmp_path / f"open-{index}.sqlite"
        with subprocess.Popen(
            [
                str(python),
                "-c",
                "import os, sqlite3, sys, time; connection = sqlite3.connect(sys.argv[1]); "
                "print(os.getpid(), flush=True); time.sleep(60)",
                str(database),
            ],
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                assert process.stdout is not None
                interpreter = psutil.Process(int(process.stdout.readline()))
            finally:
                stop(process)
            assert not interpreter.is_running()
        database.unlink()


def test_installer_pins_its_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "uv.lock").touch()
    (tmp_path / "packages").mkdir()
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "unrelated-environment")
    run = Mock()
    monkeypatch.setattr("scripts.install.subprocess.run", run)
    result = install(tmp_path, "uv")
    command = run.call_args.args[0]
    assert command == [
        "uv",
        "sync",
        "--project",
        str(tmp_path.resolve()),
        "--locked",
        "--no-dev",
        "--all-packages",
    ]
    assert run.call_args.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv")
    assert run.call_args.kwargs["check"] is True
    assert result == tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")


def test_installer_rejects_incomplete_archive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="complete Dinkster backend archive"):
        install(tmp_path, "uv")


@pytest.mark.parametrize("has_installer", [True, False])
def test_release_manifest_binds_archive_to_resolved_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_installer: bool,
    windows_runtime_source: dict[str, str],
) -> None:
    commit = "a" * 40
    monkeypatch.setattr("scripts.build_release.subprocess.check_output", Mock(return_value=commit))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/desktop_windows_runtime.json").write_text("invalid checkout metadata")

    def archive(root: Path, revision: str, destination: Path) -> None:
        assert root == tmp_path
        assert revision == commit
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("pyproject.toml", '[project]\nrequires-python = ">=3.11,<3.14"\n')
            archive.writestr(
                "packages/dinkster-workers/src/dinkster_workers/boundary.py",
                "PROTOCOL_VERSION = 9\n",
            )
            if has_installer:
                archive.writestr("scripts/install.py", "")
            for path, content in windows_runtime_source.items():
                archive.writestr(path, content)

    monkeypatch.setattr("scripts.build_release.git_archive", archive)
    monkeypatch.setattr("scripts.build_release.vendor_identity", Mock(return_value="b" * 40))
    output = tmp_path / "output"
    if not has_installer:
        with pytest.raises(ValueError, match="does not contain scripts/install.py"):
            build(tmp_path, output, "HEAD")
        assert not (output / "backend-release.json").exists()
        return
    artifact = build(tmp_path, output, "HEAD")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = json.loads((output / "backend-release.json").read_text())
    profile_path = "scripts/desktop_windows_runtime.json"
    assert manifest["desktopWindowsRuntime"] == json.loads(windows_runtime_source[profile_path])
    with zipfile.ZipFile(artifact) as release:
        assert release.read(f"dinkster-backend-{commit}/{profile_path}") == windows_runtime_source[
            profile_path
        ].encode("utf-8")
    assert manifest["repository"] == "Kosinkadink/Dinkster"
    assert manifest["commit"] == commit
    assert manifest["releaseTag"] == f"backend-{commit}"
    assert manifest["workerProtocol"] == 9
    assert manifest["requiresPython"] == ">=3.11,<3.14"
    assert manifest["bootstrap"] == {
        "tool": "uv",
        "python": "3.12",
        "requiresInternet": True,
        "requiresGit": False,
    }
    assert manifest["install"] == (
        f"uv run --no-project --python {manifest['bootstrap']['python']} scripts/install.py"
    )
    assert manifest["identityCommit"] == "b" * 40
    assert manifest["archive"] == artifact.name
    assert manifest["sha256"] == digest
    assert manifest["size"] == artifact.stat().st_size
    assert (output / "SHA256SUMS").read_text() == f"{digest}  {artifact.name}\n"


@pytest.mark.parametrize(
    ("path", "old", "new", "message"),
    [
        (
            "scripts/desktop_windows_runtime.json",
            '"version": "0.6.0"',
            '"version": "0.7.0"',
            "aimdo.version",
        ),
        (
            "scripts/desktop_windows_runtime.json",
            "dinkster_aimdo-0.6.0-cp39-abi3-win_amd64.whl",
            "missing.whl",
            "aimdo.archive",
        ),
        (
            "scripts/desktop_windows_runtime.json",
            "d" * 64,
            "f" * 64,
            "aimdo.sha256",
        ),
        ("uv.lock", 'version = "2.14.0"', 'version = "2.15.0"', "cudaTorch.version"),
        (
            "packages/dinkster-vision-birefnet/dinkster-pack.toml",
            "torchvision==0.29.0",
            "torchvision==0.30.0",
            r"torchvisionVersion.*dinkster-vision-birefnet/dinkster-pack.toml \[pack\].requires",
        ),
        (
            "packages/dinkster-vision-birefnet/dinkster-pack.toml",
            ', "torchvision==0.29.0"',
            "",
            r"torchvisionVersion.*dinkster-vision-birefnet/dinkster-pack.toml \[pack\].requires",
        ),
        (
            "packages/dinkster-vision-birefnet/dinkster-pack.toml",
            "torchvision==0.29.0",
            "torchvision>=0.29.0",
            r"torchvisionVersion.*dinkster-vision-birefnet/dinkster-pack.toml \[pack\].requires",
        ),
    ],
)
def test_windows_runtime_rejects_stale_source_pins(
    tmp_path: Path,
    windows_runtime_source: dict[str, str],
    path: str,
    old: str,
    new: str,
    message: str,
) -> None:
    windows_runtime_source[path] = windows_runtime_source[path].replace(old, new)
    for name, content in windows_runtime_source.items():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        desktop_windows_runtime(tmp_path)


def test_release_rejects_unknown_worker_protocol(tmp_path: Path) -> None:
    boundary = tmp_path / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("PROTOCOL_VERSION = compute_version()\n")
    with pytest.raises(ValueError, match="worker protocol"):
        worker_protocol(tmp_path)


@pytest.mark.parametrize("result", ["local", "changed-version", "git-source"])
def test_identity_bundle_preserves_versions_and_removes_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: str
) -> None:
    commit = "b" * 40
    git_source = f"https://github.com/Kosinkadink/dinkster-identity.git?rev={commit}#{commit}"
    lock = (
        '[[package]]\nname = "dinkster-identity"\nversion = "0.1.0"\n'
        f'source = {{ git = "{git_source}" }}\n'
    )
    (tmp_path / "uv.lock").write_text(lock)
    server = tmp_path / "packages/dinkster-server/pyproject.toml"
    server.parent.mkdir(parents=True)
    server.write_text(
        "[project]\ndependencies = ["
        f'"dinkster-identity @ git+https://github.com/Kosinkadink/dinkster-identity.git@{commit}"'
        "]\n[tool.uv.sources]\n"
    )

    def archive(root: Path, revision: str, destination: Path) -> None:
        assert revision == commit
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr(
                "pyproject.toml", '[project]\nname = "dinkster-identity"\nversion = "0.1.0"\n'
            )

    def relock(command: list[str], *, check: bool) -> None:
        assert check
        assert command == ["uv", "lock", "--project", str(tmp_path)]
        assert "dinkster-identity==0.1.0" in server.read_text()
        assert 'path = "../../vendor/dinkster-identity"' in server.read_text()
        released = lock
        if result != "git-source":
            released = released.replace(
                f'git = "{git_source}"', 'directory = "vendor/dinkster-identity"'
            )
        if result == "changed-version":
            released = released.replace('version = "0.1.0"', 'version = "0.2.0"')
        (tmp_path / "uv.lock").write_text(released)

    monkeypatch.setattr("scripts.build_release.git_archive", archive)
    monkeypatch.setattr("scripts.build_release.subprocess.run", relock)
    if result == "local":
        assert vendor_identity(tmp_path, tmp_path, tmp_path, "uv") == commit
        assert (tmp_path / "vendor/dinkster-identity/pyproject.toml").is_file()
    else:
        message = "dependency versions" if result == "changed-version" else "Git dependency"
        with pytest.raises(ValueError, match=message):
            vendor_identity(tmp_path, tmp_path, tmp_path, "uv")
