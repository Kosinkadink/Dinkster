"""The release builder produces one complete, version-bound artifact set."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml
from test_frontend_artifact import artifact

from scripts.build_release import (
    build,
    build_source_archive,
    development_version,
    download_external_wheels,
    extend_release,
    read_base_manifest,
    release_version,
    stage_external_wheels,
    versioned_sources,
    workspace_projects,
    write_constraints,
)
from scripts.download_frontend import verify_and_extract

ROOT = Path(__file__).resolve().parents[1]


def _distribution_wheels(tmp_path: Path, project: str) -> tuple[Path, Path]:
    direct = tmp_path / "direct"
    subprocess.run(
        ["uv", "build", "--package", project, "--wheel", "--out-dir", str(direct)],
        cwd=ROOT,
        check=True,
    )
    source = tmp_path / "source"
    subprocess.run(
        ["uv", "build", "--package", project, "--sdist", "--out-dir", str(source)],
        cwd=ROOT,
        check=True,
    )
    extracted = tmp_path / "isolated"
    with tarfile.open(next(source.glob("*.tar.gz"))) as archive:
        archive.extractall(extracted, filter="data")
    [package] = extracted.iterdir()
    rebuilt = tmp_path / "rebuilt"
    subprocess.run(
        ["uv", "build", str(package), "--wheel", "--out-dir", str(rebuilt)],
        cwd=extracted,
        check=True,
    )
    return next(direct.glob("*.whl")), next(rebuilt.glob("*.whl"))


@pytest.mark.parametrize("project", ["dinkster", "dinkster-api", "dinkster-supervisor"])
def test_source_distribution_rebuilds_the_complete_direct_wheel(
    tmp_path: Path, project: str
) -> None:
    direct, rebuilt = _distribution_wheels(tmp_path, project)
    with (
        zipfile.ZipFile(direct) as expected,
        zipfile.ZipFile(rebuilt) as actual,
    ):
        assert set(actual.namelist()) == set(expected.namelist())
        for name in expected.namelist():
            assert actual.read(name) == expected.read(name), name


@pytest.mark.network("pypi.org", "files.pythonhosted.org")
def test_rebuilt_api_wheel_installs_without_workspace_sources(tmp_path: Path) -> None:
    _, rebuilt = _distribution_wheels(tmp_path, "dinkster-api")
    environment = tmp_path / "standalone-api"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        cwd=tmp_path,
        check=True,
    )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-sources",
            str(rebuilt),
        ],
        cwd=tmp_path,
        check=True,
    )
    result = subprocess.check_output(
        [
            str(python),
            "-I",
            "-c",
            "from dinkster_api.v1 import Node; print(Node.__name__)",
        ],
        cwd=tmp_path,
        text=True,
    )
    assert result.strip() == "Node"


def test_built_frontend_wheel_is_discovered_without_a_checkout(tmp_path: Path) -> None:
    source = tmp_path / "frontend-source"
    source.mkdir()
    payload, pin = artifact()
    published = source / "frontend.zip"
    published.write_bytes(payload)
    uv = shutil.which("uv")
    assert uv is not None, "release packaging requires uv"
    with versioned_sources(ROOT, "0.0.1") as staged:
        verify_and_extract(payload, pin, staged / "src/dinkster/frontend_bundle/static")
        subprocess.run(
            [uv, "build", "--package", "dinkster", "--wheel", "--out-dir", str(tmp_path)],
            cwd=staged,
            check=True,
        )
    [wheel] = tmp_path.glob("*.whl")
    installed = tmp_path / "site-packages"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    shutil.rmtree(source)
    discovered = subprocess.check_output(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path[:0] = {[str(installed), str(ROOT / 'src')]!r}; "
            "from dinkster import frontend; "
            f"frontend.__file__ = {str(tmp_path / 'engine/src/dinkster/frontend.py')!r}; "
            "print(frontend.discover_frontend_bundle())",
        ],
        cwd=tmp_path,
        text=True,
    ).strip()
    assert discovered == str(installed / "dinkster/frontend_bundle/static")


def test_release_workflow_validates_tag_before_building_and_publishing() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    assert workflow[True]["push"] == {"branches": ["main"], "tags": ["v*.*.*"]}
    assert ".github/workflows/release.yml" in workflow[True]["pull_request"]["paths"]
    assert "src/dinkster/frontend.py" in workflow[True]["pull_request"]["paths"]
    assert set(workflow["jobs"]) == {
        "validation",
        "build",
        "tools",
        "finalize",
        "install",
        "release",
        "publish",
        "publish-api",
        "publish-supervisor",
    }
    assert workflow["jobs"]["validation"] == {
        "uses": "./.github/workflows/ci.yml",
    }
    assert workflow["jobs"]["build"]["needs"] == "validation"
    assert workflow["jobs"]["tools"]["needs"] == "build"
    assert workflow["jobs"]["finalize"]["needs"] == ["build", "tools"]
    assert workflow["jobs"]["install"]["needs"] == "finalize"
    assert workflow["jobs"]["tools"]["strategy"]["matrix"]["include"] == [
        {"os": "linux", "runner": "ubuntu-latest"},
        {"os": "windows", "runner": "windows-latest"},
    ]
    for job in ("install", "release", "publish", "publish-api", "publish-supervisor"):
        downloads = [
            step["with"]["name"]
            for step in workflow["jobs"][job]["steps"]
            if step.get("uses") == "actions/download-artifact@v4"
        ]
        assert downloads == ["engine-release"]
    assert workflow["jobs"]["release"]["needs"] == "install"
    assert workflow["jobs"]["release"]["if"] == "github.event_name == 'push'"
    assert workflow["jobs"]["release"]["permissions"] == {"contents": "write"}
    assert workflow["jobs"]["release"]["outputs"] == {
        "engine-id": "${{ steps.identity.outputs.engine-id }}",
        "engine-tag": "${{ steps.identity.outputs.tag }}",
    }
    preflight = next(
        step for step in workflow["jobs"]["release"]["steps"] if step.get("id") == "identity"
    )
    assert preflight["uses"] == "actions/github-script@v7"
    assert "getReleaseByTag" in preflight["with"]["script"]
    assert "Refusing to overwrite an existing release" in preflight["with"]["script"]
    assert next(
        step
        for step in workflow["jobs"]["release"]["steps"]
        if step.get("uses") == "softprops/action-gh-release@v2"
    ) == {
        "uses": "softprops/action-gh-release@v2",
        "with": {
            "tag_name": "${{ steps.identity.outputs.tag }}",
            "target_commitish": "${{ github.sha }}",
            "prerelease": "${{ github.ref == 'refs/heads/main' }}",
            "make_latest": "${{ github.ref != 'refs/heads/main' }}",
            "generate_release_notes": True,
            "fail_on_unmatched_files": True,
            "files": "dist/*",
        },
    }


@pytest.mark.parametrize(
    ("job", "project", "environment"),
    [
        ("publish", "dinkster", "pypi"),
        ("publish-api", "dinkster-api", "pypi-dinkster-api"),
        ("publish-supervisor", "dinkster-supervisor", "pypi-dinkster-supervisor"),
    ],
)
def test_release_publication_is_tag_only_and_uses_trusted_publishing(
    job: str, project: str, environment: str, tmp_path: Path
) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    assert workflow["jobs"]["release"]["if"] == "github.event_name == 'push'"
    publish = workflow["jobs"][job]
    assert publish["needs"] == "release"
    assert publish["if"] == ("github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')")
    assert "pypi.org/pypi/" not in json.dumps(workflow)
    assert "pypi-ready" not in json.dumps(workflow)
    assert publish["permissions"] == {"id-token": "write"}
    assert publish["environment"] == {
        "name": environment,
        "url": f"https://pypi.org/project/{project}/",
    }
    action = publish["steps"][-1]
    assert action["uses"] == "pypa/gh-action-pypi-publish@release/v1"
    assert action["with"] == {"packages-dir": "publish/", "skip-existing": True}
    assert "secrets." not in json.dumps(publish)
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in (
        "dinkster-1.2.3-py3-none-any.whl",
        "dinkster_api-1.2.3-py3-none-any.whl",
        "dinkster_supervisor-1.2.3-py3-none-any.whl",
        "dinkster_inference-1.2.3-py3-none-any.whl",
        "official-packs.lock",
        "dinkster-source-1.2.3.zip",
    ):
        (dist / name).write_bytes(name.encode())
    selection = next(step["run"] for step in publish["steps"] if "run" in step)
    bash = (
        str(Path(os.environ["ProgramFiles"]) / "Git/bin/bash.exe")
        if sys.platform == "win32"
        else "bash"
    )
    subprocess.run([bash, "-e", "-c", selection], cwd=tmp_path, check=True)
    selected = list((tmp_path / "publish").iterdir())
    assert [path.name for path in selected] == [
        f"{project.replace('-', '_')}-1.2.3-py3-none-any.whl"
    ]
    assert selected[0].read_bytes() == (dist / selected[0].name).read_bytes()
    assert all(
        "latest" in row["runner"]
        for row in workflow["jobs"]["install"]["strategy"]["matrix"]["include"]
    )


def test_release_workflow_builds_all_wheels_and_checks_tag_metadata() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    build = workflow["jobs"]["build"]
    assert build["permissions"] == {"actions": "read", "contents": "read"}
    commands = "\n".join(step.get("run", "") for step in build["steps"])
    assert "-m scripts.build_release" in commands
    assert "--download-external-wheels" in commands
    assert '--tag "$RELEASE_TAG"' in commands
    assert "sha256sum -c SHA256SUMS" in commands
    assert "--identity-root" not in commands
    assert "uvx twine check dist/*.whl" in commands
    assert "scripts/download_frontend.py" in commands
    assert "--frontend-artifact .release/frontend.zip" in commands
    assert "from scripts.build_official_packs import build" in commands
    assert 'arguments+=(--artifact "$payload")' in commands
    assert (
        "uv sync --locked --no-dev --package dinkster-registry --package dinkster-workers"
        in commands
    )
    assert "pnpm" not in commands
    assert "node" not in commands
    assert all(
        step.get("uses") not in {"pnpm/action-setup@v4", "actions/setup-node@v4"}
        and step.get("with", {}).get("repository") != "Kosinkadink/Dinkster-Frontend"
        for step in build["steps"]
    )
    frontend_download = next(
        step for step in build["steps"] if step.get("name") == "Download the pinned frontend bundle"
    )
    assert frontend_download["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert all(
        step.get("with", {}).get("repository") != "Kosinkadink/Dinkster-Frontend"
        for step in build["steps"]
        if step.get("uses") == "./.github/actions/configure-private-repository"
    )
    assert all(
        step.get("with", {}).get("repository") != "Kosinkadink/dinkster-identity"
        for step in build["steps"]
    )


def test_release_resolves_the_public_token_verifier() -> None:
    server = tomllib.loads(
        (ROOT / "packages/dinkster-server/pyproject.toml").read_text(encoding="utf-8")
    )
    verifier = "dinkster-token-verifier==0.1.2"
    assert verifier in server["project"]["dependencies"]
    assert "dinkster-identity" not in server["tool"]["uv"]["sources"]


def test_release_install_matrix_covers_supported_desktop_platforms() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    assert workflow["jobs"]["install"]["strategy"]["matrix"] == {
        "include": [
            {"os": "linux", "runner": "ubuntu-latest", "python": "3.14.10"},
            {"os": "windows", "runner": "windows-latest", "python": "3.14.10"},
            {"os": "macos", "runner": "macos-latest", "python": "3.12"},
        ]
    }
    bootstrap = workflow["jobs"]["install"]["steps"][0]
    assert bootstrap["name"] == "Expose Git Bash on Windows"
    assert bootstrap["if"] == "runner.os == 'Windows'"
    assert bootstrap["shell"] == "pwsh"
    assert "Git/bin" in bootstrap["run"]
    assert "bash.exe" in bootstrap["run"]
    assert "GITHUB_PATH" in bootstrap["run"]
    setup = next(
        step
        for step in workflow["jobs"]["install"]["steps"]
        if step.get("uses") == "astral-sh/setup-uv@v5"
    )
    assert setup["with"]["python-version"] == "${{ matrix.python }}"
    install = next(
        step
        for step in workflow["jobs"]["install"]["steps"]
        if step.get("name") == "Install and launch from wheels"
    )
    assert install["env"]["RELEASE_PYTHON"] == "${{ matrix.python }}"
    command = next(
        step["run"]
        for step in workflow["jobs"]["install"]["steps"]
        if step.get("name") == "Install and launch from wheels"
    )
    assert 'uv venv --seed --python "$RELEASE_PYTHON"' in command
    assert '"$python" -m pip install' in command
    assert '"$python" -m pip check' in command
    assert "--require-hashes" in command
    assert "--find-links" in command
    assert "--requirement" in command
    assert "from dinkster.frontend import discover_frontend_bundle" in command
    assert all(
        step.get("uses") != "./.github/actions/configure-dinkster-identity"
        for step in workflow["jobs"]["install"]["steps"]
    )


def test_release_install_launches_without_private_registry_inputs() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["install"]["steps"]
    assert "secrets." not in (ROOT / ".github/workflows/release.yml").read_text()
    assert all(
        step.get("with", {}).get("repository") != "Kosinkadink/dinkster-registry" for step in steps
    )
    wheel_install = next(
        step for step in steps if step.get("name") == "Install and launch from wheels"
    )
    assert "scripts/verify_release_install.py" in wheel_install["run"]
    assert "--require-frontend" in wheel_install["run"]
    assert "--registry-command" not in wheel_install["run"]


def test_independent_release_launch_serves_composition_and_nodes(tmp_path: Path) -> None:
    import sys

    from scripts.verify_release_install import verify_launch

    state = tmp_path / "release"
    verify_launch(state, Path(sys.executable))
    assert json.loads((state / "nodes.json").read_text())["nodes"] == {}
    assert (state / "composition.json").is_file()


@pytest.mark.parametrize("has_root", [True, False])
def test_release_http_check_requires_the_application_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_root: bool
) -> None:
    from scripts.verify_release_install import verify_launch

    module = tmp_path / "dinkster"
    shutil.copytree(ROOT / "src/dinkster", module)
    (module / "frontend_bundle/static").mkdir(parents=True)
    document = '<html><div id="root"></div></html>' if has_root else "<html>Error</html>"
    (module / "frontend_bundle/static/index.html").write_text(document)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    state = tmp_path / "release"
    if has_root:
        verify_launch(state, Path(sys.executable), require_frontend=True)
        assert (state / "index.html").read_text() == document
    else:
        with pytest.raises(AssertionError, match="application root"):
            verify_launch(state, Path(sys.executable), require_frontend=True)


def test_release_failure_prints_service_and_publication_log_tails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.verify_release_install import print_log_tails

    (tmp_path / "registry.log").write_text("\n".join(f"line {i}" for i in range(125)))
    (tmp_path / "backend.log").write_text("backend failed\n")
    (tmp_path / "publish.log").write_text("publication result\n")
    (tmp_path / "registry.db").write_text("not a log")
    print_log_tails(tmp_path)
    output = capsys.readouterr().err
    assert "backend failed" in output and "publication result" in output
    assert "line 24\n" not in output and "line 25\n" in output and "line 124\n" in output
    assert "not a log" not in output


def test_windows_runtime_admits_the_pinned_interpreter_before_installing_dependencies() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/full-validation.yml").read_text())
    steps = workflow["jobs"]["windows-runtime"]["steps"]
    setup_index = next(
        index for index, step in enumerate(steps) if step.get("uses") == "astral-sh/setup-uv@v5"
    )
    admission_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses") == "./.github/actions/windows-python"
    )
    sync_index = next(index for index, step in enumerate(steps) if "uv sync" in step.get("run", ""))
    assert steps[setup_index]["with"]["python-version"] == "3.14.10"
    assert setup_index < admission_index < sync_index
    assert "if" not in steps[admission_index]


def test_registry_install_keeps_state_and_uploads_it_even_on_failure() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/full-validation.yml").read_text())
    steps = workflow["jobs"]["registry-install"]["steps"]
    command = next(
        step["run"] for step in steps if "scripts/verify_release_install.py" in step.get("run", "")
    )
    assert '--state "$RUNNER_TEMP/registry-install-state"' in command
    upload = steps[-1]
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "${{ runner.temp }}/registry-install-state"


def test_probe_jail_is_configured_before_registry_validation_only() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/full-validation.yml").read_text())
    steps = workflow["jobs"]["registry-install"]["steps"]
    setup_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("run") == "bash scripts/setup_probe_jail.sh"
    )
    validation_index = next(
        index for index, step in enumerate(steps) if "uv sync" in step.get("run", "")
    )
    assert setup_index < validation_index
    assert "if" not in steps[setup_index]
    fast = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert all(
        "setup_probe_jail.sh" not in step.get("run", "") for step in fast["jobs"]["fast"]["steps"]
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CI probe-jail setup")
@pytest.mark.parametrize(
    "scenario, update_attempts, install_attempts, succeeds",
    [("retry", 2, 2, True), ("update-fails", 2, 0, False), ("install-fails", 1, 2, False)],
)
def test_probe_jail_apt_is_bounded_and_retries_once(
    scenario: str, update_attempts: int, install_attempts: int, succeeds: bool
) -> None:
    command = """
update_attempts=0
install_attempts=0
sudo() {
    printf 'sudo'; printf ' <%s>' "$@"; printf '\\n'
    if [[ "$1" != timeout ]]; then return 0; fi
    if [[ "$*" == *" update" ]]; then
        update_attempts=$((update_attempts + 1))
        if [[ "$scenario" == update-fails ]] ||
           [[ "$scenario" == retry && "$update_attempts" == 1 ]]; then return 124; fi
    else
        install_attempts=$((install_attempts + 1))
        if [[ "$scenario" == install-fails ]] ||
           [[ "$scenario" == retry && "$install_attempts" == 1 ]]; then return 124; fi
    fi
}
bwrap() { echo jail-preflight; }
source "$1"
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            'scenario="$2"\n' + command,
            "setup-test",
            str(ROOT / "scripts/setup_probe_jail.sh"),
            scenario,
        ],
        capture_output=True,
        text=True,
    )
    apt_calls = [line for line in result.stdout.splitlines() if line.startswith("sudo <timeout>")]
    assert sum(line.endswith("<update>") for line in apt_calls) == update_attempts
    assert sum("<install>" in line for line in apt_calls) == install_attempts
    for line in apt_calls:
        assert "<timeout> <180> <apt-get>" in line
        assert "<Acquire::Retries=3>" in line
        assert "<Acquire::http::Timeout=30>" in line
        assert "<Acquire::https::Timeout=30>" in line
    assert (result.returncode == 0) is succeeds
    assert ("jail-preflight" in result.stdout) is succeeds
    assert "failed or exceeded 180 seconds" in result.stderr


@pytest.mark.parametrize("tag", ["0.0.1", "v0.0", "v0.0.1-rc1", "backend-0.0.1"])
def test_release_version_rejects_non_version_tags(tag: str) -> None:
    with pytest.raises(ValueError, match="vX.Y.Z"):
        release_version(ROOT, tag)


def test_release_version_matches_every_workspace_project(tmp_path: Path) -> None:
    (tmp_path / "packages/one").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "root"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    package = tmp_path / "packages/one/pyproject.toml"
    package.write_text('[project]\nname = "one"\nversion = "1.2.3"\n', encoding="utf-8")
    assert release_version(tmp_path, "v1.2.3") == "1.2.3"
    package.write_text('[project]\nname = "one"\nversion = "1.2.4"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="one=1.2.4"):
        release_version(tmp_path, "v1.2.3")


def test_repository_versions_match_first_release_tag() -> None:
    assert release_version(ROOT, "v0.0.2") == "0.0.2"
    with pytest.raises(ValueError, match="does not match workspace metadata"):
        release_version(ROOT, "v0.0.1")
    projects = workspace_projects(ROOT)
    assert "dinkster-nodes-std" not in projects
    dev = tomllib.loads(projects["dinkster-nodes-dev"].read_text(encoding="utf-8"))
    assert "Private :: Do Not Upload" in dev["project"]["classifiers"]
    frontend = json.loads((ROOT / "scripts/release_sources.json").read_text(encoding="utf-8"))
    assert frontend["repository"] == "Kosinkadink/Dinkster-Frontend"
    assert len(frontend["commit"]) == 40
    assert len(frontend["sha256"]) == 64
    assert set(frontend) == {"repository", "commit", "sha256"}


def test_release_wheels_pin_the_workspace_and_frontend_without_changing_the_lock(
    tmp_path: Path,
) -> None:
    version = development_version(ROOT)
    assert version.startswith("0.0.3.dev")
    assert version.endswith(
        "+g" + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    original_lock = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    output = tmp_path / "wheels"
    with versioned_sources(ROOT, version) as staged:
        for project in ("dinkster", "dinkster-api"):
            subprocess.run(
                ["uv", "build", "--wheel", "--package", project, "--out-dir", str(output)],
                cwd=staged,
                check=True,
            )
    [wheel] = output.glob("dinkster-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        [name] = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = archive.read(name).decode()
        wheel_modules = {
            name
            for name in archive.namelist()
            if name.startswith("dinkster/") and name.endswith(".py")
        }
        for member in (
            "graph",
            "engine",
            "server",
            "native",
            "collab",
            "compat-comfy",
            "nodes-generation",
        ):
            namespace = "dinkster_" + member.replace("-", "_")
            source = ROOT / "packages" / f"dinkster-{member}" / "src"
            assert {
                name
                for name in archive.namelist()
                if name.startswith(namespace + "/") and name.endswith(".py")
            } == {path.relative_to(source).as_posix() for path in source.rglob("*.py")}
    [api_wheel] = output.glob("dinkster_api-*.whl")
    with zipfile.ZipFile(api_wheel) as archive:
        [name] = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        api_metadata = archive.read(name).decode()
        assert "Requires-Dist: torch" not in api_metadata
        assert {
            line
            for line in api_metadata.splitlines()
            if line.startswith("Requires-Dist: dinkster-")
        } == {"Requires-Dist: dinkster-opentimelineio==0.19.0"}
        for member in (
            "api",
            "values",
            "schema",
            "protocol",
            "memory",
            "assets",
            "caches",
            "inference-wire",
            "image-document",
            "video",
            "workers",
            "registry",
        ):
            namespace = "dinkster_" + member.replace("-", "_")
            source = ROOT / "packages" / f"dinkster-{member}" / "src"
            assert {
                name
                for name in archive.namelist()
                if name.startswith(namespace + "/")
                and "/preview/" not in name
                and name.endswith(".py")
            } == {path.relative_to(source).as_posix() for path in source.rglob("*.py")}
        assert "dinkster_video_preview/__init__.py" in archive.namelist()
        assert (
            "dinkster-doctor = dinkster_workers.doctor:main"
            in archive.read(f"dinkster_api-{version}.dist-info/entry_points.txt").decode()
        )
    source_archive = build_source_archive(ROOT, version, tmp_path)
    with zipfile.ZipFile(source_archive) as archive:
        source_modules = {
            name.removeprefix("src/")
            for name in archive.namelist()
            if name.startswith("src/dinkster/") and name.endswith(".py")
        }
    assert "dinkster/benchmark.py" in wheel_modules
    assert source_modules == wheel_modules
    assert f"Version: {version}\n" in metadata
    assert f"Requires-Dist: dinkster-api=={version}\n" in metadata
    assert "dinkster-frontend" not in metadata
    assert "Requires-Dist: dinkster-inference==0.1.0\n" in metadata
    assert "Requires-Dist: aiohttp>=3.10\n" in metadata
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(wheel)!r}); "
            "import dinkster; from importlib.metadata import version; "
            f"assert dinkster.__version__ == version('dinkster') == {version!r}",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest() == original_lock


def test_engine_constraints_include_the_locked_base_but_not_the_dev_tools(tmp_path: Path) -> None:
    constraints = write_constraints(ROOT, tmp_path, [], "uv").read_text()
    requirements = constraints.replace("\\\n", "").splitlines()
    for requirement in (
        "numpy==2.5.1",
        "opencv-contrib-python-headless==5.0.0.93",
    ):
        assert any(
            line.startswith(requirement + " ") and "--hash=sha256:" in line for line in requirements
        )
    assert "opencv-python-headless==" not in constraints
    assert "pytest==" not in constraints
    assert "pyright==" not in constraints
    assert "--index-url" not in constraints


@pytest.mark.parametrize("python", ["3.13.9", "3.14.10"])
def test_external_downloads_select_target_wheels_and_enforce_exported_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, python: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    records: dict[str, list[dict[str, object]]] = {}

    def wheel(package: str, tag: str) -> Path:
        path = source / f"{package}-1.0-{tag}.whl"
        path.write_bytes(path.name.encode())
        records.setdefault(package, []).append(
            {
                "url": path.as_uri(),
                "size": path.stat().st_size,
                "hash": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        return path

    tag = f"cp{python.split('.')[0]}{python.split('.')[1]}"
    shared = wheel("shared", "py3-none-any")
    win = wheel("native", f"{tag}-{tag}-win_amd64")
    linux = wheel("native", f"{tag}-{tag}-manylinux_2_28_x86_64")
    mac = wheel("native", f"{tag}-{tag}-macosx_14_0_arm64")
    wheel("native", "cp313-cp313-win_amd64")
    wheel("native", f"{tag}-{tag}-macosx_15_0_arm64")
    wheel("native", f"{tag}-{tag}-macosx_14_0_x86_64")
    wheel("native", "cp312-abi3-win_amd64")
    legacy = wheel("legacy", "py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64")
    oldest = wheel("oldest", "py3-none-manylinux1_x86_64.manylinux_2_5_x86_64")
    wheel("legacy", "py3-none-manylinux_2_29_x86_64")
    win_only = wheel("windows_only", "py3-none-any")
    mac_only = wheel("mac_only", "py3-none-any")
    lock = ""
    requirements = []
    for package, wheels in records.items():
        lock += f'[[package]]\nname = "{package}"\nversion = "1.0"\nwheels = [\n'
        for row in wheels:
            lock += (
                "{"
                + ", ".join(f"{key} = {json.dumps(value)}" for key, value in row.items())
                + "},\n"
            )
        lock += "]\n"
        marker = (
            f' ; sys_platform == "win32" and python_full_version >= "{python}"'
            if package == "windows_only"
            else (
                ' ; sys_platform == "darwin" and platform_machine == "arm64"'
                ' and platform_system == "Darwin"'
            )
            if package == "mac_only"
            else ' ; sys_platform == "linux"'
            if package in {"legacy", "oldest"}
            else ""
        )
        requirements.append(
            f"{package}==1.0{marker} " + " ".join(f"--hash={row['hash']}" for row in wheels)
        )
    (tmp_path / "uv.lock").write_text(lock)

    def export(root: Path, output: Path, wheels: list[Path], uv: str) -> Path:
        path = output / "constraints.txt"
        path.write_text("\n".join(requirements) + "\n")
        return path

    monkeypatch.setattr("scripts.build_release.write_constraints", export)
    manifest = {
        "format": "dinkster.base/1",
        "variant": "win-cpu",
        "platform": {"os": "windows", "arch": "x86_64"},
        "python": python,
        "interpreter": {},
        "wheels": [],
        "replaces": [],
        "workerProtocol": 1,
        "nvidia": None,
        "archive": {"format": "tar+zstd", "parts": []},
    }
    identity = {key: value for key, value in manifest.items() if key != "id"}
    identity["archive"] = {"format": "tar+zstd"}
    manifest["id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    base_manifest = tmp_path / "base-manifest.json"
    base_manifest.write_text(json.dumps(manifest))
    linux_manifest = {
        **manifest,
        "variant": "linux-cpu",
        "platform": {"os": "linux", "arch": "x86_64"},
    }
    identity = {key: value for key, value in linux_manifest.items() if key != "id"}
    identity["archive"] = {"format": "tar+zstd"}
    linux_manifest["id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    linux_base_manifest = tmp_path / "linux-base-manifest.json"
    linux_base_manifest.write_text(json.dumps(linux_manifest))
    mac_manifest = {
        **manifest,
        "variant": "mac-arm64",
        "platform": {"os": "macos", "arch": "arm64"},
    }
    identity = {key: value for key, value in mac_manifest.items() if key != "id"}
    identity["archive"] = {"format": "tar+zstd"}
    mac_manifest["id"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    )
    mac_base_manifest = tmp_path / "mac-base-manifest.json"
    mac_base_manifest.write_text(json.dumps(mac_manifest))
    base_manifests = [base_manifest, linux_base_manifest, mac_base_manifest]
    output = tmp_path / "downloads"
    result = download_external_wheels(tmp_path, output, "uv", base_manifests)
    assert {path.name for path in result} == {
        shared.name,
        win.name,
        linux.name,
        mac.name,
        legacy.name,
        oldest.name,
        win_only.name,
        mac_only.name,
    }
    assert all(path.read_bytes() == (source / path.name).read_bytes() for path in result)
    original = requirements[0]
    requirements[0] = "shared==1.0 --hash=sha256:" + "0" * 64
    with pytest.raises(ValueError, match=f"no locked {tag} wheel for shared"):
        download_external_wheels(tmp_path, tmp_path / "unauthorized", "uv", base_manifests)
    requirements[0] = original
    linux.unlink()
    with pytest.raises(OSError):
        download_external_wheels(tmp_path, tmp_path / "unavailable", "uv", base_manifests)
    assert not (tmp_path / "unavailable" / linux.name).exists()
    linux.write_bytes(linux.name.encode())
    shared.write_bytes(b"x" * shared.stat().st_size)
    with pytest.raises(ValueError, match="does not match lock"):
        download_external_wheels(tmp_path, tmp_path / "corrupt", "uv", base_manifests)
    assert not (tmp_path / "corrupt" / shared.name).exists()
    (tmp_path / "uv.lock").write_text(
        lock.replace(
            linux.as_uri(),
            linux.as_uri().replace(f"{tag}-{tag}-manylinux", "cp312-cp312-manylinux"),
        )
    )
    with pytest.raises(ValueError, match=f"no locked {tag} wheel for native on manylinux"):
        download_external_wheels(tmp_path, tmp_path / "missing-target", "uv", base_manifests)
    requirements[:] = ["shared @ https://example.invalid/shared.whl"]
    with pytest.raises(ValueError, match="unsupported direct wheel requirement"):
        download_external_wheels(tmp_path, tmp_path / "url-requirement", "uv", base_manifests)


@pytest.mark.parametrize(
    "change",
    [
        "plan",
        "multipart",
        "largest-part",
        "legacy-parts",
        "missing-archive",
        "archive-format",
        "parts-type",
        "part-type",
        "part-name",
        "part-order",
        "part-zero",
        "part-negative",
        "part-limit",
        "part-bool",
        "part-hash",
        "part-hash-type",
        "legacy-id",
        "parts-in-id",
    ],
)
def test_base_archive_contract_and_canonical_identity(tmp_path: Path, change: str) -> None:
    manifest: dict = {
        "format": "dinkster.base/1",
        "variant": "linux-cpu",
        "platform": {"os": "linux", "arch": "x86_64"},
        "python": "3.14.10",
        "interpreter": {},
        "wheels": [],
        "replaces": [],
        "workerProtocol": 11,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    manifest["id"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    part = {"filename": "base.tar.zst.001", "size": 1, "sha256": "a" * 64}
    archive = manifest["archive"]
    archive["parts"] = [part]
    if change == "plan":
        archive["parts"] = []
    elif change == "multipart":
        archive["parts"].append({**part, "filename": "base.tar.zst.002"})
    elif change == "largest-part":
        part["size"] = 1_999_999_999
    elif change == "legacy-parts":
        manifest["parts"] = manifest.pop("archive")["parts"]
    elif change == "missing-archive":
        del manifest["archive"]
    elif change == "archive-format":
        archive["format"] = "7z"
    elif change == "parts-type":
        archive["parts"] = {}
    elif change == "part-type":
        archive["parts"] = [None]
    elif change == "part-name":
        part["filename"] = "base.7z.001"
    elif change == "part-order":
        part["filename"] = "base.tar.zst.002"
    elif change.startswith("part-") and change.removeprefix("part-") in {
        "zero",
        "negative",
        "limit",
        "bool",
    }:
        part["size"] = {"zero": 0, "negative": -1, "limit": 2_000_000_000, "bool": True}[
            change.removeprefix("part-")
        ]
    elif change == "part-hash":
        part["sha256"] = "A" * 64
    elif change == "part-hash-type":
        part["sha256"] = None
    elif change in {"legacy-id", "parts-in-id"}:
        identity = {key: value for key, value in manifest.items() if key != "id"}
        if change == "legacy-id":
            del identity["archive"]
        manifest["id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest()
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    if change in {"plan", "multipart", "largest-part"}:
        assert read_base_manifest(path) == manifest
    else:
        with pytest.raises(ValueError, match="archive|id does not match"):
            read_base_manifest(path)


def test_external_payload_omits_only_exact_members_shared_by_all_target_bases(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidates"
    source.mkdir()
    wheels = []
    records = []
    for name in ("common", "transitive", "cpu_only", "changed_hash", "changed_size"):
        wheel = source / f"{name}-1.0-py3-none-any.whl"
        wheel.write_bytes(name.encode())
        wheels.append(wheel)
        records.append(
            {
                "name": name,
                "version": "1.0",
                "filename": wheel.name,
                "size": wheel.stat().st_size,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "index": "https://pypi.org/simple",
                "url": f"https://example.invalid/{wheel.name}",
            }
        )
    (tmp_path / "uv.lock").write_text(
        "[[package]]\nname = 'fixture'\nversion = '1.0'\nwheels = [\n"
        + "\n".join(
            f"{{url = '{row['url']}', size = {row['size']}, hash = 'sha256:{row['sha256']}'}},"
            for row in records
        )
        + "\n{url = 'https://example.invalid/unsized.whl', hash = 'sha256:"
        + "0" * 64
        + "'},"
        + "\n]\n"
    )
    manifests = []
    for variant in ("cpu", "cuda"):
        members = [dict(row) for row in records]
        if variant == "cuda":
            members = [row for row in members if row["name"] != "cpu_only"]
            next(row for row in members if row["name"] == "changed_hash")["sha256"] = "0" * 64
            next(row for row in members if row["name"] == "changed_size")["size"] += 1
        manifest = {
            "format": "dinkster.base/1",
            "variant": variant,
            "platform": {"os": "linux", "arch": "x86_64"},
            "python": "3.14.0",
            "interpreter": {},
            "wheels": members,
            "replaces": [row["name"] for row in members],
            "workerProtocol": 1,
            "nvidia": None,
            "archive": {"format": "tar+zstd"},
        }
        manifest["id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest()
        )
        manifest["archive"]["parts"] = [
            {"filename": "base.tar.zst.001", "size": 1, "sha256": "0" * 64}
        ]
        manifest["signature"] = None
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps(manifest))
        manifests.append(path)

    for label, bases, targets, expected in (
        ("shared", manifests, {"cpu", "cuda"}, {wheel.name for wheel in wheels[2:]}),
        ("none", [], {"cpu", "cuda"}, {wheel.name for wheel in wheels}),
        ("partial", manifests[:1], {"cpu", "cuda"}, {wheel.name for wheel in wheels}),
    ):
        output = tmp_path / label
        output.mkdir()
        staged = stage_external_wheels(tmp_path, output, wheels, bases, targets)
        assert {wheel.name for wheel in staged} == expected
        assert {wheel.name for wheel in output.iterdir()} == expected
        assert all(wheel.read_bytes() == (source / wheel.name).read_bytes() for wheel in staged)

    plans = []
    for path in manifests:
        manifest = json.loads(path.read_text())
        manifest["archive"]["parts"] = []
        plan = tmp_path / f"plan-{path.name}"
        plan.write_text(json.dumps(manifest))
        plans.append(plan)
    output = tmp_path / "plans"
    output.mkdir()
    assert {
        path.name
        for path in stage_external_wheels(tmp_path, output, wheels, plans, {"cpu", "cuda"})
    } == {wheel.name for wheel in wheels}

    with pytest.raises(ValueError, match="unexpected or duplicate base variant"):
        stage_external_wheels(tmp_path, tmp_path, [], manifests, {"cpu"})
    with pytest.raises(ValueError, match="unexpected or duplicate base variant"):
        stage_external_wheels(tmp_path, tmp_path, [], manifests[:1] * 2, {"cpu"})
    damaged = json.loads(manifests[0].read_text())
    damaged["python"] = "3.14.1"
    manifests[0].write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="id does not match"):
        stage_external_wheels(tmp_path, tmp_path, [], manifests[:1], {"cpu"})
    damaged["format"] = "another.base/1"
    manifests[0].write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="unsupported base manifest format"):
        stage_external_wheels(tmp_path, tmp_path, [], manifests[:1], {"cpu"})
    wheels[0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match a locked artifact"):
        stage_external_wheels(tmp_path, tmp_path, wheels[:1], [], set())


def test_release_payloads_are_hashed_but_never_published_as_workspace_wheels(
    tmp_path: Path,
) -> None:
    from dinkster_registry import LockedPack, Lockfile, build_artifact

    root = tmp_path / "source"
    frontend = tmp_path / "frontend.zip"
    frontend_payload, frontend_pin = artifact()
    frontend.write_bytes(frontend_payload)
    files = {
        "pyproject.toml": (
            '[project]\nname = "dinkster"\nversion = "1.2.3"\n'
            "[tool.uv.workspace]\n"
            'members = ["packages/dinkster-api", "packages/dinkster-supervisor"]\n'
            "[dependency-groups]\nengine-base = []\n"
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        ),
        "src/dinkster/__init__.py": "",
        "packages/dinkster-workers/src/dinkster_workers/boundary.py": "PROTOCOL_VERSION = 1\n",
        "scripts/release_sources.json": json.dumps(frontend_pin),
    }
    for member in ("api", "supervisor"):
        files[f"packages/dinkster-{member}/pyproject.toml"] = (
            f'[project]\nname = "dinkster-{member}"\nversion = "1.2.3"\n'
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        )
        files[f"packages/dinkster-{member}/src/dinkster_{member}/__init__.py"] = ""
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    subprocess.run(["uv", "lock", "--project", str(root)], check=True)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    tree = subprocess.check_output(["git", "-C", str(root), "write-tree"], text=True).strip()
    (root / ".git/HEAD").write_text(tree + "\n")
    payload = tmp_path / "official-pack-1.2.3.tar.gz"
    payload.write_bytes(b"finished pack archive")
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-nodes-foundation"\npublisher = "dinkster"\n'
        'namespaces = ["std"]\n[pack.entry]\nnodes = "nodes:NODES"\n'
    )
    (pack / "nodes.py").write_text("NODES = ()\n")
    archive = tmp_path / "dinkster-nodes-foundation-1.2.3.zip"
    digest = build_artifact(pack, archive)
    catalog = tmp_path / "official-packs.lock"
    catalog.write_text(
        Lockfile.of(
            [
                LockedPack(
                    "dinkster-nodes-foundation", "1.2.3", digest, "dinkster", ("std",), "release"
                )
            ]
        ).record_json()
    )
    bootstrap = tmp_path / ("dinkster-tools-1.2.4.dev123+g" + "a" * 40 + "-linux-x86_64.tar.gz")
    bootstrap.write_bytes(b"finished interpreter and supervisor payload")
    output = tmp_path / "release"
    payloads = [payload, bootstrap, archive, catalog]
    built = build(root, output, "v1.2.3", frontend, payload_artifacts=payloads)
    manifest = json.loads((output / "release-manifest.json").read_text())
    assert built == manifest
    assert set(manifest["workspaceWheels"]) == {
        "dinkster-1.2.3-py2.py3-none-any.whl",
        "dinkster_api-1.2.3-py2.py3-none-any.whl",
        "dinkster_supervisor-1.2.3-py2.py3-none-any.whl",
    }
    with zipfile.ZipFile(output / "dinkster-1.2.3-py2.py3-none-any.whl") as wheel:
        assert {
            name.removeprefix("dinkster/official_packs/"): wheel.read(name)
            for name in wheel.namelist()
            if name.startswith("dinkster/official_packs/")
        } == {path.name: path.read_bytes() for path in (archive, catalog)}
    for candidate in payloads:
        assert (output / candidate.name).read_bytes() == candidate.read_bytes()
        row = next(row for row in manifest["artifacts"] if row["name"] == candidate.name)
        assert row == {
            "name": candidate.name,
            "size": len(candidate.read_bytes()),
            "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
    sums = dict(
        line.split("  ", 1)[::-1] for line in (output / "SHA256SUMS").read_text().splitlines()
    )
    assert set(sums) == {row["name"] for row in manifest["artifacts"]} | {"release-manifest.json"}
    assert all(
        hashlib.sha256((output / name).read_bytes()).hexdigest() == value
        for name, value in sums.items()
    )
    for index, name in enumerate(
        (
            "constraints.txt",
            "RELEASE-MANIFEST.JSON",
            "SHA256SUMS",
            payload.name.upper(),
            "extra.whl",
            "escaped%2fpack.zip",
        )
    ):
        candidate = tmp_path / name
        candidate.write_bytes(b"must not replace a release asset or bypass locked wheels")
        with pytest.raises(ValueError, match="invalid or duplicate release payload name"):
            build(
                root,
                tmp_path / f"invalid-{index}",
                "v1.2.3",
                frontend,
                payload_artifacts=[payload, candidate],
            )


@pytest.mark.parametrize("corruption", [None, "payload", "sums", "extra", "duplicate"])
def test_native_payload_finalization_preserves_and_verifies_existing_release(
    tmp_path: Path, corruption: str | None
) -> None:
    output = tmp_path / "release"
    output.mkdir()
    wheel = output / "dinkster_supervisor-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"release-built supervisor wheel")
    constraints = output / "constraints.txt"
    constraints.write_bytes(b"pinned constraints\n")
    original = [constraints, wheel]
    manifest = {
        "format": "dinkster.engine/1",
        "commit": "a" * 40,
        "version": "1.2.3",
        "tag": "v1.2.3",
        "workerProtocol": 1,
        "workspaceWheels": [wheel.name],
        "artifacts": [
            {
                "name": path.name,
                "size": len(path.read_bytes()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in original
        ],
    }
    metadata = output / "release-manifest.json"
    metadata.write_text(json.dumps(manifest) + "\n")
    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in [*original, metadata]
        )
    )
    candidate = tmp_path / "dinkster-tools-windows-x86_64.zip"
    candidate.write_bytes(b"native archive")
    if corruption == "payload":
        wheel.write_bytes(b"wrong supervisor wheel")
    elif corruption == "sums":
        sums.write_text(sums.read_text().replace("  constraints.txt", "  wrong.txt"))
    elif corruption == "extra":
        (output / "unlisted.zip").write_bytes(b"unadmitted")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    if corruption:
        match = "payload name" if corruption == "duplicate" else "mismatch|unlisted"
        with pytest.raises(ValueError, match=match):
            extend_release(
                output, [candidate, candidate] if corruption == "duplicate" else [candidate]
            )
        assert {path.name: path.read_bytes() for path in output.iterdir()} == before
        return
    result = extend_release(output, [candidate])
    assert {key: value for key, value in result.items() if key != "artifacts"} == {
        key: value for key, value in manifest.items() if key != "artifacts"
    }
    assert result["artifacts"] == sorted(
        [
            *manifest["artifacts"],
            {
                "name": candidate.name,
                "size": 14,
                "sha256": hashlib.sha256(b"native archive").hexdigest(),
            },
        ],
        key=lambda row: row["name"],
    )
    for path in original:
        assert path.read_bytes() == before[path.name]
    assert (output / candidate.name).read_bytes() == b"native archive"
    final_sums = dict(line.split("  ", 1)[::-1] for line in sums.read_text().splitlines())
    assert set(final_sums) == {path.name for path in [*original, metadata, candidate]}
    for name, digest in final_sums.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert json.loads(metadata.read_bytes()) == result


def test_main_release_retention_preserves_stable_drafts_and_newest_fifty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.prune_engine_prereleases import prune

    releases = {
        number: {
            "id": number,
            "tag_name": f"main-{number:040x}",
            "published_at": f"2026-10-28T00:00:{number:02d}Z",
            "prerelease": True,
            "draft": False,
        }
        for number in range(1, 54)
    }
    for number, fields in (
        (101, {"prerelease": False}),
        (102, {"draft": True}),
        (103, {"tag_name": "other-prerelease"}),
    ):
        releases[number] = {**releases[1], "id": number, **fields}

    def read_api(command: list[str], *, text: bool) -> str:
        if "--paginate" in command:
            rows = list(releases.values())
            return json.dumps([rows[:20], rows[20:]])
        number = int(command[-1].rsplit("/", 1)[1])
        if number == 1:
            releases[number]["prerelease"] = False
        return json.dumps(releases[number])

    def delete_api(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["gh", "api", "--method", "DELETE"]
        number = int(command[-1].rsplit("/", 1)[1])
        del releases[number]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "check_output", read_api)
    monkeypatch.setattr(subprocess, "run", delete_api)
    assert set(prune("Kosinkadink/Dinkster")) == {2, 3}
    assert set(releases) == {1, *range(4, 54), 101, 102, 103}
    assert releases[1]["prerelease"] is False


@pytest.mark.parametrize("system", ["windows", "macos"])
@pytest.mark.parametrize("python", ["3.13.9", "3.14.10"])
def test_selected_published_bases_are_qualified_and_preserved_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, python: str, system: str
) -> None:
    from copy import deepcopy

    from scripts.acquire_engine_bases import acquire

    boundary = tmp_path / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("PROTOCOL_VERSION = 11\n")
    pins = {"torch": "2.14.1", "torchvision": "0.29.1", "numpy": "2.5.1"}
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "dinkster"\nrequires-python = ">=3.12"\n'
        "[dependency-groups]\nengine-base = " + json.dumps(["numpy==2.5.1"])
    )
    tag = f"cp{python.split('.')[0]}{python.split('.')[1]}"
    windows = system == "windows"
    variant = "win-cpu" if windows else "mac-arm64"
    target = "win_amd64" if windows else "macosx_14_0_arm64"
    triple = "x86_64-pc-windows-msvc" if windows else "aarch64-apple-darwin"
    executable = "python/python.exe" if windows else "python/bin/python3"
    lock = []
    wheels = []
    for name, version in pins.items():
        torch = windows and name in {"torch", "torchvision"}
        filename = f"{name}-{version}{'+cpu' if torch else ''}-{tag}-{tag}-{target}.whl"
        digest = hashlib.sha256(name.encode()).hexdigest()
        url = f"https://example.invalid/{filename}"
        wheels.append(
            dict(
                name=name,
                version=version + ("+cpu" if torch else ""),
                filename=filename,
                size=17,
                sha256=digest,
                url=url,
                index="https://download.pytorch.org/whl/cpu"
                if torch
                else "https://pypi.org/simple",
                verification="index" if torch else "lock",
                lockedVersion=version,
            )
        )
        lock.append(
            f"[[package]]\nname = '{name}'\nversion = '{version}'\n"
            f"wheels = [{{url = '{url}', size = 17, hash = 'sha256:{digest}'}}]\n"
        )
    (tmp_path / "uv.lock").write_text("\n".join(lock))
    original: dict = dict(
        format="dinkster.base/1",
        variant=variant,
        platform=dict(os=system, arch="x86_64" if windows else "arm64"),
        python=python,
        interpreter=dict(
            filename=f"cpython-{python}+20261026-{triple}-install_only_stripped.tar.gz",
            release="20261026",
            executable=executable,
            size=23,
            sha256="a" * 64,
            url="https://github.com/astral-sh/python-build-standalone/releases/download/20261026/"
            f"cpython-{python}%2B20261026-{triple}-install_only_stripped.tar.gz",
        ),
        wheels=wheels,
        replaces=list(pins),
        workerProtocol=11,
        nvidia=None,
        archive=dict(format="tar+zstd"),
    )
    part_digest = hashlib.sha256(b"archive part").hexdigest()
    mutation = "valid"
    payload = b""
    extras_payload = b""
    requested = ""

    def prepare(change: str) -> str:
        nonlocal payload, extras_payload, mutation, requested
        mutation = change
        manifest = deepcopy(original)
        if change == "version":
            manifest["wheels"][-1]["version"] = "2.5.0"
            manifest["wheels"][-1]["filename"] = f"numpy-2.5.0-{tag}-{tag}-{target}.whl"
        elif change == "wheel-hash":
            manifest["wheels"][-1]["sha256"] = "0" * 64
        elif change == "torch-index":
            manifest["wheels"][0]["index"] = "https://download.pytorch.org/whl/cu132"
        elif change == "protocol":
            manifest["workerProtocol"] = 10
        elif change == "python":
            manifest["python"] = "3.11.9"
        elif change == "interpreter":
            manifest["interpreter"]["executable"] = (
                "python/bin/python3" if windows else "python/python.exe"
            )
        elif change == "wheel-platform":
            manifest["wheels"][0]["filename"] = (
                f"torch-2.14.1+cpu-{tag}-{tag}-manylinux_2_28_x86_64.whl"
            )
        elif change == "missing-pin":
            manifest["wheels"] = manifest["wheels"][:-1]
        identity = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        manifest["id"] = "sha256:" + identity
        manifest["archive"]["parts"] = (
            []
            if change == "plan"
            else [dict(filename="base.tar.zst.001", size=12, sha256=part_digest)]
        )
        manifest["signature"] = None
        requested = f"base-{variant}-{identity}"
        if change == "identity":
            manifest["workerProtocol"] = 12
        payload = (json.dumps(manifest, indent=3) + "\n").encode()
        extras = dict(format="dinkster.extras/1", base=manifest["id"], variant=variant, wheels=[])
        if change == "extras-base":
            extras["base"] = "sha256:" + "0" * 64
        elif change == "extras-variant":
            extras["variant"] = "linux-cpu"
        elif change == "extras-format":
            extras["format"] = "dinkster.extras/2"
        extras_payload = b"{" if change == "extras-json" else json.dumps(extras).encode()
        return requested

    def transport(command: list[str]) -> bytes:
        if "/releases/tags/" not in command[-1]:
            return extras_payload if command[-1].endswith("/2") else payload
        if mutation == "missing-release":
            raise subprocess.CalledProcessError(1, command)
        assets: list[dict] = [
            dict(name="manifest.json", id=1, size=len(payload), state="uploaded"),
            dict(
                name="extras.json",
                id=2,
                size=len(extras_payload) + (mutation == "extras-size"),
                state="uploaded",
                digest=None
                if mutation == "stream"
                else "sha256:"
                + (
                    "0" * 64
                    if mutation == "extras-hash"
                    else hashlib.sha256(extras_payload).hexdigest()
                ),
            ),
        ]
        if mutation != "missing-part":
            assets.append(
                dict(
                    name="base.tar.zst.001",
                    id=3,
                    size=12,
                    state="uploaded",
                    digest=None
                    if mutation.startswith("stream")
                    else "sha256:" + ("0" * 64 if mutation == "part-hash" else part_digest),
                )
            )
        return json.dumps(
            dict(
                tag_name=requested,
                draft=mutation == "draft",
                published_at="2026-10-28T00:00:00Z",
                assets=assets,
            )
        ).encode()

    monkeypatch.setattr(subprocess, "check_output", transport)
    popen = subprocess.Popen

    def stream_part(command: list[str], **kwargs):
        data = b"short" if mutation == "stream-short" else b"archive part"
        if mutation == "stream-corrupt":
            data = b"archive port"
        return popen(
            [
                sys.executable,
                "-c",
                f"import sys; sys.stdout.buffer.write({data!r}); "
                f"sys.exit({1 if mutation == 'stream-error' else 0})",
            ],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", stream_part)
    output = tmp_path / "engine-base-manifests"
    paths = acquire(tmp_path, [prepare("valid")], output)
    assert paths == [output / variant / "manifest.json"]
    assert paths[0].read_bytes() == payload
    streamed = acquire(tmp_path, [prepare("stream")], tmp_path / "streamed")
    assert streamed[0].read_bytes() == payload
    for change in (
        "version",
        "wheel-hash",
        "torch-index",
        "protocol",
        "python",
        "interpreter",
        "wheel-platform",
        "missing-pin",
        "plan",
        "identity",
        "missing-release",
        "missing-part",
        "part-hash",
        "draft",
        "extras-base",
        "extras-variant",
        "extras-format",
        "extras-json",
        "extras-size",
        "extras-hash",
        "stream-corrupt",
        "stream-short",
        "stream-error",
    ):
        destination = tmp_path / change
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            acquire(tmp_path, [prepare(change)], destination)
        assert not destination.exists()
    duplicate = tmp_path / "duplicate"
    tag = prepare("valid")
    with pytest.raises(FileExistsError):
        acquire(tmp_path, [tag, tag], duplicate)
    assert not duplicate.exists()
    with pytest.raises(ValueError, match="exact variant/id"):
        acquire(tmp_path, ["latest"], tmp_path / "latest")
    assert not list(tmp_path.glob("dinkster-base-selection-*"))


def test_maintainer_source_archive_excludes_non_release_material(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    files = {
        "src/package.py": "released\n",
        "tests/test_package.py": "excluded\n",
        "tools/generate.py": "excluded\n",
        "benchmarks/measure.py": "excluded\n",
        "scripts/comfyui_benchmark_nodes/fixture.py": "excluded\n",
        "scripts/run_benchmark.py": "released\n",
        "src/benchmark.py": "released\n",
        "src/package/benchmarks/runtime.py": "released\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    tree = subprocess.run(
        ["git", "-C", str(root), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (root / ".git/HEAD").write_text(f"{tree}\n", encoding="ascii")
    archive = build_source_archive(root, "1.2.3", tmp_path)
    with zipfile.ZipFile(archive) as source:
        assert {name for name in source.namelist() if not name.endswith("/")} == {
            name for name, content in files.items() if content == "released\n"
        }
