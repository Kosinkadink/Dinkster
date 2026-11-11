"""The release builder produces one complete, version-bound artifact set."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml
from test_frontend_artifact import artifact

from scripts.build_release import (
    build,
    build_source_archive,
    check_release,
    development_version,
    extend_release,
    inputs_changed,
    read_base_manifest,
    release_version,
    select_dependency_wheels,
    validate_engine_wheels,
    versioned_sources,
    wheel_metadata,
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


@pytest.mark.parametrize(
    "project",
    ["dinkster", "dinkster-api", "dinkster-engine", "dinkster-supervisor", "dinkster-workers"],
)
def test_source_distribution_rebuilds_the_complete_direct_wheel(
    tmp_path: Path, project: str
) -> None:
    direct, rebuilt = _distribution_wheels(tmp_path, project)
    with (
        zipfile.ZipFile(direct) as expected,
        zipfile.ZipFile(rebuilt) as actual,
    ):
        if project == "dinkster":
            assert expected.read("dinkster/py.typed") == b""
            assert actual.read("dinkster/py.typed") == b""
        assert set(actual.namelist()) == set(expected.namelist())
        for name in expected.namelist():
            assert actual.read(name) == expected.read(name), name
    if project not in {"dinkster", "dinkster-engine"}:
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                f"import sys; sys.path.insert(0, {str(rebuilt)!r}); "
                "from pathlib import Path; from dinkster_store import object_path; "
                f"root = Path({str(tmp_path / 'store')!r}); "
                "first = object_path(root, 'sha256:' + 'a' * 64, label='same-label'); "
                "second = object_path(root, 'sha256:' + 'b' * 64, label='same-label'); "
                "assert first == root / 'same-label'; "
                "assert second == root / 'same-label--2'; "
                "assert object_path(root, 'sha256:' + 'a' * 64, label='renamed') == first",
            ],
            cwd=tmp_path,
            check=True,
        )


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
        "changes",
        "validation",
        "build",
        "finalize",
        "install",
        "release",
        "channel",
        "publish",
        "publish-api",
        "publish-engine",
        "publish-supervisor",
    }
    assert workflow["jobs"]["validation"] == {
        "needs": "changes",
        "if": "needs.changes.outputs.engine == 'true' && startsWith(github.ref, 'refs/tags/v')",
        "uses": "./.github/workflows/ci.yml",
    }
    assert workflow["jobs"]["build"]["needs"] == ["changes", "validation"]
    assert workflow["jobs"]["build"]["if"] == (
        "always() && !cancelled() && needs.changes.outputs.engine == 'true' && "
        "(needs.validation.result == 'success' || (needs.validation.result == 'skipped' && "
        "(github.event_name == 'pull_request' || "
        "(github.event_name == 'push' && github.ref == 'refs/heads/main'))))"
    )
    assert workflow["jobs"]["finalize"]["needs"] == "build"
    assert workflow["jobs"]["finalize"]["if"] == (
        "${{ !cancelled() && needs.build.result == 'success' }}"
    )
    assert workflow["jobs"]["install"]["needs"] == "finalize"
    assert workflow["jobs"]["install"]["if"] == (
        "${{ !cancelled() && needs.finalize.result == 'success' }}"
    )
    from types import SimpleNamespace

    for job, parent in (("finalize", "build"), ("install", "finalize")):
        condition = workflow["jobs"][job]["if"].removeprefix("${{").removesuffix("}}")
        condition = condition.replace("&&", " and ").replace("!", "not ").strip()
        for result in ("success", "failure", "skipped"):
            for cancelled in (False, True):
                assert eval(
                    condition,
                    {
                        "needs": SimpleNamespace(**{parent: SimpleNamespace(result=result)}),
                        "cancelled": lambda cancelled=cancelled: cancelled,
                    },
                ) is (result == "success" and not cancelled)
    commands = "\n".join(
        step.get("run", "") for job in workflow["jobs"].values() for step in job.get("steps", [])
    )
    assert "build_official_packs" not in commands
    assert "build_release_pack_locks" not in commands
    assert "official-packs" not in commands
    assert "pack-locks" not in commands
    assert "lock-matrix" not in workflow["jobs"]["build"].get("outputs", {})
    admission = next(step["run"] for step in workflow["jobs"]["finalize"]["steps"] if "run" in step)
    assert "uv sync --locked --no-dev --package dinkster-supervisor" in admission
    assert "build_native_tools" not in commands
    assert ".release/tools" not in commands
    for job in ("install", "release", "publish", "publish-api", "publish-supervisor"):
        downloads = [
            step["with"]["name"]
            for step in workflow["jobs"][job]["steps"]
            if step.get("uses") == "actions/download-artifact@v4"
        ]
        assert downloads == ["engine-release"]
    assert workflow["jobs"]["release"]["needs"] == "install"
    assert workflow["jobs"]["release"]["if"] == (
        "${{ !cancelled() && needs.install.result == 'success' && github.event_name == 'push' }}"
    )
    assert workflow["jobs"]["release"]["permissions"] == {"contents": "write"}
    channel = workflow["jobs"]["channel"]
    assert channel["needs"] == "release"
    assert channel["permissions"] == {"contents": "write"}
    assert channel["if"] == (
        "${{ !cancelled() && needs.release.result == 'success' && github.event_name == 'push' }}"
    )
    assert channel["concurrency"]["cancel-in-progress"] is False
    publication = next(
        step
        for step in channel["steps"]
        if "--command python -m scripts.publish_environment_channel" in step.get("run", "")
    )
    retention = next(step for step in channel["steps"] if "--scope engine" in step.get("run", ""))
    assert channel["steps"].index(retention) > channel["steps"].index(publication)
    assert retention["if"] == "github.ref == 'refs/heads/main'"
    assert "--dry-run" not in retention["run"]
    assert "--old-naming" in retention["run"]
    assert "--output retention-report.json" in retention["run"]
    assert "continue-on-error" not in retention
    assert (
        sum(
            "--scope engine" in step.get("run", "")
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
        )
        == 1
    )
    assert "prune_engine_prereleases" not in commands
    assert "retention.py" in publication["run"]
    assert next(
        step["with"]
        for step in channel["steps"]
        if step.get("with", {}).get("repository") == "Kosinkadink/Dinkster-Environments"
    ) == {
        "repository": "Kosinkadink/Dinkster-Environments",
        "ref": "022b8234976065d0d3344a9717dd150ec6347bfa",
        "path": ".release/retention",
        "persist-credentials": False,
    }
    assert "uv sync --locked --only-group release-publication" in publication["run"]
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]
    assert "dinkster-supervisor" in groups["release-publication"]
    assert any(requirement.startswith("boto3") for requirement in groups["release-publication"])
    assert "-m scripts.publish_environment_channel" in publication["run"]
    assert "--minimum-desktop-version 0.2.0" in publication["run"]
    assert publication["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert publication["env"]["R2_ACCESS_KEY_ID"] == "${{ secrets.R2_ACCESS_KEY_ID }}"
    assert any(
        "dinkster_supervisor.environment_channels" in step.get("run", "")
        for step in workflow["jobs"]["finalize"]["steps"]
    )
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
    release_steps = workflow["jobs"]["release"]["steps"]
    collision_check = next(step for step in release_steps if "--preflight" in step.get("run", ""))
    assert release_steps.index(collision_check) < release_steps.index(preflight)
    assert "R2_BUCKET" in collision_check["env"]
    assert "digest.slice(0, 8)" in preflight["with"]["script"]
    assert "`v${manifest.version}`" in preflight["with"]["script"]
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


def test_release_acquires_each_selected_base_without_pack_lock_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import acquire_engine_bases

    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    selection = next(
        step["run"]
        for step in workflow["jobs"]["build"]["steps"]
        if step.get("name") == "Acquire the selected published base manifests"
    )
    paths = []
    for variant, system in (
        ("linux-cpu", "linux"),
        ("linux-nvidia", "linux"),
        ("mac-arm64", "macos"),
        ("win-cpu", "windows"),
        ("win-nvidia", "windows"),
        ("win-nvidia-cu126", "windows"),
    ):
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps({"variant": variant, "platform": {"os": system}}))
        paths.append(path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/base_bundles.json").write_text(
        json.dumps({path.stem: str(index) for index, path in enumerate(paths)})
    )
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.chdir(tmp_path)
    requested = []

    def acquire(root, references, destination):
        requested.extend(references)
        assert root == tmp_path
        assert destination == tmp_path / ".release/base-manifests"
        return paths

    monkeypatch.setattr(acquire_engine_bases, "acquire", acquire)
    script = selection.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    exec(compile(script, "release base selection", "exec"), {})
    assert requested == [
        "base-linux-cpu-0",
        "base-linux-nvidia-1",
        "base-mac-arm64-2",
        "base-win-cpu-3",
        "base-win-nvidia-4",
        "base-win-nvidia-cu126-5",
    ]
    assert not output.exists()


@pytest.mark.parametrize(
    ("job", "project", "environment"),
    [
        ("publish", "dinkster", "pypi"),
        ("publish-api", "dinkster-api", "pypi-dinkster-api"),
        ("publish-engine", "dinkster-engine", "pypi-dinkster-engine"),
        ("publish-supervisor", "dinkster-supervisor", "pypi-dinkster-supervisor"),
    ],
)
def test_release_publication_is_tag_only_and_uses_trusted_publishing(
    job: str, project: str, environment: str, tmp_path: Path
) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    assert workflow["jobs"]["release"]["if"] == (
        "${{ !cancelled() && needs.install.result == 'success' && github.event_name == 'push' }}"
    )
    publish = workflow["jobs"][job]
    assert publish["needs"] == "release"
    assert publish["if"] == (
        "${{ !cancelled() && needs.release.result == 'success' && github.event_name == 'push' "
        "&& startsWith(github.ref, 'refs/tags/v') }}"
    )
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
        "dinkster_engine-1.2.3-py3-none-any.whl",
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


def test_default_pack_admission_uses_complete_version_history() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/default-pack-admission.yml").read_text(encoding="utf-8")
    )
    triggers = workflow[True]
    assert "push" not in triggers
    assert triggers["workflow_run"] == {
        "workflows": ["Independent official pack publication"],
        "types": ["completed"],
        "branches": ["main"],
    }
    assert "pull_request" in triggers
    selection = workflow["jobs"]["select"]
    assert "github.event.workflow_run.event == 'workflow_run'" in selection["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in selection["if"]
    assert (
        "github.event.workflow_run.head_repository.full_name == 'Kosinkadink/Dinkster'"
        in (selection["if"])
    )
    publication = next(step for step in selection["steps"] if step.get("id") == "publication")
    assert "job.name === 'publish' && job.conclusion === 'success'" in publication["with"]["script"]
    identity = next(step for step in selection["steps"] if step.get("id") == "identity")
    assert (
        "core.setOutput('engine', context.payload.pull_request.head.sha)"
        in identity["with"]["script"]
    )
    assert "core.setOutput('engine', engine.commit)" in identity["with"]["script"]
    assert "engine.repository !== 'Kosinkadink/Dinkster'" in identity["with"]["script"]
    checkout = next(
        step
        for step in workflow["jobs"]["admission"]["steps"]
        if step.get("uses") == "actions/checkout@v4"
    )
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["ref"] == "${{ needs.select.outputs.engine }}"
    for job in (selection, workflow["jobs"]["prepare"]):
        download = next(
            step for step in job["steps"] if step.get("uses") == "actions/download-artifact@v4"
        )
        assert download["with"]["name"] == "official-publication-inputs"
        assert download["with"]["run-id"] == "${{ github.event.workflow_run.id }}"
    command = next(
        step["run"] for step in workflow["jobs"]["prepare"]["steps"] if step.get("id") == "sources"
    )
    assert '--candidate-base "$CANDIDATE_BASE"' in command
    assert "--published-sources .publication/sources/official-packs.lock" in command


@pytest.mark.parametrize(
    ("changed", "pull_request", "all_candidates"),
    [
        (".github/workflows/official-packs.yml", True, True),
        (".github/workflows/default-pack-admission.yml", True, False),
        ("packages/dinkster-text/dinkster-pack.toml", True, False),
        (".github/workflows/official-packs.yml", False, False),
    ],
)
def test_default_admission_workflow_selects_publication_candidates(
    tmp_path: Path, changed: str, pull_request: bool, all_candidates: bool
) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    base = subprocess.check_output(["git", "mktree"], cwd=tmp_path, input="", text=True).strip()
    path = tmp_path / changed
    path.parent.mkdir(parents=True)
    path.write_text("changed\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    tree = subprocess.check_output(["git", "write-tree"], cwd=tmp_path, text=True).strip()
    original = subprocess.check_output(["git", "cat-file", "-p", "HEAD"], cwd=ROOT, text=True)
    metadata = "\n".join(
        line for line in original.splitlines() if line.startswith(("author ", "committer "))
    )
    head = (
        subprocess.check_output(
            ["git", "hash-object", "-w", "-t", "commit", "--stdin"],
            cwd=tmp_path,
            input=f"tree {tree}\n{metadata}\n\nPublication fixture\n".encode(),
        )
        .decode("ascii")
        .strip()
    )
    subprocess.run(["git", "update-ref", "HEAD", head], cwd=tmp_path, check=True)
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/default-pack-admission.yml").read_text(encoding="utf-8")
    )
    command = next(
        step["run"] for step in workflow["jobs"]["prepare"]["steps"] if step.get("id") == "sources"
    )
    stub = """
    uv() {
      if [ "$4" = "-c" ]; then
        echo development-engine
      elif [ "$4" = "-" ]; then
        "$TEST_PYTHON" -
      else
        printf '%s\\0' "$@" > admission-arguments
        echo '{"groups": [["dinkster-image"], ["dinkster-audio", "dinkster-video"]]}'
      fi
    }
    """
    bash = (
        str(Path(os.environ["ProgramFiles"]) / "Git/bin/bash.exe")
        if sys.platform == "win32"
        else "bash"
    )
    subprocess.run(
        [bash, "-e", "-o", "pipefail", "-c", stub + command],
        cwd=tmp_path,
        env=os.environ
        | {
            "CANDIDATE_BASE": base if pull_request else "",
            "PUBLISHED_VERSION": "1.2.3",
            "GITHUB_OUTPUT": str(tmp_path / "outputs"),
            "TEST_PYTHON": sys.executable,
        },
        check=True,
    )
    arguments = (tmp_path / "admission-arguments").read_bytes().decode().split("\0")[:-1]
    assert arguments == [
        "run",
        "--no-sync",
        "python",
        "-m",
        "scripts.check_default_pack_admission",
        "--engine-version",
        "development-engine" if pull_request else "1.2.3",
        "--candidate-base" if pull_request else "--published-sources",
        ("0" * 40 if all_candidates else base)
        if pull_request
        else ".publication/sources/official-packs.lock",
        "--prepare-output",
        "admission-sources",
    ]
    groups = json.loads((tmp_path / "outputs").read_text().removeprefix("groups="))
    assert groups == [
        {"group": 1, "packs": '["dinkster-image"]'},
        {"group": 2, "packs": '["dinkster-audio", "dinkster-video"]'},
    ]
    prepared = tmp_path / "admission-sources"
    prepared.mkdir()
    (prepared / "engine.json").write_text('{"engineVersion": "prepared-engine"}')
    command = workflow["jobs"]["admission"]["steps"][-1]["run"]
    subprocess.run(
        [bash, "-e", "-o", "pipefail", "-c", stub + command],
        cwd=tmp_path,
        env=os.environ | {"TEST_PACKS": groups[1]["packs"]},
        check=True,
    )
    assert (tmp_path / "admission-arguments").read_bytes().decode().split("\0")[:-1] == [
        "run",
        "--no-sync",
        "python",
        "-m",
        "scripts.check_default_pack_admission",
        "--engine-version",
        "prepared-engine",
        "--prepared-sources",
        "admission-sources",
        "--pack",
        "dinkster-audio",
        "--pack",
        "dinkster-video",
    ]


def test_release_workflow_builds_all_wheels_and_checks_tag_metadata() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    build = workflow["jobs"]["build"]
    assert build["permissions"] == {"actions": "read", "contents": "read"}
    checkout = next(step for step in build["steps"] if step.get("uses") == "actions/checkout@v4")
    assert checkout["with"]["fetch-depth"] == 0
    commands = "\n".join(step.get("run", "") for step in build["steps"])
    assert "-m scripts.build_release" in commands
    assert "--download-external-wheels" not in commands
    assert 'arguments+=(--base-manifest "$manifest")' in commands
    final_checks = "\n".join(step.get("run", "") for step in workflow["jobs"]["finalize"]["steps"])
    assert "--check --max-bytes 100000000" in final_checks
    assert '--tag "$RELEASE_TAG"' in commands
    assert "sha256sum -c SHA256SUMS" in commands
    assert "--identity-root" not in commands
    assert "uvx twine check dist/*.whl" in commands
    assert "scripts/download_frontend.py" in commands
    assert "--frontend-artifact .release/frontend.zip" in commands
    assert "build_official_packs" not in commands
    assert "--artifact" not in commands
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
    assert 'uv pip install --python "$python"' in command
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
    assert "secrets." not in yaml.safe_dump(
        {
            "env": workflow.get("env"),
            "defaults": workflow.get("defaults"),
            "install": workflow["jobs"]["install"],
        }
    )
    assert all(
        step.get("with", {}).get("repository") != "Kosinkadink/dinkster-registry" for step in steps
    )
    wheel_install = next(
        step for step in steps if step.get("name") == "Install and launch from wheels"
    )
    assert "scripts/verify_release_launch.py" in wheel_install["run"]
    assert "--require-frontend" in wheel_install["run"]
    assert "--registry-command" not in wheel_install["run"]


def test_independent_release_launch_serves_composition_and_nodes(tmp_path: Path) -> None:
    import sys

    from scripts.verify_release_launch import verify_launch

    state = tmp_path / "release"
    verify_launch(state, Path(sys.executable))
    assert json.loads((state / "nodes.json").read_text())["nodes"] == {}
    assert (state / "composition.json").is_file()


@pytest.mark.parametrize("has_root", [True, False])
def test_release_http_check_requires_the_application_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_root: bool
) -> None:
    from scripts.verify_release_launch import verify_launch

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


def test_release_failure_prints_backend_log_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.verify_release_launch import print_log_tails

    (tmp_path / "backend.log").write_text("\n".join(f"line {i}" for i in range(125)))
    (tmp_path / "nodes.json").write_text("not a log")
    print_log_tails(tmp_path)
    output = capsys.readouterr().err
    assert "backend.log" in output
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


def test_probe_jail_is_configured_before_live_tests_and_required_by_main() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/full-validation.yml").read_text())
    steps = workflow["jobs"]["probe-jail"]["steps"]
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
    command = steps[validation_index]["run"]
    assert "tests/test_sandbox.py::TestLiveJail" in command
    assert "tests/test_doctor.py::TestLiveJailedProbe" in command
    main_status = workflow["jobs"]["main-status"]
    assert "probe-jail" in main_status["needs"]
    gate = main_status["steps"][-1]
    assert gate["env"]["PROBE_JAIL_RESULT"] == "${{ needs.probe-jail.result }}"
    assert 'test "$PROBE_JAIL_RESULT" = success' in gate["run"]
    fast = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert all(
        "setup_probe_jail.sh" not in step.get("run", "")
        for job in fast["jobs"].values()
        for step in job["steps"]
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
    supervisor = tmp_path / "packages/supervisor/pyproject.toml"
    supervisor.parent.mkdir()
    supervisor.write_text('[project]\nname = "dinkster-supervisor"\nversion = "7.8.9"\n')
    for name, manifest in (
        ("dinkster-nodes-remote", "dinkster-pack.toml"),
        ("dinkster-nodes-vision", "hed_pack/dinkster-pack.toml"),
    ):
        pack = tmp_path / "packages" / name
        (pack / manifest).parent.mkdir(parents=True)
        (pack / manifest).write_text('[pack]\nversion = "4.5.6"\n')
        (pack / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "4.5.6"\n')
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "root"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    package = tmp_path / "packages/one/pyproject.toml"
    package.write_text('[project]\nname = "one"\nversion = "1.2.3"\n', encoding="utf-8")
    assert release_version(tmp_path, "v1.2.3") == "1.2.3"
    package.write_text('[project]\nname = "one"\nversion = "1.2.4"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="one=1.2.4"):
        release_version(tmp_path, "v1.2.3")
    metadata = package.read_text()
    package.write_text(metadata + "\n[tool.dinkster]\nofficial-library = true\n")
    assert release_version(tmp_path, "v1.2.3") == "1.2.3"
    package.write_text(metadata + "\n[tool.dinkster]\nofficial-library = false\n")
    with pytest.raises(ValueError, match="one=1.2.4"):
        release_version(tmp_path, "v1.2.3")


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ("supervisor", False),
        ("supervisor-lock", False),
        ("packages/dinkster-supervisor/shared/dinkster_store/__init__.py", True),
        ("engine-lock", True),
        ("engine", True),
        ("scripts/build_native_tools.py", False),
        ("scripts/build_environment_tools.py", False),
        ("scripts/build_tools_release.py", False),
        ("scripts/publish_tools_release.py", False),
        (".github/workflows/release-tools.yml", False),
    ],
)
def test_engine_publication_ignores_only_supervisor_changes(
    tmp_path: Path, changed: str, expected: bool
) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    lock = (
        '[[package]]\nname = "dinkster-supervisor"\nversion = "1.0"\n'
        '[[package]]\nname = "aiohttp"\nversion = "2.0"\n'
    )
    (tmp_path / "uv.lock").write_text(lock)

    def snapshot() -> str:
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        return subprocess.check_output(["git", "write-tree"], cwd=tmp_path, text=True).strip()

    before = snapshot()
    supervisor = tmp_path / "packages/dinkster-supervisor/client.py"
    supervisor.parent.mkdir(parents=True)
    supervisor.write_text("new_behavior = True\n")
    if changed.endswith("lock"):
        (tmp_path / "uv.lock").write_text(
            lock.replace('version = "1.0"', 'version = "1.1"')
            if changed == "supervisor-lock"
            else lock.replace('version = "2.0"', 'version = "2.1"')
        )
    elif changed == "engine":
        (tmp_path / "engine.py").write_text("new_engine = True\n")
    elif "/" in changed:
        source = tmp_path / changed
        source.parent.mkdir(parents=True)
        source.write_text("tools input\n")
    assert inputs_changed(tmp_path, before, snapshot()) is expected


def test_repository_versions_match_engine_release_tag() -> None:
    assert release_version(ROOT, "v0.0.3") == "0.0.3"
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from pathlib import Path; from scripts.build_release import release_version; "
            "assert release_version(Path.cwd(), 'v0.0.3') == '0.0.3'",
        ],
        cwd=ROOT,
        check=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )
    with pytest.raises(ValueError, match="does not match workspace metadata"):
        release_version(ROOT, "v0.0.2")
    projects = workspace_projects(ROOT)
    assert "dinkster-nodes-std" not in projects
    dev = tomllib.loads(projects["dinkster-nodes-dev"].read_text(encoding="utf-8"))
    assert "Private :: Do Not Upload" in dev["project"]["classifiers"]
    frontend = json.loads((ROOT / "scripts/release_sources.json").read_text(encoding="utf-8"))
    assert frontend["repository"] == "Kosinkadink/Dinkster-Frontend"
    assert len(frontend["commit"]) == 40
    assert len(frontend["sha256"]) == 64
    assert set(frontend) == {"repository", "commit", "sha256"}


def test_development_version_counts_commits_since_the_last_version_tag(tmp_path: Path) -> None:
    root = tmp_path / "history"
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "1.2.3"\n')
    tree = subprocess.check_output(["git", "-C", str(root), "mktree"], input="", text=True).strip()
    original = subprocess.check_output(
        ["git", "-C", str(ROOT), "cat-file", "-p", "HEAD"], text=True
    )
    metadata = "\n".join(
        line for line in original.splitlines() if line.startswith(("author ", "committer "))
    )
    head = ""
    for number in range(3):
        parent = f"parent {head}\n" if head else ""
        head = (
            subprocess.check_output(
                ["git", "-C", str(root), "hash-object", "-w", "-t", "commit", "--stdin"],
                input=f"tree {tree}\n{parent}{metadata}\n\nFixture commit {number}\n".encode(),
            )
            .decode("ascii")
            .strip()
        )
    subprocess.run(["git", "-C", str(root), "update-ref", "HEAD", head], check=True)
    assert development_version(root) == f"1.2.4.dev3+g{head}"
    subprocess.run(["git", "-C", str(root), "update-ref", "refs/tags/v1.2.3", "HEAD~2"], check=True)
    assert development_version(root) == f"1.2.4.dev2+g{head}"
    subprocess.run(
        ["git", "-C", str(root), "update-ref", "refs/tags/main-latest", "HEAD"], check=True
    )
    assert development_version(root) == f"1.2.4.dev2+g{head}"
    subprocess.run(["git", "-C", str(root), "update-ref", "refs/tags/v1.2.4", "HEAD~1"], check=True)
    assert development_version(root) == f"1.2.4.dev1+g{head}"
    subprocess.run(["git", "-C", str(root), "update-ref", "refs/tags/v1.2.5", "HEAD"], check=True)
    assert development_version(root) == f"1.2.4.dev0+g{head}"


def test_release_wheels_pin_the_workspace_and_frontend_without_changing_the_lock(
    tmp_path: Path,
) -> None:
    version = development_version(ROOT)
    assert version.startswith("0.0.4.dev")
    assert version.endswith(
        "+g" + subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    original_lock = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    output = tmp_path / "wheels"
    with versioned_sources(ROOT, version) as staged:
        for member in ("remote", "image", "dev"):
            relative = f"packages/dinkster-nodes-{member}/pyproject.toml"
            committed = tomllib.loads(
                subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT, text=True)
            )["project"]["version"]
            actual = tomllib.loads((staged / relative).read_text())["project"]["version"]
            assert actual == (committed if member in {"remote", "image"} else version)
        helper = tomllib.loads(
            (staged / "packages/dinkster-pack-helpers/pyproject.toml").read_text()
        )["project"]
        assert helper["version"] == "0.0.2"
        models = tomllib.loads(
            (staged / "packages/dinkster-nodes-models/pyproject.toml").read_text()
        )["project"]
        assert "dinkster-pack-helpers==0.0.2" in models["dependencies"]
        for project in ("dinkster", "dinkster-api", "dinkster-engine", "dinkster-supervisor"):
            subprocess.run(
                ["uv", "build", "--wheel", "--package", project, "--out-dir", str(output)],
                cwd=staged,
                check=True,
            )
    validate_engine_wheels(list(output.glob("*.whl")))
    [wheel] = output.glob("dinkster-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        [name] = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = archive.read(name).decode()
        supervisor_version = tomllib.loads(
            (ROOT / "packages/dinkster-supervisor/pyproject.toml").read_text()
        )["project"]["version"]
        assert f"Requires-Dist: dinkster-supervisor=={supervisor_version}\n" in metadata
        wheel_modules = {
            name
            for name in archive.namelist()
            if name.startswith("dinkster/") and name.endswith(".py")
        }
        for member in (
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
    [supervisor_wheel] = output.glob("dinkster_supervisor-*.whl")
    assert supervisor_wheel.name == f"dinkster_supervisor-{supervisor_version}-py3-none-any.whl"
    [api_wheel] = output.glob("dinkster_api-*.whl")
    with zipfile.ZipFile(api_wheel) as archive:
        [name] = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        api_metadata = archive.read(name).decode()
        assert "Requires-Dist: torch" not in api_metadata
        assert {
            line
            for line in api_metadata.splitlines()
            if line.startswith("Requires-Dist: dinkster-")
        } == {
            "Requires-Dist: dinkster-opentimelineio==0.19.0",
        }
        assert "dinkster_store/__init__.py" in archive.namelist()
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
    assert f"Requires-Dist: dinkster-engine=={version}\n" in metadata
    assert "dinkster-frontend" not in metadata
    assert "Requires-Dist: dinkster-inference==0.1.1\n" not in metadata
    assert "Requires-Dist: dinkster-inference==0.1.1; extra == 'inference'\n" in metadata
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


@pytest.mark.parametrize("namespace", ["dinkster_engine", "dinkster_graph"])
@pytest.mark.parametrize("owner", ["dinkster", "dinkster-api", "dinkster-supervisor"])
def test_release_refuses_duplicate_engine_modules(
    tmp_path: Path, namespace: str, owner: str
) -> None:
    wheel = tmp_path / f"{owner.replace('-', '_')}-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{owner.replace('-', '_')}-1.2.3.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {owner}\nVersion: 1.2.3\n",
        )
        archive.writestr(f"{namespace}/__init__.py", "")
    with pytest.raises(ValueError, match="must ship only in dinkster-engine"):
        validate_engine_wheels([wheel])


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
def test_dependency_metadata_selects_target_wheels_and_enforces_exported_hashes(
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
        lock += (
            f'[[package]]\nname = "{package}"\nversion = "1.0"\n'
            'source = { registry = "https://pypi.org/simple" }\nwheels = [\n'
        )
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
    result = select_dependency_wheels(tmp_path, "uv", base_manifests)
    assert {row["filename"] for selection in result.values() for row in selection["wheels"]} == {
        shared.name,
        win.name,
        linux.name,
        mac.name,
        legacy.name,
        oldest.name,
        win_only.name,
        mac_only.name,
    }
    for selection in result.values():
        for row in selection["wheels"]:
            assert row["size"] == (source / row["filename"]).stat().st_size
            assert (
                row["sha256"] == hashlib.sha256((source / row["filename"]).read_bytes()).hexdigest()
            )
            assert row["index"] == "https://pypi.org/simple"
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    linux.unlink()
    assert select_dependency_wheels(tmp_path, "uv", base_manifests) == result
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before - {
        linux.relative_to(tmp_path)
    }
    original = requirements[0]
    requirements[0] = "shared==1.0 --hash=sha256:" + "0" * 64
    with pytest.raises(ValueError, match=f"no locked {tag} wheel for shared"):
        select_dependency_wheels(tmp_path, "uv", base_manifests)
    requirements[0] = original
    (tmp_path / "uv.lock").write_text(
        lock.replace(
            linux.as_uri(),
            linux.as_uri().replace(f"{tag}-{tag}-manylinux", "cp312-cp312-manylinux"),
        )
    )
    with pytest.raises(ValueError, match=f"no locked {tag} wheel for native on manylinux"):
        select_dependency_wheels(tmp_path, "uv", base_manifests)
    requirements[:] = ["shared @ https://example.invalid/shared.whl"]
    with pytest.raises(ValueError, match="unsupported direct wheel requirement"):
        select_dependency_wheels(tmp_path, "uv", base_manifests)


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


def test_dependency_metadata_uses_producer_ownership_names_per_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
            transitive = next(row for row in members if row["name"] == "transitive")
            transitive["version"] = "1.2"
            transitive["filename"] = transitive["filename"].replace("-1.0-", "-1.2-")
            transitive["url"] = transitive["url"].replace("-1.0-", "-1.2-")
        manifest = {
            "format": "dinkster.base/1",
            "variant": variant,
            "platform": {"os": "linux", "arch": "x86_64"},
            "python": "3.14.0",
            "interpreter": {},
            "wheels": members,
            "replaces": [row["name"].replace("_", "-") for row in records],
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

    def export(root: Path, output: Path, wheels: list[Path], uv: str) -> Path:
        path = output / "constraints.txt"
        path.write_text(
            "\n".join(f"{row['name']}==1.0 --hash=sha256:{row['sha256']}" for row in records)
        )
        return path

    monkeypatch.setattr("scripts.build_release.write_constraints", export)
    # The two bases deliberately disagree on bytes and membership; ownership is by name.
    for path in manifests:
        manifest = json.loads(path.read_text())
        for row in manifest["wheels"]:
            row["name"] = row["name"].replace("_", "-")
        identity = {key: value for key, value in manifest.items() if key not in {"id", "signature"}}
        identity["archive"] = {"format": "tar+zstd"}
        manifest["id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest()
        )
        path.write_text(json.dumps(manifest))
    result = select_dependency_wheels(tmp_path, "uv", manifests)
    assert (
        next(row for row in result["cuda"]["wheels"] if row["name"] == "transitive")["version"]
        == "1.2"
    )
    assert {row["name"] for row in result["cpu"]["wheels"]} == {
        "common",
        "transitive",
        "cpu-only",
        "changed-hash",
        "changed-size",
    }
    assert {row["name"] for row in result["cuda"]["wheels"]} == {
        "common",
        "transitive",
        "changed-hash",
        "changed-size",
    }
    assert (
        next(row for row in result["cuda"]["wheels"] if row["name"] == "changed-hash")["sha256"]
        == "0" * 64
    )
    assert (
        next(row for row in result["cuda"]["wheels"] if row["name"] == "changed-size")["size"]
        == records[-1]["size"] + 1
    )
    with pytest.raises(ValueError, match="select at least one"):
        select_dependency_wheels(tmp_path, "uv", [])
    with pytest.raises(ValueError, match="duplicate base variant"):
        select_dependency_wheels(tmp_path, "uv", manifests[:1] * 2)
    damaged = json.loads(manifests[0].read_text())
    damaged["python"] = "3.14.1"
    manifests[0].write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="id does not match"):
        select_dependency_wheels(tmp_path, "uv", manifests[:1])
    damaged["format"] = "another.base/1"
    manifests[0].write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="unsupported base manifest format"):
        select_dependency_wheels(tmp_path, "uv", manifests[:1])


@pytest.mark.parametrize(
    ("tag", "version", "public_version"),
    [
        ("v1.2.3", "1.2.3", "1.2.3"),
        (None, "1.2.4.dev412+g" + "a" * 40, "1.2.4.dev412"),
    ],
)
def test_release_payloads_are_hashed_but_never_published_as_workspace_wheels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tag: str | None,
    version: str,
    public_version: str,
) -> None:
    from dinkster_registry import LockedPack, Lockfile, build_artifact

    root = tmp_path / "source"
    frontend = tmp_path / "frontend.zip"
    frontend_payload, frontend_pin = artifact()
    frontend.write_bytes(frontend_payload)
    inference_wheel = tmp_path / "dinkster_inference-0.1.1-py3-none-any.whl"
    with zipfile.ZipFile(inference_wheel, "w") as wheel:
        metadata = "dinkster_inference-0.1.1.dist-info"
        wheel.writestr(
            f"{metadata}/METADATA",
            "Metadata-Version: 2.1\nName: dinkster-inference\nVersion: 0.1.1\n",
        )
        wheel.writestr(
            f"{metadata}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        wheel.writestr(f"{metadata}/RECORD", "")
    monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
    files = {
        "pyproject.toml": (
            '[project]\nname = "dinkster"\nversion = "1.2.3"\n'
            'dependencies = ["dinkster-engine==1.2.3"]\n'
            '[project.optional-dependencies]\ninference = ["dinkster-inference==0.1.1"]\n'
            "[tool.uv.workspace]\n"
            'members = ["packages/dinkster-api", "packages/dinkster-engine", '
            '"packages/dinkster-supervisor"]\n'
            "[tool.uv.sources]\ndinkster-engine = { workspace = true }\n"
            "[dependency-groups]\nengine-base = []\n"
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        ),
        "src/dinkster/__init__.py": "",
        "src/dinkster/default-packs.toml": '[packs]\nfixture-pack = "1.2.3"\n',
        "packages/dinkster-workers/src/dinkster_workers/boundary.py": "PROTOCOL_VERSION = 1\n",
        "scripts/release_sources.json": json.dumps(frontend_pin),
    }
    for member in ("api", "engine", "supervisor"):
        files[f"packages/dinkster-{member}/pyproject.toml"] = (
            f'[project]\nname = "dinkster-{member}"\nversion = "1.2.3"\n'
            '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        )
        files[f"packages/dinkster-{member}/src/dinkster_{member}/__init__.py"] = ""
    files["packages/dinkster-engine/pyproject.toml"] += (
        '[tool.hatch.build.targets.wheel.force-include]\n"src/dinkster_graph" = "dinkster_graph"\n'
    )
    files["packages/dinkster-engine/pyproject.toml"] = files[
        "packages/dinkster-engine/pyproject.toml"
    ].replace(
        "[build-system]",
        'dependencies = ["dinkster-api==1.2.3"]\n'
        "[tool.uv.sources]\ndinkster-api = { workspace = true }\n[build-system]",
    )
    files["packages/dinkster-engine/src/dinkster_graph/__init__.py"] = ""
    files["packages/dinkster-engine/src/dinkster_engine/testing.py"] = ""
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
    monkeypatch.setattr("scripts.build_release.development_version", lambda root: version)
    built = build(root, output, tag, frontend, payload_artifacts=payloads)
    manifest = json.loads((output / "release-manifest.json").read_text())
    assert built == manifest
    built_at = datetime.fromisoformat(manifest["builtAt"])
    assert built_at.tzinfo == UTC
    assert manifest["label"] == f"{built_at.date().isoformat()}-{public_version}"
    assert manifest["version"] == version
    assert manifest["defaultPacks"] == {"fixture-pack": "1.2.3"}
    assert "tag" not in manifest
    assert set(manifest["workspaceWheels"]) == {
        f"dinkster-{version}-py2.py3-none-any.whl",
        f"dinkster_api-{version}-py2.py3-none-any.whl",
        f"dinkster_engine-{version}-py2.py3-none-any.whl",
        "dinkster_supervisor-1.2.3-py2.py3-none-any.whl",
    }
    with zipfile.ZipFile(output / f"dinkster-{version}-py2.py3-none-any.whl") as wheel:
        assert not any(name.startswith("dinkster/official_packs/") for name in wheel.namelist())
        assert not any(name.endswith((".zip", ".pack-lock.json")) for name in wheel.namelist())
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


def test_finalized_release_rejects_external_wheels_and_enforces_exact_byte_budget(
    tmp_path: Path,
) -> None:
    from scripts.build_release import write_manifest

    names = [
        "dinkster-1.0-py3-none-any.whl",
        "dinkster_api-1.0-py3-none-any.whl",
        "dinkster_engine-1.0-py3-none-any.whl",
        "dinkster_supervisor-1.0-py3-none-any.whl",
        "constraints.txt",
    ]
    for project in ("dinkster", "dinkster-api", "dinkster-engine", "dinkster-supervisor"):
        distribution = project.replace("-", "_")
        companion = {"dinkster": "dinkster-engine", "dinkster-engine": "dinkster-api"}.get(project)
        metadata = f"Metadata-Version: 2.1\nName: {project}\nVersion: 1.0\n"
        if companion:
            metadata += f"Requires-Dist: {companion}==1.0\n"
        with zipfile.ZipFile(tmp_path / f"{distribution}-1.0-py3-none-any.whl", "w") as wheel:
            wheel.writestr(f"{distribution}-1.0.dist-info/METADATA", metadata)
            if project == "dinkster-engine":
                for module in (
                    "dinkster_engine/__init__.py",
                    "dinkster_engine/testing.py",
                    "dinkster_graph/__init__.py",
                ):
                    wheel.writestr(module, "")
    (tmp_path / "constraints.txt").write_text("pins\n")
    workspace_wheels = names[:4]
    manifest: dict[str, object] = {"workspaceWheels": workspace_wheels}
    write_manifest(tmp_path, manifest, [tmp_path / name for name in names])
    expected = sum((tmp_path / name).stat().st_size for name in names)
    expected += len((tmp_path / "release-manifest.json").read_bytes())
    expected += len((tmp_path / "SHA256SUMS").read_bytes())
    assert check_release(tmp_path, expected) == expected
    with pytest.raises(ValueError, match="exceeds budget"):
        check_release(tmp_path, expected - 1)
    for name in (
        "dinkster-tools-windows-x86_64.zip",
        "official-packs.lock",
        "dinkster-nodes-foundation-1.0.zip",
        "dinkster-nodes-foundation-1.0-linux-cpu.pack-lock.json",
    ):
        forbidden = tmp_path / name
        forbidden.write_text('{"packs": []}')
        write_manifest(tmp_path, manifest, [*(tmp_path / name for name in names), forbidden])
        with pytest.raises(ValueError, match="unexpected release payload"):
            check_release(tmp_path, 100_000_000)
        forbidden.unlink()
    external = "numpy-1.0-py3-none-any.whl"
    (tmp_path / external).write_bytes(b"external")
    write_manifest(tmp_path, manifest, [*(tmp_path / name for name in names), tmp_path / external])
    with pytest.raises(ValueError, match="only the four Dinkster workspace wheels"):
        check_release(tmp_path, 100_000_000)
    workspace_wheels.append(external)
    write_manifest(tmp_path, manifest, [*(tmp_path / name for name in names), tmp_path / external])
    with pytest.raises(ValueError, match="only the four Dinkster workspace wheels"):
        check_release(tmp_path, 100_000_000)


@pytest.mark.parametrize("platform", ["windows-x86_64", "linux-x86_64", "macos-arm64"])
@pytest.mark.parametrize("corruption", [None, "payload", "sums", "extra", "duplicate"])
def test_native_payload_finalization_preserves_and_verifies_existing_release(
    tmp_path: Path, corruption: str | None, platform: str
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
        "label": "2026-10-31-1.2.3",
        "builtAt": "2026-10-31T12:00:00Z",
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
    candidate = tmp_path / f"dinkster-tools-{platform}.zip"
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


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "corruption", [None, "size", "hash", "upstream-size", "upstream-hash", "missing"]
)
def test_native_pack_lock_urls_do_not_depend_on_manifest_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
    corruption: str | None,
    published: bool,
) -> None:
    from dinkster_registry import LockedPack, Lockfile
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock

    from scripts import build_release_pack_locks as builder

    release = tmp_path / "release"
    release.mkdir()
    filename = "example_dependency-2.0+native-py3-none-any.whl"
    wheel = release / filename
    wheel.write_bytes(b"verified native wheel")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    upstream = "https://files.example.test/locked/example_dependency-2.0%2Bnative-py3-none-any.whl"
    (tmp_path / "uv.lock").write_text(
        f'[[package]]\nname="example-dependency"\nversion="2.0+native"\n'
        f'[[package.wheels]]\nurl="{upstream}"\n'
        f"size={wheel.stat().st_size + (1 if corruption == 'upstream-size' else 0)}\n"
        f'hash="sha256:{"a" * 64 if corruption == "upstream-hash" else digest}"\n'
        if corruption != "missing"
        else "package=[]\n"
    )
    (release / "constraints.txt").write_text("example-dependency==2.0+native\n")
    engine = {
        "version": "1.2.3",
        "label": "2026-10-31-1.2.3",
        "workerProtocol": 11,
        "artifacts": [{"name": filename, "size": wheel.stat().st_size, "sha256": digest}],
    }
    if legacy:
        engine.pop("label")
        engine["tag"] = "main-" + "c" * 40
    metadata = release / "release-manifest.json"
    base = {
        "id": "sha256:" + "b" * 64,
        "variant": "linux-cpu",
        "workerProtocol": 11,
        "platform": {"os": "linux"},
        "interpreter": {"executable": "python/bin/python3"},
        "archive": {"parts": [{"filename": "base.tar.zst.001", "size": 17, "sha256": "a" * 64}]},
        "wheels": [],
        "nvidia": None,
    }
    engine["dependencies"] = {"linux-cpu": {"base": base["id"], "wheels": []}}
    metadata.write_text(json.dumps(engine))
    if not legacy:
        base["label"] = "2026-10-31-py3.14.10-torch2.14.1-bbbbbbbb"
    manifest = tmp_path / "base.json"
    manifest.write_text(json.dumps(base))
    monkeypatch.setattr(builder, "qualify", lambda *args: base)
    monkeypatch.setattr(builder.platform, "system", lambda: "Linux")
    monkeypatch.setattr(builder, "extract_base", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "effective_constraints", lambda *args, **kwargs: "constraints\n")

    def fetch(sources, filename, destination, digest, size):
        name = base.get("label", "b" * 64)
        assert sources == (
            "https://github.com/Kosinkadink/Dinkster-Environments/releases/download/"
            f"base-linux-cpu-{name}",
        )
        return destination

    monkeypatch.setattr(builder, "fetch_artifact", fetch)

    def build_pack(root, output, **kwargs):
        output.mkdir()
        path = output / "example-1.2.3-linux-cpu.pack-lock.json"
        PackLock(
            "example",
            "linux-cpu",
            ">=3.12",
            ("example-dependency==2.0+native",),
            ("https://pypi.org/simple",),
            (
                LockedDistribution(
                    "example-dependency",
                    "2.0+native",
                    (
                        LockedWheel(
                            filename,
                            wheel.stat().st_size + (1 if corruption == "size" else 0),
                            "a" * 64 if corruption == "hash" else digest,
                            wheel.as_uri(),
                        ),
                    ),
                ),
            ),
        ).write(path)
        return (path,)

    monkeypatch.setattr(builder, "build", build_pack)
    sources = tmp_path / "sources" if published else None
    if sources is not None:
        sources.mkdir()
        (sources / "official-packs.lock").write_text(
            Lockfile.of(
                [LockedPack("example", "1.2.3", "blake3:" + "a" * 64, "dinkster", ())]
            ).record_json()
        )
    if corruption in ("size", "hash") or (corruption and not legacy and not published):
        with pytest.raises(ValueError, match="portable.*provenance"):
            builder.build_locks(tmp_path, release, manifest, tmp_path / "locks", sources=sources)
        assert not (tmp_path / "locks").exists()
        return
    before = hashlib.sha256(metadata.read_bytes()).hexdigest()
    [first] = builder.build_locks(tmp_path, release, manifest, tmp_path / "locks", sources=sources)
    first_lock = PackLock.read(first)
    expected = (
        "https://github.com/Kosinkadink/Dinkster/releases/download/main-"
        + "c" * 40
        + "/example_dependency-2.0%2Bnative-py3-none-any.whl"
        if legacy
        else upstream
    )
    if published and not legacy:
        expected = (
            "https://github.com/Kosinkadink/Dinkster/releases/download/"
            f"main-2026-10-31-1.2.3-{before[:8]}/"
            "example_dependency-2.0%2Bnative-py3-none-any.whl"
        )
    assert first_lock.packages[0].wheels[0].url == expected
    assert first_lock.packages[0].wheels[0].sha256 == digest
    if published:
        assert hashlib.sha256(metadata.read_bytes()).hexdigest() == before
        return
    shutil.copyfile(first, release / first.name)
    engine["artifacts"].append(
        {
            "name": first.name,
            "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            "size": first.stat().st_size,
        }
    )
    # Adding lock bytes changes identity, but must not change their own provenance.
    metadata.write_text(json.dumps(engine))
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() != before
    [second] = builder.build_locks(tmp_path, release, manifest, tmp_path / "second-locks")
    assert second.read_bytes() == first.read_bytes()


@pytest.mark.parametrize(
    "mode", ["independent", "legacy", "archive", "catalog", "empty", "engine", "protocol", "host"]
)
def test_native_pack_locks_preserve_independent_sources_and_legacy_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import platform

    from blake3 import blake3
    from dinkster_registry import LockedPack, Lockfile
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock

    from scripts import build_release_pack_locks as native

    release, sources = tmp_path / "release", tmp_path / "sources"
    release.mkdir()
    sources.mkdir()
    versions = {"first-pack": "1.2.3", "second-pack": "7.0rc1"}
    release_versions = {**versions, "unselected-pack": "0.5.0"}
    (tmp_path / "uv.lock").write_text("package = []\n")

    def artifacts(directory, selected):
        directory.mkdir(exist_ok=True)
        paths, entries = [], []
        for name, version in selected.items():
            path = directory / f"{name}-{version}.zip"
            data = f"{name}@{version}".encode()
            path.write_bytes(data)
            paths.append(path)
            entries.append(
                LockedPack(name, version, "blake3:" + blake3(data).hexdigest(), "dinkster", ())
            )
        catalog = directory / "official-packs.lock"
        catalog.write_text(Lockfile.of(entries).record_json() + "\n")
        return (*paths, catalog)

    artifacts(sources, versions)
    artifacts(release, release_versions)
    if mode == "archive":
        (sources / "second-pack-7.0rc1.zip").write_bytes(b"changed")
    elif mode == "catalog":
        catalog = sources / "official-packs.lock"
        catalog.write_text(catalog.read_text().replace("7.0rc1", "6.0"))
    elif mode == "empty":
        (sources / "official-packs.lock").write_text(Lockfile.of([]).record_json())
    engine = {
        "version": "9.0+engine",
        "tag": "v9.0",
        "workerProtocol": 11,
        "artifacts": [
            {"name": "constraints.txt", "size": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        ],
    }
    (release / "constraints.txt").write_bytes(b"changed" if mode == "engine" else b"")
    (release / "release-manifest.json").write_text(json.dumps(engine))
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[platform.system()]
    wheel = "fixture_dependency-2.0-py3-none-any.whl"
    base = {
        "variant": "fixture-base",
        "id": "sha256:" + "a" * 64,
        "platform": {"os": "other" if mode == "host" else system},
        "workerProtocol": 12 if mode == "protocol" else 11,
        "archive": {"parts": []},
        "interpreter": {"executable": "python"},
        "wheels": [
            {
                "filename": wheel,
                "size": 17,
                "sha256": "b" * 64,
                "url": "https://example.test/fixture.whl",
            }
        ],
        "nvidia": None,
    }
    manifest = tmp_path / "base.json"
    manifest.write_text(json.dumps(base))
    monkeypatch.setattr(native, "qualify", lambda *args: base)
    monkeypatch.setattr(native, "extract_base", lambda *args, **kwargs: None)
    monkeypatch.setattr(native, "engine_constraints", lambda *args: "constraints")
    monkeypatch.setattr(native, "effective_constraints", lambda *args, **kwargs: "constraints")

    def build(root, output, *, version=None, packs, variant, **kwargs):
        selected = {name: version or release_versions[name] for name in packs or release_versions}
        payloads = artifacts(output, selected)
        locks = []
        for name, own_version in selected.items():
            path = output / f"{name}-{own_version}-{variant}.pack-lock.json"
            PackLock(
                name,
                variant,
                ">=3.12",
                (),
                ("https://pypi.org/simple",),
                (
                    LockedDistribution(
                        "fixture-dependency",
                        "2.0",
                        (LockedWheel(wheel, 17, "b" * 64, "file:///native/" + wheel),),
                    ),
                ),
            ).write(path)
            locks.append(path)
        return (*payloads, *locks)

    monkeypatch.setattr(native, "build", build)
    before = {
        path: path.read_bytes() for directory in (release, sources) for path in directory.iterdir()
    }
    output = tmp_path / "locks"
    if mode not in {"independent", "legacy"}:
        with pytest.raises(ValueError):
            native.build_locks(tmp_path, release, manifest, output, sources=sources)
        assert not output.exists()
    else:
        paths = native.build_locks(
            tmp_path, release, manifest, output, sources=None if mode == "legacy" else sources
        )
        expected = release_versions if mode == "legacy" else versions
        assert {path.name for path in paths} == {
            f"{name}-{version}-fixture-base.pack-lock.json" for name, version in expected.items()
        }
        for path in paths:
            lock = PackLock.read(path)
            assert lock.pack in expected and lock.variant == "fixture-base"
            assert lock.packages[0].wheels[0].url == "https://example.test/fixture.whl"
    assert {path: path.read_bytes() for path in before} == before
    assert not list(tmp_path.glob("release-pack-locks-*"))


@pytest.mark.parametrize("system", ["windows", "macos", "linux"])
@pytest.mark.parametrize("python", ["3.13.9", "3.14.10"])
@pytest.mark.parametrize("publication", ["legacy", "labelled", "alias"])
def test_selected_published_bases_are_qualified_and_preserved_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, python: str, system: str, publication: str
) -> None:
    from copy import deepcopy
    from io import BytesIO

    from scripts import acquire_engine_bases
    from scripts.acquire_engine_bases import acquire

    boundary = tmp_path / "packages/dinkster-workers/src/dinkster_workers/boundary.py"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("PROTOCOL_VERSION = 11\n")
    pins = {
        "torch": "2.14.1",
        "torchvision": "0.29.1",
        "numpy": "2.5.1",
        "cuda-toolkit": "13.0.3.0",
        "sympy": "1.14",
    }
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "dinkster"\nrequires-python = ">=3.12"\n'
        "[dependency-groups]\nengine-base = " + json.dumps(["numpy==2.5.1"])
    )
    tag = f"cp{python.split('.')[0]}{python.split('.')[1]}"
    windows = system == "windows"
    variant = {"windows": "win-cpu", "macos": "mac-arm64", "linux": "linux-nvidia"}[system]
    target = {
        "windows": "win_amd64",
        "macos": "macosx_14_0_arm64",
        "linux": "manylinux1_x86_64.manylinux_2_5_x86_64",
    }[system]
    triple = {
        "windows": "x86_64-pc-windows-msvc",
        "macos": "aarch64-apple-darwin",
        "linux": "x86_64-unknown-linux-gnu",
    }[system]
    executable = "python/python.exe" if windows else "python/bin/python3"
    lock = []
    wheels = []
    for name, version in pins.items():
        torch = system != "macos" and name in {"torch", "torchvision"}
        backend = "+cpu" if windows else "+cu132"
        actual = "13.2.2" if name == "cuda-toolkit" else version + (backend if torch else "")
        filename = f"{name.replace('-', '_')}-{actual}-{tag}-{tag}-{target}.whl"
        digest = hashlib.sha256(name.encode()).hexdigest()
        url = f"https://example.invalid/{filename}"
        wheels.append(
            dict(
                name=name,
                version=actual,
                filename=filename,
                size=17,
                sha256=digest,
                url=url,
                index=f"https://download.pytorch.org/whl/{backend[1:]}"
                if torch
                else "https://pypi.org/simple",
                verification="index" if torch or name == "cuda-toolkit" else "lock",
                lockedVersion=version,
            )
        )
        locked_url = (
            f"https://example.invalid/cuda_toolkit-{version}-{tag}-{tag}-{target}.whl"
            if name == "cuda-toolkit"
            else url
        )
        lock.append(
            f"[[package]]\nname = '{name}'\nversion = '{version}'\n"
            f"wheels = [{{url = '{locked_url}', size = 17, hash = 'sha256:{digest}'}}]\n"
        )
    lock.append(
        "[[package]]\nname = 'consumer'\nversion = '1.0'\n"
        "dependencies = [{name = 'torch'}, {name = 'torchvision'}, {name = 'numpy'}]\n"
        "[package.optional-dependencies]\n"
        "images = [{name = 'sympy', marker = \"extra == 'images' and python_version >= '3.13'\"},"
        "{name = 'cuda-toolkit', marker = \"sys_platform == 'win32' "
        "and python_version < '3.13'\"}]\n"
        "[[package]]\nname = 'inactive-consumer'\nversion = '1.0'\n"
        "resolution-markers = [\"python_version < '3.13'\"]\n"
        "dependencies = [{name = 'cuda-toolkit'}]\n"
    )
    (tmp_path / "uv.lock").write_text("\n".join(lock))
    original: dict = dict(
        format="dinkster.base/1",
        variant=variant,
        platform=dict(os=system, arch="arm64" if system == "macos" else "x86_64"),
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
        numpy = next(row for row in manifest["wheels"] if row["name"] == "numpy")
        if change == "version":
            numpy["version"] = "2.5.0"
            numpy["filename"] = f"numpy-2.5.0-{tag}-{tag}-{target}.whl"
        elif change == "wheel-hash":
            numpy["sha256"] = "0" * 64
        elif change in {"historical-index", "historical-lock"}:
            toolkit = next(row for row in manifest["wheels"] if row["name"] == "cuda-toolkit")
            toolkit["lockedVersion"] = "12.9.0" if change == "historical-index" else "13.2.2"
            toolkit["verification"] = "index" if change == "historical-index" else "lock"
        elif change == "optional-interface-version":
            sympy = next(row for row in manifest["wheels"] if row["name"] == "sympy")
            sympy.update(
                version="1.15",
                filename=f"sympy-1.15-{tag}-{tag}-{target}.whl",
                verification="index",
            )
        elif change == "torch-index":
            manifest["wheels"][0]["index"] = "https://download.pytorch.org/whl/cu126"
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
            manifest["wheels"] = [row for row in manifest["wheels"] if row["name"] != "numpy"]
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
        if publication != "legacy" or change.startswith("label-"):
            label = f"2026-10-31-py{python}-torch2.14.1-{identity[:8]}"
            if change == "label-hash":
                label = label[:-8] + "0" * 8
            elif change == "label-path":
                label = "../" + label
            manifest["label"] = label
            requested = f"base-{variant}-{label}"
            if change == "label-tag":
                requested = requested.replace("2026-10-31", "2026-10-30")
            elif change == "label-variant":
                requested = requested.replace(f"base-{variant}-", "base-other-")
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
        if change == "pinned-id":
            return f"base-{variant}-{identity[:8]}{'0' * 56}"
        return f"base-{variant}-{identity}" if publication == "alias" else requested

    def index_response(request, **kwargs):
        assert request.full_url == f"https://releases.dinkster.org/base/{variant}/index.json"
        assert request.get_header("User-agent") == "dinkster-env"
        manifest = json.loads(payload)
        identity = manifest["id"]
        if mutation == "pinned-id":
            identity = identity[:15] + "0" * 56
        rows = [{"id": identity, "path": f"base/{variant}/{identity[7:]}"}]
        if publication == "alias" or mutation == "pinned-id":
            rows.append(
                {
                    "id": identity,
                    "label": manifest["label"],
                    "path": f"base/{variant}/{manifest['label']}",
                }
            )
        return BytesIO(json.dumps(rows).encode())

    def transport(command: list[str]) -> bytes:
        if "/releases/tags/" not in command[-1]:
            return extras_payload if command[-1].endswith("/2") else payload
        if mutation in {"valid", "stream", "historical-index", "historical-lock", "pinned-id"}:
            assert command[-1].endswith("/" + requested)
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

    monkeypatch.setattr(acquire_engine_bases, "urlopen", index_response)
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
    if publication == "alias":
        assert json.loads(paths[0].read_bytes())["label"] in requested
    streamed = acquire(tmp_path, [prepare("stream")], tmp_path / "streamed")
    assert streamed[0].read_bytes() == payload
    for change in ("historical-index", "historical-lock"):
        historical = acquire(tmp_path, [prepare(change)], tmp_path / change)
        assert historical[0].read_bytes() == payload
    for change in (
        "version",
        "wheel-hash",
        "optional-interface-version",
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
        "label-hash",
        "label-path",
        "label-tag",
        "label-variant",
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
    if publication == "alias":
        with pytest.raises(ValueError, match="pinned full identity"):
            acquire(tmp_path, [prepare("pinned-id")], tmp_path / "pinned-id")
        assert not (tmp_path / "pinned-id").exists()
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


def test_disposable_registry_acquisition_uses_public_source_for_all_pull_requests(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/official-packs.yml").read_text())
    assert "pull_request_target" not in workflow[True]
    assert "packages/**/pyproject.toml" in workflow[True]["pull_request"]["paths"]
    job = workflow["jobs"]["registry-service-image"]
    assert "registry-service-unavailable" not in workflow["jobs"]
    assert "secrets." not in yaml.safe_dump(job)
    for event in ("pull_request", "workflow_run", "workflow_dispatch", "push"):
        for fork in (False, True):
            for actor in ("contributor", "dependabot[bot]"):
                values = {
                    "github.event_name": event,
                    "github.event.pull_request.head.repo.full_name": "fork/Dinkster"
                    if fork
                    else "Kosinkadink/Dinkster",
                    "github.repository": "Kosinkadink/Dinkster",
                    "github.actor": actor,
                }
                condition = job["if"]
                for name, value in values.items():
                    condition = condition.replace(name, repr(value))
                condition = condition.replace("&&", " and ").replace("||", " or ")
                assert eval(condition, {"__builtins__": {}}) is (event == "pull_request")
    checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v4")
    assert checkout["with"]["repository"] == "Kosinkadink/dinkster-registry-service"
    pin = checkout["with"]["ref"]
    assert len(pin) == 40 and all(char in "0123456789abcdef" for char in pin)
    assert checkout["with"]["persist-credentials"] is False
    assert "ssh-key" not in checkout["with"]
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="ascii")
    docker.chmod(0o755)
    service_build = next(
        step["run"]
        for step in job["steps"]
        if "docker build --tag" in step.get("run", "")
        and ".disposable-registry-source" in step["run"]
    )
    build = subprocess.run(
        ["bash", "-c", service_build],
        env={"PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    assert build.stdout.splitlines() == [
        "build",
        "--tag",
        "dinkster-disposable-registry:local",
        checkout["with"]["path"],
    ]
    assert job["timeout-minutes"] == 15
    acquisition = workflow["jobs"]["probe-index"]["steps"]
    assert any("check_publication_controls prepare" in step.get("run", "") for step in acquisition)
    upload = next(
        step["with"]
        for step in acquisition
        if step.get("with", {}).get("name") == "disposable-publication-controls"
    )
    assert upload["include-hidden-files"] is True
    isolated = workflow["jobs"]["disposable-publication"]["steps"]
    controls = next(
        index
        for index, step in enumerate(isolated)
        if "check_publication_controls check" in step.get("run", "")
    )
    publication = next(
        index
        for index, step in enumerate(isolated)
        if "python3 scripts/check_disposable_publication.py" in step.get("run", "")
    )
    assert controls < publication
    assert any(
        step.get("with", {}).get("name") == "disposable-publication-controls"
        for step in isolated[:controls]
    )
    # The isolated runner must not install host packages to execute staged controls.
    help_result = subprocess.run(
        [sys.executable, "-S", "-m", "scripts.check_publication_controls", "check", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--fixtures" in help_result.stdout


@pytest.mark.parametrize(
    "event",
    [
        "initial",
        "engine-only",
        "tag",
        "lockstep",
        "main-success",
        "main-failure",
        "main-skipped",
        "main-workflow",
        "main-unrelated",
        "main-served",
        "main-other-versions",
        "main-schema1",
        "main-invalid",
        "main-substitution",
        "main-error",
        "workflow_dispatch",
    ],
)
def test_official_publication_workflow_selects_committed_versions_without_engine_coupling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, event: str
) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/official-packs.yml").read_text())
    trigger = workflow[True]
    assert trigger["push"] == {"tags": ["pack/**"]}
    assert trigger["workflow_run"] == {
        "workflows": ["Versioned wheel release"],
        "types": ["completed"],
        "branches": ["main"],
    }
    assert "pull_request_target" not in trigger
    assert trigger["workflow_dispatch"]["inputs"]["head"]["required"] is True
    assert trigger["workflow_dispatch"]["inputs"]["head"]["type"] == "string"
    jobs = workflow["jobs"]
    checkout = next(
        step for step in jobs["sources"]["steps"] if step.get("uses") == "actions/checkout@v4"
    )
    assert checkout["with"]["fetch-depth"] == 0
    select = next(step for step in jobs["select"]["steps"] if step.get("id") == "select")
    script = select["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    manifest = tmp_path / "packages/dinkster-nodes-foundation/dinkster-pack.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '[pack]\nname = "dinkster-nodes-foundation"\npublisher = "dinkster"\n'
        'version = "2.7rc1"\nengine = ">=0.2,<1"\n'
        '[pack.entry]\nnodes = "nodes:NODES"\n'
    )
    if event.startswith("main-"):
        library = tmp_path / "packages/dinkster-pack-helpers/pyproject.toml"
        library.parent.mkdir()
        library.write_text(
            '[project]\nname = "dinkster-pack-helpers"\nversion = "1.3"\n'
            "[tool.dinkster]\nofficial-library = true\n"
        )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    tree = subprocess.check_output(["git", "-C", str(tmp_path), "write-tree"], text=True).strip()
    (tmp_path / ".git/HEAD").write_text(tree + "\n")
    after = tree
    if event == "lockstep" or event.startswith("main-"):
        changed = tmp_path / (
            ".github/workflows/official-packs.yml"
            if event == "main-workflow"
            else "unrelated-test.py"
            if event
            in {
                "main-unrelated",
                "main-served",
                "main-other-versions",
                "main-schema1",
                "main-invalid",
                "main-substitution",
                "main-error",
            }
            else "uv.lock"
        )
        changed.parent.mkdir(parents=True, exist_ok=True)
        changed.write_text("version = 1\n")
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
        after = subprocess.check_output(
            ["git", "-C", str(tmp_path), "write-tree"], text=True
        ).strip()
        (tmp_path / ".git/HEAD").write_text(after + "\n")
        original = subprocess.check_output

        def read(command, **kwargs):
            if command == ["git", "rev-parse", "HEAD^"]:
                return tree + "\n"
            return original(command, **kwargs)

        monkeypatch.setattr(subprocess, "check_output", read)
    manifest.write_text("invalid uncommitted declaration")
    output = tmp_path / "outputs"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv(
        "EVENT",
        "workflow_run"
        if event.startswith("main-")
        else "push"
        if event == "tag"
        else "workflow_dispatch"
        if event == "workflow_dispatch"
        else "pull_request",
    )
    monkeypatch.setenv("CONCLUSION", "failure" if event == "main-failure" else "success")
    monkeypatch.setenv("ENGINE_PUBLISHED", "false" if event == "main-skipped" else "true")
    monkeypatch.setenv("BEFORE", "0" * 40 if event == "initial" else tree)
    monkeypatch.setenv(
        "PACK_TAG", "pack/dinkster-nodes-foundation/v2.7rc1" if event == "tag" else ""
    )

    def read_metadata(registry, path):
        assert event.startswith("main-") and event not in {"main-failure", "main-skipped"}
        assert registry == "https://registry.dinkster.org"
        if event == "main-error":
            raise OSError("registry unavailable")
        if event == "main-invalid":
            return b'{"schemaVersion": 2, "items": [null]}'
        if event == "main-schema1":
            return json.dumps(
                {
                    "schemaVersion": 1,
                    "items": [
                        {"name": "dinkster-nodes-foundation", "version": "2.7rc1"},
                        {"name": "dinkster-pack-helpers", "version": "1.2"},
                    ],
                }
            ).encode()
        versions = (
            {"dinkster-nodes-foundation": ["2.7rc1", "9.0"], "dinkster-pack-helpers": ["1.3"]}
            if event in {"main-served", "main-substitution"}
            else {"dinkster-nodes-foundation": ["0.4", "9.0"], "dinkster-pack-helpers": ["1.3"]}
            if event == "main-other-versions"
            else {}
        )
        if path == "static/packs/index.json":
            return json.dumps(
                {
                    "schemaVersion": 2,
                    "items": [
                        {
                            "name": name,
                            "latestVersion": values[-1],
                            "url": f"packs/{name}/index.json",
                        }
                        for name, values in versions.items()
                    ],
                }
            ).encode()
        name = path.split("/")[2]
        return json.dumps(
            {
                "schemaVersion": 2,
                "name": "substituted" if event == "main-substitution" else name,
                "items": [{"name": name, "version": version} for version in versions[name]],
            }
        ).encode()

    monkeypatch.setattr("dinkster_supervisor.environment_downloads.read_metadata", read_metadata)
    if event in {"main-invalid", "main-substitution", "main-error"}:
        with pytest.raises(
            OSError if event == "main-error" else ValueError, match="index|unavailable"
        ):
            exec(compile(script, "official-packs.yml", "exec"), {})
        assert not output.exists()
        return
    if event == "main-unrelated":
        monkeypatch.setenv("ENGINE_PUBLISHED", "false")
        exec(compile(script, "official-packs.yml", "exec"), {})
        skipped = dict(line.split("=", 1) for line in output.read_text().splitlines())
        assert json.loads(skipped["packs"]) == [] and skipped["applicable"] == "false"
        output.unlink()
        monkeypatch.setenv("ENGINE_PUBLISHED", "true")
    exec(compile(script, "official-packs.yml", "exec"), {})
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(outputs["packs"]) == (
        []
        if event in {"engine-only", "main-failure", "main-skipped", "main-served"}
        else ["dinkster-pack-helpers"]
        if event == "main-schema1"
        else ["dinkster-nodes-foundation", "dinkster-pack-helpers"]
        if event.startswith("main-") and event != "main-other-versions"
        else ["dinkster-nodes-foundation"]
    )
    assert outputs["applicable"] == (
        "false" if event in {"main-failure", "main-skipped"} else "true"
    )
    if event in {"tag", "lockstep"}:
        assert "candidate-head engine dependency pins" in capsys.readouterr().out
    if event in {"main-failure", "main-skipped"}:
        assert "same-commit engine release did not succeed" in capsys.readouterr().out
    if event in {"main-success", "main-workflow", "workflow_dispatch"}:
        assert outputs["engine"] == after
        assert "candidate-head engine dependency pins" not in capsys.readouterr().out
    assert jobs["sources"]["if"] == (
        "needs.select.outputs.packs != '[]' && needs.select.outputs.applicable == 'true'"
        " && !needs.select.outputs.recovery_run"
    )
    continuation = next(
        step for step in jobs["select"]["steps"] if step.get("id") == "continuation"
    )
    assert jobs["select"]["steps"].index(select) < jobs["select"]["steps"].index(continuation)
    assert continuation["env"]["PACKS"] == "${{ steps.select.outputs.packs }}"
    assert '--packs "$PACKS"' in continuation["run"]
    assert jobs["locks"]["needs"] == ["select", "sources"]
    assert jobs["locks"]["runs-on"] == "${{ matrix.runner }}"
    native = next(step for step in jobs["locks"]["steps"] if "--sources" in step.get("run", ""))
    assert "--release .official-publication/engine" in native["run"]
    assert '--base-manifest ".official-publication/bases/$VARIANT/manifest.json"' in native["run"]
    assert "scripts.build_release_pack_locks" in native["run"]
    prepare = next(step for step in jobs["sources"]["steps"] if step.get("id") == "prepare")
    assert "scripts.build_release import" not in prepare["run"]
    assert "prepare_inputs(" in prepare["run"]
    assert "ENGINE_COMMIT" in prepare["env"]
    assert prepare["env"]["CANDIDATE"] == (
        "${{ github.event_name != 'workflow_run' && github.event_name != 'workflow_dispatch' }}"
    )
    assert "version=" not in prepare["run"]
    assert jobs["publish"]["needs"] == ["select", "sources", "locks", "recovery"]
    for job in ("sources", "locks", "publish"):
        checkout = next(
            step for step in jobs[job]["steps"] if step.get("uses") == "actions/checkout@v4"
        )
        assert checkout["with"]["ref"] == (
            "${{ github.event_name == 'workflow_dispatch' && github.sha"
            " || needs.select.outputs.engine }}"
            if job == "publish"
            else "${{ needs.select.outputs.engine }}"
        )
    [submission] = [
        step
        for job in jobs.values()
        for step in job["steps"]
        if "secrets.DINKSTER_REGISTRY_TOKEN" in json.dumps(step)
    ]
    assert submission["if"] == (
        "github.repository == 'Kosinkadink/Dinkster' && "
        "((github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main') || "
        "(github.event_name == 'workflow_run' && github.event.workflow_run.event == 'push' && "
        "github.event.workflow_run.head_branch == 'main' && "
        "github.event.workflow_run.head_repository.full_name == 'Kosinkadink/Dinkster' && "
        "github.event.workflow_run.conclusion == 'success'))"
    )
    assert "--publish" in submission["run"]
    preflight = next(
        step
        for step in jobs["publish"]["steps"]
        if "--output .official-publication/preflight" in step.get("run", "")
    )
    assert "--publish" not in preflight["run"]
    assert "--check-published" in preflight["run"]
    assert jobs["publish"]["steps"].index(preflight) < jobs["publish"]["steps"].index(submission)


@pytest.mark.parametrize(
    "problem",
    [None, "repository", "head", "event", "workflow", "expired", "locks", "engine"],
)
def test_recovery_dispatch_requires_trusted_exact_head_artifacts(problem: str | None) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/official-packs.yml").read_text())
    script = next(
        step["with"]["script"]
        for step in workflow["jobs"]["select"]["steps"]
        if step.get("id") == "engine_release"
    )
    head = "a" * 40
    run = {
        "id": 70,
        "head_sha": head,
        "head_branch": "main",
        "head_repository": {"full_name": "Kosinkadink/Dinkster"},
        "path": ".github/workflows/official-packs.yml",
        "event": "workflow_run",
        "conclusion": "success",
    }
    fixture = {
        "head": head,
        "run": run,
        "engineRun": {**run, "event": "push"},
        "engine": problem != "engine",
        "artifact": {
            "id": 100,
            "name": "official-publication-inputs",
            "expired": problem == "expired",
            "workflow_run": {"id": 70},
        },
        "locks": [
            {"id": index, "name": "publication-locks-" + variant, "expired": False}
            for index, variant in enumerate(
                json.loads((ROOT / "scripts/base_bundles.json").read_bytes()), 101
            )
        ],
        "script": script,
    }
    if problem == "repository":
        run["head_repository"] = {"full_name": "other/Dinkster"}
    elif problem == "head":
        run["head_sha"] = "b" * 40
    elif problem == "event":
        run["event"] = "pull_request"
    elif problem == "workflow":
        run["path"] = ".github/workflows/other.yml"
    elif problem == "locks":
        fixture["locks"] = []
    # Execute the workflow's actual script, replacing only GitHub transport and outputs.
    harness = r"""
const fs = require('fs');
const fixture = JSON.parse(fs.readFileSync(0, 'utf8'));
const outputs = {};
const context = {eventName: 'workflow_dispatch', repo: {owner: 'Kosinkadink', repo: 'Dinkster'},
  payload: {inputs: {head: fixture.head, recovery_artifact: '100'}}};
const core = {setOutput: (key, value) => outputs[key] = value};
const actions = {getArtifact: async () => ({data: fixture.artifact}),
  getWorkflowRun: async () => ({data: fixture.run}), listWorkflowRuns: () => {},
  listJobsForWorkflowRun: () => {}, listWorkflowRunArtifacts: () => {}};
const github = {rest: {repos: {compareCommitsWithBasehead: async () =>
  ({data: {status: 'ahead'}})}, actions}, paginate: async (operation) => {
    if (operation === actions.listWorkflowRuns) return [fixture.engineRun];
    if (operation === actions.listJobsForWorkflowRun)
      return fixture.engine ? [{name: 'release', conclusion: 'success'}] : [];
    if (operation === actions.listWorkflowRunArtifacts) return fixture.locks;
    throw new Error('unexpected endpoint');
  }};
new Function('context', 'core', 'github', 'return (async () => {' + fixture.script + '})()')
  (context, core, github).then(() => console.log(JSON.stringify({outputs})))
  .catch(error => console.log(JSON.stringify({error: error.message})));
"""
    result = json.loads(
        subprocess.check_output(
            ["node", "-e", harness], input=json.dumps(fixture), text=True, timeout=10
        )
    )
    if problem is None:
        assert result == {
            "outputs": {
                "recovery_run": 70,
                "recovery_locks": "101,102,103,104,105,106",
                "published": True,
            }
        }
    else:
        expected = (
            "Catch-up publication requires a successful same-commit engine release"
            if problem == "engine"
            else "Recovery requires unexpired unique native lock artifacts"
            if problem == "locks"
            else "Recovery requires original successful main publication inputs at head"
        )
        assert result == {"error": expected}


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_official_publication_recovery_reuses_inputs_without_resolving_locks(
    recovery: bool, failed: bool
) -> None:
    from types import SimpleNamespace

    workflow = yaml.safe_load((ROOT / ".github/workflows/official-packs.yml").read_text())
    jobs = workflow["jobs"]
    needs = SimpleNamespace(
        select=SimpleNamespace(result="success"),
        recovery=SimpleNamespace(
            result="failure" if failed else "success" if recovery else "skipped"
        ),
        sources=SimpleNamespace(result="skipped" if recovery else "success"),
        locks=SimpleNamespace(result="skipped" if recovery else "failure" if failed else "success"),
    )
    condition = (
        jobs["publish"]["if"].replace("&&", " and ").replace("||", " or ").replace("!", "not ")
    )
    assert eval(
        condition, {"needs": needs, "always": lambda: True, "cancelled": lambda: False}
    ) is (not failed)
    steps = jobs["recovery"]["steps"]
    assert not any("build_release_pack_locks" in step.get("run", "") for step in steps)
    downloads = [
        step["with"] for step in steps if step.get("uses") == "actions/download-artifact@v4"
    ]
    assert [step["artifact-ids"] for step in downloads] == [
        "${{ needs.select.outputs.recovery_artifact }}",
        "${{ needs.select.outputs.recovery_locks }}",
    ]
    assert all(step["run-id"] == "${{ needs.select.outputs.recovery_run }}" for step in downloads)
    assert downloads[0]["path"] == ".official-publication"
    assert downloads[1]["path"] == ".official-publication/locks"
    uploads = [step["with"] for step in steps if step.get("uses") == "actions/upload-artifact@v4"]
    assert uploads[0]["path"].splitlines() == [
        ".official-publication/sources/",
        ".official-publication/bases/",
        ".official-publication/engine/",
    ]
    assert uploads[1]["path"] == ".official-publication/locks/*.pack-lock.json"


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "failed-lock",
        "missing-lock",
        "duplicate",
        "expired",
        "foreign-artifact",
        "ancestor",
        "completed",
        "foreign-run",
        "source-failure",
        "missing-artifact",
        "artifact-head",
        "invalid-head",
        "foreign-branch",
        "foreign-path",
        "foreign-event",
        "failed-continuation",
        "failure",
    ],
)
def test_interrupted_publication_recovery_requires_original_complete_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, problem: str | None
) -> None:
    from scripts import recover_official_publication
    from scripts.recover_official_publication import recovery_inputs

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as inputs:
        inputs.writestr("sources/official-packs.lock", "{}")
        inputs.writestr("engine/release-manifest.json", "{}")
    monkeypatch.setattr(
        recover_official_publication, "validate_recovered_sources", lambda *args: None
    )
    variants = json.loads((ROOT / "scripts/base_bundles.json").read_bytes())
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/base_bundles.json").write_text(json.dumps(variants))
    head = "a" * 40
    run = {
        "id": 70,
        "head_sha": head,
        "head_branch": "main",
        "head_repository": {"full_name": "Kosinkadink/Dinkster"},
        "path": ".github/workflows/official-packs.yml",
        "event": "workflow_run",
        "conclusion": "cancelled",
    }
    jobs = [{"name": "sources", "conclusion": "success"}] + [
        {"name": f"locks ({v}, runner)", "conclusion": "success"} for v in variants
    ]
    artifacts = [
        {"id": i, "name": name, "expired": False, "workflow_run": {"id": 70, "head_sha": head}}
        for i, name in enumerate(
            ["official-publication-inputs", *(f"publication-locks-{v}" for v in variants)], 100
        )
    ]
    runs = [run]
    if problem == "failed-lock":
        jobs[-1]["conclusion"] = "failure"
    elif problem == "source-failure":
        jobs[0]["conclusion"] = "failure"
    elif problem == "missing-lock":
        jobs.pop()
    elif problem == "duplicate":
        artifacts.append(artifacts[-1])
    elif problem == "expired":
        artifacts[-1]["expired"] = True
    elif problem == "foreign-artifact":
        artifacts[-1]["workflow_run"] = {"id": 69, "head_sha": head}
    elif problem == "artifact-head":
        artifacts[-1]["workflow_run"] = {"id": 70, "head_sha": "b" * 40}
    elif problem == "missing-artifact":
        artifacts.pop()
    elif problem == "invalid-head":
        run["head_sha"] = "main"
    elif problem == "foreign-branch":
        run["head_branch"] = "feature"
    elif problem == "foreign-path":
        run["path"] = ".github/workflows/other.yml"
    elif problem == "foreign-event":
        run["event"] = "pull_request"
    elif problem == "foreign-run":
        run["head_repository"] = {"full_name": "other/Dinkster"}
    elif problem == "completed":
        runs.insert(0, {**run, "id": 80, "conclusion": "success"})
    elif problem == "failed-continuation":
        runs.insert(0, {**run, "id": 80, "conclusion": "failure"})
    elif problem == "failure":
        run["conclusion"] = "failure"

    def read(command, **kwargs):
        assert command[:2] == ["gh", "api"]
        path = command[-1]
        if path.endswith("/zip"):
            return archive.getvalue()
        if "/workflows/" in path:
            return json.dumps([{"workflow_runs": runs}]).encode()
        if "/80/" in path:
            if problem == "failed-continuation":
                return json.dumps(
                    [
                        {"jobs": [{"name": "sources", "conclusion": "skipped"}]}
                        if "/jobs?" in path
                        else {"artifacts": []}
                    ]
                ).encode()
            return json.dumps([{"artifacts": [{"name": "publication-completed-70"}]}]).encode()
        return json.dumps(
            [{"jobs": jobs} if "/jobs?" in path else {"artifacts": artifacts}]
        ).encode()

    monkeypatch.setattr(subprocess, "check_output", read)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 1 if problem == "ancestor" else 0),
    )
    if problem in {"completed", "foreign-run", "foreign-branch", "foreign-path", "foreign-event"}:
        assert recovery_inputs(tmp_path, 90) == {}
    elif problem == "source-failure":
        assert recovery_inputs(tmp_path, 90) == {}
        assert "source job did not succeed" in capsys.readouterr().out
    elif problem in {"failed-lock", "missing-lock"}:
        assert recovery_inputs(tmp_path, 90) == {}
        assert (
            "continuation refused; all six native lock jobs must succeed" in capsys.readouterr().out
        )
    elif problem not in {None, "failed-continuation", "failure"}:
        assert problem is not None
        with pytest.raises(
            ValueError,
            match={
                "duplicate": "unexpired and unique",
                "expired": "unexpired and unique",
                "foreign-artifact": "artifact run identity",
                "artifact-head": "artifact run identity",
                "missing-artifact": "unexpired and unique",
                "invalid-head": "source commit must belong to main",
                "ancestor": "source commit must belong to main",
            }[problem],
        ):
            recovery_inputs(tmp_path, 90)
    else:
        assert recovery_inputs(tmp_path, 90) == {
            "recovery_run": "70",
            "recovery_head": head,
            "recovery_artifact": "100",
            "recovery_locks": "101,102,103,104,105,106",
        }


@pytest.mark.parametrize(
    "change", [None, "code", "older-valid", "pack-tests", "removed", "engine", "catalog", "bases"]
)
def test_publication_continuation_requires_unchanged_selected_git_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, change: str | None
) -> None:
    from dinkster_registry import LockedPack, Lockfile

    from scripts.recover_official_publication import recovery_inputs, validate_recovered_sources

    root = tmp_path / "repo"
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    manifest = root / "packages/dinkster-nodes-foundation/dinkster-pack.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '[pack]\nname = "dinkster-nodes-foundation"\npublisher = "dinkster"\n'
        'version = "2.7rc1"\nengine = ">=0.2,<1"\n'
        '[pack.entry]\nnodes = "nodes:NODES"\n'
    )
    code = manifest.parent / "nodes.py"
    code.write_text("NODES = {}\n")
    if change == "older-valid":
        unchanged = root / "packages/dinkster-nodes-unchanged/dinkster-pack.toml"
        unchanged.parent.mkdir()
        unchanged.write_text(manifest.read_text().replace("foundation", "unchanged"))
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    head = subprocess.check_output(["git", "-C", str(root), "write-tree"], text=True).strip()
    (root / ".git/HEAD").write_text(head + "\n")
    if change in {"code", "older-valid"}:
        code.write_text("NODES = {'changed': 1}\n")
    elif change == "pack-tests":
        (manifest.parent / "tests").mkdir()
        (manifest.parent / "tests/test_nodes.py").write_text("assert True\n")
    elif change == "removed":
        manifest.unlink()
    elif change == "bases":
        (root / "scripts").mkdir()
        (root / "scripts/base_bundles.json").write_text('{"different": "base"}')
    else:
        (root / "unrelated-test.py").write_text("assert True\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    after = subprocess.check_output(["git", "-C", str(root), "write-tree"], text=True).strip()
    (root / ".git/HEAD").write_text(after + "\n")
    work = tmp_path / "original-inputs"
    (work / "sources").mkdir(parents=True)
    (work / "engine").mkdir()
    catalog = Lockfile.of(
        [
            LockedPack(
                "dinkster-nodes-foundation",
                "2.7rc1",
                "blake3:" + "1" * 64,
                "dinkster",
                (),
                source=f"git:Kosinkadink/Dinkster@{after if change == 'catalog' else head}",
            )
        ]
    )
    (work / "sources/official-packs.lock").write_text(catalog.record_json())
    (work / "engine/release-manifest.json").write_text(
        json.dumps(
            {
                "repository": "Kosinkadink/Dinkster",
                "commit": after if change == "engine" else head,
            }
        )
    )
    if change in {None, "pack-tests"}:
        validate_recovered_sources(root, work, head)
    else:
        assert change is not None
        with pytest.raises(
            ValueError,
            match={
                "code": "selected source bytes changed",
                "older-valid": "selected source bytes changed",
                "removed": "source identity removed",
                "engine": "engine identity differs",
                "catalog": "catalog must identify",
                "bases": "base selection changed",
            }[change],
        ):
            validate_recovered_sources(root, work, head)

    variants = json.loads((ROOT / "scripts/base_bundles.json").read_bytes())
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts/base_bundles.json").write_text(json.dumps(variants))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as inputs:
        for name in ("sources/official-packs.lock", "engine/release-manifest.json"):
            inputs.write(work / name, name)
    older_archive = io.BytesIO()
    if change == "older-valid":
        with zipfile.ZipFile(older_archive, "w") as inputs:
            inputs.writestr(
                "sources/official-packs.lock",
                Lockfile.of(
                    [
                        LockedPack(
                            "dinkster-nodes-unchanged",
                            "2.7rc1",
                            "blake3:" + "2" * 64,
                            "dinkster",
                            (),
                            source=f"git:Kosinkadink/Dinkster@{head}",
                        )
                    ]
                ).record_json(),
            )
            inputs.write(work / "engine/release-manifest.json", "engine/release-manifest.json")
    jobs = [{"name": "sources", "conclusion": "success"}] + [
        {"name": f"locks ({v}, runner)", "conclusion": "success"} for v in variants
    ]
    artifacts = [
        {"id": i, "name": name, "expired": False, "workflow_run": {"id": 70, "head_sha": head}}
        for i, name in enumerate(
            ["official-publication-inputs", *(f"publication-locks-{v}" for v in variants)], 100
        )
    ]
    original_read = subprocess.check_output
    original_run = subprocess.run

    def read(command, **kwargs):
        if command[:2] != ["gh", "api"]:
            return original_read(command, **kwargs)
        path = command[-1]
        if path.endswith("/zip"):
            if "/200/" in path:
                return older_archive.getvalue()
            return archive.getvalue()
        if "/workflows/" in path:
            pages = [
                {
                    "workflow_runs": [
                        {
                            "id": 70,
                            "head_sha": head,
                            "head_branch": "main",
                            "head_repository": {"full_name": "Kosinkadink/Dinkster"},
                            "path": ".github/workflows/official-packs.yml",
                            "event": "workflow_run",
                            "conclusion": "cancelled",
                        }
                    ]
                }
            ]
            if change == "older-valid":
                pages[0]["workflow_runs"].append({**pages[0]["workflow_runs"][0], "id": 60})
            return json.dumps(pages).encode()
        if "/60/" in path:
            return json.dumps(
                [
                    {"jobs": jobs}
                    if "/jobs?" in path
                    else {
                        "artifacts": [
                            {**a, "id": a["id"] + 100, "workflow_run": {"id": 60, "head_sha": head}}
                            for a in artifacts
                        ]
                    }
                ]
            ).encode()
        return json.dumps(
            [{"jobs": jobs} if "/jobs?" in path else {"artifacts": artifacts}]
        ).encode()

    def run(command, **kwargs):
        if "merge-base" in command:
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", read)
    monkeypatch.setattr(subprocess, "run", run)
    if change in {None, "pack-tests"}:
        assert recovery_inputs(root, 90)["recovery_run"] == "70"
        assert (
            recovery_inputs(root, 90, packs=["dinkster-nodes-foundation"])["recovery_run"] == "70"
        )
        assert recovery_inputs(root, 90, packs=[]) == {}
        assert (
            recovery_inputs(
                root, 90, packs=["dinkster-nodes-foundation", "missing-predecessor-pack"]
            )
            == {}
        )
    elif change == "older-valid":
        assert recovery_inputs(root, 90)["recovery_run"] == "60"
        assert "publication run 70: continuation skipped; recovered" in capsys.readouterr().out
    elif change in {"code", "removed", "bases"}:
        assert recovery_inputs(root, 90) == {}
        assert "publication run 70: continuation skipped; recovered" in capsys.readouterr().out
    else:
        with pytest.raises(ValueError, match="identity differs|catalog must identify"):
            recovery_inputs(root, 90)


@pytest.mark.parametrize(
    ("repository", "event", "ref", "allowed"),
    [
        ("Kosinkadink/Dinkster", "workflow_run", "refs/heads/main", True),
        ("Kosinkadink/Dinkster", "workflow_dispatch", "refs/heads/main", True),
        ("someone/Dinkster", "workflow_dispatch", "refs/heads/main", False),
        ("Kosinkadink/Dinkster", "workflow_dispatch", "refs/heads/feature", False),
        ("Kosinkadink/Dinkster", "workflow_dispatch", "refs/tags/pack/demo/v1.0", False),
        ("Kosinkadink/Dinkster", "push", "refs/heads/main", False),
        ("Kosinkadink/Dinkster", "pull_request", "refs/heads/main", False),
        ("someone/Dinkster", "push", "refs/heads/main", False),
        ("Kosinkadink/Dinkster", "push", "refs/tags/pack/demo/v1.0", False),
        ("Kosinkadink/Dinkster", "push", "refs/heads/feature", False),
    ],
)
def test_official_publication_submits_only_from_trusted_main(
    repository: str, event: str, ref: str, allowed: bool
) -> None:
    from types import SimpleNamespace

    workflow = yaml.safe_load((ROOT / ".github/workflows/official-packs.yml").read_text())
    submission = next(
        step for step in workflow["jobs"]["publish"]["steps"] if "secrets." in json.dumps(step)
    )
    github = SimpleNamespace(
        repository=repository,
        event_name=event,
        ref=ref,
        event=SimpleNamespace(
            workflow_run=SimpleNamespace(
                event="push",
                head_branch="main" if ref == "refs/heads/main" else "feature",
                head_repository=SimpleNamespace(full_name=repository),
                conclusion="success",
            )
        ),
    )
    assert (
        eval(submission["if"].replace("&&", " and ").replace("||", " or "), {"github": github})
        is allowed
    )
    if event == "workflow_dispatch":
        assert (
            eval(
                workflow["jobs"]["select"]["if"].replace("&&", " and ").replace("||", " or "),
                {"github": github},
            )
            is allowed
        )


def test_official_publication_refuses_missing_credential_before_submission(tmp_path: Path) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/official-packs.yml").read_text())
    submission = next(
        step for step in workflow["jobs"]["publish"]["steps"] if "secrets." in json.dumps(step)
    )
    environment = {**os.environ, "DINKSTER_REGISTRY_TOKEN": ""}
    bash = (
        str(Path(os.environ["ProgramFiles"]) / "Git/bin/bash.exe")
        if sys.platform == "win32"
        else "bash"
    )
    result = subprocess.run(
        [bash, "-e", "-c", submission["run"]],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "Official pack submission refused: DINKSTER_REGISTRY_TOKEN is unavailable"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("labelled", [False, True])
@pytest.mark.parametrize(
    "problem",
    [
        None,
        "range",
        "tag",
        "commit",
        "missing",
        "digest",
        "candidate",
        "index-commit",
        "index-digest",
        "index-path",
        "index-duplicate",
    ],
)
def test_official_publication_workflow_prepares_verified_published_engine_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None, labelled: bool
) -> None:
    from dinkster_registry import Lockfile
    from dinkster_supervisor import environment_downloads
    from test_installer import _serve_artifacts

    from scripts import prepare_official_publication as preparation

    tag = "main-" + "c" * 40
    (tmp_path / "scripts").mkdir()
    selections = {"linux-cpu": "a" * 64, "win-nvidia": "b" * 64}
    (tmp_path / "scripts/base_bundles.json").write_text(json.dumps(selections))

    def acquire(root, tags, output):
        assert set(tags) == {
            f"base-{variant}-{identity}" for variant, identity in selections.items()
        }
        paths = []
        for variant, system in (("linux-cpu", "linux"), ("win-nvidia", "windows")):
            path = output / variant / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"variant": variant, "platform": {"os": system}}))
            paths.append(path)
        return paths

    engine = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "version": "0.1" if problem == "range" else "0.0.4.dev123",
        "tag": "wrong" if problem == "tag" else tag,
        "commit": ("d" if problem == "commit" else "c") * 40,
        "workerProtocol": 11,
        "artifacts": [
            {"name": "constraints.txt", "size": 4, "sha256": hashlib.sha256(b"pins").hexdigest()}
        ],
    }
    if labelled:
        del engine["tag"]
        engine.update(label="2026-11-03-0.0.4.dev123+build", builtAt="2026-11-03T10:41:18Z")
    engine_bytes = json.dumps(engine).encode()
    identity = hashlib.sha256(engine_bytes).hexdigest()
    indexed_digest = "f" * 64 if problem == "index-digest" else identity
    if labelled:
        tag = f"main-2026-11-03-0.0.4.dev123+build-{indexed_digest[:8]}"
    entry = {
        "commit": ("e" if problem == "index-commit" else "c") * 40,
        "sha256": indexed_digest,
        "label": engine.get("label"),
        "version": engine["version"],
        "path": "engine/"
        + (f"2026-11-03-0.0.4.dev123+build-{indexed_digest[:8]}" if labelled else indexed_digest),
    }
    if problem == "index-path":
        entry["path"] = "engine/" + "a" * 64

    def metadata(mirror, path):
        assert problem != "candidate"
        assert mirror == "https://releases.dinkster.org" and path == "engine/index.json"
        entries = [{**entry, "commit": "b" * 40}, entry]
        if problem == "index-duplicate":
            entries.append(entry)
        return json.dumps(entries).encode()

    old_check_output, old_run = subprocess.check_output, subprocess.run

    def read(command, **kwargs):
        if command[0] != "gh":
            return old_check_output(command, **kwargs)
        if command[-1] != f"repos/Kosinkadink/Dinkster/releases/tags/{quote(tag, safe='')}":
            raise subprocess.CalledProcessError(1, command, stderr=b"Not Found (HTTP 404)")
        return json.dumps(
            {
                "tag_name": "wrong" if labelled and problem == "tag" else tag,
                "draft": problem == "missing",
                "published_at": "2026-01-01T00:00:00Z",
            }
        ).encode()

    def run(command, **kwargs):
        if command[0] != "gh":
            return old_run(command, **kwargs)
        assert command[3] == tag
        directory = Path(command[-1])
        directory.mkdir()
        (directory / "release-manifest.json").write_bytes(engine_bytes)
        return subprocess.CompletedProcess(command, 0)

    original_fetch = environment_downloads.fetch_artifact
    server = _serve_artifacts({"/constraints.txt": b"fake" if problem == "digest" else b"pins"}, [])

    def fetch(mirrors, name, destination, digest, size):
        assert mirrors == (
            "https://github.com/Kosinkadink/Dinkster/releases/download/" + quote(tag, safe=""),
        )
        return original_fetch(
            (f"http://127.0.0.1:{server.server_port}",), name, destination, digest, size
        )

    original_build = preparation.build
    monkeypatch.setattr(preparation, "read_metadata", metadata, raising=False)
    monkeypatch.setattr(preparation, "acquire", acquire)
    monkeypatch.setattr(subprocess, "check_output", read)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(preparation, "fetch_artifact", fetch)
    candidate_dependencies = {"linux-cpu": {"base": "sha256:" + "a" * 64, "wheels": []}}
    if problem == "candidate":
        monkeypatch.setattr(preparation, "development_version", lambda root: "0.0.4.dev456")
        monkeypatch.setattr(
            preparation,
            "versioned_sources",
            lambda root, version: versioned_sources(ROOT, version),
        )

        def candidate_wheels(root, uv, bases):
            assert root == tmp_path and uv == "uv" and len(bases) == 2
            return candidate_dependencies

        monkeypatch.setattr(preparation, "select_dependency_wheels", candidate_wheels)
    monkeypatch.setattr(
        preparation,
        "build",
        lambda root, output, **kwargs: original_build(ROOT, output, **kwargs),
    )
    output = tmp_path / "outputs"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("PACKS", '["dinkster-image"]')
    try:
        if problem not in {None, "candidate"}:
            with pytest.raises((ValueError, environment_downloads.DownloadError)):
                preparation.prepare_inputs(
                    tmp_path,
                    tmp_path / ".official-publication",
                    "c" * 40,
                    ["dinkster-image"],
                )
            assert not output.exists()
        else:
            assert preparation.prepare_inputs(
                tmp_path,
                tmp_path / ".official-publication",
                "c" * 40,
                ["dinkster-image"],
                candidate=problem == "candidate",
            ) == [
                {"variant": "linux-cpu", "runner": "ubuntu-latest"},
                {"variant": "win-nvidia", "runner": "windows-latest"},
            ]
            work = tmp_path / ".official-publication"
            if problem == "candidate":
                manifest = json.loads((work / "engine/release-manifest.json").read_bytes())
                assert manifest["dependencies"] == candidate_dependencies
                assert manifest["version"] == "0.0.4.dev456"
                assert manifest["commit"] == "c" * 40
                api = work / "engine/dinkster_api-0.0.4.dev456-py3-none-any.whl"
                assert wheel_metadata(api) == ("dinkster-api", "0.0.4.dev456")
                assert manifest["artifacts"] == [
                    {
                        "name": api.name,
                        "size": api.stat().st_size,
                        "sha256": hashlib.sha256(api.read_bytes()).hexdigest(),
                    }
                ]
            else:
                assert (work / "engine/constraints.txt").read_bytes() == b"pins"
            catalog = Lockfile.from_record_json((work / "sources/official-packs.lock").read_text())
            assert [(pack.pack, pack.version) for pack in catalog.packs] == [
                ("dinkster-image", "0.0.5")
            ]
    finally:
        server.shutdown()
        server.server_close()
