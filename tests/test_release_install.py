"""The release builder produces one complete, version-bound artifact set."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml

from scripts.build_release import build_source_archive, release_version, workspace_projects

ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_is_tag_only_and_publishes_after_platform_installs() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    assert workflow[True] == {"push": {"tags": ["v*.*.*"]}}
    assert set(workflow["jobs"]) == {"build", "install", "release"}
    assert workflow["jobs"]["install"]["needs"] == "build"
    assert workflow["jobs"]["release"]["needs"] == "install"
    assert workflow["jobs"]["release"]["permissions"] == {"contents": "write"}
    preflight = workflow["jobs"]["release"]["steps"][-2]
    assert preflight["uses"] == "actions/github-script@v7"
    assert "getReleaseByTag" in preflight["with"]["script"]
    assert "Refusing to overwrite an existing release" in preflight["with"]["script"]
    assert workflow["jobs"]["release"]["steps"][-1] == {
        "uses": "softprops/action-gh-release@v2",
        "with": {
            "generate_release_notes": True,
            "fail_on_unmatched_files": True,
            "files": "dist/*",
        },
    }


def test_release_workflow_builds_all_wheels_and_checks_tag_metadata() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    build = workflow["jobs"]["build"]
    commands = "\n".join(step.get("run", "") for step in build["steps"])
    assert "scripts/build_release.py" in commands
    assert '--tag "$GITHUB_REF_NAME"' in commands
    assert "--identity-root .release/identity" in commands
    assert "uvx twine check dist/*.whl" in commands
    assert "pnpm --filter @dinkster/app build" in commands
    frontend_checkout = next(
        step
        for step in build["steps"]
        if step.get("uses") == "actions/checkout@v4"
        and step.get("with", {}).get("repository") == "Kosinkadink/Dinkster-Frontend"
    )
    assert frontend_checkout["with"]["ref"] == "${{ steps.frontend.outputs.ref }}"
    assert frontend_checkout["with"]["persist-credentials"] is False
    identity_checkout = next(
        step
        for step in build["steps"]
        if step.get("uses") == "actions/checkout@v4"
        and step.get("with", {}).get("repository") == "Kosinkadink/dinkster-identity"
    )
    assert identity_checkout["with"]["ref"] == "${{ steps.identity.outputs.ref }}"
    assert identity_checkout["with"]["persist-credentials"] is False


def test_release_install_matrix_covers_supported_desktop_platforms() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    assert workflow["jobs"]["install"]["strategy"]["matrix"] == {
        "include": [
            {"os": "linux", "labels": ["self-hosted", "linux", "x64"]},
            {"os": "windows", "labels": ["self-hosted", "windows", "x64"]},
            {"os": "macos", "labels": ["self-hosted", "macos", "arm64"]},
        ]
    }
    bootstrap = workflow["jobs"]["install"]["steps"][0]
    assert bootstrap["name"] == "Expose Git Bash on Windows"
    assert bootstrap["if"] == "runner.os == 'Windows'"
    assert bootstrap["shell"] == "pwsh"
    assert "Git/bin" in bootstrap["run"]
    assert "bash.exe" in bootstrap["run"]
    assert "GITHUB_PATH" in bootstrap["run"]
    command = next(
        step["run"]
        for step in workflow["jobs"]["install"]["steps"]
        if step.get("name") == "Install and launch from wheels"
    )
    assert "--no-deps" in command
    assert "--require-hashes" in command
    assert "--find-links" in command
    assert "--requirement" in command
    assert "from dinkster_frontend import bundle_path" in command
    assert all(
        step.get("uses") != "./.github/actions/configure-dinkster-identity"
        for step in workflow["jobs"]["install"]["steps"]
    )


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
    assert release_version(ROOT, "v0.0.1") == "0.0.1"
    assert len(workspace_projects(ROOT)) == 33
    frontend = json.loads((ROOT / "scripts/release_sources.json").read_text(encoding="utf-8"))
    assert frontend["repository"] == "Kosinkadink/Dinkster-Frontend"
    assert len(frontend["commit"]) == 40
    assert frontend["identityRepository"] == "Kosinkadink/dinkster-identity"
    assert len(frontend["identityCommit"]) == 40


def test_maintainer_source_archive_excludes_non_release_material(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    files = {
        "src/package.py": "released\n",
        "tests/test_package.py": "excluded\n",
        "tools/generate.py": "excluded\n",
        "benchmarks/measure.py": "excluded\n",
        "scripts/run_benchmark.py": "excluded\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True)
    archive = build_source_archive(root, "1.2.3", tmp_path)
    with zipfile.ZipFile(archive) as source:
        assert source.namelist() == ["scripts/", "src/", "src/package.py"]
