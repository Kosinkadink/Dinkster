from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from dinkster_workers import boundary

from tools.evidence_paths import EVIDENCE_ROOT, validate_evidence_revision

validate_evidence_revision()
ROOT = Path(__file__).resolve().parents[1]


def _toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_workspace_and_pack_fixture_share_python_floor() -> None:
    package_manifests = sorted((ROOT / "packages").glob("*/pyproject.toml"))
    assert package_manifests
    manifests = [
        ROOT / "pyproject.toml",
        *package_manifests,
        EVIDENCE_ROOT / "packages/dinkster-acceptance/pyproject.toml",
    ]
    for manifest in manifests:
        project = _toml(manifest)["project"]
        assert isinstance(project, dict)
        assert project["requires-python"] == ">=3.12", manifest

    root = _toml(ROOT / "pyproject.toml")
    tool = root["tool"]
    assert isinstance(tool, dict)
    assert tool["ruff"]["target-version"] == "py312"  # type: ignore[index]
    assert tool["pyright"]["pythonVersion"] == "3.12"  # type: ignore[index]

    fixture = _toml(ROOT / "tests/fixtures/pack-contract/pyproject.toml")
    fixture_project = fixture["project"]
    fixture_tool = fixture["tool"]
    assert isinstance(fixture_project, dict) and isinstance(fixture_tool, dict)
    assert fixture_project["requires-python"] == ">=3.12"
    assert fixture_tool["ruff"]["target-version"] == "py312"  # type: ignore[index]
    assert fixture_tool["pyright"]["pythonVersion"] == "3.12"  # type: ignore[index]


def test_ci_exercises_python_312_without_narrowing_package_support() -> None:
    for name in ("ci.yml", "full-validation.yml"):
        workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
        assert 'python-version: "3.12"' in workflow
        assert '"3.13"' not in workflow
        assert '"3.11"' not in workflow
    assert _toml(ROOT / "uv.lock")["requires-python"] == ">=3.12"


@pytest.mark.skipif(os.name != "posix", reason="POSIX resource_tracker contract")
@pytest.mark.parametrize(("version", "track"), [((3, 12), None), ((3, 13), False)])
def test_shared_memory_attach_preserves_version_specific_tracking(
    version: tuple[int, int], track: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    unregistered: list[tuple[str, str]] = []

    class Segment:
        pass

    def shared_memory(**kwargs: object) -> Segment:
        calls.append(kwargs)
        return Segment()

    monkeypatch.setattr("dinkster_workers.boundary.sys.version_info", version)
    monkeypatch.setattr(boundary, "SharedMemory", shared_memory)
    monkeypatch.setattr(
        "multiprocessing.resource_tracker.unregister",
        lambda name, kind: unregistered.append((name, kind)),
    )

    assert isinstance(boundary._attach_segment("segment"), Segment)
    expected: dict[str, object] = {"name": "segment"}
    if track is not None:
        expected["track"] = track
    assert calls == [expected]
    assert unregistered == ([("/segment", "shared_memory")] if version < (3, 13) else [])
