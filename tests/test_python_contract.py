from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from dinkster_workers import boundary

from dinkster import port
from tools.evidence_paths import EVIDENCE_ROOT

ROOT = Path(__file__).resolve().parents[1]


def _toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_workspace_and_generated_metadata_share_python_floor() -> None:
    manifests = [
        ROOT / "pyproject.toml",
        *sorted((ROOT / "packages").glob("*/pyproject.toml")),
        EVIDENCE_ROOT / "packages/dinkster-acceptance/pyproject.toml",
    ]
    assert len(manifests) == 37
    for manifest in manifests:
        project = _toml(manifest)["project"]
        assert isinstance(project, dict)
        assert project["requires-python"] == ">=3.12", manifest

    root = _toml(ROOT / "pyproject.toml")
    tool = root["tool"]
    assert isinstance(tool, dict)
    assert tool["ruff"]["target-version"] == "py312"  # type: ignore[index]
    assert tool["pyright"]["pythonVersion"] == "3.12"  # type: ignore[index]

    torch = _toml(ROOT / "packages/dinkster-inference-torch/pyproject.toml")
    torch_tool = torch["tool"]
    assert isinstance(torch_tool, dict)
    assert torch_tool["pyright"]["pythonVersion"] == "3.12"  # type: ignore[index]

    training_torch = _toml(ROOT / "packages/dinkster-training-torch/pyproject.toml")
    training_torch_tool = training_torch["tool"]
    assert isinstance(training_torch_tool, dict)
    assert training_torch_tool["pyright"]["pythonVersion"] == "3.12"  # type: ignore[index]

    template = _toml(ROOT / "templates/pack/pyproject.toml")
    template_project = template["project"]
    template_tool = template["tool"]
    assert isinstance(template_project, dict) and isinstance(template_tool, dict)
    assert template_project["requires-python"] == ">=3.12"
    assert template_tool["ruff"]["target-version"] == "py312"  # type: ignore[index]
    assert template_tool["pyright"]["pythonVersion"] == "3.12"  # type: ignore[index]

    generated = tomllib.loads(port._emit_pyproject("sample-pack", "source"))
    assert generated["project"]["requires-python"] == ">=3.12"
    assert generated["tool"]["ruff"]["target-version"] == "py312"
    assert generated["tool"]["pyright"]["pythonVersion"] == "3.12"


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
