from __future__ import annotations

import hashlib
import json
import tomllib
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from scripts.record_opentimelineio_wheels import record


@pytest.mark.parametrize(
    "platform,target", [("win_amd64", "windows"), ("manylinux_2_17_x86_64", "linux")]
)
def test_opentimelineio_release_records_exact_binary_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, target: str
) -> None:
    pin = json.loads((Path(__file__).parents[1] / "scripts/opentimelineio_source.json").read_text())
    replies = iter([pin["commit"] + "\n", " " + "1" * 40 + " src/deps/example (v1)\n"])
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: next(replies))
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    path = wheels / f"opentimelineio-0.19.0.dev1-cp314-cp314-{platform}.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            "opentimelineio-0.19.0.dev1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: OpenTimelineIO\nVersion: 0.19.0.dev1\n",
        )
    payload = path.read_bytes()
    record(tmp_path / "source", wheels, tmp_path / "output", target)
    result = json.loads((tmp_path / "output" / f"opentimelineio-{target}.json").read_text())
    assert result["commit"] == pin["commit"]
    assert result["wheels"] == [
        {"filename": path.name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    ]
    notes = (tmp_path / "output" / f"opentimelineio-{target}.md").read_text()
    assert pin["commit"] in notes
    assert hashlib.sha256(payload).hexdigest() in notes


@pytest.mark.parametrize(
    "tag,metadata_version",
    [("cp313-cp313-win_amd64", "0.19.0.dev1"), ("cp314-cp314-win_amd64", "0.18.1")],
)
def test_opentimelineio_release_rejects_wrong_interpreter_or_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str, metadata_version: str
) -> None:
    pin = json.loads((Path(__file__).parents[1] / "scripts/opentimelineio_source.json").read_text())
    replies = iter([pin["commit"] + "\n", " " + "1" * 40 + " src/deps/example (v1)\n"])
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: next(replies))
    with zipfile.ZipFile(tmp_path / f"opentimelineio-0.19.0.dev1-{tag}.whl", "w") as wheel:
        wheel.writestr(
            "opentimelineio-0.19.0.dev1.dist-info/METADATA",
            f"Name: OpenTimelineIO\nVersion: {metadata_version}\n",
        )
    with pytest.raises(ValueError, match="wheel identity|metadata disagrees"):
        record(tmp_path / "source", tmp_path, tmp_path / "output", "windows")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "version,implementation,platform,machine,expected",
    [
        ("3.14", "cpython", "linux", "x86_64", "0.19.0.dev1"),
        ("3.14", "cpython", "win32", "AMD64", "0.19.0.dev1"),
        ("3.12", "cpython", "linux", "x86_64", "0.18.1"),
        ("3.13", "cpython", "win32", "AMD64", "0.18.1"),
        ("3.15", "cpython", "linux", "x86_64", "0.18.1"),
        ("3.14", "pypy", "linux", "x86_64", "0.18.1"),
        ("3.14", "cpython", "linux", "aarch64", "0.18.1"),
        ("3.14", "cpython", "win32", "ARM64", "0.18.1"),
        ("3.14", "cpython", "darwin", "arm64", "0.18.1"),
    ],
)
def test_opentimelineio_requirements_select_one_publishable_target_version(
    version: str, implementation: str, platform: str, machine: str, expected: str
) -> None:
    root = Path(__file__).parents[1]
    environment = {
        "python_version": version,
        "implementation_name": implementation,
        "sys_platform": platform,
        "platform_machine": machine,
    }
    umbrella = tomllib.loads((root / "pyproject.toml").read_text())
    video = tomllib.loads((root / "packages/dinkster-video/pyproject.toml").read_text())
    for requirements in (
        umbrella["dependency-groups"]["dev"],
        video["project"]["dependencies"],
    ):
        candidates = [Requirement(value) for value in requirements]
        selected = [
            value
            for value in candidates
            if value.name == "opentimelineio"
            and (value.marker is None or value.marker.evaluate(environment))
        ]
        assert len(selected) == 1
        assert selected[0].url is None
        assert str(selected[0].specifier) == f"=={expected}"
