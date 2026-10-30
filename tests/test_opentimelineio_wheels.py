from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_opentimelineio_requirements_use_the_published_distribution() -> None:
    root = Path(__file__).parents[1]
    umbrella = tomllib.loads((root / "pyproject.toml").read_text())
    video = tomllib.loads((root / "packages/dinkster-video/pyproject.toml").read_text())
    api = tomllib.loads((root / "packages/dinkster-api/pyproject.toml").read_text())
    for metadata, requirements in (
        (umbrella, umbrella["dependency-groups"]["dev"]),
        (video, video["project"]["dependencies"]),
        (api, api["project"]["dependencies"]),
    ):
        candidates = [Requirement(value) for value in requirements]
        selected = [value for value in candidates if "opentimelineio" in value.name]
        assert len(selected) == 1
        assert selected[0].name == "dinkster-opentimelineio"
        assert selected[0].url is None
        assert selected[0].marker is None
        assert str(selected[0].specifier) == "==0.19.0"
        sources = metadata.get("tool", {}).get("uv", {}).get("sources", {})
        assert not any("opentimelineio" in name for name in sources)
