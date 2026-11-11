from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tools.verify_pinned_inference import pinned_requirement, verify_source_contracts

PIN = "0.1.1"
REQUIREMENT = f"dinkster-inference=={PIN}"


def test_pack_requirements_use_the_engine_extra_pin() -> None:
    root = Path(__file__).resolve().parents[1]
    assert pinned_requirement(root) == (REQUIREMENT, PIN)
    for path in root.glob("packages/**/*pack.toml"):
        manifest = tomllib.loads(path.read_text())
        for value in manifest.get("pack", {}).get("requires", []):
            if Requirement(value).name == "dinkster-inference":
                assert value == "dinkster-inference", str(path.relative_to(root))


def test_runtime_dependencies_do_not_require_git() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "*.toml", "uv.lock", "**/uv.lock"], cwd=root
        )
        .decode()
        .split("\0")
    )
    for relative in filter(None, paths):
        document = tomllib.loads(root.joinpath(relative).read_text())
        sources = [
            document.get("pack", {}).get("requires", []),
            document.get("tool", {}).get("uv", {}).get("sources", {}),
        ]
        if Path(relative).name == "uv.lock":
            sources = [document]
        while sources:
            source = sources.pop()
            if isinstance(source, dict):
                assert "git" not in source, f"{relative}: Git source {source}"
                sources.extend(source.values())
            elif isinstance(source, list):
                sources.extend(source)
            elif isinstance(source, str):
                assert "git+" not in source, f"{relative}: Git requirement {source}"


def test_pinned_requirement_requires_exact_release(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(f'[project.optional-dependencies]\ninference = ["{REQUIREMENT}"]\n')
    assert pinned_requirement(tmp_path) == (REQUIREMENT, PIN)

    project.write_text(
        '[project.optional-dependencies]\ninference = ["dinkster-inference>=0.1.0"]\n'
    )
    with pytest.raises(RuntimeError, match="no exact dinkster-inference release"):
        pinned_requirement(tmp_path)


@pytest.mark.parametrize(
    ("relative", "metadata"),
    [
        ("packages/example/pyproject.toml", '[project]\ndependencies = ["{requirement}"]\n'),
        ("packages/example/dinkster-pack.toml", '[pack]\nrequires = ["{requirement}"]\n'),
        (
            "pyproject.toml",
            '[project.optional-dependencies]\ninference = ["dinkster-inference==0.1.1"]\n'
            '[tool.uv]\nconstraint-dependencies = ["{requirement}"]\n',
        ),
    ],
)
@pytest.mark.parametrize("requirement", [REQUIREMENT, "dinkster-inference>=0.1.0"])
def test_release_build_rejects_a_second_inference_pin(
    tmp_path: Path, relative: str, metadata: str, requirement: str
) -> None:
    from scripts.build_release import build

    tmp_path.joinpath("pyproject.toml").write_text(
        f'[project.optional-dependencies]\ninference = ["{REQUIREMENT}"]\n'
    )
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.format(requirement=requirement))
    with pytest.raises(RuntimeError, match="one authored pin"):
        build(tmp_path, tmp_path / "release", None, tmp_path / "frontend.zip")
    assert not tmp_path.joinpath("release").exists()


def test_source_contracts_accept_upstream_loaders_and_reject_missing_sampler(
    tmp_path: Path,
) -> None:
    package = tmp_path / "dinkster_inference"
    package.mkdir()
    package.joinpath("sd.py").write_text(
        "def load_checkpoint_guess_config(): pass\n"
        "def load_diffusion_model(): pass\n"
        "def load_clip(): pass\n"
        "class VAE: pass\n"
    )
    package.joinpath("sample.py").write_text(
        "def sample(): pass\ndef prepare_noise(): pass\ndef fix_empty_latent_channels(): pass\n"
    )
    verify_source_contracts(package)
    package.joinpath("sample.py").write_text("def prepare_noise(): pass\n")
    with pytest.raises(RuntimeError, match="missing sample"):
        verify_source_contracts(package)
