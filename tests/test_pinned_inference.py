from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tools.verify_pinned_inference import pinned_requirement, verify_source_contracts

PIN = "1" * 40
REQUIREMENT = (
    f"dinkster-inference @ git+https://github.com/Kosinkadink/dinkster-inference.git@{PIN}"
)


def test_pack_requirements_match_setup_script_pin() -> None:
    root = Path(__file__).resolve().parents[1]
    requirement, _ = pinned_requirement(root)
    for relative in (
        "packages/dinkster-native/dinkster-pack.toml",
        "packages/dinkster-compat-comfy/dinkster-pack.toml",
        "packages/dinkster-compat-comfy/dinkster-legacy-pack.toml",
    ):
        manifest = tomllib.loads(root.joinpath(relative).read_text())
        assert manifest["pack"]["requires"] == [requirement]


def _write_setup_scripts(root: Path, shell_pin: str, powershell_pin: str) -> None:
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "setup_envs.sh").write_text(f'dinkster_inference_requirement="{shell_pin}"\n')
    (scripts / "setup_envs.ps1").write_text(f'$DinksterInferenceRequirement = "{powershell_pin}"\n')


def test_pinned_requirement_requires_matching_exact_commits(tmp_path: Path) -> None:
    _write_setup_scripts(tmp_path, REQUIREMENT, REQUIREMENT)
    assert pinned_requirement(tmp_path) == (REQUIREMENT, PIN)

    other = REQUIREMENT[:-40] + "2" * 40
    (tmp_path / "scripts/setup_envs.ps1").write_text(f'$DinksterInferenceRequirement = "{other}"\n')
    with pytest.raises(RuntimeError, match="pin different inference commits"):
        pinned_requirement(tmp_path)


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
