from __future__ import annotations

from pathlib import Path

import pytest

from tools.verify_pinned_inference import pinned_requirement, verify_source_contracts

PIN = "1" * 40
REQUIREMENT = (
    f"dinkster-inference @ git+https://github.com/Kosinkadink/dinkster-inference.git@{PIN}"
)


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


def test_source_contracts_reject_an_inference_without_window_execution(tmp_path: Path) -> None:
    package = tmp_path / "dinkster_inference"
    package.mkdir()
    for name in ("window_plan", "window_execution", "sd", "gguf", "gguf_ops"):
        package.joinpath(f"{name}.py").write_text("")

    with pytest.raises(RuntimeError, match="IntegerAffineIndexMap"):
        verify_source_contracts(package)
