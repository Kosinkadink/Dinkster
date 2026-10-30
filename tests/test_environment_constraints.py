"""Variant base membership replaces locked requirements without resolution."""

from __future__ import annotations

import pytest
from dinkster_supervisor.environment_constraints import effective_constraints
from dinkster_supervisor.environment_state import EnvironmentError


def _base(version: str = "2.14.1+cpu") -> dict[str, object]:
    return {
        "replaces": ["nvidia-cublas", "torch"],
        "wheels": [
            {
                "name": "torch",
                "version": version,
                "filename": f"torch-{version}-cp314-cp314-manylinux_2_28_x86_64.whl",
                "sha256": "b" * 64,
                "lockedVersion": "2.14.1",
            }
        ],
    }


def test_cpu_replaces_cuda_closure_and_keeps_unrelated_markers_and_all_hashes():
    engine = (
        "torch==2.14.1 --hash=sha256:" + "a" * 64 + "\n"
        "nvidia-cublas==13.0 --hash=sha256:" + "c" * 64 + "\n"
        'different_cuda_tool==1.2 ; sys_platform == "linux" \\\n'
        "    --hash=sha256:" + "d" * 64 + " \\\n"
        "    --hash=sha256:" + "e" * 64 + "\n"
    )
    merged = effective_constraints(engine, _base())
    assert merged == (
        'different_cuda_tool==1.2 ; sys_platform == "linux" '
        "    --hash=sha256:" + "d" * 64 + " "
        "    --hash=sha256:" + "e" * 64 + "\n"
        "torch==2.14.1+cpu --hash=sha256:" + "b" * 64 + "\n"
    )
    assert "nvidia-cublas" not in merged
    assert "--hash=sha256:" + "a" * 64 not in merged


def test_selected_base_requirement_absent_from_engine_is_not_resolved():
    assert effective_constraints("", _base()) == (
        "torch==2.14.1+cpu --hash=sha256:" + "b" * 64 + "\n"
    )


@pytest.mark.parametrize("version", ["2.15.0+cpu", "2.14.1.post1+cpu"])
def test_variant_version_mismatch_fails_instead_of_overriding_engine_pin(version: str):
    engine = "torch==2.14.1 --hash=sha256:" + "a" * 64
    with pytest.raises(EnvironmentError, match="disagrees with engine lock"):
        effective_constraints(engine, _base(version))


def test_non_torch_local_versions_do_not_escape_the_exact_pin():
    base = {
        "replaces": ["numpy"],
        "wheels": [
            {
                "name": "numpy",
                "version": "2.5.1+other",
                "filename": "numpy-2.5.1+other-cp314-cp314-manylinux_2_28_x86_64.whl",
                "sha256": "b" * 64,
                "lockedVersion": "2.5.1",
            }
        ],
    }
    with pytest.raises(EnvironmentError, match="disagrees with engine lock"):
        effective_constraints("numpy==2.5.1 --hash=sha256:" + "a" * 64, base)


@pytest.mark.parametrize(
    "requirement",
    [
        "torch==2.14.1",
        "torch>=2.14.1 --hash=sha256:" + "a" * 64,
        "torch==2.14.* --hash=sha256:" + "a" * 64,
        "torch @ https://example.org/torch.whl --hash=sha256:" + "a" * 64,
        "--index-url https://example.org/simple",
    ],
)
def test_unhashed_resolving_or_url_requirements_are_refused(requirement: str):
    with pytest.raises(EnvironmentError):
        effective_constraints(requirement, _base())


@pytest.mark.parametrize(
    "membership", [[], ["torch", "torch"], ["torch", "nvidia-cublas"], ["Torch"]]
)
def test_invalid_replacement_membership_is_not_inferred(membership: list[str]):
    base = _base()
    base["replaces"] = membership
    with pytest.raises(EnvironmentError):
        effective_constraints("", base)


def test_selected_wheel_cannot_claim_an_unowned_distribution_or_false_filename():
    base = _base()
    base["replaces"] = ["other"]
    with pytest.raises(EnvironmentError, match="replacement ownership"):
        effective_constraints("", base)
    base = _base()
    base["wheels"] = [
        {
            "name": "torch",
            "version": "2.14.1+cpu",
            "filename": "torch-2.15.0-cp314-cp314-manylinux_2_28_x86_64.whl",
            "sha256": "b" * 64,
        }
    ]
    with pytest.raises(EnvironmentError, match="identity disagrees with filename"):
        effective_constraints("", base)
