"""Variant base membership replaces locked requirements without resolution."""

from __future__ import annotations

import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from dinkster_supervisor.environment_constraints import (
    effective_constraints,
    validate_base_requirements,
)
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


def test_generic_base_only_versions_and_provenance_do_not_override_selected_artifacts():
    base = {
        "python": "3.14.10",
        "replaces": ["cuda-toolkit"],
        "wheels": [
            {
                "name": "cuda-toolkit",
                "version": "13.2.2",
                "lockedVersion": "13.0.3.0",
                "filename": "cuda_toolkit-13.2.2-py3-none-any.whl",
                "sha256": "b" * 64,
            }
        ],
    }
    exact = "cuda-toolkit==13.2.2 --hash=sha256:" + "b" * 64 + "\n"
    generic = "cuda-toolkit==13.0.3.0 --hash=sha256:" + "a" * 64 + "\n"
    assert effective_constraints("", base) == exact
    assert effective_constraints(generic, base, interfaces=set()) == exact
    assert effective_constraints(exact, base, strict=True) == exact
    with pytest.raises(EnvironmentError, match="disagrees with engine lock"):
        effective_constraints(generic, base, interfaces={"cuda-toolkit"})
    with pytest.raises(EnvironmentError, match="require strict=False"):
        effective_constraints(generic, base, interfaces=set(), strict=True)
    marked = (
        'cuda-toolkit==13.2.2 ; python_version >= "3.14" --hash=sha256:' + "b" * 64 + "\n"
        'cuda-toolkit==99.0 ; python_version < "3.14" --hash=sha256:' + "c" * 64 + "\n"
    )
    assert effective_constraints(marked, base, interfaces={"cuda-toolkit"}) == exact


def test_variant_install_lock_requires_exact_backend_version_and_hash():
    exact = "torch==2.14.1+cpu --hash=sha256:" + "b" * 64 + "\n"
    assert effective_constraints(exact, _base(), strict=True) == exact
    for wrong in (
        "torch==2.14.1 --hash=sha256:" + "b" * 64,
        "torch==2.14.1+cpu --hash=sha256:" + "a" * 64,
        "",
    ):
        with pytest.raises(EnvironmentError, match="engine lock"):
            effective_constraints(wrong, _base(), strict=True)


@pytest.mark.parametrize(
    ("requirement", "extra", "compatible"),
    [
        ("torch>=2.13,<2.15", False, True),
        ("torch>=2.15", False, False),
        ("torch==2.14.1", False, True),
        ("nvidia-cublas>=13", False, False),
        ('torch>=2.15; python_version < "3.14"', False, True),
        ('torch>=2.15; python_version >= "3.14"', False, False),
        ('torch>=2.15; extra == "accelerated"', False, True),
        ('torch>=2.15; extra == "accelerated"', True, False),
    ],
)
def test_base_compatibility_uses_wheel_specifiers_target_python_and_requested_extras(
    tmp_path: Path, requirement: str, extra: bool, compatible: bool
):
    base = {**_base(), "python": "3.14.10"}
    paths = []
    for name, dependency in (
        ("adapter", requirement),
        ("consumer", "adapter[accelerated]" if extra else "adapter"),
    ):
        path = tmp_path / f"{name}-1.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(
                f"{name}-1.0.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\nRequires-Dist: {dependency}\n",
            )
        paths.append(path)
    if compatible:
        validate_base_requirements(tuple(paths), base)
    else:
        with pytest.raises(
            EnvironmentError,
            match="adapter requires .*base provides (torch 2.14.1\\+cpu|nvidia-cublas absent)",
        ):
            validate_base_requirements(tuple(paths), base)


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


@pytest.mark.parametrize("version", ["2.14.1+cu130", "2.15.0.dev1+vendor"])
def test_explicit_custom_constraints_replace_both_closures_without_mutating_manifests(version: str):
    custom = _base(version)
    custom.update(source="user-defined", python="3.14.10")
    custom["replaces"] = ["sympy", "torch"]
    custom["wheels"] = [
        {
            "name": "torch",
            "version": version,
            "filename": f"torch-{version}-cp314-cp314-manylinux_2_28_x86_64.whl",
            "sha256": "b" * 64,
        },
        {
            "name": "sympy",
            "version": "1.15",
            "filename": "sympy-1.15-py3-none-any.whl",
            "sha256": "c" * 64,
        },
    ]
    published = _base()
    before = deepcopy((custom, published))
    engine = (
        "torch==2.14.1+cpu --hash=sha256:" + "a" * 64 + "\n"
        "sympy==1.14 --hash=sha256:" + "d" * 64 + "\n"
        "nvidia-cublas==13.0 --hash=sha256:" + "e" * 64 + "\n"
        'other==3.0 ; python_version >= "3.12" --hash=sha256:' + "f" * 64 + "\n"
    )
    assert effective_constraints(engine, custom, custom=True, published_base=published) == (
        'other==3.0 ; python_version >= "3.12" --hash=sha256:' + "f" * 64 + "\n"
        "sympy==1.15 --hash=sha256:" + "c" * 64 + "\n"
        f"torch=={version} --hash=sha256:" + "b" * 64 + "\n"
    )
    assert (custom, published) == before
    with pytest.raises(EnvironmentError, match="disagrees with engine lock"):
        effective_constraints(engine, custom)


@pytest.mark.parametrize(
    ("present", "required"), [("2.14.0+cu130", "2.14.1"), ("2.15.0rc1+vendor", "2.15.0")]
)
def test_custom_torch_minimum_reports_versions_and_rebuild_instruction(present: str, required: str):
    custom = _base(present)
    custom.update(source="user-defined", python="3.14.10")
    with pytest.raises(EnvironmentError) as error:
        effective_constraints(
            f"torch=={required} --hash=sha256:" + "a" * 64,
            custom,
            custom=True,
            published_base=_base(),
        )
    assert f"requires torch {required}" in str(error.value)
    assert f"provides torch {present}" in str(error.value)
    assert "rebuild with dinkster-env update --variant NAME --variants PATH --yes" in str(
        error.value
    )


def test_custom_torch_minimum_uses_target_python_markers():
    custom = _base()
    custom.update(source="user-defined", python="3.14.10")
    engine = (
        'torch==2.14.1 ; python_version >= "3.14" --hash=sha256:' + "a" * 64 + "\n"
        'torch==99.0 ; python_version < "3.14" --hash=sha256:' + "c" * 64 + "\n"
    )
    assert effective_constraints(engine, custom, custom=True) == (
        "torch==2.14.1+cpu --hash=sha256:" + "b" * 64 + "\n"
    )
    custom["python"] = "3.13.10"
    with pytest.raises(EnvironmentError, match="requires torch 99.0"):
        effective_constraints(engine, custom, custom=True)


def test_custom_context_cannot_weaken_strict_published_alignment():
    custom = _base()
    custom.update(source="user-defined", python="3.14.10")
    with pytest.raises(EnvironmentError, match="strict=False"):
        effective_constraints("", custom, custom=True, strict=True)
    with pytest.raises(EnvironmentError, match="user-defined"):
        effective_constraints("", _base(), custom=True)
    with pytest.raises(EnvironmentError, match="requires custom=True"):
        effective_constraints("", _base(), published_base=_base())
    engine = "torch==2.14.1+cpu --hash=sha256:" + "b" * 64
    assert effective_constraints(engine, _base(), strict=True) == engine + "\n"
    with pytest.raises(EnvironmentError, match="exact engine lock"):
        effective_constraints(engine.replace("b" * 64, "c" * 64), _base(), strict=True)
    with pytest.raises(EnvironmentError, match="disagrees with engine lock"):
        effective_constraints(engine.replace("2.14.1+cpu", "2.14.1"), _base(), strict=True)
