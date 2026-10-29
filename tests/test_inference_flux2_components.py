"""Torch-free Flux2 per-component planning and identity tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import BFLOAT16, FLOAT32, ComponentBinding, ComponentPlan
from dinkster_inference import flux2_assembly as assembly
from dinkster_inference.weights import WeightEntry, WeightSource


@dataclass(frozen=True)
class _Source:
    path: Path
    asset_size: int | None
    asset_digest: str | None

    def keys(self) -> tuple[str, ...]:
        return ()

    def entry(self, key: str) -> WeightEntry:
        raise KeyError(key)

    def metadata(self) -> dict[str, str]:
        return {}


@dataclass(frozen=True)
class _BareSource:
    """A geometry source that carries no asset identity."""

    path: Path

    def keys(self) -> tuple[str, ...]:
        return ()

    def entry(self, key: str) -> WeightEntry:
        raise KeyError(key)

    def metadata(self) -> dict[str, str]:
        return {}


def _plan(role: str, path: Path) -> ComponentPlan[object]:
    return ComponentPlan(role, path, object(), {}, {}, {})


def test_split_component_binds_asset_identity_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "text.safetensors"
    digest = "blake3:" + "1" * 64
    monkeypatch.setattr(
        assembly,
        "plan_flux2_component",
        lambda _source, role: ("dinkster.flux2_klein_4b", _plan(role, path)),
    )
    source = _Source(path, 10, digest)

    planned = assembly.plan_flux2_split_component(
        cast("WeightSource", source), role="qwen3_4b", path=path
    )
    identity = assembly.flux2_component_runtime_identity(planned, BFLOAT16)

    assert planned.role == "qwen3_4b"
    assert planned.family_id == "dinkster.flux2_klein_4b"
    assert f"asset_digest={digest}" in planned.plan.identity_facts
    assert "asset_size=10" in planned.plan.identity_facts
    ComponentBinding("qwen3_4b", "dinkster.flux2_klein_4b", identity)
    assert identity == assembly.flux2_component_runtime_identity(planned, BFLOAT16)
    assert identity.startswith("native:dinkster.flux2_klein_4b:")


def test_same_geometry_different_bytes_never_share_an_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "text.safetensors"
    monkeypatch.setattr(
        assembly,
        "plan_flux2_component",
        lambda _source, role: ("dinkster.flux2_klein_4b", _plan(role, path)),
    )
    first = assembly.plan_flux2_split_component(
        cast("WeightSource", _Source(path, 10, "blake3:" + "1" * 64)),
        role="qwen3_4b",
        path=path,
    )
    second = assembly.plan_flux2_split_component(
        cast("WeightSource", _Source(path, 10, "blake3:" + "2" * 64)),
        role="qwen3_4b",
        path=path,
    )

    assert assembly.flux2_component_runtime_identity(
        first, BFLOAT16
    ) != assembly.flux2_component_runtime_identity(second, BFLOAT16)


def test_component_identity_binds_compute_dtype_per_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vae.safetensors"
    monkeypatch.setattr(
        assembly,
        "plan_flux2_component",
        lambda _source, role: ("dinkster.flux2", _plan(role, path)),
    )
    planned = assembly.plan_flux2_split_component(
        cast("WeightSource", _Source(path, 7, "blake3:" + "3" * 64)), role="vae", path=path
    )

    assert assembly.flux2_component_runtime_identity(
        planned, FLOAT32
    ) != assembly.flux2_component_runtime_identity(planned, BFLOAT16)


def test_split_component_refuses_path_mismatch_and_missing_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"
    monkeypatch.setattr(
        assembly,
        "plan_flux2_component",
        lambda _source, role: ("dinkster.flux2_dev", _plan(role, path)),
    )
    with pytest.raises(assembly.Flux2ComponentAssemblyError, match="path differs"):
        assembly.plan_flux2_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "4" * 64)),
            role="diffusion",
            path=tmp_path / "other.safetensors",
        )
    with pytest.raises(assembly.Flux2ComponentAssemblyError, match="must carry asset identity"):
        assembly.plan_flux2_split_component(
            cast("WeightSource", _BareSource(path)), role="diffusion", path=path
        )
    with pytest.raises(assembly.Flux2ComponentAssemblyError, match="must carry asset identity"):
        assembly.plan_flux2_split_component(
            cast("WeightSource", _Source(path, 5, None)), role="diffusion", path=path
        )
    with pytest.raises(assembly.Flux2ComponentAssemblyError, match="must carry asset identity"):
        assembly.plan_flux2_split_component(
            cast("WeightSource", _Source(path, None, "blake3:" + "4" * 64)),
            role="diffusion",
            path=path,
        )


def test_split_component_wraps_geometry_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"

    def refuse(_source: object, _role: object) -> tuple[str, ComponentPlan[object]]:
        raise ValueError("weight geometry is not a Flux2 diffusion component")

    monkeypatch.setattr(assembly, "plan_flux2_component", refuse)

    with pytest.raises(assembly.Flux2ComponentAssemblyError, match="weight geometry"):
        assembly.plan_flux2_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "5" * 64)),
            role="diffusion",
            path=path,
        )


def test_planned_component_requires_the_matching_plan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires the vae plan"):
        assembly.Flux2PlannedComponent(
            "vae", "dinkster.flux2", _plan("diffusion", tmp_path / "vae.safetensors")
        )
