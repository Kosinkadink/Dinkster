"""Torch-free classic LTX-Video split-component identity tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT32,
    LTXV_2B_V09_CONFIG,
    LTXV_2B_V09_VAE_CONFIG,
    T5_XXL_CONFIG,
    ComponentBinding,
    ComponentPlan,
    LTXVStandaloneComponentPlan,
)
from dinkster_inference import ltx_component as component
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
    path: Path

    def keys(self) -> tuple[str, ...]:
        return ()

    def entry(self, key: str) -> WeightEntry:
        raise KeyError(key)

    def metadata(self) -> dict[str, str]:
        return {}


def _plan(role: str, path: Path) -> LTXVStandaloneComponentPlan:
    config = {
        "diffusion": LTXV_2B_V09_CONFIG,
        "t5xxl": T5_XXL_CONFIG,
        "vae": LTXV_2B_V09_VAE_CONFIG,
    }[role]
    plan = ComponentPlan(role, path, config, {}, {}, {})
    return LTXVStandaloneComponentPlan(cast("Any", role), cast("Any", plan))


def test_split_component_binds_asset_identity_and_compute_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vae.safetensors"
    digest = "blake3:" + "1" * 64
    monkeypatch.setattr(
        component,
        "plan_ltxv_standalone_component",
        lambda _source, role: _plan(role, path),
    )

    planned = component.plan_ltxv_split_component(
        cast("WeightSource", _Source(path, 10, digest)), role="vae", path=path
    )
    identity = component.ltxv_component_runtime_identity(planned, FLOAT32)

    assert f"asset_digest={digest}" in planned.component.identity_facts
    assert "asset_size=10" in planned.component.identity_facts
    assert identity != component.ltxv_component_runtime_identity(planned, BFLOAT16)
    ComponentBinding("vae", "dinkster.ltxv", identity)


def test_same_geometry_with_different_bytes_has_different_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "diffusion.safetensors"
    monkeypatch.setattr(
        component,
        "plan_ltxv_standalone_component",
        lambda _source, role: _plan(role, path),
    )
    first = component.plan_ltxv_split_component(
        cast("WeightSource", _Source(path, 10, "blake3:" + "1" * 64)),
        role="diffusion",
        path=path,
    )
    second = component.plan_ltxv_split_component(
        cast("WeightSource", _Source(path, 10, "blake3:" + "2" * 64)),
        role="diffusion",
        path=path,
    )

    assert component.ltxv_component_runtime_identity(
        first, BFLOAT16
    ) != component.ltxv_component_runtime_identity(second, BFLOAT16)


def test_split_component_refuses_path_mismatch_and_missing_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"
    monkeypatch.setattr(
        component,
        "plan_ltxv_standalone_component",
        lambda _source, role: _plan(role, path),
    )

    with pytest.raises(component.LTXVComponentAssemblyError, match="path differs"):
        component.plan_ltxv_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "3" * 64)),
            role="diffusion",
            path=tmp_path / "other.safetensors",
        )
    with pytest.raises(component.LTXVComponentAssemblyError, match="must carry asset identity"):
        component.plan_ltxv_split_component(
            cast("WeightSource", _BareSource(path)), role="diffusion", path=path
        )


def test_split_component_wraps_geometry_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"

    def refuse(_source: object, _role: object) -> object:
        raise ValueError("not an exact classic LTX-Video component")

    monkeypatch.setattr(component, "plan_ltxv_standalone_component", refuse)

    with pytest.raises(component.LTXVComponentAssemblyError, match="exact classic LTX-Video"):
        component.plan_ltxv_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "4" * 64)),
            role="t5xxl",
            path=path,
        )
