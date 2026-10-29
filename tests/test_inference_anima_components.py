"""Torch-free Anima split-component planning and identity tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import BFLOAT16, FLOAT32, ComponentBinding, ComponentPlan
from dinkster_inference import anima_component as component
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


def _plan(role: str, path: Path) -> ComponentPlan[object]:
    return ComponentPlan(role, path, object(), {}, {}, {})


def test_split_component_binds_asset_identity_and_compute_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "text.safetensors"
    digest = "blake3:" + "1" * 64
    monkeypatch.setattr(
        component,
        "plan_anima_component",
        lambda _source, role: _plan(role, path),
    )

    planned = component.plan_anima_split_component(
        cast("WeightSource", _Source(path, 10, digest)), role="qwen3_06b", path=path
    )
    identity = component.anima_component_runtime_identity(planned, "qwen3_06b", BFLOAT16)

    assert f"asset_digest={digest}" in planned.identity_facts
    assert "asset_size=10" in planned.identity_facts
    assert identity != component.anima_component_runtime_identity(planned, "qwen3_06b", FLOAT32)
    ComponentBinding("qwen3_06b", "dinkster.anima", identity)


def test_same_geometry_different_bytes_never_share_an_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "diffusion.safetensors"
    monkeypatch.setattr(
        component,
        "plan_anima_component",
        lambda _source, role: _plan(role, path),
    )
    first = component.plan_anima_split_component(
        cast("WeightSource", _Source(path, 10, "blake3:" + "1" * 64)),
        role="diffusion",
        path=path,
    )
    second = component.plan_anima_split_component(
        cast("WeightSource", _Source(path, 10, "blake3:" + "2" * 64)),
        role="diffusion",
        path=path,
    )

    assert component.anima_component_runtime_identity(
        first, "diffusion", BFLOAT16
    ) != component.anima_component_runtime_identity(second, "diffusion", BFLOAT16)


def test_split_component_refuses_path_mismatch_and_missing_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"
    monkeypatch.setattr(
        component,
        "plan_anima_component",
        lambda _source, role: _plan(role, path),
    )

    with pytest.raises(component.AnimaComponentAssemblyError, match="path differs"):
        component.plan_anima_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "3" * 64)),
            role="diffusion",
            path=tmp_path / "other.safetensors",
        )
    with pytest.raises(component.AnimaComponentAssemblyError, match="must carry asset identity"):
        component.plan_anima_split_component(
            cast("WeightSource", _BareSource(path)), role="diffusion", path=path
        )
    with pytest.raises(component.AnimaComponentAssemblyError, match="must carry asset identity"):
        component.plan_anima_split_component(
            cast("WeightSource", _Source(path, None, "blake3:" + "4" * 64)),
            role="diffusion",
            path=path,
        )


def test_split_component_wraps_geometry_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"

    def refuse(_source: object, _role: object) -> ComponentPlan[object]:
        raise ValueError("not the exact Anima 2B layout")

    monkeypatch.setattr(component, "plan_anima_component", refuse)

    with pytest.raises(component.AnimaComponentAssemblyError, match="exact Anima 2B layout"):
        component.plan_anima_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "5" * 64)),
            role="diffusion",
            path=path,
        )


def test_component_identity_requires_the_matching_role(tmp_path: Path) -> None:
    planned = _plan("diffusion", tmp_path / "diffusion.safetensors")
    with pytest.raises(ValueError, match="requires the qwen3_06b plan"):
        component.anima_component_runtime_identity(planned, "qwen3_06b", BFLOAT16)
