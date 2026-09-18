"""Torch-free TripoSplat per-component planning and identity tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    ComponentBinding,
    ComponentPlan,
    TensorGeometry,
    TripoSplatComponentRole,
    WeightEntry,
    plan_triposplat_component,
)
from dinkster_inference import triposplat_assembly as assembly
from dinkster_inference.weights import WeightSource

GOLDENS = Path(__file__).parent / "goldens" / "triposplat_headers.json"


def golden_shapes(section: str) -> dict[str, tuple[int, ...]]:
    payload = json.loads(GOLDENS.read_text())
    return {key: tuple(shape) for key, shape in payload[section].items()}


class PathHeaderSource:
    def __init__(self, shapes: Mapping[str, tuple[int, ...]], path: Path) -> None:
        self.shapes = dict(shapes)
        self.path = path

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], FLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


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


def test_plan_dit_accepts_bare_and_prefixed_split_files(tmp_path: Path) -> None:
    shapes = golden_shapes("diffusion")
    path = tmp_path / "dit.safetensors"

    bare = plan_triposplat_component(cast("WeightSource", PathHeaderSource(shapes, path)), "dit")
    prefixed_shapes = {f"model.diffusion_model.{key}": shape for key, shape in shapes.items()}
    prefixed = plan_triposplat_component(
        cast("WeightSource", PathHeaderSource(prefixed_shapes, path)), "dit"
    )

    assert bare.component == "dit"
    assert prefixed.component == "dit"
    assert set(bare.keys) == set(prefixed.keys)
    assert bare.keys["input_layer.weight"] == "input_layer.weight"
    assert prefixed.keys["input_layer.weight"] == "model.diffusion_model.input_layer.weight"


def test_plan_accepts_the_bare_conditioner_and_decoder_headers(tmp_path: Path) -> None:
    vision = plan_triposplat_component(
        cast(
            "WeightSource",
            PathHeaderSource(golden_shapes("dinov3_vith"), tmp_path / "dino.safetensors"),
        ),
        "dinov3-vision-conditioner",
    )
    decoder = plan_triposplat_component(
        cast(
            "WeightSource",
            PathHeaderSource(golden_shapes("gaussian_decoder"), tmp_path / "vae.safetensors"),
        ),
        "gaussian-decoder",
    )

    assert vision.component == "dinov3-vision-conditioner"
    assert decoder.component == "gaussian-decoder"


def test_plan_refuses_cross_role_headers(tmp_path: Path) -> None:
    decoder_source = PathHeaderSource(
        golden_shapes("gaussian_decoder"), tmp_path / "vae.safetensors"
    )
    with pytest.raises(ValueError, match="not a TripoSplat DiT"):
        plan_triposplat_component(cast("WeightSource", decoder_source), "dit")
    with pytest.raises(ValueError, match="not a DINOv3"):
        plan_triposplat_component(cast("WeightSource", decoder_source), "dinov3-vision-conditioner")
    dit_source = PathHeaderSource(golden_shapes("diffusion"), tmp_path / "dit.safetensors")
    with pytest.raises(ValueError, match="not a TripoSplat gaussian decoder"):
        plan_triposplat_component(cast("WeightSource", dit_source), "gaussian-decoder")


def test_split_component_binds_asset_identity_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dit.safetensors"
    digest = "blake3:" + "1" * 64
    monkeypatch.setattr(
        assembly,
        "plan_triposplat_component",
        lambda _source, role: _plan(role, path),
    )
    source = _Source(path, 10, digest)

    planned = assembly.plan_triposplat_split_component(
        cast("WeightSource", source), role="dit", path=path
    )
    identity = assembly.triposplat_component_runtime_identity(planned, FLOAT16)

    assert planned.role == "dit"
    assert planned.family_id == "dinkster.triposplat"
    assert f"asset_digest={digest}" in planned.plan.identity_facts
    assert "asset_size=10" in planned.plan.identity_facts
    ComponentBinding("dit", "dinkster.triposplat", identity)
    assert identity == assembly.triposplat_component_runtime_identity(planned, FLOAT16)
    assert identity.startswith("native:dinkster.triposplat:")


def test_same_geometry_different_bytes_never_share_an_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dino.safetensors"
    monkeypatch.setattr(
        assembly,
        "plan_triposplat_component",
        lambda _source, role: _plan(role, path),
    )
    first = assembly.plan_triposplat_split_component(
        cast("WeightSource", _Source(path, 10, "blake3:" + "1" * 64)),
        role="dinov3-vision-conditioner",
        path=path,
    )
    second = assembly.plan_triposplat_split_component(
        cast("WeightSource", _Source(path, 10, "blake3:" + "2" * 64)),
        role="dinov3-vision-conditioner",
        path=path,
    )

    assert assembly.triposplat_component_runtime_identity(
        first, FLOAT16
    ) != assembly.triposplat_component_runtime_identity(second, FLOAT16)


def test_component_identity_binds_compute_dtype_per_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"
    monkeypatch.setattr(
        assembly,
        "plan_triposplat_component",
        lambda _source, role: _plan(role, path),
    )

    def planned_as(role: str) -> assembly.TripoSplatPlannedComponent:
        return assembly.plan_triposplat_split_component(
            cast("WeightSource", _Source(path, 7, "blake3:" + "3" * 64)),
            role=cast("TripoSplatComponentRole", role),
            path=path,
        )

    decoder = planned_as("gaussian-decoder")
    assert assembly.triposplat_component_runtime_identity(
        decoder, FLOAT32
    ) != assembly.triposplat_component_runtime_identity(decoder, BFLOAT16)
    # The dtype occupies a role-specific identity slot, so the same
    # dtype on different roles never collides.
    identities = {
        assembly.triposplat_component_runtime_identity(planned_as(role), FLOAT16)
        for role in ("dit", "dinov3-vision-conditioner", "gaussian-decoder")
    }
    assert len(identities) == 3


def test_split_component_refuses_path_mismatch_and_missing_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"
    monkeypatch.setattr(
        assembly,
        "plan_triposplat_component",
        lambda _source, role: _plan(role, path),
    )
    with pytest.raises(assembly.TripoSplatComponentAssemblyError, match="path differs"):
        assembly.plan_triposplat_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "4" * 64)),
            role="dit",
            path=tmp_path / "other.safetensors",
        )
    with pytest.raises(
        assembly.TripoSplatComponentAssemblyError, match="must carry asset identity"
    ):
        assembly.plan_triposplat_split_component(
            cast("WeightSource", _BareSource(path)), role="dit", path=path
        )
    with pytest.raises(
        assembly.TripoSplatComponentAssemblyError, match="must carry asset identity"
    ):
        assembly.plan_triposplat_split_component(
            cast("WeightSource", _Source(path, 5, None)), role="dit", path=path
        )
    with pytest.raises(
        assembly.TripoSplatComponentAssemblyError, match="must carry asset identity"
    ):
        assembly.plan_triposplat_split_component(
            cast("WeightSource", _Source(path, None, "blake3:" + "4" * 64)),
            role="dit",
            path=path,
        )


def test_split_component_wraps_geometry_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"

    def refuse(_source: object, _role: object) -> ComponentPlan[object]:
        raise ValueError("weight geometry is not a TripoSplat DiT component")

    monkeypatch.setattr(assembly, "plan_triposplat_component", refuse)

    with pytest.raises(assembly.TripoSplatComponentAssemblyError, match="weight geometry"):
        assembly.plan_triposplat_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "5" * 64)),
            role="dit",
            path=path,
        )


def test_planned_component_requires_the_matching_plan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires the gaussian-decoder plan"):
        assembly.TripoSplatPlannedComponent(
            "gaussian-decoder", _plan("dit", tmp_path / "vae.safetensors")
        )
