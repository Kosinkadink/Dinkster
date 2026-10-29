from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT32,
    DType,
    Ideogram4ComponentAssemblyError,
    ideogram4_layout,
    ideogram4_text_layout,
    plan_ideogram4_component,
)
from dinkster_inference.weights import TensorGeometry, WeightEntry
from dinkster_inference_torch import ideogram4_component as component
from dinkster_inference_torch.ideogram4_dit import Ideogram4DiT
from dinkster_inference_torch.operations import INITLESS


class _FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


class _MissingResolver:
    def resolve(self, digest: str) -> None:
        del digest
        return None


def _asset(path: Path) -> AssetRef:
    return AssetRef(
        digest_file(path), path.name, path.stat().st_size, resolver=_FixedResolver(path)
    )


def _component_path(tmp_path: Path) -> Path:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    return path


def _patch_planning(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_identity: str,
    seen: dict[str, object],
) -> SimpleNamespace:
    def read_configuration(file: BinaryIO, key: str, *, limit: int) -> bytes:
        assert limit == 65_536
        seen.update(configuration_file=file, configuration_key=key)
        return b'{"format":"int8_tensorwise"}'

    source = SimpleNamespace(read_uint8_configuration_from_file=read_configuration)
    planned = SimpleNamespace(component="diffusion", dtypes={})

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path)
        return source

    def plan(candidate: object, *, role: str, path: Path) -> object:
        seen.update(planner_source=candidate, planner_role=role, planner_path=path)
        planned.component = role
        planned.dtypes = (
            {"double_stream_blocks.0.img_attn.proj.weight": FLOAT8_E4M3}
            if role == "diffusion"
            else {}
        )
        return planned

    def identity(candidate: object, role: str, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_role=role, identity_dtype=dtype)
        return expected_identity

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_ideogram4_split_component", plan)
    monkeypatch.setattr(component, "ideogram4_component_runtime_identity", identity)
    return planned


@pytest.mark.parametrize(
    ("role", "builder"),
    (
        ("diffusion", component._build_ideogram4_diffusion),  # pyright: ignore[reportPrivateUsage]
        ("qwen3vl_8b", component._build_ideogram4_text),  # pyright: ignore[reportPrivateUsage]
    ),
)
def test_component_loader_preserves_identity_and_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    builder: object,
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    expected_identity = "native:dinkster.ideogram4:" + "a" * 64
    seen: dict[str, object] = {}
    planned = _patch_planning(monkeypatch, expected_identity=expected_identity, seen=seen)
    module = torch.nn.Identity()

    def load(candidate: object, load_builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast(SimpleNamespace, seen["planner_source"])
        seen.update(load_plan=candidate, load_builder=load_builder, load_kwargs=kwargs)
        assert kwargs["source_file"] is seen["header_handle"]
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        assert pinned.read_uint8_configuration("layer.comfy_quant") == (
            b'{"format":"int8_tensorwise"}'
        )
        return module

    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_ideogram4_component(
        path,
        asset=asset,
        expected_role=cast("object", role),  # pyright: ignore[reportArgumentType]
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
    )

    assert loaded.role == role
    assert loaded.module is module
    assert loaded.plan is planned
    assert loaded.runtime_identity == expected_identity
    assert seen["header_path"] == path
    assert seen["planner_role"] == role
    assert seen["identity_plan"] is planned
    assert seen["identity_role"] == role
    assert seen["identity_dtype"] is BFLOAT16
    assert seen["load_plan"] is planned
    assert seen["load_builder"] is builder
    assert seen["configuration_file"] is seen["header_handle"]
    assert seen["configuration_key"] == "layer.comfy_quant"
    kwargs = cast("dict[str, object]", seen["load_kwargs"])
    assert kwargs["compute_dtype"] is torch.bfloat16
    assert kwargs["fp8_matmul"] is (role == "diffusion")


def test_component_loader_refuses_identity_mismatch_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    seen: dict[str, object] = {}
    _patch_planning(
        monkeypatch,
        expected_identity="native:dinkster.ideogram4:" + "b" * 64,
        seen=seen,
    )

    def fail_load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        pytest.fail("payload loading must not run")

    monkeypatch.setattr(component, "_load_component", fail_load)

    with pytest.raises(Ideogram4ComponentAssemblyError, match="expected component identity"):
        component.load_ideogram4_component(
            path,
            asset=_asset(path),
            expected_role="diffusion",
            expected_identity="native:dinkster.ideogram4:" + "c" * 64,
            compute_dtype=torch.bfloat16,
        )


def test_component_loader_refuses_size_unavailable_and_invalid_arguments(
    tmp_path: Path,
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    identity = "native:dinkster.ideogram4:" + "d" * 64
    mismatched = AssetRef(asset.digest, asset.name, asset.size + 1, resolver=_FixedResolver(path))
    with pytest.raises(Ideogram4ComponentAssemblyError, match="byte size differs"):
        component.load_ideogram4_component(
            path,
            asset=mismatched,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    unavailable = AssetRef(asset.digest, asset.name, asset.size, resolver=_MissingResolver())
    with pytest.raises(Ideogram4ComponentAssemblyError, match="artifact is unavailable"):
        component.load_ideogram4_component(
            path,
            asset=unavailable,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(TypeError, match="must be an AssetRef"):
        component.load_ideogram4_component(
            path,
            asset=cast("object", SimpleNamespace()),  # pyright: ignore[reportArgumentType]
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="requires an expected identity"):
        component.load_ideogram4_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity="",
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(TypeError, match="compute dtype must be"):
        component.load_ideogram4_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.float64,
        )


def test_builders_refuse_non_exact_profiles() -> None:
    with pytest.raises(Ideogram4ComponentAssemblyError, match="supported profile"):
        component._build_ideogram4_text(  # pyright: ignore[reportPrivateUsage]
            SimpleNamespace(), operations=cast("Any", None)
        )
    with pytest.raises(Ideogram4ComponentAssemblyError, match="supported profile"):
        component._build_ideogram4_diffusion(  # pyright: ignore[reportPrivateUsage]
            SimpleNamespace(), operations=cast("Any", None)
        )


class _HeaderSource:
    def __init__(
        self, path: Path, layout: Mapping[str, tuple[int, ...]], dtype: DType = BFLOAT16
    ) -> None:
        self.path = path
        self._geometries = {key: TensorGeometry(shape, dtype) for key, shape in layout.items()}

    def keys(self) -> tuple[str, ...]:
        return tuple(self._geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self._geometries[key]
        return WeightEntry(key=key, geometry=geometry, offset=0, nbytes=geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return {}


def _build_text_module(config: object) -> torch.nn.Module:
    return component._build_ideogram4_text(  # pyright: ignore[reportPrivateUsage]
        config, operations=INITLESS
    )


def _build_dit_module(config: object) -> torch.nn.Module:
    return Ideogram4DiT(cast("Any", config), operations=INITLESS)


@pytest.mark.parametrize(
    ("role", "layout", "build", "storage_dtype"),
    [
        ("qwen3vl_8b", ideogram4_text_layout(), _build_text_module, BFLOAT16),
        ("diffusion", ideogram4_layout(), _build_dit_module, FLOAT32),
    ],
)
def test_plan_model_keys_match_the_loaded_module_exactly(
    role: str,
    layout: Mapping[str, tuple[int, ...]],
    build: Callable[[object], torch.nn.Module],
    storage_dtype: DType,
) -> None:
    plan = plan_ideogram4_component(
        cast("Any", _HeaderSource(Path("/fake/component.safetensors"), layout, storage_dtype)),
        cast("Any", role),
    )
    assert set(plan.dtypes.values()) == {storage_dtype}
    module = build(plan.config)
    assert set(plan.keys) == set(module.state_dict())
