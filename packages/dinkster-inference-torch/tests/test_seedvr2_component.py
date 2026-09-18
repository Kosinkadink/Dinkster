from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import FLOAT16, SeedVR2ComponentAssemblyError
from dinkster_inference_torch import seedvr2_component as component
from dinkster_inference_torch.operations import CastOperations


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
    def read_configuration(file: BinaryIO, key: str) -> bytes:
        seen.update(configuration_file=file, configuration_key=key)
        return b'{"format":"int8_tensorwise"}'

    source = SimpleNamespace(read_uint8_configuration_from_file=read_configuration)
    planned = SimpleNamespace(component="diffusion", quant={})

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path, header_source=source)
        return source

    def plan(candidate: object, *, role: str, path: Path) -> object:
        seen.update(planner_source=candidate, planner_role=role, planner_path=path)
        planned.component = role
        return planned

    def identity(candidate: object, role: str, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_role=role, identity_dtype=dtype)
        return expected_identity

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_seedvr2_split_component", plan)
    monkeypatch.setattr(component, "seedvr2_component_runtime_identity", identity)
    return planned


@pytest.mark.parametrize(
    ("role", "builder"),
    (
        ("diffusion", component._build_diffusion),  # pyright: ignore[reportPrivateUsage]
        ("vae", component._build_vae),  # pyright: ignore[reportPrivateUsage]
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
    expected_identity = "native:dinkster.seedvr2:" + "a" * 64
    seen: dict[str, object] = {}
    planned = _patch_planning(monkeypatch, expected_identity=expected_identity, seen=seen)
    module = torch.nn.Identity()
    source_parameter: torch.nn.Parameter | None = None
    source_buffer: torch.Tensor | None = None
    cast_parameter: torch.nn.Parameter | None = None
    if role == "diffusion":
        source_parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        module.register_parameter("source_parameter", source_parameter)
        source_buffer = torch.ones(1, dtype=torch.bfloat16)
        module.register_buffer("source_buffer", source_buffer)
        module.register_buffer("positive_conditioning", torch.ones(1))
        module.register_buffer("negative_conditioning", torch.zeros(1))
        linear = CastOperations(torch.float16).linear(2, 2, bias=False)
        cast_parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.bfloat16))
        linear.weight = cast_parameter
        module.add_module("linear", linear)

    def load(candidate: object, load_builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast(SimpleNamespace, seen["planner_source"])
        seen.update(load_plan=candidate, load_builder=load_builder, load_kwargs=kwargs)
        assert kwargs["preserve_equal_width_cast_storage"] is (role == "diffusion")
        assert kwargs["source_file"] is seen["header_handle"]
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        assert pinned.read_uint8_configuration("layer.comfy_quant") == (
            b'{"format":"int8_tensorwise"}'
        )
        return module

    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_seedvr2_component(
        path,
        asset=asset,
        expected_role=cast("object", role),  # pyright: ignore[reportArgumentType]
        expected_identity=expected_identity,
        compute_dtype=torch.float16,
    )

    assert loaded.role == role
    assert loaded.module is module
    assert loaded.plan is planned
    assert loaded.runtime_identity == expected_identity
    if role == "diffusion":
        assert tuple(module.state_dict()) == ("source_parameter", "source_buffer", "linear.weight")
        assert module.get_parameter("linear.weight") is cast_parameter
        assert source_parameter is not None
        assert module.get_parameter("source_parameter").untyped_storage().data_ptr() != (
            source_parameter.untyped_storage().data_ptr()
        )
        assert source_buffer is not None
        assert module.get_buffer("source_buffer").untyped_storage().data_ptr() != (
            source_buffer.untyped_storage().data_ptr()
        )
        assert tuple(name for name, _value in module.named_buffers()) == (
            "source_buffer",
            "positive_conditioning",
            "negative_conditioning",
        )
    assert seen["header_path"] == path
    assert seen["planner_role"] == role
    assert seen["identity_plan"] is planned
    assert seen["identity_role"] == role
    assert seen["identity_dtype"] is FLOAT16
    assert seen["load_plan"] is planned
    assert seen["load_builder"] is builder
    assert seen["configuration_file"] is seen["header_handle"]
    assert seen["configuration_key"] == "layer.comfy_quant"
    kwargs = cast("dict[str, object]", seen["load_kwargs"])
    assert kwargs["compute_dtype"] is torch.float16
    assert kwargs["fp8_matmul"] is False


def test_component_loader_refuses_identity_mismatch_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    seen: dict[str, object] = {}
    _patch_planning(
        monkeypatch,
        expected_identity="native:dinkster.seedvr2:" + "b" * 64,
        seen=seen,
    )

    def fail_load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        pytest.fail("payload loading must not run")

    monkeypatch.setattr(component, "_load_component", fail_load)

    with pytest.raises(SeedVR2ComponentAssemblyError, match="expected component identity"):
        component.load_seedvr2_component(
            path,
            asset=_asset(path),
            expected_role="diffusion",
            expected_identity="native:dinkster.seedvr2:" + "c" * 64,
            compute_dtype=torch.float16,
        )


def test_component_loader_refuses_size_unavailable_and_invalid_arguments(tmp_path: Path) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    identity = "native:dinkster.seedvr2:" + "d" * 64
    mismatched = AssetRef(asset.digest, asset.name, asset.size + 1, resolver=_FixedResolver(path))
    with pytest.raises(SeedVR2ComponentAssemblyError, match="byte size differs"):
        component.load_seedvr2_component(
            path,
            asset=mismatched,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.float16,
        )
    unavailable = AssetRef(asset.digest, asset.name, asset.size, resolver=_MissingResolver())
    with pytest.raises(SeedVR2ComponentAssemblyError, match="artifact is unavailable"):
        component.load_seedvr2_component(
            path,
            asset=unavailable,
            expected_role="vae",
            expected_identity=identity,
            compute_dtype=torch.float16,
        )
    with pytest.raises(TypeError, match="must be an AssetRef"):
        component.load_seedvr2_component(
            path,
            asset=cast("object", SimpleNamespace()),  # pyright: ignore[reportArgumentType]
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.float16,
        )
    with pytest.raises(ValueError, match="requires an expected identity"):
        component.load_seedvr2_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity="",
            compute_dtype=torch.float16,
        )
    with pytest.raises(TypeError, match="compute dtype must be"):
        component.load_seedvr2_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.float64,
        )
