from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import FLOAT16, TripoSplatComponentAssemblyError
from dinkster_inference_torch import triposplat_assembly as assembly


class FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


class MissingResolver:
    def resolve(self, digest: str) -> None:
        del digest
        return None


def _asset(path: Path) -> AssetRef:
    return AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=FixedResolver(path))


def _component_path(tmp_path: Path) -> Path:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    return path


def _patch_planner_seam(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_identity: str,
    seen: dict[str, object],
) -> SimpleNamespace:
    """Stub header/plan/identity so loader tests exercise only the seam."""

    source = SimpleNamespace(keys=lambda: ())
    planned = SimpleNamespace(plan=SimpleNamespace(), family_id="dinkster.triposplat")

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path)
        return source

    def plan_component(candidate: object, *, role: str, path: Path) -> object:
        seen.update(planner_source=candidate, planner_role=role, planner_path=path)
        return planned

    def component_identity(candidate: object, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_dtype=dtype)
        return expected_identity

    monkeypatch.setattr(assembly, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(assembly, "plan_triposplat_split_component", plan_component)
    monkeypatch.setattr(assembly, "triposplat_component_runtime_identity", component_identity)
    return planned


@pytest.mark.parametrize(
    ("role", "builder"),
    [
        ("dit", assembly.TripoSplatModel),
        ("dinov3-vision-conditioner", assembly.DINOv3ViTModel),
        ("gaussian-decoder", assembly.OctreeGaussianDecoder),
    ],
)
def test_component_loader_preserves_dispatch_identity_and_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, builder: object
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    expected_identity = "native:dinkster.triposplat:" + "a" * 64
    seen: dict[str, object] = {}
    planned = _patch_planner_seam(monkeypatch, expected_identity=expected_identity, seen=seen)
    module = torch.nn.Identity()

    def load(candidate: object, load_builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast(SimpleNamespace, kwargs["source"])
        seen.update(load_plan=candidate, load_builder=load_builder, load_kwargs=kwargs)
        assert kwargs["source_file"] is seen["header_handle"]
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return module

    monkeypatch.setattr(assembly, "_load_component", load)

    loaded = assembly.load_triposplat_component(
        path,
        asset=asset,
        expected_role=cast("Any", role),
        expected_identity=expected_identity,
        compute_dtype=torch.float16,
    )

    assert loaded.role == role
    assert loaded.family_id == "dinkster.triposplat"
    assert loaded.module is module
    assert loaded.plan is planned.plan
    assert loaded.runtime_identity == expected_identity
    assert seen["header_path"] == path
    assert seen["planner_role"] == role
    assert seen["planner_path"] == path
    assert cast(SimpleNamespace, seen["planner_source"]).file is seen["header_handle"]
    assert seen["identity_plan"] is planned
    assert seen["identity_dtype"] is FLOAT16
    assert seen["load_plan"] is planned.plan
    assert seen["load_builder"] is builder
    kwargs = cast("dict[str, object]", seen["load_kwargs"])
    assert kwargs["compute_dtype"] is torch.float16
    assert kwargs["fp8_matmul"] is False


def test_component_loader_refuses_identity_mismatch_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    seen: dict[str, object] = {}
    _patch_planner_seam(
        monkeypatch, expected_identity="native:dinkster.triposplat:" + "c" * 64, seen=seen
    )

    def fail_load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        pytest.fail("payload loading must not run on identity mismatch")

    monkeypatch.setattr(assembly, "_load_component", fail_load)

    with pytest.raises(TripoSplatComponentAssemblyError, match="expected component identity"):
        assembly.load_triposplat_component(
            path,
            asset=_asset(path),
            expected_role="dit",
            expected_identity="native:dinkster.triposplat:" + "d" * 64,
            compute_dtype=torch.float16,
        )


def test_component_loader_refuses_byte_size_disagreeing_with_asset_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    asset = AssetRef(
        digest_file(path), path.name, path.stat().st_size + 1, resolver=FixedResolver(path)
    )

    def fail_header(*_args: object, **_kwargs: object) -> object:
        pytest.fail("header planning must not run on size mismatch")

    monkeypatch.setattr(assembly, "load_safetensors_header_from_file", fail_header)

    with pytest.raises(TripoSplatComponentAssemblyError, match="byte size differs"):
        assembly.load_triposplat_component(
            path,
            asset=asset,
            expected_role="dit",
            expected_identity="native:dinkster.triposplat:" + "e" * 64,
            compute_dtype=torch.float16,
        )


def test_component_loader_reports_unavailable_artifact(tmp_path: Path) -> None:
    path = _component_path(tmp_path)
    asset = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=MissingResolver())
    with pytest.raises(TripoSplatComponentAssemblyError, match="artifact is unavailable"):
        assembly.load_triposplat_component(
            path,
            asset=asset,
            expected_role="dit",
            expected_identity="native:dinkster.triposplat:" + "f" * 64,
            compute_dtype=torch.float16,
        )


def test_component_loader_validates_arguments_before_touching_the_artifact(
    tmp_path: Path,
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    identity = "native:dinkster.triposplat:" + "0" * 64
    with pytest.raises(TypeError, match="must be an AssetRef"):
        assembly.load_triposplat_component(
            path,
            asset=cast("Any", SimpleNamespace(digest="x", size=1)),
            expected_role="dit",
            expected_identity=identity,
            compute_dtype=torch.float16,
        )
    with pytest.raises(ValueError, match="requires an expected identity"):
        assembly.load_triposplat_component(
            path,
            asset=asset,
            expected_role="dit",
            expected_identity="",
            compute_dtype=torch.float16,
        )
    with pytest.raises(TypeError, match="compute dtype must be"):
        assembly.load_triposplat_component(
            path,
            asset=asset,
            expected_role="dit",
            expected_identity=identity,
            compute_dtype=torch.float64,
        )
