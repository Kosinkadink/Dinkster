from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import LTXVComponentAssemblyError, LTXVStandaloneComponentRole
from dinkster_inference_torch import ltx_component as component


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
    source = SimpleNamespace()
    planned = SimpleNamespace(role="t5xxl", component=SimpleNamespace())

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path, header_source=source)
        return source

    def plan(candidate: object, *, role: str, path: Path) -> object:
        seen.update(planner_source=candidate, planner_role=role, planner_path=path)
        planned.role = role
        planned.component.component = role
        return planned

    def identity(candidate: object, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_dtype=dtype)
        return expected_identity

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_ltxv_split_component", plan)
    monkeypatch.setattr(component, "ltxv_component_runtime_identity", identity)
    return planned


@pytest.mark.parametrize(
    ("role", "builder_name"),
    [
        ("diffusion", "LTXVModel"),
        ("t5xxl", "T5TextModel"),
        ("vae", "LTXVideoVAE"),
    ],
)
def test_component_loader_preserves_identity_and_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    builder_name: str,
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    expected_identity = "native:dinkster.ltxv:" + "a" * 64
    seen: dict[str, object] = {}
    planned = _patch_planning(monkeypatch, expected_identity=expected_identity, seen=seen)
    module = torch.nn.Identity()

    def load(candidate: object, load_builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast(SimpleNamespace, seen["planner_source"])
        seen.update(load_plan=candidate, load_builder=load_builder, load_kwargs=kwargs)
        assert kwargs["source_file"] is seen["header_handle"]
        assert kwargs["source"] is seen["header_source"]
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return module

    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_ltxv_component(
        path,
        asset=asset,
        expected_role=cast("LTXVStandaloneComponentRole", role),
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
    )

    assert loaded.role == role
    assert loaded.module is module
    assert loaded.plan is planned.component
    assert loaded.runtime_identity == expected_identity
    assert seen["header_path"] == path
    assert seen["planner_role"] == role
    assert seen["identity_plan"] is planned
    assert getattr(seen["identity_dtype"], "name", None) == "bfloat16"
    assert getattr(seen["load_builder"], "__name__", None) == builder_name


def test_component_loader_refuses_identity_mismatch_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    seen: dict[str, object] = {}
    _patch_planning(
        monkeypatch,
        expected_identity="native:dinkster.ltxv:" + "b" * 64,
        seen=seen,
    )

    def fail_load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        pytest.fail("payload loading must not run")

    monkeypatch.setattr(component, "_load_component", fail_load)

    with pytest.raises(LTXVComponentAssemblyError, match="expected component identity"):
        component.load_ltxv_component(
            path,
            asset=_asset(path),
            expected_role="diffusion",
            expected_identity="native:dinkster.ltxv:" + "c" * 64,
            compute_dtype=torch.bfloat16,
        )


def test_component_loader_refuses_size_unavailable_and_invalid_dtype(tmp_path: Path) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    identity = "native:dinkster.ltxv:" + "d" * 64
    mismatched = AssetRef(
        asset.digest,
        asset.name,
        asset.size + 1,
        resolver=_FixedResolver(path),
    )
    with pytest.raises(LTXVComponentAssemblyError, match="byte size differs"):
        component.load_ltxv_component(
            path,
            asset=mismatched,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    unavailable = AssetRef(
        asset.digest,
        asset.name,
        asset.size,
        resolver=_MissingResolver(),
    )
    with pytest.raises(LTXVComponentAssemblyError, match="artifact is unavailable"):
        component.load_ltxv_component(
            path,
            asset=unavailable,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(TypeError, match="compute dtype must be"):
        component.load_ltxv_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.float64,
        )
