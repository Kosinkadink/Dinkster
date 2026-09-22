from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import BFLOAT16
from dinkster_inference_torch import z_image_component as component
from dinkster_inference_torch.z_image import ZImage


class _FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def test_component_loader_preserves_identity_and_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "z_image.safetensors"
    path.write_bytes(b"component")
    asset = AssetRef(
        digest_file(path), path.name, path.stat().st_size, resolver=_FixedResolver(path)
    )
    identity = "native:dinkster.z_image:" + "1" * 64
    source = SimpleNamespace()
    planned = SimpleNamespace(component="diffusion")
    module = cast("ZImage", torch.nn.Identity())
    seen: dict[str, object] = {}

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path, header_source=source)
        return source

    def plan(candidate: object, *, role: str, path: Path) -> object:
        seen.update(planner_source=candidate, planner_role=role, planner_path=path)
        return planned

    def runtime_identity(candidate: object, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_dtype=dtype)
        return identity

    def load(candidate: object, builder: object, **kwargs: object) -> ZImage:
        pinned = cast(SimpleNamespace, seen["planner_source"])
        seen.update(load_plan=candidate, load_builder=builder, load_kwargs=kwargs)
        assert kwargs["source_file"] is seen["header_handle"]
        assert kwargs["source"] is seen["header_source"]
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return module

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_z_image_split_component", plan)
    monkeypatch.setattr(component, "z_image_component_runtime_identity", runtime_identity)
    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_z_image_component(
        path,
        asset=asset,
        expected_role="diffusion",
        expected_identity=identity,
        compute_dtype=torch.bfloat16,
    )

    assert loaded.module is module
    assert loaded.runtime_identity == identity
    assert seen["header_path"] == path
    assert seen["planner_role"] == "diffusion"
    assert seen["identity_plan"] is planned
    assert seen["identity_dtype"] is BFLOAT16
    assert seen["load_plan"] is planned
    assert seen["load_builder"] is component.ZImage
    kwargs = cast("dict[str, object]", seen["load_kwargs"])
    assert kwargs["compute_dtype"] is torch.bfloat16
    assert kwargs["fp8_matmul"] is False
