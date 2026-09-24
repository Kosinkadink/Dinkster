from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    BFLOAT16,
    WAN21_CAUSAL_AR_1_3B,
    WAN21_HUMO_17B,
    WAN22_S2V_14B,
    WAN22_WANDANCER_14B,
    Wan21ComponentAssemblyError,
    Wan21StandaloneComponentRole,
)
from dinkster_inference_torch import wan21_component as component
from dinkster_inference_torch.wan21_causal import Wan21CausalModel
from dinkster_inference_torch.wan21_humo import Wan21HumoModel
from dinkster_inference_torch.wan22_dancer import Wan22DancerModel
from dinkster_inference_torch.wan22_s2v import Wan22S2VModel


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
    tokenizer: bytes = b"sentencepiece",
) -> SimpleNamespace:
    source = SimpleNamespace()
    planned = SimpleNamespace(
        role="umt5xxl",
        component=SimpleNamespace(),
        tokenizer_source_key="spiece_model",
    )

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path, header_source=source)
        return source

    def plan(candidate: object, *, role: str, path: Path, family: str) -> object:
        seen.update(
            planner_source=candidate,
            planner_role=role,
            planner_path=path,
            planner_family=family,
        )
        planned.role = role
        planned.tokenizer_source_key = "spiece_model" if role == "umt5xxl" else ""
        return planned

    def identity(candidate: object, dtype: object) -> str:
        seen.update(identity_plan=candidate, identity_dtype=dtype)
        return expected_identity

    def load_tokenizer(
        handle: BinaryIO, candidate: object, keys: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        key = keys[0]
        seen.update(tokenizer_handle=handle, tokenizer_source=candidate, tokenizer_key=key)
        return {key: torch.tensor(tuple(tokenizer), dtype=torch.uint8)}

    def plan_wan21(candidate: object, *, role: str, path: Path) -> object:
        return plan(candidate, role=role, path=path, family="wan21")

    def plan_wan22(candidate: object, *, role: str, path: Path) -> object:
        return plan(candidate, role=role, path=path, family="wan22")

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_wan21_split_component", plan_wan21)
    monkeypatch.setattr(component, "plan_wan22_split_component", plan_wan22)
    monkeypatch.setattr(component, "wan21_component_runtime_identity", identity)
    monkeypatch.setattr(component, "load_tensors_from_file", load_tokenizer)
    return planned


def test_component_loader_selects_wan22_planner_from_expected_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    identity = "native:dinkster.wan22:" + "2" * 64
    seen: dict[str, object] = {}
    _patch_planning(monkeypatch, expected_identity=identity, seen=seen)

    def load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        return torch.nn.Identity()

    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_wan21_component(
        path,
        asset=_asset(path),
        expected_role="diffusion",
        expected_identity=identity,
        compute_dtype=torch.bfloat16,
    )

    assert loaded.runtime_identity == identity
    assert seen["planner_family"] == "wan22"


@pytest.mark.parametrize(
    ("role", "builder_name"),
    [
        ("diffusion", "Wan21Model"),
        ("umt5xxl", "T5TextModel"),
        ("vae", "_build_vae"),
    ],
)
def test_component_loader_preserves_identity_open_descriptor_and_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    builder_name: str,
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    expected_identity = "native:dinkster.wan21:" + "a" * 64
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

    loaded = component.load_wan21_component(
        path,
        asset=asset,
        expected_role=cast("Wan21StandaloneComponentRole", role),
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
    assert seen["identity_dtype"] is BFLOAT16
    assert getattr(seen["load_builder"], "__name__", None) == builder_name
    tokenizer_attribute = component._WAN21_TOKENIZER_ATTRIBUTE  # pyright: ignore[reportPrivateUsage]
    if role == "umt5xxl":
        assert module.__dict__[tokenizer_attribute] == b"sentencepiece"
        assert seen["tokenizer_handle"] is seen["header_handle"]
        assert seen["tokenizer_source"] is seen["header_source"]
        assert seen["tokenizer_key"] == "spiece_model"
    else:
        assert tokenizer_attribute not in module.__dict__


@pytest.mark.parametrize(
    ("config", "expected_builder"),
    (
        (WAN21_CAUSAL_AR_1_3B, Wan21CausalModel),
        (WAN21_HUMO_17B, Wan21HumoModel),
        (WAN22_S2V_14B, Wan22S2VModel),
        (WAN22_WANDANCER_14B, Wan22DancerModel),
    ),
)
def test_component_loader_selects_variant_model_from_exact_planned_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: object,
    expected_builder: type[torch.nn.Module],
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    identity = "native:dinkster.wan21:" + "5" * 64
    seen: dict[str, object] = {}
    planned = _patch_planning(monkeypatch, expected_identity=identity, seen=seen)
    planned.component.config = config
    module = torch.nn.Identity()

    def load(_candidate: object, builder: object, **_kwargs: object) -> torch.nn.Module:
        seen["builder"] = builder
        return module

    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_wan21_component(
        path,
        asset=asset,
        expected_role="diffusion",
        expected_identity=identity,
        compute_dtype=torch.bfloat16,
    )

    assert loaded.module is module
    assert seen["builder"] is expected_builder


def test_component_loader_refuses_identity_mismatch_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    seen: dict[str, object] = {}
    _patch_planning(
        monkeypatch,
        expected_identity="native:dinkster.wan21:" + "b" * 64,
        seen=seen,
    )

    def fail_load(*_args: object, **_kwargs: object) -> torch.nn.Module:
        pytest.fail("payload loading must not run")

    monkeypatch.setattr(component, "_load_component", fail_load)

    with pytest.raises(Wan21ComponentAssemblyError, match="expected component identity"):
        component.load_wan21_component(
            path,
            asset=_asset(path),
            expected_role="diffusion",
            expected_identity="native:dinkster.wan21:" + "c" * 64,
            compute_dtype=torch.bfloat16,
        )


def test_component_loader_refuses_size_unavailable_and_invalid_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _component_path(tmp_path)
    asset = _asset(path)
    identity = "native:dinkster.wan21:" + "d" * 64
    mismatched = AssetRef(
        asset.digest,
        asset.name,
        asset.size + 1,
        resolver=_FixedResolver(path),
    )
    with pytest.raises(Wan21ComponentAssemblyError, match="byte size differs"):
        component.load_wan21_component(
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
    with pytest.raises(Wan21ComponentAssemblyError, match="artifact is unavailable"):
        component.load_wan21_component(
            path,
            asset=unavailable,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.bfloat16,
        )
    with pytest.raises(TypeError, match="compute dtype must be"):
        component.load_wan21_component(
            path,
            asset=asset,
            expected_role="diffusion",
            expected_identity=identity,
            compute_dtype=torch.float64,
        )
