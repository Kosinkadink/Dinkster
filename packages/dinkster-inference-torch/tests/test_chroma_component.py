from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import BFLOAT16, ChromaComponentAssemblyError
from dinkster_inference_torch import CastOperations, ModuleStateStore
from dinkster_inference_torch import chroma_component as component


class _FixedResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _asset(path: Path) -> AssetRef:
    return AssetRef(
        digest_file(path), path.name, path.stat().st_size, resolver=_FixedResolver(path)
    )


@pytest.mark.parametrize(
    ("role", "builder"),
    (
        ("diffusion", component._build_diffusion),  # pyright: ignore[reportPrivateUsage]
        ("t5xxl", component.T5TextModel),
        ("vae", component.AutoencoderKL),
    ),
)
def test_component_loader_preserves_identity_and_pinned_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    builder: object,
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    asset = _asset(path)
    expected_identity = "native:dinkster.chroma:" + "a" * 64
    seen: dict[str, object] = {}

    def read_configuration(handle: BinaryIO, key: str) -> bytes:
        seen.update(configuration_handle=handle, configuration_key=key)
        return b'{"format":"float8_e4m3fn"}'

    source = SimpleNamespace(read_uint8_configuration_from_file=read_configuration)
    planned = SimpleNamespace(component=role, quant={})

    def load_header(handle: BinaryIO, *, path: Path) -> object:
        seen.update(header_handle=handle, header_path=path)
        return source

    def plan(candidate: object, *, role: str, path: Path) -> object:
        pinned = cast("component._PinnedSource", candidate)  # pyright: ignore[reportPrivateUsage]
        seen.update(
            planner_source=pinned,
            planner_role=role,
            planner_path=path,
            configuration=pinned.read_uint8_configuration("quant.config"),
        )
        return planned

    def identity(candidate: object, role: str, dtype: object, **kwargs: object) -> str:
        seen.update(
            identity_plan=candidate,
            identity_role=role,
            identity_dtype=dtype,
            identity_kwargs=kwargs,
        )
        return expected_identity

    module = CastOperations(torch.bfloat16).linear(2, 2)
    module.load_state_dict(
        {
            "weight": torch.ones(2, 2, dtype=torch.bfloat16),
            "bias": torch.ones(2, dtype=torch.bfloat16),
        },
        strict=True,
        assign=True,
    )

    def load(candidate: object, load_builder: object, **kwargs: object) -> torch.nn.Module:
        pinned = cast("component._PinnedSource", seen["planner_source"])  # pyright: ignore[reportPrivateUsage]
        seen.update(load_plan=candidate, load_builder=load_builder, load_kwargs=kwargs)
        assert pinned.file is kwargs["source_file"]
        assert pinned.asset_digest == asset.digest
        assert pinned.asset_size == asset.size
        return module

    monkeypatch.setattr(component, "load_safetensors_header_from_file", load_header)
    monkeypatch.setattr(component, "plan_chroma_split_component", plan)
    monkeypatch.setattr(component, "chroma_component_runtime_identity", identity)
    monkeypatch.setattr(component, "_load_component", load)

    loaded = component.load_chroma_component(
        path,
        asset=asset,
        expected_role=cast("object", role),  # pyright: ignore[reportArgumentType]
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
        attention_backend=cast(
            "object",
            {"diffusion": "flux", "t5xxl": "t5", "vae": "vae"}[role],
        ),  # pyright: ignore[reportArgumentType]
    )

    assert loaded.role == role
    assert loaded.module is module
    assert loaded.plan is planned
    assert loaded.runtime_identity == expected_identity
    assert loaded.attention_status.primary == "sdpa"
    assert seen["header_path"] == path
    assert seen["planner_role"] == role
    assert seen["identity_plan"] is planned
    assert seen["identity_role"] == role
    assert seen["identity_dtype"] is BFLOAT16
    assert seen["load_plan"] is planned
    load_builder = seen["load_builder"]
    assert isinstance(load_builder, partial)
    assert load_builder.func is builder
    assert load_builder.keywords["attention_kernel"] is not None
    assert seen["identity_kwargs"] == {
        "attention_policy": "auto",
        "attention_route_token": None,
    }
    assert seen["configuration"] == b'{"format":"float8_e4m3fn"}'
    assert seen["configuration_handle"] is seen["header_handle"]
    assert seen["configuration_key"] == "quant.config"
    kwargs = cast("dict[str, object]", seen["load_kwargs"])
    assert kwargs["compute_dtype"] is torch.bfloat16
    assert kwargs["fp8_matmul"] is False
    assert kwargs["source"] is source
    store = ModuleStateStore(loaded.module)
    assert store.max_materialized_itemsize("weight") == 2
    assert store.max_materialized_itemsize("bias") == 2
    assert module.weight.requires_grad is (role != "t5xxl")
    assert module.bias is not None
    assert module.bias.requires_grad is (role != "t5xxl")


@pytest.mark.parametrize(
    ("role", "config", "load_device", "supported", "expected_bound", "support_checked"),
    (
        ("diffusion", "quant.config", torch.device("cuda:0"), True, True, True),
        ("diffusion", "quant.config", torch.device("cpu"), False, False, True),
        ("diffusion", None, torch.device("cuda:0"), True, False, False),
        ("t5xxl", "quant.config", torch.device("cuda:0"), True, False, False),
    ),
)
def test_component_loader_binds_only_checkpoint_declared_diffusion_fp8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    config: str | None,
    load_device: torch.device,
    supported: bool,
    expected_bound: bool,
    support_checked: bool,
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    expected_identity = "native:dinkster.chroma:" + "a" * 64
    layer = component.Fp8Linear(2, 2, compute_dtype=torch.bfloat16)
    module = torch.nn.Module()
    module.add_module("projection", layer)
    planned = SimpleNamespace(
        component=role,
        quant={"projection": SimpleNamespace(config=config)},
    )
    support_calls: list[torch.device] = []
    bind_calls: list[bool] = []

    def load_header(_handle: BinaryIO, *, path: Path) -> object:
        return SimpleNamespace(path=path)

    def plan(_source: object, *, role: str, path: Path) -> object:
        del role, path
        return planned

    def identity(_plan: object, _role: str, _dtype: object, **_kwargs: object) -> str:
        return expected_identity

    def load(_plan: object, _builder: object, **_kwargs: object) -> torch.nn.Module:
        return module

    monkeypatch.setattr(
        component,
        "load_safetensors_header_from_file",
        load_header,
    )
    monkeypatch.setattr(component, "plan_chroma_split_component", plan)
    monkeypatch.setattr(
        component,
        "chroma_component_runtime_identity",
        identity,
    )
    monkeypatch.setattr(component, "_load_component", load)

    def fp8_supported(device: torch.device) -> bool:
        support_calls.append(device)
        return supported

    def bind_fp8(_layer: component.Fp8Linear, enabled: bool) -> None:
        bind_calls.append(enabled)

    monkeypatch.setattr(component, "supports_fp8_matmul", fp8_supported)
    monkeypatch.setattr(component.Fp8Linear, "bind_fp8_matmul", bind_fp8)

    loaded = component.load_chroma_component(
        path,
        asset=_asset(path),
        expected_role=cast("object", role),  # pyright: ignore[reportArgumentType]
        expected_identity=expected_identity,
        compute_dtype=torch.bfloat16,
        load_device=load_device,
        attention_backend=cast(
            "object",
            {"diffusion": "flux", "t5xxl": "t5", "vae": "vae"}[role],
        ),  # pyright: ignore[reportArgumentType]
    )

    assert loaded.module is module
    assert bind_calls == ([True] if expected_bound else [])
    assert support_calls == ([load_device] if support_checked else [])


def test_component_loader_refuses_identity_mismatch_before_payload_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "component.safetensors"
    path.write_bytes(b"component")
    source = SimpleNamespace()
    planned = SimpleNamespace(component="diffusion")

    def fake_header(_handle: BinaryIO, *, path: Path) -> object:
        del path
        return source

    def fake_plan(_source: object, *, role: str, path: Path) -> object:
        del role, path
        return planned

    def fake_identity(_plan: object, _role: str, _dtype: object, **_kwargs: object) -> str:
        return "native:dinkster.chroma:" + "b" * 64

    def fail_payload(*_args: object, **_kwargs: object) -> None:
        pytest.fail("payload loading must not run")

    monkeypatch.setattr(
        component,
        "load_safetensors_header_from_file",
        fake_header,
    )
    monkeypatch.setattr(
        component,
        "plan_chroma_split_component",
        fake_plan,
    )
    monkeypatch.setattr(
        component,
        "chroma_component_runtime_identity",
        fake_identity,
    )
    monkeypatch.setattr(
        component,
        "_load_component",
        fail_payload,
    )

    with pytest.raises(ChromaComponentAssemblyError, match="expected component identity"):
        component.load_chroma_component(
            path,
            asset=_asset(path),
            expected_role="diffusion",
            expected_identity="native:dinkster.chroma:" + "c" * 64,
            compute_dtype=torch.bfloat16,
            attention_backend="flux",
        )
