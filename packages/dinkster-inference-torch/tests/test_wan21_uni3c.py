"""Native Wan 2.1 Uni3C model, assembly, and binding contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import WAN21_UNI3C, PercentRange, wan21_uni3c_model_layout
from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
from dinkster_inference_torch import assemble as assemble_module
from dinkster_inference_torch import assemble_wan21_uni3c
from dinkster_inference_torch import wan21_uni3c as uni3c_module
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.wan21_uni3c import (
    Wan21Uni3C,
    Wan21Uni3CBindingError,
    Wan21Uni3CExecution,
    Wan21Uni3CLayerNormZero,
    snapshot_wan21_uni3c_execution,
    validate_wan21_uni3c_resource,
    wan21_uni3c_resource_digest,
    wan21_uni3c_tensor_digest,
)


def _fill_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(((values + index) % 13 - 6) / 32)


def _small_sealed_model() -> tuple[Wan21Uni3C, str]:
    model = Wan21Uni3C.__new__(Wan21Uni3C)
    torch.nn.Module.__init__(model)
    test_model = cast(Any, model)
    test_model.layer = assemble_module.INITLESS.linear(2, 2, bias=False)
    model.load_state_dict(
        {"layer.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        strict=True,
        assign=True,
    )
    digest = wan21_uni3c_resource_digest("blake3:" + "0" * 64, torch.bfloat16)
    uni3c_module._bind_wan21_uni3c_resource(  # pyright: ignore[reportPrivateUsage]
        model, digest
    )
    return model, digest


def test_exact_meta_layout_matches_the_maintained_checkpoint_contract() -> None:
    with torch.device("meta"):
        model = Wan21Uni3C()
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    assert model.config is WAN21_UNI3C
    assert actual == wan21_uni3c_model_layout()
    assert len(actual) == 490
    assert actual["controlnet_patch_embedding.weight"] == (5120, 36, 1, 2, 2)
    assert actual["controlnet_blocks.19.self_attn.o.weight"] == (1024, 1024)
    assert actual["proj_out.19.weight"] == (5120, 1024)
    assert model.controlnet_mask_embedding.mask_proj[1].eps == 1e-5


def test_layer_norm_zero_matches_reference_affine_and_gate_math() -> None:
    layer = Wan21Uni3CLayerNormZero(4, 6)
    _fill_parameters(layer)
    hidden = torch.randn((2, 3, 6), requires_grad=True)
    temb = torch.randn((2, 4), requires_grad=True)

    shift, scale, gate = layer.linear(layer.silu(temb)).chunk(3, dim=1)
    expected = layer.norm(hidden) * (1.0 + scale[:, None]) + shift[:, None]
    actual, actual_gate = layer(hidden, temb)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_gate, gate[:, None])
    (actual.sum() + actual_gate.sum()).backward()
    assert hidden.grad is not None
    assert temb.grad is not None


def test_assembly_builds_exact_profile_and_seals_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = SimpleNamespace(config=WAN21_UNI3C)

    def load(
        value: object,
        build: Any,
        *,
        compute_dtype: torch.dtype,
        fp8_matmul: bool,
    ) -> torch.nn.Module:
        assert value is component
        assert compute_dtype == torch.bfloat16
        assert fp8_matmul is False
        with torch.device("meta"):
            model = build(WAN21_UNI3C, operations=assemble_module.INITLESS)
        assert type(model) is Wan21Uni3C
        assert set(model.state_dict()) == set(wan21_uni3c_model_layout())
        return model

    monkeypatch.setattr(assemble_module, "_load_component", load)
    plan = cast(
        Any,
        SimpleNamespace(patch=component, asset_digest="blake3:" + "1" * 64),
    )
    assembled = assemble_wan21_uni3c(plan)

    assert assembled.compute_dtype == torch.bfloat16
    assert assembled.attention_status.role == "flux"
    assert assembled.resource_digest == wan21_uni3c_resource_digest(
        plan.asset_digest, torch.bfloat16
    )


def test_resource_binding_accepts_residency_load_and_unload() -> None:
    model, digest = _small_sealed_model()
    test_model = cast(Any, model)
    original = test_model.layer.weight
    mechanism = enroll_component(
        model,
        load_device="cpu",
        offload_device="cpu",
        patch_set=PatchSet(
            {"layer.weight": (PatchEntry(DiffPatch(torch.ones_like(test_model.layer.weight))),)}
        ),
    )

    mechanism.partially_load(None)
    assert test_model.layer.weight is not original
    validate_wan21_uni3c_resource(model, digest)

    mechanism.unload()
    assert test_model.layer.weight is original
    validate_wan21_uni3c_resource(model, digest)


def test_execution_owns_render_snapshot_and_rejects_mutation() -> None:
    model, digest = _small_sealed_model()
    render = torch.zeros((1, 16, 2, 3, 4))
    render_digest = wan21_uni3c_tensor_digest(render)
    execution = Wan21Uni3CExecution(
        model,
        render,
        -1.25,
        PercentRange(0.25, 0.75),
        digest,
        render_digest,
    )
    snapshot = snapshot_wan21_uni3c_execution(execution)

    render.fill_(1.0)
    assert torch.count_nonzero(snapshot.render_latent).item() == 0
    with pytest.raises(Wan21Uni3CBindingError, match="identity changed"):
        Wan21Uni3CExecution(
            model,
            render,
            execution.strength,
            execution.window,
            digest,
            render_digest,
        )


def test_resource_binding_rejects_untracked_parameter_mutation() -> None:
    model, digest = _small_sealed_model()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(Wan21Uni3CBindingError, match="binding-mismatch"):
        validate_wan21_uni3c_resource(model, digest)
