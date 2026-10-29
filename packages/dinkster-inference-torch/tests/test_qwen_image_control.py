"""Reduced proofs for native Qwen Image ControlNet execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    QWEN_IMAGE_DIFFSYNTH_INPAINT,
    QWEN_IMAGE_FUN_CONTROL,
    QWEN_IMAGE_INSTANTX_INPAINT_CONTROL,
    ControlApplication,
    PayloadReference,
    PercentRange,
    QwenImageConfig,
    QwenImageControlConfig,
)
from dinkster_inference_torch import assemble as assemble_module
from dinkster_inference_torch import assemble_qwen_image_control, assemble_qwen_image_diffsynth
from dinkster_inference_torch import qwen_image_control as control_module
from dinkster_inference_torch.qwen_image import QwenImage
from dinkster_inference_torch.qwen_image_control import (
    QwenImageControlBindingError,
    QwenImageControlConditioning,
    QwenImageDiffSynthBlock,
    QwenImageDiffSynthPatch,
    QwenImageFunControlNet,
    QwenImageInstantXControlNet,
    qwen_image_control_hint_digest,
    qwen_image_control_resource_digest,
    qwen_image_diffsynth_resource_digest,
    snapshot_qwen_image_control_conditioning,
)


@dataclass(frozen=True)
class _ReducedConfig:
    transformer_blocks: int
    hidden_width: int = 12
    attention_heads: int = 2
    attention_head_dim: int = 6
    text_width: int = 6
    pooled_width: int = 4
    patchified_input_channels: int = 8
    output_latent_channels: int = 2
    patch: tuple[int, int] = (2, 2)
    rope_axes: tuple[int, int, int] = (2, 2, 2)
    default_ref_method: str = "index"
    use_additional_t_cond: bool = False


def reduced_config(blocks: int) -> QwenImageConfig:
    return cast(QwenImageConfig, _ReducedConfig(blocks))


def fill_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_(((values + index) % 17 - 8) / 64)


def test_instantx_layout_omits_final_layer_and_adds_exact_control_state() -> None:
    with torch.device("meta"):
        model = QwenImageInstantXControlNet(reduced_config(2), extra_condition_channels=4)
    state = model.state_dict()
    assert "norm_out.linear.weight" not in state
    assert "proj_out.weight" not in state
    assert state["controlnet_x_embedder.weight"].shape == (12, 12)
    assert state["controlnet_blocks.0.weight"].shape == (12, 12)
    assert state["controlnet_blocks.1.bias"].shape == (12,)
    assert state["transformer_blocks.1.attn.to_q.weight"].shape == (12, 12)


def test_instantx_produces_one_post_block_residual_per_base_block() -> None:
    config = reduced_config(2)
    model = QwenImageInstantXControlNet(config)
    fill_parameters(model)
    latent = torch.randn(1, 2, 1, 3, 5, requires_grad=True)
    context = torch.randn(1, 2, 6, requires_grad=True)
    hint = torch.randn_like(latent, requires_grad=True)
    residuals = model(latent, torch.tensor([0.25]), context, hint)
    assert len(residuals) == 2
    assert all(residual.shape == (1, 6, 12) for residual in residuals)
    assert not torch.equal(residuals[0], residuals[1])
    torch.stack(tuple(residual.sum() for residual in residuals)).sum().backward()
    assert latent.grad is not None
    assert context.grad is not None
    assert hint.grad is not None


def test_base_model_applies_control_to_only_the_declared_image_prefix() -> None:
    model = QwenImage(reduced_config(1))
    fill_parameters(model)
    latent = torch.randn(1, 2, 1, 2, 4)
    context = torch.randn(1, 2, 6)
    seen: list[torch.Tensor] = []
    hook = model.norm_out.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].detach().clone())
    )
    model(latent, torch.tensor([0.5]), context)
    baseline = seen.pop()
    residual = torch.full((1, 1, 12), 0.25)
    model(latent, torch.tensor([0.5]), context, control_residuals=(residual,))
    controlled = seen.pop()
    hook.remove()
    assert torch.equal(controlled[:, :1], baseline[:, :1] + residual)
    assert torch.equal(controlled[:, 1:], baseline[:, 1:])
    with pytest.raises(ValueError, match="block count"):
        model(latent, torch.tensor([0.5]), context, control_residuals=())


def test_base_model_applies_block_patches_in_declaration_order() -> None:
    model = QwenImage(reduced_config(1))
    fill_parameters(model)
    latent = torch.randn(1, 2, 1, 2, 4)
    context = torch.randn(1, 2, 6)
    seen: list[torch.Tensor] = []
    hook = model.norm_out.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].detach().clone())
    )
    calls: list[tuple[str, int]] = []

    def first(image: torch.Tensor, index: int) -> torch.Tensor:
        calls.append(("first", index))
        return image + 1

    def second(image: torch.Tensor, index: int) -> torch.Tensor:
        calls.append(("second", index))
        return image * 2

    model(latent, torch.tensor([0.5]), context)
    baseline = seen.pop()
    residual = torch.full((1, 1, 12), 0.25)
    model(
        latent,
        torch.tensor([0.5]),
        context,
        control_residuals=(residual,),
        block_patches=(first, second),
    )
    patched = seen.pop()
    hook.remove()
    assert calls == [("first", 0), ("second", 0)]
    expected = (baseline + 1) * 2
    expected[:, :1] += residual
    assert torch.equal(patched, expected)


def test_diffsynth_block_matches_reference_math_and_preserves_autograd() -> None:
    block = QwenImageDiffSynthBlock(12)
    fill_parameters(block)
    image = torch.randn(1, 3, 12, requires_grad=True)
    condition = torch.randn(1, 3, 12, requires_grad=True)
    expected = block.output_proj(
        block.act(block.input_proj(block.x_rms(image) + block.y_rms(condition)))
    )
    actual = block(image, condition)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert image.grad is not None
    assert condition.grad is not None


def test_diffsynth_layout_and_condition_patchification_match_reference() -> None:
    with torch.device("meta"):
        model = QwenImageDiffSynthPatch(input_features=68)
    state = model.state_dict()
    assert len(state) == 362
    assert state["img_in.weight"].shape == (3072, 68)
    assert state["controlnet_blocks.0.y_rms.weight"].shape == (3072,)
    assert state["controlnet_blocks.59.output_proj.bias"].shape == (3072,)

    class Capture(torch.nn.Module):
        in_features = 68

        def __init__(self) -> None:
            super().__init__()
            self.value: torch.Tensor | None = None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.value = value
            return value

    capture = Capture()
    model.img_in = cast(Any, capture)
    latent = torch.arange(17 * 3 * 5, dtype=torch.float32).reshape(1, 17, 1, 3, 5)
    result = model.prepare_condition(latent)
    expected = (
        torch.nn.functional.pad(latent, (0, 1, 0, 1, 0, 0), mode="circular")
        .view(1, 17, 2, 2, 3, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(1, 6, 68)
    )
    assert torch.equal(result, expected)
    assert capture.value is result


def test_fun_layout_and_sparse_injection_sites_match_maintained_checkpoint() -> None:
    config = reduced_config(60)
    with torch.device("meta"):
        model = QwenImageFunControlNet(config)
    state = model.state_dict()
    assert state["control_img_in.weight"].shape == (12, 132)
    assert state["control_blocks.0.before_proj.weight"].shape == (12, 12)
    assert "control_blocks.1.before_proj.weight" not in state
    assert state["control_blocks.4.after_proj.bias"].shape == (12,)

    base = QwenImage(config)
    model = QwenImageFunControlNet(config)
    fill_parameters(base)
    fill_parameters(model)
    residuals = model(
        base,
        torch.randn(1, 2, 1, 3, 5),
        torch.tensor([0.25]),
        torch.randn(1, 2, 6),
        torch.randn(1, 16, 3, 5),
    )
    assert len(residuals) == 60
    assert tuple(index for index, residual in enumerate(residuals) if residual is not None) == (
        0,
        12,
        24,
        36,
        48,
    )
    assert all(residual is None or residual.shape == (1, 6, 12) for residual in residuals)


@pytest.mark.parametrize(
    ("config", "expected_type", "expected_input"),
    (
        (QWEN_IMAGE_INSTANTX_INPAINT_CONTROL, QwenImageInstantXControlNet, 68),
        (QWEN_IMAGE_FUN_CONTROL, QwenImageFunControlNet, 132),
    ),
)
def test_control_assembly_builds_exact_profile_and_seals_provenance(
    monkeypatch: pytest.MonkeyPatch,
    config: QwenImageControlConfig,
    expected_type: type[torch.nn.Module],
    expected_input: int,
) -> None:
    component = SimpleNamespace(config=config)

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
            model = build(config, operations=assemble_module.INITLESS)
        assert type(model) is expected_type
        typed_model = cast(Any, model)
        input_width = (
            typed_model.control_img_in.in_features
            if type(model) is QwenImageFunControlNet
            else typed_model.controlnet_x_embedder.in_features
        )
        assert input_width == expected_input
        return model

    monkeypatch.setattr(assemble_module, "_load_component", load)
    plan = cast(
        Any,
        SimpleNamespace(
            control=component,
            asset_digest="blake3:" + "0" * 64,
        ),
    )
    assembled = assemble_qwen_image_control(plan)
    assert assembled.kind == config.kind
    assert assembled.resource_digest == assembled.control.resource_digest
    assert assembled.attention_status.role == "qwen"


def test_diffsynth_assembly_builds_exact_profile_and_seals_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = SimpleNamespace(config=QWEN_IMAGE_DIFFSYNTH_INPAINT)

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
            model = build(component.config, operations=assemble_module.INITLESS)
        assert type(model) is QwenImageDiffSynthPatch
        assert model.img_in.in_features == 68
        return model

    monkeypatch.setattr(assemble_module, "_load_component", load)
    plan = cast(
        Any,
        SimpleNamespace(patch=component, asset_digest="blake3:" + "2" * 64),
    )
    assembled = assemble_qwen_image_diffsynth(plan)
    assert assembled.kind == "diffsynth_inpaint"
    assert assembled.resource_digest == assembled.patch.resource_digest
    assert assembled.resource_digest == qwen_image_diffsynth_resource_digest(
        plan.asset_digest, "diffsynth_inpaint", torch.bfloat16
    )


def test_control_conditioning_binds_model_and_owned_hint_identity() -> None:
    model = QwenImageFunControlNet(reduced_config(60))
    fill_parameters(model)
    model_digest = qwen_image_control_resource_digest("blake3:" + "0" * 64, "fun", torch.float32)
    control_module._bind_qwen_image_control_resource(  # pyright: ignore[reportPrivateUsage]
        model, model_digest
    )
    hint = torch.zeros((1, 16, 3, 5))
    hint_digest = qwen_image_control_hint_digest(hint)
    conditioning = QwenImageControlConditioning(
        ControlApplication("qwen-fun", PayloadReference(hint_digest), 2.25, PercentRange(0.0, 1.0)),
        model,
        "fun",
        hint,
        model_digest,
        hint_digest,
    )
    snapshot = snapshot_qwen_image_control_conditioning(conditioning)
    hint.fill_(1.0)
    assert torch.count_nonzero(snapshot.hint).item() == 0
    with pytest.raises(QwenImageControlBindingError, match="content changed"):
        QwenImageControlConditioning(
            conditioning.application,
            model,
            "fun",
            hint,
            model_digest,
            hint_digest,
        )
