"""Classic SD1.5 ControlNet module, normalization, and assembly proofs."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch
import torch.nn.functional as F
from attention_spy import CallableModuleKernel
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    FLOAT64,
    SDXL,
    ComponentPlan,
    Conditioning,
    ConstantGainCurve,
    ContributionGain,
    ControlApplication,
    ControlNetAssemblyPlan,
    DiscreteSigmas,
    EffectMaskInput,
    MaskMediaPlacement,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    SD15ControlNetConfig,
    SDControlMode,
    SDXLControlLoRAAssemblyPlan,
    SDXLControlLoRAConfig,
    SDXLControlNetConfig,
    SDXLControlNetUnionConfig,
    sd15_controlnet_layout,
    sdxl_control_lora_layout,
    sdxl_controlnet_layout,
    sdxl_controlnet_union_layout,
)
from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
from dinkster_inference_torch import (
    AssembledControlNet,
    AssembledSD,
    AutoencoderKL,
    ControlResourceBindingError,
    DenoiseError,
    InitlessOperations,
    SD15ControlNet,
    SDControlConditioning,
    SDControlResiduals,
    SDDenoiser,
    SDEffectMaskSource,
    SDXLControlLoRA,
    SDXLControlNet,
    SDXLControlNetUnion,
    UNetModel,
    assemble_sd15_controlnet,
    assemble_sdxl_control_lora,
    assemble_sdxl_controlnet_union,
    compile_sd_effect_mask,
    normalize_control_hint,
    sd15_controlnet_resource_digest,
    sd_control_hint_digest,
    sd_effect_mask_source_digest,
)
from dinkster_inference_torch import assemble as assemble_module
from dinkster_inference_torch.attention import builtin_sdpa_kernel, select_attention
from dinkster_inference_torch.controlnet import (
    _bind_sd15_controlnet_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_base_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_control_lora_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_controlnet_resource,  # pyright: ignore[reportPrivateUsage]
    _bind_sdxl_controlnet_union_resource,  # pyright: ignore[reportPrivateUsage]
    _ControlAddEmbedding,  # pyright: ignore[reportPrivateUsage]
    _ControlLoRAConv2d,  # pyright: ignore[reportPrivateUsage]
    _ControlLoRALinear,  # pyright: ignore[reportPrivateUsage]
    _matches_sealed_resource_tensor,  # pyright: ignore[reportPrivateUsage]
    _resource_tensor_seal,  # pyright: ignore[reportPrivateUsage]
    _snapshot_sd_control_conditioning,  # pyright: ignore[reportPrivateUsage]
    _UnionAttention,  # pyright: ignore[reportPrivateUsage]
    _validate_sdxl_base_resource,  # pyright: ignore[reportPrivateUsage]
    _validate_sdxl_control_lora_resource,  # pyright: ignore[reportPrivateUsage]
    _validate_sdxl_controlnet_resource,  # pyright: ignore[reportPrivateUsage]
    _validate_sdxl_controlnet_union_resource,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.module_residency import ModuleStateStore, enroll_component
from dinkster_inference_torch.operations import INITLESS
from dinkster_inference_torch.sd_denoise import SDControlGain
from dinkster_inference_torch.t2i_adapter import (
    SD15T2IAdapter,
    _bind_sd15_t2i_adapter_resource,  # pyright: ignore[reportPrivateUsage]
    validate_sd15_t2i_adapter_resource,
)
from dinkster_inference_torch.unet import CrossAttention, SpatialTransformer, timestep_embedding


def test_controlnet_surface_is_public() -> None:
    assert SD15ControlNet is not None
    assert SDControlConditioning is not None
    assert SDControlResiduals is not None
    assert AssembledControlNet is not None
    assert assemble_sd15_controlnet is not None
    assert normalize_control_hint is not None


def test_sdxl_control_lora_module_matches_the_exact_artifact_layout() -> None:
    layout = sdxl_control_lora_layout(SDXLControlLoRAConfig())
    with torch.device("meta"):
        model = SDXLControlLoRA()
    expected_model_keys: set[str] = set()
    for key in layout:
        if key == "lora_controlnet":
            continue
        if key.endswith((".up", ".down")):
            expected_model_keys.add(key.rsplit(".", 1)[0] + ".weight")
        else:
            expected_model_keys.add(key)
    assert set(model.state_dict()) == expected_model_keys
    assert len(tuple(model.input_blocks)) == 9
    assert len(tuple(model.zero_convs)) == 9


def test_sdxl_controlnet_union_module_matches_the_exact_artifact_layout() -> None:
    config = SDXLControlNetUnionConfig()
    with torch.device("meta"):
        model = SDXLControlNetUnion(config, attention_kernel=builtin_sdpa_kernel())
    expected = sdxl_controlnet_union_layout(config)
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == expected
    assert len(tuple(model.input_blocks)) == 9
    assert len(tuple(model.zero_convs)) == 9


def test_sdxl_controlnet_union_routes_every_attention_call_through_injected_kernel() -> None:
    kernel = CallableModuleKernel(builtin_sdpa_kernel())
    with torch.device("meta"):
        model = SDXLControlNetUnion(attention_kernel=kernel)

    assert all(
        module._attention_kernel is kernel  # pyright: ignore[reportPrivateUsage]
        for module in model.modules()
        if isinstance(module, CrossAttention)
    )
    spatial = tuple(module for module in model.modules() if isinstance(module, SpatialTransformer))
    assert spatial
    union_attention = cast(Any, model.transformer_layes[0]).attn
    assert union_attention._attention_kernel is kernel  # pyright: ignore[reportPrivateUsage]


def test_sdxl_controlnet_union_task_attention_executes_injected_kernel() -> None:
    def nondefault_kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        return q + 2 * k + 3 * v

    kernel = CallableModuleKernel(nondefault_kernel)
    attention = _UnionAttention(4, 2, INITLESS, kernel)
    identity = torch.eye(4)
    with torch.no_grad():
        attention.in_proj.weight.copy_(torch.cat((identity, 2 * identity, -identity)))
        attention.in_proj.bias.zero_()
        attention.out_proj.weight.copy_(identity)
        attention.out_proj.bias.zero_()
    inputs = torch.tensor([[[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]]])

    output = attention(inputs)

    assert len(kernel.calls) == 1
    assert kernel.calls[0]["q_shape"] == (1, 2, 2, 2)
    torch.testing.assert_close(output, 2 * inputs, rtol=0, atol=0)


def test_sdxl_controlnet_union_assembly_forwards_central_unet_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = CallableModuleKernel(builtin_sdpa_kernel())
    statuses = {"unet": select_attention("unet").status}
    captured: list[object] = []

    def select_runtime(*_args: object):
        return {"unet": kernel}, statuses

    def capture_component(_plan: object, build: object, **_kwargs: object) -> object:
        captured.append(build)
        raise RuntimeError("captured builder")

    monkeypatch.setattr(assemble_module, "_select_attention_runtime", select_runtime)
    monkeypatch.setattr(assemble_module, "_load_component", capture_component)

    plan = SimpleNamespace(controlnet_union=object())
    with pytest.raises(RuntimeError, match="captured builder"):
        assemble_sdxl_controlnet_union(cast(Any, plan), attention_backend="unet")
    builder = cast(Any, captured[0])
    assert builder.keywords["attention_kernel"] is kernel


def test_classic_sdxl_controlnet_module_matches_the_exact_artifact_layout() -> None:
    config = SDXLControlNetConfig()
    with torch.device("meta"):
        model = SDXLControlNet(config)
    expected = sdxl_controlnet_layout(config)
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == expected
    assert len(tuple(model.input_blocks)) == 9
    assert len(tuple(model.zero_convs)) == 9


def test_classic_sdxl_controlnet_requires_sealed_provenance_and_no_mode() -> None:
    hint = torch.zeros(1, 3, 64, 64)
    hint_digest = sd_control_hint_digest(hint)
    model_digest = "d" * 64
    application = ControlApplication(
        "sdxl-canny",
        PayloadReference(hint_digest),
        1.0,
        PercentRange(0.0, 1.0),
    )
    with torch.device("meta"):
        model = SDXLControlNet()
    with pytest.raises(ControlResourceBindingError, match="provenance"):
        SDControlConditioning(application, model, hint, model_digest, hint_digest)
    _bind_sdxl_controlnet_resource(model, model_digest)
    assert SDControlConditioning(application, model, hint, model_digest, hint_digest).model is model
    with pytest.raises(ValueError, match="non-Union"):
        SDControlConditioning(
            replace(application, mode=SDControlMode("sdxl-controlnet-union", "canny")),
            model,
            hint,
            model_digest,
            hint_digest,
        )


def test_sdxl_controlnet_union_mode_compatibility_refuses_before_execution() -> None:
    hint = torch.zeros(1, 3, 64, 64)
    hint_digest = sd_control_hint_digest(hint)
    model_digest = "d" * 64
    with torch.device("meta"):
        union = SDXLControlNetUnion(attention_kernel=builtin_sdpa_kernel())
        limited = SDXLControlNetUnion(
            SDXLControlNetUnionConfig(mode_capacity=6),
            attention_kernel=builtin_sdpa_kernel(),
        )
        control_lora = SDXLControlLoRA()
    _bind_sdxl_controlnet_union_resource(union, model_digest)
    _bind_sdxl_controlnet_union_resource(limited, model_digest)
    application = ControlApplication(
        "union-canny",
        PayloadReference(hint_digest),
        1.0,
        PercentRange(0.0, 1.0),
        mode=SDControlMode("sdxl-controlnet-union", "canny"),
    )
    assert SDControlConditioning(application, union, hint, model_digest, hint_digest).model is union
    auto = SDControlConditioning(
        replace(application, mode=None), union, hint, model_digest, hint_digest
    )
    assert auto.application.mode is None
    with pytest.raises(ValueError, match="exceeds the detected artifact capacity"):
        SDControlConditioning(
            replace(application, mode=SDControlMode("sdxl-controlnet-union", "repaint")),
            limited,
            hint,
            model_digest,
            hint_digest,
        )
    _bind_sdxl_control_lora_resource(control_lora, model_digest)
    with pytest.raises(ValueError, match="non-Union"):
        SDControlConditioning(application, control_lora, hint, model_digest, hint_digest)


def test_union_task_embedding_has_a_residency_route() -> None:
    with torch.device("meta"):
        model = SDXLControlNetUnion(attention_kernel=builtin_sdpa_kernel())
    model.task_embedding = torch.nn.Parameter(torch.arange(8 * 320).reshape(8, 320).float())
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    binding = model.residency_binding()
    assert binding is not None
    assert binding.key("task_embedding") == "task_embedding"
    with binding.lease() as lease:
        torch.testing.assert_close(
            lease.get("task_embedding", dtype=torch.float32), model.task_embedding
        )
    mechanism.unload()


def test_union_control_embedding_matches_comfyui_empty_and_named_type_arithmetic() -> None:
    embedding = _ControlAddEmbedding(8, 4, INITLESS)
    with torch.no_grad():
        embedding.linear_1.weight.copy_(
            torch.arange(4 * 8 * 256, dtype=torch.float32).reshape(4, 8 * 256) / 8192
        )
        embedding.linear_1.bias.copy_(torch.linspace(-0.25, 0.25, 4))
        embedding.linear_2.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)
        embedding.linear_2.bias.copy_(torch.linspace(0.5, -0.5, 4))

    for mode_index in (None, *range(8)):
        control_type = torch.zeros(8)
        if mode_index is not None:
            control_type[mode_index] = 1.0
        source_embedding = timestep_embedding(control_type, 256).reshape(1, -1)
        expected = embedding.linear_2(F.silu(embedding.linear_1(source_embedding)))
        actual = embedding(mode_index, torch.float32, torch.device("cpu"))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_union_auto_uses_ordinary_hint_encoder_without_task_fusion() -> None:
    with torch.device("meta"):
        model = SDXLControlNetUnion(attention_kernel=builtin_sdpa_kernel())
    captured: dict[str, torch.Tensor] = {}

    class TimeEmbedding(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            assert value.shape == (1, 320)
            return torch.zeros(1, 1280)

    class ControlEmbedding(torch.nn.Module):
        def forward(
            self, mode_index: int | None, dtype: torch.dtype, device: torch.device
        ) -> torch.Tensor:
            assert mode_index is None
            return torch.ones(1, 1280, dtype=dtype, device=device)

    class HintEncoder(torch.nn.Module):
        def forward(
            self, hint: torch.Tensor, embedding: torch.Tensor, context: torch.Tensor
        ) -> torch.Tensor:
            assert hint.shape == (1, 3, 64, 64)
            assert torch.count_nonzero(embedding == 1) == embedding.numel()
            assert context.shape == (1, 1, 2048)
            return torch.full((1, 320, 8, 8), 2.0)

    class LabelEmbedding(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            assert value.shape == (1, 2816)
            return torch.full((1, 1280), 3.0)

    class RefuseTaskFusion(torch.nn.Module):
        def forward(self, _value: torch.Tensor) -> torch.Tensor:
            raise AssertionError("automatic Union mode must skip task fusion")

    def run_control(
        _self: SDXLControlNetUnion,
        _x: torch.Tensor,
        embedding: torch.Tensor,
        guided_hint: torch.Tensor,
        _context: torch.Tensor,
    ) -> SDControlResiduals:
        captured["embedding"] = embedding
        captured["guided_hint"] = guided_hint
        return cast(SDControlResiduals, object())

    model.__dict__["time_embed"] = TimeEmbedding()
    model.__dict__["control_add_embedding"] = ControlEmbedding()
    model.__dict__["input_hint_block"] = HintEncoder()
    model.__dict__["label_emb"] = LabelEmbedding()
    model.__dict__["transformer_layes"] = RefuseTaskFusion()
    model.__dict__["spatial_ch_projs"] = RefuseTaskFusion()
    model.__dict__["_run_control"] = MethodType(run_control, model)
    result = model(
        torch.zeros(1, 4, 8, 8),
        torch.zeros(1, 3, 64, 64),
        torch.zeros(1),
        torch.zeros(1, 1, 2048),
        torch.zeros(1, 2816),
        None,
    )
    assert type(result) is object
    assert torch.count_nonzero(captured["embedding"] == 4) == 1280
    assert torch.count_nonzero(captured["guided_hint"] == 2) == 320 * 8 * 8


@pytest.mark.parametrize(
    ("mode", "expected_index"),
    (
        (None, None),
        (SDControlMode("sdxl-controlnet-union", "canny"), 3),
    ),
)
def test_denoiser_passes_union_auto_and_named_mode_indices(
    mode: SDControlMode | None, expected_index: int | None
) -> None:
    indices: list[int | None] = []
    down_channels = (320, 320, 320, 320, 640, 640, 640, 1280, 1280)
    down_scales = (1, 1, 1, 2, 2, 2, 4, 4, 4)

    class CapturingUNet(torch.nn.Module):
        config = SDXLControlLoRAConfig().base

        def forward(
            self,
            input: torch.Tensor,
            _timesteps: torch.Tensor,
            *,
            context: torch.Tensor,
            y: torch.Tensor | None = None,
            control: SDControlResiduals | None = None,
            attention_guidance: object | None = None,
        ) -> torch.Tensor:
            assert context.shape == (1, 1, 2048)
            assert y is not None and y.shape == (1, 2816)
            assert control is not None
            return torch.zeros_like(input)

    def union_forward(
        _self: SDXLControlNetUnion,
        x: torch.Tensor,
        _hint: torch.Tensor,
        _timesteps: torch.Tensor,
        _context: torch.Tensor,
        _y: torch.Tensor,
        mode_index: int | None,
    ) -> SDControlResiduals:
        indices.append(mode_index)
        down = tuple(
            torch.zeros(x.shape[0], channels, x.shape[2] // scale, x.shape[3] // scale)
            for channels, scale in zip(down_channels, down_scales, strict=True)
        )
        return SDControlResiduals(
            down,
            torch.zeros(x.shape[0], 1280, x.shape[2] // 4, x.shape[3] // 4),
            down_channels,
            down_scales,
        )

    with torch.device("meta"):
        union = SDXLControlNetUnion(attention_kernel=builtin_sdpa_kernel())
    union.__dict__["forward"] = MethodType(union_forward, union)
    denoiser = SDDenoiser(
        cast(UNetModel, CapturingUNet()),
        DiscreteSigmas.linear_beta(),
        Conditioning(torch.zeros(1, 1, 2048)),
        adm_cond=torch.zeros(1, 2816),
        control_model=union,
        control_hint=torch.zeros(1, 3, 64, 64),
        control_mode=mode,
        compute_dtype=torch.float32,
    )
    denoiser.set_control_gain(1.0)

    denoiser(torch.zeros(1, 4, 8, 8), 1.0)

    assert indices == [expected_index]


ResourceBinder = Callable[[torch.nn.Module, str], None]
ResourceValidator = Callable[[torch.nn.Module, str], None]


def test_resource_tensor_seal_accepts_real_residency_transitions() -> None:
    model = INITLESS.linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    original = model.weight
    seal = _resource_tensor_seal("parameter:weight", original)
    mechanism = enroll_component(
        model,
        load_device="cpu",
        offload_device="cpu",
        patch_set=PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones_like(original))),)}),
    )

    mechanism.partially_load(None)
    loaded = model.weight
    assert loaded is not original
    assert _matches_sealed_resource_tensor(model, "parameter:weight", loaded, seal)
    with torch.no_grad():
        loaded.add_(1)
    assert not _matches_sealed_resource_tensor(model, "parameter:weight", loaded, seal)

    mechanism.unload()
    assert model.weight is original
    assert _matches_sealed_resource_tensor(model, "parameter:weight", model.weight, seal)


def _sealed_control_resource(
    kind: str,
) -> tuple[torch.nn.Module, str, ResourceValidator]:
    with torch.device("meta"):
        if kind == "base":
            model = UNetModel(SDXLControlLoRAConfig().base)
            digest = "blake3:" + "a" * 64
            binder = cast(ResourceBinder, _bind_sdxl_base_resource)
            validator = cast(ResourceValidator, _validate_sdxl_base_resource)
        elif kind == "control-lora":
            model = SDXLControlLoRA()
            digest = "a" * 64
            binder = cast(ResourceBinder, _bind_sdxl_control_lora_resource)
            validator = cast(ResourceValidator, _validate_sdxl_control_lora_resource)
        elif kind == "controlnet":
            model = SDXLControlNet()
            digest = "a" * 64
            binder = cast(ResourceBinder, _bind_sdxl_controlnet_resource)
            validator = cast(ResourceValidator, _validate_sdxl_controlnet_resource)
        elif kind == "union":
            model = SDXLControlNetUnion(attention_kernel=builtin_sdpa_kernel())
            digest = "a" * 64
            binder = cast(ResourceBinder, _bind_sdxl_controlnet_union_resource)
            validator = cast(ResourceValidator, _validate_sdxl_controlnet_union_resource)
        else:
            assert kind == "t2i-adapter"
            model = SD15T2IAdapter()
            digest = "a" * 64
            binder = cast(ResourceBinder, _bind_sd15_t2i_adapter_resource)
            validator = cast(ResourceValidator, validate_sd15_t2i_adapter_resource)
    binder(model, digest)
    return model, digest, validator


@pytest.mark.parametrize("kind", ("base", "control-lora", "controlnet", "union", "t2i-adapter"))
def test_control_resource_seals_accept_actual_residency_assignments(kind: str) -> None:
    model, digest, validate = _sealed_control_resource(kind)
    name, original = next(model.named_parameters())
    store = ModuleStateStore(model)
    model.__dict__["_dinkster_residency_state_store"] = store
    store._set_authorized(name, original.detach().clone())  # pyright: ignore[reportPrivateUsage]

    validate(model, digest)

    current = dict(model.named_parameters())[name]
    current.data = current.detach().clone()
    with pytest.raises(ControlResourceBindingError):
        validate(model, digest)


@pytest.mark.parametrize("kind", ("base", "control-lora", "controlnet", "union", "t2i-adapter"))
def test_control_resource_seals_reject_version_mutation(kind: str) -> None:
    model, digest, validate = _sealed_control_resource(kind)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(ControlResourceBindingError):
        validate(model, digest)


def test_sdxl_control_lora_refuses_unbound_base_asset() -> None:
    with torch.device("meta"):
        base = UNetModel(SDXLControlLoRAConfig().base)
    assemble_module = importlib.import_module("dinkster_inference_torch.assemble")
    validate = assemble_module._validate_sdxl_base_resource
    with pytest.raises(ControlResourceBindingError, match="base assembly provenance"):
        validate(base, "blake3:" + "a" * 64)


@pytest.mark.parametrize("compute_dtype", (torch.float16, torch.float32))
def test_sdxl_control_lora_assembly_resolves_state_dtype_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    compute_dtype: torch.dtype,
) -> None:
    layout = sdxl_control_lora_layout()
    component = ComponentPlan(
        "control_lora",
        Path("control-lora.safetensors"),
        SDXLControlLoRAConfig(),
        {key: key for key in layout},
        {key: FLOAT32 if key == "lora_controlnet" else FLOAT16 for key in layout},
        {},
    )
    plan = SDXLControlLoRAAssemblyPlan(
        component,
        "blake3:" + "a" * 64,
        "blake3:" + "b" * 64,
        tuple(sorted(layout)),
    )
    with torch.device("meta"):
        base_model = UNetModel(SDXLControlLoRAConfig().base).to(compute_dtype)
        tensors = {
            key: torch.empty(
                shape,
                dtype=torch.float32 if key == "lora_controlnet" else torch.float16,
            )
            for key, shape in layout.items()
        }
    _bind_sdxl_base_resource(base_model, plan.base_asset_digest)
    base = AssembledSD(
        SDXL,
        base_model,
        None,
        None,
        cast(AutoencoderKL, object()),
    )
    assemble_module = importlib.import_module("dinkster_inference_torch.assemble")

    def load_tensors(_path: Path, _keys: object) -> dict[str, torch.Tensor]:
        return tensors

    monkeypatch.setattr(assemble_module, "load_tensors", load_tensors)

    assembled = assemble_sdxl_control_lora(plan, base, control_lora_dtype=compute_dtype)

    model = assembled.control_lora
    expected_keys: set[str] = set()
    for key in layout:
        if key == "lora_controlnet":
            continue
        expected_keys.add(key)
        if key.endswith((".up", ".down")):
            expected_keys.add(key.rsplit(".", 1)[0] + ".weight")
    assert set(model.state_dict()) == expected_keys
    assert {parameter.dtype for parameter in model.parameters()} == {compute_dtype}
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert assembled.compute_dtype is compute_dtype
    assert assembled.resource_digest == model.resource_digest
    base_state = base_model.state_dict()
    for key, tensor in model.state_dict().items():
        source = tensors[key] if key in tensors else base_state[key]
        if source.dtype == compute_dtype:
            assert tensor.untyped_storage() is source.untyped_storage(), key

    _validate_sdxl_control_lora_resource(model, assembled.resource_digest)
    with torch.no_grad():
        base_state["time_embed.0.weight"].add_(1)
    with pytest.raises(ControlResourceBindingError):
        _validate_sdxl_control_lora_resource(model, assembled.resource_digest)


@pytest.mark.parametrize("placement", ("unmanaged", "offloaded", "loaded"))
def test_sdxl_control_lora_factors_match_comfyui_weight_materialization(placement: str) -> None:
    linear = _ControlLoRALinear(3, 2, bias=False)
    linear.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    linear.up = torch.nn.Parameter(torch.tensor([[2.0], [-1.0]]))
    linear.down = torch.nn.Parameter(torch.tensor([[0.5, 1.5, -2.0]]))
    if placement != "unmanaged":
        mechanism = enroll_component(linear, load_device="cpu", offload_device="cpu")
        if placement == "loaded":
            mechanism.partially_load(None)
    value = torch.tensor([[1.0, 2.0, -1.0]])
    expected_weight = linear.weight + linear.up @ linear.down
    torch.testing.assert_close(linear(value), F.linear(value, expected_weight), rtol=0, atol=0)

    conv = _ControlLoRAConv2d(1, 1, 3, padding=1, bias=False)
    conv.weight = torch.nn.Parameter(torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3))
    conv.up = torch.nn.Parameter(torch.tensor([[[[2.0]]]]))
    conv.down = torch.nn.Parameter(torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3))
    if placement != "unmanaged":
        mechanism = enroll_component(conv, load_device="cpu", offload_device="cpu")
        if placement == "loaded":
            mechanism.partially_load(None)
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    expected_weight = conv.weight + torch.mm(
        conv.up.flatten(start_dim=1), conv.down.flatten(start_dim=1)
    ).reshape_as(conv.weight)
    torch.testing.assert_close(
        conv(image), F.conv2d(image, expected_weight, padding=1), rtol=0, atol=0
    )


def test_control_lora_routes_all_parameter_owners() -> None:
    with torch.device("meta"):
        model = SDXLControlLoRA()
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.unload()


def test_resolved_control_conditioning_is_frozen_and_exact_typed() -> None:
    model_digest = "a" * 64
    with torch.device("meta"):
        model = SD15ControlNet()
    _bind_sd15_controlnet_resource(model, model_digest)
    hint = torch.zeros(1, 3, 16, 16)
    hint_digest = sd_control_hint_digest(hint)
    application = ControlApplication(
        "canny",
        PayloadReference(hint_digest),
        1.0,
        PercentRange(0.0, 1.0),
    )
    control = SDControlConditioning(
        application,
        model,
        hint,
        model_digest,
        hint_digest,
        ContributionGain(ConstantGainCurve(1.0), 1.0),
    )
    with pytest.raises(FrozenInstanceError):
        control.gain = None  # type: ignore[misc]
    with pytest.raises(TypeError, match="exact ControlApplication"):
        SDControlConditioning(
            cast(ControlApplication, object()),
            model,
            hint,
            model_digest,
            hint_digest,
        )
    with pytest.raises(ValueError, match=r"\[batch x 3 x H x W\]"):
        bad_hint = torch.zeros(1, 4, 16, 16)
        SDControlConditioning(
            application,
            model,
            bad_hint,
            model_digest,
            sd_control_hint_digest(bad_hint),
        )


def test_control_conditioning_refuses_unbound_resources() -> None:
    hint = torch.zeros(1, 3, 16, 16)
    hint_digest = sd_control_hint_digest(hint)
    model_digest = "a" * 64
    application = ControlApplication(
        "canny", PayloadReference(hint_digest), 1.0, PercentRange(0.0, 1.0)
    )
    with torch.device("meta"):
        unbound_model = SD15ControlNet()
    with pytest.raises(ControlResourceBindingError, match="assemble_sd15_controlnet"):
        SDControlConditioning(application, unbound_model, hint, model_digest, hint_digest)

    with torch.device("meta"):
        model = SD15ControlNet()
    _bind_sd15_controlnet_resource(model, model_digest)
    with pytest.raises(ControlResourceBindingError, match="declared model digest"):
        SDControlConditioning(application, model, hint, "b" * 64, hint_digest)
    with pytest.raises(ControlResourceBindingError, match="does not resolve"):
        SDControlConditioning(
            replace(application, hint=PayloadReference("b" * 64)),
            model,
            hint,
            model_digest,
            hint_digest,
        )
    with pytest.raises(ControlResourceBindingError, match="materialized hint"):
        SDControlConditioning(
            application,
            model,
            torch.ones_like(hint),
            model_digest,
            hint_digest,
        )


def test_control_admission_snapshots_hint_and_refuses_changed_model_state() -> None:
    model_digest = "a" * 64
    with torch.device("meta"):
        model = SD15ControlNet()
    _bind_sd15_controlnet_resource(model, model_digest)
    hint = torch.zeros(1, 3, 16, 16)
    hint_digest = sd_control_hint_digest(hint)
    control = SDControlConditioning(
        ControlApplication("canny", PayloadReference(hint_digest), 1.0, PercentRange(0.0, 1.0)),
        model,
        hint,
        model_digest,
        hint_digest,
    )
    snapshot = _snapshot_sd_control_conditioning(control)
    assert snapshot.hint is not hint
    hint.fill_(1.0)
    assert torch.count_nonzero(snapshot.hint) == 0
    with pytest.raises(ControlResourceBindingError, match="materialized hint tensor"):
        _snapshot_sd_control_conditioning(control)

    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(ControlResourceBindingError, match="state changed after assembly"):
        _snapshot_sd_control_conditioning(snapshot)


def test_control_resource_accepts_only_residency_owned_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlnet_module = importlib.import_module("dinkster_inference_torch.controlnet")
    model_digest = "a" * 64
    with torch.device("meta"):
        model = SD15ControlNet()
    _bind_sd15_controlnet_resource(model, model_digest)
    name, original = next(iter(model.named_parameters()))
    module_name, _, parameter_name = name.rpartition(".")
    owner = model.get_submodule(module_name)
    replacement = torch.nn.Parameter(torch.empty_like(original), requires_grad=False)
    with torch.no_grad():
        replacement.add_(1.0)
    setattr(owner, parameter_name, replacement)

    with pytest.raises(ControlResourceBindingError, match="state changed after assembly"):
        controlnet_module._validate_sd15_controlnet_resource(model, model_digest)

    authorized_version = int(replacement._version)

    def authorize(_model: torch.nn.Module, key: str, tensor: torch.Tensor) -> int | None:
        return authorized_version if key == name and tensor is replacement else None

    def authorize_generation(_model: torch.nn.Module, key: str, tensor: torch.Tensor) -> int | None:
        return 1 if key == name and tensor is replacement else None

    residency_module = importlib.import_module("dinkster_inference_torch.module_residency")
    monkeypatch.setattr(
        residency_module,
        "_residency_assignment_generation",
        authorize_generation,
    )
    monkeypatch.setattr(residency_module, "_residency_assignment_version", authorize)
    controlnet_module._validate_sd15_controlnet_resource(model, model_digest)

    with torch.no_grad():
        replacement.add_(1.0)
    with pytest.raises(ControlResourceBindingError, match="state changed after assembly"):
        controlnet_module._validate_sd15_controlnet_resource(model, model_digest)

    setattr(owner, parameter_name, original)
    controlnet_module._validate_sd15_controlnet_resource(model, model_digest)
    original.data = original.detach().clone()
    with pytest.raises(ControlResourceBindingError, match="state changed after assembly"):
        controlnet_module._validate_sd15_controlnet_resource(model, model_digest)


def _effect_mask_source(mask: torch.Tensor, site: str) -> SDEffectMaskSource:
    digest = sd_effect_mask_source_digest(mask)
    declaration = EffectMaskInput(
        PayloadDescriptor(PayloadReference(digest), tuple(mask.shape), "float32", "mask"),
        digest,
        ("batch", "height", "width"),
        MaskMediaPlacement.FULL_DOMAIN,
        site,
        "sd15.control-effect-mask.bilinear-align-corners-false.v1",
    )
    return SDEffectMaskSource(declaration, mask)


def test_effect_mask_source_is_snapshotted_and_compiles_to_declared_site() -> None:
    mask = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    source = _effect_mask_source(mask, "sd15.unet.input_skip.00.v1")
    field = compile_sd_effect_mask(source, latent_height=8, latent_width=8)

    expected = F.interpolate(mask.unsqueeze(1), (8, 8), mode="bilinear", align_corners=False)
    torch.testing.assert_close(field.mask, expected, rtol=0, atol=0)
    assert field.compiled.target_segment == "sd15.unet.input_skip.00.v1"
    assert field.compiled.source_digest == source.declaration.source_digest
    assert field.compiled.input_digest == source.declaration.digest
    assert field.compiled.table.rows == 64
    assert field.compiled.table.row_shape == (1,)

    model_digest = "a" * 64
    with torch.device("meta"):
        model = SD15ControlNet()
    _bind_sd15_controlnet_resource(model, model_digest)
    hint = torch.zeros(1, 3, 16, 16)
    hint_digest = sd_control_hint_digest(hint)
    control = SDControlConditioning(
        ControlApplication("canny", PayloadReference(hint_digest), 1.0, PercentRange(0.0, 1.0)),
        model,
        hint,
        model_digest,
        hint_digest,
        effect_masks=(source,),
    )
    snapshot = _snapshot_sd_control_conditioning(control)
    mask.fill_(1.0)
    assert torch.equal(
        snapshot.effect_masks[0].mask,
        torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
    )
    with pytest.raises(ControlResourceBindingError, match="source digest"):
        _snapshot_sd_control_conditioning(control)


def test_control_admission_snapshots_and_binds_the_complete_chain() -> None:
    with torch.device("meta"):
        first_model = SD15ControlNet()
        second_model = SD15ControlNet()
    _bind_sd15_controlnet_resource(first_model, "a" * 64)
    _bind_sd15_controlnet_resource(second_model, "b" * 64)
    first_hint = torch.zeros(1, 3, 16, 16)
    second_hint = torch.ones(1, 3, 16, 16)
    first_hint_digest = sd_control_hint_digest(first_hint)
    second_hint_digest = sd_control_hint_digest(second_hint)
    first_application = ControlApplication(
        "depth", PayloadReference(first_hint_digest), 1.0, PercentRange(0.0, 1.0)
    )
    first = SDControlConditioning(
        first_application, first_model, first_hint, "a" * 64, first_hint_digest
    )
    second_application = ControlApplication(
        "canny",
        PayloadReference(second_hint_digest),
        0.5,
        PercentRange(0.25, 0.75),
        first_application,
    )
    second = SDControlConditioning(
        second_application,
        second_model,
        second_hint,
        "b" * 64,
        second_hint_digest,
        previous=first,
    )
    snapshot = _snapshot_sd_control_conditioning(second)
    assert snapshot.hint is not second_hint
    assert snapshot.previous is not None
    assert snapshot.previous.hint is not first_hint
    first_hint.fill_(2.0)
    second_hint.fill_(3.0)
    assert torch.count_nonzero(snapshot.previous.hint) == 0
    assert torch.count_nonzero(snapshot.hint == 1.0) == snapshot.hint.numel()

    with pytest.raises(ControlResourceBindingError, match="application chain"):
        SDControlConditioning(
            second_application,
            second_model,
            snapshot.hint,
            "b" * 64,
            second_hint_digest,
        )


def test_control_resource_digests_bind_content_and_load_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zeros = torch.zeros(1, 3, 16, 16)

    def refuse_tolist(_self: torch.Tensor) -> list[object]:
        raise AssertionError("digest must use a direct buffer")

    monkeypatch.setattr(torch.Tensor, "tolist", refuse_tolist)

    def refuse_numpy(_self: torch.Tensor) -> object:
        raise AssertionError("digest must not require NumPy")

    monkeypatch.setattr(torch.Tensor, "numpy", refuse_numpy)
    assert sd_control_hint_digest(zeros) == (
        "ada2d368703899b42786d4baec898f1b7e4535d7c236390f991f09b8748910da"
    )
    assert sd_control_hint_digest(zeros) == sd_control_hint_digest(zeros.clone())
    assert sd_control_hint_digest(zeros) != sd_control_hint_digest(torch.ones_like(zeros))
    assert sd_control_hint_digest(zeros) != sd_control_hint_digest(zeros.to(torch.bfloat16))
    assert sd_control_hint_digest(zeros) != sd_control_hint_digest(zeros.to(torch.float64))
    assert sd_control_hint_digest(zeros) != sd_control_hint_digest(torch.zeros(1, 3, 8, 32))

    backing = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    lazy_negative = torch.as_strided(
        torch.complex(torch.zeros_like(backing), backing).conj().imag,
        backing.shape,
        backing.stride(),
    )
    opposite = -lazy_negative
    assert lazy_negative.is_neg() and lazy_negative.is_contiguous()
    assert not torch.equal(lazy_negative, opposite)
    assert sd_control_hint_digest(lazy_negative) == sd_control_hint_digest(
        lazy_negative.resolve_neg()
    )
    assert sd_control_hint_digest(lazy_negative) != sd_control_hint_digest(opposite)

    source_digest = "blake3:" + "a" * 64
    base = sd15_controlnet_resource_digest(source_digest, "canonical", torch.float16)
    assert base != sd15_controlnet_resource_digest("blake3:" + "b" * 64, "canonical", torch.float16)
    assert base != sd15_controlnet_resource_digest(source_digest, "diffusers", torch.float16)
    assert base != sd15_controlnet_resource_digest(source_digest, "canonical", torch.float32)


def test_full_module_state_names_and_shapes_match_torch_free_layout() -> None:
    layout = sd15_controlnet_layout(SD15ControlNetConfig())
    with torch.device("meta"):
        model = SD15ControlNet()
    actual = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    assert actual == dict(layout.keys)
    assert len(model.input_blocks) == 12
    assert len(model.zero_convs) == 12
    assert [
        cast(torch.nn.Conv2d, cast(torch.nn.Sequential, conv)[0]).out_channels
        for conv in model.zero_convs
    ] == list(layout.zero_conv_channels)
    assert model.middle_block_out[0].out_channels == 1280


def test_normalize_hint_center_crops_resizes_repeats_and_caches() -> None:
    first = torch.arange(24, dtype=torch.float64).reshape(1, 3, 2, 4)
    second = first + 100
    hint = torch.cat((first, second))
    cache: dict[tuple[object, ...], torch.Tensor] = {}
    normalized = normalize_control_hint(
        hint,
        latent_height=1,
        latent_width=1,
        batch=3,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        cache=cache,  # type: ignore[arg-type]
    )
    cropped = hint[..., 1:3]
    resized = F.interpolate(cropped, size=(8, 8), mode="nearest-exact")
    expected = torch.cat((resized, resized[:1])).to(torch.float32)
    assert torch.equal(normalized, expected)
    assert normalized.dtype is torch.float32
    assert len(cache) == 1
    assert (
        normalize_control_hint(
            hint,
            latent_height=1,
            latent_width=1,
            batch=3,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            cache=cache,  # type: ignore[arg-type]
        )
        is normalized
    )


def test_adapter_rgb_hint_normalization_preserves_float32_mean_and_cache_identity() -> None:
    hint = torch.tensor([2048.0, 1.0, -2048.0], dtype=torch.float16).reshape(1, 3, 1, 1)
    hint = hint.expand(1, 3, 2, 4).contiguous()
    cache: dict[tuple[object, ...], torch.Tensor] = {}
    kwargs = {
        "latent_height": 1,
        "latent_width": 1,
        "batch": 1,
        "device": torch.device("cpu"),
        "compute_dtype": torch.float32,
        "cache": cache,
    }
    rgb = normalize_control_hint(hint, **kwargs)  # type: ignore[arg-type]
    gray = normalize_control_hint(hint, expected_channels=1, **kwargs)  # type: ignore[arg-type]
    resized = F.interpolate(hint[..., 1:3], size=(8, 8), mode="nearest-exact")
    expected = resized.float().mean(dim=1, keepdim=True)
    assert torch.equal(gray, expected)
    assert not torch.equal(
        gray, hint.mean(dim=1, keepdim=True).float()[..., :1, :1].expand_as(gray)
    )
    assert rgb.shape == (1, 3, 8, 8)
    assert gray.shape == (1, 1, 8, 8)
    assert len(cache) == 2
    assert normalize_control_hint(hint, expected_channels=1, **kwargs) is gray  # type: ignore[arg-type]

    model, digest, _validate = _sealed_control_resource("t2i-adapter")
    hint_digest = sd_control_hint_digest(hint)
    application = ControlApplication(
        "adapter", PayloadReference(hint_digest), 1.0, PercentRange(0.0, 1.0)
    )
    control = SDControlConditioning(
        application, cast(SD15T2IAdapter, model), hint, digest, hint_digest
    )
    assert control.hint is hint


def test_normalize_hint_refuses_ambiguous_batch_and_bad_geometry() -> None:
    with pytest.raises(ValueError, match="cannot unambiguously map"):
        normalize_control_hint(
            torch.empty(3, 3, 8, 8),
            latent_height=1,
            latent_width=1,
            batch=2,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
        )
    with pytest.raises(TypeError, match="compute dtype must be floating"):
        normalize_control_hint(
            torch.empty(1, 3, 8, 8),
            latent_height=1,
            latent_width=1,
            batch=1,
            device=torch.device("cpu"),
            compute_dtype=torch.int32,
        )
    with pytest.raises(ValueError, match=r"\[batch x 3 x H x W\]"):
        normalize_control_hint(
            torch.empty(1, 4, 8, 8),
            latent_height=1,
            latent_width=1,
            batch=1,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
        )


def _residuals(
    dtype: torch.dtype = torch.float32, *, value: float | None = None, batch: int = 1
) -> SDControlResiduals:
    channels = (320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280)
    scales = (1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8)

    def make(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        if value is None:
            return torch.empty(*shape, dtype=dtype)
        return torch.full(shape, value, dtype=dtype)

    down = tuple(
        make(batch, channel, 8 // scale, 8 // scale, dtype=dtype)
        for channel, scale in zip(channels, scales, strict=True)
    )
    return SDControlResiduals(down, make(batch, 1280, 1, 1, dtype=dtype))


def test_denoiser_executes_and_adds_control_chain_oldest_to_newest() -> None:
    calls: list[str] = []

    class FixedControl(torch.nn.Module):
        def __init__(self, name: str, value: float) -> None:
            super().__init__()
            self.name = name
            self.value = value

        def forward(self, *_args: object) -> SDControlResiduals:
            calls.append(self.name)
            return _residuals(torch.float16, value=self.value)

    class CapturingUNet(torch.nn.Module):
        config = type("Config", (), {"in_channels": 4, "adm_in_channels": None})()

        def __init__(self) -> None:
            super().__init__()
            self.control: SDControlResiduals | None = None

        def forward(
            self,
            input: torch.Tensor,
            _timesteps: torch.Tensor,
            *,
            context: torch.Tensor,
            y: torch.Tensor | None = None,
            control: SDControlResiduals | None = None,
            attention_guidance: object | None = None,
        ) -> torch.Tensor:
            assert context.shape == (1, 1, 1)
            assert y is None
            self.control = control
            return torch.zeros_like(input)

    model = CapturingUNet()
    controls = tuple(
        cast(SD15ControlNet, FixedControl(name, value))
        for name, value in (("oldest", 2048.0), ("middle", -2048.0), ("newest", 1.0))
    )
    denoiser = SDDenoiser(
        cast(UNetModel, model),
        DiscreteSigmas.linear_beta(),
        Conditioning(torch.zeros(1, 1, 1)),
        control_model=controls,
        control_hint=tuple(torch.zeros(1, 3, 64, 64) for _ in controls),
        compute_dtype=torch.float16,
    )
    denoiser.set_control_gains((1.0, 1.0, 1.0))
    denoiser(torch.zeros(1, 4, 8, 8, dtype=torch.float16), 1.0)
    assert calls == ["oldest", "middle", "newest"]
    assert model.control is not None
    assert torch.count_nonzero(model.control.down[0] == 1.0) == model.control.down[0].numel()
    assert torch.count_nonzero(model.control.middle == 1.0) == model.control.middle.numel()


def test_denoiser_applies_resolved_site_and_lane_gains_before_injection() -> None:
    class FixedControl(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, input: torch.Tensor, *_args: object) -> SDControlResiduals:
            self.calls += 1
            return _residuals(torch.float32, value=1.0, batch=input.shape[0])

    class CapturingUNet(torch.nn.Module):
        config = type("Config", (), {"in_channels": 4, "adm_in_channels": None})()

        def __init__(self) -> None:
            super().__init__()
            self.control: SDControlResiduals | None = None

        def forward(
            self,
            input: torch.Tensor,
            _timesteps: torch.Tensor,
            *,
            context: torch.Tensor,
            y: torch.Tensor | None = None,
            control: SDControlResiduals | None = None,
            attention_guidance: object | None = None,
        ) -> torch.Tensor:
            self.control = control
            return torch.zeros_like(input)

    model = CapturingUNet()
    control_model = FixedControl()
    denoiser = SDDenoiser(
        cast(UNetModel, model),
        DiscreteSigmas.linear_beta(),
        control_model=cast(SD15ControlNet, control_model),
        control_hint=torch.zeros(1, 3, 64, 64),
        compute_dtype=torch.float32,
    )
    positive = denoiser.prepare_conditioning(Conditioning(torch.ones(1, 1, 1)), lane_id="positive")
    denoiser.set_control_gain_rows((SDControlGain(("positive",), ((0.0,),) * 13),))
    denoiser.evaluate_conditioning_batch(
        torch.zeros(1, 4, 8, 8),
        1.0,
        (positive,),
    )
    assert control_model.calls == 0

    site_lane_gains = ((2.0, 3.0),) + ((1.0, 1.0),) * 11 + ((5.0, 7.0),)
    denoiser.set_control_gain_rows((SDControlGain(("positive", "negative"), site_lane_gains),))
    negative = denoiser.prepare_conditioning(Conditioning(torch.zeros(1, 1, 1)), lane_id="negative")

    denoiser.evaluate_conditioning_batch(
        torch.zeros(1, 4, 8, 8),
        1.0,
        (negative, positive),
    )

    assert control_model.calls == 1
    assert model.control is not None
    assert torch.all(model.control.down[0][0] == 3.0)
    assert torch.all(model.control.down[0][1] == 2.0)
    assert torch.all(model.control.down[1] == 1.0)
    assert torch.all(model.control.middle[0] == 7.0)
    assert torch.all(model.control.middle[1] == 5.0)


def test_denoiser_multiplies_ordered_effect_masks_with_site_lane_gains() -> None:
    class FixedControl(torch.nn.Module):
        def forward(self, input: torch.Tensor, *_args: object) -> SDControlResiduals:
            return _residuals(torch.float32, value=1.0, batch=input.shape[0])

    class CapturingUNet(torch.nn.Module):
        config = type("Config", (), {"in_channels": 4, "adm_in_channels": None})()

        def __init__(self) -> None:
            super().__init__()
            self.control: SDControlResiduals | None = None

        def forward(
            self,
            input: torch.Tensor,
            _timesteps: torch.Tensor,
            *,
            context: torch.Tensor,
            y: torch.Tensor | None = None,
            control: SDControlResiduals | None = None,
            attention_guidance: object | None = None,
        ) -> torch.Tensor:
            self.control = control
            return torch.zeros_like(input)

    source = _effect_mask_source(
        torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        "sd15.unet.input_skip.00.v1",
    )
    second_source = _effect_mask_source(
        torch.tensor([[[0.25, 0.5], [0.75, 1.0]]]),
        "sd15.unet.input_skip.00.v1",
    )
    one_source = _effect_mask_source(
        torch.ones(1, 2, 2),
        "sd15.unet.input_skip.00.v1",
    )
    field = compile_sd_effect_mask(source, latent_height=8, latent_width=8)
    second_field = compile_sd_effect_mask(second_source, latent_height=8, latent_width=8)
    one_field = compile_sd_effect_mask(one_source, latent_height=8, latent_width=8)
    expected = field.mask.clone() * second_field.mask.clone()
    model = CapturingUNet()
    denoiser = SDDenoiser(
        cast(UNetModel, model),
        DiscreteSigmas.linear_beta(),
        control_model=cast(SD15ControlNet, FixedControl()),
        control_hint=torch.zeros(1, 3, 64, 64),
        control_effect_masks=((field, one_field, second_field),),
        compute_dtype=torch.float32,
    )
    field.mask.fill_(1.0)
    second_field.mask.fill_(1.0)
    one_field.mask.fill_(0.0)
    with pytest.raises(DenoiseError, match="structured control gain"):
        denoiser.set_control_gain(1.0)
    with pytest.raises(DenoiseError, match="residual-site layout"):
        denoiser.set_control_gain_rows((SDControlGain(("positive",), ((1.0,),) * 10),))
    with pytest.raises(DenoiseError, match="unresolved effect-mask"):
        denoiser.set_control_gain_rows((SDControlGain(("positive",), ((1.0,),) * 13, ("f" * 64,)),))
    positive = denoiser.prepare_conditioning(Conditioning(torch.ones(1, 1, 1)), lane_id="positive")
    negative = denoiser.prepare_conditioning(Conditioning(torch.zeros(1, 1, 1)), lane_id="negative")
    denoiser.set_control_gain_rows(
        (
            SDControlGain(
                ("positive", "negative"),
                ((2.0, 3.0),) + ((1.0, 1.0),) * 12,
                (
                    source.declaration.digest,
                    one_source.declaration.digest,
                    second_source.declaration.digest,
                ),
            ),
        )
    )

    denoiser.evaluate_conditioning_batch(
        torch.zeros(1, 4, 8, 8),
        1.0,
        (negative, positive),
    )

    assert model.control is not None
    expected = expected.repeat(2, 1, 1, 1)
    expected = expected * torch.tensor([3.0, 2.0]).reshape(2, 1, 1, 1)
    torch.testing.assert_close(model.control.down[0], expected.expand(-1, 320, -1, -1))
    assert torch.all(model.control.down[1] == 1.0)
    assert torch.all(model.control.middle == 1.0)

    wrong_family_source = _effect_mask_source(
        torch.ones(1, 2, 2),
        "sdxl.unet.input_skip.00.v1",
    )
    wrong_family_field = compile_sd_effect_mask(
        wrong_family_source, latent_height=8, latent_width=8
    )
    with pytest.raises(DenoiseError, match="does not belong"):
        SDDenoiser(
            cast(UNetModel, CapturingUNet()),
            DiscreteSigmas.linear_beta(),
            control_model=cast(SD15ControlNet, FixedControl()),
            control_hint=torch.zeros(1, 3, 64, 64),
            control_effect_masks=((wrong_family_field,),),
            compute_dtype=torch.float32,
        )

    corrupted = compile_sd_effect_mask(source, latent_height=8, latent_width=8)
    corrupted.mask.fill_(1.0)
    with pytest.raises(ControlResourceBindingError, match="changed after compilation"):
        SDDenoiser(
            cast(UNetModel, CapturingUNet()),
            DiscreteSigmas.linear_beta(),
            control_model=cast(SD15ControlNet, FixedControl()),
            control_hint=torch.zeros(1, 3, 64, 64),
            control_effect_masks=((corrupted,),),
            compute_dtype=torch.float32,
        )

    corrupted_shape = compile_sd_effect_mask(one_source, latent_height=8, latent_width=8)
    corrupted_shape.mask.resize_(1, 1, 4, 16)
    with pytest.raises(ControlResourceBindingError, match="tensor shape changed"):
        SDDenoiser(
            cast(UNetModel, CapturingUNet()),
            DiscreteSigmas.linear_beta(),
            control_model=cast(SD15ControlNet, FixedControl()),
            control_hint=torch.zeros(1, 3, 64, 64),
            control_effect_masks=((corrupted_shape,),),
            compute_dtype=torch.float32,
        )

    corrupted_dtype = compile_sd_effect_mask(one_source, latent_height=8, latent_width=8)
    corrupted_dtype.mask.data = corrupted_dtype.mask.to(torch.float16)
    with pytest.raises(ControlResourceBindingError, match="dtype or layout changed"):
        SDDenoiser(
            cast(UNetModel, CapturingUNet()),
            DiscreteSigmas.linear_beta(),
            control_model=cast(SD15ControlNet, FixedControl()),
            control_hint=torch.zeros(1, 3, 64, 64),
            control_effect_masks=((corrupted_dtype,),),
            compute_dtype=torch.float32,
        )


def test_residual_carrier_is_frozen_and_validates_exact_sequence() -> None:
    residuals = _residuals()
    with pytest.raises(FrozenInstanceError):
        residuals.middle = torch.empty(1)  # type: ignore[misc]
    bad = list(residuals.down)
    bad[4] = torch.empty(1, 320, 4, 4)
    with pytest.raises(ValueError, match="down residual 4 shape"):
        SDControlResiduals(tuple(bad), residuals.middle)


@pytest.mark.parametrize("dtype", (torch.int32, torch.complex64))
def test_residual_carrier_refuses_non_real_floating_dtypes(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="residuals require a floating dtype"):
        _residuals(dtype)


class _CheapConv(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        self.test_kernel_size = kernel_size
        self.test_stride = stride
        self.test_padding = padding
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            device="meta",
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        height = (
            input.shape[-2] + 2 * self.test_padding - self.test_kernel_size
        ) // self.test_stride + 1
        width = (
            input.shape[-1] + 2 * self.test_padding - self.test_kernel_size
        ) // self.test_stride + 1
        base = F.interpolate(input.mean(dim=1, keepdim=True), size=(height, width), mode="nearest")
        return base.expand(input.shape[0], self.out_channels, height, width)


class _CheapLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias, device="meta")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.mean(dim=-1, keepdim=True).expand(*input.shape[:-1], self.out_features)


class _CheapOperations(InitlessOperations):
    def conv1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> torch.nn.Conv1d:
        return torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def conv_transpose1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
    ) -> torch.nn.ConvTranspose1d:
        return torch.nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ) -> torch.nn.Conv2d:
        if (
            not isinstance(kernel_size, int)
            or not isinstance(stride, int)
            or not isinstance(padding, int)
            or dilation != 1
            or groups != 1
            or padding_mode != "zeros"
        ):
            return super().conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )
        return _CheapConv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def conv3d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> torch.nn.Conv3d:
        return torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def group_norm(
        self,
        num_channels: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
    ) -> torch.nn.GroupNorm:
        return torch.nn.GroupNorm(num_groups, num_channels, eps=eps)

    def linear(self, in_features: int, out_features: int, *, bias: bool = True) -> torch.nn.Linear:
        return _CheapLinear(in_features, out_features, bias=bias)

    def layer_norm(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.nn.LayerNorm:
        return torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def embedding(self, num_embeddings: int, embedding_dim: int) -> torch.nn.Embedding:
        return torch.nn.Embedding(num_embeddings, embedding_dim)

    def rms_norm(self, normalized_shape: int, *, eps: float | None = None) -> torch.nn.RMSNorm:
        return torch.nn.RMSNorm(normalized_shape, eps=eps)


def test_cpu_forward_returns_commissioned_channels_and_scales() -> None:
    model = SD15ControlNet(operations=_CheapOperations()).eval()
    x = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8) / 100
    hint = torch.arange(3 * 40 * 80, dtype=torch.float64).reshape(1, 3, 40, 80)
    result = model(
        x,
        hint,
        torch.tensor([500.0]),
        torch.arange(5 * 768, dtype=torch.float32).reshape(1, 5, 768) / 100,
    )
    assert isinstance(result, SDControlResiduals)
    assert [tuple(value.shape) for value in result.down] == [
        (1, 320, 8, 8),
        (1, 320, 8, 8),
        (1, 320, 8, 8),
        (1, 320, 4, 4),
        (1, 640, 4, 4),
        (1, 640, 4, 4),
        (1, 640, 2, 2),
        (1, 1280, 2, 2),
        (1, 1280, 2, 2),
        (1, 1280, 1, 1),
        (1, 1280, 1, 1),
        (1, 1280, 1, 1),
    ]
    assert tuple(result.middle.shape) == (1, 1280, 1, 1)
    assert not hasattr(model, "hint_cache")


def test_forward_clears_invocation_hint_cache_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SD15ControlNet(operations=_CheapOperations()).eval()
    controlnet_module = importlib.import_module("dinkster_inference_torch.controlnet")
    original = controlnet_module.normalize_control_hint
    seen_cache: dict[object, torch.Tensor] | None = None

    def capture_cache(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal seen_cache
        seen_cache = cast(dict[object, torch.Tensor], kwargs["cache"])
        return original(*args, **kwargs)

    def fail_hint_block(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("stop")

    monkeypatch.setattr(controlnet_module, "normalize_control_hint", capture_cache)
    monkeypatch.setattr(model.input_hint_block, "forward", fail_hint_block)
    with pytest.raises(RuntimeError, match="stop"):
        model(
            torch.empty(1, 4, 8, 8),
            torch.empty(1, 3, 64, 64),
            torch.empty(1),
            torch.empty(1, 2, 768),
        )
    assert seen_cache == {}


def test_forward_strictly_validates_batch_dtype_and_spatial_shape() -> None:
    with torch.device("meta"):
        model = SD15ControlNet()
    valid = {
        "x": torch.empty(1, 4, 8, 8),
        "hint": torch.empty(1, 3, 64, 64),
        "timesteps": torch.empty(1),
        "context": torch.empty(1, 2, 768),
    }
    with pytest.raises(ValueError, match="timesteps"):
        model(**{**valid, "timesteps": torch.empty(2)})
    with pytest.raises(ValueError, match="dtype must match"):
        model(**{**valid, "context": torch.empty(1, 2, 768, dtype=torch.float64)})
    with pytest.raises(ValueError, match="divisible by 8"):
        model(**{**valid, "x": torch.empty(1, 4, 9, 8)})


def test_assembly_uses_planned_component_and_compute_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = sd15_controlnet_layout(SD15ControlNetConfig())
    source_path = tmp_path / "controlnet.safetensors"
    source_path.write_bytes(b"controlnet source")
    component = ComponentPlan(
        component="controlnet",
        path=source_path,
        config=layout.config,
        keys={key: key for key in layout.keys},
        dtypes={key: FLOAT32 for key in layout.keys},
        quant={},
    )
    plan = ControlNetAssemblyPlan(
        component,
        layout,
        "canonical",
        "blake3:" + "a" * 64,
        tuple(sorted(layout.keys)),
    )
    seen: dict[str, object] = {}

    def fake_load(component_arg: object, build: object, **kwargs: object) -> SD15ControlNet:
        seen.update(component=component_arg, build=build, **kwargs)
        assert build is SD15ControlNet
        return build(operations=_CheapOperations())

    assemble_module = importlib.import_module("dinkster_inference_torch.assemble")
    monkeypatch.setattr(assemble_module, "_load_component", fake_load)
    source_path.unlink()
    assembled = assemble_sd15_controlnet(plan, controlnet_dtype=torch.float32)
    assert assembled == AssembledControlNet(assembled.controlnet, torch.float32)
    assert assembled.resource_digest == sd15_controlnet_resource_digest(
        "blake3:" + "a" * 64, "canonical", torch.float32
    )
    assert seen == {
        "component": component,
        "build": seen["build"],
        "compute_dtype": torch.float32,
        "fp8_matmul": False,
    }


@pytest.mark.parametrize("component_name", ("controlnet", "controlnet_union", "diffusion"))
def test_floating_storage_cast_is_diagnosed(
    component_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    component = ComponentPlan(
        component=component_name,
        path=Path(f"{component_name}.safetensors"),
        config=object(),
        keys={"weight": "weight"},
        dtypes={"weight": FLOAT64},
        quant={},
    )
    stored = torch.ones((1, 1), dtype=torch.float64)
    assemble_module = importlib.import_module("dinkster_inference_torch.assemble")

    def load_tensors(_path: Path, keys: object) -> dict[str, torch.Tensor]:
        assert set(keys) == {"weight"}  # type: ignore[arg-type]
        return {"weight": stored}

    def build(_config: object, *, operations: object) -> torch.nn.Linear:
        del operations
        return torch.nn.Linear(1, 1, bias=False, device="meta")

    monkeypatch.setattr(assemble_module, "load_tensors", load_tensors)
    caplog.set_level("INFO", logger="dinkster.inference_torch.assemble")

    loaded = assemble_module._load_component(
        component,
        build,
        compute_dtype=torch.float32,
        fp8_matmul=False,
    )

    assert loaded.weight.dtype is torch.float32
    assert torch.equal(loaded.weight, stored.float())
    assert (
        f"{component_name}: casting checkpoint storage dtype torch.float64"
        " to compute dtype torch.float32"
    ) in caplog.messages
