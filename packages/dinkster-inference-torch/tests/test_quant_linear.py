"""Module-held fp8 weight storage: Fp8Linear.

Proves the no-wrapper stance holds through stock nn.Module mechanics:
state-dict layout and round trips, assign-loading, dequant-forward
numerics against the Fp8ScaledWeight value type, the patch/requantize
bridge, autograd through the dequant path, torch.compile fullgraph,
and worker-thread execution. The fp8 MATMUL path needs hardware
(torch._scaled_mm); its numerics live in test_gpu.py - here only its
binding rules and the rank fallback are pinned.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from dinkster_inference_torch import (
    Fp8Linear,
    Int8Embedding,
    Int8Linear,
    default_fp8_matmul,
    supports_fp8_matmul,
)
from dinkster_inference_torch import quant as quant_mod
from dinkster_inference_torch import quant_linear as quant_linear_mod
from dinkster_inference_torch._nvfp4_diagnostics import (
    Nvfp4DiagnosticsRecorder,
    nvfp4_runtime_status,
)
from dinkster_inference_torch.quant import (
    Fp8ScaledWeight,
    Int8PackedWeight,
    Nvfp4PackedWeight,
    quantize_fp8_scaled,
    requantize_int8,
    requantize_nvfp4,
)
from dinkster_inference_torch.quant_linear import (
    Int8ExecutionError,
    Nvfp4ExecutionError,
    Nvfp4Linear,
    linear_input_act,
)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2


def test_int8_convrot_embedding_matches_kitchen() -> None:
    generator = torch.Generator().manual_seed(20260830)
    weight = torch.randint(-100, 101, (11, 256), generator=generator, dtype=torch.int8)
    scale = torch.rand((11, 1), generator=generator, dtype=torch.float32) / 100
    indices = torch.tensor([[1, 7, 4], [10, 0, 3]])
    layer = Int8Embedding(
        11,
        256,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)

    expected = torch.ops.dinkster_kitchen.dequantize_int8_embedding(weight, scale, indices, 256, 0)
    torch.testing.assert_close(layer(indices), expected, rtol=0, atol=0)
    assert set(layer.state_dict()) == {"weight", "weight_scale"}


@pytest.mark.parametrize("rank", [2, 3])
def test_int8_convrot_linear_matches_kitchen(rank: int) -> None:
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

    generator = torch.Generator().manual_seed(41)
    input_shape = (2, 3, 256) if rank == 3 else (6, 256)
    input = torch.randn(input_shape, generator=generator)
    weight = torch.randint(-100, 101, (7, 256), generator=generator, dtype=torch.int8)
    scale = torch.rand((7, 1), generator=generator, dtype=torch.float32) / 100
    bias = torch.randn(7, generator=generator)
    layer = Int8Linear(
        256,
        7,
        bias=True,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale, "bias": bias}, assign=True)

    expected = dinkster_kitchen.int8_linear(
        input,
        weight,
        scale,
        bias,
        out_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
    )
    torch.testing.assert_close(layer(input), expected, rtol=0, atol=0)
    assert layer.state_dict()["weight"].dtype == torch.int8
    assert layer.state_dict()["weight_scale"].shape == (7, 1)


def test_int8_convrot_requantize_is_bit_exact_with_kitchen_wrapper() -> None:
    from dinkster_kitchen.tensor import (  # pyright: ignore[reportMissingTypeStubs]
        QuantizedTensor,
        TensorWiseINT8Layout,
    )

    generator = torch.Generator().manual_seed(20260822)
    source = torch.randn((7, 256), generator=generator)
    qdata, params = TensorWiseINT8Layout.quantize(
        source,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=256,
    )
    stored = Int8PackedWeight(qdata, params.scale, torch.float32, True, 256)
    patched = stored.dequantize(torch.float32) + torch.randn(
        stored.shape, generator=generator
    ).mul_(0.01)
    seed = 609766

    wrapped = QuantizedTensor(
        stored.qdata,
        "TensorWiseINT8Layout",
        TensorWiseINT8Layout.Params(
            scale=stored.scale,
            orig_dtype=stored.orig_dtype,
            orig_shape=stored.shape,
            is_weight=True,
            convrot=stored.convrot,
            convrot_groupsize=stored.convrot_groupsize,
        ),
    )
    assert torch.equal(stored.dequantize(torch.float32), wrapped.dequantize())
    expected = wrapped.requantize_from_float(
        patched,
        scale="recalculate",
        stochastic_rounding=seed,
        inplace_ops=True,
    )
    actual = requantize_int8(stored, patched, seed=seed)
    assert torch.equal(actual.qdata, expected._qdata)  # pyright: ignore[reportPrivateUsage]
    assert torch.equal(
        actual.scale,
        expected._params.scale,  # pyright: ignore[reportPrivateUsage]
    )
    assert actual.orig_dtype == stored.orig_dtype
    assert actual.convrot == stored.convrot
    assert actual.convrot_groupsize == stored.convrot_groupsize


def test_int8_native_matmul_supported_is_a_device_type_predicate() -> None:
    supported = quant_linear_mod._int8_native_matmul_supported  # pyright: ignore[reportPrivateUsage]
    assert supported(torch.device("cpu"))
    assert supported(torch.device("cuda", 0))
    assert supported(torch.device("xpu", 0))
    assert not supported(torch.device("mps"))
    assert not supported(torch.device("mps", 0))


@pytest.mark.parametrize("convrot", [False, True])
def test_int8_unsupported_native_device_routes_through_dequant(
    monkeypatch: pytest.MonkeyPatch, convrot: bool
) -> None:
    """Devices without a native INT8 matmul (MPS) execute the dequant
    route even when the layer plans the native route, and never reach
    kitchen int8_linear."""
    pytest.importorskip("dinkster_kitchen")
    generator = torch.Generator().manual_seed(46)
    input = torch.randn((2, 256), generator=generator)
    weight = torch.randint(-100, 101, (5, 256), generator=generator, dtype=torch.int8)
    scale_shape = (5, 1) if convrot else ()
    scale = torch.rand(scale_shape, generator=generator, dtype=torch.float32) / 100
    bias = torch.randn(5, generator=generator)
    layer = Int8Linear(
        256,
        5,
        bias=True,
        compute_dtype=torch.float32,
        convrot=convrot,
        convrot_groupsize=256,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale, "bias": bias}, assign=True)
    assert not layer.full_precision_matmul

    def refuse_native(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("kitchen int8_linear must not run on an unsupported device")

    def unsupported(device: torch.device) -> bool:
        return False

    monkeypatch.setattr(quant_linear_mod, "_int8_linear", refuse_native)
    monkeypatch.setattr(quant_linear_mod, "_int8_native_matmul_supported", unsupported)
    expected_weight = quant_linear_mod._dequantize_int8(  # pyright: ignore[reportPrivateUsage]
        weight, scale, dtype=torch.float32, convrot=convrot, convrot_groupsize=256
    )

    with torch.no_grad():
        output = layer(input)
    torch.testing.assert_close(
        output, torch.nn.functional.linear(input, expected_weight, bias), rtol=0, atol=0
    )


def test_int8_unsupported_native_device_refuses_grad_enabled_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(48)
    input = torch.randn((2, 16), generator=generator, requires_grad=True)
    weight = torch.randint(-100, 101, (4, 16), generator=generator, dtype=torch.int8)
    scale = torch.rand((), generator=generator, dtype=torch.float32) / 100
    layer = Int8Linear(
        16,
        4,
        bias=False,
        compute_dtype=torch.float32,
        convrot=False,
        convrot_groupsize=16,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)

    def unexpected_predicate(device: torch.device) -> bool:
        raise AssertionError("gradient refusal must precede the native-device predicate")

    monkeypatch.setattr(quant_linear_mod, "_int8_native_matmul_supported", unexpected_predicate)

    with pytest.raises(Int8ExecutionError, match="inference-only"):
        layer(input)


def test_int8_supported_native_device_still_uses_kitchen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("dinkster_kitchen")
    generator = torch.Generator().manual_seed(47)
    input = torch.randn((2, 256), generator=generator)
    weight = torch.randint(-100, 101, (5, 256), generator=generator, dtype=torch.int8)
    scale = torch.rand((5, 1), generator=generator, dtype=torch.float32) / 100
    layer = Int8Linear(
        256,
        5,
        bias=False,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)

    seen: list[torch.device] = []
    sentinel = torch.zeros(2, 5)
    original_predicate = quant_linear_mod._int8_native_matmul_supported  # pyright: ignore[reportPrivateUsage]

    def spying_predicate(device: torch.device) -> bool:
        seen.append(device)
        return original_predicate(device)

    def sentinel_linear(*args: object, **kwargs: object) -> torch.Tensor:
        return sentinel

    monkeypatch.setattr(quant_linear_mod, "_int8_native_matmul_supported", spying_predicate)
    monkeypatch.setattr(quant_linear_mod, "_int8_linear", sentinel_linear)

    assert layer(input) is sentinel
    assert seen == [input.device]


def test_int8_full_precision_convrot_dequantizes_before_matmul() -> None:
    kitchen = pytest.importorskip("dinkster_kitchen")
    assert kitchen is not None

    generator = torch.Generator().manual_seed(42)
    input = torch.randn((2, 256), generator=generator, requires_grad=True)
    weight = torch.randint(-100, 101, (5, 256), generator=generator, dtype=torch.int8)
    scale = torch.rand((5, 1), generator=generator, dtype=torch.float32) / 100
    layer = Int8Linear(
        256,
        5,
        bias=False,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
        full_precision_matmul=True,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)
    expected_weight = torch.ops.dinkster_kitchen.dequantize_int8_convrot_weight(weight, scale, 256)

    output = layer(input)
    torch.testing.assert_close(output, torch.nn.functional.linear(input, expected_weight))
    output.sum().backward()
    assert input.grad is not None


def test_int8_fused_training_refuses_off_cuda_before_mps_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input = torch.randn((2, 64), requires_grad=True)
    layer = Int8Linear(
        64,
        8,
        bias=False,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=64,
    )
    layer.bind_fused_training(True)

    def unexpected_predicate(device: torch.device) -> bool:
        raise AssertionError("fused-training refusal must precede the native-device predicate")

    monkeypatch.setattr(quant_linear_mod, "_int8_native_matmul_supported", unexpected_predicate)

    with pytest.raises(Int8ExecutionError, match="requires CUDA input"):
        layer(input)


def test_int8_fused_training_refuses_full_precision_checkpoint() -> None:
    layer = Int8Linear(
        64,
        9,
        bias=False,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=64,
        full_precision_matmul=True,
    )

    with pytest.raises(ValueError, match="pinned to full-precision"):
        layer.bind_fused_training(True)


@pytest.mark.parametrize(
    ("in_features", "out_features", "convrot", "convrot_groupsize", "message"),
    [
        (63, 8, False, 64, "input features divisible by 16"),
        (64, 8, True, 0, "ConvRot group size to be a power of 4 >= 4, got 0"),
        (64, 8, True, 32, "ConvRot group size to be a power of 4 >= 4, got 32"),
        (80, 8, True, 64, "input features divisible by ConvRot group size 64"),
    ],
)
def test_int8_fused_training_refuses_unsupported_layout(
    in_features: int,
    out_features: int,
    convrot: bool,
    convrot_groupsize: int,
    message: str,
) -> None:
    layer = Int8Linear(
        in_features,
        out_features,
        bias=False,
        compute_dtype=torch.bfloat16,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
    )

    with pytest.raises(ValueError, match=message):
        layer.bind_fused_training(True)


@pytest.mark.parametrize("convrot_groupsize", [0, 32])
def test_int8_fused_training_refuses_group_made_invalid_after_binding(
    convrot_groupsize: int,
) -> None:
    layer = Int8Linear(
        64,
        8,
        bias=False,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=64,
    )
    layer.bind_fused_training(True)
    layer.convrot_groupsize = convrot_groupsize

    with pytest.raises(Int8ExecutionError, match="ConvRot group size to be a power of 4 >= 4"):
        layer(torch.randn((2, 64)))


def test_int8_fused_training_refuses_trainable_bias() -> None:
    layer = Int8Linear(
        64,
        8,
        bias=True,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=64,
    )
    assert layer.bias is not None
    layer.bias.requires_grad_(True)

    with pytest.raises(ValueError, match="does not support trainable bias"):
        layer.bind_fused_training(True)


def test_int8_fused_training_refuses_bias_made_trainable_after_binding() -> None:
    layer = Int8Linear(
        64,
        8,
        bias=True,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=64,
    )
    layer.bind_fused_training(True)
    assert layer.bias is not None
    layer.bias.requires_grad_(True)

    with pytest.raises(Int8ExecutionError, match="does not support trainable bias"):
        layer(torch.randn((2, 64)))


def test_int8_training_chunk_bounds_real_h3_adaln_temporary_storage() -> None:
    weight = torch.empty((96_768, 2_688), dtype=torch.int8, device="meta")

    assert (
        quant_linear_mod._int8_training_chunk_features(  # pyright: ignore[reportPrivateUsage]
            weight,
            dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=64,
        )
        == 128
    )


@pytest.mark.parametrize("input_act", ["swiglu", "gelu_tanh"])
def test_int8_full_precision_convrot_input_activation_supports_backward(
    input_act: Literal["gelu_tanh", "swiglu"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("dinkster_kitchen")
    generator = torch.Generator().manual_seed(44)
    width = 256
    input_width = width * 2 if input_act == "swiglu" else width
    input = torch.randn((3, input_width), generator=generator, requires_grad=True)
    weight = torch.randint(-100, 101, (5, width), generator=generator, dtype=torch.int8)
    scale = torch.rand((5, 1), generator=generator) / 100
    layer = Int8Linear(
        width,
        5,
        bias=False,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
        full_precision_matmul=True,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)
    dequantized = torch.ops.dinkster_kitchen.dequantize_int8_convrot_weight(weight, scale, 256)
    if input_act == "swiglu":
        gate, up = input.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
    else:
        activated = torch.nn.functional.gelu(input, approximate="tanh")

    forward_inputs: list[torch.Tensor] = []
    original_forward = layer.forward

    def record_forward(forwarded_input: torch.Tensor) -> torch.Tensor:
        forward_inputs.append(forwarded_input)
        return original_forward(forwarded_input)

    monkeypatch.setattr(layer, "forward", record_forward)
    output = quant_linear_mod.linear_input_act(layer, input, input_act)
    assert len(forward_inputs) == 1
    assert forward_inputs[0] is not input
    assert torch.equal(forward_inputs[0], activated)
    torch.testing.assert_close(output, torch.nn.functional.linear(activated, dequantized))
    output.sum().backward()
    assert input.grad is not None


@pytest.mark.parametrize("input_act", ["swiglu", "gelu_tanh"])
def test_int8_convrot_folds_input_activation_with_kitchen_by_default(
    input_act: Literal["gelu_tanh", "swiglu"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

    generator = torch.Generator().manual_seed(43)
    width = 256
    input_width = width * 2 if input_act == "swiglu" else width
    input = torch.randn((6, input_width), generator=generator)
    weight = torch.randint(-100, 101, (7, width), generator=generator, dtype=torch.int8)
    scale = torch.rand((7, 1), generator=generator) / 100
    layer = Int8Linear(
        width,
        7,
        bias=False,
        compute_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)
    forward_calls: list[tuple[torch.Tensor, Literal["gelu_tanh", "swiglu"] | None]] = []
    original_forward = layer._forward  # pyright: ignore[reportPrivateUsage]

    def record_forward(
        forwarded_input: torch.Tensor,
        forwarded_input_act: Literal["gelu_tanh", "swiglu"] | None = None,
    ) -> torch.Tensor:
        forward_calls.append((forwarded_input, forwarded_input_act))
        return original_forward(forwarded_input, forwarded_input_act)

    monkeypatch.setattr(layer, "_forward", record_forward)

    expected = dinkster_kitchen.int8_linear(
        input,
        weight,
        scale,
        out_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=256,
        input_act=input_act,
    )

    actual = quant_linear_mod.linear_input_act(layer, input, input_act)

    assert len(forward_calls) == 1
    forwarded_input, forwarded_input_act = forward_calls[0]
    assert forwarded_input is input
    assert forwarded_input_act == input_act
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def bits(tensor: torch.Tensor) -> torch.Tensor:
    """Bitwise view for exact fp8 comparison (fp8 eq kernels are not
    guaranteed on CPU; bytes always are)."""
    return tensor.reshape(-1).view(torch.uint8)


def make_layer(
    in_features: int = 4,
    out_features: int = 3,
    *,
    bias: bool = True,
    fp8_dtype: torch.dtype = E4M3,
    compute_dtype: torch.dtype = torch.float32,
    scale: float = 2.0,
    seed: int = 7,
) -> Fp8Linear:
    layer = Fp8Linear(
        in_features,
        out_features,
        bias=bias,
        fp8_dtype=fp8_dtype,
        compute_dtype=compute_dtype,
    )
    generator = torch.Generator().manual_seed(seed)
    state = {
        "weight": torch.randn(out_features, in_features, generator=generator).to(fp8_dtype),
        "weight_scale": torch.tensor(scale, dtype=torch.float32),
        "input_scale": torch.ones((), dtype=torch.float32),
    }
    if bias:
        state["bias"] = torch.randn(out_features, generator=generator).to(compute_dtype)
    layer.load_state_dict(state, assign=True)
    return layer


def make_nvfp4_layer(
    *, pre_quant_scale: bool = False, full_precision_matmul: bool = False
) -> Nvfp4Linear:
    layer = Nvfp4Linear(
        16,
        16,
        bias=True,
        compute_dtype=torch.float32,
        pre_quant_scale=pre_quant_scale,
        full_precision_matmul=full_precision_matmul,
    )
    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]
    weight = torch.linspace(-2.0, 2.0, 256).reshape(16, 16)
    tensor_scale = torch.tensor(weight.abs().amax().item() / (448.0 * 6.0), dtype=torch.float32)
    qweight, block_scale = kitchen.quantize(weight, tensor_scale)
    state = {
        "weight": qweight,
        "weight_scale": block_scale,
        "weight_scale_2": tensor_scale,
        "input_scale": torch.tensor(0.25, dtype=torch.float32),
        "bias": torch.linspace(-0.2, 0.2, 16),
    }
    if pre_quant_scale:
        state["pre_quant_scale"] = torch.linspace(0.5, 1.5, 16)
    layer.load_state_dict(state, strict=True, assign=True)
    return layer


def test_nvfp4_state_dict_is_ordinary_state_and_assign_round_trips() -> None:
    layer = make_nvfp4_layer(pre_quant_scale=True)
    assert set(layer.state_dict()) == {
        "weight",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
        "pre_quant_scale",
        "bias",
    }
    assert layer.weight.dtype == torch.uint8
    assert layer.weight.shape == (16, 8)
    assert layer.weight_scale.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (128, 4)
    fresh = Nvfp4Linear(
        16,
        16,
        bias=True,
        compute_dtype=torch.float32,
        pre_quant_scale=True,
    )
    fresh.load_state_dict(layer.state_dict(), strict=True, assign=True)
    for key, value in layer.state_dict().items():
        assert torch.equal(bits(fresh.state_dict()[key]), bits(value))


def test_nvfp4_cpu_dequant_reference_and_pre_scale_order() -> None:
    layer = make_nvfp4_layer(pre_quant_scale=True)
    x = torch.randn(2, 3, 16)
    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]
    weight = kitchen.dequantize(
        layer.weight,
        layer.weight_scale_2,
        layer.weight_scale,
        output_type=torch.float32,
    )[:16, :16]
    assert layer.pre_quant_scale is not None
    expected = torch.nn.functional.linear(
        x * layer.pre_quant_scale,
        weight,
        layer.bias,
    )
    assert torch.equal(layer(x), expected)


def test_nvfp4_synthetic_cpu_direct_layer_fallback_acceptance() -> None:
    layer = make_nvfp4_layer(pre_quant_scale=True)
    recorder = Nvfp4DiagnosticsRecorder()
    layer._bind_diagnostics(recorder)  # pyright: ignore[reportPrivateUsage]
    runtime = SimpleNamespace(
        assembled=SimpleNamespace(diffusion=SimpleNamespace(_nvfp4_diagnostics=recorder))
    )
    generator = torch.Generator().manual_seed(20260807)
    x = torch.randn(2, 3, 16, generator=generator)
    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]
    weight = kitchen.dequantize(
        layer.weight,
        layer.weight_scale_2,
        layer.weight_scale,
        output_type=torch.float32,
    )[:16, :16]
    assert layer.pre_quant_scale is not None
    expected = torch.nn.functional.linear(
        x * layer.pre_quant_scale,
        weight,
        layer.bias,
    )

    assert torch.equal(layer(x), expected)
    status = nvfp4_runtime_status(runtime)
    assert status.active == 0
    assert len(status.completed) == 1
    assert status.completed[0].terminal == "success"
    assert status.lifetime["route_non_cuda"] == 1
    assert status.lifetime["dequantize_success"] == 1
    assert status.lifetime.get("route_native", 0) == 0
    assert status.lifetime.get("quantize_success", 0) == 0
    assert status.lifetime.get("scaled_mm_success", 0) == 0


def test_nvfp4_route_refuses_amd_hip_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A HIP torch reports device type "cuda" and a gfx-generation
    capability major (gfx1200 reports 12), so the HIP refusal must fire
    before the SM 10 check can misread it as native-capable."""
    route = quant_linear_mod._nvfp4_route  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(torch.version, "hip", "7.1.44064", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(major=12),  # pyright: ignore[reportUnknownLambdaType]
    )

    assert route(False, torch.device("cuda"), 2) == (False, "route_hip")


def test_nvfp4_route_native_needs_nvidia_sm10(monkeypatch: pytest.MonkeyPatch) -> None:
    route = quant_linear_mod._nvfp4_route  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    properties = SimpleNamespace(major=12)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: properties,  # pyright: ignore[reportUnknownLambdaType]
    )

    assert route(False, torch.device("cuda"), 2) == (True, "route_native")

    properties.major = 8
    assert route(False, torch.device("cuda"), 2) == (False, "route_pre_sm10")


def test_nvfp4_route_early_refusals_never_read_device_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = quant_linear_mod._nvfp4_route  # pyright: ignore[reportPrivateUsage]

    def poisoned(device: torch.device) -> SimpleNamespace:
        raise AssertionError("early refusals must not read device properties")

    monkeypatch.setattr(torch.cuda, "get_device_properties", poisoned)

    assert route(True, torch.device("cuda"), 2) == (False, "route_full_precision")
    assert route(False, torch.device("cpu"), 2) == (False, "route_non_cuda")
    monkeypatch.setattr(torch.version, "hip", "7.1.44064", raising=False)
    assert route(False, torch.device("cuda"), 1) == (False, "route_hip")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert route(False, torch.device("cuda"), 1) == (False, "route_rank")


def test_nvfp4_seed_609766_comfyui_2eb609766_stochastic_golden() -> None:
    """Input: float32 linspace(-3, 3, 256).reshape(16, 16); seed: 609766."""
    tensor = torch.linspace(-3, 3, 256, dtype=torch.float32).reshape(16, 16)
    original = tensor.clone()
    stored = Nvfp4PackedWeight(
        torch.zeros(16, 8, dtype=torch.uint8),
        torch.zeros(128, 4, dtype=torch.float8_e4m3fn),
        torch.tensor(1.0),
        (16, 16),
        torch.float32,
    )
    result = requantize_nvfp4(stored, tensor, seed=609766)
    assert result.qdata.flatten().tolist() == [
        255,
        255,
        239,
        254,
        238,
        255,
        254,
        254,
        255,
        239,
        239,
        254,
        239,
        238,
        254,
        255,
        254,
        255,
        255,
        255,
        239,
        238,
        255,
        239,
        255,
        255,
        254,
        254,
        239,
        238,
        239,
        239,
        255,
        255,
        255,
        254,
        239,
        255,
        238,
        254,
        239,
        255,
        254,
        255,
        239,
        254,
        238,
        222,
        255,
        255,
        254,
        255,
        237,
        221,
        222,
        238,
        255,
        238,
        237,
        237,
        221,
        219,
        186,
        153,
        1,
        35,
        68,
        84,
        102,
        101,
        103,
        103,
        85,
        102,
        102,
        102,
        119,
        118,
        103,
        119,
        102,
        103,
        103,
        118,
        119,
        119,
        119,
        103,
        86,
        118,
        103,
        102,
        102,
        102,
        119,
        119,
        102,
        103,
        103,
        102,
        118,
        119,
        118,
        119,
        119,
        119,
        119,
        119,
        119,
        103,
        119,
        119,
        118,
        119,
        119,
        119,
        103,
        119,
        118,
        118,
        119,
        119,
        102,
        119,
        119,
        119,
        118,
        119,
    ]
    assert (
        result.block_scale.view(torch.uint8).flatten()[:32].tolist()
        == [126] + [0] * 15 + [124] + [0] * 15
    )
    assert result.tensor_scale.item() == 0.0011160714784637094
    assert torch.equal(tensor, original)

    sliced_qdata, sliced_scale = quant_mod._stochastic_quantize_nvfp4(  # pyright: ignore[reportPrivateUsage]
        tensor,
        result.tensor_scale,
        609766,
        block_size=64,
    )
    digest = hashlib.sha256(
        bytes(sliced_qdata.flatten().tolist())
        + bytes(sliced_scale.view(torch.uint8).flatten().tolist())
    ).hexdigest()
    assert digest == "8cd586aca0d7893dab437a0b12cf76188f9240771f1b01cdcab2e64815b5a7a0"


@pytest.mark.parametrize("shape", ((16,), (2, 3, 4, 16)))
def test_nvfp4_linear_accepts_rank_1_and_rank_4(shape: tuple[int, ...]) -> None:
    layer = make_nvfp4_layer()
    assert layer(torch.randn(shape)).shape == shape[:-1] + (16,)


def test_nvfp4_optional_input_scale_state() -> None:
    layer = Nvfp4Linear(16, 16, bias=False, compute_dtype=torch.float32, input_scale=False)
    assert layer.input_scale is None
    assert "input_scale" not in layer.state_dict()


def test_nvfp4_cpu_and_full_precision_never_request_native_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = make_nvfp4_layer(full_precision_matmul=True)
    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]

    def forbidden_quantize(
        _x: torch.Tensor,
        _per_tensor_scale: torch.Tensor,
        _epsilon: float = 0.0,
        _pad_16x: bool = False,
        _hi_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("native NVFP4 route was called")

    def forbidden_mm(
        _a: torch.Tensor,
        _b: torch.Tensor,
        _tensor_scale_a: torch.Tensor,
        _tensor_scale_b: torch.Tensor,
        _block_scale_a: torch.Tensor,
        _block_scale_b: torch.Tensor,
        _bias: torch.Tensor | None = None,
        _out_dtype: torch.dtype | None = None,
        _alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise AssertionError("native NVFP4 route was called")

    monkeypatch.setattr(
        quant_linear_mod,
        "_nvfp4_kitchen",
        quant_linear_mod._Nvfp4Kitchen(  # pyright: ignore[reportPrivateUsage]
            forbidden_quantize,
            kitchen.dequantize,
            forbidden_mm,
            kitchen.registry,
        ),
    )
    assert layer(torch.randn(2, 16)).shape == (2, 16)


def test_nvfp4_missing_kitchen_is_a_named_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = Nvfp4Linear(16, 16, bias=False, compute_dtype=torch.float32)
    monkeypatch.setattr(quant_linear_mod, "_nvfp4_kitchen_probed", True)
    monkeypatch.setattr(quant_linear_mod, "_nvfp4_kitchen", None)
    with pytest.raises(Nvfp4ExecutionError, match="requires dinkster-kitchen"):
        layer(torch.randn(1, 16))


class _CudaRegistry:
    def get_capable_backend(self, operation: str, _kwargs: object) -> str:
        assert operation in {"quantize_nvfp4", "scaled_mm_nvfp4"}
        return "cuda"


def test_nvfp4_fast_call_contract_padding_crop_and_rank_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = Nvfp4Linear(32, 13, bias=True, compute_dtype=torch.bfloat16)
    calls: list[tuple[str, object]] = []

    def quantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        epsilon: float = 0.0,
        pad_16x: bool = False,
        hi_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(("quantize", (tuple(x.shape), scale, epsilon, pad_16x, hi_first)))
        return torch.zeros((16, 16), dtype=torch.uint8), torch.ones(
            (128, 4), dtype=torch.float8_e4m3fn
        )

    def dequantize(
        _qx: torch.Tensor,
        _per_tensor_scale: torch.Tensor,
        _block_scales: torch.Tensor,
        _output_type: torch.dtype = torch.bfloat16,
        _hi_first: bool = True,
    ) -> torch.Tensor:
        raise AssertionError("dequantize route was called")

    def scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        tensor_scale_a: torch.Tensor,
        tensor_scale_b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        calls.append(
            (
                "scaled_mm",
                {
                    "a": a,
                    "b": b,
                    "tensor_scale_a": tensor_scale_a,
                    "tensor_scale_b": tensor_scale_b,
                    "block_scale_a": block_scale_a,
                    "block_scale_b": block_scale_b,
                    "bias": bias,
                    "out_dtype": out_dtype,
                    "alpha": alpha,
                },
            )
        )
        return torch.arange(16 * 16, dtype=torch.bfloat16).reshape(16, 16)

    monkeypatch.setattr(
        quant_linear_mod,
        "_nvfp4_kitchen",
        quant_linear_mod._Nvfp4Kitchen(  # pyright: ignore[reportPrivateUsage]
            quantize,
            dequantize,
            scaled_mm,
            _CudaRegistry(),
        ),
    )
    monkeypatch.setattr(quant_linear_mod, "_nvfp4_kitchen_probed", True)
    x = torch.randn(1, 3, 32)
    out = layer._fast_forward(  # pyright: ignore[reportPrivateUsage]
        x,
        layer.weight,
        layer.weight_scale,
        layer.weight_scale_2,
        layer.input_scale,
        layer.bias,
    )
    assert out is not None
    assert out.shape == (1, 3, 13)
    assert calls[0][0] == "quantize"
    assert calls[0][1][0] == (3, 32)  # type: ignore[index]
    assert calls[0][1][2:] == (0.0, True, True)  # type: ignore[index]
    mm = calls[1][1]
    assert isinstance(mm, dict)
    assert mm["a"].shape == (16, 16)  # type: ignore[union-attr]
    assert mm["b"] is layer.weight
    assert mm["tensor_scale_a"] is layer.input_scale
    assert mm["tensor_scale_b"] is layer.weight_scale_2
    assert mm["block_scale_a"].shape == (128, 4)  # type: ignore[union-attr]
    assert mm["block_scale_b"] is layer.weight_scale
    assert mm["bias"] is layer.bias
    assert mm["out_dtype"] is torch.bfloat16
    assert mm["alpha"] is None


def test_nvfp4_fast_gradient_backend_and_kernel_failures_are_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = Nvfp4Linear(16, 16, bias=False, compute_dtype=torch.float32)
    with pytest.raises(Nvfp4ExecutionError, match="inference-only"):
        layer._fast_forward(  # pyright: ignore[reportPrivateUsage]
            torch.randn(1, 16, requires_grad=True),
            layer.weight,
            layer.weight_scale,
            layer.weight_scale_2,
            layer.input_scale,
            None,
        )

    class WrongRegistry:
        def get_capable_backend(self, _operation: str, _kwargs: object) -> str:
            return "eager"

    def unused_quantize(
        _x: torch.Tensor,
        _per_tensor_scale: torch.Tensor,
        _epsilon: float = 0.0,
        _pad_16x: bool = False,
        _hi_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty((16, 8), dtype=torch.uint8),
            torch.empty((128, 4), dtype=torch.float8_e4m3fn),
        )

    def unused_dequantize(
        _qx: torch.Tensor,
        _per_tensor_scale: torch.Tensor,
        _block_scales: torch.Tensor,
        _output_type: torch.dtype = torch.bfloat16,
        _hi_first: bool = True,
    ) -> torch.Tensor:
        return torch.empty(0)

    def unused_scaled_mm(
        _a: torch.Tensor,
        _b: torch.Tensor,
        _tensor_scale_a: torch.Tensor,
        _tensor_scale_b: torch.Tensor,
        _block_scale_a: torch.Tensor,
        _block_scale_b: torch.Tensor,
        _bias: torch.Tensor | None = None,
        _out_dtype: torch.dtype | None = None,
        _alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty(0)

    monkeypatch.setattr(
        quant_linear_mod,
        "_nvfp4_kitchen",
        quant_linear_mod._Nvfp4Kitchen(  # pyright: ignore[reportPrivateUsage]
            unused_quantize,
            unused_dequantize,
            unused_scaled_mm,
            WrongRegistry(),
        ),
    )
    monkeypatch.setattr(quant_linear_mod, "_nvfp4_kitchen_probed", True)
    with torch.no_grad():
        assert (
            layer._fast_forward(  # pyright: ignore[reportPrivateUsage]
                torch.randn(1, 16),
                layer.weight,
                layer.weight_scale,
                layer.weight_scale_2,
                layer.input_scale,
                None,
            )
            is None
        )

    kernel_layer = Nvfp4Linear(32, 16, bias=False, compute_dtype=torch.bfloat16)

    def dequantize(
        _qx: torch.Tensor,
        _per_tensor_scale: torch.Tensor,
        _block_scales: torch.Tensor,
        _output_type: torch.dtype = torch.bfloat16,
        _hi_first: bool = True,
    ) -> torch.Tensor:
        raise AssertionError("dequantize route was called")

    def quantize_failure(
        x: torch.Tensor,
        per_tensor_scale: torch.Tensor,
        epsilon: float = 0.0,
        pad_16x: bool = False,
        hi_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del x, per_tensor_scale, epsilon, pad_16x, hi_first
        raise RuntimeError("quantize exploded")

    def valid_quantize(
        x: torch.Tensor,
        per_tensor_scale: torch.Tensor,
        epsilon: float = 0.0,
        pad_16x: bool = False,
        hi_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del x, per_tensor_scale, epsilon, pad_16x, hi_first
        return torch.empty((16, 16), dtype=torch.uint8), torch.empty(
            (128, 4), dtype=torch.float8_e4m3fn
        )

    def scaled_mm_failure(
        a: torch.Tensor,
        b: torch.Tensor,
        tensor_scale_a: torch.Tensor,
        tensor_scale_b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            a,
            b,
            tensor_scale_a,
            tensor_scale_b,
            block_scale_a,
            block_scale_b,
            bias,
            out_dtype,
            alpha,
        )
        raise RuntimeError("matrix multiply exploded")

    for quantize, match in (
        (quantize_failure, "quantize_nvfp4 failed"),
        (valid_quantize, "scaled_mm_nvfp4 failed"),
    ):
        monkeypatch.setattr(
            quant_linear_mod,
            "_nvfp4_kitchen",
            quant_linear_mod._Nvfp4Kitchen(  # pyright: ignore[reportPrivateUsage]
                quantize,
                dequantize,
                scaled_mm_failure,
                _CudaRegistry(),
            ),
        )
        with torch.no_grad(), pytest.raises(Nvfp4ExecutionError, match=match):
            kernel_layer._fast_forward(  # pyright: ignore[reportPrivateUsage]
                torch.randn(1, 32),
                kernel_layer.weight,
                kernel_layer.weight_scale,
                kernel_layer.weight_scale_2,
                kernel_layer.input_scale,
                None,
            )


@pytest.mark.parametrize(
    "failure",
    (torch.OutOfMemoryError("oom"), KeyboardInterrupt()),
    ids=("oom", "cancellation"),
)
def test_nvfp4_selected_native_failure_preserves_oom_and_cancellation(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    layer = Nvfp4Linear(16, 16, bias=False, compute_dtype=torch.float32)

    def fail(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise failure

    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        quant_linear_mod,
        "_nvfp4_kitchen",
        quant_linear_mod._Nvfp4Kitchen(  # pyright: ignore[reportPrivateUsage]
            fail,
            kitchen.dequantize,
            kitchen.scaled_mm,
            _CudaRegistry(),
        ),
    )
    monkeypatch.setattr(quant_linear_mod, "_nvfp4_kitchen_probed", True)
    with (
        torch.no_grad(),
        pytest.raises(type(failure), match=str(failure) if str(failure) else None),
    ):
        layer._fast_forward(  # pyright: ignore[reportPrivateUsage]
            torch.randn(1, 16),
            layer.weight,
            layer.weight_scale,
            layer.weight_scale_2,
            layer.input_scale,
            None,
        )


# ------------------------------------------------------- construction


def test_constructor_refuses_non_fp8_storage() -> None:
    with pytest.raises(ValueError, match="not an fp8 dtype"):
        Fp8Linear(4, 3, fp8_dtype=torch.float16, compute_dtype=torch.float32)


def test_state_dict_is_the_comfy_quant_layer_layout() -> None:
    layer = make_layer()
    state = layer.state_dict()
    assert set(state) == {"weight", "weight_scale", "input_scale", "bias"}
    assert state["weight"].dtype == E4M3
    assert state["weight_scale"].dtype == torch.float32
    assert state["weight_scale"].shape == ()
    assert state["input_scale"].dtype == torch.float32
    assert state["bias"].dtype == torch.float32


def test_bias_false_registers_none() -> None:
    layer = make_layer(bias=False)
    assert layer.bias is None
    assert set(layer.state_dict()) == {
        "weight",
        "weight_scale",
        "input_scale",
    }


def test_weight_never_requires_grad() -> None:
    layer = make_layer()
    assert not layer.weight.requires_grad


# ------------------------------------------------- state-dict round trip


def test_assign_load_round_trip_preserves_exact_storage() -> None:
    layer = make_layer(scale=0.375)
    fresh = Fp8Linear(4, 3, fp8_dtype=E4M3, compute_dtype=torch.float32)
    fresh.load_state_dict(layer.state_dict(), assign=True)
    assert torch.equal(bits(fresh.weight), bits(layer.weight))
    assert torch.equal(fresh.weight_scale, layer.weight_scale)
    assert torch.equal(fresh.input_scale, layer.input_scale)
    assert layer.bias is not None and fresh.bias is not None
    assert torch.equal(fresh.bias, layer.bias)
    x = torch.randn(2, 4)
    assert torch.equal(fresh(x), layer(x))


def test_apply_preserves_dtypes_and_values() -> None:
    layer = make_layer()
    before = layer.state_dict()

    def clone(t: torch.Tensor) -> torch.Tensor:
        return t.clone()

    layer._apply(clone)  # pyright: ignore[reportPrivateUsage]
    after = layer.state_dict()
    for key in before:
        assert after[key].dtype == before[key].dtype
        assert torch.equal(bits(after[key]), bits(before[key]))


# ------------------------------------------------------ dequant forward


def test_dequant_forward_is_the_scaled_product() -> None:
    layer = make_layer(scale=2.5)
    x = torch.randn(5, 4)
    expected = torch.nn.functional.linear(
        x,
        layer.weight.to(torch.float32) * layer.weight_scale,
        layer.bias,
    )
    assert torch.equal(layer(x), expected)


def test_dequant_forward_matches_value_type_dequantize() -> None:
    layer = make_layer(scale=0.75, bias=False)
    x = torch.randn(3, 4)
    expected = torch.nn.functional.linear(x, layer.stored().dequantize())
    assert torch.equal(layer(x), expected)


def test_plain_fp8_neutral_scale_is_a_direct_cast() -> None:
    layer = make_layer(scale=1.0, bias=False)
    x = torch.randn(2, 4)
    expected = torch.nn.functional.linear(x, layer.weight.to(torch.float32))
    assert torch.equal(layer(x), expected)


def test_e5m2_storage_dequantizes() -> None:
    layer = make_layer(fp8_dtype=E5M2, scale=2.0, bias=False)
    x = torch.randn(2, 4)
    expected = torch.nn.functional.linear(x, layer.weight.to(torch.float32) * layer.weight_scale)
    assert torch.equal(layer(x), expected)


def test_compute_dtype_decides_output_dtype() -> None:
    layer = make_layer(compute_dtype=torch.bfloat16)
    out = layer(torch.randn(2, 4, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


# ------------------------------------------------------ matmul binding


def test_bind_refuses_e5m2() -> None:
    layer = make_layer(fp8_dtype=E5M2)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        layer.bind_fp8_matmul(True)
    layer.bind_fp8_matmul(False)
    assert not layer.fp8_matmul


def test_bind_refuses_full_precision_pin() -> None:
    layer = Fp8Linear(
        4,
        3,
        fp8_dtype=E4M3,
        compute_dtype=torch.float32,
        full_precision_matmul=True,
    )
    with pytest.raises(ValueError, match="full-precision"):
        layer.bind_fp8_matmul(True)


def test_prepare_fp8_runtime_resolves_matmul_and_quantizer_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        quant_linear_mod,
        "select_fp8_matmul_backend",
        lambda: calls.append("matmul") or "kitchen",
    )
    monkeypatch.setattr(
        quant_linear_mod,
        "_probe_kitchen_quantize_per_tensor_fp8",
        lambda: calls.append("kitchen-quantizer") or object(),
    )

    assert quant_linear_mod.prepare_fp8_matmul_runtime() == "kitchen"
    assert calls == ["matmul", "kitchen-quantizer"]


def test_high_rank_input_falls_back_to_dequant() -> None:
    layer = make_layer()
    layer.bind_fp8_matmul(True)  # capability is the caller's check
    x = torch.randn(2, 2, 3, 4)
    expected = torch.nn.functional.linear(
        x,
        layer.weight.to(torch.float32) * layer.weight_scale,
        layer.bias,
    )
    assert torch.equal(layer(x), expected)


def test_fp8_matmul_route_refuses_gradient_input_before_quantizer_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch._scaled_mm has no backward; the route raises BEFORE the
    # matmul, so the refusal is provable off-CUDA too.
    layer = make_layer()
    layer.bind_fp8_matmul(True)
    kitchen_probe = quant_linear_mod._probe_kitchen_quantize_per_tensor_fp8  # pyright: ignore[reportPrivateUsage]

    def unexpected_probe() -> None:
        raise AssertionError("gradient refusal must precede quantizer probes")

    monkeypatch.setattr(
        quant_linear_mod, "_probe_kitchen_quantize_per_tensor_fp8", unexpected_probe
    )
    x = torch.randn(2, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        layer(x)
    monkeypatch.setattr(quant_linear_mod, "_probe_kitchen_quantize_per_tensor_fp8", kitchen_probe)
    # Under no_grad the same input is not a gradient consumer: the
    # refusal must not fire (whether the backend then supports the
    # matmul is the platform's business, not this gate's).
    with torch.no_grad():
        try:
            layer(x)
        except RuntimeError as exc:
            assert "inference-only" not in str(exc)


def test_supports_fp8_matmul_is_false_off_cuda() -> None:
    assert supports_fp8_matmul(torch.device("cpu")) is False


@pytest.mark.parametrize(
    ("cuda_version", "major", "minor", "torch_version", "windows", "expected"),
    [
        (None, 9, 0, "2.13.0", False, False),  # non-NVIDIA
        ("13.0", 9, 0, "2.2.0", False, True),
        ("13.0", 7, 5, "2.13.0", False, False),
        ("13.0", 8, 0, "2.13.0", False, False),
        ("13.0", 8, 9, "2.2.0", False, False),
        ("13.0", 8, 9, "2.3.0", False, True),
        ("13.0", 8, 9, "2.3.0", True, False),
        ("13.0", 8, 9, "2.4.0", True, True),
    ],
)
def test_default_fp8_matmul_matches_upstream_gate_matrix(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str | None,
    major: int,
    minor: int,
    torch_version: str,
    windows: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)

    def properties(_device: torch.device | str | int | None) -> SimpleNamespace:
        return SimpleNamespace(major=major, minor=minor)

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        properties,
    )
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(quant_linear_mod.sys, "platform", "win32" if windows else "linux")
    device = torch.device("cuda:0")
    assert default_fp8_matmul(device) is False  # upstream --fast is opt-in
    assert default_fp8_matmul(device, requested=True) is expected


def test_fp8_support_override_matches_upstream_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert supports_fp8_matmul(torch.device("cpu"), support_override=True)
    assert default_fp8_matmul(
        torch.device("cpu"),
        requested=True,
        support_override=True,
    )


def test_kitchen_unavailable_uses_eager_scaled_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quant_linear_mod, "_kitchen_probed", False)
    monkeypatch.setattr(quant_linear_mod, "_kitchen_scaled_mm_v2", None)

    def lacks_capability(_name: str) -> SimpleNamespace:
        def has_scaled_mm_v2() -> bool:
            return False

        def scaled_mm_v2(*_args: object, **_kwargs: object) -> None:
            return None

        return SimpleNamespace(
            has_scaled_mm_v2=has_scaled_mm_v2,
            scaled_mm_v2=scaled_mm_v2,
        )

    monkeypatch.setattr(
        quant_linear_mod.importlib,
        "import_module",
        lacks_capability,
    )
    calls: list[dict[str, object]] = []

    def eager(input: torch.Tensor, weight: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append({"input": input, "weight": weight, **kwargs})
        return torch.zeros((input.shape[0], weight.shape[1]), dtype=torch.float32)

    monkeypatch.setattr(torch, "_scaled_mm", eager)
    layer = make_layer(bias=False, scale=0.5)
    layer.bind_fp8_matmul(True)
    assert layer._fp8_matmul_backend == "torch"  # pyright: ignore[reportPrivateUsage]
    with torch.no_grad():
        result = layer(torch.randn(2, 4))
    assert result.shape == (2, 3)
    assert len(calls) == 1
    assert calls[0]["scale_a"] is layer.input_scale
    assert calls[0]["scale_b"] is layer.weight_scale


def test_fp8_matmul_uses_kitchen_input_quantizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantize_calls: list[tuple[torch.Tensor, torch.Tensor, torch.dtype]] = []

    def quantize(input: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        quantize_calls.append((input, scale, dtype))
        return input.to(dtype)

    def eager(input: torch.Tensor, weight: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return torch.zeros((input.shape[0], weight.shape[1]), dtype=torch.float32)

    monkeypatch.setattr(quant_linear_mod, "_kitchen_quantize_probed", True)
    monkeypatch.setattr(quant_linear_mod, "_kitchen_quantize_per_tensor_fp8", quantize)
    monkeypatch.setattr(torch, "_scaled_mm", eager)
    layer = make_layer(bias=False)
    layer.bind_fp8_matmul(True)
    input = torch.randn(2, 4)

    with torch.no_grad():
        result = layer(input)

    assert result.shape == (2, 3)
    assert quantize_calls == [(input, layer.input_scale, torch.float8_e4m3fn)]


def test_kitchen_probe_failure_is_an_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quant_linear_mod, "_kitchen_probed", False)
    monkeypatch.setattr(quant_linear_mod, "_kitchen_scaled_mm_v2", None)

    def unavailable(_name: str) -> object:
        raise RuntimeError("broken optional kitchen install")

    monkeypatch.setattr(quant_linear_mod.importlib, "import_module", unavailable)
    layer = make_layer(bias=False)
    layer.bind_fp8_matmul(True)
    assert layer._fp8_matmul_backend == "torch"  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------- storage bridge


def test_stored_load_stored_round_trip() -> None:
    layer = make_layer(scale=3.0)
    stored = layer.stored()
    assert stored.orig_dtype == torch.float32
    assert torch.equal(stored.scale, layer.weight_scale)
    replacement = Fp8ScaledWeight(
        torch.randn(3, 4).to(E4M3),
        torch.tensor(0.5, dtype=torch.float32),
        torch.float32,
    )
    layer.load_stored(replacement)
    assert torch.equal(bits(layer.weight), bits(replacement.qdata))
    assert layer.weight_scale.item() == 0.5


def test_load_stored_refuses_dtype_and_shape_mismatch() -> None:
    layer = make_layer()
    with pytest.raises(ValueError, match="float8_e5m2"):
        layer.load_stored(
            Fp8ScaledWeight(
                torch.randn(3, 4).to(E5M2),
                torch.ones((), dtype=torch.float32),
                torch.float32,
            )
        )
    with pytest.raises(ValueError, match="shape"):
        layer.load_stored(
            Fp8ScaledWeight(
                torch.randn(4, 4).to(E4M3),
                torch.ones((), dtype=torch.float32),
                torch.float32,
            )
        )


def test_set_weight_requantizes_like_the_reference() -> None:
    layer = make_layer()
    fresh = torch.randn(3, 4) * 7.0
    expected = quantize_fp8_scaled(fresh.clone(), E4M3, seed=0)
    layer.set_weight(fresh.clone(), seed=0)
    assert torch.equal(bits(layer.weight), bits(expected.qdata))
    assert torch.equal(layer.weight_scale, expected.scale)


def test_set_weight_seeded_rounding_is_deterministic() -> None:
    a = make_layer()
    b = make_layer()
    fresh = torch.randn(3, 4)
    a.set_weight(fresh.clone(), seed=41)
    b.set_weight(fresh.clone(), seed=41)
    assert torch.equal(bits(a.weight), bits(b.weight))
    assert torch.equal(a.weight_scale, b.weight_scale)


# ------------------------------------------------- ecosystem behavior


def test_mixed_model_composes_with_ordinary_linears() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        make_layer(4, 3),
    )
    state = model.state_dict()
    assert state["0.weight"].dtype == torch.float32
    assert state["1.weight"].dtype == E4M3
    out = model(torch.randn(2, 4))
    assert out.shape == (2, 3)
    assert not out.isnan().any()


def test_autograd_flows_to_input_through_dequant() -> None:
    layer = make_layer()
    x = torch.randn(2, 4, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()
    assert layer.weight.grad is None


def test_compile_fullgraph_dequant_parity() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable on this build")
    layer = make_layer()
    x = torch.randn(2, 4)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    assert torch.equal(compiled(x), layer(x))


@pytest.mark.parametrize("input_act", ["swiglu", "gelu_tanh"])
def test_linear_input_act_matches_inference_values_and_supports_backward(
    input_act: Literal["gelu_tanh", "swiglu"],
) -> None:
    torch.manual_seed(7)
    linear = torch.nn.Linear(4, 3, bias=False)
    input_width = 8 if input_act == "swiglu" else 4
    input = torch.randn(2, input_width)

    with torch.no_grad():
        expected = linear_input_act(linear, input.clone(), input_act)

    source = input.clone().requires_grad_(True)
    output = linear_input_act(linear, source, input_act)
    assert torch.equal(output, expected)

    output.square().mean().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert bool(torch.count_nonzero(source.grad))
    assert linear.weight.grad is not None
    assert bool(torch.count_nonzero(linear.weight.grad))


def test_worker_thread_forward_parity() -> None:
    layer = make_layer()
    x = torch.randn(8, 4)
    expected = layer(x)

    def run(_: int) -> torch.Tensor:
        return layer(x)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(16)))
    for result in results:
        assert torch.equal(result, expected)
