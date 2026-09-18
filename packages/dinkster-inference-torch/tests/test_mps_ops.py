"""Apple Silicon (MPS) parity for the kitchen eager operations Dinkster
dispatches during inference.

Everything here is capability-gated: without an MPS device the whole
module skips, so the CPU-only `.venv-torch` gate and the CUDA suite are
unaffected. No model artifacts are needed; inputs are small seeded
tensors.

Tolerances are pinned to measured behavior on real hardware (M4 Mac
mini, torch 2.13.0, comfy-kitchen 0.2.31 eager backend): apply_rope,
adaln, rms_adaln, ConvRot INT8 dequantization, and the Int8Linear
dequant route are exact against the same computation on CPU (rtol=0,
atol=0). The fused rms_rope variants normalize in fp16 and then rotate
in the rotation table's dtype. These tests use bounded float32 cos/sin
tables, the configuration the z_image and cosmos_predict2 call sites
pass; MiniMax H3 instead casts its table to the runtime compute dtype
(bfloat16 or float32), and the bfloat16-table pairing is not asserted
here. With float32 rotation the device-dependent steps are the fp16
rms_norm rounding and multiply-add contraction inside the rotation,
making the drift absolute at the unit-rms working scale rather than
relative to each output element (a near-cancelling rotation of
correctly rounded inputs can leave a tiny output with large relative
error, which is why no rtol is asserted). Measured max abs drift is
2**-9 for the pinned seeds and 2**-8 across a 100-seed sweep of the
same shapes; the assertions use rtol=0, atol=2**-8 - twice the
pinned-seed drift, equal to the sweep maximum. Per the numerical parity
discipline a mismatch beyond that is a finding to investigate, never a
tolerance to widen.
"""

from __future__ import annotations

import pytest
import torch

mps_available = torch.backends.mps.is_available()

pytestmark = pytest.mark.skipif(not mps_available, reason="MPS device required (see README)")

RMS_ROPE_ATOL = 2**-8  # see module docstring


def _seeded(shape: tuple[int, ...], seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)


def test_apply_rope_matches_torch_reference_and_cpu_exactly() -> None:
    import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]
    from dinkster_inference_torch.flux import (
        _apply_rope_torch,  # pyright: ignore[reportPrivateUsage]
        rope,
    )

    xq = _seeded((2, 4, 8, 16), 101, torch.float16).to("mps")
    xk = _seeded((2, 4, 8, 16), 102, torch.float16).to("mps")
    positions = torch.arange(8, device="mps", dtype=torch.float32)[None, :]
    freqs = rope(positions, 16, 10000)[:, None]
    assert freqs.device.type == "mps"

    with torch.no_grad():
        got_q, got_k = comfy_kitchen.apply_rope(xq, xk, freqs)
    want_q, want_k = _apply_rope_torch(xq, xk, freqs)
    torch.testing.assert_close(got_q, want_q, rtol=0, atol=0)
    torch.testing.assert_close(got_k, want_k, rtol=0, atol=0)

    cpu_q, cpu_k = comfy_kitchen.apply_rope(xq.cpu(), xk.cpu(), freqs.cpu())
    torch.testing.assert_close(got_q.cpu(), cpu_q, rtol=0, atol=0)
    torch.testing.assert_close(got_k.cpu(), cpu_k, rtol=0, atol=0)


def _rotation_table(dim: int) -> torch.Tensor:
    """float32 [1, 1, 8, dim/2, 2, 2] cos/sin rotation blocks over 8
    positions, the shape and dtype of the rope tables the z_image and
    cosmos_predict2 fused rms_rope call sites pass."""
    from dinkster_inference_torch.flux import rope

    positions = torch.arange(8, dtype=torch.float32)[None, :]
    return rope(positions, dim, 10000)[:, None]


def _rms_scale(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (0.5 + torch.rand((16,), generator=generator, dtype=torch.float32)).to(torch.float16)


def test_rms_rope_matches_cpu_within_fp16_rounding_tolerance() -> None:
    import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]

    q = _seeded((2, 4, 8, 16), 103, torch.float16)
    k = _seeded((2, 4, 8, 16), 104, torch.float16)
    freqs = _rotation_table(16)
    q_scale = _rms_scale(106)
    k_scale = _rms_scale(107)

    with torch.no_grad():
        got_q, got_k = comfy_kitchen.rms_rope(
            q.to("mps"),
            k.to("mps"),
            freqs.to("mps"),
            q_scale.to("mps"),
            k_scale.to("mps"),
            epsilon=1e-6,
        )
        want_q, want_k = comfy_kitchen.rms_rope(q, k, freqs, q_scale, k_scale, epsilon=1e-6)
    assert got_q.dtype == want_q.dtype == torch.float16
    torch.testing.assert_close(got_q.cpu(), want_q, rtol=0, atol=RMS_ROPE_ATOL)
    torch.testing.assert_close(got_k.cpu(), want_k, rtol=0, atol=RMS_ROPE_ATOL)


def test_rms_rope_split_half_matches_cpu_within_fp16_rounding_tolerance() -> None:
    import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]

    q = _seeded((2, 4, 8, 16), 108, torch.float16)
    k = _seeded((2, 4, 8, 16), 109, torch.float16)
    freqs = _rotation_table(8)
    q_scale = _rms_scale(111)
    k_scale = _rms_scale(112)

    def run(device: str) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return comfy_kitchen.rms_rope_split_half_(
                q.clone().to(device),
                k.clone().to(device),
                freqs.to(device),
                q_scale.to(device),
                k_scale.to(device),
                epsilon=1e-6,
                rot_dim=8,
            )

    got_q, got_k = run("mps")
    want_q, want_k = run("cpu")
    torch.testing.assert_close(got_q.cpu(), want_q, rtol=0, atol=RMS_ROPE_ATOL)
    torch.testing.assert_close(got_k.cpu(), want_k, rtol=0, atol=RMS_ROPE_ATOL)


@pytest.mark.parametrize("operation", ["adaln", "rms_adaln"])
def test_adaln_matches_cpu_exactly(operation: str) -> None:
    import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]

    kernel = getattr(comfy_kitchen, operation)
    x = _seeded((2, 5, 16), 113, torch.float16)
    scale = _seeded((2, 1, 16), 114, torch.float16)
    shift = _seeded((2, 1, 16), 115, torch.float16)

    with torch.no_grad():
        got = kernel(x.to("mps"), scale.to("mps"), shift.to("mps"))
        want = kernel(x, scale, shift)
    assert got.dtype == torch.float16
    torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


def test_convrot_int8_dequantization_matches_cpu_exactly() -> None:
    pytest.importorskip("comfy_kitchen")
    generator = torch.Generator().manual_seed(116)
    q = torch.randint(-128, 128, (512, 512), generator=generator, dtype=torch.int8)
    scale = torch.rand((512, 1), generator=generator, dtype=torch.float32) / 100

    got = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(q.to("mps"), scale.to("mps"), 256)
    want = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(q, scale, 256)
    torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


@pytest.mark.parametrize("convrot", [False, True])
def test_int8_linear_executes_on_mps_through_the_dequant_route(convrot: bool) -> None:
    pytest.importorskip("comfy_kitchen")
    from dinkster_inference_torch import Int8Linear
    from dinkster_inference_torch.quant_linear import (
        _dequantize_int8,  # pyright: ignore[reportPrivateUsage]
    )

    generator = torch.Generator().manual_seed(117)
    input = torch.randn((3, 256), generator=generator, dtype=torch.float32)
    weight = torch.randint(-100, 101, (5, 256), generator=generator, dtype=torch.int8)
    scale_shape = (5, 1) if convrot else ()
    scale = torch.rand(scale_shape, generator=generator, dtype=torch.float32) / 100
    bias = torch.randn(5, generator=generator, dtype=torch.float32)
    layer = Int8Linear(
        256,
        5,
        bias=True,
        compute_dtype=torch.float32,
        convrot=convrot,
        convrot_groupsize=256,
    )
    layer.load_state_dict({"weight": weight, "weight_scale": scale, "bias": bias}, assign=True)
    layer.to("mps")
    assert not layer.full_precision_matmul

    with torch.no_grad():
        got = layer(input.to("mps"))
        expected_weight = _dequantize_int8(
            layer.weight,
            layer.weight_scale,
            dtype=torch.float32,
            convrot=convrot,
            convrot_groupsize=256,
        )
        want = torch.nn.functional.linear(input.to("mps"), expected_weight, layer.bias)
    assert got.device.type == "mps"
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "policy", ["auto", "sdpa", "flash", "xformers", "sage", "sage3", "sol", "comfy_kitchen_int8"]
)
def test_authenticated_attention_fallback_matches_mps_sdpa(
    policy: str, caplog: pytest.LogCaptureFixture
) -> None:
    from dinkster_inference_torch.attention import (
        builtin_sdpa_kernel,
        discover_attention_route_token,
        resolve_role_attention,
    )
    from dinkster_protocol.attention import ATTENTION_ROLES, validate_attention_policy

    requested = validate_attention_policy(policy)
    token = discover_attention_route_token(requested)
    assert token.device_kind == "mps"
    assert token.version == (1 if policy in ("auto", "sdpa") else 3)
    q, k, v = (_seeded((1, 2, 4, 16), seed, torch.float32).to("mps") for seed in (201, 202, 203))
    expected = builtin_sdpa_kernel()(q, k, v)
    for role in ATTENTION_ROLES:
        selection = resolve_role_attention(role, requested, token)
        assert selection.status.authenticated
        assert selection.status.requested_policy == policy
        assert selection.status.primary == "sdpa"
        assert selection.status.fallback is None
        assert torch.equal(selection.kernel(q, k, v), expected)
    if policy not in ("auto", "sdpa"):
        assert "using SDPA" in caplog.text


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fp8_matmul_fallback_executes_on_mps(
    scaled: bool, partial: bool, dtype: torch.dtype, caplog: pytest.LogCaptureFixture
) -> None:
    from dinkster_inference_torch import (
        CastOperations,
        DeviceMemory,
        Fp8Linear,
        MemoryPolicy,
        ResidencyManager,
        enroll_component,
    )
    from dinkster_inference_torch.operations import bind_fp8_matmul_layer

    weight = _seeded((16, 32), 204, torch.float32).to(torch.float8_e4m3fn)
    bias = _seeded((16,), 205, dtype)
    state = {"weight": weight, "bias": bias}
    if scaled:
        layer = Fp8Linear(32, 16, bias=True, compute_dtype=dtype)
        state.update(weight_scale=torch.tensor(0.625), input_scale=torch.tensor(1.25))
    else:
        layer = CastOperations(dtype).linear(32, 16)
    layer.load_state_dict(state, assign=True)
    if isinstance(layer, Fp8Linear):
        layer.bind_fp8_matmul(True)
    else:
        assert bind_fp8_matmul_layer(layer, True)
    model = torch.nn.Sequential(layer)
    mechanism = enroll_component(model, load_device="mps", offload_device="cpu")
    assert "using CPU dequantization" in caplog.text
    if partial:
        manager = ResidencyManager(
            policy=MemoryPolicy(inference_reserve=100, physical_headroom=0),
            free_memory=lambda _device: DeviceMemory(50, 0),
        )
        manager.load([mechanism])
        assert mechanism.loaded_bytes() == 0
        assert "attempting memory-budgeted loading" in caplog.text
    else:
        mechanism.partially_load(None)
        assert mechanism.offloaded_bytes() == 0
    input = _seeded((3, 32), 206, dtype).to("mps")
    expected_weight = weight.to(dtype)
    if scaled:
        expected_weight *= state["weight_scale"].to(dtype)
    with torch.no_grad():
        expected = torch.nn.functional.linear(input, expected_weight.to("mps"), bias.to("mps"))
        actual = model(input)
    assert actual.device.type == "mps"
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    mechanism.unload()
