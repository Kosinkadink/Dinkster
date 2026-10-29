"""Reduced CPU proofs for the unregistered MiniMax H3 audio codec source."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import gc
import hashlib
import math
import weakref
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from dinkster_inference_torch import (
    INITLESS,
    Activation1d,
    AMPBlock1,
    BigVGAN,
    CastOperations,
    DACEncoder,
    DownSample1d,
    MiniMaxH3AudioVAE,
    Operations,
    ResidencyRouted,
    SnakeBeta,
    UpSample1d,
    enroll_component,
    kaiser_sinc_filter1d,
    snake,
)
from dinkster_inference_torch.minimax_h3_audio import CausalAttention
from dinkster_inference_torch.residency import WeightLease
from dinkster_inference_torch.residency_timing import PartialResidencyTiming


def _fill(module: torch.nn.Module, value: float = 0.05) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _tiny_codec() -> MiniMaxH3AudioVAE:
    class Encoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.avg_pool1d(x, 800)

    class MeanProjection(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.repeat(1, 32, 1)

    class Decoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1].repeat_interleave(800, dim=-1)

    with torch.device("meta"):
        codec = MiniMaxH3AudioVAE()
    codec.encoder = Encoder()
    codec.pre_block = torch.nn.Identity()
    codec.mean_proj = MeanProjection()
    codec.dec_in_proj = torch.nn.Identity()
    codec.decoder = Decoder()
    codec.latents_mean = torch.zeros(32)
    codec.latents_std = torch.ones(32)
    return codec


REDUCED_STATE_COUNT = 917
REDUCED_STATE_DIGEST = "ad4180e18b2e21ed9a78175cec5fcc83665780dbab5de27bec48857bfee2aed1"


def _filled_reduced_codec(
    operations: Operations = INITLESS,
    *,
    storage_dtype: torch.dtype = torch.float32,
) -> MiniMaxH3AudioVAE:
    codec = MiniMaxH3AudioVAE(
        encoder_dim=2,
        latent_dim=8,
        decoder_dim=128,
        operations=operations,
    )
    generator = torch.Generator().manual_seed(20260812)
    state = {}
    for key, value in codec.state_dict().items():
        if key == "latents_std" or key.endswith(
            ("norm.weight", "norm1.weight", "norm2.weight", "norm3.weight")
        ):
            filled = torch.ones_like(value, dtype=storage_dtype)
        elif key.endswith(("alpha", "beta")):
            filled = torch.full_like(value, 0.01, dtype=storage_dtype)
        else:
            filled = torch.randn(value.shape, generator=generator, dtype=storage_dtype) * 0.02
        state[key] = filled
    codec.load_state_dict(state, strict=True, assign=True)
    return codec


def _prefetch_requests(codec: MiniMaxH3AudioVAE) -> dict[str, torch.dtype | None]:
    requests: dict[str, torch.dtype | None] = {}
    for module in codec.modules():
        if not isinstance(module, ResidencyRouted):
            continue
        prefetch = module.residency_prefetch()
        if prefetch is None:
            continue
        for key, dtype in prefetch[1]:
            assert key not in requests
            requests[key] = dtype
    return requests


def test_snake_is_exact_and_does_not_mutate_input() -> None:
    x = torch.tensor([[[-1.0, 0.0, 2.0]]])
    original = x.clone()
    alpha = torch.tensor([[[0.75]]])
    beta = torch.tensor([[[1.25]]])
    expected = x + torch.sin(alpha * x).square() / (beta + 1e-9)
    torch.testing.assert_close(snake(x, alpha, beta), expected)
    assert torch.equal(x, original)


def test_kaiser_filter_is_normalized_symmetric_and_validated() -> None:
    filter_ = kaiser_sinc_filter1d(0.25, 0.3, 12)
    assert filter_.shape == (1, 1, 12)
    torch.testing.assert_close(filter_.sum(), torch.tensor(1.0))
    torch.testing.assert_close(filter_, filter_.flip(-1))
    with pytest.raises(ValueError, match="cutoff"):
        kaiser_sinc_filter1d(0.0, 0.3, 12)
    with pytest.raises(ValueError, match="kernel_size"):
        kaiser_sinc_filter1d(0.25, 0.3, 1)


def test_alias_free_resamplers_have_exact_rate_and_constant_response() -> None:
    constant = torch.ones(1, 2, 17)
    up = UpSample1d(ratio=2, kernel_size=12)(constant)
    down = DownSample1d(ratio=2, kernel_size=12)(up)
    assert up.shape == (1, 2, 34)
    assert down.shape == constant.shape
    torch.testing.assert_close(up, torch.ones_like(up), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(down, constant, rtol=1e-5, atol=1e-5)


def test_activation_and_amp_block_preserve_odd_length_and_gradients() -> None:
    activation = Activation1d(SnakeBeta(2))
    block = AMPBlock1(2, kernel_size=3, dilation=(1,))
    _fill(activation)
    _fill(block)
    x = torch.randn(1, 2, 9, requires_grad=True)
    assert activation(x).shape == x.shape
    output = block(x)
    assert output.shape == x.shape
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dac_encoder_reduced_shape_uses_pinned_stride_geometry() -> None:
    encoder = DACEncoder(d_model=2, strides=(2, 2), d_latent=8)
    _fill(encoder)
    assert encoder(torch.randn(3, 1, 8)).shape == (3, 8, 2)
    assert encoder(torch.randn(3, 1, 9)).shape == (3, 8, 2)


def test_bigvgan_reduced_decoder_multiplies_frames_and_clamps() -> None:
    decoder = BigVGAN(
        num_mels=8,
        upsample_initial_channel=8,
        upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
    )
    _fill(decoder, 0.2)
    output = decoder(torch.full((2, 8, 3), 10.0))
    assert output.shape == (2, 1, 12)
    assert output.abs().max().item() <= 1.0


def test_codec_exact_stereo_contract_rate_and_frame_math() -> None:
    codec = _tiny_codec()
    assert codec.sample_rate == codec.output_sample_rate == 32_000
    assert codec.samples_per_latent == 800
    assert codec.latents_per_second == 40

    waveform = torch.randn(2, 2, 1_601)
    latent = codec.encode(waveform, sample_rate=32_000)
    assert latent.shape == (2, 32, 2, math.ceil(1_601 / 800))
    assert codec.encode_output_shape(waveform.shape) == tuple(latent.shape)
    decoded = codec.decode(latent)
    assert decoded.shape == (2, 2, latent.shape[-1] * 800)


def test_codec_stereo_batch_order_and_normalization_are_exact() -> None:
    class Encoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.avg_pool1d(x, 800)

    class MeanProjection(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.repeat(1, 32, 1)

    class Decoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1].repeat_interleave(800, dim=-1)

    codec = _tiny_codec()
    codec.encoder = Encoder()
    codec.pre_block = torch.nn.Identity()
    codec.mean_proj = MeanProjection()
    codec.dec_in_proj = torch.nn.Identity()
    codec.decoder = Decoder()
    with torch.no_grad():
        codec.latents_mean.fill_(2.0)
        codec.latents_std.fill_(4.0)

    waveform = torch.stack((torch.full((1_600,), 6.0), torch.full((1_600,), 10.0))).unsqueeze(0)
    latent = codec.encode(waveform, sample_rate=32_000)
    assert torch.equal(latent[:, :, 0], torch.ones(1, 32, 2))
    assert torch.equal(latent[:, :, 1], torch.full((1, 32, 2), 2.0))
    assert torch.equal(codec.decode(latent), waveform)


def test_codec_default_configuration_is_exact_h3_profile() -> None:
    with torch.device("meta"):
        codec = MiniMaxH3AudioVAE()
    assert codec.sample_rate == codec.output_sample_rate == 32_000
    assert codec.samples_per_latent == 800
    assert codec.latents_per_second == 40
    assert codec.vae_latent_channels == 32


def test_codec_default_state_dict_layout_is_pinned() -> None:
    with torch.device("meta"):
        codec = MiniMaxH3AudioVAE()
    entries = tuple((name, tuple(tensor.shape)) for name, tensor in codec.state_dict().items())
    digest = hashlib.sha256(repr(entries).encode("ascii")).hexdigest()
    assert len(entries) == 917
    assert digest == "d0608cb7a20deb5046b7d077700d7f08ead745bf67a18c7fad29d088b0718a39"


def test_reduced_state_layout_is_exact_and_whole_root_enrolls() -> None:
    codec = _filled_reduced_codec()
    entries = tuple((name, tuple(tensor.shape)) for name, tensor in codec.state_dict().items())
    assert len(entries) == REDUCED_STATE_COUNT
    assert hashlib.sha256(repr(entries).encode("ascii")).hexdigest() == REDUCED_STATE_DIGEST
    mechanism = enroll_component(codec, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(0)
    assert mechanism.loaded_bytes() == 0
    with pytest.raises(RuntimeError, match="already enrolled"):
        enroll_component(codec, load_device="cpu", offload_device="cpu")


@pytest.mark.parametrize(
    ("operations", "storage_dtype"),
    ((INITLESS, torch.float32), (CastOperations(torch.float32), torch.float16)),
    ids=("initless", "cast-at-use"),
)
def test_whole_root_prefetch_and_offloaded_encode_decode_are_exact(
    operations: Operations,
    storage_dtype: torch.dtype,
) -> None:
    codec = _filled_reduced_codec(operations, storage_dtype=storage_dtype)
    assert {value.dtype for value in codec.state_dict().values()} == {storage_dtype}
    mechanism = enroll_component(codec, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    waveform = torch.linspace(-0.1, 0.1, 1_600).reshape(1, 2, 800)
    latent = torch.linspace(-0.25, 0.25, 64).reshape(1, 32, 2, 1)
    resident_encode = codec.encode(waveform.clone())
    resident_decode = codec.decode(latent.clone())
    assert _prefetch_requests(codec) == {}

    mechanism.unload()
    expected_dtype = torch.float32 if isinstance(operations, CastOperations) else storage_dtype
    expected_prefetch = {key: expected_dtype for key in codec.state_dict()}
    assert _prefetch_requests(codec) == expected_prefetch
    requested: list[tuple[str, torch.dtype]] = []
    original_lease = mechanism.lease

    def tracked_lease(unit: str) -> AbstractContextManager[WeightLease]:
        inner = original_lease(unit)

        @contextmanager
        def bracket() -> Generator[WeightLease]:
            with inner as lease:

                class TrackingLease:
                    def get(self, key: str, *, dtype: torch.dtype) -> torch.Tensor:
                        requested.append((key, dtype))
                        return lease.get(key, dtype=dtype)

                    def get_stored(self, key: str) -> object:
                        return lease.get_stored(key)

                    def timing_collector(self) -> PartialResidencyTiming | None:
                        return lease.timing_collector()

                yield cast(WeightLease, TrackingLease())

        return bracket()

    object.__setattr__(mechanism, "lease", tracked_lease)
    offloaded_encode = codec.encode(waveform.clone())
    offloaded_decode = codec.decode(latent.clone())
    assert torch.equal(offloaded_encode, resident_encode)
    assert torch.equal(offloaded_decode, resident_decode)
    unused = {"logs_proj.weight", "logs_proj.bias"}
    assert {key for key, _dtype in requested} == set(expected_prefetch) - unused
    assert all(expected_prefetch[key] == dtype for key, dtype in requested)
    assert all(dtype == expected_dtype for _key, dtype in requested)
    assert mechanism.loaded_bytes() == 0


def test_default_direct_state_preserves_input_dtype() -> None:
    signal = torch.linspace(-0.2, 0.2, 16, dtype=torch.float64).reshape(1, 1, 16)
    upsample = UpSample1d()
    activation = SnakeBeta(1)
    with torch.no_grad():
        activation.alpha.fill_(0.25)
        activation.beta.fill_(0.5)
    assert upsample(signal).dtype == torch.float64
    assert activation(signal).dtype == torch.float64


def test_offloaded_failure_closes_every_lease_and_allows_immediate_reuse() -> None:
    codec = _filled_reduced_codec()
    mechanism = enroll_component(codec, load_device="cpu", offload_device="cpu")
    mechanism.unload()
    original_lease = mechanism.lease
    opened: list[WeightLease] = []

    def tracked_lease(unit: str) -> AbstractContextManager[WeightLease]:
        inner = original_lease(unit)

        @contextmanager
        def bracket() -> Generator[WeightLease]:
            with inner as lease:
                opened.append(lease)
                yield lease

        return bracket()

    object.__setattr__(mechanism, "lease", tracked_lease)
    attention = cast(CausalAttention, codec.pre_block.attn)  # type: ignore[attr-defined]
    original = attention._attention_kernel

    def fail_attention(*_args: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected audio attention failure")

    object.__setattr__(attention, "_attention_kernel", fail_attention)
    waveform = torch.zeros(1, 2, 800)
    with pytest.raises(RuntimeError, match="injected audio attention failure"):
        codec.encode(waveform.clone())
    assert mechanism.loaded_bytes() == 0
    assert opened
    for lease in opened:
        with pytest.raises(RuntimeError, match="closed"):
            lease.get("unused", dtype=torch.float32)

    object.__setattr__(attention, "_attention_kernel", original)
    opened.clear()
    assert torch.isfinite(codec.encode(waveform.clone())).all()
    assert mechanism.loaded_bytes() == 0
    assert opened
    for lease in opened:
        with pytest.raises(RuntimeError, match="closed"):
            lease.get("unused", dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_minimax_h3_audio_residency_cuda_resident_offloaded_and_cleanup() -> None:
    device = torch.device("cuda:0")

    def run() -> None:
        codec = _filled_reduced_codec(CastOperations(torch.float32), storage_dtype=torch.float16)
        mechanism = enroll_component(codec, load_device=device, offload_device="cpu")
        waveform = torch.linspace(-0.1, 0.1, 1_600, device=device).reshape(1, 2, 800)
        latent = torch.linspace(-0.25, 0.25, 64, device=device).reshape(1, 32, 2, 1)
        mechanism.partially_load(None)
        resident_encode = codec.encode(waveform.clone())
        resident_decode = codec.decode(latent.clone())
        mechanism.unload()
        offloaded_encode = codec.encode(waveform.clone())
        offloaded_decode = codec.decode(latent.clone())
        assert torch.equal(offloaded_encode, resident_encode)
        assert torch.equal(offloaded_decode, resident_decode)
        assert mechanism.loaded_bytes() == 0

    run()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    run()
    gc.collect()
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated(device) == baseline


def test_codec_refuses_non_h3_frame_geometry() -> None:
    with pytest.raises(ValueError, match="encoder rates"):
        MiniMaxH3AudioVAE(encoder_rates=(2, 2))


def test_causal_attention_binds_rank4_qkv_and_causal_kernel() -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool]] = []
    values: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        assert mask is None
        assert scale is None
        assert enable_gqa is False
        calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), causal))
        values.append((q.clone(), k.clone(), v.clone()))
        return q + 2 * k + 3 * v

    attention = CausalAttention(4, 2, 2, attention_kernel=kernel)
    with torch.no_grad():
        attention.qkv.weight.copy_(torch.arange(48).reshape(12, 4).div(40))
        attention.q_bias.copy_(torch.tensor((0.1, 0.2, 0.3, 0.4)))
        attention.zero_k_bias.zero_()
        attention.v_bias.copy_(torch.tensor((-0.1, -0.2, -0.3, -0.4)))
        attention.proj.weight.copy_(torch.tensor(((0.5, -0.25), (0.75, 0.125))))
        attention.proj.bias.copy_(torch.tensor((0.2, -0.3)))
    input_ = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).div(10)
    output = attention(input_)
    assert calls == [((2, 2, 3, 2), (2, 2, 3, 2), (2, 2, 3, 2), True)]
    fused = F.linear(
        input_,
        attention.qkv.weight,
        torch.cat((attention.q_bias, attention.zero_k_bias, attention.v_bias)),
    )
    expected_qkv = fused.reshape(2, 3, 3, 2, 2).permute(2, 0, 3, 1, 4).unbind(0)
    for actual, expected in zip(values[0], expected_qkv, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        output,
        torch.tensor(
            (
                ((0.6825, 1.5488), (2.0525, 7.3038), (3.4225, 13.0587)),
                ((4.7925, 18.8137), (6.1625, 24.5687), (7.5325, 30.3238)),
            )
        ),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    ("operations", "storage_dtype"),
    ((INITLESS, torch.float32), (CastOperations(torch.float32), torch.float16)),
    ids=("initless", "cast-at-use"),
)
def test_causal_attention_routed_bias_matches_fused_reference(
    operations: Operations,
    storage_dtype: torch.dtype,
) -> None:
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        captured.append((q.clone(), k.clone(), v.clone()))
        return q + 2 * k + 3 * v

    attention = CausalAttention(8, 4, 2, operations=operations, attention_kernel=kernel)
    generator = torch.Generator().manual_seed(20260812)
    state = {
        key: torch.randn(value.shape, generator=generator, dtype=storage_dtype)
        for key, value in attention.state_dict().items()
    }
    attention.load_state_dict(state, strict=True, assign=True)
    input_ = torch.randn((2, 5, 8), generator=generator)
    actual = attention(input_)

    compute = {key: value.to(torch.float32) for key, value in state.items()}
    fused = F.linear(
        input_,
        compute["qkv.weight"],
        torch.cat((compute["q_bias"], compute["zero_k_bias"], compute["v_bias"])),
    )
    query, key, value = fused.reshape(2, 5, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
    for routed, reference in zip(captured[0], (query, key, value), strict=True):
        assert torch.equal(routed, reference)
    attended = query + 2 * key + 3 * value
    pooled = F.adaptive_avg_pool1d(torch.mean(attended, dim=1), 4)
    expected = F.linear(pooled, compute["proj.weight"], compute["proj.bias"])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "waveform",
    (
        torch.empty(2, 9),
        torch.empty(0, 2, 9),
        torch.empty(1, 1, 9),
        torch.empty(1, 3, 9),
        torch.empty(1, 2, 0),
    ),
)
def test_encode_refuses_bad_geometry_before_encoder_work(waveform: torch.Tensor) -> None:
    codec = _tiny_codec()
    calls = 0

    def counted(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = codec.encoder.register_forward_pre_hook(counted)
    with pytest.raises(ValueError, match="waveform"):
        codec.encode(waveform, sample_rate=32_000)
    handle.remove()
    assert calls == 0


def test_encode_refuses_wrong_rate_before_encoder_work() -> None:
    codec = _tiny_codec()
    calls = 0

    def counted(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = codec.encoder.register_forward_pre_hook(counted)
    with pytest.raises(ValueError, match="32000"):
        codec.encode(torch.empty(1, 2, 8), sample_rate=44_100)
    handle.remove()
    assert calls == 0


@pytest.mark.parametrize(
    "latent",
    (
        torch.empty(1, 32, 4),
        torch.empty(0, 32, 2, 4),
        torch.empty(1, 31, 2, 4),
        torch.empty(1, 32, 1, 4),
        torch.empty(1, 32, 2, 0),
    ),
)
def test_decode_refuses_bad_geometry_before_decoder_work(latent: torch.Tensor) -> None:
    codec = _tiny_codec()
    calls = 0

    def counted(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = codec.decoder.register_forward_pre_hook(counted)
    with pytest.raises(ValueError, match="latent"):
        codec.decode(latent)
    handle.remove()
    assert calls == 0


@pytest.mark.parametrize("bad_std", (-1.0, 0.0, float("nan"), float("inf")))
def test_codec_refuses_malformed_normalization_before_model_work(bad_std: float) -> None:
    codec = _tiny_codec()
    with torch.no_grad():
        codec.latents_std[0] = bad_std
    with pytest.raises(ValueError, match="latents_std"):
        codec.encode(torch.empty(1, 2, 8), sample_rate=32_000)
    with pytest.raises(ValueError, match="latents_std"):
        codec.decode(torch.empty(1, 32, 2, 2))


def test_codec_refuses_nonfloating_normalization_before_model_work() -> None:
    codec = _tiny_codec()
    codec.latents_mean = torch.zeros(32, dtype=torch.int64)
    with pytest.raises(ValueError, match="latents_mean"):
        codec.encode(torch.empty(1, 2, 800), sample_rate=32_000)
    codec.latents_mean = torch.zeros(32)
    codec.latents_std = torch.ones(32, dtype=torch.int64)
    with pytest.raises(ValueError, match="latents_std"):
        codec.decode(torch.empty(1, 32, 2, 1))


@pytest.mark.parametrize("offloaded", (False, True))
def test_cast_codec_refuses_nonfloating_normalization_before_model_work(
    offloaded: bool,
) -> None:
    codec = _filled_reduced_codec(CastOperations(torch.float32), storage_dtype=torch.float16)
    codec.latents_mean = torch.zeros(32, dtype=torch.int64)
    mechanism = enroll_component(codec, load_device="cpu", offload_device="cpu")
    if offloaded:
        mechanism.unload()
    else:
        mechanism.partially_load(None)
    calls = 0

    def counted(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        nonlocal calls
        calls += 1

    handle = codec.encoder.register_forward_pre_hook(counted)
    with pytest.raises(ValueError, match="latents_mean"):
        codec.encode(torch.empty(1, 2, 800), sample_rate=32_000)
    handle.remove()
    assert calls == 0
    assert mechanism.loaded_bytes() == (0 if offloaded else mechanism.total_bytes())


def test_encode_right_padding_is_bounded_to_one_frame_remainder() -> None:
    codec = _tiny_codec()
    observed: list[int] = []

    def record(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        observed.append(args[0].shape[-1])

    handle = codec.encoder.register_forward_pre_hook(record)
    codec.encode(torch.randn(1, 2, 1_600), sample_rate=32_000)
    codec.encode(torch.randn(1, 2, 1_601), sample_rate=32_000)
    handle.remove()
    assert observed == [1_600, 2_400]


def test_temporary_stereo_batch_releases_after_downstream_exception() -> None:
    codec = _tiny_codec()
    temporary: weakref.ReferenceType[torch.Tensor] | None = None

    def capture_and_fail(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        nonlocal temporary
        temporary = weakref.ref(args[0])
        raise RuntimeError("stop")

    handle = codec.encoder.register_forward_pre_hook(capture_and_fail)
    with pytest.raises(RuntimeError, match="stop"):
        codec.encode(torch.randn(1, 2, 801), sample_rate=32_000)
    handle.remove()
    gc.collect()
    assert temporary is not None
    assert temporary() is None
