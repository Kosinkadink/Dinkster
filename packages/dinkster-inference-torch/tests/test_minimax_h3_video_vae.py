"""Reduced CPU proofs for the unregistered MiniMax H3 video VAE source."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import gc
import hashlib
import weakref
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import cast

import dinkster_inference_torch.minimax_h3_video_vae as vae_module
import pytest
import torch
import torch.nn.functional as F
from dinkster_inference_torch import (
    INITLESS,
    CastOperations,
    CausalConv3d,
    MiniMaxH3VideoVAE,
    MiniMaxH3VideoVAEConfig,
    ResidencyRouted,
    RotaryEmbeddingND,
    TemporalIsolatedGroupNorm,
    ViT3DDecoder,
    apply_rope_split_half,
    create_token_ids,
    enroll_component,
)
from dinkster_inference_torch.minimax_h3_video_vae import Downsample3D, TransformerBlock
from dinkster_inference_torch.residency import WeightLease
from dinkster_inference_torch.residency_timing import PartialResidencyTiming


def reduced_config(**overrides: object) -> MiniMaxH3VideoVAEConfig:
    fields: dict[str, object] = {
        "ch": 32,
        "embed_dim": 2,
        "z_channels": 2,
        "ch_mult": (1,),
        "num_res_blocks": 1,
        "space_down": (1,),
        "time_down": (1,),
        "clip_length": 3,
        "token_drop": 1,
        "tiling": False,
        "decoder_num_layers": 1,
        "decoder_heads": 2,
        "decoder_dim_head": 6,
        "decoder_rope_dim_ratio": 1.0,
        "decoder_num_register_tokens": 1,
    }
    fields.update(overrides)
    return MiniMaxH3VideoVAEConfig(**fields)  # type: ignore[arg-type]


REDUCED_STATE_LAYOUT = (
    ("latents_mean", (2,)),
    ("latents_std", (2,)),
    ("encoder.conv_in.weight", (32, 3, 3, 3, 3)),
    ("encoder.conv_in.bias", (32,)),
    ("encoder.down.0.block.0.norm1.weight", (32,)),
    ("encoder.down.0.block.0.norm1.bias", (32,)),
    ("encoder.down.0.block.0.norm2.weight", (32,)),
    ("encoder.down.0.block.0.norm2.bias", (32,)),
    ("encoder.down.0.block.0.conv1.weight", (32, 32, 3, 3, 3)),
    ("encoder.down.0.block.0.conv1.bias", (32,)),
    ("encoder.down.0.block.0.conv2.weight", (32, 32, 3, 3, 3)),
    ("encoder.down.0.block.0.conv2.bias", (32,)),
    ("encoder.norm_out.weight", (32,)),
    ("encoder.norm_out.bias", (32,)),
    ("encoder.conv_out.weight", (4, 32, 3, 3, 3)),
    ("encoder.conv_out.bias", (4,)),
    ("quant_conv.weight", (4, 4, 1, 1, 1)),
    ("quant_conv.bias", (4,)),
    ("post_quant_conv.weight", (2, 2, 1, 1, 1)),
    ("post_quant_conv.bias", (2,)),
    ("decoder.register_tokens", (1, 1, 12)),
    ("decoder.mask_token", (1, 1, 12)),
    ("decoder.x_embedder.weight", (12, 2)),
    ("decoder.x_embedder.bias", (12,)),
    ("decoder.transformer_blocks.0.scale1", (12,)),
    ("decoder.transformer_blocks.0.scale2", (12,)),
    ("decoder.transformer_blocks.0.norm1.weight", (12,)),
    ("decoder.transformer_blocks.0.attn.to_qkv.weight", (36, 12)),
    ("decoder.transformer_blocks.0.attn.to_qkv.bias", (36,)),
    ("decoder.transformer_blocks.0.attn.to_out.weight", (12, 12)),
    ("decoder.transformer_blocks.0.attn.to_out.bias", (12,)),
    ("decoder.transformer_blocks.0.norm2.weight", (12,)),
    ("decoder.transformer_blocks.0.ff.w1.weight", (96, 12)),
    ("decoder.transformer_blocks.0.ff.w1.bias", (96,)),
    ("decoder.transformer_blocks.0.ff.w2.weight", (12, 48)),
    ("decoder.transformer_blocks.0.ff.w2.bias", (12,)),
    ("decoder.norm_out.weight", (12,)),
    ("decoder.norm_out.bias", (12,)),
    ("decoder.proj_out.weight", (3, 12)),
    ("decoder.proj_out.bias", (3,)),
)


def test_reduced_state_layout_is_exact_and_whole_root_enrolls() -> None:
    vae = MiniMaxH3VideoVAE(reduced_config())
    assert tuple((key, tuple(value.shape)) for key, value in vae.state_dict().items()) == (
        REDUCED_STATE_LAYOUT
    )
    mechanism = enroll_component(vae, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(0)
    assert mechanism.loaded_bytes() == 0
    assert isinstance(vae, ResidencyRouted)
    with pytest.raises(RuntimeError, match="already enrolled"):
        enroll_component(vae, load_device="cpu", offload_device="cpu")


def test_default_state_layout_matches_prechange_golden() -> None:
    with torch.device("meta"):
        vae = MiniMaxH3VideoVAE()
    layout = "\n".join(
        f"{key}:{tuple(value.shape)}:{value.dtype}" for key, value in vae.state_dict().items()
    )
    assert len(vae.state_dict()) == 562
    assert hashlib.sha256(layout.encode("ascii")).hexdigest() == (
        "3ef3857c98539fb000ded8d8a5914c37393eef6fbda992e5d52e105024da68a1"
    )


def filled_reduced_vae(
    operations: object = INITLESS,
    *,
    storage_dtype: torch.dtype = torch.float32,
    config: MiniMaxH3VideoVAEConfig | None = None,
) -> MiniMaxH3VideoVAE:
    vae = MiniMaxH3VideoVAE(config or reduced_config(), operations=operations)  # type: ignore[arg-type]
    generator = torch.Generator().manual_seed(20260812)
    state = {}
    for key, value in vae.state_dict().items():
        if key == "latents_std" or key.endswith(("norm1.weight", "norm2.weight")):
            filled = torch.ones_like(value, dtype=storage_dtype)
        elif key.endswith(("scale1", "scale2")):
            filled = torch.full_like(value, 0.01, dtype=storage_dtype)
        else:
            filled = torch.randn(value.shape, generator=generator, dtype=storage_dtype) * 0.02
        state[key] = filled
    vae.load_state_dict(state, strict=True, assign=True)
    return vae


def _prefetch_requests(vae: MiniMaxH3VideoVAE) -> dict[str, torch.dtype | None]:
    requests: dict[str, torch.dtype | None] = {}
    for module in vae.modules():
        if not isinstance(module, ResidencyRouted):
            continue
        prefetch = module.residency_prefetch()
        if prefetch is None:
            continue
        for key, dtype in prefetch[1]:
            assert key not in requests
            requests[key] = dtype
    return requests


@pytest.mark.parametrize(
    ("operations", "storage_dtype"),
    ((INITLESS, torch.float32), (CastOperations(torch.float32), torch.float16)),
    ids=("initless", "cast-at-use"),
)
def test_whole_root_prefetch_and_offloaded_encode_decode_are_exact(
    operations: object, storage_dtype: torch.dtype
) -> None:
    vae = filled_reduced_vae(operations, storage_dtype=storage_dtype)
    assert tuple((key, tuple(value.shape)) for key, value in vae.state_dict().items()) == (
        REDUCED_STATE_LAYOUT
    )
    assert {value.dtype for value in vae.state_dict().values()} == {storage_dtype}
    assert "pixel_mean" not in vae.state_dict()
    assert "pixel_std" not in vae.state_dict()
    assert all("inv_freq" not in key for key in vae.state_dict())

    mechanism = enroll_component(vae, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    content = torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 1, 2, 2)
    latent = torch.linspace(-0.25, 0.25, 8).reshape(1, 2, 1, 2, 2)
    resident_encode = vae.encode(content.clone())
    resident_decode = vae.decode(latent.clone())
    assert _prefetch_requests(vae) == {}

    mechanism.unload()
    expected_dtype = torch.float32 if isinstance(operations, CastOperations) else storage_dtype
    expected_prefetch = {key: expected_dtype for key in vae.state_dict()}
    assert _prefetch_requests(vae) == expected_prefetch
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
    offloaded_encode = vae.encode(content.clone())
    offloaded_decode = vae.decode(latent.clone())
    assert torch.equal(offloaded_encode, resident_encode)
    assert torch.equal(offloaded_decode, resident_decode)
    executed_prefetch = {
        key: dtype for key, dtype in expected_prefetch.items() if key != "decoder.mask_token"
    }
    assert dict(requested) == executed_prefetch
    assert all(expected_prefetch[key] == dtype for key, dtype in requested)
    assert all(dtype == expected_dtype for _key, dtype in requested)
    assert mechanism.loaded_bytes() == 0


def test_offloaded_failure_closes_every_lease_and_allows_immediate_reuse() -> None:
    vae = filled_reduced_vae()
    mechanism = enroll_component(vae, load_device="cpu", offload_device="cpu")
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
    attention = cast(TransformerBlock, vae.decoder.transformer_blocks[0]).attn
    original = attention._attention_kernel

    def fail_attention(*_args: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected attention failure")

    object.__setattr__(attention, "_attention_kernel", fail_attention)
    with pytest.raises(RuntimeError, match="injected attention failure"):
        vae.decode(torch.zeros(1, 2, 1, 2, 2))
    assert mechanism.loaded_bytes() == 0
    assert opened
    for lease in opened:
        with pytest.raises(RuntimeError, match="closed"):
            lease.get("unused", dtype=torch.float32)

    object.__setattr__(attention, "_attention_kernel", original)
    opened.clear()
    assert torch.isfinite(vae.decode(torch.zeros(1, 2, 1, 2, 2))).all()
    assert mechanism.loaded_bytes() == 0
    assert opened
    for lease in opened:
        with pytest.raises(RuntimeError, match="closed"):
            lease.get("unused", dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_minimax_h3_video_residency_cuda_resident_offloaded_and_cleanup() -> None:
    device = torch.device("cuda:0")

    def run() -> None:
        vae = filled_reduced_vae(CastOperations(torch.float32), storage_dtype=torch.float16)
        mechanism = enroll_component(vae, load_device=device, offload_device="cpu")
        content = torch.linspace(-1.0, 1.0, 12, device=device).reshape(1, 3, 1, 2, 2)
        latent = torch.linspace(-0.25, 0.25, 8, device=device).reshape(1, 2, 1, 2, 2)
        mechanism.partially_load(None)
        resident_encode = vae.encode(content.clone())
        resident_decode = vae.decode(latent.clone())
        mechanism.unload()
        offloaded_encode = vae.encode(content.clone())
        offloaded_decode = vae.decode(latent.clone())
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


def test_default_config_pins_upstream_architecture_and_chunk_geometry() -> None:
    config = MiniMaxH3VideoVAEConfig()
    assert config.ch_mult == (1, 2, 2, 4, 4, 8)
    assert config.space_down == (2, 2, 2, 2, 1, 1)
    assert config.time_down == (1, 2, 2, 1, 1, 1)
    assert config.num_res_blocks == 2
    assert config.embed_dim == config.z_channels == 24
    assert config.clip_length == 17
    assert config.token_drop == 3
    assert config.tile_size == 256
    assert config.tile_overlap_min == 64
    assert config.decoder_num_layers == 36
    assert config.decoder_heads == 32
    assert config.decoder_dim_head == 64
    assert config.decoder_rope_dim_ratio == 0.75
    assert config.decoder_num_register_tokens == 4


def test_causal_conv_single_frame_matches_explicit_front_zero_padding() -> None:
    conv = CausalConv3d(1, 1, kernel_size=3, padding=1)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3))
        assert conv.bias is not None
        conv.bias.fill_(0.25)
    sample = torch.arange(9, dtype=torch.float32).reshape(1, 1, 1, 3, 3)
    spatial = torch.nn.functional.pad(sample, (1, 1, 1, 1, 0, 0), mode="reflect")
    explicit = torch.nn.functional.conv3d(
        torch.nn.functional.pad(spatial, (0, 0, 0, 0, 2, 0)),
        conv.weight,
        conv.bias,
    )
    torch.testing.assert_close(conv(sample), explicit)


@pytest.mark.parametrize("fused", (False, True), ids=("fallback", "fused"))
def test_causal_conv_norm_pad_and_residual_match_unfused_equation(
    monkeypatch: pytest.MonkeyPatch, fused: bool
) -> None:
    conv = CausalConv3d(8, 8, kernel_size=3, padding=1)
    norm = TemporalIsolatedGroupNorm(4, 8, eps=1e-6)
    with torch.no_grad():
        conv.weight.copy_(torch.linspace(-0.2, 0.3, conv.weight.numel()).reshape_as(conv.weight))
        assert conv.bias is not None and norm.weight is not None and norm.bias is not None
        conv.bias.copy_(torch.linspace(-0.1, 0.1, 8))
        norm.weight.copy_(torch.linspace(0.6, 1.3, 8))
        norm.bias.copy_(torch.linspace(-0.3, 0.2, 8))
    sample = torch.linspace(-1.7, 2.1, 1 * 8 * 2 * 3 * 4).reshape(1, 8, 2, 3, 4)
    residual = torch.linspace(0.4, -0.2, sample.numel()).reshape_as(sample)

    normalized = F.silu(norm(sample))
    padded = F.pad(normalized, (1, 1, 1, 1, 0, 0), mode="reflect")
    padded = F.pad(padded, (0, 0, 0, 0, 2, 0))
    expected = F.conv3d(padded, conv.weight, conv.bias).add(residual)

    def kitchen_supported(_input: torch.Tensor) -> bool:
        return fused

    monkeypatch.setattr(vae_module, "_kitchen_ndhwc", kitchen_supported)
    calls: list[torch.Tensor | None] = []

    def fused_conv(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_residual: torch.Tensor | None,
        stride: tuple[int, int, int],
    ) -> torch.Tensor:
        calls.append(conv_residual)
        assert conv_residual is not None
        return F.conv3d(input, weight, bias, stride).add(conv_residual)

    if fused:
        monkeypatch.setattr(vae_module, "_fp16_accum_conv", fused_conv)
    actual = conv(sample, pre_norm=norm, residual=residual)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert calls == ([residual] if fused else [])


def test_causal_conv_custom_spatial_pad_matches_downsample_equation() -> None:
    conv = CausalConv3d(1, 1, kernel_size=3, stride=(1, 2, 2), padding=(1, 0, 0))
    with torch.no_grad():
        conv.weight.copy_(torch.linspace(-0.5, 0.4, conv.weight.numel()).reshape_as(conv.weight))
        assert conv.bias is not None
        conv.bias.fill_(0.125)
    sample = torch.linspace(-1.0, 1.0, 1 * 1 * 2 * 4 * 6).reshape(1, 1, 2, 4, 6)
    spatial = F.pad(sample, (0, 1, 0, 1, 0, 0), mode="reflect")
    expected = F.conv3d(F.pad(spatial, (0, 0, 0, 0, 2, 0)), conv.weight, conv.bias, conv.stride)

    torch.testing.assert_close(conv(sample, spatial_pad=(0, 1, 0, 1)), expected)


def test_temporal_group_norm_isolates_each_frame_statistics() -> None:
    norm = TemporalIsolatedGroupNorm(2, 4, affine=False)
    first = torch.arange(16, dtype=torch.float32).reshape(1, 4, 1, 2, 2)
    second = torch.flip(first, dims=(-1,))
    pair = torch.cat((first, second), dim=2)
    changed = torch.cat((first, second * 1000.0 + 500.0), dim=2)
    torch.testing.assert_close(norm(pair)[:, :, :1], norm(changed)[:, :, :1])


def test_temporal_group_norm_preserves_direct_stock_constructor() -> None:
    norm = TemporalIsolatedGroupNorm(2, 4, dtype=torch.float64)
    assert norm.weight is not None and torch.equal(norm.weight, torch.ones(4, dtype=torch.float64))
    assert norm.bias is not None and torch.equal(norm.bias, torch.zeros(4, dtype=torch.float64))


class TileDecodeProbe(MiniMaxH3VideoVAE):
    def __init__(self) -> None:
        super().__init__(
            reduced_config(
                tile_size=4,
                tile_overlap_min=2,
                tiling=True,
                space_down=(1,),
                time_down=(1,),
            )
        )
        self.decode_batch_sizes: list[int] = []
        self.decoded_references: list[weakref.ReferenceType[torch.Tensor]] = []
        self.previous_alive: list[bool] = []

    def _decode_pixels(self, latent: torch.Tensor) -> torch.Tensor:
        if self.decoded_references:
            self.previous_alive.append(self.decoded_references[-1]() is not None)
        self.decode_batch_sizes.append(latent.shape[0])
        corner = latent[:, :1, :, :1, :1]
        decoded = corner.expand(-1, -1, -1, latent.shape[-2], latent.shape[-1]).clone()
        self.decoded_references.append(weakref.ref(decoded))
        return decoded


def _set_free_tile_memory(monkeypatch: pytest.MonkeyPatch, bytes_free: int) -> None:
    class Memory:
        free_total = bytes_free

    def free_memory(_device: torch.device) -> Memory:
        return Memory()

    monkeypatch.setattr(vae_module, "get_free_memory", free_memory)


def test_tiled_decode_batches_rows_from_free_memory_without_changing_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent = torch.arange(1 * 2 * 1 * 4 * 8, dtype=torch.float32).reshape(1, 2, 1, 4, 8)
    _set_free_tile_memory(monkeypatch, 4 * 128 * 2**20)
    batched = TileDecodeProbe()
    batched_pixels = batched.tiled_decode(latent)
    assert batched.decode_batch_sizes == [3]

    _set_free_tile_memory(monkeypatch, 0)
    single = TileDecodeProbe()
    single_pixels = single.tiled_decode(latent)
    assert single.decode_batch_sizes == [1, 1, 1]
    assert single.previous_alive == [False, False]
    torch.testing.assert_close(batched_pixels, single_pixels)


def test_tiled_decode_blends_against_composited_top_and_left_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_free_tile_memory(monkeypatch, 0)
    latent = torch.zeros((1, 2, 1, 6, 6), dtype=torch.float32)
    latent[:, 0] = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1) * 10 + torch.arange(
        6, dtype=torch.float32
    ).view(1, 1, 1, 6)
    vae = TileDecodeProbe()

    actual = vae.tiled_decode(latent)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 11.0, 12.0, 12.0],
            [20.0, 20.0, 20.0, 21.0, 22.0, 22.0],
            [20.0, 20.0, 20.0, 21.0, 22.0, 22.0],
        ]
    ).reshape(1, 1, 1, 6, 6)
    torch.testing.assert_close(actual, expected)
    del actual
    gc.collect()
    assert all(reference() is None for reference in vae.decoded_references)


def test_temporal_group_norm_refuses_ambiguous_factory_placement() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        TemporalIsolatedGroupNorm(2, 4, dtype=torch.float64, operations=INITLESS)


def test_token_grid_and_split_half_rope_match_reference_layout() -> None:
    ids = create_token_ids((1, 2, 2), torch.device("cpu"), torch.float32)
    expected = torch.tensor(
        [[[-0.0, -0.5, -0.5], [-0.0, -0.5, 0.5], [-0.0, 0.5, -0.5], [-0.0, 0.5, 0.5]]]
    )
    torch.testing.assert_close(ids, expected)

    q = torch.tensor([[[[1.0, 2.0, 10.0, 20.0]]]])
    k = q + 1.0
    table = torch.tensor([[[[[[0.0, -1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]]])
    q_rot, k_rot = apply_rope_split_half(q, k, table)
    torch.testing.assert_close(q_rot, torch.tensor([[[[-10.0, 2.0, 1.0, 20.0]]]]))
    torch.testing.assert_close(k_rot, torch.tensor([[[[-11.0, 3.0, 2.0, 21.0]]]]))


def test_transformer_fused_boundaries_match_unfused_equations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def attention_kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        del mask, causal, scale, enable_gqa
        return q + k * 2.0 + v * 3.0

    block = TransformerBlock(2, 8, attention_kernel=attention_kernel)
    with torch.no_grad():
        block.norm1.weight.copy_(torch.linspace(0.6, 1.4, 16))
        block.norm2.weight.copy_(torch.linspace(1.3, 0.5, 16))
        block.scale1.copy_(torch.linspace(-0.2, 0.3, 16))
        block.scale2.copy_(torch.linspace(0.4, -0.1, 16))
        linears = (block.attn.to_qkv, block.attn.to_out, block.ff.w1, block.ff.w2)
        for index, linear in enumerate(linears):
            linear.weight.copy_(
                torch.linspace(
                    -0.05 + index * 0.01, 0.07 + index * 0.01, linear.weight.numel()
                ).reshape_as(linear.weight)
            )
            assert linear.bias is not None
            linear.bias.copy_(torch.linspace(-0.03, 0.02, linear.bias.numel()))
    source = torch.linspace(-1.2, 1.7, 1 * 3 * 16).reshape(1, 3, 16)
    ids = create_token_ids((1, 1, 3), source.device, source.dtype)
    rotary = RotaryEmbeddingND(6)(ids)

    normalized1 = F.rms_norm(source, (16,), block.norm1.weight, block.norm1.eps)
    qkv = F.linear(normalized1, block.attn.to_qkv.weight, block.attn.to_qkv.bias).view(1, 3, 2, 24)
    query, key, value = qkv.chunk(3, dim=-1)
    query = F.rms_norm(query, (8,), None, block.attn.norm_q.eps)
    key = F.rms_norm(key, (8,), None, block.attn.norm_k.eps)
    rotated = rotary.shape[-3] * 2
    query_prefix, key_prefix = apply_rope_split_half(
        query[..., :rotated], key[..., :rotated], rotary
    )
    query = torch.cat((query_prefix, query[..., rotated:]), dim=-1)
    key = torch.cat((key_prefix, key[..., rotated:]), dim=-1)
    attended = attention_kernel(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    projected = F.linear(
        attended.transpose(1, 2).reshape(1, 3, 16),
        block.attn.to_out.weight,
        block.attn.to_out.bias,
    )
    after_attention = source + projected * block.scale1
    normalized2 = F.rms_norm(after_attention, (16,), block.norm2.weight, block.norm2.eps)
    gate, up = F.linear(normalized2, block.ff.w1.weight, block.ff.w1.bias).chunk(2, dim=-1)
    fed_forward = F.linear(F.silu(gate) * up, block.ff.w2.weight, block.ff.w2.bias)
    expected = after_attention + fed_forward * block.scale2

    original = vae_module.dinkster_kitchen.rms_rope_split_half_
    scale_devices: list[torch.device] = []

    def fused_norm_rope(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        scale = cast(torch.Tensor, args[3])
        scale_devices.append(scale.device)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(vae_module.dinkster_kitchen, "rms_rope_split_half_", fused_norm_rope)
    with torch.no_grad():
        actual = block(source.clone(), rotary)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert scale_devices == [source.device]


def test_reduced_vit3d_decoder_preserves_exact_patch_reassembly() -> None:
    decoder = ViT3DDecoder(
        patch_size=2,
        patch_size_t=2,
        in_channels=3,
        out_channels=2,
        num_layers=1,
        heads=2,
        dim_head=6,
        rope_dim_ratio=1.0,
        num_register_tokens=1,
    )
    for parameter in decoder.parameters():
        torch.nn.init.constant_(parameter, 0.01)
    latent = torch.randn(1, 3, 2, 3, 4)
    output = decoder(latent)
    assert output.shape == (1, 2, 4, 6, 8)
    assert torch.isfinite(output).all()
    assert isinstance(decoder.pos_embed, RotaryEmbeddingND)
    assert len(decoder.transformer_blocks) == 1
    state = decoder.state_dict()
    expected_shapes = {
        "mask_token": (1, 1, 12),
        "register_tokens": (1, 1, 12),
        "x_embedder.weight": (12, 3),
        "x_embedder.bias": (12,),
        "transformer_blocks.0.scale1": (12,),
        "transformer_blocks.0.scale2": (12,),
        "transformer_blocks.0.norm1.weight": (12,),
        "transformer_blocks.0.attn.to_qkv.weight": (36, 12),
        "transformer_blocks.0.attn.to_out.weight": (12, 12),
        "transformer_blocks.0.norm2.weight": (12,),
        "transformer_blocks.0.ff.w1.weight": (96, 12),
        "transformer_blocks.0.ff.w2.weight": (12, 48),
        "norm_out.weight": (12,),
        "proj_out.weight": (16, 12),
        "proj_out.bias": (16,),
    }
    assert {key: tuple(state[key].shape) for key in expected_shapes} == expected_shapes
    assert "transformer_blocks.0.attn.norm_q.weight" not in state
    assert "transformer_blocks.0.attn.norm_k.weight" not in state


class ProbeVAE(MiniMaxH3VideoVAE):
    def __init__(self, config: MiniMaxH3VideoVAEConfig | None = None) -> None:
        super().__init__(config or reduced_config())
        self.encode_parts: list[torch.Tensor] = []
        self.decode_parts: list[torch.Tensor] = []

    def _adaptive_encode(self, content: torch.Tensor) -> torch.Tensor:
        self.encode_parts.append(content.detach().clone())
        base = content.mean(dim=1, keepdim=True)
        return base.repeat(1, self.config.embed_dim * 2, 1, 1, 1)

    def _adaptive_decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_parts.append(latent.detach().clone())
        base = latent.mean(dim=1, keepdim=True)
        return base.repeat(
            1,
            self.config.out_channels,
            self.vae_ratio_t,
            self.vae_ratio,
            self.vae_ratio,
        )


def test_pixel_and_latent_transforms_are_pinned() -> None:
    vae = ProbeVAE()
    pixels = (
        torch.tensor([-1.0, 1.0], dtype=torch.float32).reshape(1, 1, 2, 1, 1).repeat(1, 3, 1, 1, 1)
    )
    normalized = vae._normalize_pixels(pixels)
    expected = ((pixels + 1.0) * 0.5 - vae.pixel_mean) / vae.pixel_std
    torch.testing.assert_close(normalized, expected)

    raw = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float64).reshape(1, 3, 1, 1, 1)
    finalized = vae._finalize_pixels(raw)
    assert finalized.dtype == torch.float32
    assert torch.equal(finalized, torch.tensor([0.0, 0.456, 1.0]).reshape(1, 3, 1, 1, 1))

    mean = torch.arange(2, dtype=torch.float32).reshape(1, 2, 1, 1, 1)
    standardized = vae._normalize_latents(mean)
    torch.testing.assert_close(vae._denormalize_latents(standardized), mean)


def test_chunked_encode_pads_remainder_with_last_frame_and_drops_tail_tokens() -> None:
    vae = ProbeVAE()
    content = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1, 1).repeat(1, 3, 1, 2, 2)
    latent = vae.encode(content)
    assert [part.shape[2] for part in vae.encode_parts] == [3, 3]
    expected_last = vae._normalize_pixels(content[:, :, -1:])
    torch.testing.assert_close(vae.encode_parts[-1][:, :, -2:], expected_last.repeat(1, 1, 2, 1, 1))
    assert latent.shape == (1, 2, 5, 2, 2)
    assert vae.encode_output_shape(content.shape) == tuple(latent.shape)


def test_stride_2_spatial_downsample_pads_space_only_and_encodes() -> None:
    down = Downsample3D(3, 4, operations=CastOperations(torch.float32))
    generator = torch.Generator().manual_seed(20260823)
    down.load_state_dict(
        {
            key: torch.randn(value.shape, generator=generator) * 0.02
            for key, value in down.state_dict().items()
        },
        strict=True,
        assign=True,
    )
    content = torch.linspace(-1.0, 1.0, 3 * 3 * 6 * 6).reshape(1, 3, 3, 6, 6)
    out = down(content)
    assert tuple(out.shape) == (1, 4, 3, 3, 3)
    reference = down.conv(F.pad(content, (0, 1, 0, 1, 0, 0), mode="reflect"))
    assert torch.equal(out, reference)

    vae = filled_reduced_vae(config=reduced_config(space_down=(2,)))
    clip = torch.linspace(-1.0, 1.0, 3 * 3 * 6 * 6).reshape(1, 3, 3, 6, 6)
    latent = vae.encode(clip)
    assert tuple(latent.shape) == vae.encode_output_shape(clip.shape)
    assert bool(torch.isfinite(latent).all())


def test_decode_chunking_is_bounded_and_uses_supplied_output_buffer() -> None:
    vae = ProbeVAE()
    latent = torch.zeros((1, 2, 5, 2, 2))
    output = torch.empty(vae.decode_output_shape(latent.shape), dtype=torch.float32)
    result = vae.decode(latent, output_buffer=output)
    assert result is output
    assert all(
        part.shape[2] <= vae.tokens_chunk_size + vae.token_overlap for part in vae.decode_parts
    )
    assert output.dtype == torch.float32
    assert bool(torch.all((output >= 0.0) & (output <= 1.0)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_decode_writes_cuda_chunks_to_cpu_output() -> None:
    vae = ProbeVAE().to("cuda")
    latent = torch.arange(5, device="cuda", dtype=torch.float32).reshape(1, 1, 5, 1, 1)
    latent = latent.repeat(1, vae.config.embed_dim, 1, 2, 2)

    expected = vae.decode(latent)
    assert expected.device.type == "cpu"
    output = torch.empty(vae.decode_output_shape(latent.shape), dtype=torch.float32)

    result = vae.decode(latent, output_buffer=output)
    assert result is output
    assert torch.equal(result, expected)


@pytest.mark.parametrize("frames", (2, 4, 5, 6, 7, 12))
def test_decode_chunking_preserves_indexed_frames_across_remainders(frames: int) -> None:
    vae = ProbeVAE()
    latent = torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1)
    latent = latent.repeat(1, vae.config.embed_dim, 1, 1, 1)
    decoded = vae.decode_temporal(latent)
    expected_raw = torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1)
    expected_raw = expected_raw.repeat(1, vae.config.out_channels, 1, 1, 1)
    torch.testing.assert_close(decoded, vae._finalize_pixels(expected_raw))
    assert all(
        part.shape[2] <= vae.tokens_chunk_size + vae.token_overlap for part in vae.decode_parts
    )


def test_two_latent_token_remainder_crops_to_exact_five_frame_plan() -> None:
    vae = ProbeVAE(MiniMaxH3VideoVAEConfig(tiling=False))
    latent = torch.zeros((1, 24, 2, 1, 1))
    assert vae.decode_output_shape(latent.shape) == (1, 3, 5, 16, 16)
    assert vae.decode(latent).shape == (1, 3, 5, 16, 16)


@pytest.mark.parametrize(
    "content",
    (
        torch.empty(3, 2, 2),
        torch.empty(1, 2, 1, 2, 2),
        torch.empty(1, 3, 0, 2, 2),
        torch.empty(1, 3, 1, 0, 2),
    ),
)
def test_encode_refuses_bad_rank_channel_or_frame_geometry_before_work(
    content: torch.Tensor,
) -> None:
    vae = ProbeVAE()
    with pytest.raises(ValueError):
        vae.encode(content)
    assert vae.encode_parts == []


@pytest.mark.parametrize(
    "latent",
    (
        torch.empty(1, 2, 2, 2),
        torch.empty(1, 3, 1, 2, 2),
        torch.empty(1, 2, 0, 2, 2),
        torch.empty(1, 2, 1, 0, 2),
    ),
)
def test_decode_refuses_bad_rank_channel_or_frame_geometry_before_work(
    latent: torch.Tensor,
) -> None:
    vae = ProbeVAE()
    with pytest.raises(ValueError):
        vae.decode(latent)
    assert vae.decode_parts == []


@pytest.mark.parametrize("bad", ("shape", "dtype", "layout", "expanded", "meta"))
def test_supplied_output_buffer_refuses_before_decoder_work(bad: str) -> None:
    vae = ProbeVAE()
    latent = torch.zeros((1, 2, 1, 2, 2))
    shape = vae.decode_output_shape(latent.shape)
    if bad == "shape":
        output = torch.empty((*shape[:-3], shape[-3] + 1, *shape[-2:]))
    elif bad == "dtype":
        output = torch.empty(shape, dtype=torch.float64)
    elif bad == "layout":
        output = torch.empty(shape).to_sparse()
    elif bad == "expanded":
        output = torch.empty((*shape[:-1], 1)).expand(shape)
    else:
        output = torch.empty(shape, device="meta")
    with pytest.raises((TypeError, ValueError)):
        vae.decode(latent, output_buffer=output)
    assert vae.decode_parts == []


def test_decode_refuses_underfilled_output_buffer() -> None:
    class UnderProducingVAE(ProbeVAE):
        def _adaptive_decode(self, latent: torch.Tensor) -> torch.Tensor:
            decoded = super()._adaptive_decode(latent)
            return decoded[:, :, :1]

    vae = UnderProducingVAE()
    with pytest.raises(RuntimeError, match="decoded 1 frames for a buffer expecting 5"):
        vae.decode(torch.zeros((1, 2, 5, 2, 2)))


def test_decode_releases_temporary_after_exception() -> None:
    released: list[bool] = []

    class FailingVAE(ProbeVAE):
        reference: weakref.ReferenceType[torch.Tensor] | None = None

        def _adaptive_decode(self, latent: torch.Tensor) -> torch.Tensor:
            temporary = super()._adaptive_decode(latent)
            self.reference = weakref.ref(temporary)
            weakref.finalize(temporary, released.append, True)
            return temporary

        def _finalize_pixels(self, content: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("finish failed")

    vae = FailingVAE()
    with pytest.raises(RuntimeError, match="finish failed"):
        vae.decode(torch.zeros((1, 2, 2, 2, 2)))
    gc.collect()
    assert vae.reference is not None and vae.reference() is None
    assert released == [True]
