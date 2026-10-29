from __future__ import annotations

import gc
import hashlib
import json
import weakref
from typing import Any, Literal

import pytest
import torch
import torch.nn.functional as F
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference_torch import select_attention
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import CastOperations, ResidencyRouted
from dinkster_inference_torch.wan21_vae import (
    LATENTS_MEAN,
    LATENTS_STD,
    AttentionBlock,
    CausalConv3d,
    Resample,
    RMSNorm,
    WanVAE,
    WanVAEConfig,
    count_cache_layers,
)


def _fill_model(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("gamma"):
                parameter.fill_(1.0)
            elif name.endswith("bias"):
                parameter.zero_()
            else:
                parameter.fill_(0.01)


def _reduced_config(*, conv_out_channels: int = 3) -> WanVAEConfig:
    return WanVAEConfig(
        dim=2,
        z_dim=2,
        dim_mult=(1, 1, 1),
        num_res_blocks=1,
        temporal_downsample=(True, True),
        conv_out_channels=conv_out_channels,
    )


def test_default_config_is_exact_base_wan21_geometry() -> None:
    config = WanVAEConfig()
    assert config.dim == 96
    assert config.z_dim == 16
    assert config.dim_mult == (1, 2, 4, 4)
    assert config.num_res_blocks == 2
    assert config.attn_scales == ()
    assert config.temporal_downsample == (False, True, True)
    assert config.image_channels == config.conv_out_channels == 3
    assert config.dropout == 0.0
    assert config.spatial_ratio == 8


def test_flow_rvs_config_accepts_rgb_input_and_one_channel_decode() -> None:
    config = WanVAEConfig(conv_out_channels=1)
    assert (config.image_channels, config.conv_out_channels) == (3, 1)


@pytest.mark.parametrize(
    "config",
    (
        {"z_dim": 0},
        {"dim_mult": (1,)},
        {"temporal_downsample": (True,)},
        {"temporal_downsample": (False, False, False)},
        {"image_channels": 4},
        {"conv_out_channels": 4},
        {"dropout": float("nan")},
        {"attn_scales": (0.0,)},
    ),
)
def test_config_refuses_non_wan21_geometry(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        WanVAEConfig(**config)


def test_causal_conv_single_frame_fast_path_matches_explicit_front_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conv = CausalConv3d(2, 3, 3, padding=1)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel()).reshape_as(conv.weight) / 1000)
        assert conv.bias is not None
        conv.bias.copy_(torch.tensor([-0.25, 0.0, 0.25]))
    content = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(1, 2, 1, 4, 5) / 10
    expected = torch.conv3d(
        content,
        conv.weight[:, :, -1:],
        conv.bias,
        stride=conv.stride,
        padding=(0, 1, 1),
        dilation=conv.dilation,
        groups=conv.groups,
    )
    dispatches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def observed_conv3d(input: torch.Tensor, weight: torch.Tensor, *args: Any) -> torch.Tensor:
        dispatches.append((tuple(input.shape), tuple(weight.shape)))
        return torch.conv3d(input, weight, *args)

    monkeypatch.setattr(F, "conv3d", observed_conv3d)
    torch.testing.assert_close(conv(content), expected)
    assert dispatches == [((1, 2, 1, 4, 5), (3, 2, 1, 3, 3))]


def test_causal_conv_single_frame_fast_path_covers_unpadded_temporal_kernel() -> None:
    conv = CausalConv3d(1, 1, (3, 1, 1), padding=0, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([1.0, 2.0, 3.0]).reshape_as(conv.weight))
    content = torch.tensor([[[[[4.0]]]]])
    expected = F.conv3d(content, conv.weight[:, :, -1:])
    torch.testing.assert_close(conv(content), expected)


def test_causal_conv_cache_matches_one_shot_execution() -> None:
    conv = CausalConv3d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(27, dtype=torch.float32).reshape_as(conv.weight) / 27)
    content = torch.arange(5 * 3 * 3, dtype=torch.float32).reshape(1, 1, 5, 3, 3) / 10
    expected = conv(content)
    first = content[:, :, :2]
    second = content[:, :, 2:]
    actual = torch.cat((conv(first), conv(second, cache_x=first[:, :, -2:])), dim=2)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("kernel_size", "padding"), ((1, 0), (3, 1)))
def test_causal_conv_releases_consumed_cache_before_convolution(
    kernel_size: int,
    padding: int,
) -> None:
    cache = torch.ones(1, 1, 2, 2, 2)
    cache_reference = weakref.ref(cache)
    cache_list: list[torch.Tensor | None] = [cache]
    del cache

    class ObservedCausalConv3d(CausalConv3d):
        def _conv_forward(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            gc.collect()
            assert cache_reference() is None
            return F.conv3d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    conv = ObservedCausalConv3d(1, 1, kernel_size, padding=padding, bias=False)
    conv(torch.ones(1, 1, 1, 2, 2), cache_list=cache_list, cache_idx=0)
    assert cache_list == [None]


def test_rms_norm_matches_reference_formula_for_image_and_video_axes() -> None:
    image = torch.arange(24, dtype=torch.float64).reshape(1, 3, 2, 4).add(1)
    image_norm = RMSNorm(3)
    with torch.no_grad():
        image_norm.gamma.copy_(torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1))
    expected_image = F.normalize(image, dim=1) * (3**0.5) * image_norm.gamma
    torch.testing.assert_close(image_norm(image), expected_image)

    video = image.unsqueeze(2)
    video_norm = RMSNorm(3, images=False, bias=True)
    with torch.no_grad():
        video_norm.gamma.fill_(2.0)
        assert video_norm.bias is not None
        video_norm.bias.fill_(0.5)
    expected_video = F.normalize(video, dim=1) * (3**0.5) * 2.0 + 0.5
    torch.testing.assert_close(video_norm(video), expected_video)


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("none", (1, 4, 3, 8, 10)),
        ("upsample2d", (1, 2, 3, 16, 20)),
        ("upsample3d", (1, 2, 3, 16, 20)),
        ("downsample2d", (1, 4, 3, 4, 5)),
        ("downsample3d", (1, 4, 3, 4, 5)),
    ),
)
def test_resample_modes_preserve_exact_channel_and_geometry_contract(
    mode: Literal["none", "upsample2d", "upsample3d", "downsample2d", "downsample3d"],
    expected: tuple[int, ...],
) -> None:
    layer = Resample(4, mode)
    _fill_model(layer)
    assert layer(torch.ones(1, 4, 3, 8, 10)).shape == expected


def test_attention_uses_injected_vae_kernel_without_registering_it() -> None:
    baseline = AttentionBlock(4)
    _fill_model(baseline)
    spy = CallableModuleKernel(select_attention("vae").kernel)
    block = AttentionBlock(4, attention_kernel=spy)
    block.load_state_dict(baseline.state_dict(), strict=True)
    content = torch.randn(1, 4, 2, 3, 5)
    torch.testing.assert_close(block(content), baseline(content))
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert (call["q_shape"], call["k_shape"], call["v_shape"]) == (
        (2, 1, 15, 4),
        (2, 1, 15, 4),
        (2, 1, 15, 4),
    )
    assert (call["q_stride"], call["k_stride"], call["v_stride"]) == (
        (60, 60, 4, 1),
        (60, 60, 4, 1),
        (60, 60, 4, 1),
    )
    assert call["q_contiguous"] and call["k_contiguous"] and call["v_contiguous"]
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)
    assert_kernel_is_not_model_state(block, spy)


def test_reduced_state_layout_and_shapes_are_checkpoint_compatible() -> None:
    model = WanVAE(_reduced_config())
    state = model.state_dict()
    assert len(state) == 122
    canonical = sorted((name, list(value.shape)) for name, value in state.items())
    digest = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    assert digest == "a615879081889fdcb301cd264a4a2bb6f60deeb5006f1b7d81d779eedc965daa"
    expected = {
        "encoder.conv1.weight": (2, 3, 3, 3, 3),
        "encoder.downsamples.0.residual.0.gamma": (2, 1, 1, 1),
        "encoder.downsamples.0.residual.2.weight": (2, 2, 3, 3, 3),
        "encoder.downsamples.1.resample.1.weight": (2, 2, 3, 3),
        "encoder.downsamples.1.time_conv.weight": (2, 2, 3, 1, 1),
        "encoder.middle.1.to_qkv.weight": (6, 2, 1, 1),
        "encoder.head.2.weight": (4, 2, 3, 3, 3),
        "conv1.weight": (4, 4, 1, 1, 1),
        "conv2.weight": (2, 2, 1, 1, 1),
        "decoder.conv1.weight": (2, 2, 3, 3, 3),
        "decoder.middle.1.proj.weight": (2, 2, 1, 1),
        "decoder.upsamples.2.time_conv.weight": (4, 2, 3, 1, 1),
        "decoder.upsamples.2.resample.1.weight": (1, 2, 3, 3),
        "decoder.head.2.weight": (3, 2, 3, 3, 3),
    }
    for name, shape in expected.items():
        assert name in state
        assert tuple(state[name].shape) == shape
    assert all("latents_" not in name and "_attention_kernel" not in name for name in state)


def test_every_state_owner_is_residency_routed_and_compute_castable() -> None:
    model = WanVAE(_reduced_config(), operations=CastOperations(torch.bfloat16))
    _fill_model(model)
    owners = [module for module in model.modules() if tuple(module.parameters(recurse=False))]
    assert owners
    assert all(isinstance(module, ResidencyRouted) for module in owners)
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    encoded = model.encode(torch.ones(1, 3, 1, 8, 8, dtype=torch.bfloat16))
    assert encoded.dtype is torch.bfloat16


def test_cache_layer_count_is_exact_for_reduced_encoder_and_decoder() -> None:
    model = WanVAE(_reduced_config())
    assert count_cache_layers(model.encoder) == 16
    assert count_cache_layers(model.decoder) == 22


def test_encode_uses_one_then_two_frame_chunks_and_enforces_4n_plus_1() -> None:
    model = WanVAE(_reduced_config())
    _fill_model(model)
    chunks: list[int] = []
    handle = model.encoder.register_forward_pre_hook(
        lambda _module, args: chunks.append(args[0].shape[2])
    )
    try:
        five = torch.linspace(-1, 1, 1 * 3 * 5 * 8 * 8).reshape(1, 3, 5, 8, 8)
        eight = torch.cat((five, torch.ones(1, 3, 3, 8, 8)), dim=2)
        latent_five = model.encode(five)
        assert chunks == [1, 2, 2]
        chunks.clear()
        latent_eight = model.encode(eight)
        assert chunks == [1, 2, 2]
        torch.testing.assert_close(latent_eight, latent_five)
        assert latent_five.shape == (1, 2, 2, 2, 2)
    finally:
        handle.remove()


def test_decode_uses_cached_one_then_two_token_chunks_and_exact_frame_count() -> None:
    model = WanVAE(_reduced_config())
    _fill_model(model)
    chunks: list[int] = []
    handle = model.decoder.register_forward_pre_hook(
        lambda _module, args: chunks.append(args[0].shape[2])
    )
    try:
        decoded = model.decode(torch.zeros(1, 2, 2, 2, 2))
    finally:
        handle.remove()
    assert chunks == [1, 1]
    assert decoded.shape == (1, 3, 5, 8, 8)


def test_flow_rvs_reduced_decode_executes_one_channel_output() -> None:
    model = WanVAE(_reduced_config(conv_out_channels=1))
    _fill_model(model)

    decoded = model.decode(torch.zeros(1, 2, 2, 2, 2))

    assert decoded.shape == (1, 1, 5, 8, 8)
    assert model.encoder.conv1.weight.shape[1] == 3
    output = model.get_submodule("decoder.head.2")
    assert isinstance(output, CausalConv3d)
    assert output.weight.shape[0] == 1


def test_decoder_recursion_releases_consumed_layer_outputs() -> None:
    model = WanVAE(_reduced_config())
    released: list[bool] = []
    reference: weakref.ReferenceType[torch.Tensor] | None = None

    class First(torch.nn.Module):
        def forward(self, content: torch.Tensor) -> torch.Tensor:
            nonlocal reference
            output = torch.empty_like(content)
            reference = weakref.ref(output)
            weakref.finalize(output, released.append, True)
            return output

    class Replace(torch.nn.Module):
        def forward(self, content: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(content)

    class Observe(torch.nn.Module):
        def forward(self, content: torch.Tensor) -> torch.Tensor:
            gc.collect()
            assert reference is not None and reference() is None
            assert released == [True]
            return content

    model.decoder.upsamples = torch.nn.Sequential(First(), Replace(), Observe())
    model.decoder.head = torch.nn.Sequential(torch.nn.Identity())
    assert model.decoder(torch.zeros(1, 2, 1, 2, 2))[0].shape == (1, 2, 1, 2, 2)


def test_wan21_normalization_constants_and_roundtrip_are_exact() -> None:
    model = WanVAE(_reduced_config())
    assert torch.equal(model.latents_mean, torch.tensor(LATENTS_MEAN))
    assert torch.equal(model.latents_std, torch.tensor(LATENTS_STD))
    latent = torch.arange(16, dtype=torch.float64).reshape(1, 16, 1, 1, 1)
    torch.testing.assert_close(model.process_out(model.process_in(latent)), latent)


@pytest.mark.parametrize(
    "content",
    (
        torch.empty(1, 3, 5, 8),
        torch.empty(1, 2, 5, 8, 8),
        torch.empty(1, 3, 0, 8, 8),
        torch.empty(1, 3, 5, 0, 8),
        torch.empty(1, 3, 5, 7, 8),
    ),
)
def test_encode_refuses_rank_channel_empty_or_spatial_geometry_before_model_work(
    content: torch.Tensor,
) -> None:
    model = WanVAE(_reduced_config())
    called: list[bool] = []

    def mark_called(_module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
        called.append(True)

    handle = model.encoder.register_forward_pre_hook(mark_called)
    try:
        with pytest.raises(ValueError):
            model.encode(content)
    finally:
        handle.remove()
    assert called == []


@pytest.mark.parametrize(
    "latent",
    (
        torch.empty(1, 2, 2, 2),
        torch.empty(1, 3, 2, 2, 2),
        torch.empty(1, 2, 0, 2, 2),
        torch.empty(1, 2, 2, 0, 2),
    ),
)
def test_decode_refuses_rank_channel_or_empty_geometry_before_model_work(
    latent: torch.Tensor,
) -> None:
    model = WanVAE(_reduced_config())
    called: list[bool] = []

    def mark_called(_module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
        called.append(True)

    handle = model.decoder.register_forward_pre_hook(mark_called)
    try:
        with pytest.raises(ValueError):
            model.decode(latent)
    finally:
        handle.remove()
    assert called == []


def test_decode_releases_cached_temporaries_after_exception() -> None:
    model = WanVAE(_reduced_config())
    _fill_model(model)
    released: list[bool] = []
    reference: weakref.ReferenceType[torch.Tensor] | None = None
    original = model.decoder.forward

    def fail_after_decode(
        x: torch.Tensor,
        feat_cache: Any = None,
        feat_idx: list[int] | None = None,
    ) -> list[torch.Tensor]:
        nonlocal reference
        chunks = original(x, feat_cache, feat_idx)
        temporary = chunks[0]
        reference = weakref.ref(temporary)
        weakref.finalize(temporary, released.append, True)
        raise RuntimeError("decode failed")

    model.decoder.forward = fail_after_decode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="decode failed"):
        model.decode(torch.zeros(1, 2, 2, 2, 2))
    gc.collect()
    assert reference is not None and reference() is None
    assert released == [True]
