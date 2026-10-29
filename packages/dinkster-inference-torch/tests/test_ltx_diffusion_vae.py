from pathlib import Path

import pytest
import torch
from dinkster_inference_torch import ltx_diffusion_vae as vae_module
from dinkster_inference_torch.ltx_diffusion_vae import (
    LinearPixelShuffleUpsample,
    NADiffusionDecoder,
    patchify,
    unpatchify,
)
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import INITLESS
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(
    Path(__file__).parent / "goldens/ltx_diffusion_vae_goldens.json",
    allow_portable_fallback=True,
)


def test_patchify_round_trip_and_channel_order() -> None:
    x = torch.arange(1 * 2 * 2 * 4 * 6).reshape(1, 2, 2, 4, 6)
    packed = patchify(x, 2, 2)
    assert packed.shape == (1, 16, 1, 2, 3)
    torch.testing.assert_close(unpatchify(packed, 2, 2), x)


def test_native_linear_pixel_shuffle() -> None:
    module = LinearPixelShuffleUpsample(1, (2, 2, 2), 1, INITLESS)
    module.proj.weight.data.fill_(1)
    module.proj.bias.data.copy_(torch.arange(8))
    out = module(torch.ones(1, 1, 1, 1, 1), drop_leading_frame=False)
    torch.testing.assert_close(out.flatten(), torch.arange(1, 9, dtype=torch.float32))


def _tiny_decoder(*, diffusion_depth: int = 0) -> NADiffusionDecoder:
    model = NADiffusionDecoder(
        in_channels=2,
        out_channels=1,
        patch_size=1,
        head_dim=4,
        stage_channels=(4, 4, 4, 4, 4),
        stage_depths=(0, 0, 0, 0, diffusion_depth),
        stage_kernels=((1, 1, 1),) * 5,
        upsamples=(((1, 1, 1), 1),) * 4,
        t_emb_dim=4,
    )
    for parameter in model.parameters():
        parameter.data.normal_(generator=torch.Generator().manual_seed(parameter.numel()))
    return model


def _reference_decoder() -> NADiffusionDecoder:
    model = NADiffusionDecoder(
        in_channels=2,
        out_channels=1,
        patch_size=1,
        head_dim=4,
        stage_channels=(4, 4, 4, 4, 4),
        stage_depths=(1, 0, 0, 0, 1),
        stage_kernels=((1, 1, 1),) * 5,
        upsamples=(((1, 1, 1), 1),) * 4,
        t_emb_dim=4,
    )
    model.load_state_dict(fill_state_dict(GOLDENS["state_dict"]), strict=True)
    return model


def test_decoder_matches_executed_reference() -> None:
    model = _reference_decoder()
    latent = hashed_input("ltx-diffusion-vae:latent", (1, 2, 2, 2, 2))
    with torch.no_grad():
        context = model.forward_pre_diffusion(latent)
        output = model(latent, generator=torch.Generator().manual_seed(13))

    assert_reference_tensor(context, torch.tensor(GOLDENS["context"]))
    assert_reference_tensor(output, torch.tensor(GOLDENS["output"]))


def test_tiny_decoder_is_deterministic_and_matches_direct_step() -> None:
    model = _tiny_decoder()
    z = torch.randn(1, 2, 1, 2, 2)
    context = model.forward_pre_diffusion(z)
    noise = torch.randn((1, 1, 1, 2, 2), generator=torch.Generator().manual_seed(7))
    expected = model.forward_diff_step(context, noise, torch.ones(1))
    actual = model(z, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, model(z, generator=torch.Generator().manual_seed(7)))


def test_state_layout_and_no_persistent_execution_state() -> None:
    model = _tiny_decoder()
    keys = set(model.state_dict())
    assert "conv_in.weight" in keys
    assert "t_embedder.mlp.0.weight" in keys
    assert "shared_adaln.proj.weight" in keys
    assert "default_inference_timesteps" not in keys
    before = set(vars(model))
    model(torch.zeros(1, 2, 1, 1, 1), generator=torch.Generator().manual_seed(0))
    assert set(vars(model)) == before


def test_diffusion_decoder_state_is_admitted_by_component_residency() -> None:
    model = _tiny_decoder()

    enroll_component(model, load_device="cpu", offload_device="cpu")


def test_chunked_projection_path_is_value_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_decoder(diffusion_depth=1)
    z = torch.randn(1, 2, 2, 2, 2)
    seed = 19
    with torch.no_grad():
        expected = model(z, generator=torch.Generator().manual_seed(seed))
        monkeypatch.setattr(vae_module, "MLP_TOKEN_CHUNK", 1)
        actual = model(z, generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(actual, expected)


def test_diffusion_decoder_autograd_keeps_inputs_differentiable() -> None:
    model = _tiny_decoder(diffusion_depth=1)
    z = torch.randn(1, 2, 1, 1, 1, requires_grad=True)

    model(z, generator=torch.Generator().manual_seed(3)).sum().backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_temporal_pixel_shuffle_drop_option_preserves_expected_frames() -> None:
    module = LinearPixelShuffleUpsample(1, (2, 1, 1), 1, INITLESS)
    module.proj.weight.data.fill_(1)
    module.proj.bias.data.copy_(torch.tensor([2.0, 4.0]))
    x = torch.ones(1, 2, 1, 1, 1)

    kept = module(x, drop_leading_frame=False)
    dropped = module(x, drop_leading_frame=True)

    assert kept.shape[1] == 4
    assert torch.equal(dropped, kept[:, 1:])
