from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from dinkster_inference import (
    Z_IMAGE_CONFIG,
    Z_IMAGE_PIXEL_CONFIG,
    z_image_layout,
    z_image_pixel_layout,
)
from dinkster_inference_torch import (
    ZImage,
    ZImageBlock,
    ZImagePixelSpace,
    enroll_component,
    z_image_timestep_embedding,
)
from dinkster_inference_torch.operations import INITLESS
from dinkster_inference_torch.z_image import (
    ZImagePixelDecoder,
    ZImagePixelEmbedder,
    ZImagePixelResBlock,
)


@dataclass(frozen=True)
class SmallConfig:
    family_id: str = "dinkster.z_image"
    hidden_width: int = 18
    caption_width: int = 8
    main_blocks: int = 1
    noise_refiner_blocks: int = 1
    context_refiner_blocks: int = 1
    attention_heads: int = 3
    kv_heads: int = 3
    attention_head_dim: int = 6
    ffn_width: int = 16
    latent_channels: int = 2
    patch: tuple[int, int] = (2, 2)
    rope_axes: tuple[int, int, int] = (2, 2, 2)
    rope_theta: float = 256.0
    qk_norm_eps: float = 1e-5
    timestep_embedding_width: int = 6
    modulation_width: int = 6
    timestep_multiplier: float = 1000.0
    block_modulation_silu: bool = False
    pad_tokens_multiple: int = 4
    learned_padding: bool = True


@dataclass(frozen=True)
class SmallPixelConfig(SmallConfig):
    latent_channels: int = 3
    decoder_hidden_width: int = 18
    decoder_blocks: int = 1
    decoder_max_frequencies: int = 2


def test_production_state_dict_matches_exact_layout() -> None:
    with torch.device("meta"):
        model = ZImage(Z_IMAGE_CONFIG)
    actual = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    assert actual == dict(z_image_layout())
    assert len(actual) == 453


def test_pixel_space_state_dict_matches_exact_layout() -> None:
    with torch.device("meta"):
        model = ZImagePixelSpace(Z_IMAGE_PIXEL_CONFIG)
    actual = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    expected = {
        name: shape
        for name, shape in z_image_pixel_layout().items()
        if name not in {"__sequential__", "__x0__"}
    }
    assert actual == expected
    assert len(actual) == 555


def test_pixel_space_forward_converts_x0_decoder_to_flow_velocity() -> None:
    model = ZImagePixelSpace(SmallPixelConfig())  # type: ignore[arg-type]
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    pixels = torch.arange(45, dtype=torch.float32).view(1, 3, 3, 5)
    output = model(pixels, torch.tensor([0.5]), torch.zeros(1, 3, 8))
    torch.testing.assert_close(output, pixels * 2)
    assert output.shape == pixels.shape


def test_pixel_space_fused_norm_rope_matches_autograd_path() -> None:
    torch.manual_seed(30)
    model = ZImagePixelSpace(SmallPixelConfig())  # type: ignore[arg-type]
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    pixels = torch.randn(1, 3, 4, 4)
    timestep = torch.tensor([0.625])
    context = torch.randn(1, 3, 8)

    expected = model(pixels, timestep, context)
    with torch.inference_mode():
        actual = model(pixels, timestep, context)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_pixel_space_dct_embedding_matches_comfyui_equations() -> None:
    torch.manual_seed(31)
    embedder = ZImagePixelEmbedder(3, 7, 3, operations=INITLESS)
    for parameter in embedder.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    inputs = torch.randn(2, 4, 3)

    output = embedder(inputs)

    positions = torch.linspace(0, 1, 2)
    position_y, position_x = torch.meshgrid(positions, positions, indexing="ij")
    frequencies = torch.linspace(0, 2, 3)
    frequency_x = frequencies[None, :, None]
    frequency_y = frequencies[None, None, :]
    dct = (
        torch.cos(position_x.reshape(-1, 1, 1) * frequency_x * torch.pi)
        * torch.cos(position_y.reshape(-1, 1, 1) * frequency_y * torch.pi)
        * (1 + frequency_x * frequency_y) ** -1
    ).view(1, 4, 9)
    projection = cast(torch.nn.Linear, embedder.embedder[0])
    expected = F.linear(
        torch.cat((inputs, dct.expand(2, -1, -1)), dim=-1),
        projection.weight,
        projection.bias,
    )
    torch.testing.assert_close(output, expected)


def test_pixel_space_decoder_matches_comfyui_equations() -> None:
    torch.manual_seed(32)
    config = SmallPixelConfig()
    decoder = ZImagePixelDecoder(config, operations=INITLESS)
    for parameter in decoder.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    pixels = torch.randn(2, 1, 12)
    condition = torch.randn(2, config.hidden_width)

    output = decoder(pixels, condition)

    hidden = decoder.input_embedder(pixels)
    modulation = F.linear(condition, decoder.cond_embed.weight, decoder.cond_embed.bias).unsqueeze(
        1
    )
    for module in decoder.res_blocks:
        block = cast(ZImagePixelResBlock, module)
        shift, scale, gate = block.adaLN_modulation(modulation).chunk(3, dim=-1)
        normalized = F.layer_norm(
            hidden,
            (config.decoder_hidden_width,),
            block.in_ln.weight,
            block.in_ln.bias,
            1e-6,
        )
        hidden = hidden + gate * block.mlp(normalized * (1 + scale) + shift)
    expected = F.linear(
        F.layer_norm(hidden, (config.decoder_hidden_width,), eps=1e-6),
        decoder.final_layer.linear.weight,
        decoder.final_layer.linear.bias,
    )
    torch.testing.assert_close(output, expected)


def test_forward_pads_segments_and_crops_circular_patch_grid() -> None:
    torch.manual_seed(8)
    model = ZImage(SmallConfig())
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.01)
    patches: list[torch.Tensor] = []
    scaled_timesteps: list[torch.Tensor] = []
    model.x_embedder.register_forward_pre_hook(
        lambda _module, inputs: patches.append(inputs[0].detach().clone())
    )
    model.t_embedder.register_forward_pre_hook(
        lambda _module, inputs: scaled_timesteps.append(inputs[0].detach().clone())
    )
    latent = torch.arange(30, dtype=torch.float32).view(1, 2, 3, 5)
    output = model(
        latent,
        torch.tensor([0.25]),
        torch.randn(1, 3, 8),
    )
    assert output.shape == (1, 2, 3, 5)
    assert torch.isfinite(output).all()
    assert patches[0][0, 2].tolist() == [4, 19, 0, 15, 9, 24, 5, 20]
    assert patches[0][0, 5].tolist() == [14, 29, 10, 25, 4, 19, 0, 15]
    torch.testing.assert_close(scaled_timesteps[0], torch.tensor([750.0]))


def test_rope_ids_start_image_after_padded_caption_and_zero_image_padding() -> None:
    model = ZImage(SmallConfig())
    captured: list[torch.Tensor] = []
    original = model.rope_embedder.forward

    def capture(ids: torch.Tensor) -> torch.Tensor:
        captured.append(ids.detach().clone())
        return original(ids)

    model.rope_embedder.forward = capture  # type: ignore[method-assign]
    model(torch.randn(1, 2, 2, 6), torch.tensor([0.5]), torch.randn(1, 3, 8))
    ids = captured[0][0]
    assert ids[:4, 0].tolist() == [1, 2, 3, 4]
    assert ids[4:7, 0].tolist() == [5, 5, 5]
    assert ids[7].tolist() == [0, 0, 0]


def test_context_refiner_is_unmodulated_and_other_blocks_are_modulated() -> None:
    model = ZImage(SmallConfig())
    context = cast(ZImageBlock, model.context_refiner[0])
    noise = cast(ZImageBlock, model.noise_refiner[0])
    main = cast(ZImageBlock, model.layers[0])
    assert context.adaLN_modulation is None
    assert noise.adaLN_modulation is not None
    assert main.adaLN_modulation is not None
    assert noise.attention.q_norm.eps == 1e-5
    assert noise.attention.k_norm.eps == 1e-5


def test_timestep_embedding_has_expected_endpoints() -> None:
    embedding = z_image_timestep_embedding(torch.tensor([0.0]), 6)
    torch.testing.assert_close(embedding, torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]))


def test_forward_casts_timestep_embedding_to_model_dtype() -> None:
    model = ZImage(SmallConfig()).to(torch.bfloat16)
    output = model(
        torch.randn(1, 2, 2, 2, dtype=torch.bfloat16),
        torch.tensor([0.5]),
        torch.randn(1, 3, 8, dtype=torch.bfloat16),
    )
    assert output.dtype == torch.bfloat16


def test_residency_enrollment_routes_learned_padding_tokens() -> None:
    model = ZImage(SmallConfig())
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    assert "" in mechanism.loaded_unit_names()
    output = model(
        torch.randn(1, 2, 2, 6),
        torch.tensor([0.5]),
        torch.randn(1, 3, 8),
    )
    assert output.shape == (1, 2, 2, 6)
