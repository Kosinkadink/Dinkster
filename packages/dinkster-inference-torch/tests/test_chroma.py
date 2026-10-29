from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from dinkster_inference import ChromaConfig, ChromaRadianceConfig
from dinkster_inference.chroma import chroma_layout, chroma_radiance_layout
from dinkster_inference_torch import Chroma, ChromaRadiance, ChromaRadianceOptions, enroll_component
from dinkster_inference_torch.chroma import (
    NerfEmbedder,
    NerfFinalLayer,
    NerfFinalLayerConv,
    NerfGLUBlock,
)
from dinkster_inference_torch.flux import apply_rope_comfy
from dinkster_inference_torch.operations import INITLESS
from dinkster_inference_torch.z_image import PixelSpaceCodec, ZImagePixelCodec
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDEN_PATH = Path(__file__).parent / "goldens" / "chroma_goldens.json"
GOLDENS = json.loads(GOLDEN_PATH.read_text())


def golden_config(spec: dict[str, Any]) -> ChromaConfig | ChromaRadianceConfig:
    common: dict[str, Any] = {
        "hidden_size": spec["hidden_size"],
        "depth": spec["depth"],
        "depth_single_blocks": spec["depth_single_blocks"],
        "num_heads": spec["num_heads"],
        "context_in_dim": spec["context_in_dim"],
        "axes_dim": tuple(spec["axes_dim"]),
        "theta": spec["theta"],
        "mlp_ratio": spec["mlp_ratio"],
        "qkv_bias": spec["qkv_bias"],
        "approximator_input_dim": spec["in_dim"],
        "approximator_hidden_dim": spec["hidden_dim"],
        "approximator_layers": spec["n_layers"],
    }
    if "nerf_hidden_size" in spec:
        return ChromaRadianceConfig(
            patch_size=spec["patch_size"],
            nerf_hidden_size=spec["nerf_hidden_size"],
            nerf_mlp_ratio=spec["nerf_mlp_ratio"],
            nerf_depth=spec["nerf_depth"],
            nerf_max_freqs=spec["nerf_max_freqs"],
            nerf_tile_size=spec["nerf_tile_size"],
            nerf_final_head_type=spec["nerf_final_head_type"],
            use_x0=spec["use_x0"],
            use_sequential_txt_ids=spec["use_sequential_txt_ids"],
            **common,
        )
    return ChromaConfig(
        latent_channels=spec["in_channels"] // spec["patch_size"] ** 2,
        patch_size=spec["patch_size"],
        out_channels=spec["out_channels"],
        **common,
    )


def decode_golden(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def small_chroma_config() -> ChromaConfig:
    return ChromaConfig(
        hidden_size=32,
        depth=2,
        depth_single_blocks=3,
        num_heads=4,
        context_in_dim=8,
        latent_channels=2,
        patch_size=2,
        out_channels=8,
        axes_dim=(2, 2, 4),
        approximator_hidden_dim=16,
        approximator_layers=1,
    )


def small_radiance_config(**changes: object) -> ChromaRadianceConfig:
    config = ChromaRadianceConfig(
        hidden_size=32,
        depth=2,
        depth_single_blocks=3,
        num_heads=4,
        patch_size=2,
        context_in_dim=8,
        axes_dim=(2, 2, 4),
        approximator_hidden_dim=16,
        approximator_layers=1,
        nerf_hidden_size=4,
        nerf_mlp_ratio=2,
        nerf_depth=2,
        nerf_max_freqs=2,
        nerf_tile_size=4,
    )
    return replace(config, **changes)


def initialize(module: torch.nn.Module, *, seed: int = 1) -> None:
    torch.manual_seed(seed)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.02)


def test_production_modules_match_torch_free_layouts() -> None:
    chroma_config = ChromaConfig(3072, 19, 38, 24)
    radiance_config = ChromaRadianceConfig(
        3072,
        19,
        38,
        24,
        16,
        nerf_final_head_type="conv",
        use_x0=True,
    )
    with torch.device("meta"):
        chroma = Chroma(chroma_config)
        radiance = ChromaRadiance(radiance_config)
    assert {key: tuple(value.shape) for key, value in chroma.state_dict().items()} == chroma_layout(
        chroma_config
    )
    assert {
        key: tuple(value.shape) for key, value in radiance.state_dict().items()
    } == chroma_radiance_layout(radiance_config)


def test_chroma_family_owns_comfy_compatible_rope_route() -> None:
    for model in (Chroma(small_chroma_config()), ChromaRadiance(small_radiance_config())):
        blocks = (*model.double_blocks, *model.single_blocks)
        assert all(block._rope_kernel is apply_rope_comfy for block in blocks)  # pyright: ignore[reportPrivateUsage]


def test_chroma_family_reuses_attention_backend_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference_torch.chroma as chroma_module

    calls: list[tuple[object, int, torch.device]] = []

    class RecordingContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    def recording_context(
        kernel: object, query_elements: int, *, device: torch.device
    ) -> RecordingContext:
        calls.append((kernel, query_elements, device))
        return RecordingContext()

    monkeypatch.setattr(chroma_module, "attention_kernel_context", recording_context)
    expected_kernels = []
    for model in (Chroma(small_chroma_config()), ChromaRadiance(small_radiance_config())):
        expected_kernels.append(model._attention_kernel)  # pyright: ignore[reportPrivateUsage]
        initialize(model)
        model(
            torch.randn(1, model.config.latent_channels, 4, 4),
            torch.tensor([0.5]),
            torch.randn(1, 2, 8),
            torch.tensor([3.5]),
        )

    assert len(calls) == 2
    assert [kernel for kernel, _, _ in calls] == expected_kernels
    assert all(elements > 0 and device == torch.device("cpu") for _, elements, device in calls)


def test_radiance_checkpoint_markers_are_residency_routed() -> None:
    model = ChromaRadiance(small_radiance_config(use_x0=True, use_sequential_txt_ids=True))
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    assert mechanism.loaded_bytes() == mechanism.total_bytes()


def test_modulation_slices_follow_single_image_text_final_layout() -> None:
    model = Chroma(small_chroma_config())
    config = model.config
    values = torch.arange(config.modulation_count, dtype=torch.float32).view(1, -1, 1)
    single = model._single_modulation(values, 2)  # pyright: ignore[reportPrivateUsage]
    assert [part.item() for part in single] == [6.0, 7.0, 8.0]
    image, text = model._double_modulation(values, 1)  # pyright: ignore[reportPrivateUsage]
    assert [[part.item() for part in modulation] for modulation in image] == [
        [15.0, 16.0, 17.0],
        [18.0, 19.0, 20.0],
    ]
    assert [[part.item() for part in modulation] for modulation in text] == [
        [27.0, 28.0, 29.0],
        [30.0, 31.0, 32.0],
    ]
    assert values[:, -2:].flatten().tolist() == [33.0, 34.0]


def test_chroma_circularly_pads_patch_tokens_and_crops_output() -> None:
    model = Chroma(small_chroma_config())
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.01)
    patches: list[torch.Tensor] = []
    model.img_in.register_forward_pre_hook(
        lambda _module, inputs: patches.append(cast(torch.Tensor, inputs[0]).detach().clone())
    )
    latent = torch.arange(30, dtype=torch.float32).view(1, 2, 3, 5)
    output = model(
        latent,
        torch.tensor([0.5]),
        torch.ones(1, 2, 8),
        torch.tensor([3.5]),
    )
    assert output.shape == latent.shape
    assert torch.isfinite(output).all()
    assert patches[0].shape == (1, 6, 8)
    assert patches[0][0, 2].tolist() == [4, 0, 9, 5, 19, 15, 24, 20]
    assert patches[0][0, 5].tolist() == [14, 10, 4, 0, 29, 25, 19, 15]


def test_nerf_dct_embedding_matches_reference_equations() -> None:
    embedder = NerfEmbedder(3, 7, 3)
    initialize(embedder, seed=2)
    inputs = torch.randn(2, 4, 3)
    output = embedder(inputs)
    positions = torch.linspace(0, 1, 2)
    position_y, position_x = torch.meshgrid(positions, positions, indexing="ij")
    frequencies = torch.linspace(0, 2, 3)
    frequency_x = frequencies[None, :, None]
    frequency_y = frequencies[None, None, :]
    basis = (
        torch.cos(position_x.reshape(-1, 1, 1) * frequency_x * torch.pi)
        * torch.cos(position_y.reshape(-1, 1, 1) * frequency_y * torch.pi)
        / (1 + frequency_x * frequency_y)
    ).view(1, 4, 9)
    projection = cast(torch.nn.Linear, embedder.embedder[0])
    expected = F.linear(
        torch.cat((inputs, basis.expand(2, -1, -1)), dim=-1),
        projection.weight,
        projection.bias,
    )
    torch.testing.assert_close(output, expected)


def test_nerf_dct_positions_reuse_the_upstream_four_entry_cache() -> None:
    embedder = NerfEmbedder(3, 7, 3)
    device = torch.device("cpu")
    first = embedder._positions(2, device=device)  # pyright: ignore[reportPrivateUsage]
    assert (
        embedder._positions(2, device=device) is first  # pyright: ignore[reportPrivateUsage]
    )
    for patch_size in range(3, 7):
        embedder._positions(  # pyright: ignore[reportPrivateUsage]
            patch_size, device=device
        )
    assert len(embedder._position_cache) == 4  # pyright: ignore[reportPrivateUsage]
    assert (2, device) not in embedder._position_cache  # pyright: ignore[reportPrivateUsage]


def test_nerf_dynamic_glu_matches_reference_equations() -> None:
    block = NerfGLUBlock(6, 4, 2)
    initialize(block, seed=3)
    inputs = torch.randn(5, 4, 4)
    states = torch.randn(5, 6)
    output = block(inputs, states)
    gate, value, final = block.param_generator(states).chunk(3, dim=-1)
    gate = F.normalize(gate.view(5, 4, 8), dim=-2)
    value = F.normalize(value.view(5, 4, 8), dim=-2)
    final = F.normalize(final.view(5, 8, 4), dim=-2)
    expected = inputs + torch.bmm(
        F.silu(torch.bmm(block.norm(inputs), gate)) * torch.bmm(block.norm(inputs), value),
        final,
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("head", ("linear", "conv"))
def test_radiance_tiled_and_untiled_heads_match_for_batched_inputs(head: str) -> None:
    model = ChromaRadiance(small_radiance_config(nerf_final_head_type=head))
    initialize(model, seed=4)
    pixels = torch.randn(2, 3, 5, 7)
    timestep = torch.tensor([0.25, 0.75])
    context = torch.randn(2, 3, 8)
    guidance = torch.tensor([2.0, 4.0])
    untiled = model(
        pixels,
        timestep,
        context,
        guidance,
        options=ChromaRadianceOptions(nerf_tile_size=0),
    )
    tiled = model(
        pixels,
        timestep,
        context,
        guidance,
        options=ChromaRadianceOptions(nerf_tile_size=1),
    )
    assert tiled.shape == pixels.shape
    torch.testing.assert_close(tiled, untiled, rtol=1e-5, atol=1e-6)


def test_radiance_linear_and_conv_final_layers_match_their_equations() -> None:
    inputs = torch.randn(2, 5, 3, 4)
    linear = NerfFinalLayer(5, 3, operations=INITLESS)
    convolution = NerfFinalLayerConv(5, 3, operations=INITLESS)
    initialize(linear, seed=5)
    initialize(convolution, seed=6)
    linear_expected = F.linear(
        linear.norm(inputs.movedim(1, -1)), linear.linear.weight, linear.linear.bias
    ).movedim(-1, 1)
    convolution_expected = F.conv2d(
        convolution.norm(inputs.movedim(1, -1)).movedim(-1, 1),
        convolution.conv.weight,
        convolution.conv.bias,
        padding=1,
    )
    torch.testing.assert_close(linear(inputs), linear_expected)
    torch.testing.assert_close(convolution(inputs), convolution_expected)


def test_radiance_x0_converts_predictions_to_flow_and_crops_padding() -> None:
    model = ChromaRadiance(small_radiance_config(use_x0=True))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    pixels = torch.arange(45, dtype=torch.float32).view(1, 3, 3, 5)
    output = model(
        pixels,
        torch.tensor([0.5]),
        torch.zeros(1, 2, 8),
        torch.zeros(1),
    )
    torch.testing.assert_close(output, pixels * 2)
    assert output.shape == pixels.shape


def test_radiance_sequential_text_ids_are_variant_or_runtime_selected() -> None:
    reference = torch.zeros(2, 3, 4, 4)
    plain = ChromaRadiance(small_radiance_config())
    _, zero_ids = plain._position_ids(  # pyright: ignore[reportPrivateUsage]
        2, 2, 2, 4, reference
    )
    _, sequential_ids = plain._position_ids(  # pyright: ignore[reportPrivateUsage]
        2, 2, 2, 4, reference, sequential_text=True
    )
    assert zero_ids[:, :, 0].tolist() == [[0.0] * 4] * 2
    assert sequential_ids[:, :, 0].tolist() == [[0.0, 1.0, 2.0, 3.0]] * 2
    assert small_radiance_config(use_sequential_txt_ids=True).use_sequential_txt_ids


def test_pixel_space_codec_uses_radiance_latent_range_and_clamps_decode() -> None:
    codec = PixelSpaceCodec()
    images = torch.tensor([0.0, 0.25, 0.5, 1.0])
    torch.testing.assert_close(codec.encode(images), torch.tensor([-1.0, -0.5, 0.0, 1.0]))
    latents = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    torch.testing.assert_close(codec.decode(latents), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_pixel_space_codec_matches_reference_compute_and_output_dtypes() -> None:
    latent = torch.tensor([0.7725079])
    original = latent.clone()
    expected = latent.to(torch.bfloat16).to(torch.float32).add_(1.0).div_(2.0)
    decoded = PixelSpaceCodec().decode(latent)
    assert torch.equal(decoded, expected)
    assert torch.equal(latent, original)

    encoded = PixelSpaceCodec().encode(torch.tensor([0.3]))
    assert torch.equal(encoded, torch.tensor([-0.4]).to(torch.bfloat16).to(torch.float32))

    decoded_float32 = PixelSpaceCodec(compute_dtype=torch.float32).decode(latent)
    assert torch.equal(decoded_float32, (latent + 1.0) / 2.0)
    assert decoded_float32.data_ptr() != latent.data_ptr()


def test_z_image_pixel_codec_preserves_existing_identity_semantics() -> None:
    codec = ZImagePixelCodec()
    values = torch.tensor([-1.0, 0.0, 1.0])
    assert codec.encode(values) is values
    assert codec.decode(values) is values


def test_radiance_options_validate_tile_size() -> None:
    assert ChromaRadianceOptions().nerf_tile_size is None
    with pytest.raises(ValueError, match="non-negative"):
        ChromaRadianceOptions(nerf_tile_size=-1)


@pytest.mark.parametrize("name", sorted(GOLDENS["layouts"]))
def test_full_size_layout_matches_executed_reference(name: str) -> None:
    with torch.device("meta"):
        if name == "chroma":
            model = Chroma(ChromaConfig(3072, 19, 38, 24))
        else:
            model = ChromaRadiance(
                ChromaRadianceConfig(
                    3072,
                    19,
                    38,
                    24,
                    16,
                    nerf_final_head_type=("conv" if name == "radiance_conv_x0" else "linear"),
                    use_x0=name == "radiance_conv_x0",
                    use_sequential_txt_ids=name == "radiance_sequential",
                )
            )
    observed = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    expected = [(key, list(shape)) for key, shape in GOLDENS["layouts"][name]]
    assert observed == expected


@pytest.mark.parametrize("case", sorted(GOLDENS["cases"]))
def test_forward_matches_executed_reference(case: str) -> None:
    spec = load_platform_golden(GOLDEN_PATH, allow_portable_fallback=True)["cases"][case]
    config = golden_config(spec["config"])
    model = ChromaRadiance(config) if isinstance(config, ChromaRadianceConfig) else Chroma(config)
    entries = [(key, list(shape)) for key, shape in spec["state_dict"]]
    model.load_state_dict(fill_state_dict(entries), strict=True)
    channels = config.latent_channels
    latent = hashed_input(f"{case}:x", (spec["batch"], channels, spec["height"], spec["width"]))
    timestep = torch.tensor(spec["timestep"], dtype=torch.float32)
    context = hashed_input(
        f"{case}:context",
        (spec["batch"], spec["context_len"], config.context_in_dim),
    )
    guidance = torch.tensor(spec["guidance"], dtype=torch.float32)
    outputs = {
        "default": model(latent, timestep, context, guidance),
    }
    with torch.inference_mode():
        inference_output = model(latent, timestep, context, guidance)
    torch.testing.assert_close(inference_output, outputs["default"], rtol=0.0, atol=0.0)
    if isinstance(model, ChromaRadiance) and "tiled" in spec["outputs"]:
        outputs["tiled"] = model(
            latent,
            timestep,
            context,
            guidance,
            options=ChromaRadianceOptions(nerf_tile_size=1),
        )
        outputs["sequential"] = model(
            latent,
            timestep,
            context,
            guidance,
            options=ChromaRadianceOptions(force_sequential_txt_ids=True),
        )
    for name, observed in outputs.items():
        assert_reference_tensor(
            observed,
            decode_golden(spec["outputs"][name]),
            rtol=0.0,
            atol=0.0,
        )
