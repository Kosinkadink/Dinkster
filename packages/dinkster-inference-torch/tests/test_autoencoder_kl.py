"""Stage 5 slice 2: the native SD/SDXL AutoencoderKL.

Every golden in goldens/kl_goldens.json was produced by RUNNING the
reference comfy.ldm.models.autoencoder.AutoencoderKL @ the audited
baseline (tools/gen_kl_goldens.py). The golden pins the reference's
exact state-dict (key, shape) listing plus encode/decode outputs;
weights come from the shared deterministic hash (kl_fill.py), so both
sides run bit-identical parameters without storing megabytes.

The vertical test goes file bytes -> torch-free header parse ->
detect_kl_config -> construct -> load_tensors -> strict load ->
encode/decode, proving the whole loading path against the executed
reference with no safetensors package anywhere.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference import (
    FLOAT32,
    KL_BATCH_NORM_EPS,
    KLMemoryEstimator,
    TensorGeometry,
    detect_kl_config,
)
from dinkster_inference.sources import load_safetensors_header
from dinkster_inference_torch import (
    INITLESS,
    AutoencoderKL,
    DiagonalGaussian,
    crop_to_multiple,
    enroll_component,
    kl_codec_plugin,
    load_tensors,
    process_input,
    process_output,
    select_attention,
)
from kl_fill import fill_state_dict

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "kl_goldens.json").read_text())

CASES = sorted(GOLDENS["cases"])


def dec(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def golden_entries(case: str) -> list[tuple[str, list[int]]]:
    return [(key, list(shape)) for key, shape in GOLDENS["cases"][case]["state_dict"]]


def build_model(case: str) -> AutoencoderKL:
    """Detect from the reference listing, construct, fill, load."""
    geometries = {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in golden_entries(case)}
    model = AutoencoderKL(detect_kl_config(geometries))
    model.load_state_dict(fill_state_dict(golden_entries(case)), strict=True)
    return model


# ------------------------------------------------------ key layout


@pytest.mark.parametrize("case", CASES)
def test_state_dict_layout_matches_executed_reference(case: str) -> None:
    geometries = {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in golden_entries(case)}
    model = AutoencoderKL(detect_kl_config(geometries))
    ours = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert ours == golden_entries(case)


def test_vae_injects_one_kernel_without_changing_state_or_output() -> None:
    baseline = build_model("x4")
    spy = CallableModuleKernel(select_attention("vae").kernel)
    geometries = {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in golden_entries("x4")}
    model = AutoencoderKL(detect_kl_config(geometries), attention_kernel=spy)
    model.load_state_dict(baseline.state_dict(), strict=True)
    content = dec(GOLDENS["cases"]["x4"]["input"])
    assert set(model.state_dict()) == set(baseline.state_dict())
    assert_kernel_is_not_model_state(model, spy)
    torch.testing.assert_close(model.encode(content), baseline.encode(content))
    assert spy.calls
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)


# ---------------------------------------------------- golden replay


@pytest.mark.parametrize("case", CASES)
def test_encode_matches_executed_reference(case: str) -> None:
    model = build_model(case)
    content = dec(GOLDENS["cases"][case]["input"])
    latent = model.encode(content)
    torch.testing.assert_close(latent, dec(GOLDENS["cases"][case]["latent"]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("case", CASES)
def test_decode_matches_executed_reference(case: str) -> None:
    model = build_model(case)
    latent = dec(GOLDENS["cases"][case]["latent"])
    decoded = model.decode(latent)
    torch.testing.assert_close(
        decoded, dec(GOLDENS["cases"][case]["decoded"]), rtol=1e-4, atol=1e-5
    )


def test_nonsquare_odd_multiple_shapes() -> None:
    """Any content whose sides are multiples of the downscale is
    legal; latent and reconstruction shapes follow the descriptor."""
    model = build_model("standard")
    content = torch.zeros(1, 3, 24, 40)
    latent = model.encode(content)
    assert latent.shape == (1, 3, 3, 5)
    assert model.decode(latent).shape == (1, 3, 24, 40)


# ------------------------------------------------- batch-norm latent


def test_batch_norm_latent_shape_and_descriptor() -> None:
    """The packed latent has 4x the embed channels on a 2x halved
    grid; the descriptor and plugin expose the external geometry."""
    model = build_model("batch_norm")
    content = dec(GOLDENS["cases"]["batch_norm"]["input"])
    assert content.shape == (1, 3, 32, 48)
    latent = model.encode(content)
    assert latent.shape == (1, 16, 2, 3)
    assert model.decode(latent).shape == (1, 3, 32, 48)

    plugin = kl_codec_plugin(model)
    assert plugin.descriptor.latent.channels == 16
    assert plugin.descriptor.latent.spatial_downscale == 16
    memory = plugin.memory
    assert isinstance(memory, KLMemoryEstimator)
    assert memory.decode_multiplier == 4.0
    baseline = kl_codec_plugin(build_model("standard")).memory
    assert isinstance(baseline, KLMemoryEstimator)
    geometry = TensorGeometry(tuple(latent.shape), FLOAT32)
    assert memory.decode_bytes(geometry) == 4 * baseline.decode_bytes(geometry)


def test_pack_unpack_latent_round_trips() -> None:
    model = build_model("batch_norm")
    latent = torch.arange(1 * 4 * 4 * 6, dtype=torch.float32).reshape(1, 4, 4, 6) / 10.0
    packed = model.pack_latent(latent)
    assert packed.shape == (1, 16, 2, 3)
    torch.testing.assert_close(model.unpack_latent(packed), latent, rtol=1e-5, atol=1e-6)


def test_batch_norm_latent_enrolls_and_leases_frozen_statistics() -> None:
    """The frozen bn owns state, so it needs a residency route:
    enrollment must accept it, prefetch must request exactly the two
    running statistics (num_batches_tracked is int64, above the
    float32 materialization ceiling), and the offloaded pack/unpack
    must reproduce the resident outputs exactly."""
    model = build_model("batch_norm")
    bn = model.bn
    assert bn is not None
    assert bn.running_mean is not None and bn.running_var is not None
    bn.running_mean.copy_(torch.linspace(-0.5, 0.7, 16))
    bn.running_var.copy_(torch.linspace(0.3, 1.9, 16))
    latent = torch.arange(1 * 4 * 4 * 6, dtype=torch.float32).reshape(1, 4, 4, 6) / 10.0
    resident_packed = model.pack_latent(latent)
    resident_unpacked = model.unpack_latent(resident_packed)

    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.unload()
    prefetch = bn.residency_prefetch()
    assert prefetch is not None
    assert [key for key, _ in prefetch[1]] == ["bn.running_mean", "bn.running_var"]

    packed = model.pack_latent(latent)
    assert torch.equal(packed, resident_packed)
    assert torch.equal(model.unpack_latent(packed), resident_unpacked)


def test_pack_latent_channel_order_is_the_reference_rearrange() -> None:
    """One 2x2 patch per channel: the packed channel index runs
    channel-major, then patch row, then patch column - the
    reference's ``c (i pi) (j pj) -> (c pi pj) i j``."""
    model = build_model("batch_norm")
    bn = model.bn
    assert bn is not None
    assert bn.running_mean is not None and bn.running_var is not None
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0 - KL_BATCH_NORM_EPS)  # identity normalize
    latent = torch.zeros(1, 4, 2, 2)
    for channel in range(4):
        latent[0, channel] = torch.tensor([[0.0, 1.0], [2.0, 3.0]]) + 10.0 * channel
    packed = model.pack_latent(latent)
    assert packed.shape == (1, 16, 1, 1)
    expected = torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat(4) + 10.0 * torch.arange(
        4, dtype=torch.float32
    ).repeat_interleave(4)
    torch.testing.assert_close(packed.flatten(), expected)


# ------------------------------------------------------- posterior


def test_encode_is_posterior_mode() -> None:
    model = build_model("x4")
    content = dec(GOLDENS["cases"]["x4"]["input"])
    posterior = model.encode_posterior(content)
    torch.testing.assert_close(model.encode(content), posterior.mode())
    torch.testing.assert_close(posterior.mode(), posterior.mean)


def test_posterior_sample_takes_explicit_generator() -> None:
    model = build_model("x4")
    content = dec(GOLDENS["cases"]["x4"]["input"])
    posterior = model.encode_posterior(content)
    a = posterior.sample(torch.Generator().manual_seed(7))
    b = posterior.sample(torch.Generator().manual_seed(7))
    c = posterior.sample(torch.Generator().manual_seed(8))
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, c)
    assert not torch.equal(a, posterior.mode())


def test_posterior_logvar_is_clamped() -> None:
    params = torch.cat([torch.zeros(1, 2, 2, 2), torch.full((1, 2, 2, 2), 99.0)], dim=1)
    posterior = DiagonalGaussian.from_parameters(params)
    assert posterior.logvar.max().item() == 20.0
    params = torch.cat([torch.zeros(1, 2, 2, 2), torch.full((1, 2, 2, 2), -99.0)], dim=1)
    assert DiagonalGaussian.from_parameters(params).logvar.min().item() == -30.0


# --------------------------------------------------- wrapper boundary


def test_process_input_is_out_of_place() -> None:
    content = torch.tensor([0.0, 0.5, 1.0])
    out = process_input(content)
    torch.testing.assert_close(out, torch.tensor([-1.0, 0.0, 1.0]))
    torch.testing.assert_close(content, torch.tensor([0.0, 0.5, 1.0]))


def test_process_output_is_in_place_and_clamped() -> None:
    content = torch.tensor([-3.0, -1.0, 0.0, 1.0, 5.0])
    out = process_output(content)
    assert out is content  # reference semantics: mutates its argument
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_codec_plugin_applies_wrapper_transforms() -> None:
    """plugin.encode/decode ARE the reference VAE wrapper: content in
    [0,1] on both sides of the golden's [-1,1] model boundary."""
    model = build_model("x4")
    plugin = kl_codec_plugin(model)
    assert plugin.id == "dinkster.autoencoder_kl"
    assert plugin.descriptor.latent.spatial_downscale == 4

    content = (dec(GOLDENS["cases"]["x4"]["input"]) + 1.0) / 2.0
    torch.testing.assert_close(
        plugin.encode(content),
        dec(GOLDENS["cases"]["x4"]["latent"]),
        rtol=1e-4,
        atol=1e-5,
    )
    expected = dec(GOLDENS["cases"]["x4"]["decoded"]).add_(1.0).div_(2.0)
    torch.testing.assert_close(
        plugin.decode(dec(GOLDENS["cases"]["x4"]["latent"])),
        expected.clamp_(0.0, 1.0),
        rtol=1e-4,
        atol=1e-5,
    )


def test_codec_plugin_tiled_decode_matches_direct() -> None:
    """When every sweep variant covers the latent in ONE tile (the KL
    decoder is not spatially local, so genuine tiling only
    approximates direct), tiled output must equal direct exactly -
    including the content_out clamp placement after the average. The
    4x6 latent needs tile (8,12): the halved variants (4,24)/(16,6)
    still cover it."""
    model = build_model("x4")
    plugin = kl_codec_plugin(model)
    latent = dec(GOLDENS["cases"]["x4"]["latent"])
    tiled = plugin.decode_tiled(latent, tile=(8, 12), overlap=(2, 2))
    torch.testing.assert_close(tiled, plugin.decode(latent), rtol=1e-4, atol=1e-5)


def test_crop_to_multiple_is_the_reference_center_crop() -> None:
    """comfy/sd.py vae_encode_crop_pixels @ 947c2749: each spatial
    dim narrows to a floor-multiple, offset (size % multiple) // 2."""
    content = torch.arange(7 * 9, dtype=torch.float32).reshape(1, 1, 7, 9)
    cropped = crop_to_multiple(content, 4)
    torch.testing.assert_close(cropped, content[:, :, 1:5, 0:8])

    exact = torch.zeros(1, 3, 8, 12)
    assert crop_to_multiple(exact, 4) is exact  # no-op on the grid

    with pytest.raises(ValueError, match="smaller than one 4x"):
        crop_to_multiple(torch.zeros(1, 3, 3, 12), 4)


def test_codec_plugin_encode_crops_non_grid_content() -> None:
    """Encode of non-divisible content matches the reference wrapper:
    center-crop to the downscale grid, then process_input, then the
    model - never the encoder's floor-truncating downsamplers."""
    model = build_model("x4")
    plugin = kl_codec_plugin(model)
    content = torch.rand(1, 3, 23, 18)  # x4 grid: crops to 20 x 16
    latent = plugin.encode(content)
    assert latent.shape == (1, model.config.embed_dim, 5, 4)
    torch.testing.assert_close(
        latent,
        model.encode(process_input(crop_to_multiple(content, 4))),
    )


def test_codec_plugin_tiled_encode_handles_non_grid_content() -> None:
    """Tiled encode of odd dimensions must not desync plan and
    encoder geometry (23 would plan round(23/4)=6 latents while the
    floor path emits 5). Tile (40, 32) makes every sweep variant a
    single covering tile, so tiled equals direct exactly."""
    model = build_model("x4")
    plugin = kl_codec_plugin(model)
    content = torch.rand(1, 3, 23, 18)
    tiled = plugin.encode_tiled(content, tile=(40, 32), overlap=(4, 4))
    torch.testing.assert_close(tiled, plugin.encode(content), rtol=1e-4, atol=1e-5)


# ------------------------------------------------------- training


def test_gradients_flow_through_encode_decode() -> None:
    model = build_model("x4")
    content = dec(GOLDENS["cases"]["x4"]["input"]).requires_grad_(True)
    loss = model.decode(model.encode(content)).square().mean()
    loss.backward()
    grad = content.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0.0
    conv_grad = model.encoder.conv_in.weight.grad
    assert conv_grad is not None
    assert torch.isfinite(conv_grad).all()


def test_gradients_flow_through_stochastic_encode() -> None:
    model = build_model("x4")
    content = dec(GOLDENS["cases"]["x4"]["input"]).requires_grad_(True)
    posterior = model.encode_posterior(content)
    sample = posterior.sample(torch.Generator().manual_seed(3))
    sample.sum().backward()
    assert content.grad is not None
    assert torch.isfinite(content.grad).all()


# ------------------------------------------------------- operations


def test_initless_reset_parameters_is_a_no_op() -> None:
    conv = INITLESS.conv2d(2, 3, 3, stride=1, padding=1)
    conv.weight.data.fill_(7.0)
    conv.reset_parameters()
    assert (conv.weight == 7.0).all()
    norm = INITLESS.group_norm(64)
    norm.weight.data.fill_(5.0)
    norm.reset_parameters()
    assert (norm.weight == 5.0).all()


def test_initless_factories_configure_reference_layers() -> None:
    conv = INITLESS.conv2d(4, 8, 3, stride=2, padding=0)
    assert isinstance(conv, torch.nn.Conv2d)
    assert conv.stride == (2, 2)
    assert conv.padding == (0, 0)
    norm = INITLESS.group_norm(64)
    assert isinstance(norm, torch.nn.GroupNorm)
    assert norm.num_groups == 32
    assert norm.eps == 1e-6
    assert norm.affine


# ------------------------------------------------------ full vertical


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.contiguous()
    return bytes(tensor.untyped_storage())[: tensor.numel() * tensor.element_size()]


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    """A real float32 safetensors file from stdlib only (the format
    is a JSON header plus packed little-endian payload bytes)."""
    header: dict[str, object] = {}
    payload = bytearray()
    for key, tensor in tensors.items():
        data = tensor_bytes(tensor)
        header[key] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(payload))
    return path


def test_full_vertical_file_to_pixels(tmp_path: Path) -> None:
    """safetensors bytes -> torch-free detect -> construct -> payload
    reads -> strict load -> encode/decode vs the executed reference."""
    path = write_safetensors(
        tmp_path / "kl_x4.safetensors",
        fill_state_dict(golden_entries("x4")),
    )
    source = load_safetensors_header(path)
    config = detect_kl_config({key: source.entry(key).geometry for key in source.keys()})
    model = AutoencoderKL(config)
    model.load_state_dict(load_tensors(path), strict=True)

    content = dec(GOLDENS["cases"]["x4"]["input"])
    torch.testing.assert_close(
        model.encode(content),
        dec(GOLDENS["cases"]["x4"]["latent"]),
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        model.decode(dec(GOLDENS["cases"]["x4"]["latent"])),
        dec(GOLDENS["cases"]["x4"]["decoded"]),
        rtol=1e-4,
        atol=1e-5,
    )
