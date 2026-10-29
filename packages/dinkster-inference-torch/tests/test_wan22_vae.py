"""Reduced CPU and executed-reference proofs for the Wan 2.2 VAE."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference.wan22_vae import Wan22VAEConfig, wan22_vae_layout
from dinkster_inference_torch import select_attention
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import (
    INITLESS,
    CastOperations,
    Operations,
    ResidencyRouted,
)
from dinkster_inference_torch.residency import WeightLease
from dinkster_inference_torch.residency_timing import PartialResidencyTiming
from dinkster_inference_torch.wan22_vae import (
    LATENTS_MEAN,
    LATENTS_STD,
    Wan22VAE,
    count_conv3d,
    patchify,
    unpatchify,
)
from kl_fill import fill_state_dict

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "wan22_vae_goldens.json").read_text())


def _config() -> Wan22VAEConfig:
    config = GOLDENS["config"]
    return Wan22VAEConfig(
        dim=config["dim"],
        decoder_dim=config["decoder_dim"],
        z_dim=config["z_dim"],
        dim_mult=tuple(config["dim_mult"]),
        num_res_blocks=config["num_res_blocks"],
        attn_scales=tuple(config["attn_scales"]),
        temporal_downsample=tuple(config["temporal_downsample"]),
        image_channels=config["image_channels"],
        conv_out_channels=config["conv_out_channels"],
        patch_size=config["patch_size"],
        dropout=config["dropout"],
    )


def _tensor(name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    payload = GOLDENS[name]
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"]).to(dtype)


def _model(
    operations: Operations = INITLESS,
    *,
    storage_dtype: torch.dtype = torch.float32,
) -> Wan22VAE:
    model = Wan22VAE(_config(), operations=operations)
    state = {
        key: value.to(storage_dtype)
        for key, value in fill_state_dict(GOLDENS["state_dict"]).items()
    }
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model


def _prefetch_requests(model: Wan22VAE) -> dict[str, torch.dtype | None]:
    requests: dict[str, torch.dtype | None] = {}
    for module in model.modules():
        if not isinstance(module, ResidencyRouted):
            continue
        prefetch = module.residency_prefetch()
        if prefetch is None:
            continue
        for key, dtype in prefetch[1]:
            assert key not in requests
            requests[key] = dtype
    return requests


def test_golden_provenance_and_reduced_state_layout_are_exact() -> None:
    assert GOLDENS["reference"]["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert GOLDENS["reference"]["file"] == "comfy/ldm/wan/vae2_2.py"
    assert GOLDENS["reference"]["configuration_file"] == "comfy/sd.py"
    model = Wan22VAE(_config())
    actual = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    assert len(actual) == 148
    assert actual == [(key, shape) for key, shape in GOLDENS["state_dict"]]


def test_default_meta_state_matches_the_official_196_tensor_layout() -> None:
    with torch.device("meta"):
        model = Wan22VAE()
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == (
        wan22_vae_layout()
    )


def test_normalization_constants_and_roundtrip_are_exact() -> None:
    model = Wan22VAE(replace(_config(), z_dim=48))
    assert torch.equal(model.latents_mean, torch.tensor(LATENTS_MEAN))
    assert torch.equal(model.latents_std, torch.tensor(LATENTS_STD))
    latent = torch.arange(48, dtype=torch.float64).reshape(1, 48, 1, 1, 1)
    torch.testing.assert_close(model.process_out(model.process_in(latent)), latent)


def test_patchify_uses_reference_channel_order_and_roundtrips() -> None:
    content = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
    packed = patchify(content, 2)
    assert packed.shape == (1, 4, 1, 1, 1)
    assert torch.equal(packed.flatten(), torch.tensor([1.0, 3.0, 2.0, 4.0]))
    assert torch.equal(unpatchify(packed, 2), content)


def test_encode_decode_match_executed_comfyui_goldens_and_chunk_plan() -> None:
    model = _model()
    encoder_chunks: list[int] = []
    decoder_chunks: list[int] = []
    encoder_handle = model.encoder.register_forward_pre_hook(
        lambda _module, args: encoder_chunks.append(args[0].shape[2])
    )
    decoder_handle = model.decoder.register_forward_pre_hook(
        lambda _module, args: decoder_chunks.append(args[0].shape[2])
    )
    try:
        with torch.no_grad():
            encoded = model.encode(_tensor("content"))
            decoded = model.decode(_tensor("latent"))
    finally:
        encoder_handle.remove()
        decoder_handle.remove()
    torch.testing.assert_close(encoded, _tensor("encoded"), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(decoded, _tensor("decoded"), rtol=1e-5, atol=1e-6)
    assert encoder_chunks == [1, 4]
    assert decoder_chunks == [1, 1]
    assert encoded.shape == (1, 2, 2, 1, 1)
    assert decoded.shape == (1, 3, 5, 16, 16)


def test_cache_layer_counts_cover_every_causal_convolution() -> None:
    model = Wan22VAE(_config())
    assert count_conv3d(model.encoder) == 18
    assert count_conv3d(model.decoder) == 26


def test_attention_uses_the_shared_injected_vae_seam_without_state() -> None:
    baseline = _model()
    spy = CallableModuleKernel(select_attention("vae").kernel)
    model = Wan22VAE(_config(), attention_kernel=spy)
    model.load_state_dict(baseline.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            model.encode(_tensor("content")), baseline.encode(_tensor("content"))
        )
        torch.testing.assert_close(
            model.decode(_tensor("latent")), baseline.decode(_tensor("latent"))
        )
    assert len(spy.calls) == 4
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)
    assert_kernel_is_not_model_state(model, spy)


@pytest.mark.parametrize(
    ("operations", "storage_dtype", "compute_dtype"),
    (
        (INITLESS, torch.float32, torch.float32),
        (CastOperations(torch.bfloat16), torch.float16, torch.bfloat16),
    ),
    ids=("resident-dtype", "cast-at-use"),
)
def test_every_state_owner_prefetches_and_offloaded_execution_is_exact(
    operations: Operations,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> None:
    model = _model(operations, storage_dtype=storage_dtype)
    owners = [module for module in model.modules() if tuple(module.parameters(recurse=False))]
    assert owners
    assert all(isinstance(module, ResidencyRouted) for module in owners)
    assert {value.dtype for value in model.state_dict().values()} == {storage_dtype}

    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    content = _tensor("content", dtype=compute_dtype)
    latent = _tensor("latent", dtype=compute_dtype)
    with torch.no_grad():
        resident_encode = model.encode(content.clone())
        resident_decode = model.decode(latent.clone())
    assert resident_encode.dtype == compute_dtype
    assert resident_decode.dtype == compute_dtype
    assert _prefetch_requests(model) == {}

    mechanism.unload()
    assert _prefetch_requests(model) == {key: compute_dtype for key in model.state_dict()}
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
    with torch.no_grad():
        offloaded_encode = model.encode(content.clone())
        offloaded_decode = model.decode(latent.clone())
    assert torch.equal(offloaded_encode, resident_encode)
    assert torch.equal(offloaded_decode, resident_decode)
    assert {key for key, _dtype in requested} == set(model.state_dict())
    assert all(dtype == compute_dtype for _key, dtype in requested)
    assert mechanism.loaded_bytes() == 0


@pytest.mark.parametrize(
    "content",
    (
        torch.empty(1, 3, 1, 16),
        torch.empty(1, 2, 1, 16, 16),
        torch.empty(1, 3, 0, 16, 16),
        torch.empty(1, 3, 1, 15, 16),
    ),
)
def test_encode_refuses_invalid_geometry_before_model_work(content: torch.Tensor) -> None:
    model = Wan22VAE(_config())
    called: list[bool] = []
    handle = model.encoder.register_forward_pre_hook(lambda _module, _args: called.append(True))
    try:
        with pytest.raises(ValueError):
            model.encode(content)
    finally:
        handle.remove()
    assert called == []
