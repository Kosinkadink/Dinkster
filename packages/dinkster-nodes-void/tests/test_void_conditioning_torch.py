"""CPU contracts for native VOID conditioning and warped noise.

Hand-derived from nodes_void.py at ComfyUI
bc5f880dbdd43e426077f42ccec53bb577ffee9f. Threshold bands quantize
to 0/63/127/255 before inversion; lengths 1/9 become 5, while 13 stays
13. Controlled encoders select channel zero every four frames/eight pixels,
then repeat the last encoded frame or truncate to match the target time.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import torch
from dinkster_api.v1 import (
    materialize_latent,
)
from dinkster_void.void_conditioning import (
    VOIDQuadmaskPreprocess,
    VOIDWarpedNoiseSource,
)


def test_isolated_void_manifest_installs_only_consumed_codecs() -> None:
    from dinkster_void import register_types
    from dinkster_void.void_sampler import VOIDSampler
    from dinkster_workers import load_manifest
    from dinkster_workers.host import load_pack

    manifest = load_manifest(Path(__file__).resolve().parents[1] / "dinkster-pack.toml")
    worker, registry, _, _ = load_pack(manifest)
    required = {
        "CONDITIONING",
        "IMAGE",
        "LATENT",
        "MASK",
        "NOISE",
        "OPTICAL_FLOW",
        "SAMPLER",
        "VAE",
    }
    for type_id in required:
        assert registry.spec(type_id).declared_codec, type_id
        if type_id == "OPTICAL_FLOW":
            assert type_id not in worker.registered_types
            assert registry.owner_of(type_id) == "dinkster-video"
        else:
            assert type_id in worker.registered_types
        assert registry.owner_of(type_id) != manifest.name
    for type_id in ("IC_LORA_PARAMETERS", "WAN_CAMERA_EMBEDDING", "TRACKS", "INTERP_MODEL", "CLIP"):
        assert type_id not in registry, type_id
    register_types(registry)
    latent = {"samples": torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)}
    codec = registry.spec("LATENT")
    decoded = materialize_latent(codec.decode(codec.encode(latent)))
    torch.testing.assert_close(decoded["samples"], latent["samples"], rtol=0, atol=0)
    sampler = VOIDSampler.execute()["SAMPLER"]
    codec = registry.spec("SAMPLER")
    assert codec.decode(codec.encode(sampler)) is sampler
    source = VOIDWarpedNoiseSource.execute(warped_noise=latent)["NOISE"]
    codec = registry.spec("NOISE")
    decoded_noise = cast(Any, codec.decode(codec.encode(source)))
    torch.testing.assert_close(
        decoded_noise.generate_noise(latent), latent["samples"], rtol=0, atol=0
    )
    for type_id in ("OPTICAL_FLOW", "VAE"):
        resident = object()
        codec = registry.spec(type_id)
        assert codec.decode(codec.encode(resident)) is resident


def test_quadmask_threshold_edges_and_normalized_input_preserve_source() -> None:
    source = torch.tensor([0, 31, 32, 95, 96, 191, 192, 255], dtype=torch.float32)
    expected = (255 - torch.tensor([0, 0, 63, 63, 127, 127, 255, 255])) / 255
    original = source.clone()
    for mask in (source.reshape(1, 2, 4), (source / 255).reshape(1, 2, 4)):
        result = VOIDQuadmaskPreprocess.execute(mask=mask)["quadmask"]
        torch.testing.assert_close(result.flatten(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)


def test_quadmask_dilation_is_spatial_and_keeps_other_frames() -> None:
    mask = torch.full((2, 4, 5), 255.0)
    mask[0, 1, 2] = 63
    expected = torch.zeros_like(mask)
    expected[0, :3, 1:4] = 1
    torch.testing.assert_close(
        VOIDQuadmaskPreprocess.execute(mask=mask, dilate_width=1)["quadmask"],
        expected,
        rtol=0,
        atol=0,
    )
    assert mask[0, 1, 2] == 63 and torch.count_nonzero(mask == 63) == 1


def test_warped_noise_source_returns_independent_cpu_noise() -> None:
    source = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 2, 4)
    noise = VOIDWarpedNoiseSource.execute(warped_noise={"samples": source})["NOISE"]
    first = noise.generate_noise({"samples": torch.ones_like(source)})
    assert noise.seed == 0 and first.device.type == "cpu"
    torch.testing.assert_close(first, source, rtol=0, atol=0)
    first.zero_()
    torch.testing.assert_close(noise.generate_noise(None), source, rtol=0, atol=0)
    assert source.flatten()[-1] == 47


def test_void_discovery_is_torch_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_void.void_conditioning import "
            "VOIDQuadmaskPreprocess, VOIDInpaintConditioning, VOIDWarpedNoiseSource; "
            "[node.schema() for node in (VOIDQuadmaskPreprocess, VOIDInpaintConditioning, "
            "VOIDWarpedNoiseSource)]; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stderr
