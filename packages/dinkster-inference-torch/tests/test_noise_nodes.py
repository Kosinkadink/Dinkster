"""Noise generation/injection nodes against goldens from pinned KJNodes.

The goldens in goldens/kj_noise_3f200542.json are produced end to end by
the pinned upstream GenerateNoise and InjectNoiseToLatent nodes
(tools/gen_kj_noise_goldens.py). Ordinary hosts enforce portable tensor
contracts; CUDA reference validation additionally enforces the recorded
values. This avoids binding std reductions and bilinear interpolation to one
CPU microarchitecture.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_inference import LatentDescriptor, MultiStreamLatentDescriptor
from golden_files import assert_reference_tensor, load_platform_golden

ROOT = Path(__file__).resolve().parents[3]
for source in sorted((ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(source))

native_arm = importlib.import_module("dinkster_compat_comfy.native_arm")

GOLDEN_PATH = Path(__file__).parent / "goldens" / "kj_noise_3f200542.json"
GOLDEN = load_platform_golden(GOLDEN_PATH, allow_portable_fallback=True)


def _tensor(record: dict[str, Any]) -> torch.Tensor:
    values = torch.tensor([float(value) for value in record["values"]], dtype=torch.float32)
    return values.reshape(record["shape"])


def _source(name: str) -> torch.Tensor:
    return _tensor(GOLDEN["sources"][name])


@dataclass(frozen=True)
class _StubHandle:
    runtime: Any

    def require_active(self) -> None:
        return None


def _model(latent: object) -> object:
    handle = _StubHandle(SimpleNamespace(family=SimpleNamespace(latent=latent)))
    return native_arm._NativeModelOverlay(handle, (), {}, None, None, ())


def _plain_model(scale_factor: float) -> object:
    return _model(LatentDescriptor(channels=4, scale_factor=scale_factor))


def test_generate_noise_matches_kj_goldens() -> None:
    for name, case in GOLDEN["generate_cases"].items():
        kwargs: dict[str, Any] = dict(case["inputs"])
        if "sigmas" in case:
            kwargs["sigmas"] = native_arm._CustomSigmasValue(
                tuple(float(value) for value in case["sigmas"])
            )
            kwargs["model"] = _plain_model(float(case["scale_factor"]))
        result = native_arm.GenerationLatentGenerateNoise.execute(**kwargs)["latent"]
        assert set(result) == {"samples"}, name
        assert_reference_tensor(result["samples"], _tensor(case["result"]))


def test_inject_noise_matches_kj_goldens() -> None:
    for name, case in GOLDEN["inject_cases"].items():
        result = native_arm.GenerationLatentInjectNoise.execute(
            latents={"samples": _source(case["latents"]), "noise_mask": _source(case["latents"])},
            noise={"samples": _source(case["noise"])},
            mask=None if case["mask"] is None else _source(case["mask"]),
            **case["inputs"],
        )["latent"]
        # The reference returns a fresh {"samples"} dict, dropping metadata
        # such as noise_mask; the port preserves that exactly.
        assert set(result) == {"samples"}, name
        assert_reference_tensor(result["samples"], _tensor(case["result"]))


def test_generate_noise_refuses_constant_batch_for_5d_shapes() -> None:
    for shape in ("BCTHW", "BTCHW"):
        with pytest.raises(ValueError, match="constant_batch_noise"):
            native_arm.GenerationLatentGenerateNoise.execute(
                width=64,
                height=48,
                batch_size=2,
                seed=123,
                multiplier=1.0,
                constant_batch_noise=True,
                normalize=False,
                shape=shape,
            )


def test_generate_noise_sigma_path_refusals() -> None:
    kwargs: dict[str, Any] = {
        "width": 64,
        "height": 48,
        "batch_size": 1,
        "seed": 123,
        "multiplier": 1.0,
        "constant_batch_noise": False,
        "normalize": False,
    }
    sigmas = native_arm._CustomSigmasValue((12.25, 0.75))
    with pytest.raises(TypeError, match="sigmas"):
        native_arm.GenerationLatentGenerateNoise.execute(
            **kwargs, sigmas=torch.tensor([1.0]), model=_plain_model(1.0)
        )
    with pytest.raises(ValueError, match="model"):
        native_arm.GenerationLatentGenerateNoise.execute(**kwargs, sigmas=sigmas)
    with pytest.raises(ValueError, match="at least one"):
        native_arm.GenerationLatentGenerateNoise.execute(
            **kwargs, sigmas=native_arm._CustomSigmasValue(()), model=_plain_model(1.0)
        )
    multistream = MultiStreamLatentDescriptor(
        (
            ("video", LatentDescriptor(channels=4)),
            ("audio", LatentDescriptor(channels=2, dimensions=1)),
        )
    )
    with pytest.raises(ValueError, match="plain-latent"):
        native_arm.GenerationLatentGenerateNoise.execute(
            **kwargs, sigmas=sigmas, model=_model(multistream)
        )


def test_inject_noise_refuses_non_rank4_mask_blend() -> None:
    video = torch.zeros((1, 4, 3, 4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="rank-4"):
        native_arm.GenerationLatentInjectNoise.execute(
            latents={"samples": video},
            strength=0.1,
            noise={"samples": video.clone()},
            normalize=False,
            average=False,
            mask=torch.zeros((1, 4, 4), dtype=torch.float32),
        )
    with pytest.raises(TypeError, match="mask"):
        native_arm.GenerationLatentInjectNoise.execute(
            latents={"samples": torch.zeros((1, 4, 4, 4))},
            strength=0.1,
            noise={"samples": torch.zeros((1, 4, 4, 4))},
            normalize=False,
            average=False,
            mask=[[0.0]],
        )
