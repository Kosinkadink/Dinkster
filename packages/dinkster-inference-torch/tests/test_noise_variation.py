"""Variation noise against goldens executed by pinned ComfyUI-Inspire-Pack.

The goldens in goldens/inspire_variation_d23db9aa.json are produced by the
reference prepare_noise / mix_noise from inspire/libs/utils.py at the pinned
commit (tools/gen_inspire_variation_goldens.py). Ordinary hosts enforce
portable tensor contracts; CUDA reference validation additionally enforces the
recorded values. This avoids binding norm/acos/sin kernels to one CPU
microarchitecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference_torch.noise_variation import (
    NoiseVariationError,
    mix_variation_noise,
    prepare_variation_noise,
)
from golden_files import assert_reference_tensor, load_platform_golden

GOLDEN_PATH = Path(__file__).parent / "goldens" / "inspire_variation_d23db9aa.json"
GOLDEN = load_platform_golden(GOLDEN_PATH, allow_portable_fallback=True)


def _tensor(record: dict[str, Any]) -> torch.Tensor:
    values = torch.tensor([float(value) for value in record["values"]], dtype=torch.float32)
    return values.reshape(record["shape"])


def test_prepare_variation_noise_matches_inspire_goldens() -> None:
    for _name, case in GOLDEN["prepare_cases"].items():
        inputs = case["inputs"]
        latent = torch.zeros((inputs["batch"], 4, 6, 8), dtype=torch.float32)
        noise_inds = inputs["noise_inds"]
        result = prepare_variation_noise(
            latent,
            inputs["seed"],
            noise_inds=None if noise_inds is None else [int(index) for index in noise_inds],
            batch_seed_mode=inputs["batch_seed_mode"],
            variation_seed=inputs["variation_seed"],
            variation_strength=inputs["variation_strength"],
            variation_method=inputs["variation_method"],
        )
        assert_reference_tensor(result, _tensor(case["result"]))


def test_noise_inds_path_ignores_variation_strength() -> None:
    skip = GOLDEN["prepare_cases"]["comfy-inds-skip"]
    ignored = GOLDEN["prepare_cases"]["comfy-inds-variation-ignored"]
    assert skip["inputs"]["variation_strength"] == 0.0
    assert ignored["inputs"]["variation_strength"] > 0.0
    assert ignored["result"] == skip["result"]


def test_mix_variation_noise_matches_inspire_goldens() -> None:
    for _name, case in GOLDEN["mix_cases"].items():
        inputs = case["inputs"]
        result = mix_variation_noise(
            _tensor(GOLDEN["mix_sources"][inputs["low"]]),
            _tensor(GOLDEN["mix_sources"][inputs["high"]]),
            inputs["strength"],
            inputs["variation_method"],
        )
        assert_reference_tensor(result, _tensor(case["result"]))


def test_slerp_guard_cases_exercise_zero_norm_rows() -> None:
    for name in ("slerp-low-guard", "slerp-high-guard"):
        inputs = GOLDEN["mix_cases"][name]["inputs"]
        zeroed = "low" if name == "slerp-low-guard" else "high"
        operand = _tensor(GOLDEN["mix_sources"][inputs[zeroed]])
        norms = torch.norm(operand.reshape(operand.shape[0], -1), dim=1)
        assert (norms == 0.0).any(), name
        result = _tensor(GOLDEN["mix_cases"][name]["result"])
        assert torch.isfinite(result).all(), name


def test_prepare_variation_noise_refusals() -> None:
    latent = torch.zeros((2, 4, 6, 8), dtype=torch.float32)
    with pytest.raises(NoiseVariationError, match="rank-4"):
        prepare_variation_noise(torch.zeros((4, 6, 8)), 1)
    with pytest.raises(NoiseVariationError, match="unknown variation method"):
        prepare_variation_noise(latent, 1, variation_method="cubic")
    with pytest.raises(NoiseVariationError, match="unknown batch seed mode"):
        prepare_variation_noise(latent, 1, batch_seed_mode="variation str inc")
    with pytest.raises(NoiseVariationError, match="unparseable variation increment"):
        prepare_variation_noise(latent, 1, batch_seed_mode="variation str inc:x")
    with pytest.raises(ValueError, match="noise indices"):
        prepare_variation_noise(latent, 1, noise_inds=[0])
    with pytest.raises(ValueError, match="noise indices"):
        prepare_variation_noise(latent, 1, noise_inds=[-1, 2])


def test_mix_variation_noise_refuses_unknown_method() -> None:
    operand = torch.zeros((1, 4), dtype=torch.float32)
    with pytest.raises(NoiseVariationError, match="unknown variation method"):
        mix_variation_noise(operand, operand, 0.5, "cubic")
