# pyright: basic
"""Replay executed per-batch noise-index draws from pinned ComfyUI.

The fixture (tools/gen_batch_index_noise_goldens.py) records
comfy.sample.prepare_noise outputs for latents carrying ``batch_index``:
the frombatch cases run the reference LatentFromBatch -> KSampler idiom
(contiguous index ranges), the direct cases exercise the repeated- and
gap-index semantics any latent dict can carry (rows before a skipped
index are drawn and discarded, repeated indices share a draw). Draws are
plain generator-fed randn over float32 latents, so the fixture carries
no CPU pin and is enforced bit-exactly on every host.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference_torch import prepare_noise
from golden_files import assert_reference_tensor, load_platform_golden

GOLDENS = load_platform_golden(
    Path(__file__).parent / "goldens" / "batch_index_noise_b78cec87.json",
    allow_portable_fallback=True,
)


def test_fixture_records_the_reference_pin() -> None:
    assert GOLDENS["comfy_baseline"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"


@pytest.mark.parametrize("name", sorted(cast("dict[str, Any]", GOLDENS["cases"])))
def test_prepare_noise_matches_reference_batch_index_draw(name: str) -> None:
    case = cast("dict[str, Any]", GOLDENS["cases"])[name]
    result = case["result"]
    expected = torch.tensor(result["values"], dtype=torch.float32).reshape(result["shape"])
    noise_inds = case["noise_inds"]
    latent = torch.zeros(tuple(result["shape"]), dtype=torch.float32)
    noise = prepare_noise(
        latent,
        case["inputs"]["seed"],
        None if noise_inds is None else tuple(noise_inds),
    )
    assert_reference_tensor(noise, expected)
