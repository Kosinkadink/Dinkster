"""CPU behavior for native ComfyUI sigma schedule operations."""

from __future__ import annotations

import pytest
import torch
from dinkster_sampling.sigma_operations import (
    ExtendIntermediateSigmas,
    FlipSigmas,
    ManualSigmas,
    SetFirstSigma,
)


def test_flip_empty_and_set_first_clone_behavior() -> None:
    empty = torch.tensor([])
    assert FlipSigmas.execute(sigmas=empty)["sigmas"] is empty
    original = torch.tensor([8.0, 2.0, 0.0])
    flipped = FlipSigmas.execute(sigmas=original)["sigmas"]
    assert torch.equal(flipped, torch.tensor([0.0001, 2.0, 8.0]))
    assert torch.equal(original, torch.tensor([8.0, 2.0, 0.0]))
    replaced = SetFirstSigma.execute(sigmas=original, sigma=11.5)["sigmas"]
    assert torch.equal(replaced, torch.tensor([11.5, 2.0, 0.0]))
    assert replaced.data_ptr() != original.data_ptr()
    assert torch.equal(original, torch.tensor([8.0, 2.0, 0.0]))


@pytest.mark.parametrize(
    ("spacing", "expected"),
    (
        ("linear", (20.0, 10.0, 8.0, 6.0, 4.0, 2.6666667, 1.3333333, 0.0)),
        (
            "cosine",
            (20.0, 10.0, 7.0, 4.8038478, 4.0, 2.0, 0.5358984, 0.0),
        ),
        ("sine", (20.0, 10.0, 9.1961524, 7.0, 4.0, 3.4641016, 2.0, 0.0)),
    ),
)
def test_extend_intermediate_sigmas_uses_asymmetric_float32_interpolation(
    spacing: str, expected: tuple[float, ...]
) -> None:
    result = ExtendIntermediateSigmas.execute(
        sigmas=torch.tensor([20.0, 10.0, 4.0, 0.0], dtype=torch.float64),
        steps=3,
        start_at_sigma=15.0,
        end_at_sigma=3.0,
        spacing=spacing,
    )["sigmas"]
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-6)


def test_extend_intermediate_sigmas_keeps_empty_and_unmatched_schedules() -> None:
    assert torch.equal(
        ExtendIntermediateSigmas.execute(
            sigmas=torch.tensor([]),
            steps=2,
            start_at_sigma=-1.0,
            end_at_sigma=12.0,
            spacing="linear",
        )["sigmas"],
        torch.tensor([], dtype=torch.float32),
    )
    source = torch.tensor([20.0, 10.0, 0.0])
    result = ExtendIntermediateSigmas.execute(
        sigmas=source, steps=4, start_at_sigma=9.0, end_at_sigma=1.0, spacing="linear"
    )["sigmas"]
    assert torch.equal(result, source)
    assert result.dtype == torch.float32
    assert torch.equal(source, torch.tensor([20.0, 10.0, 0.0]))


def test_manual_sigmas_matches_reference_number_extraction_and_float32_output() -> None:
    result = ManualSigmas.execute(sigmas="sigma=-1.5, +.25; 2.0 and .75")["sigmas"]
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([-1.5, 0.25, 2.0, 0.75], dtype=torch.float32))
