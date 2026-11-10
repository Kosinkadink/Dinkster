from __future__ import annotations

import pytest
import torch
from dinkster_flux.sigma_schedulers import Flux2Scheduler


@pytest.mark.parametrize(
    ("width", "height", "midpoint"),
    [(1280, 860, 0.9057320952415466), (1280, 861, 0.765912652015686)],
)
def test_flux2_empirical_schedule_changes_at_sequence_length_boundary(
    width: int, height: int, midpoint: float
) -> None:
    sigmas = Flux2Scheduler.execute(steps=12, width=width, height=height)["SIGMAS"]
    assert sigmas.shape == (13,) and sigmas.dtype == torch.float32
    assert float(sigmas[0]) == 1.0 and float(sigmas[-1]) == 0.0
    assert float(sigmas[6]) == midpoint
    assert torch.all(sigmas[:-1] > sigmas[1:])
