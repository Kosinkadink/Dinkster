from __future__ import annotations

from collections.abc import Mapping

import torch
from dinkster_api.v1 import (
    materialize_latent,
)
from dinkster_misc_models.model_specific_schedulers import Ideogram4Scheduler


def _sigmas(result: Mapping[str, object]) -> torch.Tensor:
    return materialize_latent(result["SIGMAS"])


def _assert_sigmas(result: Mapping[str, object], expected: list[float]) -> None:
    torch.testing.assert_close(
        _sigmas(result), torch.tensor(expected, dtype=torch.float32), rtol=0.0, atol=1e-7
    )


def test_ideogram4_schedule_has_exact_aspect_shift_and_spread_values() -> None:
    _assert_sigmas(
        Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=0.0, std=1.75),
        [0.99987662, 0.76501203, 0.5, 0.234988, 0.0],
    )
    _assert_sigmas(
        Ideogram4Scheduler.execute(steps=4, width=1024, height=512, mu=0.0, std=1.75),
        [0.99987662, 0.82155681, 0.58578646, 0.30284566, 0.0],
    )
    _assert_sigmas(
        Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=1.0, std=1.75),
        [0.99987662, 0.89847165, 0.7310586, 0.45503256, 0.0],
    )
    _assert_sigmas(
        Ideogram4Scheduler.execute(steps=4, width=512, height=512, mu=0.0, std=0.5),
        [0.99987662, 0.58352113, 0.5, 0.41647887, 0.0],
    )
