"""CPU behavior for the native VOID DDIM sampler."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest
from dinkster_api.v1 import SAMPLER_TYPE, TypeExpr
from dinkster_nodes_comfy.void_sampler import VOIDSampler


def test_void_sampler_schema_is_torch_free() -> None:
    schema = VOIDSampler.define_schema()

    assert schema.node_type == "VOIDSampler"
    assert schema.category == "model/sampling/samplers"
    assert schema.inputs == ()
    assert [(output.id, output.type) for output in schema.outputs] == [
        ("SAMPLER", TypeExpr.concrete(SAMPLER_TYPE))
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.void_sampler import VOIDSampler; "
            "VOIDSampler.schema(); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("schedule_dtype", "positive_result"),
    (("float32", 0.29626840353012085), ("float64", 0.2962684631347656)),
)
def test_void_sampler_runs_alpha_space_ddim_through_public_sampler_boundary(
    schedule_dtype: str, positive_result: float
) -> None:
    import torch

    sampler = VOIDSampler.execute()["SAMPLER"]
    from dinkster_inference.samplers import Sampler

    assert isinstance(sampler, Sampler)

    noise = torch.tensor([[[[2.0, -3.0]]]], dtype=torch.float64)
    original_noise = noise.clone()
    sigmas = torch.tensor((2.0, 1.0, 0.25, 0.0), dtype=getattr(torch, schedule_dtype))
    model_options = {"marker": "preserved"}
    calls: list[tuple[torch.Tensor, torch.Tensor, object, object]] = []
    callbacks: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []

    def model(
        x: torch.Tensor, sigma: torch.Tensor, *, model_options: object, seed: object
    ) -> torch.Tensor:
        calls.append((x.clone(), sigma.clone(), model_options, seed))
        return x * 0.25 + sigma.reshape((-1, 1, 1, 1)) * 0.125

    def callback(index: int, denoised: torch.Tensor, x: torch.Tensor, total: int) -> None:
        callbacks.append((index, denoised.clone(), x.clone(), total))

    result = sampler.sample(
        model,
        sigmas,
        {"model_options": model_options, "seed": 947},
        callback,
        noise,
        latent_image=torch.full_like(noise, 17.0),
        denoise_mask=torch.ones_like(noise),
        disable_pbar=True,
    )

    # Hand-derived from F bc5f880dbdd43e426077f42ccec53bb577ffee9f,
    # nodes_void.py:411-445. Eliminating epsilon gives an affine update:
    # x1 = ((sqrt(5)/2 + 1/8)*x0 + 1/8)/sqrt(2),
    # x2 = ((sqrt(2) + 3/4)*x1 + 3/8)/sqrt(17), x3 = x2/4 + 1/32.
    # Independently round scalar operations via struct.pack, using alpha=1/5,1/2,16/17.
    # Float32 torch.pow roots are 0.44721361994743347,0.7071067690849304,0.9701425433158875;
    # float64 scalars use double roots before conversion. No second sampler is the oracle.
    expected = torch.tensor([positive_result, -0.2804364860057831]).reshape_as(noise)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype is torch.float32
    assert torch.equal(noise, original_noise)
    assert [(index, total) for index, _, _, total in callbacks] == [(0, 3), (1, 3), (2, 3)]
    for index, denoised, x, _total in callbacks:
        torch.testing.assert_close(x, calls[index][0], rtol=0, atol=0)
        torch.testing.assert_close(
            denoised,
            calls[index][0] * 0.25 + calls[index][1].reshape((-1, 1, 1, 1)) * 0.125,
            rtol=0,
            atol=0,
        )
    assert [tuple(sigma.tolist()) for _, sigma, _, _ in calls] == [(2.0,), (1.0,), (0.25,)]
    assert [(options, seed) for _, _, options, seed in calls] == [(model_options, 947)] * 3

    ignored_result = sampler.sample(
        model,
        sigmas,
        {"model_options": model_options, "seed": 947},
        None,
        original_noise,
        latent_image=torch.full_like(noise, -91.0),
        denoise_mask=torch.zeros_like(noise),
        disable_pbar=True,
    )
    torch.testing.assert_close(ignored_result, result, rtol=0, atol=0)


def test_void_sampler_returns_float32_noise_without_steps() -> None:
    import torch

    sampler = VOIDSampler.execute()["SAMPLER"]
    noise = torch.tensor([[[[1.25, -4.5]]]], dtype=torch.float64)
    original_noise = noise.clone()
    calls: list[Any] = []

    def model(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append((args, kwargs))
        raise AssertionError("the no-step sampler must not call the model")

    result = sampler.sample(
        model,
        torch.tensor((0.0,), dtype=torch.float32),
        {},
        lambda *args: calls.append(args),
        noise,
        latent_image=torch.ones_like(noise),
        denoise_mask=torch.ones_like(noise),
        disable_pbar=True,
    )

    torch.testing.assert_close(result, noise.to(torch.float32), rtol=0, atol=0)
    assert result.dtype is torch.float32
    assert torch.equal(noise, original_noise)
    assert calls == []
