"""Complete checkpoint node sampling matches executed ComfyUI flow goldens."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import FLUX2_DEV, LUMINA2
from dinkster_inference_torch import Flux2Runtime, Lumina2Runtime
from dinkster_inference_torch.schedules import torch_scheduler_registry
from dinkster_inference_torch.solvers import torch_sampler_registry
from dinkster_native import native_arm as arm

GOLDEN = json.loads((Path(__file__).parent / "goldens/complete_sampling_shifts.json").read_text())


class ArithmeticDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.guidance_in = None
        self.vector_in = None
        self.config = SimpleNamespace(vec_in_dim=None, context_in_dim=8)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: object = None,
        guidance: object = None,
        *,
        ref_latents: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        return latent * 0.17 + timestep.reshape(-1, 1, 1, 1).square() * 0.07


@pytest.mark.parametrize("case", GOLDEN["cases"])
def test_complete_checkpoint_sampling_shift_reference(
    monkeypatch: pytest.MonkeyPatch, case: dict[str, Any]
) -> None:
    flux = case["family"] == "flux2"
    runtime: Any = object.__new__(Flux2Runtime if flux else Lumina2Runtime)

    def compute_dtype(role: str) -> torch.dtype:
        return torch.float32

    def stage(*args: object, **kwargs: object) -> nullcontext[None]:
        return nullcontext()

    def preview(*args: object, **kwargs: object) -> None:
        return None

    runtime.assembled = SimpleNamespace(
        family=FLUX2_DEV if flux else LUMINA2,
        diffusion=ArithmeticDiffusion(),
        compute_dtype=compute_dtype,
    )
    runtime._runtime_identity = "native:complete-sampling-test"
    runtime._samplers = torch_sampler_registry()
    runtime._schedulers = torch_scheduler_registry()
    runtime._guidance = None
    handle = SimpleNamespace(
        runtime=runtime,
        recipe=SimpleNamespace(
            family_id=runtime.family.id, sources=(SimpleNamespace(role="checkpoint"),)
        ),
        load_device=torch.device("cpu"),
        stage=stage,
    )

    def native_model(*args: object) -> Any:
        return handle, (), {}, None, case["shift"], (), None, ()

    monkeypatch.setattr(arm, "_native_model", native_model)
    monkeypatch.setattr(arm, "sampling_preview_emitter", preview)
    original = runtime.sample_custom
    calls = []

    def sample_custom(*args: Any, **kwargs: Any) -> Any:
        request, noise = kwargs["request"], kwargs["noise"]
        assert list(request.sigmas) == case["sigmas"]
        assert hashlib.sha256(noise.numpy().tobytes()).hexdigest() == case["noise_hash"]
        calls.append(request)
        return original(*args, **kwargs)

    runtime.sample_custom = sample_custom
    rows = [[torch.ones((1, 3, 8 if flux else 2304)), {}]]
    output = cast("Any", arm.NativeKSampler).execute(
        model=object(),
        seed=7,
        steps=3,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=rows,
        negative=rows,
        latent_image={"samples": torch.zeros((1, 128 if flux else 16, 2, 2))},
        denoise=1.0,
    )["latent"]["samples"]
    assert len(calls) == 1
    assert hashlib.sha256(output.numpy().tobytes()).hexdigest() == case["output_hash"]
