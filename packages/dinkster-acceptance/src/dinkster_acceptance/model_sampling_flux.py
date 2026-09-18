from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from collections.abc import Mapping
from typing import Any

from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import CORE_STRING

from .fixtures import TINY_FLUX, assembled_flux, tiny_latent
from .golden_files import load_model_sampling_flux_golden

STRING = TypeExpr.concrete(CORE_STRING)


def _digest(tensor: Any) -> str:
    import torch

    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _build_runtime(device: Any) -> tuple[Any, Any]:
    import torch
    from dinkster_inference import (
        FLOAT32,
        ReconstructionRecipe,
        RuntimeKnobs,
        WeightSourceBinding,
        WeightSourceRef,
    )
    from dinkster_inference_torch import FluxRuntime

    recipe = ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "checkpoint",
                WeightSourceRef("blake3:" + "0" * 64, "deterministic-tiny-flux", 0),
            ),
        ),
        family_id="dinkster.flux_dev",
        component_identity=("family=dinkster.flux_dev", "deterministic-weighted-runtime"),
        knobs=RuntimeKnobs(
            diffusion_dtype=FLOAT32.name,
            text_dtype=FLOAT32.name,
            vae_dtype=FLOAT32.name,
            fp8_matmul=False,
        ),
    )
    bundle = assembled_flux()
    bundle.diffusion.to(device=device, dtype=torch.float32)
    runtime = FluxRuntime(bundle, runtime_identity=recipe.runtime_identity)
    return runtime, recipe


def run_acceptance() -> dict[str, object]:
    import torch
    from dinkster_compat_comfy.native_arm import GenerationModelSamplingFlux
    from dinkster_compat_comfy.native_residency import NativeRuntimeHandle
    from dinkster_inference import Conditioning, FluxFlowSigmas, builtin_sampler_registry
    from dinkster_inference_torch import BrownianTreeNoise
    from dinkster_inference_torch.sampling_execution import build_custom_sampling_schedule
    from dinkster_inference_torch.wiring import brownian_step_noise

    device = torch.device(os.environ.get("DINKSTER_ACCEPTANCE_DEVICE", "cpu"))
    runtime, recipe = _build_runtime(device)
    handle = NativeRuntimeHandle(runtime, device, recipe=recipe)
    golden_document = load_model_sampling_flux_golden()
    golden = golden_document["case"]
    overlay = GenerationModelSamplingFlux.execute(
        model=handle,
        max_shift=golden["max_shift"],
        base_shift=golden["base_shift"],
        width=golden["width"],
        height=golden["height"],
    )["model"]
    patched = overlay.runtime.with_sampling_space(overlay.sampling_space)
    reference = runtime.with_sampling_space(FluxFlowSigmas(shift=golden["shift"]))
    sampler = builtin_sampler_registry().get("dinkster.dpmpp_2m_sde")
    assert sampler is not None
    scheduler_id = "dinkster.normal"
    steps = 4
    seed = 23
    candidate_sigmas = patched.custom_sampling_sigmas(scheduler_id, steps, 1.0)
    control_sigmas = reference.custom_sampling_sigmas(scheduler_id, steps, 1.0)
    schedule = build_custom_sampling_schedule(
        candidate_sigmas,
        overlay.sampling_space,
        sampler,
        flow=True,
    )
    offset_sigmas = torch.tensor(schedule.sigmas, dtype=torch.float32).tolist()
    expected = golden["schedule"]
    like = torch.zeros(1, 2, 2, 2)
    noise = brownian_step_noise(sampler, schedule, like, seed=seed)
    if noise is None:
        noise = BrownianTreeNoise(
            like,
            min(sigma for sigma in schedule.pre_offset if sigma > 0),
            max(schedule.pre_offset),
            seed=seed,
            cpu=True,
        )
    draws = [
        noise(schedule.sigmas[index], schedule.sigmas[index + 1])
        for index in range(len(schedule.sigmas) - 2)
    ]

    latent = tiny_latent().to(device)
    generator = torch.Generator("cpu").manual_seed(1341)
    assert TINY_FLUX.context_in_dim is not None
    assert TINY_FLUX.vec_in_dim is not None
    cond = Conditioning(
        embeddings=torch.randn(1, 3, TINY_FLUX.context_in_dim, generator=generator).to(device),
        pooled=torch.randn(1, TINY_FLUX.vec_in_dim, generator=generator).to(device),
    )

    def sample(active: Any) -> Any:
        return active.sample(
            latent,
            cond=cond,
            sampler_id=sampler.id,
            scheduler_id=scheduler_id,
            steps=steps,
            seed=seed,
            compute_dtype=torch.float32,
            device=device,
        )

    candidate = sample(patched)
    repeated = sample(patched)
    control = sample(reference)
    expected_draws = [
        torch.tensor(draw, dtype=torch.float32) for draw in expected["brownian_draws"]
    ]
    verdicts = {
        "overlay_shift_matches_comfyui": overlay.sampling_space.shift == golden["shift"],
        "schedule_matches_comfyui": list(candidate_sigmas) == expected["sigmas"],
        "pre_offset_schedule_matches_comfyui": list(schedule.pre_offset) == expected["sigmas"],
        "offset_schedule_matches_comfyui": offset_sigmas == expected["snr_sigmas"],
        "brownian_bounds_match_comfyui": (
            min(sigma for sigma in schedule.pre_offset if sigma > 0) == expected["brownian_min"]
            and max(schedule.pre_offset) == expected["brownian_max"]
        ),
        "brownian_draws_match_comfyui": all(
            torch.equal(actual, wanted)
            for actual, wanted in zip(draws, expected_draws, strict=True)
        ),
        "patched_matches_control": torch.equal(candidate, control),
        "patched_is_deterministic": torch.equal(candidate, repeated),
        "control_schedule_matches": candidate_sigmas == control_sigmas,
        "finite": bool(torch.isfinite(candidate).all()),
    }
    if not all(verdicts.values()):
        raise AssertionError(json.dumps(verdicts, sort_keys=True))
    return {
        "schema": "dinkster-model-sampling-flux-cross-machine-acceptance/1",
        "commit": os.environ["DINKSTER_ACCEPTANCE_COMMIT"],
        "comfyui_commit": golden_document["comfyui_commit"],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None
        ),
        "overlay_node": GenerationModelSamplingFlux.define_schema().node_type,
        "runtime": type(patched).__qualname__,
        "sampler": sampler.id,
        "scheduler": scheduler_id,
        "steps": steps,
        "seed": seed,
        "geometry": {"width": golden["width"], "height": golden["height"]},
        "shift": overlay.sampling_space.shift,
        "sigmas_pre_offset": list(schedule.pre_offset),
        "sigmas_offset": offset_sigmas,
        "brownian_min": min(sigma for sigma in schedule.pre_offset if sigma > 0),
        "brownian_max": max(schedule.pre_offset),
        "brownian_draw_sha256": [_digest(draw) for draw in draws],
        "candidate_sha256": _digest(candidate),
        "control_sha256": _digest(control),
        "repeat_sha256": _digest(repeated),
        "verdicts": verdicts,
    }


class ModelSamplingFluxAcceptance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="acceptance.model_sampling_flux",
            display_name="ModelSamplingFlux cross-machine acceptance",
            category="acceptance",
            inputs=(),
            outputs=(OutputSpec("report", STRING),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(report=json.dumps(run_acceptance(), sort_keys=True))


NODES = (ModelSamplingFluxAcceptance,)
