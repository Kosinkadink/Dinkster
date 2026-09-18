"""Persistent native Dinkster adapter for the canonical Ovis workload."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import cast


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-json", required=True)
    args = parser.parse_args()
    workload = json.loads(args.workload_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(workload["device_ordinal"])
    for source in sorted((args.repo / "packages").glob("*/src")):
        sys.path.insert(0, str(source))

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from dinkster_inference import (
        Registry,
        SamplingGuidance,
        SchedulerDescriptor,
        load_safetensors_header,
    )
    from dinkster_inference_torch import FluxRuntime, load_runtime

    def aura_flow_simple(steps: int, _space: object) -> tuple[float, ...]:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        timesteps = torch.arange(1, 1001, dtype=torch.float32) / 1000
        shift = workload["model_sampling_shift"]
        multiplier = workload["model_sampling_multiplier"]
        sigmas = multiplier * (shift * timesteps) / (1.0 + (shift - 1.0) * timesteps)
        stride = len(sigmas) / steps
        return (
            *(float(sigmas[-(1 + int(index * stride))]) for index in range(steps)),
            0.0,
        )

    schedulers = Registry()
    schedulers.register(
        SchedulerDescriptor(
            id="dinkster.simple",
            display_name="Ovis AuraFlow simple",
            make_sigmas=aura_flow_simple,
        )
    )
    runtime = None
    cond = None
    uncond = None
    latent = None
    device = torch.device("cuda:0")
    for line in sys.stdin:
        request = json.loads(line)
        phase = request["phase"]
        load_ns = 0
        if runtime is None:
            start = time.perf_counter_ns()
            runtime = cast(
                FluxRuntime,
                load_runtime(
                    diffusion=load_safetensors_header(Path(workload["diffusion_model"])),
                    qwen3_2b=load_safetensors_header(Path(workload["text_encoder"])),
                    vae=load_safetensors_header(Path(workload["codec"])),
                    diffusion_dtype=torch.bfloat16,
                    text_dtype=torch.float32,
                    vae_dtype=torch.bfloat16,
                    scheduler_registry=schedulers,
                    registry_token="ovis-auraflow-shift3-simple-v1",
                ),
            )
            assert runtime.assembled.qwen3_2b is not None
            runtime.assembled.qwen3_2b.to(device)
            with torch.inference_mode():
                cond = runtime.encode_text(workload["prompt"])
                uncond = runtime.encode_text(workload["negative_prompt"])
            runtime.assembled.qwen3_2b.to("cpu")
            runtime.assembled.diffusion.to(device)
            runtime.assembled.vae.to(device)
            torch.cuda.empty_cache()
            latent = torch.zeros(
                (
                    workload["batch"],
                    16,
                    workload["height"] // 8,
                    workload["width"] // 8,
                ),
                dtype=torch.float32,
                device=device,
            )
            torch.cuda.synchronize()
            load_ns = time.perf_counter_ns() - start
        assert runtime is not None and cond is not None and uncond is not None
        assert latent is not None
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            sampled = runtime.sample(
                latent,
                cond=cond,
                cfg=SamplingGuidance(uncond, workload["cfg"]),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=workload["steps"],
                denoise=workload["denoise"],
                seed=request["seed"],
                compute_dtype=torch.bfloat16,
                device=device,
            )
            image = runtime.codec.decode_tiled(
                sampled.to(torch.bfloat16),
                output_device="cpu",
                dtype=torch.float32,
            ).permute(0, 2, 3, 1)
        torch.cuda.synchronize()
        generation_ns = time.perf_counter_ns() - start
        latent_output = args.output_dir / f"dinkster-{phase}-latent.npy"
        image_output = args.output_dir / f"dinkster-{phase}-image.npy"
        np.save(latent_output, sampled.detach().float().cpu().numpy(), allow_pickle=False)
        np.save(image_output, image.detach().float().cpu().numpy(), allow_pickle=False)
        print(
            json.dumps(
                {
                    "cold_load_ns": load_ns,
                    "decode_mode": "tiled",
                    "generation_ns": generation_ns,
                    "outputs": {"image": str(image_output), "latent": str(latent_output)},
                    "phase": phase,
                    "torch": torch.__version__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
