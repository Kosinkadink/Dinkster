"""Persistent native Dinkster adapter for the canonical SD1.5 workload."""

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
    from dinkster_inference import SamplingGuidance, load_safetensors_header
    from dinkster_inference_torch import SDRuntime, load_runtime

    runtime = None
    cond = None
    uncond = None
    latent = None
    text_parameter_dtype = None
    device = torch.device("cuda:0")
    for line in sys.stdin:
        request = json.loads(line)
        phase = request["phase"]
        load_ns = 0
        if runtime is None:
            start = time.perf_counter_ns()
            source = load_safetensors_header(args.artifact_root / workload["checkpoint"])
            text_dtype = torch.float32
            runtime = cast(
                SDRuntime,
                load_runtime(
                    checkpoint=source,
                    diffusion_dtype=torch.float16,
                    text_dtype=text_dtype,
                    vae_dtype=torch.float32,
                ),
            )
            runtime.assembled.diffusion.to(device)
            assert runtime.assembled.clip_l is not None
            runtime.assembled.clip_l.to(device=device, dtype=text_dtype)
            text_parameter_dtype = str(
                next(runtime.assembled.clip_l.parameters()).dtype
            ).removeprefix("torch.")
            runtime.assembled.vae.to(device)
            with torch.inference_mode():
                cond = runtime.encode_text(workload["prompt"])
                uncond = runtime.encode_text(workload["negative_prompt"])
            latent = torch.zeros(
                (workload["batch"], 4, workload["height"] // 8, workload["width"] // 8),
                dtype=torch.float32,
                device=device,
            )
            torch.cuda.synchronize()
            load_ns = time.perf_counter_ns() - start
        assert (
            cond is not None
            and uncond is not None
            and latent is not None
            and text_parameter_dtype is not None
        )
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            sampled = runtime.sample(
                latent,
                cond=cond,
                cfg=SamplingGuidance(uncond, workload["cfg"]),
                sampler_id=workload["sampler"],
                scheduler_id=workload["scheduler"],
                steps=workload["steps"],
                denoise=workload["denoise"],
                seed=request["seed"],
                compute_dtype=torch.float16,
                device=device,
            )
            image = runtime.decode_latent(sampled).permute(0, 2, 3, 1)
        torch.cuda.synchronize()
        generation_ns = time.perf_counter_ns() - start
        output = args.output_dir / f"dinkster-{phase}.npy"
        np.save(output, image.detach().float().cpu().numpy(), allow_pickle=False)
        print(
            json.dumps(
                {
                    "cold_load_ns": load_ns,
                    "generation_ns": generation_ns,
                    "output_path": str(output),
                    "phase": phase,
                    "text_parameter_dtype": text_parameter_dtype,
                    "torch": torch.__version__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
