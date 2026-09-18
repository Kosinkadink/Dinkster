"""Persistent native Dinkster adapter for the canonical SDXL v-pred workload."""

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
    parser.add_argument("--construct-only", action="store_true")
    args = parser.parse_args()
    workload = json.loads(args.workload_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if args.construct_only else str(workload["device_ordinal"])
    )
    for source in sorted((args.repo / "packages").glob("*/src")):
        sys.path.insert(0, str(source))

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from dinkster_inference import (
        Parameterization,
        SamplingGuidance,
        load_safetensors_header,
        plan_sd_assembly,
        probe_native,
    )
    from dinkster_inference_torch import SDRuntime, load_runtime

    checkpoint_path = args.artifact_root / workload["checkpoint"]
    if args.construct_only:
        source = load_safetensors_header(checkpoint_path)
        capability = probe_native(source)
        plan = plan_sd_assembly(checkpoint=source)
        sampling = plan.sampling
        if (
            not capability.native
            or capability.family_id != "dinkster.sdxl"
            or sampling is None
            or sampling.parameterization is not Parameterization.V_PREDICTION
            or not sampling.zsnr
        ):
            raise RuntimeError(
                f"SDXL v-pred/zsnr checkpoint was not accepted: {capability}; {sampling}"
            )
        print(
            "READY dinkster family=dinkster.sdxl; parameterization=v_prediction; "
            f"zsnr=True; cuda_initialized={torch.cuda.is_initialized()}",
            flush=True,
        )
        return 0

    runtime = None
    cond = None
    uncond = None
    latent = None
    text_parameter_dtype = None
    device = torch.device("cuda:0")
    diffusion_dtype = torch.float16
    text_dtype = torch.float32
    vae_dtype = torch.float32
    for line in sys.stdin:
        request = json.loads(line)
        phase = request["phase"]
        load_ns = 0
        if runtime is None:
            start = time.perf_counter_ns()
            runtime = cast(
                SDRuntime,
                load_runtime(
                    checkpoint=load_safetensors_header(checkpoint_path),
                    diffusion_dtype=diffusion_dtype,
                    text_dtype=text_dtype,
                    vae_dtype=vae_dtype,
                ),
            )
            runtime.assembled.diffusion.to(
                device=device,
                dtype=diffusion_dtype,
            )
            assert runtime.assembled.clip_l is not None
            assert runtime.assembled.clip_g is not None
            runtime.assembled.clip_l.to(device=device, dtype=text_dtype)
            runtime.assembled.clip_g.to(device=device, dtype=text_dtype)
            text_dtypes = {
                str(next(runtime.assembled.clip_l.parameters()).dtype).removeprefix("torch."),
                str(next(runtime.assembled.clip_g.parameters()).dtype).removeprefix("torch."),
            }
            if len(text_dtypes) != 1:
                raise RuntimeError(f"SDXL text parameter dtypes differ: {sorted(text_dtypes)}")
            text_parameter_dtype = text_dtypes.pop()
            runtime.assembled.vae.to(device=device, dtype=vae_dtype)
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
        assert runtime is not None and cond is not None and uncond is not None
        assert latent is not None and text_parameter_dtype is not None
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
                compute_dtype=diffusion_dtype,
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
                    "decode_mode": "regular",
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
