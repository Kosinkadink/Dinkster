"""Persistent Dinkster adapter for the SDXL GGUF parity workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, cast


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _ids(value: str, kind: str) -> str:
    aliases = {"euler": "dinkster.euler", "simple": "dinkster.simple"}
    result = aliases.get(value, value)
    if "." not in result:
        raise ValueError(f"unsupported {kind}: {value}")
    return result


def _tensor_map(authority: Any) -> list[dict[str, Any]]:
    return [
        {
            "key": tensor.model_key,
            "source_key": tensor.source_name,
            "type": tensor.ggml_type.name,
            "shape": list(tensor.logical_shape),
        }
        for tensor in authority.component_map.tensors.values()
    ]


def _parameter_dtype(*modules: Any) -> str:
    dtypes = {
        str(parameter.dtype).removeprefix("torch.")
        for module in modules
        for parameter in module.parameters()
    }
    if len(dtypes) != 1:
        raise RuntimeError(f"expected one parameter dtype, got {sorted(dtypes)}")
    return dtypes.pop()


def _actual_dtypes(runtime: Any, dequant: str) -> dict[str, str]:
    return {
        "codec": _parameter_dtype(runtime.assembled.vae),
        "dequant": dequant,
        "diffusion": _parameter_dtype(runtime.assembled.diffusion),
        "text": _parameter_dtype(runtime.assembled.clip_l, runtime.assembled.clip_g),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-json", required=True)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()
    workload = json.loads(args.workload_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if args.reference_output else str(workload["device_ordinal"])
    )
    for source in sorted((args.repo / "packages").glob("*/src")):
        sys.path.insert(0, str(source))

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from dinkster_inference import (
        SamplingGuidance,
        load_gguf_weight_source,
        load_safetensors_header,
    )
    from dinkster_inference_torch import SDRuntime, load_runtime
    from dinkster_inference_torch.sources import load_gguf_tensors

    args.output_dir.mkdir(parents=True, exist_ok=True)
    authority = load_gguf_weight_source(_path(args.artifact_root, workload["checkpoint"]))
    sources = {
        "diffusion": authority,
        "clip_l": load_safetensors_header(_path(args.artifact_root, workload["clip_l"])),
        "clip_g": load_safetensors_header(_path(args.artifact_root, workload["clip_g"])),
        "vae": load_safetensors_header(_path(args.artifact_root, workload["vae"])),
    }

    def load(device: torch.device, dtype: torch.dtype) -> Any:
        runtime = cast(
            SDRuntime,
            load_runtime(
                **sources,
                diffusion_dtype=dtype,
                text_dtype=torch.float32,
                vae_dtype=torch.float32,
                attention_policy="auto",
            ),
        )
        runtime.assembled.diffusion.to(device=device, dtype=dtype)
        assert runtime.assembled.clip_l is not None
        assert runtime.assembled.clip_g is not None
        runtime.assembled.clip_l.to(device=device, dtype=torch.float32)
        runtime.assembled.clip_g.to(device=device, dtype=torch.float32)
        runtime.assembled.vae.to(device=device, dtype=torch.float32)
        return runtime

    if args.reference_output:
        reference = {**workload, **workload.get("reference", {})}
        reference.update(width=64, height=64, batch=1, steps=1, cfg=1.0, seed=240)
        runtime = load(torch.device("cpu"), torch.float32)
        with torch.inference_mode():
            cond = runtime.encode_text(reference["prompt"])
            uncond = runtime.encode_text(reference.get("negative_prompt", ""))
            latent = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
            sampled = runtime.sample(
                latent,
                cond=cond,
                cfg=SamplingGuidance(uncond, 1.0),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=1,
                denoise=reference.get("denoise", 1.0),
                seed=240,
                compute_dtype=torch.float32,
                device=torch.device("cpu"),
            )
            image = runtime.decode_latent(sampled).permute(0, 2, 3, 1).clamp(0, 1)
        image_path = args.reference_output.with_suffix(".image.npy")
        latent_path = args.reference_output.with_suffix(".latent.npy")
        np.save(image_path, image.float().numpy(), allow_pickle=False)
        np.save(latent_path, sampled.float().numpy(), allow_pickle=False)
        digests = []
        for tensor in authority.component_map.tensors.values():
            if tensor.ggml_type.name != "Q8_0":
                continue
            value = load_gguf_tensors(authority, [tensor.source_name])[tensor.source_name]
            raw = value.contiguous().float().numpy().tobytes()
            digests.append(
                {
                    "key": tensor.model_key,
                    "shape": list(tensor.logical_shape),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        receipt = {
            "engine": "dinkster",
            "python": platform.python_version(),
            "runtime_facts": list(authority.runtime_facts),
            "torch": torch.__version__,
            "attention_backend": runtime.attention_status["unet"].primary,
            "actual_dtypes": _actual_dtypes(runtime, "float32"),
            "image_path": str(image_path),
            "latent_path": str(latent_path),
            "q8_tensors": digests,
            "tensor_map": _tensor_map(authority),
        }
        args.reference_output.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return 0

    runtime = cond = uncond = latent = None
    device = torch.device("cuda:0")
    for line in sys.stdin:
        request = json.loads(line)
        load_ns = 0
        if runtime is None:
            start = time.perf_counter_ns()
            runtime = load(device, torch.float16)
            with torch.inference_mode():
                cond = runtime.encode_text(workload["prompt"])
                uncond = runtime.encode_text(workload.get("negative_prompt", ""))
            latent = torch.zeros(
                (workload["batch"], 4, workload["height"] // 8, workload["width"] // 8),
                device=device,
            )
            torch.cuda.synchronize()
            load_ns = time.perf_counter_ns() - start
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            sampled = runtime.sample(
                latent,
                cond=cond,
                cfg=SamplingGuidance(uncond, request.get("cfg", workload["cfg"])),
                sampler_id=_ids(workload["sampler"], "sampler"),
                scheduler_id=_ids(workload["scheduler"], "scheduler"),
                steps=workload["steps"],
                denoise=workload["denoise"],
                seed=request["seed"],
                compute_dtype=torch.float16,
                device=device,
            )
        torch.cuda.synchronize()
        sampling_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        with torch.inference_mode():
            image = runtime.decode_latent(sampled).permute(0, 2, 3, 1).clamp(0, 1)
        torch.cuda.synchronize()
        decode_ns = time.perf_counter_ns() - start
        output = args.output_dir / f"dinkster-{request['phase']}.npy"
        np.save(output, image.float().cpu().numpy(), allow_pickle=False)
        receipt = {
            "phase": request["phase"],
            "cold_load_ns": load_ns,
            "generation_ns": sampling_ns + decode_ns,
            "sampling_ns": sampling_ns,
            "decode_ns": decode_ns,
            "decode_mode": "regular",
            "output_path": str(output),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "attention_backend": runtime.attention_status["unet"].primary,
            "actual_dtypes": _actual_dtypes(runtime, "float16"),
            "memory_policy": "resident",
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
