"""Measure fork-backed Flux window scattering against serial window execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import secrets
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

PROMPT = "A red fox sitting beside a mossy stone in a sunlit forest, detailed photograph"
WORKLOADS = {
    "two-windows": (tuple(range(48)), tuple(range(16, 64))),
    "three-windows": (tuple(range(28)), tuple(range(18, 46)), tuple(range(36, 64))),
}
ATTEMPT_GROUP = "flux-window-multigpu-receipt"
ATTEMPT = 1


class ReceiptError(RuntimeError):
    pass


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


class _WindowProfiler:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.reset()

    def reset(self) -> None:
        self.events: dict[str, list[tuple[Any, Any]]] = {
            "window_compute": [],
            "collective": [],
        }
        self.cpu_seconds = {
            "collective": 0.0,
            "failure_sync": 0.0,
            "process_group_init": 0.0,
            "process_group_lookup": 0.0,
        }
        self.calls = {
            "window_evaluations": 0,
            "collectives": 0,
            "failure_syncs": 0,
            "process_group": 0,
        }

    def cuda_call(self, category: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        start = self.torch.cuda.Event(enable_timing=True)
        stop = self.torch.cuda.Event(enable_timing=True)
        start.record()
        cpu_started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.cpu_seconds[category] += time.perf_counter() - cpu_started
            stop.record()
            self.events[category].append((start, stop))

    def snapshot(self, sample_seconds: float) -> dict[str, Any]:
        gpu_seconds = {
            category: sum(start.elapsed_time(stop) for start, stop in events) / 1000.0
            for category, events in self.events.items()
        }
        accounted = gpu_seconds["window_compute"] + gpu_seconds["collective"]
        return {
            "window_compute_gpu_seconds": gpu_seconds["window_compute"],
            "collective_gpu_seconds": gpu_seconds["collective"],
            "collective_cpu_seconds": self.cpu_seconds["collective"],
            "failure_sync_cpu_seconds": self.cpu_seconds["failure_sync"],
            "process_group_init_cpu_seconds": self.cpu_seconds["process_group_init"],
            "process_group_lookup_cpu_seconds": self.cpu_seconds["process_group_lookup"],
            "other_sample_wall_seconds": max(0.0, sample_seconds - accounted),
            "calls": dict(self.calls),
        }


def _install_window_profiler(torch: Any) -> _WindowProfiler:
    from dinkster_inference.window_execution import WindowPlanExecutor
    from dinkster_native import attention, multigpu

    profiler = _WindowProfiler(torch)
    evaluate_window = WindowPlanExecutor.evaluate_window
    ensure_process_group = attention._ensure_process_group  # pyright: ignore[reportPrivateUsage]
    raise_group_failure = multigpu._raise_group_failure  # pyright: ignore[reportPrivateUsage]
    broadcast = torch.distributed.broadcast
    all_gather = torch.distributed.all_gather

    def profiled_evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
        profiler.calls["window_evaluations"] += 1
        return profiler.cuda_call("window_compute", evaluate_window, self, *args, **kwargs)

    def profiled_group() -> Any:
        initialized = attention._process_group_config is not None  # pyright: ignore[reportPrivateUsage]
        started = time.perf_counter()
        try:
            return ensure_process_group()
        finally:
            category = "process_group_lookup" if initialized else "process_group_init"
            profiler.cpu_seconds[category] += time.perf_counter() - started
            profiler.calls["process_group"] += 1

    def profiled_failure(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return raise_group_failure(*args, **kwargs)
        finally:
            profiler.cpu_seconds["failure_sync"] += time.perf_counter() - started
            profiler.calls["failure_syncs"] += 1

    def profiled_collective(function: Any, *args: Any, **kwargs: Any) -> Any:
        profiler.calls["collectives"] += 1
        return profiler.cuda_call("collective", function, *args, **kwargs)

    WindowPlanExecutor.evaluate_window = profiled_evaluate
    attention._ensure_process_group = profiled_group  # pyright: ignore[reportPrivateUsage]
    multigpu._raise_group_failure = profiled_failure  # pyright: ignore[reportPrivateUsage]
    torch.distributed.broadcast = lambda *args, **kwargs: profiled_collective(
        broadcast, *args, **kwargs
    )
    torch.distributed.all_gather = lambda *args, **kwargs: profiled_collective(
        all_gather, *args, **kwargs
    )
    return profiler


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *arguments), text=True, timeout=30
    ).strip()


def _asset(path: Path, digest: str) -> Any:
    from dinkster_assets import AssetRef

    return AssetRef(digest, path.name, path.stat().st_size, resolver=_Resolver(path))


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.view(cast("Any", importlib.import_module("torch")).uint8).numpy().tobytes()
    ).hexdigest()


def _route() -> tuple[Any, Any]:
    from dinkster_native.attention import create_attention_runtime
    from dinkster_protocol import AttentionPolicyConfig, derive_attention_route_token

    runtime = create_attention_runtime()
    token = derive_attention_route_token(
        runtime.capabilities,
        AttentionPolicyConfig(requested_policy="sdpa"),
    )
    return runtime, token


def _window_plan(name: str) -> object:
    from dinkster_native.fork_nodes import GenerationExplicitWindowPlan

    windows = ";".join(",".join(str(index) for index in window) for window in WORKLOADS[name])
    return GenerationExplicitWindowPlan.execute(
        axis="width",
        windows=windows,
        wrap=False,
        fuse_method="flat",
    )["plan"]


def _sample(
    model: object,
    positive: object,
    latent: dict[str, object],
    window_plan: object,
    args: argparse.Namespace,
    profiler: _WindowProfiler,
) -> dict[str, Any]:
    from dinkster_native.fork_nodes import GenerationKSampler

    torch = cast("Any", importlib.import_module("torch"))
    profiler.reset()
    torch.cuda.synchronize()
    started = time.perf_counter()
    sampled = cast(
        "dict[str, Any]",
        GenerationKSampler.execute(
            model=model,
            seed=args.seed,
            steps=args.steps,
            cfg=1.0,
            sampler_name="euler",
            scheduler="normal",
            positive=positive,
            negative=[],
            latent_image=latent,
            denoise=1.0,
            window_plan=window_plan,
        )["latent"],
    )
    torch.cuda.synchronize()
    sample_seconds = time.perf_counter() - started
    return {
        "latent": sampled["samples"],
        "latent_sha256": _tensor_sha256(sampled["samples"]),
        "sample_seconds": sample_seconds,
        "profile": profiler.snapshot(sample_seconds),
    }


def _job(
    model: object,
    clip: Any,
    vae: Any,
    latent: dict[str, object],
    window_plan: object,
    args: argparse.Namespace,
    profiler: _WindowProfiler,
) -> dict[str, Any]:
    torch = cast("Any", importlib.import_module("torch"))
    torch.cuda.synchronize()
    job_started = time.perf_counter()
    conditioning_started = job_started
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    hooks = cast("Any", importlib.import_module("dinkster_inference.hooks"))
    positive = hooks.conditioning_set_values(conditioning, {"guidance": 3.5})
    torch.cuda.synchronize()
    conditioning_seconds = time.perf_counter() - conditioning_started
    sampled = _sample(model, positive, latent, window_plan, args, profiler)
    decode_started = time.perf_counter()
    image = vae.decode(sampled["latent"])
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    return {
        "latent_sha256": sampled["latent_sha256"],
        "image_sha256": _tensor_sha256(image),
        "conditioning_seconds": conditioning_seconds,
        "sample_seconds": sampled["sample_seconds"],
        "decode_seconds": decode_seconds,
        "whole_job_seconds": time.perf_counter() - job_started,
        "profile": sampled["profile"],
    }


def run_worker(args: argparse.Namespace) -> int:
    worker_started = time.perf_counter()
    torch = cast("Any", importlib.import_module("torch"))
    from dinkster_native.attention import (
        activate_distributed_attention,
        release_distributed_attention,
    )
    from dinkster_native.fork_nodes import GenerationLoadCheckpoint
    from dinkster_workers import ExecutionContext, use_execution_context

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ReceiptError("Flux receipt workers require exactly one visible CUDA device")
    runtime, token = _route()
    load_started = time.perf_counter()
    loaded = GenerationLoadCheckpoint.execute(
        checkpoint=_asset(args.checkpoint, args.checkpoint_digest)
    )
    model = loaded["model"]
    clip = cast("Any", loaded["clip"])
    vae = cast("Any", loaded["vae"])
    compute_dtype = cast("Any", model).model.get_dtype_inference()
    if compute_dtype is not torch.bfloat16:
        raise ReceiptError(f"Flux receipt compute dtype must be bfloat16, got {compute_dtype}")
    model_load_seconds = time.perf_counter() - load_started
    latent = {
        "samples": torch.zeros((1, 16, args.height // 8, args.width // 8), dtype=torch.float32)
    }
    window_plan = _window_plan(args.workload)
    profiler = _install_window_profiler(torch)
    distributed = "DINKSTER_SINGLE_JOB_RANK" in os.environ
    context = ExecutionContext(
        arm="native",
        expected_execution_identity=None,
        attention_policy="sdpa",
        attention_route_token=token,
        attention_capabilities=runtime.capabilities,
        attention_runtime=runtime,
    )
    if distributed:
        activate_distributed_attention(ATTEMPT_GROUP, ATTEMPT)
    try:
        with use_execution_context(context):
            cold = _job(model, clip, vae, latent, window_plan, args, profiler)
            cold_workflow_seconds = time.perf_counter() - worker_started
            warmups = [
                _job(model, clip, vae, latent, window_plan, args, profiler)
                for _ in range(args.warmups)
            ]
            measured = [
                _job(model, clip, vae, latent, window_plan, args, profiler)
                for _ in range(args.repeats)
            ]
    finally:
        if distributed:
            release_distributed_attention(ATTEMPT_GROUP, ATTEMPT)
    sample_seconds = [float(run["sample_seconds"]) for run in measured]
    whole_job_seconds = [float(run["whole_job_seconds"]) for run in measured]
    median_seconds = statistics.median(sample_seconds)
    median_whole_job_seconds = statistics.median(whole_job_seconds)
    profile_keys = tuple(key for key in measured[0]["profile"] if key != "calls")
    properties = torch.cuda.get_device_properties(0)
    result = {
        "status": "PASS",
        "distributed": distributed,
        "rank": int(os.environ["DINKSTER_SINGLE_JOB_RANK"]) if distributed else None,
        "world_size": int(os.environ.get("DINKSTER_SINGLE_JOB_WORLD_SIZE", "1")),
        "output_hashes": {
            "cold": {
                "latent": cold["latent_sha256"],
                "image": cold["image_sha256"],
            },
            "warmups": [
                {"latent": run["latent_sha256"], "image": run["image_sha256"]} for run in warmups
            ],
            "measured": [
                {"latent": run["latent_sha256"], "image": run["image_sha256"]} for run in measured
            ],
        },
        "timing": {
            "worker_wall_seconds": time.perf_counter() - worker_started,
            "model_load_seconds": model_load_seconds,
            "cold_workflow_seconds": cold_workflow_seconds,
            "cold_job": {key: value for key, value in cold.items() if not key.endswith("sha256")},
            "warmup_jobs": [
                {key: value for key, value in run.items() if not key.endswith("sha256")}
                for run in warmups
            ],
            "measured_jobs": [
                {key: value for key, value in run.items() if not key.endswith("sha256")}
                for run in measured
            ],
            "sample_seconds": sample_seconds,
            "median_sample_seconds": median_seconds,
            "median_sample_step_seconds": median_seconds / args.steps,
            "whole_job_seconds": whole_job_seconds,
            "median_whole_job_seconds": median_whole_job_seconds,
            "jobs_per_hour": 3600.0 / median_whole_job_seconds,
            "median_profile": {
                key: statistics.median(float(run["profile"][key]) for run in measured)
                for key in profile_keys
            },
            "median_profile_per_step": {
                key: statistics.median(float(run["profile"][key]) for run in measured) / args.steps
                for key in profile_keys
            },
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            ),
            "gpu": properties.name,
            "gpu_uuid": str(properties.uuid),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "compute_dtype": str(compute_dtype),
        },
        "distributions": {
            name: importlib.metadata.version(name)
            for name in ("comfy-kitchen", "dinkster-inference")
        },
    }
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _source_environment(dinkster_root: Path, fork_root: Path) -> dict[str, str]:
    sources = (
        str(dinkster_root / "src"),
        *(str(path) for path in sorted((dinkster_root / "packages").glob("*/src"))),
        str(fork_root),
    )
    current = os.environ.get("PYTHONPATH")
    return {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((*sources, *((current,) if current else ()))),
    }


def _spawn_worker(
    args: argparse.Namespace,
    result: Path,
    log: Path,
    environment: dict[str, str],
) -> subprocess.Popen[bytes]:
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--checkpoint",
        str(args.checkpoint),
        "--checkpoint-digest",
        args.checkpoint_digest,
        "--result",
        str(result),
        "--workload",
        args.workload,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
    )
    with log.open("wb") as output:
        return subprocess.Popen(
            command,
            cwd=args.dinkster_root,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )


def _wait(workers: Sequence[tuple[str, subprocess.Popen[bytes], Path]]) -> None:
    failures = []
    for name, worker, log in workers:
        returncode = worker.wait()
        if returncode:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
            failures.append(f"{name} exited {returncode}:\n{tail}")
    if failures:
        raise ReceiptError("; ".join(failures))


def _load(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text()))


def _log_receipt(path: Path) -> dict[str, Any]:
    contents = path.read_bytes()
    lines = contents.decode(errors="replace").splitlines()
    terms = ("NCCL", "P2P", "NET/", "Channel", "NVLS", "SHM")
    return {
        "sha256": hashlib.sha256(contents).hexdigest(),
        "bytes": len(contents),
        "transport_lines": [line for line in lines if any(term in line for term in terms)][-500:],
    }


def _run_serial(args: argparse.Namespace, scratch: Path) -> dict[str, Any]:
    result = scratch / "serial.json"
    log = scratch / "serial.log"
    environment = {
        key: value
        for key, value in _source_environment(args.dinkster_root, args.fork_root).items()
        if not key.startswith("DINKSTER_SINGLE_JOB_")
    }
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu[0]
    worker = _spawn_worker(args, result, log, environment)
    _wait((("serial", worker, log),))
    loaded = _load(result)
    loaded["worker_log"] = _log_receipt(log)
    return loaded


def _run_distributed(args: argparse.Namespace, scratch: Path) -> list[dict[str, Any]]:
    token = secrets.token_hex(16)
    rendezvous = scratch / "rendezvous"
    workers = []
    for rank, gpu in enumerate(args.gpu):
        result = scratch / f"rank{rank}.json"
        log = scratch / f"rank{rank}.log"
        environment = _source_environment(args.dinkster_root, args.fork_root)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "DINKSTER_SINGLE_JOB_RANK": str(rank),
                "DINKSTER_SINGLE_JOB_WORLD_SIZE": str(len(args.gpu)),
                "DINKSTER_SINGLE_JOB_MULTI_GPU_MODE": "window",
                "DINKSTER_SINGLE_JOB_RENDEZVOUS": f"file://{rendezvous}",
                "DINKSTER_SINGLE_JOB_TOKEN": token,
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH",
            }
        )
        workers.append((f"rank{rank}", _spawn_worker(args, result, log, environment), log))
    _wait(workers)
    results = []
    for rank in range(len(args.gpu)):
        loaded = _load(scratch / f"rank{rank}.json")
        loaded["worker_log"] = _log_receipt(scratch / f"rank{rank}.log")
        results.append(loaded)
    return results


def _nvidia_smi(*arguments: str) -> str:
    return subprocess.check_output(("nvidia-smi", *arguments), text=True, timeout=30).strip()


def run_mint(args: argparse.Namespace) -> int:
    from dinkster_assets import digest_file

    args.dinkster_root = args.dinkster_root.resolve()
    args.fork_root = args.fork_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if len(args.gpu) not in (2, 3, 4):
        raise ReceiptError("Flux window receipts require two, three, or four GPU UUIDs")
    if not args.checkpoint.is_file():
        raise ReceiptError(f"checkpoint does not exist: {args.checkpoint}")
    for root in (args.dinkster_root, args.fork_root):
        if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ReceiptError(f"source checkout is dirty: {root}")
    args.checkpoint_digest = digest_file(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flux-window-multigpu-") as directory:
        scratch = Path(directory)
        arm_order = (
            ("candidate", "reference") if args.candidate_first else ("reference", "candidate")
        )
        arms: dict[str, Any] = {}
        for arm in arm_order:
            arms[arm] = (
                _run_serial(args, scratch)
                if arm == "reference"
                else _run_distributed(args, scratch)
            )
    reference = cast("dict[str, Any]", arms["reference"])
    candidate = cast("list[dict[str, Any]]", arms["candidate"])
    hashes = {
        json.dumps(result["output_hashes"], sort_keys=True) for result in (reference, *candidate)
    }
    if len(hashes) != 1:
        raise ReceiptError("serial and distributed output hashes differ")
    reference_seconds = float(reference["timing"]["median_sample_seconds"])
    candidate_seconds = max(
        float(result["timing"]["median_sample_seconds"]) for result in candidate
    )
    reference_job_seconds = float(reference["timing"]["median_whole_job_seconds"])
    candidate_job_seconds = max(
        float(result["timing"]["median_whole_job_seconds"]) for result in candidate
    )
    receipt = {
        "schema": "dinkster.flux-window-multigpu-receipt.v1",
        "status": "PASS",
        "source": {
            "dinkster_head": _git(args.dinkster_root, "rev-parse", "HEAD"),
            "fork_head": _git(args.fork_root, "rev-parse", "HEAD"),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "blake3": args.checkpoint_digest.removeprefix("blake3:"),
        },
        "workload": {
            "name": args.workload,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "prompt": PROMPT,
            "guidance": 3.5,
            "windows": WORKLOADS[args.workload],
            "window_indices": "latent width; retained Flux token indices multiplied by two",
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "execution": {
            "mode": "window",
            "execution_provider": "fp8-weights-bf16-compute",
            "compute_dtype": "bfloat16",
            "attention_policy": "sdpa",
            "world_size": len(candidate),
            "arm_order": list(arm_order),
            "hash_contract": "bit-identical between serial and every distributed rank",
            "performance": {
                "reference_median_sample_seconds": reference_seconds,
                "candidate_median_sample_seconds": candidate_seconds,
                "sample_speedup": reference_seconds / candidate_seconds,
                "reference_median_sample_step_seconds": reference_seconds / args.steps,
                "candidate_median_sample_step_seconds": candidate_seconds / args.steps,
                "sample_step_speedup": reference_seconds / candidate_seconds,
                "reference_median_whole_job_seconds": reference_job_seconds,
                "candidate_median_whole_job_seconds": candidate_job_seconds,
                "whole_job_speedup": reference_job_seconds / candidate_job_seconds,
                "reference_jobs_per_hour": 3600.0 / reference_job_seconds,
                "candidate_jobs_per_hour": 3600.0 / candidate_job_seconds,
                "reference_gpu_seconds_per_job": reference_job_seconds,
                "candidate_gpu_seconds_per_job": candidate_job_seconds * len(candidate),
            },
        },
        "environment": {
            "hostname": os.uname().nodename,
            "nvidia_smi_topology": _nvidia_smi("topo", "-m"),
            "nvidia_smi_p2p_read": _nvidia_smi("topo", "-p2p", "r"),
            "nvidia_smi_p2p_write": _nvidia_smi("topo", "-p2p", "w"),
            "nvidia_smi_inventory": _nvidia_smi(
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
                "--format=csv,noheader",
            ),
            "requested_gpu_uuids": args.gpu,
        },
        "reference": reference,
        "candidate_ranks": candidate,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {args.output}")
    return 0


def _workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    _workload_arguments(worker)
    worker.add_argument("--checkpoint-digest", required=True)
    worker.add_argument("--result", type=Path, required=True)
    worker.set_defaults(function=run_worker)
    mint = commands.add_parser("mint")
    _workload_arguments(mint)
    mint.add_argument("--dinkster-root", type=Path, required=True)
    mint.add_argument("--fork-root", type=Path, required=True)
    mint.add_argument("--gpu", action="append", required=True)
    mint.add_argument("--output", type=Path, required=True)
    mint.add_argument("--candidate-first", action="store_true")
    mint.set_defaults(function=run_mint)
    return parser


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
