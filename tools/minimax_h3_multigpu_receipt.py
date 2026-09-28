"""Measure fork-backed MiniMax H3 single-job multi-GPU execution."""

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

BASELINES = {
    ("RipperPC", "guidance", "sdpa", "short"): 1.974,
    ("RipperPC", "guidance", "sdpa", "production"): 1.802,
    ("RipperPC", "sequence", "sdpa", "production"): 1.5404,
    ("RipperPC", "sequence", "dinkster_kitchen_int8", "production"): 1.436792013659844,
    ("X570", "sequence", "dinkster_kitchen_int8", "production"): 1.327,
}
POLICIES = ("sdpa", "dinkster_kitchen_int8")
MODES = ("guidance", "sequence", "sparse")
ATTEMPT_GROUP = "minimax-h3-multigpu-receipt"
ATTEMPT = 1


class ReceiptError(RuntimeError):
    pass


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *arguments), text=True, timeout=30
    ).strip()


def _tensor_sha256(value: Any) -> str:
    torch = cast("Any", importlib.import_module("torch"))
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _latent_hashes(samples: Any) -> dict[str, str]:
    return {role: _tensor_sha256(samples.by_role(role)) for role in samples.roles}


def _asset(path: Path, digest: str) -> Any:
    from dinkster_assets import AssetRef

    return AssetRef(digest, path.name, path.stat().st_size, resolver=_Resolver(path))


def _conditioning(torch: Any, seed: int) -> list[list[object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn((1, 32, 5120), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    return [[value, {}]]


def _route(policy: str) -> tuple[Any, Any]:
    from dinkster_native.attention import create_attention_runtime
    from dinkster_protocol import (
        AttentionPolicy,
        AttentionPolicyConfig,
        derive_attention_route_token,
    )

    runtime = create_attention_runtime()
    token = derive_attention_route_token(
        runtime.capabilities,
        AttentionPolicyConfig(requested_policy=cast("AttentionPolicy", policy)),
    )
    return runtime, token


def _inventory() -> dict[str, Any]:
    from dinkster.compose import default_pack_specs

    packs = []
    for spec in default_pack_specs():
        for pack, info in sorted((spec.packs or {}).items()):
            packs.append(
                {
                    "pack": pack,
                    "artifact_digest": info.artifact_digest,
                    "source": info.source,
                    "version": info.version,
                }
            )
    distributions = {}
    for name in ("comfy-kitchen", "dinkster-comfy"):
        distributions[name] = importlib.metadata.version(name)
    return {
        "executable": sys.executable,
        "packs": packs,
        "distributions": distributions,
    }


def _run_sample(
    model: Any,
    latent: dict[str, object],
    positive: object,
    negative: object,
    cfg: float,
    steps: int,
    seed: int,
) -> tuple[dict[str, str], float]:
    from dinkster_native.fork_nodes import GenerationKSampler

    torch = cast("Any", importlib.import_module("torch"))
    torch.cuda.synchronize()
    started = time.perf_counter()
    sampled = cast(
        "dict[str, Any]",
        GenerationKSampler.execute(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0,
        )["latent"],
    )
    torch.cuda.synchronize()
    return _latent_hashes(sampled["samples"]), time.perf_counter() - started


def run_worker(args: argparse.Namespace) -> int:
    worker_started = time.perf_counter()
    inventory = _inventory()
    torch = cast("Any", importlib.import_module("torch"))
    from dinkster_native.attention import (
        activate_distributed_attention,
        release_distributed_attention,
    )
    from dinkster_native.fork_nodes import (
        GenerationLoadDiffusionModel,
        NativeBlockSparseAttention,
        NativeEmptyMiniMaxH3AV,
    )
    from dinkster_workers import ExecutionContext, use_execution_context

    if not torch.cuda.is_available():
        raise ReceiptError("MiniMax H3 receipt worker requires CUDA")
    setup_started = time.perf_counter()
    runtime, token = _route(args.policy)
    load_started = time.perf_counter()
    model = GenerationLoadDiffusionModel.execute(
        diffusion_model=_asset(args.model, args.model_digest), weight_dtype="default"
    )["model"]
    model_load_seconds = time.perf_counter() - load_started
    if args.sparse_enabled:
        model = NativeBlockSparseAttention.execute(
            model=model,
            selection="sol-attn",
            start_percent=0.2,
            end_percent=1.0,
            dense_blocks="",
            min_tokens=12_288,
            extra_tokens=256,
            sink_conditioning="exact_kv_and_rows",
            tau=1.3,
            keep_percent=10.0,
        )["MODEL"]
    latent = cast(
        "dict[str, object]",
        NativeEmptyMiniMaxH3AV.execute(
            width=args.width, height=args.height, frame_count=args.frames
        )["latent"],
    )
    positive = _conditioning(torch, 220)
    negative = _conditioning(torch, 221) if args.mode == "guidance" else []
    cfg = 2.0 if args.mode == "guidance" else 1.0
    distributed = "DINKSTER_SINGLE_JOB_RANK" in os.environ
    context = ExecutionContext(
        arm="native",
        expected_execution_identity=None,
        attention_policy=cast("Any", args.policy),
        attention_route_token=token,
        attention_capabilities=runtime.capabilities,
        attention_runtime=runtime,
    )
    setup_seconds = time.perf_counter() - setup_started
    if distributed:
        activate_distributed_attention(ATTEMPT_GROUP, ATTEMPT)
    try:
        with use_execution_context(context):
            cold_hashes, cold_seconds = _run_sample(
                model, latent, positive, negative, cfg, args.steps, args.seed
            )
            cold_workflow_seconds = time.perf_counter() - worker_started
            warmup_runs = [
                _run_sample(model, latent, positive, negative, cfg, args.steps, args.seed)
                for _ in range(args.warmups)
            ]
            measured_runs = [
                _run_sample(model, latent, positive, negative, cfg, args.steps, args.seed)
                for _ in range(args.repeats)
            ]
    finally:
        if distributed:
            release_distributed_attention(ATTEMPT_GROUP, ATTEMPT)
    warmup_seconds = [run[1] for run in warmup_runs]
    sample_seconds = [run[1] for run in measured_runs]
    median_sample_seconds = statistics.median(sample_seconds)
    properties = torch.cuda.get_device_properties(0)
    result = {
        "status": "PASS",
        "mode": args.mode,
        "policy": args.policy,
        "distributed": distributed,
        "rank": int(os.environ["DINKSTER_SINGLE_JOB_RANK"]) if distributed else None,
        "world_size": int(os.environ["DINKSTER_SINGLE_JOB_WORLD_SIZE"]) if distributed else 1,
        "output_hashes": {
            "cold": cold_hashes,
            "warmups": [run[0] for run in warmup_runs],
            "measured": [run[0] for run in measured_runs],
        },
        "timing": {
            "worker_wall_seconds": time.perf_counter() - worker_started,
            "setup_seconds": setup_seconds,
            "model_load_seconds": model_load_seconds,
            "cold_workflow_seconds": cold_workflow_seconds,
            "cold_sample_seconds": cold_seconds,
            "warmup_sample_seconds": warmup_seconds,
            "sample_seconds": sample_seconds,
            "median_warm_workflow_seconds": median_sample_seconds,
            "median_sample_seconds": median_sample_seconds,
            "jobs_per_hour": 3600.0 / median_sample_seconds,
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
        },
        "inventory": inventory,
    }
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _source_environment(dinkster_root: Path, fork_root: Path) -> dict[str, str]:
    package_sources = sorted(
        str(path) for path in (dinkster_root / "packages").glob("*/src") if path.is_dir()
    )
    sources = (str(dinkster_root / "src"), *package_sources, str(fork_root))
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
        "--model",
        str(args.model),
        "--model-digest",
        args.model_digest,
        "--result",
        str(result),
        "--mode",
        args.mode,
        "--policy",
        args.policy,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--frames",
        str(args.frames),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        *(("--sparse-enabled",) if args.sparse_enabled else ()),
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


def _run_serial(
    args: argparse.Namespace, scratch: Path, *, sparse_enabled: bool = False
) -> dict[str, Any]:
    name = "sparse" if sparse_enabled else "serial"
    result = scratch / f"{name}.json"
    log = scratch / f"{name}.log"
    environment = _source_environment(args.dinkster_root, args.fork_root)
    environment = {
        key: value
        for key, value in environment.items()
        if not key.startswith("DINKSTER_SINGLE_JOB_")
    }
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu[0]
    args.sparse_enabled = sparse_enabled
    worker = _spawn_worker(args, result, log, environment)
    _wait(((name, worker, log),))
    loaded = _load(result)
    loaded["worker_log"] = _log_receipt(log)
    return loaded


def _run_distributed(args: argparse.Namespace, scratch: Path) -> list[dict[str, Any]]:
    if args.mode == "sparse":
        raise ReceiptError("sparse attention is measured as a one-GPU registry route")
    token = secrets.token_hex(16)
    rendezvous = scratch / "rendezvous"
    workers = []
    for rank, gpu in enumerate(args.gpu):
        result = scratch / f"rank{rank}.json"
        environment = _source_environment(args.dinkster_root, args.fork_root)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "DINKSTER_SINGLE_JOB_RANK": str(rank),
                "DINKSTER_SINGLE_JOB_WORLD_SIZE": str(len(args.gpu)),
                "DINKSTER_SINGLE_JOB_MULTI_GPU_MODE": args.mode,
                "DINKSTER_SINGLE_JOB_RENDEZVOUS": f"file://{rendezvous}",
                "DINKSTER_SINGLE_JOB_TOKEN": token,
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH",
            }
        )
        log = scratch / f"rank{rank}.log"
        args.sparse_enabled = False
        workers.append((f"rank{rank}", _spawn_worker(args, result, log, environment), log))
    _wait(workers)
    results = []
    for rank in range(len(args.gpu)):
        loaded = _load(scratch / f"rank{rank}.json")
        loaded["worker_log"] = _log_receipt(scratch / f"rank{rank}.log")
        results.append(loaded)
    return results


def _log_receipt(path: Path) -> dict[str, Any]:
    contents = path.read_bytes()
    lines = contents.decode(errors="replace").splitlines()
    transport_terms = ("NCCL", "P2P", "NET/", "Channel", "NVLS", "SHM")
    return {
        "sha256": hashlib.sha256(contents).hexdigest(),
        "bytes": len(contents),
        "transport_lines": [
            line for line in lines if any(term in line for term in transport_terms)
        ][-500:],
    }


def _nvidia_smi(*arguments: str) -> str:
    return subprocess.check_output(("nvidia-smi", *arguments), text=True, timeout=30).strip()


def _classify_host(gpus: Sequence[dict[str, Any]]) -> str:
    names = {gpu["environment"]["gpu"] for gpu in gpus}
    if names == {"NVIDIA GeForce RTX 4090"}:
        return "X570"
    if names == {"NVIDIA RTX PRO 6000 Blackwell Workstation Edition"}:
        return "RipperPC"
    return os.uname().nodename


def _median_seconds(result: dict[str, Any]) -> float:
    return float(result["timing"]["median_sample_seconds"])


def _performance(
    reference: dict[str, Any], candidate: Sequence[dict[str, Any]]
) -> dict[str, float]:
    reference_seconds = _median_seconds(reference)
    candidate_seconds = max(_median_seconds(result) for result in candidate)
    world_size = len(candidate)
    return {
        "reference_median_seconds": reference_seconds,
        "candidate_median_seconds": candidate_seconds,
        "speedup": reference_seconds / candidate_seconds,
        "reference_jobs_per_hour": 3600.0 / reference_seconds,
        "candidate_jobs_per_hour": 3600.0 / candidate_seconds,
        "reference_gpu_seconds_per_job": reference_seconds,
        "candidate_gpu_seconds_per_job": candidate_seconds * world_size,
    }


def run_mint(args: argparse.Namespace) -> int:
    from dinkster_assets import digest_file

    args.dinkster_root = args.dinkster_root.resolve()
    args.fork_root = args.fork_root.resolve()
    args.model = args.model.resolve()
    if len(args.gpu) != 2 and args.mode != "sparse":
        raise ReceiptError("guidance and U2R1 receipt minting require exactly two GPU UUIDs")
    if args.mode == "sparse" and len(args.gpu) != 1:
        raise ReceiptError("sparse attention receipt minting requires exactly one GPU UUID")
    if args.mode == "sparse" and args.policy != "sdpa":
        raise ReceiptError("sparse attention uses SDPA as its dense reference policy")
    if not args.model.is_file():
        raise ReceiptError(f"model does not exist: {args.model}")
    for root in (args.dinkster_root, args.fork_root):
        if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ReceiptError(f"source checkout is dirty: {root}")
    args.model_digest = digest_file(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minimax-h3-multigpu-") as directory:
        scratch = Path(directory)
        arm_order = (
            ("candidate", "reference") if args.candidate_first else ("reference", "candidate")
        )
        arms: dict[str, Any] = {}
        for arm in arm_order:
            if arm == "reference":
                arms[arm] = _run_serial(args, scratch)
            elif args.mode == "sparse":
                arms[arm] = [_run_serial(args, scratch, sparse_enabled=True)]
            else:
                arms[arm] = _run_distributed(args, scratch)
    serial = cast("dict[str, Any]", arms["reference"])
    candidate = cast("list[dict[str, Any]]", arms["candidate"])
    if args.mode != "sparse":
        hashes = {
            json.dumps(result["output_hashes"], sort_keys=True)
            for result in (serial, *candidate)
        }
        if len(hashes) != 1:
            raise ReceiptError("serial and distributed output hashes differ")
    performance = _performance(serial, candidate)
    speedup = performance["speedup"]
    host = _classify_host(candidate)
    workload = (
        "production"
        if (args.width, args.height, args.frames, args.steps) == (1344, 768, 124, 20)
        else "short"
    )
    baseline = BASELINES.get((host, args.mode, args.policy, workload))
    receipt = {
        "schema": "dinkster.minimax-h3-multigpu-receipt.v2",
        "status": "PASS",
        "source": {
            "dinkster_head": _git(args.dinkster_root, "rev-parse", "HEAD"),
            "fork_head": _git(args.fork_root, "rev-parse", "HEAD"),
        },
        "model": {
            "path": str(args.model),
            "bytes": args.model.stat().st_size,
            "blake3": args.model_digest.removeprefix("blake3:"),
        },
        "workload": {
            "name": workload,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": args.steps,
            "seed": args.seed,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "sampler": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "guidance": 2.0 if args.mode == "guidance" else 1.0,
            "conditioning": "synthetic 32-token bfloat16 context with seed 220",
            "sparse_config": (
                {
                    "selection": "sol-attn",
                    "start_percent": 0.2,
                    "end_percent": 1.0,
                    "dense_blocks": [],
                    "min_tokens": 12_288,
                    "extra_tokens": 256,
                    "sink_conditioning": "exact_kv_and_rows",
                    "tau": 1.3,
                }
                if args.mode == "sparse"
                else None
            ),
        },
        "execution": {
            "mode": args.mode,
            "execution_provider": "bf16-linear",
            "compute_dtype": "bfloat16",
            "attention_policy": args.policy,
            "attention_provider": {
                "sdpa": "torch-sdpa-priority-v1",
                "dinkster_kitchen_int8": "comfy-kitchen-int8-attention-v1",
            }[args.policy],
            "sparse_backend": "comfy_kitchen_sol_chunked" if args.mode == "sparse" else None,
            "world_size": len(candidate),
            "arm_order": list(arm_order),
            "measurement_boundaries": {
                "cold_workflow": (
                    "worker process start through model load, setup, and first KSampler output"
                ),
                "warm_workflow": "one warm KSampler node invocation from resident inputs",
                "sampling_only": "the same KSampler invocation; VAE decode is excluded",
                "wall_time": "synchronized host wall clock around each KSampler invocation",
            },
            "reference_output_hashes": serial["output_hashes"],
            "candidate_output_hashes": candidate[0]["output_hashes"],
            "hash_contract": (
                "all run hashes retained; sparse output may differ from dense reference"
                if args.mode == "sparse"
                else "bit-identical between serial and every distributed rank"
            ),
            "performance": performance,
            "pre_reset_speedup": baseline,
            "speedup_ratio_to_pre_reset": speedup / baseline if speedup and baseline else None,
        },
        "environment": {
            "host": host,
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
        "reference": serial,
        "candidate_ranks": candidate,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {args.output}")
    return 0


def _workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sparse-enabled", action="store_true", help=argparse.SUPPRESS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    _workload_arguments(worker)
    worker.add_argument("--model-digest", required=True)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
