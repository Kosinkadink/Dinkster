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
    runtime, token = _route(args.policy)
    model = GenerationLoadDiffusionModel.execute(
        diffusion_model=_asset(args.model, args.model_digest), weight_dtype="default"
    )["model"]
    if args.mode == "sparse":
        model = NativeBlockSparseAttention.execute(
            model=model,
            selection="sol-attn",
            start_percent=0.0,
            end_percent=1.0,
            dense_blocks="",
            min_tokens=1,
            extra_tokens=128,
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
    if distributed:
        activate_distributed_attention(ATTEMPT_GROUP, ATTEMPT)
    try:
        hashes = []
        durations = []
        with use_execution_context(context):
            for repeat in range(args.warmups + args.repeats):
                output_hashes, duration = _run_sample(
                    model, latent, positive, negative, cfg, args.steps, args.seed
                )
                hashes.append(output_hashes)
                if repeat >= args.warmups:
                    durations.append(duration)
    finally:
        if distributed:
            release_distributed_attention(ATTEMPT_GROUP, ATTEMPT)
    if any(value != hashes[0] for value in hashes[1:]):
        raise ReceiptError("MiniMax H3 output was not deterministic across repeats")
    properties = torch.cuda.get_device_properties(0)
    result = {
        "status": "PASS",
        "mode": args.mode,
        "policy": args.policy,
        "distributed": distributed,
        "rank": int(os.environ["DINKSTER_SINGLE_JOB_RANK"]) if distributed else None,
        "world_size": int(os.environ["DINKSTER_SINGLE_JOB_WORLD_SIZE"]) if distributed else 1,
        "hashes": hashes[0],
        "sample_seconds": durations,
        "median_sample_seconds": statistics.median(durations),
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
    )
    return subprocess.Popen(command, cwd=args.dinkster_root, env=environment)


def _wait(workers: Sequence[tuple[str, subprocess.Popen[bytes]]]) -> None:
    failures = []
    for name, worker in workers:
        returncode = worker.wait()
        if returncode:
            failures.append(f"{name} exited {returncode}")
    if failures:
        raise ReceiptError("; ".join(failures))


def _load(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text()))


def _run_serial(args: argparse.Namespace, scratch: Path) -> dict[str, Any]:
    result = scratch / "serial.json"
    environment = _source_environment(args.dinkster_root, args.fork_root)
    environment = {
        key: value
        for key, value in environment.items()
        if not key.startswith("DINKSTER_SINGLE_JOB_")
    }
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu[0]
    worker = _spawn_worker(args, result, environment)
    _wait((("serial", worker),))
    return _load(result)


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
            }
        )
        workers.append((f"rank{rank}", _spawn_worker(args, result, environment)))
    _wait(workers)
    return [_load(scratch / f"rank{rank}.json") for rank in range(len(args.gpu))]


def _nvidia_smi(*arguments: str) -> str:
    return subprocess.check_output(("nvidia-smi", *arguments), text=True, timeout=30).strip()


def _classify_host(gpus: Sequence[dict[str, Any]]) -> str:
    names = {gpu["environment"]["gpu"] for gpu in gpus}
    if names == {"NVIDIA GeForce RTX 4090"}:
        return "X570"
    if names == {"NVIDIA RTX PRO 6000 Blackwell Workstation Edition"}:
        return "RipperPC"
    return os.uname().nodename


def run_mint(args: argparse.Namespace) -> int:
    from dinkster_assets import digest_file

    args.dinkster_root = args.dinkster_root.resolve()
    args.fork_root = args.fork_root.resolve()
    args.model = args.model.resolve()
    if len(args.gpu) != 2 and args.mode != "sparse":
        raise ReceiptError("guidance and U2R1 receipt minting require exactly two GPU UUIDs")
    if not args.model.is_file():
        raise ReceiptError(f"model does not exist: {args.model}")
    for root in (args.dinkster_root, args.fork_root):
        if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ReceiptError(f"source checkout is dirty: {root}")
    args.model_digest = digest_file(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minimax-h3-multigpu-") as directory:
        scratch = Path(directory)
        serial = _run_serial(args, scratch)
        ranks = [] if args.mode == "sparse" else _run_distributed(args, scratch)
    compared = ranks or [serial]
    hashes = {json.dumps(result["hashes"], sort_keys=True) for result in (serial, *ranks)}
    if len(hashes) != 1:
        raise ReceiptError("serial and distributed output hashes differ")
    distributed_seconds = (
        max(float(rank["median_sample_seconds"]) for rank in ranks) if ranks else None
    )
    serial_seconds = float(serial["median_sample_seconds"])
    speedup = serial_seconds / distributed_seconds if distributed_seconds is not None else None
    host = _classify_host(compared)
    workload = (
        "production"
        if (args.width, args.height, args.frames, args.steps) == (1344, 768, 124, 20)
        else "short"
    )
    baseline = BASELINES.get((host, args.mode, args.policy, workload))
    receipt = {
        "schema": "dinkster.minimax-h3-multigpu-receipt.v1",
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
        },
        "execution": {
            "mode": args.mode,
            "attention_policy": args.policy,
            "world_size": len(ranks) if ranks else 1,
            "output_hashes": serial["hashes"],
            "serial_median_seconds": serial_seconds,
            "distributed_median_seconds": distributed_seconds,
            "speedup": speedup,
            "pre_reset_speedup": baseline,
            "speedup_ratio_to_pre_reset": speedup / baseline if speedup and baseline else None,
        },
        "environment": {
            "host": host,
            "hostname": os.uname().nodename,
            "nvidia_smi_topology": _nvidia_smi("topo", "-m"),
            "nvidia_smi_inventory": _nvidia_smi(
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
                "--format=csv,noheader",
            ),
            "requested_gpu_uuids": args.gpu,
        },
        "serial": serial,
        "distributed_ranks": ranks,
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
    mint.set_defaults(function=run_mint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
