"""Mint distributed window-scatter receipts for Flux dev BF16.

The mint drives production sampling end to end: every rank process
configures the single-job environment, loads the runtime, and calls
``FluxRuntime.sample`` with a multi-window plan so manifest consensus,
fences, scatter, gather, and canonical merge all execute exactly as
they do in production. The acceptance criterion is exact final latent
bytes: every rank of every group size must match the same windowed plan
sampled serially on one GPU.

Window scatter is circumstantial acceleration. Its measured speedups
are informational only and are never counted toward the core multi-GPU
performance goals tracked by issues #121 and #298.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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
from typing import Any

CHECKPOINT_BYTES = 17_246_524_772
CHECKPOINT_SHA256 = "8e91b68084b53a7fc44ed2a3756d821e355ac1a7b6fe29be760c1db532f3d88a"
CHECKPOINT_REVISION = "40a8a3d745c7d7adb077cb19879a975aa19c847b"
CHECKPOINT_URL = (
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/"
    f"{CHECKPOINT_REVISION}/flux1-dev-fp8.safetensors"
)

PROMPT = "A red fox sitting beside a mossy stone in a sunlit forest, detailed photograph"
SEED = 424242
STEPS = 4
GUIDANCE = 3.5
WIDTH = 512
HEIGHT = 512
WIDTH_TOKENS = WIDTH // 16
MEASURED_RUNS = 5
FAMILY_ID = "dinkster.flux_dev"
INTEGRATION_FACTS = ("topology=window", "window_evaluation=full-replica-window-scatter")
WORLD_SIZES = (2, 4)
ATTEMPT_GROUP = "flux-window-receipt"
ATTEMPT = 1
WORKER_TIMEOUT_SECONDS = 3600
CIRCUMSTANTIAL_STATEMENT = (
    "Window scatter is circumstantial acceleration: it parallelizes only the"
    " per-step window evaluations a windowed plan already performs. Its measured"
    " performance is informational only and is never counted toward the core"
    " multi-GPU performance goals tracked by issues #121 and #298."
)

# Two overlapping windows exercise the minimum scatter; three windows
# leave one rank idle at world size 4 and split unevenly at world size
# 2, proving zero-window and uneven assignments merge identically.
WORKLOADS: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = (
    ("two-windows", (tuple(range(24)), tuple(range(8, 32)))),
    ("three-windows", (tuple(range(14)), tuple(range(9, 23)), tuple(range(18, 32)))),
)


class ReceiptMintError(RuntimeError):
    """The mint environment or a measured result is invalid."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _checkpoint_receipt(checkpoint: Path) -> dict[str, Any]:
    size = checkpoint.stat().st_size
    if size != CHECKPOINT_BYTES:
        raise ReceiptMintError(f"checkpoint size must be {CHECKPOINT_BYTES} bytes, got {size}")
    digest = _file_sha256(checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ReceiptMintError(f"checkpoint sha256 must be {CHECKPOINT_SHA256}, got {digest}")
    return {
        "bytes": size,
        "path": str(checkpoint.resolve()),
        "revision": CHECKPOINT_REVISION,
        "sha256": digest,
        "url": CHECKPOINT_URL,
    }


def _driver_version() -> str:
    completed = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=True,
        capture_output=True,
        text=True,
    )
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise ReceiptMintError(f"expected one NVIDIA driver version, got {sorted(versions)}")
    return versions.pop()


def _gpu_inventory() -> list[dict[str, str]]:
    completed = subprocess.run(
        ("nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"),
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, name, uuid = (part.strip() for part in line.split(",", maxsplit=2))
        inventory.append({"index": index, "name": name, "uuid": uuid})
    return inventory


def _add_dinkster_sources(dinkster_root: Path) -> None:
    for source in sorted((dinkster_root / "packages").glob("*/src")):
        sys.path.insert(0, str(source))


def run_worker(args: argparse.Namespace) -> int:
    dinkster_root = args.dinkster_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    _add_dinkster_sources(dinkster_root)

    import torch
    from dinkster_inference import (
        AccumulationDType,
        IntegerAffineIndexMap,
        KindAxisMap,
        LayerWindow,
        MediaAxis,
        MergeDeclaration,
        WindowIndexList,
        WindowKind,
        WindowPlanLayer,
        WindowWeightKind,
        WindowWeightProfile,
        compile_window_plan,
        load_safetensors_header,
    )
    from dinkster_inference_torch import load_runtime
    from dinkster_inference_torch.distributed import (
        activate_distributed_sampling_attempt,
        release_distributed_sampling_attempt,
    )
    from dinkster_inference_torch.flux_window_distributed import window_receipt_identity

    distributed = "DINKSTER_SINGLE_JOB_RANK" in os.environ
    if distributed:
        activate_distributed_sampling_attempt(ATTEMPT_GROUP, ATTEMPT)
    try:
        device = torch.device("cuda:0")
        if torch.cuda.device_count() != 1:
            raise ReceiptMintError(
                f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
            )
        checkpoint = load_safetensors_header(checkpoint_path)
        runtime = load_runtime(
            checkpoint=checkpoint,
            diffusion_dtype=torch.bfloat16,
            text_dtype=torch.float32,
            vae_dtype=torch.float32,
            fp8_matmul=False,
            attention_policy="auto",
        )
        if runtime.assembled.compute_dtype("diffusion") is not torch.bfloat16:
            raise ReceiptMintError("diffusion compute dtype must be bfloat16")
        assert runtime.assembled.clip_l is not None
        assert runtime.assembled.t5xxl is not None
        runtime.assembled.clip_l.to(device)
        runtime.assembled.t5xxl.to(device)
        with torch.inference_mode():
            cond = runtime.encode_text(PROMPT)
        runtime.assembled.clip_l.to("cpu")
        runtime.assembled.t5xxl.to("cpu")
        torch.cuda.empty_cache()
        runtime.assembled.diffusion.to(device)

        width_axis = MediaAxis("width", WIDTH_TOKENS)
        kinds = (
            WindowKind(
                "latent_image",
                (KindAxisMap("width", WIDTH_TOKENS, IntegerAffineIndexMap(1)),),
            ),
            WindowKind("text", invariant_axes=("width",)),
        )

        def make_plan(windows: tuple[tuple[int, ...], ...]) -> Any:
            return compile_window_plan(
                axes=(width_axis,),
                kinds=kinds,
                layers=(
                    WindowPlanLayer(
                        ("width",),
                        tuple(LayerWindow((WindowIndexList(indices),)) for indices in windows),
                        (WindowWeightProfile(WindowWeightKind.FLAT),),
                        MergeDeclaration(AccumulationDType.FLOAT32),
                    ),
                ),
            )

        latent = torch.zeros((1, 16, HEIGHT // 8, WIDTH // 8), dtype=torch.float32, device=device)

        def sample(plan: Any) -> Any:
            with torch.inference_mode():
                return runtime.sample(
                    latent,
                    cond=cond,
                    sampler_id="dinkster.euler",
                    scheduler_id="dinkster.normal",
                    steps=STEPS,
                    denoise=1.0,
                    seed=SEED,
                    guidance=GUIDANCE,
                    window_plan=plan,
                    compute_dtype=torch.bfloat16,
                    device=device,
                )

        workloads: dict[str, Any] = {}
        for name, windows in WORKLOADS:
            plan = make_plan(windows)
            sample(plan)
            durations: list[float] = []
            hashes: set[str] = set()
            for _ in range(MEASURED_RUNS):
                torch.cuda.synchronize(device)
                start = time.perf_counter_ns()
                sampled = sample(plan)
                torch.cuda.synchronize(device)
                durations.append((time.perf_counter_ns() - start) / 1e9)
                array = sampled.detach().float().cpu().numpy()
                hashes.add(hashlib.sha256(array.tobytes(order="C")).hexdigest())
            if len(hashes) != 1:
                raise ReceiptMintError(
                    f"workload {name} was not deterministic across repeated runs: {sorted(hashes)}"
                )
            workloads[name] = {
                "latent_sha256": hashes.pop(),
                "median_sample_seconds": statistics.median(durations),
                "sample_seconds": durations,
            }

        properties = torch.cuda.get_device_properties(device)
        result = {
            "claimed_receipt_identity": window_receipt_identity(
                runtime.family.id,
                INTEGRATION_FACTS,
                torch.bfloat16,
                int(os.environ.get("DINKSTER_SINGLE_JOB_WORLD_SIZE", "2")),
                ("cuda", torch.cuda.get_device_capability(device)),
            ),
            "distributed": distributed,
            "environment": {
                "cuda": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device_capability": ".".join(
                    str(value) for value in torch.cuda.get_device_capability(device)
                ),
                "gpu": properties.name,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
            },
            "rank": int(os.environ["DINKSTER_SINGLE_JOB_RANK"]) if distributed else None,
            "workloads": workloads,
            "world_size": (int(os.environ["DINKSTER_SINGLE_JOB_WORLD_SIZE"]) if distributed else 1),
        }
    finally:
        if distributed:
            release_distributed_sampling_attempt(ATTEMPT_GROUP, ATTEMPT)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _spawn_worker(
    dinkster_root: Path,
    checkpoint: Path,
    result_path: Path,
    environment: dict[str, str],
) -> subprocess.Popen[bytes]:
    command = (
        sys.executable,
        "-m",
        "tools.inference_parity.flux_window_receipts",
        "worker",
        "--checkpoint",
        str(checkpoint),
        "--dinkster-root",
        str(dinkster_root),
        "--result",
        str(result_path),
    )
    return subprocess.Popen(command, cwd=dinkster_root, env=environment)


def _base_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DINKSTER_SINGLE_JOB_")
    }
    return environment


def _wait_all(workers: Sequence[tuple[str, subprocess.Popen[bytes]]]) -> None:
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    failures = []
    for name, worker in workers:
        remaining = max(1.0, deadline - time.monotonic())
        try:
            returncode = worker.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            for _, victim in workers:
                victim.kill()
            raise ReceiptMintError(f"worker {name} timed out") from None
        if returncode != 0:
            failures.append(f"{name} exited {returncode}")
    if failures:
        raise ReceiptMintError("; ".join(failures))


def _run_serial_reference(dinkster_root: Path, checkpoint: Path, scratch: Path) -> dict[str, Any]:
    result_path = scratch / "serial.json"
    environment = _base_environment()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    worker = _spawn_worker(dinkster_root, checkpoint, result_path, environment)
    _wait_all((("serial", worker),))
    return json.loads(result_path.read_text())


def _run_distributed_group(
    dinkster_root: Path, checkpoint: Path, scratch: Path, world_size: int
) -> list[dict[str, Any]]:
    token = secrets.token_hex(16)
    rendezvous = scratch / f"rendezvous-w{world_size}"
    workers = []
    for rank in range(world_size):
        environment = _base_environment()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank)
        environment["DINKSTER_SINGLE_JOB_RANK"] = str(rank)
        environment["DINKSTER_SINGLE_JOB_WORLD_SIZE"] = str(world_size)
        environment["DINKSTER_SINGLE_JOB_MULTI_GPU_MODE"] = "window"
        environment["DINKSTER_SINGLE_JOB_RENDEZVOUS"] = f"file://{rendezvous}"
        environment["DINKSTER_SINGLE_JOB_TOKEN"] = token
        result_path = scratch / f"w{world_size}-rank{rank}.json"
        workers.append(
            (
                f"w{world_size}-rank{rank}",
                _spawn_worker(dinkster_root, checkpoint, result_path, environment),
            )
        )
    _wait_all(workers)
    return [
        json.loads((scratch / f"w{world_size}-rank{rank}.json").read_text())
        for rank in range(world_size)
    ]


def _expected_identity(dinkster_root: Path, world_size: int) -> tuple[str, tuple[str, ...]]:
    _add_dinkster_sources(dinkster_root)
    import torch
    from dinkster_inference_torch.flux_window_distributed import window_receipt_identity

    identity = window_receipt_identity(FAMILY_ID, INTEGRATION_FACTS, torch.bfloat16, world_size)
    preimage = (
        "domain=dinkster.distributed.window-receipt.v2",
        f"family={FAMILY_ID}",
        *INTEGRATION_FACTS,
        "compute_dtype=bfloat16",
        f"world_size={world_size}",
    )
    return identity, preimage


def _validate_group(
    reference: dict[str, Any],
    ranks: list[dict[str, Any]],
    world_size: int,
    expected_identity: str,
) -> list[dict[str, Any]]:
    workload_records = []
    identities = {rank["claimed_receipt_identity"] for rank in ranks}
    if identities != {expected_identity}:
        raise ReceiptMintError(
            f"world {world_size} runtime claims {sorted(identities)} do not match"
            f" the measured execution identity {expected_identity}"
        )
    for name, windows in WORKLOADS:
        expected = reference["workloads"][name]["latent_sha256"]
        observed = {rank["workloads"][name]["latent_sha256"] for rank in ranks}
        matched = observed == {expected}
        workload_records.append(
            {
                "expected_sha256": expected,
                "height": HEIGHT,
                "name": name,
                "result": "PASS" if matched else "FAIL",
                "steps": STEPS,
                "width": WIDTH,
                "windows": [list(window) for window in windows],
            }
        )
        if not matched:
            raise ReceiptMintError(
                f"world {world_size} workload {name} diverged from the serial"
                f" reference: expected {expected}, observed {sorted(observed)}"
            )
    return workload_records


def _performance_record(reference: dict[str, Any], ranks: list[dict[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": CIRCUMSTANTIAL_STATEMENT,
        "predeclared_criterion": (
            "informational only: no pass/fail performance gate; parity is the"
            " sole acceptance criterion"
        ),
        "workloads": {},
    }
    for name, _windows in WORKLOADS:
        serial = float(reference["workloads"][name]["median_sample_seconds"])
        distributed = max(float(rank["workloads"][name]["median_sample_seconds"]) for rank in ranks)
        record["workloads"][name] = {
            "distributed_median_sample_seconds": distributed,
            "serial_median_sample_seconds": serial,
            "speedup_serial_over_distributed": serial / distributed,
        }
    return record


def _write_record(
    output_dir: Path,
    world_size: int,
    identity: str,
    preimage: tuple[str, ...],
    checkpoint_receipt: dict[str, Any],
    proof_workloads: list[dict[str, Any]],
    performance: dict[str, Any],
    reference: dict[str, Any],
    ranks: list[dict[str, Any]],
    dinkster_root: Path,
) -> Path:
    record_dir = output_dir / f"distributed-flux-dev-bf16-window-w{world_size}-d1"
    record_dir.mkdir(parents=True, exist_ok=True)
    validation = {
        "distributed_ranks": ranks,
        "serial_reference": reference,
    }
    validation_path = record_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    diff_command = (
        "git -c core.abbrev=40 diff --full-index --binary --no-ext-diff --no-textconv HEAD -- ."
    )
    diff_text = _git(
        dinkster_root,
        "-c",
        "core.abbrev=40",
        "diff",
        "--full-index",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
        ".",
    )
    receipt = {
        "schema": "dinkster.distributed-window-receipt.v1",
        "status": "PASS",
        "receipt_identity": identity,
        "hashed_preimage": list(preimage),
        "circumstantial_acceleration": CIRCUMSTANTIAL_STATEMENT,
        "proof": {
            "date_utc": datetime.datetime.now(datetime.UTC).date().isoformat(),
            "measured_code_head": _git(dinkster_root, "rev-parse", "HEAD"),
            "measured_worktree_diff_sha256": hashlib.sha256(diff_text.encode("utf-8")).hexdigest(),
            "measured_worktree_diff_command": diff_command,
            "checkpoint_sha256": checkpoint_receipt["sha256"],
            "checkpoint_bytes": checkpoint_receipt["bytes"],
            "seed": SEED,
            "prompt": PROMPT,
            "guidance": GUIDANCE,
            "sampler": "dinkster.euler",
            "scheduler": "dinkster.normal",
            "mode": "window",
            "route": (
                "single-job rank environment -> FluxRuntime.sample -> manifest"
                " consensus -> fenced window scatter -> gathered canonical merge"
            ),
            "criterion": {
                "reference": (
                    "the same multi-window plans sampled serially on one GPU"
                    " without a distributed group"
                ),
                "acceptance": ("exact final latent bytes on every rank for every workload"),
                "tolerance_widening": False,
            },
            "workloads": proof_workloads,
        },
        "performance": performance,
        "environment": {
            "host": os.uname().nodename,
            "gpus": _gpu_inventory(),
            "driver": _driver_version(),
            "rank_environments": [rank["environment"] for rank in ranks],
            "serial_environment": reference["environment"],
        },
        "evidence": {
            "issue": "https://github.com/Kosinkadink/Dinkster/issues/268",
            "validation": "validation.json",
            "validation_sha256": _file_sha256(validation_path),
        },
    }
    (record_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n")
    windows_note = "\n".join(
        f"- {name}: windows {[list(window) for window in windows]}" for name, windows in WORKLOADS
    )
    (record_dir / "README.md").write_text(
        f"# Flux dev BF16 window-scatter receipt (world size {world_size})\n\n"
        f"{CIRCUMSTANTIAL_STATEMENT}\n\n"
        "Minted by tools/inference_parity/flux_window_receipts.py. Every rank of"
        f" the {world_size}-rank single-job group produced final latent bytes"
        " identical to the serial one-GPU reference for each workload:\n\n"
        f"{windows_note}\n\n"
        "The three-window workload leaves ranks without windows at world size 4"
        " and splits unevenly at world size 2, so idle-rank collectives and"
        " uneven deterministic assignment are both covered.\n\n"
        "See receipt.json for the proof and validation.json for the raw"
        " per-rank measurements.\n"
    )
    return record_dir


def run_mint(args: argparse.Namespace) -> int:
    dinkster_root = args.dinkster_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    inventory = _gpu_inventory()
    if len(inventory) < max(WORLD_SIZES):
        raise ReceiptMintError(f"minting needs {max(WORLD_SIZES)} GPUs, found {len(inventory)}")
    checkpoint_receipt = _checkpoint_receipt(checkpoint)
    with tempfile.TemporaryDirectory(prefix="flux-window-receipts-") as scratch_text:
        scratch = Path(scratch_text)
        print("running serial one-GPU reference", flush=True)
        reference = _run_serial_reference(dinkster_root, checkpoint, scratch)
        for world_size in WORLD_SIZES:
            print(f"running {world_size}-rank window-scatter group", flush=True)
            ranks = _run_distributed_group(dinkster_root, checkpoint, scratch, world_size)
            identity, preimage = _expected_identity(dinkster_root, world_size)
            proof_workloads = _validate_group(reference, ranks, world_size, identity)
            performance = _performance_record(reference, ranks)
            record_dir = _write_record(
                output_dir,
                world_size,
                identity,
                preimage,
                checkpoint_receipt,
                proof_workloads,
                performance,
                reference,
                ranks,
                dinkster_root,
            )
            print(f"world {world_size} PASS -> {record_dir}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--dinkster-root", type=Path, required=True)
    worker.add_argument("--result", type=Path, required=True)
    worker.set_defaults(function=run_worker)

    mint = subparsers.add_parser("mint")
    mint.add_argument("--checkpoint", type=Path, required=True)
    mint.add_argument("--dinkster-root", type=Path, required=True)
    mint.add_argument("--output-dir", type=Path, required=True)
    mint.set_defaults(function=run_mint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
