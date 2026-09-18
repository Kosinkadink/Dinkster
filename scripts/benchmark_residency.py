"""Measure one residency-mechanism A/B cell on real hardware.

One invocation enrolls a synthetic weight working set in module
residency under one mechanism and measures steady-state offloaded
forward passes:

    .venv-gpu/bin/python scripts/benchmark_residency.py \\
        --mechanism aimdo --regime open --json aimdo-open.json

The workload reproduces the residency A/B probe from issue #258 (the
evidence base for issue #826): --blocks stacked blocks of
--layers-per-block Linear(--features, --features, bias=False) layers,
1024 MiB of float16 weights at the defaults, offloaded to the CPU and
streamed to the load device on every forward. --weights q8_0 swaps the
dense layers for GgufEncodedLinear layers whose weights stay resident
as encoded Q8_0 uint8 blocks (544 MiB at the defaults) and decode on
the load device each forward, so the streamed bytes are the stored
blocks, not materialized floats. Mechanisms: eager
(ResidentWeights) and aimdo (AimdoWeights demand paging).
--queue adds the block-loop prefetch queue, which streams each
block's weights one block ahead of the consuming forward. Outputs are
verified bit-identical against a fully resident reference before
anything is measured; a mismatch fails the run.

Regimes: open leaves free device memory alone. constrained allocates a
ballast tensor before mechanism setup so that roughly --leave-free-mib
MiB of the device stays free - below the working set at the defaults -
forcing eviction or streaming thrash. The ballast precedes mechanism
activation so demand paging budgets against the constrained figure.

Windows shared-memory observations: on WDDM the NVIDIA driver spills over-budget
allocations into shared system memory instead of failing, a performance
cliff no error surfaces and naive free-memory checks miss. Each cell
therefore records per-pass wall times with a max/median timing-outlier flag,
per-pass free device memory, and - when the GPU Process Memory
performance counters are reachable (native Windows, or WSL through
powershell.exe) - the counters' Shared Usage bytes at three points:
before the cold pass, after warmup, and after the measured passes.
Pinned host staging also counts as WDDM Shared Usage. Neither the cold
nor the warm delta identifies its allocation owner, and stable usage can
hide spill that already happened. shared_spill_detected is therefore
null (unassessed), not a pass. Spill acceptance needs separate process,
staging-owner, device-budget and timing evidence; machine-wide usage
alone cannot attribute changes to this process.

Timing receipts: the measured passes run under one
partial-residency-timing collection window, so the report carries the
TRANSFER / DEQUANT / EXPOSED_STALL split, transfer bytes, and the
lease-versus-prefetch transfer attribution for the whole measured
phase.

The JSON report is one cell of the mechanism comparison recorded on
issue #826. Exits nonzero when the bit-identity check or the report
validation fails.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

SCHEMA = "dinkster-residency-probe/4"
MECHANISMS = ("eager", "aimdo")
WEIGHT_FORMATS = ("dense", "q8_0")
REGIMES = ("open", "constrained")
SPILL_SCOPES = ("auto", "process", "machine", "off")
_MIB = 1024 * 1024
# GGUF Q8_0 stores 32 elements per 34-byte block (2-byte fp16 scale +
# 32 int8 quants); mirrored here so working-set math stays torch-free.
_Q8_0_BLOCK_ELEMENTS = 32
_Q8_0_BLOCK_BYTES = 34


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mechanism", required=True, choices=MECHANISMS)
    parser.add_argument(
        "--weights",
        choices=WEIGHT_FORMATS,
        default="dense",
        help="dense float layers, or encoded GGUF Q8_0 block layers",
    )
    parser.add_argument("--queue", action="store_true", help="stream one block ahead")
    parser.add_argument("--regime", choices=REGIMES, default="open")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--layers-per-block", type=int, default=4)
    parser.add_argument("--features", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=826)
    parser.add_argument(
        "--leave-free-mib",
        type=int,
        default=768,
        help="constrained regime: MiB of device memory the ballast leaves free",
    )
    parser.add_argument(
        "--cliff-ratio",
        type=float,
        default=4.0,
        help="max/median pass-time ratio above which a timing outlier is flagged, not spill proof",
    )
    parser.add_argument(
        "--spill-scope",
        choices=SPILL_SCOPES,
        default="auto",
        help=(
            "GPU Process Memory counter instances to sum: process (this pid), "
            "machine (all instances; the only usable scope under WSL, whose "
            "pids the host counters do not know), off, or auto"
        ),
    )
    parser.add_argument("--json", type=str, default=None, help="report path (default: stdout)")
    return parser.parse_args(list(argv))


def working_set_bytes(arguments: argparse.Namespace) -> int:
    elements = arguments.blocks * arguments.layers_per_block * arguments.features**2
    if arguments.weights == "q8_0":
        return elements // _Q8_0_BLOCK_ELEMENTS * _Q8_0_BLOCK_BYTES
    element = {"float16": 2, "bfloat16": 2, "float32": 4}[arguments.dtype]
    return elements * element


# -- Windows sysmem-spill axis ---------------------------------------------


def pass_time_cliff(pass_ms: Sequence[float], ratio: float) -> tuple[float | None, bool]:
    """Observed max/median pass-time ratio and whether it crosses ratio."""
    if not pass_ms:
        return None, False
    median = statistics.median(pass_ms)
    if median <= 0:
        return None, False
    observed = max(pass_ms) / median
    return observed, observed > ratio


def shared_usage_command(scope: str, pid: int) -> str:
    """Print validated byte counts only after every expected counter sample succeeds."""
    instance = f"pid_{pid}_*" if scope == "process" else "*"
    return (
        "$ErrorActionPreference = 'Stop'; "
        "$samples = @((Get-Counter '\\GPU Process Memory(" + instance + ")\\Shared Usage' "
        "-ErrorAction Stop).CounterSamples); "
        "if ($samples.Count -eq 0) { exit 1 }; "
        "foreach ($sample in $samples) { "
        "if ($sample.Status -ne 0 -or $null -eq $sample.CookedValue -or "
        "$sample.CookedValue -lt 0 -or "
        "[double]::IsNaN($sample.CookedValue) -or "
        "[double]::IsInfinity($sample.CookedValue)) { exit 1 } }; "
        "$samples | ForEach-Object { [long]$_.CookedValue }"
    )


def sum_counter_lines(text: str) -> int | None:
    """Sum integral counter lines; None when nothing parseable arrived."""
    total = 0
    seen = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError:
            return None
        if value < 0:
            return None
        total += value
        seen = True
    return total if seen else None


def gpu_shared_usage_bytes(
    scope: str,
    pid: int,
    run: Callable[..., Any] = subprocess.run,
) -> int | None:
    """Total GPU shared (sysmem) usage in bytes, or None when unreachable."""
    command = shared_usage_command(scope, pid)
    for executable in ("powershell.exe", "powershell"):
        try:
            completed = run(
                [executable, "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception:  # noqa: BLE001 - the counters are best-effort evidence
            continue
        if completed.returncode != 0:
            continue
        total = sum_counter_lines(completed.stdout)
        if total is not None:
            return total
    return None


def resolve_spill_scope(scope: str, system: str, release: str) -> str:
    """Map auto to the scope the platform can actually attribute."""
    if scope != "auto":
        return scope
    if system == "Windows":
        return "process"
    if release.endswith("-Microsoft") or release.endswith("microsoft-standard-WSL2"):
        return "machine"
    return "off"


# -- report ------------------------------------------------------------------


def build_report(
    *,
    host: Mapping[str, Any],
    config: Mapping[str, Any],
    pass_ms: Sequence[float],
    pass_free_bytes: Sequence[int | None],
    bit_identical: bool,
    receipt: Mapping[str, Any],
    memory: Mapping[str, Any],
    spill: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "host": dict(host),
        "config": dict(config),
        "results": {
            "pass_ms": list(pass_ms),
            "median_ms": statistics.median(pass_ms) if pass_ms else None,
            "min_ms": min(pass_ms) if pass_ms else None,
            "max_ms": max(pass_ms) if pass_ms else None,
            "pass_free_bytes": list(pass_free_bytes),
            "bit_identical": bit_identical,
            "receipt": dict(receipt),
            "memory": dict(memory),
            "spill": dict(spill),
        },
    }


_HOST_KEYS = (
    "system",
    "release",
    "python",
    "torch",
    "device",
    "device_name",
    "total_bytes",
)
_CONFIG_KEYS = (
    "mechanism",
    "weights",
    "queue",
    "regime",
    "device",
    "dtype",
    "blocks",
    "layers_per_block",
    "features",
    "batch",
    "warmup",
    "passes",
    "seed",
    "leave_free_mib",
    "cliff_ratio",
    "spill_scope",
    "working_set_bytes",
)
_RECEIPT_KEYS = (
    "transfer_ms",
    "exposed_stall_ms",
    "dequant_ms",
    "compute_ms",
    "transfer_bytes",
    "leased_transfers",
    "leased_forwards",
    "prefetched_transfers",
    "prefetch_bytes",
)
_MEMORY_KEYS = (
    "ballast_bytes",
    "free_after_setup_bytes",
    "free_after_passes_bytes",
    "allocated_peak_bytes",
    "reserved_peak_bytes",
)
_SPILL_KEYS = (
    "cliff_ratio_observed",
    "cliff_suspected",
    "scope",
    "shared_before_bytes",
    "shared_warm_bytes",
    "shared_after_bytes",
    "shared_growth_bytes",
    "shared_spill_detected",
)


def validate_residency_report(report: Mapping[str, Any]) -> list[str]:
    """Structural problems that make the report unusable as evidence."""
    problems: list[str] = []
    if report.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}, got {report.get('schema')!r}")
    for section in ("host", "config", "results"):
        if not isinstance(report.get(section), Mapping):
            problems.append(f"missing section {section!r}")
    host = report.get("host")
    if isinstance(host, Mapping):
        for key in _HOST_KEYS:
            if key not in host:
                problems.append(f"host missing {key!r}")
    config = report.get("config")
    if isinstance(config, Mapping):
        for key in _CONFIG_KEYS:
            if key not in config:
                problems.append(f"config missing {key!r}")
        if config.get("mechanism") not in MECHANISMS:
            problems.append(f"config mechanism must be one of {MECHANISMS}")
        if config.get("weights") not in WEIGHT_FORMATS:
            problems.append(f"config weights must be one of {WEIGHT_FORMATS}")
        if config.get("regime") not in REGIMES:
            problems.append(f"config regime must be one of {REGIMES}")
    results = report.get("results")
    if isinstance(results, Mapping):
        pass_ms = results.get("pass_ms")
        if not isinstance(pass_ms, list) or not all(
            isinstance(value, (int, float)) for value in pass_ms
        ):
            problems.append("results pass_ms must be a list of numbers")
        elif isinstance(config, Mapping) and len(pass_ms) != config.get("passes"):
            problems.append("results pass_ms length must equal config passes")
        for key in ("median_ms", "min_ms", "max_ms"):
            if not isinstance(results.get(key), (int, float)):
                problems.append(f"results {key} must be a number")
        pass_free = results.get("pass_free_bytes")
        if not isinstance(pass_free, list) or not all(
            value is None or isinstance(value, int) for value in pass_free
        ):
            problems.append("results pass_free_bytes must be a list of ints or None")
        elif isinstance(config, Mapping) and len(pass_free) != config.get("passes"):
            problems.append("results pass_free_bytes length must equal config passes")
        if not isinstance(results.get("bit_identical"), bool):
            problems.append("results bit_identical must be a bool")
        for section, keys in (
            ("receipt", _RECEIPT_KEYS),
            ("memory", _MEMORY_KEYS),
            ("spill", _SPILL_KEYS),
        ):
            mapping = results.get(section)
            if not isinstance(mapping, Mapping):
                problems.append(f"results missing {section!r}")
                continue
            for key in keys:
                if key not in mapping:
                    problems.append(f"results {section} missing {key!r}")
        spill = results.get("spill")
        if isinstance(spill, Mapping) and spill.get("shared_spill_detected") is not None:
            problems.append("shared_spill_detected must be null: shared usage cannot assess spill")
    return problems


# -- measurement -------------------------------------------------------------


def _probe_model_class(torch_module: Any, weights: str, compute_dtype: Any) -> type:
    nn = torch_module.nn
    from dinkster_inference_torch import INITLESS, GgufEncodedLinear

    def make_layer(features: int) -> Any:
        if weights == "q8_0":
            return GgufEncodedLinear(
                features,
                features,
                ggml_type="Q8_0",
                bias=False,
                compute_dtype=compute_dtype,
            )
        return INITLESS.linear(features, features, bias=False)

    class ProbeBlock(nn.Module):
        def __init__(self, layers: int, features: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList(make_layer(features) for _ in range(layers))

        def forward(self, x: Any) -> Any:
            for layer in self.layers:
                x = layer(x)
            return x

    class ProbeModel(nn.Module):
        def __init__(self, blocks: int, layers: int, features: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(ProbeBlock(layers, features) for _ in range(blocks))

        def forward(self, x: Any, queue: Any = None) -> Any:
            from dinkster_inference_torch.model_prefetch import prefetch_queue_pop

            for block in self.blocks:
                prefetch_queue_pop(queue, block)
                x = block(x)
            prefetch_queue_pop(queue, None)
            return x

    return ProbeModel


def _mechanism_factory(name: str) -> Any:
    if name == "eager":
        from dinkster_inference_torch import ResidentWeights

        return ResidentWeights
    from dinkster_inference_torch import AimdoWeights
    from dinkster_inference_torch.aimdo_activation import ensure_visible_aimdo_devices

    ready = ensure_visible_aimdo_devices()
    if ready is not True:
        raise RuntimeError(f"aimdo device activation returned {ready!r}")
    return AimdoWeights


def _free_bytes(torch_module: Any, device: Any) -> int | None:
    if device.type != "cuda":
        return None
    free, _total = torch_module.cuda.mem_get_info(device)
    return int(free)


def _synchronize(torch_module: Any, device: Any) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)
    elif device.type == "mps":
        torch_module.mps.synchronize()


def _bootstrap_aimdo() -> None:
    """Run the comfy-aimdo native bootstrap, which must precede torch import."""
    if "torch" in sys.modules:
        raise RuntimeError("comfy-aimdo control.init() must run before torch is imported")
    from comfy_aimdo import control

    if not control.init():
        raise RuntimeError("comfy-aimdo native bootstrap init() failed")


def measure(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.mechanism == "aimdo":
        _bootstrap_aimdo()

    import copy

    import torch
    from dinkster_inference_torch import collect_partial_residency_timing, enroll_component
    from dinkster_inference_torch.model_prefetch import close_prefetch_queue, make_prefetch_queue

    device = torch.device(arguments.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[arguments.dtype]
    torch.manual_seed(arguments.seed)
    model_class = _probe_model_class(torch, arguments.weights, dtype)
    model = model_class(arguments.blocks, arguments.layers_per_block, arguments.features)
    # Layers skip parameter init; fill deterministically (raw
    # uninitialized memory can hold NaN, which breaks torch.equal).
    # The 1/sqrt(fan-in) scale keeps activations near unit variance so the
    # stacked matmuls stay inside the fp16 range.
    with torch.no_grad():
        if arguments.weights == "q8_0":
            # One fp16 scale per block sized so decoded weights stay at
            # the same 1/sqrt(fan-in) magnitude as the dense fill.
            scale = float(arguments.features) ** -0.5 / 127.0
            scale_bytes = torch.tensor([scale], dtype=torch.float16).view(torch.uint8)
            generator = torch.Generator().manual_seed(arguments.seed)
            for _name, buffer in model.named_buffers():
                quants = torch.randint(
                    -127,
                    128,
                    (buffer.shape[0], _Q8_0_BLOCK_ELEMENTS),
                    dtype=torch.int8,
                    generator=generator,
                )
                buffer[:, :2] = scale_bytes
                buffer[:, 2:] = quants.view(torch.uint8)
        else:
            for parameter in model.parameters():
                scale = float(parameter.shape[-1]) ** -0.5
                parameter.copy_(torch.randn(parameter.shape, dtype=torch.float32) * scale)
    model = model.to(dtype)
    model.eval()
    x = torch.randn(arguments.batch, arguments.features, dtype=dtype)

    # Fully resident reference, computed and released before any regime
    # shaping so the ballast sizing sees a clean device.
    reference_model = copy.deepcopy(model).to(device)
    x_device = x.to(device)
    with torch.inference_mode():
        reference = reference_model(x_device)
    _synchronize(torch, device)
    del reference_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Chunked so a fragmented allocator cannot fail one huge allocation.
    ballast: list[Any] = []
    ballast_bytes = 0
    if arguments.regime == "constrained":
        if device.type != "cuda":
            raise RuntimeError("the constrained regime requires a CUDA device")
        free, _total = torch.cuda.mem_get_info(device)
        remaining = max(0, int(free) - arguments.leave_free_mib * _MIB)
        ballast_bytes = remaining
        while remaining > 0:
            chunk = min(remaining, 1024 * _MIB)
            ballast.append(torch.empty(chunk, dtype=torch.uint8, device=device))
            remaining -= chunk

    factory = _mechanism_factory(arguments.mechanism)
    mechanism = enroll_component(
        model,
        load_device=device,
        offload_device="cpu",
        mechanism_factory=factory,
    )

    def one_pass() -> Any:
        queue = make_prefetch_queue(model.blocks) if arguments.queue else None
        try:
            with torch.inference_mode():
                return model(x_device, queue=queue)
        finally:
            close_prefetch_queue(queue)

    system = platform.system()
    release = platform.uname().release
    scope = resolve_spill_scope(arguments.spill_scope, system, release)
    pid = os.getpid()
    shared_before = gpu_shared_usage_bytes(scope, pid) if scope != "off" else None

    output = one_pass()
    _synchronize(torch, device)
    bit_identical = bool(torch.equal(output, reference))
    free_after_setup = _free_bytes(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(arguments.warmup):
        one_pass()
    _synchronize(torch, device)
    shared_warm = gpu_shared_usage_bytes(scope, pid) if scope != "off" else None

    pass_ms: list[float] = []
    pass_free_bytes: list[int | None] = []
    with collect_partial_residency_timing() as timing:
        for _ in range(arguments.passes):
            _synchronize(torch, device)
            started = time.perf_counter()
            one_pass()
            _synchronize(torch, device)
            pass_ms.append((time.perf_counter() - started) * 1000.0)
            pass_free_bytes.append(_free_bytes(torch, device))
    timing_report = timing.report()

    shared_after = gpu_shared_usage_bytes(scope, pid) if scope != "off" else None
    shared_growth = (
        shared_after - shared_warm if shared_after is not None and shared_warm is not None else None
    )
    cliff_observed, cliff_suspected = pass_time_cliff(pass_ms, arguments.cliff_ratio)

    free_after_passes = _free_bytes(torch, device)
    allocated_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    reserved_peak = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    mechanism.unload()
    del ballast

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    total_bytes = int(torch.cuda.mem_get_info(device)[1]) if device.type == "cuda" else None
    host = {
        "system": system,
        "release": release,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": device_name,
        "total_bytes": total_bytes,
    }
    config = {
        "mechanism": arguments.mechanism,
        "weights": arguments.weights,
        "queue": arguments.queue,
        "regime": arguments.regime,
        "device": arguments.device,
        "dtype": arguments.dtype,
        "blocks": arguments.blocks,
        "layers_per_block": arguments.layers_per_block,
        "features": arguments.features,
        "batch": arguments.batch,
        "warmup": arguments.warmup,
        "passes": arguments.passes,
        "seed": arguments.seed,
        "leave_free_mib": arguments.leave_free_mib,
        "cliff_ratio": arguments.cliff_ratio,
        "spill_scope": scope,
        "working_set_bytes": working_set_bytes(arguments),
    }
    receipt = {
        "transfer_ms": timing_report.transfer_ms,
        "exposed_stall_ms": timing_report.exposed_stall_ms,
        "dequant_ms": timing_report.dequant_ms,
        "compute_ms": timing_report.compute_ms,
        "transfer_bytes": timing_report.transfer_bytes,
        "leased_transfers": timing_report.leased_transfers,
        "leased_forwards": timing_report.leased_forwards,
        "prefetched_transfers": timing_report.prefetched_transfers,
        "prefetch_bytes": timing_report.prefetch_bytes,
    }
    memory = {
        "ballast_bytes": ballast_bytes,
        "free_after_setup_bytes": free_after_setup,
        "free_after_passes_bytes": free_after_passes,
        "allocated_peak_bytes": allocated_peak,
        "reserved_peak_bytes": reserved_peak,
    }
    spill = {
        "cliff_ratio_observed": cliff_observed,
        "cliff_suspected": cliff_suspected,
        "scope": scope,
        "shared_before_bytes": shared_before,
        "shared_warm_bytes": shared_warm,
        "shared_after_bytes": shared_after,
        "shared_growth_bytes": shared_growth,
        "shared_spill_detected": None,
    }
    return build_report(
        host=host,
        config=config,
        pass_ms=pass_ms,
        pass_free_bytes=pass_free_bytes,
        bit_identical=bit_identical,
        receipt=receipt,
        memory=memory,
        spill=spill,
    )


def main(argv: Sequence[str]) -> int:
    arguments = parse_arguments(argv)
    report = measure(arguments)
    problems = validate_residency_report(report)
    rendered = json.dumps(report, indent=2)
    if arguments.json is None:
        print(rendered)
    else:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        print(f"wrote {arguments.json}")
    if problems:
        for problem in problems:
            print(f"invalid report: {problem}", file=sys.stderr)
        return 1
    if not report["results"]["bit_identical"]:
        print("bit-identity check failed against the resident reference", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
