"""Report what torch and Dinkster's torch layer can do on an Intel XPU GPU.

Run under the isolated XPU env (scripts/setup_env_xpu.ps1 or .sh):

    .venv-xpu/bin/python scripts/xpu_smoke.py --json xpu-report.json
    .venv-xpu\\Scripts\\python.exe scripts\\xpu_smoke.py --json xpu-report.json

Intel GPUs are their own torch backend: torch.xpu and torch.device("xpu"),
with oneAPI Level Zero underneath; CUDA namespace APIs do not apply.
Prints a human-readable capability report and optionally writes the JSON
evidence report that dinkster_workers.backend_env.validate_smoke_report
checks: host identity, driver and Level Zero identity, torch/XPU runtime
versions, device name and architecture, fp32/fp16/bf16 storage and
matmul, fp16 SDPA, memory APIs, cache control, attention route evidence,
and GGUF/INT8/FP8 dequantization parity against the CPU baseline. Exits
nonzero when torch has no XPU support, no device is visible, a required
compute, memory, routing, policy, or dequantization probe fails, or the
report is incomplete; unsupported quant storage and cast probes are
reported, not fatal - refusing them cleanly is the expected state.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from dinkster_workers.backend_env import (
    SMOKE_REPORT_VERSION,
    format_smoke_receipt,
    smoke_gate_problems,
    validate_smoke_report,
)


def _probe(action: Callable[[], object]) -> tuple[bool, str]:
    """Run one capability probe, reporting failure as data, not a crash."""
    try:
        action()
        return True, "ok"
    except Exception as error:  # the failure text is the probe's result
        # The compact message keeps only the first line; preserve the full
        # traceback in the log for nested or multi-line errors.
        traceback.print_exc()
        message = str(error).split("\n")[0].strip() or type(error).__name__
        return False, message[:120]


def _storage_roundtrip(dtype: torch.dtype, device: torch.device) -> None:
    """Weights may rest in a dtype when host bytes survive a device roundtrip."""
    host = torch.arange(16, dtype=torch.float32).to(dtype)
    back = host.to(device).cpu()
    if back.view(torch.uint8).ne(host.view(torch.uint8)).any():
        raise RuntimeError("roundtrip changed bytes")


def _matmul(dtype: torch.dtype, device: torch.device) -> None:
    square = torch.ones((8, 8), dtype=dtype, device=device)
    torch.matmul(square, square).sum().item()


def _cast_on_device(dtype: torch.dtype, device: torch.device) -> None:
    torch.arange(16, dtype=torch.float32).to(dtype).to(device).to(torch.float16).cpu()


def _sdpa(device: torch.device) -> None:
    q = torch.ones((1, 2, 4, 8), dtype=torch.float16, device=device)
    torch.nn.functional.scaled_dot_product_attention(q, q, q).sum().item()


def _dequant_parity(cpu_value: torch.Tensor, device_value: torch.Tensor) -> dict[str, object]:
    """Compare a device dequantization against its CPU baseline."""
    if not bool(torch.isfinite(cpu_value).all()):
        raise RuntimeError("CPU baseline dequantization produced non-finite values")
    moved = device_value.cpu()
    if not bool(torch.isfinite(moved).all()):
        raise RuntimeError("device dequantization produced non-finite values")
    exact = bool(torch.equal(cpu_value, moved))
    diff = 0.0 if exact else float((cpu_value - moved).abs().max())
    if diff > 1e-6:
        raise RuntimeError(f"device dequantization diverged from the CPU baseline by {diff}")
    return {"bit_exact": exact, "max_abs_diff": diff}


def _kitchen_evidence(device: torch.device) -> dict[str, object] | None:
    """dinkster-kitchen per-backend authentication facts for this device.

    None when dinkster-kitchen is not installed (the environment recipe
    pins it, so absence is itself a finding). Records
    which kitchen backends registered, which backend the registry
    selects for INT8 linear and ConvRot INT8 dequantization on this
    device, and whether executing them returns finite output; failures
    are recorded as data, never raised.
    """
    try:
        import dinkster_kitchen
    except ImportError:
        return None
    evidence: dict[str, object] = {}
    try:
        _kitchen_attempts(dinkster_kitchen, device, evidence)
    except Exception as error:  # noqa: BLE001 - the failure text is the evidence
        evidence["error"] = str(error).split("\n")[0].strip()[:200]
    return evidence


def _kitchen_attempts(
    dinkster_kitchen: Any, device: torch.device, evidence: dict[str, object]
) -> None:
    generator = torch.Generator().manual_seed(591)
    x = torch.randn(8, 256, generator=generator).to(device=device, dtype=torch.float16)
    weight = torch.randint(-127, 128, (16, 256), dtype=torch.int8, generator=generator).to(device)
    scalar_scale = torch.tensor(0.02, dtype=torch.float32, device=device)
    row_scale = torch.full((16, 1), 0.02, dtype=torch.float32, device=device)

    def run_linear() -> None:
        out = dinkster_kitchen.int8_linear(x, weight, scalar_scale, None, out_dtype=torch.float16)
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("int8_linear returned non-finite values")

    def run_convrot() -> None:
        value = torch.ops.dinkster_kitchen.dequantize_int8_convrot_weight(weight, row_scale, 256)
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("ConvRot dequantization returned non-finite values")

    try:
        kitchen_version = importlib.metadata.version("dinkster-kitchen")
    except importlib.metadata.PackageNotFoundError:
        kitchen_version = str(getattr(dinkster_kitchen, "__version__", "unknown"))
    evidence["version"] = kitchen_version
    evidence["backends"] = {
        str(name): bool(status.get("available", False))
        for name, status in dinkster_kitchen.registry.list_backends().items()
    }
    attempts: tuple[tuple[str, dict[str, object], Callable[[], None]], ...] = (
        (
            "int8_linear",
            {
                "x": x,
                "weight": weight,
                "weight_scale": scalar_scale,
                "bias": None,
                "out_dtype": torch.float16,
                "convrot": False,
                "convrot_groupsize": 256,
                "input_act": None,
            },
            run_linear,
        ),
        (
            "dequantize_int8_convrot_weight",
            {"q": weight, "scale": row_scale, "group_size": 256},
            run_convrot,
        ),
    )
    for name, kwargs, run in attempts:
        entry: dict[str, object] = {}
        try:
            entry["backend"] = str(dinkster_kitchen.registry.get_capable_backend(name, kwargs))
        except Exception as error:  # the refusal text is the evidence
            entry["backend"] = None
            entry["backend_error"] = str(error).split("\n")[0].strip()[:120]
        executed, detail = _probe(run)
        entry["executed"] = executed
        entry["detail"] = detail
        evidence[name] = entry


def _driver_identity(properties: object) -> str:
    """Best available display-driver and Level Zero identity."""
    parts: list[str] = []
    for attribute in ("driver_version", "platform_name"):
        value = str(getattr(properties, attribute, "") or "").strip()
        if value:
            parts.append(f"{attribute}={value}")
    if sys.platform == "win32":
        command = (
            "Get-CimInstance Win32_VideoController | "
            "ForEach-Object { $_.Name + ' driver ' + $_.DriverVersion }"
        )
        try:
            output = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            parts.extend(lines)
        except Exception:
            pass
    if parts:
        return "; ".join(parts)
    return "unknown (no torch driver properties and no host driver query)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    arguments = parser.parse_args()

    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"python:   {platform.python_version()}")
    print(f"torch:    {torch.__version__}")

    build_xpu = str(getattr(torch.version, "xpu", "") or "").strip()
    print(f"xpu build: {build_xpu or 'not reported'}")
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        print("error: torch cannot see an XPU device on this machine", file=sys.stderr)
        return 1
    device = torch.device("xpu:0")

    devices: list[dict[str, object]] = []
    first_properties: object = None
    for index in range(torch.xpu.device_count()):
        properties = torch.xpu.get_device_properties(index)
        if first_properties is None:
            first_properties = properties
        # No fallback: an XPU device with no architecture identity is an
        # identity gap, and the report validator must fail on it.
        architecture = ""
        for attribute in ("architecture", "device_id", "platform_name"):
            value = getattr(properties, attribute, None)
            if value:
                architecture = f"{attribute}={value}"
                break
        entry: dict[str, object] = {
            "index": index,
            "name": properties.name,
            "architecture": architecture,
            "total_memory": int(properties.total_memory),
        }
        devices.append(entry)
        print(f"device {index}: {properties.name} ({architecture}, {properties.total_memory} B)")

    driver = _driver_identity(first_properties)
    print(f"driver:   {driver}")

    # Runtime identity comes from the device the runtime actually reports,
    # not from the torch build string (which only names the compiled-against
    # oneAPI toolchain).
    backend_runtime = ""
    for attribute in ("version", "driver_version"):
        value = str(getattr(first_properties, attribute, "") or "").strip()
        if value:
            backend_runtime = f"xpu {attribute}={value}"
            break
    print(f"runtime:  {backend_runtime or 'not reported'}")

    probes: dict[str, dict[str, object]] = {}

    def record(name: str, action: Callable[[], object]) -> bool:
        ok, detail = _probe(action)
        probes[name] = {"ok": ok, "detail": detail}
        print(f"  {name:<18} {'ok' if ok else 'NO (' + detail + ')'}")
        return ok

    print("capability probes:")
    baseline_ok = True
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        suffix = str(dtype).removeprefix("torch.")
        baseline_ok &= record(f"storage_{suffix}", lambda d=dtype: _storage_roundtrip(d, device))
        baseline_ok &= record(f"matmul_{suffix}", lambda d=dtype: _matmul(d, device))
    for dtype in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2):
        suffix = str(dtype).removeprefix("torch.")
        record(f"storage_{suffix}", lambda d=dtype: _storage_roundtrip(d, device))
        record(f"cast_{suffix}", lambda d=dtype: _cast_on_device(d, device))
    baseline_ok &= record("sdpa_float16", lambda: _sdpa(device))

    memory: dict[str, object] = {}

    def read_memory() -> None:
        mem_get_info = getattr(torch.xpu, "mem_get_info", None)
        if mem_get_info is not None:
            free, total = mem_get_info(device)
            memory["mem_get_info_free"] = int(free)
            memory["mem_get_info_total"] = int(total)
        memory["allocator_allocated"] = int(torch.xpu.memory_allocated(device))
        memory["allocator_reserved"] = int(torch.xpu.memory_reserved(device))
        if mem_get_info is None:
            raise RuntimeError("torch.xpu.mem_get_info is not exposed by this torch")

    record("mem_get_info", read_memory)
    record("empty_cache", torch.xpu.empty_cache)
    record("synchronize", lambda: torch.xpu.synchronize(device))
    if "mem_get_info_total" in memory:
        gib = 1024**3
        print(
            f"memory: total={memory['mem_get_info_total'] / gib:.1f} GiB"
            f" free={memory['mem_get_info_free'] / gib:.1f} GiB"
        )

    attention_route: dict[str, object] = {}

    def read_route() -> None:
        from dinkster_inference_torch.attention import discover_attention_route_token

        token = discover_attention_route_token()
        attention_route["device_kind"] = token.device_kind
        attention_route["device_sm"] = token.device_sm
        attention_route["providers"] = [list(pair) for pair in token.provider_versions]

    record("attention_route", read_route)
    if attention_route:
        print(f"attention route: {attention_route}")

    dtype_policy: dict[str, object] = {}

    def read_dtype_policy() -> None:
        from dinkster_inference_torch import (
            bf16_support,
            fp16_support,
            lora_compute_dtype,
            supports_fp8_matmul,
        )

        fp16 = fp16_support(device)
        bf16 = bf16_support(device)
        dtype_policy["fp16"] = {"storage": fp16.storage, "compute": fp16.compute}
        dtype_policy["bf16"] = {"storage": bf16.storage, "compute": bf16.compute}
        dtype_policy["fp8_native_matmul"] = supports_fp8_matmul(device)
        dtype_policy["lora_patch_dtype"] = str(lora_compute_dtype(device)).removeprefix("torch.")

    record("dtype_policy", read_dtype_policy)
    if dtype_policy:
        print(f"dtype policy: {dtype_policy}")

    quant_dequant: dict[str, object] = {}

    def read_quant_dequant() -> None:
        from dinkster_inference_torch.gguf_linear import (
            GGUF_BLOCK_DECODERS,
            GGUF_BLOCK_SHAPES,
            synthetic_gguf_blocks,
        )
        from dinkster_inference_torch.quant import Fp8ScaledWeight, Int8PackedWeight

        gguf: dict[str, object] = {}
        for layout in sorted(GGUF_BLOCK_DECODERS):
            decode = GGUF_BLOCK_DECODERS[layout]
            elements, _ = GGUF_BLOCK_SHAPES[layout]
            blocks = synthetic_gguf_blocks(layout, 24, seed=591)
            shape = (24, elements)
            gguf[layout] = _dequant_parity(decode(blocks, shape), decode(blocks.to(device), shape))
        quant_dequant["gguf"] = gguf

        generator = torch.Generator().manual_seed(591)
        qdata = torch.randint(-127, 128, (32, 48), dtype=torch.int8, generator=generator)
        scale = torch.tensor(0.0123, dtype=torch.float32)
        int8_cpu = Int8PackedWeight(
            qdata=qdata,
            scale=scale,
            orig_dtype=torch.float32,
            convrot=False,
            convrot_groupsize=0,
        )
        int8_device = Int8PackedWeight(
            qdata=qdata.to(device),
            scale=scale.to(device),
            orig_dtype=torch.float32,
            convrot=False,
            convrot_groupsize=0,
        )
        quant_dequant["int8"] = _dequant_parity(int8_cpu.dequantize(), int8_device.dequantize())

        fp8_data = torch.randn(32, 48, generator=generator).clamp(-448, 448)
        fp8_qdata = fp8_data.to(torch.float8_e4m3fn)
        fp8_scale = torch.tensor(0.5, dtype=torch.float32)
        fp8_cpu = Fp8ScaledWeight(qdata=fp8_qdata, scale=fp8_scale, orig_dtype=torch.float32)
        fp8_device = Fp8ScaledWeight(
            qdata=fp8_qdata.to(device), scale=fp8_scale.to(device), orig_dtype=torch.float32
        )
        quant_dequant["fp8_e4m3fn"] = _dequant_parity(fp8_cpu.dequantize(), fp8_device.dequantize())

        quant_dequant["kitchen"] = _kitchen_evidence(device)

    record("quant_dequant", read_quant_dequant)
    if quant_dequant:
        print(f"quant dequant: {quant_dequant}")

    report: dict[str, object] = {
        "report_version": SMOKE_REPORT_VERSION,
        "accelerator": "xpu",
        "host": {
            "platform": platform.platform(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "driver": driver,
        "torch": {
            "version": str(torch.__version__),
            "backend_runtime": backend_runtime,
            "build_xpu": build_xpu,
        },
        "devices": devices,
        "memory": memory,
        "attention_route": attention_route,
        "dtype_policy": dtype_policy,
        "quant_dequant": quant_dequant,
        "probes": probes,
        "baseline_ok": baseline_ok,
    }

    report_problems = validate_smoke_report(report, accelerator="xpu")
    gate_problems = smoke_gate_problems(report, accelerator="xpu")
    for problem in report_problems:
        print(f"error: incomplete report: {problem}", file=sys.stderr)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report written: {arguments.json}")

    print(format_smoke_receipt(report, accelerator="xpu"))
    if gate_problems:
        for problem in gate_problems:
            if problem not in report_problems:
                print(f"error: smoke gate: {problem}", file=sys.stderr)
        return 1
    print("baseline float compute on XPU: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
