"""Report what torch and Dinkster's torch layer can do on Apple Silicon.

Run under the torch test env on a Mac:

    .venv-torch/bin/python scripts/mps_smoke.py

Prints an observed capability report in the DeviceCapabilities
vocabulary (dinkster_inference.devices): which dtypes the MPS device can
store versus compute in, plus the memory introspection and attention
route evidence dinkster-inference-torch derives on this machine. Exits
nonzero when torch cannot see an MPS device or a baseline float probe
(fp32/fp16/bf16 storage or matmul, fp16 SDPA) fails; unsupported quant
dtypes are reported, not fatal - refusing them cleanly is the expected
state.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

import torch


def _installed(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _probe(action: Callable[[], object]) -> tuple[bool, str]:
    """Run one capability probe, reporting failure as data, not a crash."""
    try:
        action()
        return True, "ok"
    except Exception as error:  # the failure text is the probe's result
        message = str(error).split("\n")[0].strip() or type(error).__name__
        return False, message[:90]


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


def main() -> int:
    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"python:   {platform.python_version()}")
    print(f"torch:    {torch.__version__}")
    print(f"dinkster-kitchen: {_installed('dinkster-kitchen')}")
    print(f"dinkster-aimdo: {_installed('dinkster-aimdo')}")
    print(f"psutil:        {_installed('psutil')}")

    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    print(f"mps: built={built} available={available}")
    if not available:
        print("error: torch cannot see an MPS device on this machine", file=sys.stderr)
        return 1
    device = torch.device("mps")

    from dinkster_inference_torch.attention import discover_attention_route_token
    from dinkster_inference_torch.memory import get_free_memory, get_total_memory, soft_empty_cache

    token = discover_attention_route_token()
    print(f"attention route: device_kind={token.device_kind} device_sm={token.device_sm}")

    gib = 1024**3
    total = get_total_memory(device)
    free = get_free_memory(device)
    print(f"memory: total={total / gib:.1f} GiB free={free.free_total / gib:.1f} GiB (unified)")
    soft_empty_cache(device)
    print("soft_empty_cache: ok")

    float_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    quant_dtypes = [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2]
    print("capability table (storage = host bytes survive a device roundtrip;")
    print("compute = matmul for float dtypes, on-device cast for quant dtypes):")
    baseline_ok = True
    for dtype in float_dtypes + quant_dtypes:
        stored, store_detail = _probe(lambda d=dtype: _storage_roundtrip(d, device))
        if dtype in float_dtypes:
            computed, compute_detail = _probe(lambda d=dtype: _matmul(d, device))
            baseline_ok = baseline_ok and stored and computed
        else:
            computed, compute_detail = _probe(lambda d=dtype: _cast_on_device(d, device))
        name = str(dtype).removeprefix("torch.")
        store_note = "" if stored else f" ({store_detail})"
        compute_note = "" if computed else f" ({compute_detail})"
        print(
            f"  {name:<14} storage={'yes' if stored else 'NO'}{store_note}"
            f" compute={'yes' if computed else 'NO'}{compute_note}"
        )
    sdpa_ok, sdpa_detail = _probe(lambda: _sdpa(device))
    print(f"  sdpa (fp16)    {'ok' if sdpa_ok else 'NO (' + sdpa_detail + ')'}")
    baseline_ok = baseline_ok and sdpa_ok

    if not baseline_ok:
        print("error: a baseline float probe failed on MPS", file=sys.stderr)
        return 1
    print("baseline float compute on MPS: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
