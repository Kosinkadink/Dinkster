"""Measure native checkpoint-load peak RSS in a fresh child process.

The checkpoint path is exposed to the child through a temporary symlink,
never a copy. The child imports the production native arm and torch, reports
its baseline, then runs the arm's ``_load_runtime`` seam and constructs the
production ``NativeRuntimeHandle``. It deliberately stops before any stage
placement, so the measured peak includes safetensors parsing, CPU module
assembly, and per-unit enrollment without requiring an available GPU.

Linux VmHWM is the child process's lifetime peak RSS. The reported factor
therefore includes the child's native-arm and torch import baseline as well
as checkpoint materialization.

Usage:

    .venv-gpu/bin/python tools/measure_native_checkpoint_ram.py \
        /home/kosin/ComfyUI/models/checkpoints/model.safetensors
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dinkster_inference import (
    FLOAT32,
    build_runtime_identity,
    default_diffusion_dtype,
    load_safetensors_header,
    plan_native,
    probe_native,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_CHILD_READY = "native-child-ready"
_CHILD_LOADED = "native-child-loaded"


def dinkster_pythonpath() -> str:
    return os.pathsep.join(sorted(str(path) for path in (REPO_ROOT / "packages").glob("*/src")))


def vm_hwm_bytes(pid: int) -> int:
    status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    raise RuntimeError(f"VmHWM not found in /proc/{pid}/status")


def native_identity(checkpoint: Path) -> str:
    source = load_safetensors_header(checkpoint)
    capability = probe_native(source)
    if not capability.native:
        raise RuntimeError("native probe refused: " + "; ".join(capability.reasons))
    plan = plan_native(source)
    return build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=default_diffusion_dtype(plan.family.id),
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
        registry_token=None,
    )


def _expect_marker(process: subprocess.Popen[str], expected: str) -> None:
    assert process.stdout is not None
    marker = process.stdout.readline().rstrip("\n")
    if marker == expected:
        return
    _, stderr = process.communicate()
    detail = stderr.strip() or f"unexpected child marker {marker!r}"
    raise RuntimeError(f"native measurement child failed: {detail}")


def _continue_child(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.write("\n")
    process.stdin.flush()


def measure(checkpoint: Path, python: Path) -> None:
    checkpoint = checkpoint.resolve(strict=True)
    file_size = checkpoint.stat().st_size
    expected_identity = native_identity(checkpoint)
    with tempfile.TemporaryDirectory(prefix="dinkster-native-checkpoint-measure-") as root_raw:
        link = Path(root_raw) / checkpoint.name
        link.symlink_to(checkpoint)
        env = os.environ.copy()
        env["PYTHONPATH"] = dinkster_pythonpath()
        process = subprocess.Popen(
            [
                str(python),
                str(Path(__file__).resolve()),
                "--native-child",
                str(link),
                expected_identity,
            ],
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _expect_marker(process, _CHILD_READY)
            baseline = vm_hwm_bytes(process.pid)
            _continue_child(process)
            _expect_marker(process, _CHILD_LOADED)
            peak = vm_hwm_bytes(process.pid)
            _continue_child(process)
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"native measurement child failed: {stderr.strip()}")
        except BaseException:
            process.kill()
            process.communicate()
            raise

    print(f"checkpoint={checkpoint}")
    print(f"file_bytes={file_size}")
    print(f"child_baseline_hwm_bytes={baseline}")
    print(f"child_peak_hwm_bytes={peak}")
    print(f"peak_to_file_factor={peak / file_size:.6f}")


def _run_child(checkpoint: Path, expected_identity: str) -> None:
    from dinkster_compat_comfy.native_arm import (
        _build_runtime_handle,
        _load_runtime,
        _torch,
    )

    torch = _torch()
    print(_CHILD_READY, flush=True)
    input()
    runtime = _load_runtime(checkpoint, expected_identity)
    handle = _build_runtime_handle(runtime, torch)
    print(_CHILD_LOADED, flush=True)
    input()
    del handle
    del runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="torch-enabled child interpreter (default: current interpreter)",
    )
    parser.add_argument("--native-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("identity", nargs="?", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.native_child:
        if args.identity is None:
            parser.error("native child requires an expected identity")
        _run_child(args.checkpoint, args.identity)
        return
    if args.identity is not None:
        parser.error("unexpected identity argument")
    if not args.python.is_file():
        parser.error(f"child interpreter does not exist: {args.python}")
    measure(args.checkpoint, args.python.absolute())


if __name__ == "__main__":
    main()
