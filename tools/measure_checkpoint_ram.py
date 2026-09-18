"""Measure compat checkpoint-load peak RSS in a real isolated worker.

The checkpoint path is exposed to the child through a temporary indexed
asset root (a symlink, never a copy), then ``dinkster.load_checkpoint`` runs
through the production compat manifest and boundary. Linux VmHWM is the
child process's lifetime peak RSS, so the reported factor includes ComfyUI
and torch import overhead as well as checkpoint materialization.

Usage:

    .venv/bin/python tools/measure_checkpoint_ram.py \
        --comfyui-root /home/kosin/ComfyUI \
        /home/kosin/ComfyUI/models/checkpoints/model.safetensors
"""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

from dinkster_assets import LocalAssetLibrary, register_asset_type
from dinkster_engine import Invocation
from dinkster_memory import GovernorReservationService, MemoryGovernor
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker, LaunchSpec
from dinkster_workers.launch import SubprocessLauncher

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPAT_MANIFEST = REPO_ROOT / "packages" / "dinkster-compat-comfy" / "dinkster-pack.toml"


class RecordingLauncher:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        self.process = await SubprocessLauncher().launch(spec)
        return self.process


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


async def measure(checkpoint: Path, comfyui_root: Path, python: Path) -> None:
    checkpoint = checkpoint.resolve(strict=True)
    file_size = checkpoint.stat().st_size
    with tempfile.TemporaryDirectory(prefix="dinkster-checkpoint-measure-") as root_raw:
        asset_root = Path(root_raw)
        link = asset_root / checkpoint.name
        link.symlink_to(checkpoint)
        library = LocalAssetLibrary(asset_root)
        library.scan()
        ref = library.ref(f"models/{checkpoint.name}")

        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry)
        launcher = RecordingLauncher()
        governor = MemoryGovernor()
        worker = IsolatedWorker(
            COMPAT_MANIFEST,
            registry,
            python=str(python),
            extra_env={
                "DINKSTER_COMFYUI_ROOT": str(comfyui_root),
                "DINKSTER_COMFY_NODES": "",
                "DINKSTER_ASSET_ROOT": str(asset_root),
                "PYTHONPATH": dinkster_pythonpath(),
            },
            reservations=GovernorReservationService(governor),
            launcher=launcher,
            start_timeout=180.0,
        )
        await worker.start()
        try:
            process = launcher.process
            if process is None or process.pid is None:
                raise RuntimeError("isolated worker launcher did not expose a pid")
            baseline = vm_hwm_bytes(process.pid)
            result = await worker.invoke(
                Invocation(
                    invocation_id="checkpoint-ram-measurement",
                    node_id="load",
                    node_type="dinkster.load_checkpoint",
                    inputs={"checkpoint": registry.wrap("dinkster.asset", ref)},
                    effective_schema=worker.schemas["dinkster.load_checkpoint"],
                )
            )
            if result.error is not None:
                raise RuntimeError(result.error.message)
            peak = vm_hwm_bytes(process.pid)
        finally:
            await worker.close()

    print(f"checkpoint={checkpoint}")
    print(f"file_bytes={file_size}")
    print(f"child_baseline_hwm_bytes={baseline}")
    print(f"child_peak_hwm_bytes={peak}")
    print(f"peak_to_file_factor={peak / file_size:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--comfyui-root",
        type=Path,
        default=os.environ.get("DINKSTER_COMFYUI_ROOT"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=os.environ.get("DINKSTER_COMFYUI_PYTHON"),
        help="ComfyUI interpreter (default: COMFYUI_ROOT/venv/bin/python)",
    )
    args = parser.parse_args()
    if args.comfyui_root is None:
        parser.error("--comfyui-root or DINKSTER_COMFYUI_ROOT is required")
    comfyui_root = args.comfyui_root.resolve(strict=True)
    python = args.python or comfyui_root / "venv" / "bin" / "python"
    asyncio.run(measure(args.checkpoint, comfyui_root, python))


if __name__ == "__main__":
    main()
