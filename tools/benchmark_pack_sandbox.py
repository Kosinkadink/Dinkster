"""Compare native and public-launcher CUDA model loading and compute.

Run under tools/gpu-claim on shared hosts. The selected Python must have
CUDA torch and safetensors; the parent must be able to import dinkster_workers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def child(model: Path, output: Path) -> None:
    import torch
    from safetensors import safe_open

    started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    initialization = time.perf_counter() - started
    started = time.perf_counter()
    with safe_open(model, framework="pt", device="cpu") as source:
        tensors = {key: source.get_tensor(key).to("cuda") for key in source.keys()}
    torch.cuda.synchronize()
    loading = time.perf_counter() - started
    left = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    for _ in range(20):
        torch.mm(left, right)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(300):
        torch.mm(left, right)
    torch.cuda.synchronize()
    computing = time.perf_counter() - started
    output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "cuda_init_s": initialization,
                "load_to_gpu_s": loading,
                "tensors": len(tensors),
                "matmul_per_s": 300 / computing,
                "vram_bytes": torch.cuda.memory_allocated(),
            }
        )
    )


async def benchmark(python: str, model: Path) -> None:
    from dinkster_workers import BubblewrapLauncher, LaunchSpec, SandboxPolicy

    launcher = BubblewrapLauncher(SandboxPolicy(gpu=True, ro_binds=(str(model.parent),)))
    script = Path(__file__).resolve()
    # Two observations per cell; reverse the order to expose warm-up drift.
    for sandboxed in (False, True, True, False):
        for cold in (True, False):
            if cold:
                descriptor = os.open(model, os.O_RDONLY)
                try:
                    os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(descriptor)
            with tempfile.TemporaryDirectory(prefix="dinkster-sandbox-benchmark-") as directory:
                endpoint = Path(directory)
                result_path = endpoint / "result.json"
                command = (python, str(script), str(model), "--child-output", str(result_path))
                started = time.perf_counter()
                if sandboxed:
                    process = await launcher.launch(
                        LaunchSpec(
                            command=command,
                            env={},
                            endpoint_dir=endpoint,
                            pack_root=script.parent,
                            python=python,
                            use_shm=True,
                        )
                    )
                    code = await process.wait()
                    if code:
                        raise RuntimeError(f"sandboxed benchmark exited {code}")
                else:
                    subprocess.run(command, check=True)
                result = json.loads(result_path.read_text())
                result.update(
                    {"sandboxed": sandboxed, "cold": cold, "wall_s": time.perf_counter() - started}
                )
                print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--python", default=os.sys.executable)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_output is not None:
        child(args.model, args.child_output)
    else:
        asyncio.run(benchmark(args.python, args.model.resolve()))


if __name__ == "__main__":
    main()
