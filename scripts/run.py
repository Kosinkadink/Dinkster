"""Prepare and launch the browser editor from a source checkout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3639)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")

    windows = platform.system() == "Windows"
    for name in ("uv", "git", "node", "npm"):
        if not shutil.which(name):
            raise RuntimeError(f"{name} is required; install it and restart your terminal")
    if int(output("node", "--version").lstrip("v").split(".")[0]) < 22:
        raise RuntimeError("Node.js 22 or newer is required: https://nodejs.org/")

    cuda_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_disabled = cuda_mask is not None and cuda_mask.strip() in ("", "-1")
    has_nvidia = (
        not cuda_disabled and platform.system() != "Darwin" and bool(shutil.which("nvidia-smi"))
    )
    if cuda_disabled:
        print("Using CPU Torch: CUDA_VISIBLE_DEVICES disables CUDA devices.", flush=True)
    if has_nvidia:
        try:
            output("nvidia-smi", "-L")
        except subprocess.CalledProcessError:
            print("Using CPU Torch: NVIDIA GPU driver is unavailable.", flush=True)
            has_nvidia = False

    npm = shutil.which("npm")
    assert npm is not None
    pnpm = (npm, "exec", "--yes", "--package=pnpm@10.31.0", "--", "pnpm")
    pin = json.loads((ROOT / "scripts/release_sources.json").read_text())
    frontend = ROOT / ".run/frontend"
    frontend.parent.mkdir(exist_ok=True)
    if not frontend.exists():
        run("git", "clone", "https://github.com/Kosinkadink/Dinkster-Frontend.git", str(frontend))
    if output("git", "status", "--porcelain", cwd=frontend):
        raise RuntimeError(".run/frontend has local changes; preserve them before rerunning")
    if output("git", "rev-parse", "HEAD", cwd=frontend) != pin["commit"]:
        run("git", "fetch", "origin", pin["commit"], cwd=frontend)
        run("git", "checkout", "--detach", pin["commit"], cwd=frontend)
    stamp = ROOT / ".run/frontend-build"
    if not (frontend / "packages/app/dist/index.html").is_file() or (
        not stamp.is_file() or stamp.read_text() != pin["commit"]
    ):
        run(*pnpm, "install", "--frozen-lockfile", cwd=frontend)
        run(*pnpm, "--filter", "@dinkster/app", "build", cwd=frontend)
        stamp.write_text(pin["commit"])

    os.environ["UV_PYTHON_PREFERENCE"] = "only-managed"
    if windows:
        run(
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts/setup_envs.ps1"),
        )
        relative_python = "Scripts/python.exe"
    else:
        run("bash", str(ROOT / "scripts/setup_envs.sh"))
        relative_python = "bin/python"
    execution = ROOT / ".venv-torch" / relative_python
    if has_nvidia:
        gpu_python = ROOT / ".venv-gpu" / relative_python
        if (
            gpu_python.is_file()
            and output(str(gpu_python), "-c", "import torch; print(torch.cuda.is_available())")
            == "True"
        ):
            execution = gpu_python
        else:
            print(
                "Using CPU Torch: NVIDIA GPU driver cannot run pinned CUDA Torch; "
                "update the driver to enable GPU execution.",
                flush=True,
            )
    os.environ["DINKSTER_EXECUTION_PYTHON"] = str(execution)
    python = str(ROOT / ".venv" / relative_python)
    run(python, "-m", "dinkster.cli", "setup")
    launch = [
        python,
        "-m",
        "dinkster.cli",
        "--port",
        str(args.port),
        "--frontend-root",
        str(frontend / "packages/app/dist"),
    ]
    if args.no_browser:
        launch.append("--no-browser")
    run(*launch)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
