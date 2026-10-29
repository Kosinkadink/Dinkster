"""Generate S4-B2.3b CFG facts by executing pinned ComfyUI source.

Usage (the interpreter must provide torch):

    .venv-gpu/bin/python tools/gen_scheduled_sampling_goldens.py \
        --comfy-git ../ComfyUI
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/scheduled_sampling_goldens.json"


def _source(comfy_git: Path) -> str:
    resolved = subprocess.run(
        ["git", "rev-parse", REFERENCE_COMMIT],
        cwd=comfy_git,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if resolved != REFERENCE_COMMIT:
        raise SystemExit(f"reference resolved to {resolved}, expected {REFERENCE_COMMIT}")
    return subprocess.run(
        ["git", "show", f"{REFERENCE_COMMIT}:comfy/samplers.py"],
        cwd=comfy_git,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _cfg_function(source: str) -> Any:
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "cfg_function"
    ]
    if len(functions) != 1:
        raise SystemExit("pinned source no longer contains exactly one cfg_function")
    namespace: dict[str, object] = {}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, f"{REFERENCE_COMMIT}:comfy/samplers.py", "exec"), namespace)
    return namespace["cfg_function"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-git", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    source = _source(args.comfy_git.resolve())
    cfg = _cfg_function(source)
    cond = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=torch.float32)
    uncond = torch.tensor([[[[-0.5, 0.25], [1.5, -2.0]]]], dtype=torch.float32)
    cases = []
    for scale in (0.0, 1.0, 2.5):
        output = cfg(None, cond, uncond, scale, cond, torch.tensor([0.5]))
        cases.append(
            {
                "scale": scale,
                "shape": list(output.shape),
                "dtype": str(output.dtype).removeprefix("torch."),
                "data": output.reshape(-1).tolist(),
            }
        )
    document = {
        "reference": {
            "commit": REFERENCE_COMMIT,
            "path": "comfy/samplers.py",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "executed_symbol": "cfg_function",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
