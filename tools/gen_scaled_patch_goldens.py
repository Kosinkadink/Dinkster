"""Generate scaled-patch goldens from pinned ComfyUI source.

Usage (the interpreter must provide torch):

    .venv-gpu/bin/python tools/gen_scaled_patch_goldens.py \
        --comfy-git ../ComfyUI

The ComfyUI checkout need not move. The generator reads the audited files with
``git show``, extracts the exact ``LoraDiff`` low-rank implementation,
``get_module_type_info``, and ``BypassForwardHook``, then executes the pinned
``h`` and default ``g(f(x) + h(x))`` combination on CPU. The full-diff case
pins the direct ``F.linear`` formula.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from golden_platform import platform_golden_path, tuple_provenance

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = platform_golden_path(
    REPO / "packages/dinkster-inference-torch/tests/goldens/scaled_patch_goldens.json",
    torch.__version__,
)


def _source(comfy_git: Path, path: str) -> str:
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
        ["git", "show", f"{REFERENCE_COMMIT}:{path}"],
        cwd=comfy_git,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


class _WeightAdapterBase:
    def bypass_forward(self, forward: Any, value: torch.Tensor, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def g(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _WeightAdapterTrainBase(nn.Module):
    def bypass_forward(self, forward: Any, value: torch.Tensor, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def g(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _reference(lora_source: str, bypass_source: str) -> dict[str, Any]:
    lora_classes = [
        node
        for node in ast.parse(lora_source).body
        if isinstance(node, ast.ClassDef) and node.name == "LoraDiff"
    ]
    bypass_nodes = [
        node
        for node in ast.parse(bypass_source).body
        if (isinstance(node, ast.FunctionDef) and node.name == "get_module_type_info")
        or isinstance(node, ast.ClassDef)
        and node.name == "BypassForwardHook"
    ]
    if len(lora_classes) != 1 or len(bypass_nodes) != 2:
        raise SystemExit("pinned source no longer contains the expected low-rank surfaces")
    namespace: dict[str, Any] = {
        "F": F,
        "Optional": __import__("typing").Optional,
        "Union": __import__("typing").Union,
        "WeightAdapterBase": _WeightAdapterBase,
        "WeightAdapterTrainBase": _WeightAdapterTrainBase,
        "logging": logging,
        "nn": nn,
        "torch": torch,
        "comfy": SimpleNamespace(
            model_management=SimpleNamespace(get_torch_device=lambda: torch.device("cpu"))
        ),
        "tucker_weight_from_conv": lambda *_args: (_ for _ in ()).throw(
            AssertionError("tucker mid path is outside the pinned scaled-patch scope")
        ),
    }
    module = ast.Module(body=lora_classes + bypass_nodes, type_ignores=[])
    exec(compile(module, f"{REFERENCE_COMMIT}:scaled-patch-reference", "exec"), namespace)
    return namespace


def _tensor(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "data": value.detach().reshape(-1).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-git", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    lora_source = _source(args.comfy_git.resolve(), "comfy/weight_adapter/lora.py")
    bypass_source = _source(args.comfy_git.resolve(), "comfy/weight_adapter/bypass.py")
    ref = _reference(lora_source, bypass_source)

    base = torch.tensor([[0.25, -0.5, 0.75], [-1.0, 0.5, 0.125]], dtype=torch.float32)
    value = torch.tensor(
        [[1.0, -2.0, 0.5], [-0.25, 0.75, 2.0], [1.5, 0.0, -1.0]],
        dtype=torch.float32,
    )
    up = torch.tensor([[0.5, -1.0], [1.25, 0.25]], dtype=torch.float32)
    down = torch.tensor([[0.75, -0.5, 1.0], [-1.5, 0.25, 0.5]], dtype=torch.float32)
    alpha = 3.0
    multiplier = 0.4

    layer = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(base)
    adapter = ref["LoraDiff"]((up, down, alpha, None, None, None))
    hook = ref["BypassForwardHook"](layer, adapter, multiplier)
    hook.original_forward = layer.forward
    base_output = layer(value)
    lora_delta = adapter.h(value, base_output)
    lora_output = hook._bypass_forward(value)

    diff = torch.tensor([[-0.125, 0.375, 0.5], [0.75, -0.25, 0.625]], dtype=torch.float32)
    strength = 0.6
    full_diff_delta = F.linear(value, diff) * strength
    scales = torch.tensor([0.0, 1.0, -0.5], dtype=torch.float32)
    full_diff_output = base_output + scales[:, None] * full_diff_delta

    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": REFERENCE_COMMIT,
            "paths": ["comfy/weight_adapter/lora.py", "comfy/weight_adapter/bypass.py"],
            "source_sha256": {
                "comfy/weight_adapter/lora.py": hashlib.sha256(
                    lora_source.encode("utf-8")
                ).hexdigest(),
                "comfy/weight_adapter/bypass.py": hashlib.sha256(
                    bypass_source.encode("utf-8")
                ).hexdigest(),
            },
            "interpreter": sys.version.splitlines()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "executed": [
                "LoraDiff.h",
                "BypassForwardHook._bypass_forward default g(f(x) + h(x))",
            ],
            "full_diff_formula": "base + scale[:,None] * (F.linear(input, diff) * strength)",
            **tuple_provenance(torch.__version__),
        },
        "inputs": {
            "base_weight": _tensor(base),
            "input": _tensor(value),
            "up": _tensor(up),
            "down": _tensor(down),
            "alpha": alpha,
            "multiplier": multiplier,
            "diff": _tensor(diff),
            "diff_strength": strength,
            "scales": _tensor(scales),
        },
        "outputs": {
            "base": _tensor(base_output),
            "lora_delta": _tensor(lora_delta),
            "lora_bypass": _tensor(lora_output),
            "full_diff_delta": _tensor(full_diff_delta),
            "full_diff_scaled": _tensor(full_diff_output),
        },
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


if __name__ == "__main__":
    main()
