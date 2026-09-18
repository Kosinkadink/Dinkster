"""Capture ModelSamplingLTXV shifts from pinned ComfyUI source.

Usage: .venv/bin/python tools/gen_ltxv_dynamic_shift_goldens.py COMFY_ROOT
Only model patch storage and schema declarations are substituted. The shift
arithmetic comes from the named, unmodified upstream node definition.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests/goldens/ltxv_dynamic_shift.json"


class NodeOutput:
    def __init__(self, *values: object) -> None:
        self.values = values


class ModelSamplingFlux:
    def __init__(self, model_config: object) -> None:
        self.model_config = model_config
        self.shift: float | None = None

    def set_parameters(self, *, shift: float) -> None:
        self.shift = shift


class CONST:
    pass


class Patcher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(model_config=object())
        self.patches: dict[str, object] = {}

    def clone(self) -> Patcher:
        result = Patcher()
        result.patches = dict(self.patches)
        return result

    def add_object_patch(self, name: str, value: object) -> None:
        self.patches[name] = value


def definition(path: Path, name: str, namespace: dict[str, Any]) -> None:
    source = ast.parse(path.read_text())
    body = [node for node in source.body if getattr(node, "name", None) == name]
    assert len(body) == 1
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def main() -> None:
    comfy = Path(sys.argv[1]).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI must be at {REFERENCE_COMMIT}")
    io = SimpleNamespace(ComfyNode=object, NodeOutput=NodeOutput)
    namespace: dict[str, Any] = {
        "io": io,
        "math": math,
        "comfy": SimpleNamespace(
            model_sampling=SimpleNamespace(ModelSamplingFlux=ModelSamplingFlux, CONST=CONST)
        ),
    }
    definition(comfy / "comfy_extras/nodes_lt.py", "ModelSamplingLTXV", namespace)
    execute = namespace["ModelSamplingLTXV"].execute
    cases = []
    for name, tokens, maximum, base in (
        ("default", None, 2.05, 0.95),
        ("low_endpoint", 1024, 3.5, 1.25),
        ("midpoint", 2560, 3.5, 1.25),
        ("high_endpoint", 4096, 3.5, 1.25),
        ("below_range", 512, 2.05, 0.95),
        ("above_range", 8192, 2.05, 0.95),
    ):
        latent = None
        if tokens is not None:
            latent = {"samples": SimpleNamespace(shape=(1, 128, 1, 1, tokens))}
        output = execute(Patcher(), maximum, base, latent)
        patched = output.values[0].patches["model_sampling"]
        cases.append(
            {
                "name": name,
                "tokens": 4096 if tokens is None else tokens,
                "max_shift": maximum,
                "base_shift": base,
                "shift": patched.shift,
            }
        )
    OUTPUT.write_text(json.dumps({"comfyui_commit": actual, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
