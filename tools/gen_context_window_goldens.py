"""Mint context-window planning goldens from the pinned ComfyUI reference.

Executes the planning functions of comfy/context_windows.py exactly as
committed at the pinned reference (read with ``git show``, never from
the working tree) and records window index lists per schedule, fuse
weights standalone and per planned window set, and FreeNoise-shuffled
index orders. The replay tests compare dinkster_inference.context_windows
and the torch windowing layer against these values.

Run with a torch-capable interpreter:

    .venv-torch/bin/python tools/gen_context_window_goldens.py \
        --comfyui ../ComfyUI

The output is bit-stable: regenerating writes byte-identical JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
GOLDEN_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "goldens"
    / ("context_windows_" + REFERENCE_COMMIT[:8] + ".json")
)

WINDOW_CASES = [
    # (schedule, num_frames, length, overlap, stride, closed_loop, steps)
    ("standard_static", 16, 16, 4, 1, False, [0]),
    ("standard_static", 33, 16, 4, 1, False, [0]),
    ("standard_static", 40, 16, 4, 1, False, [0]),
    ("standard_static", 7, 16, 4, 1, False, [0]),
    ("standard_static", 41, 21, 8, 1, False, [0]),
    ("batched", 33, 16, 0, 1, False, [0]),
    ("batched", 32, 16, 0, 1, False, [0]),
    ("batched", 5, 16, 0, 1, False, [0]),
    ("standard_uniform", 33, 16, 4, 1, False, [0, 1, 2, 3, 7, 19]),
    ("standard_uniform", 40, 16, 4, 1, False, [0, 1, 5, 11]),
    ("standard_uniform", 64, 16, 4, 2, False, [0, 1, 2, 9]),
    ("standard_uniform", 16, 16, 4, 1, False, [0, 3]),
    ("looped_uniform", 33, 16, 4, 1, False, [0, 1, 2, 3, 7, 19]),
    ("looped_uniform", 33, 16, 4, 1, True, [0, 1, 2, 5]),
    ("looped_uniform", 64, 16, 4, 2, True, [0, 1, 9]),
    ("looped_uniform", 12, 16, 4, 1, False, [0]),
]

WEIGHT_CASES = [
    # (fuse_method, num_frames, length, overlap, index_list)
    ("flat", 33, 16, 4, list(range(16))),
    ("pyramid", 33, 16, 4, list(range(16))),
    ("pyramid", 33, 15, 4, list(range(15))),
    ("pyramid", 33, 1, 0, [0]),
    ("pyramid", 33, 2, 0, [0, 1]),
    ("overlap-linear", 33, 16, 4, list(range(16))),
    ("overlap-linear", 33, 16, 4, list(range(4, 20))),
    ("overlap-linear", 33, 16, 4, list(range(17, 33))),
    ("overlap-linear", 33, 16, 1, list(range(4, 20))),
]

FREENOISE_CASES = [
    # (num_frames, length, overlap, seed)
    (33, 16, 4, 0),
    (33, 16, 4, 7),
    (40, 16, 4, 42),
    (16, 16, 4, 0),
    (17, 16, 4, 3),
]

ORDERED_HALVING_STEPS = [0, 1, 2, 3, 4, 5, 7, 11, 19, 100]


def load_reference(comfyui: Path) -> types.ModuleType:
    source = subprocess.run(
        ["git", "-C", str(comfyui), "show", f"{REFERENCE_COMMIT}:comfy/context_windows.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    package = types.ModuleType("comfy")
    package.__path__ = []
    sys.modules.setdefault("comfy", package)
    for name in ("comfy.utils", "comfy.model_management", "comfy.patcher_extension", "comfy.conds"):
        stub = types.ModuleType(name)
        sys.modules.setdefault(name, stub)
        setattr(sys.modules["comfy"], name.split(".")[1], stub)
    module = types.ModuleType("reference_context_windows")
    sys.modules["reference_context_windows"] = module
    exec(compile(source, "comfy/context_windows.py", "exec"), module.__dict__)
    return module


def stub_handler(reference: types.ModuleType, case: tuple) -> object:
    schedule, _num_frames, length, overlap, stride, closed_loop, _steps = case
    handler = types.SimpleNamespace(
        context_schedule=reference.get_matching_context_schedule(schedule),
        context_length=length,
        context_overlap=overlap,
        context_stride=stride,
        closed_loop=closed_loop,
        _step=0,
    )
    return handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "-C", str(args.comfyui), "rev-parse", f"{REFERENCE_COMMIT}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != REFERENCE_COMMIT:
        raise SystemExit(f"reference commit resolution mismatch: {head}")
    reference = load_reference(args.comfyui)
    import torch

    windows_out = []
    for case in WINDOW_CASES:
        schedule, num_frames, length, overlap, stride, closed_loop, steps = case
        handler = stub_handler(reference, case)
        per_step = []
        for step in steps:
            handler._step = step
            windows = handler.context_schedule.func(num_frames, handler, {})
            per_step.append({"step": step, "windows": [[int(i) for i in w] for w in windows]})
        windows_out.append(
            {
                "schedule": schedule,
                "num_frames": num_frames,
                "length": length,
                "overlap": overlap,
                "stride": stride,
                "closed_loop": closed_loop,
                "plans": per_step,
            }
        )

    weights_out = []
    for fuse, num_frames, length, overlap, index_list in WEIGHT_CASES:
        method = reference.get_matching_fuse_method(fuse)
        values = method.func(
            len(index_list),
            full_length=num_frames,
            idxs=index_list,
            context_overlap=overlap,
        )
        if isinstance(values, torch.Tensor):
            values = values.tolist()
        weights_out.append(
            {
                "fuse_method": fuse,
                "num_frames": num_frames,
                "length": length,
                "overlap": overlap,
                "index_list": index_list,
                "weights": [float(v) for v in values],
            }
        )

    plan_weights_out = []
    for case in WINDOW_CASES:
        schedule, num_frames, length, overlap, stride, closed_loop, steps = case
        handler = stub_handler(reference, case)
        handler._step = steps[0]
        windows = handler.context_schedule.func(num_frames, handler, {})
        for fuse in ("flat", "pyramid", "overlap-linear"):
            if fuse == "overlap-linear" and overlap < 1:
                continue
            method = reference.get_matching_fuse_method(fuse)
            per_window = []
            for window in windows:
                values = method.func(
                    len(window),
                    full_length=num_frames,
                    idxs=[int(i) for i in window],
                    context_overlap=overlap,
                )
                if isinstance(values, torch.Tensor):
                    values = values.tolist()
                per_window.append([float(v) for v in values])
            plan_weights_out.append(
                {
                    "schedule": schedule,
                    "num_frames": num_frames,
                    "length": length,
                    "overlap": overlap,
                    "stride": stride,
                    "closed_loop": closed_loop,
                    "step": steps[0],
                    "fuse_method": fuse,
                    "weights": per_window,
                }
            )

    freenoise_out = []
    for num_frames, length, overlap, seed in FREENOISE_CASES:
        base = torch.arange(num_frames, dtype=torch.float32).reshape(1, 1, num_frames, 1, 1)
        shuffled = reference.apply_freenoise(base.clone(), 2, length, overlap, seed)
        freenoise_out.append(
            {
                "num_frames": num_frames,
                "length": length,
                "overlap": overlap,
                "seed": seed,
                "shuffled_indices": [int(v) for v in shuffled.flatten().tolist()],
            }
        )

    halving_out = [
        {"step": step, "value": reference.ordered_halving(step)} for step in ORDERED_HALVING_STEPS
    ]

    payload = {
        "_meta": {
            "reference": "comfy/context_windows.py",
            "reference_commit": REFERENCE_COMMIT,
            "generator": "tools/gen_context_window_goldens.py",
        },
        "ordered_halving": halving_out,
        "windows": windows_out,
        "weights": weights_out,
        "plan_weights": plan_weights_out,
        "freenoise": freenoise_out,
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    GOLDEN_PATH.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {GOLDEN_PATH}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
