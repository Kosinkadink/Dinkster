"""Generate brownian-tree noise goldens from the reference stack.

Runs the REFERENCE implementations - numpy.random.SeedSequence,
torchsde.BrownianTree (0.2.6), and comfy/k_diffusion/sampling.py
BrownianTreeNoiseSampler @ the audited baseline - and writes
packages/dinkster-inference-torch/tests/goldens/brownian_goldens.json.
dinkster_inference_torch.brownian is pinned against these outputs - the
oracle is the reference code itself, never a re-derivation.

Three layers, matching the transcription's structure:

- seedseq_cases: numpy SeedSequence generate_state word streams for
  int entropy + int spawn keys (the shapes torchsde produces).
- tree_cases: torchsde.BrownianTree interval queries, including
  repeated and out-of-query-order pairs (the tree memoizes; replay
  must be bit-identical) in float32 and float64.
- sampler_cases: BrownianTreeNoiseSampler(cpu=True) step noise over
  descending schedules, including a query at the RAW sigma_max (the
  tree bounds round inward at construction, so this exercises the
  reference's clamp path) and one ascending pair (the sign path).

All expected tensors are exact: float32/float64 values round-trip
through JSON as shortest-repr float64.

Usage (needs a torch interpreter with numpy + torchsde + the pinned
ComfyUI checkout's import closure; the workspace root venv is
deliberately torch-free and the validation venvs deliberately carry
neither numpy nor torchsde):

    PYTHONPATH=../ComfyUI:../comfy-aimdo:../comfy-kitchen \
        /path/to/ref-venv/bin/python tools/gen_brownian_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torchsde

# comfy.k_diffusion.sampling drags in comfy.model_management, which
# probes CUDA at import time; ask for CPU explicitly so the generator
# also runs on CPU-only torch builds.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.k_diffusion.sampling as kds  # noqa: E402
from golden_platform import platform_golden_path, platform_provenance  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = platform_golden_path(
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "brownian_goldens.json",
    torch.__version__,
)

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def enc(x: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.flatten().tolist(),
    }


def f32(value: float) -> float:
    """The wrapper's query-time dtype path: float32 truncation
    widened back to a Python float."""
    return float(torch.tensor(value, dtype=torch.float32))


def seedseq_cases() -> list[dict[str, object]]:
    cases = []
    for entropy, spawn_key, pool_size, n_words in [
        (0, [], 24, 8),
        (1, [], 24, 8),
        (42, [], 24, 4),
        (2**31 - 1, [], 24, 4),
        (2**63 - 1, [], 24, 4),
        (2**64 + 17, [], 24, 4),
        (5, [], 4, 8),
        (42, [0, 1], 24, 4),
        (42, [5, 3], 24, 4),
        (2**63 - 1, [2**40, 27], 24, 4),
        (7, [123456789, 24], 24, 4),
        (7, [1, 2], 4, 4),
    ]:
        state = np.random.SeedSequence(
            entropy, spawn_key=tuple(spawn_key), pool_size=pool_size
        ).generate_state(n_words)
        cases.append(
            {
                "entropy": entropy,
                "spawn_key": spawn_key,
                "pool_size": pool_size,
                "expected": [int(w) for w in state],
            }
        )
    return cases


def tree_cases() -> list[dict[str, object]]:
    cases = []
    for name, t0, t1, shape, dtype_name, entropy, queries in [
        # SD-like sigma span, repeated + out-of-order queries.
        (
            "sd_span_f32",
            0.0292,
            14.6146,
            [2, 3],
            "float32",
            42,
            [
                [7.0, 14.6146],
                [0.5, 7.0],
                [0.5, 7.0],  # exact repeat: memoized replay
                [0.0292, 0.5],
                [3.0, 10.0],  # overlaps earlier splits out of order
                [0.0292, 14.6146],  # the whole interval
            ],
        ),
        # Flow-like unit span.
        (
            "unit_span_f32",
            0.003,
            1.0,
            [1, 4, 2, 2],
            "float32",
            0,
            [
                [0.7, 1.0],
                [0.35, 0.7],
                [0.1, 0.35],
                [0.003, 0.1],
                [0.003, 1.0],
            ],
        ),
        # float64 tensor path with a >32-bit entropy.
        (
            "wide_span_f64",
            0.1,
            9.7,
            [3],
            "float64",
            2**62 + 5,
            [
                [4.0, 9.7],
                [0.1, 4.0],
                [2.0, 6.0],
                [4.0, 9.7],  # repeat after more splits
            ],
        ),
    ]:
        dtype = DTYPES[dtype_name]
        w0 = torch.zeros(shape, dtype=dtype)
        tree = torchsde.BrownianTree(
            torch.tensor(f32(t0)),
            w0,
            torch.tensor(f32(t1)),
            entropy=entropy,
        )
        expected = []
        for qa, qb in queries:
            w = tree(torch.tensor(f32(qa)), torch.tensor(f32(qb)))
            expected.append(enc(w))
        cases.append(
            {
                "name": name,
                "t0": f32(t0),
                "t1": f32(t1),
                "shape": shape,
                "dtype": dtype_name,
                "entropy": entropy,
                "queries": [[f32(a), f32(b)] for a, b in queries],
                "expected": expected,
            }
        )
    return cases


def sampler_cases() -> list[dict[str, object]]:
    cases = []
    for name, sigma_min, sigma_max, shape, dtype_name, seed, queries in [
        # SD-like descending schedule; first query at the RAW
        # sigma_max overhangs the rounded tree bound (clamp path).
        (
            "sd_schedule",
            0.0292,
            14.6146,
            [2, 4, 8, 8],
            "float32",
            42,
            [
                [14.6146, 8.0],
                [8.0, 3.5],
                [3.5, 1.2],
                [1.2, 0.4],
                [0.4, 0.0292],
            ],
        ),
        # Flow-like schedule, big seed, plus one ASCENDING pair to
        # exercise the sort/sign path.
        (
            "flow_schedule",
            0.05,
            0.9999,
            [1, 16, 4, 4],
            "float32",
            2**63 - 1,
            [
                [0.9999, 0.7],
                [0.7, 0.4],
                [0.4, 0.7],  # ascending: sign flip
                [0.4, 0.15],
                [0.15, 0.05],
            ],
        ),
        # Repeat queries must replay bit-identically.
        (
            "replay",
            0.1,
            5.0,
            [3, 2],
            "float32",
            7,
            [
                [5.0, 2.0],
                [2.0, 0.5],
                [5.0, 2.0],
                [2.0, 0.5],
            ],
        ),
    ]:
        dtype = DTYPES[dtype_name]
        x = torch.zeros(shape, dtype=dtype)
        with warnings.catch_warnings():
            # the raw-bound query overhangs the rounded tree bound;
            # the reference warns and clamps, and the clamp IS the
            # behavior under pin.
            warnings.simplefilter("ignore")
            sampler = kds.BrownianTreeNoiseSampler(
                x,
                torch.tensor(sigma_min),
                torch.tensor(sigma_max),
                seed=seed,
                cpu=True,
            )
            expected = []
            for s_from, s_to in queries:
                w = sampler(torch.tensor(s_from), torch.tensor(s_to))
                expected.append(enc(w))
        cases.append(
            {
                "name": name,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "shape": shape,
                "dtype": dtype_name,
                "seed": seed,
                "queries": queries,
                "expected": expected,
            }
        )
    return cases


def main() -> None:
    commit = subprocess.run(
        ["git", "-C", str(REPO.parent / "ComfyUI"), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    seedseq = seedseq_cases()
    trees = tree_cases()
    samplers = sampler_cases()
    OUT.write_text(
        json.dumps(
            {
                "_meta": {
                    "reference_commit": commit,
                    "torchsde": torchsde.__version__,
                    "numpy": np.__version__,
                    "generator": "tools/gen_brownian_goldens.py",
                    **platform_provenance(torch.__version__, pin_cpu=True),
                },
                "seedseq_cases": seedseq,
                "tree_cases": trees,
                "sampler_cases": samplers,
            },
            indent=1,
        )
        + "\n",
        newline="\n",
    )
    print(
        f"wrote {len(seedseq)} seedseq + {len(trees)} tree + {len(samplers)} sampler cases to {OUT}"
    )


if __name__ == "__main__":
    main()
