"""Exhaustive fp32 evaluation-order bound for the lora stack golden cases.

The four linear stack cases compare apply_patches output against the
executed-reference golden with torch.equal. The only host-dependent
operation on the path is torch.mm on 8xr @ rx6 (r in {1,2,3}); every
scaling and composition step is a host-independent single rounding.

Each output element of the composite is the independent chain
w + d1 + d2 + d3 (entry order), where each dk is the elementwise scaling
of that element's dot result. This instrument enumerates EVERY fp32
evaluation strategy a standard CPU fp32-accumulating kernel can produce
for each dot: all plain summation trees over the rounded products, every
single-FMA-at-root variant, and every FMA chain - computed in exact
Fraction arithmetic with one round-to-nearest-even fp32 step per
operation (FMA = fl32(a*b + c), a single rounding). Candidate deltas are
then scaled through the real torch op so the chain semantics match the
tested code exactly.

For each case and element it walks the cartesian product of per-entry
candidates against the golden value and reports the maximum ULP distance
over ALL candidates and whether every candidate satisfies the
adjacent-ULP contract.

Model exclusions (stated, not hidden): torch CPU sgemm accumulates fp32
with FMA and no extended precision; kernels accumulating in fp64 or
internally in bf16 are outside the model and do not exist for fp32 CPU
sgemm in torch 2.13.
"""

from __future__ import annotations

import itertools
import json
import math
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from dinkster_inference.patches import AdapterPatch, PatchEntry
from dinkster_inference_torch import LoRAAdapter, apply_patches

REPO = Path(__file__).resolve().parent.parent
DOC = json.loads(
    (REPO / "packages/dinkster-inference-torch/tests/goldens/lora_stack_goldens.json").read_text()
)


def round32(x: Fraction) -> Fraction:
    """Round an exact Fraction to float32, round-to-nearest-even."""
    if x == 0:
        return Fraction(0.0)
    sign = -1 if x < 0 else 1
    x = abs(x)
    approx = float(x)
    if approx == 0.0:
        return Fraction(sign * 0.0)
    e = math.floor(math.log2(approx))
    best = None
    for ee in (e - 1, e, e + 1):
        scaled = x * Fraction(2) ** (23 - ee)
        m_floor = scaled.numerator // scaled.denominator
        for m in (m_floor, m_floor + 1):
            if not (2**23 <= m <= 2**24):
                continue
            value = Fraction(m) * Fraction(2) ** (ee - 23)
            key = (abs(value - x), m)
            if best is None or key < best[0]:
                best = (key, value)
    assert best is not None
    return sign * best[1]


def dot_candidates(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    """All candidate fp32 roundings of sum(a_i*b_i) over standard kernel
    evaluation strategies (plain trees, single root FMA, FMA chains).
    Every returned Fraction is an exactly representable fp32 value."""
    n = len(a)
    exact_ab = [ai * bi for ai, bi in zip(a, b)]
    rounded = [round32(p) for p in exact_ab]

    if n == 1:
        return [rounded[0]]

    results: list[Fraction] = []
    for perm in itertools.permutations(range(n)):
        acc = rounded[perm[0]]
        for idx in perm[1:]:
            acc = round32(acc + rounded[idx])
        results.append(acc)

    if n == 2:
        results.append(round32(exact_ab[0] + rounded[1]))
        results.append(round32(exact_ab[1] + rounded[0]))
        return results

    for i in range(n):
        j, k = [idx for idx in range(n) if idx != i]
        base = round32(rounded[j] + rounded[k])
        results.append(round32(exact_ab[i] + base))
    for perm in itertools.permutations(range(n)):
        acc = rounded[perm[0]]
        for idx in perm[1:]:
            acc = round32(exact_ab[idx] + acc)
        results.append(acc)
    return results


def ulp_distance(x: float, y: float) -> int:
    (xb,) = struct.unpack("<i", struct.pack("<f", x))
    (yb,) = struct.unpack("<i", struct.pack("<f", y))

    def key(bits: int) -> int:
        return (-2) * bits - 1 if bits < 0 else 2 * bits

    return abs(key(xb) - key(yb))


def adjacent_or_equal(candidate: float, expected: float) -> bool:
    if candidate == expected:
        return True
    t_c = torch.tensor(candidate, dtype=torch.float32)
    t_e = torch.tensor(expected, dtype=torch.float32)
    return bool(torch.equal(torch.nextafter(t_e, t_c), t_c))


def run_case(case: dict) -> dict:
    weight = np.array(case["weight"]["data"], dtype=np.float32).reshape(case["weight"]["shape"])
    expected = np.array(case["expected"]["data"], dtype=np.float32).reshape(case["expected"]["shape"])

    # Local (host) apply_patches result for sanity: must equal the golden.
    entries = []
    for entry in case["entries"]:
        v = entry["value"]
        up = torch.tensor(v["up"]["data"], dtype=torch.float32).reshape(v["up"]["shape"])
        down = torch.tensor(v["down"]["data"], dtype=torch.float32).reshape(v["down"]["shape"])
        entries.append(
            PatchEntry(
                AdapterPatch(LoRAAdapter(up, down, alpha=v["alpha"])),
                strength=entry["strength"],
                strength_model=entry["strength_model"],
            )
        )
    w = torch.tensor(case["weight"]["data"], dtype=torch.float32).reshape(case["weight"]["shape"])
    local = apply_patches(w.clone(), entries, key=case["key"], intermediate_dtype=torch.float32)
    local_ulp = sum(
        ulp_distance(float(local[i, j]), float(expected[i, j]))
        for i in range(expected.shape[0])
        for j in range(expected.shape[1])
    )

    # Per-element candidate deltas per entry. Scaling goes through the real
    # torch op (host-independent single rounding) per candidate dot value.
    entry_candidates: list[list[list[list[float]]]] = []
    entry_meta = []
    for entry in case["entries"]:
        v = entry["value"]
        up = np.array(v["up"]["data"], dtype=np.float32).reshape(v["up"]["shape"])
        down = np.array(v["down"]["data"], dtype=np.float32).reshape(v["down"]["shape"])
        alpha = v["alpha"]
        r = down.shape[0]
        alpha_val = 1.0 if alpha is None else alpha / r
        scalar = entry["strength"] * alpha_val
        cands: list[list[list[float]]] = []
        for i in range(up.shape[0]):
            row = []
            for j in range(down.shape[1]):
                a = [Fraction(x) for x in up[i].tolist()]
                b = [Fraction(x) for x in down[:, j].tolist()]
                dots = dot_candidates(a, b)
                scaled = set()
                for d in dots:
                    t = torch.tensor([[float(d)]], dtype=torch.float32)
                    scaled.add(float(((scalar * t) * 1.0).item()))
                row.append(sorted(scaled))
            cands.append(row)
        entry_candidates.append(cands)
        entry_meta.append((up.shape, r, len(cands[0][0])))

    max_ulp = 0
    worst_info = None
    chains = 0
    beyond = 0
    exact_chains = 0
    examples_beyond = []
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            w0 = float(weight[i, j])
            exp = float(expected[i, j])
            for combo in itertools.product(*(entry_candidates[e][i][j] for e in range(len(entry_candidates)))):
                s = np.float32(w0)
                for d in combo:
                    s = np.float32(float(s) + d)
                chains += 1
                dist = ulp_distance(float(s), exp)
                if dist == 0:
                    exact_chains += 1
                if dist > max_ulp:
                    max_ulp = dist
                    worst_info = (i, j, exp, float(s), dist)
                if dist > 1:
                    beyond += 1
                    if len(examples_beyond) < 5:
                        examples_beyond.append((i, j, exp, float(s), dist))

    print(f"case {case['name']}:")
    print(f"  entries (shape, rank, dot candidates/element): {entry_meta}")
    print(f"  local apply_patches vs golden total ULP: {local_ulp} (0 = bit-exact)")
    print(f"  enumerated candidate chains: {chains}")
    print(f"  max ULP distance from golden over ALL candidates: {max_ulp}")
    if worst_info and max_ulp > 0:
        i, j, exp, val, dist = worst_info
        print(f"  worst: element ({i},{j}) expected={exp!r} candidate={val!r} dist={dist}")
    print(f"  candidate chains ending beyond 1 ULP: {beyond}")
    print(f"  candidate chains ending bit-identical to golden: {exact_chains}")
    if examples_beyond:
        for i, j, exp, val, dist in examples_beyond:
            print(f"    beyond-ULP example: ({i},{j}) expected={exp!r} candidate={val!r} dist={dist}")
    print()
    return {"max_ulp": max_ulp, "beyond": beyond, "chains": chains, "local_ulp": local_ulp}


summary = {}
for case in DOC["apply_cases"]:
    if case["name"] == "lora_stack_convolution":
        continue
    summary[case["name"]] = run_case(case)

print("SUMMARY: " + json.dumps(summary, sort_keys=True))
