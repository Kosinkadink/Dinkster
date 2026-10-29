"""Issue-403 diagnostic: replay the four failing LoRA-stack golden cases and
report exact expected-vs-actual divergence (first index, ULP distance) under
the current GEMM configuration. Scratch tool; not part of the test suite."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = (
    ROOT
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "lora_stack_goldens.json"
)
document = json.loads(GOLDEN_PATH.read_text())
meta = document["_meta"]

sys.path.insert(0, str(ROOT / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(ROOT / "packages" / "dinkster-inference-torch" / "src"))
sys.path.insert(0, str(ROOT))

import dinkster_inference_torch as dit  # noqa: E402
from test_patches import build_entry  # noqa: E402

CASES = {
    "lora_stack_linear",
    "lora_stack_linear_reversed",
    "lora_stack_repeated_fractional",
    "lora_stack_repeated_combined",
}


def dec(spec):
    data = torch.tensor(spec["data"], dtype=torch.float32)
    return data.reshape(spec["shape"]).to(getattr(torch, spec["dtype"]))


print("meta:", json.dumps(meta, sort_keys=True))
print("runtime torch:", torch.__version__)

failures = 0
for case in document["apply_cases"]:
    if case["name"] not in CASES:
        continue
    result = dit.apply_patches(
        dec(case["weight"]).clone(),
        [build_entry(entry) for entry in case["entries"]],
        key=case["key"],
        intermediate_dtype=torch.float32,
    )
    expected = dec(case["expected"])
    different = result != expected
    count = int(torch.count_nonzero(different).item())
    if count:
        first = int(torch.argmax(different.flatten()).item())
        a = result.flatten()[first].item()
        b = expected.flatten()[first].item()
        # ULP distance: how many representable floats between expected and actual
        lo, hi = (b, a) if b <= a else (a, b)
        steps = 0
        cur = lo
        while cur < hi and steps < 64:
            cur = math.nextafter(cur, math.inf)
            steps += 1
        failures += 1
        print(
            f"{case['name']}: MISMATCH count={count}/{result.numel()} "
            f"first_index={first} expected={b!r} actual={a!r} ulp_steps={steps}"
        )
    else:
        print(f"{case['name']}: exact match ({result.numel()} values)")

print("mismatched_cases:", failures)
sys.exit(0)
