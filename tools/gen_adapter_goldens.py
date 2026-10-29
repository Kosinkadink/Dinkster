"""Generate adapter-math goldens from the ComfyUI reference checkout.

Runs the REFERENCE comfy/weight_adapter/*.calculate_weight @ the
audited baseline on small seeded tensors and writes
packages/dinkster-inference-torch/tests/goldens/adapter_goldens.json.
dinkster_inference_torch.adapters is pinned against these outputs - the
oracle is the reference code itself. Constraint cases pass the
trainer's effective ``alpha * out_dim`` bound to that reference while
recording the raw alpha stored in adapter files.

Input tensors are stored IN the golden file (float32/float16 values
round-trip exactly through JSON doubles), so the pin does not depend
on RNG stability across torch versions.

Usage (needs a torch interpreter that imports the pinned checkout;
the workspace root venv is deliberately torch-free):

    PYTHONPATH=../ComfyUI /path/to/torch-venv/bin/python tools/gen_adapter_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch  # noqa: E402
from comfy.weight_adapter.boft import BOFTAdapter  # noqa: E402
from comfy.weight_adapter.glora import GLoRAAdapter  # noqa: E402
from comfy.weight_adapter.loha import LoHaAdapter  # noqa: E402
from comfy.weight_adapter.lokr import LoKrAdapter  # noqa: E402
from comfy.weight_adapter.lora import LoRAAdapter  # noqa: E402
from comfy.weight_adapter.oft import OFTAdapter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "adapter_goldens.json"

_gen = torch.Generator().manual_seed(0x40B)


def t(*shape: int, dtype: torch.dtype = torch.float32, scale: float = 1.0):
    x = torch.randn(shape, generator=_gen, dtype=torch.float32) * scale
    return x.to(dtype)


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


def enc_opt(x):
    return None if x is None else enc(x)


def make(cls, weights):
    """Reference adapters read only self.weights in calculate_weight;
    bypass __init__ (which does train-mode shape probing)."""
    ad = object.__new__(cls)
    ad.weights = weights
    ad.loaded_keys = set()
    return ad


CASES: list[dict] = []
CONSTRAINT_CONTRACT_CASES = ["oft_constraint", "boft_rescale_strength"]


def case(
    name: str,
    cls,
    weights: tuple,
    weight: torch.Tensor,
    *,
    strength: float,
    slots: list[str],
    constraint_out_dim: int | None = None,
) -> None:
    """Run the reference calculate_weight and record inputs+output."""
    reference_weights = weights
    if constraint_out_dim is not None:
        raw_alpha = weights[2]
        if not isinstance(raw_alpha, int | float):
            raise TypeError(f"{name}: constraint alpha must be numeric")
        reference_weights = (*weights[:2], raw_alpha * constraint_out_dim, *weights[3:])
    result = make(cls, reference_weights).calculate_weight(
        weight.clone(),
        key=name,
        strength=strength,
        strength_model=1.0,
        offset=None,
        function=lambda x: x,
        intermediate_dtype=torch.float32,
    )
    if not torch.isfinite(result.to(torch.float32)).all():
        raise RuntimeError(f"{name}: non-finite reference output")
    if torch.equal(result.to(torch.float32), weight.to(torch.float32)) and not name.endswith(
        "_noop"
    ):
        raise RuntimeError(f"{name}: reference left the weight unpatched (silent error?)")
    named: dict[str, object] = {}
    for slot, value in zip(slots, weights, strict=True):
        if isinstance(value, torch.Tensor):
            named[slot] = enc(value)
        else:
            named[slot] = value
    CASES.append(
        {
            "name": name,
            "adapter": cls.__name__,
            "weights": named,
            "weight": enc(weight),
            "strength": strength,
            "expected": enc(result),
        }
    )


LORA_SLOTS = ["up", "down", "alpha", "mid", "dora_scale", "reshape"]
LOHA_SLOTS = ["w1_a", "w1_b", "alpha", "w2_a", "w2_b", "t1", "t2", "dora_scale"]
LOKR_SLOTS = ["w1", "w2", "alpha", "w1_a", "w1_b", "w2_a", "w2_b", "t2", "dora_scale"]
GLORA_SLOTS = ["a1", "a2", "b1", "b2", "alpha", "dora_scale"]
OFT_SLOTS = ["blocks", "rescale", "alpha", "dora_scale"]


def gen_lora() -> None:
    w = t(8, 6)
    up, down = t(8, 3, scale=0.4), t(3, 6, scale=0.4)
    case(
        "lora_linear_plain",
        LoRAAdapter,
        (up, down, 2.0, None, None, None),
        w,
        strength=1.0,
        slots=LORA_SLOTS,
    )
    case(
        "lora_linear_strength",
        LoRAAdapter,
        (up, down, 2.0, None, None, None),
        w,
        strength=0.6,
        slots=LORA_SLOTS,
    )
    case(
        "lora_linear_noalpha",
        LoRAAdapter,
        (up, down, None, None, None, None),
        w,
        strength=1.0,
        slots=LORA_SLOTS,
    )
    wc = t(8, 4, 3, 3)
    case(
        "lora_conv_plain",
        LoRAAdapter,
        (t(8, 3, 1, 1, scale=0.4), t(3, 4, 3, 3, scale=0.4), 1.5, None, None, None),
        wc,
        strength=0.8,
        slots=LORA_SLOTS,
    )
    case(
        "lora_conv_mid",
        LoRAAdapter,
        (
            t(8, 3, 1, 1, scale=0.4),
            t(3, 4, 1, 1, scale=0.4),
            1.5,
            t(3, 3, 3, 3, scale=0.4),
            None,
            None,
        ),
        wc,
        strength=0.8,
        slots=LORA_SLOTS,
    )
    case(
        "lora_reshape_pad",
        LoRAAdapter,
        (t(8, 3, scale=0.4), t(3, 6, scale=0.4), None, None, None, [8, 6]),
        t(6, 4),
        strength=1.0,
        slots=LORA_SLOTS,
    )
    case(
        "lora_dora_out_axis",
        LoRAAdapter,
        (up, down, 2.0, None, t(8, 1, scale=0.5).abs() + 0.5, None),
        w,
        strength=0.7,
        slots=LORA_SLOTS,
    )
    case(
        "lora_dora_in_axis",
        LoRAAdapter,
        (up, down, 2.0, None, (t(1, 6, scale=0.5).abs() + 0.5), None),
        w,
        strength=1.0,
        slots=LORA_SLOTS,
    )
    case(
        "lora_fp16",
        LoRAAdapter,
        (
            t(8, 3, dtype=torch.float16, scale=0.4),
            t(3, 6, dtype=torch.float16, scale=0.4),
            2.0,
            None,
            None,
            None,
        ),
        t(8, 6, dtype=torch.float16),
        strength=0.5,
        slots=LORA_SLOTS,
    )


def gen_loha() -> None:
    w = t(8, 6)
    w1a, w1b = t(8, 3, scale=0.4), t(3, 6, scale=0.4)
    w2a, w2b = t(8, 3, scale=0.4), t(3, 6, scale=0.4)
    case(
        "loha_plain",
        LoHaAdapter,
        (w1a, w1b, 1.5, w2a, w2b, None, None, None),
        w,
        strength=0.9,
        slots=LOHA_SLOTS,
    )
    case(
        "loha_tucker",
        LoHaAdapter,
        (
            t(3, 8, scale=0.4),
            t(3, 4, scale=0.4),
            2.0,
            t(3, 8, scale=0.4),
            t(3, 4, scale=0.4),
            t(3, 3, 3, 3, scale=0.4),
            t(3, 3, 3, 3, scale=0.4),
            None,
        ),
        t(8, 4, 3, 3),
        strength=1.0,
        slots=LOHA_SLOTS,
    )
    case(
        "loha_dora",
        LoHaAdapter,
        (w1a, w1b, 1.5, w2a, w2b, None, None, t(8, 1, scale=0.5).abs() + 0.5),
        w,
        strength=0.5,
        slots=LOHA_SLOTS,
    )


def gen_lokr() -> None:
    w = t(8, 6)
    w1, w2 = t(2, 3, scale=0.4), t(4, 2, scale=0.4)
    case(
        "lokr_full",
        LoKrAdapter,
        (w1, w2, 2.0, None, None, None, None, None, None),
        w,
        strength=1.0,
        slots=LOKR_SLOTS,
    )
    case(
        "lokr_decomposed",
        LoKrAdapter,
        (
            None,
            None,
            1.5,
            t(2, 3, scale=0.4),
            t(3, 3, scale=0.4),
            t(4, 3, scale=0.4),
            t(3, 2, scale=0.4),
            None,
            None,
        ),
        w,
        strength=0.8,
        slots=LOKR_SLOTS,
    )
    # NOTE deliberately no lokr tucker (t2) golden: at 947c2749 the
    # reference ALWAYS fails on that path (the einsum-produced w2 is
    # non-contiguous and torch.kron raises a view error; logged +
    # weight silently unpatched, verified on torch 2.9.1). Pinned as
    # an explicit loud-failure test instead of a golden.
    case(
        "lokr_dora",
        LoKrAdapter,
        (w1, w2, None, None, None, None, None, None, t(8, 1, scale=0.5).abs() + 0.5),
        w,
        strength=0.5,
        slots=LOKR_SLOTS,
    )


def gen_glora() -> None:
    w = t(8, 6)
    case(
        "glora_old",
        GLoRAAdapter,
        (
            t(3, 6, scale=0.4),
            t(6, 3, scale=0.4),
            t(3, 6, scale=0.4),
            t(8, 3, scale=0.4),
            1.5,
            None,
        ),
        w,
        strength=0.9,
        slots=GLORA_SLOTS,
    )
    case(
        "glora_new",
        GLoRAAdapter,
        (
            t(6, 3, scale=0.4),
            t(3, 6, scale=0.4),
            t(8, 3, scale=0.4),
            t(3, 6, scale=0.4),
            1.5,
            None,
        ),
        w,
        strength=0.9,
        slots=GLORA_SLOTS,
    )
    case(
        "glora_new_conv",
        GLoRAAdapter,
        (
            t(6, 3, scale=0.4),
            t(3, 6, scale=0.4),
            t(8, 3, scale=0.4),
            t(3, 6, 3, 3, scale=0.4),
            None,
            None,
        ),
        t(8, 6, 3, 3),
        strength=1.0,
        slots=GLORA_SLOTS,
    )
    case(
        "glora_dora",
        GLoRAAdapter,
        (
            t(3, 6, scale=0.4),
            t(6, 3, scale=0.4),
            t(3, 6, scale=0.4),
            t(8, 3, scale=0.4),
            1.5,
            t(8, 1, scale=0.5).abs() + 0.5,
        ),
        w,
        strength=0.5,
        slots=GLORA_SLOTS,
    )


def gen_oft() -> None:
    w = t(8, 6)
    blocks = t(2, 4, 4, scale=0.1)
    case(
        "oft_plain",
        OFTAdapter,
        (blocks, None, None, None),
        w,
        strength=1.0,
        slots=OFT_SLOTS,
    )
    case(
        "oft_constraint",
        OFTAdapter,
        (blocks, None, 0.05, None),
        w,
        strength=0.7,
        slots=OFT_SLOTS,
        constraint_out_dim=blocks.shape[0] * blocks.shape[1],
    )
    case(
        "oft_dora",
        OFTAdapter,
        (blocks, None, None, t(8, 1, scale=0.5).abs() + 0.5),
        w,
        strength=0.5,
        slots=OFT_SLOTS,
    )


def gen_boft() -> None:
    w = t(8, 6)
    blocks = t(2, 2, 4, 4, scale=0.1)
    case(
        "boft_plain",
        BOFTAdapter,
        (blocks, None, 0.0, None),
        w,
        strength=1.0,
        slots=OFT_SLOTS,
    )
    case(
        "boft_rescale_strength",
        BOFTAdapter,
        (blocks, t(8, 1, scale=0.3).abs() + 0.7, 0.05, None),
        w,
        strength=0.6,
        slots=OFT_SLOTS,
        constraint_out_dim=blocks.shape[1] * blocks.shape[2],
    )
    # NOTE fp16 golden only at strength == 1: at 947c2749 the
    # reference ALWAYS fails for fp16 weight + strength != 1 (the
    # strength interpolation promotes bi to float32 via the
    # intermediate-dtype eye, then einsum rejects the fp32/fp16 mix;
    # logged + weight silently unpatched, verified on torch 2.9.1).
    # Pinned as an explicit loud-failure test instead of a golden.
    case(
        "boft_fp16",
        BOFTAdapter,
        (blocks, None, 0.0, None),
        t(8, 6, dtype=torch.float16),
        strength=1.0,
        slots=OFT_SLOTS,
    )
    case(
        "boft_dora",
        BOFTAdapter,
        (blocks, None, 0.0, t(8, 1, scale=0.5).abs() + 0.5),
        w,
        strength=0.5,
        slots=OFT_SLOTS,
    )


def main() -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO.parent / "ComfyUI"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    gen_lora()
    gen_loha()
    gen_lokr()
    gen_glora()
    gen_oft()
    gen_boft()
    out = {
        "_meta": {
            "constraint_contract_cases": CONSTRAINT_CONTRACT_CASES,
            "reference_commit": commit,
            "torch": torch.__version__,
            "generator": "tools/gen_adapter_goldens.py",
        },
        "cases": CASES,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(CASES)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
