"""Generate patch-application goldens from the ComfyUI reference.

Runs the REFERENCE comfy/lora.py calculate_weight, the reference
weight_adapter loads, comfy/float.py stochastic_rounding (manual
path) and comfy/utils.py string_to_seed @ the audited baseline, and
writes packages/dinkster-inference-torch/tests/goldens/
patch_goldens.json. dinkster_inference_torch.apply / .materialize /
.rounding are pinned against these outputs - the oracle is the
reference code itself. The constrained OFT case passes the trainer's
effective ``alpha * out_dim`` bound to that reference while recording
the raw alpha stored in the adapter file.

Rounding cases come in two flavors matching the two upstream paths:
``rounding_cases`` pin the manual torch path (comfy-kitchen kernel
force-disabled while generating them), ``rounding_kitchen_cases`` pin
the comfy-kitchen accelerated path (kernel re-enabled; requires
comfy_kitchen importable, e.g. PYTHONPATH=../comfy-kitchen). The two
paths draw randomness differently and are not bit-equal even
upstream - but the accelerated path's seeded uint8 ``torch.randint``
stream IS byte-identical across the torch builds Dinkster spans
(verified 2.9.1 vs 2.13.0), so kitchen cases replay bit-exact on any
build, unlike the manual float16 ``torch.rand`` cases.

``quant_cases`` pin comfy/quant_ops.py _TensorCoreFP8LayoutBase
quantize (scale="recalculate", inplace_ops=True; nearest and
stochastic) + kitchen dequantize_per_tensor_fp8 - the scaled-fp8
storage semantics Dinkster ports in quant.py.

Input tensors are stored IN the golden file, so the pin does not
depend on RNG stability across torch versions. Stochastic-rounding
cases DO execute torch.rand at replay time with a pinned seed - and
torch.rand(dtype=float16) on CPU verifiably changed streams between
torch 2.9 (this generator's venv) and 2.13 (the test venv), so the
test replays them bitwise only on the recorded torch build and
otherwise checks the exact adjacent-fp8-grid contract (see
test_patches.py::test_rounding_golden).

Usage (needs a torch interpreter that imports the pinned checkout;
the workspace root venv is deliberately torch-free):

    PYTHONPATH=../ComfyUI /path/to/torch-venv/bin/python tools/gen_patch_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import comfy_aimdo
import torch

# The reference venv may carry a comfy_aimdo predating the pinned
# checkout's comfy_aimdo.host_buffer / .vram_buffer modules. They are
# imported at module scope by comfy/memory_management.py and
# comfy/model_management.py but only CALLED inside disk-read /
# cast-buffer helpers no golden case touches; stub them rather than
# mutate the live venv.
for _sub in ("host_buffer", "vram_buffer"):
    _name = f"comfy_aimdo.{_sub}"
    try:
        __import__(_name)
    except ModuleNotFoundError:
        _stub = types.ModuleType(_name)
        setattr(comfy_aimdo, _sub, _stub)
        sys.modules[_name] = _stub

import comfy.float  # noqa: E402
import comfy.quant_ops  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.lora import calculate_weight  # noqa: E402
from comfy.weight_adapter.loha import LoHaAdapter  # noqa: E402
from comfy.weight_adapter.lokr import LoKrAdapter  # noqa: E402
from comfy.weight_adapter.lora import LoRAAdapter  # noqa: E402
from comfy.weight_adapter.oft import OFTAdapter  # noqa: E402

# The kitchen kernel must have imported for the accelerated cases;
# manual cases toggle it off temporarily (see module docstring).
if not comfy.float._CK_STOCHASTIC_ROUNDING_AVAILABLE:
    raise RuntimeError(
        "comfy_kitchen.stochastic_rounding_fp8 unavailable - run with"
        " PYTHONPATH including ../comfy-kitchen"
    )

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "patch_goldens.json"

_gen = torch.Generator().manual_seed(0x4B2)


def t(*shape: int, dtype: torch.dtype = torch.float32, scale: float = 1.0):
    x = torch.randn(shape, generator=_gen, dtype=torch.float32) * scale
    return x.to(dtype)


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


def make_lora(up: torch.Tensor, down: torch.Tensor, alpha, dora_scale=None):
    """A reference LoRAAdapter around a bare weights tuple (bypasses
    the train-mode __init__, like gen_adapter_goldens.py)."""
    ad = object.__new__(LoRAAdapter)
    ad.weights = (up, down, alpha, None, dora_scale, None)
    ad.loaded_keys = set()
    return ad


# Named pinned callables shared with the Dinkster replay side
# (test_patches.py FUNCTIONS/CONVERTS): entry "function" values are
# the per-entry delta hook (patch tuple slot 4), nested "convert"
# values the donor convert_func. Goldens reference them by name so
# the JSON stays plain data; the replay failing loudly is the pin
# that both registries agree.
FUNCTIONS = {
    "halve": lambda a: a * 0.5,
    "negate": lambda a: -a,
}
CONVERTS = {
    "double": lambda a, inplace=False: a.mul_(2.0) if inplace else a * 2.0,
}


# ---------------------------------------------------------------- apply

# Each entry descriptor is JSON the Dinkster test rebuilds typed entries
# from; here it also builds the reference patch tuple.

APPLY_CASES: list[dict] = []


def ref_value(value: dict):
    kind = value["kind"]
    if kind == "diff":
        diff = dec_local(value["diff"])
        if value["pad_weight"]:
            return ("diff", (diff, {"pad_weight": True}))
        return (diff,)
    if kind == "set":
        return ("set", (dec_local(value["value"]),))
    if kind == "model_as_lora":
        return ("model_as_lora", (dec_local(value["target"]),))
    if kind == "lora_adapter":
        dora = value.get("dora_scale")
        return make_lora(
            dec_local(value["up"]),
            dec_local(value["down"]),
            value["alpha"],
            dora_scale=None if dora is None else dec_local(dora),
        )
    if kind == "nested":
        inner = [ref_entry(e) for e in value["entries"]]
        convert = CONVERTS.get(value.get("convert"), lambda a, **kw: a)
        return [(dec_local(value["base"]), convert)] + inner
    raise ValueError(kind)


def ref_entry(entry: dict):
    offset = entry["offset"]
    return (
        entry["strength"],
        ref_value(entry["value"]),
        entry["strength_model"],
        None if offset is None else tuple(offset),
        FUNCTIONS.get(entry.get("function")),
    )


def dec_local(spec: dict) -> torch.Tensor:
    data = torch.tensor(spec["data"], dtype=torch.float32)
    return data.reshape(spec["shape"]).to(getattr(torch, spec["dtype"]))


def apply_case(
    name: str,
    weight: torch.Tensor,
    entries: list[dict],
    *,
    original: torch.Tensor | None = None,
) -> None:
    key = f"model.{name}.weight"
    original_weights = None
    if original is not None:
        original_weights = {key: [(original, None)]}
    result = calculate_weight(
        [ref_entry(e) for e in entries],
        weight.clone(),
        key,
        intermediate_dtype=torch.float32,
        original_weights=original_weights,
    )
    if not torch.isfinite(result.to(torch.float32)).all():
        raise RuntimeError(f"{name}: non-finite reference output")
    if torch.equal(result.to(torch.float32), weight.to(torch.float32)):
        raise RuntimeError(f"{name}: reference left the weight unpatched")
    APPLY_CASES.append(
        {
            "name": name,
            "key": key,
            "weight": enc(weight),
            "original": None if original is None else enc(original),
            "entries": entries,
            "expected": enc(result),
        }
    )


def e_diff(
    diff,
    *,
    strength=1.0,
    strength_model=1.0,
    offset=None,
    pad=False,
    function=None,
):
    entry = {
        "strength": strength,
        "strength_model": strength_model,
        "offset": offset,
        "value": {"kind": "diff", "diff": enc(diff), "pad_weight": pad},
    }
    if function is not None:
        entry["function"] = function
    return entry


def e_lora(
    up,
    down,
    alpha,
    *,
    strength=1.0,
    strength_model=1.0,
    offset=None,
    dora_scale=None,
    function=None,
):
    entry = {
        "strength": strength,
        "strength_model": strength_model,
        "offset": offset,
        "value": {
            "kind": "lora_adapter",
            "up": enc(up),
            "down": enc(down),
            "alpha": alpha,
        },
    }
    if dora_scale is not None:
        entry["value"]["dora_scale"] = enc(dora_scale)
    if function is not None:
        entry["function"] = function
    return entry


def gen_apply() -> None:
    w = t(8, 6)
    d = t(8, 6, scale=0.3)

    apply_case("diff_full", w, [e_diff(d)])
    apply_case("diff_partial_strength", w, [e_diff(d, strength=0.55)])
    apply_case(
        "diff_pad_weight",
        t(6, 4),
        [e_diff(t(8, 6, scale=0.3), pad=True)],
    )
    apply_case(
        "set_replaces",
        w,
        [
            {
                "strength": 1.0,
                "strength_model": 1.0,
                "offset": None,
                "value": {"kind": "set", "value": enc(t(8, 6))},
            }
        ],
    )
    orig = t(8, 6)
    apply_case(
        "model_as_lora",
        orig,
        [
            {
                "strength": 0.65,
                "strength_model": 1.0,
                "offset": None,
                "value": {"kind": "model_as_lora", "target": enc(t(8, 6))},
            }
        ],
        original=orig,
    )
    apply_case(
        "lora_entry",
        w,
        [e_lora(t(8, 3, scale=0.4), t(3, 6, scale=0.4), 1.5, strength=0.8)],
    )
    apply_case(
        "strength_model_scales_base",
        w,
        [e_diff(d, strength=0.5, strength_model=0.7)],
    )
    apply_case(
        "offset_diff_window",
        w,
        [e_diff(t(3, 6, scale=0.3), strength=0.9, offset=[0, 2, 3])],
    )
    apply_case(
        "offset_lora_window",
        w,
        [
            e_lora(
                t(4, 2, scale=0.4),
                t(2, 6, scale=0.4),
                None,
                strength=0.6,
                offset=[0, 4, 4],
            )
        ],
    )
    apply_case(
        "sequential_entries",
        w,
        [
            e_diff(d, strength=0.4),
            e_lora(t(8, 3, scale=0.4), t(3, 6, scale=0.4), 2.0, strength=0.7),
            e_diff(t(8, 6, scale=0.2), strength=1.0),
        ],
    )
    apply_case(
        "nested_donor_model",
        w,
        [
            {
                "strength": 0.7,
                "strength_model": 1.0,
                "offset": None,
                "value": {
                    "kind": "nested",
                    "base": enc(t(8, 6, scale=0.1)),
                    "entries": [
                        e_diff(t(8, 6, scale=0.2), strength=1.0),
                        e_lora(
                            t(8, 3, scale=0.3),
                            t(3, 6, scale=0.3),
                            1.0,
                            strength=0.5,
                        ),
                    ],
                },
            }
        ],
    )
    apply_case(
        "fp16_weight_lora",
        t(8, 6, dtype=torch.float16),
        [e_lora(t(8, 3, scale=0.4), t(3, 6, scale=0.4), 1.0, strength=0.5)],
    )

    # ---- per-entry delta hook (patch tuple slot 4) + nested convert.
    # These cases draw from a DEDICATED generator: gen_materialize /
    # gen_rounding continue the shared stream after gen_apply, so any
    # extra shared-stream draw here would silently churn every later
    # case's recorded inputs.
    global _gen
    _saved_gen = _gen
    _gen = torch.Generator().manual_seed(0x4B3)
    apply_case(
        "diff_with_function",
        w,
        [e_diff(d, strength=0.8, function="halve")],
    )
    apply_case(
        "model_as_lora_with_function",
        orig,
        [
            {
                "strength": 0.7,
                "strength_model": 1.0,
                "offset": None,
                "function": "negate",
                "value": {"kind": "model_as_lora", "target": enc(t(8, 6))},
            }
        ],
        original=orig,
    )
    apply_case(
        "lora_with_function",
        w,
        [
            e_lora(
                t(8, 3, scale=0.4),
                t(3, 6, scale=0.4),
                1.25,
                strength=0.9,
                function="halve",
            )
        ],
    )
    apply_case(
        "dora_with_function",
        w,
        [
            e_lora(
                t(8, 3, scale=0.4),
                t(3, 6, scale=0.4),
                1.0,
                strength=0.6,
                dora_scale=t(8, 1, scale=0.1) + 1.0,
                function="halve",
            )
        ],
    )
    apply_case(
        "nested_with_convert",
        w,
        [
            {
                "strength": 0.5,
                "strength_model": 1.0,
                "offset": None,
                "value": {
                    "kind": "nested",
                    "base": enc(t(8, 6, scale=0.1)),
                    "convert": "double",
                    "entries": [e_diff(t(8, 6, scale=0.2), strength=0.8)],
                },
            }
        ],
    )
    _gen = _saved_gen


# ---------------------------------------------------------- materialize

# Mini "lora files" loaded by the REFERENCE loads (comfy/lora.py
# load_lora alpha .item() + weight_adapter .load), then applied. The
# Dinkster test builds the matching stage-4a spec by hand and goes
# through materialize_adapter - pinning the scalar-read semantics.

MATERIALIZE_CASES: list[dict] = []
CONSTRAINT_CONTRACT_CASES = ["oft_alpha_constraint"]


def materialize_case(
    name: str,
    adapter_cls,
    stem: str,
    file: dict[str, torch.Tensor],
    spec: dict,
    weight: torch.Tensor,
    *,
    strength: float,
    constraint_out_dim: int | None = None,
) -> None:
    alpha = None
    alpha_name = f"{stem}.alpha"
    if alpha_name in file:
        alpha = file[alpha_name].item()
    dora_scale = file.get(f"{stem}.dora_scale")
    adapter = adapter_cls.load(stem, dict(file), alpha, dora_scale, set())
    if adapter is None:
        raise RuntimeError(f"{name}: reference load refused the file")
    if constraint_out_dim is not None:
        blocks, rescale, raw_alpha, loaded_dora_scale = adapter.weights
        if not isinstance(raw_alpha, int | float):
            raise TypeError(f"{name}: constraint alpha must be numeric")
        adapter.weights = (
            blocks,
            rescale,
            raw_alpha * constraint_out_dim,
            loaded_dora_scale,
        )
    result = adapter.calculate_weight(
        weight.clone(),
        key=name,
        strength=strength,
        strength_model=1.0,
        offset=None,
        function=lambda x: x,
        intermediate_dtype=torch.float32,
    )
    if result is None or torch.equal(result.to(torch.float32), weight.to(torch.float32)):
        raise RuntimeError(f"{name}: reference did not patch")
    MATERIALIZE_CASES.append(
        {
            "name": name,
            "adapter": adapter_cls.__name__,
            "spec": spec,
            "file": {k: enc(v) for k, v in file.items()},
            "weight": enc(weight),
            "strength": strength,
            "expected": enc(result),
        }
    )


def gen_materialize() -> None:
    w = t(8, 6)

    up, down, alpha = t(8, 3, scale=0.4), t(3, 6, scale=0.4), torch.tensor(1.75)
    materialize_case(
        "lora_kohya_alpha",
        LoRAAdapter,
        "m",
        {"m.lora_up.weight": up, "m.lora_down.weight": down, "m.alpha": alpha},
        {
            "kind": "lora",
            "up": "m.lora_up.weight",
            "down": "m.lora_down.weight",
            "alpha": "m.alpha",
        },
        w,
        strength=0.9,
    )

    # reshape_weight read via .tolist() inside the reference load
    materialize_case(
        "lora_reshape",
        LoRAAdapter,
        "m",
        {
            "m.lora_up.weight": t(8, 2, scale=0.4),
            "m.lora_down.weight": t(2, 6, scale=0.4),
            "m.reshape_weight": torch.tensor([8, 6]),
        },
        {
            "kind": "lora",
            "up": "m.lora_up.weight",
            "down": "m.lora_down.weight",
            "reshape": "m.reshape_weight",
        },
        t(6, 4),
        strength=1.0,
    )

    materialize_case(
        "loha_alpha",
        LoHaAdapter,
        "m",
        {
            "m.hada_w1_a": t(8, 3, scale=0.4),
            "m.hada_w1_b": t(3, 6, scale=0.4),
            "m.hada_w2_a": t(8, 3, scale=0.4),
            "m.hada_w2_b": t(3, 6, scale=0.4),
            "m.alpha": torch.tensor(2.5),
        },
        {
            "kind": "loha",
            "w1_a": "m.hada_w1_a",
            "w1_b": "m.hada_w1_b",
            "w2_a": "m.hada_w2_a",
            "w2_b": "m.hada_w2_b",
            "alpha": "m.alpha",
        },
        w,
        strength=0.8,
    )

    materialize_case(
        "lokr_decomposed_alpha",
        LoKrAdapter,
        "m",
        {
            "m.lokr_w1_a": t(2, 2, scale=0.4),
            "m.lokr_w1_b": t(2, 4, scale=0.4),
            "m.lokr_w2_a": t(4, 3, scale=0.4),
            "m.lokr_w2_b": t(3, 3, scale=0.4),
            "m.alpha": torch.tensor(1.25),
        },
        {
            "kind": "lokr",
            "w1_a": "m.lokr_w1_a",
            "w1_b": "m.lokr_w1_b",
            "w2_a": "m.lokr_w2_a",
            "w2_b": "m.lokr_w2_b",
            "alpha": "m.alpha",
        },
        t(8, 12),
        strength=1.0,
    )

    oft_blocks = t(2, 4, 4, scale=0.1)
    materialize_case(
        "oft_alpha_constraint",
        OFTAdapter,
        "m",
        {
            "m.oft_blocks": oft_blocks,
            "m.alpha": torch.tensor(0.05),
        },
        {"kind": "oft", "blocks": "m.oft_blocks", "alpha": "m.alpha"},
        w,
        strength=1.0,
        constraint_out_dim=oft_blocks.shape[0] * oft_blocks.shape[1],
    )


# -------------------------------------------------------------- rounding

ROUNDING_CASES: list[dict] = []
ROUNDING_KITCHEN_CASES: list[dict] = []
SEED_CASES: list[dict] = []


def gen_rounding() -> None:
    for key in [
        "diffusion_model.input_blocks.0.0.weight",
        "diffusion_model.output_blocks.11.1.bias",
        "m",
    ]:
        SEED_CASES.append({"key": key, "seed": comfy.utils.string_to_seed(key)})

    # Manual torch path: kitchen kernel force-disabled while these run.
    comfy.float._CK_STOCHASTIC_ROUNDING_AVAILABLE = False
    for name, dtype in [
        ("e4m3fn", torch.float8_e4m3fn),
        ("e5m2", torch.float8_e5m2),
    ]:
        for seed_key in ["m", "diffusion_model.input_blocks.0.0.weight"]:
            value = t(8, 6, scale=0.5)
            seed = comfy.utils.string_to_seed(seed_key)
            result = comfy.float.stochastic_rounding(value.clone(), dtype, seed=seed)
            assert result.dtype == dtype
            ROUNDING_CASES.append(
                {
                    "name": f"round_{name}_{seed_key.split('.')[0]}",
                    "value": enc(value),
                    "dtype": str(dtype).removeprefix("torch."),
                    "seed": seed,
                    "expected": enc(result),
                }
            )
    # non-fp8 targets are plain casts
    value = t(4, 4)
    result = comfy.float.stochastic_rounding(value.clone(), torch.float16, seed=7)
    ROUNDING_CASES.append(
        {
            "name": "round_fp16_plain_cast",
            "value": enc(value),
            "dtype": "float16",
            "seed": 7,
            "expected": enc(result),
        }
    )
    comfy.float._CK_STOCHASTIC_ROUNDING_AVAILABLE = True

    # Accelerated kitchen path: seeded uint8 randint stream, byte-
    # identical across torch builds -> these replay bit-exact anywhere
    # comfy-kitchen is installed.
    for name, dtype in [
        ("e4m3fn", torch.float8_e4m3fn),
        ("e5m2", torch.float8_e5m2),
    ]:
        for seed_key in ["m", "diffusion_model.input_blocks.0.0.weight"]:
            value = t(8, 6, scale=0.5)
            seed = comfy.utils.string_to_seed(seed_key)
            result = comfy.float.stochastic_rounding(value.clone(), dtype, seed=seed)
            assert result.dtype == dtype
            ROUNDING_KITCHEN_CASES.append(
                {
                    "name": f"ck_round_{name}_{seed_key.split('.')[0]}",
                    "value": enc(value),
                    "dtype": str(dtype).removeprefix("torch."),
                    "seed": seed,
                    "expected": enc(result),
                }
            )


# ------------------------------------------------------------------ quant

QUANT_CASES: list[dict] = []

_FP8_LAYOUTS = {
    "float8_e4m3fn": comfy.quant_ops.TensorCoreFP8E4M3Layout,
    "float8_e5m2": comfy.quant_ops.TensorCoreFP8E5M2Layout,
}


def quant_case(
    name: str,
    tensor: torch.Tensor,
    dtype_name: str,
    *,
    seed: int,
) -> None:
    """Reference scaled-fp8 storage semantics: quantize with a
    recalculated per-tensor scale (nearest when seed == 0, seeded
    stochastic otherwise) + dequantize back to the source dtype."""
    layout = _FP8_LAYOUTS[dtype_name]
    qdata, params = layout.quantize(
        tensor.clone(),
        scale="recalculate",
        stochastic_rounding=seed,
        inplace_ops=True,
    )
    import comfy_kitchen as ck

    dequant = ck.dequantize_per_tensor_fp8(qdata, params.scale, tensor.dtype)
    QUANT_CASES.append(
        {
            "name": name,
            "value": enc(tensor),
            "dtype": dtype_name,
            "seed": seed,
            "expected_qdata": enc(qdata),
            "expected_scale": params.scale.item(),
            "expected_dequant": enc(dequant),
        }
    )


def gen_quant() -> None:
    quant_case("q_e4m3_nearest", t(8, 6, scale=2.0), "float8_e4m3fn", seed=0)
    quant_case("q_e5m2_nearest", t(8, 6, scale=2.0), "float8_e5m2", seed=0)
    quant_case(
        "q_e4m3_stochastic",
        t(8, 6, scale=2.0),
        "float8_e4m3fn",
        seed=comfy.utils.string_to_seed("model.q.weight"),
    )
    # fp16 source exercises the too-small-scale correction branch
    quant_case(
        "q_e4m3_fp16_source",
        t(8, 6, dtype=torch.float16, scale=0.05),
        "float8_e4m3fn",
        seed=0,
    )


def main() -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO.parent / "ComfyUI"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    gen_apply()
    gen_materialize()
    gen_rounding()
    gen_quant()
    out = {
        "_meta": {
            "constraint_contract_cases": CONSTRAINT_CONTRACT_CASES,
            "reference_commit": commit,
            "torch": torch.__version__,
            "generator": "tools/gen_patch_goldens.py",
        },
        "apply_cases": APPLY_CASES,
        "materialize_cases": MATERIALIZE_CASES,
        "rounding_cases": ROUNDING_CASES,
        "rounding_kitchen_cases": ROUNDING_KITCHEN_CASES,
        "quant_cases": QUANT_CASES,
        "seed_cases": SEED_CASES,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    n = (
        len(APPLY_CASES)
        + len(MATERIALIZE_CASES)
        + len(ROUNDING_CASES)
        + len(ROUNDING_KITCHEN_CASES)
        + len(QUANT_CASES)
    )
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes, {n} cases)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
