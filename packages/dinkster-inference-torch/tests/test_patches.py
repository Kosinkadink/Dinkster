"""Materialization and PatchSet application vs executed references.

Recorded inputs come from running the reference @ 947c2749 through
tools/gen_patch_goldens.py: comfy/lora.py calculate_weight for apply cases,
weight_adapter loads for materialize cases, comfy/float.py stochastic_rounding
(manual torch path), and comfy/utils.py string_to_seed for rounding cases. The
constrained OFT expectation uses the trainer's ``alpha * out_dim`` contract.
Other cases require numeric agreement; stochastic-rounding cases require bit
agreement with the same seed and RNG draw order.

Deliberate loud deviations (documented in apply.py / materialize.py)
are tested separately.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import Any

import dinkster_inference_torch.rounding as rounding_mod
import pytest
import torch
from dinkster_inference.lora import (
    BOFTSpec,
    DiffPatchRef,
    LoHaSpec,
    LoKrSpec,
    LoRASpec,
    OFTSpec,
    PatchTarget,
    SetPatchRef,
)
from dinkster_inference.patches import (
    AdapterPatch,
    DiffPatch,
    ModelAsLoraPatch,
    NestedPatch,
    PatchEntry,
    PatchOffset,
    PatchPayloadError,
    PatchSet,
    SetPatch,
    WeightConverter,
    patch_payloads,
    rebuild_patch_entries,
)
from dinkster_inference_torch import (
    Fp8ScaledWeight,
    LoRAAdapter,
    MaterializeError,
    PatchApplyError,
    apply_patches,
    build_patch_set,
    materialize_adapter,
    materialize_value,
    patch_weights,
    quantize_fp8_scaled,
    requantize_fp8_scaled,
    restore_weights,
    stochastic_rounding,
    string_to_seed,
)
from dinkster_inference_torch import quant_linear as quant_linear_mod
from dinkster_inference_torch.quant import (
    Int8PackedWeight,
    Nvfp4PackedWeight,
    requantize_int8,
    requantize_nvfp4,
)
from golden_files import load_platform_golden

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "patch_goldens.json").read_text())
STACK_GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "lora_stack_goldens.json")


def dec(spec: dict[str, Any]) -> torch.Tensor:
    data = torch.tensor(spec["data"], dtype=torch.float32)
    return data.reshape(spec["shape"]).to(getattr(torch, spec["dtype"]))


# Named pinned callables mirroring tools/gen_patch_goldens.py
# FUNCTIONS/CONVERTS - golden entries reference them by name; a
# registry mismatch fails the replay loudly.
def _fn_halve(a: torch.Tensor) -> torch.Tensor:
    return a * 0.5


def _fn_negate(a: torch.Tensor) -> torch.Tensor:
    return -a


def _cv_double(a: torch.Tensor, /, *, inplace: bool = False) -> torch.Tensor:
    return a.mul_(2.0) if inplace else a * 2.0


FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "halve": _fn_halve,
    "negate": _fn_negate,
}
CONVERTS: dict[str, WeightConverter[torch.Tensor]] = {
    "double": _cv_double,
}


def build_value(value: dict[str, Any]):
    kind = value["kind"]
    if kind == "diff":
        return DiffPatch(dec(value["diff"]), pad_weight=value["pad_weight"])
    if kind == "set":
        return SetPatch(dec(value["value"]))
    if kind == "model_as_lora":
        return ModelAsLoraPatch(dec(value["target"]))
    if kind == "lora_adapter":
        dora = value.get("dora_scale")
        return AdapterPatch(
            LoRAAdapter(
                dec(value["up"]),
                dec(value["down"]),
                alpha=value["alpha"],
                dora_scale=None if dora is None else dec(dora),
            )
        )
    if kind == "nested":
        return NestedPatch(
            dec(value["base"]),
            tuple(build_entry(e) for e in value["entries"]),
            convert=CONVERTS.get(value.get("convert", "")),
        )
    raise AssertionError(f"unknown value kind {kind}")


def build_entry(entry: dict[str, Any]) -> PatchEntry[torch.Tensor]:
    offset = entry["offset"]
    return PatchEntry(
        build_value(entry["value"]),
        strength=entry["strength"],
        strength_model=entry["strength_model"],
        offset=None if offset is None else PatchOffset(*offset),
        function=FUNCTIONS.get(entry.get("function", "")),
    )


def test_patch_payload_walk_rebuilds_all_values_and_preserves_metadata() -> None:
    payloads = tuple(torch.full((2, 2), float(index)) for index in range(1, 8))
    nested_entry = PatchEntry(
        SetPatch(payloads[6]),
        strength=0.25,
        strength_model=0.75,
        offset=PatchOffset(0, 0, 1),
        function=_fn_halve,
    )
    entries = (
        PatchEntry(DiffPatch(payloads[0], pad_weight=True), strength=0.1),
        PatchEntry(SetPatch(payloads[1]), strength_model=0.2),
        PatchEntry(ModelAsLoraPatch(payloads[2]), function=_fn_negate),
        PatchEntry(AdapterPatch(LoRAAdapter(payloads[3], payloads[4]))),
        PatchEntry(NestedPatch(payloads[5], (nested_entry,), convert=_cv_double)),
    )
    assert patch_payloads(entries) == payloads

    replacements = tuple(payload.add(10) for payload in payloads)
    rebuilt = rebuild_patch_entries(entries, replacements)
    assert patch_payloads(rebuilt) == replacements
    for original, replacement in zip(entries, rebuilt, strict=True):
        assert replacement.strength == original.strength
        assert replacement.strength_model == original.strength_model
        assert replacement.offset is original.offset
        assert replacement.function is original.function
    rebuilt_nested = rebuilt[-1].value
    assert isinstance(rebuilt_nested, NestedPatch)
    assert rebuilt_nested.convert is _cv_double
    assert rebuilt_nested.entries[0].offset is nested_entry.offset
    assert rebuilt_nested.entries[0].function is _fn_halve

    with pytest.raises(PatchPayloadError, match="too few"):
        rebuild_patch_entries(entries, replacements[:-1])
    with pytest.raises(PatchPayloadError, match="too many"):
        rebuild_patch_entries(entries, (*replacements, torch.ones(1)))


def assert_matches(result: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    assert result.dtype == expected.dtype, name
    assert tuple(result.shape) == tuple(expected.shape), name
    if expected.dtype == torch.float16:
        rtol, atol = 1e-2, 1e-3
    else:
        rtol, atol = 1e-5, 1e-6
    assert torch.allclose(result.float(), expected.float(), rtol=rtol, atol=atol), (
        name,
        (result.float() - expected.float()).abs().max().item(),
    )


# ------------------------------------------------------- apply goldens


@pytest.mark.parametrize(
    "case",
    GOLDENS["apply_cases"],
    ids=[c["name"] for c in GOLDENS["apply_cases"]],
)
def test_apply_golden(case: dict[str, Any]) -> None:
    weight = dec(case["weight"])
    original = None if case["original"] is None else dec(case["original"])
    result = apply_patches(
        weight.clone(),
        [build_entry(e) for e in case["entries"]],
        key=case["key"],
        intermediate_dtype=torch.float32,
        original_weight=original,
    )
    assert_matches(result, dec(case["expected"]), case["name"])


@pytest.mark.parametrize(
    "case",
    STACK_GOLDENS["apply_cases"],
    ids=[case["name"] for case in STACK_GOLDENS["apply_cases"]],
)
def test_lora_stack_matches_comfyui_golden(case: dict[str, Any]) -> None:
    assert STACK_GOLDENS["_meta"]["generator"] == "tools/gen_lora_stack_goldens.py"
    assert STACK_GOLDENS["_meta"]["reference_commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    if sys.platform.startswith("linux"):
        assert STACK_GOLDENS["_meta"] == {
            "generator": "tools/gen_lora_stack_goldens.py",
            "reference_commit": "b78cec879b9460d5cb25228a83a942fb78d2cd24",
            "torch": "2.13.0+cu130",
        }
    result = apply_patches(
        dec(case["weight"]).clone(),
        [build_entry(entry) for entry in case["entries"]],
        key=case["key"],
        intermediate_dtype=torch.float32,
    )
    expected = dec(case["expected"])
    if case["name"] == "lora_stack_convolution":
        # One audited CPU provider changed 1/180 values by one float32 ULP
        # (2.9802322e-08). Same-provider ComfyUI and Dinkster were bit-exact;
        # CPU GEMM operation ordering explains the drift. This adds no headroom.
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype == torch.float32
        assert torch.all(torch.isfinite(result))
        assert torch.all(torch.isfinite(expected))
        different = result != expected
        assert torch.count_nonzero(different).item() <= 1
        if torch.any(different):
            assert torch.equal(
                torch.nextafter(expected[different], result[different]),
                result[different],
            )
    else:
        assert torch.equal(result, expected), case["name"]


def _assert_composition_delta_is_eps_scale(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    # Composition order and strength splitting stay visible in the
    # reference execution, but only at float32-eps scale: the max
    # absolute delta is exactly 2**-23 on every audited reference
    # platform (linux, darwin, win32). Which elements move, and by how
    # many ULPs for magnitudes below 1.0, is a platform kernel
    # incidental, so element counts are not pinned.
    assert (lhs - rhs).abs().max().item() == 1.1920928955078125e-07


def test_comfyui_lora_stack_golden_pins_order_and_strength_composition() -> None:
    expected = {case["name"]: dec(case["expected"]) for case in STACK_GOLDENS["apply_cases"]}
    _assert_composition_delta_is_eps_scale(
        expected["lora_stack_linear"], expected["lora_stack_linear_reversed"]
    )
    _assert_composition_delta_is_eps_scale(
        expected["lora_stack_repeated_fractional"], expected["lora_stack_repeated_combined"]
    )


# ------------------------------------------------- materialize goldens


def build_spec(spec: dict[str, Any]):
    kind = spec.pop("kind")
    if kind == "lora":
        return LoRASpec(**spec)
    if kind == "loha":
        return LoHaSpec(**spec)
    if kind == "lokr":
        return LoKrSpec(**spec)
    if kind == "oft":
        return OFTSpec(**spec)
    raise AssertionError(f"unknown spec kind {kind}")


@pytest.mark.parametrize(
    "case",
    GOLDENS["materialize_cases"],
    ids=[c["name"] for c in GOLDENS["materialize_cases"]],
)
def test_materialize_golden(case: dict[str, Any]) -> None:
    tensors = {k: dec(v) for k, v in case["file"].items()}
    adapter = materialize_adapter(build_spec(dict(case["spec"])), tensors)
    result = adapter.calculate(dec(case["weight"]).clone(), strength=case["strength"])
    assert_matches(result, dec(case["expected"]), case["name"])


# --------------------------------------------------- rounding goldens


@pytest.mark.parametrize(
    "case",
    GOLDENS["seed_cases"],
    ids=[c["key"] for c in GOLDENS["seed_cases"]],
)
def test_string_to_seed_golden(case: dict[str, Any]) -> None:
    assert string_to_seed(case["key"]) == case["seed"]


def fp8_grid(dtype: torch.dtype) -> torch.Tensor:
    """All finite representable values of an fp8 dtype, sorted."""
    vals = torch.arange(256, dtype=torch.uint8).view(dtype).float()
    return torch.unique(vals[torch.isfinite(vals)]).sort().values


def assert_on_adjacent_grid(result: torch.Tensor, value: torch.Tensor, dtype: torch.dtype) -> None:
    """Every element must be one of the two fp8 grid neighbors of the
    half-cast input - the exact contract of stochastic rounding,
    independent of which neighbor the RNG picked."""
    grid = fp8_grid(dtype)
    inf = torch.finfo(dtype)
    x = value.half().float().clamp(inf.min, inf.max).flatten()
    r = result.float().flatten()
    hi = torch.searchsorted(grid, x, right=False).clamp(0, len(grid) - 1)
    lo = (hi - 1).clamp(0)
    ok = (r == grid[hi]) | (r == grid[lo]) | (r == x)
    assert bool(ok.all()), (r[~ok], x[~ok])


@pytest.mark.parametrize(
    "case",
    GOLDENS["rounding_cases"],
    ids=[c["name"] for c in GOLDENS["rounding_cases"]],
)
def test_rounding_golden_manual(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual-path pin (kitchen kernel disabled for the call): bit
    agreement with the reference manual path when running the same
    torch build the golden was generated on. torch.rand(dtype=
    float16) changed its CPU RNG stream between torch 2.9 and 2.13
    (verified on this workspace's two venvs), so on other builds the
    draws differ and the check degrades to the exact semantic
    contract: dtype + every element lands on an adjacent fp8 grid
    point of the half-cast input. The port itself was verified
    bit-equal to comfy/float.py @ 947c2749 on the reference
    interpreter (torch 2.9.1). A tolerance comparison would be
    meaningless - stochastic up-vs-down rounding differs by a whole
    fp8 ulp."""
    monkeypatch.setattr(rounding_mod, "_ck_stochastic_rounding_fp8", None)
    dtype = getattr(torch, case["dtype"])
    value = dec(case["value"])
    result = stochastic_rounding(value, dtype, seed=case["seed"])
    assert result.dtype == dtype
    if torch.__version__ == GOLDENS["_meta"]["torch"]:
        assert torch.equal(result.float(), dec(case["expected"]).float()), case["name"]
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        assert_on_adjacent_grid(result, value, dtype)
        assert_on_adjacent_grid(dec(case["expected"]), value, dtype)
    else:
        # non-fp8 targets are deterministic plain casts on any build
        assert torch.equal(result.float(), dec(case["expected"]).float()), case["name"]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_rounding_boundary_values_stay_on_grid(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression pin for Dinkster's DELIBERATE DIVERGENCE from the
    reference manual path (docs/comfyui-issues/
    stochastic-rounding-fp16-log2-boundary.md): for inputs just below
    a power of two, the reference's fp16 log2 rounds up to the
    integer, the exponent comes out one too high, and low RNG draws
    land one fp8 ulp BELOW the correct lower neighbor - off the
    adjacent grid. Dinkster's frexp exponent keeps every draw on the
    grid. Enough repeated draws that the old code fails with
    near-certainty (each bad element misrounds with p ~ 2^-8 per
    draw; 4096 draws per dtype)."""
    monkeypatch.setattr(rounding_mod, "_ck_stochastic_rounding_fp8", None)
    boundary = [7.99614334, -0.249910235, 0.0624428205, -0.0312358364]
    value = torch.tensor(boundary * 128)
    for seed in range(8):
        result = stochastic_rounding(value, dtype, seed=seed)
        assert_on_adjacent_grid(result, value, dtype)


kitchen_available = (
    rounding_mod._probe_kitchen_fp8_kernel()  # pyright: ignore[reportPrivateUsage]
    is not None
)


@pytest.mark.skipif(
    not kitchen_available,
    reason="comfy-kitchen not installed in this venv (see README)",
)
@pytest.mark.parametrize(
    "case",
    GOLDENS["rounding_kitchen_cases"],
    ids=[c["name"] for c in GOLDENS["rounding_kitchen_cases"]],
)
def test_rounding_golden_kitchen(case: dict[str, Any]) -> None:
    """Accelerated-path pin: BIT agreement with the reference kitchen
    path on ANY torch build - the seeded uint8 torch.randint stream is
    byte-identical across the builds Dinkster spans (verified 2.9.1 vs
    2.13.0) and the kitchen kernel itself is deterministic given the
    rng tensor."""
    dtype = getattr(torch, case["dtype"])
    value = dec(case["value"])
    result = stochastic_rounding(value, dtype, seed=case["seed"])
    assert result.dtype == dtype
    assert torch.equal(result.float(), dec(case["expected"]).float()), case["name"]


def test_rounding_prefers_kitchen_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability probe honored: when the kernel attribute is set, fp8
    rounding routes through it (upstream prefers the accelerated path
    whenever comfy_kitchen exposes it)."""
    calls: list[tuple[torch.Size, torch.dtype]] = []

    def spy(value: torch.Tensor, rng: torch.Tensor, dtype: torch.dtype):
        assert rng.dtype == torch.uint8
        assert rng.shape == value.shape
        calls.append((value.shape, dtype))
        return torch.zeros_like(value, dtype=dtype)

    monkeypatch.setattr(rounding_mod, "_ck_stochastic_rounding_fp8", spy)
    out = stochastic_rounding(torch.randn(4, 3), torch.float8_e4m3fn, seed=9)
    assert calls == [(torch.Size((4, 3)), torch.float8_e4m3fn)]
    assert out.dtype == torch.float8_e4m3fn
    # non-fp8 targets never touch the kernel
    stochastic_rounding(torch.randn(4, 3), torch.float16, seed=9)
    assert len(calls) == 1


def test_training_import_defers_kitchen_backends_until_fp8_use() -> None:
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import dinkster_training_torch"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "comfy_kitchen" not in result.stderr


def test_kitchen_probe_runs_only_on_first_fp8_use(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    def probe() -> None:
        calls.append(None)
        return None

    monkeypatch.setattr(rounding_mod, "_probe_kitchen_fp8_kernel", probe)
    monkeypatch.setattr(
        rounding_mod,
        "_ck_stochastic_rounding_fp8",
        rounding_mod._UNPROBED,  # pyright: ignore[reportPrivateUsage]
    )
    stochastic_rounding(torch.randn(2, 2), torch.float16)
    assert calls == []
    stochastic_rounding(torch.randn(2, 2), torch.float8_e4m3fn)
    stochastic_rounding(torch.randn(2, 2), torch.float8_e4m3fn)
    assert calls == [None]


def test_kitchen_probe_runs_once_during_concurrent_first_fp8_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[None] = []
    first_probe = Event()
    second_probe = Event()
    release_probe = Event()

    def probe() -> None:
        calls.append(None)
        (first_probe if len(calls) == 1 else second_probe).set()
        assert release_probe.wait(timeout=5)
        return None

    monkeypatch.setattr(rounding_mod, "_probe_kitchen_fp8_kernel", probe)
    monkeypatch.setattr(
        rounding_mod,
        "_ck_stochastic_rounding_fp8",
        rounding_mod._UNPROBED,  # pyright: ignore[reportPrivateUsage]
    )
    threads = [
        Thread(
            target=rounding_mod._kitchen_fp8_kernel  # pyright: ignore[reportPrivateUsage]
        )
        for _ in range(2)
    ]
    threads[0].start()
    assert first_probe.wait(timeout=5)
    threads[1].start()
    try:
        assert not second_probe.wait(timeout=0.2)
    finally:
        release_probe.set()
        for thread in threads:
            thread.join(timeout=5)
    assert calls == [None]
    assert not any(thread.is_alive() for thread in threads)


def test_rounding_manual_fallback_without_kitchen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel absent -> the manual torch path still produces a valid
    stochastic rounding (grid-adjacent, seed-deterministic)."""
    monkeypatch.setattr(rounding_mod, "_ck_stochastic_rounding_fp8", None)
    value = torch.randn(8, 6) * 0.5
    a = stochastic_rounding(value, torch.float8_e4m3fn, seed=3)
    b = stochastic_rounding(value, torch.float8_e4m3fn, seed=3)
    assert torch.equal(a.float(), b.float())
    assert_on_adjacent_grid(a, value, torch.float8_e4m3fn)


def test_kitchen_probe_is_capability_based() -> None:
    """The probe keys off the kernel attribute existing, never a
    version: a comfy_kitchen build predating the kernel selects the
    manual path."""
    fake = types.ModuleType("comfy_kitchen")
    real = sys.modules.get("comfy_kitchen")
    sys.modules["comfy_kitchen"] = fake
    try:
        assert (
            rounding_mod._probe_kitchen_fp8_kernel()  # pyright: ignore[reportPrivateUsage]
            is None
        )
    finally:
        if real is not None:
            sys.modules["comfy_kitchen"] = real
        else:
            del sys.modules["comfy_kitchen"]


def test_golden_meta_pins_reference() -> None:
    assert GOLDENS["_meta"]["reference_commit"].startswith("b78cec87")
    assert GOLDENS["_meta"]["constraint_contract_cases"] == ["oft_alpha_constraint"]


# ---------------------------------------------- rounding behavior


def test_rounding_deterministic_per_seed() -> None:
    value = torch.randn(16, 8)
    a = stochastic_rounding(value, torch.float8_e4m3fn, seed=11)
    b = stochastic_rounding(value, torch.float8_e4m3fn, seed=11)
    c = stochastic_rounding(value, torch.float8_e4m3fn, seed=12)
    assert torch.equal(a.float(), b.float())
    assert not torch.equal(a.float(), c.float())


def test_rounding_is_stochastic_not_nearest() -> None:
    """The load-bearing property: fp8 writeback must NOT be a plain
    nearest cast, or small LoRA deltas vanish."""
    value = torch.randn(64, 32) * 0.5
    rounded = stochastic_rounding(value, torch.float8_e4m3fn, seed=5)
    nearest = value.half().to(torch.float8_e4m3fn)
    assert not torch.equal(rounded.float(), nearest.float())


def test_rounding_preserves_small_delta_in_expectation() -> None:
    """A delta far below one fp8 ulp survives on average across seeds:
    the reason stochastic rounding is load-bearing for patching
    quantized storage."""
    base = torch.full((1024,), 1.0)
    delta = 0.01  # 1.0's fp8 e4m3 neighbor is 1.0625: delta << 1 ulp
    patched = base + delta
    assert torch.equal(  # nearest rounding erases the LoRA entirely
        patched.half().to(torch.float8_e4m3fn).float(), base
    )
    means = torch.stack(
        [stochastic_rounding(patched, torch.float8_e4m3fn, seed=s).float().mean() for s in range(8)]
    )
    assert abs(means.mean().item() - (1.0 + delta)) < delta / 2


def test_rounding_plain_casts_for_non_fp8() -> None:
    value = torch.randn(4, 4)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        out = stochastic_rounding(value, dtype, seed=3)
        assert out.dtype == dtype
        assert torch.equal(out.float(), value.to(dtype).float())


# ------------------------------------------------ materialize behavior


def test_materialize_missing_key_raises() -> None:
    spec = LoRASpec(up="m.up", down="m.down", alpha="m.alpha")
    tensors = {"m.up": torch.zeros(8, 3), "m.down": torch.zeros(3, 6)}
    with pytest.raises(MaterializeError, match="m.alpha"):
        materialize_adapter(spec, tensors)


def test_materialize_reshape_dims_skips_tensor_read() -> None:
    """A spec with decode-time reshape_dims must not demand the
    reshape tensor from the source."""
    spec = LoRASpec(up="m.up", down="m.down", reshape="m.gone", reshape_dims=(8, 6))
    adapter = materialize_adapter(spec, {"m.up": torch.zeros(8, 3), "m.down": torch.zeros(3, 6)})
    assert isinstance(adapter, LoRAAdapter)
    assert adapter.reshape == (8, 6)


def test_materialize_value_diff_set_refs() -> None:
    tensors = {"d": torch.randn(2, 2), "s": torch.randn(2, 2)}
    diff = materialize_value(DiffPatchRef(key="d"), tensors)
    assert isinstance(diff, DiffPatch)
    assert diff.value is tensors["d"]
    replace = materialize_value(SetPatchRef(key="s"), tensors)
    assert isinstance(replace, SetPatch)
    assert replace.value is tensors["s"]


def test_materialize_boft_spec() -> None:
    from dinkster_inference_torch import BOFTAdapter

    adapter = materialize_adapter(
        BOFTSpec(blocks="m.blocks"), {"m.blocks": torch.zeros(2, 2, 4, 4)}
    )
    assert isinstance(adapter, BOFTAdapter)


def test_build_patch_set_groups_and_carries_offsets() -> None:
    offset = PatchOffset(0, 2, 3)
    patches = {
        PatchTarget(key="model.a.weight"): DiffPatchRef(key="d1"),
        PatchTarget(key="model.a.weight", offset=offset): DiffPatchRef(key="d2"),
        PatchTarget(key="model.b.weight"): SetPatchRef(key="s"),
    }
    tensors = {
        "d1": torch.randn(8, 6),
        "d2": torch.randn(3, 6),
        "s": torch.randn(4, 4),
    }
    patch_set = build_patch_set(patches, tensors, strength=0.5, strength_model=0.9)
    a_entries = patch_set.entries("model.a.weight")
    assert len(a_entries) == 2
    assert {e.offset for e in a_entries} == {None, offset}
    assert all(e.strength == 0.5 for e in a_entries)
    assert all(e.strength_model == 0.9 for e in a_entries)
    (b_entry,) = patch_set.entries("model.b.weight")
    assert isinstance(b_entry.value, SetPatch)


def test_build_patch_set_threads_declared_structural_digest() -> None:
    digest = "a" * 64
    patch_set = build_patch_set(
        {PatchTarget("weight"): DiffPatchRef("diff")},
        {"diff": torch.ones(1)},
        structural_digest=digest,
    )
    assert patch_set.structural_digest == digest


# ----------------------------------------------------- apply behavior


def test_apply_no_entries_returns_weight_unchanged() -> None:
    weight = torch.randn(4, 4)
    result = apply_patches(weight, [])
    assert result is weight


def test_apply_mutates_passed_tensor_like_reference() -> None:
    weight = torch.zeros(4, 4)
    result = apply_patches(weight, [PatchEntry(DiffPatch(torch.ones(4, 4)))])
    assert result is weight
    assert torch.equal(weight, torch.ones(4, 4))


def test_apply_diff_shape_mismatch_raises() -> None:
    with pytest.raises(PatchApplyError, match="does not match"):
        apply_patches(
            torch.zeros(4, 4),
            [PatchEntry(DiffPatch(torch.ones(2, 2)))],
            key="model.x.weight",
        )


def test_apply_model_as_lora_without_original_raises() -> None:
    with pytest.raises(PatchApplyError, match="original"):
        apply_patches(
            torch.zeros(4, 4),
            [PatchEntry(ModelAsLoraPatch(torch.ones(4, 4)))],
            key="model.x.weight",
        )


def test_apply_offset_only_touches_window() -> None:
    weight = torch.zeros(6, 4)
    apply_patches(
        weight,
        [PatchEntry(DiffPatch(torch.ones(2, 4)), offset=PatchOffset(0, 1, 2))],
    )
    assert torch.equal(weight[1:3], torch.ones(2, 4))
    assert torch.equal(weight[:1], torch.zeros(1, 4))
    assert torch.equal(weight[3:], torch.zeros(3, 4))


# --------------------------------------- patch_weights / backup-restore


def make_store() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(99)
    return {
        "model.a.weight": torch.randn(8, 6, generator=gen),
        "model.b.weight": torch.randn(8, 6, generator=gen).half(),
        "model.c.weight": torch.randn(4, 4, generator=gen),
    }


def diff_set(key: str, diff: torch.Tensor) -> PatchSet[torch.Tensor]:
    return PatchSet({key: (PatchEntry(DiffPatch(diff)),)})


def test_patch_weights_applies_and_backs_up() -> None:
    weights = make_store()
    originals = dict(weights)
    diff = torch.randn(8, 6, generator=torch.Generator().manual_seed(7))
    backup = patch_weights(weights, diff_set("model.a.weight", diff))

    assert set(backup) == {"model.a.weight"}
    # backup holds the exact original tensor object, untouched
    assert backup["model.a.weight"] is originals["model.a.weight"]
    assert torch.equal(weights["model.a.weight"], originals["model.a.weight"] + diff)
    # unpatched keys untouched
    assert weights["model.b.weight"] is originals["model.b.weight"]

    restore_weights(weights, backup)
    assert weights["model.a.weight"] is originals["model.a.weight"]


def test_patch_weights_backup_device_same_device_is_identity() -> None:
    """backup_device matching the store's device changes nothing: the
    backup holds the original object (Parameter registration intact)
    and restore puts it back untouched."""
    key = "model.a.weight"
    weights = make_store()
    original = torch.nn.Parameter(weights[key], requires_grad=False)
    weights[key] = original
    diff = torch.randn(8, 6, generator=torch.Generator().manual_seed(7))

    backup = patch_weights(weights, diff_set(key, diff), backup_device=torch.device("cpu"))
    assert backup[key] is original
    assert torch.equal(weights[key], original + diff)

    restore_weights(weights, backup)
    assert weights[key] is original


def test_patch_weights_writes_back_storage_dtype() -> None:
    weights = make_store()
    diff = torch.randn(8, 6) * 0.1
    patch_weights(weights, diff_set("model.b.weight", diff))
    assert weights["model.b.weight"].dtype == torch.float16


def test_patch_weights_fp8_uses_seeded_stochastic_rounding() -> None:
    key = "model.q.weight"
    original = (torch.randn(16, 8) * 0.5).to(torch.float8_e4m3fn)
    weights = {key: original}
    diff = torch.randn(16, 8) * 0.05
    patch_weights(weights, diff_set(key, diff))

    out = weights[key]
    assert out.dtype == torch.float8_e4m3fn
    expected = stochastic_rounding(
        apply_patches(
            original.to(dtype=torch.float32, copy=True),
            [PatchEntry(DiffPatch(diff))],
            key=key,
        ),
        torch.float8_e4m3fn,
        seed=string_to_seed(key),
    )
    assert torch.equal(out.float(), expected.float())


def test_patch_weights_missing_key_rolls_back() -> None:
    weights = make_store()
    originals = dict(weights)
    diff = torch.randn(8, 6)
    patch_set = PatchSet(
        {
            "model.a.weight": (PatchEntry(DiffPatch(diff)),),
            "model.missing.weight": (PatchEntry(DiffPatch(diff)),),
        }
    )
    with pytest.raises(PatchApplyError, match="model.missing.weight"):
        patch_weights(weights, patch_set)
    # every key restored to the exact original object
    for key, tensor in originals.items():
        assert weights[key] is tensor


def test_patch_weights_skips_empty_entry_tuples() -> None:
    weights = make_store()
    backup = patch_weights(weights, PatchSet({"model.a.weight": ()}))
    assert backup == {}


def test_patch_weights_original_survives_for_model_as_lora() -> None:
    key = "model.a.weight"
    weights = make_store()
    original = weights[key]
    target = torch.randn(8, 6)
    patch_set = PatchSet({key: (PatchEntry(ModelAsLoraPatch(target)),)})
    patch_weights(weights, patch_set)
    torch.testing.assert_close(weights[key], original + (target - original), rtol=1e-5, atol=1e-6)


# --------------------------------------- per-entry function / convert


def test_entry_function_transforms_diff_delta() -> None:
    """PatchEntry.function receives the strength-scaled delta and its
    return value is what gets added (upstream patch tuple slot 4)."""
    weight = torch.zeros(4, 4)
    diff = torch.full((4, 4), 2.0)
    entry = PatchEntry(DiffPatch(diff), strength=0.5, function=lambda a: a * 10.0)
    out = apply_patches(weight.clone(), [entry], key="k")
    torch.testing.assert_close(out, torch.full((4, 4), 10.0))


def test_entry_function_not_routed_for_set() -> None:
    """SetPatch never routes through function (the reference's set
    branch copies directly)."""
    weight = torch.zeros(4, 4)
    value = torch.full((4, 4), 3.0)
    entry = PatchEntry(SetPatch(value), function=lambda a: a * 100.0)
    out = apply_patches(weight.clone(), [entry], key="k")
    torch.testing.assert_close(out, value)


def test_nested_convert_gets_owned_copy() -> None:
    """NestedPatch.convert runs on an owned intermediate-dtype copy of
    the donor base (upstream cast copy=True + convert inplace=True);
    the caller's base tensor is never mutated."""
    base = torch.full((4, 4), 1.0)
    base_snapshot = base.clone()
    inner = PatchEntry(DiffPatch(torch.full((4, 4), 0.5)))

    def convert(a: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
        return a.mul_(2.0) if inplace else a * 2.0

    entry = PatchEntry(NestedPatch(base, (inner,), convert=convert), strength=1.0)
    weight = torch.zeros(4, 4)
    out = apply_patches(weight.clone(), [entry], key="k")
    # patched donor = convert(base) + 0.5 = 2.5, added as a diff
    torch.testing.assert_close(out, torch.full((4, 4), 2.5))
    torch.testing.assert_close(base, base_snapshot)


# ------------------------------------------- fp8-scaled weight store


def fp8_store_key() -> tuple[str, Fp8ScaledWeight]:
    key = "model.q.weight"
    source = torch.randn(16, 8, generator=torch.Generator().manual_seed(41))
    return key, quantize_fp8_scaled(source.clone(), torch.float8_e4m3fn)


def test_patch_weights_fp8_scaled_writeback() -> None:
    """An Fp8ScaledWeight store entry round-trips through dequantize ->
    patch -> requantize with a recalculated scale and the per-key seed;
    the writeback is bit-reproducible."""
    key, stored = fp8_store_key()
    weights: dict[str, Fp8ScaledWeight] = {key: stored}
    diff = torch.randn(16, 8) * 0.05
    backup = patch_weights(weights, diff_set(key, diff))

    assert backup[key] is stored  # exact original object
    out = weights[key]
    assert isinstance(out, Fp8ScaledWeight)
    assert out.qdata.dtype == stored.qdata.dtype
    assert out.orig_dtype == stored.orig_dtype

    expected = requantize_fp8_scaled(
        stored,
        apply_patches(
            stored.dequantize(torch.float32),
            [PatchEntry(DiffPatch(diff))],
            key=key,
        ),
        seed=string_to_seed(key),
    )
    assert torch.equal(out.qdata.float(), expected.qdata.float())
    assert torch.equal(out.scale, expected.scale)

    restore_weights(weights, backup)
    assert weights[key] is stored


def test_patch_weights_fp8_scaled_model_as_lora_gets_dequantized_original() -> None:
    """For an fp8-scaled store the original handed to model-as-lora
    entries is a fresh dequantized copy (extension - upstream's
    quantized path never passes original_weights)."""
    key, stored = fp8_store_key()
    weights: dict[str, Fp8ScaledWeight] = {key: stored}
    target = torch.randn(16, 8)
    patch_set: PatchSet[torch.Tensor] = PatchSet({key: (PatchEntry(ModelAsLoraPatch(target)),)})
    patch_weights(weights, patch_set)
    out = weights[key]
    # mirror the pipeline: original + (target - original), where the
    # original is the dequantized copy, then seeded requantization
    expected = requantize_fp8_scaled(
        stored,
        apply_patches(
            stored.dequantize(torch.float32),
            [PatchEntry(ModelAsLoraPatch(target))],
            key=key,
            original_weight=stored.dequantize(torch.float32),
        ),
        seed=string_to_seed(key),
    )
    assert torch.equal(out.qdata.float(), expected.qdata.float())


def test_patch_weights_mixed_store_rolls_back_fp8() -> None:
    """Rollback restores the exact original Fp8ScaledWeight object in
    a store that mixes plain tensors and fp8-scaled storage."""
    key, stored = fp8_store_key()
    plain = torch.randn(8, 6)
    weights: dict[str, torch.Tensor | Fp8ScaledWeight] = {
        key: stored,
        "model.a.weight": plain,
    }
    diff = torch.randn(16, 8) * 0.05
    patch_set: PatchSet[torch.Tensor] = PatchSet(
        {
            key: (PatchEntry(DiffPatch(diff)),),
            "model.missing.weight": (PatchEntry(DiffPatch(torch.zeros(2, 2))),),
        }
    )
    with pytest.raises(PatchApplyError, match="model.missing.weight"):
        patch_weights(weights, patch_set)
    assert weights[key] is stored
    assert weights["model.a.weight"] is plain


def nvfp4_store_key() -> tuple[str, Nvfp4PackedWeight]:
    key = "model.nvfp4.weight"
    source = torch.linspace(-2.0, 2.0, 256).reshape(16, 16)
    scale = torch.amax(source.abs()).to(torch.float32) / (448.0 * 6.0)
    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]
    qdata, block_scale = kitchen.quantize(source, scale, pad_16x=False)
    return key, Nvfp4PackedWeight(qdata, block_scale, scale, (16, 16), torch.float32)


def int8_store_key() -> tuple[str, Int8PackedWeight]:
    from comfy_kitchen.tensor import (  # pyright: ignore[reportMissingTypeStubs]
        TensorWiseINT8Layout,
    )

    key = "model.int8.weight"
    source = torch.linspace(-2.0, 2.0, 7 * 256).reshape(7, 256)
    qdata, params = TensorWiseINT8Layout.quantize(
        source,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=256,
    )
    return key, Int8PackedWeight(qdata, params.scale, torch.float32, True, 256)


def test_patch_weights_int8_requantizes_without_mutating_and_restores_exactly() -> None:
    key, stored = int8_store_key()
    qdata = stored.qdata.clone()
    scale = stored.scale.clone()
    delta = torch.linspace(-0.02, 0.02, stored.qdata.numel()).reshape(stored.shape)
    patch_set = PatchSet({key: (PatchEntry(DiffPatch(delta)),)})
    weights: dict[str, Int8PackedWeight] = {key: stored}

    backup = patch_weights(weights, patch_set)
    assert backup[key] is stored
    assert weights[key] is not stored
    assert torch.equal(stored.qdata, qdata)
    assert torch.equal(stored.scale, scale)

    expected_float = apply_patches(
        stored.dequantize(torch.float32),
        patch_set.entries(key),
        key=key,
        original_weight=stored.dequantize(torch.float32),
    )
    expected = requantize_int8(stored, expected_float, seed=string_to_seed(key))
    assert torch.equal(weights[key].qdata, expected.qdata)
    assert torch.equal(weights[key].scale, expected.scale)

    restore_weights(weights, backup)
    assert weights[key] is stored


@pytest.mark.parametrize("model_as_lora", (False, True))
def test_patch_weights_nvfp4_requantizes_without_mutating_and_restores_exactly(
    model_as_lora: bool,
) -> None:
    key, stored = nvfp4_store_key()
    qdata = stored.qdata.clone()
    block_scale = stored.block_scale.clone()
    tensor_scale = stored.tensor_scale.clone()
    target = stored.dequantize() + 0.125
    value = ModelAsLoraPatch(target) if model_as_lora else DiffPatch(torch.full((16, 16), 0.125))
    patch_set = PatchSet({key: (PatchEntry(value),)})
    weights: dict[str, Nvfp4PackedWeight] = {key: stored}

    backup = patch_weights(weights, patch_set)
    assert backup[key] is stored
    assert weights[key] is not stored
    assert torch.equal(stored.qdata, qdata)
    assert torch.equal(stored.block_scale.view(torch.uint8), block_scale.view(torch.uint8))
    assert torch.equal(stored.tensor_scale, tensor_scale)

    expected_float = apply_patches(
        stored.dequantize(torch.float32),
        patch_set.entries(key),
        key=key,
        original_weight=stored.dequantize(torch.float32),
    )
    expected = requantize_nvfp4(stored, expected_float, seed=string_to_seed(key))
    assert torch.equal(weights[key].qdata, expected.qdata)
    assert torch.equal(
        weights[key].block_scale.view(torch.uint8),
        expected.block_scale.view(torch.uint8),
    )
    assert torch.equal(weights[key].tensor_scale, expected.tensor_scale)

    restore_weights(weights, backup)
    assert weights[key] is stored
