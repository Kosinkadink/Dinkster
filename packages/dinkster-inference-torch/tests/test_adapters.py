"""Adapter tensor math vs executed references.

The recorded inputs come from running comfy/weight_adapter/*.calculate_weight
@ 947c2749 through tools/gen_adapter_goldens.py. Constrained OFT/BOFT expected
values use the trainer's ``alpha * out_dim`` contract and are independently
derived in focused tests below. Loud AdapterMathError behavior and the
reference-broken paths in docs/comfyui-issues/ are also tested separately.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference_torch import (
    AdapterMathError,
    BOFTAdapter,
    GLoRAAdapter,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    OFTAdapter,
    pad_tensor_to_shape,
)

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "adapter_goldens.json").read_text())

DTYPES = {"float32": torch.float32, "float16": torch.float16}


def dec(spec: dict[str, Any]) -> torch.Tensor:
    data = torch.tensor(spec["data"], dtype=torch.float32)
    return data.reshape(spec["shape"]).to(DTYPES[spec["dtype"]])


def dec_opt(spec: dict[str, Any] | None) -> torch.Tensor | None:
    return None if spec is None else dec(spec)


def build(case: dict[str, Any]):
    w = case["weights"]
    kind = case["adapter"]
    if kind == "LoRAAdapter":
        reshape = w["reshape"]
        return LoRAAdapter(
            dec(w["up"]),
            dec(w["down"]),
            alpha=w["alpha"],
            mid=dec_opt(w["mid"]),
            dora_scale=dec_opt(w["dora_scale"]),
            reshape=None if reshape is None else tuple(reshape),
        )
    if kind == "LoHaAdapter":
        return LoHaAdapter(
            dec(w["w1_a"]),
            dec(w["w1_b"]),
            dec(w["w2_a"]),
            dec(w["w2_b"]),
            alpha=w["alpha"],
            t1=dec_opt(w["t1"]),
            t2=dec_opt(w["t2"]),
            dora_scale=dec_opt(w["dora_scale"]),
        )
    if kind == "LoKrAdapter":
        return LoKrAdapter(
            w1=dec_opt(w["w1"]),
            w2=dec_opt(w["w2"]),
            w1_a=dec_opt(w["w1_a"]),
            w1_b=dec_opt(w["w1_b"]),
            w2_a=dec_opt(w["w2_a"]),
            w2_b=dec_opt(w["w2_b"]),
            t2=dec_opt(w["t2"]),
            alpha=w["alpha"],
            dora_scale=dec_opt(w["dora_scale"]),
        )
    if kind == "GLoRAAdapter":
        return GLoRAAdapter(
            dec(w["a1"]),
            dec(w["a2"]),
            dec(w["b1"]),
            dec(w["b2"]),
            alpha=w["alpha"],
            dora_scale=dec_opt(w["dora_scale"]),
        )
    if kind == "OFTAdapter":
        return OFTAdapter(
            dec(w["blocks"]),
            rescale=dec_opt(w["rescale"]),
            alpha=w["alpha"],
            dora_scale=dec_opt(w["dora_scale"]),
        )
    if kind == "BOFTAdapter":
        return BOFTAdapter(
            dec(w["blocks"]),
            rescale=dec_opt(w["rescale"]),
            alpha=w["alpha"],
            dora_scale=dec_opt(w["dora_scale"]),
        )
    raise AssertionError(f"unknown adapter kind {kind}")


# ------------------------------------------------------------- goldens


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=[c["name"] for c in GOLDENS["cases"]])
def test_golden(case: dict[str, Any]) -> None:
    adapter = build(case)
    weight = dec(case["weight"])
    expected = dec(case["expected"])
    result = adapter.calculate(weight.clone(), strength=case["strength"])
    assert result.dtype == expected.dtype
    assert tuple(result.shape) == tuple(expected.shape)
    if expected.dtype == torch.float16:
        rtol, atol = 1e-2, 1e-3
    else:
        rtol, atol = 1e-5, 1e-6
    assert torch.allclose(result.float(), expected.float(), rtol=rtol, atol=atol), (
        case["name"],
        (result.float() - expected.float()).abs().max().item(),
    )


def test_golden_meta_pins_reference() -> None:
    assert GOLDENS["_meta"]["reference_commit"].startswith("b78cec87")
    assert GOLDENS["_meta"]["constraint_contract_cases"] == [
        "oft_constraint",
        "boft_rescale_strength",
    ]


def _cayley_rotation_at_bound(blocks: torch.Tensor, bound: float) -> torch.Tensor:
    q = blocks - blocks.transpose(-1, -2)
    q_norm = torch.norm(q) + 1e-8
    if q_norm > bound:
        q = q * bound / q_norm
    eye = torch.eye(blocks.shape[-1], dtype=blocks.dtype)
    return torch.linalg.solve((eye - q).transpose(-1, -2), (eye + q).transpose(-1, -2)).transpose(
        -1, -2
    )


def test_oft_constraint_uses_alpha_times_out_dim() -> None:
    blocks = torch.tensor(
        [
            [[0.0, 0.04], [0.0, 0.0]],
            [[0.0, 0.0], [0.03, 0.0]],
        ]
    )
    weight = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3) / 10
    alpha = 0.05
    block_num, block_size = blocks.shape[:2]

    effective_bound = alpha * block_num * block_size
    q_norm = torch.norm(blocks - blocks.transpose(-1, -2)) + 1e-8
    assert alpha < q_norm < effective_bound
    expected = torch.bmm(
        _cayley_rotation_at_bound(blocks, effective_bound).transpose(1, 2),
        weight.unflatten(0, (block_num, block_size)),
    ).flatten(0, 1)
    raw_alpha_result = torch.bmm(
        _cayley_rotation_at_bound(blocks, alpha).transpose(1, 2),
        weight.unflatten(0, (block_num, block_size)),
    ).flatten(0, 1)

    result = OFTAdapter(blocks, alpha=alpha).calculate(weight.clone(), strength=1.0)

    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(result, raw_alpha_result, rtol=1e-5, atol=1e-6)


def test_boft_constraint_uses_alpha_times_out_dim() -> None:
    blocks = torch.tensor(
        [
            [
                [[0.0, 0.04], [0.0, 0.0]],
                [[0.0, 0.0], [0.03, 0.0]],
            ]
        ]
    )
    weight = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3) / 10
    alpha = 0.05
    _boft_m, block_num, boft_b = blocks.shape[:3]

    effective_bound = alpha * block_num * boft_b
    q_norm = torch.norm(blocks - blocks.transpose(-1, -2)) + 1e-8
    assert alpha < q_norm < effective_bound
    expected = torch.bmm(
        _cayley_rotation_at_bound(blocks[0], effective_bound),
        weight.unflatten(0, (block_num, boft_b)),
    ).flatten(0, 1)
    raw_alpha_result = torch.bmm(
        _cayley_rotation_at_bound(blocks[0], alpha),
        weight.unflatten(0, (block_num, boft_b)),
    ).flatten(0, 1)

    result = BOFTAdapter(blocks, alpha=alpha).calculate(weight.clone(), strength=1.0)

    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(result, raw_alpha_result, rtol=1e-5, atol=1e-6)


def test_oft_and_boft_dora_receive_raw_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    import dinkster_inference_torch.adapters as adapters_mod

    received_alphas: list[float] = []

    def record_alpha(
        dora_scale: torch.Tensor,
        weight: torch.Tensor,
        lora_diff: torch.Tensor,
        alpha: float,
        strength: float,
        intermediate_dtype: torch.dtype,
        function: Any,
    ) -> torch.Tensor:
        del dora_scale, lora_diff, strength, intermediate_dtype, function
        received_alphas.append(alpha)
        return weight

    monkeypatch.setattr(adapters_mod, "weight_decompose", record_alpha)
    alpha = 0.05
    weight = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3) / 10
    oft_blocks = torch.zeros(2, 2, 2)
    boft_blocks = torch.zeros(1, 2, 2, 2)

    OFTAdapter(oft_blocks, alpha=alpha, dora_scale=torch.ones(4, 1)).calculate(
        weight.clone(), strength=1.0
    )
    BOFTAdapter(boft_blocks, alpha=alpha, dora_scale=torch.ones(4, 1)).calculate(
        weight.clone(), strength=1.0
    )

    assert received_alphas == [alpha, alpha]


def test_adapters_satisfy_stage1_protocol() -> None:
    """Each torch adapter must be a WeightAdapter[torch.Tensor]
    (stage-1 contract). The annotated assignments below are verified
    statically by the package pyright gate; the runtime assertions
    just keep the test observable."""
    from dinkster_inference import WeightAdapter

    adapters: list[WeightAdapter[torch.Tensor]] = [
        LoRAAdapter(torch.zeros(8, 3), torch.zeros(3, 6)),
        LoHaAdapter(
            torch.zeros(8, 3),
            torch.zeros(3, 6),
            torch.zeros(8, 3),
            torch.zeros(3, 6),
        ),
        LoKrAdapter(w1=torch.zeros(2, 3), w2=torch.zeros(4, 2)),
        GLoRAAdapter(
            torch.zeros(3, 6),
            torch.zeros(6, 3),
            torch.zeros(3, 6),
            torch.zeros(8, 3),
        ),
        OFTAdapter(torch.zeros(2, 4, 4)),
        BOFTAdapter(torch.zeros(2, 2, 4, 4)),
    ]
    for adapter in adapters:
        assert adapter.target_shape((8, 6)) == (8, 6)


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=[c["name"] for c in GOLDENS["cases"]])
def test_adapters_rebuild_equivalent_payload_state(case: dict[str, Any]) -> None:
    from dinkster_inference import PreparedWeightAdapter

    adapter: PreparedWeightAdapter[torch.Tensor] = build(case)
    payloads = adapter.payload_tensors()
    replacements = tuple(payload.clone() for payload in payloads)
    rebuilt = adapter.rebuild_payloads(replacements)
    assert rebuilt is not adapter
    assert rebuilt.payload_tensors() == replacements
    payload_names = {
        name for name, value in vars(adapter).items() if isinstance(value, torch.Tensor)
    }
    assert {name: value for name, value in vars(rebuilt).items() if name not in payload_names} == {
        name: value for name, value in vars(adapter).items() if name not in payload_names
    }
    with pytest.raises(ValueError, match="payload replacements"):
        adapter.rebuild_payloads(replacements[:-1])


# ------------------------------------------- shape prediction + mutation


def test_target_shape_reshape_vs_base() -> None:
    plain = LoRAAdapter(torch.zeros(8, 3), torch.zeros(3, 6))
    assert plain.target_shape((8, 6)) == (8, 6)
    padded = LoRAAdapter(torch.zeros(8, 3), torch.zeros(3, 6), reshape=(8, 6))
    assert padded.target_shape((6, 4)) == (8, 6)


def test_calculate_mutates_in_place_like_reference() -> None:
    adapter = LoRAAdapter(torch.randn(8, 3), torch.randn(3, 6), alpha=2.0)
    weight = torch.zeros(8, 6)
    result = adapter.calculate(weight, strength=1.0)
    assert result is weight  # += path returns the same tensor
    assert not torch.equal(weight, torch.zeros(8, 6))


def test_reshape_path_returns_new_tensor() -> None:
    adapter = LoRAAdapter(torch.randn(8, 3), torch.randn(3, 6), reshape=(8, 6))
    weight = torch.zeros(6, 4)
    result = adapter.calculate(weight, strength=1.0)
    assert result is not weight
    assert tuple(result.shape) == (8, 6)
    assert torch.equal(weight, torch.zeros(6, 4))  # original untouched


# ------------------------------------------------- deliberate loud errors
# The reference catches these, logs, and silently returns the weight
# unpatched; Dinkster raises instead (documented deviation).


def test_lora_rank_mismatch_raises() -> None:
    adapter = LoRAAdapter(torch.randn(8, 3), torch.randn(4, 6))
    with pytest.raises(AdapterMathError, match="lora"):
        adapter.calculate(torch.zeros(8, 6), strength=1.0)


def test_lokr_incomplete_decomposition_raises() -> None:
    adapter = LoKrAdapter(w2=torch.randn(4, 2))
    with pytest.raises(AdapterMathError, match="w1_a/w1_b"):
        adapter.calculate(torch.zeros(8, 6), strength=1.0)


def test_glora_alpha_without_orientation_raises() -> None:
    adapter = GLoRAAdapter(
        torch.randn(3, 5),
        torch.randn(4, 2),
        torch.randn(2, 6),
        torch.randn(7, 3),
        alpha=1.0,
    )
    with pytest.raises(AdapterMathError, match="orientation"):
        adapter.calculate(torch.zeros(8, 6), strength=1.0)


def test_pad_smaller_shape_raises_valueerror() -> None:
    # parity: upstream raises the ValueError OUTSIDE its try/except
    with pytest.raises(ValueError, match="larger"):
        pad_tensor_to_shape(torch.zeros(8, 6), (6, 4))


def test_lokr_tucker_conv_fixed_beyond_reference() -> None:
    """Upstream-broken path FIXED in Dinkster (docs/comfyui-issues/
    lokr-tucker-kron-noncontiguous.md): at 947c2749 the reference
    lokr+t2 path ALWAYS fails (einsum output is non-contiguous,
    torch.kron raises a view error) and silently leaves the weight
    unpatched, so no oracle golden can exist. Verified independently
    instead: expected weight computed in float64 with explicit loops
    (Tucker reconstruction + Kronecker product from their
    definitions). When upstream fixes this, add a reference golden
    cross-check (ROADMAP pin)."""
    gen = torch.Generator().manual_seed(7)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=gen)

    w1 = rnd(2, 3)
    w2_a, w2_b, t2 = rnd(3, 4), rnd(2, 2), rnd(3, 2, 3, 3)
    alpha, strength = 1.5, 0.75
    weight = rnd(8, 6, 3, 3)

    adapter = LoKrAdapter(w1=w1.clone(), w2_a=w2_a, w2_b=w2_b, t2=t2, alpha=alpha)
    got = adapter.calculate(weight.clone(), strength=strength)

    # Independent float64 reference, loops only.
    t2d, w2ad, w2bd = t2.double(), w2_a.double(), w2_b.double()
    p_dim, r_dim = w2ad.shape[1], w2bd.shape[1]
    i_dim, j_dim, kk, ll = t2d.shape
    w2 = torch.zeros(p_dim, r_dim, kk, ll, dtype=torch.float64)
    for p in range(p_dim):
        for r in range(r_dim):
            for i in range(i_dim):
                for j in range(j_dim):
                    w2[p, r] += t2d[i, j] * w2bd[j, r] * w2ad[i, p]
    w1d = w1.double()
    m, n = w1d.shape
    diff = torch.zeros(m * p_dim, n * r_dim, kk, ll, dtype=torch.float64)
    for a in range(m):
        for b in range(n):
            diff[a * p_dim : (a + 1) * p_dim, b * r_dim : (b + 1) * r_dim] = w1d[a, b] * w2
    scale = strength * (alpha / w2_b.shape[0])  # dim = w2_b.shape[0]
    expected = weight.double() + scale * diff

    assert torch.allclose(got.double(), expected, rtol=1e-4, atol=1e-5)


def test_boft_fp16_partial_strength_fixed_beyond_reference() -> None:
    """Upstream-broken path FIXED in Dinkster (docs/comfyui-issues/
    boft-fp16-partial-strength-dtype-mismatch.md): at 947c2749 the
    reference BOFT path ALWAYS fails for fp16 weight + strength != 1
    (interpolating with the fp32 eye promotes bi, the einsum rejects
    the fp32/fp16 mix) and silently leaves the weight unpatched, so
    no oracle golden can exist. Verified independently instead: the
    same inputs run at float32 (a path that IS golden-covered) must
    agree within fp16 tolerance. When upstream fixes this, add a
    reference golden cross-check (ROADMAP pin)."""
    gen = torch.Generator().manual_seed(11)
    blocks = torch.randn((2, 2, 4, 4), generator=gen) * 0.1
    weight = torch.randn((8, 6), generator=gen)

    got16 = BOFTAdapter(blocks.clone(), alpha=0.0).calculate(weight.to(torch.float16), strength=0.5)
    got32 = BOFTAdapter(blocks.clone(), alpha=0.0).calculate(weight.clone(), strength=0.5)

    assert got16.dtype == torch.float16
    assert torch.allclose(got16.float(), got32, rtol=5e-3, atol=5e-3)
    # and the patch actually did something
    assert not torch.equal(got16, weight.to(torch.float16))


# ---------------------------------------------------------- compile smoke


def test_pure_lora_diff_compiles_without_graph_break() -> None:
    """Patch application is mutable setup work and stays outside
    compiled forwards; the pure diff math must still be
    compile-clean (fullgraph=True fails loudly on graph breaks).
    backend="eager" on purpose: graph-break detection is dynamo's,
    and inductor CPU codegen needs system Python headers. The real
    GPU/Triton worker-thread gate stays deferred (ROADMAP)."""

    def lora_diff(up: torch.Tensor, down: torch.Tensor, scale: float) -> torch.Tensor:
        return scale * torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1))

    up, down = torch.randn(8, 3), torch.randn(3, 6)
    eager = lora_diff(up, down, 0.5)
    compiled = torch.compile(lora_diff, fullgraph=True, backend="eager")(up, down, 0.5)
    assert torch.allclose(eager, compiled, rtol=1e-5, atol=1e-6)
