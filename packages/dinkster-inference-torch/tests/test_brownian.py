"""Stage 5: brownian-tree step noise vs the executed reference.

Every golden in goldens/brownian_goldens.json was produced by RUNNING
the reference stack (tools/gen_brownian_goldens.py): numpy
SeedSequence word streams, torchsde 0.2.6 BrownianTree interval
queries, and comfy/k_diffusion/sampling.py BrownianTreeNoiseSampler
(cpu=True) @ 947c2749. The native transcription must agree EXACTLY -
the whole point of the port is a bit-identical noise stream without
numpy or torchsde in the environment.

Deliberate loud deviations (documented in brownian.py) are tested
directly: required int seed, refusal beyond one tree-grid step of
bound overhang, refusal on reversed tree bounds. Reference-legal
edges are tested too: the zero-width (equal-bounds) tree the SDE
solvers build for one-step schedules, and the 45-entry increment
cache bound with value-neutral eviction.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference_torch import BrownianNoiseError, BrownianTreeNoise
from dinkster_inference_torch.brownian import (
    SeedSequence,
    _BrownianTree,  # pyright: ignore[reportPrivateUsage]
)
from golden_files import load_platform_golden

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "brownian_goldens.json")


def dec(spec: dict[str, Any]) -> torch.Tensor:
    # JSON floats are float64; decode there first so float64 goldens
    # stay exact, then cast (exact for values born float32).
    data = torch.tensor(spec["data"], dtype=torch.float64)
    return data.reshape(spec["shape"]).to(getattr(torch, spec["dtype"]))


def test_goldens_are_from_the_audited_baseline() -> None:
    assert GOLDENS["_meta"]["reference_commit"].startswith("b78cec87")


class TestSeedSequence:
    @pytest.mark.parametrize(
        "case",
        GOLDENS["seedseq_cases"],
        ids=lambda c: f"e{c['entropy']}_k{c['spawn_key']}_p{c['pool_size']}",
    )
    def test_word_stream_matches_numpy(self, case: dict[str, Any]) -> None:
        got = SeedSequence(
            case["entropy"],
            spawn_key=tuple(case["spawn_key"]),
            pool_size=case["pool_size"],
        ).generate_state(len(case["expected"]))
        assert got == case["expected"]

    def test_bool_entropy_refuses(self) -> None:
        with pytest.raises(BrownianNoiseError, match="int"):
            SeedSequence(True)

    def test_negative_entropy_refuses(self) -> None:
        with pytest.raises(BrownianNoiseError, match="non-negative"):
            SeedSequence(-1)


def make_tree(case: dict[str, Any]) -> _BrownianTree:
    return _BrownianTree(
        t0=case["t0"],
        t1=case["t1"],
        size=tuple(case["shape"]),
        dtype=getattr(torch, case["dtype"]),
        device=torch.device("cpu"),
        entropy=case["entropy"],
    )


class TestBrownianTree:
    @pytest.mark.parametrize("case", GOLDENS["tree_cases"], ids=lambda c: c["name"])
    def test_queries_match_torchsde(self, case: dict[str, Any]) -> None:
        tree = make_tree(case)
        for (qa, qb), expected in zip(case["queries"], case["expected"], strict=True):
            got = tree.query(qa, qb)
            want = dec(expected)
            assert got.dtype == want.dtype
            assert torch.equal(got, want), f"query [{qa}, {qb}] diverged"

    def test_partition_adds_up_to_the_whole(self) -> None:
        """W(a,c) == W(a,b) + W(b,c): increments over a partition sum
        to the whole interval's increment (the reference's defining
        tree invariant), regardless of query order."""
        tree = make_tree(GOLDENS["tree_cases"][0])
        whole = tree.query(0.5, 12.0)
        left = tree.query(0.5, 4.0)
        right = tree.query(4.0, 12.0)
        assert torch.allclose(left + right, whole, atol=1e-5)

    def test_zero_width_query_is_zeros(self) -> None:
        tree = make_tree(GOLDENS["tree_cases"][0])
        w = tree.query(3.0, 3.0)
        assert torch.equal(w, torch.zeros_like(w))

    def test_query_leaving_the_interval_refuses(self) -> None:
        case = GOLDENS["tree_cases"][0]
        tree = make_tree(case)
        with pytest.raises(BrownianNoiseError, match="leaves the tree"):
            tree.query(case["t0"], case["t1"] + 1.0)
        with pytest.raises(BrownianNoiseError, match="leaves the tree"):
            tree.query(case["t0"] - 1.0, case["t1"])

    def test_zero_width_tree_constructs_and_reversed_refuses(self) -> None:
        """t0 == t1 is legal like the reference (torchsde only refuses
        t0 > t1): the SDE solvers build a zero-width tree for a
        one-step (sigma, 0) schedule and never query it. Its root
        increment is exactly zero (randn * sqrt(0))."""
        tree = _BrownianTree(
            t0=1.0,
            t1=1.0,
            size=(2,),
            dtype=torch.float32,
            device=torch.device("cpu"),
            entropy=0,
        )
        assert torch.equal(tree.root_w, torch.zeros(2))
        assert torch.equal(tree.query(1.0, 1.0), torch.zeros(2))
        with pytest.raises(BrownianNoiseError, match="t0 <= t1"):
            _BrownianTree(
                t0=2.0,
                t1=1.0,
                size=(2,),
                dtype=torch.float32,
                device=torch.device("cpu"),
                entropy=0,
            )

    def test_increment_cache_stays_bounded_and_eviction_is_value_neutral(self) -> None:
        """The reference's 45-entry LRU transcribed: querying many
        distinct schedule boundaries must not retain more than 45
        child increments (the root's W lives outside the cache), and
        recomputing an evicted increment reproduces the exact tensor
        (each node's noise is deterministic in its stored seed)."""
        case = GOLDENS["tree_cases"][0]
        tree = make_tree(case)
        t0, t1 = float(case["t0"]), float(case["t1"])
        first = tree.query(t0, t1 / 2)
        # Drive far more than 45 distinct nodes into existence.
        for i in range(1, 200):
            ta = t0 + (t1 - t0) * i / 401
            tb = t0 + (t1 - t0) * (i + 1) / 401
            tree.query(ta, tb)
        assert len(tree.cache) <= 45
        # The early increment has long been evicted; the re-query
        # recomputes it bit-identically from the parent chain.
        assert torch.equal(tree.query(t0, t1 / 2), first)

    def test_global_rng_untouched(self) -> None:
        state = torch.random.get_rng_state()
        tree = make_tree(GOLDENS["tree_cases"][0])
        tree.query(1.0, 9.0)
        assert torch.equal(state, torch.random.get_rng_state())


class TestBrownianTreeNoise:
    @pytest.mark.parametrize("case", GOLDENS["sampler_cases"], ids=lambda c: c["name"])
    def test_step_noise_matches_reference(self, case: dict[str, Any]) -> None:
        like = torch.zeros(tuple(case["shape"]), dtype=getattr(torch, case["dtype"]))
        sampler = BrownianTreeNoise(like, case["sigma_min"], case["sigma_max"], seed=case["seed"])
        for (s_from, s_to), expected in zip(case["queries"], case["expected"], strict=True):
            got = sampler(s_from, s_to)
            want = dec(expected)
            assert got.dtype == want.dtype
            assert torch.equal(got, want), f"step ({s_from} -> {s_to}) diverged"

    def test_same_seed_replays_identically(self) -> None:
        like = torch.zeros(2, 3)
        a = BrownianTreeNoise(like, 0.1, 5.0, seed=11)
        b = BrownianTreeNoise(like, 0.1, 5.0, seed=11)
        assert torch.equal(a(5.0, 2.0), b(5.0, 2.0))
        assert torch.equal(a(2.0, 0.5), b(2.0, 0.5))

    def test_different_seeds_differ(self) -> None:
        like = torch.zeros(2, 3)
        a = BrownianTreeNoise(like, 0.1, 5.0, seed=11)
        b = BrownianTreeNoise(like, 0.1, 5.0, seed=12)
        assert not torch.equal(a(5.0, 2.0), b(5.0, 2.0))

    def test_output_matches_like_dtype_and_device(self) -> None:
        like = torch.zeros(2, 2, dtype=torch.float64)
        sampler = BrownianTreeNoise(like, 0.1, 5.0, seed=3)
        out = sampler(5.0, 1.0)
        assert out.dtype == torch.float64
        assert out.device == like.device
        assert out.shape == like.shape

    def test_equal_bounds_construct_a_zero_width_tree(self) -> None:
        """sigma_min == sigma_max is the reference's legal one-step
        (sigma, 0) schedule: BrownianTreeNoiseSampler(x, s, s)
        constructs (validated live against the reference @ 947c2749:
        sort yields sign -1 and torchsde accepts t0 == t1) and the
        solver never queries it. The zero-width tree's root increment
        is exactly zero."""
        like = torch.zeros(2)
        sampler = BrownianTreeNoise(like, 5.0, 5.0, seed=0)
        tree = sampler._tree  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(tree.root_w, torch.zeros(2))

    def test_global_rng_untouched(self) -> None:
        state = torch.random.get_rng_state()
        sampler = BrownianTreeNoise(torch.zeros(2, 3), 0.1, 5.0, seed=11)
        sampler(5.0, 2.0)
        assert torch.equal(state, torch.random.get_rng_state())
