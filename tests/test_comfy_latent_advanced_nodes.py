"""Advanced latent nodes preserve the native ComfyUI contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
from dinkster_nodes_comfy.latent_advanced import (
    LATENT_ADVANCED_NODES,
    LatentAdd,
    LatentBatch,
    LatentBatchSeedBehavior,
    LatentConcat,
    LatentCut,
    LatentCutToBatch,
    LatentInterpolate,
    LatentMultiply,
    LatentSubtract,
)

torch: Any = pytest.importorskip("torch")


def _latent(output: object) -> dict[str, Any]:
    return cast(dict[str, Any], cast(dict[str, object], output)["LATENT"])


def test_advanced_latent_schemas_have_exact_core_interfaces() -> None:
    expected = (
        ("LatentAdd", "Latent Add", "model/latent/advanced", ("samples1", "samples2")),
        ("LatentSubtract", "Latent Subtract", "model/latent/advanced", ("samples1", "samples2")),
        ("LatentMultiply", "Latent Multiply", "model/latent/advanced", ("samples", "multiplier")),
        (
            "LatentInterpolate",
            "Latent Interpolate",
            "model/latent/advanced",
            ("samples1", "samples2", "ratio"),
        ),
        ("LatentConcat", "Latent Concat", "model/latent/advanced", ("samples1", "samples2", "dim")),
        ("LatentCut", "Latent Cut", "model/latent/advanced", ("samples", "dim", "index", "amount")),
        (
            "LatentCutToBatch",
            "Latent Cut To Batch",
            "model/latent/advanced",
            ("samples", "dim", "slice_size"),
        ),
        (
            "LatentBatch",
            "Batch Latents (DEPRECATED)",
            "model/latent/batch",
            ("samples1", "samples2"),
        ),
        (
            "LatentBatchSeedBehavior",
            "Latent Batch Seed Behavior",
            "model/latent/advanced",
            ("samples", "seed_behavior"),
        ),
    )
    assert len(LATENT_ADVANCED_NODES) == len(expected)
    for node, (node_type, display_name, category, inputs) in zip(
        LATENT_ADVANCED_NODES, expected, strict=True
    ):
        schema = node.schema()
        assert (schema.node_type, schema.display_name, schema.category) == (
            node_type,
            display_name,
            category,
        )
        assert tuple(input_spec.id for input_spec in schema.inputs) == inputs
        assert tuple(output_spec.type.types for output_spec in schema.outputs) == (("LATENT",),)


def test_arithmetic_resizes_repeats_and_preserves_first_metadata() -> None:
    first = {"samples": torch.ones((2, 1, 2, 2)), "batch_index": [4, 8], "note": "first"}
    second = {"samples": torch.full((1, 1, 1, 1), 2.0), "note": "second"}
    assert torch.equal(
        _latent(LatentAdd.execute(samples1=first, samples2=second))["samples"],
        torch.full((2, 1, 2, 2), 3.0),
    )
    assert torch.equal(
        _latent(LatentSubtract.execute(samples1=first, samples2=second))["samples"],
        torch.full((2, 1, 2, 2), -1.0),
    )
    multiplied = _latent(LatentMultiply.execute(samples=first, multiplier=-2.0))
    assert torch.equal(multiplied["samples"], torch.full((2, 1, 2, 2), -2.0))
    assert multiplied["note"] == "first"


def test_interpolation_edge_ratios_and_layout_nodes() -> None:
    first = {"samples": torch.tensor([[[[3.0, 4.0]]]])}
    second = {"samples": torch.tensor([[[[0.0, 10.0]]]])}
    assert torch.allclose(
        _latent(LatentInterpolate.execute(samples1=first, samples2=second, ratio=1.0))["samples"],
        first["samples"],
    )
    assert torch.allclose(
        _latent(LatentInterpolate.execute(samples1=first, samples2=second, ratio=0.0))["samples"],
        second["samples"],
    )
    concat = _latent(LatentConcat.execute(samples1=first, samples2=second, dim="-x"))["samples"]
    assert torch.equal(concat, torch.tensor([[[[0.0, 10.0, 3.0, 4.0]]]]))
    cut = _latent(LatentCut.execute(samples={"samples": concat}, dim="x", index=-2, amount=9))[
        "samples"
    ]
    assert torch.equal(cut, torch.tensor([[[[3.0, 4.0]]]]))
    split = _latent(LatentCutToBatch.execute(samples={"samples": concat}, dim="x", slice_size=2))[
        "samples"
    ]
    assert torch.equal(split, torch.tensor([[[[0.0, 10.0]]], [[[3.0, 4.0]]]]))


def test_batch_and_seed_behavior_preserve_batch_index_rules() -> None:
    first = {"samples": torch.ones((2, 1, 1, 1)), "batch_index": [7, 8]}
    second = {"samples": torch.ones((1, 1, 2, 2))}
    batched = _latent(LatentBatch.execute(samples1=first, samples2=second))
    assert batched["samples"].shape == (3, 1, 1, 1)
    assert batched["batch_index"] == [7, 8, 0]
    fixed = _latent(LatentBatchSeedBehavior.execute(samples=batched, seed_behavior="fixed"))
    assert fixed["batch_index"] == [7, 7, 7]
    random = _latent(LatentBatchSeedBehavior.execute(samples=fixed, seed_behavior="random"))
    assert "batch_index" not in random
