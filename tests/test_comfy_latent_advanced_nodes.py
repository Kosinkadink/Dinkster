"""Advanced latent nodes preserve the native ComfyUI contracts."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
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


class ArrayTensor:
    def __init__(self, values: object) -> None:
        self.array = np.asarray(values, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    def detach(self) -> ArrayTensor:
        return self

    def __getitem__(self, key: object) -> ArrayTensor:
        return ArrayTensor(self.array[cast(Any, key)])

    def __add__(self, other: object) -> ArrayTensor:
        return ArrayTensor(self.array + _array(other))

    def __sub__(self, other: object) -> ArrayTensor:
        return ArrayTensor(self.array - _array(other))

    def __mul__(self, other: object) -> ArrayTensor:
        return ArrayTensor(self.array * _array(other))

    def __truediv__(self, other: object) -> ArrayTensor:
        with np.errstate(divide="ignore", invalid="ignore"):
            return ArrayTensor(self.array / _array(other))

    def clone(self) -> ArrayTensor:
        return ArrayTensor(self.array.copy())

    def movedim(self, source: int, destination: int) -> ArrayTensor:
        return ArrayTensor(np.moveaxis(self.array, source, destination))

    def narrow(self, dim: int, start: int, length: int) -> ArrayTensor:
        slices = [slice(None)] * self.ndim
        slices[dim] = slice(start, start + length)
        return ArrayTensor(self.array[tuple(slices)])

    def repeat(self, *shape: object) -> ArrayTensor:
        dimensions = shape[0] if len(shape) == 1 else shape
        return ArrayTensor(np.tile(self.array, cast(tuple[int, ...], dimensions)))

    def reshape(self, shape: list[int] | tuple[int, ...]) -> ArrayTensor:
        return ArrayTensor(self.array.reshape(shape))


def _array(value: object) -> Any:
    return value.array if isinstance(value, ArrayTensor) else value


def _latent(output: object) -> dict[str, Any]:
    return cast(dict[str, Any], cast(dict[str, object], output)["LATENT"])


@pytest.fixture(autouse=True)
def tensor_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def repeat_to_batch_size(tensor: ArrayTensor, batch_size: int) -> ArrayTensor:
        if tensor.shape[0] > batch_size:
            return tensor.narrow(0, 0, batch_size)
        if tensor.shape[0] < batch_size:
            return tensor.repeat(
                math.ceil(batch_size / tensor.shape[0]), *([1] * (tensor.ndim - 1))
            ).narrow(0, 0, batch_size)
        return tensor

    def common_upscale(
        tensor: ArrayTensor, width: int, height: int, _method: str, _crop: str
    ) -> ArrayTensor:
        return ArrayTensor(np.resize(tensor.array, (*tensor.shape[:-2], height, width)))

    torch = SimpleNamespace(
        cat=lambda values, dim: ArrayTensor(
            np.concatenate([value.array for value in values], axis=dim)
        ),
        narrow=lambda tensor, dim, start, length: tensor.narrow(dim, start, length),
        nan_to_num=lambda tensor: ArrayTensor(np.nan_to_num(tensor.array)),
        linalg=SimpleNamespace(
            vector_norm=lambda tensor, dim: ArrayTensor(
                np.linalg.vector_norm(tensor.array, axis=dim)
            )
        ),
    )
    utils = SimpleNamespace(
        common_upscale=common_upscale, repeat_to_batch_size=repeat_to_batch_size
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "dinkster_inference", SimpleNamespace(utils=utils))
    monkeypatch.setitem(sys.modules, "dinkster_inference.utils", utils)


def test_advanced_latent_schemas_have_exact_core_interfaces() -> None:
    expected = (
        ("LatentAdd", "LatentAdd", "model/latent/advanced", ("samples1", "samples2")),
        ("LatentSubtract", "LatentSubtract", "model/latent/advanced", ("samples1", "samples2")),
        ("LatentMultiply", "LatentMultiply", "model/latent/advanced", ("samples", "multiplier")),
        (
            "LatentInterpolate",
            "LatentInterpolate",
            "model/latent/advanced",
            ("samples1", "samples2", "ratio"),
        ),
        ("LatentConcat", "LatentConcat", "model/latent/advanced", ("samples1", "samples2", "dim")),
        ("LatentCut", "LatentCut", "model/latent/advanced", ("samples", "dim", "index", "amount")),
        (
            "LatentCutToBatch",
            "LatentCutToBatch",
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
            "LatentBatchSeedBehavior",
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
    batch_schema = LatentBatch.schema()
    assert batch_schema.deprecation is not None
    assert batch_schema.search_visibility == "deprecated"


def test_arithmetic_preserves_first_metadata_and_repeats_second_batch() -> None:
    first = {"samples": ArrayTensor(np.ones((2, 1, 2, 2))), "batch_index": [4, 8], "note": "first"}
    second = {"samples": ArrayTensor(np.full((1, 1, 2, 2), 2.0)), "note": "second"}
    assert _latent(LatentAdd.execute(samples1=first, samples2=second))[
        "samples"
    ].array.tolist() == [
        [[[3.0, 3.0], [3.0, 3.0]]],
        [[[3.0, 3.0], [3.0, 3.0]]],
    ]
    assert _latent(LatentSubtract.execute(samples1=first, samples2=second))[
        "samples"
    ].array.tolist() == [
        [[[-1.0, -1.0], [-1.0, -1.0]]],
        [[[-1.0, -1.0], [-1.0, -1.0]]],
    ]
    multiplied = _latent(LatentMultiply.execute(samples=first, multiplier=-2.0))
    assert np.all(multiplied["samples"].array == -2.0)
    assert multiplied["note"] == "first"


@pytest.mark.parametrize("dim,axis", (("x", -1), ("y", -2), ("t", -3)))
def test_cut_handles_negative_and_overbound_indices_for_5d_latents(dim: str, axis: int) -> None:
    source = ArrayTensor(np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6))
    negative = _latent(
        LatentCut.execute(samples={"samples": source}, dim=dim, index=-99, amount=2)
    )["samples"]
    overbound = _latent(
        LatentCut.execute(samples={"samples": source}, dim=dim, index=99, amount=9)
    )["samples"]
    assert negative.shape[axis] == 2
    assert overbound.shape[axis] == 1


def test_interpolation_handles_zero_norm_and_non_endpoint_ratio() -> None:
    zeros = {"samples": ArrayTensor(np.zeros((1, 1, 2, 2)))}
    values = {"samples": ArrayTensor(np.array([[[[3.0, 4.0], [0.0, 5.0]]]]))}
    assert np.all(
        _latent(LatentInterpolate.execute(samples1=zeros, samples2=zeros, ratio=0.4))[
            "samples"
        ].array
        == 0
    )
    blended = _latent(LatentInterpolate.execute(samples1=values, samples2=zeros, ratio=0.4))[
        "samples"
    ]
    assert np.isfinite(blended.array).all()
    assert not np.array_equal(blended.array, values["samples"].array)


def test_concat_cut_to_batch_and_batch_index_metadata() -> None:
    first = {"samples": ArrayTensor(np.ones((2, 1, 1, 3))), "batch_index": [7, 8]}
    second = {"samples": ArrayTensor(np.full((1, 1, 1, 3), 2.0))}
    concat = _latent(LatentConcat.execute(samples1=first, samples2=second, dim="-x"))["samples"]
    assert concat.array[0, 0, 0].tolist() == [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
    cut_to_batch = _latent(
        LatentCutToBatch.execute(
            samples={"samples": ArrayTensor(np.arange(10).reshape(1, 1, 1, 10))},
            dim="x",
            slice_size=3,
        )
    )["samples"]
    assert cut_to_batch.shape == (3, 1, 1, 3)
    batched = _latent(LatentBatch.execute(samples1=first, samples2=second))
    assert batched["batch_index"] == [7, 8, 0]
    assert _latent(LatentBatchSeedBehavior.execute(samples=batched, seed_behavior="fixed"))[
        "batch_index"
    ] == [7, 7, 7]
    assert "batch_index" not in _latent(
        LatentBatchSeedBehavior.execute(samples=batched, seed_behavior="random")
    )
