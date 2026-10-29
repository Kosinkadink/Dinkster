"""dinkster.latent decode materializes torch tensors bit-exactly.

The torch-free half (encode, validation, and transport) is covered in the
root suite; this proves the lazy torch decoder reconstructs exact tensor
bytes, dtypes, shapes, and multi-stream topology.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from dinkster_inference import (
    LATENT_TYPE_ID,
    MultiStreamLatent,
    SparseLatent,
    register_inference_types,
)
from dinkster_inference_torch import make_sparse_support, pack_sparse_latent
from dinkster_values import TypeRegistry, TypeSpec


def _spec() -> TypeSpec:
    registry = TypeRegistry()
    register_inference_types(registry)
    return registry.spec(LATENT_TYPE_ID)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_latent_round_trip_is_bit_exact(dtype: torch.dtype) -> None:
    spec = _spec()
    generator = torch.Generator().manual_seed(7)
    samples = torch.randn(2, 4, 8, 8, generator=generator).to(dtype)
    decoded = spec.decode(spec.encode({"samples": samples, "batch_index": [0, 1]}))
    assert isinstance(decoded, dict)
    assert decoded["batch_index"] == [0, 1]
    result = decoded["samples"]
    assert isinstance(result, torch.Tensor)
    assert result.dtype == dtype
    assert torch.equal(result, samples)


def test_latent_round_trips_zero_element_tensors() -> None:
    spec = _spec()
    samples = torch.empty(0, 4, 8, 8, dtype=torch.float32)
    decoded = spec.decode(spec.encode({"samples": samples}))
    assert isinstance(decoded, dict)
    result = decoded["samples"]
    assert isinstance(result, torch.Tensor)
    assert result.shape == samples.shape
    assert result.dtype == samples.dtype


def test_dinkster_latent_round_trip_preserves_multistream_tensors() -> None:
    samples = MultiStreamLatent.from_pairs(
        (
            ("video", torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)),
            ("audio", torch.arange(6, dtype=torch.bfloat16).reshape(1, 2, 3)),
        )
    )
    noise_mask = MultiStreamLatent.from_pairs(
        (
            ("video", torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])),
            ("audio", torch.tensor([[[1.0, 0.0, 1.0]]])),
        )
    )
    source = {
        "samples": samples,
        "noise_mask": noise_mask,
        "batch_index": [4],
        "metadata": {"fps": 24.0, "roles": ("video", "audio")},
    }
    spec = _spec()
    result = cast("dict[str, object]", spec.decode(spec.encode(source)))
    result_samples = cast("MultiStreamLatent[torch.Tensor]", result["samples"])
    result_mask = cast("MultiStreamLatent[torch.Tensor]", result["noise_mask"])

    assert result_samples.roles == samples.roles == ("video", "audio")
    assert result_mask.roles == noise_mask.roles == samples.roles
    for role in samples.roles:
        assert torch.equal(result_samples.by_role(role), samples.by_role(role))
        assert torch.equal(result_mask.by_role(role), noise_mask.by_role(role))
    assert result["batch_index"] == [4]
    assert result["metadata"] == {"fps": 24.0, "roles": ("video", "audio")}


def test_dinkster_latent_round_trip_authenticates_sparse_support() -> None:
    coordinates = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 2, 3], [1, 0, 1, 2]],
        dtype=torch.int32,
    )
    support = make_sparse_support(
        coordinates,
        (2, 1),
        8,
        (-0.5, -0.5, -0.5),
        (0.125, 0.125, 0.125),
    )
    source = pack_sparse_latent(
        support,
        torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
    )
    spec = _spec()
    result = cast("dict[str, object]", spec.decode(spec.encode({"samples": source})))
    decoded = result["samples"]

    assert type(decoded) is SparseLatent
    decoded = cast("SparseLatent[torch.Tensor]", decoded)
    assert decoded.support.support_id == support.support_id
    assert torch.equal(cast("torch.Tensor", decoded.support.coordinates), coordinates)
    assert torch.equal(decoded.features, source.features)


def test_dinkster_latent_refuses_mutated_sparse_support() -> None:
    coordinates = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    support = make_sparse_support(
        coordinates,
        (1,),
        8,
        (-0.5, -0.5, -0.5),
        (0.125, 0.125, 0.125),
    )
    source = pack_sparse_latent(support, torch.ones((1, 4)))
    coordinates[0, 1] = 7

    with pytest.raises(ValueError, match="digest"):
        _spec().encode({"samples": source})
