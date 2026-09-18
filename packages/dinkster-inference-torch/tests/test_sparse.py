from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from dinkster_inference_torch.sparse import (
    authenticate_sparse_support,
    make_sparse_support,
    pack_sparse_latent,
    sparse_support_id,
    unpack_sparse_latent,
)


def coordinates() -> torch.Tensor:
    return torch.tensor(
        ((0, 1, 2, 3), (0, 4, 5, 6), (1, 7, 8, 9)),
        dtype=torch.int32,
    )


def support():
    return make_sparse_support(
        coordinates(),
        (2, 1),
        64,
        (-0.5, -0.5, -0.5),
        (1.0 / 64, 1.0 / 64, 1.0 / 64),
    )


def test_support_id_is_value_stable_across_integer_dtype() -> None:
    value = coordinates()
    assert sparse_support_id(value) == sparse_support_id(value.to(torch.int64))


def test_support_authentication_refuses_noncontiguous_batches() -> None:
    value = coordinates().index_select(0, torch.tensor((0, 2, 1)))
    with pytest.raises(ValueError, match="contiguous"):
        make_sparse_support(
            value,
            (2, 1),
            64,
            (-0.5, -0.5, -0.5),
            (1.0 / 64, 1.0 / 64, 1.0 / 64),
        )


def test_support_authentication_refuses_mutation_and_forgery() -> None:
    value = support()
    value.coordinates[0, 1] = 10
    with pytest.raises(ValueError, match="digest"):
        authenticate_sparse_support(value)
    with pytest.raises(ValueError, match="digest"):
        authenticate_sparse_support(replace(support(), support_id="sha256:" + "0" * 64))


def test_sparse_pack_unpack_preserves_exact_support() -> None:
    expected_support = support()
    features = torch.randn(3, 8)
    latent = pack_sparse_latent(expected_support, features)
    actual_support, actual_features = unpack_sparse_latent(latent)
    assert actual_support is expected_support
    assert actual_features is features
    with pytest.raises(ValueError, match="one row per support coordinate"):
        pack_sparse_latent(expected_support, torch.randn(2, 8))
