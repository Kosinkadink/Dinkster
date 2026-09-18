"""Authenticated torch realization of sparse latent support."""

from __future__ import annotations

import hashlib
import struct

import torch
from dinkster_inference import SparseLatent, SparseSupport

_COORDINATE_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)
_DIGEST_DOMAIN = b"dinkster.sparse-support.v1\0"


def sparse_support_id(coordinates: torch.Tensor) -> str:
    """Hash coordinate values in a dtype- and device-independent format."""
    if type(coordinates) is not torch.Tensor:
        raise TypeError("sparse coordinates must be an exact torch.Tensor")
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("sparse coordinates must have shape (points, 4)")
    if coordinates.dtype not in _COORDINATE_DTYPES:
        raise TypeError("sparse coordinates must have an integer dtype")
    canonical = coordinates.detach().to(device="cpu", dtype=torch.int64).contiguous()
    digest = hashlib.sha256()
    digest.update(_DIGEST_DOMAIN)
    digest.update(struct.pack("<Q", canonical.shape[0]))
    digest.update(canonical.numpy().astype("<i8", copy=False).tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _validate_coordinate_layout(
    coordinates: torch.Tensor,
    batch_counts: tuple[int, ...],
    resolution: int,
) -> None:
    if coordinates.shape[0] != sum(batch_counts):
        raise ValueError("sparse batch counts must account for every coordinate row")
    offset = 0
    for batch, count in enumerate(batch_counts):
        if not bool(torch.all(coordinates[offset : offset + count, 0] == batch).item()):
            raise ValueError("sparse coordinate rows must be contiguous and match batch counts")
        offset += count
    spatial = coordinates[:, 1:]
    if spatial.numel() and bool(torch.any((spatial < 0) | (spatial >= resolution)).item()):
        raise ValueError("sparse spatial coordinates must be inside the declared resolution")


def make_sparse_support(
    coordinates: torch.Tensor,
    batch_counts: tuple[int, ...],
    resolution: int,
    origin: tuple[float, float, float],
    voxel_size: tuple[float, float, float],
) -> SparseSupport[torch.Tensor]:
    """Create support only after authenticating coordinate order and bounds."""
    support = SparseSupport(
        coordinates,
        batch_counts,
        resolution,
        origin,
        voxel_size,
        sparse_support_id(coordinates),
    )
    _validate_coordinate_layout(coordinates, batch_counts, resolution)
    return support


def authenticate_sparse_support(value: object) -> SparseSupport[torch.Tensor]:
    """Reject mutated, forged, or non-contiguous torch support."""
    if type(value) is not SparseSupport:
        raise TypeError("sparse support must be an exact SparseSupport")
    support = value
    if type(support.coordinates) is not torch.Tensor:
        raise TypeError("sparse support coordinates must be an exact torch.Tensor")
    _validate_coordinate_layout(support.coordinates, support.batch_counts, support.resolution)
    if sparse_support_id(support.coordinates) != support.support_id:
        raise ValueError("sparse support coordinate digest does not match its authenticated id")
    return support


def pack_sparse_latent(
    support: SparseSupport[torch.Tensor],
    features: torch.Tensor,
) -> SparseLatent[torch.Tensor]:
    """Bind sampled feature rows to authenticated support."""
    authenticated = authenticate_sparse_support(support)
    if type(features) is not torch.Tensor:
        raise TypeError("sparse latent features must be an exact torch.Tensor")
    if not features.is_floating_point():
        raise TypeError("sparse latent features must have a floating-point dtype")
    return SparseLatent(authenticated, features)


def unpack_sparse_latent(
    value: object,
) -> tuple[SparseSupport[torch.Tensor], torch.Tensor]:
    """Return authenticated support and its corresponding feature rows."""
    if type(value) is not SparseLatent:
        raise TypeError("sparse latent must be an exact SparseLatent")
    support = authenticate_sparse_support(value.support)
    if type(value.features) is not torch.Tensor:
        raise TypeError("sparse latent features must be an exact torch.Tensor")
    if value.features.ndim != 2 or value.features.shape[0] != support.point_count:
        raise ValueError("sparse latent features must have one row per support coordinate")
    if not value.features.is_floating_point():
        raise TypeError("sparse latent features must have a floating-point dtype")
    return support, value.features


__all__ = [
    "authenticate_sparse_support",
    "make_sparse_support",
    "pack_sparse_latent",
    "sparse_support_id",
    "unpack_sparse_latent",
]
