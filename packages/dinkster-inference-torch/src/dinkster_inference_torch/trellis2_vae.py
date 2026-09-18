"""Native TRELLIS.2 structure, shape, and texture decoders."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from dinkster_inference import (
    PBR_CHANNELS,
    SUBDIVISION_CHANNELS,
    SparseLatent,
    SparseSubdivisionGuides,
    SparseSupport,
    SparseVolume,
    TriangleMesh,
)

from .memory import get_free_memory
from .operations import CastOperations, Operations, ResidencyRouted
from .ops import cast_weight
from .sparse import make_sparse_support, unpack_sparse_latent


@dataclass
class _Sparse:
    support: SparseSupport[torch.Tensor]
    feats: torch.Tensor
    cache: dict[str, torch.Tensor] = field(default_factory=dict)

    def replace(self, feats: torch.Tensor) -> _Sparse:
        return _Sparse(self.support, feats, self.cache)


def _flat_coordinates(coordinates: torch.Tensor, resolution: int) -> torch.Tensor:
    return (
        coordinates[:, 0].long() * resolution**3
        + coordinates[:, 1].long() * resolution**2
        + coordinates[:, 2].long() * resolution
        + coordinates[:, 3].long()
    )


def _neighbor_map(value: _Sparse, kernel_size: int, dilation: int) -> torch.Tensor:
    cache_key = f"neighbors-{kernel_size}-{dilation}"
    cached = value.cache.get(cache_key)
    if cached is not None:
        return cached
    coordinates = value.support.coordinates.to(value.feats.device)
    resolution = value.support.resolution
    keys = _flat_coordinates(coordinates, resolution)
    sorted_keys, order = keys.sort()
    offsets = torch.tensor(
        [
            (x, y, z)
            for x in range(kernel_size)
            for y in range(kernel_size)
            for z in range(kernel_size)
        ],
        dtype=torch.long,
        device=value.feats.device,
    )
    offsets = (offsets - kernel_size // 2) * dilation
    volume = offsets.shape[0]
    neighbors = torch.empty(
        (coordinates.shape[0], volume), dtype=torch.int32, device=value.feats.device
    )
    chunk = max(1, min(coordinates.shape[0], int(0.5 * 1024**3 / (volume * 40))))
    for start in range(0, coordinates.shape[0], chunk):
        end = min(start + chunk, coordinates.shape[0])
        points = coordinates[start:end, None, 1:].long() + offsets[None]
        valid = ((points >= 0) & (points < resolution)).all(dim=-1)
        flat = (
            coordinates[start:end, None, 0].long() * resolution**3
            + points[..., 0] * resolution**2
            + points[..., 1] * resolution
            + points[..., 2]
        )
        flat = torch.where(valid, flat, torch.full_like(flat, -1)).reshape(-1)
        positions = torch.searchsorted(sorted_keys, flat)
        safe = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
        found = positions < sorted_keys.numel()
        if sorted_keys.numel():
            found &= sorted_keys[safe] == flat
        found_indices = found.nonzero(as_tuple=True)[0]
        chunk_neighbors = torch.full_like(positions, -1, dtype=torch.int32)
        chunk_neighbors[found_indices] = order[safe[found_indices]].to(torch.int32)
        neighbors[start:end] = chunk_neighbors.reshape(end - start, volume)
    value.cache[cache_key] = neighbors
    return neighbors


class Trellis2SparseConv3d(ResidencyRouted, torch.nn.Module):
    """Submanifold Conv3d with the checkpoint's [out,k,k,k,in] layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels)
        )
        self.bias = torch.nn.Parameter(torch.empty(out_channels))

    def _convolve(self, value: _Sparse, weight: torch.Tensor, bias: torch.Tensor) -> _Sparse:
        neighbors = _neighbor_map(value, self.kernel_size, self.dilation)
        rows = value.feats.shape[0]
        padded = torch.cat((value.feats, value.feats.new_zeros((1, value.feats.shape[1]))))
        indices = torch.where(neighbors < 0, rows, neighbors)
        output = value.feats.new_empty((rows, weight.shape[0]))
        volume = self.kernel_size**3
        matrix = weight.reshape(weight.shape[0], -1).transpose(0, 1)
        bytes_per_row = volume * value.feats.shape[1] * value.feats.element_size()
        free = get_free_memory(value.feats.device).free_total
        chunk_budget = max(0.25 * 1024**3, min(free * 0.2, 2.0 * 1024**3))
        chunk = max(1, min(rows, int(chunk_budget // max(bytes_per_row, 1))))
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            gathered = padded[indices[start:end]].reshape(end - start, -1)
            output[start:end] = gathered @ matrix
        output.add_(bias)
        return value.replace(output)

    def forward(self, value: _Sparse) -> _Sparse:
        binding = self._offloaded_residency()
        if binding is None:
            return self._convolve(
                value,
                cast_weight(self.weight, dtype=value.feats.dtype, device=value.feats.device),
                cast_weight(self.bias, dtype=value.feats.dtype, device=value.feats.device),
            )
        with binding.lease() as lease:
            return self._convolve(
                value,
                lease.get("weight", dtype=value.feats.dtype),
                lease.get("bias", dtype=value.feats.dtype),
            )


class Trellis2SparseConvNeXt(torch.nn.Module):
    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv = Trellis2SparseConv3d(channels, channels)
        self.norm = operations.layer_norm(channels, eps=1e-6)
        self.mlp = torch.nn.Sequential(
            operations.linear(channels, channels * 4),
            torch.nn.SiLU(inplace=True),
            operations.linear(channels * 4, channels),
        )

    def forward(self, value: _Sparse) -> _Sparse:
        features = self.mlp(self.norm(self.conv(value).feats))
        return value.replace(features + value.feats.to(features))


def _upsample_sparse(value: _Sparse, subdivision: torch.Tensor) -> _Sparse:
    selected = subdivision.nonzero(as_tuple=False)
    source_rows, child = selected[:, 0], selected[:, 1]
    coordinates = value.support.coordinates.to(value.feats.device)[source_rows].clone()
    coordinates[:, 1:] *= 2
    coordinates[:, 1] += child % 2
    coordinates[:, 2] += child // 2 % 2
    coordinates[:, 3] += child // 4 % 2
    features = value.feats.reshape(value.feats.shape[0], 8, -1)[source_rows, child]
    counts: list[int] = []
    for batch_slice in value.support.batch_slices:
        counts.append(int(subdivision[batch_slice].sum().item()))
    support = make_sparse_support(
        coordinates,
        tuple(counts),
        value.support.resolution * 2,
        value.support.origin,
        (
            value.support.voxel_size[0] / 2,
            value.support.voxel_size[1] / 2,
            value.support.voxel_size[2] / 2,
        ),
    )
    return _Sparse(support, features)


class Trellis2SparseUpsampleBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        *,
        predict_subdivision: bool,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.norm1 = operations.layer_norm(channels, eps=1e-6)
        self.norm2 = operations.layer_norm(out_channels, eps=1e-6, elementwise_affine=False)
        self.conv1 = Trellis2SparseConv3d(channels, out_channels * 8)
        self.conv2 = Trellis2SparseConv3d(out_channels, out_channels)
        self.to_subdiv = operations.linear(channels, 8) if predict_subdivision else None

    def forward(
        self, value: _Sparse, guide: torch.Tensor | None = None
    ) -> tuple[_Sparse, _Sparse | None]:
        subdivision = self.to_subdiv(value.feats) if self.to_subdiv is not None else guide
        assert subdivision is not None
        hidden = value.replace(F.silu(self.norm1(value.feats), inplace=True))
        hidden = self.conv1(hidden)
        selected = subdivision > 0
        hidden = _upsample_sparse(hidden, selected)
        skip = _upsample_sparse(value, selected)
        hidden = self.conv2(hidden.replace(F.silu(self.norm2(hidden.feats), inplace=True)))
        repeat = self.out_channels // skip.feats.shape[1]
        features = hidden.feats.view(hidden.feats.shape[0], skip.feats.shape[1], repeat)
        features = (features + skip.feats.to(features).unsqueeze(-1)).reshape(
            hidden.feats.shape[0], -1
        )
        predicted = value.replace(subdivision) if self.to_subdiv is not None else None
        return hidden.replace(features), predicted


class Trellis2SparseDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        out_channels: int,
        predict_subdivision: bool,
        operations: Operations,
        microsoft_split_precision: bool = False,
    ) -> None:
        super().__init__()
        channels = (1024, 512, 256, 128, 64)
        counts = (4, 16, 8, 4, 0)
        self.predict_subdivision = predict_subdivision
        self.microsoft_split_precision = microsoft_split_precision
        boundary_operations = (
            CastOperations(torch.float32) if microsoft_split_precision else operations
        )
        self.from_latent = boundary_operations.linear(32, channels[0])
        self.output_layer = boundary_operations.linear(channels[-1], out_channels)
        stages: list[torch.nn.ModuleList] = []
        for stage, (channel, count) in enumerate(zip(channels, counts, strict=True)):
            blocks: list[torch.nn.Module] = [
                Trellis2SparseConvNeXt(channel, operations=operations) for _ in range(count)
            ]
            if stage < len(channels) - 1:
                blocks.append(
                    Trellis2SparseUpsampleBlock(
                        channel,
                        channels[stage + 1],
                        predict_subdivision=predict_subdivision,
                        operations=operations,
                    )
                )
            stages.append(torch.nn.ModuleList(blocks))
        self.blocks = torch.nn.ModuleList(stages)

    def forward(
        self,
        latent: SparseLatent[torch.Tensor],
        guides: tuple[_Sparse, ...] = (),
    ) -> tuple[_Sparse, tuple[_Sparse, ...]]:
        support, features = unpack_sparse_latent(latent)
        if self.microsoft_split_precision:
            features = features.float()
        value = _Sparse(support, self.from_latent(features))
        if self.microsoft_split_precision:
            value = value.replace(value.feats.half())
        subdivisions: list[_Sparse] = []
        for stage in range(len(self.blocks)):
            blocks = self.blocks[stage]
            assert isinstance(blocks, torch.nn.ModuleList)
            for index in range(len(blocks)):
                block = blocks[index]
                if isinstance(block, Trellis2SparseUpsampleBlock):
                    guide = None if self.predict_subdivision else guides[stage].feats > 0
                    value, predicted = block(value, guide)
                    if predicted is not None:
                        subdivisions.append(predicted)
                else:
                    assert isinstance(block, Trellis2SparseConvNeXt)
                    value = block(value)
        features = value.feats.float() if self.microsoft_split_precision else value.feats
        features = self.output_layer(F.layer_norm(features, features.shape[-1:]))
        return value.replace(features), tuple(subdivisions)

    def upsample(
        self,
        latent: SparseLatent[torch.Tensor],
        upsample_times: int,
    ) -> torch.Tensor:
        support, features = unpack_sparse_latent(latent)
        value = _Sparse(support, self.from_latent(features))
        for stage in range(len(self.blocks)):
            if stage == upsample_times:
                return value.support.coordinates
            blocks = self.blocks[stage]
            assert isinstance(blocks, torch.nn.ModuleList)
            for block in blocks:
                if isinstance(block, Trellis2SparseUpsampleBlock):
                    value, _ = block(value, None)
                else:
                    assert isinstance(block, Trellis2SparseConvNeXt)
                    value = block(value)
        raise ValueError("TRELLIS.2 upsample count exceeds the shape decoder depth")

    def decode_shape(
        self, latent: SparseLatent[torch.Tensor], *, frame: str
    ) -> tuple[
        tuple[TriangleMesh[torch.Tensor], ...],
        SparseSubdivisionGuides[torch.Tensor],
    ]:
        if not self.predict_subdivision:
            raise TypeError("TRELLIS.2 texture decoder cannot decode shape")
        decoded, subdivisions = self(latent)
        return _decoded_shape(decoded, subdivisions, frame=frame)

    def upsample_shape(
        self,
        latent: SparseLatent[torch.Tensor],
        *,
        upsample_times: int,
    ) -> torch.Tensor:
        if not self.predict_subdivision:
            raise TypeError("TRELLIS.2 texture decoder cannot upsample shape")
        return self.upsample(latent, upsample_times)

    def decode_texture(
        self,
        latent: SparseLatent[torch.Tensor],
        subdivisions: SparseSubdivisionGuides[torch.Tensor],
    ) -> SparseVolume[torch.Tensor]:
        if self.predict_subdivision:
            raise TypeError("TRELLIS.2 shape decoder cannot decode texture")
        decoded, _ = self(
            latent,
            tuple(_Sparse(level.support, level.features) for level in subdivisions.levels),
        )
        return SparseVolume(decoded.support, decoded.feats * 0.5 + 0.5, PBR_CHANNELS)


def _pixel_shuffle_3d(value: torch.Tensor) -> torch.Tensor:
    batch, channels, depth, height, width = value.shape
    value = value.reshape(batch, channels // 8, 2, 2, 2, depth, height, width)
    return value.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
        batch, channels // 8, depth * 2, height * 2, width * 2
    )


class Trellis2ChannelLayerNorm(ResidencyRouted, torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(channels))
        self.bias = torch.nn.Parameter(torch.empty(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        channels_last = value.movedim(1, -1)
        binding = self._offloaded_residency()
        if binding is None:
            weight = cast_weight(self.weight, dtype=value.dtype, device=value.device)
            bias = cast_weight(self.bias, dtype=value.dtype, device=value.device)
            return F.layer_norm(channels_last, (value.shape[1],), weight, bias, 1e-6).movedim(-1, 1)
        with binding.lease() as lease:
            return F.layer_norm(
                channels_last,
                (value.shape[1],),
                lease.get("weight", dtype=value.dtype),
                lease.get("bias", dtype=value.dtype),
                1e-6,
            ).movedim(-1, 1)


class Trellis2DenseResBlock(torch.nn.Module):
    def __init__(self, channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.norm1 = Trellis2ChannelLayerNorm(channels)
        self.norm2 = Trellis2ChannelLayerNorm(channels)
        self.conv1 = operations.conv3d(channels, channels, 3, padding=1)
        self.conv2 = operations.conv3d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        return value + self.conv2(F.silu(self.norm2(hidden)))


class Trellis2DenseUpsample(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv = operations.conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _pixel_shuffle_3d(self.conv(value))


class Trellis2StructureDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        operations: Operations,
        microsoft_split_precision: bool = False,
    ) -> None:
        super().__init__()
        self.microsoft_split_precision = microsoft_split_precision
        boundary_operations = (
            CastOperations(torch.float32) if microsoft_split_precision else operations
        )
        self.input_layer = boundary_operations.conv3d(8, 512, 3, padding=1)
        self.middle_block = torch.nn.Sequential(
            Trellis2DenseResBlock(512, operations=operations),
            Trellis2DenseResBlock(512, operations=operations),
        )
        blocks: list[torch.nn.Module] = []
        for index, channels in enumerate((512, 128, 32)):
            blocks.extend(Trellis2DenseResBlock(channels, operations=operations) for _ in range(2))
            if index < 2:
                blocks.append(
                    Trellis2DenseUpsample(channels, (128, 32)[index], operations=operations)
                )
        self.blocks = torch.nn.ModuleList(blocks)
        self.out_layer = torch.nn.Sequential(
            Trellis2ChannelLayerNorm(32),
            torch.nn.SiLU(),
            boundary_operations.conv3d(32, 1, 3, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self.microsoft_split_precision:
            latent = latent.float()
        value = self.input_layer(latent)
        if self.microsoft_split_precision:
            value = value.half()
        value = self.middle_block(value)
        for block in self.blocks:
            value = block(value)
        if self.microsoft_split_precision:
            value = value.float()
        return self.out_layer(value)

    def decode_structure(self, latent: torch.Tensor) -> torch.Tensor:
        return self(latent)


def _lookup_indices(
    coordinates: torch.Tensor, queries: torch.Tensor, resolution: int
) -> torch.Tensor:
    keys = _flat_coordinates(
        torch.cat((torch.zeros_like(coordinates[:, :1]), coordinates), dim=1), resolution
    )
    query_keys = (
        queries[:, 0].long() * resolution**2
        + queries[:, 1].long() * resolution
        + queries[:, 2].long()
    )
    sorted_keys, order = keys.sort()
    positions = torch.searchsorted(sorted_keys, query_keys)
    safe = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
    found = (positions < sorted_keys.numel()) & (sorted_keys[safe] == query_keys)
    result = torch.full_like(positions, -1)
    result[found] = order[safe[found]]
    return result


def _mesh_from_grid(
    coordinates: torch.Tensor,
    offsets: torch.Tensor,
    intersections: torch.Tensor,
    split_weight: torch.Tensor,
    resolution: int,
    frame: str,
) -> TriangleMesh[torch.Tensor]:
    edge_offsets = torch.tensor(
        (
            ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
            ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
            ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
        ),
        device=coordinates.device,
    )
    rows, axes = intersections.nonzero(as_tuple=True)
    connected = coordinates[rows, None] + edge_offsets[axes]
    indices = _lookup_indices(coordinates, connected.reshape(-1, 3), resolution).reshape(-1, 4)
    quads = indices[(indices >= 0).all(dim=1)]
    split_a = torch.tensor((0, 1, 2, 0, 2, 3), device=coordinates.device)
    split_b = torch.tensor((0, 1, 3, 3, 1, 2), device=coordinates.device)
    use_a = split_weight[quads[:, 0]] * split_weight[quads[:, 2]] > (
        split_weight[quads[:, 1]] * split_weight[quads[:, 3]]
    )
    faces = torch.where(use_a[:, None], quads[:, split_a], quads[:, split_b]).reshape(-1, 3)
    vertices = (coordinates.float() + offsets) / resolution - 0.5
    return TriangleMesh(vertices, faces, frame)


def _decoded_shape(
    decoded: _Sparse,
    subdivisions: tuple[_Sparse, ...],
    *,
    frame: str,
) -> tuple[
    tuple[TriangleMesh[torch.Tensor], ...],
    SparseSubdivisionGuides[torch.Tensor],
]:
    vertices = torch.sigmoid(decoded.feats[:, :3]) * 2.0 - 0.5
    intersections = decoded.feats[:, 3:6] > 0
    split_weight = F.softplus(decoded.feats[:, 6])
    meshes = tuple(
        _mesh_from_grid(
            decoded.support.coordinates[batch_slice, 1:].to(decoded.feats.device),
            vertices[batch_slice],
            intersections[batch_slice],
            split_weight[batch_slice],
            decoded.support.resolution,
            frame,
        )
        for batch_slice in decoded.support.batch_slices
    )
    guides = SparseSubdivisionGuides(
        tuple(
            SparseVolume(subdivision.support, subdivision.feats, SUBDIVISION_CHANNELS)
            for subdivision in subdivisions
        )
    )
    return meshes, guides


class Trellis2ShapeVae(torch.nn.Module):
    def __init__(self, *, operations: Operations) -> None:
        super().__init__()
        self.shape_dec = Trellis2SparseDecoder(
            out_channels=7, predict_subdivision=True, operations=operations
        )
        self.struct_dec = Trellis2StructureDecoder(operations=operations)

    def decode_structure(self, latent: torch.Tensor) -> torch.Tensor:
        return self.struct_dec(latent)

    def decode_shape(
        self, latent: SparseLatent[torch.Tensor], *, frame: str
    ) -> tuple[
        tuple[TriangleMesh[torch.Tensor], ...],
        SparseSubdivisionGuides[torch.Tensor],
    ]:
        return self.shape_dec.decode_shape(latent, frame=frame)

    def upsample_shape(
        self,
        latent: SparseLatent[torch.Tensor],
        *,
        upsample_times: int,
    ) -> torch.Tensor:
        return self.shape_dec.upsample(latent, upsample_times)


class Trellis2TextureVae(torch.nn.Module):
    def __init__(self, *, operations: Operations) -> None:
        super().__init__()
        self.txt_dec = Trellis2SparseDecoder(
            out_channels=6, predict_subdivision=False, operations=operations
        )

    def decode_texture(
        self,
        latent: SparseLatent[torch.Tensor],
        subdivisions: SparseSubdivisionGuides[torch.Tensor],
    ) -> SparseVolume[torch.Tensor]:
        return self.txt_dec.decode_texture(latent, subdivisions)


__all__ = [
    "Trellis2ShapeVae",
    "Trellis2SparseDecoder",
    "Trellis2StructureDecoder",
    "Trellis2TextureVae",
]
