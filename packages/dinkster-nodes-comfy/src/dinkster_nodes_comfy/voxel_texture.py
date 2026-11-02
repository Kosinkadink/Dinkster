"""Bake sparse voxel attributes into an existing mesh UV layout."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    MESH_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_mesh,
    materialize_voxel,
    report_progress,
)


def _sample(positions: Any, coords: Any, colors: Any, resolution: int) -> Any:
    import torch

    r = int(resolution)
    try:
        result, valid = _trilinear(positions, coords, colors, r)
    except Exception as error:
        management = importlib.import_module("dinkster_inference.model_management")
        management.raise_non_oom(error)
        # CPU double arithmetic matches the reference's OOM fallback without zero bleed.
        result, valid = _trilinear(
            positions.cpu().double(),
            coords.cpu(),
            colors.cpu().float().double(),
            r,
        )
        result = result.to(device=positions.device, dtype=torch.float32)
        valid = valid.to(positions.device)
    missing = torch.nonzero(~valid, as_tuple=True)[0]
    if missing.numel():
        result[missing] = _nearest(positions[missing], coords, colors, r)
    return result.clamp_(0.0, 1.0)


def _trilinear(positions: Any, coords: Any, colors: Any, r: int) -> tuple[Any, Any]:
    import torch

    dtype = torch.int32 if r**3 <= torch.iinfo(torch.int32).max else torch.int64
    coords = coords.to(device=positions.device, dtype=dtype)
    colors = colors.to(device=positions.device, dtype=positions.dtype)
    keys, order = ((coords[:, 0] * r + coords[:, 1]) * r + coords[:, 2]).sort()
    grid = (positions + 0.5) / (1.0 / r) - 0.5
    base = grid.floor().to(coords.dtype)
    frac = grid - base.to(positions.dtype)
    result = colors.new_zeros((len(positions), colors.shape[1]))
    weight_sum = colors.new_zeros((len(positions), 1))
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                cell = base + base.new_tensor([dx, dy, dz])
                key = (cell[:, 0] * r + cell[:, 1]) * r + cell[:, 2]
                index = torch.searchsorted(keys, key).clamp(max=len(coords) - 1)
                match = ((cell >= 0) & (cell < r)).all(1) & (keys[index] == key)
                weight = torch.where(match, wx * wy * wz, 0.0)[:, None]
                result.add_(colors[order[index]] * weight)
                weight_sum += weight
    valid = weight_sum[:, 0] > 1e-8
    result[valid] /= weight_sum[valid].clamp_min(1e-8)
    return result, valid


def _nearest(positions: Any, coords: Any, colors: Any, r: int) -> Any:
    import torch

    dtype = torch.int32 if r**3 <= torch.iinfo(torch.int32).max else torch.int64
    coords = coords.to(device=positions.device, dtype=dtype)
    colors = colors.to(device=positions.device, dtype=torch.float32)
    keys, order = ((coords[:, 0] * r + coords[:, 1]) * r + coords[:, 2]).sort()
    count = len(positions)
    found = torch.zeros(count, dtype=torch.bool, device=positions.device)
    nearest = torch.zeros(count, dtype=torch.long, device=positions.device)
    for radius in (1, 4):
        missing = torch.nonzero(~found, as_tuple=True)[0]
        if not missing.numel():
            break
        query = positions[missing]
        center = ((query + 0.5) * r - 0.5).round().to(coords.dtype)
        distance = query.new_full((len(query),), 1e30)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    cell = center + center.new_tensor([dx, dy, dz])
                    key = (cell[:, 0] * r + cell[:, 1]) * r + cell[:, 2]
                    index = torch.searchsorted(keys, key).clamp(max=len(coords) - 1)
                    match = ((cell >= 0) & (cell < r)).all(1) & (keys[index] == key)
                    d = (((cell.float() + 0.5) / r - 0.5 - query) ** 2).sum(1)
                    take = match & (d < distance)
                    distance = torch.where(take, d, distance)
                    nearest[missing] = torch.where(take, order[index], nearest[missing])
                    found[missing] |= match
    missing = torch.nonzero(~found, as_tuple=True)[0]
    if 0 < len(missing) <= 8192:
        query = positions[missing]
        distance = query.new_full((len(query),), 1e30)
        centers = (coords.float() + 0.5) / r - 0.5
        chunk = max(1, (1 << 26) // len(query))
        for start in range(0, len(coords), chunk):
            d = (query[:, None] - centers[None, start : start + chunk]).square().sum(-1)
            minimum, index = d.min(1)
            take = minimum < distance
            distance = torch.where(take, minimum, distance)
            nearest[missing] = torch.where(take, index + start, nearest[missing])
    elif len(missing):
        tree = importlib.import_module("scipy.spatial").cKDTree
        centers = ((coords.float() + 0.5) / r - 0.5).cpu().numpy()
        _, index = tree(centers).query(positions[missing].cpu().numpy(), k=1, workers=-1)
        nearest[missing] = torch.as_tensor(index, device=positions.device)
    return colors[nearest]


def _project(positions: Any, mask: Any, vertices: Any, faces: Any) -> Any:
    import torch

    from .mesh_bake_ops import closest_points
    from .mesh_render_ops import build_bvh

    triangles = vertices[faces]
    bvh = build_bvh(triangles)
    size = positions.shape[0]
    stride = max(1, math.ceil(size / 1024.0))
    if stride == 1 or not bool(mask[::stride, ::stride].any()):
        positions[mask] = closest_points(positions[mask], triangles, bvh)
        return positions
    low, low_mask = positions[::stride, ::stride], mask[::stride, ::stride]
    correction = torch.zeros_like(low)
    correction[low_mask] = closest_points(low[low_mask], triangles, bvh) - low[low_mask]
    # Match the reference's exact nearest extension before bilinear upsampling.
    ndi = importlib.import_module("scipy.ndimage")
    indices = ndi.distance_transform_edt(
        ~low_mask.cpu().numpy(), return_distances=False, return_indices=True
    )
    correction = correction[tuple(torch.as_tensor(i, device=positions.device) for i in indices)]
    correction = torch.nn.functional.interpolate(
        correction.permute(2, 0, 1)[None], size=(size, size), mode="bilinear", align_corners=False
    )[0].permute(1, 2, 0)
    positions[mask] += correction[mask]
    return positions


class BakeTextureFromVoxel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="BakeTextureFromVoxel",
            display_name="Bake Texture From Voxel",
            category="3d/texturing",
            description=(
                "Bakes PBR textures onto the mesh's existing UV layout (trilinear-sample the "
                "sparse voxel volume). Does NOT unwrap - connect a UV unwrap node upstream. "
                "Outputs base_color + metallic/roughness grayscale IMAGEs (black if no PBR); "
                "feed them to ApplyTextureToMesh (SAME mesh) for SaveGLB."
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec("voxel_colors", TypeExpr.concrete(VOXEL_TYPE)),
                InputSpec(
                    "texture_size",
                    TypeExpr.concrete(CORE_INT),
                    display_name="resolution",
                    default=2048,
                    widget=NumberWidget(min=64, max=8192),
                    doc="Square UV atlas resolution.",
                ),
                InputSpec(
                    "reference_mesh",
                    TypeExpr.concrete(MESH_TYPE),
                    required=False,
                    doc="Optional dense pre-decimation mesh; back-projects each texel onto its "
                    "true surface before sampling, removing faceted baking on coarse meshes.",
                ),
            ),
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(IMAGE_TYPE))
                for name in ("base_color", "metallic", "roughness")
            ),
        )

    @classmethod
    def execute(
        cls, *, mesh: object, voxel_colors: object, texture_size: int, reference_mesh: object = None
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        from .mesh_bake_ops import fill_gutters, rasterize_uv
        from .mesh_texture import normalize_uvs

        mesh = materialize_mesh(mesh)
        voxel = materialize_voxel(voxel_colors)
        reference = materialize_mesh(reference_mesh) if reference_mesh is not None else None
        if mesh.uvs is None:
            raise ValueError(
                "BakeTextureFromVoxel: input mesh has no UVs. This node bakes onto the mesh's "
                "existing UV layout and never unwraps - connect a UV unwrap node before it."
            )
        device = management.get_torch_device()
        batch, size = mesh.vertices.shape[0], int(texture_size)
        reference_faces = reference if reference is not None else mesh
        face_count = (
            sum(int(count) for count in reference_faces.face_counts)
            if reference_faces.face_counts is not None
            else reference_faces.faces.numel() // 3
        )
        management.free_memory(
            max(size * size * 512, face_count * 512) + management.minimum_inference_memory(), device
        )
        output: dict[str, list[Any]] = {
            name: [] for name in ("base_color", "metallic", "roughness")
        }
        for i in range(batch):
            n = (
                int(mesh.vertex_counts[i])
                if mesh.vertex_counts is not None
                else mesh.vertices.shape[1]
            )
            f = int(mesh.face_counts[i]) if mesh.face_counts is not None else mesh.faces.shape[1]
            coords, colors = voxel.data, voxel.voxel_colors
            if coords.shape[-1] == 4:
                selected = coords[:, 0].long() == i
                coords, colors = coords[selected, 1:], colors[selected]
            maps = [torch.zeros((size, size, 3)) for _ in range(3)]
            if f and len(coords):
                vertices, faces = (
                    mesh.vertices[i, :n].to(device).float(),
                    mesh.faces[i, :f].to(device).long(),
                )
                uv = mesh.uvs[i, :n] if mesh.uvs.ndim == 3 else mesh.uvs[:n]
                uv = torch.from_numpy(normalize_uvs(uv.detach().cpu().numpy())).to(device)
                index, bary, mask = rasterize_uv(faces, uv, size)
                positions = vertices.new_zeros((size, size, 3))
                positions[mask] = (vertices[faces[index[mask]]] * bary[mask, :, None]).sum(1)
                if reference is not None and bool(mask.any()):
                    ri = i if reference.vertices.shape[0] > 1 else 0
                    rn = (
                        int(reference.vertex_counts[ri])
                        if reference.vertex_counts is not None
                        else reference.vertices.shape[1]
                    )
                    rf = (
                        int(reference.face_counts[ri])
                        if reference.face_counts is not None
                        else reference.faces.shape[1]
                    )
                    positions = _project(
                        positions,
                        mask,
                        reference.vertices[ri, :rn].to(device).float(),
                        reference.faces[ri, :rf].to(device).long(),
                    )
                attributes = vertices.new_zeros((size, size, colors.shape[-1]))
                if bool(mask.any()):
                    attributes[mask] = _sample(positions[mask], coords, colors, voxel.resolution)
                attributes = fill_gutters(attributes, mask)
                maps[0] = attributes[..., :3].float().cpu()
                if colors.shape[-1] >= 5:
                    maps[1] = attributes[..., 3:4].expand(-1, -1, 3).float().cpu()
                    maps[2] = attributes[..., 4:5].expand(-1, -1, 3).float().cpu()
            for name, value in zip(output, maps, strict=True):
                output[name].append(value)
            report_progress(i + 1, batch)
        return cls.outputs(**{name: torch.stack(values) for name, values in output.items()})
