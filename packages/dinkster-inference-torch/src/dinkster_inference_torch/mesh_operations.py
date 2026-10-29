"""Native Torch and NumPy mesh operations matching ComfyUI model3d nodes.

Adapted from ComfyUI commit 6338e4bd428247a4a8843496aa98fb7f2a9d3632,
primarily ``comfy_extras/nodes_mesh_postprocess.py``.
"""

# pyright: basic, reportArgumentType=false, reportAssignmentType=false
# pyright: reportAttributeAccessIssue=false, reportMissingTypeStubs=false
# pyright: reportOperatorIssue=false, reportOptionalSubscript=false, reportPrivateUsage=false
# ruff: noqa: B007, B905, E501, E741

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import torch
from dinkster_inference import GIBIBYTE, MEBIBYTE, DenseVoxelGrid, SparseVolume, TriangleMeshBatch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

from .mesh_ops.qem_decimate import (
    QEMConfig,
    _compute_vertex_normals,
    qem_cluster_decimate,
    qem_decimate_simplify,
)
from .mesh_ops.remesh import _point_tri_closest, remesh_narrow_band_dc
from .mesh_ops.uv_unwrap import mesh as _uv_mesh
from .mesh_ops.uv_unwrap import pack as _uv_pack
from .mesh_ops.uv_unwrap import parameterize as _uv_param
from .mesh_ops.uv_unwrap import segment as _uv_seg


def _compute_device(value: torch.Tensor | None = None) -> torch.device:
    if value is not None and value.device.type != "cpu":
        return value.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Progress:
    def update(self, value: int = 1) -> None:
        del value

    def update_absolute(self, value: int, total: int | None = None) -> None:
        del value, total


def _validate_mesh(mesh: TriangleMeshBatch[torch.Tensor], operation: str) -> None:
    if type(mesh) is not TriangleMeshBatch:
        raise TypeError(f"{operation}: mesh must be an exact TriangleMeshBatch")
    vertices = mesh.vertices
    faces = mesh.faces
    if vertices.ndim != 3 or vertices.shape[0] < 1 or vertices.shape[2] != 3:
        raise ValueError(f"{operation}: vertices must have shape (batch, vertices, 3)")
    if faces.ndim != 3 or faces.shape[0] != vertices.shape[0] or faces.shape[2] != 3:
        raise ValueError(f"{operation}: faces must have shape (batch, triangles, 3)")
    batch = vertices.shape[0]
    if mesh.vertex_counts is None:
        vertex_counts = [vertices.shape[1]] * batch
        face_counts = [faces.shape[1]] * batch
    else:
        if mesh.face_counts is None:
            raise ValueError(f"{operation}: vertex and face counts must be provided together")
        if mesh.vertex_counts.shape != (batch,) or mesh.face_counts.shape != (batch,):
            raise ValueError(f"{operation}: mesh counts must have shape ({batch},)")
        vertex_counts = [int(value) for value in mesh.vertex_counts.tolist()]
        face_counts = [int(value) for value in mesh.face_counts.tolist()]
    for index, (vertex_count, face_count) in enumerate(zip(vertex_counts, face_counts)):
        if not 0 <= vertex_count <= vertices.shape[1]:
            raise ValueError(
                f"{operation}: vertex count {vertex_count} for batch {index} is outside "
                f"[0, {vertices.shape[1]}]"
            )
        if not 0 <= face_count <= faces.shape[1]:
            raise ValueError(
                f"{operation}: face count {face_count} for batch {index} is outside "
                f"[0, {faces.shape[1]}]"
            )
        item_faces = faces[index, :face_count]
        if item_faces.numel() and (
            int(item_faces.min().item()) < 0 or int(item_faces.max().item()) >= vertex_count
        ):
            raise ValueError(
                f"{operation}: face indices for batch {index} must be in [0, {vertex_count})"
            )
    for name, widths in (
        ("uvs", (2,)),
        ("vertex_colors", (3, 4)),
        ("normals", (3,)),
        ("tangents", (4,)),
    ):
        value = getattr(mesh, name)
        if value is None:
            continue
        if value.ndim != 3 or value.shape[:2] != vertices.shape[:2] or value.shape[2] not in widths:
            expected = " or ".join(str(width) for width in widths)
            raise ValueError(f"{operation}: {name} must have shape (batch, vertices, {expected})")


def get_mesh_batch_item(
    mesh: TriangleMeshBatch[torch.Tensor], index: int
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    vertex_count = (
        mesh.vertices.shape[1]
        if mesh.vertex_counts is None
        else int(mesh.vertex_counts[index].item())
    )
    face_count = (
        mesh.faces.shape[1] if mesh.face_counts is None else int(mesh.face_counts[index].item())
    )
    return (
        mesh.vertices[index, :vertex_count],
        mesh.faces[index, :face_count],
        None if mesh.vertex_colors is None else mesh.vertex_colors[index, :vertex_count],
        None if mesh.uvs is None else mesh.uvs[index, :vertex_count],
        None if mesh.normals is None else mesh.normals[index, :vertex_count],
    )


def pack_variable_mesh_batch(
    vertices: list[torch.Tensor],
    faces: list[torch.Tensor],
    colors: list[torch.Tensor] | None = None,
    uvs: list[torch.Tensor] | None = None,
    normals: list[torch.Tensor] | None = None,
    tangents: list[torch.Tensor] | None = None,
    **attributes: Any,
) -> TriangleMeshBatch[torch.Tensor]:
    batch = len(vertices)
    max_vertices = max((value.shape[0] for value in vertices), default=0)
    max_faces = max((value.shape[0] for value in faces), default=0)

    def padded(values: list[torch.Tensor], length: int) -> torch.Tensor:
        result = values[0].new_zeros((batch, length, values[0].shape[1]))
        for index, value in enumerate(values):
            result[index, : value.shape[0]] = value
        return result

    return TriangleMeshBatch(
        vertices=padded(vertices, max_vertices),
        faces=padded(faces, max_faces),
        vertex_colors=None if colors is None else padded(colors, max_vertices),
        uvs=None if uvs is None else padded(uvs, max_vertices),
        normals=None if normals is None else padded(normals, max_vertices),
        tangents=None if tangents is None else padded(tangents, max_vertices),
        vertex_counts=torch.tensor(
            [value.shape[0] for value in vertices], device=vertices[0].device
        ),
        face_counts=torch.tensor([value.shape[0] for value in faces], device=faces[0].device),
        **attributes,
    )


def _mesh_face_count(mesh):
    if mesh.face_counts is not None:
        return sum(int(count) for count in mesh.face_counts)
    if isinstance(mesh.faces, list):
        return sum(int(faces.shape[0]) for faces in mesh.faces)
    return int(mesh.faces.numel() // 3)


def _prepare_gpu_mesh_processing(device, memory_required):
    del device, memory_required


def paint_mesh_with_voxels(mesh, voxel_coords, voxel_colors, resolution):
    """Paint a mesh using nearest-neighbor colors from a sparse voxel field."""
    device = _compute_device(mesh.vertices)

    origin = torch.tensor([-0.5, -0.5, -0.5], device=device)
    voxel_size = 1.0 / resolution

    voxel_pos = voxel_coords.to(device).float() * voxel_size + origin
    verts = mesh.vertices.to(device).squeeze(0)
    voxel_colors = voxel_colors.to(device)

    voxel_pos_np = voxel_pos.cpu().numpy()
    verts_np = verts.cpu().numpy()

    tree = cKDTree(voxel_pos_np)
    _, nearest_idx_np = tree.query(verts_np, k=1, workers=-1)

    nearest_idx = torch.from_numpy(nearest_idx_np).long().to(voxel_colors.device)
    v_colors = voxel_colors[nearest_idx]
    # Voxel field may carry full PBR; vertex colors use only base_color RGB.
    if v_colors.shape[-1] > 3:
        v_colors = v_colors[:, :3]

    srgb_colors = v_colors.clamp(0, 1)

    # to Linear RGB (required for GLTF)
    linear_colors = torch.pow(srgb_colors, 2.2)

    final_colors = linear_colors.unsqueeze(0)

    out_mesh = copy.deepcopy(mesh)
    out_mesh.vertex_colors = final_colors

    return out_mesh


def paint_mesh_default_colors(mesh):
    out_mesh = copy.copy(mesh)
    vertex_count = mesh.vertices.shape[1]
    out_mesh.vertex_colors = mesh.vertices.new_zeros((1, vertex_count, 3))
    return out_mesh


def _rasterize_uv_barycentric(faces_np, uvs_np, texture_size):
    """Rasterize the mesh in UV space (tiled point-in-triangle, pure torch). Returns per-texel
    face index [H,W], barycentric coords [H,W,3] and coverage mask [H,W], on the torch device.
    Interpolate any per-vertex attribute from these with _interp_vertex_attr."""
    dev = _compute_device()
    H = W = int(texture_size)
    face_idx = torch.zeros((H, W), dtype=torch.long, device=dev)
    bary = torch.zeros((H, W, 3), device=dev)
    cov = torch.zeros((H, W), dtype=torch.bool, device=dev)
    if faces_np.shape[0] == 0:
        return face_idx, bary, cov

    uvs = torch.from_numpy(np.ascontiguousarray(uvs_np, dtype=np.float32)).to(dev)
    faces = torch.from_numpy(np.ascontiguousarray(faces_np).astype(np.int64)).to(dev)

    # GL convention: window coord = uv * resolution, coverage tested at texel centre.
    tri_uv = (uvs * float(W))[faces]  # [F,3,2]
    x0, y0 = tri_uv[:, 0, 0], tri_uv[:, 0, 1]
    x1, y1 = tri_uv[:, 1, 0], tri_uv[:, 1, 1]
    x2, y2 = tri_uv[:, 2, 0], tri_uv[:, 2, 1]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegen = denom.abs() > 1e-20

    xmin = torch.minimum(torch.minimum(x0, x1), x2).floor().clamp_(0, W - 1).long()
    xmax = torch.maximum(torch.maximum(x0, x1), x2).ceil().clamp_(0, W - 1).long()
    ymin = torch.minimum(torch.minimum(y0, y1), y2).floor().clamp_(0, H - 1).long()
    ymax = torch.maximum(torch.maximum(y0, y1), y2).ceil().clamp_(0, H - 1).long()

    # Tile so point-in-triangle only runs over the triangles whose bbox hits the tile.
    TILE = 64
    eps = 1e-6
    for ty in range(0, H, TILE):
        ty_end = min(ty + TILE, H)
        for tx in range(0, W, TILE):
            tx_end = min(tx + TILE, W)
            tri_mask = nondegen & (xmin < tx_end) & (xmax >= tx) & (ymin < ty_end) & (ymax >= ty)
            if not tri_mask.any():
                continue
            idx = torch.nonzero(tri_mask, as_tuple=True)[0]
            ys = torch.arange(ty, ty_end, dtype=torch.float32, device=dev) + 0.5
            xs = torch.arange(tx, tx_end, dtype=torch.float32, device=dev) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [th,tw]
            sx0, sy0 = x0[idx][:, None, None], y0[idx][:, None, None]
            sx1, sy1 = x1[idx][:, None, None], y1[idx][:, None, None]
            sx2, sy2 = x2[idx][:, None, None], y2[idx][:, None, None]
            sden = denom[idx][:, None, None]
            b0 = ((sy1 - sy2) * (xx - sx2) + (sx2 - sx1) * (yy - sy2)) / sden
            b1 = ((sy2 - sy0) * (xx - sx2) + (sx0 - sx2) * (yy - sy2)) / sden
            b2 = 1.0 - b0 - b1
            inside = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)  # [K,th,tw]
            if not inside.any():
                continue
            hit = inside.any(dim=0)  # [th,tw]
            sel = inside.int().argmax(dim=0)  # [th,tw] first covering local tri
            bsel = torch.stack(
                [
                    b0.gather(0, sel[None]).squeeze(0),
                    b1.gather(0, sel[None]).squeeze(0),
                    b2.gather(0, sel[None]).squeeze(0),
                ],
                dim=-1,
            )  # [th,tw,3]
            face_idx[ty:ty_end, tx:tx_end][hit] = idx[sel][hit]  # slice is a view -> writes through
            bary[ty:ty_end, tx:tx_end][hit] = bsel[hit]
            cov[ty:ty_end, tx:tx_end] |= hit

    return face_idx, bary, cov


def _interp_vertex_attr(attr_v, faces, face_idx, bary, mask):
    """Interpolate a per-vertex attribute [N,C] into a [H,W,C] map via a rasterized
    (face_idx, bary, mask). Uncovered texels stay zero."""
    H, W = mask.shape
    out = torch.zeros((H, W, attr_v.shape[1]), device=attr_v.device, dtype=attr_v.dtype)
    if mask.any():
        vtri = attr_v[faces[face_idx[mask]]]  # [K,3,C]
        out[mask] = (bary[mask][:, :, None] * vtri).sum(1)
    return out


def _bake_position_map(verts_np, faces_np, uvs_np, texture_size):
    """Barycentric-interpolate a per-vertex vec3 (world position, or any vec3 e.g. normals)
    at each covered texel. Returns (attr_map [H,W,3] float32, mask [H,W] bool)."""
    dev = _compute_device()
    H = W = int(texture_size)
    if faces_np.shape[0] == 0:
        return np.zeros((H, W, 3), dtype=np.float32), np.zeros((H, W), dtype=bool)

    face_idx, bary, mask = _rasterize_uv_barycentric(faces_np, uvs_np, texture_size)
    verts = torch.from_numpy(np.ascontiguousarray(verts_np, dtype=np.float32)).to(dev)
    faces = torch.from_numpy(np.ascontiguousarray(faces_np).astype(np.int64)).to(dev)
    attr = _interp_vertex_attr(verts, faces, face_idx, bary, mask)
    del face_idx, bary, verts, faces
    return attr.cpu().numpy(), mask.cpu().numpy()


def _trilinear_sample_sparse(positions, voxel_coords_np, color_np, resolution):
    """Normalized trilinear over a SPARSE voxel field (only occupied corners of the 8,
    renormalized; matches official o_voxel.to_glb but without dense-volume zero-bleed).
    Returns (vals [K,C] float64, ok [K] bool); ok=False where no corner is occupied."""
    R = int(resolution)
    origin = -0.5
    voxel_size = 1.0 / R
    # Cell-CENTER convention: coord c sits at origin+(c+0.5)*voxel_size (matches
    # official grid_sample_3d); the -0.5 puts integer gc on centres so the 8 corners
    # bracket the query (omitting it bleeds colour at boundaries/thin features).
    gc = (positions.astype(np.float64) - origin) / voxel_size - 0.5
    base = np.floor(gc).astype(np.int64)
    frac = gc - base

    vc = voxel_coords_np.astype(np.int64)
    occ_keys = (vc[:, 0] * R + vc[:, 1]) * R + vc[:, 2]
    order = np.argsort(occ_keys)
    occ_sorted = occ_keys[order]

    K = positions.shape[0]
    C = color_np.shape[1]
    acc = np.zeros((K, C), dtype=np.float64)
    wsum = np.zeros((K, 1), dtype=np.float64)
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                cx = base[:, 0] + dx
                cy = base[:, 1] + dy
                cz = base[:, 2] + dz
                inb = (cx >= 0) & (cx < R) & (cy >= 0) & (cy < R) & (cz >= 0) & (cz < R)
                key = (cx * R + cy) * R + cz
                ins = np.clip(np.searchsorted(occ_sorted, key), 0, len(occ_sorted) - 1)
                matched = inb & (occ_sorted[ins] == key)
                idx = order[ins]  # garbage where !matched
                w = np.where(matched, wx * wy * wz, 0.0)[:, None]
                acc += w * color_np[idx]  # w=0 cancels garbage rows
                wsum += w
    ok = wsum[:, 0] > 1e-8
    vals = np.zeros((K, C), dtype=np.float64)
    vals[ok] = acc[ok] / wsum[ok]
    return vals, ok


def _trilinear_sample_sparse_gpu(positions, voxel_coords_np, color_np, resolution):
    """GPU port of `_trilinear_sample_sparse`. Returns (vals [K,C] float32, ok [K] bool)."""
    dev = _compute_device()
    R = int(resolution)
    origin = -0.5
    voxel_size = 1.0 / R
    index_dtype = torch.int32 if R**3 <= torch.iinfo(torch.int32).max else torch.int64
    P = torch.from_numpy(np.ascontiguousarray(positions)).to(dev).float()
    VC = torch.from_numpy(np.ascontiguousarray(voxel_coords_np)).to(device=dev, dtype=index_dtype)
    col = torch.from_numpy(np.ascontiguousarray(color_np)).to(dev).float()
    K, C = P.shape[0], col.shape[1]
    M = VC.shape[0]
    # Cell-CENTER convention (see NumPy path): -0.5 to bracket the query.
    gc = (P - origin) / voxel_size - 0.5
    base = torch.floor(gc).to(index_dtype)
    frac = gc - base.float()
    key = (VC[:, 0] * R + VC[:, 1]) * R + VC[:, 2]
    skey, order = key.sort()
    acc = torch.zeros((K, C), device=dev)
    wsum = torch.zeros((K, 1), device=dev)
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                cx = base[:, 0] + dx
                cy = base[:, 1] + dy
                cz = base[:, 2] + dz
                inb = (cx >= 0) & (cx < R) & (cy >= 0) & (cy < R) & (cz >= 0) & (cz < R)
                qk = (cx * R + cy) * R + cz
                ins = torch.searchsorted(skey, qk).clamp(max=M - 1)
                matched = inb & (skey[ins] == qk)
                idx = order[ins]  # garbage where !matched
                w = torch.where(matched, wx * wy * wz, torch.zeros_like(wx))[:, None]
                weighted = col[idx]
                weighted.mul_(w)  # w=0 cancels garbage rows
                acc.add_(weighted)
                wsum += w
    ok = wsum[:, 0] > 1e-8
    vals = torch.zeros((K, C), device=dev)
    vals[ok] = acc[ok] / wsum[ok].clamp_min(1e-8)
    return vals.cpu().numpy(), ok.cpu().numpy()


# Above this many grid-scan stragglers, the O(N*M) GPU brute force (and its chunk loop)
# is slower than a one-off cKDTree build, so the nearest fallback defers them to cKDTree.
_BRUTE_NEAREST_MAX = 8192


def _nearest_voxel_sample_gpu(positions, voxel_coords_np, color_np, resolution):
    """GPU nearest-occupied-voxel lookup via sorted-key grid scan. Returns (vals [K,C]
    float32, found [K] bool); `found` is False for stragglers left to the caller's cKDTree."""
    dev = _compute_device()
    R = int(resolution)
    index_dtype = torch.int32 if R**3 <= torch.iinfo(torch.int32).max else torch.int64
    P = torch.from_numpy(np.ascontiguousarray(positions)).to(dev).float()
    VC = torch.from_numpy(np.ascontiguousarray(voxel_coords_np)).to(device=dev, dtype=index_dtype)
    col = torch.from_numpy(np.ascontiguousarray(color_np)).to(dev).float()
    M, K = VC.shape[0], P.shape[0]
    key = (VC[:, 0] * R + VC[:, 1]) * R + VC[:, 2]
    skey, order = key.sort()

    def _search(idx, radius):
        """Nearest occupied voxel within +/-radius cells, for query subset P[idx]."""
        Ps = P[idx]
        # Cell-CENTER convention: nearest coord = round((p+0.5)*R-0.5) (matches official).
        rc = ((Ps + 0.5) * R - 0.5).round().to(index_dtype)
        n = idx.shape[0]
        bd = torch.full((n,), 1e30, device=dev)
        bi = torch.zeros(n, dtype=torch.long, device=dev)
        fnd = torch.zeros(n, dtype=torch.bool, device=dev)
        rng = range(-radius, radius + 1)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    cc = rc + torch.tensor([dx, dy, dz], dtype=index_dtype, device=dev)
                    inb = ((cc >= 0) & (cc < R)).all(1)
                    qk = (cc[:, 0] * R + cc[:, 1]) * R + cc[:, 2]
                    ins = torch.searchsorted(skey, qk).clamp(max=M - 1)
                    match = inb & (skey[ins] == qk)
                    dd = (((cc.float() + 0.5) / R - 0.5 - Ps) ** 2).sum(1)
                    upd = match & (dd < bd)
                    bd = torch.where(upd, dd, bd)
                    bi = torch.where(upd, order[ins], bi)
                    fnd |= match
        return bi, fnd

    def _brute_nearest(idx):
        """Exact nearest occupied voxel for the few grid-scan stragglers, chunked GPU
        brute force (avoids a seconds-long cKDTree build over all M voxels)."""
        Ps = P[idx]  # [N,3] world
        N = Ps.shape[0]
        vox_pos = (VC.float() + 0.5) / R - 0.5  # [M,3] voxel centres
        best_d = torch.full((N,), 1e30, device=dev)
        best_j = torch.zeros(N, dtype=torch.long, device=dev)
        # Bound the N*chunk matrix to ~64M elements.
        chunk = max(1, (1 << 26) // max(1, N))
        for s in range(0, M, chunk):
            vc = vox_pos[s : s + chunk]  # [B,3]
            dd = (Ps[:, None, :] - vc[None, :, :]).pow(2).sum(-1)  # [N,B]
            md, mj = dd.min(1)
            upd = md < best_d
            best_d = torch.where(upd, md, best_d)
            best_j = torch.where(upd, mj + s, best_j)
        return best_j

    all_idx = torch.arange(K, device=dev)
    best_i = torch.zeros(K, dtype=torch.long, device=dev)
    found = torch.zeros(K, dtype=torch.bool, device=dev)
    # Pass 1: radius 1 over everything; Pass 2: radius 4 on misses; Pass 3: brute force.
    bi1, fnd1 = _search(all_idx, 1)
    best_i[all_idx] = bi1
    found[all_idx] = fnd1
    miss = torch.nonzero(~found, as_tuple=True)[0]
    if miss.numel() > 0:
        bi2, fnd2 = _search(miss, 4)
        best_i[miss] = bi2
        found[miss] = fnd2
    # Pass 3: stragglers >4 cells from any voxel. A handful -> GPU brute force; many
    # (coarse mesh, texels far from the voxel shell) -> leave unfound for the caller's
    # cKDTree, since brute force is O(N*M) and its chunk loop blows up at large N.
    miss2 = torch.nonzero(~found, as_tuple=True)[0]
    if 0 < miss2.numel() <= _BRUTE_NEAREST_MAX:
        best_i[miss2] = _brute_nearest(miss2)
        found[miss2] = True
    vals = col[best_i]
    return vals.cpu().numpy(), found.cpu().numpy()


def _sample_voxel_attrs_per_texel(position_map, mask, voxel_coords, voxel_colors, resolution):
    """Sample all voxel attribute channels at every masked texel. Returns (H,W,C)
    float32 in [0,1] (C = feature width: 3 color, 6 PBR). Normalized trilinear over
    occupied voxels (matches official), nearest fallback where all 8 corners empty."""
    H, W, _ = position_map.shape
    color_np = voxel_colors.detach().cpu().numpy().astype(np.float32)
    C = color_np.shape[-1]
    out = np.zeros((H, W, C), dtype=np.float32)
    if not mask.any():
        return out

    coords_np = voxel_coords.detach().cpu().numpy()
    valid_positions = position_map[mask]

    def _nearest(query):
        # Grid scan + small-N brute tail (on the compute device). Only a large count of far
        # stragglers (coarse mesh, or a surface off the voxel shell) is left unfound -> resolve
        # those with one cKDTree, since GPU brute force is O(N*M) and blows up at large N.
        vals, found = _nearest_voxel_sample_gpu(query, coords_np, color_np, resolution)
        if not found.all():
            origin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
            voxel_size = 1.0 / float(resolution)
            voxel_pos = (coords_np.astype(np.float32) + 0.5) * voxel_size + origin
            tree = cKDTree(voxel_pos)
            _, nearest_idx = tree.query(query[~found], k=1, workers=-1)
            vals[~found] = color_np[nearest_idx]
        return vals

    try:
        vals, ok = _trilinear_sample_sparse_gpu(valid_positions, coords_np, color_np, resolution)
    except torch.OutOfMemoryError as e:
        logging.warning(
            f"[BakeTextureFromVoxel] GPU trilinear ran out of memory ({e}); falling back to CPU"
        )
        vals, ok = _trilinear_sample_sparse(valid_positions, coords_np, color_np, resolution)
    if not ok.all():
        vals[~ok] = _nearest(valid_positions[~ok])  # no occupied neighbour
    np.clip(vals, 0.0, 1.0, out=vals)
    out[mask] = vals
    return out


def _msb_int64(x):
    """floor(log2(x)) elementwise for int64 x >= 1 (bit-search, no float)."""
    r = torch.zeros_like(x)
    xx = x.clone()
    for s in (32, 16, 8, 4, 2, 1):
        sh = xx >> s
        m = sh > 0
        r = torch.where(m, r + s, r)
        xx = torch.where(m, sh, xx)
    return r


def _morton_expand21(v):
    """Spread the low 21 bits of v across every 3rd bit (for a 63-bit Morton code)."""
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def _build_triangle_bvh(tri):
    """Linear BVH (Karras 2012) over triangle AABBs, pure torch, no external deps
    (the cuMesh approach, in torch). Internal nodes 0..T-2; leaves encoded LEAF+i,
    leaf i holds triangle order[i]. Returns dict(LEAF, left, right, nmin, nmax over
    2T entries, order, T)."""
    dev = tri.device
    T = tri.shape[0]
    amin = tri.amin(1)
    amax = tri.amax(1)
    cent = (amin + amax) * 0.5
    lo = cent.amin(0)
    hi = cent.amax(0)
    span = (hi - lo).clamp_min(1e-12)
    q = (((cent - lo) / span).clamp(0, 1) * float((1 << 21) - 1)).long()
    morton = (
        _morton_expand21(q[:, 0]) << 2 | _morton_expand21(q[:, 1]) << 1 | _morton_expand21(q[:, 2])
    ).long()
    order_long = torch.argsort(morton)
    msort = morton[order_long]
    order = order_long.to(torch.int32)

    # delta(i,j): common-prefix length of (morton, index) keys of leaves i,j (index
    # breaks ties so duplicate codes still split); -1 if OOB.
    def delta(i, j):
        ok = (j >= 0) & (j < T)
        jj = j.clamp(0, T - 1)
        x = msort[i] ^ msort[jj]
        same = x == 0
        cp = torch.where(same, torch.full_like(x, 63), 62 - _msb_int64(x.clamp_min(1)))
        xi = i ^ jj
        cpi = torch.where(xi == 0, torch.full_like(x, 32), 31 - _msb_int64(xi.clamp_min(1)))
        return torch.where(
            ok, cp + torch.where(same, cpi, torch.zeros_like(cp)), torch.full_like(x, -1)
        )

    I = torch.arange(T - 1, dtype=torch.int32, device=dev)
    dplus = delta(I, I + 1)
    dminus = delta(I, I - 1)
    direction = torch.where(dplus >= dminus, torch.ones_like(I), -torch.ones_like(I))
    dmin = torch.minimum(dplus, dminus)
    # range length: exponential probe then binary search
    lmax = torch.full_like(I, 2)
    while True:
        cond = delta(I, I + lmax * direction) > dmin
        if not bool(cond.any()):
            break
        lmax = torch.where(cond, lmax * 2, lmax)
        if int(lmax.max()) > 2 * T:
            break
    l = torch.zeros_like(I)
    t = lmax.clone()
    while True:
        t = t // 2
        if int(t.max()) == 0:
            break
        cond = delta(I, I + (l + t) * direction) > dmin
        l = torch.where(cond, l + t, l)
    j = I + l * direction
    first = torch.minimum(I, j)
    last = torch.maximum(I, j)
    # split position: binary search on delta within [first, last]
    dnode = delta(first, last)
    s = torch.zeros_like(I)
    div = torch.full_like(I, 2)
    rng = last - first
    while True:
        step = (rng + div - 1) // div
        cond = delta(first, (first + s + step).clamp(max=T - 1)) > dnode
        s = torch.where(cond, s + step, s)
        if int(step.max()) <= 1:
            cond1 = delta(first, (first + s + 1).clamp(max=T - 1)) > dnode
            s = torch.where(cond1, s + 1, s)
            break
        div = div * 2
    gamma = first + s
    LEAF = T
    left = torch.where(gamma == first, LEAF + gamma, gamma)
    right = torch.where(gamma + 1 == last, LEAF + gamma + 1, gamma + 1)

    del cent, lo, hi, span, q, morton, order_long
    del I, dplus, dminus, direction, dmin, lmax, cond, l, t, j
    del first, last, dnode, s, div, rng, step, cond1, gamma
    delta = None
    msort = None

    # node AABBs: leaves seeded, internal unioned bottom-up (~log2(T) passes; cap is a backstop).
    nmin = torch.empty((2 * T, 3), device=dev)
    nmax = torch.empty((2 * T, 3), device=dev)
    nmin[LEAF:] = amin[order]
    nmax[LEAF:] = amax[order]
    del amin, amax
    setm = torch.zeros(2 * T, dtype=torch.bool, device=dev)
    setm[LEAF:] = True
    for _ in range(128):
        need = ~setm[: T - 1]
        if not bool(need.any()):
            break
        idx = torch.nonzero(need, as_tuple=True)[0]
        ii = idx[setm[left[idx]] & setm[right[idx]]]
        if ii.numel() == 0:
            break
        nmin[ii] = torch.minimum(nmin[left[ii]], nmin[right[ii]])
        nmax[ii] = torch.maximum(nmax[left[ii]], nmax[right[ii]])
        setm[ii] = True
    return dict(LEAF=LEAF, left=left, right=right, nmin=nmin, nmax=nmax, order=order, T=T)


def _closest_points_on_mesh_bvh(Q, tri, bvh, max_stack=64, return_face=False):
    """Exact closest surface point per query via per-query BVH stack traversal
    (nearest-child-first), pure torch. Returns [N,3], or (points [N,3], face_idx [N])
    when return_face=True (face_idx indexes `tri`). `max_stack` bounds the stack
    (= tree height); overflow is counted+warned, not silently wrong."""
    dev = Q.device
    N = Q.shape[0]
    LEAF = bvh["LEAF"]
    nmin = bvh["nmin"]
    nmax = bvh["nmax"]
    left = bvh["left"]
    right = bvh["right"]
    order = bvh["order"]
    stack = torch.full((N, max_stack), -1, dtype=torch.int32, device=dev)
    sp = torch.ones(N, dtype=torch.long, device=dev)
    stack[:, 0] = 0
    best = torch.full((N,), 1e30, device=dev)
    bestp = Q.clone()
    bestf = torch.full((N,), -1, dtype=torch.long, device=dev)
    active = torch.arange(N, device=dev)
    overflow = 0

    def aabb_d2(node, q):
        d = (nmin[node] - q).clamp_min(0) + (q - nmax[node]).clamp_min(0)
        return (d * d).sum(-1)

    while active.numel() > 0:
        a = active
        qa = Q[a]
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        within = aabb_d2(node, qa) < best[a]
        isleaf = node >= LEAF
        lv = within & isleaf
        if bool(lv.any()):
            ga = a[lv]
            fidx = order[node[lv] - LEAF]  # triangle index of each leaf
            tt = tri[fidx]
            cp, d2 = _point_tri_closest(qa[lv], tt)
            upd = d2 < best[ga]
            gu = ga[upd]
            best[gu] = d2[upd]
            bestp[gu] = cp[upd]
            bestf[gu] = fidx[upd].long()
        iv = within & ~isleaf
        if bool(iv.any()):
            gi = a[iv]
            qi = qa[iv]
            lc = left[node[iv]]
            rc = right[node[iv]]
            dl = aabb_d2(lc, qi)
            dr = aabb_d2(rc, qi)
            near = torch.where(dl <= dr, lc, rc)
            far = torch.where(dl <= dr, rc, lc)
            s0 = sp[gi]
            stack[gi, s0.clamp(max=max_stack - 1)] = far
            sp[gi] = (s0 + 1).clamp(max=max_stack)
            s1 = sp[gi]
            overflow += int((s1 >= max_stack).sum())
            stack[gi, s1.clamp(max=max_stack - 1)] = near
            sp[gi] = (s1 + 1).clamp(max=max_stack)
        active = a[sp[a] > 0]
    if overflow:
        logging.warning(
            f"[back-project] BVH stack overflow on {overflow} pushes "
            f"(max_stack={max_stack}); a few texels may be slightly off -- "
            f"raise max_stack if this is large."
        )
    if return_face:
        return bestp, bestf
    return bestp


def _back_project_positions(position_map, mask, ref_v, ref_f, max_query_res=1024):
    """Snap covered texels onto the reference mesh's true surface (pure-torch BVH, no
    cumesh/scipy/trimesh) so the voxel field is sampled at full detail, not along flat
    triangle chords. Returns a new position_map.
    """
    if not mask.any():
        return position_map

    dev = _compute_device()
    rv = ref_v.detach().to(dev).float()
    rf = ref_f.detach().to(device=dev, dtype=torch.int32)
    tri = rv[rf]
    bvh = _build_triangle_bvh(tri)
    del rv, rf

    H, W, _ = position_map.shape
    stride = max(1, int(math.ceil(max(H, W) / float(max_query_res))))
    if stride == 1 or not mask[::stride, ::stride].any():
        out = position_map.copy()
        query = torch.from_numpy(np.ascontiguousarray(position_map[mask], dtype=np.float32)).to(dev)
        closest = (
            _closest_points_on_mesh_bvh(query, tri, bvh).cpu().numpy().astype(position_map.dtype)
        )
        del query, tri, bvh
        out[mask] = closest
        return out

    # Low-res correction, then bilinear upsample to full resolution.
    pos_lo = position_map[::stride, ::stride]
    mask_lo = mask[::stride, ::stride]
    Hl, Wl = mask_lo.shape
    corr_lo = np.zeros((Hl, Wl, 3), dtype=np.float32)
    query = torch.from_numpy(np.ascontiguousarray(pos_lo[mask_lo], dtype=np.float32)).to(dev)
    closest = _closest_points_on_mesh_bvh(query, tri, bvh).cpu().numpy().astype(np.float32)
    del query, tri, bvh
    corr_lo[mask_lo] = closest - pos_lo[mask_lo].astype(np.float32)
    inds = ndi.distance_transform_edt(~mask_lo, return_distances=False, return_indices=True)
    corr_lo = corr_lo[tuple(inds)]  # extrapolate into gutter (nearest)
    corr = (
        torch.nn.functional.interpolate(
            torch.from_numpy(np.ascontiguousarray(corr_lo)).permute(2, 0, 1)[None].to(dev),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    out = position_map.copy()
    out[mask] = position_map[mask] + corr[mask]
    return out


def _ray_tri_hit(o, d, tri, tmin, tmax):
    """Moller-Trumbore any-hit per (ray, triangle) pair, double-sided. Returns bool [N]."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = b - a, c - a
    p = torch.cross(d, e2, dim=-1)
    det = (e1 * p).sum(-1)
    inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
    tvec = o - a
    u = (tvec * p).sum(-1) * inv
    q = torch.cross(tvec, e1, dim=-1)
    v = (d * q).sum(-1) * inv
    t = (e2 * q).sum(-1) * inv
    return (det.abs() > 1e-20) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > tmin) & (t < tmax)


def _any_hit_rays_bvh(orig, dirs, tri, bvh, tmin=0.0, tmax=1e30, max_stack=64):
    """Any-hit ray test over the BVH (slab cull + Moller-Trumbore), pure torch. Returns bool
    [N]: True if the ray hits any triangle in (tmin, tmax). Rays early-out once they hit."""
    dev = orig.device
    N = orig.shape[0]
    LEAF = bvh["LEAF"]
    nmin, nmax = bvh["nmin"], bvh["nmax"]
    left, right, order = bvh["left"], bvh["right"], bvh["order"]
    inv = 1.0 / torch.where(dirs.abs() < 1e-20, torch.full_like(dirs, 1e-20), dirs)
    tmaxN = (
        tmax if torch.is_tensor(tmax) else torch.full((N,), float(tmax), device=dev)
    )  # per-ray far bound
    hit = torch.zeros(N, dtype=torch.bool, device=dev)
    # int32 stack: node indices fit in 31 bits and this [N, max_stack] array dominates memory.
    stack = torch.full((N, max_stack), -1, dtype=torch.int32, device=dev)
    sp = torch.ones(N, dtype=torch.long, device=dev)
    stack[:, 0] = 0
    active = torch.arange(N, device=dev)

    def slab(node, o, i, tmx):
        t1 = (nmin[node] - o) * i
        t2 = (nmax[node] - o) * i
        tnear = torch.minimum(t1, t2).amax(-1)
        tfar = torch.maximum(t1, t2).amin(-1)
        return (tfar >= tnear.clamp_min(tmin)) & (tnear <= tmx) & (tfar >= tmin)

    while active.numel() > 0:
        a = active
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        within = slab(node, orig[a], inv[a], tmaxN[a])
        isleaf = node >= LEAF
        lv = within & isleaf
        if bool(lv.any()):
            ga = a[lv]
            tt = tri[order[node[lv] - LEAF]]
            h = _ray_tri_hit(orig[ga], dirs[ga], tt, tmin, tmaxN[ga])
            hit[ga[h]] = True
        iv = within & ~isleaf
        if bool(iv.any()):
            gi = a[iv]
            s0 = sp[gi]
            stack[gi, s0.clamp(max=max_stack - 1)] = left[node[iv]].to(torch.int32)
            sp[gi] = (s0 + 1).clamp(max=max_stack)
            s1 = sp[gi]
            stack[gi, s1.clamp(max=max_stack - 1)] = right[node[iv]].to(torch.int32)
            sp[gi] = (s1 + 1).clamp(max=max_stack)
        active = a[(sp[a] > 0) & ~hit[a]]  # drop finished + already-hit rays
    return hit


def _ray_tri_intersect(o, d, tri, tmin, tmax, cull_backface=False):
    """Moller-Trumbore per (ray, triangle) pair. Returns (hit [N], t [N]) where t is the ray
    parameter and hit means the meeting is in (tmin, tmax). With cull_backface, drops faces whose
    outward (winding) normal points along the ray -- i.e. only keep surfaces facing the origin."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = b - a, c - a
    p = torch.cross(d, e2, dim=-1)
    det = (e1 * p).sum(-1)
    inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
    tvec = o - a
    u = (tvec * p).sum(-1) * inv
    q = torch.cross(tvec, e1, dim=-1)
    v = (d * q).sum(-1) * inv
    t = (e2 * q).sum(-1) * inv
    hit = (det.abs() > 1e-20) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > tmin) & (t < tmax)
    if cull_backface:
        hit = hit & ((torch.cross(e1, e2, dim=-1) * d).sum(-1) < 0)  # keep only front-facing
    return hit, t


def _closest_hit_rays_bvh(
    orig, dirs, tri, bvh, tmin=0.0, tmax=1e30, max_stack=64, cull_backface=False
):
    """Nearest-hit ray cast over the BVH, pure torch. Returns (t [N], face [N] long, -1 on
    miss; hit [N] bool) -- the closest intersection in (tmin, tmax), pruning nodes past best_t."""
    dev = orig.device
    N = orig.shape[0]
    LEAF = bvh["LEAF"]
    nmin, nmax = bvh["nmin"], bvh["nmax"]
    left, right, order = bvh["left"], bvh["right"], bvh["order"]
    inv = 1.0 / torch.where(dirs.abs() < 1e-20, torch.full_like(dirs, 1e-20), dirs)
    best_t = torch.full((N,), float(tmax), device=dev)
    best_f = torch.full((N,), -1, dtype=torch.long, device=dev)
    stack = torch.full((N, max_stack), -1, dtype=torch.int32, device=dev)
    sp = torch.ones(N, dtype=torch.long, device=dev)
    stack[:, 0] = 0
    active = torch.arange(N, device=dev)

    while active.numel() > 0:
        a = active
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        t1 = (nmin[node] - orig[a]) * inv[a]
        t2 = (nmax[node] - orig[a]) * inv[a]
        tnear = torch.minimum(t1, t2).amax(-1)
        tfar = torch.maximum(t1, t2).amin(-1)
        within = (
            (tfar >= tnear.clamp_min(tmin)) & (tfar >= tmin) & (tnear < best_t[a])
        )  # prune past best
        isleaf = node >= LEAF
        lv = within & isleaf
        if bool(lv.any()):
            ga = a[lv]
            fidx = order[node[lv] - LEAF]
            h, t = _ray_tri_intersect(orig[ga], dirs[ga], tri[fidx], tmin, tmax, cull_backface)
            upd = h & (t < best_t[ga])
            gu = ga[upd]
            best_t[gu] = t[upd]
            best_f[gu] = fidx[upd].long()
        iv = within & ~isleaf
        if bool(iv.any()):
            gi = a[iv]
            s0 = sp[gi]
            stack[gi, s0.clamp(max=max_stack - 1)] = left[node[iv]].to(torch.int32)
            sp[gi] = (s0 + 1).clamp(max=max_stack)
            s1 = sp[gi]
            stack[gi, s1.clamp(max=max_stack - 1)] = right[node[iv]].to(torch.int32)
            sp[gi] = (s1 + 1).clamp(max=max_stack)
        active = a[sp[a] > 0]
    return best_t, best_f, best_f >= 0


def _onb(n):
    """Branchless orthonormal basis (t, b) around unit normals n [N,3]."""
    up = torch.where(
        n[..., 2:3].abs() < 0.999,
        torch.tensor([0.0, 0.0, 1.0], device=n.device).expand_as(n),
        torch.tensor([1.0, 0.0, 0.0], device=n.device).expand_as(n),
    )
    t = torch.nn.functional.normalize(torch.cross(up, n, dim=-1), dim=-1, eps=1e-6)
    return t, torch.cross(n, t, dim=-1)


def _bake_ambient_occlusion(
    high_v,
    high_f,
    low_v_np,
    low_f_np,
    low_uv_np,
    low_n,
    resolution,
    num_samples=64,
    max_distance=0.5,
    strength=1.0,
    bias=0.01,
    ray_chunk=None,
    pbar=None,
    pbar_range=None,
):
    """Bake high-poly ambient occlusion into the low-poly's UV layout: per texel, cosine-weight
    a hemisphere of rays around the normal and cast them at the high-poly. AO = 1 - hit-fraction
    (cosine weighting makes the hit-fraction the estimator). Returns ao_img [H,W,3] in [0,1].

    ray_chunk caps rays cast at once (the per-chunk BVH stack is its dominant transient VRAM);
    None auto-sizes it to a slice of free VRAM -- big chunks (fast) on large GPUs, small (safe)
    on small ones."""
    dev = _compute_device()
    H = W = int(resolution)
    S = int(num_samples)
    if ray_chunk is None:
        # ~376 B/ray (int32 stack max_stack*4 + a few [N,3] ray buffers); spend a quarter of free
        # device memory. Speed saturates around 4M rays/chunk, so cap there (about 2 GB peak) rather than
        # grow memory for no gain; floor keeps tiny GPUs from thrashing into too many chunks.
        free = torch.cuda.mem_get_info(dev)[0] if dev.type == "cuda" else 4 * GIBIBYTE
        ray_chunk = int(min(4 * MEBIBYTE, max(MEBIBYTE, (free * 0.25) / (num_samples * 4 + 200))))
    face_idx, bary_uv, mask = _rasterize_uv_barycentric(low_f_np, low_uv_np, resolution)
    if not mask.any():
        return np.ones((H, W, 3), dtype=np.float32)
    lf = torch.from_numpy(np.ascontiguousarray(low_f_np).astype(np.int64)).to(dev)
    lv = torch.from_numpy(np.ascontiguousarray(low_v_np, dtype=np.float32)).to(dev)
    low_n = low_n.to(dev).float()
    m = mask
    vtri = lf[face_idx[m]]  # [K,3] vertex ids
    bsel = bary_uv[m]  # [K,3]
    P = (bsel[:, :, None] * lv[vtri]).sum(1)  # [K,3]
    Nl = torch.nn.functional.normalize((bsel[:, :, None] * low_n[vtri]).sum(1), dim=-1, eps=1e-6)

    hv = high_v.to(dev).float()
    hf = high_f.to(device=dev, dtype=torch.int32)
    tri = hv[hf]
    bvh = _build_triangle_bvh(tri)
    diag = float((hv.amax(0) - hv.amin(0)).norm().clamp_min(1e-6))
    biasw = max(1e-5, float(bias) * diag)
    tmax = float(max_distance) * diag

    # Back-project onto the high surface, then lift along the normal: the low-poly chord can sit
    # below the high surface, and casting from below floods false self-occlusion (dark blotches).
    bp = _closest_points_on_mesh_bvh(P, tri, bvh)
    origins = bp + Nl * biasw

    K = P.shape[0]
    T, B = _onb(Nl)
    occ = torch.zeros(K, device=dev)
    tex_per_chunk = max(1, int(ray_chunk) // max(1, S))
    n_chunks = max(1, (K + tex_per_chunk - 1) // tex_per_chunk)
    for ci, s in enumerate(range(0, K, tex_per_chunk)):
        e = min(s + tex_per_chunk, K)
        kk = e - s
        o, n, t, b = origins[s:e], Nl[s:e], T[s:e], B[s:e]
        r1 = torch.rand(kk, S, device=dev)
        r2 = torch.rand(kk, S, device=dev)
        sr = r1.sqrt()
        lz = r1.mul_(-1.0).add_(1.0).clamp_min_(0.0).sqrt_()  # sqrt(1-r1) (r1 dead after sr)
        ang = r2.mul_(2.0 * math.pi)  # in place (r2 dead)
        lx = ang.cos().mul_(sr)
        ly = ang.sin().mul_(sr)
        d = t[:, None, :] * lx[..., None]  # cosine-weighted hemisphere,
        d.addcmul_(b[:, None, :], ly[..., None])  # fused d += b*ly
        d.addcmul_(n[:, None, :], lz[..., None])  # fused d += n*lz  (no extra temps)
        d = torch.nn.functional.normalize(d.reshape(-1, 3), dim=-1, eps=1e-6)
        oo = o[:, None, :].expand(-1, S, -1).reshape(-1, 3)
        hit = _any_hit_rays_bvh(oo, d, tri, bvh, tmin=biasw, tmax=tmax)
        occ[s:e] = (
            hit.reshape(kk, S).sum(1, dtype=torch.float32).div_(float(S))
        )  # mean without a float copy
        if pbar is not None and pbar_range is not None:
            lo, hi = pbar_range
            pbar.update_absolute(lo + ((hi - lo) * (ci + 1)) // n_chunks, 1000)

    ao = (
        occ.mul_(-float(strength)).add_(1.0).clamp_(0.0, 1.0)
    )  # 1 - occ*strength, in place (occ is dead)
    out = torch.ones((H, W), device=dev)
    out[m] = ao
    out3 = np.repeat(out.cpu().numpy()[..., None], 3, axis=2)
    return _jfa_fill_gpu(np.ascontiguousarray(out3, dtype=np.float32), mask.cpu().numpy())


def _camera_basis(eye, center, up_hint):
    """Forward/right/up for a camera at `eye` looking at `center` (each [3])."""
    f = torch.nn.functional.normalize(center - eye, dim=-1, eps=1e-6)
    # Fall back to +Z up when looking near-vertical (f parallel +Y gives a degenerate right vector).
    up = (
        up_hint
        if float(torch.abs((f * up_hint).sum())) < 0.99
        else torch.tensor([0.0, 0.0, 1.0], device=f.device)
    )
    r = torch.nn.functional.normalize(torch.cross(f, up, dim=-1), dim=-1, eps=1e-6)
    return f, r, torch.cross(r, f, dim=-1)


def _apply_model_transform(values, transform, is_normal=False):
    def _vec(data):
        return values.new_tensor([data["x"], data["y"], data["z"]])

    scale = _vec(transform["scale"])
    position = _vec(transform["position"])
    quaternion = transform["quaternion"]
    x, y, z, w = (values.new_tensor(quaternion[key]) for key in ("x", "y", "z", "w"))
    rotation = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    )
    if is_normal:
        return torch.nn.functional.normalize((values / scale) @ rotation.T, dim=-1, eps=1e-6)
    return (values * scale) @ rotation.T + position


def _sample_image01(img_hwc, uv01):
    """Bilinear-sample img [H,W,C] at uv01 [K,2] in [0,1] (u=x/col, v=y/row). Returns [K,C]."""
    g = (uv01 * 2.0 - 1.0).view(1, 1, -1, 2)
    s = torch.nn.functional.grid_sample(
        img_hwc.permute(2, 0, 1)[None].float(),
        g,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return s[0, :, 0, :].t()


def _render_view(
    tri,
    bvh,
    uv,
    faces,
    texture_hwc,
    eye,
    f,
    r,
    u,
    fov,
    H,
    W,
    ray_chunk=1 << 22,
    vertex_colors=None,
    vertex_normals=None,
    render_normal=False,
):
    """Ray-cast render: per pixel, nearest-hit triangle -> colour it. With `render_normal`, output the
    view-space normal (OpenGL: x=right, y=up, z=toward camera; smooth `vertex_normals` if given, else
    the face normal). Otherwise colour source in order: `texture_hwc` (sampled via interpolated UVs),
    else `vertex_colors` (barycentric), else neutral clay shaded by facing angle. Returns
    (img, hit_mask, depth)."""
    dev = tri.device
    ys = 1.0 - (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H * 2.0  # row 0 = +up
    xs = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W * 2.0 - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    tn = math.tan(0.5 * fov)
    aspect = W / H
    d = torch.nn.functional.normalize(
        (r * (gx * tn * aspect)[..., None] + u * (gy * tn)[..., None] + f).reshape(-1, 3),
        dim=-1,
        eps=1e-6,
    )
    o = eye[None, :].expand(H * W, 3)
    img = torch.zeros((H * W, 3), device=dev)
    depth = torch.full((H * W,), float("inf"), device=dev)
    hit_all = torch.zeros(H * W, dtype=torch.bool, device=dev)
    for s in range(0, H * W, ray_chunk):
        e = min(s + ray_chunk, H * W)
        t_hit, face, hit = _closest_hit_rays_bvh(o[s:e], d[s:e], tri, bvh, tmin=1e-5, tmax=1e30)
        if bool(hit.any()):
            fh = face[hit].clamp_min(0)
            P = o[s:e][hit] + t_hit[hit, None] * d[s:e][hit]
            bary = _barycentric(P, tri[fh])
            local = torch.zeros((e - s, 3), device=dev)
            if render_normal:
                if vertex_normals is not None:
                    nrm = torch.nn.functional.normalize(
                        (bary[:, :, None] * vertex_normals[faces[fh]]).sum(1), dim=-1, eps=1e-6
                    )
                else:  # face normal, oriented toward camera
                    nrm = torch.nn.functional.normalize(
                        torch.cross(
                            tri[fh][:, 1] - tri[fh][:, 0], tri[fh][:, 2] - tri[fh][:, 0], dim=-1
                        ),
                        dim=-1,
                        eps=1e-6,
                    )
                    nrm = torch.where((nrm * -d[s:e][hit]).sum(-1, keepdim=True) < 0, -nrm, nrm)
                nv = torch.stack([(nrm * r).sum(-1), (nrm * u).sum(-1), (nrm * -f).sum(-1)], dim=-1)
                local[hit] = (nv * 0.5 + 0.5).clamp(0.0, 1.0)  # view-space OpenGL normal encode
            elif texture_hwc is not None and uv is not None:
                uvh = (bary[:, :, None] * uv[faces[fh]]).sum(1)
                local[hit] = _sample_image01(texture_hwc, uvh)
            elif vertex_colors is not None:
                local[hit] = (bary[:, :, None] * vertex_colors[faces[fh]]).sum(1)
            else:
                # Neutral clay, headlight-shaded (|n*view|) so silhouette-plus-form reads, not a flat blob.
                fn = torch.nn.functional.normalize(
                    torch.cross(
                        tri[fh][:, 1] - tri[fh][:, 0], tri[fh][:, 2] - tri[fh][:, 0], dim=-1
                    ),
                    dim=-1,
                    eps=1e-6,
                )
                ndl = (fn * -d[s:e][hit]).sum(-1).abs().clamp(0.15, 1.0)
                local[hit] = torch.tensor([0.72, 0.72, 0.72], device=dev) * ndl[:, None]
            img[s:e] = local
            dloc = torch.full((e - s,), float("inf"), device=dev)
            dloc[hit] = t_hit[hit]
            depth[s:e] = dloc
            hit_all[s:e] = hit
    img = img.reshape(H, W, 3)
    depth = depth.reshape(H, W)
    hit_all = hit_all.reshape(H, W)
    # Dilate the object color into the background so bilinear sampling near the silhouette doesn't
    # bleed black (a cross-view seam source) -- and gives the upscaler a coherent, edge-free image.
    if bool(hit_all.any()) and not bool(hit_all.all()):
        img = torch.from_numpy(_jfa_fill_gpu(img.cpu().numpy(), hit_all.cpu().numpy())).to(dev)
    return img, hit_all, depth


def _smooth_vertex_normals(vertices_np, faces_np, weld=True):
    """Area-weighted per-vertex normals (unit length), fully smooth, no vertex splitting."""
    tris = vertices_np[faces_np]  # (M, 3, 3)
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    if weld and vertices_np.shape[0]:
        # Group coincident positions (quantized to ~1e-5 of the bbox) into one shared normal.
        lo = vertices_np.min(0)
        inv_tol = 1.0 / (max(float((vertices_np.max(0) - lo).max()), 1e-9) * 1e-5)
        q = np.round((vertices_np - lo) * inv_tol).astype(np.int64)
        _, group = np.unique(q, axis=0, return_inverse=True)
        acc = np.zeros((int(group.max()) + 1, 3), dtype=np.float64)
        for k in range(3):
            np.add.at(acc, group[faces_np[:, k]], face_n)
        normals = acc[group]  # welded normal back to each vertex
    else:
        normals = np.zeros((vertices_np.shape[0], 3), dtype=np.float64)
        for k in range(3):
            np.add.at(normals, faces_np[:, k], face_n)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lens > 1e-12, lens, 1.0)
    return normals.astype(np.float32)


def _compute_vertex_face_normals(vertices_np, faces_np, crease_angle=None):
    """Compute per-vertex normals, returning (vertices, faces_uint32, normals, remap).

    crease_angle is None (or >= 180) -> fully smooth normals; vertices/faces are
    returned unchanged and remap is None.

    Otherwise vertices are split along edges whose dihedral angle exceeds
    crease_angle (degrees) so hard creases stay sharp while smooth regions still
    interpolate. remap maps each output vertex back to its source index, so the
    caller can duplicate any per-vertex attributes (uvs / colors) to match."""
    faces_i = faces_np.astype(np.int64)
    if crease_angle is None or crease_angle >= 180.0:
        return (
            vertices_np,
            faces_i.astype(np.uint32),
            _smooth_vertex_normals(vertices_np, faces_i),
            None,
        )

    M = faces_i.shape[0]
    tris = vertices_np[faces_i]
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = np.linalg.norm(face_n, axis=1, keepdims=True)
    face_unit = face_n / np.where(areas > 1e-12, areas, 1.0)
    cos_thresh = math.cos(math.radians(crease_angle))

    # Union faces that share an edge whose dihedral angle is below the crease
    # threshold; each connected component becomes one smoothing group.
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edge_faces = {}
    for fi in range(M):
        a, b, c = int(faces_i[fi, 0]), int(faces_i[fi, 1]), int(faces_i[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((u, v) if u < v else (v, u), []).append(fi)
    for fl in edge_faces.values():
        if len(fl) == 2 and float(np.dot(face_unit[fl[0]], face_unit[fl[1]])) >= cos_thresh:
            ra, rb = find(fl[0]), find(fl[1])
            if ra != rb:
                parent[ra] = rb

    # Emit one output vertex per (original vertex, smoothing group) pair.
    new_index = {}
    remap = []
    out_faces = np.empty((M, 3), dtype=np.int64)
    for fi in range(M):
        g = find(fi)
        for k in range(3):
            ov = int(faces_i[fi, k])
            key = (ov, g)
            ni = new_index.get(key)
            if ni is None:
                ni = len(remap)
                new_index[key] = ni
                remap.append(ov)
            out_faces[fi, k] = ni

    remap = np.asarray(remap, dtype=np.int64)
    normals = np.zeros((remap.shape[0], 3), dtype=np.float64)
    for k in range(3):
        np.add.at(normals, out_faces[:, k], face_n)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lens > 1e-12, lens, 1.0)
    return (vertices_np[remap], out_faces.astype(np.uint32), normals.astype(np.float32), remap)


def _compute_vertex_tangents(verts, faces, uvs, normals):
    """Per-vertex tangents (Lengyel) orthonormalized against `normals`. Returns [N,4]:
    unit tangent xyz + handedness w (the bitangent is w * cross(N, T)). Pure torch."""
    N = verts.shape[0]
    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    e1, e2 = verts[i1] - verts[i0], verts[i2] - verts[i0]
    d1, d2 = uvs[i1] - uvs[i0], uvs[i2] - uvs[i0]
    denom = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = 1.0 / torch.where(denom.abs() < 1e-20, torch.full_like(denom, 1e-20), denom)
    tan = (d2[:, 1:2] * e1 - d1[:, 1:2] * e2) * r[:, None]  # [F,3]
    bit = (d1[:, 0:1] * e2 - d2[:, 0:1] * e1) * r[:, None]
    tacc = torch.zeros((N, 3), device=verts.device, dtype=verts.dtype)
    bacc = torch.zeros((N, 3), device=verts.device, dtype=verts.dtype)
    for idx in (i0, i1, i2):
        tacc.scatter_add_(0, idx[:, None].expand(-1, 3), tan)
        bacc.scatter_add_(0, idx[:, None].expand(-1, 3), bit)
    n = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
    # Gram-Schmidt: drop the normal component, then renormalize.
    t = torch.nn.functional.normalize(tacc - n * (n * tacc).sum(-1, keepdim=True), dim=-1, eps=1e-6)
    w = torch.sign((torch.cross(n, t, dim=-1) * bacc).sum(-1))
    w = torch.where(w == 0, torch.ones_like(w), w)  # degenerate -> right-handed
    return torch.cat([t, w[:, None]], dim=-1)


def _vertex_tangents_for_item(lv, lf, uv, low_n_attr_i, dev):
    """Per-item shading normals + tangents. Shared by the bake (BakeNormalMapFromMesh) and the
    export attach (ApplyTextureToMesh) so their basis can't diverge. `low_n_attr_i` is the
    mesh's per-item normals or None (then computed). Returns (low_n [N,3], tangents [N,4])."""
    low_n = (
        low_n_attr_i.to(dev).float()
        if low_n_attr_i is not None
        else _compute_vertex_normals(lv, lf)
    )
    tangents = _compute_vertex_tangents(lv, lf, uv.to(dev).float(), low_n)
    return low_n, tangents


def _barycentric(p, tri):
    """Barycentric coords [N,3] of points p [N,3] wrt triangles tri [N,3,3] (per-pair)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    denom = torch.where(denom.abs() < 1e-20, torch.full_like(denom, 1e-20), denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return torch.stack([1.0 - v - w, v, w], dim=-1)


def _bake_normal_map(
    high_v,
    high_f,
    high_n,
    low_v_np,
    low_f_np,
    low_uv_np,
    low_n,
    tangents,
    resolution,
    cage_distance=0.05,
    ignore_backfaces=True,
):
    """Tangent-space normal map (glTF/OpenGL +Y) of the high-poly baked into the low-poly's UV
    layout. Per texel a cage ray (along the normal, over cage_distance * bbox-diagonal) finds the
    matching high-poly surface, whose normal is projected into the texel's TBN frame.
    ignore_backfaces skips surfaces facing away (crevices/enclosures); misses fall back to
    closest-point. Returns [H,W,3] in [0,1]."""
    dev = _compute_device()
    H = W = int(resolution)
    flat = np.array([0.5, 0.5, 1.0], dtype=np.float32)

    # One rasterization, then interpolate position/normal/tangent/handedness by indexing it.
    face_idx, bary_uv, mask = _rasterize_uv_barycentric(low_f_np, low_uv_np, resolution)
    if not mask.any():
        return np.tile(flat, (H, W, 1))
    lf = torch.from_numpy(np.ascontiguousarray(low_f_np).astype(np.int64)).to(dev)
    lv = torch.from_numpy(np.ascontiguousarray(low_v_np, dtype=np.float32)).to(dev)
    low_n = low_n.to(dev).float()
    tangents = tangents.to(dev).float()
    m = mask
    fsel = face_idx[m]  # [K] source face per texel
    bsel = bary_uv[m]  # [K,3]
    vtri = lf[fsel]  # [K,3] vertex ids

    def _interp(attr):  # attr [N,C] -> [K,C]
        return (bsel[:, :, None] * attr[vtri]).sum(1)

    P = _interp(lv)  # [K,3] world pos
    Nl = torch.nn.functional.normalize(_interp(low_n), dim=-1, eps=1e-6)
    Tl = _interp(tangents[:, :3])
    Wl = _interp(tangents[:, 3:4])[:, 0]

    hv = high_v.to(dev).float()
    hf = high_f.to(device=dev, dtype=torch.int32)
    tri = hv[hf]
    bvh = _build_triangle_bvh(tri)

    # Cage ray-cast: from cage outward, march back along -normal and take the nearest (outermost)
    # hit. Closest-point is the fallback where the ray misses.
    diag = float((hv.amax(0) - hv.amin(0)).norm().clamp_min(1e-6))
    cage = max(1e-6, float(cage_distance) * diag)
    origin = P + Nl * cage
    t_hit, f_hit, ray_hit = _closest_hit_rays_bvh(
        origin, -Nl, tri, bvh, tmin=0.0, tmax=2.0 * cage, cull_backface=bool(ignore_backfaces)
    )
    bface = f_hit.clamp_min(0)
    hitpoint = origin - t_hit[:, None] * Nl
    # Closest-point fallback only for texels the ray missed (usually few) -- running it over every
    # texel wastes a full BVH traversal on the ones already resolved by the ray.
    miss = ~ray_hit
    if bool(miss.any()):
        bp_m, bf_m = _closest_points_on_mesh_bvh(P[miss], tri, bvh, return_face=True)
        bface = bface.clone()
        hitpoint = hitpoint.clone()
        bface[miss] = bf_m.clamp_min(0)
        hitpoint[miss] = bp_m

    htri = tri[bface]  # [K,3,3]
    bary = _barycentric(hitpoint, htri)
    hn_tri = high_n.to(dev).float()[hf[bface]]  # [K,3,3] vertex normals
    Nh = torch.nn.functional.normalize((bary[:, :, None] * hn_tri).sum(1), dim=-1, eps=1e-6)

    # Per-texel TBN (Gram-Schmidt tangent against the interpolated normal).
    T = torch.nn.functional.normalize(Tl - Nl * (Nl * Tl).sum(-1, keepdim=True), dim=-1, eps=1e-6)
    Bn = Wl[:, None] * torch.cross(Nl, T, dim=-1)
    nz = (Nh * Nl).sum(-1)  # reused as z-channel and the back-face test
    ts = torch.stack([(Nh * T).sum(-1), (Nh * Bn).sum(-1), nz], dim=-1)
    ts = torch.nn.functional.normalize(ts, dim=-1, eps=1e-6)
    # Safety net: if the matched high normal faces away from the texel (a back surface the fallback
    # grabbed in a deep crevice), use the flat base normal rather than a wrong one.
    ts[nz < 0.0] = torch.tensor([0.0, 0.0, 1.0], device=dev)
    enc = ts.mul_(0.5).add_(0.5).clamp_(0.0, 1.0)  # encode in place (ts is dead)

    out = torch.from_numpy(np.tile(flat, (H, W, 1))).to(dev)
    out[m] = enc
    # Dilate into the UV gutter so bilinear/mip sampling at chart edges doesn't bleed flat blue.
    return _jfa_fill_gpu(out.cpu().numpy(), mask.cpu().numpy())


def _jfa_fill_gpu(img01, mask):
    """Fill uncovered texels with nearest covered value via GPU Jump Flooding
    (O(log n) passes; replaces cv2.inpaint). img01 [H,W,C] float, mask [H,W] bool."""
    if not mask.any():
        return img01
    dev = _compute_device()
    it = torch.from_numpy(np.ascontiguousarray(img01)).to(dev).float()
    mm = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
    H, W = mm.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.int32, device=dev),
        torch.arange(W, dtype=torch.int32, device=dev),
        indexing="ij",
    )
    by = torch.where(mm, yy, torch.full_like(yy, -1))
    bx = torch.where(mm, xx, torch.full_like(xx, -1))
    step = 1 << ((max(H, W) - 1).bit_length() - 1)
    while step >= 1:
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                ny = (yy + dy).clamp(0, H - 1)
                nx = (xx + dx).clamp(0, W - 1)
                cby = by[ny, nx]
                cbx = bx[ny, nx]
                valid = cby >= 0
                dc = torch.where(valid, (yy - cby) ** 2 + (xx - cbx) ** 2, GIBIBYTE)
                db = torch.where(by >= 0, (yy - by) ** 2 + (xx - bx) ** 2, GIBIBYTE)
                take = valid & (dc < db)
                by = torch.where(take, cby, by)
                bx = torch.where(take, cbx, bx)
        step //= 2
    flat_idx = by.clamp_min_(0).long().mul_(W).add_(bx.clamp_min_(0))
    filled = it.reshape(-1, it.shape[-1])[flat_idx.reshape(-1)].reshape(H, W, -1)
    return filled.cpu().numpy()


def _seam_fill(img01, mask):
    """Fill UV-gutter texels (so seams don't pull in black) via JFA nearest-coverage."""
    return _jfa_fill_gpu(img01, mask)


def _normalize_uvs_to_unit(uv_np, normalize=True, log_prefix=None):
    """Uniformly fit a UV bbox into [0,1] when it spills outside (preserves aspect;
    no-op if already in [0,1]; not a UDIM de-tiler). Shared deterministic helper --
    bake and ApplyTextureToMesh both call it so UVs agree (keep both paths in sync).
    Returns float32 [N,2]."""
    uv_np = uv_np.astype(np.float32)
    uv_min = uv_np.min(axis=0)
    uv_max = uv_np.max(axis=0)
    out_of_unit = (uv_min.min() < -1e-4) or (uv_max.max() > 1.0001)
    if not (normalize and out_of_unit):
        return uv_np
    extent = float((uv_max - uv_min).max())
    span = max(float(uv_max[0] - uv_min[0]), float(uv_max[1] - uv_min[1]))
    if span > 1.5 and log_prefix:
        logging.warning(
            f"{log_prefix} UV span {span:.2f} looks like a tiled/UDIM layout; "
            f"uniform-fitting it into [0,1] will overlap tiles. Re-unwrap upstream instead."
        )
    if extent > 0:
        uv_np = ((uv_np - uv_min) / extent).astype(np.float32)
        if log_prefix:
            logging.info(f"{log_prefix} normalized UVs into [0,1] (uniform scale 1/{extent:.4f})")
    return uv_np


def bake_texture_from_voxel_fn(
    vertices,
    faces,
    voxel_coords,
    voxel_colors,
    resolution,
    texture_size,
    uvs,
    normalize_uvs=True,
    reference=None,
    pbar=None,
):
    """Bake a baseColor (+ optional metallicRoughness) texture: rasterize in UV space,
    sample each texel from the sparse voxel volume. `uvs` (N,2) is the existing layout,
    1:1 with `vertices` (never unwraps). Returns (v, f, uvs, texture, mr). Ticks `pbar`
    once per stage; size it 5 per bake."""
    # _tick fires once per stage boundary, including no-op stages, so the 5-tick pbar stays aligned.
    _tq = tqdm(total=5, desc="Bake texture", leave=False)

    def _tick(name):
        _tq.set_postfix_str(name)
        _tq.update(1)
        if pbar is not None:
            pbar.update(1)

    v_np = vertices.detach().cpu().numpy().astype(np.float32)
    f_np = faces.detach().cpu().numpy().astype(np.uint32)

    uv_np = uvs.detach().cpu().numpy().astype(np.float32)
    if uv_np.shape[0] != v_np.shape[0]:
        raise ValueError(
            f"BakeTextureFromVoxel: UVs ({uv_np.shape[0]}) must be 1:1 "
            f"with vertices ({v_np.shape[0]})."
        )
    uv_np = _normalize_uvs_to_unit(uv_np, normalize_uvs, log_prefix="[BakeTextureFromVoxel]  ")
    new_verts, new_faces, new_uvs = v_np, f_np, uv_np

    _tick("uvs")

    position_map, mask = _bake_position_map(new_verts, new_faces, new_uvs, texture_size)
    _tick("rasterize")

    if reference is not None:
        # Back-project onto the dense surface before sampling (smooth bake on coarse
        # meshes, not along flat triangle chords).
        position_map = _back_project_positions(position_map, mask, reference[0], reference[1])
    _tick("back-project")

    attrs = _sample_voxel_attrs_per_texel(
        position_map,
        mask,
        voxel_coords,
        voxel_colors,
        resolution,
    )
    _tick("sample")

    # PBR layout (upstream pbr_attr_layout): 0:3 base_color, 3 metallic, 4 roughness, 5 alpha.
    C = attrs.shape[-1]
    base_color = np.ascontiguousarray(attrs[..., 0:3])
    has_pbr = C >= 5
    # alpha (idx 5) ignored -- meshes kept opaque (upstream OPAQUE alpha_mode).

    mr_image = None
    if has_pbr:
        # glTF metallicRoughness: R unused, G=roughness, B=metallic.
        mr = np.empty((*attrs.shape[:-1], 3), dtype=attrs.dtype)
        mr[..., 0] = 0.0
        mr[..., 1] = attrs[..., 4]
        mr[..., 2] = attrs[..., 3]
    del attrs

    base_color = _seam_fill(base_color, mask)
    if has_pbr:
        mr_image = _seam_fill(mr, mask)

    device = vertices.device
    out_v = torch.from_numpy(new_verts).to(device=device, dtype=torch.float32)
    out_f = torch.from_numpy(new_faces.astype(np.int32)).to(device=device, dtype=torch.int32)
    out_uvs = torch.from_numpy(new_uvs).to(device=device, dtype=torch.float32)
    out_tex = torch.from_numpy(np.ascontiguousarray(base_color)).to(
        device=device, dtype=torch.float32
    )
    out_mr = (
        torch.from_numpy(np.ascontiguousarray(mr_image)).to(device=device, dtype=torch.float32)
        if mr_image is not None
        else None
    )
    _tick("finalize")
    _tq.close()
    return out_v, out_f, out_uvs, out_tex, out_mr


def _mr_channel(packed_mr, ch, ref):
    """Pull one channel (G=roughness idx 1, B=metallic idx 2) from a packed glTF MR map
    as 3-channel grayscale [H,W,3] in [0,1]. Black sized like `ref` if no MR map."""
    if packed_mr is None:
        return torch.zeros((*ref.shape[:-1], 1), dtype=torch.float32).expand(-1, -1, 3)
    m = packed_mr[..., ch : ch + 1].float().clamp(0.0, 1.0).cpu()
    return m.expand(-1, -1, 3)


def _pack_uv_meshes(vs, fs, uvs, colors):
    """Pack per-item (verts, faces, uvs[, colors]) into a MESH; stack if single, else pad."""
    if len(vs) == 1:
        m = TriangleMeshBatch(
            vertices=vs[0].unsqueeze(0),
            faces=fs[0].unsqueeze(0),
            uvs=uvs[0].unsqueeze(0),
        )
        if colors is not None:
            m.vertex_colors = colors[0].unsqueeze(0)
        return m
    bsz = len(vs)
    dev = vs[0].device
    maxv = max(v.shape[0] for v in vs)
    maxf = max(f.shape[0] for f in fs)
    pv = vs[0].new_zeros((bsz, maxv, 3))
    pf = fs[0].new_zeros((bsz, maxf, 3))
    pu = uvs[0].new_zeros((bsz, maxv, 2))
    for i, (v, f, u) in enumerate(zip(vs, fs, uvs)):
        pv[i, : v.shape[0]] = v
        pf[i, : f.shape[0]] = f
        pu[i, : u.shape[0]] = u
    vc = torch.tensor([v.shape[0] for v in vs], device=dev, dtype=torch.int64)
    fc = torch.tensor([f.shape[0] for f in fs], device=dev, dtype=torch.int64)
    m = TriangleMeshBatch(
        vertices=pv,
        faces=pf,
        uvs=pu,
        vertex_counts=vc,
        face_counts=fc,
    )
    if colors is not None:
        pc = colors[0].new_zeros((bsz, maxv, colors[0].shape[1]))
        for i, c in enumerate(colors):
            pc[i, : c.shape[0]] = c
        m.vertex_colors = pc
    return m


def _uv_weld_vertices(v, f, weld_distance):
    """Merge coincident verts; returns (welded_v, welded_f, welded_to_orig); last None if no welding."""
    v_np = v.cpu().numpy()
    f_np = f.cpu().numpy()
    if v_np.size == 0:
        return v, f, None
    extent = float(np.linalg.norm(v_np.max(axis=0) - v_np.min(axis=0)))
    tol = weld_distance if weld_distance > 0.0 else 1e-5 * extent
    if tol <= 0.0:
        return v, f, None
    keys = np.round(v_np / tol).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_unique = int(inv.max()) + 1
    if n_unique >= v_np.shape[0]:
        return v, f, None
    v_welded = np.zeros((n_unique, 3), dtype=np.float32)
    counts = np.zeros(n_unique, dtype=np.int64)
    np.add.at(v_welded, inv, v_np)
    np.add.at(counts, inv, 1)
    v_welded /= counts[:, None]
    welded_to_orig = np.empty(n_unique, dtype=np.int64)
    welded_to_orig[inv] = np.arange(v_np.shape[0], dtype=np.int64)
    v_new = torch.from_numpy(v_welded).to(v.dtype).to(v.device)
    f_new = torch.from_numpy(inv[f_np]).to(f.dtype).to(f.device)
    return v_new, f_new, welded_to_orig


def _uv_unwrap(positions, indices, segmenter, resolution, padding, weld_distance):
    """UV-unwrap a single mesh; returns (vmapping, indices, uvs); vmapping maps each output
    vertex to an input vertex (seam verts duplicated)."""
    t_start = time.perf_counter()
    # phase-weighted node progress: weld/mesh 2%, segment 33%, extract 5%, param 25%, pack 33%
    pbar = _Progress()
    v_in = positions.to(torch.float32)
    f_in = indices.to(torch.long).reshape(-1, 3)
    v_in, f_in, welded_to_orig = _uv_weld_vertices(v_in, f_in, weld_distance)

    # drop degenerate faces (repeated index; corrupt edge adjacency)
    degen = (f_in[:, 0] == f_in[:, 1]) | (f_in[:, 1] == f_in[:, 2]) | (f_in[:, 2] == f_in[:, 0])
    if bool(degen.any()):
        f_in = f_in[~degen]

    mesh = _uv_mesh.build_mesh(v_in, f_in)
    ff = mesh.face_face
    if ff.numel() and float((ff >= 0).float().mean().item()) < 0.25:
        logging.warning(
            "[uv_unwrap] mesh face-adjacency < 25% -- vertices appear un-welded "
            "(triangle soup); UV charts will be per-face. Raise weld_distance."
        )

    pbar.update_absolute(20, 1000)

    def _seg_progress(done, total):
        pbar.update_absolute(20 + (330 * done) // max(total, 1), 1000)

    if segmenter == "pec":
        face_chart = _uv_seg.cluster_charts_pec(mesh, max_cost=1.0, progress_callback=_seg_progress)
    elif segmenter == "adaptive":
        face_chart = _uv_seg.segment_charts(mesh, max_cost=2.0, progress_callback=_seg_progress)
    else:
        raise ValueError(f"unknown segmenter '{segmenter}'. valid: pec, adaptive")
    pbar.update_absolute(350, 1000)

    n_charts = int(face_chart.max().item()) + 1 if face_chart.numel() else 0
    areas_cpu = _uv_mesh.chart_3d_areas(mesh.face_area, face_chart, n_charts).detach().cpu()

    if n_charts == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
        )

    # vectorized chart extraction: one global sort/unique replaces per-chart unique/searchsorted
    face_chart_np = face_chart.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()
    vertices_np = mesh.vertices.cpu().numpy()
    face_face_np = mesh.face_face.cpu().numpy()
    order = np.argsort(face_chart_np, kind="stable")
    chart_counts_np = np.bincount(face_chart_np, minlength=n_charts)
    chart_offsets_np = np.zeros(n_charts + 1, dtype=np.int64)
    np.cumsum(chart_counts_np, out=chart_offsets_np[1:])
    faces_sorted = faces_np[order]
    chart_sorted = face_chart_np[order]
    n_verts_in = max(vertices_np.shape[0], 1)
    chart_of_slot = np.repeat(chart_sorted, 3)
    uniq_keys, local_flat = np.unique(
        chart_of_slot * n_verts_in + faces_sorted.reshape(-1), return_inverse=True
    )
    used_verts_all = uniq_keys % n_verts_in  # per-chart sorted unique verts, concatenated
    vert_counts = np.bincount(uniq_keys // n_verts_in, minlength=n_charts)
    vert_offsets = np.zeros(n_charts + 1, dtype=np.int64)
    np.cumsum(vert_counts, out=vert_offsets[1:])
    local_faces_all = (local_flat - vert_offsets[chart_of_slot]).reshape(-1, 3)
    pos_in_chart = np.empty(order.size, dtype=np.int64)
    pos_in_chart[order] = np.arange(order.size) - chart_offsets_np[chart_sorted]
    ff_sorted = face_face_np[order]
    ff_safe = np.maximum(ff_sorted, 0)
    keep = (ff_sorted >= 0) & (face_chart_np[ff_safe] == chart_sorted[:, None])
    local_ff_all = np.where(keep, pos_in_chart[ff_safe], -1)

    pbar.update_absolute(400, 1000)

    # parameterize (batched): ortho-project every chart at once, batched stretch metrics
    # decide acceptance, rejected charts solve ABF/LSCM in dense per-size-bucket batches
    chart_of_vert = (uniq_keys // n_verts_in).astype(np.int64)
    verts_concat = vertices_np[used_verts_all].astype(np.float64)
    gl_faces = local_faces_all + vert_offsets[chart_sorted][:, None]
    face_pos = pos_in_chart[order]  # row of each (sorted) face in its chart
    uv0 = _uv_param.ortho_project_concat(verts_concat, chart_of_vert, n_charts)
    rms, mx, n_flip, n_zero = _uv_param.stretch_metrics_concat(
        verts_concat, uv0, gl_faces, chart_sorted, n_charts
    )
    valid_chart = (vert_counts >= 3) & (chart_counts_np > 0)
    auto = valid_chart & (chart_counts_np <= 5)  # tiny charts always keep ortho
    flip_ok = (n_flip == 0) | (n_flip == chart_counts_np)
    cand = valid_chart & ~auto & flip_ok & (n_zero == 0) & (rms <= 1.5) & (mx <= 2.0)
    param_done = int(auto.sum())
    pbar.update_absolute(400 + (250 * param_done) // n_charts, 1000)

    ortho_ok = auto.copy()
    cand_ids = np.nonzero(cand)[0]
    for c in tqdm(cand_ids, desc="unwrap: ortho checks", unit="chart", leave=False):
        f0, f1 = chart_offsets_np[c], chart_offsets_np[c + 1]
        v0, v1 = vert_offsets[c], vert_offsets[c + 1]
        if not _uv_param._uv_boundary_self_intersects(
            uv0[v0:v1], local_faces_all[f0:f1], local_ff_all[f0:f1]
        ):
            ortho_ok[c] = True
        param_done += 1
        pbar.update_absolute(400 + (250 * param_done) // n_charts, 1000)

    lscm_mask = valid_chart & ~ortho_ok
    batchable = vert_counts <= _uv_param.LSCM_BATCH_MAX_VERTS
    lscm_ids = np.nonzero(lscm_mask & batchable)[0]
    big_ids = np.nonzero(lscm_mask & ~batchable)[0]
    lscm_uv = _uv_param.lscm_charts_batch(
        verts_concat,
        uv0,
        gl_faces,
        face_pos,
        chart_sorted,
        chart_of_vert,
        vert_offsets,
        lscm_ids,
        n_charts,
        device=_compute_device(),
    )
    param_done += int(lscm_ids.size)
    pbar.update_absolute(400 + (250 * param_done) // n_charts, 1000)

    uvs_np_list: list = [None] * n_charts
    uv0_f32 = uv0.astype(np.float32)
    for c in tqdm(big_ids, desc="unwrap: LSCM (large charts)", unit="chart", leave=False):
        f0, f1 = chart_offsets_np[c], chart_offsets_np[c + 1]
        v0, v1 = vert_offsets[c], vert_offsets[c + 1]
        uvs_t = _uv_param.lscm_chart(
            torch.from_numpy(verts_concat[v0:v1]),
            torch.from_numpy(local_faces_all[f0:f1]),
            torch.from_numpy(local_ff_all[f0:f1]),
            pin_positions=uv0[v0:v1],
        )
        lscm_uv[int(c)] = uvs_t.detach().cpu().numpy().astype(np.float32)
        param_done += 1
        pbar.update_absolute(400 + (250 * param_done) // n_charts, 1000)
    for c in range(n_charts):
        v0, v1 = vert_offsets[c], vert_offsets[c + 1]
        if ortho_ok[c]:
            uvs_np_list[c] = uv0_f32[v0:v1]
            continue
        u = lscm_uv.get(int(c))
        if u is not None and np.all(np.isfinite(u)) and u.size:
            # collapsed UV island (aspect > 100:1) blows up packing scale; keep ortho instead
            bbox = u.max(axis=0) - u.min(axis=0)
            if max(float(bbox.max()), 1e-12) / max(float(bbox.min()), 1e-12) <= 100.0:
                uvs_np_list[c] = u
                continue
        uvs_np_list[c] = (
            uv0_f32[v0:v1] if valid_chart[c] else np.zeros((v1 - v0, 2), dtype=np.float32)
        )

    # per-chart UV areas in one pass over all faces
    uvs_all_np = np.concatenate(uvs_np_list)
    ua, ub, uc = uvs_all_np[gl_faces[:, 0]], uvs_all_np[gl_faces[:, 1]], uvs_all_np[gl_faces[:, 2]]
    tri_uv_area = 0.5 * np.abs(
        (ub[:, 0] - ua[:, 0]) * (uc[:, 1] - ua[:, 1])
        - (uc[:, 0] - ua[:, 0]) * (ub[:, 1] - ua[:, 1])
    )
    uv_area_np = np.bincount(
        chart_sorted, weights=tri_uv_area.astype(np.float64), minlength=n_charts
    )

    areas_3d_np = areas_cpu.numpy().astype(np.float64)

    # auto-tune texel density toward `resolution` (~0.62 pack fill)
    total_3d_area = float(areas_3d_np.sum()) or 1.0
    target_dim = float(resolution) if resolution > 0 else 1024.0
    tex_per_unit = math.sqrt((target_dim * target_dim) * 0.62 / total_3d_area)

    with tqdm(total=2 * n_charts, desc="unwrap: pack", unit="chart", leave=False) as tq_pack:

        def _pack_progress(done, total):
            tq_pack.update(done - tq_pack.n)
            pbar.update_absolute(650 + (340 * done) // max(total, 1), 1000)

        p_x, p_y, p_sw, p_th, p_sc, p_chh, atlas_w, atlas_h = _uv_pack.pack_bitmap_concat(
            uvs_all_np,
            vert_offsets,
            local_faces_all,
            chart_offsets_np,
            areas_3d_np,
            uv_area_np,
            texels_per_unit=tex_per_unit,
            padding_texels=padding,
            progress_callback=_pack_progress,
        )
    pbar.update_absolute(1000, 1000)

    # assembly: output verts are the per-chart used-vert lists concatenated in chart order,
    # so vert_offsets doubles as the output vertex cursor
    n_in_faces = mesh.faces.shape[0]
    out_indices = np.zeros((n_in_faces, 3), dtype=np.int64)
    out_indices[order] = gl_faces
    vmapping_out = used_verts_all if welded_to_orig is None else welded_to_orig[used_verts_all]
    uvs_out = _uv_pack.apply_placements_concat(
        uvs_all_np, vert_offsets, p_x, p_y, p_sw, p_th, p_sc, p_chh, atlas_w, atlas_h
    )
    logging.info(
        f"[uv_unwrap] {mesh.faces.shape[0]} faces -> {n_charts} charts, "
        f"atlas {atlas_w}x{atlas_h}, {time.perf_counter() - t_start:.1f}s"
    )
    return vmapping_out, out_indices, uvs_out


def _uv_sorted_edge_keys(indices: np.ndarray):
    """Sorted undirected edge keys; returns (sorted_keys, face_id, lo, hi, first_mask)."""
    a = indices.ravel().astype(np.int64)
    b = np.roll(indices, -1, axis=1).ravel().astype(np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    V = int(indices.max()) + 1
    key = lo * V + hi
    order = np.argsort(key, kind="stable")
    sk = key[order]
    fid = (np.arange(a.size, dtype=np.int64) // 3)[order]
    first = np.ones(sk.size, dtype=bool)
    first[1:] = sk[1:] != sk[:-1]
    return sk, fid, lo[order], hi[order], first


def _uv_faces_to_chart_ids(indices: np.ndarray) -> np.ndarray:
    """Chart = connected component of faces sharing a (non-seam-duplicated) UV vertex."""
    F = indices.shape[0]
    if F == 0:
        return np.empty(0, dtype=np.int64)
    _sk, fid, _lo, _hi, first = _uv_sorted_edge_keys(indices)
    group_id = np.cumsum(first) - 1
    starts = np.nonzero(first)[0]
    rows = fid[starts[group_id[~first]]]
    cols = fid[~first]
    if rows.size == 0:
        return np.arange(F, dtype=np.int64)
    adj = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(F, F))
    _, labels = connected_components(adj, directed=False)
    return labels.astype(np.int64)


_UV_TAB20 = np.array(
    [
        [0.121568627, 0.466666667, 0.705882353],
        [0.682352941, 0.780392157, 0.909803922],
        [1.000000000, 0.498039216, 0.054901961],
        [1.000000000, 0.733333333, 0.470588235],
        [0.172549020, 0.627450980, 0.172549020],
        [0.596078431, 0.874509804, 0.541176471],
        [0.839215686, 0.152941176, 0.156862745],
        [1.000000000, 0.596078431, 0.588235294],
        [0.580392157, 0.403921569, 0.741176471],
        [0.772549020, 0.690196078, 0.835294118],
        [0.549019608, 0.337254902, 0.294117647],
        [0.768627451, 0.611764706, 0.580392157],
        [0.890196078, 0.466666667, 0.760784314],
        [0.968627451, 0.713725490, 0.823529412],
        [0.498039216, 0.498039216, 0.498039216],
        [0.780392157, 0.780392157, 0.780392157],
        [0.737254902, 0.741176471, 0.133333333],
        [0.858823529, 0.858823529, 0.552941176],
        [0.090196078, 0.745098039, 0.811764706],
        [0.619607843, 0.854901961, 0.898039216],
    ],
    dtype=np.float32,
)


def _uv_palette(n: int) -> np.ndarray:
    rng = np.random.RandomState(42)
    perm = rng.permutation(max(1, n))
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        out[i] = _UV_TAB20[perm[i % len(perm)] % 20]
    return out


def _uv_render_atlas(
    uvs_np, indices_np, resolution, device, bg=(0.13, 0.13, 0.13), edge=(0.0, 0.0, 0.0)
):
    """Tile-based torch rasterizer of the UV atlas (charts colored, borders outlined); (H,W,3)."""
    w = h = int(resolution)
    chart_ids_np = _uv_faces_to_chart_ids(indices_np)
    uvs = torch.from_numpy(uvs_np).to(device=device, dtype=torch.float32)
    indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
    chart_ids = torch.from_numpy(chart_ids_np).to(device=device, dtype=torch.long)

    img = torch.tensor(bg, dtype=torch.float32, device=device).expand(h, w, 3).contiguous()
    if indices.numel() == 0:
        return img

    n_charts = int(chart_ids.max().item()) + 1 if chart_ids.numel() else 1
    colors = torch.from_numpy(_uv_palette(n_charts)).to(device=device, dtype=torch.float32)

    uv_px = uvs.clone()
    uv_px[:, 0] = uv_px[:, 0].clamp(0.0, 1.0) * (w - 1)
    uv_px[:, 1] = uv_px[:, 1].clamp(0.0, 1.0) * (h - 1)

    tri = uv_px[indices]
    x0 = tri[:, 0, 0]
    y0 = tri[:, 0, 1]
    x1 = tri[:, 1, 0]
    y1 = tri[:, 1, 1]
    x2 = tri[:, 2, 0]
    y2 = tri[:, 2, 1]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegen = denom.abs() > 1e-20

    xmin = torch.minimum(torch.minimum(x0, x1), x2).floor().clamp_(0, w - 1).long()
    xmax = torch.maximum(torch.maximum(x0, x1), x2).ceil().clamp_(0, w - 1).long()
    ymin = torch.minimum(torch.minimum(y0, y1), y2).floor().clamp_(0, h - 1).long()
    ymax = torch.maximum(torch.maximum(y0, y1), y2).ceil().clamp_(0, h - 1).long()

    # full point-in-tri over all pairs is O(H*W*F); tile and test only bbox-overlapping tris
    TILE = 64
    eps = 1e-6
    for ty in range(0, h, TILE):
        ty_end = min(ty + TILE, h)
        for tx in range(0, w, TILE):
            tx_end = min(tx + TILE, w)
            tri_mask = nondegen & (xmin < tx_end) & (xmax >= tx) & (ymin < ty_end) & (ymax >= ty)
            if not tri_mask.any():
                continue
            idx = torch.nonzero(tri_mask, as_tuple=True)[0]
            ys = torch.arange(ty, ty_end, dtype=torch.float32, device=device) + 0.5
            xs = torch.arange(tx, tx_end, dtype=torch.float32, device=device) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            sub_x0 = x0[idx][:, None, None]
            sub_y0 = y0[idx][:, None, None]
            sub_x1 = x1[idx][:, None, None]
            sub_y1 = y1[idx][:, None, None]
            sub_x2 = x2[idx][:, None, None]
            sub_y2 = y2[idx][:, None, None]
            sub_den = denom[idx][:, None, None]
            bx = ((sub_y1 - sub_y2) * (xx - sub_x2) + (sub_x2 - sub_x1) * (yy - sub_y2)) / sub_den
            by = ((sub_y2 - sub_y0) * (xx - sub_x2) + (sub_x0 - sub_x2) * (yy - sub_y2)) / sub_den
            bz = 1.0 - bx - by
            inside = (bx >= -eps) & (by >= -eps) & (bz >= -eps)
            if not inside.any():
                continue
            hit_any = inside.any(dim=0)
            best_tri = idx[inside.int().argmax(dim=0)]
            tile_color = colors[chart_ids[best_tri]]
            tile_img = img[ty:ty_end, tx:tx_end]
            tile_img[hit_any] = tile_color[hit_any]
            img[ty:ty_end, tx:tx_end] = tile_img

    # chart outlines: UV-space borders are open boundaries (edges with 1 incident face)
    _sk, _fid, lo, hi, first = _uv_sorted_edge_keys(indices_np)
    starts = np.nonzero(first)[0]
    counts = np.diff(np.append(starts, first.size))
    boundary = counts == 1
    uv_cpu = uv_px.cpu().numpy()
    px_xs, px_ys = [], []
    for a, b in zip(lo[starts[boundary]], hi[starts[boundary]]):
        p0 = uv_cpu[a]
        p1 = uv_cpu[b]
        steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) + 1)
        if steps <= 1:
            continue
        ts = np.linspace(0.0, 1.0, steps)
        xs = (p0[0] + (p1[0] - p0[0]) * ts).astype(np.int32)
        ys = (p0[1] + (p1[1] - p0[1]) * ts).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        px_xs.append(xs[valid])
        px_ys.append(ys[valid])
    if px_xs:
        xs_all = torch.from_numpy(np.concatenate(px_xs)).to(device=device, dtype=torch.long)
        ys_all = torch.from_numpy(np.concatenate(px_ys)).to(device=device, dtype=torch.long)
        img[ys_all, xs_all] = torch.tensor(edge, dtype=torch.float32, device=device)

    return img


def voxel_to_mesh(voxels, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cpu")
    voxels = voxels.to(device)

    binary = (voxels > threshold).float()
    padded = torch.nn.functional.pad(binary, (1, 1, 1, 1, 1, 1), "constant", 0)

    D, H, W = binary.shape

    neighbors = torch.tensor(
        [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]], device=device
    )

    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    voxel_indices = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)

    solid_mask = binary.flatten() > 0
    solid_indices = voxel_indices[solid_mask]

    corner_offsets = [
        torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]], device=device),
        torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], device=device),
        torch.tensor([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]], device=device),
        torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]], device=device),
        torch.tensor([[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], device=device),
        torch.tensor([[0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]], device=device),
    ]

    all_vertices = []
    all_indices = []

    vertex_count = 0

    for face_idx, offset in enumerate(neighbors):
        neighbor_indices = solid_indices + offset

        padded_indices = neighbor_indices + 1

        is_exposed = padded[padded_indices[:, 0], padded_indices[:, 1], padded_indices[:, 2]] == 0

        if not is_exposed.any():
            continue

        exposed_indices = solid_indices[is_exposed]

        corners = corner_offsets[face_idx].unsqueeze(0)

        face_vertices = exposed_indices.unsqueeze(1) + corners

        all_vertices.append(face_vertices.reshape(-1, 3))

        num_faces = exposed_indices.shape[0]
        face_indices = torch.arange(
            vertex_count, vertex_count + 4 * num_faces, device=device
        ).reshape(-1, 4)

        all_indices.append(
            torch.stack([face_indices[:, 0], face_indices[:, 1], face_indices[:, 2]], dim=1)
        )
        all_indices.append(
            torch.stack([face_indices[:, 0], face_indices[:, 2], face_indices[:, 3]], dim=1)
        )

        vertex_count += 4 * num_faces

    if len(all_vertices) > 0:
        vertices = torch.cat(all_vertices, dim=0)
        faces = torch.cat(all_indices, dim=0)
    else:
        vertices = torch.zeros((1, 3))
        faces = torch.zeros((1, 3))

    v_min = 0
    v_max = max(voxels.shape)

    vertices = vertices - (v_min + v_max) / 2

    scale = (v_max - v_min) / 2
    if scale > 0:
        vertices = vertices / scale

    vertices = torch.fliplr(vertices)
    return vertices, faces


def voxel_to_mesh_surfnet(voxels, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cpu")
    voxels = voxels.to(device)

    D, H, W = voxels.shape

    padded = torch.nn.functional.pad(voxels, (1, 1, 1, 1, 1, 1), "constant", 0)
    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    cell_positions = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)

    corner_offsets = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        device=device,
    )

    pos = cell_positions.unsqueeze(1) + corner_offsets.unsqueeze(0)
    z_idx, y_idx, x_idx = pos.unbind(-1)
    corner_values = padded[z_idx, y_idx, x_idx]

    corner_signs = corner_values > threshold
    has_inside = torch.any(corner_signs, dim=1)
    has_outside = torch.any(~corner_signs, dim=1)
    contains_surface = has_inside & has_outside

    active_cells = cell_positions[contains_surface]
    active_signs = corner_signs[contains_surface]
    active_values = corner_values[contains_surface]

    if active_cells.shape[0] == 0:
        return torch.zeros((0, 3), device=device), torch.zeros(
            (0, 3), dtype=torch.long, device=device
        )

    edges = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ],
        device=device,
    )

    cell_vertices = {}
    progress = _Progress()

    for edge_idx, (e1, e2) in enumerate(edges):
        progress.update(1)
        crossing = active_signs[:, e1] != active_signs[:, e2]
        if not crossing.any():
            continue

        cell_indices = torch.nonzero(crossing, as_tuple=True)[0]

        v1 = active_values[cell_indices, e1]
        v2 = active_values[cell_indices, e2]

        t = torch.zeros_like(v1, device=device)
        denom = v2 - v1
        valid = denom != 0
        t[valid] = (threshold - v1[valid]) / denom[valid]
        t[~valid] = 0.5

        p1 = corner_offsets[e1].float()
        p2 = corner_offsets[e2].float()

        intersection = p1.unsqueeze(0) + t.unsqueeze(1) * (p2.unsqueeze(0) - p1.unsqueeze(0))

        for i, point in zip(cell_indices.tolist(), intersection):
            if i not in cell_vertices:
                cell_vertices[i] = []
            cell_vertices[i].append(point)

    # Calculate the final vertices as the average of intersection points for each cell
    vertices = []
    vertex_lookup = {}

    vert_progress_mod = round(len(cell_vertices) / 50)

    for i, points in cell_vertices.items():
        if not i % vert_progress_mod:
            progress.update(1)

        if points:
            vertex = torch.stack(points).mean(dim=0)
            vertex = vertex + active_cells[i].float()
            vertex_lookup[tuple(active_cells[i].tolist())] = len(vertices)
            vertices.append(vertex)

    if not vertices:
        return torch.zeros((0, 3), device=device), torch.zeros(
            (0, 3), dtype=torch.long, device=device
        )

    final_vertices = torch.stack(vertices)

    inside_corners_mask = active_signs
    outside_corners_mask = ~active_signs

    inside_counts = inside_corners_mask.sum(dim=1, keepdim=True).float()
    outside_counts = outside_corners_mask.sum(dim=1, keepdim=True).float()

    inside_pos = torch.zeros((active_cells.shape[0], 3), device=device)
    outside_pos = torch.zeros((active_cells.shape[0], 3), device=device)

    for i in range(8):
        mask_inside = inside_corners_mask[:, i].unsqueeze(1)
        mask_outside = outside_corners_mask[:, i].unsqueeze(1)
        inside_pos += corner_offsets[i].float().unsqueeze(0) * mask_inside
        outside_pos += corner_offsets[i].float().unsqueeze(0) * mask_outside

    inside_pos /= inside_counts
    outside_pos /= outside_counts
    gradients = inside_pos - outside_pos

    pos_dirs = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device)

    cross_products = [
        torch.linalg.cross(pos_dirs[i].float(), pos_dirs[j].float())
        for i in range(3)
        for j in range(i + 1, 3)
    ]

    faces = []
    all_keys = set(vertex_lookup.keys())

    face_progress_mod = round(len(active_cells) / 38 * 3)

    for pair_idx, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        dir_i = pos_dirs[i]
        dir_j = pos_dirs[j]
        cross_product = cross_products[pair_idx]

        ni_positions = active_cells + dir_i
        nj_positions = active_cells + dir_j
        diag_positions = active_cells + dir_i + dir_j

        alignments = torch.matmul(gradients, cross_product)

        valid_quads = []
        quad_indices = []

        for idx, active_cell in enumerate(active_cells):
            if not idx % face_progress_mod:
                progress.update(1)
            cell_key = tuple(active_cell.tolist())
            ni_key = tuple(ni_positions[idx].tolist())
            nj_key = tuple(nj_positions[idx].tolist())
            diag_key = tuple(diag_positions[idx].tolist())

            if (
                cell_key in all_keys
                and ni_key in all_keys
                and nj_key in all_keys
                and diag_key in all_keys
            ):
                v0 = vertex_lookup[cell_key]
                v1 = vertex_lookup[ni_key]
                v2 = vertex_lookup[nj_key]
                v3 = vertex_lookup[diag_key]

                valid_quads.append((v0, v1, v2, v3))
                quad_indices.append(idx)

        for q_idx, (v0, v1, v2, v3) in enumerate(valid_quads):
            cell_idx = quad_indices[q_idx]
            if alignments[cell_idx] > 0:
                faces.append(torch.tensor([v0, v1, v3], device=device, dtype=torch.long))
                faces.append(torch.tensor([v0, v3, v2], device=device, dtype=torch.long))
            else:
                faces.append(torch.tensor([v0, v3, v1], device=device, dtype=torch.long))
                faces.append(torch.tensor([v0, v2, v3], device=device, dtype=torch.long))

    if faces:
        faces = torch.stack(faces)
    else:
        faces = torch.zeros((0, 3), dtype=torch.long, device=device)

    v_min = 0
    v_max = max(D, H, W)

    final_vertices = final_vertices - (v_min + v_max) / 2

    scale = (v_max - v_min) / 2
    if scale > 0:
        final_vertices = final_vertices / scale

    final_vertices = torch.fliplr(final_vertices)

    return final_vertices, faces


def voxel_grid_to_mesh(
    voxel: DenseVoxelGrid[torch.Tensor],
    algorithm: str = "surface net",
    threshold: float = 0.6,
) -> TriangleMeshBatch[torch.Tensor]:
    if type(voxel) is not DenseVoxelGrid:
        raise TypeError("voxel_to_mesh: voxel must be an exact DenseVoxelGrid")
    if algorithm not in ("surface net", "basic"):
        raise ValueError("voxel_to_mesh: algorithm must be 'surface net' or 'basic'")
    function = voxel_to_mesh_surfnet if algorithm == "surface net" else voxel_to_mesh
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    for item in voxel.data:
        item_vertices, item_faces = function(item, threshold=threshold, device=torch.device("cpu"))
        vertices.append(item_vertices)
        faces.append(item_faces)
    if all(value.shape == vertices[0].shape for value in vertices) and all(
        value.shape == faces[0].shape for value in faces
    ):
        return TriangleMeshBatch(vertices=torch.stack(vertices), faces=torch.stack(faces))
    return pack_variable_mesh_batch(vertices, faces)


def _map_geometry(
    mesh: TriangleMeshBatch[torch.Tensor],
    operation: str,
    function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ],
) -> TriangleMeshBatch[torch.Tensor]:
    _validate_mesh(mesh, operation)
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    colors: list[torch.Tensor] = []
    for index in range(mesh.vertices.shape[0]):
        item_vertices, item_faces, item_colors, _, _ = get_mesh_batch_item(mesh, index)
        out_vertices, out_faces, out_colors = function(item_vertices, item_faces, item_colors)
        vertices.append(out_vertices)
        faces.append(out_faces)
        if out_colors is not None:
            colors.append(out_colors)
    result = pack_variable_mesh_batch(
        vertices,
        faces,
        colors=colors if len(colors) == len(vertices) else None,
    )
    result.texture = mesh.texture
    result.metallic_roughness = mesh.metallic_roughness
    result.normal_map = mesh.normal_map
    result.occlusion_in_mr = mesh.occlusion_in_mr
    result.material = mesh.material
    result.emissive = mesh.emissive
    result.unlit = mesh.unlit
    return result


def decimate_mesh(
    mesh: TriangleMeshBatch[torch.Tensor],
    target_face_count: int = 200_000,
    placement_mode: str = "midpoint",
) -> TriangleMeshBatch[torch.Tensor]:
    if placement_mode not in ("midpoint", "qem"):
        raise ValueError("decimate_mesh: placement_mode must be 'midpoint' or 'qem'")
    config = QEMConfig(placement_mode=placement_mode)
    compute_device = _compute_device(mesh.vertices)

    def decimate(vertices, faces, colors):
        if target_face_count <= 0 or faces.shape[0] <= target_face_count:
            return vertices, faces, colors
        source_device = vertices.device
        out_vertices, out_faces, out_colors, _, _ = qem_decimate_simplify(
            vertices.to(compute_device),
            faces.to(compute_device),
            int(target_face_count),
            colors=None if colors is None else colors.to(compute_device),
            config=config,
        )
        return (
            out_vertices.to(source_device),
            out_faces.to(source_device),
            None if out_colors is None else out_colors.to(source_device),
        )

    return _map_geometry(mesh, "decimate_mesh", decimate)


def remesh_mesh(
    mesh: TriangleMeshBatch[torch.Tensor],
    resolution: int = 512,
    sign_mode: str = "udf",
    qef: bool = False,
    drop_inverted_components: bool = False,
    drop_enclosed_components: bool = False,
    manifold: bool = False,
    band: float = 1.0,
    project_back: float = 0.0,
    fix_poles: bool = False,
    smooth_iters: int = 0,
    drop_small_components: float = 0.01,
    precluster_max_verts: int = 20_000_000,
    cancelled: Callable[[], bool] | None = None,
) -> TriangleMeshBatch[torch.Tensor]:
    if sign_mode not in ("udf", "sdf"):
        raise ValueError("remesh_mesh: sign_mode must be 'udf' or 'sdf'")
    compute_device = _compute_device(mesh.vertices)

    def remesh(vertices, faces, colors):
        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise RuntimeError("remesh_mesh: cancelled")

        check_cancelled()
        source_device = vertices.device
        working_vertices = vertices.to(compute_device).float()
        working_faces = faces.to(device=compute_device, dtype=torch.int64)
        working_colors = None if colors is None else colors.to(compute_device).float()
        if precluster_max_verts > 0 and working_vertices.shape[0] > precluster_max_verts:
            working_vertices, working_faces, working_colors = qem_cluster_decimate(
                working_vertices,
                working_faces,
                target_verts=int(precluster_max_verts),
                colors=working_colors,
            )
        scale = (resolution + 3.0 * band) / resolution
        out_vertices, out_faces, out_colors = remesh_narrow_band_dc(
            working_vertices,
            working_faces,
            resolution=int(resolution),
            band=float(band),
            project_back=float(project_back),
            qef=bool(qef),
            sign_mode=sign_mode,
            manifold=bool(manifold),
            fix_poles=bool(fix_poles),
            smooth_iters=int(smooth_iters),
            drop_small_components=float(drop_small_components),
            drop_inverted_components=bool(drop_inverted_components),
            progress_callback=check_cancelled,
            drop_enclosed_components=bool(drop_enclosed_components),
            scale=scale,
            center=torch.zeros(3, dtype=working_vertices.dtype, device=compute_device),
            colors=working_colors,
        )
        return (
            out_vertices.to(source_device),
            out_faces.to(device=source_device, dtype=torch.int32),
            None if out_colors is None else out_colors.to(source_device),
        )

    return _map_geometry(mesh, "remesh_mesh", remesh)


def smooth_mesh_normals(
    mesh: TriangleMeshBatch[torch.Tensor], crease_angle: float = 180.0
) -> TriangleMeshBatch[torch.Tensor]:
    _validate_mesh(mesh, "smooth_mesh_normals")
    if crease_angle >= 180.0:
        normals = torch.zeros_like(mesh.vertices)
        for index in range(mesh.vertices.shape[0]):
            vertices, faces, _, _, _ = get_mesh_batch_item(mesh, index)
            if vertices.numel() and faces.numel():
                value = _smooth_vertex_normals(
                    vertices.cpu().numpy().astype(np.float32),
                    faces.cpu().numpy().astype(np.int64),
                )
                normals[index, : value.shape[0]] = torch.from_numpy(value).to(mesh.vertices)
        result = copy.copy(mesh)
        result.normals = normals
        return result
    vertices_out: list[torch.Tensor] = []
    faces_out: list[torch.Tensor] = []
    normals_out: list[torch.Tensor] = []
    colors_out: list[torch.Tensor] | None = [] if mesh.vertex_colors is not None else None
    uvs_out: list[torch.Tensor] | None = [] if mesh.uvs is not None else None
    tangents_out: list[torch.Tensor] | None = [] if mesh.tangents is not None else None
    for index in range(mesh.vertices.shape[0]):
        vertices, faces, colors, uvs, _ = get_mesh_batch_item(mesh, index)
        out_vertices, out_faces, out_normals, remap = _compute_vertex_face_normals(
            vertices.cpu().numpy().astype(np.float32),
            faces.cpu().numpy().astype(np.int64),
            float(crease_angle),
        )
        remap_tensor = torch.from_numpy(remap).to(vertices.device)
        vertices_out.append(torch.from_numpy(out_vertices).to(vertices))
        faces_out.append(torch.from_numpy(out_faces).to(faces))
        normals_out.append(torch.from_numpy(out_normals).to(vertices))
        if colors_out is not None and colors is not None:
            colors_out.append(colors[remap_tensor])
        if uvs_out is not None and uvs is not None:
            uvs_out.append(uvs[remap_tensor])
        if tangents_out is not None and mesh.tangents is not None:
            tangents_out.append(mesh.tangents[index, : vertices.shape[0]][remap_tensor])
    return pack_variable_mesh_batch(
        vertices_out,
        faces_out,
        colors=colors_out,
        uvs=uvs_out,
        normals=normals_out,
        tangents=tangents_out,
        texture=mesh.texture,
        metallic_roughness=mesh.metallic_roughness,
        unlit=mesh.unlit,
        normal_map=mesh.normal_map,
        occlusion_in_mr=mesh.occlusion_in_mr,
        material=mesh.material,
        emissive=mesh.emissive,
    )


def unwrap_mesh(
    mesh: TriangleMeshBatch[torch.Tensor],
    segmenter: str = "pec",
    resolution: int = 1024,
    padding: int = 1,
    weld_distance: float = 0.0,
) -> TriangleMeshBatch[torch.Tensor]:
    _validate_mesh(mesh, "unwrap_mesh")
    if segmenter not in ("pec", "adaptive"):
        raise ValueError("unwrap_mesh: segmenter must be 'pec' or 'adaptive'")
    segment_device = _compute_device(mesh.vertices) if segmenter == "pec" else torch.device("cpu")
    vertices_out: list[torch.Tensor] = []
    faces_out: list[torch.Tensor] = []
    uvs_out: list[torch.Tensor] = []
    colors_out: list[torch.Tensor] = []
    for index in range(mesh.vertices.shape[0]):
        vertices, faces, colors, _, _ = get_mesh_batch_item(mesh, index)
        vertices_np = vertices.detach().cpu().numpy().astype(np.float32)
        extent = (
            float(np.linalg.norm(vertices_np.max(0) - vertices_np.min(0)))
            if len(vertices_np)
            else 0.0
        )
        weld_absolute = weld_distance * extent if weld_distance > 0.0 else 0.0
        mapping, out_faces, out_uvs = _uv_unwrap(
            vertices.to(segment_device).float(),
            faces.to(segment_device).long(),
            segmenter,
            int(resolution),
            int(padding),
            weld_absolute,
        )
        out_uvs = out_uvs.copy()
        out_uvs[:, 1] = 1.0 - out_uvs[:, 1]
        vertices_out.append(torch.from_numpy(vertices_np[mapping]).to(vertices.device))
        faces_out.append(torch.from_numpy(out_faces).to(device=faces.device, dtype=torch.int32))
        uvs_out.append(torch.from_numpy(out_uvs.astype(np.float32)).to(vertices.device))
        if colors is not None:
            colors_out.append(colors[torch.from_numpy(mapping).to(colors.device)])
    result = _pack_uv_meshes(
        vertices_out,
        faces_out,
        uvs_out,
        colors_out if len(colors_out) == len(vertices_out) else None,
    )
    result.texture = mesh.texture
    return result


def paint_mesh(
    mesh: TriangleMeshBatch[torch.Tensor], voxel_colors: SparseVolume[torch.Tensor]
) -> TriangleMeshBatch[torch.Tensor]:
    _validate_mesh(mesh, "paint_mesh")
    if type(voxel_colors) is not SparseVolume:
        raise TypeError("paint_mesh: voxel_colors must be an exact SparseVolume")
    coords = voxel_colors.data
    colors = voxel_colors.voxel_colors
    batch_index = coords[:, 0].long()
    vertices_out: list[torch.Tensor] = []
    faces_out: list[torch.Tensor] = []
    colors_out: list[torch.Tensor] = []
    for index in range(mesh.vertices.shape[0]):
        vertices, faces, _, _, _ = get_mesh_batch_item(mesh, index)
        selected = batch_index == index
        item_mesh = TriangleMeshBatch(vertices=vertices[None], faces=faces[None])
        if not bool(selected.any()):
            painted = paint_mesh_default_colors(item_mesh)
        else:
            painted = paint_mesh_with_voxels(
                item_mesh,
                coords[selected, 1:],
                colors[selected],
                voxel_colors.resolution,
            )
        vertices_out.append(vertices)
        faces_out.append(faces)
        assert painted.vertex_colors is not None
        colors_out.append(painted.vertex_colors[0])
    return pack_variable_mesh_batch(vertices_out, faces_out, colors_out)


def bake_texture_from_voxel(
    mesh: TriangleMeshBatch[torch.Tensor],
    voxel_colors: SparseVolume[torch.Tensor],
    texture_size: int = 2048,
    reference_mesh: TriangleMeshBatch[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_mesh(mesh, "bake_texture_from_voxel")
    if reference_mesh is not None:
        _validate_mesh(reference_mesh, "bake_texture_from_voxel reference")
    if type(voxel_colors) is not SparseVolume:
        raise TypeError("bake_texture_from_voxel: voxel_colors must be an exact SparseVolume")
    if mesh.uvs is None:
        raise ValueError("BakeTextureFromVoxel: input mesh has no UVs")
    coords = voxel_colors.data
    batch_index = coords[:, 0].long()
    base_maps: list[torch.Tensor] = []
    mr_maps: list[torch.Tensor | None] = []
    for index in range(mesh.vertices.shape[0]):
        selected = batch_index == index
        vertices, faces, _, uvs, _ = get_mesh_batch_item(mesh, index)
        if not bool(selected.any()) or not faces.numel():
            base_maps.append(torch.zeros((texture_size, texture_size, 3)))
            mr_maps.append(None)
            continue
        reference = None
        if reference_mesh is not None:
            reference_index = index if reference_mesh.vertices.shape[0] > 1 else 0
            ref_vertices, ref_faces, _, _, _ = get_mesh_batch_item(reference_mesh, reference_index)
            reference = (ref_vertices, ref_faces)
        assert uvs is not None
        _, _, _, base, mr = bake_texture_from_voxel_fn(
            vertices,
            faces,
            coords[selected, 1:],
            voxel_colors.voxel_colors[selected],
            resolution=voxel_colors.resolution,
            texture_size=texture_size,
            uvs=uvs,
            reference=reference,
            pbar=_Progress(),
        )
        base_maps.append(base.float().clamp(0.0, 1.0).cpu())
        mr_maps.append(mr)
    metallic = [_mr_channel(value, 2, base_maps[0]) for value in mr_maps]
    roughness = [_mr_channel(value, 1, base_maps[0]) for value in mr_maps]
    return torch.stack(base_maps), torch.stack(metallic), torch.stack(roughness)


def bake_normal_map_from_mesh(
    low_poly: TriangleMeshBatch[torch.Tensor],
    high_poly: TriangleMeshBatch[torch.Tensor],
    resolution: int = 1024,
    cage_distance: float = 0.05,
    ignore_backfaces: bool = True,
) -> torch.Tensor:
    _validate_mesh(low_poly, "bake_normal_map_from_mesh low_poly")
    _validate_mesh(high_poly, "bake_normal_map_from_mesh high_poly")
    if low_poly.uvs is None:
        raise ValueError("BakeNormalMapFromMesh: low_poly has no UVs")
    device = _compute_device(low_poly.vertices)
    images: list[torch.Tensor] = []
    for index in range(low_poly.vertices.shape[0]):
        vertices, faces, _, uvs, normals = get_mesh_batch_item(low_poly, index)
        if not faces.numel():
            images.append(torch.full((resolution, resolution, 3), 0.5))
            continue
        assert uvs is not None
        uv_np = _normalize_uvs_to_unit(uvs.detach().cpu().numpy())
        low_vertices = vertices.to(device).float()
        low_faces = faces.to(device).long()
        low_normals, tangents = _vertex_tangents_for_item(
            low_vertices,
            low_faces,
            torch.from_numpy(uv_np).to(device),
            normals,
            device,
        )
        high_index = index if high_poly.vertices.shape[0] > 1 else 0
        high_vertices, high_faces, _, _, high_normals = get_mesh_batch_item(high_poly, high_index)
        high_vertices = high_vertices.to(device).float()
        high_faces = high_faces.to(device).long()
        if high_normals is None:
            high_normals = _compute_vertex_normals(high_vertices, high_faces)
        image = _bake_normal_map(
            high_vertices,
            high_faces,
            high_normals.to(device).float(),
            low_vertices.cpu().numpy(),
            low_faces.cpu().numpy().astype(np.uint32),
            uv_np,
            low_normals,
            tangents,
            resolution,
            cage_distance=float(cage_distance),
            ignore_backfaces=bool(ignore_backfaces),
        )
        images.append(torch.from_numpy(np.ascontiguousarray(image)).float())
    return torch.stack(images).clamp(0.0, 1.0)


def bake_ambient_occlusion(
    low_poly: TriangleMeshBatch[torch.Tensor],
    high_poly: TriangleMeshBatch[torch.Tensor],
    resolution: int = 1024,
    samples: int = 64,
    max_distance: float = 0.5,
    strength: float = 1.0,
    bias: float = 0.01,
) -> torch.Tensor:
    _validate_mesh(low_poly, "bake_ambient_occlusion low_poly")
    _validate_mesh(high_poly, "bake_ambient_occlusion high_poly")
    if low_poly.uvs is None:
        raise ValueError("BakeAmbientOcclusion: low_poly has no UVs")
    device = _compute_device(low_poly.vertices)
    images: list[torch.Tensor] = []
    for index in range(low_poly.vertices.shape[0]):
        vertices, faces, _, uvs, normals = get_mesh_batch_item(low_poly, index)
        if not faces.numel():
            images.append(torch.ones((resolution, resolution, 3)))
            continue
        assert uvs is not None
        uv_np = _normalize_uvs_to_unit(uvs.detach().cpu().numpy())
        low_vertices = vertices.to(device).float()
        low_faces = faces.to(device).long()
        if normals is None:
            normals = _compute_vertex_normals(low_vertices, low_faces)
        high_index = index if high_poly.vertices.shape[0] > 1 else 0
        high_vertices, high_faces, _, _, _ = get_mesh_batch_item(high_poly, high_index)
        image = _bake_ambient_occlusion(
            high_vertices.to(device).float(),
            high_faces.to(device).long(),
            low_vertices.cpu().numpy(),
            low_faces.cpu().numpy().astype(np.uint32),
            uv_np,
            normals.to(device).float(),
            resolution,
            num_samples=int(samples),
            max_distance=float(max_distance),
            strength=float(strength),
            bias=float(bias),
            pbar=_Progress(),
            pbar_range=(0, 1000),
        )
        images.append(torch.from_numpy(np.ascontiguousarray(image)).float())
    return torch.stack(images).clamp(0.0, 1.0)


def render_uv_atlas(mesh: TriangleMeshBatch[torch.Tensor], resolution: int = 1024) -> torch.Tensor:
    _validate_mesh(mesh, "render_uv_atlas")
    if mesh.uvs is None:
        raise RuntimeError("mesh has no UVs to render. Run UnwrapMesh first.")
    images: list[torch.Tensor] = []
    device = _compute_device(mesh.vertices)
    for index in range(mesh.vertices.shape[0]):
        _, faces, _, uvs, _ = get_mesh_batch_item(mesh, index)
        assert uvs is not None
        image = _uv_render_atlas(
            np.ascontiguousarray(uvs.detach().cpu().numpy(), dtype=np.float32),
            np.ascontiguousarray(faces.detach().cpu().numpy(), dtype=np.int64),
            int(resolution),
            device,
        )
        images.append(image.detach().cpu())
    return torch.stack(images)


def apply_texture_to_mesh(
    mesh: TriangleMeshBatch[torch.Tensor],
    base_color: torch.Tensor,
    metallic: torch.Tensor | None = None,
    roughness: torch.Tensor | None = None,
    occlusion: torch.Tensor | None = None,
    normal_map: torch.Tensor | None = None,
) -> TriangleMeshBatch[torch.Tensor]:
    _validate_mesh(mesh, "apply_texture_to_mesh")
    if mesh.uvs is None:
        raise ValueError("ApplyTextureToMesh: mesh has no UVs")
    new_uvs = mesh.uvs.clone()
    for index in range(mesh.vertices.shape[0]):
        vertices, _, _, uvs, _ = get_mesh_batch_item(mesh, index)
        assert uvs is not None
        normalized = _normalize_uvs_to_unit(uvs.detach().cpu().numpy())
        new_uvs[index, : vertices.shape[0]] = torch.from_numpy(normalized).to(new_uvs)
    result = copy.copy(mesh)
    result.uvs = new_uvs
    result.texture = base_color.float().clamp(0.0, 1.0).cpu()
    if normal_map is not None:
        device = _compute_device(mesh.vertices)
        tangents = torch.zeros((*mesh.vertices.shape[:2], 4), dtype=torch.float32)
        normals = torch.zeros_like(mesh.vertices, device=torch.device("cpu"), dtype=torch.float32)
        for index in range(mesh.vertices.shape[0]):
            vertices, faces, _, uvs, source_normals = get_mesh_batch_item(mesh, index)
            if not faces.numel():
                continue
            count = vertices.shape[0]
            item_normals, item_tangents = _vertex_tangents_for_item(
                vertices.to(device).float(),
                faces.to(device).long(),
                new_uvs[index, :count],
                source_normals,
                device,
            )
            tangents[index, :count] = item_tangents.cpu()
            normals[index, :count] = item_normals.cpu()
        result.normal_map = normal_map.float().clamp(0.0, 1.0).cpu()
        result.tangents = tangents
        result.normals = normals
    provided = [value for value in (metallic, roughness, occlusion) if value is not None]
    if provided:
        batch = int(provided[0].shape[0])
        height = max(int(value.shape[1]) for value in provided)
        width = max(int(value.shape[2]) for value in provided)

        def channel(image: torch.Tensor | None, default: float) -> torch.Tensor:
            if image is None:
                return torch.full((batch, height, width, 1), default)
            value = image[..., :1].float().clamp(0.0, 1.0).cpu()
            if value.shape[1:3] != (height, width):
                value = torch.nn.functional.interpolate(
                    value.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            return value

        result.metallic_roughness = torch.cat(
            [channel(occlusion, 1.0), channel(roughness, 1.0), channel(metallic, 0.0)],
            dim=-1,
        )
        result.occlusion_in_mr = occlusion is not None
    return result


__all__ = [
    "apply_texture_to_mesh",
    "bake_ambient_occlusion",
    "bake_normal_map_from_mesh",
    "bake_texture_from_voxel",
    "decimate_mesh",
    "paint_mesh",
    "remesh_mesh",
    "render_uv_atlas",
    "smooth_mesh_normals",
    "unwrap_mesh",
    "voxel_grid_to_mesh",
]
