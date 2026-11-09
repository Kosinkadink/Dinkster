"""Surface Nets extraction adapted from ComfyUI bc5f880d nodes_gaussian_splat.

Per-gaussian covariance windows, accelerator Surface Nets and connected
components preserve the reference's bounded-memory fast paths.
"""

# pyright: basic

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import torch
from dinkster_api.v1 import TriangleMeshBatch
from dinkster_inference import model_management
from scipy.ndimage import map_coordinates
from scipy.ndimage import maximum as ndi_maximum
from scipy.ndimage import minimum as ndi_minimum
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _inverse_covariance(scale, quat):
    q = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    rotation = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    return torch.einsum("nij,nj,nkj->nik", rotation, 1.0 / scale.clamp_min(1e-8) ** 2, rotation)


def _splat_density(xyz, opacity, scale, quat, rgb, res, kernel, device, color_sharpen, progress):
    pad = 4.0 * scale.median()
    lo, hi = xyz.amin(0) - pad, xyz.amax(0) + pad
    voxel = ((hi - lo).max() / res).clamp_min(1e-8)
    dx, dy, dz = (torch.ceil((hi - lo) / voxel).long() + 1).tolist()
    sharp = color_sharpen != 1.0
    volume_bytes = dx * dy * dz * (4 + 3 * 2 + (2 if sharp else 0))
    if (
        not model_management.is_device_cpu(device)
        and cast(float, model_management.get_free_memory(device)) < volume_bytes
    ):
        device = torch.device("cpu")
        xyz, opacity, scale, quat, rgb, lo, voxel = (
            value.cpu() for value in (xyz, opacity, scale, quat, rgb, lo, voxel)
        )
    sinv = _inverse_covariance(scale, quat)
    kreq = torch.ceil(3.0 * scale.amax(-1) / voxel).long().clamp(1, int(kernel))
    vol = torch.zeros(dx * dy * dz, device=device)
    colvol = torch.zeros(dx * dy * dz, 3, device=device, dtype=torch.float16)
    wcol = torch.zeros(dx * dy * dz, device=device, dtype=torch.float16) if sharp else None
    done = 0
    for k in range(1, int(kernel) + 1):
        sel = (kreq == k).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        rng = torch.arange(-k, k + 1, device=device, dtype=torch.float32)
        off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
        for start in range(0, sel.numel(), 4096):
            gi = sel[start : start + 4096]
            centers = xyz[gi]
            idx = ((centers - lo) / voxel).round()[:, None, :] + off[None]
            delta = lo + idx * voxel - centers[:, None, :]
            quad = torch.einsum("bmi,bij,bmj->bm", delta, sinv[gi], delta)
            weight = opacity[gi, None] * torch.exp(-0.5 * quad)
            weight = torch.where(quad < 9.0, weight, torch.zeros_like(weight))
            ii = idx.long()
            ix, iy, iz = (
                ii[..., 0].clamp(0, dx - 1),
                ii[..., 1].clamp(0, dy - 1),
                ii[..., 2].clamp(0, dz - 1),
            )
            flat = (ix * (dy * dz) + iy * dz + iz).reshape(-1)
            vol.index_add_(0, flat, weight.reshape(-1))
            wp = weight.pow(color_sharpen) if sharp else weight
            colvol.index_add_(0, flat, (wp[..., None] * rgb[gi, None, :]).reshape(-1, 3).half())
            if wcol is not None:
                wcol.index_add_(0, flat, wp.reshape(-1).half())
            done += gi.numel()
            progress(min(1.0, done / max(1, xyz.shape[0])))
    norm = (wcol if wcol is not None else vol).reshape(dx, dy, dz)
    return (
        vol.reshape(dx, dy, dz),
        colvol.reshape(dx, dy, dz, 3),
        norm,
        lo.cpu().numpy(),
        float(voxel),
    )


def _connected_components_gpu(faces, nv):
    a, b = torch.cat([faces[:, 0], faces[:, 1]]), torch.cat([faces[:, 1], faces[:, 2]])
    labels = torch.arange(nv, device=faces.device)
    while True:
        gp = labels[labels]
        ga, gb = gp[a], gp[b]
        new = labels.clone()
        new.scatter_reduce_(0, labels[a], gb, "amin", include_self=True)
        new.scatter_reduce_(0, labels[b], ga, "amin", include_self=True)
        new.scatter_reduce_(0, a, gb, "amin", include_self=True)
        new.scatter_reduce_(0, b, ga, "amin", include_self=True)
        new = new[new]
        if torch.equal(new, labels):
            return labels
        labels = new


def _clean_components_gpu(verts, faces, min_verts, device):
    vt, ft = torch.as_tensor(verts, device=device), torch.as_tensor(faces, device=device)
    nv = vt.shape[0]
    _, label = torch.unique(_connected_components_gpu(ft, nv), return_inverse=True)
    ncomp = int(label.max()) + 1
    flabel = label[ft[:, 0]]
    keep = torch.bincount(label, minlength=ncomp) >= min_verts
    if int(keep.sum()) > 1:
        fcount = torch.bincount(flabel, minlength=ncomp)
        largest = int(torch.where(keep, fcount, fcount.new_tensor(-1)).argmax())
        v0, v1, v2 = vt[ft[:, 0]], vt[ft[:, 1]], vt[ft[:, 2]]
        cvol = torch.zeros(ncomp, device=device).scatter_add_(
            0, flabel, (v0 * torch.linalg.cross(v1, v2)).sum(-1)
        )
        idx3 = label[:, None].expand(-1, 3)
        cmin = torch.full((ncomp, 3), float("inf"), device=device).scatter_reduce_(
            0, idx3, vt, "amin", include_self=True
        )
        cmax = torch.full((ncomp, 3), float("-inf"), device=device).scatter_reduce_(
            0, idx3, vt, "amax", include_self=True
        )
        tol = 1e-4 * (cmax[largest] - cmin[largest]).max()
        enclosed = (cmin >= cmin[largest] - tol).all(1) & (cmax <= cmax[largest] + tol).all(1)
        inner = (
            enclosed
            & (torch.sign(cvol) != torch.sign(cvol[largest]))
            & (torch.arange(ncomp, device=device) != largest)
        )
        keep &= ~inner
    faces_k = ft[keep[flabel]]
    if faces_k.shape[0] == 0:
        return verts[:0], faces[:0]
    used = torch.unique(faces_k)
    remap = torch.full((nv,), -1, dtype=torch.int64, device=device)
    remap[used] = torch.arange(used.shape[0], device=device)
    return vt[used].cpu().numpy(), remap[faces_k].cpu().numpy()


def _clean_components(verts, faces, min_verts, device):
    if (
        not model_management.is_device_cpu(device)
        and model_management.get_free_memory(device) > 10 * faces.size * 8
    ):
        return _clean_components_gpu(verts, faces, min_verts, device)
    nv = len(verts)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    ncomp, label = connected_components(
        coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(nv, nv)),
        directed=False,
    )
    flabel = label[faces[:, 0]]
    keep = np.bincount(label, minlength=ncomp) >= min_verts
    if keep.sum() > 1:
        fcount = np.bincount(flabel, minlength=ncomp)
        largest = np.where(keep, fcount, -1).argmax()
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cvol = np.bincount(
            flabel, weights=np.einsum("ij,ij->i", v0, np.cross(v1, v2)), minlength=ncomp
        )
        cidx = np.arange(ncomp)
        cmin = np.stack([ndi_minimum(verts[:, axis], label, cidx) for axis in range(3)], 1)
        cmax = np.stack([ndi_maximum(verts[:, axis], label, cidx) for axis in range(3)], 1)
        tol = 1e-4 * (cmax[largest] - cmin[largest]).max()
        enclosed = (cmin >= cmin[largest] - tol).all(1) & (cmax <= cmax[largest] + tol).all(1)
        inner = enclosed & (np.sign(cvol) != np.sign(cvol[largest])) & (np.arange(ncomp) != largest)
        keep &= ~inner
    faces = faces[keep[flabel]]
    if len(faces) == 0:
        return verts[:0], faces
    used = np.unique(faces)
    remap = np.full(nv, -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces]


def _surface_nets(vol, level, voxel, origin, device):
    vol = vol.to(device=device, dtype=torch.float32)
    dx, dy, dz = vol.shape
    origin_t = torch.as_tensor(origin, device=device, dtype=torch.float32)
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))
    if dx < 2 or dy < 2 or dz < 2:
        return empty
    inside = vol >= level
    corners = [
        inside[ox : ox + dx - 1, oy : oy + dy - 1, oz : oz + dz - 1]
        for ox, oy, oz in (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        )
    ]
    any_in = (
        corners[0]
        | corners[1]
        | corners[2]
        | corners[3]
        | corners[4]
        | corners[5]
        | corners[6]
        | corners[7]
    )
    all_in = (
        corners[0]
        & corners[1]
        & corners[2]
        & corners[3]
        & corners[4]
        & corners[5]
        & corners[6]
        & corners[7]
    )
    active = any_in & ~all_in
    nv = int(active.sum())
    if nv == 0:
        return empty
    del any_in, all_in, corners
    ac = active.nonzero(as_tuple=False)
    offs = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        device=device,
    )
    offf = offs.to(torch.float32)
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
    e0, e1 = edges[:, 0], edges[:, 1]
    oe0, oe1 = offf[e0], offf[e1]
    loc = []
    for start in range(0, nv, 1 << 18):
        ci = ac[start : start + (1 << 18), None, :] + offs[None]
        cval = vol[ci[..., 0], ci[..., 1], ci[..., 2]]
        csl = cval >= level
        v0, v1 = cval[:, e0], cval[:, e1]
        cross = (csl[:, e0] != csl[:, e1])[..., None].to(torch.float32)
        denom = v1 - v0
        t = torch.where(
            denom.abs() > 1e-12, (level - v0) / denom, torch.full_like(denom, 0.5)
        ).clamp(0, 1)
        pts = torch.lerp(oe0, oe1, t[..., None])
        loc.append((pts * cross).sum(1) / cross.sum(1).clamp_min(1.0))
    local = torch.cat(loc, 0) if len(loc) > 1 else loc[0]
    verts = origin_t + (ac.to(torch.float32) + local) * voxel
    del loc, local, ac
    vid = torch.full((dx - 1, dy - 1, dz - 1), -1, dtype=torch.int32, device=device)
    vid[active] = torch.arange(nv, dtype=torch.int32, device=device)
    del active
    faces = []

    def emit(cr, sol, a, b, d, c):
        valid = cr & (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        if not bool(valid.any()):
            return
        a, b, c, d, sol = a[valid], b[valid], c[valid], d[valid], sol[valid]
        p2, p4 = torch.where(sol, b, c), torch.where(sol, c, b)
        faces.append(torch.stack([a, p2, d], 1))
        faces.append(torch.stack([a, d, p4], 1))

    a = inside[0 : dx - 1, 1 : dy - 1, 1 : dz - 1]
    emit(
        a != inside[1:dx, 1 : dy - 1, 1 : dz - 1],
        a,
        vid[:, 0 : dy - 2, 0 : dz - 2],
        vid[:, 1 : dy - 1, 0 : dz - 2],
        vid[:, 1 : dy - 1, 1 : dz - 1],
        vid[:, 0 : dy - 2, 1 : dz - 1],
    )
    a = inside[1 : dx - 1, 0 : dy - 1, 1 : dz - 1]
    emit(
        a != inside[1 : dx - 1, 1:dy, 1 : dz - 1],
        a,
        vid[0 : dx - 2, :, 0 : dz - 2],
        vid[0 : dx - 2, :, 1 : dz - 1],
        vid[1 : dx - 1, :, 1 : dz - 1],
        vid[1 : dx - 1, :, 0 : dz - 2],
    )
    a = inside[1 : dx - 1, 1 : dy - 1, 0 : dz - 1]
    emit(
        a != inside[1 : dx - 1, 1 : dy - 1, 1:dz],
        a,
        vid[0 : dx - 2, 0 : dy - 2, :],
        vid[1 : dx - 1, 0 : dy - 2, :],
        vid[1 : dx - 1, 1 : dy - 1, :],
        vid[0 : dx - 2, 1 : dy - 1, :],
    )
    if not faces:
        return empty
    return verts.cpu().numpy().astype(np.float32), torch.cat(faces, 0).cpu().numpy().astype(
        np.int64
    )


def _otsu_level(values):
    hist, edges = np.histogram(values, bins=256)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight = np.cumsum(hist)
    mean = np.cumsum(hist * centers)
    other = weight[-1] - weight
    mb = mean / np.where(weight > 0, weight, 1.0)
    mf = (mean[-1] - mean) / np.where(other > 0, other, 1.0)
    variance = weight * other * (mb - mf) ** 2
    variance[(weight <= 0) | (other <= 0)] = -1.0
    return float(centers[int(np.argmax(variance))])


def _taubin_smooth(verts, faces, iters):
    if iters <= 0 or len(verts) == 0 or len(faces) == 0:
        return verts
    nv = len(verts)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    edges = np.concatenate([edges, edges[:, ::-1]], 0)
    adj = coo_matrix(
        (np.ones(len(edges), np.float32), (edges[:, 0], edges[:, 1])), shape=(nv, nv)
    ).tocsr()
    adj.data[:] = 1.0
    deg = np.clip(np.asarray(adj.sum(1)).ravel(), 1.0, None).astype(np.float32)[:, None]
    value = verts.astype(np.float32)
    for _ in range(int(iters)):
        for factor in (0.5, -0.53):
            value = value + np.float32(factor) * ((adj @ value) / deg - value)
    return np.ascontiguousarray(value)


def _sample_vertex_colours_gpu(colvol, colnorm, verts, origin, voxel, device):
    size = torch.tensor(colnorm.shape, device=device, dtype=torch.float32)
    vt, org = (
        torch.as_tensor(verts, device=device, dtype=torch.float32),
        torch.as_tensor(origin, device=device, dtype=torch.float32),
    )
    grid_index = 2.0 * ((vt - org) / voxel) / (size - 1).clamp_min(1.0) - 1.0
    grid = torch.stack([grid_index[:, 2], grid_index[:, 1], grid_index[:, 0]], -1)[None, None, None]

    def sample(value):
        inp = value.to(device).permute(3, 0, 1, 2)[None].float()
        out = torch.nn.functional.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return out[0, :, 0, 0, :]

    return (sample(colvol) / sample(colnorm[..., None]).clamp_min(1e-8)).T.cpu().numpy()


def _gaussian_to_mesh(
    splat,
    index,
    res,
    kernel,
    smooth,
    level_bias,
    min_component,
    min_opacity,
    color_sharpen,
    device,
    progress,
):
    end = int(splat.counts[index].item()) if splat.counts is not None else splat.positions.shape[1]
    xyz = splat.positions[index, :end].to(device=device, dtype=torch.float32)
    scale = splat.scales[index, :end].to(device=device, dtype=torch.float32)
    quat = splat.rotations[index, :end].to(device=device, dtype=torch.float32)
    opacity = splat.opacities[index, :end].reshape(-1).to(device=device, dtype=torch.float32)
    rgb = (
        splat.sh[index, :end, 0, :].to(device=device, dtype=torch.float32) * 0.28209479177387814
        + 0.5
    ).clamp(0, 1)
    keep = opacity >= min_opacity
    xyz, scale, quat, opacity, rgb = xyz[keep], scale[keep], quat[keep], opacity[keep], rgb[keep]
    if xyz.shape[0] == 0:
        return None
    vol, colvol, colnorm, origin, voxel = _splat_density(
        xyz,
        opacity,
        scale,
        quat,
        rgb,
        res,
        kernel,
        device,
        color_sharpen,
        lambda value: progress(0.25 * value),
    )
    device = vol.device
    colour_gpu = (
        not model_management.is_device_cpu(device)
        and cast(float, model_management.get_free_memory(device)) > 6 * vol.numel() * 4
    )
    if colour_gpu:
        colvol_cpu, colnorm_cpu = colvol.cpu(), colnorm.half().cpu()
    else:
        colvol_np, colnorm_np = (
            colvol.cpu().numpy().astype(np.float32),
            colnorm.cpu().numpy().astype(np.float32),
        )
    del colvol, colnorm
    progress(0.40)
    vmin, vmax = float(vol.min()), float(vol.max())
    occupied = vol[vol > vmax * 1e-3]
    if occupied.numel() == 0:
        return None
    level = min(
        max(_otsu_level(occupied.cpu().numpy()) * level_bias, vmin + 1e-6 * (vmax - vmin)),
        vmax - 1e-6 * (vmax - vmin),
    )
    sn_device = device
    if (
        not model_management.is_device_cpu(device)
        and cast(float, model_management.get_free_memory(device)) < 6 * vol.numel() * 4
    ):
        sn_device, vol = torch.device("cpu"), vol.cpu()
    verts, faces = _surface_nets(vol, level, voxel, origin, sn_device)
    del vol
    progress(0.55)
    if min_component > 0 and len(faces) > 0:
        verts, faces = _clean_components(verts, faces, min_component, device)
    if len(verts) == 0 or len(faces) == 0:
        return None
    verts = _taubin_smooth(verts, faces, smooth)
    progress(0.7)
    if colour_gpu:
        col = _sample_vertex_colours_gpu(colvol_cpu, colnorm_cpu, verts, origin, voxel, device)
    else:
        coords = ((verts - origin) / voxel).T
        num = np.stack(
            [
                map_coordinates(colvol_np[..., channel], coords, order=1, mode="nearest")
                for channel in range(3)
            ],
            -1,
        )
        den = map_coordinates(colnorm_np, coords, order=1, mode="nearest")
        col = num / np.clip(den, 1e-8, None)[:, None]
    progress(1.0)
    col = np.clip(col, 0, 1)
    col = np.where(col <= 0.04045, col / 12.92, ((col + 0.055) / 1.055) ** 2.4).astype(np.float32)
    verts = np.ascontiguousarray(verts * np.array([1.0, -1.0, -1.0], dtype=np.float32))
    return torch.from_numpy(verts), torch.from_numpy(faces), torch.from_numpy(col)


def splat_to_mesh(
    splat,
    resolution,
    kernel,
    smooth,
    level,
    min_component,
    min_opacity,
    color_sharpen,
    report_progress,
):
    device = model_management.get_torch_device()
    batch = splat.positions.shape[0]
    vertices, faces, colors = [], [], []
    for index in range(batch):

        def progress(value, base=index * 1000):
            report_progress(base + int(min(max(value, 0.0), 1.0) * 1000), batch * 1000)

        result = _gaussian_to_mesh(
            splat,
            index,
            resolution,
            kernel,
            smooth,
            level,
            min_component,
            min_opacity,
            color_sharpen,
            device,
            progress,
        )
        if result is None:
            logging.warning(
                "SplatToMesh: splat %d produced no surface; emitting an empty mesh.", index
            )
            result = (
                torch.zeros((0, 3)),
                torch.zeros((0, 3), dtype=torch.int64),
                torch.zeros((0, 3)),
            )
        vertex, face, color = result
        vertices.append(vertex)
        faces.append(face)
        colors.append(color)
        report_progress((index + 1) * 1000, batch * 1000)

    def padded(values):
        count = max(value.shape[0] for value in values)
        result = values[0].new_zeros((batch, count, 3))
        for index, value in enumerate(values):
            result[index, : value.shape[0]] = value
        return result

    return TriangleMeshBatch(
        vertices=padded(vertices),
        faces=padded(faces),
        vertex_colors=padded(colors),
        vertex_counts=torch.tensor([value.shape[0] for value in vertices]),
        face_counts=torch.tensor([value.shape[0] for value in faces]),
        unlit=True,
    )
