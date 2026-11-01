"""Gaussian density and Surface Nets from ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f."""

# pyright: basic
from __future__ import annotations

import importlib
from typing import Any

from .gaussian_splat import _quaternion_matrix


def density(
    xyz: Any,
    opacity: Any,
    scale: Any,
    quat: Any,
    rgb: Any,
    resolution: int,
    kernel: int,
    device: Any,
    sharpen: float,
    progress: Any,
) -> Any:
    torch = importlib.import_module("torch")
    pad = 4.0 * scale.median()
    lo, hi = xyz.amin(0) - pad, xyz.amax(0) + pad
    voxel = ((hi - lo).max() / resolution).clamp_min(1e-8)
    dx, dy, dz = (torch.ceil((hi - lo) / voxel).long() + 1).tolist()
    matrix = _quaternion_matrix(torch, quat)
    inverse = torch.einsum("nij,nj,nkj->nik", matrix, 1.0 / scale.clamp_min(1e-8) ** 2, matrix)
    widths = torch.ceil(3.0 * scale.amax(-1) / voxel).long().clamp(1, kernel)
    sharp = sharpen != 1.0
    vol = torch.zeros(dx * dy * dz, device=device)
    colors = torch.zeros(dx * dy * dz, 3, device=device, dtype=torch.float16)
    weights: Any = torch.zeros(dx * dy * dz, device=device, dtype=torch.float16) if sharp else None
    done = 0
    for half_width in range(1, kernel + 1):
        selected = (widths == half_width).nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            continue
        rng = torch.arange(-half_width, half_width + 1, device=device, dtype=torch.float32)
        offsets = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
        for start in range(0, selected.numel(), 4096):
            indices = selected[start : start + 4096]
            centers = xyz[indices]
            coords = ((centers - lo) / voxel).round()[:, None, :] + offsets[None]
            delta = lo + coords * voxel - centers[:, None, :]
            quad = torch.einsum("bmi,bij,bmj->bm", delta, inverse[indices], delta)
            weight = opacity[indices, None] * torch.exp(-0.5 * quad)
            weight = torch.where(quad < 9.0, weight, torch.zeros_like(weight))
            coords = coords.long()
            ix, iy, iz = (
                coords[..., axis].clamp(0, size - 1) for axis, size in enumerate((dx, dy, dz))
            )
            flat = (ix * (dy * dz) + iy * dz + iz).reshape(-1)
            vol.index_add_(0, flat, weight.reshape(-1))
            color_weight = weight.pow(sharpen) if sharp else weight
            colors.index_add_(
                0,
                flat,
                (color_weight[..., None] * rgb[indices, None, :]).reshape(-1, 3).to(torch.float16),
            )
            if sharp:
                weights.index_add_(0, flat, color_weight.reshape(-1).to(torch.float16))
            done += indices.numel()
            progress(min(1.0, done / max(1, xyz.shape[0])))
    normalizer = (weights if sharp else vol).reshape(dx, dy, dz)
    return (
        vol.reshape(dx, dy, dz),
        colors.reshape(dx, dy, dz, 3),
        normalizer,
        lo.cpu().numpy(),
        float(voxel),
    )


def _components_gpu(faces: Any, count: int) -> Any:
    torch = importlib.import_module("torch")
    a, b = torch.cat([faces[:, 0], faces[:, 1]]), torch.cat([faces[:, 1], faces[:, 2]])
    labels = torch.arange(count, device=faces.device)
    while True:
        grandparents = labels[labels]
        ga, gb = grandparents[a], grandparents[b]
        new = labels.clone()
        new.scatter_reduce_(0, labels[a], gb, "amin", include_self=True)
        new.scatter_reduce_(0, labels[b], ga, "amin", include_self=True)
        new.scatter_reduce_(0, a, gb, "amin", include_self=True)
        new.scatter_reduce_(0, b, ga, "amin", include_self=True)
        new = new[new]
        if torch.equal(new, labels):
            return labels
        labels = new


def _clean_gpu(vertices: Any, faces: Any, minimum: int, device: Any) -> Any:
    torch = importlib.import_module("torch")
    vt, ft = torch.as_tensor(vertices, device=device), torch.as_tensor(faces, device=device)
    nv = vt.shape[0]
    _, labels = torch.unique(_components_gpu(ft, nv), return_inverse=True)
    count = int(labels.max()) + 1
    flabel = labels[ft[:, 0]]
    keep = torch.bincount(labels, minlength=count) >= minimum
    if int(keep.sum()) > 1:
        fcount = torch.bincount(flabel, minlength=count)
        largest = int(torch.where(keep, fcount, fcount.new_tensor(-1)).argmax())
        v0, v1, v2 = vt[ft[:, 0]], vt[ft[:, 1]], vt[ft[:, 2]]
        volume = torch.zeros(count, device=device).scatter_add_(
            0, flabel, (v0 * torch.linalg.cross(v1, v2)).sum(-1)
        )
        idx = labels[:, None].expand(-1, 3)
        lo = torch.full((count, 3), float("inf"), device=device).scatter_reduce_(
            0, idx, vt, "amin", include_self=True
        )
        hi = torch.full((count, 3), float("-inf"), device=device).scatter_reduce_(
            0, idx, vt, "amax", include_self=True
        )
        tolerance = 1e-4 * (hi[largest] - lo[largest]).max()
        enclosed = (lo >= lo[largest] - tolerance).all(1) & (hi <= hi[largest] + tolerance).all(1)
        inner = (
            enclosed
            & (torch.sign(volume) != torch.sign(volume[largest]))
            & (torch.arange(count, device=device) != largest)
        )
        keep &= ~inner
    kept = ft[keep[flabel]]
    if kept.shape[0] == 0:
        return vertices[:0], faces[:0]
    used = torch.unique(kept)
    remap = torch.full((nv,), -1, dtype=torch.int64, device=device)
    remap[used] = torch.arange(used.shape[0], device=device)
    return vt[used].cpu().numpy(), remap[kept].cpu().numpy()


def clean_components(vertices: Any, faces: Any, minimum: int, device: Any) -> Any:
    np = importlib.import_module("numpy")
    management = importlib.import_module("dinkster_inference.model_management")
    if (
        not management.is_device_cpu(device)
        and management.get_free_memory(device) > 10 * faces.size * 8
    ):
        return _clean_gpu(vertices, faces, minimum, device)
    sparse = importlib.import_module("scipy.sparse")
    csgraph = importlib.import_module("scipy.sparse.csgraph")
    ndi = importlib.import_module("scipy.ndimage")
    nv = len(vertices)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    count, labels = csgraph.connected_components(
        sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(nv, nv)),
        directed=False,
    )
    flabel = labels[faces[:, 0]]
    keep = np.bincount(labels, minlength=count) >= minimum
    if keep.sum() > 1:
        fcount = np.bincount(flabel, minlength=count)
        largest = np.where(keep, fcount, -1).argmax()
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        volume = np.bincount(
            flabel, weights=np.einsum("ij,ij->i", v0, np.cross(v1, v2)), minlength=count
        )
        cidx = np.arange(count)
        lo = np.stack([ndi.minimum(vertices[:, axis], labels, cidx) for axis in range(3)], 1)
        hi = np.stack([ndi.maximum(vertices[:, axis], labels, cidx) for axis in range(3)], 1)
        tolerance = 1e-4 * (hi[largest] - lo[largest]).max()
        enclosed = (lo >= lo[largest] - tolerance).all(1) & (hi <= hi[largest] + tolerance).all(1)
        inner = (
            enclosed & (np.sign(volume) != np.sign(volume[largest])) & (np.arange(count) != largest)
        )
        keep &= ~inner
    faces = faces[keep[flabel]]
    if len(faces) == 0:
        return vertices[:0], faces
    used = np.unique(faces)
    remap = np.full(nv, -1, np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def surface_nets(volume: Any, level: float, voxel: float, origin: Any, device: Any) -> Any:
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    volume = volume.to(device=device, dtype=torch.float32)
    dx, dy, dz = volume.shape
    origin = torch.as_tensor(origin, device=device, dtype=torch.float32)
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))
    if min(dx, dy, dz) < 2:
        return empty
    inside = volume >= level
    corners = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )
    cs = [inside[x : x + dx - 1, y : y + dy - 1, z : z + dz - 1] for x, y, z in corners]
    any_in = cs[0] | cs[1] | cs[2] | cs[3] | cs[4] | cs[5] | cs[6] | cs[7]
    all_in = cs[0] & cs[1] & cs[2] & cs[3] & cs[4] & cs[5] & cs[6] & cs[7]
    active = any_in & ~all_in
    count = int(active.sum())
    if count == 0:
        return empty
    del any_in, all_in, cs
    ac = active.nonzero(as_tuple=False)
    offsets = torch.tensor(corners, device=device)
    offf = offsets.to(torch.float32)
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
    locations = []
    for start in range(0, count, 1 << 18):
        ci = ac[start : start + (1 << 18), None, :] + offsets[None]
        values = volume[ci[..., 0], ci[..., 1], ci[..., 2]]
        signs = values >= level
        v0, v1 = values[:, e0], values[:, e1]
        cross = (signs[:, e0] != signs[:, e1])[..., None].to(torch.float32)
        denominator = v1 - v0
        t = torch.where(
            denominator.abs() > 1e-12, (level - v0) / denominator, torch.full_like(denominator, 0.5)
        ).clamp(0, 1)
        points = torch.lerp(oe0, oe1, t[..., None])
        locations.append((points * cross).sum(1) / cross.sum(1).clamp_min(1.0))
    local = torch.cat(locations, 0) if len(locations) > 1 else locations[0]
    vertices = origin + (ac.to(torch.float32) + local) * voxel
    del locations, local, ac
    vertex_ids = torch.full((dx - 1, dy - 1, dz - 1), -1, dtype=torch.int32, device=device)
    vertex_ids[active] = torch.arange(count, dtype=torch.int32, device=device)
    del active
    faces = []

    def emit(cross: Any, solid: Any, a: Any, b: Any, d: Any, c: Any) -> None:
        valid = cross & (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        if not bool(valid.any()):
            return
        a, b, c, d, solid = a[valid], b[valid], c[valid], d[valid], solid[valid]
        p2, p4 = torch.where(solid, b, c), torch.where(solid, c, b)
        faces.append(torch.stack([a, p2, d], 1))
        faces.append(torch.stack([a, d, p4], 1))

    a = inside[: dx - 1, 1 : dy - 1, 1 : dz - 1]
    emit(
        a != inside[1:dx, 1 : dy - 1, 1 : dz - 1],
        a,
        vertex_ids[:, : dy - 2, : dz - 2],
        vertex_ids[:, 1 : dy - 1, : dz - 2],
        vertex_ids[:, 1 : dy - 1, 1 : dz - 1],
        vertex_ids[:, : dy - 2, 1 : dz - 1],
    )
    a = inside[1 : dx - 1, : dy - 1, 1 : dz - 1]
    emit(
        a != inside[1 : dx - 1, 1:dy, 1 : dz - 1],
        a,
        vertex_ids[: dx - 2, :, : dz - 2],
        vertex_ids[: dx - 2, :, 1 : dz - 1],
        vertex_ids[1 : dx - 1, :, 1 : dz - 1],
        vertex_ids[1 : dx - 1, :, : dz - 2],
    )
    a = inside[1 : dx - 1, 1 : dy - 1, : dz - 1]
    emit(
        a != inside[1 : dx - 1, 1 : dy - 1, 1:dz],
        a,
        vertex_ids[: dx - 2, : dy - 2, :],
        vertex_ids[1 : dx - 1, : dy - 2, :],
        vertex_ids[1 : dx - 1, 1 : dy - 1, :],
        vertex_ids[: dx - 2, 1 : dy - 1, :],
    )
    if not faces:
        return empty
    return vertices.cpu().numpy().astype(np.float32), torch.cat(faces, 0).cpu().numpy().astype(
        np.int64
    )


def _otsu(values: Any) -> float:
    np = importlib.import_module("numpy")
    histogram, edges = np.histogram(values, bins=256)
    histogram = histogram.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    weights, means = np.cumsum(histogram), np.cumsum(histogram * centers)
    rest = weights[-1] - weights
    mb = means / np.where(weights > 0, weights, 1.0)
    mf = (means[-1] - means) / np.where(rest > 0, rest, 1.0)
    variance = weights * rest * (mb - mf) ** 2
    variance[(weights <= 0) | (rest <= 0)] = -1.0
    return float(centers[int(np.argmax(variance))])


def _smooth(vertices: Any, faces: Any, iterations: int) -> Any:
    np = importlib.import_module("numpy")
    if iterations <= 0 or len(vertices) == 0 or len(faces) == 0:
        return vertices
    sparse = importlib.import_module("scipy.sparse")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], 0)
    edges = np.concatenate([edges, edges[:, ::-1]], 0)
    adj = sparse.coo_matrix(
        (np.ones(len(edges), np.float32), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    adj.data[:] = 1.0
    degree = np.clip(np.asarray(adj.sum(1)).ravel(), 1.0, None).astype(np.float32)[:, None]
    value = vertices.astype(np.float32)
    for _ in range(iterations):
        for factor in (0.5, -0.53):
            value = value + np.float32(factor) * ((adj @ value) / degree - value)
    return np.ascontiguousarray(value)


def _colors_gpu(
    colors: Any, weights: Any, vertices: Any, origin: Any, voxel: float, device: Any
) -> Any:
    torch = importlib.import_module("torch")
    dx, dy, dz = weights.shape
    vt = torch.as_tensor(vertices, device=device, dtype=torch.float32)
    org = torch.as_tensor(origin, device=device, dtype=torch.float32)
    gi = (vt - org) / voxel
    size = torch.tensor([dx, dy, dz], device=device, dtype=torch.float32)
    g = 2.0 * gi / (size - 1).clamp_min(1.0) - 1.0
    grid = torch.stack([g[:, 2], g[:, 1], g[:, 0]], -1)[None, None, None]

    def sample(value: Any) -> Any:
        value = value.to(device).permute(3, 0, 1, 2)[None].float()
        out = torch.nn.functional.grid_sample(
            value, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return out[0, :, 0, 0, :]

    return (sample(colors) / sample(weights[..., None]).clamp_min(1e-8)).T.cpu().numpy()


def mesh_item(
    value: Any,
    index: int,
    resolution: int,
    kernel: int,
    smooth: int,
    level_bias: float,
    minimum: int,
    min_opacity: float,
    sharpen: float,
    device: Any,
    progress: Any,
) -> Any:
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    management = importlib.import_module("dinkster_inference.model_management")
    count = (
        int(value.counts[index].item()) if value.counts is not None else value.positions.shape[1]
    )
    xyz, scale, quat = (
        a[index, :count].to(device=device, dtype=torch.float32)
        for a in (value.positions, value.scales, value.rotations)
    )
    opacity = value.opacities[index, :count].reshape(-1).to(device=device, dtype=torch.float32)
    rgb = (
        value.sh[index, :count, 0, :].to(device=device, dtype=torch.float32) * 0.28209479177387814
        + 0.5
    ).clamp(0, 1)
    keep = opacity >= min_opacity
    xyz, scale, quat, opacity, rgb = xyz[keep], scale[keep], quat[keep], opacity[keep], rgb[keep]
    if xyz.shape[0] == 0:
        return None
    volume, colors, weights, origin, voxel = density(
        xyz,
        opacity,
        scale,
        quat,
        rgb,
        resolution,
        kernel,
        device,
        sharpen,
        lambda f: progress(0.25 * f),
    )
    color_gpu = (
        not management.is_device_cpu(device)
        and management.get_free_memory(device) > 6 * volume.numel() * 4
    )
    if color_gpu:
        colors_cpu, weights_cpu = colors.cpu(), weights.half().cpu()
    else:
        colors_np, weights_np = (
            colors.cpu().numpy().astype(np.float32),
            weights.cpu().numpy().astype(np.float32),
        )
    del colors, weights
    progress(0.40)
    vmin, vmax = float(volume.min()), float(volume.max())
    occupied = volume[volume > vmax * 1e-3]
    if occupied.numel() == 0:
        return None
    level = min(
        max(_otsu(occupied.cpu().numpy()) * level_bias, vmin + 1e-6 * (vmax - vmin)),
        vmax - 1e-6 * (vmax - vmin),
    )
    sn_device = device
    if (
        not management.is_device_cpu(device)
        and management.get_free_memory(device) < 6 * volume.numel() * 4
    ):
        sn_device = torch.device("cpu")
        volume = volume.cpu()
    vertices, faces = surface_nets(volume, level, voxel, origin, sn_device)
    del volume
    progress(0.55)
    if minimum > 0 and len(faces) > 0:
        vertices, faces = clean_components(vertices, faces, minimum, device)
    if len(vertices) == 0 or len(faces) == 0:
        return None
    vertices = _smooth(vertices, faces, smooth)
    progress(0.7)
    if color_gpu:
        color = _colors_gpu(colors_cpu, weights_cpu, vertices, origin, voxel, device)
    else:
        map_coordinates = importlib.import_module("scipy.ndimage").map_coordinates
        coords = ((vertices - origin) / voxel).T
        num = np.stack(
            [map_coordinates(colors_np[..., c], coords, order=1, mode="nearest") for c in range(3)],
            -1,
        )
        den = map_coordinates(weights_np, coords, order=1, mode="nearest")
        color = num / np.clip(den, 1e-8, None)[:, None]
    progress(1.0)
    color = np.clip(color, 0, 1)
    color = np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )
    vertices = np.ascontiguousarray(vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32))
    return torch.from_numpy(vertices), torch.from_numpy(faces), torch.from_numpy(color)
