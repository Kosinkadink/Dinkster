"""Torch triangle BVH traversal and single-view mesh shading."""

from __future__ import annotations

import math
from typing import Any, cast


def _msb(x: Any) -> Any:
    import torch

    result = torch.zeros_like(x)
    value = x.clone()
    for shift in (32, 16, 8, 4, 2, 1):
        shifted = value >> shift
        active = shifted > 0
        result = torch.where(active, result + shift, result)
        value = torch.where(active, shifted, value)
    return result


def _morton(value: Any) -> Any:
    value = value & 0x1FFFFF
    value = (value | (value << 32)) & 0x1F00000000FFFF
    value = (value | (value << 16)) & 0x1F0000FF0000FF
    value = (value | (value << 8)) & 0x100F00F00F00F00F
    value = (value | (value << 4)) & 0x10C30C30C30C30C3
    return (value | (value << 2)) & 0x1249249249249249


def build_bvh(tri: Any) -> dict[str, Any]:
    import torch

    count = tri.shape[0]
    amin, amax = tri.amin(1), tri.amax(1)
    if count == 1:
        return dict(
            LEAF=0,
            left=torch.empty(0, dtype=torch.int32, device=tri.device),
            right=torch.empty(0, dtype=torch.int32, device=tri.device),
            nmin=amin,
            nmax=amax,
            order=torch.zeros(1, dtype=torch.int32, device=tri.device),
            T=count,
        )
    center = (amin + amax) * 0.5
    lo, hi = center.amin(0), center.amax(0)
    quantized = (
        ((center - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1) * float((1 << 21) - 1)
    ).long()
    morton = (
        _morton(quantized[:, 0]) << 2 | _morton(quantized[:, 1]) << 1 | _morton(quantized[:, 2])
    ).long()
    order = torch.argsort(morton)
    sorted_codes = morton[order]
    order = order.to(torch.int32)

    def delta(i: Any, j: Any) -> Any:
        valid = (j >= 0) & (j < count)
        jj = j.clamp(0, count - 1)
        x = cast(Any, sorted_codes)[i] ^ cast(Any, sorted_codes)[jj]
        same = x == 0
        prefix = torch.where(same, torch.full_like(x, 63), 62 - _msb(x.clamp_min(1)))
        xi = i ^ jj
        tie = torch.where(xi == 0, torch.full_like(x, 32), 31 - _msb(xi.clamp_min(1)))
        return torch.where(
            valid, prefix + torch.where(same, tie, torch.zeros_like(prefix)), torch.full_like(x, -1)
        )

    indices = torch.arange(count - 1, dtype=torch.int32, device=tri.device)
    plus, minus = delta(indices, indices + 1), delta(indices, indices - 1)
    direction = torch.where(plus >= minus, torch.ones_like(indices), -torch.ones_like(indices))
    minimum = torch.minimum(plus, minus)
    maximum = torch.full_like(indices, 2)
    while True:
        condition = delta(indices, indices + maximum * direction) > minimum
        if not bool(condition.any()):
            break
        maximum = torch.where(condition, maximum * 2, maximum)
        if int(maximum.max()) > 2 * count:
            break
    length = torch.zeros_like(indices)
    probe = maximum.clone()
    while True:
        probe = probe // 2
        if int(probe.max()) == 0:
            break
        condition = delta(indices, indices + (length + probe) * direction) > minimum
        length = torch.where(condition, length + probe, length)
    end = indices + length * direction
    first, last = torch.minimum(indices, end), torch.maximum(indices, end)
    node_prefix = delta(first, last)
    split = torch.zeros_like(indices)
    divisor = torch.full_like(indices, 2)
    span = last - first
    while True:
        step = (span + divisor - 1) // divisor
        condition = delta(first, (first + split + step).clamp(max=count - 1)) > node_prefix
        split = torch.where(condition, split + step, split)
        if int(step.max()) <= 1:
            condition = delta(first, (first + split + 1).clamp(max=count - 1)) > node_prefix
            split = torch.where(condition, split + 1, split)
            break
        divisor = divisor * 2
    gamma = first + split
    left = torch.where(gamma == first, count + gamma, gamma)
    right = torch.where(gamma + 1 == last, count + gamma + 1, gamma + 1)
    del center, lo, hi, quantized, morton
    del indices, plus, minus, direction, minimum, maximum, condition, length, probe, end
    del first, last, node_prefix, split, divisor, span, step, gamma
    del delta
    sorted_codes = None
    nmin = torch.empty((2 * count, 3), device=tri.device)
    nmax = torch.empty((2 * count, 3), device=tri.device)
    nmin[count:], nmax[count:] = amin[order], amax[order]
    del amin, amax
    ready = torch.zeros(2 * count, dtype=torch.bool, device=tri.device)
    ready[count:] = True
    for _ in range(128):
        need = ~ready[: count - 1]
        if not bool(need.any()):
            break
        idx = torch.nonzero(need, as_tuple=True)[0]
        active = idx[ready[left[idx]] & ready[right[idx]]]
        if active.numel() == 0:
            break
        nmin[active] = torch.minimum(nmin[left[active]], nmin[right[active]])
        nmax[active] = torch.maximum(nmax[left[active]], nmax[right[active]])
        ready[active] = True
    return dict(
        LEAF=count,
        left=left,
        right=right,
        nmin=nmin,
        nmax=nmax,
        order=order.to(torch.int32),
        T=count,
    )


def ray_triangle(
    o: Any, d: Any, tri: Any, tmin: float, tmax: Any, cull_backface: bool = False
) -> tuple[Any, Any]:
    import torch

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
        hit = hit & ((torch.cross(e1, e2, dim=-1) * d).sum(-1) < 0)
    return hit, t


def closest_hit(
    orig: Any,
    dirs: Any,
    tri: Any,
    bvh: dict[str, Any],
    tmin: float = 1e-5,
    tmax: float = 1e30,
    cull_backface: bool = False,
) -> tuple[Any, Any, Any]:
    import torch

    count, max_stack = orig.shape[0], 64
    leaf = bvh["LEAF"]
    inv = 1.0 / torch.where(dirs.abs() < 1e-20, torch.full_like(dirs, 1e-20), dirs)
    best_t = torch.full((count,), float(tmax), device=orig.device)
    best_f = torch.full((count,), -1, dtype=torch.long, device=orig.device)
    stack = torch.full((count, max_stack), -1, dtype=torch.int32, device=orig.device)
    sp = torch.ones(count, dtype=torch.long, device=orig.device)
    stack[:, 0] = 0
    active = torch.arange(count, device=orig.device)
    while active.numel() > 0:
        a = active
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        t1 = (bvh["nmin"][node] - orig[a]) * inv[a]
        t2 = (bvh["nmax"][node] - orig[a]) * inv[a]
        near = torch.minimum(t1, t2).amax(-1)
        far = torch.maximum(t1, t2).amin(-1)
        within = (far >= near.clamp_min(tmin)) & (far >= tmin) & (near < best_t[a])
        isleaf = node >= leaf
        leaves = within & isleaf
        if bool(leaves.any()):
            ga = a[leaves]
            face = bvh["order"][node[leaves] - leaf]
            hit, t = ray_triangle(orig[ga], dirs[ga], tri[face], tmin, tmax, cull_backface)
            update = hit & (t < best_t[ga])
            gu = ga[update]
            best_t[gu], best_f[gu] = t[update], face[update].long()
        internal = within & ~isleaf
        if bool(internal.any()):
            gi = a[internal]
            s0 = sp[gi]
            stack[gi, s0.clamp(max=max_stack - 1)] = bvh["left"][node[internal]].to(torch.int32)
            sp[gi] = (s0 + 1).clamp(max=max_stack)
            s1 = sp[gi]
            stack[gi, s1.clamp(max=max_stack - 1)] = bvh["right"][node[internal]].to(torch.int32)
            sp[gi] = (s1 + 1).clamp(max=max_stack)
        active = a[sp[a] > 0]
    return best_t, best_f, best_f >= 0


def camera_basis(eye: Any, center: Any, up_hint: Any) -> tuple[Any, Any, Any]:
    import torch
    from torch.nn.functional import normalize

    forward = normalize(center - eye, dim=-1, eps=1e-6)
    up = (
        up_hint
        if float(torch.abs((forward * up_hint).sum())) < 0.99
        else torch.tensor([0.0, 0.0, 1.0], device=forward.device)
    )
    right = normalize(torch.cross(forward, up, dim=-1), dim=-1, eps=1e-6)
    return forward, right, torch.cross(right, forward, dim=-1)


def model_transform(values: Any, transform: Any, is_normal: bool = False) -> Any:
    import torch

    def vec(data: Any) -> Any:
        return values.new_tensor([data["x"], data["y"], data["z"]])

    scale, position = vec(transform["scale"]), vec(transform["position"])
    q = transform["quaternion"]
    x, y, z, w = (values.new_tensor(q[key]) for key in ("x", "y", "z", "w"))
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


def barycentric(p: Any, tri: Any) -> Any:
    import torch

    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    denominator = d00 * d11 - d01 * d01
    denominator = torch.where(
        denominator.abs() < 1e-20, torch.full_like(denominator, 1e-20), denominator
    )
    v, w = (d11 * d20 - d01 * d21) / denominator, (d00 * d21 - d01 * d20) / denominator
    return torch.stack((1.0 - v - w, v, w), dim=-1)


def render_view(
    tri: Any,
    bvh: dict[str, Any],
    uv: Any,
    faces: Any,
    texture: Any,
    eye: Any,
    forward: Any,
    right: Any,
    up: Any,
    fov: float,
    height: int,
    width: int,
    vertex_colors: Any = None,
    vertex_normals: Any = None,
    render_normal: bool = False,
) -> tuple[Any, Any, Any]:
    import torch
    from torch.nn.functional import normalize

    ys = 1.0 - (torch.arange(height, device=tri.device, dtype=torch.float32) + 0.5) / height * 2.0
    xs = (torch.arange(width, device=tri.device, dtype=torch.float32) + 0.5) / width * 2.0 - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    tangent = math.tan(0.5 * fov)
    directions = normalize(
        (
            right * (gx * tangent * (width / height))[..., None]
            + up * (gy * tangent)[..., None]
            + forward
        ).reshape(-1, 3),
        dim=-1,
        eps=1e-6,
    )
    origins = eye[None, :].expand(height * width, 3)
    image = torch.zeros((height * width, 3), device=tri.device)
    depth = torch.full((height * width,), float("inf"), device=tri.device)
    hit_all = torch.zeros(height * width, dtype=torch.bool, device=tri.device)
    for start in range(0, height * width, 1 << 22):
        end = min(start + (1 << 22), height * width)
        t_hit, face, hit = closest_hit(origins[start:end], directions[start:end], tri, bvh)
        if not bool(hit.any()):
            continue
        fh = face[hit].clamp_min(0)
        points = origins[start:end][hit] + t_hit[hit, None] * directions[start:end][hit]
        bary = barycentric(points, tri[fh])
        local = torch.zeros((end - start, 3), device=tri.device)
        if render_normal:
            if vertex_normals is not None:
                normal = normalize(
                    (bary[:, :, None] * vertex_normals[faces[fh]]).sum(1), dim=-1, eps=1e-6
                )
            else:
                normal = normalize(
                    torch.cross(
                        tri[fh][:, 1] - tri[fh][:, 0], tri[fh][:, 2] - tri[fh][:, 0], dim=-1
                    ),
                    dim=-1,
                    eps=1e-6,
                )
                normal = torch.where(
                    (normal * -directions[start:end][hit]).sum(-1, keepdim=True) < 0,
                    -normal,
                    normal,
                )
            view_normal = torch.stack(
                [(normal * right).sum(-1), (normal * up).sum(-1), (normal * -forward).sum(-1)],
                dim=-1,
            )
            local[hit] = (view_normal * 0.5 + 0.5).clamp(0.0, 1.0)
        elif texture is not None and uv is not None:
            sampled_uv = (bary[:, :, None] * uv[faces[fh]]).sum(1)
            grid = (sampled_uv * 2.0 - 1.0).view(1, 1, -1, 2)
            sample = torch.nn.functional.grid_sample(
                texture.permute(2, 0, 1)[None].float(),
                grid,
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )
            local[hit] = sample[0, :, 0, :].t()
        elif vertex_colors is not None:
            local[hit] = (bary[:, :, None] * vertex_colors[faces[fh]]).sum(1)
        else:
            normal = normalize(
                torch.cross(tri[fh][:, 1] - tri[fh][:, 0], tri[fh][:, 2] - tri[fh][:, 0], dim=-1),
                dim=-1,
                eps=1e-6,
            )
            light = (normal * -directions[start:end][hit]).sum(-1).abs().clamp(0.15, 1.0)
            local[hit] = torch.tensor([0.72, 0.72, 0.72], device=tri.device) * light[:, None]
        image[start:end] = local
        local_depth = torch.full((end - start,), float("inf"), device=tri.device)
        local_depth[hit] = t_hit[hit]
        depth[start:end], hit_all[start:end] = local_depth, hit
    # The reference fills uncovered colors before replacing every uncovered pixel
    # with the background. That fill cannot affect the returned image or mask.
    return (
        image.reshape(height, width, 3),
        hit_all.reshape(height, width),
        depth.reshape(height, width),
    )
