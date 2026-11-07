"""UV coverage, surface projection and ambient-occlusion rays."""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any

from dinkster_api.v1 import report_progress


def vertex_normals(vertices: Any, faces: Any) -> Any:
    torch = importlib.import_module("torch")

    triangles = vertices[faces]
    normal = torch.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
    )
    lo = vertices.min(0).values
    inverse = 1.0 / (float((vertices.max(0).values - lo).max().clamp_min(1e-9)) * 1e-5)
    quantized = ((vertices - lo) * inverse).round().to(torch.int64)
    _, group = torch.unique(quantized, dim=0, return_inverse=True)
    accumulated = vertices.new_zeros((int(group.max()) + 1, 3))
    for corner in range(3):
        accumulated.scatter_add_(0, group[faces[:, corner], None].expand_as(normal), normal)
    return torch.nn.functional.normalize(accumulated[group], p=2, dim=-1, eps=1e-6)


def rasterize_uv(faces: Any, uvs: Any, size: int) -> tuple[Any, Any, Any]:
    import torch

    device = uvs.device
    face_index = torch.zeros((size, size), dtype=torch.long, device=device)
    bary = torch.zeros((size, size, 3), device=device)
    coverage = torch.zeros((size, size), dtype=torch.bool, device=device)
    if faces.shape[0] == 0:
        return face_index, bary, coverage
    triangles = (uvs * float(size))[faces]
    x0, y0 = triangles[:, 0, 0], triangles[:, 0, 1]
    x1, y1 = triangles[:, 1, 0], triangles[:, 1, 1]
    x2, y2 = triangles[:, 2, 0], triangles[:, 2, 1]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegenerate = denominator.abs() > 1e-20
    xmin = torch.minimum(torch.minimum(x0, x1), x2).floor().clamp_(0, size - 1).long()
    xmax = torch.maximum(torch.maximum(x0, x1), x2).ceil().clamp_(0, size - 1).long()
    ymin = torch.minimum(torch.minimum(y0, y1), y2).floor().clamp_(0, size - 1).long()
    ymax = torch.maximum(torch.maximum(y0, y1), y2).ceil().clamp_(0, size - 1).long()
    for ty in range(0, size, 64):
        end_y = min(ty + 64, size)
        for tx in range(0, size, 64):
            end_x = min(tx + 64, size)
            candidates = (
                nondegenerate & (xmin < end_x) & (xmax >= tx) & (ymin < end_y) & (ymax >= ty)
            )
            if not bool(candidates.any()):
                continue
            indices = torch.nonzero(candidates, as_tuple=True)[0]
            ys = torch.arange(ty, end_y, dtype=torch.float32, device=device) + 0.5
            xs = torch.arange(tx, end_x, dtype=torch.float32, device=device) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            sx0, sy0 = x0[indices, None, None], y0[indices, None, None]
            sx1, sy1 = x1[indices, None, None], y1[indices, None, None]
            sx2, sy2 = x2[indices, None, None], y2[indices, None, None]
            divisor = denominator[indices, None, None]
            b0 = ((sy1 - sy2) * (xx - sx2) + (sx2 - sx1) * (yy - sy2)) / divisor
            b1 = ((sy2 - sy0) * (xx - sx2) + (sx0 - sx2) * (yy - sy2)) / divisor
            b2 = 1.0 - b0 - b1
            inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6)
            if not bool(inside.any()):
                continue
            hit = inside.any(dim=0)
            selected = inside.int().argmax(dim=0)
            weights = torch.stack(
                [value.gather(0, selected[None]).squeeze(0) for value in (b0, b1, b2)], dim=-1
            )
            face_index[ty:end_y, tx:end_x][hit] = indices[selected][hit]
            bary[ty:end_y, tx:end_x][hit] = weights[hit]
            coverage[ty:end_y, tx:end_x] |= hit
    return face_index, bary, coverage


def closest_points(
    points: Any, triangles: Any, bvh: dict[str, Any], return_face: bool = False
) -> Any:
    import torch

    from .mesh_remesh_ops import point_tri_closest

    count, stack_size = points.shape[0], 64
    stack = torch.full((count, stack_size), -1, dtype=torch.int32, device=points.device)
    sp = torch.ones(count, dtype=torch.long, device=points.device)
    stack[:, 0] = 0
    best = torch.full((count,), 1e30, device=points.device)
    result = points.clone()
    best_face = (
        torch.full((count,), -1, dtype=torch.long, device=points.device) if return_face else None
    )
    active = torch.arange(count, device=points.device)
    overflow = 0

    def distance(node: Any, query: Any) -> Any:
        delta = (bvh["nmin"][node] - query).clamp_min(0) + (query - bvh["nmax"][node]).clamp_min(0)
        return (delta * delta).sum(-1)

    while active.numel() > 0:
        a = active
        query = points[a]
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        within = distance(node, query) < best[a]
        is_leaf = node >= bvh["LEAF"]
        leaves = within & is_leaf
        if bool(leaves.any()):
            ga = a[leaves]
            faces = bvh["order"][node[leaves] - bvh["LEAF"]]
            projected, squared = point_tri_closest(query[leaves], triangles[faces])
            update = squared < best[ga]
            chosen = ga[update]
            best[chosen], result[chosen] = squared[update], projected[update]
            if best_face is not None:
                best_face[chosen] = faces[update].long()
        internal = within & ~is_leaf
        if bool(internal.any()):
            gi = a[internal]
            left, right = bvh["left"][node[internal]], bvh["right"][node[internal]]
            dl, dr = distance(left, query[internal]), distance(right, query[internal])
            near, far = torch.where(dl <= dr, left, right), torch.where(dl <= dr, right, left)
            s0 = sp[gi]
            stack[gi, s0.clamp(max=stack_size - 1)] = far
            sp[gi] = (s0 + 1).clamp(max=stack_size)
            s1 = sp[gi]
            overflow += int((s1 >= stack_size).sum())
            stack[gi, s1.clamp(max=stack_size - 1)] = near
            sp[gi] = (s1 + 1).clamp(max=stack_size)
        active = a[sp[a] > 0]
    if overflow:
        logging.warning("Mesh back-projection BVH stack overflow on %d pushes", overflow)
    return (result, best_face) if return_face else result


def any_hit(
    origins: Any, directions: Any, triangles: Any, bvh: Any, tmin: float, tmax: float
) -> Any:
    import torch

    from .mesh_render_ops import ray_triangle

    count, stack_size = origins.shape[0], 64
    inverse = 1.0 / torch.where(
        directions.abs() < 1e-20, torch.full_like(directions, 1e-20), directions
    )
    far_bound = torch.full((count,), float(tmax), device=origins.device)
    hit = torch.zeros(count, dtype=torch.bool, device=origins.device)
    stack = torch.full((count, stack_size), -1, dtype=torch.int32, device=origins.device)
    sp = torch.ones(count, dtype=torch.long, device=origins.device)
    stack[:, 0] = 0
    active = torch.arange(count, device=origins.device)
    while active.numel() > 0:
        a = active
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        t1 = (bvh["nmin"][node] - origins[a]) * inverse[a]
        t2 = (bvh["nmax"][node] - origins[a]) * inverse[a]
        near, far = torch.minimum(t1, t2).amax(-1), torch.maximum(t1, t2).amin(-1)
        within = (far >= near.clamp_min(tmin)) & (near <= far_bound[a]) & (far >= tmin)
        is_leaf = node >= bvh["LEAF"]
        leaves = within & is_leaf
        if bool(leaves.any()):
            ga = a[leaves]
            tri = triangles[bvh["order"][node[leaves] - bvh["LEAF"]]]
            found, _ = ray_triangle(origins[ga], directions[ga], tri, tmin, far_bound[ga])
            hit[ga[found]] = True
        internal = within & ~is_leaf
        if bool(internal.any()):
            gi = a[internal]
            s0 = sp[gi]
            stack[gi, s0.clamp(max=stack_size - 1)] = bvh["left"][node[internal]].to(torch.int32)
            sp[gi] = (s0 + 1).clamp(max=stack_size)
            s1 = sp[gi]
            stack[gi, s1.clamp(max=stack_size - 1)] = bvh["right"][node[internal]].to(torch.int32)
            sp[gi] = (s1 + 1).clamp(max=stack_size)
        active = a[(sp[a] > 0) & ~hit[a]]
    return hit


def fill_gutters(image: Any, mask: Any) -> Any:
    import torch

    if not bool(mask.any()):
        return image
    height, width = mask.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.int32, device=mask.device),
        torch.arange(width, dtype=torch.int32, device=mask.device),
        indexing="ij",
    )
    by, bx = torch.where(mask, yy, -1), torch.where(mask, xx, -1)
    step = 1 << ((max(height, width) - 1).bit_length() - 1)
    while step >= 1:
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = (yy + dy).clamp(0, height - 1), (xx + dx).clamp(0, width - 1)
                cy, cx = by[ny, nx], bx[ny, nx]
                valid = cy >= 0
                # Squared pixel distances need a sentinel above the supported image diagonal.
                candidate = torch.where(valid, (yy - cy) ** 2 + (xx - cx) ** 2, 1_073_741_824)
                current = torch.where(by >= 0, (yy - by) ** 2 + (xx - bx) ** 2, 1_073_741_824)
                take = valid & (candidate < current)
                by, bx = torch.where(take, cy, by), torch.where(take, cx, bx)
        step //= 2
    indices = by.clamp_min_(0).long().mul_(width).add_(bx.clamp_min_(0))
    return image.reshape(-1, image.shape[-1])[indices.reshape(-1)].reshape(height, width, -1)


def bake_occlusion(
    high_vertices: Any,
    high_faces: Any,
    low_vertices: Any,
    low_faces: Any,
    uvs: Any,
    normals: Any,
    size: int,
    samples: int,
    max_distance: float,
    strength: float,
    bias: float,
    progress_start: int,
    progress_end: int,
) -> Any:
    import torch

    from .mesh_render_ops import build_bvh

    model_management = importlib.import_module("dinkster_inference.model_management")
    device = low_vertices.device
    free = model_management.get_free_memory(device)
    # Bounds count rays, after converting the free-memory estimate to the same unit.
    ray_chunk = int(min(4_194_304, max(1_048_576, (free * 0.25) / (samples * 4 + 200))))
    face_index, bary, mask = rasterize_uv(low_faces, uvs, size)
    if not bool(mask.any()):
        return torch.ones((size, size, 3), device=device)
    indices, weights = low_faces[face_index[mask]], bary[mask]
    points = (weights[:, :, None] * low_vertices[indices]).sum(1)
    normal = torch.nn.functional.normalize(
        (weights[:, :, None] * normals[indices]).sum(1), dim=-1, eps=1e-6
    )
    triangles = high_vertices[high_faces.to(torch.int32)]
    bvh = build_bvh(triangles)
    diagonal = float((high_vertices.amax(0) - high_vertices.amin(0)).norm().clamp_min(1e-6))
    lift = max(1e-5, float(bias) * diagonal)
    far_bound = float(max_distance) * diagonal
    origins = closest_points(points, triangles, bvh) + normal * lift
    up = torch.where(
        normal[..., 2:3].abs() < 0.999,
        torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(normal),
        torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(normal),
    )
    tangent = torch.nn.functional.normalize(torch.cross(up, normal, dim=-1), dim=-1, eps=1e-6)
    bitangent = torch.cross(normal, tangent, dim=-1)
    count = points.shape[0]
    occluded = torch.zeros(count, device=device)
    per_chunk = max(1, ray_chunk // max(1, samples))
    chunks = max(1, (count + per_chunk - 1) // per_chunk)
    for chunk, start in enumerate(range(0, count, per_chunk)):
        end = min(start + per_chunk, count)
        n = end - start
        r1, r2 = torch.rand(n, samples, device=device), torch.rand(n, samples, device=device)
        radius = r1.sqrt()
        z = r1.mul_(-1.0).add_(1.0).clamp_min_(0.0).sqrt_()
        angle = r2.mul_(2.0 * math.pi)
        x, y = angle.cos().mul_(radius), angle.sin().mul_(radius)
        directions = tangent[start:end, None, :] * x[..., None]
        directions.addcmul_(bitangent[start:end, None, :], y[..., None])
        directions.addcmul_(normal[start:end, None, :], z[..., None])
        directions = torch.nn.functional.normalize(directions.reshape(-1, 3), dim=-1, eps=1e-6)
        rays = origins[start:end, None, :].expand(-1, samples, -1).reshape(-1, 3)
        hit = any_hit(rays, directions, triangles, bvh, lift, far_bound)
        occluded[start:end] = (
            hit.reshape(n, samples).sum(1, dtype=torch.float32).div_(float(samples))
        )
        report_progress(
            progress_start + ((progress_end - progress_start) * (chunk + 1)) // chunks, 1000
        )
    ao = occluded.mul_(-float(strength)).add_(1.0).clamp_(0.0, 1.0)
    image = torch.ones((size, size), device=device)
    image[mask] = ao
    return fill_gutters(image[..., None].repeat(1, 1, 3), mask)


def bake_normal_map(
    high_vertices: Any,
    high_faces: Any,
    high_normals: Any,
    vertices: Any,
    faces: Any,
    uvs: Any,
    normals: Any,
    tangents: Any,
    resolution: int,
    cage_distance: float,
    ignore_backfaces: bool,
) -> Any:
    import torch

    from .mesh_render_ops import barycentric, build_bvh, closest_hit

    face_index, weights, mask = rasterize_uv(faces, uvs, resolution)
    output = vertices.new_tensor([0.5, 0.5, 1.0]).repeat(resolution, resolution, 1)
    if not bool(mask.any()):
        return output
    vtri, bary = faces[face_index[mask]], weights[mask]

    def interpolate(attribute: Any) -> Any:
        return (bary[:, :, None] * attribute[vtri]).sum(1)

    points = interpolate(vertices)
    normal = torch.nn.functional.normalize(interpolate(normals), dim=-1, eps=1e-6)
    tangent = interpolate(tangents[:, :3])
    handedness = interpolate(tangents[:, 3:4])[:, 0]
    triangles = high_vertices[high_faces.to(torch.int32)]
    bvh = build_bvh(triangles)
    diagonal = float((high_vertices.amax(0) - high_vertices.amin(0)).norm().clamp_min(1e-6))
    cage = max(1e-6, cage_distance * diagonal)
    origin = points + normal * cage
    distance, hit_face, hit = closest_hit(
        origin, -normal, triangles, bvh, 0.0, 2.0 * cage, ignore_backfaces
    )
    selected = hit_face.clamp_min(0)
    hitpoint = origin - distance[:, None] * normal
    miss = ~hit
    # Already resolved cage rays must not pay for a second full BVH traversal.
    if bool(miss.any()):
        projected, projected_face = closest_points(points[miss], triangles, bvh, return_face=True)
        selected = selected.clone()
        hitpoint = hitpoint.clone()
        selected[miss], hitpoint[miss] = projected_face.clamp_min(0), projected
    high_bary = barycentric(hitpoint, triangles[selected])
    matched = torch.nn.functional.normalize(
        (high_bary[:, :, None] * high_normals[high_faces[selected]]).sum(1), dim=-1, eps=1e-6
    )
    tangent = torch.nn.functional.normalize(
        tangent - normal * (normal * tangent).sum(-1, keepdim=True), dim=-1, eps=1e-6
    )
    bitangent = handedness[:, None] * torch.cross(normal, tangent, dim=-1)
    z = (matched * normal).sum(-1)
    encoded = torch.stack([(matched * tangent).sum(-1), (matched * bitangent).sum(-1), z], dim=-1)
    encoded = torch.nn.functional.normalize(encoded, dim=-1, eps=1e-6)
    encoded[z < 0.0] = vertices.new_tensor([0.0, 0.0, 1.0])
    output[mask] = encoded.mul_(0.5).add_(0.5).clamp_(0.0, 1.0)
    return fill_gutters(output, mask)
