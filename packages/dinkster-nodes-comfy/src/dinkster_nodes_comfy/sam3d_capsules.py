"""Bounded analytic ray-capsule rendering for OpenPose and SCAIL bodies."""

from __future__ import annotations

import importlib
from typing import Any

from .sam3d_pose import (
    OPENPOSE_18_PAIRS,
    OPENPOSE_HAND_COLORS_21,
    OPENPOSE_HAND_PAIRS,
    OPENPOSE_RAINBOW_18,
    SCAIL_LIMB_COLORS_17,
    openpose_render_keypoints,
)

np = importlib.import_module("numpy")
torch = importlib.import_module("torch")
management = importlib.import_module("dinkster_inference.model_management")


def _build_specs(
    persons: list[dict[str, Any]],
    pose: Any,
    include_hands: bool,
    palette: str,
    person_brightness_falloff: float,
) -> tuple[Any, Any, Any, Any]:
    starts: list[Any] = []
    ends: list[Any] = []
    colors: list[Any] = []
    hands: list[bool] = []
    body_colors = (
        SCAIL_LIMB_COLORS_17
        if palette == "scail"
        else OPENPOSE_RAINBOW_18[: len(OPENPOSE_18_PAIRS)]
    ).astype(np.float32)
    hand_colors = OPENPOSE_HAND_COLORS_21.astype(np.float32)
    falloff = max(0.0, min(1.0, float(person_brightness_falloff)))
    for k, person in enumerate(persons):
        translation = person.get("pred_cam_t")
        body = openpose_render_keypoints(person, pose, "body", dim=3)
        if body is None or translation is None:
            continue
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        body = body + translation[None, :]
        pastel = 0.0 if k == 0 else 1.0 - falloff**k

        def append_limb(a: Any, b: Any, rgb: Any, hand: bool, mix: float = pastel) -> None:
            starts.append(a)
            ends.append(b)
            hands.append(hand)
            rgb = rgb if mix <= 0 else rgb * (1.0 - mix) + mix
            colors.append(np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32))

        spine_dir = None
        spine_length = 0.0
        if palette == "scail":
            spine = body[1] - 0.5 * (body[8] + body[11])
            spine_length = float(np.linalg.norm(spine))
            if np.all(np.isfinite(spine)) and spine_length > 1e-6:
                spine_dir = spine / spine_length
        count = 13 if palette == "scail" else len(OPENPOSE_18_PAIRS)
        for index, (a, b) in enumerate(OPENPOSE_18_PAIRS[:count]):
            start, end = body[a], body[b]
            if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
                continue
            if start[2] <= 0 or end[2] <= 0:
                continue
            if palette == "scail" and index == 12:
                nose = end - start
                length = float(np.linalg.norm(nose))
                if length > 1e-6 and spine_dir is not None:
                    mixed = 0.6 * spine_dir + 0.4 * nose / length
                    mixed = mixed / max(float(np.linalg.norm(mixed)), 1e-6)
                    end = start + mixed * (length * 0.5)
                elif length > 1e-6:
                    end = start + nose * 0.5
                elif spine_dir is not None:
                    end = start + spine_dir * (spine_length * 0.3)
            append_limb(start, end, body_colors[index], False)
        if include_hands:
            hand_points = [
                openpose_render_keypoints(person, pose, part, dim=3)
                for part in ("hand_r", "hand_l")
            ]
            hand_points = [
                points + translation[None, :] for points in hand_points if points is not None
            ]
            for a, b in OPENPOSE_HAND_PAIRS:
                for points in hand_points:
                    start, end = points[a], points[b]
                    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
                        continue
                    if start[2] <= 0 or end[2] <= 0:
                        continue
                    append_limb(start, end, hand_colors[(a + b) % len(hand_colors)], True)
    if not starts:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0, 4), np.float32),
            np.zeros((0,), bool),
        )
    return (
        np.stack(starts).astype(np.float32),
        np.stack(ends).astype(np.float32),
        np.stack(colors).astype(np.float32),
        np.asarray(hands, dtype=bool),
    )


def _ray_capsule_t(rays: Any, starts: Any, ends: Any, axis: Any, length: Any, radius: Any) -> Any:
    radius_sq = radius * radius
    dn, da, db = (
        rays @ axis.transpose(0, 1),
        rays @ starts.transpose(0, 1),
        rays @ ends.transpose(0, 1),
    )
    an = (starts * axis).sum(-1)
    a_sq, b_sq = (starts * starts).sum(-1), (ends * ends).sum(-1)
    a = 1.0 - dn * dn
    b = -2.0 * (da - dn * an)
    c = a_sq - an * an - radius_sq
    disc = b * b - 4.0 * a * c
    cylinder = (-b - torch.sqrt(disc.clamp(min=0.0))) / (2.0 * a.clamp(min=1e-9))
    projection = cylinder * dn - an
    valid = (
        (disc >= 0) & (a > 1e-7) & (cylinder > 1e-6) & (projection >= 0.0) & (projection <= length)
    )
    cylinder = torch.where(valid, cylinder, torch.full_like(cylinder, float("inf")))
    disc_a = da * da - (a_sq - radius_sq)
    sphere_a = da - torch.sqrt(disc_a.clamp(min=0.0))
    valid_a = (disc_a >= 0) & (sphere_a > 1e-6) & (sphere_a * dn - an <= 0.0)
    sphere_a = torch.where(valid_a, sphere_a, torch.full_like(sphere_a, float("inf")))
    disc_b = db * db - (b_sq - radius_sq)
    sphere_b = db - torch.sqrt(disc_b.clamp(min=0.0))
    valid_b = (disc_b >= 0) & (sphere_b > 1e-6) & (sphere_b * dn - an >= length)
    sphere_b = torch.where(valid_b, sphere_b, torch.full_like(sphere_b, float("inf")))
    return torch.minimum(torch.minimum(cylinder, sphere_a), sphere_b)


def _render_capsules(
    starts: Any,
    ends: Any,
    colors: Any,
    height: int,
    width: int,
    focal: float,
    radius: Any,
    background: Any,
    device: Any,
    flat_shade: bool,
) -> Any:
    count = int(starts.shape[0])
    if count == 0:
        return (
            background.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            if background is not None
            else torch.zeros(height, width, 3, dtype=torch.float32, device=device)
        )
    radius = torch.as_tensor(radius, device=device, dtype=torch.float32)
    if radius.ndim == 0:
        radius = radius.expand(count)
    segment = ends - starts
    length = torch.linalg.norm(segment, dim=1).clamp(min=1e-6)
    axis = segment / length.unsqueeze(1)
    near = (torch.minimum(starts[:, 2].amin(), ends[:, 2].amin()) - radius.amax()).clamp(min=0.05)
    sz, ez = torch.maximum(starts[:, 2], near), torch.maximum(ends[:, 2], near)
    cx, cy = width * 0.5, height * 0.5
    sx, sy = starts[:, 0] * focal / sz + cx, starts[:, 1] * focal / sz + cy
    ex, ey = ends[:, 0] * focal / ez + cx, ends[:, 1] * focal / ez + cy
    projected_radius = radius * focal / torch.minimum(sz, ez)
    xmin = (torch.minimum(sx, ex) - projected_radius - 2.0).floor().long().clamp(min=0, max=width)
    xmax = (torch.maximum(sx, ex) + projected_radius + 2.0).ceil().long().clamp(min=0, max=width)
    ymin = (torch.minimum(sy, ey) - projected_radius - 2.0).floor().long().clamp(min=0, max=height)
    ymax = (torch.maximum(sy, ey) + projected_radius + 2.0).ceil().long().clamp(min=0, max=height)
    stride = width + 1
    diff = torch.zeros((height + 1) * stride, dtype=torch.int32, device=device)
    corners = torch.cat(
        [ymin * stride + xmin, ymax * stride + xmax, ymin * stride + xmax, ymax * stride + xmin]
    )
    signs = torch.cat(
        [
            torch.ones(count, dtype=torch.int32, device=device),
            torch.ones(count, dtype=torch.int32, device=device),
            -torch.ones(count, dtype=torch.int32, device=device),
            -torch.ones(count, dtype=torch.int32, device=device),
        ]
    )
    diff.index_add_(0, corners, signs)
    coarse = diff.view(height + 1, width + 1).cumsum(0).cumsum(1)[:height, :width] > 0
    indices = torch.nonzero(coarse.view(-1), as_tuple=False).squeeze(1)
    y = torch.div(indices, width, rounding_mode="floor").to(torch.float32)
    x = (indices % width).to(torch.float32)
    rays = torch.stack([(x - cx) / focal, (y - cy) / focal, torch.ones_like(x)], dim=-1)
    rays = rays / torch.linalg.norm(rays, dim=-1, keepdim=True)
    distances = torch.full((indices.shape[0],), float("inf"), device=device, dtype=torch.float32)
    objects = torch.full((indices.shape[0],), -1, device=device, dtype=torch.long)
    if indices.numel() > 0:
        chunk_max = max(1, int(4_000_000 / max(count, 1)))
        for start in range(0, indices.numel(), chunk_max):
            hits = _ray_capsule_t(
                rays[start : start + chunk_max], starts, ends, axis, length, radius
            )
            distance, obj = hits.min(dim=1)
            hit_indices = torch.nonzero(distance < float("inf"), as_tuple=False).squeeze(1)
            distances[start + hit_indices] = distance[hit_indices]
            objects[start + hit_indices] = obj[hit_indices]
    out = torch.zeros((height * width, 3), dtype=torch.float32, device=device)
    if background is not None:
        out = background.to(device=device, dtype=torch.float32).reshape(height * width, 3).clone()
    hit_indices = torch.nonzero(objects >= 0, as_tuple=False).squeeze(1)
    if hit_indices.numel() > 0:
        pixels, directions, hit_objects = (
            indices[hit_indices],
            rays[hit_indices],
            objects[hit_indices],
        )
        points = directions * distances[hit_indices].unsqueeze(-1)
        a, normal_axis, limb_length = starts[hit_objects], axis[hit_objects], length[hit_objects]
        projection = ((points - a) * normal_axis).sum(-1).clamp(min=0.0)
        projection = torch.minimum(projection, limb_length)
        closest = a + projection.unsqueeze(-1) * normal_axis
        normals = points - closest
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        color = colors[hit_objects, :3]
        if flat_shade:
            out[pixels] = color
            return out.view(height, width, 3).clamp(0.0, 1.0)
        diffuse = 0.45 + 0.55 * torch.clamp(-normals[:, 2], min=0.0)
        half = -directions.clone()
        half[:, 2] -= 1.0
        half = half / half.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        specular = torch.clamp((normals * half).sum(-1), min=0.0).pow(32)
        z = points[:, 2]
        fade = 1.0 - (z - z.amin()) / (z.amax() - z.amin()).clamp(min=1e-6)
        depth = 0.85 + 0.15 * fade
        out[pixels] = color * diffuse.unsqueeze(-1) * depth.unsqueeze(-1) + (
            0.5 * specular * depth
        ).unsqueeze(-1)
    return out.view(height, width, 3).clamp(0.0, 1.0)


def render_capsules(
    pose_data: Any,
    *,
    frame_idx: int,
    W: int,
    H: int,
    background: Any = None,
    composite: str = "over",
    radius_m: float = 0.025,
    include_hands: bool = False,
    palette: str = "scail",
    person_brightness_falloff: float = 0.0,
    flat_shade: bool = False,
    hand_radius_scale: float = 0.4,
) -> Any:
    persons: list[dict[str, Any]] = pose_data["frames"][frame_idx]
    device = management.get_torch_device()
    focal = float(min(H, W))
    for person in persons:
        value = person.get("focal_length")
        if value is not None:
            focal = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
            break
    starts, ends, colors, hands = _build_specs(
        persons, pose_data, include_hands, palette, person_brightness_falloff
    )
    bg = None
    if composite == "over" and background is not None:
        bg = background.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if bg.shape[:2] != (H, W):
            bg = (
                torch.nn.functional.interpolate(
                    bg.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .contiguous()
            )
    if starts.shape[0] == 0:
        return bg if bg is not None else torch.zeros(H, W, 3, dtype=torch.float32, device=device)
    radius = np.where(hands, radius_m * hand_radius_scale, radius_m).astype(np.float32)
    return _render_capsules(
        torch.from_numpy(starts).to(device=device, dtype=torch.float32),
        torch.from_numpy(ends).to(device=device, dtype=torch.float32),
        torch.from_numpy(colors).to(device=device, dtype=torch.float32),
        H,
        W,
        focal,
        torch.from_numpy(radius).to(device=device, dtype=torch.float32),
        bg,
        device,
        flat_shade,
    )
