"""SAM3D mesh shading with a bounded, hard z-buffer triangle rasterizer."""

from __future__ import annotations

import importlib
from typing import Any

np = importlib.import_module("numpy")

_CANONICAL_PRESETS = {"rainbow", "rainbow_face_normal", "rainbow_face_semantic"}


def rainbow_colors_from_canonical(
    positions: Any, tilt_x_deg: float = 0.0, tilt_z_deg: float = 0.0
) -> Any:
    theta_x, theta_z = np.deg2rad(tilt_x_deg), np.deg2rad(tilt_z_deg)
    axis = np.array(
        [np.sin(theta_z), np.cos(theta_z) * np.cos(theta_x), np.cos(theta_z) * np.sin(theta_x)],
        dtype=np.float32,
    )
    s = positions @ axis
    s = (s - s.min()) / max(float(s.max() - s.min()), 1e-8)
    s = np.clip(s * 0.98, 0.0, 1.0).astype(np.float32)
    r = np.interp(s, [0.0, 0.35, 0.66, 0.89, 1.0], [0.0, 0.0, 1.0, 1.0, 0.5])
    g = np.interp(s, [0.0, 0.125, 0.375, 0.64, 0.91, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    b = np.interp(s, [0.0, 0.11, 0.34, 0.65, 1.0], [0.5, 1.0, 1.0, 0.0, 0.0])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _vertex_normals(verts: Any, faces: Any) -> Any:
    torch = importlib.import_module("torch")
    v0, v1, v2 = (verts[faces[:, i]] for i in range(3))
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vn = torch.zeros_like(verts)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn)
    return vn / vn.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def build_vertex_colors(canonical: Any, preset: str, tilt_x: float, tilt_z: float) -> Any:
    positions = np.asarray(canonical.get("positions"), dtype=np.float32)
    color = rainbow_colors_from_canonical(positions, tilt_x, tilt_z).copy()
    if preset in ("rainbow_face_normal", "rainbow_face_semantic"):
        mask = canonical.get("face_mask")
        if mask is not None and mask.any():
            if preset == "rainbow_face_normal":
                color[mask] = np.asarray(canonical["norm"], dtype=np.float32)[mask]
            else:
                semantic = np.asarray(canonical["face_region_rgb"], dtype=np.float32)
                assigned = semantic.sum(axis=1) > 0
                color[assigned] = semantic[assigned]
    return color


def _rasterize_chunk(
    fv_pix: Any,
    fv_z: Any,
    min_x: Any,
    max_x: Any,
    min_y: Any,
    max_y: Any,
    extent: int,
    width: int,
) -> tuple[Any, Any, Any, Any]:
    torch = importlib.import_module("torch")
    device = fv_pix.device
    sx, sy = max_x - min_x, max_y - min_y
    x_off = torch.arange(extent, device=device)
    y_off = torch.arange(extent, device=device)
    px = (min_x[:, None, None] + x_off[None, None, :]).float() + 0.5
    py = (min_y[:, None, None] + y_off[None, :, None]).float() + 0.5
    in_bb = (x_off[None, None, :] < sx[:, None, None]) & (y_off[None, :, None] < sy[:, None, None])
    ax, ay = fv_pix[:, 0, 0][:, None, None], fv_pix[:, 0, 1][:, None, None]
    bx, by = fv_pix[:, 1, 0][:, None, None], fv_pix[:, 1, 1][:, None, None]
    cx, cy = fv_pix[:, 2, 0][:, None, None], fv_pix[:, 2, 1][:, None, None]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    ea = (bx - px) * (cy - py) - (by - py) * (cx - px)
    eb = (cx - px) * (ay - py) - (cy - py) * (ax - px)
    ec = (ax - px) * (by - py) - (ay - py) * (bx - px)
    nondegen = area.abs() > 1e-6
    inside = (ea * area >= 0) & (eb * area >= 0) & (ec * area >= 0) & in_bb & nondegen
    inv = torch.where(nondegen, 1.0 / area, torch.zeros_like(area))
    wa, wb, wc = ea * inv, eb * inv, ec * inv
    z = wa * fv_z[:, 0, None, None] + wb * fv_z[:, 1, None, None] + wc * fv_z[:, 2, None, None]
    fi, yi, xi = inside.nonzero(as_tuple=True)
    pixels = ((min_y[fi] + yi) * width + min_x[fi] + xi).long()
    bary = torch.stack([wa[fi, yi, xi], wb[fi, yi, xi], wc[fi, yi, xi]], dim=-1)
    return pixels, z[fi, yi, xi], fi.long(), bary


def _rasterize_person(
    vertices: Any,
    faces: Any,
    focal: float,
    width: int,
    height: int,
    z_buf: Any,
    color_buf: Any,
    mask_buf: Any,
    shade: Any,
) -> None:
    torch = importlib.import_module("torch")
    valid = vertices[:, 2] > 0.05
    z = vertices[:, 2].clamp(min=0.05)
    px = 0.5 * width + focal * vertices[:, 0] / z
    py = 0.5 * height + focal * vertices[:, 1] / z
    face_pixels = torch.stack([px, py], dim=-1)[faces]
    face_z = vertices[faces][..., 2]
    x, y = face_pixels[..., 0], face_pixels[..., 1]
    min_x = x.amin(dim=-1).floor().long().clamp(min=0, max=width)
    max_x = (x.amax(dim=-1).ceil().long() + 1).clamp(min=0, max=width)
    min_y = y.amin(dim=-1).floor().long().clamp(min=0, max=height)
    max_y = (y.amax(dim=-1).ceil().long() + 1).clamp(min=0, max=height)
    sx, sy = max_x - min_x, max_y - min_y
    keep = torch.where(valid[faces].all(dim=-1) & (sx > 0) & (sy > 0))[0]
    if keep.numel() == 0:
        return
    bbox_size = torch.maximum(sx, sy)[keep]
    lower, limit = 0, 1
    max_extent = max(width, height)
    while lower < max_extent:
        selected = keep[(bbox_size > lower) & (bbox_size <= limit)]
        chunk_size = min(8192, max(1, 4_000_000 // (limit * limit)))
        for start in range(0, selected.shape[0], chunk_size):
            chunk = selected[start : start + chunk_size]
            pixels, depth, face_local, bary = _rasterize_chunk(
                face_pixels[chunk],
                face_z[chunk],
                min_x[chunk],
                max_x[chunk],
                min_y[chunk],
                max_y[chunk],
                limit,
                width,
            )
            face_global = chunk[face_local]
            old = z_buf[pixels].clone()
            z_buf.scatter_reduce_(0, pixels, depth, reduce="amin", include_self=True)
            new = z_buf[pixels]
            is_min = (depth == new) & (new < old)
            surviving_pixels, surviving_faces, surviving_bary = (
                pixels[is_min],
                face_global[is_min],
                bary[is_min],
            )
            # The first fragment wins exact depth ties, including across chunks.
            order = torch.argsort(surviving_pixels, stable=True)
            sorted_pixels = surviving_pixels[order]
            first = torch.ones_like(sorted_pixels, dtype=torch.bool)
            first[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
            chosen = order[first]
            written = surviving_pixels[chosen]
            color_buf[written] = shade(surviving_faces[chosen], surviving_bary[chosen])
            mask_buf[written] = True
        lower, limit = limit, min(limit * 2, max_extent)


def _make_shade_fn(
    preset: str,
    composite: str,
    normals: Any,
    positions: Any,
    colors: Any,
    faces: Any,
    base_color: Any,
    light_dir: Any,
    pastel_mix: float,
) -> Any:
    torch = importlib.import_module("torch")
    device = normals.device
    base = torch.as_tensor(base_color, dtype=torch.float32, device=device)
    light = -torch.as_tensor(light_dir, dtype=torch.float32, device=device)
    light = light / light.norm().clamp(min=1e-8)

    def pastel(rgb: Any) -> Any:
        return rgb if pastel_mix <= 0 else rgb * (1.0 - pastel_mix) + pastel_mix

    def gather_n(fi: Any, bary: Any) -> Any:
        n = (normals[faces[fi]] * bary.unsqueeze(-1)).sum(dim=1)
        return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def gather_color(fi: Any, bary: Any) -> Any:
        return (colors[faces[fi]] * bary.unsqueeze(-1)).sum(dim=1)

    if composite == "silhouette":

        def shade_silhouette(fi: Any, bary: Any) -> Any:
            return torch.ones((fi.shape[0], 3), device=device)

        return shade_silhouette
    if preset == "normals":

        def shade_normals(fi: Any, bary: Any) -> Any:
            return pastel(((gather_n(fi, bary) + 1.0) * 0.5).clamp(0.0, 1.0))

        return shade_normals
    if preset not in _CANONICAL_PRESETS or colors is None:

        def shade(fi: Any, bary: Any) -> Any:
            n = gather_n(fi, bary)
            v = -(positions[faces[fi]] * bary.unsqueeze(-1)).sum(dim=1)
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            diffuse = (n * light).sum(dim=-1).clamp(min=0)
            rim = (1.0 - (n * v).sum(dim=-1).clamp(min=0)).pow(3.0)
            return pastel(
                0.25 * base + 0.75 * base * diffuse.unsqueeze(-1) + 0.35 * rim.unsqueeze(-1)
            )

        return shade
    if preset == "rainbow":

        def shade_rainbow(fi: Any, bary: Any) -> Any:
            diffuse = (gather_n(fi, bary) * light).sum(dim=-1).clamp(min=0)
            return pastel(gather_color(fi, bary) * (0.65 + 0.35 * diffuse).unsqueeze(-1))

        return shade_rainbow
    fill = torch.stack([-light[0], light[1].abs(), -light[2]])
    half = light + torch.tensor([0.0, 0.0, 1.0], device=device)
    half = half / half.norm().clamp(min=1e-8)

    def shade_face(fi: Any, bary: Any) -> Any:
        n = gather_n(fi, bary)
        key_diffuse = (n * light).sum(dim=-1).clamp(min=0)
        fill_diffuse = (n * fill).sum(dim=-1).clamp(min=0)
        rim = (1.0 - n[..., 2].clamp(min=0)).pow(2.5) * 0.30
        value = (0.45 + 0.45 * key_diffuse + 0.15 * fill_diffuse + rim * 0.5).clamp(
            min=0.0, max=1.25
        )
        spec = (n * half).sum(dim=-1).clamp(min=0).pow(48) * 0.12
        return pastel(gather_color(fi, bary) * value.unsqueeze(-1) + spec.unsqueeze(-1))

    return shade_face


def render_mesh(
    pose_data: Any,
    *,
    frame_idx: int,
    W: int,
    H: int,
    background: Any = None,
    composite: str = "over",
    opacity: float = 1.0,
    shader_preset: str = "default",
    base_color: Any = (0.68, 0.71, 0.78),
    light_dir: Any = (0.4, -0.7, -0.6),
    rainbow_tilt_x_deg: float = 0.0,
    rainbow_tilt_z_deg: float = 0.0,
    person_brightness_falloff: float = 0.6,
    cache: dict[Any, Any] | None = None,
) -> Any:
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    device = management.get_torch_device()
    cache = {} if cache is None else cache
    persons: list[dict[str, Any]] = (
        pose_data["frames"][frame_idx] if frame_idx < len(pose_data["frames"]) else []
    )
    if not persons:
        if composite == "over" and background is not None:
            return torch.as_tensor(background, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        return torch.zeros((H, W, 3), device=device, dtype=torch.float32)
    faces_source = pose_data["faces"]
    faces_key = ("faces", id(faces_source), device)
    if faces_key not in cache:
        # Keep the source alive so its identity cannot be recycled within a render.
        cache[faces_key] = (
            faces_source,
            torch.as_tensor(np.asarray(faces_source, dtype=np.int64), device=device),
        )
    faces = cache[faces_key][1]
    canonical = pose_data.get("canonical_colors")
    using_canonical = shader_preset in _CANONICAL_PRESETS
    if using_canonical and canonical is None:
        shader_preset, using_canonical = "default", False
    colors = None
    if using_canonical:
        key = (
            "vcolor",
            id(canonical),
            shader_preset,
            float(rainbow_tilt_x_deg),
            float(rainbow_tilt_z_deg),
            device,
        )
        colors = cache.get(key)
        if colors is None:
            colors = torch.as_tensor(
                build_vertex_colors(
                    canonical, shader_preset, rainbow_tilt_x_deg, rainbow_tilt_z_deg
                ),
                dtype=torch.float32,
                device=device,
            )
            cache[key] = colors
    falloff = max(0.0, min(1.0, float(person_brightness_falloff)))
    pastels = [0.0 if k == 0 else (1.0 - falloff**k) for k in range(len(persons))]
    order = sorted(
        range(len(persons)),
        key=lambda i: -float(np.asarray(persons[i]["pred_cam_t"]).reshape(-1)[2]),
    )
    z_buf = torch.full((H * W,), float("inf"), device=device, dtype=torch.float32)
    color_buf = torch.zeros((H * W, 3), device=device, dtype=torch.float32)
    mask_buf = torch.zeros(H * W, device=device, dtype=torch.bool)
    for idx in order:
        person = persons[idx]
        vertices = np.asarray(person["pred_vertices"], dtype=np.float32).reshape(-1, 3)
        translation = np.asarray(person["pred_cam_t"], dtype=np.float32).reshape(3)
        world = torch.as_tensor(vertices + translation[None, :], device=device, dtype=torch.float32)
        focal = float(np.asarray(person.get("focal_length", 5000.0)).reshape(-1)[0])
        view = torch.stack([world[:, 0], -world[:, 1], -world[:, 2]], dim=-1)
        normals = _vertex_normals(world, faces)
        normals = torch.stack([normals[:, 0], -normals[:, 1], -normals[:, 2]], dim=-1)
        person_colors = colors if colors is not None and colors.shape[0] == world.shape[0] else None
        preset = (
            "default"
            if shader_preset in _CANONICAL_PRESETS and person_colors is None
            else shader_preset
        )
        shade = _make_shade_fn(
            preset,
            composite,
            normals,
            view,
            person_colors,
            faces,
            base_color,
            light_dir,
            pastels[idx],
        )
        _rasterize_person(world, faces, focal, W, H, z_buf, color_buf, mask_buf, shade)
    if shader_preset == "depth":
        mask, depth = mask_buf.reshape(H, W), z_buf.reshape(H, W)
        near = torch.where(mask, depth, torch.full_like(depth, float("inf"))).amin()
        far = torch.where(mask, depth, torch.full_like(depth, float("-inf"))).amax()
        normalized = torch.where(
            mask, 1.0 - (depth - near) / (far - near).clamp(min=1e-6), torch.zeros_like(depth)
        )
        rendered, mask_f = torch.stack([normalized] * 3, dim=-1), mask.float()
    else:
        rendered = color_buf.reshape(H, W, 3).clamp(0.0, 1.0)
        mask_f = mask_buf.reshape(H, W).float()
    if composite == "over" and background is not None:
        bg = torch.as_tensor(background, dtype=torch.float32, device=device)
        alpha = mask_f.unsqueeze(-1)
        if opacity != 1.0:
            alpha = alpha * float(opacity)
        return torch.lerp(bg, rendered, alpha)
    return rendered * float(opacity) if opacity != 1.0 else rendered
