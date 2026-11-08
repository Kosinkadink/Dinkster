"""Perspective-correct anisotropic splat rasterization.

Adapted from ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f,
comfy_extras/nodes_gaussian_splat.py. Imported only during execution.
"""

# pyright: basic
from __future__ import annotations

import importlib
import math
from typing import Any

from .camera_info import _camera_info
from .gaussian_splat import _quaternion_matrix, _quaternion_product


def quantile(value: Any, q: float) -> Any:
    torch = importlib.import_module("torch")
    limit = 1 << 24
    if value.numel() > limit:
        value = value[:: value.numel() // limit + 1]
    return torch.quantile(value, q)


def _srgb_to_linear(value: Any) -> Any:
    torch = importlib.import_module("torch")
    return torch.where(
        value <= 0.04045, value / 12.92, ((value.clamp_min(0) + 0.055) / 1.055) ** 2.4
    )


def _linear_to_srgb(value: Any) -> Any:
    torch = importlib.import_module("torch")
    return torch.where(
        value <= 0.0031308, value * 12.92, 1.055 * value.clamp_min(0) ** (1 / 2.4) - 0.055
    )


def _camera_basis(camera: Any, device: Any) -> Any:
    torch = importlib.import_module("torch")

    def vector(value: Any) -> Any:
        return torch.tensor(
            [float(value.get("x", 0)), -float(value.get("y", 0)), -float(value.get("z", 0))],
            device=device,
        )

    def mapped(value: Any) -> Any:
        return torch.stack([value[0], -value[1], -value[2]])

    def normalized(value: Any) -> Any:
        return value / value.norm().clamp_min(1e-8)

    eye, target = vector(camera.get("position", {})), vector(camera.get("target", {}))
    quaternion = camera.get("quaternion")
    if quaternion:
        q = torch.tensor(
            [float(quaternion.get(axis, 1 if axis == "w" else 0)) for axis in "wxyz"], device=device
        )
        matrix = _quaternion_matrix(torch, q[None])[0]
        return (
            eye,
            target,
            normalized(mapped(matrix[:, 0])),
            normalized(mapped(-matrix[:, 1])),
            normalized(mapped(-matrix[:, 2])),
        )
    forward = normalized(target - eye)
    yaw = math.atan2(-float(forward[0]), float(forward[2]))
    pitch = math.asin(max(-1.0, min(1.0, float(forward[1]))))
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device)
    rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], device=device)
    matrix = rx @ ry
    return eye, target, matrix[0], matrix[1], matrix[2]


def default_camera(center: Any, distance: float, device: Any) -> Any:
    torch = importlib.import_module("torch")
    yaw, pitch = math.radians(35), math.radians(30)
    forward = torch.tensor(
        [-math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)],
        device=device,
    )
    pos = center - distance * forward
    position = [float(pos[0]), -float(pos[1]), -float(pos[2])]
    target = [float(center[0]), -float(center[1]), -float(center[2])]
    return _camera_info(
        torch,
        {"mode": "look_at", **{f"position_{axis}": position[i] for i, axis in enumerate("xyz")}},
        target,
        0,
        35,
        1,
        "perspective",
        device,
    )


def orbit_camera(camera: Any, angle: float, device: Any) -> Any:
    torch = importlib.import_module("torch")
    angle = math.radians(angle)
    ca, sa = math.cos(angle), math.sin(angle)

    def vector(value: Any) -> Any:
        return torch.tensor([float(value.get(axis, 0)) for axis in "xyz"], device=device)

    pos, target = vector(camera.get("position", {})), vector(camera.get("target", {}))
    rotation = torch.tensor([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], device=device)
    pos = target + rotation @ (pos - target)
    q = camera.get("quaternion") or {}
    current = torch.tensor(
        [float(q.get(axis, 1 if axis == "w" else 0)) for axis in "wxyz"], device=device
    )
    turn = torch.tensor([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0], device=device)
    result = _quaternion_product(torch, turn[None], current[None])[0]
    return {
        **camera,
        "position": {axis: float(pos[i]) for i, axis in enumerate("xyz")},
        "quaternion": {axis: float(result[i]) for i, axis in enumerate("wxyz")},
    }


def _blur(value: Any, sigma: float, device: Any) -> Any:
    torch = importlib.import_module("torch")
    radius = max(1, int(round(3 * sigma)))
    kernel = torch.exp(
        -0.5 * (torch.arange(-radius, radius + 1, device=device, dtype=torch.float32) / sigma) ** 2
    )
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    value = torch.nn.functional.conv2d(
        value,
        kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
        padding=(0, radius),
        groups=channels,
    )
    return torch.nn.functional.conv2d(
        value,
        kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
        padding=(radius, 0),
        groups=channels,
    )


def render_gaussian(
    xyz: Any,
    rgb: Any,
    opacity: Any,
    scale: Any,
    rot: Any,
    width: int,
    height: int,
    splat_scale: float,
    bg: Any,
    camera_info: Any,
    sharpen: float,
    headlight_shading: float,
    render_style: str,
) -> Any:
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    dev = management.get_torch_device()

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=dev)

    idev, idtype = management.intermediate_device(), management.intermediate_dtype()
    xyz, rgb, opacity = tensor(xyz), tensor(rgb).clamp(0, 1), tensor(opacity).reshape(-1)
    scale, rot = tensor(scale) * float(splat_scale), tensor(rot)
    do_linear = render_style == "color"
    if do_linear:
        rgb = _srgb_to_linear(rgb)
    flat = width * height
    bg_t = tensor(bg)
    bg_comp = _srgb_to_linear(bg_t) if do_linear else bg_t
    need_depth = render_style == "depth"
    need_normal = render_style in ("normal", "clay") or headlight_shading > 0

    def background_only() -> Any:
        image = (
            bg_t.expand(height, width, 3)
            if render_style == "color"
            else torch.zeros(height, width, 3, device=dev)
        )
        return image.to(idev, idtype), torch.zeros(height, width, device=idev, dtype=idtype)

    if xyz.shape[0] == 0:
        return background_only()
    eye, target, right, up, forward = _camera_basis(camera_info, dev)
    matrix = torch.stack([right, up, forward], 0)
    cam = (xyz - eye) @ matrix.T
    fov = float(camera_info.get("fov", 0) or 0) or 35.0
    zoom = float(camera_info.get("zoom", 1) or 1)
    is_ortho = str(camera_info.get("cameraType", "")).lower().startswith("ortho")
    xc, yc, zc = cam.unbind(-1)
    keep = zc > 1e-2
    xc, yc, zc, rgb, opacity, scale, rot = (a[keep] for a in (xc, yc, zc, rgb, opacity, scale, rot))
    if xc.shape[0] == 0:
        return background_only()
    if render_style == "clay":
        rgb = torch.full_like(rgb, 0.75)
    focal = (min(width, height) / 2) / math.tan(math.radians(fov) / 2) * zoom
    cx0, cy0 = width / 2, height / 2
    mw = matrix[None] @ _quaternion_matrix(torch, rot)
    cam_cov = (mw * scale.square()[:, None, :]) @ mw.transpose(1, 2)
    cam_cov = cam_cov + (cam_cov.diagonal(dim1=-2, dim2=-1).mean(-1) * 1e-3)[
        :, None, None
    ] * torch.eye(3, device=dev)
    mu = torch.stack([xc, yc, zc], -1)
    inverse = torch.linalg.inv(cam_cov)
    simu = (inverse @ mu[:, :, None])[:, :, 0]
    musimu = (mu * simu).sum(-1)
    s00, s01, s02 = inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 0, 2]
    s11, s12, s22 = inverse[:, 1, 1], inverse[:, 1, 2], inverse[:, 2, 2]
    simu0, simu1, simu2 = simu.unbind(-1)
    if need_normal:
        nrm = mw[torch.arange(mw.shape[0], device=dev), :, scale.argmin(-1)]
        nrm = nrm * torch.where(nrm[:, 2:3] > 0, -1.0, 1.0)
    jacobian = torch.zeros(xc.shape[0], 2, 3, device=dev)
    if is_ortho:
        projection_scale = focal / float((target - eye).norm().clamp_min(1e-6))
        cx, cy = cx0 + projection_scale * xc, cy0 + projection_scale * yc
        jacobian[:, 0, 0] = projection_scale
        jacobian[:, 1, 1] = projection_scale
    else:
        invz = 1.0 / zc
        cx, cy = cx0 + focal * xc * invz, cy0 + focal * yc * invz
        jacobian[:, 0, 0], jacobian[:, 0, 2] = focal * invz, -focal * xc * invz.square()
        jacobian[:, 1, 1], jacobian[:, 1, 2] = focal * invz, -focal * yc * invz.square()
    cov2 = jacobian @ cam_cov @ jacobian.transpose(1, 2)
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    max_eig = (a + c) * 0.5 + (((a - c) * 0.5).square() + b * b).clamp_min(0).sqrt()
    radius = 3.0 * max_eig.clamp_min(1e-8).sqrt()
    cap = int(
        min(max(24, min(width, height) // 16), max(2, math.ceil(quantile(radius, 0.995).item())))
    )
    # Tight kernel buckets avoid paying the largest footprint for every splat.
    levels = [level for level in (16, 64, 256) if level < cap] + [cap]
    levels_t = torch.tensor(levels, device=dev, dtype=torch.float32)
    grids = []
    for level in levels:
        rng = torch.arange(-level, level + 1, device=dev, dtype=torch.float32)
        gy, gx = torch.meshgrid(rng, rng, indexing="ij")
        grids.append((gx.reshape(-1), gy.reshape(-1)))
    bucket = torch.bucketize(radius * (4.0 / 3.0), levels_t).clamp_(max=len(levels) - 1)
    count = zc.shape[0]
    slabs = int(min(256, max(1, count // 1000)))
    nl = len(levels)
    order = torch.argsort(zc)
    bounds = torch.linspace(0, count, slabs + 1, device=dev).round().long()
    rank = torch.empty(count, dtype=torch.long, device=dev)
    rank[order] = torch.arange(count, device=dev)
    slab_id = (torch.searchsorted(bounds, rank, right=True) - 1).clamp_(0, slabs - 1)
    key = slab_id * nl + bucket
    order = torch.argsort(key)
    key = key[order]
    cxr, cyr = cx[order].round(), cy[order].round()
    s00, s01, s02 = s00[order], s01[order], s02[order]
    s11, s12, s22 = s11[order], s12[order], s22[order]
    s01b, s02b, s12b = s01 * 2, s02 * 2, s12 * 2
    simu0, simu1, simu2, musimu = simu0[order], simu1[order], simu2[order], musimu[order]
    opacity, rgb = opacity[order], rgb[order]
    zc_o: Any = zc[order] if need_depth else None
    nrm_o: Any = nrm[order] if need_normal else None
    mux_o, muy_o, muz_o = (xc[order], yc[order], zc[order]) if is_ortho else (None, None, None)
    common = [cxr, cyr, s00, s11, s22, s01b, s02b, s12b, opacity]
    packed = torch.stack(
        common + ([s02, s12, mux_o, muy_o, muz_o] if is_ortho else [simu0, simu1, simu2, musimu])
    )
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=dev),
            (key[1:] != key[:-1]).nonzero().flatten() + 1,
        ]
    )
    keys = key[starts]
    run_lo = starts.tolist() + [count]
    run_levels = (keys % nl).tolist()
    run_slabs = torch.div(keys, nl, rounding_mode="floor").tolist()
    slab_runs: Any = [[] for _ in range(slabs)]
    for run in range(len(run_levels)):
        slab_runs[run_slabs[run]].append((run_lo[run], run_lo[run + 1], run_levels[run]))

    def splat(lo: int, hi: int, ox: Any, oy: Any) -> Any:
        cols = packed[:, lo:hi, None].unbind(0)
        cxr_, cyr_, a00, a11, a22, b01, b02, b12, opa = cols[:9]
        px, py = cxr_ + ox[None, :], cyr_ + oy[None, :]
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if is_ortho:
            c02, c12, mx, my, mz = cols[9:]
            rx, ry, rz = (px - cx0) / projection_scale - mx, (py - cy0) / projection_scale - my, -mz
            a22rz = a22 * rz
            inx = torch.addcmul(b02 * rz, a00, rx).addcmul_(b01, ry)
            rsr = torch.addcmul(a22rz * rz, rx, inx).addcmul_(ry, torch.addcmul(b12 * rz, a11, ry))
            dsr = torch.addcmul(a22rz, c02, rx).addcmul_(c12, ry)
            q = torch.addcdiv(rsr, dsr * dsr, a22.clamp_min(1e-12), value=-1).clamp_min_(0)
        else:
            su0, su1, su2, mus = cols[9:]
            dx, dy = (px - cx0) / focal, (py - cy0) / focal
            dsid = torch.addcmul(a22, dx, torch.addcmul(b02, a00, dx))
            dsid = dsid.addcmul_(dy, torch.addcmul(b12, a11, dy)).addcmul_(b01 * dx, dy)
            dsimu = torch.addcmul(su2, dx, su0).addcmul_(dy, su1)
            q = torch.addcdiv(mus, dsimu * dsimu, dsid.clamp_min(1e-12), value=-1).clamp_min_(0)
        alpha = (opa * torch.exp(-0.5 * q) * valid).clamp_(0, 0.999)
        idx = py.long().clamp(0, height - 1) * width + px.long().clamp(0, width - 1)
        return idx, alpha

    sharp = sharpen != 1.0
    cacc, trans = torch.zeros((flat, 3), device=dev), torch.ones((flat,), device=dev)
    a_buf, tau_buf = torch.zeros((flat,), device=dev), torch.zeros((flat,), device=dev)
    crgb = torch.zeros((flat, 3), device=dev)
    wbuf: Any = torch.zeros((flat,), device=dev) if sharp else None
    dacc: Any = torch.zeros((flat,), device=dev) if need_depth else None
    nacc: Any = torch.zeros((flat, 3), device=dev) if need_normal else None
    zslab: Any = torch.zeros((flat,), device=dev) if need_depth else None
    nslab: Any = torch.zeros((flat, 3), device=dev) if need_normal else None
    stale = 0
    for slab in range(slabs):
        runs = slab_runs[slab]
        if not runs:
            continue
        a_buf.zero_()
        tau_buf.zero_()
        crgb.zero_()
        if sharp:
            wbuf.zero_()
        if need_depth:
            zslab.zero_()
        if need_normal:
            nslab.zero_()
        for run_start, run_end, level in runs:
            ox, oy = grids[level]
            chunk = max(2048, 10_000_000 // ox.shape[0])
            for lo in range(run_start, run_end, chunk):
                hi = min(lo + chunk, run_end)
                idx, alpha = splat(lo, hi, ox, oy)
                idx, af = idx.reshape(-1), alpha.reshape(-1)
                a_buf.index_add_(0, idx, af)
                tau_buf.index_add_(0, idx, (-torch.log1p(-alpha)).reshape(-1))
                weight = alpha.pow(sharpen) if sharp else alpha
                crgb.index_add_(0, idx, (weight[:, :, None] * rgb[lo:hi, None, :]).reshape(-1, 3))
                if sharp:
                    wbuf.index_add_(0, idx, weight.reshape(-1))
                if need_depth:
                    zslab.index_add_(0, idx, (alpha * zc_o[lo:hi, None]).reshape(-1))
                if need_normal:
                    nslab.index_add_(
                        0, idx, (alpha[:, :, None] * nrm_o[lo:hi, None, :]).reshape(-1, 3)
                    )
        slab_alpha = 1 - torch.exp(-tau_buf)
        front = trans * slab_alpha
        denominator = wbuf if sharp else a_buf
        cacc.addcmul_(front[:, None], crgb / denominator.clamp_min(1e-8)[:, None])
        if need_depth or need_normal:
            inverse_alpha = a_buf.clamp_min(1e-8)
            if need_depth:
                dacc.addcmul_(front, zslab / inverse_alpha)
            if need_normal:
                nacc.addcmul_(front[:, None], nslab / inverse_alpha[:, None])
        trans.mul_(1 - slab_alpha)
        # Two occluded checkpoints terminate without a GPU synchronization per slab.
        if slab % 8 == 7:
            if float(front.max()) < 1e-3:
                stale += 1
                if stale >= 2:
                    break
            else:
                stale = 0
    coverage = 1 - trans
    coverage_grid = coverage.reshape(height, width)
    silhouette: Any = coverage_grid > 0.5 if render_style in ("depth", "normal") else None
    depth: Any = (dacc / coverage.clamp_min(1e-6)).reshape(height, width) if need_depth else None
    normal_map: Any = None
    if need_normal:
        nb = nacc.reshape(height, width, 3).permute(2, 0, 1)[None]
        cb = coverage.reshape(1, 1, height, width)
        nb, cb = _blur(nb, 1.2, dev), _blur(cb, 1.2, dev)
        normal = (nb / cb.clamp_min(1e-6))[0].permute(1, 2, 0)
        normal_map = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if render_style == "depth":
        value = torch.zeros(height, width, device=dev)
        if bool(silhouette.any()):
            lo, hi = depth[silhouette].min(), depth[silhouette].max()
            value = torch.where(
                silhouette, ((hi - depth) / (hi - lo).clamp_min(1e-6)).clamp(0, 1), value
            )
        image = value[:, :, None].expand(height, width, 3)
    elif render_style == "normal":
        image = (normal_map * tensor([1.0, -1.0, -1.0]) * 0.5 + 0.5).clamp(0, 1) * silhouette[
            :, :, None
        ]
    else:
        image = cacc.reshape(height, width, 3)
        if render_style == "clay":
            light = tensor([-0.4, -0.7, -0.6])
            light = light / light.norm()
            lambert = (0.5 * (normal_map * light).sum(-1) + 0.5).clamp(0, 1)
            image = image * (0.35 + 0.65 * lambert * lambert)[:, :, None]
        elif headlight_shading > 0:
            strength = float(headlight_shading)
            lambert = (-normal_map[:, :, 2]).clamp(0, 1)
            image = image * (1 - 0.6 * strength + 0.6 * strength * lambert)[:, :, None]
        image = image.addcmul_(trans.reshape(height, width, 1), bg_comp)
        if do_linear:
            image = _linear_to_srgb(image)
    return image.clamp(0, 1).to(idev, idtype), coverage_grid.clamp(0, 1).to(idev, idtype)
