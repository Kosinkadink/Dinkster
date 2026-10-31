"""
Pure-PyTorch GPU QEM mesh simplification.

Parallel rounds of independent edge collapses, gated by normal-flip, link-condition and
skinny-triangle checks, with mesh cleanup before and after.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

from dinkster_api.v1 import report_progress

from .mesh_holes import connected_labels as _connected_labels
from .mesh_weld import weld_vertices


@dataclass
class QEMConfig:
    stabilizer_scale: float = 1e-3  # Tikhonov term = mesh_scale squared * this
    wander_threshold: float = 2.0  # midpoint fallback beyond this times the edge length
    clamp_v_to_edge: bool = True  # project v* onto the edge segment (qem mode only)

    # Midpoint uses a cost threshold; QEM uses optimal placement and a ratio driver.
    placement_mode: str = "midpoint"

    sampling_cap: int = 10_000_000  # about this many edges evaluated per round
    max_collapses_fraction: float = 0.25  # of remaining faces-to-remove
    max_collapses_floor: int = 10_000
    max_collapses_ceiling: int = 1_000_000
    max_collapses_relative_cap: float = 0.10  # of current faces

    max_iterations: int = 5_000

    boundary_weight: float = 1000.0  # line quadrics pinning boundary edges
    line_quadric_weight: float = 0.0  # more uniform verts; 0 disables
    line_quadric_skip_opposite_normals_cos: float = (
        0.0  # skip edges with endpoint normal cos below this
    )

    # line quadrics on interior edges sharper than the min dihedral; 0 disables
    feature_edge_quadric_weight: float = 0.0
    feature_edge_min_dihedral_deg: float = 30.0

    # Reject collapses that flip a 1-ring normal.
    flip_cos_threshold: float = 0.0  # 0 = any sign reversal (dihedral > 90 degrees)
    flip_check_max_degree: int = 16  # torch fallback only; the kitchen op is exact

    skinny_weight: float = 1e-3  # needle/sliver penalty

    enforce_link_condition: bool = True  # keep topology, until that alone blocks the target

    area_weighted_quadrics: bool = False  # Garland-Heckbert area weighting

    lambda_edge_length: float = 1e-2  # cost += weight * length squared, favors short edges

    threshold_start: float = 1e-8  # midpoint scale squared; multiply by 10 below 1% removal
    repair_nonmanifold: bool = True  # split sheets fused at non-manifold edges after decimation

    postclean: bool = True  # remove slivers, tiny components, unused verts left by collapse
    postclean_min_angle_deg: float = 0.5
    postclean_max_aspect_ratio: float = 100.0
    postclean_min_component_faces: int = 8

    preclean_weld_epsilon_rel: float = 1e-5  # fraction of bbox diagonal

    @property
    def threshold_driver(self) -> bool:
        """True when the cost-threshold driver is used (midpoint placement)."""
        return self.placement_mode == "midpoint"


def _sorted_edge_halfedges(
    faces: Any,
    num_verts: int,
) -> tuple[Any, Any, Any]:
    """Return half-edge keys min(a,b)*(V+1)+max(a,b), face ids and slot ids in key order."""
    torch = importlib.import_module("torch")
    device = faces.device
    F = faces.shape[0]
    e_all = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e_sorted, _ = torch.sort(e_all, dim=1)
    P = num_verts + 1
    key = e_sorted[:, 0].long() * P + e_sorted[:, 1].long()
    face_per_he = torch.arange(F, device=device, dtype=torch.long).repeat(3)
    slot_per_he = torch.arange(3, device=device, dtype=torch.long).repeat_interleave(F)
    sort_idx = torch.argsort(key)
    return key[sort_idx], face_per_he[sort_idx], slot_per_he[sort_idx]


def _manifold_edge_pairs(
    sorted_keys: Any,
    sorted_faces: Any,
) -> tuple[Any, Any, Any]:
    """Edges shared by exactly 2 faces; returns (pair_keys, fa, fb)."""
    torch = importlib.import_module("torch")
    if sorted_keys.shape[0] < 2:
        empty = sorted_keys.new_empty(0)
        return empty, empty, empty
    pair_mask = sorted_keys[:-1] == sorted_keys[1:]
    if not pair_mask.any():
        empty = sorted_keys.new_empty(0)
        return empty, empty, empty
    pair_starts = torch.nonzero(pair_mask, as_tuple=True)[0]
    # manifold iff neither neighbour half-edge shares the key
    cur = sorted_keys[pair_starts]
    prev_ok = (pair_starts == 0) | (sorted_keys[(pair_starts - 1).clamp_min(0)] != cur)
    nxt_idx = (pair_starts + 2).clamp(max=sorted_keys.shape[0] - 1)
    nxt_ok = (pair_starts + 2 >= sorted_keys.shape[0]) | (sorted_keys[nxt_idx] != cur)
    pair_starts = pair_starts[prev_ok & nxt_ok]
    return (sorted_keys[pair_starts], sorted_faces[pair_starts], sorted_faces[pair_starts + 1])


_FACE_CHUNK = 1 << 22  # faces per pass of the quadric build
_EDGE_CHUNK = 1 << 21  # edges per pass of the collapse errors


def _line_quadric_planes(pa: Any, pb: Any) -> tuple[Any, Any, Any]:
    """Return two planes per edge whose squared distances sum to distance from its line."""
    torch = importlib.import_module("torch")
    e = pb - pa
    elen = torch.norm(e, dim=-1, keepdim=True).clamp_min(1e-12)
    e_unit = e / elen
    m = 0.5 * (pa + pb)
    # least-aligned axis, orthogonalised against e_unit
    helper = torch.zeros_like(e_unit)
    helper.scatter_(-1, e_unit.abs().argmin(dim=-1, keepdim=True), 1.0)
    u = helper - (helper * e_unit).sum(-1, keepdim=True) * e_unit
    u = u / torch.norm(u, dim=-1, keepdim=True).clamp_min(1e-12)
    w = torch.cross(e_unit, u, dim=-1)
    d_u = -(u * m).sum(-1, keepdim=True)
    d_w = -(w * m).sum(-1, keepdim=True)
    p_u = torch.cat([u, d_u], dim=-1)
    p_w = torch.cat([w, d_w], dim=-1)
    return p_u, p_w, elen.squeeze(-1)


def _add_line_quadrics(
    verts: Any,
    faces: Any,
    face_areas: Any,
    Q_flat: Any,
    weight: float,
    skip_he_mask: Any = None,
) -> Any:
    """Accumulate area-weighted half-edge line quadrics, excluding masked edges."""
    torch = importlib.import_module("torch")
    a_all = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]], dim=0).long()
    b_all = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]], dim=0).long()
    pa = verts[a_all]
    pb = verts[b_all]
    p_u, p_w, _ = _line_quadric_planes(pa, pb)
    area_per_edge = face_areas.repeat(3)
    w_per_edge = area_per_edge * weight
    if skip_he_mask is not None:
        w_per_edge = torch.where(skip_he_mask, torch.zeros_like(w_per_edge), w_per_edge)
    w_per_edge = w_per_edge.unsqueeze(-1).unsqueeze(-1)
    K_line = (
        p_u.unsqueeze(-1) * p_u.unsqueeze(-2) + p_w.unsqueeze(-1) * p_w.unsqueeze(-2)
    ) * w_per_edge
    K_flat = K_line.reshape(-1, 16)
    Q_flat.scatter_add_(0, a_all.unsqueeze(1).expand(-1, 16), K_flat)
    Q_flat.scatter_add_(0, b_all.unsqueeze(1).expand(-1, 16), K_flat)
    return Q_flat


def _build_quadrics(
    verts: Any,
    faces: Any,
    cfg: QEMConfig,
    b_edges: Any,
) -> Any:
    """Per-vertex face-plane quadrics with configured line, boundary and feature terms."""
    torch = importlib.import_module("torch")
    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    Q_flat = torch.zeros((V, 16), dtype=dtype, device=device)
    area: Any = None
    n_norm: Any = None

    if faces.numel() > 0:
        F = faces.shape[0]
        area = torch.empty(F, dtype=dtype, device=device)
        n_norm = (
            torch.empty((F, 3), dtype=dtype, device=device) if cfg.line_quadric_weight > 0 else None
        )
        # a chunk at a time: the per-face quadrics are the decimation's largest transient
        for s in range(0, F, _FACE_CHUNK):
            f = faces[s : s + _FACE_CHUNK]
            v0 = verts[f[:, 0]]
            n = torch.cross(verts[f[:, 1]] - v0, verts[f[:, 2]] - v0, dim=-1)
            a = torch.norm(n, dim=-1)
            # degenerate faces get a zero plane
            nn = torch.where(
                (a > 1e-12).unsqueeze(-1), n / a.unsqueeze(-1).clamp_min(1e-12), n.new_zeros(())
            )
            p = torch.cat([nn, -(nn * v0).sum(dim=-1, keepdim=True)], dim=-1)  # (n, 4)
            K = torch.einsum("fi,fj->fij", p, p)
            if cfg.area_weighted_quadrics:
                K.mul_(a[:, None, None])
            K_flat = K.reshape(-1, 16)
            for corner in range(3):
                Q_flat.scatter_add_(0, f[:, corner].unsqueeze(1).expand(-1, 16), K_flat)
            area[s : s + f.shape[0]] = a
            if n_norm is not None:
                n_norm[s : s + f.shape[0]] = nn

    # line quadrics on all half-edges, skipping thin-shell rim edges whose endpoint normals oppose
    if cfg.line_quadric_weight > 0 and faces.numel() > 0:
        v_norm = torch.zeros((V, 3), dtype=dtype, device=device)
        n_weighted = n_norm * area.unsqueeze(-1)
        for corner in range(3):
            v_norm.scatter_add_(0, faces[:, corner].unsqueeze(-1).expand(-1, 3), n_weighted)
        v_norm = torch.nn.functional.normalize(v_norm, p=2, dim=-1, eps=1e-12)
        a_he = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]], dim=0).long()
        b_he = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]], dim=0).long()
        cos_endpoints = (v_norm[a_he] * v_norm[b_he]).sum(dim=-1)
        skip_he_sharp = cos_endpoints < cfg.line_quadric_skip_opposite_normals_cos
        Q_flat = _add_line_quadrics(
            verts, faces, area, Q_flat, cfg.line_quadric_weight, skip_he_mask=skip_he_sharp
        )

    # pin boundary verts to their boundary edge lines
    if b_edges.shape[0] > 0:
        ba = b_edges[:, 0]
        bb = b_edges[:, 1]
        pa = verts[ba]
        pb = verts[bb]
        p_u, p_w, _ = _line_quadric_planes(pa, pb)
        K_b = (
            torch.einsum("ei,ej->eij", p_u, p_u) + torch.einsum("ei,ej->eij", p_w, p_w)
        ) * cfg.boundary_weight
        K_b_flat = K_b.reshape(-1, 16)
        Q_flat.scatter_add_(0, ba.unsqueeze(1).expand(-1, 16), K_b_flat)
        Q_flat.scatter_add_(0, bb.unsqueeze(1).expand(-1, 16), K_b_flat)

    # feature-edge line quadrics, weighted by 1 - cos(dihedral)
    if cfg.feature_edge_quadric_weight > 0 and faces.numel() > 0:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
        fn = torch.nn.functional.normalize(fn, p=2, dim=-1, eps=1e-12)
        sorted_keys_fe, sorted_faces_fe, _ = _sorted_edge_halfedges(faces, V)
        pair_keys, f1_idx, f2_idx = _manifold_edge_pairs(sorted_keys_fe, sorted_faces_fe)
        if pair_keys.numel() > 0:
            P = V + 1
            edge_a = pair_keys // P
            edge_b = pair_keys % P
            cos_dihedral = (fn[f1_idx] * fn[f2_idx]).sum(dim=-1)
            cos_thresh = math.cos(math.radians(cfg.feature_edge_min_dihedral_deg))
            sharp = cos_dihedral < cos_thresh
            if sharp.any():
                fa = edge_a[sharp]
                fb = edge_b[sharp]
                p_u, p_w, _ = _line_quadric_planes(verts[fa], verts[fb])
                sharpness = (1.0 - cos_dihedral[sharp]).clamp_min(0.0)
                avg_area = 0.5 * (area[f1_idx[sharp]] + area[f2_idx[sharp]])
                w = (
                    (avg_area * sharpness * cfg.feature_edge_quadric_weight)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                K_feat = (
                    p_u.unsqueeze(-1) * p_u.unsqueeze(-2) + p_w.unsqueeze(-1) * p_w.unsqueeze(-2)
                ) * w
                K_flat = K_feat.reshape(-1, 16)
                Q_flat.scatter_add_(0, fa.unsqueeze(1).expand(-1, 16), K_flat)
                Q_flat.scatter_add_(0, fb.unsqueeze(1).expand(-1, 16), K_flat)

    return Q_flat.reshape(V, 4, 4)


def _edge_errors(
    verts: Any,
    Q: Any,
    edges: Any,
    stabilizer: Any,
    max_err: Any,
    mesh_scale_sq: Any,
    cfg: QEMConfig,
    vert_is_boundary: Any = None,
) -> tuple[Any, Any, Any]:
    """Per-edge positions, costs and validity, snapping midpoints to boundary endpoints."""
    torch = importlib.import_module("torch")
    n_edges = edges.shape[0]
    dtype = verts.dtype
    device = verts.device

    verts_pair = verts[edges]  # (E, 2, 3)
    pa = verts_pair[:, 0]
    pb = verts_pair[:, 1]
    edge_vec = pb - pa
    el = torch.norm(edge_vec, dim=-1)

    # snap to the boundary endpoint when exactly one endpoint is on the boundary
    if vert_is_boundary is not None:
        ba = vert_is_boundary[edges[:, 0]]
        bb = vert_is_boundary[edges[:, 1]]
        w_a = torch.where(
            ba & ~bb,
            torch.ones_like(el),
            torch.where(~ba & bb, torch.zeros_like(el), torch.full_like(el, 0.5)),
        )
        midpoint = pa * w_a.unsqueeze(-1) + pb * (1.0 - w_a).unsqueeze(-1)
    else:
        midpoint = torch.lerp(pa, pb, 0.5)

    Qe = Q[edges].sum(dim=1)  # (E, 4, 4)

    if cfg.placement_mode == "midpoint":
        opt = midpoint
    else:
        A = Qe[:, :3, :3] + torch.eye(3, device=device, dtype=dtype) * stabilizer
        b = -Qe[:, :3, 3].unsqueeze(-1)

        # stabilizer keeps A invertible; where() fallback avoids a host sync
        sol = torch.linalg.solve(A, b)
        dets = torch.det(A)
        good = (dets.abs() > 1e-12).unsqueeze(-1)
        opt = torch.where(good, sol.squeeze(-1), midpoint)

        if cfg.clamp_v_to_edge:
            # subsumes the wander check
            edge_len_sq = (edge_vec * edge_vec).sum(dim=-1) + 1e-20
            t = ((opt - pa) * edge_vec).sum(dim=-1) / edge_len_sq
            t = t.clamp(0.0, 1.0).unsqueeze(-1)
            opt = torch.lerp(pa, pb, t)
        else:
            dist_a = torch.norm(opt - pa, dim=-1)
            dist_b = torch.norm(opt - pb, dim=-1)
            wander_bad = (
                (dist_a > cfg.wander_threshold * el) | (dist_b > cfg.wander_threshold * el)
            ).unsqueeze(-1)
            opt = torch.where(wander_bad, midpoint, opt)

    v4 = torch.cat([opt, torch.ones((n_edges, 1), device=device, dtype=dtype)], dim=1)
    err = torch.abs(torch.einsum("ei,eij,ej->e", v4, Qe, v4))

    length_ok = el * el > mesh_scale_sq * 1e-10
    error_ok = err < max_err
    nan_ok = ~torch.isnan(opt).any(dim=-1) & ~torch.isnan(err)
    valid = length_ok & error_ok & nan_ok

    return opt, err + cfg.lambda_edge_length * el * el, valid


def _greedy_matching(
    edges: Any,
    err: Any,
    v_alive: Any,
    max_select: int,
) -> Any:
    """Select independent edges whose (cost, index) key wins at both endpoints."""
    torch = importlib.import_module("torch")
    device = edges.device
    n_edges = edges.shape[0]
    if n_edges == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    va = edges[:, 0]
    vb = edges[:, 1]
    num_verts = v_alive.shape[0]

    err32 = err.to(torch.float32).clamp(min=0).contiguous()
    err_bits = err32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    edge_idx = torch.arange(n_edges, device=device, dtype=torch.int64)
    key = (err_bits << 32) | edge_idx

    INT64_MAX = torch.iinfo(torch.int64).max
    best_key = torch.full((num_verts,), INT64_MAX, dtype=torch.int64, device=device)
    best_key.scatter_reduce_(0, va, key, reduce="amin", include_self=True)
    best_key.scatter_reduce_(0, vb, key, reduce="amin", include_self=True)

    is_winner = (key == best_key[va]) & (key == best_key[vb]) & v_alive[va] & v_alive[vb]
    sel = torch.nonzero(is_winner, as_tuple=True)[0]

    if sel.numel() > max_select:
        sel_err = err[sel]
        top = torch.topk(sel_err, max_select, largest=False).indices
        sel = sel[top]
    return sel


def _build_vert_to_faces_pad(
    faces: Any,
    num_verts: int,
    max_deg: int,
) -> Any:
    """(V, max_deg) incident-face table, -1 padded; faces beyond max_deg are dropped."""
    torch = importlib.import_module("torch")
    device = faces.device
    F = faces.shape[0]
    if F == 0:
        return torch.full((num_verts, max_deg), -1, dtype=torch.int64, device=device)
    v_rep = faces.flatten().long()
    f_rep = torch.arange(F, device=device, dtype=torch.int64).repeat_interleave(3)
    sort_idx = v_rep.argsort()
    sorted_v = v_rep[sort_idx]
    sorted_f = f_rep[sort_idx]
    offsets = torch.searchsorted(
        sorted_v, torch.arange(num_verts + 1, device=device, dtype=sorted_v.dtype)
    )
    slot = torch.arange(sorted_v.shape[0], device=device, dtype=torch.int64) - offsets[sorted_v]
    keep = slot < max_deg
    table = torch.full((num_verts, max_deg), -1, dtype=torch.int64, device=device)
    table[sorted_v[keep], slot[keep]] = sorted_f[keep]
    return table


def _quality_checks_fused(
    verts: Any,
    faces: Any,
    edges: Any,
    opt: Any,
    vert_to_faces: Any,
    cos_threshold: float = 0.0,
    want_link: bool = False,
    chunk_size: int = 100_000,
) -> tuple[Any, Any, Any]:
    """Return normal flips, skinny penalties and optional link validity for each collapse."""
    torch = importlib.import_module("torch")
    E = edges.shape[0]
    device = verts.device
    flip_out = torch.zeros(E, dtype=torch.int32, device=device)
    skinny_out = torch.zeros(E, dtype=verts.dtype, device=device)
    link_out: Any = torch.ones(E, dtype=torch.bool, device=device) if want_link else None

    D = vert_to_faces.shape[1]
    a_all = edges[:, 0]
    b_all = edges[:, 1]
    sqrt3_4 = 4.0 * math.sqrt(3.0)

    for start in range(0, E, chunk_size):
        stop = min(start + chunk_size, E)
        Ec = stop - start
        a = a_all[start:stop]
        b = b_all[start:stop]

        fa = vert_to_faces[a]
        fb = vert_to_faces[b]
        all_f = torch.cat([fa, fb], dim=1)  # (Ec, 2D)
        valid_f = all_f >= 0
        fv = faces[all_f.clamp(min=0)]  # (Ec, 2D, 3)
        a_b = a.view(Ec, 1)
        b_b = b.view(Ec, 1)

        oc = opt[start:stop]
        s0_a = fv[..., 0] == a_b
        s0_b = fv[..., 0] == b_b
        s1_a = fv[..., 1] == a_b
        s1_b = fv[..., 1] == b_b
        s2_a = fv[..., 2] == a_b
        s2_b = fv[..., 2] == b_b
        contains_a = s0_a | s1_a | s2_a
        contains_b = s0_b | s1_b | s2_b
        affected = (contains_a ^ contains_b) & valid_f
        p0 = verts[fv[..., 0]]
        p1 = verts[fv[..., 1]]
        p2 = verts[fv[..., 2]]
        opt_b = oc.view(Ec, 1, 3).expand(-1, 2 * D, -1)
        p0n = torch.where((s0_a | s0_b).unsqueeze(-1), opt_b, p0)
        p1n = torch.where((s1_a | s1_b).unsqueeze(-1), opt_b, p1)
        p2n = torch.where((s2_a | s2_b).unsqueeze(-1), opt_b, p2)

        # The new normal's length is twice the new area for the skinny term.
        e01 = p1n - p0n
        e02 = p2n - p0n
        e12 = p2n - p1n
        n_new = torch.cross(e01, e02, dim=-1)
        nlen_new = torch.norm(n_new, dim=-1)
        edge_sum_sq = (e01 * e01).sum(-1) + (e02 * e02).sum(-1) + (e12 * e12).sum(-1)

        d01, d02, d12 = p1 - p0, p2 - p0, p2 - p1
        n_old = torch.cross(d01, d02, dim=-1)
        nlen_old = torch.norm(n_old, dim=-1)
        old_sum_sq = (d01 * d01).sum(-1) + (d02 * d02).sum(-1) + (d12 * d12).sum(-1)
        # faces too thin to have a normal never count as flipped (scale-relative test)
        usable = (nlen_old > 1e-6 * old_sum_sq) & (nlen_new > 1e-6 * edge_sum_sq)
        flip = (
            ((n_old * n_new).sum(dim=-1) < cos_threshold * nlen_old * nlen_new) & affected & usable
        )
        flip_out[start:stop] = flip.sum(dim=-1).to(torch.int32)

        shape = (sqrt3_4 * 0.5 * nlen_new) / edge_sum_sq.clamp_min(1e-20)
        term = torch.where(affected, 1.0 - shape.clamp(0.0, 1.0), torch.zeros_like(shape))
        skinny_out[start:stop] = term.sum(dim=-1) / affected.sum(dim=-1).clamp_min(1).to(term.dtype)

        if want_link:
            # link condition: common neighbours of a and b must not outnumber the faces on edge ab
            fa_ok = valid_f[:, :D]
            fb_ok = valid_f[:, D:]
            fav = fv[:, :D]
            fbv = fv[:, D:]
            an1 = torch.where(fav[..., 0] == a_b, fav[..., 1], fav[..., 0])
            an2 = torch.where(fav[..., 2] == a_b, fav[..., 1], fav[..., 2])
            bn1 = torch.where(fbv[..., 0] == b_b, fbv[..., 1], fbv[..., 0])
            bn2 = torch.where(fbv[..., 2] == b_b, fbv[..., 1], fbv[..., 2])
            na = torch.stack([an1, an2], dim=-1).reshape(Ec, 2 * D)
            nb = torch.stack([bn1, bn2], dim=-1).reshape(Ec, 2 * D)
            fa_okx = fa_ok.repeat_interleave(2, dim=1)
            fb_okx = fb_ok.repeat_interleave(2, dim=1)
            na[(na == a_b) | (na == b_b) | ~fa_okx] = -1
            nb[(nb == a_b) | (nb == b_b) | ~fb_okx] = -1
            in_b = (na[:, :, None] == nb[:, None, :]) & (na[:, :, None] >= 0)
            na_common = torch.where(in_b.any(dim=2), na, torch.full_like(na, -1))
            cs, _ = na_common.sort(dim=1)
            count_common = ((cs[:, 1:] != cs[:, :-1]) & (cs[:, 1:] >= 0)).sum(dim=1) + (
                cs[:, :1] >= 0
            ).sum(dim=1)
            count_faces = ((fav == b[:, None, None]).any(dim=2) & fa_ok).sum(dim=1)
            link_out[start:stop] = count_common <= count_faces

    return flip_out, skinny_out, link_out


def _gated_errors(
    verts: Any,
    faces: Any,
    edges: Any,
    opt: Any,
    err: Any,
    cfg: QEMConfig,
    link: bool,
) -> Any:
    """Gate flips/link violations and add skinny penalties; Torch caps checked vertex degree."""
    collapse_checks = _kitchen_op("edge_collapse_checks", verts)
    if collapse_checks is not None:
        flips, skinny, link_ok = collapse_checks(verts, faces, edges, opt, cfg.flip_cos_threshold)
    else:
        v_to_f = _build_vert_to_faces_pad(faces, verts.shape[0], cfg.flip_check_max_degree)
        flips, skinny, link_ok = _quality_checks_fused(
            verts, faces, edges, opt, v_to_f, cos_threshold=cfg.flip_cos_threshold, want_link=link
        )
    if link:
        err = err.masked_fill(~link_ok, float("inf"))
    err = err.masked_fill(flips > 0, float("inf"))
    # Multiply by length squared to match the QEM cost scale.
    el_sq = (verts[edges[:, 1]] - verts[edges[:, 0]]).pow(2).sum(dim=-1)
    return err + cfg.skinny_weight * skinny * el_sq


def _drop_degenerate_faces(
    verts: Any,
    faces: Any,
    min_area: float = 1e-14,
) -> Any:
    """Drop faces with repeated indices or area below min_area."""
    torch = importlib.import_module("torch")
    if faces.numel() == 0:
        return faces
    idx_bad = (
        (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    )
    f_good = faces[~idx_bad]
    v0 = verts[f_good[:, 0]]
    v1 = verts[f_good[:, 1]]
    v2 = verts[f_good[:, 2]]
    e0 = v1 - v0
    e2 = v0 - v2
    area = 0.5 * torch.norm(torch.cross(e0, -e2, dim=-1), dim=-1)
    return f_good[area >= min_area]


def _collapse_slivers(
    verts: Any,
    faces: Any,
    min_angle_deg: float = 0.0,
    max_aspect_ratio: float = 0.0,
) -> Any:
    """Resolve sliver triangles by collapsing each sliver's shortest edge (no holes)."""
    torch = importlib.import_module("torch")
    if faces.numel() == 0 or (min_angle_deg <= 0 and max_aspect_ratio <= 0):
        return faces

    fl = faces.long()
    v0 = verts[fl[:, 0]]
    v1 = verts[fl[:, 1]]
    v2 = verts[fl[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    l0 = torch.norm(e0, dim=-1)
    l1 = torch.norm(e1, dim=-1)
    l2 = torch.norm(e2, dim=-1)
    area = 0.5 * torch.norm(torch.cross(e0, -e2, dim=-1), dim=-1)

    bad = torch.zeros(faces.shape[0], dtype=torch.bool, device=verts.device)
    if max_aspect_ratio > 0:
        max_edge = torch.maximum(torch.maximum(l0, l1), l2)
        aspect = max_edge * max_edge / (2.0 * area + 1e-12)
        bad = bad | (aspect > max_aspect_ratio)
    if min_angle_deg > 0:
        cos_a = (l1 * l1 + l2 * l2 - l0 * l0) / (2 * l1 * l2 + 1e-12)
        cos_b = (l0 * l0 + l2 * l2 - l1 * l1) / (2 * l0 * l2 + 1e-12)
        cos_c = (l0 * l0 + l1 * l1 - l2 * l2) / (2 * l0 * l1 + 1e-12)
        cos_all = torch.stack([cos_a, cos_b, cos_c], dim=-1)
        angles_deg = torch.acos(torch.clamp(cos_all, -1, 1)) * (180.0 / math.pi)
        bad = bad | (angles_deg.min(dim=-1).values < min_angle_deg)

    if not bad.any():
        return faces

    # chained collapses all merge into the lowest vertex
    bad_idx = torch.nonzero(bad, as_tuple=True)[0]
    slot = torch.stack([l0, l1, l2], dim=-1)[bad_idx].argmin(dim=-1)
    merge = torch.stack([fl[bad_idx, slot], fl[bad_idx, (slot + 1) % 3]], dim=1)
    new_faces = _connected_labels(merge, verts.shape[0])[fl]
    nondeg = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    return new_faces[nondeg].to(dtype=faces.dtype)


def _drop_duplicate_faces(faces: Any) -> Any:
    """Remove duplicate vertex sets, preserving the first occurrence's winding."""
    torch = importlib.import_module("torch")
    if faces.shape[0] <= 1:
        return faces
    groups, inv = torch.unique(torch.sort(faces, dim=1)[0], dim=0, return_inverse=True)
    if groups.shape[0] == faces.shape[0]:
        return faces
    arange = torch.arange(faces.shape[0], dtype=torch.int32, device=faces.device)
    first = torch.full((groups.shape[0],), faces.shape[0], dtype=torch.int32, device=faces.device)
    first.scatter_reduce_(0, inv, arange, reduce="amin", include_self=True)
    return faces[first]


def _folded_faces(faces: Any) -> Any:
    """(F,) bool: faces sharing their vertex set with another face.
    Marks every mixed-winding face and all but the first same-winding repeat.
    """
    torch = importlib.import_module("torch")
    key_sorted, order = torch.sort(faces, dim=1)
    groups, inv = torch.unique(key_sorted, dim=0, return_inverse=True)
    n_groups = groups.shape[0]
    # winding = parity of the sorting permutation
    odd = (
        (order[:, 0] > order[:, 1]).long()
        + (order[:, 0] > order[:, 2]).long()
        + (order[:, 1] > order[:, 2]).long()
    ) % 2
    n_odd = torch.zeros(n_groups, dtype=torch.long, device=faces.device).scatter_add_(0, inv, odd)
    mixed = (n_odd > 0) & (n_odd < torch.bincount(inv, minlength=n_groups))
    arange = torch.arange(faces.shape[0], dtype=torch.int32, device=faces.device)
    first = torch.full((n_groups,), faces.shape[0], dtype=torch.int32, device=faces.device)
    first.scatter_reduce_(0, inv, arange, reduce="amin")
    return mixed[inv] | (first[inv] != arange)


def _drop_unused_verts(
    verts: Any,
    faces: Any,
    colors: Any = None,
) -> tuple[Any, Any, Any]:
    """Remove vertices not referenced by any face; remap faces and filter colors."""
    torch = importlib.import_module("torch")
    if verts.shape[0] == 0 or faces.numel() == 0:
        return verts, faces, colors
    used = torch.zeros(verts.shape[0], dtype=torch.bool, device=verts.device)
    used[faces[:, 0]] = True
    used[faces[:, 1]] = True
    used[faces[:, 2]] = True
    remap = used.long().cumsum(0) - 1
    new_verts = verts[used]
    new_faces = remap[faces.long()]
    new_colors = colors[used] if colors is not None else None
    return new_verts, new_faces, new_colors


def _split_touching_sheets(v: Any, f: Any, max_passes: int = 4) -> tuple[Any, Any]:
    """Split touching sheets, pairing forward/backward fan faces by normal agreement.

    Conflicting pairings flip the least certain member up to max_passes times.
    A closed, consistently oriented mesh stays closed.
    """
    torch = importlib.import_module("torch")
    dev = f.device
    m = f.shape[0]
    n = int(f.max()) + 1
    fv = f.reshape(-1)

    def nxt(c: Any) -> Any:
        return c - c % 3 + (c + 1) % 3

    fe = f.roll(-1, 1).reshape(-1)
    key = torch.minimum(fv, fe) * n + torch.maximum(fv, fe)
    del fe
    order = torch.argsort(key)
    ks = key[order]
    edge_cnt = torch.unique_consecutive(ks, return_counts=True)[1]
    cnt = torch.repeat_interleave(edge_cnt, edge_cnt)
    pair = torch.nonzero(ks[1:] == ks[:-1], as_tuple=True)[0]
    pair = pair[cnt[pair] == 2]
    h1, h2 = order[pair], order[pair + 1]
    del pair

    fans = bool((cnt > 2).any())
    h: Any = None
    size: Any = None
    fan_key: Any = None
    g: Any = None
    swap: Any = None
    pick_f: Any = None
    pick_b: Any = None
    nf: Any = None
    nb: Any = None
    g_start: Any = None
    margin: Any = None
    hf: Any = None
    gf: Any = None
    rf: Any = None
    if fans:
        h = order[cnt > 2]
        size = edge_cnt[edge_cnt > 2]
        fan_key = ks[torch.cumsum(edge_cnt, 0) - edge_cnt][edge_cnt > 2]
        g = torch.repeat_interleave(torch.arange(size.numel(), device=dev), size)
        d = (fv[h] > fv[nxt(h)]).long()
        o = torch.argsort((g * 2 + d) * (3 * m) + h)
        h, g, d = h[o], g[o], d[o]
        per_dir = torch.bincount(g * 2 + d, minlength=2 * size.numel()).view(-1, 2)
        g_start = torch.cumsum(size, 0) - size
        rank = torch.arange(h.numel(), device=dev) - g_start[g] - d * per_dir[g, 0]
        # dot[g, i, j]: normal agreement of the i-th forward and j-th backward face
        first2 = rank < 2
        tri = v[f[h[first2] // 3]]
        fn = torch.nn.functional.normalize(
            torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
        )
        nrm = torch.zeros((size.numel(), 2, 2, 3), dtype=v.dtype, device=dev)
        nrm[g[first2], d[first2], rank[first2]] = fn
        dot = torch.einsum("gic,gjc->gij", nrm[:, 0], nrm[:, 1])
        nf, nb = per_dir[:, 0], per_dir[:, 1]
        swap = ((dot[:, 0, 1] + dot[:, 1, 0]) > (dot[:, 0, 0] + dot[:, 1, 1])).long()
        pick_f = (dot[:, 1, 0] > dot[:, 0, 0]).long()
        pick_b = (dot[:, 0, 1] > dot[:, 0, 0]).long()
        # how sure each choice is; larger fans pair by order and gain nothing from a flip
        margin = torch.full_like(dot[:, 0, 0], float("inf"))
        margin = torch.where(
            (nf == 2) & (nb == 2),
            (dot[:, 0, 1] + dot[:, 1, 0] - dot[:, 0, 0] - dot[:, 1, 1]).abs(),
            margin,
        )
        margin = torch.where((nf == 2) & (nb == 1), (dot[:, 1, 0] - dot[:, 0, 0]).abs(), margin)
        margin = torch.where((nf == 1) & (nb == 2), (dot[:, 0, 1] - dot[:, 0, 0]).abs(), margin)
        hf, gf, rf = h[d == 0], g[d == 0], rank[d == 0]
    del key, order, ks, edge_cnt, cnt

    def join(flip: Any) -> Any:
        a1, a2 = h1, h2
        if fans:
            sw, pf, pb = swap ^ flip, pick_f ^ flip, pick_b ^ flip
            partner = torch.where(rf < nb[gf], rf, -1)  # larger fans pair by order
            partner = torch.where((nf[gf] == 2) & (nb[gf] == 2), rf ^ sw[gf], partner)
            partner = torch.where(
                (nf[gf] == 2) & (nb[gf] == 1), torch.where(rf == pf[gf], 0, -1), partner
            )
            partner = torch.where((nf[gf] == 1) & (nb[gf] == 2), pb[gf], partner)
            ok = partner >= 0
            a1 = torch.cat([a1, hf[ok]])
            a2 = torch.cat([a2, h[g_start[gf[ok]] + nf[gf[ok]] + partner[ok]]])
        same_dir = fv[a1] == fv[a2]
        a = torch.cat([a1, nxt(a1)])
        b = torch.cat([torch.where(same_dir, a2, nxt(a2)), torch.where(same_dir, nxt(a2), a2)])
        return _connected_labels(torch.stack([a, b], 1), 3 * m)

    flip = 0
    label: Any = None
    for _ in range(max_passes):
        label = join(flip)
        if not fans:
            break
        # Corners merge within one vertex, so only a fan's own half-edges can repeat a label pair.
        lk = torch.minimum(label[h], label[nxt(h)]) * (3 * m) + torch.maximum(
            label[h], label[nxt(h)]
        )
        _, inv, c = torch.unique(lk, return_inverse=True, return_counts=True)
        stuck = torch.zeros(size.numel(), dtype=torch.bool, device=dev)
        stuck[g[c[inv] > 2]] = True
        stuck &= torch.isfinite(margin)
        if not bool(stuck.any()):
            break
        # Flip the least certain fan per group; flipping a whole ring keeps it inconsistent.
        idx = stuck.nonzero().squeeze(1)
        va, vb = fan_key[idx] // n, fan_key[idx] % n
        grp = _connected_labels(torch.stack([va, vb], 1), n)[va]
        least = torch.full((n,), float("inf"), dtype=margin.dtype, device=dev).scatter_reduce(
            0, grp, margin[idx], "amin"
        )
        flip = flip ^ torch.zeros_like(fan_key).index_fill(0, idx[margin[idx] == least[grp]], 1)
    # labels are the smallest corner of their component, so the roots in order are the new vertices
    root = label == torch.arange(3 * m, device=dev)
    return (root.cumsum(0) - 1)[label].view(m, 3), fv[root]


def _repair_nonmanifold_edges(
    verts: Any,
    faces: Any,
) -> tuple[Any, Any, Any]:
    """Explode corners, then re-merge only across manifold edges; retain source indices."""
    torch = importlib.import_module("torch")
    if faces.numel() == 0:
        return verts, faces, torch.arange(verts.shape[0], device=verts.device)
    nf = faces.shape[0]
    corner_vert = faces.long().reshape(-1)  # (3F,) original vertex per corner
    corner = torch.arange(3 * nf, device=faces.device).view(nf, 3)

    # half-edges keyed by (vmin, vmax), with the corners at each endpoint
    va = corner_vert
    vb = faces.long()[:, [1, 2, 0]].reshape(-1)
    ca = corner.reshape(-1)
    cb = corner[:, [1, 2, 0]].reshape(-1)
    swap = va > vb
    keys = torch.minimum(va, vb) * (verts.shape[0] + 1) + torch.maximum(va, vb)
    keys, order = torch.sort(keys)
    cmin = torch.where(swap, cb, ca)[order]
    cmax = torch.where(swap, ca, cb)[order]
    _, cnt = torch.unique_consecutive(keys, return_counts=True)
    man = (torch.cumsum(cnt, 0) - cnt)[cnt == 2]  # manifold edges (exactly 2 incident faces)
    pairs = torch.cat(
        [torch.stack([cmin[man], cmin[man + 1]], 1), torch.stack([cmax[man], cmax[man + 1]], 1)]
    )
    roots, labels = torch.unique(_connected_labels(pairs, 3 * nf), return_inverse=True)
    src = corner_vert[roots]
    return verts[src], labels.view(nf, 3).to(faces.dtype), src


def _kitchen_op(name: str, tensor: Any) -> Any:
    """Return the kitchen op only when its CUDA backend can execute it."""
    torch = importlib.import_module("torch")
    ck = getattr(importlib.import_module("dinkster_inference.quant_ops"), "ck", None)
    if ck is None or not hasattr(ck, name) or not tensor.is_cuda:
        return None
    cuda = ck.list_backends().get("cuda", {})
    if not (
        cuda.get("available", False)
        and not cuda.get("disabled", True)
        and name in cuda.get("capabilities", ())
    ):
        return None
    floor = ck.registry.get_constraints("cuda", name).min_compute_capability
    if floor is not None and torch.cuda.get_device_capability(tensor.device) < floor:
        return None
    return getattr(ck, name)


def _drop_small_components(
    verts: Any,
    faces: Any,
    min_faces: int,
) -> Any:
    """Faces of the connected components with at least min_faces faces."""
    torch = importlib.import_module("torch")
    if faces.numel() == 0 or min_faces <= 1:
        return faces
    labels = _connected_labels(torch.cat([faces[:, :2], faces[:, 1:]]), verts.shape[0])
    face_label = labels[faces[:, 0]]
    unique_labels, counts = torch.unique(face_label, return_counts=True)
    big_labels = unique_labels[counts >= min_faces]
    # never drop every component
    if big_labels.shape[0] in (0, unique_labels.shape[0]):
        return faces
    return faces[torch.isin(face_label, big_labels)]


def clean_mesh(
    verts: Any,
    faces: Any,
    colors: Any = None,
    weld_epsilon_rel: float = 0.0,
    drop_duplicates: bool = True,
    min_component_faces: int = 0,
    min_angle_deg: float = 0.0,
    max_aspect_ratio: float = 0.0,
) -> tuple[Any, Any, Any]:
    """Weld and remove degenerate faces, slivers, duplicates, tiny components and unused verts."""
    torch = importlib.import_module("torch")
    f = faces.long() if faces.numel() > 0 else faces
    if weld_epsilon_rel > 0:
        eps = (
            torch.norm(verts.max(dim=0)[0] - verts.min(dim=0)[0]) * weld_epsilon_rel
        )  # 0-d, no sync
        verts, f, colors = weld_vertices(verts, f, colors, 0.0, eps)
    f = _drop_degenerate_faces(verts, f)
    f = _collapse_slivers(verts, f, min_angle_deg=min_angle_deg, max_aspect_ratio=max_aspect_ratio)
    if drop_duplicates:
        f = _drop_duplicate_faces(f)
    f = _drop_small_components(verts, f, min_component_faces)
    return _drop_unused_verts(verts, f, colors)


def qem_simplify(
    vertices: Any,
    faces: Any,
    target_faces: int,
    colors: Any = None,
    config: QEMConfig | None = None,
) -> tuple[Any, Any, Any]:
    """QEM-simplify a mesh to about target_faces faces; returns (v, f, colors)."""
    torch = importlib.import_module("torch")
    cfg = config or QEMConfig()

    device = vertices.device
    in_v_dtype = vertices.dtype
    in_f_dtype = faces.dtype
    in_c_dtype = colors.dtype if colors is not None else None

    # preclean on copies, since collapses edit verts and colors in place
    verts, faces, colors_w = clean_mesh(
        vertices.to(dtype=torch.float32, copy=True),
        faces.to(dtype=torch.int64),
        colors.to(dtype=torch.float32, copy=True) if colors is not None else None,
        weld_epsilon_rel=cfg.preclean_weld_epsilon_rel,
    )

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    if num_faces <= target_faces or num_verts < 4:
        return (
            verts.to(in_v_dtype),
            faces.to(in_f_dtype),
            (colors_w.to(in_c_dtype) if colors_w is not None else None),
        )

    v_alive = torch.ones(num_verts, dtype=torch.bool, device=device)
    f_alive = torch.ones(num_faces, dtype=torch.bool, device=device)

    bbox = verts.max(dim=0)[0] - verts.min(dim=0)[0]
    mesh_scale = torch.norm(bbox)  # 0-d tensor; never .item()'d
    mesh_scale_sq = mesh_scale * mesh_scale
    stabilizer = mesh_scale_sq * cfg.stabilizer_scale
    # Reject costs above the squared doubled bbox diagonal, or 1 on a tiny bbox.
    max_err = torch.where(mesh_scale < 5e-7, torch.ones_like(mesh_scale), 4.0 * mesh_scale_sq)
    quadrics: Any = None

    thresh = float(cfg.threshold_start) * float(mesh_scale_sq) if cfg.threshold_driver else 0.0

    merge_map = torch.arange(num_verts, device=device)
    gen = torch.Generator(device=device).manual_seed(
        0
    )  # the edge sample must not depend on the global RNG state

    # dropped once only the link condition keeps the threshold driver from the target
    link = cfg.enforce_link_condition

    # Python-int face estimate to avoid host syncs in the loop, re-synced at compaction
    py_n_faces = num_faces

    iteration = 0

    _start_faces = num_faces
    _prog_total = max(1, _start_faces - int(target_faces))

    def _qreport() -> None:
        pct = min(100, max(0, int(100 * (_start_faces - py_n_faces) / _prog_total)))
        report_progress(pct, 100)

    while True:
        if py_n_faces <= target_faces:
            break
        _qreport()

        if faces.shape[0] == 0:
            break

        # unique edges; those used by one face are boundary
        packed = torch.cat(
            [
                torch.add(
                    torch.maximum(faces[:, i], faces[:, j]),
                    torch.minimum(faces[:, i], faces[:, j]),
                    alpha=num_verts,
                )
                for i, j in ((0, 1), (1, 2), (2, 0))
            ]
        )
        packed, counts = torch.unique(packed, return_counts=True)
        edges_orig = torch.stack([packed // num_verts, packed % num_verts], dim=1)
        b_edges = edges_orig[counts == 1]
        del packed, counts

        # Midpoint rebuilds quadrics each round; QEM accumulates them.
        if quadrics is None:
            quadrics = _build_quadrics(verts, faces, cfg, b_edges)

        if edges_orig.shape[0] > cfg.sampling_cap:
            # a Bernoulli sample: randperm would sort every edge
            edges_orig = edges_orig[
                torch.rand(edges_orig.shape[0], device=device, generator=gen)
                < cfg.sampling_cap / edges_orig.shape[0]
            ]

        if cfg.placement_mode != "qem":
            vib = torch.zeros(num_verts, dtype=torch.bool, device=device)
            vib[b_edges.flatten()] = True
        else:
            vib = None
        parts = [
            _edge_errors(
                verts,
                quadrics,
                edges_orig[s : s + _EDGE_CHUNK],
                stabilizer,
                max_err,
                mesh_scale_sq,
                cfg,
                vert_is_boundary=vib,
            )
            for s in range(0, edges_orig.shape[0], _EDGE_CHUNK)
        ]
        optimal, err, valid = (torch.cat(p) for p in zip(*parts, strict=True))
        del parts
        if cfg.threshold_driver:
            quadrics = None  # Collapse and the next edge extraction run without these arrays.
        valid_idx = torch.nonzero(valid, as_tuple=True)[0]
        edges_orig = edges_orig[valid_idx]
        optimal = optimal[valid_idx]
        err = err[valid_idx]

        faces_to_remove = py_n_faces - target_faces
        n_faces_round_start = py_n_faces
        # ~2 faces removed per collapse
        cap_to_target = max(1, faces_to_remove // 2)

        if cfg.threshold_driver:
            cand = err <= thresh
            esc = 0
            while not bool(cand.any()) and esc < 50:
                thresh *= 10.0
                cand = err <= thresh
                esc += 1
            full_band = bool(cand.all())
            cand_idx = torch.nonzero(cand, as_tuple=True)[0]
            ce = edges_orig[cand_idx]
            copt = optimal[cand_idx]
            cerr = err[cand_idx].clone()
            if ce.shape[0] > 0:
                cerr = _gated_errors(verts, faces, ce, copt, cerr, cfg, link)
                # penalties can push edges out of the band
                keep = cerr <= thresh
                ce = ce[keep]
                copt = copt[keep]
                cerr = cerr[keep]
            edges_orig = ce
            optimal = copt
            sel = _greedy_matching(ce, cerr, v_alive, cap_to_target)
            if link and full_band and 2 * sel.numel() < min(0.01 * py_n_faces, 2 * cap_to_target):
                # whole band gated yet < 1% removed: the link condition protects topology the target
                # can't afford (e.g. tiny handles of a damaged mesh)
                link = False
            if sel.numel() == 0:
                thresh *= 10.0
                iteration += 1
                if iteration >= cfg.max_iterations:
                    break
                continue
        else:
            max_collapses = min(
                cfg.max_collapses_ceiling,
                max(cfg.max_collapses_floor, int(faces_to_remove * cfg.max_collapses_fraction)),
            )
            # relative cap avoids overshooting the target in one round
            rel_cap = max(1, int(py_n_faces * cfg.max_collapses_relative_cap))
            max_collapses = min(max_collapses, rel_cap, cap_to_target)

            if edges_orig.shape[0] > 0:
                err = _gated_errors(verts, faces, edges_orig, optimal, err, cfg, link)
                # +inf edges must not win the matching where all their neighbours are rejected too
                keep = torch.isfinite(err)
                edges_orig = edges_orig[keep]
                optimal = optimal[keep]
                err = err[keep]

            sel = _greedy_matching(edges_orig, err, v_alive, max_collapses)

            if sel.numel() == 0:
                break

        ed_sel = edges_orig[sel]
        v_a = ed_sel[:, 0]
        v_b = ed_sel[:, 1]
        new_pos = optimal[sel]
        # Release round arrays before collapse and the next round allocate.
        del b_edges, edges_orig, optimal, err, valid, sel, ed_sel

        if colors_w is not None:
            pa_sel = verts[v_a]
            pb_sel = verts[v_b]
            edge_vec = pb_sel - pa_sel
            edge_len_sq = (edge_vec * edge_vec).sum(dim=-1) + 1e-20
            t = ((new_pos - pa_sel) * edge_vec).sum(dim=-1) / edge_len_sq
            colors_w[v_a] = torch.lerp(
                colors_w[v_a], colors_w[v_b], t.clamp(0.0, 1.0).unsqueeze(-1)
            )

        verts[v_a] = new_pos
        v_alive[v_b] = False
        if not cfg.threshold_driver:
            quadrics[v_a] += quadrics[v_b]

        merge_map[v_b] = v_a
        for s in range(0, faces.shape[0], _FACE_CHUNK):  # in place, a chunk at a time
            faces[s : s + _FACE_CHUNK] = merge_map[faces[s : s + _FACE_CHUNK]]
        merge_map[v_b] = v_b  # back to identity

        bad = (
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        )
        f_alive.masked_fill_(bad, False)
        py_n_faces -= 2 * v_a.numel()
        del v_a, v_b, new_pos, bad
        if cfg.enforce_link_condition and not link:
            # Drop folds after collapsing handles so counts reflect the actual surface.
            alive_idx = torch.nonzero(f_alive, as_tuple=True)[0]
            f_alive[alive_idx[_folded_faces(faces[alive_idx])]] = False
            py_n_faces = int(f_alive.sum())

        if cfg.threshold_driver:
            removed = n_faces_round_start - py_n_faces
            if removed < 0.01 * n_faces_round_start:
                thresh *= 10.0

        iteration += 1

        # The next round needs only alive faces, so discard dead faces now.
        faces = faces[f_alive]
        num_faces = faces.shape[0]
        f_alive = torch.ones(num_faces, dtype=torch.bool, device=device)
        py_n_faces = num_faces

        if iteration >= cfg.max_iterations:
            break

    _qreport()

    # alive faces are non-degenerate and only use alive verts
    final_v = verts[v_alive]
    final_c = colors_w[v_alive] if colors_w is not None else None
    remap = v_alive.long().cumsum(0) - 1
    final_f = _drop_duplicate_faces(remap[faces[f_alive]])

    # split fused sheets after dedup and before postclean pruning
    if cfg.repair_nonmanifold and final_f.numel() > 0:
        final_f, _src = _split_touching_sheets(final_v, final_f)
        final_v, final_f, _src2 = _repair_nonmanifold_edges(final_v[_src], final_f)
        if final_c is not None:
            final_c = final_c[_src[_src2]]

    # postclean; already welded and deduplicated
    if cfg.postclean and final_f.numel() > 0:
        final_v, final_f, final_c = clean_mesh(
            final_v,
            final_f,
            final_c,
            drop_duplicates=False,
            min_component_faces=cfg.postclean_min_component_faces,
            min_angle_deg=cfg.postclean_min_angle_deg,
            max_aspect_ratio=cfg.postclean_max_aspect_ratio,
        )

    return (
        final_v.to(in_v_dtype),
        final_f.to(in_f_dtype),
        final_c.to(in_c_dtype) if final_c is not None else None,
    )
