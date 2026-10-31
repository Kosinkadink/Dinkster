"""Render mesh UV charts with deterministic colors and boundary outlines."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    MESH_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_mesh,
)


def _sorted_edges(indices: Any) -> tuple[Any, Any, Any, Any]:
    np = importlib.import_module("numpy")
    a = indices.ravel().astype(np.int64)
    b = np.roll(indices, -1, axis=1).ravel().astype(np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys = lo * (int(indices.max()) + 1) + hi
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    face_ids = (np.arange(a.size, dtype=np.int64) // 3)[order]
    first = np.ones(sorted_keys.size, dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return face_ids, lo[order], hi[order], first


def _chart_ids(indices: Any) -> Any:
    np = importlib.import_module("numpy")
    if indices.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    sparse = importlib.import_module("scipy.sparse")
    csgraph = importlib.import_module("scipy.sparse.csgraph")
    face_ids, _lo, _hi, first = _sorted_edges(indices)
    groups = np.cumsum(first) - 1
    starts = np.nonzero(first)[0]
    rows = face_ids[starts[groups[~first]]]
    cols = face_ids[~first]
    if rows.size == 0:
        return np.arange(indices.shape[0], dtype=np.int64)
    adjacency = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(indices.shape[0], indices.shape[0]),
    )
    _, labels = csgraph.connected_components(adjacency, directed=False)
    return labels.astype(np.int64)


def _palette(count: int) -> Any:
    np = importlib.import_module("numpy")
    colors = np.array(
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
    permutation = np.random.RandomState(42).permutation(max(1, count))
    result = np.empty((count, 3), dtype=np.float32)
    for index in range(count):
        result[index] = colors[permutation[index % len(permutation)] % 20]
    return result


def _render(uvs_np: Any, indices_np: Any, resolution: int, device: Any) -> Any:
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    chart_ids = torch.from_numpy(_chart_ids(indices_np)).to(device=device, dtype=torch.long)
    uvs = torch.from_numpy(uvs_np).to(device=device, dtype=torch.float32)
    indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
    image = torch.tensor([0.13, 0.13, 0.13], dtype=torch.float32, device=device)
    image = image.expand(resolution, resolution, 3).contiguous()
    if indices.numel() == 0:
        return image
    count = int(chart_ids.max().item()) + 1 if chart_ids.numel() else 1
    colors = torch.from_numpy(_palette(count)).to(device=device, dtype=torch.float32)
    pixels = uvs.clone()
    pixels[:, 0] = pixels[:, 0].clamp(0.0, 1.0) * (resolution - 1)
    pixels[:, 1] = pixels[:, 1].clamp(0.0, 1.0) * (resolution - 1)
    triangles = pixels[indices]
    x0, y0 = triangles[:, 0, 0], triangles[:, 0, 1]
    x1, y1 = triangles[:, 1, 0], triangles[:, 1, 1]
    x2, y2 = triangles[:, 2, 0], triangles[:, 2, 1]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegenerate = denominator.abs() > 1e-20
    xmin = torch.minimum(torch.minimum(x0, x1), x2).floor().clamp_(0, resolution - 1).long()
    xmax = torch.maximum(torch.maximum(x0, x1), x2).ceil().clamp_(0, resolution - 1).long()
    ymin = torch.minimum(torch.minimum(y0, y1), y2).floor().clamp_(0, resolution - 1).long()
    ymax = torch.maximum(torch.maximum(y0, y1), y2).ceil().clamp_(0, resolution - 1).long()
    # Test only triangles overlapping each tile, rather than every pixel/face pair.
    for ty in range(0, resolution, 64):
        end_y = min(ty + 64, resolution)
        for tx in range(0, resolution, 64):
            end_x = min(tx + 64, resolution)
            mask = nondegenerate & (xmin < end_x) & (xmax >= tx) & (ymin < end_y) & (ymax >= ty)
            if not mask.any():
                continue
            selected = torch.nonzero(mask, as_tuple=True)[0]
            ys = torch.arange(ty, end_y, dtype=torch.float32, device=device) + 0.5
            xs = torch.arange(tx, end_x, dtype=torch.float32, device=device) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            ax, ay = x0[selected][:, None, None], y0[selected][:, None, None]
            bx, by = x1[selected][:, None, None], y1[selected][:, None, None]
            cx, cy = x2[selected][:, None, None], y2[selected][:, None, None]
            divisor = denominator[selected][:, None, None]
            u = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / divisor
            v = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / divisor
            w = 1.0 - u - v
            inside = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)
            if not inside.any():
                continue
            hit = inside.any(dim=0)
            best = selected[inside.int().argmax(dim=0)]
            tile_colors = colors[chart_ids[best]]
            tile = image[ty:end_y, tx:end_x]
            tile[hit] = tile_colors[hit]
            image[ty:end_y, tx:end_x] = tile
    # Only edges with one incident face outline a chart; shared edges stay colored.
    _face_ids, lo, hi, first = _sorted_edges(indices_np)
    starts = np.nonzero(first)[0]
    counts = np.diff(np.append(starts, first.size))
    boundary = counts == 1
    pixels_np = pixels.cpu().numpy()
    px_xs: list[Any] = []
    px_ys: list[Any] = []
    for a, b in zip(lo[starts[boundary]], hi[starts[boundary]], strict=True):
        p0, p1 = pixels_np[a], pixels_np[b]
        steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) + 1)
        if steps <= 1:
            continue
        ts = np.linspace(0.0, 1.0, steps)
        xs = (p0[0] + (p1[0] - p0[0]) * ts).astype(np.int32)
        ys = (p0[1] + (p1[1] - p0[1]) * ts).astype(np.int32)
        valid = (xs >= 0) & (xs < resolution) & (ys >= 0) & (ys < resolution)
        px_xs.append(xs[valid])
        px_ys.append(ys[valid])
    if px_xs:
        xs_all = torch.from_numpy(np.concatenate(px_xs)).to(device=device, dtype=torch.long)
        ys_all = torch.from_numpy(np.concatenate(px_ys)).to(device=device, dtype=torch.long)
        image[ys_all, xs_all] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    return image


class RenderUVAtlas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RenderUVAtlas",
            display_name="Render UV Atlas",
            category="3d/texturing",
            description="Renders a mesh's UV layout as an image.",
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=64, max=4096, step=64),
                ),
            ),
            outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls, *, mesh: object, resolution: int) -> Mapping[str, object]:
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        source = materialize_mesh(mesh)
        if source.uvs is None:
            raise RuntimeError("mesh has no UVs to render. Run UnwrapMesh first.")
        uvs = source.uvs.detach().cpu().numpy()
        if uvs.ndim == 3:
            uvs = uvs[0]
        faces = source.faces
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        if faces.ndim == 3:
            faces = faces[0]
        result = _render(
            np.ascontiguousarray(uvs, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.int64),
            int(resolution),
            management.get_torch_device(),
        )
        return cls.outputs(image=result.detach().cpu().unsqueeze(0))
