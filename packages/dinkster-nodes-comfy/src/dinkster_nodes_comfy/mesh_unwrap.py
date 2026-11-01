"""Chart segmentation, parameterization and packing for native mesh UVs."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    MESH_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    initialize_inference_process,
    materialize_mesh,
    report_progress,
)


def _weld(vertices: Any, faces: Any, distance: float) -> tuple[Any, Any, Any]:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    v, f = vertices.cpu().numpy(), faces.cpu().numpy()
    if v.size == 0:
        return vertices, faces, None
    extent = float(np.linalg.norm(v.max(0) - v.min(0)))
    tolerance = distance if distance > 0 else 1e-5 * extent
    if tolerance <= 0:
        return vertices, faces, None
    _, inverse = np.unique(np.round(v / tolerance).astype(np.int64), axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    if count >= v.shape[0]:
        return vertices, faces, None
    welded = np.zeros((count, 3), dtype=np.float32)
    counts = np.zeros(count, dtype=np.int64)
    np.add.at(welded, inverse, v)
    np.add.at(counts, inverse, 1)
    welded /= counts[:, None]
    # The reference retains the last source vertex in each quantized group.
    original = np.empty(count, dtype=np.int64)
    original[inverse] = np.arange(v.shape[0], dtype=np.int64)
    return (
        torch.from_numpy(welded).to(vertices),
        torch.from_numpy(inverse[f]).to(faces),
        original,
    )


def _unwrap(
    vertices: Any,
    faces: Any,
    segmenter: str,
    resolution: int,
    padding: int,
    distance: float,
    device: Any,
) -> tuple[Any, Any, Any]:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    mesh_ops = importlib.import_module("dinkster_native.model3d.mesh_ops.uv_unwrap.mesh")
    segment = importlib.import_module("dinkster_native.model3d.mesh_ops.uv_unwrap.segment")
    parameterize = importlib.import_module(
        "dinkster_native.model3d.mesh_ops.uv_unwrap.parameterize"
    )
    pack = importlib.import_module("dinkster_native.model3d.mesh_ops.uv_unwrap.pack")
    vertices, faces, original = _weld(vertices.float(), faces.long().reshape(-1, 3), distance)
    degenerate = (
        (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    )
    faces = faces[~degenerate]
    mesh = mesh_ops.build_mesh(vertices, faces)

    def segment_progress(done: int, total: int) -> None:
        report_progress(20 + 330 * done // max(total, 1), 1000)

    if segmenter == "pec":
        charts = segment.cluster_charts_pec(mesh, max_cost=1.0, progress_callback=segment_progress)
    elif segmenter == "adaptive":
        charts = segment.segment_charts(mesh, max_cost=2.0, progress_callback=segment_progress)
    else:
        raise ValueError(f"unknown segmenter '{segmenter}'. valid: pec, adaptive")
    count = int(charts.max().item()) + 1 if charts.numel() else 0
    areas = mesh_ops.chart_3d_areas(mesh.face_area, charts, count).detach().cpu()
    if count == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.zeros((0, 3), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
        )

    charts_np = charts.cpu().numpy()
    faces_np, vertices_np = mesh.faces.cpu().numpy(), mesh.vertices.cpu().numpy()
    neighbors = mesh.face_face.cpu().numpy()
    order = np.argsort(charts_np, kind="stable")
    face_counts = np.bincount(charts_np, minlength=count)
    face_offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(face_counts, out=face_offsets[1:])
    sorted_faces, sorted_charts = faces_np[order], charts_np[order]
    vertex_count = max(vertices_np.shape[0], 1)
    slot_charts = np.repeat(sorted_charts, 3)
    keys, local_flat = np.unique(
        slot_charts * vertex_count + sorted_faces.reshape(-1), return_inverse=True
    )
    used = keys % vertex_count
    vertex_counts = np.bincount(keys // vertex_count, minlength=count)
    vertex_offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(vertex_counts, out=vertex_offsets[1:])
    local_faces = (local_flat - vertex_offsets[slot_charts]).reshape(-1, 3)
    positions = np.empty(order.size, dtype=np.int64)
    positions[order] = np.arange(order.size) - face_offsets[sorted_charts]
    sorted_neighbors = neighbors[order]
    safe_neighbors = np.maximum(sorted_neighbors, 0)
    keep = (sorted_neighbors >= 0) & (charts_np[safe_neighbors] == sorted_charts[:, None])
    local_neighbors = np.where(keep, positions[safe_neighbors], -1)

    vertex_charts = (keys // vertex_count).astype(np.int64)
    concatenated = vertices_np[used].astype(np.float64)
    global_faces = local_faces + vertex_offsets[sorted_charts][:, None]
    face_positions = positions[order]
    uv0 = parameterize.ortho_project_concat(concatenated, vertex_charts, count)
    rms, maximum, flips, zeros = parameterize.stretch_metrics_concat(
        concatenated, uv0, global_faces, sorted_charts, count
    )
    valid = (vertex_counts >= 3) & (face_counts > 0)
    automatic = valid & (face_counts <= 5)
    flip_ok = (flips == 0) | (flips == face_counts)
    candidates = valid & ~automatic & flip_ok & (zeros == 0) & (rms <= 1.5) & (maximum <= 2.0)
    ortho_ok = automatic.copy()
    for chart in np.nonzero(candidates)[0]:
        f0, f1 = face_offsets[chart : chart + 2]
        v0, v1 = vertex_offsets[chart : chart + 2]
        if not parameterize._uv_boundary_self_intersects(
            uv0[v0:v1], local_faces[f0:f1], local_neighbors[f0:f1]
        ):
            ortho_ok[chart] = True
    needs_lscm = valid & ~ortho_ok
    batchable = vertex_counts <= parameterize.LSCM_BATCH_MAX_VERTS
    solved = parameterize.lscm_charts_batch(
        concatenated,
        uv0,
        global_faces,
        face_positions,
        sorted_charts,
        vertex_charts,
        vertex_offsets,
        np.nonzero(needs_lscm & batchable)[0],
        count,
        device=device,
    )
    for chart in np.nonzero(needs_lscm & ~batchable)[0]:
        f0, f1 = face_offsets[chart : chart + 2]
        v0, v1 = vertex_offsets[chart : chart + 2]
        uv = parameterize.lscm_chart(
            torch.from_numpy(concatenated[v0:v1]),
            torch.from_numpy(local_faces[f0:f1]),
            torch.from_numpy(local_neighbors[f0:f1]),
            pin_positions=uv0[v0:v1],
        )
        solved[int(chart)] = uv.detach().cpu().numpy().astype(np.float32)
    uv0 = uv0.astype(np.float32)
    chart_uvs: list[Any] = []
    for chart in range(count):
        v0, v1 = vertex_offsets[chart : chart + 2]
        uv = solved.get(chart) if not ortho_ok[chart] else None
        if uv is not None and np.all(np.isfinite(uv)) and uv.size:
            bounds = uv.max(0) - uv.min(0)
            if max(float(bounds.max()), 1e-12) / max(float(bounds.min()), 1e-12) <= 100:
                chart_uvs.append(uv)
                continue
        chart_uvs.append(uv0[v0:v1] if valid[chart] else np.zeros((v1 - v0, 2), dtype=np.float32))
    uvs = np.concatenate(chart_uvs)
    a, b, c = uvs[global_faces[:, 0]], uvs[global_faces[:, 1]], uvs[global_faces[:, 2]]
    triangle_areas = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    uv_areas = np.bincount(
        sorted_charts, weights=triangle_areas.astype(np.float64), minlength=count
    )
    areas_np = areas.numpy().astype(np.float64)
    target = float(resolution) if resolution > 0 else 1024.0
    density = math.sqrt(target * target * 0.62 / (float(areas_np.sum()) or 1.0))

    def pack_progress(done: int, total: int) -> None:
        report_progress(650 + 340 * done // max(total, 1), 1000)

    placement = pack.pack_bitmap_concat(
        uvs,
        vertex_offsets,
        local_faces,
        face_offsets,
        areas_np,
        uv_areas,
        texels_per_unit=density,
        padding_texels=padding,
        progress_callback=pack_progress,
    )
    output_faces = np.zeros((mesh.faces.shape[0], 3), dtype=np.int64)
    output_faces[order] = global_faces
    mapping = used if original is None else original[used]
    return mapping, output_faces, pack.apply_placements_concat(uvs, vertex_offsets, *placement)


class UnwrapMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="UnwrapMesh",
            display_name="Unwrap Mesh UVs",
            category="3d/texturing",
            description=(
                "Generates a UV atlas: segments the surface into charts, parameterizes each, "
                "packs into a [0,1] atlas. Vertices on chart seams are duplicated (one copy "
                "per chart, same position, own UV), so the output mesh has more vertices "
                "than the input. Run after Remesh/Decimate, before texture baking."
            ),
            inputs=(
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                InputSpec(
                    "segmenter",
                    TypeExpr.concrete(CORE_COMBO),
                    default="pec",
                    widget=ComboWidget(("pec", "adaptive")),
                    doc="pec: fast parallel-edge-collapse charting on GPU. adaptive: CPU, slower.",
                ),
                InputSpec(
                    "resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=0, max=8192, step=256),
                    doc="Target atlas resolution for texel-density auto-scale "
                    "(0 = fit-to-content).",
                ),
                InputSpec(
                    "padding",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=0, max=16),
                    doc="Texel padding between charts.",
                ),
                InputSpec(
                    "weld_distance",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.0001),
                    doc="Coincident-vert merge radius as a fraction of mesh extent (0 = auto). "
                    "Raise to ~0.001 if you get per-triangle charts (unwelded input).",
                ),
            ),
            outputs=(OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, mesh: object, segmenter: str, resolution: int, padding: int, weld_distance: float
    ) -> Mapping[str, object]:
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        compute = management.get_torch_device()
        device = compute if segmenter == "pec" else torch.device("cpu")
        source = materialize_mesh(mesh)
        face_count = (
            int(source.face_counts.sum())
            if source.face_counts is not None
            else source.faces.numel() // 3
        )
        management.free_memory(
            face_count * 14 * 1024 + management.minimum_inference_memory(), compute
        )
        items: dict[str, list[Any]] = {"vertices": [], "faces": [], "uvs": []}
        if source.vertex_colors is not None:
            items["vertex_colors"] = []
        for index in range(source.vertices.shape[0]):
            nv = (
                int(source.vertex_counts[index])
                if source.vertex_counts is not None
                else source.vertices.shape[1]
            )
            nf = (
                int(source.face_counts[index])
                if source.face_counts is not None
                else source.faces.shape[1]
            )
            vertices, faces = source.vertices[index, :nv], source.faces[index, :nf]
            vertices_np = vertices.detach().cpu().numpy().astype(np.float32)
            extent = float(np.linalg.norm(vertices_np.max(0) - vertices_np.min(0))) if nv else 0.0
            distance = weld_distance * extent if weld_distance > 0 else 0.0
            mapping, indices, uvs = _unwrap(
                vertices.to(device).float(),
                faces.to(device).long(),
                segmenter,
                int(resolution),
                int(padding),
                distance,
                compute,
            )
            uvs = uvs.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]
            items["vertices"].append(torch.from_numpy(vertices_np[mapping]).to(vertices.device))
            items["faces"].append(
                torch.from_numpy(indices).to(device=vertices.device, dtype=torch.int32)
            )
            items["uvs"].append(torch.from_numpy(uvs.astype(np.float32)).to(vertices.device))
            if source.vertex_colors is not None:
                colors = source.vertex_colors[index, :nv].detach().cpu().numpy()
                items["vertex_colors"].append(
                    torch.from_numpy(np.ascontiguousarray(colors[mapping])).to(vertices.device)
                )
        packed: dict[str, Any] = {}
        for name, values in items.items():
            maximum = max(value.shape[0] for value in values)
            output = values[0].new_zeros((len(values), maximum, values[0].shape[1]))
            for index, value in enumerate(values):
                output[index, : value.shape[0]] = value
            packed[name] = output
        if source.vertices.shape[0] > 1:
            packed["vertex_counts"] = torch.tensor(
                [v.shape[0] for v in items["vertices"]],
                device=source.vertices.device,
                dtype=torch.int64,
            )
            packed["face_counts"] = torch.tensor(
                [f.shape[0] for f in items["faces"]],
                device=source.vertices.device,
                dtype=torch.int64,
            )
        report_progress(
            1,
            1,
            text=f"UV: {items['vertices'][0].shape[0]} verts / "
            f"{items['faces'][0].shape[0]} faces - atlas ~{resolution}px",
        )
        return cls.outputs(mesh=TriangleMeshBatch(**packed, texture=source.texture))
