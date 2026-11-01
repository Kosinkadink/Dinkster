"""Whole-body pose rasterization using batched disks and clipped Pillow polygons."""

from __future__ import annotations

import colorsys
import importlib
import math
from typing import Any

np = importlib.import_module("numpy")
Image = importlib.import_module("PIL.Image")
ImageDraw = importlib.import_module("PIL.ImageDraw")

HEAD_EDGES = [[2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
BODY_EDGES = [
    [2, 3],
    [2, 6],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [2, 9],
    [9, 10],
    [10, 11],
    [2, 12],
    [12, 13],
    [13, 14],
]
HAND_EDGES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
]
COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]


def _disk_offsets(radius: float) -> tuple[Any, Any]:
    radius_int = int(np.ceil(radius))
    yy, xx = np.mgrid[-radius_int : radius_int + 1, -radius_int : radius_int + 1]
    disk = xx * xx + yy * yy <= radius * radius
    return xx[disk], yy[disk]


def _draw_disks(
    canvas: Any,
    points: Any,
    radius: float,
    colors: Any,
    offsets: tuple[Any, Any] | None = None,
) -> None:
    if not len(points):
        return
    color_array = np.asarray(colors)
    uniform = color_array.ndim == 1
    points = np.asarray(points, dtype=np.int32)
    h, w = canvas.shape[:2]
    offset_x, offset_y = _disk_offsets(radius) if offsets is None else offsets
    chunk_size = max(1, 1_000_000 // len(offset_x))
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        xx = chunk[:, 0, None] + offset_x
        yy = chunk[:, 1, None] + offset_y
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        if uniform:
            canvas[yy[valid], xx[valid]] = colors
        else:
            values = np.broadcast_to(
                color_array[start : start + chunk_size, None, :], (*xx.shape, 3)
            )
            canvas[yy[valid], xx[valid]] = values[valid]


def _line_points(pt1: Any, pt2: Any) -> Any:
    x0, y0, x1, y1 = *pt1, *pt2
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err, x, y = dx - dy, x0, y0
    points: list[tuple[int, int]] = []
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err, x = err - dy, x + sx
        if e2 < dx:
            err, y = err + dx, y + sy
    return np.asarray(points, dtype=np.int32)


def _ellipse_polygon(center: Any, axes: Any, angle: int) -> Any:
    axes = (axes[0] + 0.5, axes[1] + 0.5)
    theta = np.deg2rad(np.minimum(np.arange(0, 364, 4), 360))
    x, y = axes[0] * np.cos(theta), axes[1] * np.sin(theta)
    angle_rad = math.radians(angle % 360)
    alpha, beta = math.cos(angle_rad), math.sin(angle_rad)
    pts = np.rint(
        np.column_stack(
            (
                center[0] + x * alpha - y * beta,
                center[1] + x * beta + y * alpha,
            )
        )
    ).astype(np.int32)
    keep = np.ones(pts.shape[0], dtype=bool)
    keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
    pts = pts[keep]
    return pts if len(pts) > 1 else np.asarray([center, center], dtype=np.int32)


def _fill_polygons(canvas: Any, polygons: list[tuple[Any, Any]]) -> None:
    if not polygons:
        return
    points = np.concatenate([polygon for polygon, _ in polygons])
    h, w = canvas.shape[:2]
    y_min, y_max = max(0, int(points[:, 1].min())), min(h, int(points[:, 1].max()) + 1)
    x_min, x_max = max(0, int(points[:, 0].min())), min(w, int(points[:, 0].max()) + 1)
    if y_max > y_min and x_max > x_min:
        roi = canvas[y_min:y_max, x_min:x_max]
        image = Image.fromarray(roi)
        draw = ImageDraw.Draw(image, "RGBA")
        offset = np.array([x_min, y_min], dtype=np.int32)
        for polygon, color in polygons:
            draw.polygon((polygon - offset).reshape(-1).tolist(), fill=tuple(color))
        roi[:] = np.asarray(image)


def draw_pose(
    canvas: Any,
    keypoints: Any,
    scores: Any,
    *,
    threshold: float,
    draw_body: bool,
    draw_head: bool,
    draw_feet: bool,
    draw_face: bool,
    draw_hands: bool,
    stick_width: int,
    face_point_size: int,
) -> Any:
    h, w = canvas.shape[:2]
    centers: list[tuple[int, int]] = []
    colors: list[list[int]] = []
    if draw_body or draw_head:
        edges = (BODY_EDGES if draw_body else []) + (HEAD_EDGES if draw_head else [])
        offset = 0 if draw_body else len(BODY_EDGES)
        polygons: list[tuple[Any, Any]] = []
        for i, (first, second) in enumerate(edges):
            a, b = first - 1, second - 1
            if scores[a] < threshold or scores[b] < threshold:
                continue
            xx = [keypoints[a][1], keypoints[b][1]]
            yy = [keypoints[a][0], keypoints[b][0]]
            mx, my = (xx[0] + xx[1]) / 2, (yy[0] + yy[1]) / 2
            length = math.sqrt((xx[0] - xx[1]) ** 2 + (yy[0] - yy[1]) ** 2)
            if length < 1:
                continue
            angle = math.degrees(math.atan2(xx[0] - xx[1], yy[0] - yy[1]))
            polygon = _ellipse_polygon(
                (int(my), int(mx)), (int(length / 2), stick_width), int(angle)
            )
            polygons.append((polygon, COLORS[(i + offset) % len(COLORS)]))
        _fill_polygons(canvas, polygons)
        centers, colors = [], []
        for i in range(18):
            head = i in {0, 14, 15, 16, 17}
            if (not draw_head and head) or (not draw_body and not head and i != 1):
                continue
            if scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < w and 0 <= y < h:
                centers.append((x, y))
                colors.append(COLORS[i % len(COLORS)])
        _draw_disks(canvas, centers, 4, colors)
    if draw_feet:
        centers, colors = [], []
        for i in range(18, 24):
            if scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < w and 0 <= y < h:
                centers.append((x, y))
                colors.append(COLORS[i % len(COLORS)])
        _draw_disks(canvas, centers, 4, colors)
    if draw_hands:
        offsets = _disk_offsets(1.5)
        for start in (92, 113):
            for ie, (first, second) in enumerate(HAND_EDGES):
                a, b = start + first, start + second
                if scores[a] < threshold or scores[b] < threshold:
                    continue
                x1, y1 = int(keypoints[a][0]), int(keypoints[a][1])
                x2, y2 = int(keypoints[b][0]), int(keypoints[b][1])
                if 0.01 < x1 < w and 0.01 < y1 < h and 0.01 < x2 < w and 0.01 < y2 < h:
                    rgb = colorsys.hsv_to_rgb(ie / float(len(HAND_EDGES)), 1.0, 1.0)
                    color = tuple(int(c * 255) for c in rgb)
                    _draw_disks(canvas, _line_points((x1, y1), (x2, y2)), 1.5, color, offsets)
            centers = []
            for i in range(start, start + 21):
                if scores[i] < threshold:
                    continue
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                if 0.01 < x < w and 0.01 < y < h:
                    centers.append((x, y))
            _draw_disks(canvas, centers, 4, (0, 0, 255))
    if draw_face:
        centers = []
        for i in range(24, 92):
            if scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0.01 < x < w and 0.01 < y < h:
                centers.append((x, y))
        _draw_disks(canvas, centers, face_point_size, (255, 255, 255))
    return canvas
