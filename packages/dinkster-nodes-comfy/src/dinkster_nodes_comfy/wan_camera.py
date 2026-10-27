"""Native Wan camera trajectory embeddings."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    WAN_CAMERA_EMBEDDING_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_MOTIONS = {
    "Static": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "Pan Up": ((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "Pan Down": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "Pan Left": ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    "Pan Right": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "Zoom In": ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
    "Zoom Out": ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
    "Anti Clockwise (ACW)": ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),
    "ClockWise (CW)": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
}


def _trajectory_rays(
    pose: str,
    width: int,
    height: int,
    length: int,
    speed: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Any:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    angle, translation = (np.array(values) for values in _MOTIONS[pose])
    absolute: list[Any] = []
    for index in range(length):
        x, y, z = (index / length) * speed * (np.pi / 3) * angle
        rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        matrix = np.eye(4)
        matrix[:3, :3] = np.dot(rz, np.dot(ry, rx))
        matrix[:3, 3] = (index / length) * speed * 1.5 * translation
        absolute.append(matrix)
    origin = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    relative = origin @ np.linalg.inv(absolute[0])
    transforms = np.array(
        [origin, *(relative @ matrix for matrix in absolute[1:])], dtype=np.float32
    )
    pose_ratio = 1280 / 720
    if pose_ratio > width / height:
        fx = height * pose_ratio * fx / width
    else:
        fy = width / pose_ratio * fy / height
    intrinsics = np.asarray(
        [[fx * width, fy * height, cx * width, cy * height]] * length, dtype=np.float32
    )
    k = torch.as_tensor(intrinsics)[None]
    c2w = torch.as_tensor(transforms)[None]
    j, i = torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=c2w.dtype),
        torch.linspace(0, width - 1, width, dtype=c2w.dtype),
        indexing="ij",
    )
    i = i.reshape(1, 1, height * width) + 0.5
    j = j.reshape(1, 1, height * width) + 0.5
    focal_x, focal_y, center_x, center_y = k.chunk(4, dim=-1)
    xs, ys = (i - center_x) / focal_x, (j - center_y) / focal_y
    directions = torch.stack((xs, ys, torch.ones_like(ys)), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays = directions @ c2w[..., :3, :3].transpose(-1, -2)
    origins = c2w[..., :3, 3][:, :, None].expand_as(rays)
    embedding = torch.cat([torch.cross(origins, rays, dim=-1), rays], dim=-1)
    return embedding.reshape(length, height, width, 6)


class WanCameraEmbedding(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanCameraEmbedding",
            category="model/conditioning/wan/camera",
            inputs=(
                InputSpec(
                    "camera_pose",
                    TypeExpr.concrete(CORE_COMBO),
                    default="Static",
                    widget=ComboWidget(tuple(_MOTIONS)),
                ),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=832,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=480,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "length",
                    TypeExpr.concrete(CORE_INT),
                    default=81,
                    widget=NumberWidget(min=1, max=16384, step=4),
                ),
                InputSpec(
                    "speed",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    required=False,
                    widget=NumberWidget(min=0, max=10.0, step=0.1),
                ),
                InputSpec(
                    "fx",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.5,
                    required=False,
                    advanced=True,
                    widget=NumberWidget(min=0, max=1, step=0.000000001),
                ),
                InputSpec(
                    "fy",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.5,
                    required=False,
                    advanced=True,
                    widget=NumberWidget(min=0, max=1, step=0.000000001),
                ),
                InputSpec(
                    "cx",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.5,
                    required=False,
                    advanced=True,
                    widget=NumberWidget(min=0, max=1, step=0.01),
                ),
                InputSpec(
                    "cy",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.5,
                    required=False,
                    advanced=True,
                    widget=NumberWidget(min=0, max=1, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec(
                    "camera_embedding",
                    TypeExpr.concrete(WAN_CAMERA_EMBEDDING_TYPE),
                    display_name="camera_embedding",
                ),
                OutputSpec("width", TypeExpr.concrete(CORE_INT), display_name="width"),
                OutputSpec("height", TypeExpr.concrete(CORE_INT), display_name="height"),
                OutputSpec("length", TypeExpr.concrete(CORE_INT), display_name="length"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        camera_pose: str,
        width: int,
        height: int,
        length: int,
        speed: float = 1.0,
        fx: float = 0.5,
        fy: float = 0.5,
        cx: float = 0.5,
        cy: float = 0.5,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        video = _trajectory_rays(camera_pose, width, height, length, speed, fx, fy, cx, cy)
        video = video.permute(3, 0, 1, 2).unsqueeze(0).to(management.intermediate_device())
        video = torch.cat(
            [torch.repeat_interleave(video[:, :, :1], repeats=4, dim=2), video[:, :, 1:]], dim=2
        ).transpose(1, 2)
        batch, frames, channels, h, w = video.shape
        video = video.contiguous().view(batch, frames // 4, 4, channels, h, w).transpose(2, 3)
        video = video.contiguous().view(batch, frames // 4, channels * 4, h, w).transpose(1, 2)
        return cls.outputs(camera_embedding=video, width=width, height=height, length=length)


WAN_CAMERA_NODES = (WanCameraEmbedding,)
