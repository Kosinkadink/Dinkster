"""Native world-space camera construction."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CAMERA_INFO_STATE_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    LOAD3D_CAMERA_TYPE,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


def _number(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    *,
    doc: str = "",
    advanced: bool = False,
) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_FLOAT),
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
        doc=doc,
        advanced=advanced,
    )


def matrix_quaternion(torch: Any, matrix: Any) -> Any:
    # Pick the largest quaternion component to avoid unstable divisions near half turns.
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    m21, m12 = matrix[..., 2, 1], matrix[..., 1, 2]
    m02, m20 = matrix[..., 0, 2], matrix[..., 2, 0]
    m10, m01 = matrix[..., 1, 0], matrix[..., 0, 1]
    q2 = torch.stack(
        [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1
    )
    candidates = torch.stack(
        [
            torch.stack([q2[..., 0], m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, q2[..., 1], m10 + m01, m02 + m20], -1),
            torch.stack([m02 - m20, m10 + m01, q2[..., 2], m12 + m21], -1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, q2[..., 3]], -1),
        ],
        -2,
    )
    selected = q2.argmax(-1)
    quaternion = torch.gather(
        candidates, -2, selected[..., None, None].expand(selected.shape + (1, 4))
    )[..., 0, :]
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _camera_info(
    torch: Any,
    mode: Mapping[str, Any],
    target: list[float],
    roll: float,
    fov: float,
    zoom: float,
    camera_type: str,
    device: Any,
) -> dict[str, object]:
    kind = mode["mode"]
    if kind == "orbit":
        yaw, pitch = math.radians(mode["yaw"]), math.radians(mode["pitch"])
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        distance = mode["distance"]
        position = [
            target[0] + distance * cp * sy,
            target[1] + distance * sp,
            target[2] + distance * cp * cy,
        ]
    else:
        position = [mode["position_x"], mode["position_y"], mode["position_z"]]
    pos = torch.as_tensor(position, dtype=torch.float32, device=device)
    if kind == "quaternion":
        qx, qy, qz, qw = (float(mode[f"quat_{axis}"]) for axis in "xyzw")
        quaternion = torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device=device)
        quaternion = quaternion / quaternion.norm().clamp_min(1e-8)
        q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w, x, y, z = q.unbind(-1)
        matrix = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        ).reshape(3, 3)
        tgt = pos - matrix[:, 2]
    else:
        tgt = torch.as_tensor(target, dtype=torch.float32, device=device)
        z_axis = pos - tgt
        z_axis = z_axis / z_axis.norm().clamp_min(1e-8)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        if z_axis.dot(up).abs() > 0.999:
            up = torch.tensor([0.0, 0.0, 1.0], device=device)
        x_axis = torch.linalg.cross(up, z_axis)
        x_axis = x_axis / x_axis.norm().clamp_min(1e-8)
        y_axis = torch.linalg.cross(z_axis, x_axis)
        matrix = torch.stack([x_axis, y_axis, z_axis], dim=1)
        quaternion = matrix_quaternion(torch, matrix[None])[0]
        if roll:
            angle = math.radians(roll)
            qz = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], device=device)
            aw, ax, ay, az = quaternion.unbind(-1)
            bw, bx, by, bz = qz.unbind(-1)
            quaternion = torch.stack(
                [
                    aw * bw - ax * bx - ay * by - az * bz,
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                ],
                dim=-1,
            )
    return {
        "position": {axis: float(pos[index]) for index, axis in enumerate("xyz")},
        "target": {axis: float(tgt[index]) for index, axis in enumerate("xyz")},
        "quaternion": {axis: float(quaternion[index]) for index, axis in enumerate("wxyz")},
        **({"useCustomUp": True} if kind == "quaternion" or roll else {}),
        "fov": float(fov),
        "cameraType": str(camera_type),
        "zoom": float(zoom),
    }


class CreateCameraInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        position_inputs = (
            _number(
                "position_x",
                4.0,
                -1000.0,
                1000.0,
                0.01,
                doc="Camera position in world space (right-handed, Y-up).",
            ),
            _number("position_y", 4.0, -1000.0, 1000.0, 0.01),
            _number("position_z", 4.0, -1000.0, 1000.0, 0.01),
        )
        return NodeSchema(
            node_type="CreateCameraInfo",
            display_name="Create Camera Info",
            search_terms=("camera position", "make camera info", "orbit camera", "look at camera"),
            category="3d",
            description=(
                "Build a camera_info. Mode 'orbit' aims with yaw/pitch/distance around the "
                "target; 'look_at' places the camera at world position. Coordinates are the "
                "viewer's world space (right-handed,Y-up)."
            ),
            combos=(
                DynamicComboSpec(
                    "mode",
                    (
                        DynamicComboOption(
                            "orbit",
                            (
                                _number("yaw", 35.0, -360.0, 360.0, 1.0),
                                _number("pitch", 30.0, -89.0, 89.0, 1.0),
                                _number(
                                    "distance",
                                    4.0,
                                    0.01,
                                    1000.0,
                                    0.01,
                                    doc="Camera distance from the target.",
                                ),
                            ),
                        ),
                        DynamicComboOption("look_at", position_inputs),
                        DynamicComboOption(
                            "quaternion",
                            (
                                *position_inputs,
                                _number("quat_x", 0.0, -1.0, 1.0, 0.001),
                                _number("quat_y", 0.0, -1.0, 1.0, 0.001),
                                _number("quat_z", 0.0, -1.0, 1.0, 0.001),
                                _number(
                                    "quat_w",
                                    1.0,
                                    -1.0,
                                    1.0,
                                    0.001,
                                    doc=(
                                        "Camera world-rotation quaternion (three.js: looks down "
                                        "local -Z). Normalized for you."
                                    ),
                                ),
                            ),
                        ),
                    ),
                    doc=(
                        "How to define the camera: orbit angles, an explicit position, "
                        "or a position + quaternion."
                    ),
                ),
            ),
            inputs=(
                _number(
                    "target_x",
                    0.0,
                    -1000.0,
                    1000.0,
                    0.01,
                    advanced=True,
                    doc=(
                        "Look-at point (orbit pivot / aim). In orbit mode, move it to "
                        "pan/translate the whole camera. Ignored in quaternion mode. "
                        "Defaults to the origin."
                    ),
                ),
                _number("target_y", 0.0, -1000.0, 1000.0, 0.01, advanced=True),
                _number("target_z", 0.0, -1000.0, 1000.0, 0.01, advanced=True),
                _number(
                    "roll", 0.0, -180.0, 180.0, 1.0, doc="Camera roll about the view axis, degrees."
                ),
                _number("fov", 35.0, 1.0, 120.0, 1.0, doc="Vertical field of view in degrees."),
                _number(
                    "zoom",
                    1.0,
                    0.01,
                    100.0,
                    0.01,
                    doc=(
                        "Digital zoom (focal-length multiplier). "
                        ">1 zooms in without moving the camera."
                    ),
                ),
                InputSpec(
                    "camera_type",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(("perspective", "orthographic")),
                    doc=(
                        "Projection used by Render Splat: perspective (foreshortening) "
                        "or orthographic (parallel)."
                    ),
                ),
                InputSpec(
                    "camera_info_state",
                    TypeExpr.concrete(CAMERA_INFO_STATE_TYPE),
                    required=False,
                    socketless=True,
                ),
            ),
            input_order=(
                "mode",
                "target_x",
                "target_y",
                "target_z",
                "roll",
                "fov",
                "zoom",
                "camera_type",
                "camera_info_state",
            ),
            outputs=(
                OutputSpec(
                    "camera_info", TypeExpr.concrete(LOAD3D_CAMERA_TYPE), display_name="camera_info"
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        mode: str | Mapping[str, Any],
        target_x: float,
        target_y: float,
        target_z: float,
        roll: float,
        fov: float,
        zoom: float = 1.0,
        camera_type: str = "perspective",
        camera_info_state: object = None,
        **values: Any,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        if isinstance(mode, str):
            mode = {
                "mode": mode,
                **{key.removeprefix("mode."): value for key, value in values.items()},
            }
        return cls.outputs(
            camera_info=_camera_info(
                torch,
                mode,
                [target_x, target_y, target_z],
                roll,
                fov,
                zoom,
                camera_type,
                management.get_torch_device(),
            )
        )
