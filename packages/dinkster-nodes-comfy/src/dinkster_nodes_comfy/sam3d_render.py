"""Render MHR and external rig poses as meshes, skeletons and capsules."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    KIMODO_POSE_DATA_TYPE,
    LOAD3D_CAMERA_TYPE,
    MHR_POSE_DATA_TYPE,
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
    materialize_image,
    materialize_vision_packet,
)


class SAM3DBody_Render(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        integer, floating = TypeExpr.concrete(CORE_INT), TypeExpr.concrete(CORE_FLOAT)
        palette = InputSpec(
            "person_palette_falloff",
            floating,
            default=0.6,
            widget=NumberWidget(min=0.1, max=1.0, step=0.05),
        )
        hands = InputSpec(
            "hand_style",
            TypeExpr.concrete(CORE_COMBO),
            default="disabled",
            widget=ComboWidget(("disabled", "dwpose", "openpose")),
        )
        face = InputSpec(
            "face_style",
            TypeExpr.concrete(CORE_COMBO),
            default="disabled",
            widget=ComboWidget(("disabled", "full", "eyes_mouth")),
        )
        tilts = tuple(
            InputSpec(
                f"rainbow_tilt_{axis}",
                floating,
                default=value,
                widget=NumberWidget(min=-90.0, max=90.0, step=0.5),
            )
            for axis, value in (("z", -35.0), ("x", 0.0))
        )
        shader = DynamicComboSpec(
            "shader",
            tuple(
                DynamicComboOption(name, tilts if name.startswith("rainbow") else ())
                for name in (
                    "default",
                    "normals",
                    "rainbow",
                    "rainbow_face_normal",
                    "rainbow_face_semantic",
                    "depth",
                )
            ),
        )
        return NodeSchema(
            "SAM3DBody_Render",
            display_name="Render 3D Body Pose",
            category="image/detection",
            search_terms=("Render SAM3D Body", "sam3d render", "kimodo render"),
            inputs=(
                InputSpec("pose_data", TypeExpr.union(MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE)),
                InputSpec("background", TypeExpr.concrete(IMAGE_TYPE), required=False),
                InputSpec(
                    "width", integer, default=0, widget=NumberWidget(min=0, max=16384, step=8)
                ),
                InputSpec(
                    "height", integer, default=0, widget=NumberWidget(min=0, max=16384, step=8)
                ),
                InputSpec("camera_info", TypeExpr.concrete(LOAD3D_CAMERA_TYPE), required=False),
            ),
            combos=(
                DynamicComboSpec(
                    "render_style",
                    (
                        DynamicComboOption(
                            "mesh",
                            (
                                shader,
                                InputSpec(
                                    "opacity",
                                    floating,
                                    default=1.0,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                                ),
                                palette,
                                InputSpec(
                                    "region",
                                    TypeExpr.concrete(CORE_COMBO),
                                    default="full_body",
                                    widget=ComboWidget(("full_body", "hands_only")),
                                ),
                            ),
                        ),
                        DynamicComboOption("silhouette"),
                        DynamicComboOption(
                            "openpose_2d",
                            (
                                InputSpec(
                                    "marker_radius_px",
                                    integer,
                                    default=4,
                                    widget=NumberWidget(min=1, max=32, step=1),
                                ),
                                InputSpec(
                                    "stick_width_px",
                                    integer,
                                    default=4,
                                    widget=NumberWidget(min=1, max=32, step=1),
                                ),
                                InputSpec(
                                    "limb_alpha",
                                    floating,
                                    default=0.6,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.05),
                                ),
                                face,
                                hands,
                                palette,
                            ),
                        ),
                        DynamicComboOption(
                            "openpose_3d",
                            (
                                InputSpec(
                                    "radius_m",
                                    floating,
                                    default=0.015,
                                    widget=NumberWidget(min=0.004, max=0.1, step=0.001),
                                ),
                                InputSpec(
                                    "include_hands", TypeExpr.concrete(CORE_BOOLEAN), default=True
                                ),
                                palette,
                            ),
                        ),
                        DynamicComboOption(
                            "scail",
                            (
                                InputSpec(
                                    "radius_m",
                                    floating,
                                    default=0.022,
                                    widget=NumberWidget(min=0.005, max=0.2, step=0.001),
                                ),
                                InputSpec(
                                    "hand_style",
                                    TypeExpr.concrete(CORE_COMBO),
                                    default="dwpose",
                                    widget=ComboWidget(("disabled", "dwpose", "openpose")),
                                ),
                                face,
                                palette,
                            ),
                        ),
                    ),
                ),
            ),
            input_order=(
                "pose_data",
                "background",
                "width",
                "height",
                "camera_info",
                "render_style",
            ),
            outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        pose_data: Any,
        background: Any = None,
        width: int = 0,
        height: int = 0,
        camera_info: Any = None,
        render_style: str = "mesh",
        **inputs: Any,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        management = importlib.import_module("dinkster_inference.model_management")
        camera = importlib.import_module(".sam3d_camera", __package__)
        mesh = importlib.import_module(".sam3d_rasterizer", __package__)
        capsules = importlib.import_module(".sam3d_capsules", __package__)
        openpose = importlib.import_module(".sam3d_openpose", __package__)
        pose = materialize_vision_packet(pose_data)
        native_h, native_w = pose["image_size"]
        h, w, px_scale = native_h, native_w, 1.0
        if width or height:
            w = int(width) if width else max(1, round(native_w * height / native_h))
            h = int(height) if height else max(1, round(native_h * width / native_w))
            pose = camera.scale_pose_data(pose, h, w)
            px_scale = min(w / native_w, h / native_h)
        if camera_info is not None:
            pose = camera.apply_camera_override(pose, camera_info, h, w)
        device, dtype = management.intermediate_device(), management.intermediate_dtype()
        if not pose["frames"]:
            return cls.outputs(image=torch.zeros(1, h, w, 3, dtype=dtype, device=device))
        bg = None
        if background is not None:
            bg = materialize_image(background).to(device=device, dtype=torch.float32)
            if tuple(bg.shape[1:3]) != (h, w):
                bg = torch.nn.functional.interpolate(
                    bg.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False
                ).movedim(1, -1)
        composite = (
            "silhouette"
            if render_style == "silhouette"
            else ("over" if bg is not None else "mesh_only")
        )
        options = {key.removeprefix("render_style."): value for key, value in inputs.items()}
        falloff = float(options.get("person_palette_falloff", 0.6))
        if render_style == "mesh" and options.get("region", "full_body") == "hands_only":
            hand_mask = pose.get("hand_vert_mask")
            if hand_mask is not None:
                faces = np.asarray(pose["faces"])
                pose = dict(pose, faces=np.ascontiguousarray(faces[hand_mask[faces].all(axis=1)]))
        cache: dict[Any, Any] = {}
        images: list[Any] = []
        for index in range(len(pose["frames"])):
            common = dict(
                frame_idx=index,
                W=w,
                H=h,
                background=None if bg is None else bg[min(index, bg.shape[0] - 1)],
                composite=composite,
                person_brightness_falloff=falloff,
            )
            if render_style == "openpose_2d":
                image = openpose.render_pose_data_openpose(
                    pose,
                    **common,
                    marker_radius_px=max(
                        1, int(round(options.get("marker_radius_px", 4) * px_scale))
                    ),
                    stick_width_px=max(1, int(round(options.get("stick_width_px", 4) * px_scale))),
                    limb_alpha=float(options.get("limb_alpha", 0.6)),
                    face_style=str(options.get("face_style", "disabled")),
                    include_hands=options.get("hand_style", "disabled") != "disabled",
                    hand_color_style=str(options.get("hand_style", "dwpose")),
                )
            elif render_style in {"openpose_3d", "scail"}:
                is_openpose = render_style == "openpose_3d"
                image = capsules.render_capsules(
                    pose,
                    **common,
                    radius_m=float(options.get("radius_m", 0.015 if is_openpose else 0.030)),
                    include_hands=bool(options.get("include_hands", True))
                    if is_openpose
                    else False,
                    palette="openpose" if is_openpose else "scail",
                    flat_shade=is_openpose,
                )
                if not is_openpose and (
                    options.get("hand_style", "disabled") != "disabled"
                    or options.get("face_style", "disabled") != "disabled"
                ):
                    overlay_px = max(1, int(round(4 * px_scale)))
                    image = openpose.render_pose_data_openpose(
                        pose,
                        frame_idx=index,
                        W=w,
                        H=h,
                        background=image,
                        composite="over",
                        include_body=False,
                        include_hands=options.get("hand_style", "disabled") != "disabled",
                        face_style=str(options.get("face_style", "disabled")),
                        marker_radius_px=overlay_px,
                        stick_width_px=overlay_px,
                        face_marker_radius_px=max(1, int(round(px_scale))),
                        hand_color_style=str(options.get("hand_style", "dwpose")),
                        person_brightness_falloff=falloff,
                    )
            elif render_style in {"mesh", "silhouette"}:
                image = mesh.render_mesh(
                    pose,
                    **common,
                    cache=cache,
                    opacity=float(options.get("opacity", 1.0)),
                    shader_preset=str(options.get("shader", "default")),
                    rainbow_tilt_x_deg=float(options.get("shader.rainbow_tilt_x", 0.0)),
                    rainbow_tilt_z_deg=float(options.get("shader.rainbow_tilt_z", -35.0)),
                )
            else:
                raise ValueError(f"Unknown SAM3D render style: {render_style!r}")
            images.append(image.to(device=device, dtype=dtype))
        return cls.outputs(image=torch.stack(images, dim=0))
