"""Construct animated GLB and BVH files from canonical body pose packets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    FILE_3D_TYPE,
    KIMODO_POSE_DATA_TYPE,
    MHR_POSE_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    AssetRef,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    digest_bytes,
    initialize_inference_process,
    materialize_vision_packet,
    produced_output_scratch,
)


class BuildPoseFile(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        floating = TypeExpr.concrete(CORE_FLOAT)
        combo = TypeExpr.concrete(CORE_COMBO)
        bone_options = (
            InputSpec(
                "bone_vis_radius_m",
                floating,
                default=0.02,
                widget=NumberWidget(min=0.005, max=0.5, step=0.005),
                advanced=True,
            ),
            InputSpec(
                "bone_vis_color",
                combo,
                default="rainbow_y",
                widget=ComboWidget(("white", "rainbow_y")),
            ),
        )
        tilts = tuple(
            InputSpec(
                f"rainbow_tilt_{axis}",
                floating,
                default=value,
                widget=NumberWidget(min=-90.0, max=90.0, step=0.5),
            )
            for axis, value in (("z", -35.0), ("x", 0.0))
        ) + (
            InputSpec(
                "person_palette_falloff",
                floating,
                default=0.6,
                widget=NumberWidget(min=0.1, max=1.0, step=0.05),
            ),
        )
        include_hands = InputSpec("include_hands", TypeExpr.concrete(CORE_BOOLEAN), default=False)
        face = InputSpec(
            "face_style",
            combo,
            default="disabled",
            widget=ComboWidget(("disabled", "full", "eyes_mouth")),
        )
        return NodeSchema(
            "BuildPoseFile",
            display_name="Create 3D Animation File",
            description="Build an animated GLB from pose data, or save a BVH mocap file.",
            search_terms=(
                "pose animation",
                "mocap",
                "glb",
                "bvh",
                "build pose file",
                "save pose file",
            ),
            category="3d",
            inputs=(
                InputSpec("pose_data", TypeExpr.union(MHR_POSE_DATA_TYPE, KIMODO_POSE_DATA_TYPE)),
                InputSpec(
                    "sam3d_body_model", TypeExpr.concrete(SAM3D_BODY_MODEL_TYPE), required=False
                ),
                InputSpec(
                    "fps", floating, default=24.0, widget=NumberWidget(min=1.0, max=240.0, step=1.0)
                ),
                InputSpec(
                    "camera_translation",
                    combo,
                    default="off",
                    widget=ComboWidget(("off", "centered", "absolute")),
                ),
                InputSpec(
                    "track_index",
                    TypeExpr.concrete(CORE_INT),
                    default=-1,
                    widget=NumberWidget(min=-1, max=15),
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "format",
                    (
                        DynamicComboOption(
                            "glb",
                            (
                                DynamicComboSpec(
                                    "mesh_style",
                                    (
                                        DynamicComboOption(
                                            "body_mesh",
                                            (
                                                DynamicComboSpec(
                                                    "bone_vis",
                                                    (
                                                        DynamicComboOption("off"),
                                                        DynamicComboOption(
                                                            "octahedrons", bone_options
                                                        ),
                                                    ),
                                                ),
                                                DynamicComboSpec(
                                                    "shader",
                                                    tuple(
                                                        DynamicComboOption(
                                                            name, () if name == "default" else tilts
                                                        )
                                                        for name in (
                                                            "default",
                                                            "rainbow",
                                                            "rainbow_face_normal",
                                                            "rainbow_face_semantic",
                                                        )
                                                    ),
                                                ),
                                            ),
                                        ),
                                        DynamicComboOption(
                                            "bones_only",
                                            (
                                                DynamicComboSpec(
                                                    "bone_vis",
                                                    (
                                                        DynamicComboOption(
                                                            "octahedrons", bone_options
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        DynamicComboOption(
                                            "openpose",
                                            (
                                                InputSpec(
                                                    "marker_radius_m",
                                                    floating,
                                                    default=0.010,
                                                    widget=NumberWidget(
                                                        min=0.005, max=0.1, step=0.001
                                                    ),
                                                    advanced=True,
                                                ),
                                                InputSpec(
                                                    "stick_radius_m",
                                                    floating,
                                                    default=0.008,
                                                    widget=NumberWidget(
                                                        min=0.002, max=0.05, step=0.001
                                                    ),
                                                    advanced=True,
                                                ),
                                                include_hands,
                                                InputSpec(
                                                    "hand_marker_radius_m",
                                                    floating,
                                                    default=0.005,
                                                    widget=NumberWidget(
                                                        min=0.001, max=0.1, step=0.001
                                                    ),
                                                    advanced=True,
                                                ),
                                                InputSpec(
                                                    "hand_stick_radius_m",
                                                    floating,
                                                    default=0.003,
                                                    widget=NumberWidget(
                                                        min=0.001, max=0.05, step=0.001
                                                    ),
                                                    advanced=True,
                                                ),
                                                face,
                                                InputSpec(
                                                    "face_marker_radius_m",
                                                    floating,
                                                    default=0.0,
                                                    widget=NumberWidget(
                                                        min=0.0, max=0.05, step=0.0005
                                                    ),
                                                    advanced=True,
                                                ),
                                            ),
                                        ),
                                        DynamicComboOption(
                                            "scail",
                                            (
                                                InputSpec(
                                                    "stick_radius_m",
                                                    floating,
                                                    default=0.022,
                                                    widget=NumberWidget(
                                                        min=0.002, max=0.1, step=0.001
                                                    ),
                                                    advanced=True,
                                                ),
                                                InputSpec(
                                                    "marker_radius_m",
                                                    floating,
                                                    default=0.0,
                                                    widget=NumberWidget(
                                                        min=0.0, max=0.1, step=0.001
                                                    ),
                                                    advanced=True,
                                                ),
                                                InputSpec(
                                                    "material_roughness",
                                                    floating,
                                                    default=0.3,
                                                    widget=NumberWidget(
                                                        min=0.0, max=1.0, step=0.05
                                                    ),
                                                    advanced=True,
                                                ),
                                                include_hands,
                                                InputSpec(
                                                    "hand_marker_radius_m",
                                                    floating,
                                                    default=0.005,
                                                    widget=NumberWidget(
                                                        min=0.001, max=0.05, step=0.001
                                                    ),
                                                    advanced=True,
                                                ),
                                                InputSpec(
                                                    "hand_stick_radius_m",
                                                    floating,
                                                    default=0.003,
                                                    widget=NumberWidget(
                                                        min=0.001, max=0.05, step=0.001
                                                    ),
                                                    advanced=True,
                                                ),
                                                face,
                                            ),
                                        ),
                                    ),
                                ),
                                InputSpec(
                                    "bone_smooth_window",
                                    TypeExpr.concrete(CORE_INT),
                                    default=0,
                                    widget=NumberWidget(min=0, max=51, step=2),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "bvh",
                            (
                                InputSpec(
                                    "units", combo, default="cm", widget=ComboWidget(("cm", "m"))
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            input_order=(
                "pose_data",
                "format",
                "sam3d_body_model",
                "fps",
                "camera_translation",
                "track_index",
            ),
            outputs=(OutputSpec("model_3d", TypeExpr.concrete(FILE_3D_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        pose_data: Any,
        format: str = "glb",
        sam3d_body_model: Any = None,
        fps: float = 24.0,
        camera_translation: str = "off",
        track_index: int = -1,
        **inputs: Any,
    ) -> Mapping[str, object]:
        pose = materialize_vision_packet(pose_data)
        options = {key.removeprefix("format."): value for key, value in inputs.items()}
        mode = str(options.get("mesh_style", "body_mesh"))
        external = "_skeleton_override" in pose
        if format == "bvh" or mode in ("body_mesh", "bones_only"):
            if sam3d_body_model is None and not external:
                if format == "bvh":
                    raise ValueError(
                        "Create 3D Animation: 'bvh' format needs the `sam3d_body_model` input OR a "
                        "`_skeleton_override` dict in pose_data (e.g. from KimodoSample)."
                    )
                raise ValueError(
                    f"BuildPoseFile: '{mode}' mode needs the `sam3d_body_model` input OR a "
                    "`_skeleton_override` dict in pose_data. Connect the SAM3DBody model "
                    "or feed pose_data from a node that supplies the override (e.g. KimodoSample)."
                )
            if sam3d_body_model is not None and not external:
                initialize_inference_process()
                importlib.import_module("dinkster_inference.model_management").load_model_gpu(
                    sam3d_body_model
                )
        common = dict(
            fps=float(fps), camera_translation=str(camera_translation), track_index=int(track_index)
        )
        if format == "bvh":
            serializer = importlib.import_module(".sam3d_bvh", __package__)
            common["track_index"] = max(0, int(track_index))
            data = serializer.build_bvh(
                pose, sam3d_body_model, units=str(options.get("units", "cm")), **common
            )
        elif format == "glb":
            smooth = int(options.get("bone_smooth_window", 0))
            style = {key.removeprefix("mesh_style."): value for key, value in options.items()}
            if mode in ("body_mesh", "bones_only"):
                serializer = importlib.import_module(".sam3d_skeletal", __package__)
                default_shape = "off" if mode == "body_mesh" else "octahedrons"
                data = serializer.build_glb_skeletal(
                    pose,
                    sam3d_body_model,
                    **common,
                    bone_smooth_window=smooth,
                    bone_vis=str(style.get("bone_vis", default_shape)),
                    bone_vis_radius_m=float(style.get("bone_vis.bone_vis_radius_m", 0.04)),
                    bone_vis_color=str(style.get("bone_vis.bone_vis_color", "white")),
                    include_body_mesh=mode == "body_mesh",
                    shader=str(style.get("shader", "default")),
                    rainbow_tilt_x_deg=float(style.get("shader.rainbow_tilt_x", 0.0)),
                    rainbow_tilt_z_deg=float(style.get("shader.rainbow_tilt_z", 0.0)),
                    person_palette_falloff=float(style.get("shader.person_palette_falloff", 0.6)),
                )
            elif mode in ("openpose", "scail"):
                serializer = importlib.import_module(".sam3d_openpose_glb", __package__)
                scail = mode == "scail"
                radius = float(style.get("stick_radius_m", 0.022 if scail else 0.008))
                marker = float(style.get("marker_radius_m", 0.0 if scail else 0.025))
                if scail and marker <= 0.0:
                    marker = radius
                data = serializer.build_glb_openpose(
                    pose,
                    **common,
                    bone_smooth_window=smooth,
                    marker_radius_m=marker,
                    stick_radius_m=radius,
                    include_hands=bool(style.get("include_hands", False)),
                    hand_marker_radius_m=float(style.get("hand_marker_radius_m", 0.005)),
                    hand_stick_radius_m=float(style.get("hand_stick_radius_m", 0.003)),
                    face_style=str(style.get("face_style", "disabled")),
                    face_marker_radius_m=0.0
                    if scail
                    else float(style.get("face_marker_radius_m", 0.0)),
                    palette="scail" if scail else "openpose",
                    shape="capsule" if scail else "ellipsoid",
                    smooth_shade=scail,
                    material_roughness=float(style.get("material_roughness", 0.3))
                    if scail
                    else 0.85,
                    material_double_sided=scail,
                )
            else:
                raise ValueError(f"BuildPoseGLB: unknown mesh_style {mode!r}")
        else:
            raise ValueError(f"Unsupported pose file format: {format!r}")
        vault = produced_output_scratch()
        if vault is None:
            raise RuntimeError("BuildPoseFile requires a produced-output store")
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        source = AssetRef(
            digest,
            f"pose.{format}",
            len(data),
            "model/gltf-binary" if format == "glb" else "application/octet-stream",
            resolver=vault,
        )
        return cls.outputs(model_3d={"format": format, "source": source})
