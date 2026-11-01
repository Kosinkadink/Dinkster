"""SAM3D rasterized pixels and source-preserving overlays."""

import asyncio
import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    KIMODO_POSE_DATA_TYPE,
    MHR_POSE_DATA_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_comfy.sam3d_openpose import render_pose_data_openpose
from dinkster_nodes_comfy.sam3d_render import SAM3DBody_Render

from dinkster.compose import PackSpec, ServingComposer


def _render_packet() -> dict[str, Any]:
    keypoints = np.full((70, 3), np.nan, np.float32)
    keypoints[69], keypoints[6] = (-0.5, 0, 0), (0.5, 0, 0)
    points_2d = np.full((70, 2), np.nan, np.float32)
    points_2d[69], points_2d[6] = (8, 12), (24, 12)
    return {
        "image_size": (24, 32),
        "frames": [
            [
                {
                    "pred_vertices": np.array(
                        [[-0.8, -0.6, 0], [0.8, -0.6, 0], [0, 0.8, 0]], np.float32
                    ),
                    "pred_cam_t": np.array([0, 0, 4], np.float32),
                    "focal_length": np.array(40, np.float32),
                    "pred_keypoints_2d": points_2d,
                    "pred_keypoints_3d": keypoints,
                    "bbox": np.array([2, 3, 29, 22], np.float32),
                }
            ],
            [],
            [],
        ],
        "faces": np.array([[0, 1, 2]], np.uint32),
        "hand_vert_mask": np.array([True, False, True]),
    }


def _assert_rendered(styles: dict[str, Any]) -> None:
    images = {name: materialize_image(value) for name, value in styles.items()}
    for image in images.values():
        assert image.shape == (3, 24, 32, 3)
        assert image.dtype == torch.float32 and image.device.type == "cpu"
        assert torch.count_nonzero(image[1:]) == 0
        assert image[0, 0, 0].tolist() == [0, 0, 0]
    assert images["silhouette"][0, 12, 16].tolist() == [1, 1, 1]
    assert set(images["silhouette"].unique().tolist()) == {0, 1}
    assert images["mesh"][0, 12, 16].tolist() == [0.5, 0.5, 0]
    torch.testing.assert_close(
        images["openpose_2d"][0, 12, 16], torch.tensor([128, 0, 0]) / 255, rtol=0, atol=0
    )
    assert images["openpose_3d"][0, 12, 16].tolist() == [1, 0, 0]
    red = images["scail"][0, 12, 16]
    assert red[0] == 1 and 0 < red[1] == red[2] < 0.5


def test_nonempty_render_modes_preserve_complete_source() -> None:
    packet = _render_packet()
    original = copy.deepcopy(packet)
    images = {
        style: SAM3DBody_Render.execute(
            pose_data=packet,
            render_style=style,
            **{
                "render_style.shader": "normals",
                "render_style.limb_alpha": 0.5,
                "render_style.radius_m": 0.08,
            },
        )["image"]
        for style in ("mesh", "silhouette", "openpose_2d", "openpose_3d", "scail")
    }
    _assert_rendered(images)
    np.testing.assert_equal(packet, original)
    hands = SAM3DBody_Render.execute(pose_data=packet, **{"render_style.region": "hands_only"})[
        "image"
    ]
    assert torch.count_nonzero(hands) == 0
    np.testing.assert_equal(packet, original)


def _external_packet() -> dict[str, Any]:
    packet = _render_packet()
    mapping = np.full((18, 2), -1, np.int64)
    mapping[:, 0] = 2
    mapping[1, 0], mapping[2, 0] = 0, 1
    packet["_skeleton_override"] = {"openpose18_joint_indices": mapping}
    person = packet["frames"][0][0]
    person["pred_joint_coords"] = np.array([[-0.5, 0, 0], [0.5, 0, 0], [np.nan] * 3], np.float32)
    del person["pred_keypoints_2d"], person["pred_keypoints_3d"]
    return packet


def test_external_rig_letterbox_and_projected_face_preserve_source() -> None:
    packet = _external_packet()
    original = copy.deepcopy(packet)
    image = SAM3DBody_Render.execute(
        pose_data=packet,
        width=64,
        height=24,
        render_style="openpose_2d",
        **{"render_style.marker_radius_px": 1, "render_style.stick_width_px": 1},
    )["image"]
    assert image.shape == (3, 24, 64, 3)
    torch.testing.assert_close(image[0, 12, 27], torch.tensor([255, 85, 0]) / 255, rtol=0, atol=0)
    assert torch.count_nonzero(image[:, :, :24]) == 0
    np.testing.assert_equal(packet, original)

    packet = _render_packet()
    person = packet["frames"][0][0]
    person["pred_vertices"] = np.array([[0, 0, 0]], np.float32)
    del person["pred_keypoints_2d"], person["pred_keypoints_3d"]
    packet["canonical_colors"] = {"positions": np.array([[0, 1.66, 0.1]], np.float32)}
    original = copy.deepcopy(packet)
    image = SAM3DBody_Render.execute(
        pose_data=packet,
        render_style="openpose_2d",
        **{"render_style.face_style": "eyes_mouth"},
    )["image"]
    assert image[0, 12, 16].tolist() == [1, 1, 1]
    assert image[0, 12, 20].tolist() == [0, 0, 0]
    np.testing.assert_equal(packet, original)


def test_camera_zoom_reprojects_keypoints_without_changing_sources() -> None:
    packet = _render_packet()
    del packet["frames"][0][0]["pred_vertices"]
    camera = {"position": {"x": 0, "y": 0, "z": 4}, "zoom": 2, "fov": 90}
    original, original_camera = copy.deepcopy(packet), copy.deepcopy(camera)
    image = SAM3DBody_Render.execute(
        pose_data=packet,
        camera_info=camera,
        render_style="openpose_2d",
        **{"render_style.marker_radius_px": 1, "render_style.stick_width_px": 1},
    )["image"]
    torch.testing.assert_close(image[0, 12, 13], torch.tensor([255, 85, 0]) / 255, rtol=0, atol=0)
    torch.testing.assert_close(image[0, 12, 19], torch.tensor([255, 170, 0]) / 255, rtol=0, atol=0)
    assert image[0, 12, 8].tolist() == [0, 0, 0]
    assert image[0, 12, 24].tolist() == [0, 0, 0]
    np.testing.assert_equal(packet, original)
    assert camera == original_camera


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "render-fixture.source",
            inputs=(),
            outputs=(
                OutputSpec("mhr", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),
                OutputSpec("kimodo", TypeExpr.concrete(KIMODO_POSE_DATA_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(mhr=_render_packet(), kimodo=_external_packet())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "render-fixture.inspect",
            inputs=tuple(
                InputSpec(style, TypeExpr.concrete(IMAGE_TYPE))
                for style in ("mesh", "silhouette", "openpose_2d", "openpose_3d", "scail")
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **images: Any) -> Any:
        _assert_rendered(images)
        return cls.outputs(result="render pixels preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_render_modes_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    module = f"render_fixture_{int(in_process)}"
    (tmp_path / f"{module}.py").write_text(
        "from test_sam3d_render_torch import Source, Inspect\n"
        "from dinkster_api.v1 import register_inference_types\n"
        "NODES = (Source, Inspect)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="render-fixture"\npublisher="dinkster"\nnamespaces=["render-fixture"]\n'
        f'[pack.entry]\nnodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="render", group_manifests=(native, fixture)
                    )
                )
            nodes = {"source": GraphNode("render-fixture.source")}
            for style in ("mesh", "silhouette", "openpose_2d", "openpose_3d", "scail"):
                variants = {"render_style": style}
                inputs: dict[str, Any] = {
                    "pose_data": Link("source", "kimodo" if style == "openpose_3d" else "mhr")
                }
                if style == "mesh":
                    variants["render_style.shader"] = "normals"
                elif style == "openpose_2d":
                    inputs["render_style.limb_alpha"] = 0.5
                elif style in {"openpose_3d", "scail"}:
                    inputs["render_style.radius_m"] = 0.08
                nodes[style] = GraphNode("SAM3DBody_Render", inputs, slot_variants=variants)
            nodes["inspect"] = GraphNode(
                "render-fixture.inspect",
                {style: Link(style, "image") for style in nodes if style != "source"},
            )
            graph = graph_from_wire(graph_to_wire(Graph(nodes=nodes)))
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == "render pixels preserved"
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_body_alpha_marker_radius_and_source_preservation() -> None:
    keypoints = np.full((70, 2), np.nan, dtype=np.float32)
    keypoints[69] = (8, 14)
    keypoints[6] = (30, 14)
    pose = {"frames": [[{"pred_keypoints_2d": keypoints}]]}
    before = keypoints.copy()
    background = torch.full((28, 40, 3), 64 / 255)
    original = background.clone()
    image = render_pose_data_openpose(
        pose,
        frame_idx=0,
        W=40,
        H=28,
        background=background,
        marker_radius_px=1,
        stick_width_px=2,
        limb_alpha=0.5,
    )
    torch.testing.assert_close(image[14, 19], torch.tensor([160, 32, 32]) / 255, rtol=0, atol=0)
    torch.testing.assert_close(image[14, 8], torch.tensor([255, 85, 0]) / 255, rtol=0, atol=0)
    torch.testing.assert_close(image[10, 8], background[10, 8], rtol=0, atol=0)
    np.testing.assert_array_equal(keypoints, before)
    torch.testing.assert_close(background, original, rtol=0, atol=0)


def test_hand_palette_one_pixel_line_and_marker_size() -> None:
    keypoints = np.full((70, 2), np.nan, dtype=np.float32)
    keypoints[41], keypoints[24] = (7, 12), (23, 12)
    pose = {"frames": [[{"pred_keypoints_2d": keypoints}]]}
    image = render_pose_data_openpose(
        pose,
        frame_idx=0,
        W=32,
        H=24,
        include_body=False,
        include_hands=True,
        hand_color_style="openpose",
        marker_radius_px=2,
        stick_width_px=1,
    )
    torch.testing.assert_close(image[12, 7], torch.tensor([100, 100, 100]) / 255, rtol=0, atol=0)
    torch.testing.assert_close(image[12, 23], torch.tensor([100, 0, 0]) / 255, rtol=0, atol=0)
    assert image[12, 15].tolist() == [1, 0, 0]
    assert image[13, 15].tolist() == [0, 0, 0]
    assert image[14, 7].tolist() == [0, 0, 0]


def test_face_rounding_clipped_disks_and_second_person_pastel() -> None:
    first = {"pred_face_keypoints_2d": np.array([[-0.6, 4.4], [np.nan, 3]], np.float32)}
    keypoints = np.full((70, 2), np.nan, dtype=np.float32)
    keypoints[6] = (17, 11)
    second = {"pred_keypoints_2d": keypoints}
    pose = {"frames": [[first, second], []]}
    original = copy.deepcopy(pose)
    image = render_pose_data_openpose(
        pose,
        frame_idx=0,
        W=24,
        H=20,
        face_style="full",
        face_marker_radius_px=1,
        marker_radius_px=1,
        person_brightness_falloff=0.5,
    )
    assert image[4, 0].tolist() == [1, 1, 1]
    assert image[4, 1].tolist() == [0, 0, 0]
    torch.testing.assert_close(image[11, 17], torch.tensor([255, 212, 127]) / 255, rtol=0, atol=0)
    empty = render_pose_data_openpose(pose, frame_idx=1, W=24, H=20)
    assert torch.count_nonzero(empty) == 0
    for person, before in zip(pose["frames"][0], original["frames"][0], strict=True):
        for key in person:
            np.testing.assert_array_equal(person[key], before[key])


def test_render_silhouette_letterbox_background_tail_and_source() -> None:
    vertices = np.array([[-0.8, -0.6, 0], [0.8, -0.6, 0], [0, 0.8, 0]], np.float32)
    pose = {
        "image_size": (24, 32),
        "frames": [
            [
                {
                    "pred_vertices": vertices,
                    "pred_cam_t": np.array([0, 0, 4], np.float32),
                    "focal_length": np.array(40, np.float32),
                }
            ],
            [],
            [],
        ],
        "faces": np.array([[0, 1, 2]], np.uint32),
    }
    original = copy.deepcopy(pose)
    background = torch.tensor([0.2, 0.4, 0.8])[None, None, None].expand(1, 12, 16, 3).clone()
    image = SAM3DBody_Render.execute(pose_data=pose, render_style="silhouette")["image"]
    assert image[0, 12, 16].tolist() == [1, 1, 1]
    assert image[0, 0, 0].tolist() == [0, 0, 0]
    assert set(image.unique().tolist()) == {0.0, 1.0}
    assert torch.count_nonzero(image[1:]) == 0
    transparent = SAM3DBody_Render.execute(
        pose_data=pose,
        width=48,
        background=background,
        **{"render_style.opacity": 0.0},
    )["image"]
    assert transparent.shape == (3, 36, 48, 3)
    torch.testing.assert_close(
        transparent,
        torch.tensor([0.2, 0.4, 0.8])[None, None, None].expand(3, 36, 48, 3),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(vertices, original["frames"][0][0]["pred_vertices"])
    np.testing.assert_array_equal(pose["faces"], original["faces"])
    assert pose["image_size"] == original["image_size"]


@pytest.mark.parametrize("style", ("mesh", "silhouette", "openpose_2d", "openpose_3d", "scail"))
def test_empty_pose_has_one_black_frame_in_each_style(style: str) -> None:
    image = SAM3DBody_Render.execute(
        pose_data={"image_size": (18, 30), "frames": []}, height=12, render_style=style
    )["image"]
    assert image.shape == (1, 12, 20, 3)
    assert image.dtype == torch.float32 and image.device.type == "cpu"
    assert torch.count_nonzero(image) == 0
