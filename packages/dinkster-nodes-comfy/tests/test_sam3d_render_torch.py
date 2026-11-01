"""SAM3D rasterized pixels and source-preserving overlays."""

import copy

import numpy as np
import pytest
import torch
from dinkster_nodes_comfy.sam3d_openpose import render_pose_data_openpose
from dinkster_nodes_comfy.sam3d_render import SAM3DBody_Render


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
