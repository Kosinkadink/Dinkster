from __future__ import annotations

import numpy as np
import pytest
from dinkster_image.drawing import DrawMask, DrawRegion, MakeImage
from dinkster_image.types import Region


def _run(node: type, **kwargs: object) -> dict[str, object]:
    return node.execute(**kwargs)  # pyright: ignore[reportUnknownMemberType]


def _image(result: dict[str, object]) -> np.ndarray:
    return np.asarray(result["image"], dtype=np.float32)


def test_make_image_solid_and_gradients_have_expected_endpoints() -> None:
    solid = _image(
        _run(
            MakeImage,
            operation="solid",
            width=3,
            height=2,
            channels="rgba",
            color_a="#ff800040",
        )
    )
    linear = _image(
        _run(
            MakeImage,
            operation="linear_gradient",
            width=3,
            height=1,
            color_a="#000000",
            color_b="#ffffff",
            angle=0.0,
        )
    )
    radial = _image(
        _run(
            MakeImage,
            operation="radial_gradient",
            width=5,
            height=5,
            color_a="#ffffff",
            color_b="#000000",
            center_x=0.5,
            center_y=0.5,
            radius=0.5,
        )
    )

    assert np.allclose(solid[0, 0, 0], [1.0, 128 / 255, 0.0, 64 / 255])
    assert np.allclose(linear[0, 0, :, 0], [1 / 6, 0.5, 5 / 6])
    assert radial[0, 2, 2, 0] == 1.0
    assert radial[0, 0, 0, 0] == 0.0


def test_draw_region_fills_outlines_and_clips() -> None:
    image = np.zeros((1, 5, 6, 3), dtype=np.float32)
    filled = _image(
        _run(
            DrawRegion,
            image=image,
            region=Region(x=-1, y=1, width=4, height=3),
            color="#ff0000",
            opacity=0.5,
            mode="fill",
        )
    )
    outlined = _image(
        _run(
            DrawRegion,
            image=image,
            region=Region(x=1, y=1, width=4, height=3),
            color="#00ff00",
            mode="outline",
            line_width=1,
        )
    )
    oob = _image(
        _run(
            DrawRegion,
            image=image,
            region=Region(x=20, y=20, width=3, height=3),
            color="#ffffff",
        )
    )

    assert np.all(filled[0, 1:4, :3, 0] == 0.5)
    assert filled[..., 1:].sum() == 0.0
    assert outlined[0, 1:4, 1:5, 1].sum() == 10.0
    assert outlined[0, 2, 2, 1] == 0.0
    assert np.array_equal(oob, image)


def test_draw_region_ellipse_is_inside_region_bounds() -> None:
    result = _image(
        _run(
            DrawRegion,
            image=np.zeros((1, 7, 7, 3), dtype=np.float32),
            region=Region(x=1, y=1, width=5, height=5),
            shape="ellipse",
            mode="fill",
            color="#ffffff",
        )
    )
    assert np.array_equal(result[0, 3, 3], [1.0, 1.0, 1.0])
    assert np.array_equal(result[0, 1, 1], [0.0, 0.0, 0.0])
    assert not np.any(result[0, 0])


def test_draw_region_normalizes_legacy_bbox_batches_and_coordinate_formats() -> None:
    image = np.zeros((3, 6, 7, 3), dtype=np.float32)
    result = _image(
        _run(
            DrawRegion,
            image=image,
            region=[[1, 1, 5, 4], {"x": 2, "y": 2, "width": 3, "height": 2}],
            coordinate_format="xyxy",
            batch_policy="pairwise_truncate",
            color="#ff0000",
        )
    )

    assert result.shape == (2, 6, 7, 3)
    assert result[0, 1:4, 1:5, 0].sum() == 10.0
    assert result[1, 2:4, 2:5, 0].sum() == 6.0
    assert result[..., 1:].sum() == 0.0

    four_values = _image(
        _run(
            DrawRegion,
            image=image,
            region=[1, 1, 3, 2],
            batch_policy="pairwise_truncate",
        )
    )
    extended_record = _image(
        _run(
            DrawRegion,
            image=image,
            region=[[1, 1, 3, 2, 99]],
            batch_policy="pairwise_truncate",
        )
    )
    assert four_values.shape == (1, 6, 7, 3)
    assert np.array_equal(extended_record, four_values)


def test_draw_mask_signed_placement_and_alpha_compositing() -> None:
    image = np.zeros((1, 3, 4, 4), dtype=np.float32)
    image[..., 3] = 1.0
    mask = np.array([[[0.5, 1.0, 0.25], [1.0, 0.0, 1.0]]], dtype=np.float32)
    result = _image(
        _run(
            DrawMask,
            image=image,
            mask=mask,
            x=-1,
            y=2,
            color="#ff000080",
            opacity=0.5,
        )
    )

    assert result.shape == (1, 3, 4, 4)
    assert np.allclose(result[0, 2, :2, 0], [0.2509804, 0.0627451])
    assert np.allclose(result[0, 2, :2, 3], 1.0)
    assert result[..., 1:3].sum() == 0.0


def test_draw_mask_batches_broadcast_only_from_singleton() -> None:
    image = np.zeros((3, 2, 2, 3), dtype=np.float32)
    mask = np.ones((1, 2, 2), dtype=np.float32)
    broadcast = _image(_run(DrawMask, image=image, mask=mask, color="#ffffff", opacity=1.0))
    assert broadcast.shape == (3, 2, 2, 3)
    assert np.all(broadcast == 1.0)

    paired = _image(
        _run(
            DrawMask,
            image=np.zeros((2, 2, 2, 3), dtype=np.float32),
            mask=np.stack([np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]),
            color="#ffffff",
            opacity=1.0,
        )
    )
    assert paired[0].sum() == 0.0
    assert paired[1].sum() == 12.0

    with pytest.raises(ValueError, match="equal or singleton"):
        _run(
            DrawMask,
            image=np.zeros((2, 2, 2, 3), dtype=np.float32),
            mask=np.zeros((3, 2, 2), dtype=np.float32),
        )


def test_draw_mask_fully_oob_preserves_image() -> None:
    image = np.full((1, 3, 4, 3), 0.25, dtype=np.float32)
    result = _image(
        _run(
            DrawMask,
            image=image,
            mask=np.ones((1, 2, 2), dtype=np.float32),
            x=20,
            y=-20,
            color="#ffffff",
        )
    )
    assert np.array_equal(result, image)


def test_draw_mask_kj_compatibility_controls() -> None:
    image = np.zeros((3, 2, 4, 4), dtype=np.float32)
    image[..., 3] = 0.25
    mask = np.array([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=np.float32)
    result = _image(
        _run(
            DrawMask,
            image=image,
            mask=mask,
            color="255, 0, 0, 128",
            opacity=1.0,
            mask_size="resize_to_image",
            batch_policy="cyclic_repeat",
            alpha_mode="max",
        )
    )
    inverted = _image(
        _run(
            DrawMask,
            image=image[:1],
            mask=mask[:1],
            color="#ff000080",
            opacity=1.0,
            invert=True,
            mask_size="resize_to_image",
            alpha_mode="max",
        )
    )

    alpha = 128 / 255
    assert np.allclose(result[0, :, :2, 0], 0.0)
    assert np.allclose(result[0, :, 2:, 0], alpha)
    assert np.allclose(result[1, :, :2, 0], alpha)
    assert np.allclose(result[1, :, 2:, 0], 0.0)
    assert np.array_equal(result[0], result[2])
    assert np.allclose(result[0, :, :2, 3], 0.25)
    assert np.allclose(result[0, :, 2:, 3], alpha)
    assert np.allclose(inverted[0, :, :2, 0], alpha)
    assert np.allclose(inverted[0, :, 2:, 0], 0.0)
