"""Observable trim/crop pixels and nonmutating edit controls."""

import numpy as np
import pytest
from dinkster_api.v1 import assemble_video, disassemble_video
from dinkster_nodes_comfy.video_edit import VideoCrop, VideoTrim


def test_trim_identity_negative_start_and_strict_boundary() -> None:
    pixels = np.arange(5 * 4 * 8 * 3, dtype=np.float32).reshape(5, 4, 8, 3) / 1000
    original = pixels.copy()
    video = assemble_video(pixels, fps=2)
    assert VideoTrim.execute(video=video, trim=None, strict_duration=True)["VIDEO"] is video
    trimmed = VideoTrim.execute(
        video=video,
        trim={"trim": {"start_time": -1.0, "duration": 0.0}},
        strict_duration=True,
    )["VIDEO"]
    np.testing.assert_array_equal(disassemble_video(trimmed)["images"], pixels[-2:])
    with pytest.raises(ValueError, match="exceeds available"):
        VideoTrim.execute(
            video=video,
            trim={"trim": {"start_time": 2, "duration": 0.51}},
            strict_duration=True,
        )
    with pytest.raises(ValueError, match="duration must be >= 0"):
        VideoTrim.execute(video=video, trim={"trim": {"duration": -0.1}}, strict_duration=False)
    assert video["edits"] == []
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize(
    ("rectangle", "expected"),
    (
        ({"x": 3, "y": 1, "width": 5, "height": 3}, (slice(0, 2), slice(2, 6))),
        ({"x": 99, "y": 99, "width": 10, "height": 10}, (slice(2, 4), slice(6, 8))),
        ({"x": 2, "y": 2, "width": 0, "height": 2}, (slice(None), slice(None))),
    ),
)
def test_crop_even_pixel_geometry_and_zero_size_identity(rectangle: dict, expected: tuple) -> None:
    pixels = np.arange(2 * 4 * 8 * 4, dtype=np.float32).reshape(2, 4, 8, 4) / 1000
    video = assemble_video(pixels, fps=3)
    cropped = VideoCrop.execute(video=video, crop={"crop": rectangle})["VIDEO"]
    np.testing.assert_array_equal(
        disassemble_video(cropped)["images"], pixels[:, expected[0], expected[1], :]
    )
    assert video["edits"] == []
