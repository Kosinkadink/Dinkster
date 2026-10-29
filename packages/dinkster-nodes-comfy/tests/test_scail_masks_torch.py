"""CPU SCAIL-2 colored mask rendering."""

from __future__ import annotations

import torch
from dinkster_api.v1 import IMAGE_TYPE, MASK_TYPE, SAM3_TRACK_DATA_TYPE
from dinkster_nodes_comfy.scail_masks import SCAIL2ColoredMask


def _pack(masks: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(8, dtype=torch.uint8)
    return (masks.to(torch.uint8).view(*masks.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def _tracks(masks: torch.Tensor, *, frames: int | None = None) -> dict[str, object]:
    return {
        "packed_masks": _pack(masks),
        "orig_size": masks.shape[-2:],
        "n_frames": masks.shape[0] if frames is None else frames,
        "scores": list(range(masks.shape[1])),
    }


def test_schema_preserves_native_order_types_defaults_and_visibility() -> None:
    schema = SCAIL2ColoredMask.define_schema()
    assert schema.node_type == "SCAIL2ColoredMask"
    assert schema.display_name == "Create SCAIL-2 Colored Mask"
    assert schema.category == "model/conditioning/wan/scail"
    assert schema.search_visibility == "normal"
    assert [item.id for item in schema.inputs] == [
        "driving_track_data",
        "ref_track_data",
        "object_indices",
        "sort_by",
        "replacement_mode",
    ]
    assert schema.inputs[0].type == SCAIL2ColoredMask.define_schema().inputs[0].type
    assert schema.inputs[0].type.types == (SAM3_TRACK_DATA_TYPE,)
    assert schema.inputs[1].type.types == (SAM3_TRACK_DATA_TYPE, MASK_TYPE)
    assert schema.inputs[1].required is False
    assert [item.default for item in schema.inputs[2:]] == ["", "left_to_right", False]
    assert [(item.id, item.type.types) for item in schema.outputs] == [
        ("pose_video_mask", (IMAGE_TYPE,)),
        ("reference_image_mask", (IMAGE_TYPE,)),
    ]


def test_sorting_filtering_overlap_palette_wrap_and_track_storage_are_asymmetric() -> None:
    masks = torch.zeros((2, 7, 2, 8), dtype=torch.bool)
    masks[0, 0, 0, 5] = True
    masks[0, 1, 0, 1] = True
    masks[1, 2, 0, :4] = True
    masks[1, 3, 0, 6] = True
    masks[1, 4, 1, 3] = True
    masks[1, 5, 1, 4] = True
    masks[1, 6, 1, 4] = True
    driving = _tracks(masks)
    packed_before = driving["packed_masks"].clone()
    result = SCAIL2ColoredMask.execute(
        driving_track_data=driving,
        object_indices="0, 1, 2, ignored",
        sort_by="left_to_right",
        replacement_mode=False,
    )
    pose = result["pose_video_mask"]
    assert pose.dtype is torch.float32 and pose.device.type == "cpu"
    assert torch.equal(pose[0, 0, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(pose[0, 0, 5], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(pose[1, 0, 2], torch.tensor([0.0, 1.0, 0.0]))
    assert not pose[0, 1].count_nonzero()
    all_objects = SCAIL2ColoredMask.execute(
        driving_track_data=driving,
        object_indices="",
        sort_by="left_to_right",
        replacement_mode=False,
    )["pose_video_mask"]
    assert torch.equal(all_objects[1, 1, 4], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(all_objects[1, 0, 6], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(driving["packed_masks"], packed_before)


def test_reference_absence_track_and_plain_mask_use_opposite_backgrounds() -> None:
    driving_masks = torch.zeros((1, 1, 2, 8), dtype=torch.bool)
    driving_masks[0, 0, 0, 0] = True
    driving = _tracks(driving_masks)
    absent = SCAIL2ColoredMask.execute(
        driving_track_data=driving,
        object_indices="",
        sort_by="none",
        replacement_mode=False,
    )
    assert torch.equal(absent["pose_video_mask"][0, 1, 7], torch.zeros(3))
    assert torch.equal(absent["reference_image_mask"], torch.ones((1, 2, 8, 3)))

    reference_masks = torch.zeros((1, 1, 2, 8), dtype=torch.bool)
    reference_masks[0, 0, 1, 7] = True
    track_reference = SCAIL2ColoredMask.execute(
        driving_track_data=driving,
        ref_track_data=_tracks(reference_masks),
        object_indices="",
        sort_by="none",
        replacement_mode=True,
    )
    assert torch.equal(track_reference["pose_video_mask"][0, 1, 7], torch.ones(3))
    assert torch.equal(
        track_reference["reference_image_mask"][0, 1, 7], torch.tensor([0.0, 0.0, 1.0])
    )
    assert torch.equal(track_reference["reference_image_mask"][0, 0, 0], torch.zeros(3))

    mask = torch.tensor([[0.5, 0.5001], [0.0, 1.0]])
    plain_reference = SCAIL2ColoredMask.execute(
        driving_track_data=driving,
        ref_track_data=mask,
        object_indices="",
        sort_by="none",
        replacement_mode=False,
    )["reference_image_mask"]
    assert torch.equal(plain_reference[0, 0, 0], torch.ones(3))
    assert torch.equal(plain_reference[0, 0, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(plain_reference[0, 1, 0], torch.ones(3))
    assert torch.equal(plain_reference[0, 1, 1], torch.tensor([0.0, 0.0, 1.0]))


def test_empty_packed_tracks_render_declared_frame_count_without_source_mutation() -> None:
    source = {
        "packed_masks": torch.empty((3, 0, 2, 1), dtype=torch.uint8),
        "orig_size": (2, 8),
        "n_frames": 99,
        "scores": [],
    }
    packed_before = source["packed_masks"].clone()
    result = SCAIL2ColoredMask.execute(
        driving_track_data=source,
        object_indices="",
        sort_by="area",
        replacement_mode=False,
    )["pose_video_mask"]
    assert result.shape == (3, 2, 8, 3)
    assert not result.count_nonzero()
    assert torch.equal(source["packed_masks"], packed_before)
