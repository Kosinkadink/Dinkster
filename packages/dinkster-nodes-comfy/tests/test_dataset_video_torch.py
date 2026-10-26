"""CPU contracts for lazy VIDEO dataset transforms."""

from __future__ import annotations

import io
from fractions import Fraction
from typing import Any, cast

import numpy as np
from dinkster_api.v1 import (
    VIDEO_TYPE,
    TypeExpr,
    assemble_video,
    disassemble_video,
    effective_video_facts,
    save_video_stream,
    video_from_source,
)
from dinkster_nodes_comfy.dataset_video import (
    DATASET_VIDEO_NODES,
    ShuffleVideoDataset,
    ShuffleVideoTextDataset,
    VideoRandomTemporalCrop,
    VideoTemporalCrop,
)


def _source() -> tuple[bytes, dict[str, object]]:
    frames = np.zeros((7, 16, 24, 3), dtype=np.float32)
    for index, value in enumerate((0.03, 0.17, 0.31, 0.46, 0.62, 0.78, 0.94)):
        frames[index, ..., 0] = value
        frames[index, ..., 1] = 1.0 - value
        frames[index, ..., 2] = value / 2
    encoded = io.BytesIO()
    save_video_stream(
        assemble_video(frames, fps=Fraction(5, 1)),
        encoded,
        codec="ffv1",
        container="mkv",
    )
    return encoded.getvalue(), video_from_source(encoded.getvalue())


def _values(video: object) -> list[float]:
    images = cast(dict[str, Any], disassemble_video(video))["images"]
    return [round(float(frame[0, 0, 0]) / 255, 2) for frame in cast(np.ndarray, images)]


def test_temporal_crop_clamps_to_source_tail_without_decoding_at_execute_time() -> None:
    source, video = _source()
    result = VideoTemporalCrop.execute(video=video, start_frame=5, length=99)["video"]
    assert result["source"] == source
    assert "components" not in result
    assert result["edits"] == [
        {
            "trim": {"start_time": Fraction(1, 1), "duration": Fraction(2, 5)},
            "strict_duration": False,
        }
    ]
    facts = effective_video_facts(result)
    assert (facts["frame_count"], facts["fps"], facts["duration"]) == (
        2,
        Fraction(5, 1),
        Fraction(2, 5),
    )
    assert _values(result) == [0.78, 0.94]


def test_temporal_crop_clamps_an_out_of_range_start_to_the_final_frame() -> None:
    _source_bytes, video = _source()
    result = VideoTemporalCrop.execute(video=video, start_frame=99999, length=16)["video"]
    assert _values(result) == [0.94]


def test_random_temporal_crop_matches_reference_seed_selection_and_stays_lazy() -> None:
    source, video = _source()
    seed = 19
    expected_start = int(np.random.RandomState(seed).randint(0, 5))
    result = VideoRandomTemporalCrop.execute(video=video, length=3, seed=seed)["video"]
    assert result["source"] == source
    assert "components" not in result
    assert result["edits"] == [
        {
            "trim": {
                "start_time": Fraction(expected_start, 5),
                "duration": Fraction(3, 5),
            },
            "strict_duration": False,
        }
    ]
    values = [0.03, 0.17, 0.31, 0.46, 0.62, 0.78, 0.94]
    assert _values(result) == values[expected_start : expected_start + 3]


def test_shuffle_video_nodes_preserve_reference_order_and_text_pairing() -> None:
    videos = [{"id": value} for value in ("one", "two", "three", "four")]
    expected = [int(index) for index in np.random.RandomState(3).permutation(len(videos))]
    shuffled = ShuffleVideoDataset.execute(videos=videos, seed=[3])["videos"]
    paired = ShuffleVideoTextDataset.execute(
        videos=videos, texts=["one", "two", "three", "four"], seed=[3]
    )
    assert shuffled == [videos[index] for index in expected]
    assert paired == {
        "videos": [videos[index] for index in expected],
        "texts": [["one", "two", "three", "four"][index] for index in expected],
    }


def test_video_dataset_schemas_keep_video_and_list_contracts() -> None:
    expected = (
        ("VideoTemporalCrop", "video/transform", ("video", "start_frame", "length"), ("VIDEO",)),
        (
            "VideoRandomTemporalCrop",
            "video/transform",
            ("video", "length", "seed"),
            ("VIDEO",),
        ),
        ("ShuffleVideoDataset", "video/batch", ("videos", "seed"), ("VIDEO",)),
        (
            "ShuffleVideoTextDataset",
            "video/batch",
            ("videos", "texts", "seed"),
            ("VIDEO", "STRING"),
        ),
    )
    for node, (node_type, category, inputs, output_types) in zip(
        DATASET_VIDEO_NODES, expected, strict=True
    ):
        schema = node.schema()
        assert (schema.node_type, schema.category) == (node_type, category)
        assert tuple(spec.id for spec in schema.inputs) == inputs
        assert tuple(spec.type for spec in schema.outputs) == tuple(
            TypeExpr.list_of(TypeExpr.concrete(type_id))
            if node_type.startswith("Shuffle")
            else TypeExpr.concrete(type_id)
            for type_id in output_types
        )
    assert VideoTemporalCrop.schema().inputs[0].type == TypeExpr.concrete(VIDEO_TYPE)
    assert ShuffleVideoDataset.schema().inputs[0].type == TypeExpr.list_of(
        TypeExpr.concrete(VIDEO_TYPE)
    )
    assert ShuffleVideoDataset.schema().inputs[1].default == [0]
