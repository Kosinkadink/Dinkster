"""Tracked crop and reinsertion behavior for image batches and VIDEO values."""

from __future__ import annotations

import asyncio
import io
from fractions import Fraction
from typing import Any, cast

import av
import dinkster_nodes_image.support as image_support
import numpy as np
import pytest
from dinkster_api.v1 import (
    Detection,
    Region,
    assemble_video,
    bind_video_value,
    disassemble_video,
    edit_video,
    save_video_stream,
    video_from_source,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_nodes_image import (
    IMAGE_NODES,
    MakeImage,
    TrackedCrop,
    TrackedUncrop,
    register_image_types,
)
from dinkster_nodes_media_io import AssembleVideo, register_media_types
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker


def _frames(count: int = 3, height: int = 6, width: int = 8, channels: int = 3) -> np.ndarray:
    values = np.arange(count * height * width * channels, dtype=np.float32)
    return values.reshape(count, height, width, channels) / max(1, values.size - 1)


def _audio() -> dict[str, object]:
    waveform = np.linspace(-1, 1, 4_800, dtype=np.float32)[None, None]
    return {"waveform": waveform, "sample_rate": 16_000}


def _video(
    frames: np.ndarray,
    *,
    fps: Fraction = Fraction(30_000, 1_001),
    audio: object = None,
) -> dict[str, object]:
    return assemble_video(
        frames,
        fps=fps,
        audio=audio,
        bit_depth="10",
        color_space="HDR PQ",
    )


def _video_frames(value: object) -> np.ndarray:
    video = cast("dict[str, Any]", value)
    return cast("np.ndarray", video["components"]["images"])


def test_mask_tracking_uses_largest_window_and_preserves_soft_masks() -> None:
    frames = _frames()
    masks = np.zeros((3, 6, 8), dtype=np.float32)
    masks[0, 1:3, 1:3] = 0.25
    masks[1, 2:6, 5:8] = np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape(4, 3)

    result = TrackedCrop.execute(media=frames, source="masks", masks=masks, threshold=0.2)
    cropped = cast("np.ndarray", result["media"])
    cropped_masks = cast("np.ndarray", result["masks"])
    regions = cast("list[Region]", result["regions"])

    assert cropped.shape == (3, 4, 3, 3)
    assert regions == [Region(0, 0, 3, 4), Region(5, 2, 3, 4), Region(0, 0, 0, 0)]
    np.testing.assert_array_equal(cropped[0], frames[0, 0:4, 0:3])
    np.testing.assert_array_equal(cropped[1], frames[1, 2:6, 5:8])
    np.testing.assert_array_equal(cropped_masks[1], masks[1, 2:6, 5:8])
    assert not cropped[2].any()
    assert not cropped_masks[2].any()


def test_boxes_clip_fractional_negative_and_border_regions_before_centering() -> None:
    frames = _frames()
    detection_mask = np.zeros((6, 8), dtype=np.float32)
    detection_mask[:, 6:] = 0.4
    boxes = [
        Region(-2.2, -1.1, 5.4, 3.2),
        Detection("subject", 0.9, Region(5.5, 1.2, 8.0, 4.1), detection_mask),
        Region(20, 20, 0, 4),
    ]

    result = TrackedCrop.execute(
        media=frames,
        source="bounding_boxes",
        bounding_boxes=boxes,
        padding=1,
    )
    cropped = cast("np.ndarray", result["media"])
    masks = cast("np.ndarray", result["masks"])
    regions = cast("list[Region]", result["regions"])

    assert cropped.shape == (3, 6, 5, 3)
    assert regions == [Region(0, 0, 5, 6), Region(3, 0, 5, 6), Region(0, 0, 0, 0)]
    assert np.all(masks[0] == 1)
    np.testing.assert_array_equal(masks[1], detection_mask[:, 3:8])
    assert not masks[2].any()


def test_all_empty_tracking_returns_minimal_temporal_placeholders() -> None:
    frames = _frames(channels=1)
    masks = np.zeros((3, 6, 8), dtype=np.float32)
    result = TrackedCrop.execute(media=frames, masks=masks)

    assert cast("np.ndarray", result["media"]).shape == (3, 1, 1, 1)
    assert cast("np.ndarray", result["masks"]).shape == (3, 1, 1)
    assert result["regions"] == [Region(0, 0, 0, 0)] * 3
    restored = TrackedUncrop.execute(
        base=frames,
        crop=result["media"],
        regions=cast("list[Region]", result["regions"]),
        masks=result["masks"],
    )
    np.testing.assert_array_equal(restored["media"], frames)


def test_zero_and_wholly_offscreen_boxes_stay_empty_even_with_padding() -> None:
    result = TrackedCrop.execute(
        media=_frames(3),
        source="bounding_boxes",
        bounding_boxes=[
            Region(2, 2, 0, 2),
            Region(-3, 1, 2, 2),
            Region(8, 1, 2, 2),
        ],
        padding=4,
    )
    assert result["regions"] == [Region(0, 0, 0, 0)] * 3
    assert not cast("np.ndarray", result["media"]).any()
    assert not cast("np.ndarray", result["masks"]).any()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"source": "masks", "masks": np.zeros((1, 6, 8), np.float32)}, "mask count"),
        ({"source": "bounding_boxes", "bounding_boxes": [Region(0, 0, 1, 1)]}, "box count"),
    ],
)
def test_crop_rejects_temporal_mismatch_without_broadcasting(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        cast("Any", TrackedCrop.execute)(media=_frames(), **kwargs)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_crop_and_uncrop_round_trip_is_exact_and_does_not_mutate_inputs(channels: int) -> None:
    frames = _frames(channels=channels)
    original = frames.copy()
    boxes = [Region(0, 0, 3, 2), Region(3, 2, 4, 3), Region(6, 4, 2, 2)]
    crop = TrackedCrop.execute(media=frames, source="bounding_boxes", bounding_boxes=boxes)
    result = TrackedUncrop.execute(
        base=frames,
        crop=crop["media"],
        regions=cast("list[Region]", crop["regions"]),
        masks=crop["masks"],
    )

    np.testing.assert_array_equal(result["media"], original)
    np.testing.assert_array_equal(frames, original)


def test_uncrop_resizes_processed_crops_and_applies_soft_masks_and_opacity() -> None:
    base = np.zeros((2, 6, 8, 3), dtype=np.float32)
    crop = np.ones((2, 2, 2, 3), dtype=np.float32)
    masks = np.stack((np.full((2, 2), 0.5, np.float32), np.ones((2, 2), np.float32)))
    regions = [Region(1, 1, 4, 3), Region(6, 4, 4, 4)]

    result = cast(
        "np.ndarray",
        TrackedUncrop.execute(
            base=base,
            crop=crop,
            regions=regions,
            masks=masks,
            opacity=0.5,
            interpolation="nearest-exact",
        )["media"],
    )

    assert np.all(result[0, 1:4, 1:5] == 0.25)
    assert np.all(result[1, 4:6, 6:8] == 0.5)
    assert not result[0, :1].any() and not result[0, :, 5:].any()
    assert not result[1, :4].any() and not result[1, :, :6].any()


@pytest.mark.parametrize(
    ("limit_name", "limit", "match"),
    [
        ("MAX_IMAGE_BYTES", 383, "output exceeds"),
        ("MAX_DIMENSION", 3, "output dimensions exceed"),
    ],
)
def test_uncrop_checks_base_allocation_limits_with_zero_regions(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    match: str,
) -> None:
    base = np.zeros((2, 4, 4, 3), dtype=np.float32)
    crop = np.zeros((2, 1, 1, 3), dtype=np.float32)
    monkeypatch.setattr(image_support, limit_name, limit)

    with pytest.raises(ValueError, match=match):
        TrackedUncrop.execute(
            base=base,
            crop=crop,
            regions=[Region(0, 0, 0, 0)] * 2,
        )


@pytest.mark.parametrize(
    ("crop_count", "region_count", "mask_count", "match"),
    [(1, 2, 2, "crop frame count"), (2, 1, 2, "region count"), (2, 2, 1, "mask count")],
)
def test_uncrop_rejects_every_temporal_mismatch(
    crop_count: int, region_count: int, mask_count: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        TrackedUncrop.execute(
            base=_frames(2),
            crop=_frames(crop_count, 2, 2),
            regions=[Region(0, 0, 2, 2)] * region_count,
            masks=np.ones((mask_count, 2, 2), np.float32),
        )


def test_component_video_preserves_exact_timeline_audio_depth_and_color() -> None:
    frames = _frames()
    audio = _audio()
    video = _video(frames, audio=audio)
    boxes = [Region(0, 0, 3, 3), Region(2, 1, 3, 3), Region(4, 2, 3, 3)]

    crop = TrackedCrop.execute(media=video, source="bounding_boxes", bounding_boxes=boxes)
    cropped_video = cast("dict[str, Any]", crop["media"])
    components = cropped_video["components"]
    assert components["fps"] == Fraction(30_000, 1_001)
    component_audio = cast("dict[str, object]", components["audio"])
    assert component_audio["sample_rate"] == audio["sample_rate"]
    assert cast("dict[str, object]", component_audio["source"])["pcm"] is audio["waveform"]
    np.testing.assert_array_equal(component_audio["waveform"], audio["waveform"])
    assert components["bit_depth"] == 10
    assert components["color_space"] == "HDR PQ"
    source_components = cast("dict[str, object]", video["components"])
    assert components["color"] == source_components["color"]
    assert cropped_video["edits"] == []

    restored = cast(
        "dict[str, Any]",
        TrackedUncrop.execute(
            base=video,
            crop=cropped_video,
            regions=cast("list[Region]", crop["regions"]),
            masks=crop["masks"],
        )["media"],
    )
    np.testing.assert_array_equal(_video_frames(restored), frames)
    assert restored["components"]["fps"] == Fraction(30_000, 1_001)
    restored_components = cast("dict[str, object]", restored["components"])
    restored_audio = cast("dict[str, object]", restored_components["audio"])
    assert cast("dict[str, object]", restored_audio["source"])["pcm"] is audio["waveform"]
    np.testing.assert_array_equal(restored_audio["waveform"], audio["waveform"])


def test_video_uncrop_accepts_image_crop_and_rejects_video_rate_mismatch() -> None:
    base = _video(_frames(2), fps=Fraction(24))
    regions = [Region(0, 0, 2, 2)] * 2
    image_crop = np.ones((2, 2, 2, 3), dtype=np.float32)
    result = TrackedUncrop.execute(base=base, crop=image_crop, regions=regions)
    assert cast("dict[str, Any]", result["media"])["components"]["fps"] == 24

    with pytest.raises(ValueError, match="timeline rates"):
        TrackedUncrop.execute(
            base=base,
            crop=_video(image_crop, fps=Fraction(25)),
            regions=regions,
        )


def test_encoded_edited_video_materializes_without_encoding_and_preserves_metadata(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    frames = _frames(4, 8, 8)
    audio = _audio()
    encoded = io.BytesIO()
    save_video_stream(assemble_video(frames, fps=Fraction(24), audio=audio), encoded)
    source = video_from_source(encoded.getvalue())
    edited = edit_video(source, {"trim": {"start_time": 0, "duration": Fraction(1, 8)}})
    edited = bind_video_value(edited, for_audio_extraction=True)
    expected = disassemble_video(edited)
    count = cast(int, expected["frame_count"])
    masks = np.ones((count, 8, 8), dtype=np.float32)

    result = TrackedCrop.execute(media=edited, masks=masks)
    output = cast("dict[str, Any]", result["media"])
    assert "components" in output and "source" not in output
    assert output["edits"] == []
    output_components = cast("dict[str, object]", output["components"])
    source_probe = cast("dict[str, object]", source["probe"])
    assert output_components["fps"] == source_probe["fps"]
    assert output_components["bit_depth"] == source_probe["bit_depth"]
    assert output_components["color_space"] == source_probe["color_space"]
    assert output_components["color"] == {
        key: source_probe[key] for key in ("primaries", "transfer", "matrix", "range")
    }
    assert output_components["audio"] is not None
    np.testing.assert_array_equal(_video_frames(output), expected["images"])


def test_untagged_encoded_video_is_rejected_without_inventing_color_metadata() -> None:
    encoded = io.BytesIO()
    with av.open(encoded, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = stream.height = 8
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((8, 8, 3), np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video = video_from_source(encoded.getvalue())
    probe = cast("dict[str, object]", video["probe"])
    assert probe["color_space"] == "unknown"
    original_probe = probe.copy()

    with pytest.raises(ValueError, match="metadata cannot be represented"):
        TrackedCrop.execute(media=video, masks=np.ones((1, 8, 8), np.float32))
    with pytest.raises(ValueError, match="metadata cannot be represented"):
        TrackedUncrop.execute(base=video, crop=_frames(1), regions=[Region(0, 0, 0, 0)])
    assert video["probe"] == original_probe


@pytest.mark.parametrize("type_id", ["dinkster.image", "comfy.VIDEO"])
def test_registered_schemas_bind_media_output_types_through_graph_execution(type_id: str) -> None:
    registered = {node.schema().node_type: node.schema() for node in IMAGE_NODES}
    crop_schema = registered["dinkster.image.tracked_crop"]
    uncrop_schema = registered["dinkster.image.tracked_uncrop"]
    expected_media = crop_schema.inputs[0].type
    assert expected_media.kind == "variable"
    assert expected_media.types == ("dinkster.image", "comfy.VIDEO")
    assert crop_schema.outputs[0].type == expected_media
    assert uncrop_schema.inputs[0].type.kind == "variable"
    assert uncrop_schema.outputs[0].type == uncrop_schema.inputs[0].type
    assert uncrop_schema.inputs[1].type.kind == "union"
    source = crop_schema.combos[0]
    assert tuple(option.key for option in source.options) == ("masks", "bounding_boxes")

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_media_types(registry)
        register_image_types(registry)
        nodes = (MakeImage, AssembleVideo, TrackedCrop, TrackedUncrop)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        frames = np.ones((2, 6, 8, 3), dtype=np.float32)
        media = Link("assemble", "video") if type_id == "comfy.VIDEO" else Link("generate", "image")
        masks = np.ones((2, 6, 8), dtype=np.float32)
        graph = Graph(
            {
                "generate": GraphNode(
                    "dinkster.image.generate",
                    {
                        "width": 8,
                        "height": 6,
                        "batch_size": 2,
                        "color_source.color_a": "#ffffff",
                    },
                    slot_variants={"color_source": "hex", "operation": "solid"},
                ),
                "assemble": GraphNode(
                    "dinkster.video.assemble", {"images": Link("generate", "image")}
                ),
                "crop": GraphNode(
                    "dinkster.image.tracked_crop",
                    {
                        "media": media,
                        "source.masks": TypedLiteral("dinkster.mask", masks.tolist()),
                    },
                    slot_variants={"source": "masks"},
                ),
                "uncrop": GraphNode(
                    "dinkster.image.tracked_uncrop",
                    {
                        "base": media,
                        "crop": Link("crop", "media"),
                        "regions": Link("crop", "regions"),
                        "masks": Link("crop", "masks"),
                    },
                ),
            }
        )
        execution = await engine.run(graph, ("crop", "uncrop"))
        for node in ("crop", "uncrop"):
            output = execution.outputs[node]["media"]
            assert output.type_id == type_id
            result = output.resolve()
            pixels = _video_frames(result) if type_id == "comfy.VIDEO" else result
            np.testing.assert_array_equal(pixels, frames)

    asyncio.run(scenario())
