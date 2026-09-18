from __future__ import annotations

import asyncio

import pytest
from dinkster_nodes_image import (
    DETECT_PROVIDER_CHOICE,
    IMAGE_NODES,
    MATTE_PROVIDER_CHOICE,
    MODEL_DEPTH_PROVIDER_CHOICE,
    MODEL_EDGE_PROVIDER_CHOICES,
    SEGMENT_PROVIDER_CHOICE,
    TEXT_SEGMENT_PROVIDER_CHOICE,
    TRACK_PROVIDER_CHOICE,
    UPSCALE_MODEL_PROVIDER_CHOICE,
    DetectObjects,
    ImageMatte,
    SegmentByText,
    SegmentDetections,
    TrackObjects,
    image_choices,
    vision_choices,
)
from dinkster_schema import ComboWidget, NumberWidget, TypeExpr

from dinkster.compose import compose_serving

CAPABILITIES = (
    (DetectObjects, "dinkster.detection.detect", DETECT_PROVIDER_CHOICE),
    (SegmentDetections, "dinkster.detection.segment", SEGMENT_PROVIDER_CHOICE),
    (SegmentByText, "dinkster.detection.segment_text", TEXT_SEGMENT_PROVIDER_CHOICE),
    (ImageMatte, "dinkster.image.matte", MATTE_PROVIDER_CHOICE),
    (TrackObjects, "dinkster.detection.track", TRACK_PROVIDER_CHOICE),
)


@pytest.mark.parametrize(("node", "node_type", "choice"), CAPABILITIES)
def test_capability_owner_schema_serves_provider_choice(
    node: type, node_type: str, choice: str
) -> None:
    schema = node.schema()
    assert schema.node_type == node_type
    assert choice == f"{node_type}.providers"
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{choice}"
    assert node in IMAGE_NODES


@pytest.mark.parametrize(("node", "node_type", "choice"), CAPABILITIES)
def test_capability_owner_schema_refuses_execution_without_provider(
    node: type, node_type: str, choice: str
) -> None:
    with pytest.raises(RuntimeError, match="requires an installed provider"):
        node.execute()


def test_vision_choices_are_declared_empty_and_pack_owned() -> None:
    expected = {
        DETECT_PROVIDER_CHOICE: (),
        SEGMENT_PROVIDER_CHOICE: (),
        TEXT_SEGMENT_PROVIDER_CHOICE: (),
        MATTE_PROVIDER_CHOICE: (),
        TRACK_PROVIDER_CHOICE: (),
    }
    assert vision_choices() == expected
    for choice, options in expected.items():
        assert image_choices()[choice] == options


def test_capability_schemas_share_detection_value_contract() -> None:
    detection = TypeExpr.concrete("dinkster.detection")
    detections = TypeExpr.list_of(detection)
    mask = TypeExpr.concrete("dinkster.mask")
    masks = TypeExpr.list_of(mask)

    detect = DetectObjects.schema()
    max_results = detect.input("max_results")
    assert max_results is not None
    assert max_results.type == TypeExpr.concrete("core.int")
    assert not max_results.required and max_results.default == -1
    assert max_results.widget == NumberWidget(step=1)
    min_score = detect.input("min_score")
    assert min_score is not None and min_score.widget == NumberWidget(step=0.01)
    result_limit_mode = detect.input("result_limit_mode")
    assert result_limit_mode is not None
    assert result_limit_mode.type == TypeExpr.concrete("core.combo")
    assert not result_limit_mode.required and result_limit_mode.default == "count"
    assert result_limit_mode.widget == ComboWidget(options=("count", "slice-stop"))
    assert [output.id for output in detect.outputs] == ["detections", "count"]
    assert detect.outputs[0].type == detections

    segment = SegmentDetections.schema()
    segment_prompts = segment.input("detections")
    assert segment_prompts is not None and segment_prompts.type == detections
    assert [output.id for output in segment.outputs] == ["detections", "masks"]
    assert segment.outputs[0].type == detections
    assert segment.outputs[1].type == masks

    text_segment = SegmentByText.schema()
    assert text_segment.input("image") == detect.input("image")
    assert text_segment.input("prompt") == detect.input("prompt")
    assert text_segment.input("prompt_mode") == detect.input("prompt_mode")
    text_min_score = text_segment.input("min_score")
    assert text_min_score is not None
    assert text_min_score.type == min_score.type
    assert text_min_score.default == min_score.default
    assert text_min_score.widget == NumberWidget(min=0.0, max=1.0, step=0.01)
    assert [output.id for output in text_segment.outputs] == ["detections", "masks"]
    assert text_segment.outputs[0].type == detections
    assert text_segment.outputs[1].type == masks

    matte = ImageMatte.schema()
    assert [output.id for output in matte.outputs] == ["mask"]
    assert matte.outputs[0].type == mask

    track = TrackObjects.schema()
    track_prompts = track.input("detections")
    assert track_prompts is not None and not track_prompts.required
    assert track_prompts.type == detections
    initial_masks = track.input("initial_masks")
    assert initial_masks is not None and not initial_masks.required
    assert initial_masks.type == mask
    assert [output.id for output in track.outputs] == ["masks", "combined"]
    assert track.outputs[0].type == masks
    assert track.outputs[1].type == mask


def test_default_composition_serves_standard_capability_choices() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            expected = {
                DETECT_PROVIDER_CHOICE: (
                    "dinkster-vision-detr",
                    "dinkster-vision-rtdetr",
                    "dinkster-vision-sam31",
                ),
                SEGMENT_PROVIDER_CHOICE: (
                    "dinkster-vision-efficient-sam",
                    "dinkster-vision-sam31",
                ),
                TEXT_SEGMENT_PROVIDER_CHOICE: ("dinkster-vision-sam31",),
                MATTE_PROVIDER_CHOICE: ("dinkster-vision-birefnet",),
                TRACK_PROVIDER_CHOICE: ("dinkster-vision-sam31",),
            }
            assert {choice: composition.choices[choice] for choice in vision_choices()} == expected
            assert composition.choices[UPSCALE_MODEL_PROVIDER_CHOICE] == (
                "dinkster-vision-upscale",
            )
            assert composition.choices[MODEL_DEPTH_PROVIDER_CHOICE] == (
                "dinkster-vision-depth-anything-v2",
                "dinkster-vision-depth-anything-v3",
            )
            for choice in MODEL_EDGE_PROVIDER_CHOICES:
                assert composition.choices[choice] == ("dinkster-vision-hed",)
        finally:
            await composition.close()

    asyncio.run(scenario())
