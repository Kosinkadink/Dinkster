from __future__ import annotations

import asyncio
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest
from dinkster_graph import Graph, GraphNode, TypedLiteral, validate
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
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker

from dinkster.compose import compose_serving

CAPABILITIES = (
    (DetectObjects, "dinkster.detection.detect", DETECT_PROVIDER_CHOICE),
    (SegmentDetections, "dinkster.detection.segment", SEGMENT_PROVIDER_CHOICE),
    (SegmentByText, "dinkster.detection.segment_text", TEXT_SEGMENT_PROVIDER_CHOICE),
    (ImageMatte, "dinkster.image.matte", MATTE_PROVIDER_CHOICE),
    (TrackObjects, "dinkster.detection.track", TRACK_PROVIDER_CHOICE),
)


def test_native_rtdetr_schemas_register_without_torch_or_inference() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_api.v1 import BOUNDING_BOX_TYPE, IMAGE_TYPE, TypeExpr\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "from dinkster_nodes_comfy.rtdetr import RTDETR_detect, DrawBBoxes\n"
            "from dinkster_nodes_comfy.sam3 import SAM3_TRACKING_NODES\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')\n"
            "for node in (RTDETR_detect, DrawBBoxes, *SAM3_TRACKING_NODES):\n"
            "    schema = node.define_schema()\n"
            "    assert node in COMFY_NODES and schema.node_type in manifest.namespaces\n"
            "    assert schema.category == 'image/detection'\n"
            "detect = RTDETR_detect.define_schema()\n"
            "assert [p.id for p in detect.inputs] == "
            "['model', 'image', 'threshold', 'class_name', 'max_detections']\n"
            "assert detect.outputs[0].type == TypeExpr.concrete(BOUNDING_BOX_TYPE)\n"
            "options = detect.inputs[3].widget.options\n"
            "assert len(options) == 81 and options[0].value == 'all'\n"
            "assert options[-1].value == 'toothbrush'\n"
            "draw = DrawBBoxes.define_schema()\n"
            "assert [p.id for p in draw.inputs] == ['image', 'bboxes']\n"
            "assert not draw.inputs[0].required and draw.inputs[1].required\n"
            "assert draw.outputs[0].type == TypeExpr.concrete(IMAGE_TYPE)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
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
    detection = TypeExpr.concrete("DETECTION")
    detections = TypeExpr.list_of(detection)
    mask = TypeExpr.concrete("MASK")
    masks = TypeExpr.list_of(mask)

    detect = DetectObjects.schema()
    max_results = detect.input("max_results")
    assert max_results is not None
    assert max_results.type == TypeExpr.concrete("INT")
    assert not max_results.required and max_results.default == -1
    assert max_results.widget == NumberWidget(step=1)
    min_score = detect.input("min_score")
    assert min_score is not None and min_score.widget == NumberWidget(step=0.01)
    result_limit_mode = detect.input("result_limit_mode")
    assert result_limit_mode is not None
    assert result_limit_mode.type == TypeExpr.concrete("COMBO")
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


@pytest.mark.parametrize(
    ("provider", "node_type", "input_id", "type_id", "obsolete_type_id"),
    (
        ("upscale", "dinkster.image.upscale_model", "upscale_model", "ASSET", "dinkster.asset"),
        (
            "sam31",
            "dinkster.detection.segment",
            "detections",
            "list<DETECTION>",
            "list<dinkster.detection>",
        ),
        (
            "efficient_sam",
            "dinkster.detection.segment",
            "detections",
            "list<DETECTION>",
            "list<dinkster.detection>",
        ),
    ),
)
def test_isolated_vision_accepts_canonical_typed_literals_without_aliases(
    provider: str, node_type: str, input_id: str, type_id: str, obsolete_type_id: str
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        import_module(f"dinkster_nodes_vision.{provider}").register_types(registry)
        manifest = (
            Path(__file__).parent.parent
            / "packages/dinkster-nodes-vision"
            / f"dinkster_vision_{provider}_pack/dinkster-pack.toml"
        )
        worker = IsolatedWorker(manifest, registry, python=sys.executable)
        await worker.start()
        try:
            for stamp in (type_id, obsolete_type_id):
                graph = Graph(
                    nodes={
                        "vision": GraphNode(
                            node_type,
                            {
                                "image": TypedLiteral("IMAGE", [[[[0.0, 0.25, 1.0]]]]),
                                input_id: TypedLiteral(
                                    stamp,
                                    []
                                    if input_id == "detections"
                                    else {
                                        "digest": "blake3:" + "a" * 64,
                                        "name": "upscale.pth",
                                    },
                                ),
                            },
                        ),
                    }
                )
                errors = [
                    d
                    for d in validate(graph, worker.schemas, ["vision"], known_types=registry)
                    if d.severity == "error"
                ]
                if stamp == type_id:
                    assert errors == []
                else:
                    assert [d.code for d in errors] == ["typed-literal-unknown-type"]
        finally:
            await worker.close()

    asyncio.run(scenario())
