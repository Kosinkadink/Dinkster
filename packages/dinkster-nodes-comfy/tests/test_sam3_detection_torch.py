"""CPU prompt routing, score boundaries and mask geometry for SAM3 detection."""

# Source contract: SAM3_Detect at F bc5f880d resizes 3x5 and 6x8 fixture
# images to 1008x1008. Point/box expectations below derive from those exact
# dimensions and the explicitly supplied coordinates, not stored F outputs.

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sam3_detection import SAM3_Detect

from dinkster.compose import PackSpec, ServingComposer


@pytest.fixture(autouse=True)
def cpu_management(monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)
    monkeypatch.setattr(management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(management, "intermediate_device", lambda: torch.device("cpu"))


def model(sam: Any) -> Any:
    return SimpleNamespace(
        model=SimpleNamespace(get_dtype=lambda: torch.float32, diffusion_model=sam)
    )


@pytest.mark.parametrize("individual", (False, True))
def test_no_prompts_returns_empty_detections_and_correct_mask_batch(individual: bool) -> None:
    # Hand-derived: two 3x5 images with no prompts produce no boxes or
    # foreground; individual mode has no objects rather than two empty unions.
    image = torch.arange(120, dtype=torch.float32).reshape(2, 3, 5, 4) / 120
    original = image.clone()
    result = SAM3_Detect.execute(
        model=model(None), image=image.numpy(), individual_masks=individual
    )
    assert result["bboxes"] == [[], []]
    assert result["masks"].shape == ((0, 3, 5) if individual else (2, 3, 5))
    assert torch.count_nonzero(result["masks"]) == 0
    torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize("multiple", (False, True))
@pytest.mark.parametrize("individual", (False, True))
def test_text_filters_strictly_orders_limits_each_prompt_and_preserves_source(
    multiple: bool, individual: bool
) -> None:
    # Hand-derived from the fixture's xyxy boxes and logits: sigmoid(0)
    # equals the excluded threshold, logits 2 and 1 sort first, and per-prompt
    # limits retain one or two boxes. All-positive coarse masks fill each frame.
    class Detector:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, frame: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
            self.calls += 1
            assert frame.shape == (1, 3, 1008, 1008)
            assert kwargs["orig_size"] == (3, 5) and kwargs["threshold"] == 0.5
            assert kwargs["text_embeddings"].shape == (1, 3, 2)
            torch.testing.assert_close(kwargs["text_mask"], torch.ones(1, 3, dtype=torch.int64))
            return {
                "boxes": torch.tensor([[[0, 0, 2, 1], [1, 0, 4, 2], [2, 1, 5, 3], [0, 1, 1, 3]]]),
                "scores": torch.tensor([[0.0, 1.0, 2.0, -1.0]]),
                "masks": torch.ones(1, 4, 2, 2),
            }

    sam = Detector()
    embedding = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    metadata = (
        {
            "sam3_multi_cond": [
                {"cond": embedding, "attention_mask": None, "max_detections": 2},
                {"cond": embedding + 5, "attention_mask": None, "max_detections": 1},
            ]
        }
        if multiple
        else {}
    )
    image = torch.arange(120, dtype=torch.float32).reshape(2, 3, 5, 4) / 120
    original = image.clone()
    result = SAM3_Detect.execute(
        model=model(sam),
        image=image,
        conditioning=[[embedding, metadata]],
        refine_iterations=0,
        individual_masks=individual,
    )
    assert sam.calls == (4 if multiple else 2)
    expected_boxes = [(2.0, 1.0, 3.0, 2.0)]
    if multiple:
        expected_boxes += [(1.0, 0.0, 3.0, 2.0), (2.0, 1.0, 3.0, 2.0)]
    for boxes in result["bboxes"]:
        assert [(b["x"], b["y"], b["width"], b["height"]) for b in boxes] == expected_boxes
        assert boxes[0]["score"] == float(torch.tensor(2.0).sigmoid())
    expected_count = 6 if multiple and individual else 2
    assert result["masks"].shape == (expected_count, 3, 5)
    assert torch.all(result["masks"] == 1)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize("iterations", (0, 3))
def test_points_scale_pixel_coordinates_and_keep_positive_negative_labels(iterations: int) -> None:
    # Hand-derived: (1,2)/(5,3)*1008 -> (201.6,672) and
    # (4,1)/(5,3)*1008 -> (806.4,336). The fixture's last three columns are positive.
    class Segmenter:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def forward_segment(self, frame: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            self.calls.append(kwargs)
            if "point_inputs" in kwargs:
                points = kwargs["point_inputs"]
                torch.testing.assert_close(
                    points["point_coords"], torch.tensor([[[201.6, 672.0], [806.4, 336.0]]])
                )
                assert points["point_labels"].tolist() == [[1, 0]]
                assert points["point_labels"].dtype == torch.int32
            else:
                assert kwargs["mask_inputs"].shape == (1, 1, 3, 5)
            return torch.tensor([[[[-1.0, -1.0, 1.0, 1.0, 1.0]]]]).expand(1, 1, 3, 5)

    sam = Segmenter()
    result = SAM3_Detect.execute(
        model=model(sam),
        image=torch.zeros(1, 3, 5, 3),
        positive_coords='[{"x": 1, "y": 2}]',
        negative_coords='[{"x": 4, "y": 1}]',
        refine_iterations=iterations,
    )
    assert len(sam.calls) == max(1, iterations)
    assert result["bboxes"] == [[]]
    torch.testing.assert_close(
        result["masks"], torch.tensor([[[0.0, 0.0, 1.0, 1.0, 1.0]]]).expand(1, 3, 5)
    )


def test_per_frame_boxes_pad_last_frame_and_empty_frames_skip_segmentation() -> None:
    # Hand-derived: xywh(1,1,3,2) in a 5x3 frame maps to corners
    # (201.6,336)/(806.4,1008). Empty first frame skips; the last box repeats.
    corners: list[torch.Tensor] = []

    def segment(frame: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        corners.append(kwargs["box_inputs"])
        return torch.ones(1, 1, 3, 5)

    result = SAM3_Detect.execute(
        model=model(SimpleNamespace(forward_segment=segment)),
        image=torch.zeros(3, 3, 5, 3),
        bboxes=[[], [{"x": 1, "y": 1, "width": 3, "height": 2}]],
        refine_iterations=0,
    )
    assert len(corners) == 2
    expected = torch.tensor([[[201.6, 336.0], [806.4, 1008.0]]])
    for box in corners:
        torch.testing.assert_close(box, expected)
    assert torch.count_nonzero(result["masks"][0]) == 0
    assert torch.all(result["masks"][1:] == 1)


@pytest.mark.parametrize("iterations", (1, 2))
def test_text_refinement_crops_original_pixels_and_unions_coarse_masks(iterations: int) -> None:
    # Hand-derived: fixture xyxy(2,1,6,4), expanded 10% and truncated,
    # covers x[1:6], y[0:4]. Its coarse mask also retains pixel(7,5).
    class Refiner:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, frame: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
            coarse = torch.full((1, 1, 6, 8), -1.0)
            coarse[0, 0, 5, 7] = 1
            return {
                "boxes": torch.tensor([[[2.0, 1.0, 6.0, 4.0]]]),
                "scores": torch.tensor([[2.0]]),
                "masks": coarse,
            }

        def forward_segment(self, frame: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            self.calls += 1
            assert frame.shape == (1, 3, 1008, 1008)
            assert torch.all(frame[:, 0] == 0.25) and torch.all(frame[:, 1] == 0.75)
            assert kwargs["mask_inputs"].shape == (1, 1, 1008, 1008)
            return torch.ones(1, 1, 2, 3)

    image = torch.ones(1, 6, 8, 3)
    image[..., 0], image[..., 1] = 0.25, 0.75
    sam = Refiner()
    result = SAM3_Detect.execute(
        model=model(sam),
        image=image,
        conditioning=[[torch.ones(1, 2, 3), {}]],
        refine_iterations=iterations,
    )
    expected = torch.zeros(1, 6, 8)
    expected[:, 0:4, 1:6] = 1
    expected[:, 5, 7] = 1
    torch.testing.assert_close(result["masks"], expected, rtol=0, atol=0)
    assert sam.calls == iterations


class ControlledDetector:
    def __init__(self) -> None:
        self.model = SimpleNamespace(get_dtype=lambda: torch.float32, diffusion_model=self)

    def __call__(self, frame: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Hand-derived fixture inputs from DetectionSource; the logit-zero
        # full mask is threshold-equal, while logit2 retains only x>=2,y>=1.
        assert kwargs["orig_size"] == (3, 5) and kwargs["threshold"] == 0.5
        torch.testing.assert_close(
            kwargs["text_embeddings"], torch.tensor([[[2.0, -3.0], [7.0, 11.0]]]), rtol=0, atol=0
        )
        torch.testing.assert_close(kwargs["text_mask"], torch.tensor([[1, 0]]), rtol=0, atol=0)
        masks = torch.ones(1, 2, 3, 5)
        masks[0, 1] = -1
        masks[0, 1, 1:, 2:] = 1
        return {
            "boxes": torch.tensor([[[0.0, 0.0, 5.0, 3.0], [2.0, 1.0, 5.0, 3.0]]]),
            "scores": torch.tensor([[0.0, 2.0]]),
            "masks": masks,
        }


class DetectionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sam3-detect-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("model", MODEL_TYPE),
                    ("image", IMAGE_TYPE),
                    ("conditioning", CONDITIONING_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        management.load_model_gpu = lambda model: None
        management.get_torch_device = lambda: torch.device("cpu")
        management.intermediate_device = lambda: torch.device("cpu")
        return cls.outputs(
            model=ControlledDetector(),
            image=torch.arange(90, dtype=torch.float32).reshape(2, 3, 5, 3).numpy() / 90,
            conditioning=[
                [
                    torch.tensor([[[2.0, -3.0], [7.0, 11.0]]]),
                    {"attention_mask": torch.tensor([[1, 0]])},
                ]
            ],
        )


class DetectionInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sam3-detect-fixture.inspect",
            inputs=(
                InputSpec("bboxes", TypeExpr.concrete(BOUNDING_BOX_TYPE)),
                InputSpec("masks", TypeExpr.concrete(MASK_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, bboxes: Any, masks: Any) -> dict[str, object]:
        # Hand-derived from ControlledDetector: xyxy(2,1,5,3) becomes
        # xywh(2,1,3,2) at sigmoid(2); union pixels occupy x>=2,y>=1 in both frames.
        expected = torch.zeros(2, 3, 5)
        expected[:, 1:, 2:] = 1
        torch.testing.assert_close(materialize_image(masks), expected, rtol=0, atol=0)
        assert len(bboxes) == 2
        for boxes in bboxes:
            assert boxes == [
                {
                    "x": 2.0,
                    "y": 1.0,
                    "width": 3.0,
                    "height": 2.0,
                    "score": float(torch.tensor(2.0).sigmoid()),
                }
            ]
        return cls.outputs(summary="transported detection masks and boxes")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_manifest_detects_and_transports_results_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"sam3_detect_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_sam3_detection_torch import DetectionSource, DetectionInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (DetectionSource, DetectionInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="sam3-detect-fixture"\npublisher="dinkster"\n'
            'namespaces=["sam3-detect-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="sam3-detect-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("sam3-detect-fixture.source"),
                    "detect": GraphNode(
                        "SAM3_Detect",
                        {
                            "model": Link("source", "model"),
                            "image": Link("source", "image"),
                            "conditioning": Link("source", "conditioning"),
                            "refine_iterations": 0,
                        },
                    ),
                    "inspect": GraphNode(
                        "sam3-detect-fixture.inspect",
                        {"bboxes": Link("detect", "bboxes"), "masks": Link("detect", "masks")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = (
                    (await engine.run(graph, ["inspect"])).outputs["inspect"]["summary"].resolve()
                )
                assert result == "transported detection masks and boxes"
        finally:
            await composer.close()

    asyncio.run(scenario())
