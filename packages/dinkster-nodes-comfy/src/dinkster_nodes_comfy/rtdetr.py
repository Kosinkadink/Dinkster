"""Native RT-DETR detection and bounding-box visualization."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    MODEL_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_BBOX = TypeExpr.concrete(BOUNDING_BOX_TYPE)


class RTDETR_detect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RTDETR_detect",
            display_name="Run Real-Time Detection (RT-DETR)",
            category="image/detection",
            search_terms=("bbox", "bounding box", "object detection", "coco"),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE), display_name="model"),
                InputSpec("image", _IMAGE, display_name="image"),
                InputSpec(
                    "threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    display_name="threshold",
                    default=0.5,
                ),
                InputSpec(
                    "class_name",
                    TypeExpr.concrete(CORE_COMBO),
                    default="all",
                    widget=ComboWidget(
                        options=tuple(ComboOption(name) for name in ("all", *_COCO_CLASSES))
                    ),
                    doc="Filter detections by class. Set to 'all' to disable filtering.",
                ),
                InputSpec(
                    "max_detections",
                    TypeExpr.concrete(CORE_INT),
                    display_name="max_detections",
                    default=100,
                    doc=(
                        "Maximum number of detections to return per image. "
                        "In order of descending confidence score."
                    ),
                ),
            ),
            outputs=(OutputSpec("bboxes", _BBOX),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, image: Any, threshold: float, class_name: str, max_detections: int
    ) -> Mapping[str, object]:
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        batch_size, height, width, _ = image.shape
        management.load_model_gpu(model)
        results: list[dict[str, Any]] = []
        for index in range(0, batch_size, 32):
            batch = image[index : index + 32]
            image_in = utils.common_upscale(
                batch.movedim(-1, 1), 640, 640, "bilinear", crop="disabled"
            )
            results.extend(model.model.diffusion_model(image_in, (width, height)))
        frames: list[list[dict[str, Any]]] = []
        for detection in results:
            keep = detection["scores"] > threshold
            boxes = detection["boxes"][keep].cpu()
            labels = detection["labels"][keep].cpu()
            scores = detection["scores"][keep].cpu()
            frame: list[dict[str, Any]] = [
                {
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "width": float(box[2] - box[0]),
                    "height": float(box[3] - box[1]),
                    "label": _COCO_CLASSES[int(label)],
                    "score": float(score),
                }
                for box, label, score in zip(boxes, labels, scores, strict=True)
                if class_name == "all" or _COCO_CLASSES[int(label)] == class_name
            ]
            frame.sort(key=lambda value: value["score"], reverse=True)
            frames.append(frame[:max_detections])
        return cls.outputs(bboxes=frames)


class DrawBBoxes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DrawBBoxes",
            display_name="Draw BBoxes",
            category="image/detection",
            search_terms=(
                "bbox",
                "bounding box",
                "object detection",
                "rt_detr",
                "visualize detections",
                "coco",
            ),
            inputs=(
                InputSpec("image", _IMAGE, required=False),
                InputSpec(
                    "bboxes",
                    _BBOX,
                    force_input=True,
                    default={"x": 0, "y": 0, "width": 512, "height": 512},
                ),
            ),
            outputs=(OutputSpec("out_image", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, bboxes: Any, image: Any = None) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        transforms = importlib.import_module("torchvision.transforms")
        management = importlib.import_module("dinkster_inference.model_management")
        batch_size = image.shape[0] if image is not None else 1
        if isinstance(bboxes, dict):
            bboxes = [[bboxes]]
        elif not isinstance(bboxes, list) or not bboxes:
            bboxes = [[]]
        elif isinstance(bboxes[0], dict):
            bboxes = [bboxes]
        bboxes = cast(list[list[dict[str, Any]]], bboxes)
        if len(bboxes) == 1:
            bboxes = bboxes * batch_size
        bboxes = (bboxes + [[]] * batch_size)[:batch_size]
        if image is None:
            batch_size = len(bboxes)
            width = max((int(d["x"] + d["width"]) for frame in bboxes for d in frame), default=640)
            height = max(
                (int(d["y"] + d["height"]) for frame in bboxes for d in frame), default=640
            )
            image = torch.zeros((batch_size, height, width, 3), dtype=torch.float32)
        outputs: list[Any] = []
        for index in range(batch_size):
            detections = bboxes[index]
            if detections:
                boxes = torch.tensor(
                    [
                        [d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]]
                        for d in detections
                    ]
                )
                labels = [
                    d.get("label") if d.get("label") in _COCO_CLASSES else None for d in detections
                ]
                scores = torch.tensor([d.get("score", 1.0) for d in detections])
            else:
                boxes = torch.zeros((0, 4))
                labels = []
                scores = torch.zeros((0,))
            rendered = transforms.ToPILImage()(image[index].movedim(-1, 0))
            if detections:
                rendered = cls.draw_detections(rendered, boxes, labels, scores)
            outputs.append(transforms.ToTensor()(rendered).unsqueeze(0).movedim(1, -1))
        return cls.outputs(out_image=torch.cat(outputs, dim=0).to(management.intermediate_device()))

    @classmethod
    def draw_detections(cls, image: Any, boxes: Any, labels: Any, scores: Any) -> Any:
        image_draw = importlib.import_module("PIL.ImageDraw")
        image_font = importlib.import_module("PIL.ImageFont")
        draw = image_draw.Draw(image)
        try:
            font = image_font.truetype("arial.ttf", 16)
        except Exception:
            font = image_font.load_default()
        colors = (
            (255, 0, 0),
            (0, 200, 0),
            (0, 0, 255),
            (255, 165, 0),
            (128, 0, 128),
            (0, 255, 255),
            (255, 20, 147),
            (100, 149, 237),
        )
        for box, label, score in sorted(
            zip(boxes, labels, scores, strict=True), key=lambda entry: entry[2].item()
        ):
            x1, y1, x2, y2 = box.tolist()
            color = colors[(_COCO_CLASSES.index(label) if label is not None else 0) % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            if label is not None:
                draw.text((x1 + 2, y1 + 2), f"{label} {score:.2f}", fill=color, font=font)
        return image


RTDETR_NODES = (RTDETR_detect, DrawBBoxes)
