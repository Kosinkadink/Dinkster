"""Native ComfyUI dataset image transform nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_MAX_RESOLUTION = 8192


def _integer(
    name: str,
    default: int,
    minimum: int = 1,
    maximum: int = _MAX_RESOLUTION,
    *,
    doc: str = "",
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        doc=doc,
        widget=NumberWidget(min=minimum, max=maximum, step=1),
    )


def _float(name: str, default: float, minimum: float, maximum: float, *, doc: str) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=default,
        doc=doc,
        widget=NumberWidget(min=minimum, max=maximum),
    )


def _schema(
    node_type: str, display_name: str, category: str, inputs: tuple[InputSpec, ...]
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=category,
        inputs=(InputSpec("images", _IMAGE, doc="Image to process."), *inputs),
        outputs=(OutputSpec("images", _IMAGE),),
    )


def _pil_from_tensor(image: Any) -> Any:
    image_module = importlib.import_module("PIL.Image")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    array = (image.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return image_module.fromarray(array)


def _tensor_from_pil(image: Any) -> Any:
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    return torch.from_numpy(numpy.array(image).astype("float32") / 255.0).unsqueeze(0)


def _spatial_transform(images: Any, transform: Any) -> Any:
    torch = importlib.import_module("torch")
    images = materialize_image(images)
    return torch.cat(
        [transform(images[index : index + 1]) for index in range(images.shape[0])], dim=0
    )


class ResizeImagesByShorterEdge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ResizeImagesByShorterEdge",
            "Resize Images by Shorter Edge (DEPRECATED)",
            "image/transform",
            (_integer("shorter_edge", 512, doc="Target dimension for the shorter edge."),),
        )

    @classmethod
    def execute(cls, *, images: Any, shorter_edge: int) -> Mapping[str, object]:
        image_module = importlib.import_module("PIL.Image")

        def resize(image: Any) -> Any:
            pil = _pil_from_tensor(image)
            width, height = pil.size
            if width < height:
                size = (shorter_edge, int(height * (shorter_edge / width)))
            else:
                size = (int(width * (shorter_edge / height)), shorter_edge)
            return _tensor_from_pil(pil.resize(size, image_module.Resampling.LANCZOS))

        return cls.outputs(images=_spatial_transform(images, resize))


class ResizeImagesByLongerEdge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "ResizeImagesByLongerEdge",
            "Resize Images by Longer Edge (DEPRECATED)",
            "image/transform",
            (_integer("longer_edge", 1024, doc="Target dimension for the longer edge."),),
        )

    @classmethod
    def execute(cls, *, images: Any, longer_edge: int) -> Mapping[str, object]:
        image_module = importlib.import_module("PIL.Image")

        def resize(image: Any) -> Any:
            pil = _pil_from_tensor(image)
            width, height = pil.size
            if width > height:
                size = (longer_edge, int(height * (longer_edge / width)))
            else:
                size = (int(width * (longer_edge / height)), longer_edge)
            return _tensor_from_pil(pil.resize(size, image_module.Resampling.LANCZOS))

        return cls.outputs(images=_spatial_transform(images, resize))


class CenterCropImages(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "CenterCropImages",
            "Crop Image (Center)",
            "image/transform",
            (
                _integer("width", 512, doc="Crop width."),
                _integer("height", 512, doc="Crop height."),
            ),
        )

    @classmethod
    def execute(cls, *, images: Any, width: int, height: int) -> Mapping[str, object]:
        def crop(image: Any) -> Any:
            pil = _pil_from_tensor(image)
            left = max(0, (pil.width - width) // 2)
            top = max(0, (pil.height - height) // 2)
            return _tensor_from_pil(
                pil.crop((left, top, min(pil.width, left + width), min(pil.height, top + height)))
            )

        return cls.outputs(images=_spatial_transform(images, crop))


class RandomCropImages(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "RandomCropImages",
            "Crop Image (Random)",
            "image/transform",
            (
                _integer("width", 512, doc="Crop width."),
                _integer("height", 512, doc="Crop height."),
                _integer("seed", 0, 0, 0xFFFFFFFFFFFFFFFF, doc="Random seed."),
            ),
        )

    @classmethod
    def execute(cls, *, images: Any, width: int, height: int, seed: int) -> Mapping[str, object]:
        numpy = importlib.import_module("numpy")

        def crop(image: Any) -> Any:
            numpy.random.seed(seed % (2**32 - 1))
            pil = _pil_from_tensor(image)
            max_left, max_top = max(0, pil.width - width), max(0, pil.height - height)
            left = numpy.random.randint(0, max_left + 1) if max_left else 0
            top = numpy.random.randint(0, max_top + 1) if max_top else 0
            return _tensor_from_pil(
                pil.crop((left, top, min(pil.width, left + width), min(pil.height, top + height)))
            )

        return cls.outputs(images=_spatial_transform(images, crop))


class NormalizeImages(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "NormalizeImages",
            "Normalize Image Colors",
            "image/color",
            (
                _float("mean", 0.5, 0.0, 1.0, doc="Mean value for normalization."),
                _float("std", 0.5, 0.001, 1.0, doc="Standard deviation for normalization."),
            ),
        )

    @classmethod
    def execute(cls, *, images: Any, mean: float, std: float) -> Mapping[str, object]:
        images = materialize_image(images)
        result = (images - mean) / std
        if images.shape[-1] == 4:
            result[..., 3] = images[..., 3]
        return cls.outputs(images=result)


class AdjustBrightness(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "AdjustBrightness",
            "Adjust Brightness",
            "image/adjustments",
            (
                _float(
                    "factor",
                    1.0,
                    0.0,
                    2.0,
                    doc="Brightness factor. 1.0 = no change, <1.0 = darker, >1.0 = brighter.",
                ),
            ),
        )

    @classmethod
    def execute(cls, *, images: Any, factor: float) -> Mapping[str, object]:
        images = materialize_image(images)
        result = (images * factor).clamp(0.0, 1.0)
        if images.shape[-1] == 4:
            result[..., 3] = images[..., 3]
        return cls.outputs(images=result)


class AdjustContrast(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "AdjustContrast",
            "Adjust Contrast",
            "image/adjustments",
            (
                _float(
                    "factor",
                    1.0,
                    0.0,
                    2.0,
                    doc=(
                        "Contrast factor. 1.0 = no change, <1.0 = less contrast, "
                        ">1.0 = more contrast."
                    ),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, images: Any, factor: float) -> Mapping[str, object]:
        images = materialize_image(images)
        result = ((images - 0.5) * factor + 0.5).clamp(0.0, 1.0)
        if images.shape[-1] == 4:
            result[..., 3] = images[..., 3]
        return cls.outputs(images=result)


DATASET_IMAGE_NODES = (
    ResizeImagesByShorterEdge,
    ResizeImagesByLongerEdge,
    CenterCropImages,
    RandomCropImages,
    NormalizeImages,
    AdjustBrightness,
    AdjustContrast,
)
