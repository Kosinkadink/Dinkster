"""Native dataset collection nodes from the fixed ComfyUI reference."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    materialize_image,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_IMAGES = TypeExpr.list_of(_IMAGE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_STRINGS = TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))


def _frames(images: list[Any]) -> list[Any]:
    frames: list[Any] = []
    for value in images:
        image = materialize_image(value)
        if image.ndim != 4:
            raise ValueError(
                f"Expected 4D image tensor, got {type(image).__name__} shape {image.shape}"
            )
        frames.extend(image[index : index + 1] for index in range(image.shape[0]))
    return frames


class ShuffleDataset(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ShuffleDataset",
            display_name="Shuffle Images List",
            category="image/batch",
            description="Randomly shuffle the order of images in a list.",
            search_terms=("shuffle", "randomize", "mix"),
            inputs=(
                InputSpec("images", _IMAGES, doc="List of images to process."),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    doc="Random seed.",
                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                ),
            ),
            outputs=(OutputSpec("images", _IMAGES, doc="Processed images"),),
        )

    @classmethod
    def execute(cls, *, images: list[Any], seed: int) -> Mapping[str, object]:
        numpy = importlib.import_module("numpy")
        images = _frames(images)
        indices = numpy.random.RandomState(seed % (2**32 - 1)).permutation(len(images))
        return cls.outputs(images=[images[index] for index in indices])


class ShuffleImageTextDataset(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ShuffleImageTextDataset",
            display_name="Shuffle Pairs of Image-Text",
            category="image/batch",
            description="Randomly shuffle the order of pairs of image-text in a list.",
            search_terms=("shuffle", "randomize", "mix"),
            inputs=(
                InputSpec("images", _IMAGES, doc="List of images to shuffle."),
                InputSpec("texts", _STRINGS, doc="List of texts to shuffle.", force_input=True),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    doc="Random seed.",
                    widget=NumberWidget(min=0, max=0xFFFFFFFFFFFFFFFF),
                ),
            ),
            outputs=(
                OutputSpec("images", _IMAGES, doc="Shuffled images"),
                OutputSpec("texts", _STRINGS, doc="Shuffled texts"),
            ),
        )

    @classmethod
    def execute(cls, *, images: list[Any], texts: list[str], seed: int) -> Mapping[str, object]:
        numpy = importlib.import_module("numpy")
        indices = tuple(
            int(index)
            for index in numpy.random.RandomState(seed % (2**32 - 1)).permutation(len(images))
        )
        return cls.outputs(
            images=[images[index] for index in indices], texts=[texts[index] for index in indices]
        )


class ImageDeduplication(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageDeduplication",
            display_name="Deduplicate Images",
            category="image/batch",
            description="Remove duplicate or very similar images from a list.",
            search_terms=("deduplicate", "remove duplicates", "similarity filter"),
            inputs=(
                InputSpec("images", _IMAGES, doc="List of images to process."),
                InputSpec(
                    "similarity_threshold",
                    _FLOAT,
                    default=0.95,
                    doc=(
                        "Similarity threshold (0-1). Higher means more similar. Images above this "
                        "threshold are considered duplicates."
                    ),
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0),
                ),
            ),
            outputs=(OutputSpec("images", _IMAGES, doc="Processed images"),),
        )

    @classmethod
    def execute(cls, *, images: list[Any], similarity_threshold: float) -> Mapping[str, object]:
        numpy = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
        images = _frames(images)
        if not images:
            return cls.outputs(images=[])

        def image_hash(value: Any) -> str:
            image = materialize_image(value)
            if image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
            pixels = list(
                image_module.fromarray((image.cpu().numpy() * 255).clip(0, 255).astype(numpy.uint8))
                .resize((8, 8), image_module.Resampling.LANCZOS)
                .convert("L")
                .getdata()
            )
            average = sum(pixels) / len(pixels)
            return "".join("1" if pixel > average else "0" for pixel in pixels)

        hashes = [image_hash(image) for image in images]
        kept: list[int] = []
        for index, value in enumerate(hashes):
            if not any(
                1.0
                - sum(left != right for left, right in zip(value, hashes[other], strict=True))
                / 64.0
                >= similarity_threshold
                for other in kept
            ):
                kept.append(index)
        return cls.outputs(images=[images[index] for index in kept])


class ImageGrid(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageGrid",
            display_name="Make Image Grid",
            category="image/batch",
            description="Arrange multiple images into a grid layout.",
            search_terms=("grid", "collage", "combine"),
            inputs=(
                InputSpec("images", _IMAGES, doc="List of images to process."),
                InputSpec("columns", _INT, default=4, widget=NumberWidget(min=1, max=20)),
                InputSpec(
                    "cell_width",
                    _INT,
                    default=256,
                    advanced=True,
                    widget=NumberWidget(min=32, max=2048),
                ),
                InputSpec(
                    "cell_height",
                    _INT,
                    default=256,
                    advanced=True,
                    widget=NumberWidget(min=32, max=2048),
                ),
                InputSpec(
                    "padding", _INT, default=4, advanced=True, widget=NumberWidget(min=0, max=50)
                ),
            ),
            outputs=(OutputSpec("images", _IMAGE, doc="Processed images"),),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: list[Any],
        columns: int,
        cell_width: int,
        cell_height: int,
        padding: int,
    ) -> Mapping[str, object]:
        image_module = importlib.import_module("PIL.Image")
        numpy = importlib.import_module("numpy")
        images = _frames(images)
        if not images:
            raise ValueError("Cannot create grid from empty image list")
        rows = (len(images) + columns - 1) // columns
        grid = image_module.new(
            "RGB",
            (
                columns * cell_width + (columns - 1) * padding,
                rows * cell_height + (rows - 1) * padding,
            ),
            (0, 0, 0),
        )
        for index, value in enumerate(images):
            image = materialize_image(value)
            if image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
            tile = image_module.fromarray(
                (image.cpu().numpy() * 255).clip(0, 255).astype(numpy.uint8)
            ).resize((cell_width, cell_height), image_module.Resampling.LANCZOS)
            grid.paste(
                tile,
                (
                    (index % columns) * (cell_width + padding),
                    (index // columns) * (cell_height + padding),
                ),
            )
        torch = importlib.import_module("torch")
        return cls.outputs(
            images=torch.from_numpy(numpy.array(grid).astype(numpy.float32) / 255.0)[None,]
        )


class MergeTextLists(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MergeTextLists",
            display_name="Merge Text Lists (DEPRECATED)",
            category="text",
            description="Concatenate multiple text lists into one.",
            deprecation=Deprecation("Deprecated in ComfyUI."),
            search_visibility="deprecated",
            inputs=(InputSpec("texts", _STRINGS, widget=StringWidget(multiline=False)),),
            outputs=(OutputSpec("texts", TypeExpr.concrete(CORE_STRING), doc="Processed texts"),),
        )

    @classmethod
    def execute(cls, *, texts: list[str]) -> Mapping[str, object]:
        return cls.outputs(texts=[texts])


DATASET_COLLECTION_NODES = (
    ShuffleDataset,
    ShuffleImageTextDataset,
    ImageDeduplication,
    ImageGrid,
    MergeTextLists,
)
