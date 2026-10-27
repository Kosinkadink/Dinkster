"""Image datasets read from invocation-scoped, granted folder selections."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_STRING,
    IMAGE_TYPE,
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    resolve_asset_selection,
)

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_ALIASES = ("load folder", "load from folder", "load dataset", "load images", "import dataset")
_IMAGES = OutputSpec(
    "images", TypeExpr.list_of(TypeExpr.concrete(IMAGE_TYPE)), doc="List of loaded images"
)
_FOLDER = InputSpec(
    "folder",
    TypeExpr.concrete(CORE_COMBO),
    widget=AssetWidget(kind="directory/input", selection=True),
    doc="The folder to load images from.",
)


def _load_images(files: list[Path]) -> list[Any]:
    if not files:
        raise ValueError("No valid images found in input")
    image_module = importlib.import_module("PIL.Image")
    image_file: Any = importlib.import_module("PIL.ImageFile")
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    images: list[Any] = []

    def scale_integer(value: float) -> float:
        return value * (1 / 255)

    for path in files:
        try:
            source = image_module.open(path)
        except (OSError, image_module.UnidentifiedImageError, ValueError):
            previous = image_file.LOAD_TRUNCATED_IMAGES
            try:
                image_file.LOAD_TRUNCATED_IMAGES = True
                source = image_module.open(path)
            finally:
                image_file.LOAD_TRUNCATED_IMAGES = previous
        with source:
            image = source.point(scale_integer) if source.mode == "I" else source
            pixels = numpy.array(image.convert("RGB")).astype(numpy.float32) / 255.0
            images.append(torch.from_numpy(pixels)[None])
    return images


class LoadImageDataSetFromFolder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadImageDataSetFromFolder",
            display_name="Load Image (from Folder)",
            category="image",
            description=(
                "Load a dataset of images from a specified folder and return a list of images. "
                "Supported formats: PNG, JPG, JPEG, WEBP."
            ),
            search_terms=_ALIASES,
            inputs=(_FOLDER,),
            outputs=(_IMAGES,),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, folder: str) -> Mapping[str, object]:
        directory = resolve_asset_selection("directory/input", folder)
        if not isinstance(directory, Path):
            raise TypeError("image dataset selection requires a directory subtree")
        files = [
            directory / name for name in os.listdir(directory) if name.lower().endswith(_EXTENSIONS)
        ]
        return cls.outputs(images=_load_images(files))


class LoadImageTextDataSetFromFolder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadImageTextDataSetFromFolder",
            display_name="Load Image-Text (from Folder)",
            category="image",
            description=(
                "Load a dataset of pairs of images and text captions from a specified folder "
                "and return them as a list. Supported formats: PNG, JPG, JPEG, WEBP."
            ),
            search_terms=_ALIASES,
            inputs=(
                InputSpec(
                    "folder",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=AssetWidget(kind="directory/input", selection=True),
                    doc="The folder to load images and text captions from.",
                ),
            ),
            outputs=(
                _IMAGES,
                OutputSpec(
                    "texts",
                    TypeExpr.list_of(TypeExpr.concrete(CORE_STRING)),
                    doc="List of text captions",
                ),
            ),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, folder: str) -> Mapping[str, object]:
        directory = resolve_asset_selection("directory/input", folder)
        if not isinstance(directory, Path):
            raise TypeError("image dataset selection requires a directory subtree")
        files: list[Path] = []
        for name in os.listdir(directory):
            path = directory / name
            if name.lower().endswith(_EXTENSIONS):
                files.append(path)
            elif path.is_dir():
                prefix = name.split("_")[0]
                repeat = int(prefix) if prefix.isdigit() else 1
                files.extend(
                    [
                        path / child
                        for child in os.listdir(path)
                        if child.lower().endswith(_EXTENSIONS)
                    ]
                    * repeat
                )
        captions: list[str] = []
        for path in files:
            caption = Path(str(path).replace(path.suffix, ".txt"))
            # The source replaces every suffix occurrence, including directory names.
            caption.relative_to(directory)
            captions.append(caption.read_text(encoding="utf-8").strip() if caption.exists() else "")
        return cls.outputs(images=_load_images(files), texts=captions)


FOLDER_DATASET_NODES = (LoadImageDataSetFromFolder, LoadImageTextDataSetFromFolder)
