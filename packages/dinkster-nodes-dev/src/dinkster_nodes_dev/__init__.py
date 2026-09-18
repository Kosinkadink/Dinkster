"""dinkster-nodes-dev: test/demo scaffolding, deliberately not user-facing.

Everything here exists so demos, engine tests, and the real-process E2E
can exercise core machinery (values, renditions, lists, scheduling, the
mounted write gate) without torch or PIL. dinkster-serve composes this set
in-process only under --dev; a user's node surface never contains dev.*
nodes. The default first-party packs are the user-facing counterpart.
"""

from dinkster_api.v1 import TypeRegistry

from .conformance import CONFORMANCE_NODES
from .gallery import (
    GALLERY_NODES,
    GalleryAssets,
    GalleryExoticOut,
    GalleryLists,
    GalleryMatch,
    GallerySockets,
    GallerySource,
    GalleryWidgets,
    combo_choices,
    register_gallery_types,
)
from .image import (
    DEV_IMAGE,
    BlendImages,
    GradientImage,
    ImageBatchToList,
    ImageListToBatch,
    ImageStats,
    InvertImage,
    SaveImagePGM,
)
from .image import register_dev_types as _register_image_types
from .util import Delay

PACK_NODES = [
    Delay,
    GradientImage,
    InvertImage,
    BlendImages,
    ImageStats,
    ImageBatchToList,
    ImageListToBatch,
    SaveImagePGM,
    *GALLERY_NODES,
]
"""Manifest-loadable dev examples. Conformance probes are excluded because
manifests can be explicitly composed outside ``--dev``."""

DEV_NODES = [*PACK_NODES, *CONFORMANCE_NODES]
"""Explicit ``--dev`` composition, including internal conformance probes."""


def register_dev_types(registry: TypeRegistry) -> None:
    """The pack's one type entry point: the dev.image value machinery plus
    the gallery's marker/shared types."""
    _register_image_types(registry)
    register_gallery_types(registry)


__all__ = [
    "DEV_IMAGE",
    "DEV_NODES",
    "CONFORMANCE_NODES",
    "GALLERY_NODES",
    "PACK_NODES",
    "BlendImages",
    "Delay",
    "GalleryAssets",
    "GalleryExoticOut",
    "GalleryLists",
    "GalleryMatch",
    "GallerySockets",
    "GallerySource",
    "GalleryWidgets",
    "GradientImage",
    "ImageBatchToList",
    "ImageListToBatch",
    "ImageStats",
    "InvertImage",
    "SaveImagePGM",
    "combo_choices",
    "register_dev_types",
    "register_gallery_types",
]
