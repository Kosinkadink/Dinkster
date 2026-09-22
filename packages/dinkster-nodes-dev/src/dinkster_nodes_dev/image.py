"""Image nodes over numpy arrays, plus the dev.image value type registration.

The type registration demonstrates the opt-in path (DESIGN 3.2): dev.image
declares a codec (npy bytes), a fast fingerprint, and interrogable meta -
because images are hot. A lazier pack could register with nothing but a name
and still work. The codec and PNG rendition are the shared image-array
contract (dinkster_values.image_codec, re-exported by the v1 API) - the same
bytes and renderer the compat surface uses for comfy.IMAGE.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    PNG_CONTAINER_VERSION,
    SAVE_TARGET_TYPE,
    AssetError,
    AssetWriter,
    InputSpec,
    MountSnapshotWriter,
    Node,
    NodeSchema,
    OutputSpec,
    SaveTargetWidget,
    TypeExpr,
    TypeRegistry,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    image_input,
    register_save_target_type,
    render_image_png,
)

DEV_IMAGE = "dev.image"
IMAGE = TypeExpr.concrete(DEV_IMAGE)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)


def register_dev_types(registry: TypeRegistry) -> None:
    # save_pgm speaks dinkster.save_target; register it unless a co-composed
    # pack (or the host) already did - the type is one shared definition.
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)
    registry.register(
        DEV_IMAGE,
        encode=encode_image_array,
        decode=decode_image_array,
        fingerprint=image_array_fingerprint(DEV_IMAGE),
        meta=image_array_meta,
        input_convert=image_input,
    )
    registry.register_rendition(
        DEV_IMAGE,
        "png",
        mime="image/png",
        render=render_image_png,
        version=PNG_CONTAINER_VERSION,
    )


class GradientImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.image.gradient",
            display_name="Gradient Image",
            category="dev/image",
            inputs=(InputSpec("width", INT, default=256), InputSpec("height", INT, default=256)),
            outputs=(OutputSpec("image", IMAGE),),
        )

    @classmethod
    def execute(cls, *, width: int, height: int) -> Mapping[str, object]:
        row = np.linspace(0.0, 1.0, num=width, dtype=np.float32)
        return cls.outputs(image=np.tile(row, (height, 1)))  # pyright: ignore[reportUnknownMemberType]


class InvertImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.image.invert",
            display_name="Invert Image",
            category="dev/image",
            inputs=(InputSpec("image", IMAGE),),
            outputs=(OutputSpec("image", IMAGE),),
        )

    @classmethod
    def execute(cls, *, image: np.ndarray) -> Mapping[str, object]:
        return cls.outputs(image=1.0 - image)


class BlendImages(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.image.blend",
            display_name="Blend Images",
            category="dev/image",
            inputs=(
                InputSpec("a", IMAGE),
                InputSpec("b", IMAGE),
                InputSpec("ratio", FLOAT, default=0.5),
            ),
            outputs=(OutputSpec("image", IMAGE),),
        )

    @classmethod
    def execute(cls, *, a: np.ndarray, b: np.ndarray, ratio: float) -> Mapping[str, object]:
        return cls.outputs(image=a * (1.0 - ratio) + b * ratio)


class ImageStats(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.image.stats",
            display_name="Image Stats",
            category="dev/image",
            inputs=(InputSpec("image", IMAGE),),
            outputs=(
                OutputSpec("mean", FLOAT),
                OutputSpec("minimum", FLOAT),
                OutputSpec("maximum", FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, image: np.ndarray) -> Mapping[str, object]:
        return cls.outputs(
            mean=float(image.mean()),
            minimum=float(image.min()),
            maximum=float(image.max()),
        )


class ImageBatchToList(Node):
    """batch -> list<image> (DESIGN 3.13): the tensor-batch-vs-python-list
    distinction as a visible node instead of folklore. Splits the leading
    axis; each element is an independent list child with its own fingerprint,
    so downstream per-item caching works."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.image.batch_to_list",
            display_name="Image Batch to List",
            category="dev/image",
            inputs=(InputSpec("batch", IMAGE),),
            outputs=(OutputSpec("images", TypeExpr.list_of(IMAGE)),),
        )

    @classmethod
    def execute(cls, *, batch: np.ndarray) -> Mapping[str, object]:
        if batch.ndim < 2:
            raise ValueError(f"a batch needs a leading batch axis; got shape {batch.shape}")
        return cls.outputs(images=[np.ascontiguousarray(item) for item in batch])


class ImageListToBatch(Node):
    """list<image> -> batch: stacks along a new leading axis. All elements
    must share one shape - mismatches are loud node errors, never silent
    padding or clamping. Empty lists have no shape to stack into."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.image.list_to_batch",
            display_name="Image List to Batch",
            category="dev/image",
            inputs=(InputSpec("images", TypeExpr.list_of(IMAGE)),),
            outputs=(OutputSpec("batch", IMAGE),),
        )

    @classmethod
    def execute(cls, *, images: Sequence[np.ndarray]) -> Mapping[str, object]:
        if not images:
            raise ValueError("cannot batch an empty list")
        shapes = {item.shape for item in images}
        if len(shapes) > 1:
            raise ValueError(
                "all images in a batch must share one shape, got "
                + ", ".join(sorted(str(s) for s in shapes))
            )
        return cls.outputs(
            batch=np.stack([np.asarray(item) for item in images])  # pyright: ignore[reportUnknownMemberType]
        )


def _mount_writer() -> AssetWriter:
    """Save nodes write through mounts, never raw paths (mounts are the
    unit of filesystem authority). The engine publishes granted mounts as
    a snapshot file; the writer re-reads it per save, so grants and
    revokes apply to the next save with no restart."""
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError(
            "saving requires filesystem mounts, but this process has no "
            "DINKSTER_MOUNTS_SNAPSHOT configured"
        )
    return AssetWriter(MountSnapshotWriter(snapshot))


class SaveImagePGM(Node):
    """Writes a grayscale PGM into a granted readwrite mount.

    The destination is a structured ``dinkster.save_target`` ({mount, prefix})
    - never a raw host path, which would let any graph write anywhere the
    server process can. Side-effecting, so declared non-idempotent: the
    engine will never serve it from cache."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.image.save_pgm",
            display_name="Save Image (PGM)",
            category="dev/image",
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("target", SAVE_TARGET, widget=SaveTargetWidget(".pgm")),
            ),
            outputs=(OutputSpec("path", STRING), OutputSpec("digest", STRING)),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, image: np.ndarray, target: object) -> Mapping[str, object]:
        gray = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)  # pyright: ignore[reportUnknownMemberType]
        pixels = (gray * 255).astype(np.uint8)
        height, width = pixels.shape
        data = f"P5 {width} {height} 255\n".encode("ascii") + pixels.tobytes()
        ref = _mount_writer().save_bytes(
            target, data, suffix=".pgm", media_type="image/x-portable-graymap"
        )
        return cls.outputs(path=ref.virtual_path, digest=ref.digest)
