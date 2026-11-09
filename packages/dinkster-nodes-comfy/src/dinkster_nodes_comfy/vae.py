"""Core image and video VAE encoding and decoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)


def _schema(node_type: str, display_name: str, *, encode: bool, tiled: bool) -> NodeSchema:
    inputs: tuple[InputSpec, ...] = (
        InputSpec(
            "pixels" if encode else "samples",
            TypeExpr.concrete(IMAGE_TYPE if encode else LATENT_TYPE),
        ),
        InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
    )
    if tiled:
        inputs += tuple(
            InputSpec(
                name,
                TypeExpr.concrete(CORE_INT),
                default=default,
                widget=NumberWidget(min=minimum, max=4096, step=step),
                advanced=True,
            )
            for name, default, minimum, step in (
                ("tile_size", 512, 64, 64 if encode else 32),
                ("overlap", 64, 0, 32),
                ("temporal_size", 64, 8, 4),
                ("temporal_overlap", 8, 4, 4),
            )
        )
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="model/latent",
        search_terms=()
        if tiled
        else (
            ("encode", "encode image", "image to latent")
            if encode
            else ("decode", "decode latent", "latent to image", "render latent")
        ),
        inputs=inputs,
        outputs=(
            OutputSpec(
                "LATENT" if encode else "IMAGE",
                TypeExpr.concrete(LATENT_TYPE if encode else IMAGE_TYPE),
            ),
        ),
    )


def _samples(samples: object) -> Any:
    latent = materialize_image(materialize_latent(samples)["samples"])
    return latent.unbind()[0] if latent.is_nested else latent


def _images(images: Any) -> Any:
    if len(images.shape) == 5:
        return images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


class VAEDecode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("VAEDecode", "VAE Decode", encode=False, tiled=False)

    @classmethod
    def execute(cls, *, vae: Any, samples: object) -> Mapping[str, object]:
        return cls.outputs(IMAGE=_images(vae.decode(_samples(samples))))


class VAEEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("VAEEncode", "VAE Encode", encode=True, tiled=False)

    @classmethod
    def execute(cls, *, vae: Any, pixels: object) -> Mapping[str, object]:
        return cls.outputs(LATENT={"samples": vae.encode(materialize_image(pixels))})


class VAEDecodeTiled(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("VAEDecodeTiled", "VAE Decode (Tiled)", encode=False, tiled=True)

    @classmethod
    def execute(
        cls,
        *,
        vae: Any,
        samples: object,
        tile_size: int = 512,
        overlap: int = 64,
        temporal_size: int = 64,
        temporal_overlap: int = 8,
    ) -> Mapping[str, object]:
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        compression = vae.temporal_compression_decode()
        if compression is not None:
            tile_t = max(2, temporal_size // compression)
            overlap_t = max(1, min(tile_t // 2, temporal_overlap // compression))
        else:
            tile_t = None
            overlap_t = None
        spatial = vae.spacial_compression_decode()
        images = vae.decode_tiled(
            _samples(samples),
            tile_x=tile_size // spatial,
            tile_y=tile_size // spatial,
            overlap=overlap // spatial,
            tile_t=tile_t,
            overlap_t=overlap_t,
        )
        return cls.outputs(IMAGE=_images(images))


class VAEEncodeTiled(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("VAEEncodeTiled", "VAE Encode (Tiled)", encode=True, tiled=True)

    @classmethod
    def execute(
        cls,
        *,
        vae: Any,
        pixels: object,
        tile_size: int = 512,
        overlap: int = 64,
        temporal_size: int = 64,
        temporal_overlap: int = 8,
    ) -> Mapping[str, object]:
        samples = vae.encode_tiled(
            materialize_image(pixels),
            tile_x=tile_size,
            tile_y=tile_size,
            overlap=overlap,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )
        return cls.outputs(LATENT={"samples": samples})


VAE_NODES = (VAEDecode, VAEEncode, VAEDecodeTiled, VAEEncodeTiled)
