"""Isolated native pack for type-equivalence crossings."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from dinkster_nodes_image import ImageResize
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import (
    CORE_STRING,
    TypeRegistry,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    prepare_image_array_encoding,
)

NATIVE_IMAGE = "dinkster.image"
NATIVE_MASK = "dinkster.mask"
STRING = TypeExpr.concrete(CORE_STRING)


def register_types(registry: TypeRegistry) -> None:
    for type_id in (NATIVE_IMAGE, NATIVE_MASK):
        registry.register(
            type_id,
            encode=encode_image_array,
            decode=decode_image_array,
            prepare_buffer_encoding=prepare_image_array_encoding,
            fingerprint=image_array_fingerprint(type_id),
            meta=image_array_meta,
        )


class NativeImageProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.test.image_producer",
            outputs=(OutputSpec("image", TypeExpr.concrete(NATIVE_IMAGE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(image=np.zeros((1, 2, 3, 3), dtype=np.float32))


class NativeMaskProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.test.mask_producer",
            outputs=(OutputSpec("mask", TypeExpr.concrete(NATIVE_MASK)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(mask=np.zeros((1, 2, 3), dtype=np.float32))


def _shape(value: np.ndarray) -> str:
    return "x".join(str(dimension) for dimension in value.shape)


class NativeImageConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.test.image_consumer",
            inputs=(InputSpec("image", TypeExpr.concrete(NATIVE_IMAGE)),),
            outputs=(OutputSpec("shape", STRING),),
        )

    @classmethod
    def execute(cls, *, image: np.ndarray) -> Mapping[str, object]:
        return cls.outputs(shape=_shape(image))


class NativeMaskConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.test.mask_consumer",
            inputs=(InputSpec("mask", TypeExpr.concrete(NATIVE_MASK)),),
            outputs=(OutputSpec("shape", STRING),),
        )

    @classmethod
    def execute(cls, *, mask: np.ndarray) -> Mapping[str, object]:
        return cls.outputs(shape=_shape(mask))


NODES = (
    NativeImageProducer,
    NativeMaskProducer,
    ImageResize,
    NativeImageConsumer,
    NativeMaskConsumer,
)
