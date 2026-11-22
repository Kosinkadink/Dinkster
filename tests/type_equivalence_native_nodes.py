"""Isolated native pack for type-equivalence crossings."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from dinkster_schema import Input, Node, NodeSchema, Output, TypeExpr
from dinkster_values import (
    CORE_STRING,
    TypeRegistry,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    prepare_image_array_encoding,
)

NATIVE_IMAGE = "IMAGE"
NATIVE_MASK = "MASK"
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
            body=(Output("image", TypeExpr.concrete(NATIVE_IMAGE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(image=np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3))


class NativeMaskProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.test.mask_producer",
            body=(Output("mask", TypeExpr.concrete(NATIVE_MASK)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(mask=np.arange(6, dtype=np.float32).reshape(1, 2, 3))


def _shape(value: np.ndarray) -> str:
    return "x".join(str(dimension) for dimension in value.shape)


class NativeImageConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.test.image_consumer",
            body=(Input("image", TypeExpr.concrete(NATIVE_IMAGE)), Output("shape", STRING)),
        )

    @classmethod
    def execute(cls, *, image: np.ndarray) -> Mapping[str, object]:
        np.testing.assert_array_equal(image, np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3))
        return cls.outputs(shape=_shape(image))


class NativeMaskConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.test.mask_consumer",
            body=(Input("mask", TypeExpr.concrete(NATIVE_MASK)), Output("shape", STRING)),
        )

    @classmethod
    def execute(cls, *, mask: np.ndarray) -> Mapping[str, object]:
        np.testing.assert_array_equal(mask, np.arange(6, dtype=np.float32).reshape(1, 2, 3))
        return cls.outputs(shape=_shape(mask))


NODES = (
    NativeImageProducer,
    NativeMaskProducer,
    NativeImageConsumer,
    NativeMaskConsumer,
)
