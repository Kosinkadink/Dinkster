"""Native Dinkster implementations of ComfyUI core nodes."""

from dinkster_api.v1 import TypeRegistry

from .primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from .text import TEXT_NODES

COMFY_NODES = (
    *TEXT_NODES,
    PrimitiveString,
    PrimitiveStringMultiline,
    PrimitiveInt,
    PrimitiveFloat,
    PrimitiveBoolean,
)


def register_comfy_types(registry: TypeRegistry) -> None:
    """Container values use the registry's general Python value codec."""
    registry.register("DICT")
    registry.register("ARRAY")
