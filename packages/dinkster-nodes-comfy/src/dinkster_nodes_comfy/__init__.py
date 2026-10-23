"""Native Dinkster implementations of ComfyUI core nodes."""

from .logic import LOGIC_NODES
from .primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from .text import TEXT_NODES
from .utilities import ComfyNumberConvert, SeedNode

COMFY_NODES = (
    *TEXT_NODES,
    PrimitiveString,
    PrimitiveStringMultiline,
    PrimitiveInt,
    PrimitiveFloat,
    PrimitiveBoolean,
    *LOGIC_NODES,
    ComfyNumberConvert,
    SeedNode,
)
