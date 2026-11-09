"""First-party ComfyUI-origin utilities nodes."""

from dinkster_api.v1 import (
    COLOR_TYPE,
    RESOLUTION_PREVIEW_TYPE,
    TypeRegistry,
    inference_nodes,
    register_box_types,
)

from .color_resolution import (
    ColorToRGBInt,
    ResolutionSelector,
)
from .curve_editor import (
    CurveEditor,
)
from .image_crop import (
    PrimitiveBoundingBox,
)
from .list_utilities import (
    CreateList,
    GetItemFromList,
)
from .logic import (
    ComfyAndNode,
    ComfyNotNode,
    ComfyOrNode,
    CustomCombo,
)
from .loop_lists import ListElement, MakeList
from .loops import (
    EndLoop,
    LoopIteration,
    LoopProgress,
    LoopResult,
    StartLoop,
)
from .math import (
    ComfyMathExpression,
)
from .primitive import (
    PrimitiveBoolean,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from .switch import (
    ComfySwitchNode,
)
from .utilities import (
    ComfyNumberConvert,
    PreviewAny,
    SeedNode,
)

NODES = (
    ColorToRGBInt,
    ResolutionSelector,
    CurveEditor,
    PrimitiveBoundingBox,
    CreateList,
    GetItemFromList,
    ComfyNotNode,
    ComfyAndNode,
    ComfyOrNode,
    CustomCombo,
    StartLoop,
    LoopIteration,
    LoopProgress,
    LoopResult,
    EndLoop,
    ComfyMathExpression,
    PrimitiveString,
    PrimitiveStringMultiline,
    PrimitiveInt,
    PrimitiveFloat,
    PrimitiveBoolean,
    ComfySwitchNode,
    ComfyNumberConvert,
    SeedNode,
    PreviewAny,
    MakeList,
    ListElement,
)


def register_types(registry: TypeRegistry) -> None:
    register_box_types(registry)
    for type_id in (COLOR_TYPE, RESOLUTION_PREVIEW_TYPE):
        if type_id not in registry:
            registry.register(type_id)


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
