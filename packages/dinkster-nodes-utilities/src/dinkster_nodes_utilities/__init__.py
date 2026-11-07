"""First-party ComfyUI-origin utilities nodes."""

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
)

__all__ = ["NODES"]
