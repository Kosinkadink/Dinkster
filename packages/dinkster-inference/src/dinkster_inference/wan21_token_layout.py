"""Torch-free Wan 2.1 target-video token geometry declarations.

The patch geometry, padding, and flatten order follow
``comfy/ldm/wan/model.py`` at ComfyUI ``b78cec87``: pad the latent to the
``(1, 2, 2)`` patch, apply Conv3d, then flatten contiguous T/H/W axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .token_layout import ModelTokenLayout, ModelTokenSegment, TokenGridTransform

__all__ = [
    "Wan21TokenLayoutError",
    "Wan21TokenLayoutPlan",
    "Wan21VideoLatentGeometry",
    "plan_wan21_token_layout",
]

_PATCH = (1, 2, 2)
_FLATTEN_ORDER = ("t", "h", "w")
_TRANSFORM = "wan21.video-patch-1x2x2-row-major-thw.v1"


class Wan21TokenLayoutError(ValueError):
    pass


def _positive_exact_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise Wan21TokenLayoutError(f"{name} must be an exact int >= 1")


@dataclass(frozen=True, slots=True)
class Wan21VideoLatentGeometry:
    """One unpadded 5D latent's T/H/W content geometry."""

    temporal: int
    height: int
    width: int

    def __post_init__(self) -> None:
        for name, value in (
            ("temporal", self.temporal),
            ("height", self.height),
            ("width", self.width),
        ):
            _positive_exact_int(name, value)

    @property
    def token_grid(self) -> tuple[int, int, int]:
        return (self.temporal, (self.height + 1) // 2, (self.width + 1) // 2)


@dataclass(frozen=True, slots=True, init=False)
class Wan21TokenLayoutPlan:
    """One target-only layout and its source-geometry transform fact."""

    layout: ModelTokenLayout
    transforms: tuple[TokenGridTransform, ...]
    patch: tuple[int, int, int] = field(default=_PATCH, init=False)
    flatten_order: tuple[str, str, str] = field(default=_FLATTEN_ORDER, init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Wan 2.1 token layout plans are produced only by the family planner")


def plan_wan21_token_layout(geometry: Wan21VideoLatentGeometry) -> Wan21TokenLayoutPlan:
    """Declare the one row-major T/H/W target segment for a latent."""

    if type(geometry) is not Wan21VideoLatentGeometry:
        raise Wan21TokenLayoutError("geometry must be an exact Wan21VideoLatentGeometry value")
    grid = geometry.token_grid
    rows = grid[0] * grid[1] * grid[2]
    layout = ModelTokenLayout(
        (ModelTokenSegment("target-video", "video", "target", 0, rows, grid),),
        padded_rows=0,
    )
    transforms = (
        TokenGridTransform(
            _TRANSFORM,
            "video",
            "target-video",
            (geometry.temporal, geometry.height, geometry.width),
            None,
        ),
    )
    plan = object.__new__(Wan21TokenLayoutPlan)
    object.__setattr__(plan, "layout", layout)
    object.__setattr__(plan, "transforms", transforms)
    object.__setattr__(plan, "patch", _PATCH)
    object.__setattr__(plan, "flatten_order", _FLATTEN_ORDER)
    return plan
