from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    MODEL_TYPE,
    ComboWidget,
    InputSpec,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    with_identity_replacement,
)
from dinkster_models.context_windows import ContextWindowsManual

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)

_COMBO = TypeExpr.concrete(CORE_COMBO)

_INT = TypeExpr.concrete(CORE_INT)

_MODEL = TypeExpr.concrete(MODEL_TYPE)

_SCHEDULES = ("standard_static", "standard_uniform", "looped_uniform", "batched")

_FUSE_METHODS = ("pyramid", "relative", "flat", "overlap-linear")


class WanContextWindowsManual(ContextWindowsManual):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.wan.wan_context_windows_manual",
            display_name="Wan Context Windows",
            category="model/patch/wan",
            description="Set context windows for Wan-like models.",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "context_length",
                    _INT,
                    default=81,
                    widget=NumberWidget(min=1, max=16384, step=4),
                ),
                InputSpec("context_overlap", _INT, default=30, widget=NumberWidget(min=0)),
                InputSpec(
                    "context_schedule",
                    _COMBO,
                    default="standard_uniform",
                    widget=ComboWidget(_SCHEDULES),
                ),
                InputSpec(
                    "context_stride", _INT, default=1, advanced=True, widget=NumberWidget(min=1)
                ),
                InputSpec("closed_loop", _BOOLEAN, default=False, advanced=True),
                InputSpec(
                    "fuse_method", _COMBO, default="pyramid", widget=ComboWidget(_FUSE_METHODS)
                ),
                InputSpec("freenoise", _BOOLEAN, default=True, advanced=True),
                InputSpec("retain_first_frame", _BOOLEAN, default=False),
                InputSpec("split_conds_to_windows", _BOOLEAN, default=False, advanced=True),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )
        return with_identity_replacement(schema, "WanContextWindowsManual")

    @classmethod
    def execute(
        cls,
        *,
        context_length: int,
        context_overlap: int,
        retain_first_frame: bool = False,
        **kwargs: Any,
    ) -> Mapping[str, object]:
        return super().execute(
            context_length=max(((context_length - 1) // 4) + 1, 1),
            context_overlap=max(context_overlap // 4, 0),
            dim=2,
            cond_retain_index_list="0" if retain_first_frame else "",
            **kwargs,
        )
