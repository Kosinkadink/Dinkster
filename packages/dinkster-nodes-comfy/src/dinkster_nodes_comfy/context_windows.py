"""Native context-window model patches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    MODEL_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
)

_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_INT = TypeExpr.concrete(CORE_INT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_SCHEDULES = ("standard_static", "standard_uniform", "looped_uniform", "batched")
_FUSE_METHODS = ("pyramid", "relative", "flat", "overlap-linear")


def _context_windows() -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.context_windows")


class ContextWindowsManual(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ContextWindowsManual",
            display_name="Context Windows (Manual)",
            category="model/patch",
            description="Manually set context windows.",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("context_length", _INT, default=16, widget=NumberWidget(min=1)),
                InputSpec("context_overlap", _INT, default=4, widget=NumberWidget(min=0)),
                InputSpec(
                    "context_schedule",
                    _COMBO,
                    default="standard_static",
                    widget=ComboWidget(_SCHEDULES),
                ),
                InputSpec("context_stride", _INT, default=1, widget=NumberWidget(min=1)),
                InputSpec("closed_loop", _BOOLEAN, default=False),
                InputSpec(
                    "fuse_method", _COMBO, default="pyramid", widget=ComboWidget(_FUSE_METHODS)
                ),
                InputSpec("dim", _INT, default=0, widget=NumberWidget(min=0, max=5)),
                InputSpec("freenoise", _BOOLEAN, default=False),
                InputSpec(
                    "cond_retain_index_list",
                    _STRING,
                    default="",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec("split_conds_to_windows", _BOOLEAN, default=False),
                InputSpec(
                    "latent_retain_index_list",
                    _STRING,
                    default="",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec("causal_window_fix", _BOOLEAN, default=True),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        context_stride: int,
        closed_loop: bool,
        fuse_method: str,
        dim: int,
        freenoise: bool,
        cond_retain_index_list: str = "",
        split_conds_to_windows: bool = False,
        latent_retain_index_list: str = "",
        causal_window_fix: bool = True,
    ) -> Mapping[str, object]:
        context_windows = _context_windows()
        result = model.clone()
        result.model_options["context_handler"] = context_windows.IndexListContextHandler(
            context_schedule=context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=context_windows.get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=dim,
            freenoise=freenoise,
            cond_retain_index_list=cond_retain_index_list,
            split_conds_to_windows=split_conds_to_windows,
            latent_retain_index_list=latent_retain_index_list,
            causal_window_fix=causal_window_fix,
        )
        context_windows.create_prepare_sampling_wrapper(result)
        if freenoise:
            context_windows.create_sampler_sample_wrapper(result)
        return cls.outputs(MODEL=result)


class WanContextWindowsManual(ContextWindowsManual):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanContextWindowsManual",
            display_name="Wan Context Windows",
            category="model/patch/wan",
            description="Set context windows for Wan-like models.",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("context_length", _INT, default=81, widget=NumberWidget(min=1, step=4)),
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


class LTXVContextWindows(ContextWindowsManual):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LTXVContextWindows",
            display_name="LTXV Context Windows",
            category="model/patch/ltxv",
            description="Set context windows for LTXV-like models.",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("context_length", _INT, default=145, widget=NumberWidget(min=1, step=8)),
                InputSpec("context_overlap", _INT, default=40, widget=NumberWidget(min=0, step=8)),
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

    @classmethod
    def execute(
        cls,
        *,
        context_length: int,
        context_overlap: int,
        retain_first_frame: bool = False,
        **kwargs: Any,
    ) -> Mapping[str, object]:
        retain = "0" if retain_first_frame else ""
        return super().execute(
            context_length=max(((context_length - 1) // 8) + 1, 1),
            context_overlap=max(context_overlap // 8, 0),
            dim=2,
            cond_retain_index_list=retain,
            latent_retain_index_list=retain,
            **kwargs,
        )


CONTEXT_WINDOW_NODES = (ContextWindowsManual, WanContextWindowsManual, LTXVContextWindows)
