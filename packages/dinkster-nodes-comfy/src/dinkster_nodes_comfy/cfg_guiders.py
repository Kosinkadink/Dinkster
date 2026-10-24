"""Native custom-sampling guider nodes backed by the pinned inference sampler."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    GUIDER_TYPE,
    MODEL_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_GUIDER = TypeExpr.concrete(GUIDER_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)


def _inference_modules() -> tuple[Any, Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("dinkster_inference.samplers"),
        importlib.import_module("dinkster_inference.sampler_helpers"),
        importlib.import_module("dinkster_inference.patcher_extension"),
        importlib.import_module("node_helpers"),
    )


def _guider_schema(
    node_type: str,
    display_name: str,
    inputs: tuple[InputSpec, ...],
    *,
    search_terms: tuple[str, ...] = (),
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="model/sampling/guiders",
        search_terms=search_terms,
        inputs=inputs,
        outputs=(OutputSpec("GUIDER", _GUIDER),),
    )


def _cfg_input(default: float) -> InputSpec:
    return InputSpec(
        "cfg",
        _FLOAT,
        default=default,
        widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
    )


def _dual_cfg_input(name: str) -> InputSpec:
    return InputSpec(
        name,
        _FLOAT,
        default=8.0,
        widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
    )


class BasicGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _guider_schema(
            "BasicGuider",
            "Basic Guider",
            (InputSpec("model", _MODEL), InputSpec("conditioning", _CONDITIONING)),
        )

    @classmethod
    def execute(cls, *, model: Any, conditioning: Any) -> Mapping[str, object]:
        samplers, _, _, _ = _inference_modules()

        class GuiderBasic(samplers.CFGGuider):
            def set_conds(self, positive: Any) -> None:
                self.inner_set_conds({"positive": positive})

        guider = GuiderBasic(model)
        guider.set_conds(conditioning)
        return cls.outputs(GUIDER=guider)


class CFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _guider_schema(
            "CFGGuider",
            "CFG Guider",
            (
                InputSpec("model", _MODEL),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                _cfg_input(8.0),
            ),
        )

    @classmethod
    def execute(
        cls, *, model: Any, positive: Any, negative: Any, cfg: float
    ) -> Mapping[str, object]:
        samplers, _, _, _ = _inference_modules()
        guider = samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return cls.outputs(GUIDER=guider)


class DualCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _guider_schema(
            "DualCFGGuider",
            "Dual CFG Guider",
            (
                InputSpec("model", _MODEL),
                InputSpec("cond1", _CONDITIONING),
                InputSpec("cond2", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                _dual_cfg_input("cfg_conds"),
                _dual_cfg_input("cfg_cond2_negative"),
                InputSpec(
                    "style",
                    _COMBO,
                    default="regular",
                    widget=ComboWidget((ComboOption("regular"), ComboOption("nested"))),
                ),
            ),
            search_terms=("dual prompt guidance",),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        cond1: Any,
        cond2: Any,
        negative: Any,
        cfg_conds: float,
        cfg_cond2_negative: float,
        style: str,
    ) -> Mapping[str, object]:
        samplers, _, _, node_helpers = _inference_modules()

        class GuiderDualCFG(samplers.CFGGuider):
            def set_cfg(self, cfg1: float, cfg2: float, nested: bool = False) -> None:
                self.cfg1 = cfg1
                self.cfg2 = cfg2
                self.nested = nested

            def set_conds(self, positive: Any, middle: Any, uncond: Any) -> None:
                middle = node_helpers.conditioning_set_values(middle, {"prompt_type": "negative"})
                self.inner_set_conds({"positive": positive, "middle": middle, "negative": uncond})

            def predict_noise(
                self,
                x: Any,
                timestep: Any,
                model_options: dict[str, Any] | None = None,
                seed: Any = None,
            ) -> Any:
                if model_options is None:
                    model_options = {}
                negative_cond = self.conds.get("negative")
                middle_cond = self.conds.get("middle")
                positive_cond = self.conds.get("positive")
                if self.nested:
                    out = samplers.calc_cond_batch(
                        self.inner_model,
                        [negative_cond, middle_cond, positive_cond],
                        x,
                        timestep,
                        model_options,
                    )
                    text = samplers.cfg_function(
                        self.inner_model,
                        out[2],
                        out[1],
                        self.cfg1,
                        x,
                        timestep,
                        model_options=model_options,
                        cond=positive_cond,
                        uncond=middle_cond,
                    )
                    return out[0] + self.cfg2 * (text - out[0])
                if not model_options.get("disable_cfg1_optimization", False) and math.isclose(
                    self.cfg2, 1.0
                ):
                    negative_cond = None
                    if math.isclose(self.cfg1, 1.0):
                        middle_cond = None
                out = samplers.calc_cond_batch(
                    self.inner_model,
                    [negative_cond, middle_cond, positive_cond],
                    x,
                    timestep,
                    model_options,
                )
                return (
                    samplers.cfg_function(
                        self.inner_model,
                        out[1],
                        out[0],
                        self.cfg2,
                        x,
                        timestep,
                        model_options=model_options,
                        cond=middle_cond,
                        uncond=negative_cond,
                    )
                    + (out[2] - out[1]) * self.cfg1
                )

        guider = GuiderDualCFG(model)
        guider.set_conds(cond1, cond2, negative)
        guider.set_cfg(cfg_conds, cfg_cond2_negative, nested=style == "nested")
        return cls.outputs(GUIDER=guider)


class DualModelGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _guider_schema(
            "DualModelGuider",
            "Dual Model CFG Guider",
            (
                InputSpec("model", _MODEL),
                InputSpec("model_negative", _MODEL, required=False),
                InputSpec("positive", _CONDITIONING),
                _cfg_input(4.0),
                InputSpec("negative", _CONDITIONING, required=False),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        positive: Any,
        cfg: float,
        model_negative: Any = None,
        negative: Any = None,
    ) -> Mapping[str, object]:
        samplers, sampler_helpers, _, _ = _inference_modules()
        torch = importlib.import_module("torch")
        base_init = samplers.CFGGuider.__init__
        base_outer_sample = samplers.CFGGuider.outer_sample
        base_inner_sample = samplers.CFGGuider.inner_sample

        class GuiderDualModel(samplers.CFGGuider):
            def __init__(self, model_patcher: Any, uncond_model_patcher: Any) -> None:
                base_init(self, model_patcher)
                self.uncond_model_patcher = uncond_model_patcher
                self.uncond_inner: Any = None

            def outer_sample(
                self,
                noise: Any,
                latent_image: Any,
                sampler: Any,
                sigmas: Any,
                denoise_mask: Any = None,
                callback: Any = None,
                disable_pbar: bool = False,
                seed: Any = None,
                latent_shapes: Any = None,
            ) -> Any:
                self.uncond_inner = None
                self.uncond_loaded: list[Any] = []
                self._uncond_neg: Any = None
                if not math.isclose(self.cfg, 1.0):
                    conds = {"negative": [condition.copy() for condition in self.conds["negative"]]}
                    self.uncond_inner, conds, self.uncond_loaded = sampler_helpers.prepare_sampling(
                        self.uncond_model_patcher,
                        noise.shape,
                        conds,
                        self.uncond_model_patcher.model_options,
                    )
                    self._uncond_neg = conds["negative"]
                    self.uncond_model_patcher.pre_run()
                try:
                    return base_outer_sample(
                        self,
                        noise,
                        latent_image,
                        sampler,
                        sigmas,
                        denoise_mask,
                        callback,
                        disable_pbar,
                        seed,
                        latent_shapes=latent_shapes,
                    )
                finally:
                    if self.uncond_inner is not None:
                        self.uncond_model_patcher.cleanup()
                        sampler_helpers.cleanup_models(
                            {"negative": self._uncond_neg}, self.uncond_loaded
                        )
                        self.uncond_inner = None

            def inner_sample(
                self,
                noise: Any,
                latent_image: Any,
                device: Any,
                sampler: Any,
                sigmas: Any,
                denoise_mask: Any,
                callback: Any,
                disable_pbar: bool,
                seed: Any,
                latent_shapes: Any = None,
            ) -> Any:
                if self.uncond_inner is not None:
                    negative_latent = latent_image
                    if negative_latent is not None and torch.count_nonzero(negative_latent) > 0:
                        negative_latent = self.uncond_inner.process_latent_in(negative_latent)
                    self._uncond_conds = samplers.process_conds(
                        self.uncond_inner,
                        noise,
                        {"negative": self._uncond_neg},
                        device,
                        negative_latent,
                        denoise_mask,
                        seed,
                        latent_shapes=latent_shapes,
                    )["negative"]
                return base_inner_sample(
                    self,
                    noise,
                    latent_image,
                    device,
                    sampler,
                    sigmas,
                    denoise_mask,
                    callback,
                    disable_pbar,
                    seed,
                    latent_shapes=latent_shapes,
                )

            def predict_noise(
                self,
                x: Any,
                timestep: Any,
                model_options: dict[str, Any] | None = None,
                seed: Any = None,
            ) -> Any:
                if model_options is None:
                    model_options = {}
                positive_cond = self.conds.get("positive")
                cond = samplers.calc_cond_batch(
                    self.inner_model, [positive_cond], x, timestep, model_options
                )[0]
                if self.uncond_inner is None or (
                    math.isclose(self.cfg, 1.0)
                    and not model_options.get("disable_cfg1_optimization", False)
                ):
                    return cond
                uncond_options = model_options
                if "multigpu_clones" in model_options:
                    uncond_options = {
                        key: value
                        for key, value in model_options.items()
                        if key != "multigpu_clones"
                    }
                uncond = samplers.calc_cond_batch(
                    self.uncond_inner, [self._uncond_conds], x, timestep, uncond_options
                )[0]
                return samplers.cfg_function(
                    self.inner_model,
                    cond,
                    uncond,
                    self.cfg,
                    x,
                    timestep,
                    model_options=model_options,
                    cond=positive_cond,
                    uncond=self._uncond_conds,
                )

        if negative is None:
            negative = [[None, {}]]
        guider = (
            GuiderDualModel(model, model_negative)
            if model_negative is not None
            else samplers.CFGGuider(model)
        )
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return cls.outputs(GUIDER=guider)


class CFGOverride(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CFGOverride",
            display_name="CFG Override",
            category="sampling/custom_sampling",
            description=(
                "Override cfg to a fixed value over a [start, end] percent (sigma) range. "
                "With multiple overrides, the one nearest the sampler wins on overlap."
            ),
            inputs=(
                InputSpec("model", _MODEL),
                _cfg_input(1.0),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, cfg: float, start_percent: float, end_percent: float
    ) -> Mapping[str, object]:
        _, _, patcher_extension, _ = _inference_modules()
        sampling = model.get_model_object("model_sampling")
        sigma_hi = sampling.percent_to_sigma(start_percent)
        sigma_lo = sampling.percent_to_sigma(end_percent)

        def predict_noise_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
            sigma = float(args[1].flatten()[0])
            if not sigma_lo <= sigma <= sigma_hi:
                return executor(*args, **kwargs)
            guider = executor.class_obj
            saved_cfg = guider.cfg
            guider.cfg = cfg
            try:
                return executor(*args, **kwargs)
            finally:
                guider.cfg = saved_cfg

        patched = model.clone()
        patched.add_wrapper(patcher_extension.WrappersMP.PREDICT_NOISE, predict_noise_wrapper)
        return cls.outputs(MODEL=patched)


CFG_GUIDER_NODES = (CFGGuider, BasicGuider, DualCFGGuider, DualModelGuider, CFGOverride)
