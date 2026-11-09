"""Separate LTXV video and audio guidance through the pinned sampling engine."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    GUIDER_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class LTXVDualCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        return NodeSchema(
            node_type="LTXVDualCFGGuider",
            display_name="LTXV Dual CFG Guider",
            category="model/sampling/guiders",
            description=(
                "Separate CFG scales for the video and audio modalities of a packed LTXV-AV latent."
            ),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                *(
                    InputSpec(
                        name,
                        TypeExpr.concrete(CORE_FLOAT),
                        default=default,
                        widget=NumberWidget(min=0.0, max=100.0, step=0.1, round=0.01),
                    )
                    for name, default in (("video_cfg", 3.0), ("audio_cfg", 7.0))
                ),
            ),
            outputs=(OutputSpec("GUIDER", TypeExpr.concrete(GUIDER_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        positive: Any,
        negative: Any,
        video_cfg: float,
        audio_cfg: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        samplers = importlib.import_module("dinkster_inference.samplers")

        class GuiderLTXAVDualCFG(samplers.CFGGuider):
            def set_cfg(self, video_cfg: float, audio_cfg: float) -> None:
                self.video_cfg = video_cfg
                self.audio_cfg = audio_cfg
                self.cfg = max(video_cfg, audio_cfg)

            def sample(self, noise: Any, latent_image: Any, *args: Any, **kwargs: Any) -> Any:
                self._v_numel = None
                if getattr(latent_image, "is_nested", False):
                    parts = latent_image.unbind()
                    if len(parts) >= 2:
                        self._v_numel = math.prod(parts[0].shape[1:])
                parent: Any = super()
                return parent.sample(noise, latent_image, *args, **kwargs)

            def predict_noise(
                self,
                x: Any,
                timestep: Any,
                model_options: dict[str, Any] | None = None,
                seed: Any = None,
            ) -> Any:
                if model_options is None:
                    model_options = {}
                split = getattr(self, "_v_numel", None)
                parent: Any = super()
                if split is None or math.isclose(self.video_cfg, self.audio_cfg):
                    self.cfg = self.video_cfg
                    return parent.predict_noise(x, timestep, model_options, seed)

                video_cfg, audio_cfg = self.video_cfg, self.audio_cfg

                def dual_cfg(args: dict[str, Any]) -> Any:
                    cond, uncond = args["cond"], args["uncond"]
                    result = uncond + (cond - uncond) * video_cfg
                    result[..., split:] = (
                        uncond[..., split:] + (cond[..., split:] - uncond[..., split:]) * audio_cfg
                    )
                    return result

                options = {
                    **model_options,
                    "sampler_cfg_function": dual_cfg,
                    "disable_cfg1_optimization": True,
                }
                return parent.predict_noise(x, timestep, options, seed)

        guider = GuiderLTXAVDualCFG(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(video_cfg, audio_cfg)
        return cls.outputs(GUIDER=guider)
