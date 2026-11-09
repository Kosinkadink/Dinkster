"""Wan Uni3C render-video conditioning through native model patches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


def _runtime() -> tuple[Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("torch"),
        importlib.import_module("dinkster_inference.utils"),
        importlib.import_module("dinkster_inference.model_management"),
    )


class WanUni3CCnetPatch:
    def __init__(
        self,
        model_patch: Any,
        render_video: Any,
        vae: Any,
        latent_format: Any,
        strength: float,
        sigma_start: float,
        sigma_end: float,
    ) -> None:
        # ModelPatcher can move patches during finalization, after import shutdown.
        self._device_type = importlib.import_module("torch").device
        self.model_patch = model_patch
        self.render_video = render_video
        self.vae = vae
        self.latent_format = latent_format
        self.strength = strength
        self.sigma_start, self.sigma_end = sigma_start, sigma_end
        self.prepared_render: Any = None
        self.temp_data: Any = None

    def encode_render_video(self, target_latent_shape: Any) -> Any:
        torch, utils, management = _runtime()
        temporal, height, width = target_latent_shape
        target_frames = (temporal - 1) * (self.vae.temporal_compression_decode() or 1) + 1
        compression = self.vae.spacial_compression_encode()
        target_height, target_width = height * compression, width * compression
        frames = self.render_video
        if frames.shape[0] > target_frames:
            frames = frames[:target_frames]
        elif frames.shape[0] < target_frames:
            frames = torch.cat(
                (frames, frames[-1:].expand(target_frames - frames.shape[0], -1, -1, -1)), dim=0
            )
        if frames.shape[1:3] != (target_height, target_width):
            frames = utils.common_upscale(
                frames.movedim(-1, 1), target_width, target_height, "bilinear", "center"
            ).movedim(1, -1)
        loaded = management.loaded_models(only_currently_used=True)
        latent = self.vae.encode(frames)
        management.load_models_gpu(loaded)
        return self.latent_format.process_in(latent)

    def build_controlnet_input(self, x: Any, dtype: Any, samples_per_cond: int) -> Any:
        torch, _, _ = _runtime()
        hidden = x[:samples_per_cond, :20].to(dtype)
        if hidden.shape[1] < 20:
            padding = list(hidden.shape)
            padding[1] = 20 - hidden.shape[1]
            hidden = torch.cat(
                (hidden, torch.zeros(padding, dtype=hidden.dtype, device=hidden.device)), dim=1
            )
        render = self.prepared_render
        if render is None or render.shape[2:] != hidden.shape[2:]:
            render = self.encode_render_video(hidden.shape[2:])
        render = render.to(device=hidden.device, dtype=dtype)
        self.prepared_render = render
        if render.shape[0] != hidden.shape[0]:
            render = render.expand(hidden.shape[0], -1, -1, -1, -1)
        return torch.cat((hidden, render), dim=1)

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        image, block = kwargs["img"], kwargs["block_index"]
        options = kwargs.get("transformer_options", {})
        if block == 0:
            self.temp_data = None
            sigmas = options.get("sigmas")
            active = sigmas is None or self.sigma_end <= sigmas[0].item() <= self.sigma_start
            if active:
                x = kwargs["x"]
                num_conds = len(options.get("cond_or_uncond", [0]))
                samples_per_cond = x.shape[0]
                if num_conds > 0 and x.shape[0] % num_conds == 0:
                    samples_per_cond //= num_conds
                temb = kwargs["vec"][:samples_per_cond]
                if temb.ndim == 3:
                    temb = temb[:, 0]
                control = self.build_controlnet_input(x, image.dtype, samples_per_cond)
                hidden, freqs = self.model_patch.model.process_input(control)
                self.temp_data = (hidden, temb.to(image.dtype), freqs)
        layers = self.model_patch.model.num_layers
        if self.temp_data is not None and block < layers:
            hidden, temb, freqs = self.temp_data
            hidden, residual = self.model_patch.model.forward_block(block, hidden, temb, freqs)
            residual = residual.to(image.dtype) * self.strength
            if residual.shape[0] != image.shape[0]:
                residual = residual.repeat(image.shape[0] // residual.shape[0], 1, 1)
            offset = kwargs.get("img_offset", 0)
            image[:, offset : offset + residual.shape[1]] += residual
            self.temp_data = None if block >= layers - 1 else (hidden, temb, freqs)
        return kwargs

    def to(self, device_or_dtype: Any) -> WanUni3CCnetPatch:
        if isinstance(device_or_dtype, self._device_type):
            if self.prepared_render is not None:
                self.prepared_render = self.prepared_render.to(device_or_dtype)
            self.temp_data = None
        return self

    def models(self) -> list[Any]:
        return [self.model_patch]


class WanUni3CControlnetApply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanUni3CControlnetApply",
            display_name="Apply Wan Uni3C ControlNet",
            category="model/patch/wan",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("model_patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("render_video", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        model_patch: Any,
        vae: Any,
        render_video: object,
        strength: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        runtime = importlib.import_module("dinkster_inference.ldm.wan.uni3c")
        if not isinstance(model_patch.model, runtime.WanUni3CControlnet):
            raise ValueError("The connected model patch is not a Uni3C ControlNet.")
        control_dim = model_patch.model.controlnet_blocks[0].norm1.linear.in_features
        model_dim = getattr(model.get_model_object("diffusion_model"), "dim", None)
        if model_dim is None:
            raise ValueError("The Uni3C ControlNet only works with Wan models.")
        if model_dim != control_dim:
            raise ValueError(
                f"This Uni3C ControlNet expects a Wan model with dim {control_dim}, "
                f"the loaded model has dim {model_dim}."
            )
        result = model.clone()
        sampling = model.get_model_object("model_sampling")
        patch = WanUni3CCnetPatch(
            model_patch,
            materialize_image(render_video)[..., :3],
            vae,
            model.get_model_object("latent_format"),
            strength,
            sampling.percent_to_sigma(start_percent),
            sampling.percent_to_sigma(end_percent),
        )
        result.set_model_double_block_patch(patch)
        return cls.outputs(MODEL=result)
