"""MiniMax H3 Fun control preparation and model block patches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
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


class MiniMaxH3FunControlPatch:
    def __init__(
        self,
        model_patch: Any,
        vae: Any,
        control_video: Any,
        mask: Any,
        source_video: Any,
        strength: float,
        sigma_start: float,
        sigma_end: float,
    ) -> None:
        self.model_patch, self.vae = model_patch, vae
        self.control_video, self.mask, self.source_video = control_video, mask, source_video
        self.strength, self.sigma_start, self.sigma_end = strength, sigma_start, sigma_end
        self.control_latent: Any = None
        self.control_latent_shape: tuple[int, ...] | None = None
        self.control_stream: Any = None
        self.pristine_stream: Any = None
        self.active = False

    def _fit_frames(self, frames: Any, frame_count: int, width: int, height: int) -> Any:
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        indices = torch.arange(frame_count, device=frames.device).clamp(max=frames.shape[0] - 1)
        return utils.common_upscale(frames[indices], width, height, "bilinear", "center")

    def _encode(self, frames: Any, target_shape: tuple[int, ...]) -> Any:
        torch = importlib.import_module("torch")
        latent = self.vae.encode(frames.movedim(1, -1)).to(torch.float32)
        if tuple(latent.shape) != target_shape:
            raise ValueError(
                f"MiniMax H3 Fun VAE output shape {tuple(latent.shape)} "
                f"does not match the target {target_shape}"
            )
        return latent

    def prepare_control_latent(self, target_shape: Any) -> None:
        shape = tuple(target_shape)
        if self.control_latent is not None and self.control_latent_shape == shape:
            return
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        management = importlib.import_module("dinkster_inference.model_management")
        latent_frames, latent_height, latent_width = shape[2:]
        frame_count = max((latent_frames - 2) // 5, 0) * 17 + 5
        compression = self.vae.spacial_compression_encode()
        width, height = latent_width * compression, latent_height * compression
        loaded = management.loaded_models(only_currently_used=True)
        try:
            hint = None
            if self.control_video is not None:
                frames = self._fit_frames(self.control_video, frame_count, width, height)
                hint = self._encode(frames, shape)
            if self.mask is not None:
                mask = (self.mask.reshape(-1, 1, *self.mask.shape[-2:]) > 0.5).to(torch.float32)
                indices = torch.arange(frame_count, device=mask.device).clamp(max=mask.shape[0] - 1)
                mask = utils.common_upscale(mask[indices], width, height, "bilinear", "center")
                visibility = 1.0 - (mask > 0.5).to(torch.float32)
                source = (
                    torch.zeros(
                        frame_count,
                        3,
                        height,
                        width,
                        dtype=visibility.dtype,
                        device=visibility.device,
                    )
                    if self.source_video is None
                    else self._fit_frames(self.source_video, frame_count, width, height)
                )
                visibility = visibility.to(source.device)
                masked = source * visibility
                if self.model_patch.model.inpaint_post_norm:
                    mean = importlib.import_module(
                        "dinkster_inference.ldm.minimax.vae"
                    ).IMAGENET_MEAN
                    masked += (1.0 - visibility) * torch.tensor(
                        mean,
                        dtype=source.dtype,
                        device=source.device,
                    ).view(1, 3, 1, 1)
                encoded = self._encode(masked, shape)
                if hint is None:
                    hint = torch.zeros_like(encoded)
                visibility_latent = torch.nn.functional.interpolate(
                    visibility.squeeze(1)[None, None],
                    size=(latent_frames, latent_height, latent_width),
                    mode="trilinear",
                    align_corners=False,
                )
                hint = torch.cat(
                    (hint, visibility_latent.to(hint.device), encoded.to(hint.device)), dim=1
                )
        finally:
            management.load_models_gpu(loaded)
        self.control_latent, self.control_latent_shape = hint, shape

    def diffusion_model_wrapper(
        self,
        executor: Any,
        x: Any,
        timestep: Any,
        context: Any,
        transformer_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        options = {} if transformer_options is None else transformer_options
        sigmas = options.get("sigmas")
        sigma = float(sigmas[0]) if sigmas is not None else float(timestep.flatten()[0]) / 1000.0
        self.active = self.sigma_end <= sigma <= self.sigma_start
        self.control_stream = None
        if self.active:
            prefetch = importlib.import_module("dinkster_inference.model_prefetch")
            with prefetch.pause_malloc_graph():
                self.prepare_control_latent(x[0].shape)
        try:
            return executor(x, timestep, context, options, **kwargs)
        finally:
            self.control_stream = None
            self.pristine_stream = None

    def before_block(self, block_index: int, args: Mapping[str, Any]) -> None:
        if self.active and block_index == self.model_patch.model.injection_layers[0]:
            self.pristine_stream = args["img"].clone()

    def after_block(self, block_index: int, args: Mapping[str, Any], out: dict[str, Any]) -> Any:
        if not self.active:
            return out
        index = self.model_patch.model.injection_layers.index(block_index)
        if index == 0:
            self.control_latent = self.control_latent.to(out["img"].device)
            self.control_stream = self.model_patch.model.init_stream(
                self.pristine_stream,
                self.control_latent,
                args["layout"],
                args["t_emb"],
            )
            self.pristine_stream = None
        self.control_stream, skip = self.model_patch.model.step(
            index,
            self.control_stream,
            args["t_emb"],
            args["mod_segments"],
            args["rope_freqs"],
            transformer_options=args["transformer_options"],
        )
        skip[args["layout"].audio_pos.to(skip.device)] = 0
        out["img"].add_(skip, alpha=self.strength)
        return out

    def to(self, device_or_dtype: Any) -> MiniMaxH3FunControlPatch:
        torch = importlib.import_module("torch")
        if isinstance(device_or_dtype, torch.device):
            if self.control_latent is not None:
                self.control_latent = self.control_latent.to(device_or_dtype)
            self.control_stream = None
        return self

    def cleanup(self) -> None:
        self.control_latent = self.control_latent_shape = None
        self.control_stream = self.pristine_stream = None
        self.active = False

    def models(self) -> list[Any]:
        return [self.model_patch]

    def register(self, model: Any) -> None:
        extension = importlib.import_module("dinkster_inference.patcher_extension")
        model.add_wrapper(extension.WrappersMP.DIFFUSION_MODEL, self.diffusion_model_wrapper)
        for block_index in self.model_patch.model.injection_layers:
            previous = (
                model.model_options.get("transformer_options", {})
                .get("patches_replace", {})
                .get("dit", {})
                .get(("double_block", block_index))
            )
            model.set_model_patch_replace(
                MiniMaxH3FunControlBlockPatch(self, block_index, previous),
                "dit",
                "double_block",
                block_index,
            )


class MiniMaxH3FunControlBlockPatch:
    def __init__(self, control_patch: Any, block_index: int, previous: Any) -> None:
        self.control_patch, self.block_index, self.previous = control_patch, block_index, previous

    def __call__(self, args: Any, extra_args: Any) -> Any:
        prefetch = importlib.import_module("dinkster_inference.model_prefetch")
        with prefetch.pause_malloc_graph():
            self.control_patch.before_block(self.block_index, args)
        result = (
            extra_args["original_block"](args)
            if self.previous is None
            else self.previous(args, extra_args)
        )
        with prefetch.pause_malloc_graph():
            return self.control_patch.after_block(self.block_index, args, result)

    def to(self, device_or_dtype: Any) -> MiniMaxH3FunControlBlockPatch:
        self.control_patch.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self) -> None:
        self.control_patch.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self) -> list[Any]:
        result = self.control_patch.models()
        if hasattr(self.previous, "models"):
            result += self.previous.models()
        return result


class MiniMaxH3FunControlNetApply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MiniMaxH3FunControlNetApply",
            display_name="Apply MiniMax H3 Fun ControlNet",
            category="model/patch/minimax",
            search_terms=("minimax controlnet", "h3 controlnet", "video inpaint controlnet"),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("model_patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    advanced=True,
                ),
                InputSpec(
                    "end_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    advanced=True,
                ),
                InputSpec("control_video", TypeExpr.concrete(IMAGE_TYPE), required=False),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE), required=False),
                InputSpec("source_video", TypeExpr.concrete(IMAGE_TYPE), required=False),
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
        strength: float,
        start_percent: float,
        end_percent: float,
        control_video: object = None,
        mask: object = None,
        source_video: object = None,
    ) -> Mapping[str, object]:
        if strength == 0 or (control_video is None and mask is None):
            return cls.outputs(MODEL=model)
        initialize_inference_process()
        patched = model.clone()
        sampling = model.get_model_object("model_sampling")
        patch = MiniMaxH3FunControlPatch(
            model_patch,
            vae,
            materialize_image(control_video)[..., :3].movedim(-1, 1)
            if control_video is not None
            else None,
            materialize_image(mask) if mask is not None else None,
            materialize_image(source_video)[..., :3].movedim(-1, 1)
            if mask is not None and source_video is not None
            else None,
            strength,
            float(sampling.percent_to_sigma(start_percent)),
            float(sampling.percent_to_sigma(end_percent)),
        )
        patch.register(patched)
        return cls.outputs(MODEL=patched)
