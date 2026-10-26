"""Qwen, Z-Image, and USO image-control model patches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
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
    materialize_clip_vision_output,
    materialize_image,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_MODEL_PATCH = TypeExpr.concrete(MODEL_PATCH_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)


def _range() -> tuple[InputSpec, InputSpec]:
    return (
        InputSpec(
            "start_percent",
            _FLOAT,
            default=0.0,
            required=False,
            advanced=True,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        ),
        InputSpec(
            "end_percent",
            _FLOAT,
            default=1.0,
            required=False,
            advanced=True,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        ),
    )


def _modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("torch"),
        importlib.import_module("dinkster_inference.utils"),
        importlib.import_module("dinkster_inference.latent_formats"),
        importlib.import_module("dinkster_inference.model_management"),
        importlib.import_module("dinkster_inference.model_prefetch"),
        importlib.import_module("dinkster_inference.patcher_extension"),
        importlib.import_module("dinkster_inference.ldm.qwen_image21.model"),
    )


def _in_sigma_range(options: Mapping[str, Any], sigma_range: tuple[float, float]) -> bool:
    sigma = float(options["sigmas"].flatten()[0])
    return sigma_range[1] <= sigma <= sigma_range[0]


class DiffSynthCnetPatch:
    def __init__(
        self,
        model_patch: Any,
        vae: Any,
        image: Any,
        strength: float,
        mask: Any = None,
        sigma_range: tuple[float, float] = (float("inf"), 0.0),
    ) -> None:
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.sigma_range = sigma_range
        self.encoded_image = model_patch.model.process_input_latent_image(
            self.encode_latent_cond(image)
        )
        self.encoded_image_size = (image.shape[1], image.shape[2])

    def encode_latent_cond(self, image: Any) -> Any:
        torch, utils, *_ = _modules()
        latent = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim <= 0:
            return latent
        if self.mask is None:
            mask = torch.ones_like(latent)[:, : self.model_patch.model.additional_in_dim // 4]
        else:
            mask = utils.common_upscale(
                self.mask.mean(dim=1, keepdim=True),
                latent.shape[-1],
                latent.shape[-2],
                "bilinear",
                "none",
            )
        return torch.cat((latent, mask), dim=1)

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not _in_sigma_range(kwargs["transformer_options"], self.sigma_range):
            return kwargs
        _, utils, _, management, *_ = _modules()
        x, image, block = kwargs["x"], kwargs["img"], kwargs["block_index"]
        compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (
            x.shape[-2] * compression,
            x.shape[-1] * compression,
        ):
            scaled = utils.common_upscale(
                self.image.movedim(-1, 1),
                x.shape[-1] * compression,
                x.shape[-2] * compression,
                "area",
                "center",
            )
            loaded = management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(
                self.encode_latent_cond(scaled.movedim(1, -1))
            )
            self.encoded_image_size = (scaled.shape[-2], scaled.shape[-1])
            management.load_models_gpu(loaded)
        image[:, : self.encoded_image.shape[1]] += (
            self.model_patch.model.control_block(
                image[:, : self.encoded_image.shape[1]], self.encoded_image.to(image.dtype), block
            )
            * self.strength
        )
        return kwargs

    def to(self, device_or_dtype: Any) -> DiffSynthCnetPatch:
        torch, *_ = _modules()
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self) -> list[Any]:
        return [self.model_patch]


class ZImageControlPatch:
    def __init__(
        self,
        model_patch: Any,
        vae: Any,
        image: Any,
        strength: float,
        inpaint_image: Any = None,
        mask: Any = None,
        sigma_range: tuple[float, float] = (float("inf"), 0.0),
    ) -> None:
        self.model_patch, self.vae, self.image = model_patch, vae, image
        self.inpaint_image, self.mask, self.strength = inpaint_image, mask, strength
        self.sigma_range = sigma_range
        self.is_inpaint = model_patch.model.additional_in_dim > 0
        if image is not None and inpaint_image is not None and image.shape != inpaint_image.shape:
            self.encoded_image = None
        else:
            self.encoded_image = self.encode_latent_cond(image, inpaint_image)
            source = inpaint_image if image is None else image
            assert source is not None
            self.encoded_image_size = (source.shape[1], source.shape[2])
        self.temp_data: Any = None

    def encode_latent_cond(self, control_image: Any = None, inpaint_image: Any = None) -> Any:
        torch, utils, latent_formats, *_ = _modules()
        latent = (
            None
            if control_image is None
            else latent_formats.Flux().process_in(self.vae.encode(control_image))
        )
        if not self.is_inpaint:
            return latent
        if inpaint_image is None:
            inpaint_image = torch.ones_like(control_image) * 0.5
        if self.mask is not None:
            mask = self.mask.view(
                self.mask.shape[0], -1, self.mask.shape[-2], self.mask.shape[-1]
            ).mean(dim=1, keepdim=True)
            mask = utils.common_upscale(
                mask, inpaint_image.shape[-2], inpaint_image.shape[-3], "bilinear", "center"
            )
            inpaint_image = (inpaint_image - 0.5) * mask.movedim(1, -1).round() + 0.5
        inpaint_latent = latent_formats.Flux().process_in(self.vae.encode(inpaint_image))
        if self.mask is None:
            mask = torch.zeros_like(inpaint_latent)[:, :1]
        else:
            mask = (
                self.mask.view(self.mask.shape[0], -1, self.mask.shape[-2], self.mask.shape[-1])
                .mean(dim=1, keepdim=True)
                .to(device=inpaint_latent.device)
            )
            mask = utils.common_upscale(
                mask, inpaint_latent.shape[-1], inpaint_latent.shape[-2], "nearest", "center"
            )
        if latent is None:
            latent = latent_formats.Flux().process_in(
                self.vae.encode(torch.ones_like(inpaint_image) * 0.5)
            )
        return torch.cat((latent, mask, inpaint_latent), dim=1)

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not _in_sigma_range(kwargs["transformer_options"], self.sigma_range):
            return kwargs
        _, utils, _, management, *_ = _modules()
        x, image, img_input, text = kwargs["x"], kwargs["img"], kwargs["img_input"], kwargs["txt"]
        pe, vec, block, block_type = (
            kwargs["pe"],
            kwargs["vec"],
            kwargs["block_index"],
            kwargs.get("block_type", ""),
        )
        compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (
            x.shape[-2] * compression,
            x.shape[-1] * compression,
        ):
            scaled = (
                None
                if self.image is None
                else utils.common_upscale(
                    self.image.movedim(-1, 1),
                    x.shape[-1] * compression,
                    x.shape[-2] * compression,
                    "area",
                    "center",
                ).movedim(1, -1)
            )
            inpaint = (
                None
                if self.inpaint_image is None
                else utils.common_upscale(
                    self.inpaint_image.movedim(-1, 1),
                    x.shape[-1] * compression,
                    x.shape[-2] * compression,
                    "area",
                    "center",
                ).movedim(1, -1)
            )
            source = inpaint if scaled is None else scaled
            assert source is not None
            self.encoded_image_size = (source.shape[-3], source.shape[-2])
            loaded = management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(scaled, inpaint)
            management.load_models_gpu(loaded)
        count = self.model_patch.model.n_control_layers
        div = round(30 / count)
        index, fractional = block // div, block / div
        kwargs.pop("img")
        kwargs.pop("txt")
        if fractional > count - 1:
            self.temp_data = None
            return kwargs
        if self.temp_data is None or self.temp_data[0] > index:
            initial = -3 if block_type == "noise_refiner" else -1
            self.temp_data = (
                initial,
                (None, self.model_patch.model(text, self.encoded_image.to(image.dtype), pe, vec)),
            )
        if block_type == "noise_refiner":
            next_layer = self.temp_data[0] + 1
            self.temp_data = (
                next_layer,
                self.model_patch.model.forward_noise_refiner_block(
                    block,
                    self.temp_data[1][1],
                    img_input[:, : self.temp_data[1][1].shape[1]],
                    None,
                    pe,
                    vec,
                ),
            )
            if self.temp_data[1][0] is not None:
                image[:, : self.temp_data[1][0].shape[1]] += self.temp_data[1][0] * self.strength
        else:
            while self.temp_data[0] < index and self.temp_data[0] + 1 < count:
                next_layer = self.temp_data[0] + 1
                self.temp_data = (
                    next_layer,
                    self.model_patch.model.forward_control_block(
                        next_layer,
                        self.temp_data[1][1],
                        img_input[:, : self.temp_data[1][1].shape[1]],
                        None,
                        pe,
                        vec,
                    ),
                )
            if fractional == self.temp_data[0]:
                image[:, : self.temp_data[1][0].shape[1]] += self.temp_data[1][0] * self.strength
                if count == self.temp_data[0] + 1:
                    self.temp_data = None
        return kwargs

    def to(self, device_or_dtype: Any) -> ZImageControlPatch:
        torch, *_ = _modules()
        if isinstance(device_or_dtype, torch.device):
            if self.encoded_image is not None:
                self.encoded_image = self.encoded_image.to(device_or_dtype)
            self.temp_data = None
        return self

    def models(self) -> list[Any]:
        return [self.model_patch]


class QwenImage21FunControlPatch:
    def __init__(
        self,
        model_patch: Any,
        vae: Any,
        image: Any,
        strength: float,
        inpaint_image: Any = None,
        mask: Any = None,
        sigma_range: tuple[float, float] = (float("inf"), 0.0),
    ) -> None:
        self.model_patch, self.vae, self.image = model_patch, vae, image
        self.inpaint_image, self.mask, self.strength, self.sigma_range = (
            inpaint_image,
            mask,
            strength,
            sigma_range,
        )
        self.active = False
        self.injection_layers: list[int] | None = None
        self.control = self.stream = self.pristine = None

    def prepare(self, height: int, width: int) -> None:
        torch, utils, latent_formats, management, *_ = _modules()
        if self.control is not None and self.control.shape[-2:] == (height, width):
            return
        image_width, image_height = (
            width * self.vae.spacial_compression_encode(),
            height * self.vae.spacial_compression_encode(),
        )
        latent_format = latent_formats.QwenImage21()
        loaded = management.loaded_models(only_currently_used=True)
        try:
            control = torch.zeros(1, latent_format.latent_channels, height, width)
            if self.image is not None:
                image = utils.common_upscale(
                    self.image[:1].movedim(-1, 1), image_width, image_height, "bicubic", "disabled"
                )
                control = (
                    latent_format.process_in(self.vae.encode(image.movedim(1, -1))).float().cpu()
                )
            regen = torch.ones(1, 1, image_height, image_width)
            if self.mask is not None:
                mask = self.mask.reshape(-1, 1, *self.mask.shape[-2:])[:1].float().cpu()
                regen = (
                    utils.common_upscale(mask, image_width, image_height, "bilinear", "disabled")
                    >= 0.5
                ).float()
            inpaint = torch.zeros_like(control)
            if self.inpaint_image is not None:
                image = (
                    utils.common_upscale(
                        self.inpaint_image[:1].movedim(-1, 1),
                        image_width,
                        image_height,
                        "bicubic",
                        "disabled",
                    )
                    .float()
                    .cpu()
                )
                inpaint = (
                    latent_format.process_in(
                        self.vae.encode((image * (1 - regen) + 0.5 * regen).movedim(1, -1))
                    )
                    .float()
                    .cpu()
                )
            keep = 1 - torch.nn.functional.interpolate(regen, size=(height, width), mode="nearest")
        finally:
            management.load_models_gpu(loaded)
        self.control = torch.cat((control, keep, inpaint), dim=1)

    def diffusion_model_wrapper(
        self,
        executor: Any,
        x: Any,
        timestep: Any,
        context: Any,
        ref_latents: Any,
        image_slots: Any,
        transformer_options: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        _, _, _, _, prefetch, _, _ = _modules()
        self.active = _in_sigma_range(transformer_options, self.sigma_range)
        if self.active:
            with prefetch.pause_malloc_graph():
                self.prepare(*x.shape[-2:])
        else:
            replacements = transformer_options.get("patches_replace", {}).get("dit", {})
            replacements = {
                key: value.previous
                if isinstance(value, QwenImage21FunControlBlockPatch)
                and value.control_patch is self
                else value
                for key, value in replacements.items()
            }
            replacements = {key: value for key, value in replacements.items() if value is not None}
            transformer_options = {
                **transformer_options,
                "patches_replace": {**transformer_options["patches_replace"], "dit": replacements},
            }
        try:
            return executor(
                x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs
            )
        finally:
            self.stream = self.pristine = None

    def before_block(self, block: int, args: dict[str, Any]) -> None:
        assert self.injection_layers is not None
        if self.active and block == self.injection_layers[0]:
            self.pristine = args["img"].clone()

    def after_block(
        self, block: int, args: dict[str, Any], output: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.active:
            return output
        assert self.injection_layers is not None
        assert self.control is not None
        index = self.injection_layers.index(block)
        if index == 0:
            self.control = self.control.to(output["img"].device, output["img"].dtype)
            self.stream = self.model_patch.model.init_stream(
                self.pristine, self.control.flatten(2).transpose(1, 2), args["prefix_len"]
            )
            self.pristine = None
        self.stream, skip = self.model_patch.model.step(
            index,
            self.stream,
            args["mod"],
            args["pe"],
            args["attn_fn"],
            args["prefix_len"],
            args["transformer_options"],
        )
        output["img"].add_(skip, alpha=self.strength)
        return output

    def to(self, device_or_dtype: Any) -> QwenImage21FunControlPatch:
        torch, *_ = _modules()
        if isinstance(device_or_dtype, torch.device):
            if self.control is not None:
                self.control = self.control.to(device_or_dtype)
            self.stream = None
        return self

    def cleanup(self) -> None:
        self.control = self.stream = self.pristine = None
        self.active = False

    def models(self) -> list[Any]:
        return [self.model_patch]

    def register(self, model: Any) -> None:
        _, _, _, _, _, extension, _ = _modules()
        base = len(model.get_model_object("diffusion_model").transformer_blocks)
        self.injection_layers = list(
            range(0, base, base // len(self.model_patch.model.control_blocks))
        )
        model.add_wrapper(extension.WrappersMP.DIFFUSION_MODEL, self.diffusion_model_wrapper)
        replacements = (
            model.model_options.get("transformer_options", {})
            .get("patches_replace", {})
            .get("dit", {})
        )
        for block in self.injection_layers:
            model.set_model_patch_replace(
                QwenImage21FunControlBlockPatch(
                    self, block, replacements.get(("single_block", block))
                ),
                "dit",
                "single_block",
                block,
            )


class QwenImage21FunControlBlockPatch:
    def __init__(
        self, control_patch: QwenImage21FunControlPatch, block: int, previous: Any
    ) -> None:
        self.control_patch, self.block, self.previous = control_patch, block, previous

    def __call__(self, args: dict[str, Any], extra_args: dict[str, Any]) -> dict[str, Any]:
        _, _, _, _, prefetch, _, _ = _modules()
        with prefetch.pause_malloc_graph():
            self.control_patch.before_block(self.block, args)
        output = (
            extra_args["original_block"](args)
            if self.previous is None
            else self.previous(args, extra_args)
        )
        with prefetch.pause_malloc_graph():
            return self.control_patch.after_block(self.block, args, output)

    def to(self, device_or_dtype: Any) -> QwenImage21FunControlBlockPatch:
        self.control_patch.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self) -> None:
        self.control_patch.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self) -> list[Any]:
        models = self.control_patch.models()
        if hasattr(self.previous, "models"):
            models += self.previous.models()
        return models


class QwenImageDiffsynthControlnet(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="QwenImageDiffsynthControlnet",
            display_name="Apply Qwen Image DiffSynth ControlNet",
            category="model/patch/qwen",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("model_patch", _MODEL_PATCH),
                InputSpec("vae", _VAE),
                InputSpec("image", _IMAGE),
                InputSpec(
                    "strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec("mask", _MASK, required=False),
                *_range(),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        model_patch: Any,
        vae: Any,
        image: object | None = None,
        strength: float = 1.0,
        inpaint_image: object | None = None,
        mask: object | None = None,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
    ) -> Mapping[str, object]:
        if strength == 0 or (image is None and inpaint_image is None and mask is None):
            return cls.outputs(MODEL=model)
        _, _, _, _, _, _, qwen = _modules()
        z_image = importlib.import_module("dinkster_inference.ldm.lumina.controlnet")
        image = None if image is None else materialize_image(image)[:, :, :, :3]
        inpaint_image = (
            None if inpaint_image is None else materialize_image(inpaint_image)[:, :, :, :3]
        )
        sampling = model.get_model_object("model_sampling")
        sigma_range = (
            float(sampling.percent_to_sigma(start_percent)),
            float(sampling.percent_to_sigma(end_percent)),
        )
        patched = model.clone()
        if isinstance(model_patch.model, qwen.QwenImage21FunControl):
            raw_mask = None if mask is None else materialize_image(mask)
            QwenImage21FunControlPatch(
                model_patch, vae, image, strength, inpaint_image, raw_mask, sigma_range
            ).register(patched)
            return cls.outputs(MODEL=patched)
        if mask is not None:
            mask_tensor = materialize_image(mask)
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(1)
            if mask_tensor.ndim == 4:
                mask_tensor = mask_tensor.unsqueeze(2)
            mask = 1.0 - mask_tensor
        if isinstance(model_patch.model, z_image.ZImage_Control):
            patch = ZImageControlPatch(
                model_patch, vae, image, strength, inpaint_image, mask, sigma_range
            )
            patched.set_model_noise_refiner_patch(patch)
            patched.set_model_double_block_patch(patch)
        else:
            patched.set_model_double_block_patch(
                DiffSynthCnetPatch(model_patch, vae, image, strength, mask, sigma_range)
            )
        return cls.outputs(MODEL=patched)


class ZImageFunControlnet(QwenImageDiffsynthControlnet):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ZImageFunControlnet",
            display_name="Apply Fun ControlNet",
            category="model/patch",
            search_terms=(
                "z-image controlnet",
                "qwen image 2.1 controlnet",
                "controlnet union",
                "fun controlnet",
            ),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("model_patch", _MODEL_PATCH),
                InputSpec("vae", _VAE),
                InputSpec(
                    "strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec("image", _IMAGE, required=False),
                InputSpec("inpaint_image", _IMAGE, required=False),
                InputSpec("mask", _MASK, required=False),
                *_range(),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )


class UsoStyleProjectorPatch:
    def __init__(self, model_patch: Any, encoded_image: Any) -> None:
        self.model_patch, self.encoded_image = model_patch, encoded_image

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        torch, *_ = _modules()
        text, ids = kwargs["txt"], kwargs["txt_ids"]
        embedding = self.model_patch.model(self.encoded_image.to(text.dtype)).to(text.dtype)
        kwargs["txt"] = torch.cat((embedding, text), dim=1)
        kwargs["txt_ids"] = torch.cat(
            (
                torch.zeros(
                    embedding.shape[0], embedding.shape[1], 3, dtype=ids.dtype, device=ids.device
                ),
                ids,
            ),
            dim=1,
        )
        return kwargs

    def to(self, device_or_dtype: Any) -> UsoStyleProjectorPatch:
        torch, *_ = _modules()
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self) -> list[Any]:
        return [self.model_patch]


class USOStyleReference(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="USOStyleReference",
            display_name="Apply USO Style Reference",
            category="model/patch/flux",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("model_patch", _MODEL_PATCH),
                InputSpec("clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, model_patch: Any, clip_vision_output: object
    ) -> Mapping[str, object]:
        torch, *_ = _modules()
        vision = materialize_clip_vision_output(clip_vision_output)
        encoded = torch.stack(
            (
                vision.all_hidden_states[:, -20],
                vision.all_hidden_states[:, -11],
                vision.penultimate_hidden_states,
            )
        )
        patched = model.clone()
        patched.set_model_post_input_patch(UsoStyleProjectorPatch(model_patch, encoded))
        return cls.outputs(MODEL=patched)


IMAGE_CONTROL_PATCH_NODES = (QwenImageDiffsynthControlnet, ZImageFunControlnet, USOStyleReference)
