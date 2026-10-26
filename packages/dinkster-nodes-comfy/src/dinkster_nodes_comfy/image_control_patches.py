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
            advanced=True,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        ),
        InputSpec(
            "end_percent",
            _FLOAT,
            default=1.0,
            advanced=True,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        ),
    )


def _in_sigma_range(options: Mapping[str, Any], sigma_range: tuple[float, float]) -> bool:
    sigma = float(options["sigmas"].flatten()[0])
    return sigma_range[1] <= sigma <= sigma_range[0]


class _DiffSynthControlPatch:
    def __init__(
        self,
        model_patch: Any,
        vae: Any,
        image: Any,
        strength: float,
        mask: Any,
        sigma_range: tuple[float, float],
    ) -> None:
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.sigma_range = sigma_range
        self.encoded_image = self._encode(image)
        self.encoded_image_size = tuple(image.shape[1:3])

    def _encode(self, image: Any) -> Any:
        torch = importlib.import_module("torch")
        latent = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim <= 0:
            return self.model_patch.model.process_input_latent_image(latent)
        if self.mask is None:
            mask = torch.ones_like(latent)[:, : self.model_patch.model.additional_in_dim // 4]
        else:
            mask = self.mask.mean(dim=1, keepdim=True)
        return self.model_patch.model.process_input_latent_image(torch.cat((latent, mask), dim=1))

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not _in_sigma_range(kwargs["transformer_options"], self.sigma_range):
            return kwargs
        image = kwargs["img"]
        block_id = kwargs["block_index"]
        encoded = self.encoded_image.to(image.dtype)
        image[:, : encoded.shape[1]] += (
            self.model_patch.model.control_block(image[:, : encoded.shape[1]], encoded, block_id)
            * self.strength
        )
        return kwargs

    def models(self) -> list[Any]:
        return [self.model_patch]


class _ZImageControlPatch:
    def __init__(
        self,
        model_patch: Any,
        vae: Any,
        image: Any,
        inpaint_image: Any,
        mask: Any,
        strength: float,
        sigma_range: tuple[float, float],
    ) -> None:
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.inpaint_image = inpaint_image
        self.mask = mask
        self.strength = strength
        self.sigma_range = sigma_range
        self.temp_data: tuple[int, tuple[Any, Any]] | None = None

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not _in_sigma_range(kwargs["transformer_options"], self.sigma_range):
            return kwargs
        image = kwargs.pop("img")
        text = kwargs.pop("txt")
        block_id = kwargs["block_index"]
        layers = self.model_patch.model.n_control_layers
        layer = block_id // round(30 / layers)
        if layer >= layers:
            self.temp_data = None
            return kwargs
        encoded = self.vae.encode(self.image if self.image is not None else self.inpaint_image)
        if self.temp_data is None or self.temp_data[0] > layer:
            self.temp_data = (
                -1,
                self.model_patch.model(text, encoded.to(image.dtype), kwargs["pe"], kwargs["vec"]),
            )
        while self.temp_data[0] < layer:
            current = self.temp_data[0] + 1
            self.temp_data = (
                current,
                self.model_patch.model.forward_control_block(
                    current,
                    self.temp_data[1][1],
                    kwargs["img_input"],
                    None,
                    kwargs["pe"],
                    kwargs["vec"],
                ),
            )
        control = self.temp_data[1][0]
        image[:, : control.shape[1]] += control * self.strength
        if layer + 1 == layers:
            self.temp_data = None
        return kwargs

    def models(self) -> list[Any]:
        return [self.model_patch]


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
        mask: Any = None,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        inpaint_image: object | None = None,
    ) -> Mapping[str, object]:
        if strength == 0 or (image is None and inpaint_image is None and mask is None):
            return cls.outputs(MODEL=model)
        source = materialize_image(image) if image is not None else None
        inpaint = materialize_image(inpaint_image) if inpaint_image is not None else None
        sampling = model.get_model_object("model_sampling")
        sigma_range = (
            float(sampling.percent_to_sigma(start_percent)),
            float(sampling.percent_to_sigma(end_percent)),
        )
        patched = model.clone()
        if hasattr(model_patch.model, "n_control_layers"):
            patch = _ZImageControlPatch(
                model_patch, vae, source, inpaint, mask, strength, sigma_range
            )
            patched.set_model_noise_refiner_patch(patch)
            patched.set_model_double_block_patch(patch)
        else:
            patched.set_model_double_block_patch(
                _DiffSynthControlPatch(model_patch, vae, source, strength, mask, sigma_range)
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


class _UsoStyleProjectorPatch:
    def __init__(self, model_patch: Any, encoded_image: Any) -> None:
        self.model_patch = model_patch
        self.encoded_image = encoded_image

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        torch = importlib.import_module("torch")
        text = kwargs["txt"]
        embedding = self.model_patch.model(self.encoded_image.to(text.dtype)).to(text.dtype)
        kwargs["txt"] = torch.cat((embedding, text), dim=1)
        kwargs["txt_ids"] = torch.cat(
            (
                torch.zeros(
                    embedding.shape[0],
                    embedding.shape[1],
                    3,
                    dtype=kwargs["txt_ids"].dtype,
                    device=kwargs["txt_ids"].device,
                ),
                kwargs["txt_ids"],
            ),
            dim=1,
        )
        return kwargs

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
        torch = importlib.import_module("torch")
        vision = materialize_clip_vision_output(clip_vision_output)
        encoded = torch.stack(
            (
                vision.all_hidden_states[:, -20],
                vision.all_hidden_states[:, -11],
                vision.penultimate_hidden_states,
            )
        )
        patched = model.clone()
        patched.set_model_post_input_patch(_UsoStyleProjectorPatch(model_patch, encoded))
        return cls.outputs(MODEL=patched)


IMAGE_CONTROL_PATCH_NODES = (QwenImageDiffsynthControlnet, ZImageFunControlnet, USOStyleReference)
