"""Qwen Image model nodes."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

ASSET = TypeExpr.concrete("dinkster.asset")
MODEL = TypeExpr.concrete("dinkster.model")
CLIP = TypeExpr.concrete("dinkster.clip")
VAE = TypeExpr.concrete("dinkster.vae")
CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
LATENT = TypeExpr.concrete("dinkster.latent")
IMAGE = TypeExpr.concrete("dinkster.image")
QWEN_IMAGE_CONTROL = TypeExpr.concrete("dinkster.qwen_image_control")
QWEN_IMAGE_DIFFSYNTH = TypeExpr.concrete("dinkster.qwen_image_diffsynth")
STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)


class LoadQwenImageControl(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_qwen_image_control",
            display_name="Load Qwen Image ControlNet",
            category="model/loaders/qwen-image",
            inputs=(
                InputSpec(
                    "control_net",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/controlnet",
                    ),
                ),
            ),
            outputs=(OutputSpec("control", QWEN_IMAGE_CONTROL),),
            search_terms=("qwen", "controlnet", "instantx", "fun"),
        )

    @classmethod
    def execute(cls, control_net: object) -> Mapping[str, object]:
        from .provider import execute_load_qwen_image_control

        return execute_load_qwen_image_control(control_net=control_net)


class ApplyQwenImageControl(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_qwen_image_control",
            display_name="Apply Qwen Image ControlNet",
            category="model/control/qwen-image",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("control", QWEN_IMAGE_CONTROL),
                InputSpec("hint", LATENT),
                InputSpec("strength", FLOAT, default=1.0, widget=NumberWidget(step=0.01)),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=("qwen", "controlnet", "instantx", "fun"),
        )

    @classmethod
    def execute(
        cls,
        model: object,
        control: object,
        hint: object,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
    ) -> Mapping[str, object]:
        from .provider import execute_apply_qwen_image_control

        return execute_apply_qwen_image_control(
            model=model,
            control=control,
            hint=hint,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
        )


class LoadQwenImageDiffSynth(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_qwen_image_diffsynth",
            display_name="Load Qwen Image DiffSynth ControlNet",
            category="model/loaders/qwen-image",
            inputs=(
                InputSpec(
                    "model_patch",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/patch",
                    ),
                ),
            ),
            outputs=(OutputSpec("patch", QWEN_IMAGE_DIFFSYNTH),),
            search_terms=("qwen", "diffsynth", "controlnet"),
        )

    @classmethod
    def execute(cls, model_patch: object) -> Mapping[str, object]:
        from .provider import execute_load_qwen_image_diffsynth

        return execute_load_qwen_image_diffsynth(model_patch=model_patch)


class ApplyQwenImageDiffSynth(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_qwen_image_diffsynth",
            display_name="Apply Qwen Image DiffSynth ControlNet",
            category="model/control/qwen-image",
            description="Applies a DiffSynth patch using a pre-encoded Qwen Image control latent.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("patch", QWEN_IMAGE_DIFFSYNTH),
                InputSpec("hint", LATENT),
                InputSpec("strength", FLOAT, default=1.0, widget=NumberWidget(step=0.01)),
            ),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=("qwen", "diffsynth", "controlnet"),
        )

    @classmethod
    def execute(
        cls,
        model: object,
        patch: object,
        hint: object,
        strength: float = 1.0,
    ) -> Mapping[str, object]:
        from .provider import execute_apply_qwen_image_diffsynth

        return execute_apply_qwen_image_diffsynth(
            model=model,
            patch=patch,
            hint=hint,
            strength=strength,
        )


class QwenImageEditEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.qwen_image_edit_encode",
            display_name="TextEncodeQwenImageEdit",
            category="model/conditioning/qwen-image",
            description="Encodes a Qwen Image Edit instruction with an optional reference.",
            inputs=(
                InputSpec("clip", CLIP),
                InputSpec("prompt", STRING),
                InputSpec("vae", VAE, required=False, default=None),
                InputSpec("image", IMAGE, required=False, default=None),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("TextEncodeQwenImageEdit",),
            search_terms=("qwen", "image edit", "reference image"),
        )

    @classmethod
    def execute(
        cls,
        clip: object,
        prompt: str,
        vae: object | None = None,
        image: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_qwen_image_edit_encode

        return execute_qwen_image_edit_encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            image=image,
        )


class QwenImageEditPlusEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.qwen_image_edit_plus_encode",
            display_name="TextEncodeQwenImageEditPlus",
            category="model/conditioning/qwen-image",
            description="Encodes a Qwen Image Edit Plus instruction with up to three references.",
            inputs=(
                InputSpec("clip", CLIP),
                InputSpec("prompt", STRING),
                InputSpec("vae", VAE, required=False, default=None),
                InputSpec("image1", IMAGE, required=False, default=None),
                InputSpec("image2", IMAGE, required=False, default=None),
                InputSpec("image3", IMAGE, required=False, default=None),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("TextEncodeQwenImageEditPlus",),
            search_terms=("qwen", "image edit", "reference image", "edit plus"),
        )

    @classmethod
    def execute(
        cls,
        clip: object,
        prompt: str,
        vae: object | None = None,
        image1: object | None = None,
        image2: object | None = None,
        image3: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_qwen_image_edit_plus_encode

        return execute_qwen_image_edit_plus_encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            images=(image1, image2, image3),
        )


class EmptyQwenImageLayeredLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_qwen_image_layered_latent",
            display_name="Empty Qwen Image Layered Latent",
            category="model/latent/qwen-image",
            inputs=(
                InputSpec(
                    "width", INT, default=640, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=640, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("layers", INT, default=3, widget=NumberWidget(min=0, max=4096, step=1)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("EmptyQwenImageLayeredLatentImage",),
            search_terms=("qwen", "layered", "empty latent"),
        )

    @classmethod
    def execute(
        cls, width: int, height: int, layers: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        from .provider import execute_empty_qwen_image_layered_latent

        return execute_empty_qwen_image_layered_latent(
            width=width,
            height=height,
            layers=layers,
            batch_size=batch_size,
        )


QWEN_IMAGE_MODEL_NODES: tuple[type[Node], ...] = (
    LoadQwenImageControl,
    ApplyQwenImageControl,
    LoadQwenImageDiffSynth,
    ApplyQwenImageDiffSynth,
    QwenImageEditEncode,
    QwenImageEditPlusEncode,
    EmptyQwenImageLayeredLatent,
)
QWEN_IMAGE_MODEL_NODE_IDS = tuple(node.schema().node_type for node in QWEN_IMAGE_MODEL_NODES)

__all__ = ["QWEN_IMAGE_MODEL_NODE_IDS", "QWEN_IMAGE_MODEL_NODES"]
