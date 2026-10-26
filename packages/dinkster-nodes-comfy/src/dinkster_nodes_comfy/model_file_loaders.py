"""Native loaders for bound ComfyUI model assets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    CLIP_VISION_TYPE,
    CONTROL_NET_TYPE,
    CORE_COMBO,
    GLIGEN_TYPE,
    MODEL_TYPE,
    STYLE_MODEL_TYPE,
    VAE_TYPE,
    AssetRef,
    AssetWidget,
    ComboOption,
    ComboWidget,
    Deprecation,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    mount_roots_for_kind,
)

_ASSET = TypeExpr.concrete(ASSET_TYPE)
_CLIP = TypeExpr.concrete(CLIP_TYPE)
_CLIP_VISION = TypeExpr.concrete(CLIP_VISION_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_CONTROL_NET = TypeExpr.concrete(CONTROL_NET_TYPE)
_GLIGEN = TypeExpr.concrete(GLIGEN_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_STYLE_MODEL = TypeExpr.concrete(STYLE_MODEL_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_CLIP_TYPES = (
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux2",
    "ovis",
    "longcat_image",
    "cogvideox",
    "lens",
    "pixeldit",
    "ideogram4",
    "boogu",
    "krea2",
    "joyimage",
    "mage",
    "minimax",
    "yue2",
    "flux3",
    "moss_image",
    "fibo2",
)
_DUAL_CLIP_TYPES = (
    "sdxl",
    "sd3",
    "flux",
    "hunyuan_video",
    "hidream",
    "hunyuan_image",
    "hunyuan_video_15",
    "kandinsky5",
    "kandinsky5_image",
    "ltxv",
    "newbie",
    "ace",
)


def _asset(name: str, kind: str) -> InputSpec:
    return InputSpec(
        name,
        _ASSET,
        widget=AssetWidget(kind=kind, accept=("application/octet-stream",)),
    )


def _path(value: object, name: str) -> str:
    if not isinstance(value, (AssetRef, LocalAssetRef)):
        raise TypeError(f"{name} must be an AssetRef or LocalAssetRef")
    return str(value.local_path())


def _embedding_roots() -> list[str]:
    return [str(root) for root in mount_roots_for_kind("model/embedding")]


def _clip_options(device: str) -> dict[str, Any]:
    if device == "default":
        return {}
    torch = importlib.import_module("torch")
    return {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}


def _clip_type(name: str) -> object:
    sd = importlib.import_module("dinkster_inference.sd")
    return getattr(sd.CLIPType, name.upper(), sd.CLIPType.STABLE_DIFFUSION)


def _combo(
    name: str, options: tuple[str, ...], default: str, *, advanced: bool = False
) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        default=default,
        advanced=advanced,
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
    )


class CheckpointLoaderSimple(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CheckpointLoaderSimple",
            display_name="Load Checkpoint",
            category="model/loaders",
            inputs=(_asset("ckpt_name", "model/checkpoint"),),
            outputs=(
                OutputSpec("MODEL", _MODEL),
                OutputSpec("CLIP", _CLIP),
                OutputSpec("VAE", _VAE),
            ),
        )

    @classmethod
    def execute(cls, *, ckpt_name: object) -> Mapping[str, object]:
        path = _path(ckpt_name, "ckpt_name")
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        model, clip, vae, *_ = sd.load_checkpoint_guess_config(
            path, output_vae=True, output_clip=True, embedding_directory=_embedding_roots()
        )
        return cls.outputs(MODEL=model, CLIP=clip, VAE=vae)


class CheckpointLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CheckpointLoader",
            display_name="Load Checkpoint With Config (DEPRECATED)",
            category="model/loaders",
            deprecation=Deprecation("Deprecated in ComfyUI."),
            inputs=(
                _asset("config_name", "model/config"),
                _asset("ckpt_name", "model/checkpoint"),
            ),
            outputs=(
                OutputSpec("MODEL", _MODEL),
                OutputSpec("CLIP", _CLIP),
                OutputSpec("VAE", _VAE),
            ),
        )

    @classmethod
    def execute(cls, *, config_name: object, ckpt_name: object) -> Mapping[str, object]:
        config_path = _path(config_name, "config_name")
        checkpoint_path = _path(ckpt_name, "ckpt_name")
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        model, clip, vae = sd.load_checkpoint(
            config_path,
            checkpoint_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=_embedding_roots(),
        )
        return cls.outputs(MODEL=model, CLIP=clip, VAE=vae)


class unCLIPCheckpointLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="unCLIPCheckpointLoader",
            display_name="Load unCLIP Checkpoint",
            category="model/loaders",
            inputs=(_asset("ckpt_name", "model/checkpoint"),),
            outputs=(
                OutputSpec("MODEL", _MODEL),
                OutputSpec("CLIP", _CLIP),
                OutputSpec("VAE", _VAE),
                OutputSpec("CLIP_VISION", _CLIP_VISION),
            ),
        )

    @classmethod
    def execute(cls, *, ckpt_name: object) -> Mapping[str, object]:
        path = _path(ckpt_name, "ckpt_name")
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        model, clip, vae, clip_vision = sd.load_checkpoint_guess_config(
            path,
            output_vae=True,
            output_clip=True,
            output_clipvision=True,
            embedding_directory=_embedding_roots(),
        )
        return cls.outputs(MODEL=model, CLIP=clip, VAE=vae, CLIP_VISION=clip_vision)


class VAELoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VAELoader",
            display_name="Load VAE",
            category="model/loaders",
            inputs=(_asset("vae_name", "model/vae"),),
            outputs=(OutputSpec("VAE", _VAE),),
        )

    @classmethod
    def execute(cls, *, vae_name: object) -> Mapping[str, object]:
        path = _path(vae_name, "vae_name")
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        sd = importlib.import_module("dinkster_inference.sd")
        state_dict, metadata = utils.load_torch_file(path, safe_load=True, return_metadata=True)
        vae = sd.VAE(sd=state_dict, metadata=metadata)
        vae.throw_exception_if_invalid()
        return cls.outputs(VAE=vae)


class ControlNetLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ControlNetLoader",
            display_name="Load ControlNet Model",
            category="model/loaders",
            inputs=(_asset("control_net_name", "model/controlnet"),),
            outputs=(OutputSpec("CONTROL_NET", _CONTROL_NET),),
            search_terms=(
                "controlnet",
                "control net",
                "cn",
                "load controlnet",
                "controlnet loader",
            ),
        )

    @classmethod
    def execute(cls, *, control_net_name: object) -> Mapping[str, object]:
        path = _path(control_net_name, "control_net_name")
        initialize_inference_process()
        controlnet = importlib.import_module("dinkster_inference.controlnet").load_controlnet(path)
        if controlnet is None:
            raise RuntimeError(
                "controlnet file is invalid and does not contain a valid controlnet model"
            )
        return cls.outputs(CONTROL_NET=controlnet)


class DiffControlNetLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DiffControlNetLoader",
            display_name="Load ControlNet Model (diff)",
            category="model/loaders",
            inputs=(
                InputSpec("model", _MODEL),
                _asset("control_net_name", "model/controlnet"),
            ),
            outputs=(OutputSpec("CONTROL_NET", _CONTROL_NET),),
        )

    @classmethod
    def execute(cls, *, model: object, control_net_name: object) -> Mapping[str, object]:
        path = _path(control_net_name, "control_net_name")
        initialize_inference_process()
        controlnet = importlib.import_module("dinkster_inference.controlnet").load_controlnet(
            path, model
        )
        return cls.outputs(CONTROL_NET=controlnet)


class UNETLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="UNETLoader",
            display_name="Load Diffusion Model",
            category="model/loaders",
            inputs=(
                _asset("unet_name", "model/diffusion"),
                _combo(
                    "weight_dtype",
                    ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"),
                    "default",
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, unet_name: object, weight_dtype: str) -> Mapping[str, object]:
        path = _path(unet_name, "unet_name")
        initialize_inference_process()
        torch = importlib.import_module("torch")
        sd = importlib.import_module("dinkster_inference.sd")
        options: dict[str, object] = {}
        if weight_dtype == "fp8_e4m3fn":
            options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            options.update(dtype=torch.float8_e4m3fn, fp8_optimizations=True)
        elif weight_dtype == "fp8_e5m2":
            options["dtype"] = torch.float8_e5m2
        return cls.outputs(MODEL=sd.load_diffusion_model(path, model_options=options))


class CLIPLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPLoader",
            display_name="Load CLIP",
            category="model/loaders",
            inputs=(
                _asset("clip_name", "model/text-encoder"),
                _combo("type", _CLIP_TYPES, "stable_diffusion"),
                InputSpec(
                    "device",
                    _COMBO,
                    required=False,
                    default="default",
                    advanced=True,
                    widget=ComboWidget((ComboOption("default"), ComboOption("cpu"))),
                ),
            ),
            outputs=(OutputSpec("CLIP", _CLIP),),
        )

    @classmethod
    def execute(
        cls, *, clip_name: object, type: str, device: str = "default"
    ) -> Mapping[str, object]:
        path = _path(clip_name, "clip_name")
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        clip = sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=_embedding_roots(),
            clip_type=_clip_type(type),
            model_options=_clip_options(device),
        )
        return cls.outputs(CLIP=clip)


class DualCLIPLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DualCLIPLoader",
            display_name="Load CLIP (Dual)",
            category="model/loaders",
            inputs=(
                _asset("clip_name1", "model/text-encoder"),
                _asset("clip_name2", "model/text-encoder"),
                _combo("type", _DUAL_CLIP_TYPES, "sdxl"),
                InputSpec(
                    "device",
                    _COMBO,
                    required=False,
                    default="default",
                    advanced=True,
                    widget=ComboWidget((ComboOption("default"), ComboOption("cpu"))),
                ),
            ),
            outputs=(OutputSpec("CLIP", _CLIP),),
        )

    @classmethod
    def execute(
        cls, *, clip_name1: object, clip_name2: object, type: str, device: str = "default"
    ) -> Mapping[str, object]:
        paths = [_path(clip_name1, "clip_name1"), _path(clip_name2, "clip_name2")]
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        clip = sd.load_clip(
            ckpt_paths=paths,
            embedding_directory=_embedding_roots(),
            clip_type=_clip_type(type),
            model_options=_clip_options(device),
        )
        return cls.outputs(CLIP=clip)


class CLIPVisionLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPVisionLoader",
            display_name="Load CLIP Vision",
            category="model/loaders",
            inputs=(_asset("clip_name", "model/clip-vision"),),
            outputs=(OutputSpec("CLIP_VISION", _CLIP_VISION),),
        )

    @classmethod
    def execute(cls, *, clip_name: object) -> Mapping[str, object]:
        path = _path(clip_name, "clip_name")
        initialize_inference_process()
        clip_vision = importlib.import_module("dinkster_inference.clip_vision").load(path)
        if clip_vision is None:
            raise RuntimeError(
                "clip vision file is invalid and does not contain a valid vision model"
            )
        return cls.outputs(CLIP_VISION=clip_vision)


class StyleModelLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StyleModelLoader",
            display_name="Load Style Model",
            category="model/loaders",
            inputs=(_asset("style_model_name", "model/style"),),
            outputs=(OutputSpec("STYLE_MODEL", _STYLE_MODEL),),
        )

    @classmethod
    def execute(cls, *, style_model_name: object) -> Mapping[str, object]:
        path = _path(style_model_name, "style_model_name")
        initialize_inference_process()
        style_model = importlib.import_module("dinkster_inference.sd").load_style_model(path)
        return cls.outputs(STYLE_MODEL=style_model)


class GLIGENLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GLIGENLoader",
            display_name="Load GLIGEN Model",
            category="model/loaders",
            inputs=(_asset("gligen_name", "model/gligen"),),
            outputs=(OutputSpec("GLIGEN", _GLIGEN),),
        )

    @classmethod
    def execute(cls, *, gligen_name: object) -> Mapping[str, object]:
        path = _path(gligen_name, "gligen_name")
        initialize_inference_process()
        gligen = importlib.import_module("dinkster_inference.sd").load_gligen(path)
        return cls.outputs(GLIGEN=gligen)


MODEL_FILE_LOADER_NODES = (
    CheckpointLoaderSimple,
    CheckpointLoader,
    unCLIPCheckpointLoader,
    VAELoader,
    ControlNetLoader,
    DiffControlNetLoader,
    UNETLoader,
    CLIPLoader,
    DualCLIPLoader,
    CLIPVisionLoader,
    StyleModelLoader,
    GLIGENLoader,
)
