"""Native loaders for bound ComfyUI model assets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    CLIP_VISION_TYPE,
    GLIGEN_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    mount_roots_for_kind,
    with_identity_replacement,
)

_ASSET = TypeExpr.concrete(ASSET_TYPE)

_CLIP = TypeExpr.concrete(CLIP_TYPE)

_CLIP_VISION = TypeExpr.concrete(CLIP_VISION_TYPE)

_GLIGEN = TypeExpr.concrete(GLIGEN_TYPE)

_MODEL = TypeExpr.concrete(MODEL_TYPE)

_VAE = TypeExpr.concrete(VAE_TYPE)


def _asset(name: str, kind: str, *, doc: str = "") -> InputSpec:
    return InputSpec(
        name,
        _ASSET,
        doc=doc,
        widget=AssetWidget(kind=kind, accept=("application/octet-stream",)),
    )


def _path(value: object, name: str) -> str:
    if not isinstance(value, (AssetRef, LocalAssetRef)):
        raise TypeError(f"{name} must be an AssetRef or LocalAssetRef")
    return str(value.local_path())


def _embedding_roots() -> list[str]:
    return [str(root) for root in mount_roots_for_kind("model/embedding")]


class unCLIPCheckpointLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.un_clip_checkpoint_loader",
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
        return with_identity_replacement(schema, "unCLIPCheckpointLoader")

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


class GLIGENLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.gligen_loader",
            display_name="Load GLIGEN Model",
            category="model/loaders",
            inputs=(_asset("gligen_name", "model/gligen"),),
            outputs=(OutputSpec("GLIGEN", _GLIGEN),),
        )
        return with_identity_replacement(schema, "GLIGENLoader")

    @classmethod
    def execute(cls, *, gligen_name: object) -> Mapping[str, object]:
        path = _path(gligen_name, "gligen_name")
        initialize_inference_process()
        gligen = importlib.import_module("dinkster_inference.sd").load_gligen(path)
        return cls.outputs(GLIGEN=gligen)


class ImageOnlyCheckpointLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.image_only_checkpoint_loader",
            display_name="Load Checkpoint Image Only (img2vid model)",
            category="model/loaders",
            inputs=(_asset("ckpt_name", "model/checkpoint"),),
            outputs=(
                OutputSpec("MODEL", _MODEL),
                OutputSpec("CLIP_VISION", _CLIP_VISION),
                OutputSpec("VAE", _VAE),
            ),
        )
        return with_identity_replacement(schema, "ImageOnlyCheckpointLoader")

    @classmethod
    def execute(cls, *, ckpt_name: object) -> Mapping[str, object]:
        path = _path(ckpt_name, "ckpt_name")
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        model, _, vae, vision = sd.load_checkpoint_guess_config(
            path,
            output_vae=True,
            output_clip=False,
            output_clipvision=True,
            embedding_directory=_embedding_roots(),
        )
        return cls.outputs(MODEL=model, CLIP_VISION=vision, VAE=vae)
