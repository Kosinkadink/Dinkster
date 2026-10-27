"""Load VAEs and Diffusers trees selected inside exact-kind grants."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    CORE_COMBO,
    MODEL_TYPE,
    VAE_TYPE,
    AssetRef,
    AssetWidget,
    Deprecation,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    mount_roots_for_kind,
    resolve_asset_selection,
)


class VAELoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VAELoader",
            display_name="Load VAE",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "vae_name",
                    TypeExpr.union(CORE_COMBO, ASSET_TYPE),
                    widget=AssetWidget(kind="model/vae", selection=True),
                ),
            ),
            outputs=(OutputSpec("VAE", TypeExpr.concrete(VAE_TYPE)),),
            # Source strings name live grants, not immutable file revisions.
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, vae_name: str | AssetRef | LocalAssetRef) -> Mapping[str, object]:
        selected = resolve_asset_selection("model/vae", vae_name)
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        sd = importlib.import_module("dinkster_inference.sd")
        metadata = None
        path = None
        state_dict: dict[str, Any]
        if selected is None:
            torch = importlib.import_module("torch")
            state_dict = {"pixel_space_vae": torch.tensor(1.0)}
        elif isinstance(selected, tuple):
            encoder, decoder = selected
            state_dict = {
                f"taesd_encoder.{key}": value
                for key, value in utils.load_torch_file(
                    str(encoder.local_path()), safe_load=True
                ).items()
            }
            state_dict.update(
                {
                    f"taesd_decoder.{key}": value
                    for key, value in utils.load_torch_file(
                        str(decoder.local_path()), safe_load=True
                    ).items()
                }
            )
            scale_shift = {
                "taesd": (0.18215, 0.0),
                "taesdxl": (0.13025, 0.0),
                "taesd3": (1.5305, 0.0609),
                "taef1": (0.3611, 0.1159),
            }
            if isinstance(vae_name, str) and vae_name in scale_shift:
                torch = importlib.import_module("torch")
                scale, shift = scale_shift[vae_name]
                state_dict["vae_scale"] = torch.tensor(scale)
                state_dict["vae_shift"] = torch.tensor(shift)
            elif vae_name == "taeqi2_1":
                latent_formats = importlib.import_module("dinkster_inference.latent_formats")
                latent_format = latent_formats.QwenImage21()
                state_dict["vae_scale"] = 1.0 / latent_format.latents_std[0]
                state_dict["vae_shift"] = latent_format.latents_mean[0]
            elif vae_name == "taef2":
                metadata = {"tae_latent_channels": 128}
        elif isinstance(selected, (AssetRef, LocalAssetRef)):
            path = str(selected.local_path())
            state_dict, metadata = utils.load_torch_file(path, safe_load=True, return_metadata=True)
        else:
            raise TypeError("VAE selection requires a file, encoder/decoder pair or pixel_space")
        vae = sd.VAE(sd=state_dict, metadata=metadata)
        vae.throw_exception_if_invalid()
        if path is not None:
            vae.patcher.cached_patcher_init = (sd.load_vae_patcher, (path, metadata, None))
        return cls.outputs(VAE=vae)


class DiffusersLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="DiffusersLoader",
            display_name="Load Diffusers Model (DEPRECATED)",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_path",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=AssetWidget(kind="model/diffusers", selection=True),
                ),
            ),
            outputs=(
                OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("VAE", TypeExpr.concrete(VAE_TYPE)),
            ),
            idempotent=False,
            deprecation=Deprecation("Deprecated in ComfyUI."),
            search_visibility="deprecated",
            search_terms=("load diffusers model",),
        )

    @classmethod
    def execute(cls, *, model_path: str) -> Mapping[str, object]:
        path = resolve_asset_selection("model/diffusers", model_path)
        if not isinstance(path, Path):
            raise TypeError("Diffusers selection requires a directory subtree")
        initialize_inference_process()
        diffusers_load = importlib.import_module("dinkster_inference.diffusers_load")
        model, clip, vae = diffusers_load.load_diffusers(
            str(path),
            output_vae=True,
            output_clip=True,
            embedding_directory=[str(root) for root in mount_roots_for_kind("model/embedding")],
        )
        return cls.outputs(MODEL=model, CLIP=clip, VAE=vae)


MODEL_SELECTION_NODES = (VAELoader, DiffusersLoader)
