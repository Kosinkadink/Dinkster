"""Load Hunyuan Video and LTXV latent-upscale checkpoints."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)

_ASSET = TypeExpr.concrete(ASSET_TYPE)
_LATENT_UPSCALE_MODEL = TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)


def _asset_path(model_name: object) -> str:
    if not isinstance(model_name, (AssetRef, LocalAssetRef)):
        raise TypeError("model_name must be an AssetRef or LocalAssetRef")
    return str(model_name.local_path())


def _hunyuan_720p_config(state_dict: Mapping[str, Any]) -> dict[str, object]:
    return {
        "in_channels": state_dict["in_conv.conv.weight"].shape[1],
        "out_channels": state_dict["out_conv.conv.weight"].shape[0],
        "hidden_channels": state_dict["in_conv.conv.weight"].shape[0],
        "num_blocks": len(
            [
                key
                for key in state_dict
                if key.startswith("blocks.") and key.endswith(".block.0.conv.weight")
            ]
        ),
        "global_residual": False,
    }


def _hunyuan_1080p_config(state_dict: Mapping[str, Any]) -> dict[str, object]:
    return {
        "z_channels": state_dict["conv_in.conv.weight"].shape[1],
        "out_channels": state_dict["conv_out.conv.weight"].shape[0],
        "block_out_channels": tuple(
            state_dict[f"up.{index}.block.0.conv1.conv.weight"].shape[0]
            for index in range(
                len(
                    [
                        key
                        for key in state_dict
                        if key.startswith("up.") and key.endswith(".block.0.conv1.conv.weight")
                    ]
                )
            )
        ),
    }


class LatentUpscaleModelLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LatentUpscaleModelLoader",
            display_name="Load Latent Upscale Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_name",
                    _ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/latent-upscaler"
                    ),
                ),
            ),
            outputs=(OutputSpec("LATENT_UPSCALE_MODEL", _LATENT_UPSCALE_MODEL),),
        )

    @classmethod
    def execute(cls, *, model_name: object) -> Mapping[str, object]:
        initialize_inference_process()
        path = _asset_path(model_name)
        utils = importlib.import_module("dinkster_inference.utils")
        storage = importlib.import_module("dinkster_inference.storage")
        state_dict, metadata = utils.load_torch_file(path, safe_load=True, return_metadata=True)
        fast_disk = storage.state_dict_fast_disk(state_dict)

        if "blocks.0.block.0.conv.weight" in state_dict:
            upsampler = importlib.import_module("dinkster_inference.ldm.hunyuan_video.upsampler")
            model = upsampler.HunyuanVideo15SRModel(
                "720p", _hunyuan_720p_config(state_dict), fast_disk=fast_disk
            )
            model.load_sd(state_dict)
        elif "up.0.block.0.conv1.conv.weight" in state_dict:
            state_dict = {
                key.replace("nin_shortcut", "nin_shortcut.conv", 1): value
                for key, value in state_dict.items()
            }
            upsampler = importlib.import_module("dinkster_inference.ldm.hunyuan_video.upsampler")
            model = upsampler.HunyuanVideo15SRModel(
                "1080p", _hunyuan_1080p_config(state_dict), fast_disk=fast_disk
            )
            model.load_sd(state_dict)
        elif "post_upsample_res_blocks.0.conv2.bias" in state_dict:
            torch = importlib.import_module("torch")
            operations = importlib.import_module("dinkster_inference.ops")
            management = importlib.import_module("dinkster_inference.model_management")
            patcher = importlib.import_module("dinkster_inference.model_patcher")
            upsampler = importlib.import_module(
                "dinkster_inference.ldm.lightricks.latent_upsampler"
            )
            config = json.loads(metadata["config"])
            loaded = upsampler.LatentUpsampler.from_config(
                config, operations=operations.disable_weight_init
            ).to(dtype=management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]))
            management.archive_model_dtypes(loaded)
            model = patcher.CoreModelPatcher(
                loaded,
                load_device=management.get_torch_device(),
                offload_device=management.unet_offload_device(),
                fast_disk=fast_disk,
            )
            loaded.load_state_dict(state_dict, assign=model.is_dynamic())
        else:
            raise ValueError("unsupported latent upscale checkpoint")

        return cls.outputs(LATENT_UPSCALE_MODEL=model)


LATENT_UPSCALE_LOADER_NODES = (LatentUpscaleModelLoader,)
