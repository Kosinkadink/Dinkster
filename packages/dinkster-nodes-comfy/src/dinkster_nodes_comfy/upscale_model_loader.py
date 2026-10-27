"""Load single-image upscale checkpoints from granted model assets."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    UPSCALE_MODEL_TYPE,
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


class UpscaleModelLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="UpscaleModelLoader",
            display_name="Load Upscale Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(accept=("application/octet-stream",), kind="model/upscaler"),
                ),
            ),
            outputs=(OutputSpec("UPSCALE_MODEL", TypeExpr.concrete(UPSCALE_MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls, *, model_name: object) -> Mapping[str, object]:
        if not isinstance(model_name, (AssetRef, LocalAssetRef)):
            raise TypeError("model_name must be an AssetRef or LocalAssetRef")
        path = str(model_name.local_path())
        initialize_inference_process()
        spandrel = importlib.import_module("spandrel")
        try:
            extra = importlib.import_module("spandrel_extra_arches")
            spandrel.MAIN_REGISTRY.add(*extra.EXTRA_REGISTRY, ignore_duplicates=True)
            logging.info("Imported optional non-commercial upscale architectures.")
        except ModuleNotFoundError as error:
            if error.name != "spandrel_extra_arches":
                logging.warning("Optional upscale architecture import failed.", exc_info=True)
        except Exception:
            logging.warning("Optional upscale architecture registration failed.", exc_info=True)
        utils = importlib.import_module("dinkster_inference.utils")
        management = importlib.import_module("dinkster_inference.model_management")
        storage = importlib.import_module("dinkster_inference.storage")
        patcher = importlib.import_module("dinkster_inference.model_patcher")
        state_dict = utils.load_torch_file(path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
            state_dict = utils.state_dict_prefix_replace(state_dict, {"module.": ""})
        model = spandrel.ModelLoader().load_from_state_dict(state_dict).eval()
        if not isinstance(model, spandrel.ImageModelDescriptor):
            raise ValueError("Upscale model must be a single-image model.")
        model.patcher = patcher.CoreModelPatcher(
            model.model,
            load_device=management.get_torch_device(),
            offload_device=management.unet_offload_device(),
            fast_disk=storage.state_dict_fast_disk(state_dict),
        )
        return cls.outputs(UPSCALE_MODEL=model)
