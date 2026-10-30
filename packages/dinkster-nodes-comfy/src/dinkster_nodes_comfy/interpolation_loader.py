"""Load FILM and RIFE frame interpolation checkpoints from granted assets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    INTERP_MODEL_TYPE,
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


class FrameInterpolationModelLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FrameInterpolationModelLoader",
            display_name="Load Frame Interpolation Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    doc=(
                        "Select a frame interpolation model to load. Models must be placed "
                        "in the 'frame_interpolation' folder."
                    ),
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/frame-interpolation"
                    ),
                ),
            ),
            outputs=(OutputSpec("INTERP_MODEL", TypeExpr.concrete(INTERP_MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls, *, model_name: object) -> Mapping[str, object]:
        if not isinstance(model_name, (AssetRef, LocalAssetRef)):
            raise TypeError("model_name must be an AssetRef or LocalAssetRef")
        path = model_name.local_path()
        initialize_inference_process()
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        management = importlib.import_module("dinkster_inference.model_management")
        storage = importlib.import_module("dinkster_inference.storage")
        patcher = importlib.import_module("dinkster_inference.model_patcher")
        state = utils.load_torch_file(str(path), safe_load=True)
        model = cls._detect_and_load(state)
        dtype = (
            torch.float16
            if management.should_use_fp16(management.get_torch_device())
            else torch.float32
        )
        model.eval().to(dtype)
        loaded = patcher.CoreModelPatcher(
            model,
            load_device=management.get_torch_device(),
            offload_device=management.unet_offload_device(),
            fast_disk=storage.state_dict_fast_disk(state),
        )
        return cls.outputs(INTERP_MODEL=loaded)

    @classmethod
    def _detect_and_load(cls, state: Any) -> Any:
        from .interpolation_models import film_model, rife_model

        if "extract.extract_sublevels.convs.0.0.conv.weight" in state:
            model = film_model()
            model.load_state_dict(state)
            return model
        utils = importlib.import_module("dinkster_inference.utils")
        state = utils.state_dict_prefix_replace(state, {"module.": "", "flownet.": ""})
        key_map: dict[str, str] = {}
        for key in state:
            for index in range(5):
                if key.startswith(f"block{index}."):
                    key_map[key] = f"blocks.{index}.{key[len(f'block{index}.') :]}"
        if key_map:
            state = {key_map.get(key, key): value for key, value in state.items()}
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(("teacher.", "caltime."))
        }
        try:
            head_ch = state["encode.cnn3.weight"].shape[1]
            channels = [
                state[f"blocks.{index}.conv0.1.0.weight"].shape[0]
                for index in range(5)
                if f"blocks.{index}.conv0.1.0.weight" in state
            ]
            if len(channels) != 5:
                raise ValueError("Unsupported RIFE model")
        except (KeyError, ValueError):
            raise ValueError("Unrecognized frame interpolation model format") from None
        model = rife_model(head_ch, channels)
        model.load_state_dict(state)
        return model
