"""Load RAFT-large optical-flow models from granted model assets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    OPTICAL_FLOW_TYPE,
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


class OpticalFlowLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="OpticalFlowLoader",
            display_name="Load Optical Flow Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/optical-flow"
                    ),
                    doc=(
                        "Optical flow model to load. Today only torchvision's "
                        "RAFT-large state dict is supported."
                    ),
                ),
            ),
            outputs=(OutputSpec("OPTICAL_FLOW", TypeExpr.concrete(OPTICAL_FLOW_TYPE)),),
        )

    @classmethod
    def execute(cls, *, model_name: object) -> Mapping[str, object]:
        if not isinstance(model_name, (AssetRef, LocalAssetRef)):
            raise TypeError("model_name must be an AssetRef or LocalAssetRef")
        path = str(model_name.local_path())
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        state_dict = utils.load_torch_file(path, safe_load=True)
        if not all(
            any(key.startswith(prefix) for key in state_dict)
            for prefix in ("feature_encoder.", "context_encoder.", "update_block.")
        ):
            raise ValueError(
                "Unrecognized optical flow model format: expected a torchvision "
                "RAFT-large state dict with 'feature_encoder.', 'context_encoder.' "
                "and 'update_block.' prefixes."
            )
        torch = importlib.import_module("torch")
        optical_flow = importlib.import_module("torchvision.models.optical_flow")
        management = importlib.import_module("dinkster_inference.model_management")
        storage = importlib.import_module("dinkster_inference.storage")
        patcher = importlib.import_module("dinkster_inference.model_patcher")
        model = optical_flow.raft_large(weights=None, progress=False)
        model.load_state_dict(state_dict)
        model.eval().to(torch.float32)
        loaded = patcher.CoreModelPatcher(
            model,
            load_device=management.get_torch_device(),
            offload_device=management.unet_offload_device(),
            fast_disk=storage.state_dict_fast_disk(state_dict),
        )
        return cls.outputs(OPTICAL_FLOW=loaded)
