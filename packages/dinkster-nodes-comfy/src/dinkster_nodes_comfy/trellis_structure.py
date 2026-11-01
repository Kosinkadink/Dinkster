"""Dense structure occupancy decoded by a Trellis VAE."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    LATENT_TYPE,
    VAE_TYPE,
    VOXEL_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)


class VaeDecodeStructureTrellis2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VaeDecodeStructureTrellis2",
            display_name="Trellis2 VAE Decode Structure",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "resolution",
                    TypeExpr.concrete(CORE_COMBO),
                    default="32",
                    widget=ComboWidget(options=("32", "64")),
                ),
            ),
            outputs=(OutputSpec("voxel", TypeExpr.concrete(VOXEL_TYPE)),),
        )

    @classmethod
    def execute(cls, *, samples: object, vae: Any, resolution: str = "32") -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        target = int(resolution)
        tensor = materialize_latent(samples)["samples"][:, :8]
        batch_number = vae.prepare_decode(tensor.shape)
        device = management.get_torch_device()
        batches: list[Any] = []
        for start in range(0, tensor.shape[0], batch_number):
            chunk = tensor[start : start + batch_number].to(device)
            batches.append(vae.first_stage_model.decode_structure(chunk.to(vae.vae_dtype)) > 0)
        decoded = torch.cat(batches, dim=0)
        current = decoded.shape[2]
        if current != target:
            ratio = current // target
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        return cls.outputs(voxel={"data": decoded.squeeze(1).float()})
