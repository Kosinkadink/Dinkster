from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    Node,
    NodeSchema,
    first_party_schema,
)
from dinkster_nodes_latent.latent_model_specific import _integer, _runtime, _schema


class EmptyLatentHunyuan3Dv2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "EmptyLatentHunyuan3Dv2",
            "model/latent/hunyuan 3d",
            (_integer("resolution", 3072, 8192), _integer("batch_size", 1, 4096)),
            "Empty Hunyuan 3D v2 Latent",
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, resolution: int, batch_size: int) -> Mapping[str, object]:
        torch, management = _runtime()
        samples = torch.zeros([batch_size, 64, resolution], device=management.intermediate_device())
        return cls.outputs(LATENT={"samples": samples, "type": "hunyuan3dv2"})
