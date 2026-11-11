from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    Node,
    NodeSchema,
    with_identity_replacement,
)
from dinkster_pack_helpers.video_latent import empty_video_latent as _empty_video_latent
from dinkster_pack_helpers.video_latent import video_latent_schema as _schema


class EmptyCosmosLatentVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "dinkster.cosmos.empty_cosmos_latent_video",
            "Empty Cosmos Latent Video",
            "model/latent/cosmos",
            width=1280,
            height=704,
            length=121,
            temporal_step=8,
        )
        return with_identity_replacement(schema, "EmptyCosmosLatentVideo")

    @classmethod
    def execute(
        cls, *, width: int, height: int, length: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        from dinkster_api.v1 import initialize_inference_process

        initialize_inference_process()
        return cls.outputs(LATENT=_empty_video_latent(batch_size, 16, length, 8, height, width, 8))
