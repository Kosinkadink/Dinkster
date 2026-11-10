from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    Node,
    NodeSchema,
    first_party_schema,
)
from dinkster_pack_helpers.video_latent import empty_video_latent as _empty_video_latent
from dinkster_pack_helpers.video_latent import video_latent_schema as _schema


class EmptyCosmosLatentVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "EmptyCosmosLatentVideo",
            "Empty Cosmos Latent Video",
            "model/latent/cosmos",
            width=1280,
            height=704,
            length=121,
            temporal_step=8,
        )
        return first_party_schema(schema)

    @classmethod
    def execute(
        cls, *, width: int, height: int, length: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        from dinkster_api.v1 import initialize_inference_process

        initialize_inference_process()
        return cls.outputs(LATENT=_empty_video_latent(batch_size, 16, length, 8, height, width, 8))
