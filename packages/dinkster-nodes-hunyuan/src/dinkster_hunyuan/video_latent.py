from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    Node,
    NodeSchema,
    with_identity_replacement,
)
from dinkster_pack_helpers.video_latent import empty_video_latent as _empty_video_latent
from dinkster_pack_helpers.video_latent import video_latent_schema as _schema


class EmptyHunyuanLatentVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "dinkster.hunyuan.empty_hunyuan_latent_video",
            "Empty Hunyuan Video 1.0 Latent",
            "model/latent/hunyuan video",
            width=848,
            height=480,
            length=25,
        )
        return with_identity_replacement(schema, "EmptyHunyuanLatentVideo")

    @classmethod
    def execute(
        cls, *, width: int, height: int, length: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        from dinkster_api.v1 import initialize_inference_process

        initialize_inference_process()
        return cls.outputs(
            LATENT=_empty_video_latent(
                batch_size,
                16,
                length,
                4,
                height,
                width,
                8,
                downscale_ratio_spacial=8,
            )
        )


class EmptyHunyuanVideo15Latent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "dinkster.hunyuan.empty_hunyuan_video15_latent",
            "Empty Hunyuan Video 1.5 Latent",
            "model/latent/hunyuan video",
            width=848,
            height=480,
            length=25,
        )
        return with_identity_replacement(schema, "EmptyHunyuanVideo15Latent")

    @classmethod
    def execute(
        cls, *, width: int, height: int, length: int, batch_size: int = 1
    ) -> Mapping[str, object]:
        from dinkster_api.v1 import initialize_inference_process

        initialize_inference_process()
        return cls.outputs(
            LATENT=_empty_video_latent(
                batch_size,
                32,
                length,
                4,
                height,
                width,
                16,
                downscale_ratio_spacial=16,
            )
        )
