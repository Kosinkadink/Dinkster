"""Worker-only test pack producing encoded media without an engine-side source."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any, cast

import av
import numpy as np
from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_assets import register_video_value_type, resolver_from_env
from dinkster_values import TypeRegistry, video_from_source


def register_types(registry: TypeRegistry) -> None:
    register_video_value_type(registry, "comfy.VIDEO", resolver_from_env())


class ProduceVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vproducer.make",
            display_name="Produce Video",
            category="test",
            inputs=(),
            outputs=(OutputSpec("video", TypeExpr.concrete("comfy.VIDEO")),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        data = io.BytesIO()
        with av.open(data, "w", format="matroska") as opened:
            container = cast(Any, opened)
            stream = container.add_stream("ffv1", rate=10)
            stream.width = stream.height = 64
            stream.pix_fmt = "bgra"
            stream.codec_context.thread_count = 1
            rng = np.random.default_rng(1250)
            for index in range(30):
                frame = av.VideoFrame.from_ndarray(
                    rng.integers(0, 256, (64, 64, 4), dtype=np.uint8), format="rgba"
                )
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return cls.outputs(video=video_from_source(data.getvalue()))


NODES = (ProduceVideo,)
