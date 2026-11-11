"""Silent audio construction."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_FLOAT,
    CORE_INT,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    with_identity_replacement,
)


class EmptyAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.audio.empty_audio",
            display_name="Empty Audio",
            category="audio",
            inputs=(
                InputSpec(
                    "duration",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=60.0,
                    widget=NumberWidget(min=0.0, max=float(0xFFFFFFFFFFFFFFFF), step=0.01),
                ),
                InputSpec(
                    "sample_rate",
                    TypeExpr.concrete(CORE_INT),
                    default=44100,
                    widget=NumberWidget(min=1, max=192000),
                    advanced=True,
                ),
                InputSpec(
                    "channels",
                    TypeExpr.concrete(CORE_INT),
                    default=2,
                    widget=NumberWidget(min=1, max=2),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("AUDIO", TypeExpr.concrete(AUDIO_TYPE)),),
        )
        return with_identity_replacement(schema, "EmptyAudio")

    @classmethod
    def execute(cls, *, duration: float, sample_rate: int, channels: int) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        waveform = torch.zeros(
            (1, channels, int(round(duration * sample_rate))), dtype=torch.float32
        )
        return cls.outputs(AUDIO={"waveform": waveform, "sample_rate": sample_rate})


AUDIO_GENERATOR_NODES = (EmptyAudio,)
