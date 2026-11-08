"""Bound audio encoder checkpoints and native music conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    AUDIO_ENCODER_TYPE,
    AUDIO_TYPE,
    CORE_COMBO,
    CORE_STRING,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    first_party_schema,
    initialize_inference_process,
    materialize_audio,
)

_AUDIO_ENCODER = TypeExpr.concrete(AUDIO_ENCODER_TYPE)

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


class SheetSage2AudioToABC(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="SheetSage2AudioToABC",
            display_name="SheetSage2 Audio to ABC",
            category="model/conditioning/yue2",
            description=(
                "Transcribes vocal and instrumental melodies from music into ABC notation. "
                "Connect abc output to YuE2 Generate Music node and use the matching mode."
            ),
            inputs=(
                InputSpec("audio_encoder", _AUDIO_ENCODER),
                InputSpec("audio", _AUDIO),
                InputSpec(
                    "mode",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(options=("melody", "full")),
                    doc=(
                        "full: generates melody and chords; melody: generates melody only, "
                        "recommended for covers."
                    ),
                ),
            ),
            outputs=(OutputSpec("abc", TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))),),
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, audio_encoder: Any, audio: object, mode: str) -> Mapping[str, object]:
        initialize_inference_process()
        source = materialize_audio(audio)
        return cls.outputs(
            abc=audio_encoder.generate_abc(
                source["waveform"], source["sample_rate"], melody_only=mode == "melody"
            )
        )
