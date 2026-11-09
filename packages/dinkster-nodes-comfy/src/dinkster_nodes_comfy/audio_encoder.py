"""Bound audio encoder checkpoints and native music conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_ENCODER_TYPE,
    AUDIO_TYPE,
    CORE_COMBO,
    CORE_STRING,
    AssetRef,
    AssetWidget,
    ComboWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
)

_AUDIO_ENCODER = TypeExpr.concrete(AUDIO_ENCODER_TYPE)
_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


class AudioEncoderLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="AudioEncoderLoader",
            display_name="Load Audio Encoder",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "audio_encoder_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/audio-encoder"
                    ),
                ),
            ),
            outputs=(OutputSpec("AUDIO_ENCODER", _AUDIO_ENCODER),),
        )

    @classmethod
    def execute(cls, *, audio_encoder_name: object) -> Mapping[str, object]:
        if not isinstance(audio_encoder_name, (AssetRef, LocalAssetRef)):
            raise TypeError("audio_encoder_name must be an AssetRef or LocalAssetRef")
        path = str(audio_encoder_name.local_path())
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        encoders = importlib.import_module("dinkster_inference.audio_encoders.audio_encoders")
        state_dict = utils.load_torch_file(path, safe_load=True)
        encoder = encoders.load_audio_encoder_from_sd(state_dict)
        if encoder is None:
            raise RuntimeError(
                "ERROR: audio encoder file is invalid and does not contain a valid model."
            )
        encoder._dinkster_resident_owner = encoder.patcher
        return cls.outputs(AUDIO_ENCODER=encoder)


class AudioEncoderEncode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="AudioEncoderEncode",
            category="model/conditioning",
            inputs=(InputSpec("audio_encoder", _AUDIO_ENCODER), InputSpec("audio", _AUDIO)),
            outputs=(
                OutputSpec("AUDIO_ENCODER_OUTPUT", TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, audio_encoder: Any, audio: object) -> Mapping[str, object]:
        initialize_inference_process()
        source = materialize_audio(audio)
        output = audio_encoder.encode_audio(source["waveform"], source["sample_rate"])
        return cls.outputs(AUDIO_ENCODER_OUTPUT=output)


class SheetSage2AudioToABC(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
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

    @classmethod
    def execute(cls, *, audio_encoder: Any, audio: object, mode: str) -> Mapping[str, object]:
        initialize_inference_process()
        source = materialize_audio(audio)
        return cls.outputs(
            abc=audio_encoder.generate_abc(
                source["waveform"], source["sample_rate"], melody_only=mode == "melody"
            )
        )


AUDIO_ENCODER_NODES = (AudioEncoderLoader, AudioEncoderEncode, SheetSage2AudioToABC)
