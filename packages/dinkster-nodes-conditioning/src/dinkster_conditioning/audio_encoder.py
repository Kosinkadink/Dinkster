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
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
    with_identity_replacement,
)

_AUDIO_ENCODER = TypeExpr.concrete(AUDIO_ENCODER_TYPE)


_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


class AudioEncoderLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.conditioning.audio_encoder_loader",
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
            ),
            "AudioEncoderLoader",
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
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.conditioning.audio_encoder_encode",
                category="model/conditioning",
                inputs=(InputSpec("audio_encoder", _AUDIO_ENCODER), InputSpec("audio", _AUDIO)),
                outputs=(
                    OutputSpec(
                        "AUDIO_ENCODER_OUTPUT", TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)
                    ),
                ),
            ),
            "AudioEncoderEncode",
        )

    @classmethod
    def execute(cls, *, audio_encoder: Any, audio: object) -> Mapping[str, object]:
        initialize_inference_process()
        source = materialize_audio(audio)
        output = audio_encoder.encode_audio(source["waveform"], source["sample_rate"])
        return cls.outputs(AUDIO_ENCODER_OUTPUT=output)
