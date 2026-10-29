"""WanDancer audio feature encoding."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    AUDIO_TYPE,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_audio,
)

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_AUDIO_ENCODER_OUTPUT = TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_STRING = TypeExpr.concrete(CORE_STRING)


class WanDancerEncodeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanDancerEncodeAudio",
            category="model/conditioning/wan/dancer",
            inputs=(
                InputSpec("audio", _AUDIO),
                InputSpec(
                    "video_frames",
                    _INT,
                    default=149,
                    widget=NumberWidget(min=1, max=16384, step=4),
                ),
                InputSpec(
                    "audio_inject_scale",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                    doc="The scale for the audio features when injected into the video model.",
                ),
            ),
            outputs=(
                OutputSpec(
                    "audio_encoder_output",
                    _AUDIO_ENCODER_OUTPUT,
                    display_name="audio_encoder_output",
                ),
                OutputSpec(
                    "fps_string",
                    _STRING,
                    display_name="fps_string",
                    doc=(
                        "The calculated fps based on the audio length and "
                        "the number of video frames. "
                        "Used in the prompt."
                    ),
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, audio: object, video_frames: int, audio_inject_scale: float
    ) -> Mapping[str, object]:
        initialize_inference_process()
        source = cast("Mapping[str, Any]", materialize_audio(audio))
        waveform = source["waveform"][0]
        sample_rate = cast("int", source["sample_rate"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=False)

        features = importlib.import_module("dinkster_nodes_comfy.wandancer_audio_features")
        start_bpm = features.quick_tempo_estimate(waveform.squeeze().cpu().numpy(), sample_rate)
        inference_audio = importlib.import_module("dinkster_inference.audio")
        waveform = inference_audio.resample(waveform, sample_rate, 30 * 512)

        np = importlib.import_module("numpy")
        waveform_np = waveform.cpu().numpy().squeeze()
        mel_spec = features._compute_mel_spectrogram(waveform_np, 22050, 2048, 512, n_mels=128)
        mel_spec_db = features.power_to_db(mel_spec, amin=1e-10, top_db=80.0, ref=1.0)
        envelope = features.compute_onset_envelope(mel_spec_db, 2048, 512)
        mfcc = features.compute_mfcc(mel_spec_db, n_mfcc=20)
        chroma = features.compute_chroma_cens(y=waveform_np, sr=22050, hop_length=512).T
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[features.detect_onset_peaks(envelope, sr=22050, hop_length=512)] = 1.0
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        tempo = features.estimate_tempo_from_onset(envelope, 22050, 512, start_bpm=start_bpm)
        beats = features.track_beats(envelope, tempo, 22050, 512, tightness=100, trim=True)
        beat_onehot[beats] = 1.0
        audio_feature = np.concatenate(
            [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]], axis=-1
        )

        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        audio_feature = (
            torch.from_numpy(audio_feature).unsqueeze(0).to(management.intermediate_device())
        )
        fps = float(30 / int(audio_feature.shape[1] / video_frames + 0.5))
        fps_string = (
            f" \u5e27\u7387\u662f{fps:.4f}"
            if int(fps + 0.5) != 30
            else ", \u5e27\u7387\u662f30fps\u3002"
        )
        return cls.outputs(
            audio_encoder_output={
                "audio_feature": audio_feature,
                "fps": fps,
                "audio_inject_scale": audio_inject_scale,
            },
            fps_string=fps_string,
        )
