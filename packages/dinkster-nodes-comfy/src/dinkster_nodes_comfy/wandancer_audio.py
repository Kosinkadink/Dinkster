"""Native WanDancer rhythm, timbre and chroma features."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

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


class WanDancerEncodeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanDancerEncodeAudio",
            category="model/conditioning/wan/dancer",
            inputs=(
                InputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),
                InputSpec(
                    "video_frames",
                    TypeExpr.concrete(CORE_INT),
                    default=149,
                    widget=NumberWidget(min=1, max=16384, step=4),
                ),
                InputSpec(
                    "audio_inject_scale",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                    doc="The scale for the audio features when injected into the video model.",
                ),
            ),
            outputs=(
                OutputSpec(
                    "audio_encoder_output",
                    TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE),
                    display_name="audio_encoder_output",
                ),
                OutputSpec(
                    "fps_string",
                    TypeExpr.concrete(CORE_STRING),
                    display_name="fps_string",
                    doc="The calculated fps based on the audio length and the number of video "
                    "frames. Used in the prompt.",
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, audio: object, video_frames: int, audio_inject_scale: float
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        audio_ops = importlib.import_module("dinkster_inference.audio")
        management = importlib.import_module("dinkster_inference.model_management")
        features = importlib.import_module("dinkster_nodes_comfy._wandancer_audio_features")
        source = materialize_audio(audio)
        waveform: Any = source["waveform"]
        waveform = waveform[0]
        sample_rate = source["sample_rate"]
        base_fps = 30
        hop_length = 512
        model_sr = 22050
        n_fft = 2048

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=False)
        start_bpm = features.quick_tempo_estimate(
            waveform.squeeze().cpu().numpy(), sample_rate, hop_length=hop_length
        )
        resample_sr = base_fps * hop_length
        waveform = audio_ops.resample(waveform, sample_rate, resample_sr)
        waveform_np = waveform.cpu().numpy().squeeze()
        mel_spec = features._compute_mel_spectrogram(
            waveform_np, model_sr, n_fft, hop_length, n_mels=128
        )
        mel_spec_db = features.power_to_db(mel_spec, amin=1e-10, top_db=80.0, ref=1.0)
        envelope = features.compute_onset_envelope(mel_spec_db, n_fft, hop_length)
        mfcc = features.compute_mfcc(mel_spec_db, n_mfcc=20)
        chroma = features.compute_chroma_cens(y=waveform_np, sr=model_sr, hop_length=hop_length).T
        peak_idxs = features.detect_onset_peaks(envelope, sr=model_sr, hop_length=hop_length)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0
        beat_tracking_tempo = features.estimate_tempo_from_onset(
            envelope, sr=model_sr, hop_length=hop_length, start_bpm=start_bpm
        )
        beat_idxs = features.track_beats(
            envelope, beat_tracking_tempo, model_sr, hop_length, tightness=100, trim=True
        )
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0
        audio_feature = np.concatenate(
            [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
            axis=-1,
        )
        audio_feature = (
            torch.from_numpy(audio_feature).unsqueeze(0).to(management.intermediate_device())
        )
        fps = float(base_fps / int(audio_feature.shape[1] / video_frames + 0.5))
        audio_encoder_output = {
            "audio_feature": audio_feature,
            "fps": fps,
            "audio_inject_scale": audio_inject_scale,
        }
        if int(fps + 0.5) != 30:
            fps_string = f" \u5e27\u7387\u662f{fps:.4f}"
        else:
            fps_string = ", \u5e27\u7387\u662f30fps\u3002"
        return cls.outputs(audio_encoder_output=audio_encoder_output, fps_string=fps_string)
