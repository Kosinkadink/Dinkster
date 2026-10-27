"""Native audio file input and waveform preview."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    AUDIO_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    SourceFilenameSpec,
    TypeExpr,
)

_ASSET = TypeExpr.concrete(ASSET_TYPE)
_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


class LoadAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadAudio",
            display_name="Load Audio",
            category="audio",
            search_terms=("import audio", "open audio", "audio file"),
            inputs=(
                InputSpec(
                    "audio",
                    _ASSET,
                    widget=AssetWidget(
                        kind="media/audio", accept=("audio/*", "video/*"), allow_upload=True
                    ),
                    source_filename=SourceFilenameSpec("media/audio", "input"),
                ),
            ),
            outputs=(OutputSpec("AUDIO", _AUDIO, preview=True),),
        )

    @classmethod
    def execute(cls, *, audio: object) -> Mapping[str, object]:
        if not isinstance(audio, (AssetRef, LocalAssetRef)):
            raise TypeError("audio must be an AssetRef or LocalAssetRef")
        av: Any = importlib.import_module("av")
        np: Any = importlib.import_module("numpy")
        with audio.open() as source, av.open(source) as container:
            if not container.streams.audio:
                raise ValueError("No audio stream found in the file.")
            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate
            channels = stream.channels
            frames: list[Any] = []
            for frame in container.decode(streams=stream.index):
                samples = frame.to_ndarray()
                if samples.shape[0] != channels:
                    samples = samples.reshape(-1, channels).T
                frames.append(samples)
            if not frames:
                raise ValueError("No audio frames decoded.")
            waveform = np.concatenate(frames, axis=1)
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / (2**15)
            elif waveform.dtype == np.int32:
                waveform = waveform.astype(np.float32) / (2**31)
            elif not np.issubdtype(waveform.dtype, np.floating):
                raise ValueError(f"Unsupported wav dtype: {waveform.dtype}")
        return cls.outputs(AUDIO={"waveform": waveform[None], "sample_rate": sample_rate})


class PreviewAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PreviewAudio",
            display_name="Preview Audio",
            category="audio",
            description="Preview the audio without saving it to the ComfyUI output directory.",
            search_terms=("preview", "preview audio", "play audio"),
            inputs=(InputSpec("audio", _AUDIO),),
            outputs=(OutputSpec("audio", _AUDIO, preview=True),),
            output_node=True,
        )

    @classmethod
    def execute(cls, *, audio: object) -> Mapping[str, object]:
        if audio is None:
            raise ValueError(
                "PreviewAudio: input audio is None (source video may have no audio track)."
            )
        return cls.outputs(audio=audio)
