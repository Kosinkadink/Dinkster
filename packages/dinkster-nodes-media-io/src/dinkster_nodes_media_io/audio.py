"""Asset-backed audio loading and bounded PyAV audio saving."""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from typing import Any, BinaryIO, cast

import av
import numpy as np
from av.codec import Codec
from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    SAVE_TARGET_TYPE,
    AssetError,
    AssetRef,
    AssetWidget,
    AssetWriter,
    AudioWindowReader,
    ComboWidget,
    InputSpec,
    MountSnapshotWriter,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SaveTargetWidget,
    SourceFilenameSpec,
    TypeExpr,
    append_audio_edit,
    audio_channel_layout,
    audio_from_source,
    effective_audio_facts,
    encode_audio,
)

from .video import AUDIO_TYPE, MAX_DECODED_AUDIO_BYTES

AUDIO = TypeExpr.concrete(AUDIO_TYPE)
AUDIO_ASSET = TypeExpr.asset_of(AUDIO)
AUDIO_ASSET_LIST = TypeExpr.list_of(AUDIO_ASSET)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
COMBO = TypeExpr.concrete(CORE_COMBO)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)

MAX_ENCODED_AUDIO_BYTES = 256 * 1024 * 1024
MAX_EMPTY_AUDIO_SECONDS = 86_400.0
MAX_SAMPLE_RATE = 192_000

# Every container the loader accepts; audio can also come from video files,
# matching the pinned ComfyUI LoadAudio listing of audio and video content.
_LOAD_ACCEPT = (
    "audio/wav",
    "audio/flac",
    "audio/mpeg",
    "audio/ogg",
    "audio/webm",
    "audio/mp4",
    "video/mp4",
    "video/webm",
)

_FORMAT_DETAILS = {
    "flac": ("flac", "flac", ".flac", "audio/flac"),
    "mp3": ("mp3", "libmp3lame", ".mp3", "audio/mpeg"),
    "opus": ("opus", "libopus", ".opus", "audio/ogg"),
}
_REQUIRED_AUDIO_ENCODERS = ("flac", "libmp3lame", "libopus")

# libmp3lame accepts exactly these input sample rates; anything else must be
# rejected loudly rather than silently resampled (declared source behavior:
# the pinned SaveAudioMP3 fails on unsupported rates too).
_MP3_RATES = frozenset({8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000})
# The FLAC stream header carries a 20-bit sample rate field.
_FLAC_MAX_RATE = 655_350
# Opus rate coercion mirrors the pinned ComfyUI AudioSaveHelper: rates above
# 48 kHz clamp down, other unsupported rates round up to the next entry.
_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)

_MP3_QUALITIES = ("V0", "128k", "320k")
_OPUS_QUALITIES = ("64k", "96k", "128k", "192k", "320k")
_BIT_RATES = {"64k": 64_000, "96k": 96_000, "128k": 128_000, "192k": 192_000, "320k": 320_000}

DEFAULT_AUDIO_TARGET = {"mount": "comfy-output", "prefix": "audio/ComfyUI"}


def require_audio_encoders() -> None:
    """Refuse an incomplete PyAV distribution before a workflow starts."""
    missing: list[str] = []
    for codec in _REQUIRED_AUDIO_ENCODERS:
        try:
            Codec(codec, "w")
        except Exception:  # noqa: BLE001 - PyAV uses several codec error classes
            missing.append(codec)
    if missing:
        raise RuntimeError(
            "Dinkster audio I/O requires PyAV encoders missing from this installation: "
            + ", ".join(missing)
            + ". Install the standard av==16.0.1 wheel with its bundled FFmpeg libraries."
        )


require_audio_encoders()


def _mount_writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError(
            "saving requires filesystem mounts, but this process has no "
            "DINKSTER_MOUNTS_SNAPSHOT configured"
        )
    return AssetWriter(MountSnapshotWriter(snapshot))


def _save_input(audio: object) -> tuple[np.ndarray, int]:
    if not isinstance(audio, Mapping):
        raise ValueError("audio must contain waveform and sample_rate")
    value = cast("Mapping[str, object]", audio)
    source = value.get("waveform")
    if not isinstance(source, np.ndarray):
        raise ValueError("audio waveform must be a numpy array")
    waveform = source
    sample_rate = value.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("audio sample_rate must be a positive integer")
    if waveform.ndim != 3 or waveform.shape[0] < 1:
        raise ValueError("audio waveform must have nonempty [B,C,T] layout")
    if waveform.shape[1] < 1:
        raise ValueError("audio waveform must have channels")
    if waveform.shape[2] < 1:
        raise ValueError("audio waveform contains no samples")
    output_bytes = math.prod(waveform.shape) * np.dtype(np.float32).itemsize
    if waveform.nbytes > MAX_DECODED_AUDIO_BYTES or output_bytes > MAX_DECODED_AUDIO_BYTES:
        raise ValueError("audio waveform exceeds the 256 MiB input limit")
    if not bool(np.isfinite(waveform).all()):
        raise ValueError("audio contains non-finite samples")
    return np.asarray(waveform, dtype=np.float32), sample_rate


def _opus_rate(sample_rate: int) -> int:
    if sample_rate > 48_000:
        return 48_000
    if sample_rate in _OPUS_RATES:
        return sample_rate
    for rate in _OPUS_RATES:
        if rate > sample_rate:
            return rate
    return 48_000


def resample_audio(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a [C,T] float32 waveform through libswresample."""
    layout = audio_channel_layout(int(waveform.shape[0]))
    resampler = av.AudioResampler(format="flt", layout=layout, rate=target_rate)
    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(waveform.T.reshape(1, -1)), format="flt", layout=layout
    )
    frame.sample_rate = source_rate
    frame.pts = 0
    pieces = [
        cast(Any, converted).to_ndarray().reshape(-1, waveform.shape[0]).T
        for converted in (*resampler.resample(frame), *resampler.resample(None))
    ]
    if not pieces:
        raise ValueError("audio resampling produced no samples")
    return np.ascontiguousarray(np.concatenate(pieces, axis=1), dtype=np.float32)


def _save_audio(
    audio: object, target: object, format_name: str, quality: str
) -> Mapping[str, object]:
    if not isinstance(audio, Mapping) or "source" not in audio:
        _save_input(audio)
    facts = effective_audio_facts(audio)
    sample_rate = facts["sample_rate"]
    if format_name == "flac" and sample_rate > _FLAC_MAX_RATE:
        raise ValueError(f"FLAC supports sample rates up to {_FLAC_MAX_RATE}, got {sample_rate}")
    if format_name == "mp3" and sample_rate not in _MP3_RATES:
        raise ValueError(
            f"MP3 encoding supports sample rates {sorted(_MP3_RATES)}, got {sample_rate}"
        )
    if format_name == "opus":
        encode_rate = _opus_rate(sample_rate)
        if encode_rate != sample_rate:
            audio = append_audio_edit(audio, {"resample": encode_rate})
            facts = effective_audio_facts(audio)
            sample_rate = encode_rate
    frames = facts["frames"]
    if frames is None:
        raise ValueError("saving unknown-duration audio requires a trim duration")
    if frames < 1 or facts["channels"] < 1 or facts["batch"] < 1:
        raise ValueError("audio waveform contains no samples")
    container_name, codec_name, suffix, media_type = _FORMAT_DETAILS[format_name]
    writer = _mount_writer()
    destination = target or DEFAULT_AUDIO_TARGET
    refs: list[AssetRef] = []
    for batch in range(facts["batch"]):
        with tempfile.TemporaryFile() as output:
            with (
                av.open(output, mode="w", format=container_name) as container,
                AudioWindowReader(audio) as reader,
            ):
                stream = cast(
                    Any, container.add_stream(codec_name, rate=sample_rate, layout=facts["layout"])
                )
                if quality in _BIT_RATES:
                    stream.bit_rate = _BIT_RATES[quality]
                elif quality == "V0":
                    stream.codec_context.qscale = 1
                chunk_size = max(1, min(65_536, 1024 * 1024 // (4 * facts["channels"])))
                for start in range(0, frames, chunk_size):
                    clip = reader.read(start, min(chunk_size, frames - start), batch_index=batch)[
                        "waveform"
                    ][0]
                    frame = av.AudioFrame.from_ndarray(
                        np.ascontiguousarray(clip.T.reshape(1, -1)),
                        format="flt",
                        layout=facts["layout"],
                    )
                    frame.sample_rate = sample_rate
                    frame.pts = start
                    container.mux(stream.encode(frame))
                container.mux(stream.encode(None))
            output.seek(0)
            refs.append(
                writer.save_stream(
                    destination,
                    cast(BinaryIO, output),
                    suffix=suffix,
                    media_type=media_type,
                    limit=os.fstat(output.fileno()).st_size,
                )
            )
    return {"audios": refs}


class LoadAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_audio",
            display_name="Load Audio",
            category="audio",
            inputs=(
                InputSpec(
                    "audio",
                    AUDIO_ASSET,
                    widget=AssetWidget(
                        accept=_LOAD_ACCEPT,
                        kind="media/audio",
                        allow_upload=True,
                    ),
                    source_filename=SourceFilenameSpec("media/audio", "input"),
                ),
                InputSpec(
                    "start_time",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, step=0.01),
                ),
                InputSpec(
                    "duration",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("audio loader", "wav", "flac", "mp3", "sound"),
        )

    @classmethod
    def execute(
        cls, *, audio: AssetRef, start_time: float = 0.0, duration: float = 0.0
    ) -> Mapping[str, object]:
        if (
            not math.isfinite(start_time)
            or not math.isfinite(duration)
            or min(start_time, duration) < 0
        ):
            raise ValueError("audio start_time and duration must be finite and nonnegative")
        value = audio_from_source(audio)
        facts = effective_audio_facts(value)
        if start_time or duration:
            start = round(start_time * facts["sample_rate"])
            count = round(duration * facts["sample_rate"]) if duration else facts["frames"]
            if count is None:
                raise ValueError(
                    "unknown-duration audio requires an explicit duration when trimming"
                )
            value = append_audio_edit(
                value, {"trim": {"start_sample": start, "sample_count": count}}
            )
        return cls.outputs(audio=value)


class EmptyAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_audio",
            display_name="Empty Audio",
            category="audio",
            inputs=(
                InputSpec(
                    "duration",
                    FLOAT,
                    required=False,
                    default=60.0,
                    widget=NumberWidget(min=0.0, max=MAX_EMPTY_AUDIO_SECONDS, step=0.01),
                ),
                InputSpec(
                    "sample_rate",
                    INT,
                    required=False,
                    default=44_100,
                    widget=NumberWidget(min=1, max=MAX_SAMPLE_RATE, step=1),
                ),
                InputSpec(
                    "channels",
                    INT,
                    required=False,
                    default=2,
                    widget=NumberWidget(min=1, step=1),
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            search_terms=("silent audio", "blank audio", "silence"),
        )

    @classmethod
    def execute(
        cls, *, duration: float = 60.0, sample_rate: object = 44_100, channels: object = 2
    ) -> Mapping[str, object]:
        seconds = float(duration)
        if not math.isfinite(seconds) or not 0.0 <= seconds <= MAX_EMPTY_AUDIO_SECONDS:
            raise ValueError(
                f"duration must be finite and in 0..{MAX_EMPTY_AUDIO_SECONDS}, got {duration!r}"
            )
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or not 1 <= sample_rate <= MAX_SAMPLE_RATE
        ):
            raise ValueError(f"sample_rate must be an integer in 1..{MAX_SAMPLE_RATE}")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        samples = int(round(seconds * sample_rate))
        if samples * channels * np.dtype(np.float32).itemsize > MAX_DECODED_AUDIO_BYTES:
            raise ValueError("empty audio exceeds the 256 MiB limit; reduce duration or rate")
        waveform = np.zeros((1, channels, samples), dtype=np.float32)
        return cls.outputs(audio={"waveform": waveform, "sample_rate": int(sample_rate)})


def _save_schema(
    node_type: str,
    display_name: str,
    *,
    quality_options: tuple[str, ...] = (),
    quality_default: str = "",
    search_terms: tuple[str, ...],
) -> NodeSchema:
    quality_inputs = (
        (
            InputSpec(
                "quality",
                COMBO,
                required=False,
                default=quality_default,
                widget=ComboWidget(options=quality_options),
            ),
        )
        if quality_options
        else ()
    )
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="audio",
        inputs=(
            InputSpec("audio", AUDIO, on_absent="fail"),
            InputSpec(
                "target",
                SAVE_TARGET,
                required=False,
                default=dict(DEFAULT_AUDIO_TARGET),
                widget=SaveTargetWidget(),
            ),
            *quality_inputs,
        ),
        outputs=(OutputSpec("audios", AUDIO_ASSET_LIST, preview=True),),
        idempotent=False,
        output_node=True,
        search_terms=search_terms,
    )


class SaveAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _save_schema(
            "dinkster.save_audio",
            "Save Audio (FLAC)",
            search_terms=("audio saver", "flac", "lossless", "export audio"),
        )

    @classmethod
    def execute(cls, *, audio: object, target: object = None) -> Mapping[str, object]:
        return cls.outputs(**_save_audio(audio, target, "flac", ""))


class SaveAudioMP3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _save_schema(
            "dinkster.save_audio_mp3",
            "Save Audio (MP3)",
            quality_options=_MP3_QUALITIES,
            quality_default="V0",
            search_terms=("audio saver", "mp3", "export audio"),
        )

    @classmethod
    def execute(
        cls, *, audio: object, target: object = None, quality: str = "V0"
    ) -> Mapping[str, object]:
        if quality not in _MP3_QUALITIES:
            raise ValueError(f"quality must be one of {_MP3_QUALITIES}, got {quality!r}")
        return cls.outputs(**_save_audio(audio, target, "mp3", quality))


class SaveAudioOpus(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _save_schema(
            "dinkster.save_audio_opus",
            "Save Audio (Opus)",
            quality_options=_OPUS_QUALITIES,
            quality_default="128k",
            search_terms=("audio saver", "opus", "export audio"),
        )

    @classmethod
    def execute(
        cls, *, audio: object, target: object = None, quality: str = "128k"
    ) -> Mapping[str, object]:
        if quality not in _OPUS_QUALITIES:
            raise ValueError(f"quality must be one of {_OPUS_QUALITIES}, got {quality!r}")
        return cls.outputs(**_save_audio(audio, target, "opus", quality))


class PreviewAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preview_audio",
            display_name="Preview Audio",
            category="audio",
            inputs=(InputSpec("audio", AUDIO, on_absent="fail"),),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            output_node=True,
            search_terms=("audio preview", "play audio", "listen"),
        )

    @classmethod
    def execute(cls, *, audio: object) -> Mapping[str, object]:
        encode_audio(audio)
        return cls.outputs(audio=audio)


AUDIO_IO_NODES: tuple[type[Node], ...] = (
    LoadAudio,
    EmptyAudio,
    SaveAudio,
    SaveAudioMP3,
    SaveAudioOpus,
    PreviewAudio,
)

__all__ = [
    "AUDIO_IO_NODES",
    "MAX_ENCODED_AUDIO_BYTES",
    "EmptyAudio",
    "LoadAudio",
    "PreviewAudio",
    "SaveAudio",
    "SaveAudioMP3",
    "SaveAudioOpus",
    "require_audio_encoders",
    "resample_audio",
]
