"""Native ComfyUI audio output nodes backed by mounted assets."""

from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Mapping
from typing import Any, BinaryIO, cast

from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_COMBO,
    CORE_STRING,
    MEBIBYTE,
    AssetError,
    AssetWriter,
    ComboWidget,
    Deprecation,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    attach_saved_artifact,
    initialize_inference_process,
    materialize_audio,
)

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_STRING = TypeExpr.concrete(CORE_STRING)
_FORMATS = ("flac", "mp3", "opus")
_MP3_QUALITIES = ("V0", "128k", "320k")
_OPUS_QUALITIES = ("64k", "96k", "128k", "192k", "320k")
_BIT_RATES = {
    "64k": 64_000,
    "96k": 96_000,
    "128k": 128_000,
    "192k": 192_000,
    "320k": 320_000,
}
_SUFFIXES = {"flac": ".flac", "mp3": ".mp3", "opus": ".opus"}
_MEDIA_TYPES = {"flac": "audio/flac", "mp3": "audio/mpeg", "opus": "audio/ogg"}
_MAX_ENCODED_BYTES = 256 * MEBIBYTE


def _writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
    return AssetWriter(MountSnapshotWriter(snapshot))


def _opus_rate(sample_rate: int) -> int:
    for rate in (8000, 12000, 16000, 24000, 48000):
        if sample_rate <= rate:
            return rate
    return 48000


def _save_audio(
    audio: object,
    filename_prefix: str,
    mount: str | None,
    format: str,
    quality: str | None,
) -> None:
    if format not in _FORMATS:
        raise ValueError(f"Unsupported audio format: {format!r}")
    initialize_inference_process()
    value = materialize_audio(audio)
    waveform: Any = value["waveform"]
    sample_rate = cast("int", value["sample_rate"])
    if format == "opus":
        encoded_rate = _opus_rate(sample_rate)
        if encoded_rate != sample_rate:
            waveform = importlib.import_module("dinkster_inference.audio").resample(
                waveform, sample_rate, encoded_rate
            )
        sample_rate = encoded_rate

    av: Any = importlib.import_module("av")
    writer = _writer()
    codec = {"flac": "flac", "mp3": "libmp3lame", "opus": "libopus"}[format]
    for batch_index in range(waveform.shape[0]):
        batch_target = writer.output_target(
            filename_prefix.replace("%batch_num%", str(batch_index)), mount=mount
        )
        clip = waveform[batch_index].detach().cpu()
        layout = "mono" if clip.shape[0] == 1 else "stereo"
        with tempfile.TemporaryFile() as output:
            with av.open(output, mode="w", format=format) as container:
                stream = container.add_stream(codec, rate=sample_rate, layout=layout)
                if quality in _BIT_RATES:
                    stream.bit_rate = _BIT_RATES[quality]
                elif quality == "V0":
                    stream.codec_context.qscale = 1
                frame = av.AudioFrame.from_ndarray(
                    clip.movedim(0, 1).reshape(1, -1).float().numpy(),
                    format="flt",
                    layout=layout,
                )
                frame.sample_rate = sample_rate
                frame.pts = 0
                container.mux(stream.encode(frame))
                container.mux(stream.encode(None))
            output.seek(0, os.SEEK_END)
            size = output.tell()
            if size > _MAX_ENCODED_BYTES:
                raise ValueError("encoded audio exceeds the 256 MiB output limit")
            output.seek(0)
            ref = writer.save_stream(
                batch_target,
                cast(BinaryIO, output),
                suffix=_SUFFIXES[format],
                media_type=_MEDIA_TYPES[format],
                limit=size,
            )
            attach_saved_artifact(ref.to_wire())


def _mount_input() -> InputSpec:
    return InputSpec(
        "mount",
        _STRING,
        required=False,
        default=None,
        widget=MountWidget(),
        adjacent_to="filename_prefix",
    )


def _audio_output() -> tuple[OutputSpec, ...]:
    return (OutputSpec("audio", _AUDIO, preview=True),)


class SaveAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveAudio",
            display_name="Save Audio (FLAC) (DEPRECATED)",
            category="audio",
            deprecation=Deprecation("Deprecated in ComfyUI."),
            search_visibility="deprecated",
            search_terms=("export flac",),
            inputs=(
                InputSpec("audio", _AUDIO),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="audio/ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                _mount_input(),
            ),
            outputs=_audio_output(),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls, *, audio: object, filename_prefix: str = "ComfyUI", mount: str | None = None
    ) -> Mapping[str, object]:
        if audio is None:
            raise ValueError(
                "SaveAudio: input audio is None (source video may have no audio track)."
            )
        _save_audio(audio, filename_prefix, mount, "flac", None)
        return cls.outputs(audio=audio)


class SaveAudioMP3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveAudioMP3",
            display_name="Save Audio (MP3) (DEPRECATED)",
            category="audio",
            deprecation=Deprecation("Deprecated in ComfyUI."),
            search_visibility="deprecated",
            search_terms=("export mp3",),
            inputs=(
                InputSpec("audio", _AUDIO),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="audio/ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "quality", _COMBO, default="V0", widget=ComboWidget(options=_MP3_QUALITIES)
                ),
                _mount_input(),
            ),
            outputs=_audio_output(),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: object,
        filename_prefix: str = "ComfyUI",
        quality: str = "128k",
        mount: str | None = None,
    ) -> Mapping[str, object]:
        if audio is None:
            raise ValueError(
                "SaveAudioMP3: input audio is None (source video may have no audio track)."
            )
        if quality not in _MP3_QUALITIES:
            raise ValueError(f"Unsupported MP3 quality: {quality!r}")
        _save_audio(audio, filename_prefix, mount, "mp3", quality)
        return cls.outputs(audio=audio)


class SaveAudioOpus(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveAudioOpus",
            display_name="Save Audio (Opus) (DEPRECATED)",
            category="audio",
            deprecation=Deprecation("Deprecated in ComfyUI."),
            search_visibility="deprecated",
            search_terms=("export opus",),
            inputs=(
                InputSpec("audio", _AUDIO),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="audio/ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "quality", _COMBO, default="128k", widget=ComboWidget(options=_OPUS_QUALITIES)
                ),
                _mount_input(),
            ),
            outputs=_audio_output(),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: object,
        filename_prefix: str = "ComfyUI",
        quality: str = "V3",
        mount: str | None = None,
    ) -> Mapping[str, object]:
        if audio is None:
            raise ValueError(
                "SaveAudioOpus: input audio is None (source video may have no audio track)."
            )
        if quality not in _OPUS_QUALITIES:
            raise ValueError(f"Unsupported Opus quality: {quality!r}")
        _save_audio(audio, filename_prefix, mount, "opus", quality)
        return cls.outputs(audio=audio)


class SaveAudioAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveAudioAdvanced",
            display_name="Save Audio (Advanced)",
            description="Saves the input audio to your ComfyUI output directory.",
            category="audio",
            search_terms=(
                "save audio",
                "export audio",
                "output audio",
                "write audio",
                "flac",
                "mp3",
                "opus",
            ),
            inputs=(
                InputSpec("audio", _AUDIO, doc="The audio to save."),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="audio/ComfyUI",
                    doc=(
                        "The prefix for the file to save. May include formatting tokens such as "
                        "%date:yyyy-MM-dd%."
                    ),
                    widget=StringWidget(multiline=False),
                ),
                _mount_input(),
            ),
            combos=(
                DynamicComboSpec(
                    "format",
                    (
                        DynamicComboOption("flac"),
                        DynamicComboOption(
                            "mp3",
                            (
                                InputSpec(
                                    "quality",
                                    _COMBO,
                                    default="V0",
                                    widget=ComboWidget(options=_MP3_QUALITIES),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "opus",
                            (
                                InputSpec(
                                    "quality",
                                    _COMBO,
                                    default="128k",
                                    widget=ComboWidget(options=_OPUS_QUALITIES),
                                ),
                            ),
                        ),
                    ),
                    doc="The file format in which to save the audio.",
                ),
            ),
            input_order=("audio", "filename_prefix", "format", "mount"),
            outputs=_audio_output(),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: object,
        filename_prefix: str,
        format: str,
        mount: str | None = None,
        quality: str | None = None,
    ) -> Mapping[str, object]:
        if format not in _FORMATS:
            raise ValueError(f"Unsupported audio format: {format!r}")
        if format == "mp3" and quality not in _MP3_QUALITIES:
            raise ValueError(f"Unsupported MP3 quality: {quality!r}")
        if format == "opus" and quality not in _OPUS_QUALITIES:
            raise ValueError(f"Unsupported Opus quality: {quality!r}")
        _save_audio(audio, filename_prefix, mount, format, quality)
        return cls.outputs(audio=audio)


__all__ = ["SaveAudio", "SaveAudioAdvanced", "SaveAudioMP3", "SaveAudioOpus"]
