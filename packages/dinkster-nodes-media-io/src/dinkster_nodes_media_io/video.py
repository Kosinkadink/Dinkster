"""Asset-backed video loading and bounded PyAV video saving."""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, BinaryIO, Protocol, cast

import av
import numpy as np
from av.codec import Codec
from dinkster_api.v1 import (
    ABSENT,
    ASSET_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    DITHERS,
    ENCODED_MEDIA_LIMIT_BYTES,
    FRAME_FORMATS,
    MEBIBYTE,
    SAVE_TARGET_TYPE,
    VIDEO_AUDIO_WORKING_SET_LIMIT_BYTES,
    VIDEO_FRAME_WORKING_SET_LIMIT_BYTES,
    AssetError,
    AssetRef,
    AssetWidget,
    AssetWriter,
    ComboWidget,
    InputSpec,
    MappingSource,
    MountSnapshotWriter,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    ReplacementCase,
    ReplacementLink,
    ReplacementMigration,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    SaveTargetWidget,
    SourceFilenameSpec,
    StringWidget,
    TypeExpr,
    read_video_metadata,
    report_value_diagnostic,
    save_video_frames,
    save_video_stream,
    video_from_source,
)

IMAGE_TYPE = "dinkster.image"
COMPAT_IMAGE_TYPE = "comfy.IMAGE"
AUDIO_TYPE = "comfy.AUDIO"
VIDEO_TYPE = "comfy.VIDEO"

IMAGE = TypeExpr.concrete(IMAGE_TYPE)
AUDIO = TypeExpr.concrete(AUDIO_TYPE)
VIDEO_ASSET = TypeExpr.asset_of(TypeExpr.concrete(VIDEO_TYPE))
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
COMBO = TypeExpr.concrete(CORE_COMBO)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)

MAX_DECODED_FRAME_BYTES = VIDEO_FRAME_WORKING_SET_LIMIT_BYTES
MAX_DECODED_AUDIO_BYTES = VIDEO_AUDIO_WORKING_SET_LIMIT_BYTES
MAX_ENCODED_VIDEO_BYTES = ENCODED_MEDIA_LIMIT_BYTES

_REQUIRED_ENCODERS = ("libx264", "libvpx-vp9", "libsvtav1", "aac", "libopus")


def require_video_encoders() -> None:
    """Refuse an incomplete PyAV distribution before a workflow starts."""
    missing: list[str] = []
    for codec in _REQUIRED_ENCODERS:
        try:
            Codec(codec, "w")
        except Exception:  # noqa: BLE001 - PyAV uses several codec error classes
            missing.append(codec)
    if missing:
        raise RuntimeError(
            "Dinkster video I/O requires PyAV encoders missing from this installation: "
            + ", ".join(missing)
            + ". Install the standard av==16.0.1 wheel with its bundled FFmpeg libraries."
        )


require_video_encoders()


class _BoundedSpool(tempfile.SpooledTemporaryFile[bytes]):
    def __init__(self, limit: int) -> None:
        super().__init__(max_size=512 * 1024, mode="w+b")
        self._limit = limit

    def write(self, s: Any) -> int:
        position = self.tell()
        self.seek(0, os.SEEK_END)
        current_size = self.tell()
        self.seek(position)
        if max(position + len(s), current_size) > self._limit:
            raise ValueError(f"encoded video exceeds the {self._limit}-byte output limit")
        return super().write(s)


class VideoByteSource(Protocol):
    """A source whose container bytes can be reopened for each read pass."""

    def open(self) -> BinaryIO: ...


def _report_saved_format(record: Mapping[str, object], *, output_id: str = "asset") -> None:
    report_value_diagnostic(
        str(record["code"]),
        {"outputId": output_id, **{key: value for key, value in record.items() if key != "code"}},
    )


def _mount_writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError(
            "saving requires filesystem mounts, but this process has no "
            "DINKSTER_MOUNTS_SNAPSHOT configured"
        )
    return AssetWriter(MountSnapshotWriter(snapshot))


def positive_finite(value: float, name: str, *, minimum: float, maximum: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in {minimum}..{maximum}, got {value!r}")
    return result


def _save_video_asset(
    video: object,
    target: object,
    output_id: str,
    *,
    container: str = "auto",
    codec: str = "auto",
    crf: int | None = None,
    metadata: Mapping[str, object] | None = None,
    profile: str = "auto",
    audio_layout: str = "preserve",
    trim_to_audio: bool = False,
) -> AssetRef:
    with _BoundedSpool(MAX_ENCODED_VIDEO_BYTES) as encoded:
        suffix, media_type = save_video_stream(
            video,
            cast(BinaryIO, encoded),
            container=container,
            codec=codec,
            crf=crf,
            metadata=metadata,
            profile=profile,
            audio_layout=audio_layout,
            trim_to_audio=trim_to_audio,
            on_diagnostic=lambda record: _report_saved_format(record, output_id=output_id),
        )
        encoded.seek(0)
        return _mount_writer().save_stream(
            target or {"mount": "comfy-output", "prefix": "video/ComfyUI"},
            cast(BinaryIO, encoded),
            suffix=suffix,
            media_type=media_type,
            limit=MAX_ENCODED_VIDEO_BYTES,
        )


def _frame_timestamp(frame: Any, stream: Any) -> Fraction:
    pts = frame.pts
    time_base = frame.time_base or stream.time_base
    if pts is None or time_base is None:
        raise ValueError("video frame has no usable timestamp")
    return Fraction(pts) * Fraction(time_base)


def _frame_time(frame: Any, stream: Any) -> float:
    timestamp = float(_frame_timestamp(frame, stream))
    if not math.isfinite(timestamp):
        raise ValueError("media frame has a non-finite timestamp")
    return timestamp


def _has_alpha(frame: Any) -> bool:
    components = getattr(frame.format, "components", ())
    return any(bool(getattr(component, "is_alpha", False)) for component in components)


def _validated_target_dimension(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 16384:
        raise ValueError(f"{name} must be zero or an integer in 1..16384")
    return value


def _resize_dimensions(
    width: int, height: int, custom_width: int, custom_height: int
) -> tuple[int, int]:
    if custom_width and custom_height:
        return custom_width, custom_height
    if custom_width:
        return custom_width, max(1, round(height * custom_width / width))
    if custom_height:
        return max(1, round(width * custom_height / height)), custom_height
    return width, height


def _image_array(frame: Any, width: int, height: int, alpha: bool) -> np.ndarray:
    channels = 4 if alpha else 3
    source_width = int(frame.width)
    source_height = int(frame.height)
    if source_width < 1 or source_height < 1:
        raise ValueError("decoded video frame has invalid dimensions")
    depth = max(component.bits for component in frame.format.components)
    dtype = np.dtype(np.uint16 if depth > 8 else np.uint8)
    if source_width * source_height * channels * dtype.itemsize > MAX_DECODED_FRAME_BYTES:
        raise ValueError("decoded source frame exceeds the 512 MiB limit")
    if width * height * channels * dtype.itemsize > MAX_DECODED_FRAME_BYTES:
        raise ValueError("resized video frame exceeds the 512 MiB limit")
    format_name = (
        ("rgba64le" if alpha else "rgb48le") if depth > 8 else ("rgba" if alpha else "rgb24")
    )
    converted = frame.reformat(width=width, height=height, format=format_name)
    array = converted.to_ndarray()
    expected = (height, width, channels)
    if array.dtype != dtype or array.shape != expected:
        raise ValueError(
            f"decoded video frame has dtype/shape {array.dtype}/{array.shape}, "
            f"expected {dtype}/{expected}"
        )
    return np.ascontiguousarray(array)


def _video_frames(container: Any, stream: Any) -> Any:
    if stream.codec_context.name != "vp9" or stream.metadata.get("alpha_mode") != "1":
        yield from container.decode(stream)
        return
    decoder = cast(Any, av.CodecContext.create("libvpx-vp9", "r"))
    decoder.extradata = stream.codec_context.extradata
    for packet in container.demux(stream):
        if packet.dts is not None:
            yield from decoder.decode(packet)
    yield from decoder.decode(None)


def decode_video_frames(
    video: VideoByteSource,
    *,
    force_rate: float,
    custom_width: object,
    custom_height: object,
    frame_load_cap: object,
    start_time: float,
    select_every_nth: object,
) -> tuple[np.ndarray, float, float]:
    target_width = _validated_target_dimension("custom_width", custom_width)
    target_height = _validated_target_dimension("custom_height", custom_height)
    if (
        isinstance(frame_load_cap, bool)
        or not isinstance(frame_load_cap, int)
        or frame_load_cap < 0
    ):
        raise ValueError("frame_load_cap must be a nonnegative integer")
    if (
        isinstance(select_every_nth, bool)
        or not isinstance(select_every_nth, int)
        or select_every_nth < 1
    ):
        raise ValueError("select_every_nth must be an integer of at least one")
    start = float(start_time)
    if not math.isfinite(start) or start < 0:
        raise ValueError("start_time must be finite and nonnegative")
    requested_rate = float(force_rate)
    if requested_rate != 0:
        positive_finite(requested_rate, "force_rate", minimum=0.01, maximum=1000.0)

    base_rate = 0.0
    selected: list[np.ndarray] = []
    with video.open() as handle, av.open(handle, mode="r") as container:
        stream = cast(Any, next(iter(container.streams.video), None))
        if stream is None:
            raise ValueError("video asset contains no video stream")
        source_rate_fraction = (
            Fraction(stream.average_rate) if stream.average_rate is not None else Fraction(0)
        )
        source_rate = float(source_rate_fraction)
        if not math.isfinite(source_rate) or source_rate_fraction <= 0:
            raise ValueError("video stream has no usable average frame rate")
        base_rate = requested_rate or source_rate
        base_rate_fraction = Fraction(base_rate).limit_denominator(1_000_000)
        start_fraction = Fraction(start).limit_denominator(1_000_000_000)
        width, height = _resize_dimensions(
            int(stream.codec_context.width),
            int(stream.codec_context.height),
            target_width,
            target_height,
        )
        if width <= 0 or height <= 0:
            raise ValueError("video stream has invalid dimensions")
        if start:
            container.seek(
                int(start / float(stream.time_base)),
                stream=stream,
                backward=True,
                any_frame=False,
            )
        selected_bytes = 0
        sample_index = 0
        previous: tuple[Fraction, Any] | None = None
        last_timestamp: Fraction | None = None
        alpha: bool | None = None

        def sample_time(index: int) -> Fraction:
            return start_fraction + Fraction(index, 1) / base_rate_fraction

        def keep(frame: Any) -> bool:
            nonlocal alpha, sample_index, selected_bytes
            retain = sample_index % select_every_nth == 0
            sample_index += 1
            if not retain:
                return False
            frame_alpha = _has_alpha(frame)
            if alpha is None:
                alpha = frame_alpha
            elif alpha != frame_alpha:
                raise ValueError("video changes alpha layout between frames")
            array = _image_array(frame, width, height, bool(alpha))
            if selected_bytes + array.nbytes > MAX_DECODED_FRAME_BYTES:
                raise ValueError(
                    "decoded video frames exceed the 512 MiB limit; use resize, start, or frame cap"
                )
            selected_bytes += int(array.nbytes)
            selected.append(array)
            return bool(frame_load_cap and len(selected) >= frame_load_cap)

        reached_cap = False
        for frame in _video_frames(container, stream):
            timestamp = _frame_timestamp(frame, stream)
            if last_timestamp is not None and timestamp <= last_timestamp:
                raise ValueError("video frame timestamps must be strictly increasing")
            last_timestamp = timestamp
            if timestamp < start_fraction:
                continue
            if previous is None:
                previous = (timestamp, frame)
                continue
            current = (timestamp, frame)
            while sample_time(sample_index) <= timestamp:
                tick = sample_time(sample_index)
                nearest = previous[1] if abs(previous[0] - tick) <= abs(timestamp - tick) else frame
                if keep(nearest):
                    reached_cap = True
                    break
            previous = current
            if reached_cap:
                break

        if previous is not None and not reached_cap:
            last_frame = previous[1]
            frame_duration = last_frame.duration
            frame_time_base = last_frame.time_base or stream.time_base
            source_end = (
                previous[0] + Fraction(frame_duration) * Fraction(frame_time_base)
                if frame_duration is not None and frame_duration > 0 and frame_time_base is not None
                else previous[0] + Fraction(1, 1) / source_rate_fraction
            )
            while sample_time(sample_index) < source_end:
                if keep(previous[1]):
                    break
        if not selected:
            raise ValueError("video selection contains no frames")
    effective_rate = base_rate / select_every_nth
    duration = len(selected) / effective_rate
    return np.stack(selected), effective_rate, duration


def audio_frame_array(frame: Any) -> np.ndarray:
    source = np.asarray(frame.to_ndarray())
    channels = len(frame.layout.channels)
    if source.ndim != 2 or channels < 1:
        raise ValueError(f"decoded audio frame has invalid shape {source.shape}")
    if frame.format.is_planar:
        if source.shape[0] != channels:
            raise ValueError("planar audio frame does not match its channel layout")
        planar = source
    else:
        if source.size % channels:
            raise ValueError("packed audio frame does not match its channel layout")
        planar = source.reshape(-1, channels).T
    if np.issubdtype(planar.dtype, np.signedinteger):
        converted = planar.astype(np.float32) / float(-np.iinfo(planar.dtype).min)
    elif np.issubdtype(planar.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(planar.dtype).max + 1) / 2.0
        converted = (planar.astype(np.float32) - midpoint) / midpoint
    else:
        converted = planar.astype(np.float32)
    return np.ascontiguousarray(converted)


def decode_video_audio(video: VideoByteSource, start_time: float, duration: float) -> object:
    last_error: Exception | None = None
    audio_indexes: list[int] = []
    with video.open() as probe_handle, av.open(probe_handle, mode="r") as probe:
        audio_indexes = [int(stream.index) for stream in probe.streams.audio]
    for stream_index in reversed(audio_indexes):
        try:
            pieces: list[np.ndarray] = []
            sample_rate: int | None = None
            channels: int | None = None
            total_bytes = 0
            interval_end = start_time + duration
            with video.open() as handle, av.open(handle, mode="r") as container:
                stream = cast(Any, container.streams[stream_index])
                if start_time:
                    container.seek(
                        int(start_time / float(stream.time_base)),
                        stream=stream,
                        backward=True,
                        any_frame=False,
                    )
                for decoded in container.decode(stream):
                    frame = cast(Any, decoded)
                    if frame.sample_rate is None or frame.sample_rate <= 0:
                        raise ValueError("audio frame has no usable sample rate")
                    rate = int(frame.sample_rate)
                    if sample_rate is None:
                        sample_rate = rate
                    elif sample_rate != rate:
                        raise ValueError("audio sample rate changes within the selected stream")
                    timestamp = _frame_time(frame, stream)
                    array = audio_frame_array(frame)
                    if channels is None:
                        channels = int(array.shape[0])
                    elif channels != int(array.shape[0]):
                        raise ValueError("audio channel count changes within the selected stream")
                    frame_end = timestamp + array.shape[1] / rate
                    if frame_end <= start_time:
                        continue
                    if timestamp >= interval_end:
                        break
                    first = max(0, math.ceil((start_time - timestamp) * rate))
                    last = min(array.shape[1], math.ceil((interval_end - timestamp) * rate))
                    if first < last:
                        piece = np.ascontiguousarray(array[:, first:last])
                        total_bytes += int(piece.nbytes)
                        if total_bytes > MAX_DECODED_AUDIO_BYTES:
                            raise ValueError("decoded audio exceeds the 256 MiB limit")
                        pieces.append(piece)
            if not pieces or sample_rate is None:
                continue
            waveform = np.concatenate(pieces, axis=1)[None, ...]
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as exc:  # noqa: BLE001 - try earlier audio streams as specified
            last_error = exc
    if audio_indexes and last_error is not None:
        raise ValueError(f"no audio stream could be decoded: {last_error}") from last_error
    return ABSENT


class LoadVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_video",
            display_name="Load Video",
            category="video",
            inputs=(
                InputSpec(
                    "video",
                    VIDEO_ASSET,
                    widget=AssetWidget(
                        accept=("video/mp4", "video/webm"),
                        kind="media/video",
                        allow_upload=True,
                    ),
                    source_filename=SourceFilenameSpec("media/video", "input"),
                ),
                InputSpec(
                    "force_rate",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1000.0, step=0.01),
                ),
                InputSpec(
                    "custom_width",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16384, step=1),
                ),
                InputSpec(
                    "custom_height",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=16384, step=1),
                ),
                InputSpec(
                    "frame_load_cap",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, step=1),
                ),
                InputSpec(
                    "start_time",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, step=0.01),
                ),
                InputSpec(
                    "select_every_nth",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, step=1),
                ),
            ),
            outputs=(
                OutputSpec("images", IMAGE, preview=True),
                OutputSpec("frame_count", INT),
                OutputSpec("audio", AUDIO, optional=True, preview=True),
                OutputSpec("fps", FLOAT),
                OutputSpec("duration", FLOAT),
            ),
            search_terms=("video loader", "frames", "audio", "VHS"),
        )

    @classmethod
    def execute(
        cls,
        *,
        video: AssetRef,
        force_rate: float = 0.0,
        custom_width: object = 0,
        custom_height: object = 0,
        frame_load_cap: object = 0,
        start_time: float = 0.0,
        select_every_nth: object = 1,
    ) -> Mapping[str, object]:
        images, fps, duration = decode_video_frames(
            video,
            force_rate=force_rate,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=frame_load_cap,
            start_time=start_time,
            select_every_nth=select_every_nth,
        )
        audio = decode_video_audio(video, float(start_time), duration)
        return cls.outputs(
            images=images,
            frame_count=int(images.shape[0]),
            audio=audio,
            fps=fps,
            duration=duration,
        )


class LoadVideoValue(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_video_value",
            display_name="Load Video Value",
            category="video",
            inputs=(
                InputSpec(
                    "video",
                    VIDEO_ASSET,
                    widget=AssetWidget(
                        accept=("video/mp4", "video/webm"),
                        kind="media/video",
                        allow_upload=True,
                    ),
                    source_filename=SourceFilenameSpec("media/video", "input"),
                ),
            ),
            outputs=(OutputSpec("video", TypeExpr.concrete(VIDEO_TYPE), preview=True),),
            search_terms=("video object", "load video value", "lazy video"),
        )

    @classmethod
    def execute(cls, *, video: AssetRef) -> Mapping[str, object]:
        if video.size > MAX_ENCODED_VIDEO_BYTES:
            raise ValueError("video asset exceeds the 1 GiB input limit")
        return cls.outputs(video=video_from_source(video))


class SaveVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        def migration_case(nested: bool) -> ReplacementCase:
            prefix = "format." if nested else ""
            return ReplacementCase.build(
                "dinkster.save_video",
                when=(
                    ReplacementPredicate.any_of(
                        *(
                            predicate("format." + key)
                            for key in ("crf", "bit_depth")
                            for predicate in (
                                ReplacementPredicate.value_present,
                                ReplacementPredicate.input_connected,
                            )
                        )
                    )
                    if nested
                    else None
                ),
                nodes={
                    "assemble": ReplacementNode(
                        "dinkster.video.assemble", values=(("color_space", "sRGB"),)
                    )
                },
                inputs={
                    **{
                        "assemble:" + key: MappingSource.copy(key)
                        for key in ("images", "fps", "audio")
                    },
                    "assemble:bit_depth": MappingSource.copy(prefix + "bit_depth"),
                    "target": MappingSource.copy("target"),
                    "format": MappingSource.copy("format"),
                    "crf": MappingSource.copy(prefix + "crf"),
                },
                links=(ReplacementLink("assemble:video", "video"),),
                outputs={"asset": "video"},
            )

        migration = ReplacementRule(
            from_type="dinkster.save_video",
            cases=(migration_case(True), migration_case(False)),
            migration=ReplacementMigration(
                ("images", "fps", "audio", "bit_depth", "format.crf", "format.bit_depth")
            ),
        )
        return NodeSchema(
            node_type="dinkster.save_video",
            aliases=("SaveVideo",),
            version=4,
            display_name="Save Video",
            category="video",
            inputs=(
                InputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default={"mount": "comfy-output", "prefix": "video/ComfyUI"},
                    widget=SaveTargetWidget(),
                ),
                InputSpec(
                    "container",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=("auto", "mp4", "mkv", "mov", "webm", "avi", "gif")),
                ),
                InputSpec(
                    "codec",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(
                        options=("auto", "h264", "hevc", "av1", "vp9", "vp8", "prores", "ffv1")
                    ),
                ),
                InputSpec(
                    "profile",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(options=("auto", "lt", "standard", "hq", "4444", "4444xq")),
                    doc="ProRes profile; auto preserves alpha using 4444.",
                ),
                InputSpec(
                    "audio_layout",
                    COMBO,
                    required=False,
                    default="preserve",
                    widget=ComboWidget(options=("preserve", "mono", "stereo")),
                    doc="Preserve declared channels, or explicitly downmix.",
                ),
                InputSpec(
                    "trim_to_audio",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    required=False,
                    default=False,
                    doc="Stop at shortest audio; otherwise pad short audio with silence.",
                ),
                InputSpec(
                    "crf",
                    INT,
                    required=False,
                    widget=NumberWidget(min=0, max=63, step=1),
                    doc="Omit for stream copy where possible; specifying CRF requests encoding.",
                ),
                InputSpec(
                    "metadata",
                    TypeExpr.concrete(CORE_STRING),
                    required=False,
                    default="{}",
                    widget=StringWidget(multiline=True),
                    doc="JSON object of container metadata tags.",
                ),
                InputSpec(
                    "format",
                    COMBO,
                    required=False,
                    advanced=True,
                    widget=ComboWidget(options=("mp4_h264", "webm_vp9", "webm_av1")),
                    doc="Legacy format; cannot combine with container or codec overrides.",
                ),
            ),
            outputs=(
                OutputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),
                OutputSpec("asset", VIDEO_ASSET, preview=True),
            ),
            idempotent=False,
            output_node=True,
            search_terms=("video encoder", "mp4", "webm", "H.264", "VP9", "AV1"),
            replacements=(migration,),
        )

    @classmethod
    def execute(
        cls,
        *,
        video: object,
        target: object = None,
        container: str = "auto",
        codec: str = "auto",
        crf: int | None = None,
        metadata: object = "{}",
        format: str | None = None,
        profile: str = "auto",
        audio_layout: str = "preserve",
        trim_to_audio: bool = False,
    ) -> Mapping[str, object]:
        if format is not None:
            if container != "auto" or codec != "auto":
                raise ValueError("legacy format cannot be combined with container or codec")
            if format in ("mp4_h264", "webm_vp9", "webm_av1"):
                container, codec = format.split("_", 1)
            else:
                codec = format
        if not isinstance(metadata, str) or len(metadata.encode("utf-8")) > MEBIBYTE:
            raise ValueError("video metadata must be a JSON object under 1 MiB")
        tags = json.loads(metadata)
        if not isinstance(tags, dict):
            raise ValueError("video metadata must be a JSON object")
        ref = _save_video_asset(
            video,
            target,
            "asset",
            container=container,
            codec=codec,
            crf=crf,
            metadata=tags,
            profile=profile,
            audio_layout=audio_layout,
            trim_to_audio=trim_to_audio,
        )
        return cls.outputs(video=video, asset=ref)


class SaveVideoValue(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_video_value",
            display_name="Save Video Value",
            category="video",
            inputs=(
                InputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default={"mount": "comfy-output", "prefix": "video/ComfyUI"},
                    widget=SaveTargetWidget(),
                ),
            ),
            outputs=(OutputSpec("video", VIDEO_ASSET, preview=True),),
            idempotent=False,
            output_node=True,
            search_terms=("save video object", "export video value"),
        )

    @classmethod
    def execute(cls, *, video: object, target: object = None) -> Mapping[str, object]:
        return cls.outputs(video=_save_video_asset(video, target, "video"))


class SaveVideoFrames(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_video_frames",
            display_name="Save Animation or PNG Sequence",
            category="video",
            inputs=(
                InputSpec("video", TypeExpr.concrete(VIDEO_TYPE)),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default={"mount": "comfy-output", "prefix": "video/ComfyUI"},
                    widget=SaveTargetWidget(),
                ),
                InputSpec(
                    "format",
                    COMBO,
                    required=False,
                    default="gif_pillow",
                    widget=ComboWidget(options=FRAME_FORMATS),
                ),
                InputSpec(
                    "loop",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=65535, step=1),
                ),
                InputSpec(
                    "dither",
                    COMBO,
                    required=False,
                    default="sierra2_4a",
                    widget=ComboWidget(options=DITHERS),
                ),
                InputSpec(
                    "lossless", TypeExpr.concrete(CORE_BOOLEAN), required=False, default=True
                ),
                InputSpec(
                    "quality",
                    INT,
                    required=False,
                    default=80,
                    widget=NumberWidget(min=0, max=100, step=1),
                ),
                InputSpec(
                    "metadata",
                    TypeExpr.concrete(CORE_STRING),
                    required=False,
                    default="{}",
                    widget=StringWidget(multiline=True),
                ),
            ),
            outputs=(OutputSpec("asset", TypeExpr.concrete(ASSET_TYPE)),),
            idempotent=False,
            output_node=True,
            search_terms=("GIF", "WebP", "PNG", "sequence", "animation"),
        )

    @classmethod
    def execute(
        cls,
        *,
        video: object,
        target: object = None,
        format: str = "gif_pillow",
        loop: int = 0,
        dither: str = "sierra2_4a",
        lossless: bool = True,
        quality: int = 80,
        metadata: object = "{}",
    ) -> Mapping[str, object]:
        if not isinstance(metadata, str) or len(metadata) > MEBIBYTE:
            raise ValueError("video metadata must be a JSON object under 1 MiB")
        tags = json.loads(metadata)
        if not isinstance(tags, dict):
            raise ValueError("video metadata must be a JSON object")
        with _BoundedSpool(MAX_ENCODED_VIDEO_BYTES) as encoded:
            suffix, mime = save_video_frames(
                video,
                cast(BinaryIO, encoded),
                format=format,
                loop=loop,
                dither=dither,
                lossless=lossless,
                quality=quality,
                metadata=tags,
                on_diagnostic=_report_saved_format,
            )
            encoded.seek(0)
            ref = _mount_writer().save_stream(
                target or {"mount": "comfy-output", "prefix": "video/ComfyUI"},
                cast(BinaryIO, encoded),
                suffix=suffix,
                media_type=mime,
                limit=MAX_ENCODED_VIDEO_BYTES,
            )
        return cls.outputs(asset=ref)


class ReadVideoMetadata(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.read_video_metadata",
            display_name="Read Video Metadata",
            category="video",
            inputs=(
                InputSpec("asset", TypeExpr.wildcard(), doc="A typed or untyped media asset."),
            ),
            outputs=(OutputSpec("metadata", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, asset: object) -> Mapping[str, object]:
        if not isinstance(asset, AssetRef):
            raise ValueError("video metadata requires an asset reference, not a filesystem path")
        return cls.outputs(metadata=json.dumps(read_video_metadata(asset)))


VIDEO_NODES = (
    LoadVideo,
    LoadVideoValue,
    SaveVideo,
    SaveVideoValue,
    SaveVideoFrames,
    ReadVideoMetadata,
)

__all__ = [
    "AUDIO_TYPE",
    "COMPAT_IMAGE_TYPE",
    "IMAGE_TYPE",
    "MAX_DECODED_AUDIO_BYTES",
    "MAX_DECODED_FRAME_BYTES",
    "MAX_ENCODED_VIDEO_BYTES",
    "VIDEO_NODES",
    "VIDEO_TYPE",
    "LoadVideo",
    "LoadVideoValue",
    "SaveVideo",
    "SaveVideoFrames",
    "ReadVideoMetadata",
    "SaveVideoValue",
    "VideoByteSource",
    "decode_video_audio",
    "decode_video_frames",
    "positive_finite",
    "require_video_encoders",
]
