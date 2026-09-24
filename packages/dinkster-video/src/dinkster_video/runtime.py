"""One ordered edit plan for explicit materialization and bounded saving."""

from __future__ import annotations
from dinkster_values import MEBIBYTE

import heapq
import io
import math
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from dinkster_values import (
    AudioWindowReader,
    annotate_image,
    append_audio_edit,
    audio_from_source,
    coerce_video,
    copy_media_semantics,
    effective_audio_facts,
    effective_video_facts,
    image_array_meta,
    media_semantics,
    open_video_source,
    video_source,
    VIDEO_DECODE_WORKING_SET_LIMIT_BYTES,
)
from dinkster_values.audio_codec import coerce_audio
from dinkster_values.storage import image_input
from dinkster_values.video_edits import (
    crop_rectangle,
    mapping,
    scale_geometry,
    seconds,
    trim_window,
)
from dinkster_values.video_probe import color_space_label

from .formats import encoded_diagnostics

if TYPE_CHECKING:
    import numpy as np

_COLORS = {"sRGB": (1, 13, 1, 1), "HDR": (9, 18, 9, 1), "HDR PQ": (9, 16, 9, 1)}
_COLOR_KEYS = ("primaries", "transfer", "matrix", "range")
_MATRIX_FOR_PRIMARIES = {values[0]: values[2] for values in _COLORS.values()}
_ENCODERS = {
    "h264": "libx264",
    "hevc": "libx265",
    "av1": "libsvtav1",
    "vp9": "libvpx-vp9",
    "vp8": "libvpx",
}
_CONTAINERS = {"mkv": "matroska"}


def assemble_video(
    images: object,
    *,
    fps: object = 24,
    audio: object = None,
    bit_depth: str = "auto",
    color_space: str | None = None,
) -> dict[str, object]:
    """Retain component storage; declaring a VIDEO never invokes an encoder."""
    import numpy as np

    if not isinstance(images, np.ndarray) or images.ndim != 4:
        raise ValueError("images must be a [B,H,W,3|4] array")
    images = cast(np.ndarray, images)
    rate = seconds(fps, "fps")
    if rate <= 0:
        raise ValueError("fps must be positive")
    if (color_space is not None and color_space not in _COLORS) or bit_depth not in (
        "auto",
        "8",
        "10",
    ):
        raise ValueError("invalid bit_depth or color_space")
    carried = mapping(image_array_meta(images)["color"], "image color")
    if color_space is None:
        matrix = carried.get("matrix", 2)
        if matrix in (0, 2):
            matrix = _MATRIX_FOR_PRIMARIES.get(cast(int, carried["primaries"]), 2)
        color = {
            "primaries": carried["primaries"],
            "transfer": carried["transfer"],
            "matrix": matrix,
            "range": 1,
        }
        color_space = color_space_label(cast(int, color["transfer"]))
    else:
        color = dict(zip(_COLOR_KEYS, _COLORS[color_space], strict=True))
    depth = (
        int(cast(int, carried.get("bit_depth", 10 if color_space in ("HDR", "HDR PQ") else 8)))
        if bit_depth == "auto"
        else int(bit_depth)
    )
    return coerce_video(
        {
            "components": {
                "images": images,
                "audio": audio,
                "fps": rate,
                "bit_depth": depth,
                "color_space": color_space,
                "color": color,
            },
            "edits": [],
        }
    )


@dataclass(frozen=True)
class _Segment:
    value: Mapping[str, object]
    start: Fraction
    duration: Fraction | None
    spatial: tuple[Mapping[str, object], ...] = ()


def _plan(value: Mapping[str, object]) -> list[_Segment]:
    probe = mapping(value["probe"], "probe")
    duration = probe["duration"]
    segments = [_Segment(value, Fraction(0), cast("Fraction | None", duration))]
    current: dict[str, object] = {**value, "edits": []}
    for raw in cast("list[object]", value["edits"]):
        edit = mapping(raw, "edit")
        if "concat" in edit:
            for child in cast("list[object]", edit["concat"]):
                segments.extend(_plan(mapping(child, "concat clip")))
        elif "trim" in edit:
            params = mapping(edit["trim"], "trim")
            length = cast("Fraction | None", effective_video_facts(current)["duration"])
            start, selected = trim_window(
                length,
                params.get("start_time", 0),
                params.get("duration", 0),
                edit.get("strict_duration", False),
            )
            requested = seconds(params.get("duration", 0), "duration")
            end = (
                start + selected
                if selected is not None
                else (start + requested if requested else None)
            )
            offset = Fraction(0)
            kept: list[_Segment] = []
            for segment in segments:
                if segment.duration is None and len(segments) != 1:
                    raise ValueError("concat materialization requires known clip durations")
                segment_end = offset + segment.duration if segment.duration is not None else None
                first = max(offset, start)
                last = (
                    min(segment_end, end)
                    if segment_end is not None and end is not None
                    else segment_end or end
                )
                if last is None or first < last:
                    kept.append(
                        replace(
                            segment,
                            start=segment.start + first - offset,
                            duration=last - first if last is not None else None,
                        )
                    )
                if segment_end is not None:
                    offset = segment_end
            segments = kept
        else:
            facts = effective_video_facts(current)
            width, height = cast(int, facts["width"]), cast(int, facts["height"])
            noop = (
                crop_rectangle(width, height, mapping(edit["crop"], "crop"))
                == (0, 0, width, height)
                if "crop" in edit
                else scale_geometry(width, height, mapping(edit["scale"], "scale"))
                == (width, height, 0, 0, width, height)
            )
            if not noop:
                segments = [replace(s, spatial=(*s.spatial, edit)) for s in segments]
        current["edits"] = [*cast("list[object]", current["edits"]), edit]
    return segments


def source_trim_window(obj: object) -> tuple[Fraction, Fraction | None]:
    """Resolve the temporal window relative to the value's original encoded source."""
    value = coerce_video(obj)
    segments = _plan(value)
    if "source" not in value or len(segments) != 1 or segments[0].value is not value:
        raise ValueError("VIDEO edits cannot be represented as one original-source trim window")
    return segments[0].start, segments[0].duration


def _alpha_pixels(frame: Any) -> np.ndarray:
    """Alpha is full-range coverage; swscale's bit-shift rescaling is not normalized coverage."""
    import numpy as np

    component = next(c for c in frame.format.components if c.is_alpha)
    if frame.format.is_planar:
        plane = frame.planes[component.plane]
        dtype = (
            "<f4"
            if "f32" in frame.format.name
            else "<f2"
            if "f16" in frame.format.name
            else "u1"
            if component.bits <= 8
            else ">u2"
            if frame.format.name.endswith("be")
            else "<u2"
        )
        array = np.frombuffer(plane, dtype=dtype).reshape(plane.height, -1)[:, : frame.width]
        return array.astype(np.float32) / (1 if "f" in dtype else 2**component.bits - 1)
    array = cast(np.ndarray, frame.to_ndarray(format="rgba64le"))
    return array[..., 3].astype(np.float32) / 65535


def _pixels(frame: Any, depth: int, alpha: bool) -> np.ndarray:
    import numpy as np

    packed = "rgba64le" if alpha else "rgb48le"
    if depth > 8 and frame.format.name == packed:
        return cast(np.ndarray, frame.to_ndarray())
    byte_output = depth <= 8 and (frame.format.is_rgb or frame.color_range == 2)
    fmt = (
        ("rgba" if alpha else "rgb24") if byte_output else ("gbrapf32le" if alpha else "gbrpf32le")
    )
    converted = (
        frame
        if frame.format.name == fmt
        else frame.reformat(
            format=fmt,
            src_colorspace=frame.colorspace if frame.colorspace not in (0, 2) else 5,
            # RGB's identity matrix also forces PyAV 16 to configure equal-range conversions.
            dst_colorspace=0,
            src_color_range=2 if frame.format.is_rgb else frame.color_range or 1,
            dst_color_range=2,
        )
    )
    # PyAV planar float conversion omits row alignment padding in to_ndarray.
    pixels = converted.to_ndarray()
    if alpha:
        alpha_pixels = _alpha_pixels(frame)
        if byte_output:
            pixels[..., 3] = np.rint(alpha_pixels * 255).astype(np.uint8)
        else:
            pixels[..., 3] = alpha_pixels
    if byte_output:
        return pixels
    maximum = 255 if depth <= 8 else 65535
    return np.rint(np.clip(pixels, 0, 1) * maximum).astype(np.uint8 if depth <= 8 else np.uint16)


def _frame(array: np.ndarray, depth: int) -> Any:
    import av
    import numpy as np

    alpha = array.shape[-1] == 4
    if depth <= 8 and array.dtype == np.uint8:
        return av.VideoFrame.from_ndarray(
            np.ascontiguousarray(array), format="rgba" if alpha else "rgb24"
        )
    if depth > 8 and not alpha and array.dtype == np.uint16:
        return av.VideoFrame.from_ndarray(
            np.ascontiguousarray(array, dtype="<u2"), format="rgb48le"
        )
    array = cast(np.ndarray, image_input(array))
    if depth <= 8:
        pixels = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        return av.VideoFrame.from_ndarray(pixels, format="rgba" if alpha else "rgb24")
    pixels = np.rint(np.clip(array, 0, 1) * 65535).astype("<u2")
    return av.VideoFrame.from_ndarray(pixels, format="rgba64le" if alpha else "rgb48le")


def _encoder_frame(array: np.ndarray, depth: int) -> Any:
    """Convert annotated IMAGE pixels to the straight-alpha representation encoders require."""
    import numpy as np

    if media_semantics(array).get("alpha") == "premultiplied":
        array = cast(np.ndarray, image_input(array)).copy()
        alpha = array[..., -1:]
        array[..., :-1] = np.divide(
            array[..., :-1], alpha, out=np.zeros_like(array[..., :-1]), where=alpha != 0
        )
    return _frame(array, depth)


def _transform(array: np.ndarray, segment: _Segment, depth: int) -> np.ndarray:
    import numpy as np

    probe = mapping(segment.value["probe"], "probe")
    premultiplied = media_semantics(array).get("alpha") == "premultiplied"
    rotation = int(cast(float, probe["rotation"]))
    if rotation:
        array = np.rot90(array, rotation // 90)
    for edit in segment.spatial:
        height, width = array.shape[:2]
        if "crop" in edit:
            x, y, w, h = crop_rectangle(width, height, mapping(edit["crop"], "crop"))
            array = array[y : y + h, x : x + w]
        else:
            params = mapping(edit["scale"], "scale")
            w, h, x, y, out_w, out_h = scale_geometry(width, height, params)
            interpolation = {
                "nearest": "POINT",
                "bilinear": "BILINEAR",
                "area": "AREA",
                "bicubic": "BICUBIC",
                "lanczos": "LANCZOS",
            }[str(params.get("interpolation", "bilinear"))]
            resized = _frame(array, depth).reformat(width=w, height=h, interpolation=interpolation)
            array = _pixels(resized, depth, array.shape[-1] == 4)
            if params.get("fit") == "crop":
                array = array[y : y + out_h, x : x + out_w]
            elif params.get("fit") == "pad":
                color = list(cast("list[float]", params.get("pad_color", [0, 0, 0, 1])))
                if len(color) == 3:
                    color.append(1)
                if premultiplied and array.shape[-1] == 4:
                    color[:3] = [sample * color[3] for sample in color[:3]]
                padded = np.empty((out_h, out_w, array.shape[-1]), dtype=array.dtype)
                if np.issubdtype(array.dtype, np.integer):
                    padded[:] = np.rint(
                        np.asarray(color[: array.shape[-1]]) * np.iinfo(array.dtype).max
                    )
                else:
                    padded[:] = color[: array.shape[-1]]
                padded[y : y + h, x : x + w] = array
                array = padded
    return np.ascontiguousarray(array)


def _decoded_video(container: Any, stream: Any) -> Iterator[Any]:
    import av

    stream.codec_context.thread_count = 1
    if (
        stream.codec_context.codec.canonical_name != "vp9"
        or stream.metadata.get("alpha_mode") != "1"
    ):
        yield from container.decode(stream)
        return
    decoder = cast(Any, av.CodecContext.create("libvpx-vp9", "r"))
    decoder.thread_count = 1
    decoder.extradata = stream.codec_context.extradata
    for packet in container.demux(stream):
        if packet.dts is not None:
            yield from decoder.decode(packet)
    yield from decoder.decode(None)


def _video_frames(segments: list[_Segment]) -> Generator[tuple[Fraction, Any]]:
    import av

    offset = Fraction(0)
    for segment in segments:
        value = segment.value
        probe = mapping(value["probe"], "probe")
        depth = int(cast(int, probe["bit_depth"]) or 8)
        end = segment.start + segment.duration if segment.duration is not None else None
        if "components" in value:
            components = mapping(value["components"], "components")
            rate = cast(Fraction, components["fps"])
            images = cast("np.ndarray", components["images"])
            for index, raw in enumerate(images):
                t = Fraction(index) / rate
                if t >= segment.start and (end is None or t < end):
                    array = copy_media_semantics(images, raw)
                    transformed = _transform(array, segment, depth)
                    yield offset + t - segment.start, copy_media_semantics(array, transformed)
        else:
            with open_video_source(video_source(value)) as handle, av.open(handle) as opened:
                container = cast(Any, opened)
                stream = container.streams.video[0]
                origin = cast(Fraction, probe["start_time"])
                if segment.start:
                    container.seek(int((origin + segment.start) / stream.time_base), stream=stream)
                for frame in _decoded_video(container, stream):
                    # VP8 cannot signal a matrix; the container outranks its decoder default.
                    if stream.codec_context.codec.canonical_name == "vp8" and probe[
                        "matrix"
                    ] not in (None, 2):
                        frame.colorspace = probe["matrix"]
                    elif frame.colorspace in (0, 2) and probe["matrix"] not in (None, 2):
                        frame.colorspace = probe["matrix"]
                    if frame.color_range == 0 and probe["range"]:
                        frame.color_range = probe["range"]
                    if frame.pts is None:
                        raise ValueError("video frame has no timestamp")
                    t = Fraction(frame.pts) * frame.time_base - origin
                    if t < segment.start:
                        continue
                    if end is not None and t >= end:
                        break
                    if segment.spatial or probe["rotation"]:
                        array = _pixels(frame, depth, bool(probe["alpha"]))
                        yield offset + t - segment.start, _transform(array, segment, depth)
                    else:
                        yield offset + t - segment.start, frame
        if segment.duration is not None:
            offset += segment.duration


def _audio_frames(segments: list[_Segment], index: int) -> Generator[tuple[Fraction, Any]]:
    import av
    import numpy as np

    offset = Fraction(0)
    for segment in segments:
        value = segment.value
        end = segment.start + segment.duration if segment.duration is not None else None
        if "components" in value:
            audio = mapping(mapping(value["components"], "components")["audio"], "audio")
            facts = effective_audio_facts(audio)
            rate = facts["sample_rate"]
            start = math.ceil(segment.start * rate)
            last = facts["frames"]
            if end is not None:
                last = math.ceil(end * rate) if last is None else min(last, math.ceil(end * rate))
            with AudioWindowReader(audio) as reader:
                while last is None or start < last:
                    count = 1024 if last is None else min(1024, last - start)
                    waveform = reader.read(start, count, batch_index=0)["waveform"][0]
                    if not waveform.shape[1]:
                        break
                    frame = av.AudioFrame.from_ndarray(
                        np.ascontiguousarray(waveform),
                        format="fltp",
                        layout=facts["layout"],
                    )
                    frame.sample_rate = rate
                    yield offset + Fraction(start, rate) - segment.start, frame
                    start += waveform.shape[1]
        else:
            with open_video_source(video_source(value)) as handle, av.open(handle) as opened:
                container = cast(Any, opened)
                stream = container.streams.audio[index]
                stream.codec_context.thread_count = 1
                origin = cast(Fraction, mapping(value["probe"], "probe")["start_time"])
                resampler = av.AudioResampler(
                    format="fltp", layout=stream.layout.name, rate=stream.rate
                )
                if segment.start:
                    container.seek(int((origin + segment.start) / stream.time_base), stream=stream)
                finished = False
                for decoded in container.decode(stream):
                    for frame in resampler.resample(decoded):
                        if frame.pts is None or frame.time_base is None:
                            raise ValueError("audio frame has no timestamp")
                        t = Fraction(frame.pts) * frame.time_base - origin
                        first = max(0, math.ceil((segment.start - t) * frame.sample_rate))
                        last = (
                            min(frame.samples, math.ceil((end - t) * frame.sample_rate))
                            if end is not None
                            else frame.samples
                        )
                        if end is not None and t >= end:
                            finished = True
                            break
                        if first >= last:
                            continue
                        array = frame.to_ndarray()[:, first:last]
                        result = av.AudioFrame.from_ndarray(
                            np.ascontiguousarray(array, dtype=np.float32),
                            format="fltp",
                            layout=frame.layout.name,
                        )
                        result.sample_rate = frame.sample_rate
                        yield (
                            offset + t + Fraction(first, frame.sample_rate) - segment.start,
                            result,
                        )
                    if finished:
                        break
        if segment.duration is not None:
            offset += segment.duration


def _audio_count(value: Mapping[str, object]) -> int:
    if "components" in value:
        return int(mapping(value["components"], "components").get("audio") is not None)
    return len(cast(list[object], mapping(value["probe"], "probe")["audio"]))


def _extract_audio(segments: list[_Segment], measured_end: Fraction) -> object:
    selected: list[object] = []
    for segment in segments:
        value = segment.value
        if not _audio_count(value):
            continue
        shift = Fraction(0)
        if "components" in value:
            audio = mapping(value["components"], "components")["audio"]
        else:
            probe = mapping(value["probe"], "probe")
            streams = cast(list[object], probe["audio"])
            position = len(streams) - 1
            stream = mapping(streams[position], "audio stream")
            shift = cast(Fraction, probe["start_time"]) - cast(Fraction, stream["start_time"] or 0)
            source = video_source(value)
            if isinstance(source, bytes):
                raise ValueError(
                    "VIDEO AUDIO extraction requires a published source; prepare inline VIDEO "
                    "with dinkster_assets.value.bind_video_value(..., for_audio_extraction=True)"
                )
            audio = audio_from_source(source, stream_index=position)
        duration = segment.duration
        if duration is None:
            if len(segments) != 1:
                raise ValueError("concat extraction requires known clip durations")
            duration = measured_end
        facts = effective_audio_facts(audio)
        rate = facts["sample_rate"]
        start = max(0, math.ceil((shift + segment.start) * rate))
        end = max(0, math.ceil((shift + segment.start + duration) * rate))
        if facts["frames"] is not None:
            start, end = min(start, facts["frames"]), min(end, facts["frames"])
        if end > start:
            selected.append(
                append_audio_edit(
                    audio, {"trim": {"start_sample": start, "sample_count": end - start}}
                )
            )
    if not selected:
        return None
    return (
        append_audio_edit(selected[0], {"concat": selected[1:]})
        if len(selected) > 1
        else selected[0]
    )


def iter_export_frames(value: Mapping[str, object], depth: int) -> Generator[tuple[Fraction, Any]]:
    """Yield RGB frames for image-only exporters through the shared edit/decoder plan."""
    import numpy as np

    if _audio_count(value):
        raise ValueError(
            "animated images and PNG sequences cannot contain audio; detach it explicitly"
        )
    probe = mapping(value["probe"], "probe")
    plan = _plan(value)
    _validate_concat(plan)
    with closing(_video_frames(plan)) as frames:
        for timestamp, item in frames:
            array = (
                cast(np.ndarray, item)
                if isinstance(item, np.ndarray)
                else _pixels(item, int(cast(int, probe["bit_depth"]) or 8), bool(probe["alpha"]))
            )
            frame = _encoder_frame(array, depth)
            del array
            yield timestamp, frame


def iter_video_pixels(obj: object) -> Generator[tuple[Fraction, Any]]:
    """Yield edited display-oriented pixels with bounded decoder ownership."""
    import numpy as np

    value = coerce_video(obj)
    probe = mapping(value["probe"], "probe")
    with closing(_video_frames(_plan(value))) as frames:
        for time, frame in frames:
            pixels = cast(
                np.ndarray,
                frame
                if isinstance(frame, np.ndarray)
                else _pixels(frame, int(cast(int, probe["bit_depth"]) or 8), bool(probe["alpha"])),
            )
            yield (
                time,
                cast(np.ndarray, image_input(cast(object, pixels))),
            )


def iter_video_audio(obj: object, index: int = 0) -> Generator[tuple[Fraction, Any]]:
    """Yield edited audio frames without accumulating the complete waveform."""
    with closing(_audio_frames(_plan(coerce_video(obj)), index)) as frames:
        yield from frames


def disassemble_video(obj: object) -> dict[str, object]:
    import numpy as np

    value = coerce_video(obj)
    probe = mapping(value["probe"], "probe")
    plan = _plan(value)
    _validate_concat(plan)
    arrays: list[np.ndarray] = []
    size = 0
    measured_end = Fraction(0)
    for timestamp, array in _video_frames(plan):
        frame_duration = (
            Fraction(array.duration) * array.time_base
            if not isinstance(array, np.ndarray) and array.duration and array.time_base
            else 1 / cast(Fraction, probe["fps"])
            if probe["fps"] is not None
            else Fraction(0)
        )
        measured_end = max(measured_end, timestamp + frame_duration)
        if not isinstance(array, np.ndarray):
            array = _pixels(array, int(cast(int, probe["bit_depth"]) or 8), bool(probe["alpha"]))
        size += array.nbytes
        if size > VIDEO_DECODE_WORKING_SET_LIMIT_BYTES:
            raise ValueError("decoded video frames exceed the 512 MiB limit")
        arrays.append(cast(np.ndarray, array))
    if not arrays:
        raise ValueError("video selection contains no frames")
    facts = effective_video_facts(value)
    duration = cast("Fraction | None", facts["duration"])
    rate = cast("Fraction | None", facts["fps"])
    if duration is None:
        duration = measured_end
    if rate is None:
        if duration <= 0:
            raise ValueError("decoded VIDEO has no usable timing information")
        rate = Fraction(len(arrays)) / duration
    images = np.stack(arrays)
    if "components" in value:
        images = copy_media_semantics(mapping(value["components"], "components")["images"], images)
    color = {"primaries": probe["primaries"], "transfer": probe["transfer"], "range": 2}
    if probe["matrix"] not in (0, 2):
        color["matrix"] = probe["matrix"]
    if probe["bit_depth"] is not None:
        color["bit_depth"] = probe["bit_depth"]
    return {
        "images": annotate_image(images, color=color),
        "audio": _extract_audio(plan, measured_end),
        "frame_count": len(arrays),
        "fps": float(rate),
        "duration": float(duration),
        "bit_depth": probe["bit_depth"],
        "color_space": probe["color_space"],
    }


def _templates(container: Any) -> tuple[object, ...]:
    result: list[object] = []
    for stream in container.streams:
        if stream.type not in ("video", "audio"):
            continue
        context = stream.codec_context
        common = (
            stream.type,
            context.codec.canonical_name,
            context.extradata,
            context.profile,
            context.format.name if context.format else None,
            stream.time_base,
            stream.metadata.get("alpha_mode"),
        )
        if stream.type == "video":
            result.append(
                (
                    *common,
                    context.width,
                    context.height,
                    context.sample_aspect_ratio,
                    context.color_primaries,
                    context.color_trc,
                    context.colorspace,
                    context.color_range,
                )
            )
        else:
            result.append((*common, context.sample_rate, context.layout.name))
    return tuple(result)


def _validate_concat(segments: list[_Segment]) -> None:
    import av

    if len(segments) < 2:
        return
    template: tuple[object, ...] | None = None
    for segment in segments:
        if "source" not in segment.value:
            raise ValueError("identical-codec concat requires encoded source clips")
        probe = mapping(segment.value["probe"], "probe")
        with open_video_source(video_source(segment.value)) as handle, av.open(handle) as opened:
            actual = (_templates(opened), probe["rotation"], probe["alpha"])
        if template is None:
            template = actual
        elif template != actual:
            raise ValueError("concat requires identical video and audio stream templates")


def _independent_packet(packet: Any) -> bool:
    context = packet.stream.codec_context
    if not packet.is_keyframe:
        return False
    if context.codec.canonical_name in ("ffv1", "mjpeg", "png", "rawvideo"):
        return True
    if context.codec.canonical_name != "h264":
        return False
    extra = cast("bytes | None", context.extradata)
    if not extra or len(extra) < 5 or extra[0] != 1:
        return False
    width = (extra[4] & 3) + 1
    data = bytes(packet)
    offset = 0
    idr = False
    while offset + width <= len(data):
        size = int.from_bytes(data[offset : offset + width], "big")
        offset += width
        if size < 1 or offset + size > len(data):
            return False
        kind = data[offset] & 31
        if kind in (1, 2, 3, 4):
            return False
        idr |= kind == 5
        offset += size
    return idr and offset == len(data)


def _copy_preflight(segments: list[_Segment]) -> bool:
    """Prove exact closed cuts; a keyframe flag alone never establishes safety."""
    import av

    safe = True
    offset = Fraction(0)
    for segment in segments:
        value = segment.value
        if "source" not in value:
            return False
        probe = mapping(value["probe"], "probe")
        with open_video_source(video_source(value)) as handle, av.open(handle) as opened:
            source = cast(Any, opened)
            if segment.spatial or segment.duration is None:
                return False
            origin = cast(Fraction, probe["start_time"])
            start, end = segment.start, segment.start + segment.duration
            if any(
                s.time_base is None or ((offset - start - origin) / s.time_base).denominator != 1
                for s in source.streams
                if s.type in ("video", "audio")
            ):
                return False
            spans: dict[int, tuple[Fraction, Fraction]] = {}
            for packet in source.demux():
                if packet.stream.type not in ("video", "audio") or not packet.size:
                    continue
                if packet.pts is None or packet.dts is None or not packet.duration:
                    safe = False
                    continue
                t = Fraction(packet.pts) * packet.time_base - origin
                stop = t + Fraction(packet.duration) * packet.time_base
                if stop <= start or t >= end:
                    continue
                if t < start or stop > end or packet.pts != packet.dts:
                    safe = False
                index = packet.stream.index
                previous = spans.get(index)
                if previous is None:
                    if packet.stream.type == "video" and not _independent_packet(packet):
                        safe = False
                    spans[index] = (t, stop)
                else:
                    if t != previous[1]:
                        safe = False
                    spans[index] = (previous[0], stop)
            indexes = {s.index for s in source.streams if s.type in ("video", "audio")}
            if set(spans) != indexes or any(span != (start, end) for span in spans.values()):
                safe = False
        offset += segment.duration
    return safe


def _copy_segments(segments: list[_Segment], output: Any, tags: Mapping[str, str]) -> None:
    import av

    streams: list[Any] = []
    offset = Fraction(0)
    for segment in segments:
        probe = mapping(segment.value["probe"], "probe")
        origin = cast(Fraction, probe["start_time"])
        assert segment.duration is not None
        with open_video_source(video_source(segment.value)) as handle, av.open(handle) as opened:
            source = cast(Any, opened)
            incoming = [s for s in source.streams if s.type in ("video", "audio")]
            if not streams:
                output.metadata.update(source.metadata)
                output.metadata.update(tags)
                streams = [output.add_stream_from_template(s) for s in incoming]
            destinations = {s.index: target for s, target in zip(incoming, streams, strict=True)}
            for packet in source.demux():
                if not packet.size or packet.stream.index not in destinations:
                    continue
                t = Fraction(packet.pts) * packet.time_base - origin
                if not segment.start <= t < segment.start + segment.duration:
                    continue
                shift = (offset - segment.start - origin) / packet.time_base
                if shift.denominator != 1:
                    raise ValueError(
                        "concat timestamps cannot be represented in the stream time base"
                    )
                packet.pts += int(shift)
                packet.dts += int(shift)
                packet.stream = destinations[packet.stream.index]
                output.mux(packet)
        offset += segment.duration


def _fit_audio(
    frames: Iterator[tuple[Fraction, Any]], duration: Fraction | None, *, pad: bool
) -> Generator[tuple[Fraction, Any]]:
    """Clip audio to the video and optionally fill gaps with bounded silence frames."""
    import av
    import numpy as np

    end = Fraction(0)
    last: Any = None

    def silence(until: Fraction, template: Any) -> Generator[tuple[Fraction, Any]]:
        nonlocal end
        rate = template.sample_rate
        while end < until:
            count = min(1024, math.ceil((until - end) * rate))
            blank = av.AudioFrame.from_ndarray(
                np.zeros((len(template.layout.channels), count), np.float32),
                format="fltp",
                layout=template.layout.name,
            )
            blank.sample_rate = rate
            yield end, blank
            end += Fraction(count, rate)

    for timestamp, frame in frames:
        if duration is not None and timestamp >= duration:
            break
        if pad and timestamp > end:
            yield from silence(timestamp, frame)
        count = frame.samples
        if duration is not None:
            count = min(count, math.ceil((duration - timestamp) * frame.sample_rate))
        if count < frame.samples:
            clipped = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(frame.to_ndarray()[:, :count]),
                format="fltp",
                layout=frame.layout.name,
            )
            clipped.sample_rate = frame.sample_rate
            frame = clipped
        yield timestamp, frame
        end = timestamp + Fraction(count, frame.sample_rate)
        last = frame
    if pad and duration is not None and last is not None:
        yield from silence(duration, last)


def _exact_component_audio_length(audio: object) -> bool:
    """Only validated PCM roots certify coverage; encoded durations are estimates."""
    value = coerce_audio(audio)
    source, probe = value["source"], mapping(value["probe"], "audio probe")
    if not (isinstance(source, Mapping) and "pcm" in source):
        if probe["codec"] != "pcm_npy":
            return False
        measured = mapping(
            audio_from_source(cast(Any, source), stream_index=cast(int, probe["stream_index"]))[
                "probe"
            ],
            "PCM probe",
        )
        if any(
            measured[key] != probe[key]
            for key in ("codec", "frames", "sample_rate", "channels", "batch")
        ):
            return False
    return all(
        _exact_component_audio_length(child)
        for edit in cast(list[dict[str, Any]], value["edits"])
        for child in edit.get("concat", [])
    )


@encoded_diagnostics
def save_video_stream(
    obj: object,
    destination: BinaryIO,
    *,
    container: str = "auto",
    codec: str = "auto",
    crf: int | None = None,
    metadata: Mapping[str, object] | None = None,
    profile: str = "auto",
    audio_layout: str = "preserve",
    trim_to_audio: bool = False,
    on_diagnostic: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[str, str]:
    """Write one container; the caller owns bounded spooling and atomic publication."""
    import av
    import numpy as np
    from av.codec.codec import UnknownCodecError
    from av.error import FFmpegError

    from .formats import metadata_tags

    value = coerce_video(obj)
    timeline_stream: tuple[dict[str, object], Iterator[Any], Iterator[Any]] | None = None
    if "timeline" in value:
        from dinkster_values.timeline_video import TimelineVideo

        from .timeline_runtime import timeline_streams

        timeline_stream = timeline_streams(cast(TimelineVideo, value))
        probe = {**timeline_stream[0], "container": None, "video_codec": None, "pix_fmt": None}
    else:
        probe = mapping(value["probe"], "probe")
    kind = str(probe["container"] or "mp4") if container == "auto" else container
    name = (
        str(
            probe["video_codec"] or ("vp9" if probe["alpha"] else "av1")
            if kind == "webm"
            else probe["video_codec"] or "h264"
        )
        if codec == "auto"
        else codec
    )
    name = "hevc" if name == "h265" else name
    from dinkster_values.video_codec import VIDEO_CONTAINERS, video_rendition_mime

    from .formats import format_diagnostic, save_video_frames

    requested = {"container": kind, "codec": name, "profile": profile, "audio_layout": audio_layout}
    if any(token in name or token in kind for token in ("{", "[", "\n")):
        raise ValueError("custom FFmpeg command JSON is refused")
    if name in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
        raise ValueError("NVENC is refused until a hardware-encoder policy exists")
    alpha, depth = bool(probe["alpha"]), int(cast(int, probe["bit_depth"]) or 8)
    if kind not in VIDEO_CONTAINERS:
        kind = "mkv"
    tags = metadata_tags(metadata)
    if audio_layout not in ("preserve", "mono", "stereo"):
        audio_layout = "preserve"
    if type(trim_to_audio) is not bool:
        raise ValueError("trim_to_audio must be boolean")
    if profile not in ("auto", "lt", "standard", "hq", "4444", "4444xq"):
        profile = "auto"
    if profile != "auto" and name != "prores":
        profile = "auto"
    if name not in (*_ENCODERS, "prores", "ffv1", "gif", "mjpeg", "png", "rawvideo"):
        name = "ffv1"
    audio_count = int(bool(probe["audio"])) if timeline_stream else _audio_count(value)
    if kind == "gif":
        if alpha or depth > 8 or audio_count or probe["transfer"] in (16, 18):
            kind, name = "mkv", "ffv1"
        else:
            return save_video_frames(
                value,
                destination,
                format="gif_ffmpeg",
                metadata=metadata,
                on_diagnostic=on_diagnostic,
            )
    if name == "prores":
        kind = "mov"
        if alpha and profile not in ("auto", "4444", "4444xq"):
            profile = "4444"
    elif name == "ffv1":
        kind = "mkv"
    if kind == "webm" and name not in ("av1", "vp9", "vp8"):
        kind = "mkv"
    if (
        alpha
        and name not in ("vp9", "prores", "ffv1", "png", "rawvideo")
        or alpha
        and depth > 8
        and name == "vp9"
    ):
        kind, name, profile = "mkv", "ffv1", "auto"
    if kind != probe["container"] or name != probe["video_codec"]:
        with av.open(io.BytesIO(), "w", format=_CONTAINERS.get(kind, kind)) as candidate:
            if name not in candidate.supported_codecs:
                kind = "mkv"
    effective = {"container": kind, "codec": name, "profile": profile, "audio_layout": audio_layout}
    if requested != effective:
        format_diagnostic(requested, effective, on_diagnostic)
    mime = video_rendition_mime({"container": kind})
    mux_options = {"movflags": "use_metadata_tags"} if kind in ("mp4", "mov") else {}
    plan = [] if timeline_stream is not None else _plan(value)
    _validate_concat(plan)
    facts = effective_video_facts(value)
    unchanged = (
        len(plan) == 1
        and plan[0].value is value
        and plan[0].start == 0
        and plan[0].duration == probe["duration"]
        and not plan[0].spatial
    )
    copy_requested = crf is None and profile == "auto" and audio_layout == "preserve"
    copy_requested = copy_requested and not trim_to_audio
    audio_aligned = timeline_stream is None and all(
        mapping(track, "audio")["duration"] == probe["duration"]
        and mapping(track, "audio")["start_time"] == probe["start_time"]
        for track in cast(list[object], probe["audio"])
    )
    if (
        "source" in value
        and unchanged
        and audio_aligned
        and name == probe["video_codec"]
        and copy_requested
    ):
        with open_video_source(video_source(value)) as handle:
            if kind == probe["container"] and not tags:
                while chunk := handle.read(MEBIBYTE):
                    destination.write(chunk)
                return "." + kind, mime
            with (
                av.open(handle) as source,
                av.open(
                    destination, "w", format=_CONTAINERS.get(kind, kind), options=mux_options
                ) as target,
            ):
                incoming, outgoing = cast(Any, source), cast(Any, target)
                outgoing.metadata.update(incoming.metadata)
                outgoing.metadata.update(tags)
                streams = {
                    s.index: outgoing.add_stream_from_template(s)
                    for s in incoming.streams
                    if s.type in ("video", "audio")
                }
                for packet in incoming.demux():
                    if packet.dts is not None and packet.stream.index in streams:
                        packet.stream = streams[packet.stream.index]
                        outgoing.mux(packet)
                return "." + kind, mime
    copy_safe = timeline_stream is None and _copy_preflight(plan)
    if copy_safe and name == probe["video_codec"] and copy_requested:
        with av.open(
            destination, "w", format=_CONTAINERS.get(kind, kind), options=mux_options
        ) as opened:
            output = cast(Any, opened)
            _copy_segments(plan, output, tags)
        return "." + kind, mime
    uncertain_audio_end = False
    for segment in plan:
        if "components" in segment.value:
            component = mapping(segment.value["components"], "components").get("audio")
            tracks = [effective_audio_facts(component)] if component is not None else []
            if tracks:
                if segment.duration is None and tracks[0]["frames"] is None:
                    raise ValueError("component AUDIO requires a finite AUDIO or VIDEO read bound")
                if trim_to_audio and not _exact_component_audio_length(component):
                    uncertain_audio_end = True
        else:
            tracks = cast(list[Any], mapping(segment.value["probe"], "probe")["audio"])
        for track in tracks:
            try:
                layout_name = str(track["layout"])
                if layout_name.endswith("c") and layout_name[:-1].isdigit():
                    raise ValueError("channel count does not declare speaker positions")
                channels = av.AudioLayout(track["layout"]).channels
                if len(channels) > 8:
                    raise ValueError("PyAV planar audio conversion exceeds eight channels")
                if audio_layout != "preserve" and any(c.name == "NONE" for c in channels):
                    raise ValueError("discrete downmix requires an explicit matrix")
            except ValueError as error:
                raise ValueError(
                    "PyAV cannot safely encode this discrete audio layout; preserve the "
                    "source bytes or apply an explicit canonical AUDIO channel_map matrix"
                ) from error
    if uncertain_audio_end:
        if facts["duration"] is None:
            raise ValueError(
                "trim_to_audio with unproven component AUDIO coverage requires finite VIDEO"
            )
        format_diagnostic(
            {"trim_to_audio": True},
            {"trim_to_audio": False},
            on_diagnostic,
            reason="component_audio_endpoint_unproven",
        )
        trim_to_audio = False
    encoder = _ENCODERS.get(name, "prores_ks" if name == "prores" else name)
    formats: set[str]
    try:
        formats = {item.name for item in av.Codec(encoder, "w").video_formats or ()}
    except UnknownCodecError:
        formats = set()
    pixel_format = "yuva420p" if alpha else "yuv420p10le" if depth > 8 else "yuv420p"
    if name == "prores":
        profile = ("4444" if alpha else "hq") if profile == "auto" else profile
        pixel_format = (
            "yuva444p10le"
            if alpha
            else "yuv444p10le"
            if profile in ("4444", "4444xq")
            else "yuv422p10le"
        )
    elif name == "ffv1":
        pixel_format = (
            ("rgba64le" if alpha else "rgb48le") if depth > 8 else "bgra" if alpha else "bgr0"
        )
    source_format = probe["pix_fmt"]
    if (
        codec == "auto"
        and isinstance(source_format, str)
        and source_format in formats
        and (not alpha or any(c.is_alpha for c in av.VideoFormat(source_format).components))
    ):
        pixel_format = source_format
    pixel = av.VideoFormat(pixel_format)
    width, height = cast(int, facts["width"]), cast(int, facts["height"])
    if (
        pixel_format not in formats
        or max((component.bits for component in pixel.components), default=0) < depth
        or (alpha and not any(component.is_alpha for component in pixel.components))
        or (pixel.chroma_width(2) == 1 and width % 2)
        or (pixel.chroma_height(2) == 1 and height % 2)
        or (name == "av1" and min(width, height) < 64)
    ):
        if name == "ffv1":
            raise ValueError("CPU FFV1 cannot retain source pixel format, depth, and alpha")
        format_diagnostic(effective, {"container": "mkv", "codec": "ffv1"}, on_diagnostic)
        return save_video_stream(
            value,
            destination,
            container="mkv",
            codec="ffv1",
            metadata=metadata,
            audio_layout=audio_layout,
            trim_to_audio=trim_to_audio,
            on_diagnostic=on_diagnostic,
        )
    limit = 51 if name in ("h264", "hevc") else 63
    if crf is not None and name not in _ENCODERS:
        format_diagnostic({"crf": crf}, {"crf": None, "codec": name}, on_diagnostic)
        crf = None
    if crf is not None and (type(crf) is not int or not 0 <= crf <= limit):
        raise ValueError(f"crf must be an integer in 0..{limit}")
    rate = cast("Fraction | None", facts["fps"])
    if rate is None:
        raise ValueError("encoding requires a known frame rate")
    color = {key: probe[key] for key in _COLOR_KEYS}
    if pixel.is_rgb:
        color.update(matrix=0, range=2)
    elif color["matrix"] == 0:
        color["matrix"] = _MATRIX_FOR_PRIMARIES.get(cast(int, color["primaries"]), 2)
    destination_matrix = 0 if pixel.is_rgb else color["matrix"] if color["matrix"] != 2 else 5
    destination_range = color["range"] or 1
    with (
        av.open(
            destination, "w", format=_CONTAINERS.get(kind, kind), options=mux_options
        ) as opened,
        ExitStack() as readers,
    ):
        output = cast(Any, opened)
        output.metadata.update(tags)
        video = output.add_stream(encoder, rate=rate)
        video.width, video.height = facts["width"], facts["height"]
        video.pix_fmt = pixel_format
        video.codec_context.thread_count = 1
        video.codec_context.max_b_frames = 0
        video.codec_context.time_base = Fraction(1, 1_000_000)
        for key, attr in zip(
            _COLOR_KEYS, ("color_primaries", "color_trc", "colorspace", "color_range"), strict=True
        ):
            setattr(video.codec_context, attr, color[key])
        if pixel.is_rgb:
            video.codec_context.colorspace, video.codec_context.color_range = 0, 2
        video.options = {"crf": str(crf if crf is not None else 23)} if name in _ENCODERS else {}
        if name == "h264":
            video.options.update({"preset": "fast", "tune": "zerolatency"})
        elif name == "hevc":
            video.codec_context.codec_tag = "hvc1"
            video.options.update(
                {
                    "preset": "medium",
                    "tune": "zerolatency",
                    "x265-params": "pools=none:frame-threads=1:log-level=error",
                }
            )
        elif name == "prores":
            video.options = {"profile": profile, "alpha_bits": "16"}
        elif name == "ffv1":
            video.options = {
                "level": "3",
                "coder": "1",
                "context": "0",
                "slicecrc": "1",
                "slices": "4",
            }
            video.gop_size = 1
        elif name in ("vp8", "vp9"):
            video.options["lag-in-frames"] = "0"
        elif name == "av1":
            parameters = "lp=1:pred-struct=1:lookahead=0:enable-tf=0" + (
                ":crf=0" if crf == 0 else ""
            )
            video.options.update({"preset": "8", "svtav1-params": parameters})
        streams = [video]
        iterators: list[Iterator[tuple[Fraction, object]]] = [
            readers.enter_context(
                closing(timeline_stream[1] if timeline_stream else _video_frames(plan))
            )
        ]
        resamplers: list[Any] = [None]
        for index in range(audio_count):
            decoded_audio = readers.enter_context(
                closing(timeline_stream[2] if timeline_stream else _audio_frames(plan, index))
            )
            iterator = readers.enter_context(
                closing(
                    _fit_audio(
                        decoded_audio,
                        cast(Fraction | None, facts["duration"]),
                        pad=not trim_to_audio,
                    )
                )
            )
            try:
                first = next(iterator)
            except StopIteration:
                if trim_to_audio:
                    raise ValueError("audio selection contains no samples") from None
                continue
            import itertools

            frame = first[1]
            layout = frame.layout.name if audio_layout == "preserve" else audio_layout
            if audio_layout == "preserve" and "source" in value:
                layout = cast(
                    str, mapping(cast(list[object], probe["audio"])[index], "audio")["layout"]
                )
            audio_encoder = (
                "libopus"
                if kind == "webm"
                else "flac"
                if name == "ffv1"
                else "pcm_s16le"
                if name == "prores"
                else "aac"
            )
            try:
                if any(channel.name == "NONE" for channel in av.AudioLayout(layout).channels):
                    raise ValueError(
                        "discrete channels require a PCM container without speaker labels"
                    )
                candidate_audio = cast(Any, av.CodecContext.create(audio_encoder, "w"))
                candidate_audio.sample_rate = 48_000 if kind == "webm" else frame.sample_rate
                candidate_audio.layout = layout
                audio_formats = av.Codec(audio_encoder, "w").audio_formats
                assert audio_formats
                candidate_audio.format = audio_formats[0]
                candidate_audio.open()
            except (ValueError, FFmpegError):
                format_diagnostic(
                    {
                        "container": kind,
                        "codec": name,
                        "audio_codec": audio_encoder,
                        "layout": layout,
                    },
                    {
                        "container": "mkv",
                        "codec": name if kind == "mkv" else "ffv1",
                        "audio_codec": "pcm_f32le",
                        "layout": layout,
                    },
                    on_diagnostic,
                )
                if kind != "mkv":
                    return save_video_stream(
                        value,
                        destination,
                        container="mkv",
                        codec="ffv1",
                        metadata=metadata,
                        audio_layout=audio_layout,
                        trim_to_audio=trim_to_audio,
                        on_diagnostic=on_diagnostic,
                    )
                audio_encoder = "pcm_f32le"
            audio = output.add_stream(
                audio_encoder,
                rate=48_000 if kind == "webm" else frame.sample_rate,
            )
            audio.layout = layout
            audio.codec_context.thread_count = 1
            streams.append(audio)
            iterators.append(itertools.chain([first], iterator))
            resamplers.append(
                av.AudioResampler(
                    format=audio.codec_context.format.name,
                    layout=cast(Any, audio.layout).name,
                    rate=audio.rate,
                )
            )
        queue: list[tuple[Fraction, int, object]] = []
        for index, iterator in enumerate(iterators):
            item = next(iterator, None)
            if item is not None:
                heapq.heappush(queue, (item[0], index, item[1]))
        video_frames = 0
        stop_at: Fraction | None = None
        while queue:
            t, index, item = heapq.heappop(queue)
            if stop_at is not None and t >= stop_at:
                continue
            if index == 0:
                video_frames += 1
            frame = (
                _encoder_frame(cast(np.ndarray, item), depth)
                if isinstance(item, np.ndarray)
                else cast(Any, item)
            )
            if index == 0:
                coverage = _alpha_pixels(frame) if alpha and pixel.is_planar else None
                frame = frame.reformat(
                    format=pixel_format,
                    src_colorspace=frame.colorspace if frame.colorspace not in (0, 2) else 5,
                    dst_colorspace=destination_matrix,
                    src_color_range=2 if frame.format.is_rgb else frame.color_range or 1,
                    dst_color_range=destination_range,
                )
                if coverage is not None:
                    component = next(c for c in pixel.components if c.is_alpha)
                    plane = frame.planes[component.plane]
                    dtype = np.dtype("u1" if component.bits <= 8 else "<u2")
                    samples = np.zeros((plane.height, plane.line_size // dtype.itemsize), dtype)
                    samples[:, : frame.width] = np.rint(coverage * (2**component.bits - 1))
                    plane.update(samples.tobytes())
            frame.time_base = (
                Fraction(1, 1_000_000) if index == 0 else Fraction(1, frame.sample_rate)
            )
            frame.pts = round(t / frame.time_base)
            frames = [frame] if index == 0 else resamplers[index].resample(frame)
            for converted in frames:
                for packet in streams[index].encode(converted):
                    output.mux(packet)
            following = next(iterators[index], None)
            if following is not None:
                heapq.heappush(queue, (following[0], index, following[1]))
            elif index and trim_to_audio:
                audio_end = t + Fraction(frame.samples, frame.sample_rate)
                stop_at = min(stop_at, audio_end) if stop_at is not None else audio_end
        if not video_frames:
            raise ValueError("video selection contains no frames")
        for index, stream in enumerate(streams):
            if index:
                for frame in resamplers[index].resample(None):
                    for packet in stream.encode(frame):
                        output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    return "." + kind, mime
