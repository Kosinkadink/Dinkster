"""Bounded timeline evaluation with explicitly supplied shared media kernels."""

from __future__ import annotations

import math
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack, closing
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol, cast

import numpy as np
from dinkster_values import (
    TIMELINE_ACTIVE_FRAME_LIMIT_BYTES,
    TIMELINE_AUDIO_WINDOW_LIMIT_BYTES,
    coerce_video,
    edit_video,
)
from dinkster_values.video_document import (
    TimelineError,
    document,
    effective_document,
    extension,
    range_seconds,
    schema,
    source_origin,
    source_video,
    time_seconds,
    walk,
)
from dinkster_values.video_edits import effective_video_facts, mapping, seconds
from numpy.typing import NDArray

from .document import clip_duration, playback_speed

Pixels = NDArray[np.float32]


class TimelineMedia(Protocol):
    """Host-owned admitted media access; URLs in the OTIO tree never reach this API."""

    def video(self, reference: Mapping[str, Any]) -> object: ...

    def frames(
        self,
        reference: Mapping[str, Any],
        start: Fraction,
        end: Fraction,
        *,
        crop: Mapping[str, Any] | None = None,
    ) -> Generator[tuple[Fraction, Pixels], None, None]: ...

    def audio_window(
        self, reference: Mapping[str, Any], start: Fraction, count: int, rate: int
    ) -> Pixels: ...


class TimelineKernels(Protocol):
    """The same CPU kernels used by graph image, layer, audio, and CURVE nodes."""

    def composite(
        self, destination: Pixels, source: Pixels, blend: str, opacity: float
    ) -> Pixels: ...

    def transition(self, first: Pixels, second: Pixels, progress: float) -> Pixels: ...

    def effect(self, node_type: str, image: Pixels, parameters: Mapping[str, Any]) -> Pixels: ...

    def curve(self, value: Mapping[str, Any], position: float) -> float: ...

    def mix_audio(self, windows: list[Pixels], gains: list[Pixels]) -> Pixels: ...


def scalar(value: Any, time: Fraction, kernels: TimelineKernels) -> float:
    if isinstance(value, dict):
        value = cast(dict[str, Any], value)
        if value.get("type") != "dinkster.curve":
            raise TimelineError("invalid_curve", "$", "expected type dinkster.curve")
        return kernels.curve(value["value"], float(time))
    return float(seconds(value, "scalar"))


def parameters(
    value: Mapping[str, Any], time: Fraction, kernels: TimelineKernels
) -> dict[str, Any]:
    return {
        key: scalar(item, time, kernels)
        if isinstance(item, dict) and cast(dict[str, Any], item).get("type") == "dinkster.curve"
        else item
        for key, item in value.items()
    }


def duration(item: dict[str, Any]) -> Fraction:
    kind = schema(item)
    if item.get("source_range") is not None or kind not in ("Stack", "Track"):
        return clip_duration(item)
    values = [duration(c) for c in item["children"]]
    return max(values, default=Fraction(0)) if kind == "Stack" else sum(values, Fraction(0))


def source_for(obj: Mapping[str, Any], item: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    source = extension(item).get("source")
    if not isinstance(source, str) or source not in obj["sources"]:
        raise TimelineError("unbound_source", path, "bind a declared asset before rendering")
    return obj["sources"][source]


def clip_video(value: object, item: Mapping[str, Any]) -> dict[str, object]:
    """Use VIDEO's own edit arithmetic, including crop sentinel and strict trim rules."""
    video = coerce_video(value)
    ext = extension(item)
    if "video_edit" not in ext and item.get("source_range") is not None:
        start, length = range_seconds(item["source_range"])
        start -= source_origin(item)
        if length <= 0:
            raise TimelineError("invalid_edit", "source_range", "empty clip")
        facts = effective_video_facts(video)
        if start != 0 or length != facts.get("duration"):
            video = edit_video(
                video, {"trim": {"start_time": start, "duration": length}, "strict_duration": True}
            )
    widget = ext.get("video_edit", {})
    for key in ("trim", "crop"):
        if widget.get(key) is not None:
            section = mapping(widget[key], f"VIDEO_EDIT {key}")
            names = ("start_time", "duration") if key == "trim" else ("x", "y", "width", "height")
            edit = {key: {name: section.get(name, 0) for name in names}}
            if key == "trim":
                edit["strict_duration"] = ext.get("strict_duration", False)
            video = edit_video(video, edit)
    return video


def single_clip_video(value: object, media: TimelineMedia) -> dict[str, object] | None:
    obj = document(value)
    stack = obj["timeline"]["tracks"]
    tracks = stack["children"]
    if len(tracks) != 1 or schema(tracks[0]) != "Track" or tracks[0]["kind"] != "Video":
        return None
    track = tracks[0]
    children = track["children"]
    if len(children) != 1 or schema(children[0]) != "Clip":
        return None
    item = children[0]
    for container in (stack, track):
        ext = extension(container)
        if (
            container.get("enabled", True) is False
            or container.get("source_range") is not None
            or container.get("effects")
            or ext
        ):
            return None
    ext = extension(item)
    if (
        item.get("enabled", True) is False
        or item.get("effects")
        or ext.get("effects")
        or set(ext) - {"source", "video_edit", "strict_duration", "proxy"}
    ):
        return None
    reference = source_for(obj, item, "tracks.children[0].children[0]")
    if reference["type"] != "comfy.VIDEO":
        return None
    return clip_video(media.video(reference), item)


@dataclass(frozen=True)
class Placement:
    item: dict[str, Any]
    path: str
    start: Fraction
    end: Fraction
    source_start: Fraction
    speed: Fraction
    clip_origin: Fraction
    tracks: tuple[dict[str, Any], ...]
    fade_in: tuple[Fraction, Fraction] | None = None
    fade_out: tuple[Fraction, Fraction] | None = None

    def source_time(self, time: Fraction) -> Fraction:
        return self.source_start + (time - self.clip_origin) * self.speed


def _speed(item: dict[str, Any], path: str) -> Fraction:
    for effect in item.get("effects", []):
        kind = schema(effect)
        if kind not in ("LinearTimeWarp", "FreezeFrame"):
            raise TimelineError("unsupported_effect", path, str(effect.get("effect_name", kind)))
    return playback_speed(item)


def compile_timeline(value: object) -> tuple[dict[str, Any], list[Placement], Fraction]:
    obj = effective_document(value)
    placements: list[Placement] = []
    stack = obj["timeline"]["tracks"]
    total = duration(stack)

    def visit(
        item: dict[str, Any],
        path: str,
        offset: Fraction,
        window: tuple[Fraction, Fraction],
        tracks: tuple[dict[str, Any], ...],
        fade_in: tuple[Fraction, Fraction] | None = None,
        fade_out: tuple[Fraction, Fraction] | None = None,
    ) -> None:
        kind = schema(item)
        if item.get("enabled", True) is False or kind in ("Gap", "Transition"):
            return
        own_duration = duration(item)
        first, last = max(offset, window[0]), min(offset + own_duration, window[1])
        if first >= last:
            return
        if kind == "Clip":
            source_for(obj, item, path)
            source_start = (
                range_seconds(item["source_range"])[0] - source_origin(item)
                if item.get("source_range")
                else 0
            )
            placements.append(
                Placement(
                    item,
                    path,
                    first,
                    last,
                    Fraction(source_start),
                    _speed(item, path),
                    offset,
                    tracks,
                    fade_in,
                    fade_out,
                )
            )
            return
        if item.get("effects") or extension(item).get("effects"):
            raise TimelineError(
                "unsupported_effect", path, "composition time/effects need lowering"
            )
        if kind == "Track":
            tracks = (*tracks, item)
        start = range_seconds(item["source_range"])[0] if item.get("source_range") else Fraction(0)
        cursor = offset - start
        children = item["children"]
        for index, child in enumerate(children):
            child_path = f"{path}.children[{index}]"
            if schema(child) == "Transition":
                if (
                    kind != "Track"
                    or index == 0
                    or index + 1 == len(children)
                    or any(schema(c) != "Clip" for c in (children[index - 1], children[index + 1]))
                ):
                    raise TimelineError("invalid_transition", child_path, "requires adjacent clips")
                if child.get("transition_type") != "SMPTE_Dissolve":
                    raise TimelineError("unsupported_transition", child_path, "only SMPTE_Dissolve")
                continue
            if child.get("enabled", True) is False:
                if kind == "Track":
                    cursor += duration(child)
                continue
            incoming = outgoing = None
            before = after = Fraction(0)
            if kind == "Track":
                if index and schema(children[index - 1]) == "Transition":
                    t = children[index - 1]
                    before, tail = time_seconds(t["in_offset"]), time_seconds(t["out_offset"])
                    incoming = (cursor - before, cursor + tail)
                if index + 1 < len(children) and schema(children[index + 1]) == "Transition":
                    t = children[index + 1]
                    head, after = time_seconds(t["in_offset"]), time_seconds(t["out_offset"])
                    cut = cursor + duration(child)
                    outgoing = (cut - head, cut + after)
            if before or after:
                if schema(child) != "Clip":
                    raise TimelineError("invalid_transition", child_path, "transition on non-clip")
                source_for(obj, child, child_path)
                source_start = range_seconds(child["source_range"])[0] - source_origin(child)
                speed = _speed(child, child_path)
                placements.append(
                    Placement(
                        child,
                        child_path,
                        max(cursor - before, first),
                        min(cursor + duration(child) + after, last),
                        source_start,
                        speed,
                        cursor,
                        tracks,
                        incoming,
                        outgoing,
                    )
                )
            else:
                visit(child, child_path, cursor, (first, last), tracks, incoming, outgoing)
            if kind == "Track":
                cursor += duration(child)

    visit(stack, "tracks", Fraction(0), (Fraction(0), total), ())
    for placement in placements:
        rate = seconds(obj["settings"]["rate"], "rate")
        first_sample = placement.source_time(Fraction(math.ceil(placement.start * rate)) / rate)
        last_sample = placement.source_time(Fraction(math.ceil(placement.end * rate) - 1) / rate)
        if min(first_sample, last_sample) < 0:
            raise TimelineError(
                "source_range_unavailable", placement.path, "missing transition handle"
            )
        reference = source_for(obj, placement.item, placement.path)
        if "video" in reference:
            length = effective_video_facts(source_video(reference))["duration"]
            if isinstance(length, Fraction):
                if max(first_sample, last_sample) >= length:
                    raise TimelineError(
                        "source_range_unavailable", placement.path, "selection exceeds bound media"
                    )
        if (
            placement.fade_in
            and placement.fade_out
            and placement.fade_in[1] > placement.fade_out[0]
        ):
            raise TimelineError("invalid_transition", placement.path, "overlapping transitions")
    return obj, placements, total


class _FrameReader:
    def __init__(self, iterator: Iterator[tuple[Fraction, Pixels]], end: Fraction | None) -> None:
        self.iterator = iterator
        self.end = end
        self.current = next(iterator, None)
        self.following = next(iterator, None)
        self.last_time: Fraction | None = None

    def at(self, time: Fraction) -> Pixels:
        if self.end is not None and time >= self.end:
            raise TimelineError(
                "source_range_unavailable", "$", "source time exceeds media duration"
            )
        if self.last_time is not None and time < self.last_time:
            raise TimelineError("unsupported_time_effect", "$", "reverse requires seekable frames")
        self.last_time = time
        while self.following is not None and self.following[0] <= time:
            self.current, self.following = self.following, next(self.iterator, None)
        if self.current is None:
            raise TimelineError("source_range_unavailable", "$", "no frame at source time")
        return self.current[1]


def _gain(placement: Placement, time: Fraction) -> float:
    gain = 1.0
    for interval, incoming in ((placement.fade_in, True), (placement.fade_out, False)):
        if interval is not None:
            a, b = interval
            if b <= a:
                raise TimelineError("invalid_transition", placement.path, "empty dissolve")
            progress = min(1.0, max(0.0, float((time - a) / (b - a))))
            gain *= progress if incoming else 1.0 - progress
    return gain


def _compose_tree(
    item: dict[str, Any],
    path: str,
    surfaces: dict[str, tuple[Placement, Pixels]],
    time: Fraction,
    kernels: TimelineKernels,
    shape: tuple[int, int, int],
) -> Pixels | None:
    if schema(item) == "Clip":
        return surfaces[path][1] if path in surfaces else None
    result: Pixels | None = None
    consumed: set[int] = set()
    children = item.get("children", [])
    for index, child in enumerate(children):
        if index in consumed:
            continue
        child_path = f"{path}.children[{index}]"
        frame = _compose_tree(child, child_path, surfaces, time, kernels, shape)
        if frame is None:
            continue
        if index + 2 < len(children) and schema(children[index + 1]) == "Transition":
            next_path = f"{path}.children[{index + 2}]"
            if next_path in surfaces:
                other, next_frame = surfaces[next_path]
                frame = kernels.transition(frame, next_frame, _gain(other, time))
                consumed.add(index + 2)
        if result is None:
            result = np.zeros(shape, dtype=np.float32)
        ext = extension(child)
        result = kernels.composite(
            result, frame, ext.get("blend", "normal"), scalar(ext.get("opacity", 1), time, kernels)
        )
    return result


def iter_timeline_frames(
    value: object, media: TimelineMedia, kernels: TimelineKernels
) -> Generator[tuple[Fraction, Pixels], None, None]:
    obj, placements, total = compile_timeline(value)
    rate = seconds(obj["settings"]["rate"], "rate")
    width, height = obj["settings"]["width"], obj["settings"]["height"]
    visual = [p for p in placements if not p.tracks or p.tracks[-1]["kind"] == "Video"]
    readers: dict[str, tuple[_FrameReader, ExitStack]] = {}
    try:
        for index in range(math.ceil(total * rate)):
            time = Fraction(index) / rate
            active = [p for p in visual if p.start <= time < p.end]
            if width * height * 4 * 4 * (4 * len(active) + 22) > TIMELINE_ACTIVE_FRAME_LIMIT_BYTES:
                raise TimelineError(
                    "document_limit", "tracks", "active frame budget exceeds 512 MiB"
                )
            active_paths = {p.path for p in active}
            for path in list(readers):
                if path not in active_paths:
                    readers.pop(path)[1].close()
            image = np.zeros((height, width, 3), dtype=np.float32)
            surfaces: dict[str, tuple[Placement, Pixels]] = {}
            for placement in active:
                source_time = placement.source_time(time)
                if placement.speed < 0 and placement.path in readers:
                    readers.pop(placement.path)[1].close()
                if placement.path not in readers:
                    resources = ExitStack()
                    reference = source_for(obj, placement.item, placement.path)
                    start = (
                        source_time
                        if placement.speed < 0
                        else placement.source_time(placement.start)
                    )
                    end = start if placement.speed < 0 else placement.source_time(placement.end)
                    try:
                        media_end = None
                        if reference["type"] == "comfy.VIDEO":
                            duration = effective_video_facts(coerce_video(media.video(reference)))[
                                "duration"
                            ]
                            if duration is None:
                                raise TimelineError(
                                    "unknown_duration",
                                    placement.path,
                                    "VIDEO duration must be known",
                                )
                            media_end = seconds(duration, "duration")
                        iterator = media.frames(
                            reference,
                            start,
                            end + 1 / rate,
                            crop=extension(placement.item).get("video_edit", {}).get("crop"),
                        )
                        resources.enter_context(closing(iterator))
                        readers[placement.path] = (_FrameReader(iterator, media_end), resources)
                    except BaseException:
                        resources.close()
                        raise
                frame = readers[placement.path][0].at(source_time)
                if frame.shape not in ((height, width, 3), (height, width, 4)):
                    raise TimelineError(
                        "geometry_mismatch",
                        placement.path,
                        "source must match canvas; add explicit scale/layer flatten",
                    )
                ext = extension(placement.item)
                for effect in ext.get("effects", []):
                    frame = kernels.effect(
                        effect["node_type"],
                        frame,
                        parameters(effect["parameters"], time - placement.clip_origin, kernels),
                    )
                surfaces[placement.path] = placement, frame
            stack = obj["timeline"]["tracks"]
            composed = _compose_tree(stack, "tracks", surfaces, time, kernels, (height, width, 4))
            if composed is not None:
                ext = extension(stack)
                image = kernels.composite(
                    image,
                    composed,
                    ext.get("blend", "normal"),
                    scalar(ext.get("opacity", 1), time, kernels),
                )
            yield time, image
    finally:
        for _, resources in readers.values():
            resources.close()


def iter_timeline_audio(
    value: object,
    media: TimelineMedia,
    kernels: TimelineKernels,
    *,
    rate: int = 48000,
    window: int = 1024,
) -> Generator[tuple[Fraction, Pixels], None, None]:
    if not 1 <= window <= 65536 or not 1 <= rate <= 384000:
        raise TimelineError("document_limit", "$", "invalid audio rate/window")
    obj, placements, total = compile_timeline(value)
    audio = [p for p in placements if p.tracks and p.tracks[-1]["kind"] == "Audio"]
    channels: int | None = None
    for first in range(0, math.ceil(total * rate), window):
        count = min(window, math.ceil(total * rate) - first)
        time = Fraction(first, rate)
        windows: list[Pixels] = []
        gains: list[Pixels] = []
        allocated = 0
        for placement in audio:
            left = max(first, math.ceil(placement.start * rate))
            right = min(first + count, math.ceil(placement.end * rate))
            if left >= right:
                continue
            if placement.speed != 1:
                raise TimelineError(
                    "unsupported_time_effect",
                    placement.path,
                    "audio retime requires a declared resampler",
                )
            reference = source_for(obj, placement.item, placement.path)
            start = placement.source_start + Fraction(left, rate) - placement.clip_origin
            samples = media.audio_window(reference, start, right - left, rate)
            if samples.ndim != 2 or samples.shape[1] != right - left:
                raise TimelineError(
                    "audio_layout_mismatch", placement.path, "expected [C,T] window"
                )
            allocated += (samples.shape[0] + 1) * count * 4
            if allocated > TIMELINE_AUDIO_WINDOW_LIMIT_BYTES:
                raise TimelineError("document_limit", "$", "active audio windows exceed 128 MiB")
            padded = np.zeros((samples.shape[0], count), dtype=np.float32)
            padded[:, left - first : right - first] = samples
            gain = np.zeros((1, count), dtype=np.float32)
            for index in range(left, right):
                position = Fraction(index, rate)
                level = _gain(placement, position)
                for track in placement.tracks:
                    mix = extension(track).get("audio_mix", {})
                    level *= scalar(mix.get("gain", 1), position - placement.clip_origin, kernels)
                gain[0, index - first] = level
            windows.append(padded)
            gains.append(gain)
        if windows:
            if channels is None:
                channels = int(windows[0].shape[0])
                for leading in range(0, first, window):
                    yield (
                        Fraction(leading, rate),
                        np.zeros((channels, min(window, first - leading)), np.float32),
                    )
            elif channels != windows[0].shape[0]:
                raise TimelineError(
                    "audio_layout_mismatch", "$", "channel count changes across clips"
                )
            yield time, kernels.mix_audio(windows, gains)
        elif channels is not None:
            yield time, np.zeros((channels, count), np.float32)


def diagnostics(value: object) -> list[dict[str, str]]:
    obj = document(value)
    result: list[dict[str, str]] = []
    for path, item in walk(obj["timeline"]["tracks"]):
        if schema(item) == "Clip":
            try:
                source_for(obj, item, path)
            except TimelineError as exc:
                result.append({"code": exc.code, "path": exc.path, "message": str(exc)})
        for effect in item.get("effects", []):
            if schema(effect) not in ("LinearTimeWarp", "FreezeFrame"):
                result.append(
                    {
                        "code": "unsupported_effect",
                        "path": path,
                        "message": "opaque OTIO effect preserved but not interpreted",
                    }
                )
    return result
