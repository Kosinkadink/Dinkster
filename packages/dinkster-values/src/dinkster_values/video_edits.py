"""Pure VIDEO edit arithmetic in presentation time and display pixels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import cast


def seconds(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(f"{name} must be finite seconds")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite seconds")
        return Fraction(str(value))
    return Fraction(value)


def integer(value: object, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in cast("Mapping[object, object]", value)
    ):
        raise ValueError(f"{name} must be an object with string keys")
    return cast("Mapping[str, object]", value)


def trim_window(
    current_duration: Fraction | None,
    start_time: object,
    duration: object,
    strict_duration: object = False,
) -> tuple[Fraction, Fraction | None]:
    """Resolve a trim against the current clip, never against discarded frames."""
    start = seconds(start_time, "start_time")
    length = seconds(duration, "duration")
    if length < 0:
        raise ValueError("duration must be nonnegative")
    if not isinstance(strict_duration, bool):
        raise ValueError("strict_duration must be a boolean")
    if current_duration is None:
        if start < 0 or strict_duration:
            raise ValueError("end-relative or strict trim requires a known duration")
        # A bound on an unknown clip is not evidence that the clip reaches it.
        return start, None
    if start < 0:
        start = max(Fraction(0), current_duration + start)
    available = max(Fraction(0), current_duration - start)
    if strict_duration and length > available:
        raise ValueError(f"requested duration {length} exceeds available duration {available}")
    selected = min(length, available) if length else available
    if selected <= 0:
        raise ValueError("video trim contains no frames")
    return start, selected


def crop_rectangle(
    width: int, height: int, crop: Mapping[str, object]
) -> tuple[int, int, int, int]:
    """Normalize ComfyUI VIDEO_EDIT pixels after display rotation."""
    if set(crop) - {"x", "y", "width", "height"}:
        raise ValueError("crop contains unknown fields")
    x = integer(crop.get("x", 0), "crop x")
    y = integer(crop.get("y", 0), "crop y")
    w = integer(crop.get("width", 0), "crop width")
    h = integer(crop.get("height", 0), "crop height")
    full = (0, 0, width, height)
    if w <= 0 or h <= 0:
        return full
    x = min(max(x, 0), width - 1) // 2 * 2
    y = min(max(y, 0), height - 1) // 2 * 2
    w = min(w, width - x)
    h = min(h, height - y)
    if (x, y, w, h) == full:
        return full
    w = w // 2 * 2
    h = h // 2 * 2
    return (x, y, w, h) if w and h else full


def scale_geometry(
    width: int, height: int, scale: Mapping[str, object]
) -> tuple[int, int, int, int, int, int]:
    """Return resized size, centered offset, and final size for one scale."""
    if set(scale) - {"width", "height", "fit", "interpolation", "pad_color"}:
        raise ValueError("scale contains unknown fields")
    out_w = integer(scale.get("width"), "scale width", 2)
    out_h = integer(scale.get("height"), "scale height", 2)
    if out_w % 2 or out_h % 2:
        raise ValueError("scale dimensions must be even")
    fit = scale.get("fit", "stretch")
    if fit not in ("stretch", "crop", "pad"):
        raise ValueError("scale fit must be stretch, crop, or pad")
    if scale.get("interpolation", "bilinear") not in (
        "nearest",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
    ):
        raise ValueError("unsupported scale interpolation")
    color = scale.get("pad_color", [0, 0, 0, 1])
    if not isinstance(color, (list, tuple)):
        raise ValueError("pad_color requires three or four samples")
    samples = cast("list[object] | tuple[object, ...]", color)
    if len(samples) not in (3, 4):
        raise ValueError("pad_color requires three or four samples")
    for sample in samples:
        if not 0 <= seconds(sample, "pad_color sample") <= 1:
            raise ValueError("pad_color samples must be in [0, 1]")
    if fit == "stretch":
        return out_w, out_h, 0, 0, out_w, out_h
    ratios = Fraction(out_w, width), Fraction(out_h, height)
    ratio = max(ratios) if fit == "crop" else min(ratios)
    rounding = math.ceil if fit == "crop" else math.floor
    resized_w = max(2, rounding(width * ratio / 2) * 2)
    resized_h = max(2, rounding(height * ratio / 2) * 2)
    return (
        resized_w,
        resized_h,
        abs(out_w - resized_w) // 2,
        abs(out_h - resized_h) // 2,
        out_w,
        out_h,
    )


def effective_video_facts(
    video: Mapping[str, object], *, _depth: int = 0, _budget: list[int] | None = None
) -> dict[str, object]:
    """Fold the ordered edits without opening, probing, or decoding the source."""
    if "timeline" in video:
        from .timeline_video import timeline_facts

        return timeline_facts(video)
    if _depth > 16:
        raise ValueError("VIDEO nesting exceeds 16")
    budget = _budget if _budget is not None else [0, 0]
    budget[1] += 1
    if budget[1] > 64:
        raise ValueError("VIDEO tree exceeds 64 clips")
    probe = mapping(video.get("probe"), "VIDEO probe")
    width = integer(probe.get("width"), "probe width", 1)
    height = integer(probe.get("height"), "probe height", 1)
    rotation = seconds(probe.get("rotation", 0), "rotation")
    if rotation % 180:
        if rotation % 90:
            raise ValueError("VIDEO display rotation must be a multiple of 90 degrees")
        width, height = height, width
    duration = probe.get("duration")
    length = seconds(duration, "probe duration") if duration is not None else None
    fps = probe.get("fps")
    rate = seconds(fps, "probe fps") if fps is not None else None
    if (rate is not None and rate <= 0) or (length is not None and length < 0):
        raise ValueError("probe fps must be positive and duration nonnegative")
    count = probe.get("frame_count")
    if count is not None:
        integer(count, "probe frame_count", 0)
    count_kind = probe.get("frame_count_kind", "unknown")
    edits = video.get("edits")
    if not isinstance(edits, list) or len(cast("list[object]", edits)) > 256:
        raise ValueError("VIDEO edits must be a list of at most 256 operations")
    budget[0] += len(cast("list[object]", edits))
    if budget[0] > 256:
        raise ValueError("VIDEO tree exceeds 256 operations")
    for raw in cast("list[object]", edits):
        edit = mapping(raw, "VIDEO edit")
        operations = set(edit) & {"trim", "crop", "scale", "concat"}
        if len(operations) != 1:
            raise ValueError("VIDEO edit must contain exactly one operation")
        op = next(iter(operations))
        if set(edit) - ({op, "strict_duration"} if op == "trim" else {op}):
            raise ValueError("VIDEO edit contains unknown fields")
        if op == "trim":
            params = mapping(edit[op], "trim")
            if set(params) - {"start_time", "duration"}:
                raise ValueError("trim contains unknown fields")
            _, length = trim_window(
                length,
                params.get("start_time", 0),
                params.get("duration", 0),
                edit.get("strict_duration", False),
            )
            count = math.ceil(length * rate) if length is not None and rate else None
            count_kind = "estimated" if count is not None else "unknown"
        elif op == "crop":
            _, _, width, height = crop_rectangle(width, height, mapping(edit[op], "crop"))
        elif op == "scale":
            *_, width, height = scale_geometry(width, height, mapping(edit[op], "scale"))
        else:
            clips = edit[op]
            if not isinstance(clips, list) or not 1 <= len(cast("list[object]", clips)) <= 64:
                raise ValueError("concat requires between 1 and 64 clips")
            for child in cast("list[object]", clips):
                facts = effective_video_facts(
                    mapping(child, "concat clip"), _depth=_depth + 1, _budget=budget
                )
                if (facts["width"], facts["height"]) != (width, height):
                    raise ValueError("concat clips must have matching effective dimensions")
                child_length = facts["duration"]
                length = (
                    length + cast("Fraction", child_length)
                    if length is not None and child_length is not None
                    else None
                )
                child_count = facts["frame_count"]
                count = (
                    cast("int", count) + cast("int", child_count)
                    if count is not None and child_count is not None
                    else None
                )
                if count is None:
                    count_kind = "unknown"
                elif facts["frame_count_kind"] != "header" or count_kind != "header":
                    count_kind = "estimated"
                if facts["fps"] != rate:
                    rate = None
    return {
        "width": width,
        "height": height,
        "duration": length,
        "fps": rate,
        "frame_count": count,
        "frame_count_kind": count_kind,
    }
