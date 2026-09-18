"""Family-neutral latent-mask timeline declarations and range parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .timeline_guides import CodecTemporalMapping


class LatentMaskError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LatentMaskMapping:
    """Map one role's latent grid to content units on a timed axis."""

    role: str
    temporal: CodecTemporalMapping
    content_rate_hz: float
    spatial_downscale: int | None = None

    def __post_init__(self) -> None:
        if type(self.role) is not str or not self.role:
            raise LatentMaskError("latent mask mapping role must be a nonempty string")
        if type(self.temporal) is not CodecTemporalMapping:
            raise TypeError("latent mask temporal mapping must be an exact CodecTemporalMapping")
        if (
            type(self.content_rate_hz) not in (int, float)
            or not math.isfinite(self.content_rate_hz)
            or self.content_rate_hz <= 0.0
        ):
            raise LatentMaskError("latent mask content rate must be finite and positive")
        if self.spatial_downscale is not None and (
            type(self.spatial_downscale) is not int or self.spatial_downscale < 1
        ):
            raise LatentMaskError("latent mask spatial downscale must be a positive integer")

    def content_ranges(self, latent_extent: int) -> tuple[tuple[int, int], ...]:
        self.temporal.content_extent(latent_extent)
        period = self.temporal.content_frames_per_latent
        cursor = 0
        ranges: list[tuple[int, int]] = []
        for index in range(latent_extent):
            stop = cursor + period[index % len(period)]
            ranges.append((cursor, stop))
            cursor = stop
        return tuple(ranges)

    def duration_seconds(self, latent_extent: int) -> float:
        return self.temporal.content_extent(latent_extent) / self.content_rate_hz


@runtime_checkable
class LatentMaskCodecRuntime(Protocol):
    """Codec-declared content geometry for authoring one role's masks."""

    @property
    def latent_mask_mapping(self) -> LatentMaskMapping: ...


def _range_items(value: str) -> tuple[str, ...]:
    if type(value) is not str:
        raise TypeError("ranges must be an exact string")
    return tuple(item.strip() for item in value.replace("\n", ",").split(",") if item.strip())


def _slice_entry(value: str, *, allow_end: bool) -> int | None:
    if not value:
        return None
    if value == "end":
        if allow_end:
            return None
        raise LatentMaskError("'end' is allowed only as a slice stop")
    try:
        return int(value)
    except ValueError as error:
        raise LatentMaskError(f"invalid frame range value: {value!r}") from error


def parse_frame_ranges(value: str, frame_count: int) -> tuple[int, ...]:
    """Resolve comma/newline-separated indices and Python slices."""

    if type(frame_count) is not int or frame_count < 1:
        raise LatentMaskError("frame count must be an exact positive integer")
    selected: set[int] = set()
    for item in _range_items(value):
        if ":" not in item:
            try:
                index = int(item)
            except ValueError as error:
                raise LatentMaskError(f"invalid frame index: {item!r}") from error
            resolved = index if index >= 0 else frame_count + index
            if not 0 <= resolved < frame_count:
                raise LatentMaskError(
                    f"frame index {index} is outside the mask's {frame_count} frames"
                )
            selected.add(resolved)
            continue
        entries = item.split(":")
        if len(entries) not in (2, 3):
            raise LatentMaskError(f"invalid frame slice: {item!r}")
        start = _slice_entry(entries[0], allow_end=False)
        stop = _slice_entry(entries[1], allow_end=True)
        step = None if len(entries) == 2 else _slice_entry(entries[2], allow_end=False)
        if step == 0:
            raise LatentMaskError("frame slice step must not be zero")
        selected.update(range(*slice(start, stop, step).indices(frame_count)))
    return tuple(sorted(selected))


def parse_time_ranges(value: str, duration_seconds: float) -> tuple[tuple[float, float], ...]:
    """Resolve half-open second ranges with optional end-relative bounds."""

    if (
        type(duration_seconds) not in (int, float)
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0.0
    ):
        raise LatentMaskError("duration must be finite and positive")
    duration = float(duration_seconds)
    ranges: list[tuple[float, float]] = []
    for item in _range_items(value):
        entries = item.split(":")
        if len(entries) != 2:
            raise LatentMaskError(f"time range must be start:stop, got {item!r}")

        def resolve(entry: str, *, stop: bool) -> float:
            if not entry or (stop and entry == "end"):
                return duration if stop else 0.0
            if entry == "end":
                raise LatentMaskError("'end' is allowed only as a time-range stop")
            try:
                parsed = float(entry)
            except ValueError as error:
                raise LatentMaskError(f"invalid time range value: {entry!r}") from error
            if not math.isfinite(parsed):
                raise LatentMaskError("time range values must be finite")
            return duration + parsed if parsed < 0.0 else parsed

        start = min(duration, max(0.0, resolve(entries[0], stop=False)))
        stop = min(duration, max(0.0, resolve(entries[1], stop=True)))
        if stop <= start:
            raise LatentMaskError(f"time range {item!r} selects no time")
        ranges.append((start, stop))
    return tuple(ranges)


__all__ = [
    "LatentMaskCodecRuntime",
    "LatentMaskError",
    "LatentMaskMapping",
    "parse_frame_ranges",
    "parse_time_ranges",
]
