"""Timeline adapters for admitted sources and the shared CPU operation kernels."""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Generator, Mapping
from contextlib import closing
from fractions import Fraction
from typing import Any, cast

import numpy as np
from dinkster_values import (
    MEBIBYTE,
    bind_video_sources,
    edit_video,
    media_semantics,
    video_from_source,
)
from dinkster_values.curve import coerce_curve
from dinkster_values.timeline_video import TimelineVideo
from dinkster_values.video_document import TimelineError, json_bytes, source_video
from dinkster_values.video_edits import seconds
from PIL import Image

from .image_math import blend as blend_samples
from .image_math import (
    composite_samples,
    masked_transition,
    overlay_rgba,
    parse_color,
    source_over,
    text_coverage,
)
from .timeline import Pixels, compile_timeline, iter_timeline_audio, iter_timeline_frames


def _straight_pixels(frame: Pixels) -> Pixels:
    if media_semantics(frame).get("alpha") != "premultiplied":
        return frame
    result = np.array(frame, copy=True)
    alpha = result[..., -1:]
    result[..., :-1] = np.divide(
        result[..., :-1], alpha, out=np.zeros_like(result[..., :-1]), where=alpha != 0
    )
    return np.ascontiguousarray(result, dtype=np.float32)


class SourceMedia:
    def __init__(self, factory: Callable[[Mapping[str, object]], Any] | None) -> None:
        self.factory = factory
        self.values: dict[str, object] = {}

    def _asset(self, wire: Mapping[str, object]) -> Any:
        if self.factory is None:
            raise TimelineError(
                "unbound_source", str(wire["digest"]), "host asset resolver required"
            )
        return self.factory(wire)

    def video(self, reference: Mapping[str, Any]) -> object:
        key = json_bytes(reference).decode()
        if key not in self.values:
            if "video" in reference:
                self.values[key] = bind_video_sources(source_video(reference), self._asset)
            else:
                self.values[key] = video_from_source(self._asset(reference["asset"]))
        return self.values[key]

    def frames(
        self,
        reference: Mapping[str, Any],
        start: Fraction,
        end: Fraction,
        *,
        crop: Mapping[str, Any] | None = None,
    ) -> Generator[tuple[Fraction, Pixels], None, None]:
        from .runtime import iter_video_pixels

        if reference["type"] == "dinkster.layers":
            from dinkster_image_document import decode_document, render_document

            with self._asset(reference["asset"]).open() as handle:
                parsed = decode_document(handle.read(MEBIBYTE))
            resources = {ref["digest"]: ref for ref in reference.get("resources", [])}
            for dep in parsed.dependencies:
                if (
                    dep["digest"] not in resources
                    or resources[dep["digest"]]["size"] != dep["byteSize"]
                ):
                    raise TimelineError(
                        "unbound_source", "$", "layer resources must be explicitly declared"
                    )

            def read_resource(digest: str) -> bytes:
                with self._asset(resources[digest]).open() as handle:
                    return handle.read(resources[digest]["size"] + 1)

            result = render_document(parsed, read_resource)
            with Image.open(io.BytesIO(result.png)) as image:
                yield start, np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
            return
        if reference["type"] == "dinkster.image":
            with self._asset(reference["asset"]).open() as handle, Image.open(handle) as image:
                if image.width * image.height > 8192 * 8192:
                    raise TimelineError("document_limit", "$", "image exceeds render budget")
                pixels = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
            yield start, pixels
            return
        if reference["type"] != "comfy.VIDEO":
            raise TimelineError(
                "unsupported_source", "$", "source needs the canonical layer adapter"
            )
        value = edit_video(
            self.video(reference),
            {"trim": {"start_time": start, "duration": end - start}, "strict_duration": False},
        )
        if crop is not None:
            value = edit_video(
                value, {"crop": {key: crop.get(key, 0) for key in ("x", "y", "width", "height")}}
            )
        probe = cast(dict[str, Any], value["probe"])
        if probe["color_space"] not in ("sRGB", "unknown"):
            raise TimelineError("color_space_mismatch", "$", "timeline CPU profile requires sRGB")
        with closing(iter_video_pixels(value)) as frames:
            for time, frame in frames:
                yield start + time, _straight_pixels(frame)

    def audio_window(
        self, reference: Mapping[str, Any], start: Fraction, count: int, rate: int
    ) -> Pixels:
        from .runtime import iter_video_audio

        if reference["type"] == "comfy.AUDIO":
            import importlib

            audio: Any = importlib.import_module("dinkster_values.audio_codec")
            if not hasattr(audio, "audio_window"):
                raise TimelineError(
                    "unsupported_source", "$", "canonical bounded AUDIO runtime required"
                )
            key = "audio:" + json_bytes(reference).decode()
            if key not in self.values:
                source = audio.audio_from_source(self._asset(reference["asset"]))
                if audio.effective_audio_facts(source)["sample_rate"] != rate:
                    source = audio.append_audio_edit(source, {"resample": rate})
                self.values[key] = source
            window = audio.audio_window(self.values[key], round(start * rate), count, batch_index=0)
            return np.asarray(window["waveform"][0], dtype=np.float32)
        if reference["type"] != "comfy.VIDEO":
            raise TimelineError("unsupported_source", "$", "source needs canonical AUDIO adapter")
        value = edit_video(
            self.video(reference),
            {
                "trim": {"start_time": start, "duration": Fraction(count, rate)},
                "strict_duration": False,
            },
        )
        output: Pixels | None = None
        with closing(iter_video_audio(value)) as frames:
            for time, frame in frames:
                if frame.sample_rate != rate:
                    raise TimelineError("audio_rate_mismatch", "$", "explicit resampling required")
                array = frame.to_ndarray()
                if output is None:
                    output = np.zeros((array.shape[0], count), dtype=np.float32)
                first = max(0, round(time * rate))
                last = min(count, first + array.shape[1])
                output[:, first:last] = array[:, : last - first]
        if output is None:
            raise TimelineError("source_range_unavailable", "$", "no audio in requested window")
        return output


class CPUKernels:
    def composite(self, destination: Pixels, source: Pixels, blend: str, opacity: float) -> Pixels:
        if not math.isfinite(opacity) or not 0 <= opacity <= 1:
            raise TimelineError("invalid_opacity", "$", "opacity must be in [0,1]")
        if source.shape[2] == 4 or destination.shape[2] == 4:
            sa = (
                source[..., 3:]
                if source.shape[2] == 4
                else np.ones((*source.shape[:2], 1), np.float32)
            )
            da = (
                destination[..., 3:]
                if destination.shape[2] == 4
                else np.ones((*destination.shape[:2], 1), np.float32)
            )
            blended = blend_samples(destination[..., :3], source[..., :3], blend)
            if blend != "normal":
                blended = composite_samples(source[..., :3], blended, da)
            rgb, alpha = source_over(
                blended,
                sa * opacity,
                destination[..., :3],
                da,
            )
            return np.ascontiguousarray(
                np.concatenate((rgb, alpha), axis=2) if destination.shape[2] == 4 else rgb,
                dtype=np.float32,
            )
        return np.ascontiguousarray(
            np.clip(
                composite_samples(destination, blend_samples(destination, source, blend), opacity),
                0,
                1,
            ),
            dtype=np.float32,
        )

    def transition(self, first: Pixels, second: Pixels, progress: float) -> Pixels:
        if first.shape[2] != second.shape[2]:
            if first.shape[2] == 3:
                first = np.concatenate((first, np.ones((*first.shape[:2], 1), np.float32)), axis=2)
            if second.shape[2] == 3:
                second = np.concatenate(
                    (second, np.ones((*second.shape[:2], 1), np.float32)), axis=2
                )
        return masked_transition(first, second, np.full(first.shape[:2], progress, np.float32))

    def curve(self, value: Mapping[str, Any], position: float) -> float:
        return coerce_curve(value).evaluate(position)

    def effect(self, node_type: str, image: Pixels, parameters: Mapping[str, Any]) -> Pixels:
        if node_type != "dinkster.image.draw_text":
            raise TimelineError("unsupported_effect", "$", node_type)
        allowed = {"text", "x", "y", "font_size", "color", "opacity", "line_spacing"}
        if set(parameters) - allowed:
            raise TimelineError("unsupported_effect", "$", "unknown draw_text parameters")
        p: dict[str, Any] = {
            "x": 0,
            "y": 0,
            "font_size": 32,
            "color": "#ffffff",
            "opacity": 1,
            "line_spacing": 4,
            **parameters,
        }
        opacity = float(seconds(p["opacity"], "opacity"))
        if not 0 <= opacity <= 1:
            raise TimelineError("invalid_opacity", "$", "opacity must be in [0,1]")
        rgba = parse_color(p["color"])
        coverage = text_coverage(
            image.shape[1],
            image.shape[0],
            p["text"],
            p["x"],
            p["y"],
            p["font_size"],
            p["line_spacing"],
        )
        return overlay_rgba(image[None], coverage * opacity * rgba[3], rgba)[0]

    def mix_audio(self, windows: list[Pixels], gains: list[Pixels]) -> Pixels:
        if any(w.shape != windows[0].shape for w in windows):
            raise TimelineError(
                "audio_layout_mismatch", "$", "audio tracks must share a channel layout"
            )
        result = np.zeros_like(windows[0])
        for window, gain in zip(windows, gains, strict=True):
            result += window * gain
        return result


def timeline_streams(
    value: TimelineVideo,
) -> tuple[dict[str, object], Generator[Any, None, None], Generator[Any, None, None]]:
    """Admit output facts and return iterators for the existing container sink."""
    import av

    obj, placements, total = compile_timeline(value["timeline"])
    media, kernels = SourceMedia(value.factory), CPUKernels()
    audio = any(p.tracks and p.tracks[-1]["kind"] == "Audio" for p in placements)
    rate = seconds(obj["settings"]["rate"], "rate")
    facts: dict[str, object] = {
        "width": obj["settings"]["width"],
        "height": obj["settings"]["height"],
        "fps": rate,
        "duration": total,
        "bit_depth": 8,
        "alpha": False,
        "primaries": 1,
        "transfer": 13,
        "matrix": 1,
        "range": 1,
        "audio": audio,
    }

    def audio_frames() -> Generator[Any, None, None]:
        for time, samples in iter_timeline_audio(obj, media, kernels):
            layout = {1: "mono", 2: "stereo"}.get(samples.shape[0])
            if layout is None:
                raise TimelineError("audio_layout_mismatch", "$", "channel layout required")
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(samples), format="fltp", layout=layout
            )
            frame.sample_rate = 48000
            yield time, frame

    return facts, iter_timeline_frames(obj, media, kernels), audio_frames()
