"""Formula facts from source bytes, without names, previews or run results."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import BinaryIO, cast

from .audio_lazy import probe_audio_handle
from .video_probe import probe_video


def decoded_file_facts(
    handle: BinaryIO, media_type: str
) -> tuple[dict[str, int | float], str | None]:
    """Read applicable facts from a borrowed, verified source descriptor."""
    handle.seek(0)
    if media_type.startswith("image/"):
        from PIL import Image

        with Image.open(handle) as image:
            width, height = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
            count = getattr(image, "n_frames", 1)
            first = 1 if image.info.get("default_image") else 0
            duration = 0.0
            timed = True
            for index in range(first, count):
                image.seek(index)
                image.load()
                delay = image.info.get("duration")
                if not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay <= 0:
                    timed = False
                else:
                    duration += delay / 1000
            frames = count - first
            facts: dict[str, int | float] = {
                "width": width,
                "height": height,
                "frames": frames,
            }
            if frames > 1 and timed and duration > 0:
                facts.update(duration=duration, fps=frames / duration)
            return facts, None
    if media_type.startswith("video/"):
        source = probe_video(handle, exact_frames=True)
        width, height = cast(int, source["width"]), cast(int, source["height"])
        rotation = cast(float, source["rotation"])
        facts = {"frames": cast(int, source["frame_count"])}
        reason = None
        if rotation % 90:
            reason = (
                f"This video is stored turned {rotation:g} degrees. Only quarter turns are "
                "supported, so its width and height are unknown."
            )
        else:
            if rotation % 180:
                width, height = height, width
            facts.update(width=width, height=height)
        for name in ("fps", "duration"):
            value = source[name]
            if value is not None:
                facts[name] = float(cast(float | Fraction, value))
        return facts, reason
    if media_type.startswith("audio/"):
        duration = probe_audio_handle(handle).get("duration")
        return ({} if duration is None else {"duration": cast(float, duration)}), None
    return {}, None
