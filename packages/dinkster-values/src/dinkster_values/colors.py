"""Color text: the forms a color field accepts, and the value a node receives."""

from __future__ import annotations

import math

_NAMES = {
    "black": "#000000",
    "blue": "#0000ff",
    "cyan": "#00ffff",
    "green": "#008000",
    "magenta": "#ff00ff",
    "red": "#ff0000",
    "transparent": "#00000000",
    "white": "#ffffff",
    "yellow": "#ffff00",
}
_FORMS = "color must be a supported name, hex RGB/RGBA, or comma-separated RGB/RGBA"


def color_channels(value: str) -> tuple[float, float, float, float]:
    """Red, green, blue and opacity from 0 to 1.

    Accepts the names above, ``#rgb``, ``#rgba``, ``#rrggbb`` and ``#rrggbbaa``
    in either case, and comma-separated RGB or RGBA where a channel above 1
    counts out of 255. Raises ValueError for anything else."""
    text = value.strip().lower()
    text = _NAMES.get(text, text)
    if text.startswith("#"):
        payload = text[1:]
        if len(payload) in (3, 4):
            payload = "".join(character * 2 for character in payload)
        if len(payload) not in (6, 8):
            raise ValueError(_FORMS)
        try:
            channels = [
                int(payload[index : index + 2], 16) / 255.0 for index in range(0, len(payload), 2)
            ]
        except ValueError as exc:
            raise ValueError(_FORMS) from exc
    else:
        try:
            channels = [float(component.strip()) for component in text.split(",")]
        except ValueError as exc:
            raise ValueError(_FORMS) from exc
        if len(channels) not in (3, 4) or not all(
            math.isfinite(component) for component in channels
        ):
            raise ValueError(_FORMS)
        channels = [component / 255.0 if component > 1.0 else component for component in channels]
        if any(component < 0.0 or component > 1.0 for component in channels):
            raise ValueError("color channels must be between 0 and 1 or between 0 and 255")
    if len(channels) == 3:
        channels.append(1.0)
    red, green, blue, opacity = channels
    return red, green, blue, opacity


def fit_color(value: str, *, alpha: bool) -> str:
    """The color a field receives: lower-case ``#rrggbbaa`` with opacity, ``#rrggbb`` without.

    Text that is not a color is returned unchanged; the node decides what it means."""
    try:
        channels = color_channels(value)
    except ValueError:
        return value
    digits = [math.floor(channel * 255 + 0.5) for channel in channels]
    return "#" + "".join(f"{digit:02x}" for digit in (digits if alpha else digits[:3]))
