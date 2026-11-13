"""Public color and resolution contracts without a Torch dependency."""

from __future__ import annotations

import pytest
from dinkster_utilities.color_resolution import ColorToRGBInt, ResolutionSelector


def test_color_values_preserve_case_alpha_and_reference_rejection() -> None:
    assert ColorToRGBInt.execute(color="#aB12CD80") == {
        "rgb_int": 11211469,
        "hex": "#aB12CD",
        "alpha": 128 / 255,
    }
    assert ColorToRGBInt.execute(color="#01020f") == {
        "rgb_int": 66063,
        "hex": "#01020f",
        "alpha": 1.0,
    }
    assert ColorToRGBInt.execute(color="#12_345") == {
        "rgb_int": 16777215,
        "hex": "#12_345",
        "alpha": 1.0,
    }
    assert ColorToRGBInt.execute(color="#00000000")["alpha"] == 0.0
    for invalid in ("", "red", "#abc", "001122", "#12345g", "#1122334455"):
        with pytest.raises(ValueError, match="Color must be in format #RRGGBB or #RRGGBBAA"):
            ColorToRGBInt.execute(color=invalid)


def test_resolution_rounds_to_requested_multiple_and_ignores_preview_payload() -> None:
    assert ResolutionSelector.execute(aspect_ratio="1:1 (Square)", megapixels=1, multiple=20) == {
        "width": 1020,
        "height": 1020,
    }
    assert ResolutionSelector.execute(
        aspect_ratio="1:1 (Square)", megapixels=0.1, multiple=8, preview={"width": 1, "height": 2}
    ) == {"width": 320, "height": 320}
    assert ResolutionSelector.execute(
        aspect_ratio="21:9 (Ultrawide)", megapixels=1, multiple=8
    ) == {"width": 1568, "height": 672}
    with pytest.raises(KeyError):
        ResolutionSelector.execute(aspect_ratio="unrecognized", megapixels=1, multiple=8)
