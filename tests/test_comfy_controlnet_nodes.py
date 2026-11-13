"""ControlNet public schemas and zero-strength behavior without Torch."""

from __future__ import annotations

from dinkster_conditioning.controlnet import (
    ControlNetApply,
    ControlNetApplyAdvanced,
)


def test_zero_strength_returns_sources_without_accessing_control_or_image() -> None:
    positive = [["positive", {"control": object(), "strength": 0.7}]]
    negative = [["negative", {"strength": 0.2}]]
    legacy = ControlNetApply.execute(
        conditioning=positive, control_net=object(), image=object(), strength=0.0
    )
    assert legacy["CONDITIONING"] is positive
    advanced = ControlNetApplyAdvanced.execute(
        positive=positive, negative=negative, control_net=object(), image=object(), strength=0.0
    )
    assert advanced["positive"] is positive
    assert advanced["negative"] is negative
