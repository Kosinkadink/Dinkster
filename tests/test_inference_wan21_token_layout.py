"""Wan 2.1 target-video token-layout tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from dinkster_inference.token_layout import TokenGridTransform, map_transforms
from dinkster_inference.wan21_token_layout import (
    Wan21TokenLayoutError,
    Wan21TokenLayoutPlan,
    Wan21VideoLatentGeometry,
    plan_wan21_token_layout,
)


def test_odd_latent_geometry_declares_one_row_major_target_segment() -> None:
    plan = plan_wan21_token_layout(Wan21VideoLatentGeometry(5, 7, 9))

    assert plan.patch == (1, 2, 2)
    assert plan.flatten_order == ("t", "h", "w")
    assert len(plan.layout.segments) == 1
    segment = plan.layout.segments[0]
    assert (
        segment.identity,
        segment.modality,
        segment.role,
        segment.start,
        segment.stop,
        segment.grid,
    ) == ("target-video", "video", "target", 0, 100, (5, 4, 5))
    assert plan.layout.padded_rows == 0

    assert plan.transforms == (
        TokenGridTransform(
            "wan21.video-patch-1x2x2-row-major-thw.v1",
            "video",
            "target-video",
            (5, 7, 9),
            None,
        ),
    )
    assert map_transforms(plan.layout, plan.transforms) == {"video": plan.transforms[0]}


def test_layout_is_deterministic_immutable_and_source_geometry_sensitive() -> None:
    first = plan_wan21_token_layout(Wan21VideoLatentGeometry(2, 4, 6))
    second = plan_wan21_token_layout(Wan21VideoLatentGeometry(2, 4, 6))
    changed = plan_wan21_token_layout(Wan21VideoLatentGeometry(2, 5, 6))

    assert first == second
    assert first.layout.digest == second.layout.digest
    assert first.transforms[0].digest == second.transforms[0].digest
    assert first.layout.digest != changed.layout.digest
    assert first.transforms[0].digest != changed.transforms[0].digest
    with pytest.raises(FrozenInstanceError):
        first.patch = (2, 2, 2)  # type: ignore[misc]
    with pytest.raises(TypeError, match="produced only by the family planner"):
        Wan21TokenLayoutPlan(first.layout, first.transforms)


@pytest.mark.parametrize(
    ("temporal", "height", "width", "message"),
    (
        (0, 1, 1, "temporal"),
        (1, 0, 1, "height"),
        (1, 1, 0, "width"),
        (True, 1, 1, "temporal"),
        (1, 2.0, 1, "height"),
    ),
)
def test_geometry_refuses_nonpositive_or_nonexact_dimensions(
    temporal: object, height: object, width: object, message: str
) -> None:
    with pytest.raises(Wan21TokenLayoutError, match=message):
        Wan21VideoLatentGeometry(temporal, height, width)  # type: ignore[arg-type]


def test_planner_refuses_foreign_equal_geometry() -> None:
    class ForeignGeometry:
        temporal = 1
        height = 2
        width = 2

    with pytest.raises(Wan21TokenLayoutError, match="exact Wan21VideoLatentGeometry"):
        plan_wan21_token_layout(ForeignGeometry())  # type: ignore[arg-type]
