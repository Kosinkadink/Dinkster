"""Declared control gain schedule and exact realization tests."""

from __future__ import annotations

import hashlib

import pytest
from dinkster_inference import (
    MONOTONE_NEAREST_ASSIGNMENT,
    ConstantGainCurve,
    ContributionGain,
    DirectGainTableCurve,
    ExecutedStepAnchors,
    ExecutedTimeline,
    GainEndpoint,
    GainInterpolation,
    GainKeyframe,
    GainOmission,
    GainRefusal,
    GainRefusalCode,
    KeyframeGainCurve,
    KeyframeHoldPolicy,
    RealizedGainTable,
    TimelineCoordinate,
    contribution_gain_slot_facts,
    realize_gain_table,
)

MASK_A = hashlib.sha256(b"mask a").hexdigest()
MASK_B = hashlib.sha256(b"mask b").hexdigest()


def _timeline(count: int = 8) -> ExecutedTimeline:
    return ExecutedTimeline(
        tuple(
            ExecutedStepAnchors(
                index,
                float(count - index),
                index / (count - 1) if count > 1 else 0.0,
            )
            for index in range(count)
        )
    )


def _keyframe_curve(
    keyframes: tuple[GainKeyframe, ...],
    *,
    coordinate: TimelineCoordinate = TimelineCoordinate.STEP_INDEX,
    interpolation: GainInterpolation = GainInterpolation.LINEAR,
    endpoint: GainEndpoint = GainEndpoint.CLAMP,
    omission: GainOmission = GainOmission.INHERIT,
    hold_policy: KeyframeHoldPolicy = KeyframeHoldPolicy.BEST_EFFORT,
) -> KeyframeGainCurve:
    return KeyframeGainCurve(coordinate, keyframes, interpolation, endpoint, omission, hold_policy)


def _gain(curve: object, **overrides: object) -> ContributionGain:
    arguments: dict[str, object] = {
        "curve": curve,
        "global_gain": 2.0,
        "site_gains": (("residual", 1.0),),
        "lane_gains": (("cond", 1.0),),
        "effect_mask_digests": (MASK_A,),
        **overrides,
    }
    return ContributionGain(**arguments)  # type: ignore[arg-type]


def test_constant_curve_realizes_base_state_on_every_row() -> None:
    gain = _gain(ConstantGainCurve(0.5))
    table = realize_gain_table(gain, _timeline(4))

    assert len(table.rows) == 4
    assert table.assignment_profile is None
    assert table.hold_policy is None
    assert table.keyframe_counts == ()
    assert table.schedule_digest == gain.digest
    for row in table.rows:
        assert row.segment == "constant"
        assert row.keyframe_id is None
        assert row.timeline_gain == 0.5
        assert row.global_gain == 2.0
        assert row.site_gains == (("residual", 1.0),)
        assert row.lane_gains == (("cond", 1.0),)
        assert row.effect_mask_digests == (MASK_A,)
    assert realize_gain_table(gain, _timeline(4)).digest == table.digest


def test_direct_table_matches_exact_rows_and_never_resamples() -> None:
    gain = _gain(DirectGainTableCurve((0.1, 0.2, 0.3)))
    table = realize_gain_table(gain, _timeline(3))
    assert tuple(row.timeline_gain for row in table.rows) == (0.1, 0.2, 0.3)
    assert all(row.segment == "table" for row in table.rows)

    for count in (2, 4):
        with pytest.raises(GainRefusal) as error:
            realize_gain_table(gain, _timeline(count))
        assert error.value.code is GainRefusalCode.INVALID_GAIN_SCHEDULE
        assert "never" in str(error.value)


def test_linear_interpolation_between_anchors() -> None:
    curve = _keyframe_curve(
        (
            GainKeyframe("start", 0.0, 0.0),
            GainKeyframe("end", 1.0, 1.0),
        ),
        coordinate=TimelineCoordinate.PROGRESS,
    )
    table = realize_gain_table(_gain(curve), _timeline(5))

    assert table.rows[0].segment == "hold:start"
    assert table.rows[4].segment == "hold:end"
    for row in table.rows[1:4]:
        assert row.segment == "segment[0]:linear.v1"
        assert row.keyframe_id is None
        assert row.timeline_gain == pytest.approx(row.progress)
    assert table.keyframe_counts == (("start", 1, 1), ("end", 1, 1))


def test_hold_and_smoothstep_interpolation_profiles() -> None:
    def gains(interpolation: GainInterpolation) -> tuple[float, ...]:
        curve = _keyframe_curve(
            (
                GainKeyframe("start", 0.0, 0.0),
                GainKeyframe("end", 1.0, 1.0),
            ),
            coordinate=TimelineCoordinate.PROGRESS,
            interpolation=interpolation,
        )
        table = realize_gain_table(_gain(curve), _timeline(5))
        return tuple(row.timeline_gain for row in table.rows[1:4])

    assert gains(GainInterpolation.HOLD) == (0.0, 0.0, 0.0)
    fractions = (0.25, 0.5, 0.75)
    expected = tuple(f * f * (3.0 - 2.0 * f) for f in fractions)
    assert gains(GainInterpolation.SMOOTHSTEP) == pytest.approx(expected)


def test_scalar_site_and_lane_gains_interpolate_per_key() -> None:
    curve = _keyframe_curve(
        (
            GainKeyframe("start", 0.0, 1.0, site_gains=(("residual", 0.0),)),
            GainKeyframe("end", 1.0, 1.0, site_gains=(("residual", 2.0),)),
        ),
        coordinate=TimelineCoordinate.PROGRESS,
    )
    table = realize_gain_table(_gain(curve), _timeline(5))
    middle = table.rows[2]
    assert middle.site_gains == (("residual", 1.0),)
    assert middle.lane_gains == (("cond", 1.0),)
    assert middle.effect_mask_digests == (MASK_A,)


def test_endpoint_clamp_and_zero_behavior() -> None:
    def rows(endpoint: GainEndpoint):
        curve = _keyframe_curve(
            (GainKeyframe("mid", 0.5, 0.8, site_gains=(("residual", 3.0),)),),
            coordinate=TimelineCoordinate.PROGRESS,
            endpoint=endpoint,
        )
        return realize_gain_table(_gain(curve), _timeline(5)).rows

    clamped = rows(GainEndpoint.CLAMP)
    assert clamped[2].segment == "hold:mid"
    for row in (clamped[0], clamped[1]):
        assert row.segment == "endpoint.before:clamp.v1"
        assert row.timeline_gain == 0.8
    for row in (clamped[3], clamped[4]):
        assert row.segment == "endpoint.after:clamp.v1"
        assert row.timeline_gain == 0.8

    zeroed = rows(GainEndpoint.ZERO)
    for row in (zeroed[0], zeroed[1], zeroed[3], zeroed[4]):
        assert row.timeline_gain == 0.0
        assert row.site_gains == (("residual", 3.0),)


def test_omission_inherit_and_reset() -> None:
    keyframes = (
        GainKeyframe("first", 0, 1.0, site_gains=(("residual", 5.0),)),
        GainKeyframe("second", 7, 1.0),
    )
    inherit = realize_gain_table(
        _gain(_keyframe_curve(keyframes, omission=GainOmission.INHERIT)), _timeline(8)
    )
    assert inherit.rows[7].site_gains == (("residual", 5.0),)

    reset = realize_gain_table(
        _gain(_keyframe_curve(keyframes, omission=GainOmission.RESET)), _timeline(8)
    )
    assert reset.rows[7].site_gains == (("residual", 1.0),)


def test_monotone_nearest_assigns_contiguous_runs_at_nearest_rows() -> None:
    curve = _keyframe_curve(
        (
            GainKeyframe("a", 1, 1.0),
            GainKeyframe("b", 6, 1.0, minimum_realized_steps=2),
        )
    )
    table = realize_gain_table(_gain(curve), _timeline(8))

    held = {row.step_index: row.keyframe_id for row in table.rows if row.keyframe_id is not None}
    # b's two-row run ties between {5,6} and {6,7}; the earliest wins.
    assert held == {1: "a", 5: "b", 6: "b"}
    assert table.keyframe_counts == (("a", 1, 1), ("b", 2, 2))
    assert table.assignment_profile == MONOTONE_NEAREST_ASSIGNMENT
    for row in table.rows:
        if row.keyframe_id == "b":
            assert row.segment == "hold:b"


def test_monotone_nearest_state_monotonicity_under_contention() -> None:
    # Duplicate anchors refuse at curve construction.
    with pytest.raises(GainRefusal):
        _keyframe_curve(
            (
                GainKeyframe("a", 3, 1.0),
                GainKeyframe("b", 3, 1.0),
            )
        )

    curve = _keyframe_curve(
        (
            GainKeyframe("a", 3, 1.0),
            GainKeyframe("b", 4, 1.0),
        )
    )
    table = realize_gain_table(_gain(curve), _timeline(8))
    a_rows = [row.step_index for row in table.rows if row.keyframe_id == "a"]
    b_rows = [row.step_index for row in table.rows if row.keyframe_id == "b"]
    assert a_rows and b_rows
    assert max(a_rows) < min(b_rows)


def test_best_effort_over_subscription_is_deterministic_and_inspectable() -> None:
    curve = _keyframe_curve(
        (
            GainKeyframe("a", 0, 1.0, minimum_realized_steps=2),
            GainKeyframe("b", 1, 1.0, minimum_realized_steps=2),
        )
    )
    table = realize_gain_table(_gain(curve), _timeline(2))
    # Keyframes-with-a-row outranks occurrences: each keyframe gets one row.
    assert table.keyframe_counts == (("a", 2, 1), ("b", 2, 1))
    held = {row.step_index: row.keyframe_id for row in table.rows}
    assert held == {0: "a", 1: "b"}


def test_zero_row_keyframes_break_ties_by_declaration_index() -> None:
    curve = _keyframe_curve(
        (
            GainKeyframe("a", 0, 1.0),
            GainKeyframe("b", 1, 1.0),
        )
    )
    table = realize_gain_table(_gain(curve), _timeline(1))
    assert table.keyframe_counts == (("a", 1, 1), ("b", 1, 0))
    assert table.rows[0].keyframe_id == "a"


def test_hard_policy_refuses_unsatisfied_minimum() -> None:
    curve = _keyframe_curve(
        (
            GainKeyframe("a", 0, 1.0, minimum_realized_steps=2),
            GainKeyframe("b", 1, 1.0, minimum_realized_steps=2),
        ),
        hold_policy=KeyframeHoldPolicy.HARD,
    )
    with pytest.raises(GainRefusal) as error:
        realize_gain_table(_gain(curve), _timeline(2))
    assert error.value.code is GainRefusalCode.UNSATISFIED_KEYFRAME_MINIMUM


def test_sigma_coordinate_with_repeated_row_sigma() -> None:
    timeline = ExecutedTimeline(
        (
            ExecutedStepAnchors(0, 4.0, 0.0),
            ExecutedStepAnchors(1, 3.0, 0.25),
            ExecutedStepAnchors(2, 2.0, 0.5),
            ExecutedStepAnchors(3, 2.0, 0.75),
            ExecutedStepAnchors(4, 1.0, 1.0),
        )
    )
    curve = _keyframe_curve(
        (
            GainKeyframe("k1", 4.0, 0.0),
            GainKeyframe("k2", 2.0, 1.0),
        ),
        coordinate=TimelineCoordinate.SIGMA,
    )
    table = realize_gain_table(_gain(curve), timeline)

    assert table.rows[0].segment == "hold:k1"
    assert table.rows[1].timeline_gain == pytest.approx(0.5)
    # The tied nearest rows at sigma 2.0 resolve to the earliest step.
    assert table.rows[2].segment == "hold:k2"
    assert table.rows[3].segment == "anchor:k2"
    assert table.rows[3].timeline_gain == 1.0
    assert table.rows[4].segment == "endpoint.after:clamp.v1"


def test_schedule_digest_binds_every_declared_fact() -> None:
    def digest(**overrides: object) -> str:
        keyframes = overrides.pop(
            "keyframes",
            (GainKeyframe("a", 1, 0.5), GainKeyframe("b", 6, 1.0)),
        )
        curve = overrides.pop("curve", None) or _keyframe_curve(
            keyframes,  # type: ignore[arg-type]
            interpolation=overrides.pop("interpolation", GainInterpolation.LINEAR),  # type: ignore[arg-type]
            endpoint=overrides.pop("endpoint", GainEndpoint.CLAMP),  # type: ignore[arg-type]
            omission=overrides.pop("omission", GainOmission.INHERIT),  # type: ignore[arg-type]
            hold_policy=overrides.pop("hold_policy", KeyframeHoldPolicy.BEST_EFFORT),  # type: ignore[arg-type]
        )
        return _gain(curve, **overrides).digest

    base = digest()
    variants = (
        digest(global_gain=3.0),
        digest(site_gains=(("residual", 0.5),)),
        digest(lane_gains=(("uncond", 1.0),)),
        digest(effect_mask_digests=(MASK_B,)),
        digest(effect_mask_digests=()),
        digest(curve=ConstantGainCurve(1.0)),
        digest(curve=DirectGainTableCurve((1.0, 1.0))),
        digest(keyframes=(GainKeyframe("a", 2, 0.5), GainKeyframe("b", 6, 1.0))),
        digest(keyframes=(GainKeyframe("a", 1, 0.75), GainKeyframe("b", 6, 1.0))),
        digest(
            keyframes=(
                GainKeyframe("a", 1, 0.5, minimum_realized_steps=2),
                GainKeyframe("b", 6, 1.0),
            )
        ),
        digest(interpolation=GainInterpolation.SMOOTHSTEP),
        digest(endpoint=GainEndpoint.ZERO),
        digest(omission=GainOmission.RESET),
        digest(hold_policy=KeyframeHoldPolicy.HARD),
    )
    digests = {base, *variants}
    assert len(digests) == len(variants) + 1
    assert digest() == base


def test_realized_digest_binds_the_executed_timeline() -> None:
    gain = _gain(ConstantGainCurve(1.0))
    base = realize_gain_table(gain, _timeline(4))
    longer = realize_gain_table(gain, _timeline(5))
    shifted = realize_gain_table(
        gain,
        ExecutedTimeline(
            tuple(ExecutedStepAnchors(index, float(9 - index), index / 3) for index in range(4))
        ),
    )
    assert len({base.digest, longer.digest, shifted.digest}) == 3


def test_slot_facts_bind_schedule_realization_and_gains() -> None:
    curve = _keyframe_curve((GainKeyframe("a", 1, 0.5),))
    gain = _gain(curve)
    table = realize_gain_table(gain, _timeline(4))
    facts = contribution_gain_slot_facts(gain, table)

    assert f"gain.schedule={gain.digest}" in facts
    assert f"gain.realized_table={table.digest}" in facts
    assert "gain.global=2" in facts
    assert "gain.coordinate=step_index" in facts
    assert "gain.interpolation=linear.v1" in facts
    assert "gain.endpoint=clamp.v1" in facts
    assert "gain.omission=inherit" in facts
    assert f"gain.assignment={MONOTONE_NEAREST_ASSIGNMENT}" in facts
    assert "gain.hold_policy=best_effort" in facts
    assert "gain.site[residual]=1" in facts
    assert "gain.lane[cond]=1" in facts
    assert f"gain.effect_mask[0]={MASK_A}" in facts

    constant_facts = contribution_gain_slot_facts(
        _gain(ConstantGainCurve(1.0)),
        realize_gain_table(_gain(ConstantGainCurve(1.0)), _timeline(4)),
    )
    assert not any(fact.startswith("gain.assignment=") for fact in constant_facts)


def test_slot_facts_refuse_a_table_from_another_schedule() -> None:
    gain = _gain(ConstantGainCurve(1.0))
    other = _gain(ConstantGainCurve(0.5))
    table = realize_gain_table(other, _timeline(4))
    with pytest.raises(GainRefusal) as error:
        contribution_gain_slot_facts(gain, table)
    assert error.value.code is GainRefusalCode.INVALID_GAIN_SCHEDULE


def test_schedule_validation_refusals() -> None:
    cases = (
        # unordered step anchors
        (GainKeyframe("a", 5, 1.0), GainKeyframe("b", 1, 1.0)),
        # duplicate anchors
        (GainKeyframe("a", 1, 1.0), GainKeyframe("b", 1, 1.0)),
        # duplicate ids
        (GainKeyframe("a", 1, 1.0), GainKeyframe("a", 2, 1.0)),
        # float anchor on the step_index coordinate
        (GainKeyframe("a", 1.0, 1.0),),
    )
    for keyframes in cases:
        with pytest.raises(GainRefusal) as error:
            _keyframe_curve(keyframes)
        assert error.value.code is GainRefusalCode.INVALID_GAIN_SCHEDULE

    with pytest.raises(GainRefusal) as sigma_error:
        _keyframe_curve(
            (GainKeyframe("a", 1.0, 1.0), GainKeyframe("b", 2.0, 1.0)),
            coordinate=TimelineCoordinate.SIGMA,
        )
    assert sigma_error.value.code is GainRefusalCode.INVALID_GAIN_SCHEDULE

    with pytest.raises(GainRefusal) as nan_error:
        GainKeyframe("a", 1, float("nan"))
    assert nan_error.value.code is GainRefusalCode.INVALID_GAIN_SCHEDULE

    with pytest.raises(GainRefusal) as profile_error:
        KeyframeGainCurve(
            TimelineCoordinate.STEP_INDEX,
            (GainKeyframe("a", 1, 1.0),),
            "linear.v1",  # type: ignore[arg-type]
            GainEndpoint.CLAMP,
            GainOmission.INHERIT,
            KeyframeHoldPolicy.BEST_EFFORT,
        )
    assert profile_error.value.code is GainRefusalCode.UNSUPPORTED_GAIN_PROFILE


def test_replacement_maps_must_cover_the_declared_key_set() -> None:
    curve = _keyframe_curve((GainKeyframe("a", 1, 1.0, site_gains=(("other", 1.0),)),))
    with pytest.raises(GainRefusal) as error:
        _gain(curve)
    assert error.value.code is GainRefusalCode.INVALID_GAIN_SCHEDULE
    assert "site key set" in str(error.value)


def test_realization_inputs_must_be_exact_values() -> None:
    gain = _gain(ConstantGainCurve(1.0))
    timeline = _timeline(2)

    class DuckGain:
        curve = gain.curve
        global_gain = gain.global_gain
        site_gains = gain.site_gains
        lane_gains = gain.lane_gains
        effect_mask_digests = gain.effect_mask_digests
        digest = gain.digest

    with pytest.raises(GainRefusal):
        realize_gain_table(DuckGain(), timeline)  # type: ignore[arg-type]
    with pytest.raises(GainRefusal):
        realize_gain_table(gain, tuple(timeline.steps))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RealizedGainTable()  # type: ignore[call-arg]


def test_timeline_validation() -> None:
    with pytest.raises(TypeError):
        ExecutedTimeline(())
    with pytest.raises(TypeError):
        ExecutedTimeline(
            (ExecutedStepAnchors(1, 1.0, 0.0),)  # step_index not dense from 0
        )
    with pytest.raises(TypeError):
        ExecutedTimeline(
            (
                ExecutedStepAnchors(0, 2.0, 0.5),
                ExecutedStepAnchors(1, 1.0, 0.5),  # progress not increasing
            )
        )
    with pytest.raises(TypeError):
        ExecutedStepAnchors(0, -1.0, 0.0)
    with pytest.raises(TypeError):
        ExecutedStepAnchors(0, 1.0, 1.5)


def test_assignment_distance_objective_is_exact_under_float_overflow() -> None:
    # Float distance sums for both candidate runs overflow to inf; the
    # exact objective requires the nearer later run {1, 2}.
    timeline = ExecutedTimeline(
        (
            ExecutedStepAnchors(0, 1.7e308, 0.0),
            ExecutedStepAnchors(1, 1.6e308, 0.5),
            ExecutedStepAnchors(2, 1.5e308, 1.0),
        )
    )
    curve = _keyframe_curve(
        (GainKeyframe("a", 0.0, 1.0, minimum_realized_steps=2),),
        coordinate=TimelineCoordinate.SIGMA,
    )
    table = realize_gain_table(_gain(curve), timeline)
    held = [row.step_index for row in table.rows if row.keyframe_id == "a"]
    assert held == [1, 2]


def test_linear_interpolation_is_overflow_safe() -> None:
    # b - a overflows for these finite gains; the interpolated value at
    # the midpoint is exactly 0.
    curve = _keyframe_curve(
        (
            GainKeyframe("a", 0.0, 1e308),
            GainKeyframe("b", 1.0, -1e308),
        ),
        coordinate=TimelineCoordinate.PROGRESS,
    )
    timeline = ExecutedTimeline(
        (
            ExecutedStepAnchors(0, 3.0, 0.0),
            ExecutedStepAnchors(1, 2.0, 0.5),
            ExecutedStepAnchors(2, 1.0, 1.0),
        )
    )
    table = realize_gain_table(_gain(curve), timeline)
    midpoint = [row for row in table.rows if row.keyframe_id is None]
    assert midpoint and all(row.timeline_gain == 0.0 for row in midpoint)
    assert all(row.timeline_gain in (1e308, -1e308, 0.0) for row in table.rows)


def test_malformed_anchors_refuse_instead_of_leaking() -> None:
    for coordinate, anchor in (
        (TimelineCoordinate.PROGRESS, "bad"),
        (TimelineCoordinate.SIGMA, "bad"),
        (TimelineCoordinate.STEP_INDEX, "bad"),
        (TimelineCoordinate.PROGRESS, 10**400),
        (TimelineCoordinate.SIGMA, 10**400),
    ):
        with pytest.raises(GainRefusal):
            _keyframe_curve(
                (GainKeyframe("a", anchor, 1.0),),  # type: ignore[arg-type]
                coordinate=coordinate,
            )


def test_huge_step_index_anchors_stay_exact_integers() -> None:
    # step_index anchors are exact nonnegative ints with no
    # float-representability bound; declaration and realization must
    # not leak OverflowError or refuse valid huge anchors.
    curve = _keyframe_curve(
        (
            GainKeyframe("a", 0, 0.0),
            GainKeyframe("b", 10**400, 1.0),
        )
    )
    timeline = ExecutedTimeline(
        (
            ExecutedStepAnchors(0, 3.0, 0.0),
            ExecutedStepAnchors(1, 2.0, 0.5),
            ExecutedStepAnchors(2, 1.0, 1.0),
        )
    )
    table = realize_gain_table(_gain(curve), timeline)
    assert [row.keyframe_id for row in table.rows] == ["a", None, "b"]
    # Row 1 interpolates at fraction 1/10**400, which rounds to 0.
    assert table.rows[1].timeline_gain == 0.0
    assert table.rows[2].timeline_gain == 1.0


def test_canonical_serialization_is_total_over_huge_integers() -> None:
    # Anchors and minimum_realized_steps beyond the interpreter's
    # int-to-str digit limit (default 4300) must digest and realize
    # without leaking ValueError; the preimage carries exact decimal.
    curve = _keyframe_curve(
        (
            GainKeyframe("a", 0, 0.5, minimum_realized_steps=10**5000),
            GainKeyframe("b", 10**5000, 1.0),
        )
    )
    gain = _gain(curve)
    expected_decimal = "1" + "0" * 5000
    assert f'"anchor":"{expected_decimal}"' in gain.canonical_preimage
    assert f'"minimum_realized_steps":"{expected_decimal}"' in gain.canonical_preimage
    table = realize_gain_table(gain, _timeline(3))
    assert f'"{expected_decimal}"' in table.canonical_preimage
    assert table.keyframe_counts[0][1] == 10**5000
    assert len(table.digest) == 64


def test_hard_policy_refusal_formats_huge_requested_counts() -> None:
    curve = _keyframe_curve(
        (GainKeyframe("a", 0, 1.0, minimum_realized_steps=10**5000),),
        hold_policy=KeyframeHoldPolicy.HARD,
    )
    with pytest.raises(GainRefusal) as excinfo:
        realize_gain_table(_gain(curve), _timeline(2))
    assert excinfo.value.code is GainRefusalCode.UNSATISFIED_KEYFRAME_MINIMUM
    assert "1" + "0" * 5000 in str(excinfo.value)


def test_adjacent_huge_step_index_anchors_distinguish() -> None:
    # 2**53 + 1 and 2**53 + 2 collapse under float conversion; exact
    # integer coordinates must keep them distinct anchors.
    curve = _keyframe_curve(
        (
            GainKeyframe("a", 2**53 + 1, 1.0),
            GainKeyframe("b", 2**53 + 2, 2.0),
        )
    )
    timeline = ExecutedTimeline(
        (
            ExecutedStepAnchors(0, 2.0, 0.0),
            ExecutedStepAnchors(1, 1.0, 1.0),
        )
    )
    table = realize_gain_table(_gain(curve), timeline)
    assert [row.keyframe_id for row in table.rows] == ["a", "b"]
    assert [row.timeline_gain for row in table.rows] == [1.0, 2.0]
