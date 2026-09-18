from __future__ import annotations

from types import MappingProxyType

import pytest
from dinkster_inference import (
    AttentionModifierSchedule,
    AttentionPlan,
    ExecutedStepAnchors,
    ExecutedTimeline,
    RealizedSamplingRow,
    RealizedSamplingTimeline,
    SamplingParameterCurve,
    SamplingTimelineSchedule,
    current_realized_sampling_row,
    realize_sampling_timeline,
)
from dinkster_inference.sampling_timeline import use_realized_sampling_timeline
from dinkster_nodes_foundation import Curve


def test_realizes_discrete_provider_window_and_sol_curve() -> None:
    curve = SamplingParameterCurve(((0.0, 1.0), (1.0, 3.0)), "linear")
    schedule = SamplingTimelineSchedule("sol", 0.25, 0.75, curve)

    timeline = realize_sampling_timeline(schedule, (4.0, 3.0, 2.0, 1.0, 0.0))

    assert tuple(row.attention_plan.provider for row in timeline.rows) == (
        "sdpa",
        "sol",
        "sol",
        "sdpa",
    )
    assert timeline.rows[1].attention_plan.parameters == MappingProxyType({"sol.tau": 5.0 / 3.0})
    assert timeline.rows[2].attention_plan.parameters == MappingProxyType({"sol.tau": 7.0 / 3.0})
    assert timeline.digest == realize_sampling_timeline(schedule, (4.0, 3.0, 2.0, 1.0, 0.0)).digest


def test_sampling_parameter_curve_matches_native_curve() -> None:
    native = Curve(((0.0, 1.0), (0.3, 4.0), (1.0, 2.0)), "monotone_cubic")
    snapshot = SamplingParameterCurve(native.points, native.interpolation)

    assert tuple(snapshot.evaluate(index / 20) for index in range(21)) == tuple(
        native.evaluate(index / 20) for index in range(21)
    )


@pytest.mark.parametrize(
    ("provider", "curve", "message"),
    [
        ("sage", SamplingParameterCurve(((0.0, 1.0),)), "only the Sol"),
        ("sol", SamplingParameterCurve(((0.0, 0.0),)), "greater than zero"),
    ],
)
def test_rejects_invalid_provider_parameters(
    provider: str,
    curve: SamplingParameterCurve,
    message: str,
) -> None:
    if provider == "sage":
        with pytest.raises(ValueError, match=message):
            SamplingTimelineSchedule("sage", 0.0, 1.0, curve)
        return
    schedule = SamplingTimelineSchedule("sol", 0.0, 1.0, curve)
    with pytest.raises(ValueError, match=message):
        realize_sampling_timeline(schedule, (1.0, 0.0))


def test_no_realized_row_leaks_without_solver_execution() -> None:
    assert current_realized_sampling_row() is None


def test_modifier_free_timeline_preserves_existing_digests() -> None:
    schedule = SamplingTimelineSchedule("sage", 0.0, 1.0)
    timeline = realize_sampling_timeline(schedule, (1.0, 0.0))

    assert schedule.digest == "2a2e3fd2dbd504865780c26552334ac924c6f533206fe7200e7018d0bbbf9691"
    assert timeline.digest == "1623d55a967ba3a1ec68ddfb796c211f8fd2cf89d076c077abb0995cb8cf64b0"


def test_realizes_categorical_attention_modifiers_without_interpolation() -> None:
    schedule = SamplingTimelineSchedule(
        "sage",
        0.0,
        1.0,
        attention_modifiers=(AttentionModifierSchedule("skip_softmax", 0.5, 1.0),),
    )

    timeline = realize_sampling_timeline(schedule, (3.0, 2.0, 1.0, 0.0))

    assert tuple(row.attention_plan.modifiers for row in timeline.rows) == (
        (),
        ("skip_softmax",),
        ("skip_softmax",),
    )


def test_realizes_sol_conditioning_sink_only_inside_provider_window() -> None:
    schedule = SamplingTimelineSchedule(
        "sol",
        0.25,
        0.75,
        attention_modifiers=(AttentionModifierSchedule("sol_conditioning_exact_kv", 0.25, 0.75),),
    )

    timeline = realize_sampling_timeline(schedule, (4.0, 3.0, 2.0, 1.0, 0.0))

    assert tuple(row.attention_plan.modifiers for row in timeline.rows) == (
        (),
        ("sol_conditioning_exact_kv",),
        ("sol_conditioning_exact_kv",),
        (),
    )
    assert timeline.digest == realize_sampling_timeline(schedule, (4.0, 3.0, 2.0, 1.0, 0.0)).digest


def test_conditioning_sink_schedule_rejects_non_sol_provider_and_outside_window() -> None:
    exact_kv = AttentionModifierSchedule("sol_conditioning_exact_kv", 0.25, 0.75)
    with pytest.raises(ValueError, match="require the Sol provider"):
        SamplingTimelineSchedule("sage", 0.0, 1.0, attention_modifiers=(exact_kv,))
    with pytest.raises(ValueError, match="require the Sol provider"):
        AttentionPlan(
            "sdpa",
            ("sol_conditioning_exact_kv",),
            MappingProxyType({}),
        )
    with pytest.raises(ValueError, match="inside the Sol provider window"):
        SamplingTimelineSchedule(
            "sol",
            0.25,
            0.75,
            attention_modifiers=(AttentionModifierSchedule("sol_conditioning_exact_kv", 0.0, 1.0),),
        )


def test_realized_row_activation_cannot_escape_timeline_scope() -> None:
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule("sage", 0.0, 1.0),
        (1.0, 0.0),
    )
    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        assert current_realized_sampling_row() is timeline.rows[0]

    with pytest.raises(RuntimeError, match="escaped its timeline scope"):
        activate(0)
    assert current_realized_sampling_row() is None


def test_realized_row_owns_immutable_parameter_snapshot() -> None:
    backing = {"sol.tau": 1.0}
    anchors = ExecutedStepAnchors(0, 1.0, 0.0)
    row = RealizedSamplingRow(
        anchors,
        AttentionPlan("sol", (), MappingProxyType(backing)),
        "0" * 64,
    )
    timeline = RealizedSamplingTimeline(
        "0" * 64,
        ExecutedTimeline((anchors,)),
        (row,),
    )
    digest = timeline.digest

    backing["sol.tau"] = 2.0

    assert timeline.digest == digest
    assert row.attention_plan.parameters == MappingProxyType({"sol.tau": 1.0})
