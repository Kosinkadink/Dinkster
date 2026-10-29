"""Parity tests for torch-free context-window planning.

Window plans, compiled per-window fuse weights, and the step jitter
replay against goldens minted from the pinned ComfyUI reference by
tools/gen_context_window_goldens.py."""

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_inference.context_windows import (
    TEMPORAL_WINDOW_AXIS,
    ContextFuseMethod,
    ContextWindowSchedule,
    ContextWindowsError,
    ContextWindowsSpec,
    _ordered_halving,
    context_window_plan,
    freenoise_plan,
    plan_windows,
    relative_bias,
    step_index_for_sigma,
)

GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "context_windows_b78cec87.json").read_text(
        encoding="utf-8"
    )
)


def _window_case_id(case: dict) -> str:
    return (
        f"{case['schedule']}-f{case['num_frames']}-l{case['length']}"
        f"-o{case['overlap']}-s{case['stride']}" + ("-closed" if case["closed_loop"] else "")
    )


@pytest.mark.parametrize("entry", GOLDENS["ordered_halving"], ids=lambda e: str(e["step"]))
def test_ordered_halving_matches_reference(entry):
    assert _ordered_halving(entry["step"]) == entry["value"]


@pytest.mark.parametrize("case", GOLDENS["windows"], ids=_window_case_id)
def test_window_plans_match_reference(case):
    spec = ContextWindowsSpec(
        schedule=ContextWindowSchedule(case["schedule"]),
        fuse_method=ContextFuseMethod.FLAT,
        length=case["length"],
        overlap=case["overlap"],
        stride=case["stride"],
        closed_loop=case["closed_loop"],
    )
    for plan in case["plans"]:
        windows = plan_windows(spec, case["num_frames"], plan["step"])
        assert [list(window) for window in windows] == plan["windows"]


@pytest.mark.parametrize(
    "case",
    GOLDENS["plan_weights"],
    ids=lambda c: f"{c['fuse_method']}-{_window_case_id(c)}",
)
def test_compiled_plan_weights_match_reference(case):
    spec = ContextWindowsSpec(
        schedule=ContextWindowSchedule(case["schedule"]),
        fuse_method=ContextFuseMethod(case["fuse_method"]),
        length=case["length"],
        overlap=case["overlap"],
        stride=case["stride"],
        closed_loop=case["closed_loop"],
    )
    plan = context_window_plan(spec, case["num_frames"], case["step"])
    assert len(plan.axes) == 1
    assert plan.axes[0].name == TEMPORAL_WINDOW_AXIS
    assert plan.axes[0].extent == case["num_frames"]
    assert len(plan.joint_windows) == len(case["weights"])
    for window, reference_weights in zip(plan.joint_windows, case["weights"], strict=True):
        assert len(window.occurrences) == len(reference_weights)
        # The reference computes overlap-linear ramps in float32; the
        # float64 compile is value-close, and flat/pyramid weights are
        # exact.
        for occurrence, reference in zip(window.occurrences, reference_weights, strict=True):
            assert math.isclose(occurrence.weight, reference, rel_tol=1e-6, abs_tol=1e-12)


def test_compiled_plan_windows_match_planner():
    spec = ContextWindowsSpec(
        schedule=ContextWindowSchedule.UNIFORM_STANDARD,
        fuse_method=ContextFuseMethod.PYRAMID,
        length=16,
        overlap=4,
    )
    windows = plan_windows(spec, 33, 1)
    plan = context_window_plan(spec, 33, 1)
    assert tuple(window.axis_indices[0][1] for window in plan.joint_windows) == windows
    assert all(weight > 0.0 for weight in plan.total_weights)


def test_every_planned_frame_is_covered():
    for case in GOLDENS["windows"]:
        spec = ContextWindowsSpec(
            schedule=ContextWindowSchedule(case["schedule"]),
            fuse_method=ContextFuseMethod.FLAT,
            length=case["length"],
            overlap=case["overlap"],
            stride=case["stride"],
            closed_loop=case["closed_loop"],
        )
        for plan in case["plans"]:
            windows = plan_windows(spec, case["num_frames"], plan["step"])
            covered = {index for window in windows for index in window}
            assert covered == set(range(case["num_frames"]))


def test_relative_bias_matches_reference_formula():
    # 1 - |idx - center| / ((last - first + 1e-2) / 2), floored at 1e-2.
    assert relative_bias(8, 0, 15) == 1 - abs(8 - 7.5) / ((15 + 1e-2) / 2)
    assert relative_bias(0, 0, 15) == max(1e-2, 1 - 7.5 / ((15 + 1e-2) / 2))
    assert relative_bias(5, 5, 5) == 1.0
    assert relative_bias(0, 0, 0) == 1.0


def test_step_index_for_sigma_matches_schedule_positions():
    sigmas = (14.61, 7.49, 3.86, 1.84, 0.68, 0.0)
    for index, sigma in enumerate(sigmas):
        assert step_index_for_sigma(sigmas, sigma) == index
    # Within the reference's rtol=1e-4 match window.
    assert step_index_for_sigma(sigmas, 7.49 * (1 + 5e-5)) == 1
    # A midpoint sigma is a solver substep: no schedule position.
    assert step_index_for_sigma(sigmas, 5.6) is None
    assert step_index_for_sigma(sigmas, 7.49 * (1 + 5e-3)) is None


def test_freenoise_plan_spans():
    # 33 frames, length 16, overlap 4: shuffle delta-12 spans until the tail.
    assert freenoise_plan(33, 16, 4) == ((0, 16, 12), (12, 28, 5))
    # Exact fit: nothing past the first window to shuffle.
    assert freenoise_plan(16, 16, 4) == ()
    assert freenoise_plan(17, 16, 4) == ((0, 16, 1),)


def _spec(
    schedule: ContextWindowSchedule = ContextWindowSchedule.STATIC_STANDARD,
    fuse_method: ContextFuseMethod = ContextFuseMethod.PYRAMID,
    length: int = 16,
    overlap: int = 4,
    stride: int = 1,
    dim: int = 0,
    cond_retain_indices: tuple[int, ...] = (),
    latent_retain_indices: tuple[int, ...] = (),
) -> ContextWindowsSpec:
    return ContextWindowsSpec(
        schedule=schedule,
        fuse_method=fuse_method,
        length=length,
        overlap=overlap,
        stride=stride,
        dim=dim,
        cond_retain_indices=cond_retain_indices,
        latent_retain_indices=latent_retain_indices,
    )


def test_spec_validation_rejections():
    _spec()
    with pytest.raises(ContextWindowsError):
        _spec(length=0)
    with pytest.raises(ContextWindowsError):
        _spec(overlap=16)
    with pytest.raises(ContextWindowsError):
        _spec(overlap=-1)
    with pytest.raises(ContextWindowsError):
        _spec(stride=0)
    with pytest.raises(ContextWindowsError):
        _spec(dim=-1)
    with pytest.raises(ContextWindowsError):
        _spec(length=True)
    with pytest.raises(ContextWindowsError):
        _spec(schedule=cast(ContextWindowSchedule, "standard_static"))
    with pytest.raises(ContextWindowsError):
        _spec(fuse_method=ContextFuseMethod.OVERLAP_LINEAR, overlap=0)


def test_spec_retain_index_validation():
    accepted = _spec(cond_retain_indices=(0, 3), latent_retain_indices=(0,))
    assert accepted.cond_retain_indices == (0, 3)
    assert accepted.latent_retain_indices == (0,)
    for name in ("cond_retain_indices", "latent_retain_indices"):
        for bad in ([0], (0.0,), (True,), (-1,), (3, 3), (3, 0), (16,)):
            overrides: dict[str, Any] = {name: bad}
            with pytest.raises(ContextWindowsError):
                _spec(**overrides)


def test_plan_windows_argument_rejections():
    spec = ContextWindowsSpec(
        schedule=ContextWindowSchedule.STATIC_STANDARD,
        fuse_method=ContextFuseMethod.PYRAMID,
        length=16,
        overlap=4,
    )
    with pytest.raises(ContextWindowsError):
        plan_windows(spec, 0, 0)
    with pytest.raises(ContextWindowsError):
        plan_windows(spec, 33, -1)
    with pytest.raises(ContextWindowsError):
        context_window_plan(
            ContextWindowsSpec(
                schedule=ContextWindowSchedule.STATIC_STANDARD,
                fuse_method=ContextFuseMethod.RELATIVE,
                length=16,
                overlap=4,
            ),
            33,
            0,
        )
