from typing import Any, cast

from tools.validate_seedvr2_e2e import case_specs, evaluate_oom_pair, evaluate_pair


def _record(value: float) -> dict[str, object]:
    return {
        "performance": {
            "cold_total_seconds": value,
            "warm_median_seconds": value,
        },
        "process_peak_cuda": {
            "peak_allocated_bytes": int(value),
            "peak_reserved_bytes": int(value),
        },
        "baseline": {"host": {"current_bytes": 100}},
        "residual": {
            "cuda": {"allocated_bytes": int(value), "reserved_bytes": int(value)},
            "host": {"current_bytes": 100 + int(value), "peak_bytes": 100 + int(value)},
        },
    }


def test_seedvr2_e2e_matrix_covers_all_math_and_media_paths() -> None:
    cases = case_specs()

    assert {case.diffusion for case in cases} == {
        "3b_swiglu",
        "7b_mlp_int8",
        "7b_mlp_sharp",
    }
    assert {case.frames > 1 for case in cases} == {False, True}
    assert sum(case.target_free_gib is not None for case in cases) == 1
    assert sum(case.expected_oom for case in cases) == 1


def test_seedvr2_e2e_pair_requires_dinkster_not_to_regress() -> None:
    assert evaluate_pair(_record(1), _record(2))["performance_memory_passed"] is True
    failed = evaluate_pair(_record(2), _record(1))

    assert failed["performance_memory_passed"] is False
    metrics = cast("dict[str, dict[str, Any]]", failed["metrics"])
    assert all(not metric["passed"] for metric in metrics.values())


def test_seedvr2_e2e_oom_pair_requires_matching_failure_fallback_and_cleanup() -> None:
    record = _record(1)
    record.update(
        status="failed",
        error={"type": "OutOfMemoryError"},
        failure={"cuda": {"driver_free_bytes": 100}},
        fallback={"diffusion": {"offloaded_bytes": 10}},
    )
    dinkster = cast("dict[str, Any]", record.copy())
    dinkster["fallback"] = {"diffusion": [{"offloaded_bytes": 10}]}

    assert evaluate_oom_pair(dinkster, cast("dict[str, Any]", record))["passed"] is True
