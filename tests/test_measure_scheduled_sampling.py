from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
TOOL_PATH = ROOT / "tools/measure_scheduled_sampling.py"
CURRENT_RECORD = (
    ROOT / "packages/dinkster-inference-torch/tests/performance/scheduled_runtime_blackwell.json"
)
HISTORICAL_MEASURED_COMMIT = "0afd90dc7498fd922765eee9b3cf4739cad8f47c"
HISTORICAL_MEASURED_SOURCE_SHA256 = (
    "sha256:b6637772d36bf37e00c43e7422ba2c68aa3d82b9d34be6a0ba90b49bf5ad9b71"
)


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_scheduled_sampling", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _telemetry() -> dict[str, object]:
    raw = []
    for phase in TOOL.PHASES:
        for boundary in ("before", "after"):
            raw.append(
                {
                    "captured_at": "2026-08-05T00:00:00.000000Z",
                    "phase": phase,
                    "boundary": boundary,
                    "gpu": {field: "1" for field in TOOL.GPU_TELEMETRY_FIELDS},
                    "host": {
                        "load_average": [0.1, 0.2, 0.3],
                        "process_max_rss_bytes": 100,
                    },
                }
            )
    telemetry = {
        "clock_policy": "default-unmodified",
        "raw": raw,
        "summary": TOOL._telemetry_summary(raw),
    }
    return telemetry


def _invocations(
    *, scheduled_wall_ns: int = 110, scheduled_resource: int = 110
) -> list[dict[str, Any]]:
    invocations: list[dict[str, Any]] = []
    for plan_item in TOOL._run_plan():
        index = int(plan_item["invocation_index"])
        engine = str(plan_item["engine"])
        family = str(plan_item["family"])
        wall = 100 if engine == "ordinary" else scheduled_wall_ns
        resource = 100 if engine == "ordinary" else scheduled_resource
        phases = {
            phase: {
                "output_sha256": "sha256:" + ("a" if family == "flux" else "b") * 64,
                "output_shape": [1, 4, 8, 8],
                "output_dtype": "float32",
                "wall_elapsed_ns": wall,
                "cuda_elapsed_ns": wall - 1,
                "peak_ram_bytes": resource,
                "peak_vram_bytes": resource,
                "telemetry_refs": [phase_index * 2, phase_index * 2 + 1],
            }
            for phase_index, phase in enumerate(TOOL.PHASES)
        }
        worker = {
            "worker_id": f"worker-{index}",
            "pid": 10_000 + index,
            "engine": engine,
            "family": TOOL.FAMILY_IDS[family],
            "environment": {
                "python": "3.12.3",
                "torch": "2.13.0+cu130",
                "cuda": "13.0",
                "device_name": "synthetic",
                "device_uuid": "synthetic-uuid",
                "driver": "595.84",
                "visible_cuda_device_count": 1,
            },
            "phase_order": list(TOOL.PHASES),
            "phases": phases,
            "telemetry": _telemetry(),
            "started_at": "2026-08-05T00:00:00.000000Z",
            "finished_at": "2026-08-05T00:00:01.000000Z",
        }
        invocations.append(
            {
                **plan_item,
                "command": TOOL._worker_command(engine, family),
                "cuda_visible_devices": "0",
                "pid": 10_000 + index,
                "return_code": 0,
                "stderr": "",
                "stdout_sha256": "sha256:" + f"{index:064x}",
                "parse_error": None,
                "attempt": 1,
                "retry_count": 0,
                "discarded": False,
                "substituted": False,
                "replacement": False,
                "started_at": "2026-08-05T00:00:00.000000Z",
                "finished_at": "2026-08-05T00:00:01.000000Z",
                "worker": worker,
            }
        )
    return invocations


def _checks(invocations: list[dict[str, Any]]) -> dict[str, bool]:
    result = TOOL._evaluate_invocations(invocations)
    checks = result["checks"]
    assert isinstance(checks, dict)
    return checks


def test_plan_is_exact_adjacent_counterbalanced_and_family_separated() -> None:
    plan = TOOL._run_plan()
    assert len(plan) == 64
    for family_index, family in enumerate(TOOL.FAMILIES):
        family_plan = plan[family_index * 32 : (family_index + 1) * 32]
        assert {item["family"] for item in family_plan} == {family}
        assert [item["order"] for item in family_plan[::2]] == list(TOOL.PAIR_ORDERS)
        assert [
            "".join("O" if item["engine"] == "ordinary" else "S" for item in pair)
            for pair in (family_plan[index : index + 2] for index in range(0, 32, 2))
        ] == list(TOOL.PAIR_ORDERS)
        assert [item["pair_index"] for item in family_plan[::2]] == list(range(1, 17))
    assert plan[31]["family"] == "flux"
    assert plan[32]["family"] == "sd"


def test_rank_uses_one_based_r12_and_exact_115_passes_with_raw_strata() -> None:
    result = TOOL._evaluate_invocations(_invocations(scheduled_wall_ns=115, scheduled_resource=115))
    assert result["overall_pass"] is True
    for family in TOOL.FAMILIES:
        for phase in TOOL.PHASES:
            timing = result["timing"][family][phase]
            assert timing["rank"] == {"one_based": 12, "value": 1.15}
            assert timing["one_sided_coverage"] == 0.9615936279296875
            assert len(timing["raw_ratios"]) == 16
            assert len(timing["strata"]["OS"]) == 8
            assert len(timing["strata"]["SO"]) == 8
            assert timing["within_15_percent"] is True
    serialized = json.dumps(result, sort_keys=True)
    assert "simultaneous" not in serialized.lower()


def test_r12_not_worst_ratio_classifies_wall_time_only() -> None:
    invocations = _invocations()
    for pair_index in range(12, 16):
        scheduled = next(
            item
            for item in invocations[pair_index * 2 : pair_index * 2 + 2]
            if item["engine"] == "scheduled"
        )
        for phase in TOOL.PHASES:
            scheduled["worker"]["phases"][phase]["wall_elapsed_ns"] = 200
    result = TOOL._evaluate_invocations(invocations)
    assert result["timing"]["flux"]["real"]["rank"]["value"] == 1.1
    assert result["timing"]["flux"]["real"]["within_15_percent"] is True
    assert result["overall_pass"] is True


def test_output_and_each_adjacent_resource_gate_fail_closed() -> None:
    output_mismatch = _invocations()
    output_mismatch[1]["worker"]["phases"]["real"]["output_sha256"] = "sha256:wrong"
    output_result = TOOL._evaluate_invocations(output_mismatch)
    assert output_result["checks"]["flux_pair_1_real_output_exact"] is False
    assert output_result["overall_pass"] is False

    resources = _invocations(scheduled_resource=116)
    resource_result = TOOL._evaluate_invocations(resources)
    assert resource_result["checks"]["flux_pair_1_warmup_ram_within_15_percent"] is False
    assert resource_result["checks"]["flux_pair_1_real_vram_within_15_percent"] is False
    assert resource_result["overall_pass"] is False


def test_symmetric_pair_output_drift_fails_family_wide_identity() -> None:
    invocations = _invocations()
    second_flux_pair = invocations[2:4]
    for invocation in second_flux_pair:
        for phase in TOOL.PHASES:
            invocation["worker"]["phases"][phase]["output_sha256"] = "sha256:" + "c" * 64
    result = TOOL._evaluate_invocations(invocations)
    assert result["checks"]["flux_pair_2_warmup_output_exact"] is True
    assert result["checks"]["flux_pair_2_real_output_exact"] is True
    assert result["checks"]["warmup_real_output_exact_each_worker"] is True
    assert result["checks"]["flux_output_exact_all_workers"] is False
    assert result["overall_pass"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        (lambda rows: rows.pop(), "worker_count_exact"),
        (lambda rows: rows[1].update(pid=rows[0]["pid"]), "parent_pids_unique"),
        (
            lambda rows: rows[1]["worker"].update(worker_id=rows[0]["worker"]["worker_id"]),
            "worker_ids_unique",
        ),
        (
            lambda rows: rows[1]["worker"].update(pid=rows[0]["worker"]["pid"]),
            "worker_pids_match_and_unique",
        ),
        (lambda rows: rows[0].update(return_code=1), "return_codes_zero"),
        (lambda rows: rows[0].update(stderr="failure\n"), "stderr_empty"),
        (lambda rows: rows[0].update(parse_error="invalid JSON"), "worker_payloads_complete"),
        (lambda rows: rows[0].update(retry_count=1), "no_retry_or_discard"),
        (lambda rows: rows[0].update(discarded=True), "no_retry_or_discard"),
        (lambda rows: rows[0].update(substituted=True), "no_retry_or_discard"),
        (lambda rows: rows[0].update(replacement=True), "no_retry_or_discard"),
        (lambda rows: rows[0].update(order="SO"), "plan_order_adjacency_family_exact"),
        (
            lambda rows: rows[0]["worker"].update(phase_order=["real", "warmup"]),
            "phase_integrity",
        ),
        (lambda rows: rows[0]["worker"]["phases"].pop("real"), "phase_integrity"),
        (lambda rows: rows[0]["worker"]["telemetry"]["raw"].pop(), "telemetry_complete"),
    ),
)
def test_integrity_failures_are_hard_overall_failures(mutation: object, failed_check: str) -> None:
    invocations = _invocations()
    mutation(invocations)  # type: ignore[operator]
    result = TOOL._evaluate_invocations(invocations)
    assert result["checks"][failed_check] is False
    assert result["overall_pass"] is False


def test_declared_worker_failure_is_hard_failure() -> None:
    invocations = _invocations()
    invocations[0]["worker"]["failure"] = "synthetic failure"
    result = TOOL._evaluate_invocations(invocations)
    assert result["checks"]["workers_declared_no_failure"] is False
    assert result["overall_pass"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda rows: rows[0]["worker"]["phases"]["real"].pop("output_sha256"),
        lambda rows: rows[0]["worker"]["phases"]["real"].pop("output_shape"),
        lambda rows: rows[0]["worker"]["phases"]["real"].pop("output_dtype"),
        lambda rows: rows[0]["worker"]["environment"].pop("device_uuid"),
        lambda rows: rows[0]["worker"]["environment"].pop("torch"),
        lambda rows: rows[0]["worker"]["phases"].update(extra={}),
    ),
)
def test_missing_output_environment_and_extra_phase_fail_closed(mutation: object) -> None:
    invocations = _invocations()
    mutation(invocations)  # type: ignore[operator]
    assert TOOL._evaluate_invocations(invocations)["overall_pass"] is False


def test_malformed_or_inconsistent_telemetry_fails_closed() -> None:
    malformed = _invocations()
    malformed[0]["worker"]["telemetry"]["raw"][0]["gpu"]["power.draw"] = "not-a-number"
    assert TOOL._evaluate_invocations(malformed)["checks"]["telemetry_complete"] is False

    inconsistent = _invocations()
    inconsistent[0]["worker"]["telemetry"]["summary"]["sample_count"] = 3
    assert TOOL._evaluate_invocations(inconsistent)["checks"]["telemetry_complete"] is False


def test_measurement_source_requires_clean_unchanged_commit_and_hashes() -> None:
    hashes = {"tool_script": "sha256:" + "a" * 64}
    stable = {
        "measured_commit": "a" * 40,
        "measured_hashes": hashes,
        "current_commit": "a" * 40,
        "current_hashes": copy.deepcopy(hashes),
        "worktree_status": "",
    }
    assert TOOL._measurement_source_is_stable(**stable) is True
    for field, replacement in (
        ("current_commit", "b" * 40),
        ("current_hashes", {"tool_script": "sha256:" + "b" * 64}),
        ("worktree_status", " M tools/measure_scheduled_sampling.py\n"),
    ):
        changed = {**stable, field: replacement}
        assert TOOL._measurement_source_is_stable(**changed) is False


def test_future_schema_has_explicit_provenance_and_deterministic_valid_json() -> None:
    document = TOOL._build_document(
        _invocations(),
        device_ordinal=0,
        measured_commit="a" * 40,
        source_base_commit="b" * 40,
        provenance_label="authorized counterbalanced baseline",
        source_hashes=TOOL._source_hashes(),
    )
    assert document["schema"] == 2
    assert document["provenance"] == {
        "measured_commit": "a" * 40,
        "source_base_commit": "b" * 40,
        "source_base_label": "authorized counterbalanced baseline",
        "ordinary_baseline": "adjacent fresh-process ordinary invocation at measured_commit",
    }
    assert "parent_commit" not in document
    assert document["acceptance"]["primary_clock"] == "synchronized perf_counter_ns wall time"
    assert document["acceptance"]["cuda_events_are_diagnostic_only"] is True
    assert document["acceptance"]["telemetry_is_diagnostic_only"] is True
    assert document["overall_pass"] is True
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert encoded == json.dumps(json.loads(encoded), indent=2, sort_keys=True) + "\n"
    assert all(ord(character) < 128 for character in encoded)


def test_historical_record_is_byte_preserved_and_pins_measured_source() -> None:
    record_bytes = CURRENT_RECORD.read_bytes()
    assert hashlib.sha256(record_bytes).hexdigest() == (
        "566ec817756378b222e2c5f5f3e9315cba4e2da471c96a2fe915c0acec9bdf2f"
    )
    record = json.loads(record_bytes)
    assert record["implementation_source_sha256"] == HISTORICAL_MEASURED_SOURCE_SHA256, (
        f"schema-1 record must remain bound to measured commit {HISTORICAL_MEASURED_COMMIT}"
    )
    assert TOOL.OUT != TOOL.CURRENT_RECORD
    assert TOOL.CURRENT_RECORD == CURRENT_RECORD


def test_command_or_environment_identity_drift_fails_closed() -> None:
    command = _invocations()
    command[0]["command"] = ["different"]
    assert _checks(command)["commands_exact"] is False

    environment = _invocations()
    environment[1]["worker"]["environment"]["driver"] = "different"
    result = TOOL._evaluate_invocations(environment)
    assert result["checks"]["flux_pair_1_identity_exact"] is False
    assert result["overall_pass"] is False

    device = _invocations()
    device[0]["cuda_visible_devices"] = "1"
    result = TOOL._evaluate_invocations(device)
    assert result["checks"]["visible_device_ordinal_exact"] is False
    assert result["overall_pass"] is False


def test_phase_output_and_timestamp_drift_fail_closed() -> None:
    output = _invocations()
    output[0]["worker"]["phases"]["warmup"]["output_sha256"] = "sha256:" + "c" * 64
    result = TOOL._evaluate_invocations(output)
    assert result["checks"]["warmup_real_output_exact_each_worker"] is False
    assert result["overall_pass"] is False

    timestamp = _invocations()
    timestamp[0]["started_at"] = ""
    result = TOOL._evaluate_invocations(timestamp)
    assert result["checks"]["timestamps_complete"] is False
    assert result["overall_pass"] is False


def test_invocation_fixture_is_independently_mutable() -> None:
    original = _invocations()
    mutated = copy.deepcopy(original)
    mutated[0]["worker"]["phases"]["real"]["wall_elapsed_ns"] = 999
    assert original[0]["worker"]["phases"]["real"]["wall_elapsed_ns"] == 100
