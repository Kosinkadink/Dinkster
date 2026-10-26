from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.workflow_benchmark import dinkster_execution_receipt
from tools.workflow_benchmark_report import (
    json_digest,
    native_job_request,
    validate_workflow_files,
    validate_workload,
    workflow_comparability_problems,
)


def test_dinkster_execution_receipt_binds_run_extensions_and_source() -> None:
    accepted = {
        "runId": "run-466",
        "extensionSnapshotDigest": "sha256:" + "a" * 64,
    }

    assert dinkster_execution_receipt(accepted, "b" * 40) == {
        "runId": "run-466",
        "extensionSnapshotDigest": "sha256:" + "a" * 64,
        "dinksterGitHead": "b" * 40,
    }


@pytest.mark.parametrize("missing", ["runId", "extensionSnapshotDigest"])
def test_dinkster_execution_receipt_rejects_missing_identity(missing: str) -> None:
    accepted = {
        "runId": "run-466",
        "extensionSnapshotDigest": "sha256:" + "a" * 64,
    }
    del accepted[missing]

    with pytest.raises(ValueError, match="missing execution provenance"):
        dinkster_execution_receipt(accepted, "b" * 40)


@pytest.fixture
def native_workload() -> dict[str, Any]:
    source = {"7": {"class_type": "RandomNoise", "inputs": {"noise_seed": 17}}}
    export = {
        "api_sha256": "a" * 64,
        "frontend_commit": "f" * 40,
        "jobs": [
            {
                "seed": seed,
                "source_graph_sha256": json_digest(
                    {"7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}}
                ),
                "request": {
                    "clientId": "editor",
                    "jobId": "captured",
                    "graph": {
                        "nodes": {
                            "7": {"nodeType": "RandomNoise", "inputs": {"noise_seed": seed}},
                            "8": {
                                "nodeType": "SaveImage",
                                "inputs": {
                                    "images": {"$link": {"node": "7", "output": "noise"}},
                                    "filename_prefix": "retained",
                                },
                            },
                        }
                    },
                    "targets": ["8"],
                },
            }
            for seed in (17, 18)
        ],
    }
    digest = hashlib.sha256(json.dumps(export).encode()).hexdigest()
    return {
        "graph": source,
        "graph_sha256": json_digest(source),
        "api_sha256": "a" * 64,
        "seed_inputs": ["7.noise_seed"],
        "seeds": [17, 18],
        "native_jobs": export,
        "native_jobs_sha256": digest,
        "provenance": {
            "api_sha256": "a" * 64,
            "native_jobs_sha256": digest,
            "template_commit": "b" * 40,
            "template_sha256": "c" * 64,
            "template_path": "workflow.json",
            "exporter": "frontend compiler",
        },
    }


def test_native_jobs_keep_compiler_graph_and_distinct_seed_requests(
    native_workload: dict[str, Any],
) -> None:
    original = copy.deepcopy(native_workload)
    validate_workload(native_workload)
    for index, seed in enumerate((17, 18)):
        request = native_job_request(native_workload, index)
        assert request == {
            **original["native_jobs"]["jobs"][index]["request"],
            "clientId": "workflow-benchmark",
            "jobId": f"run-{index}",
        }
        assert request["graph"]["nodes"]["7"]["inputs"]["noise_seed"] == seed
        request["graph"]["nodes"]["8"]["inputs"]["filename_prefix"] = "changed"
    assert native_workload == original


def test_native_job_metadata_does_not_change_source_workflow_comparison(
    native_workload: dict[str, Any],
) -> None:
    ours = {
        "system": "dinkster",
        "workload": native_workload,
        "artifacts": [],
        "sources": {name: {"commit": "a" * 40} for name in ("harness", "dinkster", "comfyui")},
    }
    theirs = copy.deepcopy(ours)
    theirs["system"] = "comfyui"
    del theirs["workload"]["native_jobs"]
    del theirs["workload"]["native_jobs_sha256"]
    assert workflow_comparability_problems(ours, theirs) == ()
    theirs["workload"]["seeds"] = [17, 19]
    assert workflow_comparability_problems(ours, theirs) == ("workload differs",)


@pytest.mark.parametrize("difference", ["source", "seed", "target", "provenance", "count"])
def test_native_job_export_rejects_mismatched_warm_run(
    native_workload: dict[str, Any],
    difference: str,
) -> None:
    export = native_workload["native_jobs"]
    row = export["jobs"][1]
    if difference == "source":
        row["source_graph_sha256"] = "0" * 64
    elif difference == "seed":
        row["request"]["graph"]["nodes"]["7"]["inputs"]["noise_seed"] = 17
    elif difference == "target":
        row["request"]["targets"] = ["missing"]
    elif difference == "provenance":
        export["frontend_commit"] = "not-a-commit"
    else:
        export["jobs"].pop()
    with pytest.raises(ValueError):
        validate_workload(native_workload)


def test_retained_native_job_requests_are_validated_on_every_run(
    native_workload: dict[str, Any],
    tmp_path: Path,
) -> None:
    (tmp_path / "native-jobs.json").write_text(json.dumps(native_workload["native_jobs"]))
    (tmp_path / "samples.jsonl").write_text('{"tree_rss_bytes": 42, "device": null}\n')
    report = {
        "system": "dinkster",
        "report_version": 1,
        "workload": native_workload,
        "sources": {"dinkster": {"commit": "d" * 40}},
        "runs": [],
        "memory": {
            "samples_file": "samples.jsonl",
            "count": 1,
            "sampled_peak_tree_rss_bytes": 42,
            "sampled_peak_device_used_bytes": None,
        },
    }
    for index in range(2):
        identity = f"job-{index}"
        body = native_job_request(native_workload, index)
        accepted = {
            "jobRef": identity,
            "runId": f"run-{index}",
            "extensionSnapshotDigest": "sha256:" + "e" * 64,
        }
        for name, value in (
            ("submitted", body),
            ("accepted", accepted),
            ("history", {"state": "completed", "jobRef": identity}),
            (
                "journal",
                {
                    "records": [
                        {
                            "name": "job_state",
                            "streamId": "execution-run/" + identity,
                            "payload": {"jobRef": identity, "state": "completed"},
                        }
                    ]
                },
            ),
        ):
            (tmp_path / f"{index}-{name}.json").write_text(json.dumps(value))
        report["runs"].append(
            {
                "submitted_file": f"{index}-submitted.json",
                "native_request_sha256": json_digest(body),
                "accepted_file": f"{index}-accepted.json",
                "history_file": f"{index}-history.json",
                "journal_file": f"{index}-journal.json",
                "job_id": identity,
                "outputs": [],
                "execution_receipt": dinkster_execution_receipt(accepted, "d" * 40),
            }
        )
    assert validate_workflow_files(report, tmp_path) == ()
    body = json.loads((tmp_path / "1-submitted.json").read_text())
    body["targets"] = ["7"]
    (tmp_path / "1-submitted.json").write_text(json.dumps(body))
    assert validate_workflow_files(report, tmp_path) == (
        "invalid retained workflow evidence: retained native request differs",
    )
