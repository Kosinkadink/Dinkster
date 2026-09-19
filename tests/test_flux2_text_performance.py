from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.inference_parity.flux2_text_performance import (
    COMFYUI_COMMIT,
    COMFYUI_KITCHEN_VERSION,
    DEVICE_UUID,
    DINKSTER_AIMDO_VERSION,
    DINKSTER_KITCHEN_VERSION,
    MEASURED_RUNS,
    TEXT_ENCODER_REVISION,
    TEXT_ENCODER_SHA256,
    TEXT_ENCODER_SIZE,
    ComparisonError,
    _workload,
    build_verdict,
    run_compare,
)


def _record(tmp_path: Path, engine: str, value: float, index: int) -> dict[str, Any]:
    output = np.full((1, 2), index % 2, dtype=np.float32)
    output_path = tmp_path / f"{engine}-{index}.npy"
    np.save(output_path, output, allow_pickle=False)
    output_digest = hashlib.sha256(output.tobytes()).hexdigest()
    runs = [
        {
            "index": 0,
            "output_sha256": output_digest,
            "phase": "cold",
            "seconds": value * 2,
        },
        {
            "index": 0,
            "output_sha256": output_digest,
            "phase": "warmup",
            "seconds": value,
        },
        *(
            {
                "index": run,
                "output_sha256": output_digest,
                "phase": "measured",
                "seconds": value,
            }
            for run in range(MEASURED_RUNS)
        ),
    ]
    receipts: dict[str, object] = {
        "artifact": {
            "revision": TEXT_ENCODER_REVISION,
            "sha256": TEXT_ENCODER_SHA256,
            "size_bytes": TEXT_ENCODER_SIZE,
        },
        "commit": COMFYUI_COMMIT if engine == "comfyui" else "candidate",
        "cuda": "13.0",
        "device_uuid": DEVICE_UUID,
        "driver": "580.95.05",
        "gpu": "NVIDIA GeForce RTX 4090",
        "gpu_total_memory": 25_250_627_584,
        "python": "3.12.3",
        "torch": "2.13.0+cu130",
    }
    if engine == "comfyui":
        receipts.update(
            {
                "comfy_aimdo": "0.4.13",
                "comfy_kitchen": COMFYUI_KITCHEN_VERSION,
                "patcher": "ModelPatcherDynamic",
                "precision": {"activation": "float32", "storage": ["bfloat16"]},
            }
        )
    else:
        receipts.update(
            {
                "dinkster_aimdo": DINKSTER_AIMDO_VERSION,
                "dinkster_kitchen": DINKSTER_KITCHEN_VERSION,
                "demand_control_sha256": output_digest,
                "hybrid_equals_demand_control": True,
                "precision": {"activation": "bfloat16", "storage": ["bfloat16"]},
            }
        )
    return {
        "engine": engine,
        "output": {
            "dtype": str(output.dtype),
            "path": str(output_path),
            "sha256": output_digest,
            "shape": list(output.shape),
        },
        "receipts": receipts,
        "summary": {"runs": runs},
        "workload": _workload([1, 2, 3]),
    }


def _records(tmp_path: Path, *, dinkster_seconds: float = 0.9) -> list[dict[str, Any]]:
    return [
        _record(tmp_path, "comfyui", 1.0, 0),
        _record(tmp_path, "dinkster", dinkster_seconds, 1),
        _record(tmp_path, "dinkster", dinkster_seconds, 3),
        _record(tmp_path, "comfyui", 1.0, 2),
    ]


def test_verdict_requires_strictly_faster_balanced_dinkster_records(tmp_path: Path) -> None:
    verdict = build_verdict(_records(tmp_path), dinkster_commit="candidate")
    assert verdict["pass"] is True
    assert verdict["performance"]["dinkster_over_comfyui_ratio"] == pytest.approx(0.9)

    tied = build_verdict(_records(tmp_path, dinkster_seconds=1.0), dinkster_commit="candidate")
    assert tied["pass"] is False


def test_verdict_rejects_runtime_and_control_drift(tmp_path: Path) -> None:
    records = _records(tmp_path)
    changed = copy.deepcopy(records)
    changed[2]["receipts"]["driver"] = "different"
    with pytest.raises(ComparisonError, match="driver"):
        build_verdict(changed, dinkster_commit="candidate")

    changed = copy.deepcopy(records)
    changed[0]["receipts"]["comfy_kitchen"] = DINKSTER_KITCHEN_VERSION
    with pytest.raises(ComparisonError, match="ComfyUI source"):
        build_verdict(changed, dinkster_commit="candidate")

    changed = copy.deepcopy(records)
    changed[1]["receipts"]["dinkster_kitchen"] = COMFYUI_KITCHEN_VERSION
    with pytest.raises(ComparisonError, match="Dinkster source"):
        build_verdict(changed, dinkster_commit="candidate")

    changed = copy.deepcopy(records)
    changed[1]["receipts"]["hybrid_equals_demand_control"] = False
    with pytest.raises(ComparisonError, match="demand-paged control"):
        build_verdict(changed, dinkster_commit="candidate")


def test_verdict_rejects_nondeterministic_process_output(tmp_path: Path) -> None:
    changed = _records(tmp_path)
    changed[1]["summary"]["runs"][2]["output_sha256"] = "different"
    with pytest.raises(ComparisonError, match="output changed between runs"):
        build_verdict(changed, dinkster_commit="candidate")


def test_compare_writes_failed_performance_verdict(tmp_path: Path) -> None:
    inputs: list[Path] = []
    for index, record in enumerate(_records(tmp_path, dinkster_seconds=1.1)):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(record))
        inputs.append(path)
    output = tmp_path / "verdict.json"

    result = run_compare(
        argparse.Namespace(inputs=inputs, output=output, dinkster_commit="candidate")
    )

    assert result == 1
    assert json.loads(output.read_text())["pass"] is False
