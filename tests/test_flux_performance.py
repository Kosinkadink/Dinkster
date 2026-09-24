from __future__ import annotations

import argparse
import copy
import json
from typing import Any

import pytest
from tools.inference_parity.flux_performance import (  # pyright: ignore[reportMissingImports]
    ComparisonError,
    _workload_receipt,
    build_verdict,
    run_compare,
)


def _record(engine: str, value: float) -> dict[str, Any]:
    return {
        "engine": engine,
        "warmup": {
            "image_sha256": f"{engine}-image",
            "latent_sha256": f"{engine}-latent",
        },
        "measured": [
            {
                "decode_seconds": value / 10,
                "end_to_end_seconds": value * 1.1,
                "image_sha256": f"{engine}-image",
                "latent_sha256": f"{engine}-latent",
                "sample_seconds": value,
            }
            for _ in range(5)
        ],
        "receipts": {
            "attention_backend": "pytorch-sdpa" if engine == "comfyui" else "sdpa",
            "checkpoint": {
                "sha256": "8e91b68084b53a7fc44ed2a3756d821e355ac1a7b6fe29be760c1db532f3d88a",
                "size_bytes": 17_246_524_772,
            },
            "commit": (
                "947c2749dd04c51ef0e21b069544d8b0b4f9b411" if engine == "comfyui" else "candidate"
            ),
            "cuda": "13.0",
            "device_uuid": "GPU-666d1242-9c20-341c-73ea-e63770947451",
            "driver": "580.95.05",
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_total_memory": 25_375_080_448,
            "initial_noise_sha256": "noise",
            "normal_schedule_float_hex": [
                "0x1.0000000000000p+0",
                "0x1.8000000000000p-1",
                "0x1.0000000000000p-1",
                "0x1.0000000000000p-2",
                "0x0.0p+0",
            ],
            "precision": (
                {
                    "diffusion_actual": "bfloat16",
                    "requested_text": "float32",
                    "requested_vae": "float32",
                    "text_actual": "float32",
                    "vae_actual": "float32",
                }
                if engine == "comfyui"
                else {
                    "clip_l_actual": "float32",
                    "diffusion_actual": "bfloat16",
                    "fp8_matmul": False,
                    "requested_diffusion": "bfloat16",
                    "requested_text": "float32",
                    "requested_vae": "float32",
                    "t5xxl_actual": "float32",
                    "vae_actual": "float32",
                }
            ),
            "python": "3.12.3",
            "torch": "2.13.0+cu130",
            **(
                {"attention_policy": "auto", "window_plan_digest": None}
                if engine == "dinkster"
                else {}
            ),
        },
        "workload": _workload_receipt(),
    }


def _build(records: list[dict[str, Any]]) -> dict[str, Any]:
    return build_verdict(records, dinkster_commit="candidate")


def test_build_verdict_accepts_balanced_records_within_noise() -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.005)]
    records += [_record("dinkster", 1.005), _record("comfyui", 1.0)]
    verdict = _build(records)
    assert verdict["pass"] is True
    assert verdict["process_records"] == records
    assert verdict["metrics"]["sample_seconds"]["dinkster_over_comfyui_ratio"] == pytest.approx(
        1.005
    )


def test_build_verdict_rejects_deficit_beyond_noise() -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.02)]
    records += [_record("dinkster", 1.02), _record("comfyui", 1.0)]
    verdict = _build(records)
    assert verdict["pass"] is False
    assert verdict["metrics"]["sample_seconds"]["pass"] is False


def test_build_verdict_rejects_runtime_drift() -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.0)]
    records += [_record("dinkster", 1.0), _record("comfyui", 1.0)]
    changed = copy.deepcopy(records)
    changed[2]["receipts"]["driver"] = "different"
    with pytest.raises(ComparisonError, match="driver"):
        _build(changed)


@pytest.mark.parametrize(
    ("record_index", "precision_key"),
    ((0, "diffusion_actual"), (0, "vae_actual"), (1, "clip_l_actual")),
)
def test_build_verdict_rejects_actual_precision_drift(
    record_index: int, precision_key: str
) -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.0)]
    records += [_record("dinkster", 1.0), _record("comfyui", 1.0)]
    changed = copy.deepcopy(records)
    changed[record_index]["receipts"]["precision"][precision_key] = "float16"
    with pytest.raises(ComparisonError, match="precision"):
        _build(changed)


def test_build_verdict_rejects_nondeterministic_output() -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.0)]
    records += [_record("dinkster", 1.0), _record("comfyui", 1.0)]
    changed = copy.deepcopy(records)
    changed[1]["measured"][1]["latent_sha256"] = "different"
    with pytest.raises(ComparisonError, match="not deterministic across processes"):
        _build(changed)


def test_build_verdict_rejects_cross_process_output_drift() -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.0)]
    records += [_record("dinkster", 1.0), _record("comfyui", 1.0)]
    changed = copy.deepcopy(records)
    for run in changed[2]["measured"]:
        run["latent_sha256"] = "different"
    with pytest.raises(ComparisonError, match="not deterministic across processes"):
        _build(changed)


def test_build_verdict_rejects_source_attention_and_run_count_drift() -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.0)]
    records += [_record("dinkster", 1.0), _record("comfyui", 1.0)]
    changed = copy.deepcopy(records)
    changed[3]["receipts"]["commit"] = "different"
    with pytest.raises(ComparisonError, match="ComfyUI source"):
        _build(changed)
    changed = copy.deepcopy(records)
    changed[1]["receipts"]["attention_backend"] = "different"
    with pytest.raises(ComparisonError, match="Dinkster source"):
        _build(changed)
    changed = copy.deepcopy(records)
    changed[0]["measured"].pop()
    with pytest.raises(ComparisonError, match="exactly 5"):
        _build(changed)


def test_build_verdict_rejects_missing_warmup_schedule_and_predeclared_commit() -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.0)]
    records += [_record("dinkster", 1.0), _record("comfyui", 1.0)]
    changed = copy.deepcopy(records)
    del changed[0]["warmup"]
    with pytest.raises(ComparisonError, match="warmup"):
        _build(changed)
    changed = copy.deepcopy(records)
    changed[0]["receipts"]["normal_schedule_float_hex"].pop()
    with pytest.raises(ComparisonError, match="5 values ending at zero"):
        _build(changed)
    with pytest.raises(ComparisonError, match="Dinkster source"):
        build_verdict(records, dinkster_commit="different")


def test_run_compare_returns_failure_for_failed_gate(tmp_path: Any) -> None:
    records = [_record("comfyui", 1.0), _record("dinkster", 1.02)]
    records += [_record("dinkster", 1.02), _record("comfyui", 1.0)]
    inputs = []
    for index, record in enumerate(records):
        path = tmp_path / f"record-{index}.json"
        path.write_text(json.dumps(record))
        inputs.append(path)
    output = tmp_path / "verdict.json"
    assert (
        run_compare(argparse.Namespace(inputs=inputs, output=output, dinkster_commit="candidate"))
        == 1
    )
    assert json.loads(output.read_text())["pass"] is False
