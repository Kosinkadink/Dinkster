from __future__ import annotations

import json
import math
import os
import statistics
from copy import deepcopy
from typing import Any

import numpy as np
from tools.inference_parity.flux_window_comparison import (
    CHECKPOINT_SHA256,
    COMFYUI_COMMIT,
    DINKSTER_COMMIT,
    MEASURED_RUNS,
    TILED_DIFFUSION_COMMIT,
    _run_metrics,
    _workload_receipt,
    build_verdict,
    compare_arrays,
)

RECORDS_ROOT = os.environ.get(
    "DINKSTER_INFERENCE_PARITY_RECORDS",
    os.path.join(os.path.dirname(__file__), "goldens", "inference_parity", "records"),
)


def _result(engine: str) -> dict[str, Any]:
    receipt = {
        "checkpoint": {"sha256": CHECKPOINT_SHA256},
        "cuda": "13.0",
        "driver": "580.95.05",
        "gpu": "NVIDIA GeForce RTX 4090",
        "gpu_total_memory": 1,
        "initial_noise_sha256": "noise",
        "normal_schedule_float_hex": [
            "0x1.0p+0",
            "0x1.8p-1",
            "0x1.0p-1",
            "0x1.0p-2",
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
        "torch": "2.13.0+cu130",
    }
    if engine == "comfyui":
        receipt.update(
            {
                "comfyui_commit": COMFYUI_COMMIT,
                "tiled_diffusion_commit": TILED_DIFFUSION_COMMIT,
            }
        )
    else:
        receipt["dinkster_commit"] = DINKSTER_COMMIT
    measured = [
        {
            "end_to_end_seconds": value + 0.1,
            "image_sha256": "image",
            "latent_sha256": "latent",
            "sample_seconds": value,
        }
        for value in (0.93, 0.94, 0.95, 0.96, 0.97)
    ]
    position_record = {
        "calls": 4,
        "height_values": list(range(32)),
        "sha256": "position-ids",
        "shape": [1, 768, 3],
        "width_values": list(range(24)),
    }
    return {
        "declared_windows": (
            [
                {"height": 64, "width": 48, "x": 0, "y": 0},
                {"height": 64, "width": 48, "x": 16, "y": 0},
            ]
            if engine == "comfyui"
            else [
                {"packed_width_indices": list(range(24))},
                {"packed_width_indices": list(range(8, 32))},
            ]
        ),
        "end_to_end_range_seconds": [1.03, 1.07],
        "measured": measured,
        "median_end_to_end_seconds": 1.05,
        "median_sample_seconds": 0.95,
        "native_control": {
            "equals_full_window_image": True,
            "equals_full_window_latent": True,
        },
        "position_id_diagnostic": (
            {
                "actual_image_position_ids": [position_record],
                "matches_measured_output": True,
            }
            if engine == "comfyui"
            else {
                "controlled_change": (
                    "replace global packed image position IDs with window-local IDs"
                ),
                "global_actual_image_position_ids": [
                    position_record,
                    {**position_record, "width_values": list(range(8, 32))},
                ],
                "global_matches_measured_output": True,
                "local_actual_image_position_ids": [position_record],
                "local_image_sha256": "local-image",
                "local_latent_sha256": "local-latent",
            }
        ),
        "receipts": receipt,
        "sample_range_seconds": [0.93, 0.97],
        "window_output": {
            "image_shape": [1, 512, 512, 3],
            "latent_shape": [1, 16, 64, 64],
        },
        "workload": _workload_receipt(),
    }


def _arrays() -> dict[str, np.ndarray]:
    arrays = {
        f"{engine}-{kind}-{value}": np.ones((1, 2), dtype=np.float32)
        for engine in ("comfyui", "dinkster")
        for kind in ("full", "native", "window")
        for value in ("image", "latent")
    }
    for value in ("image", "latent"):
        arrays[f"comfyui-window-{value}"] = np.array([[0.0, 1.0]], dtype=np.float32)
        arrays[f"dinkster-window-{value}"] = np.array([[1.0, 0.0]], dtype=np.float32)
        arrays[f"dinkster-local-window-{value}"] = np.array([[0.0, 1.0]], dtype=np.float32)
    return arrays


def test_compare_arrays_pins_exact_and_nonexact_metrics() -> None:
    left = np.array([0.0, 1.0], dtype=np.float32)
    exact = compare_arrays(left, left.copy())
    changed = compare_arrays(left, np.array([0.0, 2.0], dtype=np.float32))

    assert exact["exact"] is True
    assert exact["max_abs"] == 0.0
    assert exact["relative_l2_to_left"] == 0.0
    assert changed["exact"] is False
    assert changed["max_abs"] == 1.0
    assert changed["mean_abs"] == 0.5
    assert changed["relative_l2_to_left"] == 1.0


def test_measured_sampling_and_decode_run_in_inference_mode() -> None:
    active = False

    class InferenceMode:
        def __enter__(self) -> None:
            nonlocal active
            assert active is False
            active = True

        def __exit__(self, *_: object) -> None:
            nonlocal active
            active = False

    class Cuda:
        @staticmethod
        def synchronize(_device: object) -> None:
            pass

        @staticmethod
        def max_memory_allocated(_device: object) -> int:
            return 1

    class Torch:
        cuda = Cuda()

        @staticmethod
        def inference_mode() -> InferenceMode:
            return InferenceMode()

    def sample() -> np.ndarray:
        assert active is True
        return np.ones((1,), dtype=np.float32)

    def decode(sampled: np.ndarray) -> np.ndarray:
        assert active is True
        return sampled

    latent, image, metrics = _run_metrics(
        Torch(), object(), sample, decode, lambda value: value, lambda value: value
    )

    assert active is False
    assert np.array_equal(latent, image)
    assert metrics["peak_allocated_bytes"] == 1


def test_verdict_requires_pins_controls_and_performance_floor() -> None:
    comfyui = _result("comfyui")
    dinkster = _result("dinkster")

    verdict = build_verdict(comfyui, dinkster, _arrays())

    assert verdict["overall_pass"] is True
    assert verdict["behavioral"] == {
        "all_outputs_finite": True,
        "deterministic_measured_outputs": True,
        "exact_initial_noise": True,
        "exact_normal_schedule": True,
        "matching_environment": True,
        "matching_geometry": True,
        "matching_precision": True,
        "matching_workload": True,
        "measured_run_count": True,
        "one_window_equals_native_within_each_engine": True,
        "pinned_sources": True,
        "position_id_evidence": True,
        "position_id_isolation": True,
        "position_id_semantics": {
            "comfyui_tiled_diffusion": (
                "each cropped Flux call derives local packed positions starting at zero"
            ),
            "dinkster": "each cropped Flux call uses its declared global packed positions",
        },
    }
    assert verdict["performance"]["pass"] is True

    mismatched = deepcopy(dinkster)
    mismatched["receipts"]["initial_noise_sha256"] = "different"
    assert build_verdict(comfyui, mismatched, _arrays())["overall_pass"] is False

    mismatched_source = deepcopy(dinkster)
    mismatched_source["receipts"]["dinkster_commit"] = "not-the-pinned-source"
    verdict = build_verdict(comfyui, mismatched_source, _arrays())
    assert verdict["behavioral"]["pinned_sources"] is False
    assert verdict["overall_pass"] is False
    verdict = build_verdict(
        comfyui,
        mismatched_source,
        _arrays(),
        dinkster_commit="not-the-pinned-source",
    )
    assert verdict["behavioral"]["pinned_sources"] is True
    assert verdict["overall_pass"] is True

    slower = deepcopy(dinkster)
    for item, value in zip(slower["measured"], (1.11, 1.12, 1.13, 1.14, 1.15), strict=True):
        item["sample_seconds"] = value
    assert build_verdict(comfyui, slower, _arrays())["overall_pass"] is False

    no_position_isolation = _arrays()
    no_position_isolation["dinkster-local-window-latent"] = no_position_isolation[
        "dinkster-window-latent"
    ]
    verdict = build_verdict(comfyui, dinkster, no_position_isolation)
    assert verdict["behavioral"]["position_id_isolation"] is False
    assert verdict["overall_pass"] is False


def test_performance_floor_accepts_unambiguously_faster_dinkster() -> None:
    comfyui = _result("comfyui")
    dinkster = _result("dinkster")
    for item, sample in zip(dinkster["measured"], (0.71, 0.72, 0.73, 0.74, 0.75), strict=True):
        item["sample_seconds"] = sample
        item["end_to_end_seconds"] = sample + 0.1

    verdict = build_verdict(comfyui, dinkster, _arrays())

    assert verdict["performance"]["sample"]["candidate_median_seconds"] == 0.73
    assert verdict["performance"]["end_to_end"]["candidate_median_seconds"] == 0.83
    assert verdict["performance"]["pass"] is True
    assert verdict["overall_pass"] is True


def test_performance_floor_rejects_overlapping_outlier_range() -> None:
    comfyui = _result("comfyui")
    dinkster = _result("dinkster")
    sample_values = (0.97, 50.0, 50.0, 50.0, 100.0)
    for item, sample in zip(dinkster["measured"], sample_values, strict=True):
        item["sample_seconds"] = sample
        item["end_to_end_seconds"] = sample + 0.1

    verdict = build_verdict(comfyui, dinkster, _arrays())

    assert min(sample_values) <= max(item["sample_seconds"] for item in comfyui["measured"])
    assert verdict["performance"]["sample"]["candidate_median_seconds"] == 50.0
    assert verdict["performance"]["sample"]["pass"] is False
    assert verdict["overall_pass"] is False


def test_verdict_rejects_unmatched_actual_diffusion_dtype() -> None:
    comfyui = _result("comfyui")
    dinkster = _result("dinkster")
    comfyui["receipts"]["precision"]["diffusion_actual"] = "float16"

    verdict = build_verdict(comfyui, dinkster, _arrays())

    assert verdict["behavioral"]["matching_precision"] is False
    assert verdict["overall_pass"] is False


def test_committed_receipt_recomputes_balanced_performance_verdict() -> None:
    path = os.path.join(RECORDS_ROOT, "w0-flux-window-comparison", "receipt.json")
    with open(path, encoding="utf-8") as receipt_file:
        receipt = json.load(receipt_file)
    combined = receipt["performance"]["combined"]

    values = {
        engine: {
            metric: [
                value
                for attempt in receipt["attempts"]
                for value in attempt["engines"][engine][f"{metric}_seconds"]
            ]
            for metric in ("decode", "end_to_end", "sample")
        }
        for engine in ("comfyui", "dinkster")
    }
    for engine, metrics in values.items():
        for metric, measured in metrics.items():
            assert len(measured) == 2 * MEASURED_RUNS
            assert math.isclose(
                statistics.median(measured),
                combined[f"{engine}_median_{metric}_seconds"],
                abs_tol=5e-10,
                rel_tol=0.0,
            )
            assert [min(measured), max(measured)] == combined[f"{engine}_{metric}_range_seconds"]

    assert combined["dinkster_over_comfyui_sample_ratio"] == (
        combined["dinkster_median_sample_seconds"] / combined["comfyui_median_sample_seconds"]
    )
    assert combined["dinkster_over_comfyui_end_to_end_ratio"] == (
        combined["dinkster_median_end_to_end_seconds"]
        / combined["comfyui_median_end_to_end_seconds"]
    )
    for engine in ("comfyui", "dinkster"):
        assert combined[f"{engine}_median_sample_step_milliseconds"] == (
            combined[f"{engine}_median_sample_seconds"] * 1000 / receipt["workload"]["steps"]
        )
        for output in ("image", "latent"):
            hashes = [
                digest
                for attempt in receipt["attempts"]
                for digest in attempt["engines"][engine][f"{output}_sha256"]
            ]
            assert len(hashes) == 2 * MEASURED_RUNS
            assert set(hashes) == {receipt["outputs"][engine][f"window_{output}_sha256"]}

    for attempt in receipt["attempts"]:
        gate = attempt["performance_gate"]
        for metric in ("sample", "end_to_end"):
            reference = attempt["engines"]["comfyui"][f"{metric}_seconds"]
            candidate = attempt["engines"]["dinkster"][f"{metric}_seconds"]
            reference_median = statistics.median(reference)
            reference_mad = statistics.median(abs(value - reference_median) for value in reference)
            allowed = max(
                reference_median * receipt["policy"]["performance_noise_floor_fraction"],
                reference_mad * receipt["policy"]["performance_noise_mad_multiplier"],
            )
            assert math.isclose(gate[metric]["allowed_regression_seconds"], allowed)
            assert gate[metric]["candidate_median_seconds"] == statistics.median(candidate)
            assert gate[metric]["pass"] is False
        assert gate["pass"] is False

    position = receipt["behavioral"]["position_id_evidence"]
    assert position["pass"] is True
    assert position["comfyui"]["matches_measured_output"] is True
    assert position["dinkster"]["global_matches_measured_output"] is True
    assert (
        position["dinkster"]["local_latent_sha256"]
        == receipt["outputs"]["dinkster"]["local_control_latent_sha256"]
    )
    global_cosine = position["global_vs_comfyui_latent"]["cosine"]
    local_cosine = position["local_control_vs_comfyui_latent"]["cosine"]
    assert local_cosine >= receipt["policy"]["position_isolation_min_cosine"]
    assert local_cosine - global_cosine >= receipt["policy"]["position_isolation_min_cosine_gain"]

    assert receipt["schema"] == 2
    assert receipt["sources"]["comfyui_commit"] == COMFYUI_COMMIT
    assert receipt["sources"]["dinkster_commit"] == DINKSTER_COMMIT
    assert receipt["sources"]["harness_commit"] == ("ec8ba0e4761e3925898fa7141a400220e76b3d60")
    assert receipt["sources"]["tiled_diffusion_commit"] == TILED_DIFFUSION_COMMIT
    assert receipt["artifact"]["sha256"] == CHECKPOINT_SHA256
    assert receipt["verdict"] == {
        "behavioral_pass": True,
        "overall_pass": False,
        "performance_pass": False,
    }
