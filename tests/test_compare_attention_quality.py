"""Output-level attention quality comparison contracts."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_backend_env import complete_benchmark_report

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "compare_attention_quality.py"
_SPEC = importlib.util.spec_from_file_location("compare_attention_quality", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
compare_attention_quality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_attention_quality)


def _capture(path: Path, array: np.ndarray, stride: int | None) -> dict[str, Any]:
    np.save(path, array.astype(np.float32))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dtype": "float32",
        "source_shape": list(array.shape),
        "captured_shape": list(array.shape),
        "spatial_stride": stride,
    }


def _report(tmp_path: Path, name: str, policy: str, offset: float = 0.0) -> dict[str, Any]:
    image = np.linspace(0.0, 1.0, 2 * 8 * 16 * 3, dtype=np.float32).reshape(2, 8, 16, 3)
    audio = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 2, 16)
    image = image + offset
    audio = audio + offset
    report = complete_benchmark_report("cuda", "minimax_h3")
    report["torch"] = {"version": "2.13.0+cu130", "backend_runtime": "cuda 13.0"}
    report["devices"] = [
        {
            "index": 0,
            "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "architecture": "sm_120",
            "total_memory": 102_642_761_728,
        }
    ]
    report["dinkster"] = {"commit": "b" * 40, "clean": True}
    providers = {
        "sdpa": [["torch", "2.13.0+cu130"]],
        "comfy_kitchen_int8": [
            ["comfy-kitchen", "0.2.32"],
            ["torch", "2.13.0+cu130"],
        ],
        "sage": [["sageattention", "2.2.0.post1"], ["torch", "2.13.0+cu130"]],
    }
    report["attention"] = {
        "requested_policy": policy,
        "scope": ["diffusion:flux"],
        "route_token": {
            "version": 1,
            "routes": [
                {
                    "role": role,
                    "primary": policy,
                    "fallback": None if policy == "sdpa" else "sdpa",
                }
                for role in ("unet", "flux", "vae", "clip", "t5", "qwen")
            ],
            "providerVersions": providers[policy],
            "adapterContractRevision": "dinkster.attention-kernel.v1",
            "deviceKind": "cuda",
            "deviceSm": 120,
            "sdpaTorchRuntime": "2.13.0",
            "requestedPolicy": policy,
        },
    }
    report["quality_capture"] = {
        "version": 1,
        "seed": report["workload"]["seed"] + report["workload"]["warm_runs"] + 1,
        "image": _capture(tmp_path / f"{name}-image.npy", image, 4),
        "audio": _capture(tmp_path / f"{name}-audio.npy", audio, None),
        "audio_sample_rate": 32_000,
    }
    return report


def test_identical_captures_have_perfect_metrics(tmp_path: Path) -> None:
    baseline = _report(tmp_path, "baseline", "sdpa")
    candidate = _report(tmp_path, "candidate", "sage")

    assert compare_attention_quality.comparability_problems(baseline, candidate) == ()
    comparison = compare_attention_quality.build_comparison(baseline, candidate)

    assert comparison["acceptance_thresholds"] is None
    assert comparison["metrics"]["image"]["cosine_similarity"] == pytest.approx(1.0)
    assert comparison["metrics"]["image"]["ssim_8x8_data_range_1"] == pytest.approx(1.0)
    assert comparison["metrics"]["image"]["max_absolute_error"] == 0.0
    assert comparison["metrics"]["audio"]["cosine_similarity"] == pytest.approx(1.0)


def test_changed_capture_reports_quality_without_a_hidden_gate(tmp_path: Path) -> None:
    baseline = _report(tmp_path, "baseline", "sdpa")
    candidate = _report(tmp_path, "candidate", "comfy_kitchen_int8", offset=0.01)

    comparison = compare_attention_quality.build_comparison(baseline, candidate)
    image = comparison["metrics"]["image"]

    assert image["cosine_similarity"] < 1.0
    assert image["ssim_8x8_data_range_1"] < 1.0
    assert image["mean_absolute_error"] == pytest.approx(0.01, abs=1e-7)
    assert comparison["acceptance_thresholds"] is None


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda report: report["dinkster"].update(commit="c" * 40), "source commits"),
        (lambda report: report["attention"].update(requested_policy="sdpa"), "candidate attention"),
        (lambda report: report["workload"].update(seed=2), "workload differs"),
        (
            lambda report: report["quality_capture"]["image"].update(spatial_stride=2),
            "image.spatial_stride",
        ),
    ],
)
def test_comparability_rejects_nonmatched_evidence(
    tmp_path: Path, mutation: Any, expected: str
) -> None:
    baseline = _report(tmp_path, "baseline", "sdpa")
    candidate = _report(tmp_path, "candidate", "sage")
    mutation(candidate)
    assert any(
        expected in problem
        for problem in compare_attention_quality.comparability_problems(baseline, candidate)
    )


def test_capture_digest_is_verified_before_metrics(tmp_path: Path) -> None:
    baseline = _report(tmp_path, "baseline", "sdpa")
    candidate = _report(tmp_path, "candidate", "sage")
    candidate_path = Path(candidate["quality_capture"]["image"]["path"])
    candidate_path.write_bytes(candidate_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="size does not match"):
        compare_attention_quality.build_comparison(baseline, candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        np.empty((0, 8, 8, 3), dtype=np.float32),
        np.full((1, 8, 8, 3), np.nan, dtype=np.float32),
    ],
)
def test_metrics_reject_empty_or_nonfinite_captures(candidate: np.ndarray) -> None:
    with pytest.raises(ValueError, match="empty|finite"):
        compare_attention_quality.array_metrics(candidate, candidate, image=True)


def test_cli_writes_structured_observational_result(tmp_path: Path) -> None:
    baseline = _report(tmp_path, "baseline", "sdpa")
    candidate = _report(tmp_path, "candidate", "sage")
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))

    assert (
        compare_attention_quality.main(
            [
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--json",
                str(output),
            ]
        )
        == 0
    )
    written = json.loads(output.read_text())
    assert written["baseline_policy"] == "sdpa"
    assert written["candidate_policy"] == "sage"
    assert written["source_commit"] == "b" * 40


def test_incomplete_report_is_rejected_before_comparison(tmp_path: Path) -> None:
    baseline = _report(tmp_path, "baseline", "sdpa")
    candidate = _report(tmp_path, "candidate", "sage")
    candidate.pop("memory")

    assert any(
        "candidate report is incomplete: memory section missing" in problem
        for problem in compare_attention_quality.comparability_problems(baseline, candidate)
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda token: token.update(providerVersions=[["fabricated-provider", "not-installed"]]),
            "sageattention provider",
        ),
        (
            lambda token: token.update(deviceKind="cpu", deviceSm=None),
            "device kind differs",
        ),
        (
            lambda token: token.update(
                providerVersions=[["sageattention", "2.2.0.post1"], ["torch", "0.0"]],
                sdpaTorchRuntime="0.0",
            ),
            "torch runtime differs",
        ),
    ],
)
def test_dinkster_authentication_rejects_forged_provider_runtime_or_device(
    tmp_path: Path, mutation: Any, expected: str
) -> None:
    baseline = _report(tmp_path, "baseline", "sdpa")
    candidate = _report(tmp_path, "candidate", "sage")
    mutation(candidate["attention"]["route_token"])

    assert any(
        expected in problem
        for problem in compare_attention_quality.comparability_problems(baseline, candidate)
    )


def test_comfyui_attention_authentication_requires_observed_provider_execution() -> None:
    report = {
        "system": "comfyui",
        "torch": {"version": "2.13.0+cu130", "backend_runtime": "cuda 13.0"},
        "attention": {
            "requested_policy": "sage",
            "selected_policy": "sage",
            "provider_versions": [
                ["torch", "2.13.0+cu130"],
                ["dinkster-kitchen", "2.2.0.post1"],
            ],
            "provider_module": {
                "module": "sageattention",
                "distribution": "dinkster-kitchen",
                "version": "2.2.0.post1",
                "authenticated": True,
            },
            "execution": {
                "policy": "sage",
                "selected_calls": 12,
                "provider_attempts": 10,
                "provider_successes": 10,
                "provider_exceptions": 0,
                "fallback_calls": 2,
            },
        },
    }
    assert (
        compare_attention_quality._attention_authentication_problems(report, "candidate", "sage")
        == []
    )

    report["attention"]["execution"]["provider_successes"] = 0
    report["attention"]["execution"]["provider_attempts"] = 0
    assert any(
        "did not execute" in problem
        for problem in compare_attention_quality._attention_authentication_problems(
            report, "candidate", "sage"
        )
    )


def test_comfyui_authentication_rejects_forged_int8_provider() -> None:
    report = {
        "system": "comfyui",
        "torch": {"version": "2.13.0+cu130", "backend_runtime": "cuda 13.0"},
        "attention": {
            "requested_policy": "comfy_kitchen_int8",
            "selected_policy": "comfy_kitchen_int8",
            "provider_versions": [["torch", "fabricated"]],
            "execution": {
                "policy": "comfy_kitchen_int8",
                "selected_calls": 10,
                "provider_attempts": 10,
                "provider_successes": 10,
                "provider_exceptions": 0,
                "fallback_calls": 0,
            },
        },
    }

    problems = compare_attention_quality._attention_authentication_problems(
        report, "candidate", "comfy_kitchen_int8"
    )
    assert any("torch runtime differs" in problem for problem in problems)
    assert any("comfy-kitchen provider version is missing" in problem for problem in problems)
