"""Compare one explicit attention provider with the same system's SDPA output.

The benchmark reports must describe the same exact source, hardware, workload,
artifacts, and output-capture geometry. Metrics are observational: this command
does not invent a pass threshold for an approximate attention provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from dinkster_inference import BUILTIN_FAMILIES_BY_ID
from dinkster_protocol import attention_route_token_from_wire
from dinkster_workers.backend_env import validate_benchmark_report

_CANDIDATE_POLICIES = ("dinkster_kitchen_int8", "sage")
_MEDIA = ("image", "audio")
_QUALITY_FIELDS = ("dtype", "source_shape", "captured_shape", "spatial_stride")
_MATCHED_REPORT_FIELDS = (
    "family",
    "accelerator",
    "mode",
    "placement",
    "execution_path",
    "host",
    "driver",
    "torch",
    "devices",
    "workload",
)


def _artifact_identity(report: Mapping[str, Any]) -> dict[str, tuple[object, ...]]:
    return {
        entry["role"]: (
            entry.get("sha256"),
            entry.get("bytes"),
            entry.get("url"),
        )
        for entry in report.get("artifacts", [])
        if isinstance(entry, Mapping) and isinstance(entry.get("role"), str)
    }


def _source_identity(report: Mapping[str, Any]) -> tuple[str, str, bool]:
    system = report.get("system")
    if system == "dinkster":
        source = report.get("dinkster") or {}
        return "dinkster", str(source.get("commit") or ""), source.get("clean") is True
    if system == "comfyui":
        source = report.get("comfyui") or {}
        return "comfyui", str(source.get("commit") or ""), source.get("clean") is True
    return "", "", False


def _provider_versions(value: object) -> dict[str, str] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    pairs: dict[str, str] = {}
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        name, version = pair
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or name in pairs
        ):
            return None
        pairs[name] = version
    return pairs


def _reported_torch_version(report: Mapping[str, Any]) -> str | None:
    torch_section = report.get("torch")
    version = torch_section.get("version") if isinstance(torch_section, Mapping) else None
    return version if isinstance(version, str) else None


def _reported_device_sm(report: Mapping[str, Any]) -> tuple[bool, int | None]:
    devices = report.get("devices")
    if not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], Mapping):
        return False, None
    architecture = devices[0].get("architecture")
    if not isinstance(architecture, str):
        return False, None
    accelerator = report.get("accelerator")
    if accelerator == "cuda" and architecture.startswith("sm_"):
        suffix = architecture.removeprefix("sm_")
        return (True, int(suffix)) if suffix.isdigit() else (False, None)
    if accelerator == "rocm" and architecture.startswith("gfx"):
        suffix = architecture.removeprefix("gfx")
        return (True, int(suffix) // 10) if suffix.isdigit() else (False, None)
    if accelerator == "xpu":
        return True, None
    return False, None


def _attention_authentication_problems(
    report: Mapping[str, Any], label: str, policy: str
) -> list[str]:
    problems: list[str] = []
    attention = report.get("attention")
    if not isinstance(attention, Mapping):
        return [f"{label} attention identity is missing"]
    if attention.get("requested_policy") != policy:
        problems.append(f"{label} requested attention policy differs")
    system = report.get("system")
    if system == "dinkster":
        token = attention.get("route_token")
        if not isinstance(token, Mapping):
            return [*problems, f"{label} authenticated route token is missing"]
        try:
            parsed_token = attention_route_token_from_wire(token)
        except (TypeError, ValueError) as error:
            return [*problems, f"{label} authenticated route token is malformed: {error}"]
        if parsed_token.requested_policy != policy:
            problems.append(f"{label} route token requested policy differs")
        providers = dict(parsed_token.provider_versions)
        required_provider = {
            "sdpa": "torch",
            "dinkster_kitchen_int8": "dinkster-kitchen",
            "sage": "sageattention",
        }[policy]
        if required_provider not in providers:
            problems.append(
                f"{label} route token does not identify the {required_provider} provider"
            )
        torch_version = _reported_torch_version(report)
        if (
            not isinstance(torch_version, str)
            or providers.get("torch") != torch_version
            or parsed_token.sdpa_torch_runtime != torch_version.split("+")[0]
        ):
            problems.append(f"{label} route token torch runtime differs from the report")
        if parsed_token.device_kind != report.get("accelerator"):
            problems.append(f"{label} route token device kind differs from the report")
        device_valid, device_sm = _reported_device_sm(report)
        if not device_valid or parsed_token.device_sm != device_sm:
            problems.append(f"{label} route token device architecture differs from the report")
        flux = next((route for route in parsed_token.routes if route.role == "flux"), None)
        expected_fallback = None if policy == "sdpa" else "sdpa"
        if flux is None or (flux.primary, flux.fallback) != (
            policy,
            expected_fallback,
        ):
            problems.append(f"{label} Flux route does not authenticate the requested policy")
        return problems
    if system != "comfyui":
        return [*problems, f"{label} system cannot authenticate attention"]
    if attention.get("selected_policy") != policy:
        problems.append(f"{label} selected attention policy differs")
    providers = _provider_versions(attention.get("provider_versions"))
    if providers is None:
        problems.append(f"{label} provider versions are incomplete")
        providers = {}
    torch_version = _reported_torch_version(report)
    if not isinstance(torch_version, str) or providers.get("torch") != torch_version:
        problems.append(f"{label} provider torch runtime differs from the report")
    if policy == "dinkster_kitchen_int8" and "comfy-kitchen" not in providers:
        problems.append(f"{label} comfy-kitchen provider version is missing")
    execution = attention.get("execution")
    if not isinstance(execution, Mapping) or execution.get("policy") != policy:
        return [*problems, f"{label} attention execution proof is missing"]
    counters = {
        name: execution.get(name)
        for name in (
            "selected_calls",
            "provider_attempts",
            "provider_successes",
            "provider_exceptions",
            "fallback_calls",
        )
    }
    if any(type(value) is not int or value < 0 for value in counters.values()):
        problems.append(f"{label} attention execution counters are malformed")
    elif counters["selected_calls"] <= 0 or counters["provider_successes"] <= 0:
        problems.append(f"{label} requested attention provider did not execute")
    elif counters["provider_attempts"] != (
        counters["provider_successes"] + counters["provider_exceptions"]
    ):
        problems.append(f"{label} attention attempt counters are inconsistent")
    elif policy == "sage" and counters["selected_calls"] != (
        counters["provider_successes"] + counters["fallback_calls"]
    ):
        problems.append(f"{label} Sage fallback counters are inconsistent")
    elif policy != "sage" and (
        counters["fallback_calls"] or counters["selected_calls"] != counters["provider_successes"]
    ):
        problems.append(f"{label} attention provider counters are inconsistent")
    if policy == "sage":
        module = attention.get("provider_module")
        if not isinstance(module, Mapping) or module.get("authenticated") is not True:
            problems.append(f"{label} Sage module distribution is not authenticated")
        else:
            distribution = module.get("distribution")
            version = module.get("version")
            if (
                module.get("module") != "sageattention"
                or not isinstance(distribution, str)
                or not isinstance(version, str)
                or providers.get(distribution) != version
            ):
                problems.append(f"{label} Sage module distribution differs from provider versions")
    return problems


def comparability_problems(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[str, ...]:
    problems: list[str] = []
    system = baseline.get("system")
    if system not in ("dinkster", "comfyui") or candidate.get("system") != system:
        problems.append("reports must come from the same recognized system")
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        accelerator = report.get("accelerator")
        if not isinstance(accelerator, str):
            problems.append(f"{label} accelerator is missing")
            continue
        family_id = report.get("family_id")
        registered = BUILTIN_FAMILIES_BY_ID.get(family_id) if isinstance(family_id, str) else None
        engine = None if registered is None else registered.engine
        try:
            incomplete = validate_benchmark_report(
                report,
                accelerator=accelerator,
                residency_route_roles=() if engine is None else engine.residency_route_roles,
                requires_accelerator_residency=(
                    False if engine is None else engine.requires_accelerator_residency
                ),
            )
        except ValueError as error:
            problems.append(f"{label} report validation failed: {error}")
        else:
            problems.extend(f"{label} report is incomplete: {problem}" for problem in incomplete)
    for name in _MATCHED_REPORT_FIELDS:
        if baseline.get(name) != candidate.get(name):
            problems.append(f"{name} differs")
    if baseline.get("family") != "minimax_h3":
        problems.append("quality comparison currently requires MiniMax H3 reports")
    if _artifact_identity(baseline) != _artifact_identity(candidate):
        problems.append("artifact identities differ")
    baseline_source = _source_identity(baseline)
    candidate_source = _source_identity(candidate)
    if not baseline_source[1] or not baseline_source[2] or baseline_source != candidate_source:
        problems.append("source commits differ or were not both verified clean")

    baseline_attention = baseline.get("attention") or {}
    candidate_attention = candidate.get("attention") or {}
    if baseline_attention.get("requested_policy") != "sdpa":
        problems.append("baseline attention policy must be explicit SDPA")
    if candidate_attention.get("requested_policy") not in _CANDIDATE_POLICIES:
        problems.append(f"candidate attention policy must be one of {_CANDIDATE_POLICIES}")
    problems.extend(_attention_authentication_problems(baseline, "baseline", "sdpa"))
    candidate_policy = candidate_attention.get("requested_policy")
    if candidate_policy in _CANDIDATE_POLICIES:
        problems.extend(
            _attention_authentication_problems(candidate, "candidate", candidate_policy)
        )

    baseline_capture = baseline.get("quality_capture") or {}
    candidate_capture = candidate.get("quality_capture") or {}
    if baseline_capture.get("version") != 1 or candidate_capture.get("version") != 1:
        problems.append("both reports require quality capture version 1")
    if baseline_capture.get("seed") != candidate_capture.get("seed"):
        problems.append("quality capture seeds differ")
    if baseline_capture.get("audio_sample_rate") != candidate_capture.get("audio_sample_rate"):
        problems.append("captured audio sample rates differ")
    for medium in _MEDIA:
        ours = baseline_capture.get(medium)
        theirs = candidate_capture.get(medium)
        if not isinstance(ours, Mapping) or not isinstance(theirs, Mapping):
            problems.append(f"both reports require {medium} quality captures")
            continue
        for field in _QUALITY_FIELDS:
            if ours.get(field) != theirs.get(field):
                problems.append(f"{medium}.{field} differs")
    return tuple(problems)


def _verified_array(metadata: Mapping[str, Any], label: str) -> np.ndarray:
    path = Path(str(metadata.get("path") or ""))
    if not path.is_file():
        raise ValueError(f"{label} capture does not exist: {path}")
    expected_bytes = metadata.get("bytes")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} capture size does not match its report")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 22), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata.get("sha256"):
        raise ValueError(f"{label} capture digest does not match its report")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or list(array.shape) != metadata.get("captured_shape"):
        raise ValueError(f"{label} capture NPY identity does not match its report")
    return array


def _cosine(dot: float, left_square: float, right_square: float) -> float:
    if left_square == 0.0 and right_square == 0.0:
        return 1.0
    if left_square == 0.0 or right_square == 0.0:
        return 0.0
    return dot / math.sqrt(left_square * right_square)


def _ssim_windows(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    if left.ndim != 3 or left.shape[2] not in (1, 3, 4):
        raise ValueError("image quality capture must contain HWC frames")
    height, width, channels = left.shape
    window = 8
    if height % window or width % window:
        raise ValueError("captured image dimensions must be divisible by the 8x8 SSIM window")
    rows = height // window
    columns = width // window
    left_windows = left.reshape(rows, window, columns, window, channels)
    right_windows = right.reshape(rows, window, columns, window, channels)
    left_mean = left_windows.mean(axis=(1, 3))
    right_mean = right_windows.mean(axis=(1, 3))
    left_centered = left_windows - left_mean[:, None, :, None, :]
    right_centered = right_windows - right_mean[:, None, :, None, :]
    left_variance = np.mean(left_centered * left_centered, axis=(1, 3))
    right_variance = np.mean(right_centered * right_centered, axis=(1, 3))
    covariance = np.mean(left_centered * right_centered, axis=(1, 3))
    c1 = 0.01**2
    c2 = 0.03**2
    values = ((2.0 * left_mean * right_mean + c1) * (2.0 * covariance + c2)) / (
        (left_mean * left_mean + right_mean * right_mean + c1)
        * (left_variance + right_variance + c2)
    )
    return float(values.sum(dtype=np.float64)), int(values.size)


def array_metrics(baseline: np.ndarray, candidate: np.ndarray, *, image: bool) -> dict[str, object]:
    if baseline.shape != candidate.shape:
        raise ValueError("captured array shapes differ")
    if baseline.size == 0:
        raise ValueError("captured arrays must not be empty")
    dot = left_square = right_square = absolute_sum = 0.0
    maximum = 0.0
    count = 0
    ssim_sum = 0.0
    ssim_count = 0
    minimum = [math.inf, math.inf]
    maximum_value = [-math.inf, -math.inf]
    for index in range(baseline.shape[0]):
        left = np.asarray(baseline[index], dtype=np.float64)
        right = np.asarray(candidate[index], dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("captured arrays must contain only finite values")
        difference = np.abs(left - right)
        dot += float(np.sum(left * right, dtype=np.float64))
        left_square += float(np.sum(left * left, dtype=np.float64))
        right_square += float(np.sum(right * right, dtype=np.float64))
        absolute_sum += float(difference.sum(dtype=np.float64))
        maximum = max(maximum, float(difference.max(initial=0.0)))
        count += int(difference.size)
        minimum[0] = min(minimum[0], float(left.min()))
        minimum[1] = min(minimum[1], float(right.min()))
        maximum_value[0] = max(maximum_value[0], float(left.max()))
        maximum_value[1] = max(maximum_value[1], float(right.max()))
        if image:
            frame_ssim, frame_count = _ssim_windows(left, right)
            ssim_sum += frame_ssim
            ssim_count += frame_count
    result: dict[str, object] = {
        "elements": count,
        "cosine_similarity": _cosine(dot, left_square, right_square),
        "mean_absolute_error": absolute_sum / count,
        "max_absolute_error": maximum,
        "baseline_range": [minimum[0], maximum_value[0]],
        "candidate_range": [minimum[1], maximum_value[1]],
    }
    if image:
        result["ssim_8x8_data_range_1"] = ssim_sum / ssim_count
        result["ssim_values"] = ssim_count
    return result


def build_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    baseline_capture = baseline["quality_capture"]
    candidate_capture = candidate["quality_capture"]
    metrics = {}
    for medium in _MEDIA:
        left = _verified_array(baseline_capture[medium], f"baseline {medium}")
        right = _verified_array(candidate_capture[medium], f"candidate {medium}")
        metrics[medium] = array_metrics(left, right, image=medium == "image")
    _system, commit, _clean = _source_identity(baseline)
    return {
        "version": 1,
        "system": baseline["system"],
        "family": baseline["family"],
        "source_commit": commit,
        "baseline_policy": baseline["attention"]["requested_policy"],
        "candidate_policy": candidate["attention"]["requested_policy"],
        "capture": {
            "seed": baseline_capture["seed"],
            "image": {field: baseline_capture["image"][field] for field in _QUALITY_FIELDS},
            "audio": {field: baseline_capture["audio"][field] for field in _QUALITY_FIELDS},
            "audio_sample_rate": baseline_capture["audio_sample_rate"],
        },
        "metrics": metrics,
        "acceptance_thresholds": None,
        "note": "metrics are observational; product acceptance thresholds are not inferred",
    }


def _load_report(path: Path, label: str) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read {label} report {path}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError(f"{label} report is not a JSON object")
    return report


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="explicit SDPA report")
    parser.add_argument("--candidate", type=Path, required=True, help="approximate-provider report")
    parser.add_argument("--json", type=Path, default=None, help="write comparison JSON here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        baseline = _load_report(arguments.baseline, "baseline")
        candidate = _load_report(arguments.candidate, "candidate")
        problems = comparability_problems(baseline, candidate)
        if problems:
            raise ValueError("; ".join(problems))
        comparison = build_comparison(baseline, candidate)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    text = json.dumps(comparison, indent=2, sort_keys=True)
    print(text)
    if arguments.json is not None:
        arguments.json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
