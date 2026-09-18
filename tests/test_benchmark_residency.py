"""CPU-testable contracts for the residency A/B probe command."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_residency.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_residency", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_residency = importlib.util.module_from_spec(_SPEC)
_PREVIOUS_TORCH = sys.modules.get("torch")
sys.modules["benchmark_residency"] = benchmark_residency
sys.modules["torch"] = ModuleType("torch")
try:
    _SPEC.loader.exec_module(benchmark_residency)
finally:
    if _PREVIOUS_TORCH is None:
        del sys.modules["torch"]
    else:
        sys.modules["torch"] = _PREVIOUS_TORCH


@pytest.mark.parametrize("name", ["eager", "aimdo"])
@pytest.mark.parametrize("ready", [False, True])
def test_mechanism_factory_uses_visible_admission_without_eager_substitution(
    monkeypatch: pytest.MonkeyPatch, name: str, ready: bool
) -> None:
    eager, aimdo = object(), object()
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference_torch",
        SimpleNamespace(ResidentWeights=eager, AimdoWeights=aimdo),
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference_torch.aimdo_activation",
        SimpleNamespace(ensure_visible_aimdo_devices=lambda: calls.append("visible") or ready),
    )
    if name == "aimdo" and not ready:
        with pytest.raises(RuntimeError, match="aimdo device activation returned False"):
            benchmark_residency._mechanism_factory(name)
    else:
        assert benchmark_residency._mechanism_factory(name) is (aimdo if name == "aimdo" else eager)
    assert calls == (["visible"] if name == "aimdo" else [])


# -- argument parsing ---------------------------------------------------------


def test_parse_arguments_defaults_reproduce_the_reference_probe_shape() -> None:
    arguments = benchmark_residency.parse_arguments(["--mechanism", "eager"])
    assert arguments.mechanism == "eager"
    assert arguments.weights == "dense"
    assert arguments.queue is False
    assert arguments.regime == "open"
    assert arguments.device == "cuda:0"
    assert arguments.dtype == "float16"
    assert (arguments.blocks, arguments.layers_per_block) == (8, 4)
    assert (arguments.features, arguments.batch) == (4096, 16)
    assert (arguments.warmup, arguments.passes) == (2, 10)
    assert arguments.leave_free_mib == 768
    assert arguments.spill_scope == "auto"
    assert benchmark_residency.working_set_bytes(arguments) == 1024 * 1024 * 1024


def test_parse_arguments_requires_a_known_mechanism() -> None:
    with pytest.raises(SystemExit):
        benchmark_residency.parse_arguments([])
    with pytest.raises(SystemExit):
        benchmark_residency.parse_arguments(["--mechanism", "vbar"])
    with pytest.raises(SystemExit):
        benchmark_residency.parse_arguments(["--mechanism", "sticky"])
    with pytest.raises(SystemExit):
        benchmark_residency.parse_arguments(["--mechanism", "aimdo", "--spill-threshold-mib", "64"])


def test_working_set_bytes_scales_with_dtype_width() -> None:
    half = benchmark_residency.parse_arguments(["--mechanism", "eager"])
    full = benchmark_residency.parse_arguments(["--mechanism", "eager", "--dtype", "float32"])
    assert benchmark_residency.working_set_bytes(full) == (
        2 * benchmark_residency.working_set_bytes(half)
    )


def test_working_set_bytes_charges_q8_0_encoded_block_bytes() -> None:
    encoded = benchmark_residency.parse_arguments(["--mechanism", "aimdo", "--weights", "q8_0"])
    # 8 * 4 * 4096^2 elements in 32-element, 34-byte Q8_0 blocks: 544 MiB.
    assert benchmark_residency.working_set_bytes(encoded) == 544 * 1024 * 1024
    dense = benchmark_residency.parse_arguments(["--mechanism", "aimdo"])
    assert benchmark_residency.working_set_bytes(encoded) < (
        benchmark_residency.working_set_bytes(dense)
    )


def test_parse_arguments_requires_a_known_weight_format() -> None:
    with pytest.raises(SystemExit):
        benchmark_residency.parse_arguments(["--mechanism", "eager", "--weights", "q4_k"])


# -- spill axis ---------------------------------------------------------------


def test_pass_time_cliff_flags_only_ratios_above_threshold() -> None:
    assert benchmark_residency.pass_time_cliff([], 4.0) == (None, False)
    observed, suspected = benchmark_residency.pass_time_cliff([10.0, 10.0, 10.0], 4.0)
    assert observed == 1.0 and suspected is False
    observed, suspected = benchmark_residency.pass_time_cliff([10.0, 10.0, 51.0], 4.0)
    assert observed == pytest.approx(5.1) and suspected is True
    observed, suspected = benchmark_residency.pass_time_cliff([10.0, 10.0, 39.0], 4.0)
    assert observed == pytest.approx(3.9) and suspected is False


def test_pass_time_cliff_ignores_degenerate_zero_medians() -> None:
    assert benchmark_residency.pass_time_cliff([0.0, 0.0, 5.0], 4.0) == (None, False)


def test_shared_usage_command_scopes_counter_instances() -> None:
    process = benchmark_residency.shared_usage_command("process", 4242)
    machine = benchmark_residency.shared_usage_command("machine", 4242)
    assert "GPU Process Memory(pid_4242_*)" in process
    assert "Shared Usage" in process
    assert "[long]$_.CookedValue" in process
    assert "GPU Process Memory(*)" in machine


def test_sum_counter_lines_sums_integral_lines_and_rejects_garbage() -> None:
    assert benchmark_residency.sum_counter_lines("123\r\n456\r\n\r\n") == 579
    assert benchmark_residency.sum_counter_lines("0\n") == 0
    assert benchmark_residency.sum_counter_lines("") is None
    assert benchmark_residency.sum_counter_lines("   \n  \n") is None
    assert benchmark_residency.sum_counter_lines("12\nnot-a-number\n") is None
    assert benchmark_residency.sum_counter_lines("12\n-1\n") is None


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ("[pscustomobject]@{Status=0; CookedValue=0}", 0),
        (
            "[pscustomobject]@{Status=0; CookedValue=12}, "
            "[pscustomobject]@{Status=0; CookedValue=30}",
            42,
        ),
        ("", None),
        (
            "[pscustomobject]@{Status=0; CookedValue=12}, "
            "[pscustomobject]@{Status=1; CookedValue=0}",
            None,
        ),
        ("[pscustomobject]@{Status=0; CookedValue=-1}", None),
        ("[pscustomobject]@{Status=0; CookedValue=$null}", None),
        ("[pscustomobject]@{Status=0; CookedValue=[double]::NaN}", None),
        ("[pscustomobject]@{Status=0; CookedValue=[double]::PositiveInfinity}", None),
    ],
)
def test_shared_usage_powershell_rejects_unavailable_samples(
    samples: str, expected: int | None
) -> None:
    executable = shutil.which("powershell.exe") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is required to execute the Windows counter parser")
    script = (
        "function Get-Counter { [CmdletBinding()] param([string]$Counter) "
        "[pscustomobject]@{CounterSamples=@("
        + samples
        + ")} }; "
        + benchmark_residency.shared_usage_command("process", 4242)
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if expected is None:
        assert result.returncode != 0
        assert not result.stdout.strip()
    else:
        assert result.returncode == 0, result.stderr
        assert benchmark_residency.sum_counter_lines(result.stdout) == expected


def test_gpu_shared_usage_bytes_parses_first_working_powershell() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="1048576\n2097152\n")

    total = benchmark_residency.gpu_shared_usage_bytes("process", 99, run=run)
    assert total == 3145728
    assert len(calls) == 1
    assert calls[0][0] == "powershell.exe"
    assert "pid_99_*" in calls[0][-1]


def test_gpu_shared_usage_bytes_returns_none_when_counters_unreachable() -> None:
    def failing_run(_argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        raise OSError("no powershell here")

    assert benchmark_residency.gpu_shared_usage_bytes("machine", 1, run=failing_run) is None

    def broken_run(_argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="")

    assert benchmark_residency.gpu_shared_usage_bytes("machine", 1, run=broken_run) is None

    def garbage_run(_argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="Get-Counter : error text\n")

    assert benchmark_residency.gpu_shared_usage_bytes("machine", 1, run=garbage_run) is None


@pytest.mark.parametrize(
    ("scope", "system", "release", "expected"),
    [
        ("process", "Linux", "6.6.0", "process"),
        ("machine", "Windows", "10", "machine"),
        ("off", "Windows", "10", "off"),
        ("auto", "Windows", "10", "process"),
        ("auto", "Linux", "5.15.167.4-microsoft-standard-WSL2", "machine"),
        ("auto", "Linux", "4.4.0-19041-Microsoft", "machine"),
        ("auto", "Linux", "6.6.0-generic", "off"),
        ("auto", "Darwin", "23.0.0", "off"),
    ],
)
def test_resolve_spill_scope_maps_auto_to_attributable_scope(
    scope: str, system: str, release: str, expected: str
) -> None:
    assert benchmark_residency.resolve_spill_scope(scope, system, release) == expected


# -- report -------------------------------------------------------------------


def _report() -> dict[str, Any]:
    return benchmark_residency.build_report(
        host={
            "system": "Linux",
            "release": "wsl2",
            "python": "3.12.0",
            "torch": "2.13.0",
            "device": "cuda:0",
            "device_name": "test",
            "total_bytes": 1,
        },
        config={
            "mechanism": "aimdo",
            "weights": "q8_0",
            "queue": True,
            "regime": "open",
            "device": "cuda:0",
            "dtype": "float16",
            "blocks": 8,
            "layers_per_block": 4,
            "features": 4096,
            "batch": 16,
            "warmup": 2,
            "passes": 3,
            "seed": 826,
            "leave_free_mib": 768,
            "cliff_ratio": 4.0,
            "spill_scope": "machine",
            "working_set_bytes": 1024 * 1024 * 1024,
        },
        pass_ms=[10.0, 11.0, 12.0],
        pass_free_bytes=[100, 100, 100],
        bit_identical=True,
        receipt={
            "transfer_ms": 1.0,
            "exposed_stall_ms": 0.5,
            "dequant_ms": 0.0,
            "compute_ms": 2.0,
            "transfer_bytes": 2048,
            "leased_transfers": 1,
            "leased_forwards": 3,
            "prefetched_transfers": 2,
            "prefetch_bytes": 1024,
        },
        memory={
            "ballast_bytes": 0,
            "free_after_setup_bytes": 200,
            "free_after_passes_bytes": 150,
            "allocated_peak_bytes": 300,
            "reserved_peak_bytes": 400,
        },
        spill={
            "cliff_ratio_observed": 1.2,
            "cliff_suspected": False,
            "scope": "machine",
            "shared_before_bytes": 0,
            "shared_warm_bytes": 0,
            "shared_after_bytes": 0,
            "shared_growth_bytes": 0,
            "shared_spill_detected": None,
        },
    )


def test_build_report_summarizes_passes_and_round_trips_as_json() -> None:
    report = _report()
    assert report["schema"] == benchmark_residency.SCHEMA
    assert report["results"]["median_ms"] == 11.0
    assert report["results"]["min_ms"] == 10.0
    assert report["results"]["max_ms"] == 12.0
    assert json.loads(json.dumps(report)) == report
    assert benchmark_residency.validate_residency_report(report) == []


@pytest.mark.parametrize("detected", [True, False])
def test_validate_residency_report_rejects_unsupported_spill_claims(detected: bool) -> None:
    report = _report()
    report["results"]["spill"]["shared_spill_detected"] = detected
    assert any(
        "shared usage cannot assess spill" in problem
        for problem in benchmark_residency.validate_residency_report(report)
    )


def test_validate_residency_report_flags_schema_and_missing_sections() -> None:
    problems = benchmark_residency.validate_residency_report({"schema": "other"})
    assert any("schema" in problem for problem in problems)
    assert any("host" in problem for problem in problems)
    assert any("config" in problem for problem in problems)
    assert any("results" in problem for problem in problems)


def test_validate_residency_report_flags_pass_count_and_type_problems() -> None:
    report = _report()
    report["results"]["pass_ms"] = [10.0, 11.0]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("length must equal config passes" in problem for problem in problems)

    report = _report()
    report["results"]["pass_ms"] = "fast"
    problems = benchmark_residency.validate_residency_report(report)
    assert any("list of numbers" in problem for problem in problems)

    report = _report()
    report["results"]["bit_identical"] = "yes"
    problems = benchmark_residency.validate_residency_report(report)
    assert any("bit_identical" in problem for problem in problems)


def test_validate_residency_report_flags_unknown_mechanism_and_regime() -> None:
    report = _report()
    report["config"]["mechanism"] = "vbar"
    report["config"]["regime"] = "tight"
    problems = benchmark_residency.validate_residency_report(report)
    assert any("mechanism" in problem for problem in problems)
    assert any("regime" in problem for problem in problems)


def test_validate_residency_report_flags_unknown_or_missing_weight_format() -> None:
    report = _report()
    report["config"]["weights"] = "q4_k"
    problems = benchmark_residency.validate_residency_report(report)
    assert any("weights must be one of" in problem for problem in problems)

    report = _report()
    del report["config"]["weights"]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("config missing 'weights'" in problem for problem in problems)


def test_validate_residency_report_flags_missing_host_keys() -> None:
    report = _report()
    report["host"] = {}
    problems = benchmark_residency.validate_residency_report(report)
    for key in ("system", "release", "python", "torch", "device", "device_name", "total_bytes"):
        assert any(f"host missing {key!r}" in problem for problem in problems)


def test_validate_residency_report_flags_missing_or_bad_summary_fields() -> None:
    report = _report()
    del report["results"]["median_ms"]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("median_ms must be a number" in problem for problem in problems)

    report = _report()
    report["results"]["min_ms"] = None
    report["results"]["max_ms"] = "slow"
    problems = benchmark_residency.validate_residency_report(report)
    assert any("min_ms must be a number" in problem for problem in problems)
    assert any("max_ms must be a number" in problem for problem in problems)


def test_validate_residency_report_flags_pass_free_bytes_problems() -> None:
    report = _report()
    del report["results"]["pass_free_bytes"]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("pass_free_bytes must be a list" in problem for problem in problems)

    report = _report()
    report["results"]["pass_free_bytes"] = [100, "many", 100]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("pass_free_bytes must be a list" in problem for problem in problems)

    report = _report()
    report["results"]["pass_free_bytes"] = [100, None, 100]
    assert benchmark_residency.validate_residency_report(report) == []

    report = _report()
    report["results"]["pass_free_bytes"] = [100, 100]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("pass_free_bytes length must equal config passes" in problem for problem in problems)


def test_validate_residency_report_flags_missing_result_sections_and_keys() -> None:
    report = _report()
    del report["results"]["spill"]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("missing 'spill'" in problem for problem in problems)

    report = _report()
    del report["results"]["receipt"]["transfer_bytes"]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("receipt missing 'transfer_bytes'" in problem for problem in problems)

    report = _report()
    del report["results"]["spill"]["shared_warm_bytes"]
    problems = benchmark_residency.validate_residency_report(report)
    assert any("spill missing 'shared_warm_bytes'" in problem for problem in problems)
