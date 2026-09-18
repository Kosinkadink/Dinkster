"""The benchmark report comparison tool's validation gating, same-cell
checks, ratio construction, and command-line entrypoint, exercised on
structurally complete reports without touching real runs."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from dinkster_workers.backend_env import validate_benchmark_report

from tests.test_backend_env import complete_benchmark_report

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "compare_benchmark_reports.py"
_SPEC = importlib.util.spec_from_file_location("compare_benchmark_reports", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
compare_benchmark_reports = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_benchmark_reports)

gate_problems = compare_benchmark_reports.gate_problems
comparability_problems = compare_benchmark_reports.comparability_problems
build_comparison = compare_benchmark_reports.build_comparison
format_comparison = compare_benchmark_reports.format_comparison
main = compare_benchmark_reports.main


def cell_pair(family: str = "sd15") -> tuple[dict[str, Any], dict[str, Any]]:
    """One validated Dinkster report and one same-cell ComfyUI report."""
    dinkster = complete_benchmark_report("cuda", family=family)
    comfyui = complete_benchmark_report("cuda", family=family, system="comfyui")
    # The runners name the same algorithms with and without the Dinkster
    # registry prefix.
    comfyui["workload"]["sampler_id"] = dinkster["workload"]["sampler_id"].removeprefix("dinkster.")
    comfyui["workload"]["scheduler_id"] = dinkster["workload"]["scheduler_id"].removeprefix(
        "dinkster."
    )
    return dinkster, comfyui


class TestGating:
    def test_complete_reports_pass_the_gate(self) -> None:
        dinkster, comfyui = cell_pair()
        assert gate_problems(dinkster, "dinkster") == ()
        assert gate_problems(comfyui, "comfyui") == ()

    def test_invalid_report_problems_carry_the_label(self) -> None:
        dinkster, _ = cell_pair()
        dinkster["report_version"] = 999
        problems = gate_problems(dinkster, "dinkster")
        assert problems
        assert all(problem.startswith("dinkster: ") for problem in problems)

    def test_unknown_accelerator_is_one_problem(self) -> None:
        dinkster, _ = cell_pair()
        dinkster["accelerator"] = "abacus"
        problems = gate_problems(dinkster, "dinkster")
        assert len(problems) == 1
        assert "accelerator" in problems[0]


class TestSameCell:
    def test_same_cell_has_no_problems(self) -> None:
        dinkster, comfyui = cell_pair()
        assert comparability_problems(dinkster, comfyui) == ()

    def test_direct_diagnostic_is_valid_standalone_but_not_canonical_evidence(self) -> None:
        dinkster, _ = cell_pair()
        dinkster["placement"] = "direct_placement_diagnostic"
        assert validate_benchmark_report(dinkster, accelerator="cuda") == ()
        problems = gate_problems(dinkster, "dinkster")
        assert any(
            "canonical evidence requires Dinkster placement" in problem for problem in problems
        )

    def test_swapped_systems_are_rejected(self) -> None:
        dinkster, comfyui = cell_pair()
        problems = comparability_problems(comfyui, dinkster)
        assert any("not 'dinkster'" in problem for problem in problems)
        assert any("not 'comfyui'" in problem for problem in problems)

    def test_family_mismatch_is_rejected(self) -> None:
        dinkster, _ = cell_pair()
        _, comfyui = cell_pair(family="sdxl")
        problems = comparability_problems(dinkster, comfyui)
        assert any("family differs" in problem for problem in problems)

    def test_compile_report_is_rejected(self) -> None:
        dinkster, comfyui = cell_pair()
        dinkster["mode"] = "compile"
        problems = comparability_problems(dinkster, comfyui)
        assert any("eager against eager" in problem for problem in problems)

    def test_failed_run_is_rejected(self) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["all_ok"] = False
        problems = comparability_problems(dinkster, comfyui)
        assert any("not all_ok" in problem for problem in problems)

    def test_workload_field_mismatch_is_rejected(self) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["workload"]["seed"] = 668
        problems = comparability_problems(dinkster, comfyui)
        assert any("workload.seed differs" in problem for problem in problems)

    def test_flux_guidance_mismatch_is_rejected(self) -> None:
        dinkster, comfyui = cell_pair(family="flux")
        comfyui["workload"]["guidance"] = 4.0
        problems = comparability_problems(dinkster, comfyui)
        assert any("workload.guidance differs" in problem for problem in problems)

    def test_anima_variant_mismatch_is_rejected(self) -> None:
        dinkster, comfyui = cell_pair("anima")
        comfyui["variant"] = "fallback_768"
        problems = comparability_problems(dinkster, comfyui)
        assert any("variant differs" in problem for problem in problems)

    def test_sampler_prefix_is_equivalent_but_algorithms_are_not(self) -> None:
        dinkster, comfyui = cell_pair()
        assert dinkster["workload"]["sampler_id"] == "dinkster.euler"
        assert comfyui["workload"]["sampler_id"] == "euler"
        assert comparability_problems(dinkster, comfyui) == ()
        comfyui["workload"]["sampler_id"] = "uni_pc"
        problems = comparability_problems(dinkster, comfyui)
        assert any("workload.sampler_id differs" in problem for problem in problems)

    def test_artifact_digest_mismatch_is_rejected(self) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["artifacts"][0]["sha256"] = "c" * 64
        problems = comparability_problems(dinkster, comfyui)
        assert any("artifact digests differ" in problem for problem in problems)

    def test_device_mismatch_is_rejected(self) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["devices"][0]["name"] = "Different GPU"
        problems = comparability_problems(dinkster, comfyui)
        assert any("devices differ" in problem for problem in problems)

    def test_attention_policy_mismatch_is_rejected(self) -> None:
        dinkster, comfyui = cell_pair("minimax_h3")
        dinkster["attention"] = {"requested_policy": "sage"}
        comfyui["attention"] = {"requested_policy": "sdpa"}
        problems = comparability_problems(dinkster, comfyui)
        assert any("attention.requested_policy differs" in problem for problem in problems)


class TestComparison:
    def test_ratios_exist_only_where_measurements_are_comparable(self) -> None:
        dinkster, comfyui = cell_pair()
        dinkster["timings"]["warm"]["median_total_s"] = 5.0
        comfyui["timings"]["warm"]["median_total_s"] = 4.0
        dinkster["timings"]["cold"]["total_s"] = 30.0
        comfyui["timings"]["cold"]["total_s"] = 20.0
        comparison = build_comparison(dinkster, comfyui)
        assert comparison["timings"]["warm"]["median_total_s"]["ratio"] == 1.25
        cold = comparison["timings"]["cold"]
        assert cold["total_s"]["ratio"] == 1.5
        for phase in ("load_s", "encode_s", "sample_s", "decode_s"):
            assert cold[phase]["ratio"] is None
            assert cold[phase]["dinkster"] is not None
            assert cold[phase]["comfyui"] is not None
        memory = comparison["memory"]
        assert memory["peak_rss_bytes"]["ratio"] is not None
        assert memory["peak_allocated_bytes"]["ratio"] is None
        assert memory["peak_reserved_bytes"]["ratio"] is None
        assert memory["residual_allocated_bytes"]["ratio"] is None

    def test_cell_and_workload_come_from_the_dinkster_report(self) -> None:
        dinkster, comfyui = cell_pair()
        comparison = build_comparison(dinkster, comfyui)
        assert comparison["cell"] == {"family": "sd15", "accelerator": "cuda", "mode": "eager"}
        assert comparison["placement"] == {
            "dinkster": "production_residency",
            "comfyui": "comfyui_model_management",
        }
        assert comparison["workload"] == dinkster["workload"]
        assert comparison["attention"] == {
            "dinkster": {"requested_policy": "auto"},
            "comfyui": {"requested_policy": "auto"},
        }

    def test_lora_cell_carries_the_lora_interval_without_a_ratio(self) -> None:
        dinkster, comfyui = cell_pair(family="lora")
        comparison = build_comparison(dinkster, comfyui)
        row = comparison["timings"]["cold"]["lora_s"]
        assert row["dinkster"] is not None
        assert row["ratio"] is None

    def test_zero_denominator_yields_no_ratio(self) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["timings"]["warm"]["median_decode_s"] = 0
        comparison = build_comparison(dinkster, comfyui)
        assert comparison["timings"]["warm"]["median_decode_s"]["ratio"] is None

    def test_formatting_states_the_cell_and_the_ratio_convention(self) -> None:
        dinkster, comfyui = cell_pair()
        text = format_comparison(build_comparison(dinkster, comfyui))
        assert "cell: sd15 / eager on cuda" in text
        assert "placement: dinkster=production_residency, comfyui=comfyui_model_management" in text
        assert "dinkster / comfyui" in text
        assert "median_total_s" in text


def write_pair(tmp_path: Path, dinkster: dict[str, Any], comfyui: dict[str, Any]) -> list[str]:
    """The two reports written to disk, as CLI arguments."""
    dinkster_path = tmp_path / "dinkster.json"
    comfyui_path = tmp_path / "comfyui.json"
    dinkster_path.write_text(json.dumps(dinkster))
    comfyui_path.write_text(json.dumps(comfyui))
    return ["--dinkster", str(dinkster_path), "--comfyui", str(comfyui_path)]


class TestCli:
    def test_valid_infinitetalk_pair_writes_audio_phase_comparison(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("wan21_infinitetalk")
        json_path = tmp_path / "comparison.json"
        arguments = [*write_pair(tmp_path, dinkster, comfyui), "--json", str(json_path)]
        assert main(arguments) == 0
        written = json.loads(json_path.read_text())
        assert written["cell"]["family"] == "wan21_infinitetalk"
        assert written["timings"]["cold"]["audio_encode_s"] == {
            "dinkster": 1.7,
            "comfyui": 1.7,
            "ratio": None,
        }
        assert "audio_encode_s" in capsys.readouterr().out

    def test_valid_humo_pair_writes_audio_phase_comparison(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("wan21_humo")
        json_path = tmp_path / "comparison.json"
        arguments = [*write_pair(tmp_path, dinkster, comfyui), "--json", str(json_path)]
        assert main(arguments) == 0
        written = json.loads(json_path.read_text())
        assert written["cell"]["family"] == "wan21_humo"
        assert written["placement"] == {
            "dinkster": "production_residency",
            "comfyui": "comfyui_model_management",
        }
        assert written["timings"]["cold"]["audio_encode_s"] == {
            "dinkster": 1.7,
            "comfyui": 1.7,
            "ratio": None,
        }
        output = capsys.readouterr().out
        assert (
            "placement: dinkster=production_residency, comfyui=comfyui_model_management" in output
        )
        assert "audio_encode_s" in output

    def test_valid_anima_pair_writes_the_pinned_comparison(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("anima")
        json_path = tmp_path / "comparison.json"
        arguments = [*write_pair(tmp_path, dinkster, comfyui), "--json", str(json_path)]
        assert main(arguments) == 0
        written = json.loads(json_path.read_text())
        assert written["cell"] == {"family": "anima", "accelerator": "cuda", "mode": "eager"}
        assert written["workload"]["seed"] == 875817230929465
        assert written["workload"]["sampler_id"] == "dinkster.er_sde"
        assert written["variant"] == "primary"
        output = capsys.readouterr().out
        assert "cell: anima / eager on cuda" in output
        assert "variant: primary" in output

    def test_valid_anima_fallback_pair_is_explicitly_labeled(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("anima")
        for report in (dinkster, comfyui):
            report["variant"] = "fallback_768"
            report["workload"]["width"] = 768
            report["workload"]["height"] = 768
        json_path = tmp_path / "comparison.json"
        arguments = [*write_pair(tmp_path, dinkster, comfyui), "--json", str(json_path)]

        assert main(arguments) == 0

        assert json.loads(json_path.read_text())["variant"] == "fallback_768"
        assert "variant: fallback_768" in capsys.readouterr().out

    def test_valid_minimax_h3_pair_writes_av_comparison(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("minimax_h3")
        json_path = tmp_path / "comparison.json"
        arguments = [*write_pair(tmp_path, dinkster, comfyui), "--json", str(json_path)]
        assert main(arguments) == 0
        written = json.loads(json_path.read_text())
        assert written["cell"]["family"] == "minimax_h3"
        assert written["workload"]["length"] == 124
        assert written["placement"] == {
            "dinkster": "production_residency",
            "comfyui": "comfyui_model_management",
        }
        assert written["execution_path"] == {
            "dinkster": "generation_ksampler_multistream",
            "comfyui": "sampler_custom_advanced",
        }
        output = capsys.readouterr().out
        assert "cell: minimax_h3 / eager on cuda" in output
        assert (
            "execution path: dinkster=generation_ksampler_multistream, "
            "comfyui=sampler_custom_advanced"
        ) in output
        assert "audio_encode_s" not in output

    def test_valid_pair_prints_the_table_and_the_caveat_notes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair()
        assert main(write_pair(tmp_path, dinkster, comfyui)) == 0
        captured = capsys.readouterr()
        assert "cell: sd15 / eager on cuda" in captured.out
        assert "dinkster / comfyui" in captured.out
        assert "only cold" in captured.out and "apples-to-apples" in captured.out
        assert "outside the caching allocator" in captured.out
        assert captured.err == ""

    def test_json_output_carries_the_comparison_and_the_notes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair()
        json_path = tmp_path / "comparison.json"
        arguments = [*write_pair(tmp_path, dinkster, comfyui), "--json", str(json_path)]
        assert main(arguments) == 0
        written = json.loads(json_path.read_text())
        assert written["cell"] == {"family": "sd15", "accelerator": "cuda", "mode": "eager"}
        assert written["placement"] == {
            "dinkster": "production_residency",
            "comfyui": "comfyui_model_management",
        }
        assert set(written) >= {"cell", "placement", "workload", "timings", "memory", "notes"}
        assert written["timings"]["cold"]["total_s"]["ratio"] is not None
        assert written["timings"]["cold"]["load_s"]["ratio"] is None
        assert any("apples-to-apples" in note for note in written["notes"])
        assert any("caching allocator" in note for note in written["notes"])
        assert str(json_path) in capsys.readouterr().out

    def _assert_rejected(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        dinkster: dict[str, Any],
        comfyui: dict[str, Any],
        expected_error: str,
    ) -> None:
        json_path = tmp_path / "comparison.json"
        arguments = [*write_pair(tmp_path, dinkster, comfyui), "--json", str(json_path)]
        assert main(arguments) == 1
        captured = capsys.readouterr()
        assert expected_error in captured.err
        assert "cell:" not in captured.out
        assert not json_path.exists()

    def test_invalid_report_is_rejected_before_comparison(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair()
        dinkster["report_version"] = 999
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "error: dinkster: ")

    def test_failed_run_is_rejected_at_the_entrypoint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["all_ok"] = False
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "all_ok")

    def test_artifact_digest_mismatch_is_rejected_at_the_entrypoint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["artifacts"][0]["sha256"] = "c" * 64
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "artifact digests differ")

    def test_workload_mismatch_is_rejected_at_the_entrypoint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["workload"]["seed"] = 668
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "workload.seed differs")

    @pytest.mark.parametrize("field", ["motion_frame_count", "audio_scale", "speaker_mask_layout"])
    def test_infinitetalk_setting_mismatch_is_rejected_at_the_entrypoint(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        field: str,
    ) -> None:
        dinkster, comfyui = cell_pair("wan21_infinitetalk")
        comfyui["workload"][field] = "different"
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, f"workload.{field}")

    def test_humo_workload_mismatch_is_rejected_at_the_entrypoint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("wan21_humo")
        comfyui["workload"]["length"] = 101
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "workload.length")

    def test_anima_artifact_mismatch_is_rejected_at_the_entrypoint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("anima")
        comfyui["artifacts"][0]["sha256"] = "f" * 64
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "pinned digest")

    def test_minimax_h3_workload_mismatch_is_rejected_at_the_entrypoint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("minimax_h3")
        comfyui["workload"]["seed"] = 0
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "workload.seed")

    def test_minimax_h3_artifact_mismatch_is_rejected_at_the_entrypoint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair("minimax_h3")
        comfyui["artifacts"][0]["sha256"] = "f" * 64
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "pinned digest")

    def test_device_mismatch_is_rejected_at_the_entrypoint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dinkster, comfyui = cell_pair()
        comfyui["devices"][0]["name"] = "Different GPU"
        self._assert_rejected(tmp_path, capsys, dinkster, comfyui, "devices differ")

    def test_missing_report_file_exits_with_an_error(self, tmp_path: Path) -> None:
        dinkster, comfyui = cell_pair()
        arguments = write_pair(tmp_path, dinkster, comfyui)
        arguments[1] = str(tmp_path / "absent.json")
        with pytest.raises(SystemExit, match="could not read dinkster report"):
            main(arguments)

    def test_non_object_report_exits_with_an_error(self, tmp_path: Path) -> None:
        dinkster, comfyui = cell_pair()
        arguments = write_pair(tmp_path, dinkster, comfyui)
        Path(arguments[3]).write_text("[1, 2]")
        with pytest.raises(SystemExit, match="not a JSON object"):
            main(arguments)
