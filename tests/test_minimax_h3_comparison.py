from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import minimax_h3_comparison as comparison


def test_official_matrix_and_stack_pins_are_exact() -> None:
    assert comparison.ROWS == (
        "t2v--no-lora",
        "t2v--fl2v-8step",
        "t2v--fl2v-4step",
        "i2v--no-lora",
        "i2v--fl2v-8step",
        "i2v--fl2v-4step",
        "r2v--no-lora",
        "r2v--ref2v-4step",
        "multiframe-reference--no-lora",
        "multiframe-reference--ref2v-4step",
    )
    assert comparison.STACK_PINS == {
        "comfyui": (
            "comfyui",
            "b5cc8830279eae909a59de030af1e50761c36751",
            "2026-09-22 8:16 PM PDT",
        ),
        "dinkster-pre-generalization": (
            "dinkster",
            "246b3ff3f5f78c410d4d77bc6deeb2a8aaccd9a0",
            "2026-09-24 1:51 PM PDT",
        ),
        "dinkster-current": (
            "dinkster",
            "1c231d929248a95409e331aea199cdc6223835a8",
            "2026-09-24 5:36 PM PDT",
        ),
    }
    assert comparison.TEMPLATE_COMMIT == "fc427f00097817d3f7d8099c5259837fa51e1267"
    assert comparison.TEMPLATE_PUSHED_PACIFIC == "2026-09-22 7:35 AM PDT"
    assert comparison.fixture_hashes()


def test_every_official_row_prepares_with_its_own_steps(tmp_path: Path) -> None:
    for row in comparison.ROWS:
        workflow, _provenance, _seed_input = comparison.prepare_workflow(row, tmp_path / row)
        graph = json.loads(workflow.read_text())
        steps = sorted(
            node["inputs"]["value"]
            for node in graph.values()
            if node["class_type"] == "PrimitiveInt"
        )
        expected_fast = (
            8 if row.split("--", 1)[0] in ("t2v", "i2v") and not row.endswith("fl2v-4step") else 4
        )
        assert steps == [expected_fast, 20]


def test_workflow_normalization_preserves_native_contract(tmp_path: Path) -> None:
    workflow, provenance, seed_input = comparison.prepare_workflow(
        "multiframe-reference--ref2v-4step", tmp_path / "prepared"
    )
    graph = json.loads(workflow.read_text())
    primitive_values = [
        node["inputs"]["value"] for node in graph.values() if node["class_type"] == "PrimitiveFloat"
    ]
    noise = next(node for node in graph.values() if node["class_type"] == "RandomNoise")
    resolution = next(node for node in graph.values() if node["class_type"] == "ResolutionSelector")
    video = next(node for node in graph.values() if node["class_type"] == "CreateVideo")
    steps = sorted(
        node["inputs"]["value"] for node in graph.values() if node["class_type"] == "PrimitiveInt"
    )
    assert primitive_values == [comparison.NOMINAL_DURATION_SECONDS, 0.6, 1.2, 2.0]
    assert noise["inputs"]["noise_seed"] == comparison.FIXED_SEED
    assert seed_input.endswith(".noise_seed")
    assert resolution["inputs"]["megapixels"] == 0.4
    assert video["inputs"]["fps"] == 24
    assert steps == [4, 20]
    record = json.loads(provenance.read_text())
    assert record["api_sha256"] == comparison.file_digest(workflow)
    assert record["normalization"]["multiframe_guide_seconds"] == [0.6, 1.2, 2.0]


def test_per_frame_differences_use_rgb_values_and_frame_order() -> None:
    reference = np.zeros((2, 1, 2, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[0, 0, 0] = [3, 4, 0]
    candidate[1, 0, 1] = [0, 0, 12]
    result = comparison.compare_frame_arrays(reference, candidate)
    assert result["comparable"] is True
    assert result["frame_count"] == 2
    assert result["maximum_absolute_difference"] == 12
    assert result["differing_pixel_fraction"] == 0.5
    assert result["frames"][0] == {
        "frame": 0,
        "mean_absolute_difference": 7 / 6,
        "root_mean_square_difference": (25 / 6) ** 0.5,
        "maximum_absolute_difference": 4,
        "differing_pixel_fraction": 0.5,
    }
    assert result["frames"][1]["root_mean_square_difference"] == 24**0.5

    swapped = comparison.compare_frame_arrays(reference, candidate[::-1])
    assert swapped["frames"] != result["frames"]
    missing = comparison.compare_frame_arrays(reference, candidate[:1])
    assert missing["comparable"] is False


def test_comparison_table_has_exactly_ten_data_lines_and_keeps_failures() -> None:
    workflows = {}
    for index, row in enumerate(comparison.ROWS):
        completed = {"status": "completed", "video": {"sha256": f"{index:064x}"}}
        failed = {"status": "failed"}
        workflows[row] = {
            "stacks": {
                "comfyui": completed,
                "dinkster-pre-generalization": failed if index == 0 else completed,
                "dinkster-current": completed,
            },
            "comparisons": {
                "dinkster-pre-generalization": None,
                "dinkster-current": {
                    "comparable": True,
                    "mean_absolute_difference": 1.25,
                    "root_mean_square_difference": 2.5,
                    "maximum_absolute_difference": 7,
                },
            },
        }
    table = comparison.comparison_table({"workflows": workflows})
    assert len(table.splitlines()) == 12
    assert table.count("\n| t2v--") == 3
    assert f"| t2v--no-lora | {'0' * 64} | FAILED | n/a |" in table
    assert "MAD 1.2500; RMSE 2.5000; max 7" in table


def test_retained_video_cannot_escape_benchmark_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    report_root = tmp_path / "benchmark"
    report_root.mkdir()
    report = {"runs": [{"outputs": [{"file": "../outside.mp4"}]}]}
    with pytest.raises(ValueError, match="escapes"):
        comparison._video_from_report(report, report_root)


def test_matrix_run_writes_stable_layout_and_continues_failed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        comparison,
        "verify_source",
        lambda stack: {
            "commit": stack.commit,
            "tree": "a" * 40,
            "pushed_pacific": stack.pushed_pacific,
        },
    )
    monkeypatch.setattr(
        comparison,
        "interpreter_inventory",
        lambda stack: {"packages": {"stack": stack.id}},
    )

    def run_stack(stack: Any, row: str, *arguments: Any) -> dict[str, Any]:
        root = arguments[4]
        port = arguments[-1]
        calls.append((row, stack.id, port))
        if row == comparison.ROWS[0] and stack.id == "dinkster-pre-generalization":
            return {"status": "failed"}
        video = root / "video.mp4"
        video.write_bytes(f"{row}:{stack.id}".encode())
        return {
            "status": "completed",
            "video": {
                "file": video.name,
                "sha256": comparison.file_digest(video),
                "bytes": video.stat().st_size,
            },
        }

    monkeypatch.setattr(comparison, "run_stack", run_stack)
    monkeypatch.setattr(
        comparison,
        "compare_videos",
        lambda *_arguments: {
            "comparable": True,
            "mean_absolute_difference": 1.0,
            "root_mean_square_difference": 2.0,
            "maximum_absolute_difference": 3,
        },
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    interpreter = tmp_path / "python"
    interpreter.touch()
    args = argparse.Namespace(
        run_id="20260925T120000Z",
        output_root=tmp_path / "results",
        artifacts=tmp_path / "artifacts.json",
        gpu_uuid="GPU-test",
        comfyui_root=checkout,
        comfyui_python=interpreter,
        dinkster_pre_root=checkout,
        dinkster_pre_python=interpreter,
        dinkster_current_root=checkout,
        dinkster_current_python=interpreter,
        port_base=19000,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
    )
    assert comparison.run(args) == 0
    result = args.output_root / args.run_id
    manifest = json.loads((result / "manifest.json").read_text())
    assert len(calls) == 30
    assert calls[0] == (comparison.ROWS[0], "comfyui", 19000)
    assert calls[-1] == (comparison.ROWS[-1], "dinkster-current", 19029)
    assert set(manifest["workflows"]) == set(comparison.ROWS)
    assert (
        manifest["workflows"][comparison.ROWS[0]]["comparisons"]["dinkster-pre-generalization"]
        is None
    )
    assert len((result / "comparison.md").read_text().splitlines()) == 12
