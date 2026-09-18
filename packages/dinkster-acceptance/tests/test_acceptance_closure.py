from __future__ import annotations

import json
from pathlib import Path

from dinkster_acceptance.golden_files import load_model_sampling_flux_golden
from dinkster_acceptance.import_check import check_imports


def test_packaged_golden_matches_inference_source() -> None:
    package = Path(__file__).resolve().parents[1]
    source = json.loads(
        (
            package.parent / "dinkster-inference-torch/tests/goldens/model_sampling_flux.json"
        ).read_text(encoding="utf-8")
    )
    selected = next(case for case in source["cases"] if case["name"] == "non_square")
    packaged = load_model_sampling_flux_golden()
    assert packaged == {
        "comfyui_commit": source["comfyui_commit"],
        "case": {
            "name": selected["name"],
            "max_shift": selected["max_shift"],
            "base_shift": selected["base_shift"],
            "width": selected["width"],
            "height": selected["height"],
            "shift": selected["shift"],
            "schedule": selected["schedules"]["normal"],
        },
    }


def test_import_closure_rejects_test_only_dependencies() -> None:
    report = check_imports()
    assert report["status"] == "ok"
    assert report["node"] == "acceptance.model_sampling_flux"
