from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import tools.compare_workflow_pngs as comparer
from tools.compare_workflow_pngs import _png_from_report, compare, main


def _report(root: Path, image: Path, extra_runs: int = 0) -> Path:
    report = root / "report.json"
    retained = root / "outputs" / "0" / image.name
    retained.parent.mkdir(parents=True)
    image.replace(retained)
    payload = {
        "runs": [
            {"outputs": [{"file": str(retained.relative_to(root))}]},
            *({"outputs": [{"file": str(retained.relative_to(root))}]} for _ in range(extra_runs)),
        ]
    }
    report.write_text(json.dumps(payload))
    return report


def test_compare_reports_distinguishes_container_bytes_from_pixels(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGBA", (2, 1), (3, 4, 5, 255)).save(reference)
    Image.new("RGBA", (2, 1), (3, 4, 5, 255)).save(candidate, compress_level=0)
    reference_report = _report(tmp_path / "reference", reference)
    candidate_report = _report(tmp_path / "candidate", candidate)

    result = compare(
        _png_from_report(json.loads(reference_report.read_text()), reference_report),
        _png_from_report(json.loads(candidate_report.read_text()), candidate_report),
    )

    assert result["pixels_comparable"] is True
    assert result["max_absolute_difference"] == 0
    assert result["png_bytes_identical"] is False


def test_compare_reports_records_pixel_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGBA", (1, 1), (0, 0, 0, 255)).save(reference)
    Image.new("RGBA", (1, 1), (4, 0, 0, 255)).save(candidate)
    reference_report = _report(tmp_path / "reference", reference)
    candidate_report = _report(tmp_path / "candidate", candidate)

    result = compare(
        _png_from_report(json.loads(reference_report.read_text()), reference_report),
        _png_from_report(json.loads(candidate_report.read_text()), candidate_report),
    )

    assert result["max_absolute_difference"] == 4


def test_compare_reports_requires_one_retained_png(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"runs": [{"outputs": []}]}))

    with pytest.raises(ValueError, match="exactly one retained PNG"):
        _png_from_report(json.loads(report.read_text()), report)


def test_compare_reports_rejects_pngs_from_multiple_runs(tmp_path: Path) -> None:
    image = tmp_path / "output.png"
    Image.new("RGBA", (1, 1), (3, 4, 5, 255)).save(image)
    report = _report(tmp_path / "report", image, extra_runs=1)

    with pytest.raises(ValueError, match="exactly one retained PNG"):
        _png_from_report(json.loads(report.read_text()), report)


def test_main_writes_a_new_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGBA", (1, 1), (3, 4, 5, 255)).save(reference)
    Image.new("RGBA", (1, 1), (3, 4, 5, 255)).save(candidate)
    reference_report = _report(tmp_path / "reference", reference)
    candidate_report = _report(tmp_path / "candidate", candidate)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(comparer, "validate_workflow_report", lambda report: ())
    monkeypatch.setattr(comparer, "validate_workflow_files", lambda report, directory: ())
    monkeypatch.setattr(comparer, "workflow_comparability_problems", lambda ours, theirs: ())

    assert main([
        "--reference-report", str(reference_report),
        "--candidate-report", str(candidate_report),
        "--output", str(output),
    ]) == 0

    assert json.loads(output.read_text())["max_absolute_difference"] == 0
