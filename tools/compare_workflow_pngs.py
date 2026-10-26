"""Compare one retained PNG from two workflow benchmark reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.workflow_benchmark_report import (
    validate_workflow_files,
    validate_workflow_report,
    workflow_comparability_problems,
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _png_from_report(report: dict[str, Any], report_path: Path) -> Path:
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{report_path}: report has no runs")
    pngs = [
        report_path.parent / output["file"]
        for run in runs
        if isinstance(run, dict)
        for outputs in [run.get("outputs")]
        if isinstance(outputs, list)
        for output in outputs
        if isinstance(output, dict)
        and isinstance(output.get("file"), str)
        and Path(output["file"]).suffix.lower() == ".png"
    ]
    if len(pngs) != 1:
        raise ValueError(f"{report_path}: expected exactly one retained PNG, found {len(pngs)}")
    image = pngs[0]
    if not image.is_file():
        raise ValueError(f"{report_path}: retained PNG is missing: {image}")
    return image


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"{path}: report is not an object")
    problems = [*validate_workflow_report(report), *validate_workflow_files(report, path.parent)]
    if problems:
        raise ValueError(f"{path}: invalid workflow evidence: {'; '.join(problems)}")
    return report


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.int16)


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_pixels = _pixels(reference)
    candidate_pixels = _pixels(candidate)
    result: dict[str, Any] = {
        "reference": {
            "file": str(reference),
            "sha256": _digest(reference),
            "bytes": reference.stat().st_size,
            "shape": list(reference_pixels.shape),
        },
        "candidate": {
            "file": str(candidate),
            "sha256": _digest(candidate),
            "bytes": candidate.stat().st_size,
            "shape": list(candidate_pixels.shape),
        },
        "png_bytes_identical": reference.read_bytes() == candidate.read_bytes(),
    }
    if reference_pixels.shape != candidate_pixels.shape:
        result.update({"pixels_comparable": False, "max_absolute_difference": None})
        return result
    difference = np.abs(reference_pixels - candidate_pixels)
    result.update(
        {
            "pixels_comparable": True,
            "max_absolute_difference": int(difference.max()) if difference.size else 0,
            "mean_absolute_difference": float(difference.mean()) if difference.size else 0.0,
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must not already exist")
    reference_report = _load_report(args.reference_report)
    candidate_report = _load_report(args.candidate_report)
    problems = workflow_comparability_problems(candidate_report, reference_report)
    if problems:
        parser.error(f"reports are not comparable: {'; '.join(problems)}")
    reference = _png_from_report(reference_report, args.reference_report)
    candidate = _png_from_report(candidate_report, args.candidate_report)
    result = compare(reference, candidate)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
