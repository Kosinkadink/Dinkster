"""Side-by-side comparison of one Dinkster and one ComfyUI inference
benchmark report for the same cell.

Workflow HTTP reports use tools/workflow_benchmark_report.py and compare the
submitted graphs, authenticated artifacts and recorded revisions, not family
labels. Retain each report beside its raw evidence files. Workflow and
historical cell reports cannot be mixed. The cell contract follows below.

Both inputs are gated through validate_benchmark_report and must
describe the same cell before any number is compared: same family and
accelerator, identical device identities, eager mode on both sides
(the head-to-head contract), an equal workload, identical input
artifacts by digest per role, and all_ok true on both reports.
sampler_id and scheduler_id are compared after stripping the "dinkster."
prefix, which is the only naming difference between the two runners
for the same algorithm.

Ratios are dinkster / comfyui, so a ratio below 1.0 means Dinkster was
faster (timings) or smaller (memory). Ratios are only computed where
the two systems measure the same thing: cold total_s, the warm run
medians, and peak_rss_bytes. Cold per-phase splits are shown but not
ratioed - ComfyUI uploads weights to the device lazily inside the
first encode/sample/decode (and applies LoRA patches lazily inside the
first sample), so only cold totals are apples-to-apples. Allocator
peaks are shown but not ratioed - ComfyUI's dynamic VRAM loading holds
weights outside the caching allocator, so its allocator peaks do not
cover the same bytes as Dinkster's; its device-global
peak_device_used_bytes is echoed as informational.

Usage:
    python scripts/compare_benchmark_reports.py \
        --dinkster /tmp/dinkster_zimage.json --comfyui /tmp/comfyui_zimage.json \
        [--json /tmp/comparison_zimage.json]
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dinkster_inference import BUILTIN_FAMILIES_BY_ID
from dinkster_workers.backend_env import BENCHMARK_ACCELERATORS, validate_benchmark_report

_COLD_PHASES = ("load_s", "encode_s", "sample_s", "decode_s", "total_s")
_AUDIO_COLD_PHASES = ("load_s", "encode_s", "audio_encode_s", "sample_s", "decode_s", "total_s")
_AUDIO_FAMILIES = frozenset({"wan21_infinitetalk", "wan21_humo"})
_WARM_MEDIANS = ("median_sample_s", "median_decode_s", "median_total_s")
#: Cross-system ratios exist only for measurements with the same
#: coverage on both sides; see the module docstring.
_RATIOED_COLD = frozenset({"total_s"})
_RATIOED_MEMORY = frozenset({"peak_rss_bytes"})
_MEMORY_FIELDS = ("peak_allocated_bytes", "peak_reserved_bytes", "peak_rss_bytes")

#: Workload fields that must be equal verbatim for the cell to match.
_EQUAL_WORKLOAD_FIELDS = (
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "width",
    "height",
    "cfg",
    "guidance",
    "warm_runs",
    "length",
    "lora_strength_model",
    "lora_strength_clip",
    "motion_frame_count",
    "audio_scale",
    "speaker_mask_layout",
)

_NOTES = (
    "ratios are dinkster / comfyui; below 1.0 means Dinkster was faster or smaller",
    "cold per-phase splits carry no ratio: ComfyUI uploads weights to the"
    " device lazily inside the first encode/sample/decode, so only cold"
    " totals are apples-to-apples",
    "allocator peaks carry no ratio: ComfyUI's dynamic VRAM loading holds"
    " weights outside the caching allocator",
)


def _algorithm(value: object) -> object:
    """The sampler/scheduler name with the Dinkster registry prefix removed."""
    if isinstance(value, str) and value.startswith("dinkster."):
        return value[len("dinkster.") :]
    return value


def gate_problems(report: dict[str, Any], label: str) -> tuple[str, ...]:
    """Validator problems for one report, each prefixed with its label."""
    if "report_kind" in report:
        from tools.workflow_benchmark_report import validate_workflow_report

        return tuple(f"{label}: {problem}" for problem in validate_workflow_report(report))
    accelerator = report.get("accelerator")
    if accelerator not in BENCHMARK_ACCELERATORS:
        return (f"{label}: accelerator is not one of {BENCHMARK_ACCELERATORS}",)
    family_id = report.get("family_id")
    registered = BUILTIN_FAMILIES_BY_ID.get(family_id) if isinstance(family_id, str) else None
    engine = None if registered is None else registered.engine
    problems = validate_benchmark_report(
        report,
        accelerator=accelerator,
        canonical_evidence=True,
        residency_route_roles=() if engine is None else engine.residency_route_roles,
        requires_accelerator_residency=(
            False if engine is None else engine.requires_accelerator_residency
        ),
    )
    return tuple(f"{label}: {problem}" for problem in problems)


def _artifact_digests(report: dict[str, Any]) -> dict[str, str]:
    return {entry["role"]: entry["sha256"] for entry in report.get("artifacts", [])}


def comparability_problems(dinkster: dict[str, Any], comfyui: dict[str, Any]) -> tuple[str, ...]:
    """Reasons the two validated reports do not describe the same cell."""
    if "report_kind" in dinkster or "report_kind" in comfyui:
        from tools.workflow_benchmark_report import workflow_comparability_problems

        if dinkster.get("report_kind") != comfyui.get("report_kind"):
            return ("cannot compare workflow HTTP and historical cell reports",)
        return workflow_comparability_problems(dinkster, comfyui)
    problems: list[str] = []
    if dinkster.get("system") != "dinkster":
        problems.append(
            f"--dinkster report records system {dinkster.get('system')!r}, not 'dinkster'"
        )
    if comfyui.get("system") != "comfyui":
        problems.append(f"--comfyui report records system {comfyui.get('system')!r}, not 'comfyui'")
    for name in ("family", "accelerator"):
        if dinkster.get(name) != comfyui.get(name):
            problems.append(
                f"{name} differs: dinkster {dinkster.get(name)!r} vs comfyui {comfyui.get(name)!r}"
            )
    if dinkster.get("variant") != comfyui.get("variant"):
        problems.append(
            f"variant differs: dinkster {dinkster.get('variant')!r} vs "
            f"comfyui {comfyui.get('variant')!r}"
        )
    if dinkster.get("devices") != comfyui.get("devices"):
        problems.append("devices differ; the runs did not measure the same hardware")
    for label, report in (("dinkster", dinkster), ("comfyui", comfyui)):
        if report.get("mode") != "eager":
            problems.append(
                f"{label} report mode is {report.get('mode')!r};"
                " the head-to-head comparison is eager against eager"
            )
        if report.get("all_ok") is not True:
            problems.append(f"{label} report is not all_ok; a failed run is not evidence")
    dinkster_workload = dinkster.get("workload") or {}
    comfyui_workload = comfyui.get("workload") or {}
    for name in _EQUAL_WORKLOAD_FIELDS:
        if dinkster_workload.get(name) != comfyui_workload.get(name):
            problems.append(
                f"workload.{name} differs: dinkster {dinkster_workload.get(name)!r}"
                f" vs comfyui {comfyui_workload.get(name)!r}"
            )
    for name in ("sampler_id", "scheduler_id"):
        ours = _algorithm(dinkster_workload.get(name))
        theirs = _algorithm(comfyui_workload.get(name))
        if ours != theirs:
            problems.append(
                f"workload.{name} differs: dinkster {dinkster_workload.get(name)!r}"
                f" vs comfyui {comfyui_workload.get(name)!r}"
            )
    if _artifact_digests(dinkster) != _artifact_digests(comfyui):
        problems.append("artifact digests differ; the runs did not measure the same inputs")
    dinkster_attention = dinkster.get("attention") or {"requested_policy": "auto"}
    comfyui_attention = comfyui.get("attention") or {"requested_policy": "auto"}
    if dinkster_attention.get("requested_policy") != comfyui_attention.get("requested_policy"):
        problems.append(
            "attention.requested_policy differs: "
            f"dinkster {dinkster_attention.get('requested_policy')!r} vs "
            f"comfyui {comfyui_attention.get('requested_policy')!r}"
        )
    return tuple(problems)


def _ratio(dinkster_value: object, comfyui_value: object) -> float | None:
    if not isinstance(dinkster_value, (int, float)) or not isinstance(comfyui_value, (int, float)):
        return None
    if comfyui_value == 0:
        return None
    return round(dinkster_value / comfyui_value, 4)


def _row(dinkster_value: object, comfyui_value: object, *, ratioed: bool) -> dict[str, Any]:
    return {
        "dinkster": dinkster_value,
        "comfyui": comfyui_value,
        "ratio": _ratio(dinkster_value, comfyui_value) if ratioed else None,
    }


def build_comparison(dinkster: dict[str, Any], comfyui: dict[str, Any]) -> dict[str, Any]:
    """The structured side-by-side comparison of two same-cell reports."""
    if "report_kind" in dinkster:
        from tools.workflow_benchmark_report import compare_workflows

        return compare_workflows(dinkster, comfyui)
    dinkster_cold = dinkster["timings"]["cold"]
    comfyui_cold = comfyui["timings"]["cold"]
    cold_phases = _AUDIO_COLD_PHASES if dinkster["family"] in _AUDIO_FAMILIES else _COLD_PHASES
    cold = {
        phase: _row(
            dinkster_cold.get(phase), comfyui_cold.get(phase), ratioed=phase in _RATIOED_COLD
        )
        for phase in cold_phases
    }
    if dinkster["family"] == "lora":
        cold["lora_s"] = _row(
            dinkster_cold.get("lora_s"), comfyui_cold.get("lora_s"), ratioed=False
        )
    dinkster_warm = dinkster["timings"]["warm"]
    comfyui_warm = comfyui["timings"]["warm"]
    warm = {
        median: _row(dinkster_warm.get(median), comfyui_warm.get(median), ratioed=True)
        for median in _WARM_MEDIANS
    }
    dinkster_memory = dinkster["memory"]
    comfyui_memory = comfyui["memory"]
    memory = {
        name: _row(
            dinkster_memory.get(name), comfyui_memory.get(name), ratioed=name in _RATIOED_MEMORY
        )
        for name in _MEMORY_FIELDS
    }
    memory["residual_allocated_bytes"] = _row(
        dinkster_memory.get("residual_allocated_bytes"),
        comfyui_memory.get("residual_allocated_bytes"),
        ratioed=False,
    )
    comparison: dict[str, Any] = {
        "cell": {
            "family": dinkster["family"],
            "accelerator": dinkster["accelerator"],
            "mode": "eager",
        },
        "placement": {
            "dinkster": dinkster.get("placement"),
            "comfyui": comfyui.get("placement"),
        },
        "attention": {
            "dinkster": dinkster.get("attention") or {"requested_policy": "auto"},
            "comfyui": comfyui.get("attention") or {"requested_policy": "auto"},
        },
        "workload": dict(dinkster["workload"]),
        "timings": {"cold": cold, "warm": warm},
        "memory": memory,
        "comfyui_peak_device_used_bytes": comfyui_memory.get("peak_device_used_bytes"),
        "notes": list(_NOTES),
    }
    if dinkster.get("variant") is not None:
        comparison["variant"] = dinkster["variant"]
    if dinkster["family"] == "minimax_h3":
        comparison["execution_path"] = {
            "dinkster": dinkster.get("execution_path"),
            "comfyui": comfyui.get("execution_path"),
        }
    return comparison


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    return f"{value}"


def format_comparison(comparison: dict[str, Any]) -> str:
    """The comparison as an aligned plain-text table."""
    if "report_kind" in comparison:
        return json.dumps(comparison, indent=2, sort_keys=True)
    cell = comparison["cell"]
    placement = comparison["placement"]
    lines = [
        f"cell: {cell['family']} / {cell['mode']} on {cell['accelerator']}",
        f"placement: dinkster={placement['dinkster']}, comfyui={placement['comfyui']}",
        f"attention: {comparison['attention']['dinkster'].get('requested_policy')}",
    ]
    if comparison.get("variant") is not None:
        lines.append(f"variant: {comparison['variant']}")
    execution_path = comparison.get("execution_path")
    if isinstance(execution_path, dict):
        lines.append(
            f"execution path: dinkster={execution_path.get('dinkster')}, "
            f"comfyui={execution_path.get('comfyui')}"
        )
    lines.append("")
    width = 30
    header = f"{'':{width}} {'dinkster':>16} {'comfyui':>16} {'ratio':>10}"
    lines.append(header)
    for section, rows in (
        ("cold", comparison["timings"]["cold"]),
        ("warm", comparison["timings"]["warm"]),
        ("memory", comparison["memory"]),
    ):
        for name, row in rows.items():
            label = f"{section}  {name}"
            lines.append(
                f"{label:{width}} {_format_value(row['dinkster']):>16}"
                f" {_format_value(row['comfyui']):>16} {_format_value(row['ratio']):>10}"
            )
    device_used = comparison.get("comfyui_peak_device_used_bytes")
    if device_used is not None:
        label = "memory  peak_device_used_bytes"
        lines.append(f"{label:{width}} {'-':>16} {device_used:>16} {'-':>10}")
    lines.append("")
    for note in comparison["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def _load_report(path: Path, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        sys.exit(f"error: could not read {label} report {path}: {error}")
    if not isinstance(loaded, dict):
        sys.exit(f"error: {label} report {path} is not a JSON object")
    return loaded


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dinkster", type=Path, required=True, help="Dinkster benchmark report JSON"
    )
    parser.add_argument("--comfyui", type=Path, required=True, help="ComfyUI benchmark report JSON")
    parser.add_argument("--json", type=Path, default=None, help="write the comparison JSON here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    dinkster = _load_report(arguments.dinkster, "dinkster")
    comfyui = _load_report(arguments.comfyui, "comfyui")
    problems = [
        *gate_problems(dinkster, "dinkster"),
        *gate_problems(comfyui, "comfyui"),
    ]
    if not problems:
        problems.extend(comparability_problems(dinkster, comfyui))
    if not problems and "report_kind" in dinkster:
        from tools.workflow_benchmark_report import validate_workflow_files

        for label, report, path in (
            ("dinkster", dinkster, arguments.dinkster),
            ("comfyui", comfyui, arguments.comfyui),
        ):
            problems.extend(f"{label}: {p}" for p in validate_workflow_files(report, path.parent))
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    comparison = build_comparison(dinkster, comfyui)
    print(format_comparison(comparison))
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
        print(f"\ncomparison written: {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
