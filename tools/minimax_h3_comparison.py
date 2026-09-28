"""Run and compare the ten official MiniMax H3 workflow variants."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import workflow_benchmark
from tools.workflow_benchmark_report import file_digest

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/minimax-h3-importer-api"
TEMPLATE_COMMIT = "fc427f00097817d3f7d8099c5259837fa51e1267"
TEMPLATE_PUSHED_PACIFIC = "2026-09-22 7:35 AM PDT"
FIXED_SEED = 1064
NOMINAL_DURATION_SECONDS = 2.0
MULTIFRAME_GUIDE_SECONDS = (0.6, 1.2, 2.0)
FRAME_RATE = 24
RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class Stack:
    id: str
    system: str
    commit: str
    pushed_pacific: str
    root: Path
    python: Path


STACK_PINS = {
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

ROWS = (
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

TEMPLATE_PROVENANCE = {
    "t2v": (
        "templates/video_minimax_h3_t2v.json",
        "aeadcae30ac27d8f3bedebada670ddbc5af03efc69ec2312cc859efbf660ff45",
    ),
    "i2v": (
        "templates/video_minimax_h3_i2v.json",
        "cb269e456bc741e659919cb92019fcc9a605476af1d23868b33fa740e75a9db8",
    ),
    "r2v": (
        "templates/video_minimax_h3_r2v.json",
        "466802086b46da86aaba5afe73e72f7d18dcc46f5ab79b6fad966131b840ccd8",
    ),
    "multiframe-reference": (
        "templates/video_minimax_h3_multiframe_reference.json",
        "343de410ba8dda40553db5da3021b78d9b6e22a33e32f848bac41acce20a1db5",
    ),
}


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, timeout=30
    ).strip()


def verify_source(stack: Stack) -> dict[str, Any]:
    commit = git(stack.root, "rev-parse", "HEAD")
    if commit != stack.commit:
        raise ValueError(f"{stack.id} checkout is {commit}, expected {stack.commit}")
    if git(stack.root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError(f"{stack.id} checkout is dirty")
    if not stack.python.is_file():
        raise ValueError(f"{stack.id} interpreter does not exist: {stack.python}")
    return {
        "commit": commit,
        "tree": git(stack.root, "rev-parse", "HEAD^{tree}"),
        "pushed_pacific": stack.pushed_pacific,
        "root": str(stack.root),
        "python": str(stack.python),
    }


def interpreter_inventory(stack: Stack) -> dict[str, Any]:
    code = """
import importlib.metadata, json, platform, sys
packages = {d.metadata.get('Name', d.name): d.version for d in importlib.metadata.distributions()}
result = {
    'executable': sys.executable,
    'python': platform.python_version(),
    'packages': dict(sorted(packages.items())),
}
if sys.argv[1] == 'dinkster':
    from dinkster.compose import default_pack_specs
    packs = []
    for spec in default_pack_specs():
        for pack, info in sorted((spec.packs or {}).items()):
            packs.append({
                'pack': pack,
                'artifact_digest': info.artifact_digest,
                'source': info.source,
                'version': info.version,
            })
    result['default_pack_lock_proof'] = packs
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        [str(stack.python), "-c", code, stack.system],
        cwd=stack.root,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    inventory = json.loads(completed.stdout)
    if stack.system == "dinkster" and not inventory.get("default_pack_lock_proof"):
        raise ValueError(f"{stack.id} produced no default-pack digest proof")
    inventory["sha256"] = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return inventory


def fixture_hashes() -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in (FIXTURE_ROOT / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    expected_names = {"SOURCE.json", *(f"{row}.json" for row in ROWS)}
    if set(expected) != expected_names:
        raise ValueError("official MiniMax H3 checksum manifest has an unexpected file set")
    actual = {name: file_digest(FIXTURE_ROOT / name) for name in expected_names}
    if actual != expected:
        raise ValueError("official MiniMax H3 fixture set or hashes differ")
    return actual


def _row_family(row: str) -> str:
    return row.split("--", 1)[0]


def _single_node(graph: dict[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (node_id, node) for node_id, node in graph.items() if node["class_type"] == class_type
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {class_type} node, found {len(matches)}")
    return matches[0]


def _linked_primitive(graph: dict[str, Any], link: Any, expected_type: str) -> dict[str, Any]:
    if not isinstance(link, list) or len(link) != 2 or link[0] not in graph:
        raise ValueError("workflow input is not a local node link")
    node = graph[link[0]]
    if node["class_type"] != expected_type:
        raise ValueError(f"workflow link does not resolve to {expected_type}")
    return node


def prepare_workflow(row: str, destination: Path) -> tuple[Path, Path, str]:
    source = FIXTURE_ROOT / f"{row}.json"
    graph = json.loads(source.read_text())
    seed_node, noise = _single_node(graph, "RandomNoise")
    noise["inputs"]["noise_seed"] = FIXED_SEED

    h3_nodes = [
        node
        for node in graph.values()
        if node["class_type"] in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo")
    ]
    if len(h3_nodes) != 1:
        raise ValueError(f"{row} does not have one MiniMax H3 conditioning node")
    length_expression = _linked_primitive(
        graph, h3_nodes[0]["inputs"]["length"], "ComfyMathExpression"
    )
    duration = _linked_primitive(graph, length_expression["inputs"]["values.a"], "PrimitiveFloat")
    if duration["inputs"].get("value") != 5:
        raise ValueError(f"{row} does not have the official five-second duration input")
    duration["inputs"]["value"] = NOMINAL_DURATION_SECONDS

    if row.startswith("multiframe-reference--"):
        guide_times = []
        for guide in graph.values():
            if guide["class_type"] != "MiniMaxH3AddGuide":
                continue
            expression = _linked_primitive(
                graph, guide["inputs"]["frame_idx"], "ComfyMathExpression"
            )
            guide_times.append(
                _linked_primitive(graph, expression["inputs"]["values.a"], "PrimitiveFloat")
            )
        guide_times.sort(key=lambda node: node["inputs"]["value"])
        if [node["inputs"].get("value") for node in guide_times] != [1.5, 3, 5]:
            raise ValueError("multiframe workflow guide timing inputs differ")
        for node, seconds in zip(guide_times, MULTIFRAME_GUIDE_SECONDS, strict=True):
            node["inputs"]["value"] = seconds

    primitive_steps = sorted(
        node["inputs"]["value"] for node in graph.values() if node["class_type"] == "PrimitiveInt"
    )
    expected_fast_steps = (
        8 if _row_family(row) in ("t2v", "i2v") and not row.endswith("fl2v-4step") else 4
    )
    if primitive_steps != [expected_fast_steps, 20]:
        raise ValueError(f"{row} no longer carries its official step counts")
    selected_fast = _single_node(graph, "PrimitiveBoolean")[1]["inputs"].get("value")
    if selected_fast is not (not row.endswith("no-lora")):
        raise ValueError(f"{row} model and step selector differs from its official variant")

    resolution = _single_node(graph, "ResolutionSelector")[1]["inputs"]
    if resolution.get("megapixels") != 0.4:
        raise ValueError(f"{row} no longer uses its native 0.4 megapixel resolution")
    video = _single_node(graph, "CreateVideo")[1]["inputs"]
    if video.get("fps") != FRAME_RATE:
        raise ValueError(f"{row} no longer uses the template 24 fps rate")

    destination.mkdir(parents=True, exist_ok=False)
    workflow_path = destination / "workflow.json"
    workflow_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
    family = _row_family(row)
    template_path, template_sha256 = TEMPLATE_PROVENANCE[family]
    provenance = {
        "template_commit": TEMPLATE_COMMIT,
        "template_commit_pushed_pacific": TEMPLATE_PUSHED_PACIFIC,
        "template_path": template_path,
        "template_sha256": template_sha256,
        "api_sha256": file_digest(workflow_path),
        "exporter": (
            "ComfyUI frontend 1.53.6 app.loadGraphData + app.graphToPrompt; "
            "duration and seed normalized by tools/minimax_h3_comparison.py"
        ),
        "source_api_sha256": file_digest(source),
        "normalization": {
            "fixed_seed": FIXED_SEED,
            "nominal_duration_seconds": NOMINAL_DURATION_SECONDS,
            "frame_rate": FRAME_RATE,
            "multiframe_guide_seconds": list(MULTIFRAME_GUIDE_SECONDS)
            if family == "multiframe-reference"
            else None,
        },
    }
    provenance_path = destination / "provenance.json"
    save(provenance_path, provenance)
    return workflow_path, provenance_path, f"{seed_node}.noise_seed"


def _video_from_report(report: dict[str, Any], report_root: Path) -> Path:
    report_root = report_root.resolve()
    outputs = report.get("runs", [{}])[0].get("outputs", [])
    candidates = [
        (report_root / output["file"]).resolve()
        for output in outputs
        if Path(output.get("file", "")).suffix.lower() in VIDEO_SUFFIXES
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one retained video, found {len(candidates)}")
    if not candidates[0].is_relative_to(report_root):
        raise ValueError("retained video path escapes its benchmark directory")
    return candidates[0]


def retain_video(report: dict[str, Any], report_root: Path, stack_root: Path) -> dict[str, Any]:
    source = _video_from_report(report, report_root)
    target = stack_root / ("video" + source.suffix.lower())
    shutil.copyfile(source, target)
    digest = file_digest(target)
    (stack_root / "video.sha256").write_text(f"{digest}  {target.name}\n")
    return {"file": target.name, "sha256": digest, "bytes": target.stat().st_size}


def probe_video(video: Path, ffprobe: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("video does not contain exactly one video stream")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": width,
        "height": height,
        "frame_rate": int(numerator) / int(denominator),
        "reported_frames": int(stream["nb_frames"])
        if stream.get("nb_frames") not in (None, "N/A")
        else None,
    }


def decode_video(video: Path, ffmpeg: str, width: int, height: int) -> Any:
    import numpy as np

    completed = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True,
        capture_output=True,
        timeout=300,
    )
    frame_bytes = width * height * 3
    if not completed.stdout or len(completed.stdout) % frame_bytes:
        raise ValueError("decoded RGB byte count is not a whole number of frames")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape((-1, height, width, 3))


def compare_frame_arrays(reference: Any, candidate: Any) -> dict[str, Any]:
    import numpy as np

    if reference.shape != candidate.shape or len(reference.shape) != 4 or reference.shape[-1] != 3:
        return {
            "comparable": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    difference = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    squared = difference.astype(np.float64) ** 2
    frames = []
    for index in range(reference.shape[0]):
        frame = difference[index]
        frames.append(
            {
                "frame": index,
                "mean_absolute_difference": float(frame.mean()),
                "root_mean_square_difference": float(math.sqrt(squared[index].mean())),
                "maximum_absolute_difference": int(frame.max()),
                "differing_pixel_fraction": float(np.any(frame != 0, axis=2).mean()),
            }
        )
    return {
        "comparable": True,
        "frame_count": int(reference.shape[0]),
        "width": int(reference.shape[2]),
        "height": int(reference.shape[1]),
        "mean_absolute_difference": float(difference.mean()),
        "root_mean_square_difference": float(math.sqrt(squared.mean())),
        "maximum_absolute_difference": int(difference.max()),
        "differing_pixel_fraction": float(np.any(difference != 0, axis=3).mean()),
        "frames": frames,
    }


def compare_videos(reference: Path, candidate: Path, ffmpeg: str, ffprobe: str) -> dict[str, Any]:
    reference_probe = probe_video(reference, ffprobe)
    candidate_probe = probe_video(candidate, ffprobe)
    if (reference_probe["width"], reference_probe["height"]) != (
        candidate_probe["width"],
        candidate_probe["height"],
    ):
        return {
            "comparable": False,
            "reference": reference_probe,
            "candidate": candidate_probe,
        }
    reference_frames = decode_video(
        reference, ffmpeg, reference_probe["width"], reference_probe["height"]
    )
    candidate_frames = decode_video(
        candidate, ffmpeg, candidate_probe["width"], candidate_probe["height"]
    )
    result = compare_frame_arrays(reference_frames, candidate_frames)
    result.update({"reference": reference_probe, "candidate": candidate_probe})
    return result


def _status_cell(result: dict[str, Any]) -> str:
    video = result.get("video")
    return video["sha256"] if result.get("status") == "completed" and video else "FAILED"


def _difference_cell(comparison: dict[str, Any] | None) -> str:
    if not comparison or not comparison.get("comparable"):
        return "n/a"
    return (
        f"MAD {comparison['mean_absolute_difference']:.4f}; "
        f"RMSE {comparison['root_mean_square_difference']:.4f}; "
        f"max {comparison['maximum_absolute_difference']}"
    )


def comparison_table(manifest: dict[str, Any]) -> str:
    lines = [
        "| Workflow | ComfyUI SHA-256 | Pre-generalization SHA-256 | "
        "Pre vs ComfyUI | Current SHA-256 | Current vs ComfyUI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in ROWS:
        result = manifest["workflows"][row]
        lines.append(
            "| "
            + " | ".join(
                (
                    row,
                    _status_cell(result["stacks"]["comfyui"]),
                    _status_cell(result["stacks"]["dinkster-pre-generalization"]),
                    _difference_cell(result["comparisons"].get("dinkster-pre-generalization")),
                    _status_cell(result["stacks"]["dinkster-current"]),
                    _difference_cell(result["comparisons"].get("dinkster-current")),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_stack(
    stack: Stack,
    row: str,
    workflow: Path,
    provenance: Path,
    seed_input: str,
    artifacts: Path,
    root: Path,
    reference: Stack,
    gpu_uuid: str,
    port: int,
) -> dict[str, Any]:
    benchmark_root = root / "benchmark"
    arguments = [
        "--workflow",
        str(workflow),
        "--workflow-provenance",
        str(provenance),
        "--artifacts",
        str(artifacts),
        "--repo",
        str(stack.root if stack.system == "dinkster" else Path(__file__).resolve().parents[1]),
        "--server-python",
        str(stack.python),
        "--reference-root",
        str(reference.root),
        "--reference-commit",
        reference.commit,
        "--family",
        "minimax-h3",
        "--seed-input",
        seed_input,
        "--seed",
        str(FIXED_SEED),
        "--warm-runs",
        "0",
        "--port",
        str(port),
        "--gpu-uuid",
        gpu_uuid,
        "--output",
        str(benchmark_root),
    ]
    exit_code = workflow_benchmark.main(stack.system, arguments)
    report = json.loads((benchmark_root / "report.json").read_text())
    result: dict[str, Any] = {
        "status": "completed" if exit_code == 0 and report.get("all_ok") else "failed",
        "benchmark_exit_code": exit_code,
        "report": "benchmark/report.json",
        "errors": report.get("errors", []),
        "validation_errors": report.get("validation_errors", []),
    }
    if result["status"] == "completed":
        result["video"] = retain_video(report, benchmark_root, root)
    save(root / "result.json", result)
    return result


def run(args: argparse.Namespace) -> int:
    run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run id must use YYYYMMDDTHHMMSSZ")
    output = args.output_root.absolute() / run_id
    output.mkdir(parents=True, exist_ok=False)
    stacks = (
        Stack(
            "comfyui",
            *STACK_PINS["comfyui"],
            args.comfyui_root.absolute(),
            args.comfyui_python.absolute(),
        ),
        Stack(
            "dinkster-pre-generalization",
            *STACK_PINS["dinkster-pre-generalization"],
            args.dinkster_pre_root.absolute(),
            args.dinkster_pre_python.absolute(),
        ),
        Stack(
            "dinkster-current",
            STACK_PINS["dinkster-current"][0],
            args.dinkster_current_commit,
            args.dinkster_current_pushed_pacific,
            args.dinkster_current_root.absolute(),
            args.dinkster_current_python.absolute(),
        ),
    )
    reference = stacks[0]
    preflight: dict[str, Any] = {"fixture_sha256": fixture_hashes(), "stacks": {}}
    for stack in stacks:
        source = verify_source(stack)
        inventory = interpreter_inventory(stack)
        inventory_file = output / "preflight" / f"{stack.id}-inventory.json"
        save(inventory_file, inventory)
        preflight["stacks"][stack.id] = {
            **source,
            "inventory": str(inventory_file.relative_to(output)),
            "inventory_sha256": file_digest(inventory_file),
        }
    save(output / "preflight.json", preflight)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "informational_only": True,
        "fixed_seed": FIXED_SEED,
        "nominal_duration_seconds": NOMINAL_DURATION_SECONDS,
        "frame_rate": FRAME_RATE,
        "gpu_uuid": args.gpu_uuid,
        "preflight": "preflight.json",
        "stacks": preflight["stacks"],
        "workflows": {},
    }
    port = args.port_base
    for row in ROWS:
        row_root = output / "workflows" / row
        workflow, provenance, seed_input = prepare_workflow(row, row_root / "input")
        row_result: dict[str, Any] = {"stacks": {}, "comparisons": {}}
        for stack in stacks:
            stack_root = row_root / stack.id
            stack_root.mkdir(parents=True)
            try:
                result = run_stack(
                    stack,
                    row,
                    workflow,
                    provenance,
                    seed_input,
                    args.artifacts.absolute(),
                    stack_root,
                    reference,
                    args.gpu_uuid,
                    port,
                )
            except Exception as error:
                result = {
                    "status": "failed",
                    "errors": [f"{type(error).__name__}: {error}"],
                    "validation_errors": [],
                }
                save(stack_root / "result.json", result)
            row_result["stacks"][stack.id] = result
            port += 1
        comfy_result = row_result["stacks"]["comfyui"]
        for stack_id in ("dinkster-pre-generalization", "dinkster-current"):
            candidate = row_result["stacks"][stack_id]
            if comfy_result["status"] != "completed" or candidate["status"] != "completed":
                row_result["comparisons"][stack_id] = None
                continue
            try:
                comparison = compare_videos(
                    row_root / "comfyui" / comfy_result["video"]["file"],
                    row_root / stack_id / candidate["video"]["file"],
                    args.ffmpeg,
                    args.ffprobe,
                )
            except Exception as error:
                comparison = {
                    "comparable": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            save(row_root / f"{stack_id}-vs-comfyui.json", comparison)
            row_result["comparisons"][stack_id] = comparison
        manifest["workflows"][row] = row_result
        save(output / "manifest.json", manifest)
    table = comparison_table(manifest)
    (output / "comparison.md").write_text(table)
    save(output / "manifest.json", manifest)
    print(f"MiniMax H3 comparison: {output}")
    print(table, end="")
    return 0


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--comfyui-python", type=Path, required=True)
    parser.add_argument("--dinkster-pre-root", type=Path, required=True)
    parser.add_argument("--dinkster-pre-python", type=Path, required=True)
    parser.add_argument("--dinkster-current-root", type=Path, required=True)
    parser.add_argument("--dinkster-current-python", type=Path, required=True)
    parser.add_argument(
        "--dinkster-current-commit",
        default=STACK_PINS["dinkster-current"][1],
        help="full current-main revision to compare",
    )
    parser.add_argument(
        "--dinkster-current-pushed-pacific",
        default=STACK_PINS["dinkster-current"][2],
        help="the current-main revision's pushed date and Pacific time",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--port-base", type=int, default=18760)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    if not 1 <= args.port_base <= 65506:
        parser.error("--port-base must leave room for all 30 executions")
    if not re.fullmatch(r"[0-9a-f]{40}", args.dinkster_current_commit):
        parser.error("--dinkster-current-commit must be a full lowercase hexadecimal revision")
    if not args.dinkster_current_pushed_pacific.endswith((" PST", " PDT")):
        parser.error("--dinkster-current-pushed-pacific must state a labeled Pacific time")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_arguments()))
