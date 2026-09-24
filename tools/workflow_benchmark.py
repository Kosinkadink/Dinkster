"""Run an exported API workflow against a fresh ordinary Dinkster or ComfyUI server.

Use scripts/benchmark_inference.py or scripts/benchmark_comfyui.py from the
dinkster-evidence checkout with --workflow to select this contract before Torch
or residency bootstrap. Example from the Dinkster checkout:

  python ../dinkster-evidence/scripts/benchmark_inference.py \
    --workflow tests/fixtures/lumina2-workflow-api.json \
    --workflow-provenance export.json --artifacts artifacts.json \
    --repo /work/Dinkster --server-python /work/Dinkster/.venv-gpu/bin/python \
    --reference-root /work/ComfyUI --reference-commit FULL_COMMIT \
    --seed-input 48:33.seed --seed 1064 --warm-runs 5 \
    --gpu-uuid GPU-UUID --port 18765 --output /evidence/dinkster-lumina

artifacts.json is a list of {path, category, name, sha256}; paths are absolute,
category is a ComfyUI model-folder category (or input), and name is its relative
workflow filename. export.json records template_commit, template_path,
template_sha256, api_sha256 and exporter. No fixed model or reference revision
is embedded here. --offline-check authenticates inputs and prepares commands
without starting servers, querying GPUs or importing Torch.

The executor currently uses POSIX session containment. Hold normal host/GPU
locks around physical runs; this tool neither grants nor steals a GPU claim.
Reports retain public histories, output files, sampled RSS/device memory and
client wall intervals. GPU reports require source-hashed per-worker allocator
windows; CPU v1 reports leave allocator fields null. Raw logs are local evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from tools.workflow_benchmark_observer import (
    ObserverInstallation,
    validate_free_response,
    validate_window,
)
from tools.workflow_benchmark_process import MemorySampler, NvmlDevice, OwnedServer
from tools.workflow_benchmark_report import (
    ALLOCATOR_PEAK_SCOPE,
    REPORT_KIND,
    comfyui_history_status,
    dinkster_journal_completed,
    file_digest,
    json_digest,
    seeded_workflow,
    summary,
    validate_workflow_files,
    validate_workflow_report,
    validate_workload,
)


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity(root: Path, expected: str | None = None) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, timeout=30
        ).strip()

    commit = git("rev-parse", "HEAD")
    if expected is not None and commit != expected:
        raise ValueError("checkout differs from the requested exact revision")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError(f"source checkout is dirty: {root}")
    return {
        "root": str(root),
        "commit": commit,
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def authenticate_artifacts(manifest: Path) -> list[dict[str, Any]]:
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("artifact manifest must be a nonempty list")
    result, seen = [], set()
    for entry in entries:
        path = Path(entry["path"])
        name, category = entry["name"], entry["category"]
        for value in (name, category):
            if (
                not isinstance(value, str)
                or not value
                or "\\" in value
                or ":" in value
                or PurePosixPath(value).is_absolute()
                or ".." in PurePosixPath(value).parts
            ):
                raise ValueError("artifact names/categories must be safe relative paths")
        if "/" in category or name == "." or category == ".":
            raise ValueError("invalid artifact category/name")
        key = (category, name)
        if key in seen:
            raise ValueError("duplicate artifact category/name")
        seen.add(key)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("artifact path must name an absolute local file")
        digest = file_digest(path)
        if digest != entry["sha256"]:
            raise ValueError(f"artifact digest differs: {category}/{name}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError("empty input artifact")
        result.append(
            {
                "path": str(path.resolve()),
                "category": category,
                "name": name,
                "sha256": digest,
                "bytes": size,
            }
        )
    return result


def family_observations(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from dinkster_inference import load_safetensors_header
    from dinkster_inference.runtime import probe_native

    observations = []
    for artifact in artifacts:
        path = Path(artifact["path"])
        if path.suffix != ".safetensors":
            continue
        try:
            capability = probe_native(load_safetensors_header(path))
            observations.append(
                {
                    "name": artifact["name"],
                    "family": capability.family_id,
                    "native": capability.native,
                    "diagnostics": list(capability.reasons),
                }
            )
        except (OSError, ValueError) as error:
            observations.append(
                {
                    "name": artifact["name"],
                    "family": None,
                    "diagnostics": [f"{type(error).__name__}: {error}"],
                }
            )
    return observations


def server_command(args: argparse.Namespace, artifacts: list[dict[str, Any]]) -> list[str]:
    output = args.output
    for name in (
        "images",
        "user",
        "temp",
        "library",
        "base/input",
        "base/models",
        "base/custom_nodes",
        "base/datasets",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)
    model_directories = (
        "audio_encoders",
        "background_removal",
        "checkpoints",
        "classifiers",
        "clip",
        "clip_vision",
        "configs",
        "controlnet",
        "detection",
        "diffusers",
        "diffusion_models",
        "embeddings",
        "frame_interpolation",
        "geometry_estimation",
        "gligen",
        "hypernetworks",
        "latent_upscale_models",
        "loras",
        "model_patches",
        "optical_flow",
        "photomaker",
        "style_models",
        "t2i_adapter",
        "text_encoders",
        "unet",
        "upscale_models",
        "vae",
        "vae_approx",
    )
    for root in (output / "base/models", output / "images"):
        for name in model_directories:
            (root / name).mkdir()
    model_paths: dict[str, str | bool] = {"is_default": True}
    if args.system == "comfyui":
        model_paths["custom_nodes"] = str(
            Path(__file__).resolve().parents[1] / "scripts/comfyui_benchmark_nodes"
        )
    for artifact in artifacts:
        category = artifact["category"]
        directory = output / "base" / ("input" if category == "input" else "models/" + category)
        target = directory / artifact["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(artifact["path"], target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            if shutil.disk_usage(target.parent).free < artifact["bytes"] + 1024**3:
                raise ValueError(
                    "insufficient space to stage an input on another filesystem"
                ) from error
            shutil.copyfile(artifact["path"], target)
        if file_digest(target) != artifact["sha256"]:
            raise ValueError("staged artifact changed during authentication")
        if category != "input":
            model_paths[category] = str(directory)
    save(output / "models.yaml", {"benchmark": model_paths})
    common = [
        "--base-directory",
        str(output / "base"),
        "--extra-model-paths-config",
        str(output / "models.yaml"),
        "--input-directory",
        str(output / "base/input"),
        "--output-directory",
        str(output / "images"),
        "--user-directory",
        str(output / "user"),
        "--temp-directory",
        str(output / "temp"),
        "--database-url",
        "sqlite:///" + str(output / "user/comfyui.db"),
        "--disable-all-custom-nodes",
        "--disable-api-nodes",
    ]
    if args.cpu and args.system == "comfyui":
        common.append("--cpu")
    if args.system == "comfyui":
        common += ["--whitelist-custom-nodes", "dinkster_benchmark_shim"]
        return [
            str(args.server_python),
            str(args.reference_root / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(args.port),
            *common,
        ]
    (output / "library/mounts.toml").write_text(
        "[mounts.comfy-output]\npath = "
        + json.dumps(str(output / "images"))
        + '\nmode = "readwrite"\n[mounts.comfy-input]\npath = '
        + json.dumps(str(output / "base/input"))
        + "\n"
    )
    return [
        str(args.server_python),
        "-m",
        "dinkster.serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--library-root",
        str(output / "library"),
        "--comfy-root",
        str(args.reference_root),
        "--execution-python",
        str(args.server_python),
        "--aimdo",
        "auto",
        *["--comfy-arg=" + value for value in common],
    ]


def request(base: str, path: str, body: Any = None, *, timeout: float = 10) -> Any:
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    query = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    # Never retry POST: a lost response leaves acceptance unknown.
    with urllib.request.urlopen(query, timeout=timeout) as response:
        return json.load(response)


def queue_resumed(response: Any) -> bool:
    return isinstance(response, dict) and response.get("paused") is False


def queue_paused(response: Any) -> bool:
    return isinstance(response, dict) and response.get("paused") is True


def _resume_queue(base: str) -> None:
    resumed = request(base, "/api/queue/resume", {})
    if not queue_resumed(resumed):
        raise RuntimeError("Dinkster queue resume acknowledgement is invalid")


@contextlib.contextmanager
def paused_queue(base: str) -> Iterator[None]:
    try:
        paused = request(base, "/api/queue/pause", {})
        if not queue_paused(paused):
            raise RuntimeError("Dinkster queue pause acknowledgement is invalid")
    except BaseException as primary:
        try:
            _resume_queue(base)
        except BaseException as resume_error:
            primary.add_note(
                f"Dinkster queue resume also failed: {type(resume_error).__name__}: {resume_error}"
            )
        raise
    try:
        yield
    except BaseException as primary:
        try:
            _resume_queue(base)
        except BaseException as resume_error:
            primary.add_note(
                f"Dinkster queue resume also failed: {type(resume_error).__name__}: {resume_error}"
            )
        raise
    else:
        _resume_queue(base)


def runtime_identity(
    python: Path, root: Path, environment: dict[str, str], system: str
) -> dict[str, Any]:
    code = """
import importlib.metadata, importlib.util, json, platform, sys
spec = importlib.util.find_spec('dinkster')
print(json.dumps({
    'python': sys.executable, 'python_version': platform.python_version(),
    'dinkster_origin': spec.origin if spec else None,
    'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
}))
"""
    value = json.loads(
        subprocess.check_output(
            [str(python), "-c", code],
            cwd=root,
            env={**environment, "CUDA_VISIBLE_DEVICES": "-1"},
            text=True,
            timeout=30,
        )
    )
    if system == "dinkster" and (
        not value["dinkster_origin"]
        or not Path(value["dinkster_origin"]).resolve().is_relative_to(root)
    ):
        raise ValueError("server interpreter does not import Dinkster from the recorded checkout")
    return value


def collect_journal(base: str, identity: str, scope: str, destination: Path) -> None:
    if not scope or scope != scope.strip() or any(character.isspace() for character in scope):
        raise ValueError("terminal history has no valid journal scope")
    after, records = 0, []
    deadline = time.monotonic() + 10
    route = (
        "/api/runs/"
        + urllib.parse.quote(identity, safe="")
        + "/journal?"
        + urllib.parse.urlencode({"scope": scope, "limit": 1000})
    )
    while time.monotonic() < deadline:
        try:
            page = request(base, route + f"&after={after}")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            time.sleep(0.1)
            continue
        records.extend(page["records"])
        save(
            destination,
            {
                "records": records,
                "latestSeq": page["latestSeq"],
                "coalescedBelow": page["coalescedBelow"],
            },
        )
        if dinkster_journal_completed(records, identity) and (
            not page["records"] or page["records"][-1]["seq"] >= page["latestSeq"]
        ):
            return
        if page["records"]:
            cursor = page["records"][-1]["seq"]
            if cursor <= after:
                raise ValueError("journal cursor did not advance")
            after = cursor
        time.sleep(0.1)
    raise TimeoutError("terminal journal evidence did not arrive")


def wait_ready(args: argparse.Namespace, server: OwnedServer, base: str) -> Any:
    deadline = time.monotonic() + args.startup_timeout
    route = "/object_info" if args.system == "comfyui" else "/api/composition"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError("server exited before readiness; inspect server.log")
        try:
            response = request(base, route, timeout=min(10, max(0.01, deadline - time.monotonic())))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(args.poll_interval)
            continue
        server.assert_listener(args.port)
        if args.system == "comfyui" or not response.get("composing"):
            if args.system == "dinkster":
                mounts = request(base, "/api/mounts")
                save(args.output / "mounts.json", mounts)
                if any(row["state"] in ("pending", "scanning") for row in mounts["mounts"]):
                    time.sleep(args.poll_interval)
                    continue
                if any(row["state"] == "failed" for row in mounts["mounts"]):
                    raise ValueError("model/input mount scan failed")
            return response
        time.sleep(args.poll_interval)
    raise TimeoutError("server readiness deadline exceeded")


def output_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("output symlinks are not benchmark evidence")
        if path.is_file():
            stat = path.stat()
            files[str(path.relative_to(root))] = (stat.st_mtime_ns, stat.st_size)
    return files


def retain_outputs(
    output: Path, index: int, before: dict[str, tuple[int, int]]
) -> list[dict[str, Any]]:
    rows = []
    for name, stat in output_snapshot(output / "images").items():
        if before.get(name) == stat:
            continue
        source = output / "images" / name
        retained = output / "outputs" / str(index) / name
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, retained)
        rows.append(
            {
                "file": str(retained.relative_to(output)),
                "source_name": name,
                "sha256": file_digest(retained),
                "bytes": retained.stat().st_size,
            }
        )
    if not rows:
        raise ValueError("job completed without new output files")
    return rows


def run_jobs(
    args: argparse.Namespace,
    report: dict[str, Any],
    server: OwnedServer,
    base: str,
    sampler: MemorySampler,
    observer: ObserverInstallation | None,
) -> None:
    workload = report["workload"]
    for index, seed in enumerate(workload["seeds"]):
        sampler.check()
        graph = seeded_workflow(workload["graph"], workload["seed_inputs"], seed)
        body = {"prompt": graph, "client_id": "workflow-benchmark"}
        prefix = str(index)
        row: dict[str, Any] = {
            "seed": seed,
            "phase": "cold" if index == 0 else "warm",
            "completed": False,
            "submitted_sha256": json_digest(graph),
            "submitted_file": prefix + "-submitted.json",
            "accepted_file": prefix + "-accepted.json",
            "history_file": prefix + "-history.json",
        }
        report["runs"].append(row)
        save(args.output / row["submitted_file"], body)
        endpoint = "/prompt" if args.system == "comfyui" else "/api/compat/comfy/prompt"
        if args.system == "dinkster":
            save(
                args.output / (prefix + "-translation.json"),
                request(base, endpoint + "?dryRun=1", body),
            )
        before = output_snapshot(args.output / "images")
        server.assert_listener(args.port)
        row["start_epoch_ns"] = time.time_ns()
        start = time.monotonic()
        row["start_monotonic_seconds"] = start
        nonce = (
            observer.begin(report["device"], require_existing=index > 0)
            if observer
            else secrets.token_hex(24)
        )
        row["allocator_window_nonce"] = nonce
        reset: dict[str, Any] | None = None
        if args.gpu_uuid and args.system == "comfyui":
            reset = request(
                base,
                "/dinkster_benchmark/reset",
                {"nonce": nonce, "device": report["device"]},
            )
            assert reset is not None
            if (
                reset.get("event") != "reset_ack"
                or reset.get("nonce") != nonce
                or reset.get("device") != report["device"]
                or reset.get("physical_device_uuid") != report["device"].get("uuid")
                or not reset.get("process_instance")
            ):
                raise RuntimeError("ComfyUI did not acknowledge the allocator reset window")
        save(args.output / "report.json", report)
        accepted = request(base, endpoint, body, timeout=min(30, args.job_timeout))
        save(args.output / row["accepted_file"], accepted)
        identity = accepted["prompt_id" if args.system == "comfyui" else "jobRef"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("server returned no job identity")
        row["job_id"] = identity
        route = (
            "/history/" if args.system == "comfyui" else "/api/jobs/by-ref/"
        ) + urllib.parse.quote(identity, safe="")
        deadline = start + args.job_timeout
        while time.monotonic() < deadline:
            sampler.check()
            if server.process.poll() is not None:
                raise RuntimeError("server exited while a job was outstanding")
            response = request(base, route, timeout=min(10, max(0.01, deadline - time.monotonic())))
            received = time.monotonic()
            history = response.get(identity) if args.system == "comfyui" else response
            if history:
                save(args.output / row["history_file"], history)
                if args.system == "comfyui":
                    done, success = comfyui_history_status(history, identity, graph)
                else:
                    if history.get("jobRef") != identity:
                        raise ValueError("history belongs to another job")
                    done = history.get("state") in ("completed", "failed", "cancelled")
                    success = history.get("state") == "completed"
                if done:
                    row["client_wall_seconds"] = received - start
                    row["end_monotonic_seconds"] = received
                    row["end_epoch_ns"] = time.time_ns()
                    if not success:
                        raise RuntimeError("job failed; inspect retained history and server.log")
                    row["outputs"] = retain_outputs(args.output, index, before)
                    if args.system == "dinkster":
                        row["journal_file"] = prefix + "-journal.json"
                        collect_journal(
                            base,
                            identity,
                            history.get("scope"),
                            args.output / row["journal_file"],
                        )
                    if args.gpu_uuid:
                        if args.system == "dinkster":
                            assert observer is not None
                            reset_instances = {
                                str(record["process_instance"])
                                for record in observer.records()
                                if record.get("event") == "reset_ack"
                                and record.get("nonce") == nonce
                            }
                            if not reset_instances:
                                raise RuntimeError(
                                    "terminal job has no nonce-bound worker reset acknowledgements"
                                )
                            observer.command(
                                "read",
                                nonce=nonce,
                                device=report["device"],
                                expected_instances=reset_instances,
                                job_ref=identity,
                                attempt=accepted.get("attemptId"),
                            )
                            all_cached = bool(history.get("nodeStates")) and all(
                                state == "cached" for state in history["nodeStates"].values()
                            )
                            row["all_cached"] = all_cached
                            window = validate_window(
                                observer.records(),
                                nonce=nonce,
                                job_ref=identity,
                                attempt=accepted.get("attemptId"),
                                device=report["device"],
                                all_cached=all_cached,
                                source_sha256=observer.source_sha256,
                            )
                        else:
                            attempt = accepted.get("number")
                            if type(attempt) not in (int, float) or history["prompt"][0] != attempt:
                                raise RuntimeError(
                                    "ComfyUI queue attempt evidence is missing or mismatched"
                                )
                            memory = request(
                                base,
                                "/dinkster_benchmark/memory?"
                                + urllib.parse.urlencode({"nonce": nonce, "job_ref": identity}),
                            )
                            window = {
                                "nonce": nonce,
                                "job_ref": identity,
                                "attempt": attempt,
                                "status": (
                                    "acknowledged"
                                    if memory.get("event") == "read_ack"
                                    and memory.get("nonce") == nonce
                                    and memory.get("job_ref") == identity
                                    and memory.get("attempt") == attempt
                                    and reset is not None
                                    and memory.get("process_instance")
                                    == reset.get("process_instance")
                                    else "invalid"
                                ),
                                "raw_records": [reset, memory],
                                "peak_allocated_bytes": memory.get("peak_allocated_bytes"),
                                "peak_reserved_bytes": memory.get("peak_reserved_bytes"),
                                "problems": [],
                            }
                            if (
                                memory.get("device") != report["device"]
                                or memory.get("physical_device_uuid")
                                != report["device"].get("uuid")
                                or not memory.get("logical_device")
                            ):
                                window["status"] = "invalid"
                                window["problems"].append(
                                    "ComfyUI allocator acknowledgement has the wrong device mapping"
                                )
                        row["allocator_file"] = prefix + "-allocator.json"
                        save(args.output / row["allocator_file"], window)
                        row["allocator"] = window
                        if window["status"] == "invalid" or window["problems"]:
                            raise RuntimeError("allocator window evidence is invalid")
                    row["completed"] = True
                    save(args.output / "report.json", report)
                    break
            time.sleep(args.poll_interval)
        else:
            raise TimeoutError("job completion deadline exceeded")


def parse_arguments(system: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "workflow",
        "workflow-provenance",
        "artifacts",
        "repo",
        "server-python",
        "reference-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--reference-commit", required=True, help="exact full reference revision actually run"
    )
    parser.add_argument(
        "--family", help="diagnostic hint only; workflow and detected plans are authoritative"
    )
    parser.add_argument(
        "--seed-input", action="append", required=True, help="literal NODE_ID.INPUT to vary"
    )
    parser.add_argument("--seed", type=int, default=1064)
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=5,
        help="warm runs after the initial cold run; zero records a one-shot smoke",
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--job-timeout", type=float, default=1800)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--cpu", action="store_true")
    device.add_argument("--gpu-uuid")
    parser.add_argument("--offline-check", action="store_true")
    args = parser.parse_args(argv)
    args.system = system
    if len(args.reference_commit) != 40 or any(
        c not in "0123456789abcdef" for c in args.reference_commit
    ):
        parser.error("--reference-commit must be a full hexadecimal revision")
    if not 1 <= args.port <= 65535 or args.warm_runs < 0:
        parser.error("port must be valid and warm runs must be nonnegative")
    for name in ("startup_timeout", "job_timeout", "poll_interval"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    for name in (
        "workflow",
        "workflow_provenance",
        "artifacts",
        "repo",
        "server_python",
        "reference_root",
        "output",
    ):
        # Preserve a venv interpreter symlink; resolving it would select the base environment.
        setattr(args, name, getattr(args, name).absolute())
    return args


def main(system: str, argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(system, argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "report_version": 1 if args.cpu else 2,
        "system": system,
        "all_ok": False,
        "cleanup_verified": False,
        "errors": [],
        "runs": [],
        "family_hint": args.family,
        "machine": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "poll_interval_seconds": args.poll_interval,
        "offline_check": args.offline_check,
    }
    server = sampler = observer = None
    try:
        report["sources"] = {
            "harness": source_identity(Path(__file__).resolve().parents[1]),
            "dinkster": source_identity(args.repo),
            "comfyui": source_identity(args.reference_root, args.reference_commit),
        }
        if (args.reference_root / "extra_model_paths.yaml").exists():
            raise ValueError(
                "reference has implicit model paths; use an isolated reference checkout"
            )
        api_bytes = args.workflow.read_bytes()
        graph = json.loads(api_bytes)
        if (
            not isinstance(graph, dict)
            or not graph
            or any(
                not isinstance(node, dict)
                or not isinstance(node.get("class_type"), str)
                or not isinstance(node.get("inputs"), dict)
                for node in graph.values()
            )
        ):
            raise ValueError("workflow must be a nonempty exported API graph")
        provenance = json.loads(args.workflow_provenance.read_text())
        if provenance.get("api_sha256") != hashlib.sha256(api_bytes).hexdigest():
            raise ValueError("workflow export digest differs from provenance")
        seeds = list(range(args.seed, args.seed + args.warm_runs + 1))
        for seed in seeds:
            seeded_workflow(graph, args.seed_input, seed)
        report["workload"] = {
            "graph": graph,
            "graph_sha256": json_digest(graph),
            "api_sha256": hashlib.sha256(api_bytes).hexdigest(),
            "provenance": provenance,
            "seed_inputs": args.seed_input,
            "seeds": seeds,
        }
        validate_workload(report["workload"])
        report["artifacts"] = authenticate_artifacts(args.artifacts)
        report["family_observations"] = family_observations(report["artifacts"])
        report["diagnostics"] = [
            "Family hints and unknown labels are not used for admission or routing."
        ]
        command = server_command(args, report["artifacts"])
        report["command"] = command
        report["device"] = {"kind": "cpu"} if args.cpu else {"kind": "cuda", "uuid": args.gpu_uuid}
        save(args.output / "report.json", report)
        if args.offline_check:
            report["cleanup_verified"] = True
            report["preflight_ok"] = True
            return 0
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("DINKSTER_") and key not in ("PYTHONPATH", "PYTHONHOME")
        }
        environment["PYTHONNOUSERSITE"] = "1"
        environment["DINKSTER_SERVING_PYTHON"] = str(args.server_python)
        environment["DINKSTER_ACCELERATOR"] = "cpu" if args.cpu else "cuda"
        environment["CUDA_VISIBLE_DEVICES"] = "-1" if args.cpu else args.gpu_uuid
        if args.gpu_uuid and system == "dinkster":
            observer = ObserverInstallation.create(args.server_python, args.output)
            environment.update(observer.environment())
            report["allocator_observer"] = {
                "source_sha256": observer.source_sha256,
                "records_file": observer.records_path.name,
            }
        cwd = args.repo if system == "dinkster" else args.reference_root
        report["runtime"] = runtime_identity(args.server_python, cwd, environment, system)
        if args.gpu_uuid:
            with NvmlDevice(args.gpu_uuid) as device:
                report["device"] = device.identity()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", args.port))
        start = time.monotonic()
        server = OwnedServer(command, cwd, environment, args.output / "server.log")
        sampler = MemorySampler(server, args.output / "memory.jsonl", args.gpu_uuid)
        sampler.start()
        base = f"http://127.0.0.1:{args.port}"
        save(args.output / "startup.json", wait_ready(args, server, base))
        report["startup_seconds"] = time.monotonic() - start
        if observer is not None:
            observer.wait_for_registration("local")
        run_jobs(args, report, server, base, sampler, observer)
        if args.gpu_uuid:
            windows = [row["allocator"] for row in report["runs"]]
            report["allocator_memory"] = {
                "allocator_peak_scope": ALLOCATOR_PEAK_SCOPE,
                "allocator_peak_bytes": max(
                    (
                        row["peak_allocated_bytes"]
                        for row in windows
                        if row["peak_allocated_bytes"] is not None
                    ),
                    default=None,
                ),
                "allocator_peak_reserved_bytes": max(
                    (
                        row["peak_reserved_bytes"]
                        for row in windows
                        if row["peak_reserved_bytes"] is not None
                    ),
                    default=None,
                ),
            }
            last = report["runs"][-1]
            if system == "comfyui":
                unloaded = request(
                    base,
                    "/dinkster_benchmark/unload",
                    {
                        "nonce": last["allocator_window_nonce"],
                        "job_ref": last["job_id"],
                    },
                    timeout=40,
                )
                if unloaded.get("event") != "unload_ack":
                    raise RuntimeError("ComfyUI did not acknowledge its full free path")
                read_record = last["allocator"]["raw_records"][-1]
                if (
                    unloaded.get("nonce") != last["allocator_window_nonce"]
                    or unloaded.get("job_ref") != last["job_id"]
                    or unloaded.get("process_instance") != read_record.get("process_instance")
                    or unloaded.get("device") != report["device"]
                ):
                    raise RuntimeError("ComfyUI unload acknowledgement changed window identity")
                report["allocator_memory"].update(
                    {
                        "unload_residual_bytes": unloaded["residual_allocated_bytes"],
                        "unload_residual_reserved_bytes": unloaded["residual_reserved_bytes"],
                        "unload_record": unloaded,
                    }
                )
            else:
                assert observer is not None
                request_id = secrets.token_hex(24)
                with paused_queue(base):
                    full_free = request(base, "/memory/free", {"requestId": request_id}, timeout=40)
                    report["allocator_full_free_file"] = "allocator-full-free.json"
                    save(
                        args.output / report["allocator_full_free_file"],
                        {"request": {"requestId": request_id}, "response": full_free},
                    )
                    workers = full_free.get("workers")
                    expected_instances = {
                        str(worker["workerInstance"])
                        for worker in workers
                        if isinstance(worker, dict)
                        and isinstance(worker.get("workerInstance"), str)
                    }
                    observer.command(
                        "residue",
                        nonce=last["allocator_window_nonce"],
                        device=report["device"],
                        expected_instances=expected_instances,
                    )
                    free_evidence = validate_free_response(
                        full_free,
                        observer.records(),
                        request_id=request_id,
                        nonce=last["allocator_window_nonce"],
                        device=report["device"],
                        windows=windows,
                        source_sha256=observer.source_sha256,
                    )
                    if free_evidence["problems"]:
                        raise RuntimeError("Dinkster full-free allocator evidence is invalid")
                    report["allocator_memory"].update(
                        {
                            "unload_residual_bytes": free_evidence["residual_allocated_bytes"],
                            "unload_residual_reserved_bytes": free_evidence[
                                "residual_reserved_bytes"
                            ],
                            "unload_record": free_evidence,
                        }
                    )
        warm_samples = [row["client_wall_seconds"] for row in report["runs"][1:]]
        report["warm_summary"] = summary(warm_samples) if warm_samples else None
        for name, root in (
            ("harness", Path(__file__).resolve().parents[1]),
            ("dinkster", args.repo),
            ("comfyui", args.reference_root),
        ):
            source_identity(root, report["sources"][name]["commit"])
        if authenticate_artifacts(args.artifacts) != report["artifacts"]:
            raise ValueError("artifacts changed during execution")
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        try:
            if sampler is not None:
                report["memory"] = sampler.stop()
                report["memory"].update(report.get("allocator_memory", {}))
                report["errors"].extend(report["memory"]["errors"])
                if not report["memory"]["cleanup_verified"]:
                    report["errors"].append("memory sampler still running; retain GPU claim")
        finally:
            try:
                if server is not None:
                    report["cleanup_verified"] = server.stop()
                    report["server_exit_code"] = server.process.poll()
            except Exception as error:
                report["errors"].append(f"server cleanup: {type(error).__name__}: {error}")
            finally:
                if observer is not None:
                    observer.close()
                save(args.output / "report.json", report)
    report["all_ok"] = not report["errors"] and report["cleanup_verified"]
    report["validation_errors"] = list(validate_workflow_report(report))
    if not report["validation_errors"]:
        report["validation_errors"] = list(validate_workflow_files(report, args.output))
    report["all_ok"] = report["all_ok"] and not report["validation_errors"]
    save(args.output / "report.json", report)
    print(f"workflow report: {args.output / 'report.json'}")
    return 0 if report["all_ok"] else 1
