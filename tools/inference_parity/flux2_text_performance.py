"""Pinned Flux2 Mistral text-encoding comparison against core ComfyUI."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

COMFYUI_COMMIT = "725e6ec60621c6f001af04769173e7dbb3c53541"
TEXT_ENCODER_REVISION = "0e1aa000db476492858269a99a33c4aa9b81c940"
TEXT_ENCODER_URL = (
    "https://huggingface.co/Comfy-Org/flux2-dev/resolve/"
    f"{TEXT_ENCODER_REVISION}/split_files/text_encoders/"
    "mistral_3_small_flux2_bf16.safetensors"
)
TEXT_ENCODER_SHA256 = "7d79902f60b1aeb3a6de2cfad02f4367b5e300a1387de3d03ac717cfa3df117c"
TEXT_ENCODER_BLAKE3 = "bccf96b33766858d043f5ce3b187a63ee6b24851048e1d8bc20e1ac91c8838fc"
TEXT_ENCODER_SIZE = 35_584_897_447
DEVICE_UUID = "GPU-666d1242-9c20-341c-73ea-e63770947451"
PROMPT = (
    "A cinematic photograph of a red fox standing in fresh snow beneath northern lights, "
    "with detailed fur, natural moonlight, and a distant pine forest."
)
MEASURED_RUNS = 5
COMFYUI_KITCHEN_VERSION = "0.2.30"
DINKSTER_KITCHEN_VERSION = "0.2.35.post1"
DINKSTER_AIMDO_VERSION = "0.5.5.post2"


class ComparisonError(RuntimeError):
    """The comparison contract or evidence is invalid."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_source(root: Path, expected_commit: str, name: str) -> None:
    actual = _git(root, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise ComparisonError(f"{name} must be at {expected_commit}, got {actual}")
    if _git(root, "status", "--porcelain"):
        raise ComparisonError(f"{name} source must be clean: {root}")


def _artifact_receipt(path: Path) -> dict[str, object]:
    size = path.stat().st_size
    digest = _file_sha256(path)
    if size != TEXT_ENCODER_SIZE or digest != TEXT_ENCODER_SHA256:
        raise ComparisonError(
            f"text encoder must be {TEXT_ENCODER_SIZE} bytes and "
            f"sha256:{TEXT_ENCODER_SHA256}; got {size} bytes and sha256:{digest}"
        )
    return {
        "blake3": TEXT_ENCODER_BLAKE3,
        "path": str(path),
        "revision": TEXT_ENCODER_REVISION,
        "sha256": digest,
        "size_bytes": size,
        "url": TEXT_ENCODER_URL,
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _device_receipt(torch: Any, device: Any) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != DEVICE_UUID:
        raise ComparisonError(f"CUDA_VISIBLE_DEVICES must be {DEVICE_UUID}")
    if torch.cuda.device_count() != 1:
        raise ComparisonError(
            f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )
    uuid = subprocess.run(
        ["nvidia-smi", f"--id={DEVICE_UUID}", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if uuid != DEVICE_UUID:
        raise ComparisonError(f"expected CUDA device {DEVICE_UUID}, got {uuid}")
    driver = subprocess.run(
        [
            "nvidia-smi",
            f"--id={DEVICE_UUID}",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    properties = torch.cuda.get_device_properties(device)
    return {
        "cuda": torch.version.cuda,
        "device_uuid": uuid,
        "driver": driver,
        "gpu": properties.name,
        "gpu_total_memory": properties.total_memory,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def _write_record(path: Path, record: dict[str, object], output: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_path = path.with_suffix(".npy")
    np.save(output_path, output, allow_pickle=False)
    record["output"] = {
        "dtype": str(output.dtype),
        "path": str(output_path.resolve()),
        "sha256": _array_sha256(output),
        "shape": list(output.shape),
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _measure_runs(
    torch: Any,
    device: Any,
    encode: Callable[[], tuple[Any, dict[str, Any]]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    latest: np.ndarray | None = None
    for phase, count in (("cold", 1), ("warmup", 1), ("measured", MEASURED_RUNS)):
        for index in range(count):
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            with torch.inference_mode():
                value, details = encode()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter_ns() - started) / 1e9
            array: np.ndarray = value.detach().float().cpu().numpy()
            latest = array
            runs.append(
                {
                    **details,
                    "index": index,
                    "output_sha256": _array_sha256(array),
                    "phase": phase,
                    "seconds": elapsed,
                }
            )
    if latest is None:
        raise ComparisonError("benchmark executed no runs")
    return latest, runs


def _summary(runs: list[dict[str, Any]]) -> dict[str, object]:
    measured = [float(run["seconds"]) for run in runs if run["phase"] == "measured"]
    median = statistics.median(measured)
    return {
        "cold_seconds": next(float(run["seconds"]) for run in runs if run["phase"] == "cold"),
        "measured_mad_seconds": statistics.median(abs(value - median) for value in measured),
        "measured_median_seconds": median,
        "measured_range_seconds": [min(measured), max(measured)],
        "runs": runs,
    }


def _workload(token_ids: list[int]) -> dict[str, object]:
    return {
        "measured_runs": MEASURED_RUNS,
        "model_role": "mistral3_24b",
        "precision_policy": "production-default",
        "prompt": PROMPT,
        "token_ids": token_ids,
        "warmup_runs_discarded": 1,
    }


def run_comfyui(args: argparse.Namespace) -> int:
    root = args.comfyui_root.resolve()
    text_encoder = args.text_encoder.resolve()
    _require_source(root, COMFYUI_COMMIT, "ComfyUI")
    artifact = _artifact_receipt(text_encoder)
    sys.path.insert(0, str(root))

    import comfy_aimdo.control as control  # pyright: ignore[reportMissingImports]

    control.init()

    import comfy.memory_management  # pyright: ignore[reportMissingImports]
    import comfy.model_management  # pyright: ignore[reportMissingImports]
    import comfy.model_patcher  # pyright: ignore[reportMissingImports]
    import comfy.sd  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not control.init_devices(((0, 0),)):
        raise ComparisonError("ComfyUI comfy-aimdo device initialization failed")
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    started = time.perf_counter_ns()
    clip = comfy.sd.load_clip([str(text_encoder)], clip_type=comfy.sd.CLIPType.FLUX2)
    load_seconds = (time.perf_counter_ns() - started) / 1e9
    tokens = clip.tokenize(PROMPT)
    token_rows = tokens.get("mistral3_24b")
    if not isinstance(token_rows, list) or len(token_rows) != 1:
        raise ComparisonError("ComfyUI did not produce one Mistral token row")
    token_ids = [int(item[0]) for item in token_rows[0]]
    activation_dtypes: set[str] = set()

    def record_activation(_module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        activation_dtypes.add(str(kwargs["x"].dtype).removeprefix("torch."))

    model = clip.cond_stage_model.mistral3_24b.transformer.model
    activation_hook = model.layers[0].register_forward_pre_hook(record_activation, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats(device)

    def encode() -> tuple[Any, dict[str, Any]]:
        output = clip.encode_from_tokens(tokens)
        return output, {"loaded_bytes": int(clip.patcher.loaded_size())}

    output, runs = _measure_runs(torch, device, encode)
    activation_hook.remove()
    if activation_dtypes != {"float32"}:
        raise ComparisonError(f"ComfyUI production activation dtype changed: {activation_dtypes}")
    record: dict[str, object] = {
        "engine": "comfyui",
        "load_seconds": load_seconds,
        "memory": {
            "gpu_free_after_bytes": int(torch.cuda.mem_get_info(device)[0]),
            "logical_loaded_bytes": int(clip.patcher.loaded_size()),
            "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "receipts": {
            **_device_receipt(torch, device),
            "artifact": artifact,
            "comfy_aimdo": _package_version("comfy-aimdo"),
            "comfy_kitchen": _package_version("comfy-kitchen"),
            "commit": COMFYUI_COMMIT,
            "patcher": type(clip.patcher).__name__,
            "precision": {
                "activation": "float32",
                "storage": sorted(
                    str(dtype).removeprefix("torch.") for dtype in clip.cond_stage_model.dtypes
                ),
            },
        },
        "summary": _summary(runs),
        "workload": _workload(token_ids),
    }
    _write_record(args.output, record, output)
    gc.collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache(force=True)
    return 0


def _add_dinkster_sources(root: Path) -> None:
    for source in sorted(root.glob("packages/*/src")):
        sys.path.insert(0, str(source))
    sys.path.insert(0, str(root / "src"))


def run_dinkster(args: argparse.Namespace) -> int:
    root = args.dinkster_root.resolve()
    text_encoder = args.text_encoder.resolve()
    _require_source(root, args.dinkster_commit, "Dinkster")
    artifact = _artifact_receipt(text_encoder)
    _add_dinkster_sources(root)

    import dinkster_aimdo.control as control  # pyright: ignore[reportMissingImports]

    control.init()

    import dinkster_inference_torch.pinned_host as pinned_host
    import torch  # pyright: ignore[reportMissingImports]
    from dinkster_assets import AssetRef
    from dinkster_inference import (
        BFLOAT16,
        flux2_component_runtime_identity,
        load_flux2_tekken_bpe,
        load_safetensors_header,
        plan_flux2_split_component,
        tokenize_flux2_dev_prompt,
    )
    from dinkster_inference_torch import (
        AimdoWeights,
        Flux2TextRuntime,
        ResidencyManager,
        collect_partial_residency_timing,
        dynamic_free_memory,
        enroll_component,
        ensure_aimdo_devices,
        load_flux2_component,
    )

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return text_encoder if digest == f"blake3:{TEXT_ENCODER_BLAKE3}" else None

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if not ensure_aimdo_devices(((0, 0),)):
        raise ComparisonError("Dinkster dinkster-aimdo device initialization failed")
    asset = AssetRef(
        f"blake3:{TEXT_ENCODER_BLAKE3}",
        text_encoder.name,
        text_encoder.stat().st_size,
        resolver=Resolver(),
    )
    started = time.perf_counter_ns()
    source = load_safetensors_header(
        text_encoder,
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    planned = plan_flux2_split_component(source, role="mistral3_24b", path=text_encoder)
    identity = flux2_component_runtime_identity(planned, BFLOAT16)
    loaded = load_flux2_component(
        text_encoder,
        asset=asset,
        expected_role="mistral3_24b",
        expected_identity=identity,
        compute_dtype=torch.bfloat16,
    )
    mechanism = enroll_component(
        loaded.module,
        load_device=device,
        offload_device=torch.device("cpu"),
        mechanism_factory=AimdoWeights,
    )
    runtime = Flux2TextRuntime(loaded.module)
    manager = ResidencyManager(free_memory=dynamic_free_memory)
    load_seconds = (time.perf_counter_ns() - started) / 1e9
    token_ids = list(tokenize_flux2_dev_prompt(PROMPT, tokenizer=load_flux2_tekken_bpe()).ids)
    activation_dtypes: set[str] = set()

    def record_activation(_module: Any, args: tuple[Any, ...]) -> None:
        activation_dtypes.add(str(args[0].dtype).removeprefix("torch."))

    activation_hook = loaded.module.layers[0].register_forward_pre_hook(record_activation)
    torch.cuda.reset_peak_memory_stats(device)

    def encode() -> tuple[Any, dict[str, Any]]:
        manager_started = time.perf_counter_ns()
        manager.load((mechanism,))
        manager_load_ms = (time.perf_counter_ns() - manager_started) / 1e6
        with collect_partial_residency_timing() as timing:
            result = runtime.encode_text(PROMPT)
        report = timing.report()
        return result.embeddings, {
            "fixed_loaded_bytes": cast(Any, mechanism)._eager_loaded_bytes(),
            "logical_loaded_bytes": mechanism.loaded_bytes(),
            "logical_offloaded_bytes": mechanism.offloaded_bytes(),
            "manager_load_ms": manager_load_ms,
            "transfer_bytes": report.transfer_bytes,
            "transfer_ms": report.transfer_ms,
        }

    output, runs = _measure_runs(torch, device, encode)
    activation_hook.remove()
    if activation_dtypes != {"bfloat16"}:
        raise ComparisonError(f"Dinkster production activation dtype changed: {activation_dtypes}")
    warm_fixed_placements = {
        int(run["fixed_loaded_bytes"]) for run in runs if run["phase"] != "cold"
    }
    if len(warm_fixed_placements) != 1:
        raise ComparisonError("Dinkster fixed residency changed after the cold encode")
    optimized_fixed = warm_fixed_placements.pop()
    optimized_loaded = mechanism.loaded_bytes()
    optimized_memory = {
        "fixed_loaded_bytes": optimized_fixed,
        "gpu_free_after_bytes": int(torch.cuda.mem_get_info(device)[0]),
        "logical_loaded_bytes": optimized_loaded,
        "logical_offloaded_bytes": mechanism.total_bytes() - optimized_loaded,
        "pinned_host_registered_bytes": pinned_host.TOTAL_PINNED_MEMORY,
        "pinned_host_storage_bytes": pinned_host.TOTAL_PINNED_STORAGE,
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    while True:
        loaded_before = mechanism.loaded_bytes()
        mechanism.partially_unload(mechanism.total_bytes())
        if mechanism.loaded_bytes() >= loaded_before:
            break
    control_loaded = mechanism.loaded_bytes()
    if control_loaded >= optimized_fixed:
        raise ComparisonError("Dinkster control did not demote promoted weights")
    with torch.inference_mode():
        control_output = runtime.encode_text(PROMPT).embeddings.detach().float().cpu().numpy()
    control_equal = np.array_equal(output, control_output)
    if not control_equal:
        raise ComparisonError("Dinkster hybrid and demand-paged outputs differ")
    record: dict[str, object] = {
        "engine": "dinkster",
        "load_seconds": load_seconds,
        "memory": optimized_memory,
        "receipts": {
            **_device_receipt(torch, device),
            "artifact": artifact,
            "dinkster_aimdo": _package_version("dinkster-aimdo"),
            "dinkster_kitchen": _package_version("dinkster-kitchen"),
            "commit": args.dinkster_commit,
            "control_fixed_loaded_bytes": control_loaded,
            "demand_control_sha256": _array_sha256(control_output),
            "hybrid_equals_demand_control": control_equal,
            "precision": {
                "activation": "bfloat16",
                "storage": sorted(
                    {
                        str(parameter.dtype).removeprefix("torch.")
                        for parameter in loaded.module.parameters()
                    }
                ),
            },
            "runtime_identity": identity,
        },
        "summary": _summary(runs),
        "workload": _workload(token_ids),
    }
    _write_record(args.output, record, output)
    mechanism.unload()
    return 0


def _load_output(record: dict[str, Any]) -> np.ndarray:
    receipt = record["output"]
    output = np.load(receipt["path"], allow_pickle=False)
    if _array_sha256(output) != receipt["sha256"]:
        raise ComparisonError(f"output digest differs: {receipt['path']}")
    if list(output.shape) != receipt["shape"] or str(output.dtype) != receipt["dtype"]:
        raise ComparisonError(f"output metadata differs: {receipt['path']}")
    return output


def _engine_metric(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(run["seconds"])
        for record in records
        for run in record["summary"]["runs"]
        if run["phase"] == "measured"
    ]
    median = statistics.median(values)
    return {
        "mad_seconds": statistics.median(abs(value - median) for value in values),
        "median_seconds": median,
        "range_seconds": [min(values), max(values)],
        "values_seconds": values,
    }


def build_verdict(records: list[dict[str, Any]], *, dinkster_commit: str) -> dict[str, Any]:
    if [record["engine"] for record in records] != ["comfyui", "dinkster", "dinkster", "comfyui"]:
        raise ComparisonError("record order must be comfyui, dinkster, dinkster, comfyui")
    reference = records[0]
    for record in records:
        measured = [run for run in record["summary"]["runs"] if run["phase"] == "measured"]
        if len(measured) != MEASURED_RUNS:
            raise ComparisonError(f"each process must record {MEASURED_RUNS} measured runs")
        if record["workload"] != reference["workload"]:
            raise ComparisonError("workload or token IDs differ")
        artifact = record["receipts"]["artifact"]
        if (
            artifact["sha256"] != TEXT_ENCODER_SHA256
            or artifact["size_bytes"] != TEXT_ENCODER_SIZE
            or artifact["revision"] != TEXT_ENCODER_REVISION
        ):
            raise ComparisonError("text encoder receipt differs")
        for key in (
            "cuda",
            "device_uuid",
            "driver",
            "gpu",
            "gpu_total_memory",
            "python",
            "torch",
        ):
            if record["receipts"][key] != reference["receipts"][key]:
                raise ComparisonError(f"runtime receipt differs for {key}")
        hashes = {run["output_sha256"] for run in record["summary"]["runs"]}
        if hashes != {record["output"]["sha256"]}:
            raise ComparisonError(f"{record['engine']} output changed between runs")
    comfyui = [records[0], records[3]]
    dinkster = [records[1], records[2]]
    if (
        any(
            record["receipts"]["commit"] != COMFYUI_COMMIT
            or record["receipts"]["comfy_kitchen"] != COMFYUI_KITCHEN_VERSION
            or record["receipts"]["patcher"] != "ModelPatcherDynamic"
            or record["receipts"]["precision"] != {"activation": "float32", "storage": ["bfloat16"]}
            for record in comfyui
        )
        or len({record["receipts"]["comfy_aimdo"] for record in comfyui}) != 1
    ):
        raise ComparisonError("ComfyUI source or dynamic residency receipt differs")
    if any(
        record["receipts"]["commit"] != dinkster_commit
        or record["receipts"]["dinkster_kitchen"] != DINKSTER_KITCHEN_VERSION
        or record["receipts"]["dinkster_aimdo"] != DINKSTER_AIMDO_VERSION
        or record["receipts"]["hybrid_equals_demand_control"] is not True
        or record["receipts"]["demand_control_sha256"] != record["output"]["sha256"]
        or record["receipts"]["precision"] != {"activation": "bfloat16", "storage": ["bfloat16"]}
        for record in dinkster
    ):
        raise ComparisonError("Dinkster source or demand-paged control receipt differs")
    for engine_records in (comfyui, dinkster):
        if len({record["output"]["sha256"] for record in engine_records}) != 1:
            raise ComparisonError(f"{engine_records[0]['engine']} output differs across processes")

    comfy_output = _load_output(comfyui[0])
    dinkster_output = _load_output(dinkster[0])
    if comfy_output.shape != dinkster_output.shape:
        raise ComparisonError(
            f"output shapes differ: ComfyUI {comfy_output.shape}, Dinkster {dinkster_output.shape}"
        )
    difference = np.abs(dinkster_output.astype(np.float64) - comfy_output.astype(np.float64))
    comfy_metric = _engine_metric(comfyui)
    dinkster_metric = _engine_metric(dinkster)
    passed = float(dinkster_metric["median_seconds"]) < float(comfy_metric["median_seconds"])
    return {
        "cross_engine_output_difference": {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "p99_abs": float(np.quantile(difference, 0.99)),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        },
        "pass": passed,
        "performance": {
            "comfyui": comfy_metric,
            "dinkster": dinkster_metric,
            "dinkster_over_comfyui_ratio": (
                float(dinkster_metric["median_seconds"]) / float(comfy_metric["median_seconds"])
            ),
            "pass": passed,
            "requirement": "Dinkster measured median must be strictly lower than ComfyUI",
        },
        "policy": {
            "engine_order": ["comfyui", "dinkster", "dinkster", "comfyui"],
            "measured_runs_per_process": MEASURED_RUNS,
            "warmup_runs_discarded_per_process": 1,
        },
        "receipts": [record["receipts"] for record in records],
        "workload": reference["workload"],
    }


def run_compare(args: argparse.Namespace) -> int:
    records = [json.loads(path.read_text()) for path in args.inputs]
    verdict = build_verdict(records, dinkster_commit=args.dinkster_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return 0 if verdict["pass"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comfyui = commands.add_parser("comfyui")
    comfyui.add_argument("--comfyui-root", type=Path, required=True)
    comfyui.add_argument("--text-encoder", type=Path, required=True)
    comfyui.add_argument("--output", type=Path, required=True)
    comfyui.set_defaults(func=run_comfyui)
    dinkster = commands.add_parser("dinkster")
    dinkster.add_argument("--dinkster-commit", required=True)
    dinkster.add_argument("--dinkster-root", type=Path, required=True)
    dinkster.add_argument("--text-encoder", type=Path, required=True)
    dinkster.add_argument("--output", type=Path, required=True)
    dinkster.set_defaults(func=run_dinkster)
    compare = commands.add_parser("compare")
    compare.add_argument("--dinkster-commit", required=True)
    compare.add_argument("--inputs", nargs=4, type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=run_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
