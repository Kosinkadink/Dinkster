"""Run one pinned ControlNet Auxiliary line/edge parity measurement arm."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib
import json
import os
import resource
import statistics
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from gen_line_edge_goldens import (
    ARTIFACTS,
    COMFYUI_COMMIT,
    CONTROLNET_AUX_COMMIT,
    SOURCE_SHA256,
    _check_artifacts,
    _check_checkout,
    _sha256,
)
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / ".e2e" / "comfyui_controlnet_aux"
COMFYUI = REPO / ".e2e" / "ComfyUI"
CHECKPOINTS = REPO / ".e2e" / "ckpts"
VAULT = REPO / ".e2e" / "vault"
MANIFEST = REPO / "packages" / "dinkster-vision-hed" / "dinkster-pack.toml"

REFERENCE_CASES: dict[str, tuple[str, dict[str, object]]] = {
    "lineart-standard": (
        "LineartStandardPreprocessor",
        {"guassian_sigma": 6.0, "intensity_threshold": 8},
    ),
    "canny": ("CannyEdgePreprocessor", {"low_threshold": 100, "high_threshold": 200}),
    "pyracanny": ("PyraCannyPreprocessor", {"low_threshold": 64, "high_threshold": 128}),
    "scribble": ("ScribblePreprocessor", {}),
    "scribble-xdog": ("Scribble_XDoG_Preprocessor", {"threshold": 32}),
    "lineart-realistic": ("LineArtPreprocessor", {"coarse": "disable"}),
    "lineart-realistic-coarse": ("LineArtPreprocessor", {"coarse": "enable"}),
    "lineart-anime": ("AnimeLineArtPreprocessor", {}),
    "lineart-manga": ("Manga2Anime_LineArt_Preprocessor", {}),
    "hed-soft": ("HEDPreprocessor", {"safe": "disable"}),
    "hed-safe": ("HEDPreprocessor", {"safe": "enable"}),
    "hed-scribble": ("FakeScribblePreprocessor", {"safe": "enable"}),
    "hed-scribble-unsafe": ("FakeScribblePreprocessor", {"safe": "disable"}),
    "teed": ("TEEDPreprocessor", {"safe_steps": 2}),
    "teed-unquantized": ("TEEDPreprocessor", {"safe_steps": 0}),
    "mlsd": ("M-LSDPreprocessor", {"score_threshold": 0.1, "dist_threshold": 0.1}),
    "anyline-standard": (
        "AnyLineArtPreprocessor_aux",
        {"merge_with_lineart": "lineart_standard"},
    ),
    "anyline-realistic": (
        "AnyLineArtPreprocessor_aux",
        {"merge_with_lineart": "lineart_realisitic"},
    ),
    "anyline-anime": (
        "AnyLineArtPreprocessor_aux",
        {"merge_with_lineart": "lineart_anime"},
    ),
    "anyline-manga": (
        "AnyLineArtPreprocessor_aux",
        {"merge_with_lineart": "manga_line"},
    ),
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, int(np.ceil(percentile * len(ordered))) - 1)
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _start_rss_sampler(process: psutil.Process) -> tuple[threading.Event, list[int]]:
    stop = threading.Event()
    samples = [process.memory_info().rss]

    def sample() -> None:
        while not stop.wait(0.002):
            samples.append(process.memory_info().rss)

    threading.Thread(target=sample, daemon=True).start()
    return stop, samples


def _setup_reference(
    case: str, image: np.ndarray, resolution: int, device: str
) -> tuple[Callable[[], np.ndarray], Callable[[], int]]:
    _check_checkout(REFERENCE, CONTROLNET_AUX_COMMIT, "comfyui_controlnet_aux")
    _check_checkout(COMFYUI, COMFYUI_COMMIT, "ComfyUI")
    _check_artifacts(CHECKPOINTS)
    os.environ["AUX_ANNOTATOR_CKPTS_PATH"] = str(CHECKPOINTS)
    os.environ["AUX_USE_SYMLINKS"] = "False"
    sys.path.insert(0, str(COMFYUI))
    sys.path.insert(0, str(REFERENCE.parent))
    sys.path.insert(0, str(REFERENCE / "src"))
    sys.argv = [sys.argv[0]]
    if device == "cpu":
        sys.argv.append("--cpu")
    comfy_options = importlib.import_module("comfy.options")
    comfy_options.enable_args_parsing()
    module = importlib.import_module("comfyui_controlnet_aux")
    node_type, parameters = REFERENCE_CASES[case]
    node_class = module.NODE_CLASS_MAPPINGS[node_type]
    node = node_class()
    execute = getattr(node, node_class.FUNCTION)
    tensor = torch.from_numpy(image.copy())

    def run() -> np.ndarray:
        output = execute(image=tensor, resolution=resolution, **parameters)[0]
        return np.ascontiguousarray(output.detach().cpu().numpy(), dtype=np.float32)

    def unload() -> int:
        nonlocal node, execute
        node = None
        execute = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0

    return run, unload


def _setup_dinkster(
    case: str, image: np.ndarray, resolution: int, _device: str
) -> tuple[Callable[[], np.ndarray], Callable[[], int]]:
    from dinkster_api.v1 import PressureSignal
    from dinkster_assets import AssetVault, install_declared_assets, use_declared_asset_pack
    from dinkster_nodes_image import (
        EdgePreprocessor,
        LineartPreprocessor,
        ScribblePreprocessor,
    )
    from dinkster_vision_hed.cache import MODEL_CACHE
    from dinkster_vision_hed.nodes import (
        AnimeLineartPreprocessor,
        AnyLinePreprocessor,
        MangaLineartPreprocessor,
        MLSDPreprocessor,
        ModelEdgePreprocessor,
        RealisticLineartPreprocessor,
        TEEDPreprocessor,
    )
    from dinkster_workers import load_manifest

    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, AssetVault(VAULT))
    provider = "dinkster-vision-hed"
    cases: dict[str, tuple[type, dict[str, object]]] = {
        "lineart-standard": (
            LineartPreprocessor,
            {"gaussian_sigma": 6.0, "intensity_threshold": 8},
        ),
        "canny": (
            EdgePreprocessor,
            {"method": "canny", "low_threshold": 100, "high_threshold": 200},
        ),
        "pyracanny": (
            EdgePreprocessor,
            {"method": "pyramid_canny", "low_threshold": 64, "high_threshold": 128},
        ),
        "scribble": (ScribblePreprocessor, {"method": "threshold", "threshold": 32}),
        "scribble-xdog": (ScribblePreprocessor, {"method": "xdog", "threshold": 32}),
        "lineart-realistic": (
            RealisticLineartPreprocessor,
            {"provider": provider, "coarse": False},
        ),
        "lineart-realistic-coarse": (
            RealisticLineartPreprocessor,
            {"provider": provider, "coarse": True},
        ),
        "lineart-anime": (AnimeLineartPreprocessor, {"provider": provider}),
        "lineart-manga": (MangaLineartPreprocessor, {"provider": provider}),
        "hed-soft": (
            ModelEdgePreprocessor,
            {"provider": provider, "safe": False, "scribble": False},
        ),
        "hed-safe": (
            ModelEdgePreprocessor,
            {"provider": provider, "safe": True, "scribble": False},
        ),
        "hed-scribble": (
            ModelEdgePreprocessor,
            {"provider": provider, "safe": True, "scribble": True},
        ),
        "hed-scribble-unsafe": (
            ModelEdgePreprocessor,
            {"provider": provider, "safe": False, "scribble": True},
        ),
        "teed": (TEEDPreprocessor, {"provider": provider, "safe_steps": 2}),
        "teed-unquantized": (
            TEEDPreprocessor,
            {"provider": provider, "safe_steps": 0},
        ),
        "mlsd": (
            MLSDPreprocessor,
            {"provider": provider, "score_threshold": 0.1, "distance_threshold": 0.1},
        ),
        "anyline-standard": (
            AnyLinePreprocessor,
            {"provider": provider, "merge_with_lineart": "lineart_standard"},
        ),
        "anyline-realistic": (
            AnyLinePreprocessor,
            {"provider": provider, "merge_with_lineart": "lineart_realisitic"},
        ),
        "anyline-anime": (
            AnyLinePreprocessor,
            {"provider": provider, "merge_with_lineart": "lineart_anime"},
        ),
        "anyline-manga": (
            AnyLinePreprocessor,
            {"provider": provider, "merge_with_lineart": "manga_line"},
        ),
    }
    node, parameters = cases[case]
    context = use_declared_asset_pack(manifest.name)
    context.__enter__()

    def run() -> np.ndarray:
        output = node.execute(image=image, resolution=resolution, **parameters)["image"]
        return np.ascontiguousarray(output, dtype=np.float32)

    def unload() -> int:
        residency = "vram:cuda:0" if torch.cuda.is_available() else "ram"
        freed = asyncio.run(
            MODEL_CACHE.shed(PressureSignal(device=residency, bytes_needed=2**63 - 1))
        )
        context.__exit__(None, None, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return freed

    return run, unload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", choices=("reference", "dinkster"))
    parser.add_argument("case", choices=tuple(REFERENCE_CASES))
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--oom-fraction", type=float)
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("samples must be positive")

    source_path = REFERENCE / "examples" / "example_anyline.png"
    if _sha256(source_path) != SOURCE_SHA256:
        raise SystemExit("official example input does not match the pinned source revision")
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32)[None] / 255.0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA arm requested but CUDA is unavailable")
    if args.device == "cpu" and torch.cuda.is_available():
        raise SystemExit("CPU arm requires CUDA to be hidden from the process")
    if torch.cuda.is_available():
        if args.oom_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.oom_fraction)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    setup = _setup_reference if args.implementation == "reference" else _setup_dinkster
    run, unload = setup(args.case, source, args.resolution, args.device)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    gpu_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    stop, rss_samples = _start_rss_sampler(process)
    timings: list[float] = []
    output: np.ndarray | None = None
    error: dict[str, str] | None = None
    sample_sha256: list[str] = []
    sample_uint8_sha256: list[str] = []
    repeat_mismatch_count_maximum = 0
    repeat_max_abs_diff = 0.0
    try:
        for _ in range(args.samples):
            started = time.perf_counter()
            try:
                current = run()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                digest = hashlib.sha256(current.tobytes()).hexdigest()
                sample_sha256.append(digest)
                sample_uint8_sha256.append(
                    hashlib.sha256(np.rint(current * 255.0).astype(np.uint8).tobytes()).hexdigest()
                )
                if output is None:
                    output = current
                else:
                    repeat_mismatch_count_maximum = max(
                        repeat_mismatch_count_maximum, int(np.count_nonzero(current != output))
                    )
                    repeat_max_abs_diff = max(
                        repeat_max_abs_diff, float(np.max(np.abs(current - output)))
                    )
            except Exception as exc:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                error = {"type": type(exc).__name__, "message": str(exc)}
                break
        allocated_resident = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        reserved_resident = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        freed = unload()
    finally:
        stop.set()
    rss_after_unload = process.memory_info().rss
    rss_samples.append(rss_after_unload)
    allocated_after_unload = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    reserved_after_unload = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    warm = timings[1:] if error is None else []
    metrics: dict[str, Any] = {
        "implementation": args.implementation,
        "case": args.case,
        "device": args.device,
        "resolution": args.resolution,
        "sampleCount": args.samples,
        "oomFraction": args.oom_fraction,
        "timingsSeconds": timings,
        "coldSeconds": timings[0] if timings else None,
        "warmSeconds": warm,
        "warmSummarySeconds": _summary(warm),
        "rssBeforeBytes": rss_before,
        "rssPeakBytes": max(
            max(rss_samples), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "rssAfterUnloadBytes": rss_after_unload,
        "gpuAllocatedBeforeBytes": gpu_before,
        "gpuAllocatedResidentBytes": allocated_resident,
        "gpuAllocatedPeakBytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "gpuAllocatedAfterUnloadBytes": allocated_after_unload,
        "gpuReservedResidentBytes": reserved_resident,
        "gpuReservedPeakBytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        ),
        "gpuReservedAfterUnloadBytes": reserved_after_unload,
        "governedBytesFreed": freed,
        "outputShape": list(output.shape) if output is not None else None,
        "outputDtype": str(output.dtype) if output is not None else None,
        "outputFloat32Sha256": sample_sha256[0] if sample_sha256 else None,
        "outputSampleFloat32Sha256": sample_sha256,
        "outputUint8Sha256": (
            hashlib.sha256(np.rint(output * 255.0).astype(np.uint8).tobytes()).hexdigest()
            if output is not None
            else None
        ),
        "outputSampleUint8Sha256": sample_uint8_sha256,
        "outputBitStable": len(set(sample_sha256)) <= 1 if sample_sha256 else None,
        "repeatMismatchCountMaximum": repeat_mismatch_count_maximum,
        "repeatMaxAbsDiff": repeat_max_abs_diff,
        "error": error,
        "source": {
            "controlnetAuxCommit": CONTROLNET_AUX_COMMIT,
            "comfyuiCommit": COMFYUI_COMMIT,
            "inputSha256": SOURCE_SHA256,
            "artifacts": {
                path: {"sizeBytes": size, "sha256": sha256}
                for path, (size, sha256) in ARTIFACTS.items()
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpuUuid": (
                str(torch.cuda.get_device_properties(0).uuid) if torch.cuda.is_available() else None
            ),
            "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if output is not None:
        np.save(args.output.with_suffix(".npy"), output, allow_pickle=False)
        image = np.rint(output[0] * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(image).save(args.output.with_suffix(".png"), optimize=True)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
