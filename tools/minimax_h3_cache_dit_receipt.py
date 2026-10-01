"""Mint whole-job MiniMax H3 Cache-DiT quality and performance receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import wave
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ((672, 384, 56), (1344, 768, 124))
STEPS = 20
FPS = 24
AUDIO_CLIP_LEVEL = 1.0
QUALITY_FLOORS = {
    "video_latent_cosine": 0.95,
    "video_latent_relative_rmse": 0.30,
    "audio_latent_cosine": 0.98,
    "audio_latent_relative_rmse": 0.20,
    "decoded_video_ssim": 0.90,
}
CASES = (
    {
        "name": "fast-pan-signage",
        "category": "motion,text",
        "seed": 48501,
        "prompt": (
            "A fast lateral camera pan follows a red rally car through a rainy neon city. "
            "A large roadside sign clearly reads DINKSTER CACHE TEST. Tires hiss and the engine "
            "revs sharply, with no music."
        ),
    },
    {
        "name": "drummer-closeup",
        "category": "faces,audio-transients",
        "seed": 48502,
        "prompt": (
            "Close-up of a focused jazz drummer under warm stage lights, face and hands visible. "
            "Four crisp snare hits alternate with two bright cymbal strikes, with room ambience."
        ),
    },
    {
        "name": "train-platform-board",
        "category": "motion,text,audio-transients",
        "seed": 48503,
        "prompt": (
            "A train rushes into a station while commuters turn toward an arrival board that reads "
            "PLATFORM 7 - 18:45. Brakes squeal, then a short two-tone door chime sounds."
        ),
    },
    {
        "name": "chef-dialogue",
        "category": "faces,audio-transients",
        "seed": 48504,
        "prompt": (
            "A chef faces the camera in a quiet kitchen and says: add the salt after the water "
            "boils. "
            "A metal spoon taps a ceramic bowl once at the end. Natural lip motion, no music."
        ),
    },
    {
        "name": "skateboard-tracking",
        "category": "motion,faces,audio-transients",
        "seed": 48505,
        "prompt": (
            "Low tracking shot beside a skateboarder landing two tricks in a concrete plaza, face "
            "visible after the second landing. Distinct wheel rumble and two sharp board impacts."
        ),
    },
    {
        "name": "newsroom-ticker",
        "category": "faces,text",
        "seed": 48506,
        "prompt": (
            "A news presenter looks directly into the camera beside a screen reading WEATHER "
            "ALERT. "
            "A lower ticker reads NORTH BRIDGE CLOSED. Clean studio speech and no background music."
        ),
    },
    {
        "name": "dog-ball-splash",
        "category": "motion,audio-transients",
        "seed": 48507,
        "prompt": (
            "A dog sprints across a beach, catches a blue ball, and splashes through one shallow "
            "wave. "
            "Footfalls accelerate, followed by one bark and a loud splash."
        ),
    },
    {
        "name": "violinist-count-in",
        "category": "faces,text,audio-transients",
        "seed": 48508,
        "prompt": (
            "A violinist in a rehearsal room looks up at sheet music titled AUTUMN STUDY, quietly "
            "counts one two three four, then plays four separated notes with clear bow attacks."
        ),
    },
)


class ReceiptError(RuntimeError):
    pass


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *arguments), text=True, timeout=30
    ).strip()


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _asset(path: Path, digest: str) -> Any:
    from dinkster_assets import AssetRef

    return AssetRef(digest, path.name, path.stat().st_size, resolver=_Resolver(path))


def _tensor_sha256(value: Any) -> str:
    torch = cast("Any", importlib.import_module("torch"))
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _source_environment(dinkster_root: Path, fork_root: Path) -> dict[str, str]:
    package_sources = sorted(
        str(path) for path in (dinkster_root / "packages").glob("*/src") if path.is_dir()
    )
    current = os.environ.get("PYTHONPATH")
    sources = (str(dinkster_root / "src"), *package_sources, str(fork_root))
    return {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((*sources, *((current,) if current else ()))),
    }


def _inventory() -> dict[str, Any]:
    import importlib.metadata

    distributions = {}
    for name in ("av", "comfy-aimdo", "comfy-kitchen", "dinkster-inference", "torch"):
        distributions[name] = importlib.metadata.version(name)
    return {"executable": sys.executable, "distributions": distributions}


def _load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    from dinkster_native.fork_nodes import (
        GenerationLoadDiffusionModel,
        NativeLoadClip,
        NativeLoadVae,
        NativeMiniMaxH3CacheDIT,
    )

    model = GenerationLoadDiffusionModel.execute(
        diffusion_model=_asset(args.diffusion, args.diffusion_digest), weight_dtype="default"
    )["model"]
    clip = NativeLoadClip.execute(
        text_encoder=_asset(args.text_encoder, args.text_encoder_digest),
        type="minimax",
        device="default",
    )["clip"]
    video_vae = NativeLoadVae.execute(vae=_asset(args.video_vae, args.video_vae_digest))["vae"]
    audio_vae = NativeLoadVae.execute(vae=_asset(args.audio_vae, args.audio_vae_digest))["vae"]
    if args.arm == "cache":
        model = NativeMiniMaxH3CacheDIT.execute(model=model, policy=args.cache_policy)["MODEL"]
        model.model_options["transformer_options"]["dinkster_h3_cache_dit"]["receipt_sink"] = (
            args.cache_receipts
        )
    return model, clip, video_vae, audio_vae


def _run_case(
    args: argparse.Namespace,
    case: Mapping[str, Any],
    model: Any,
    clip: Any,
    video_vae: Any,
    audio_vae: Any,
) -> dict[str, Any]:
    from dinkster_native.fork_nodes import (
        GenerationKSampler,
        GenerationVAEDecode,
        NativeMiniMaxH3ImageToVideo,
        NativeVAEDecodeAudio,
    )

    torch = cast("Any", importlib.import_module("torch"))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    conditioned = NativeMiniMaxH3ImageToVideo.execute(
        clip=clip,
        vae=video_vae,
        prompt=case["prompt"],
        width=args.width,
        height=args.height,
        length=args.frames,
    )
    conditioned_seconds = time.perf_counter() - started
    sampled_started = time.perf_counter()
    sampled = cast(
        "dict[str, Any]",
        GenerationKSampler.execute(
            model=model,
            seed=case["seed"],
            steps=STEPS,
            cfg=1.0,
            sampler_name="res_multistep",
            scheduler="simple",
            positive=conditioned["positive"],
            negative=[],
            latent_image=conditioned["latent"],
            denoise=1.0,
        )["latent"],
    )
    torch.cuda.synchronize()
    sample_seconds = time.perf_counter() - sampled_started
    decode_started = time.perf_counter()
    video = cast("Any", GenerationVAEDecode.execute(samples=sampled, vae=video_vae)["image"])
    audio = cast(
        "dict[str, Any]",
        NativeVAEDecodeAudio.execute(samples=sampled, vae=audio_vae)["audio"],
    )
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    whole_job_seconds = time.perf_counter() - started
    streams = sampled["samples"]
    evidence = {
        "latents": {role: streams.by_role(role).detach().cpu() for role in ("video", "audio")},
        "decoded_video": video.detach().cpu().clamp(0.0, 1.0).mul(255).round().to(torch.uint8),
        "decoded_audio": audio["waveform"].detach().cpu().float(),
        "audio_sample_rate": int(audio["sample_rate"]),
    }
    evidence_path = args.evidence_directory / f"{case['name']}.pt"
    torch.save(evidence, evidence_path)
    return {
        "name": case["name"],
        "category": case["category"],
        "prompt": case["prompt"],
        "seed": case["seed"],
        "timing": {
            "conditioning_seconds": conditioned_seconds,
            "sample_seconds": sample_seconds,
            "decode_seconds": decode_seconds,
            "whole_job_seconds": whole_job_seconds,
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "hashes": {
            "video_latent": _tensor_sha256(evidence["latents"]["video"]),
            "audio_latent": _tensor_sha256(evidence["latents"]["audio"]),
            "decoded_video": _tensor_sha256(evidence["decoded_video"]),
            "decoded_audio": _tensor_sha256(evidence["decoded_audio"]),
        },
        "evidence": _file_receipt(evidence_path),
    }


def _deliberate_invalidation() -> dict[str, Any]:
    torch = cast("Any", importlib.import_module("torch"))
    from dinkster_inference.ldm.minimax.model import _cache_dit_key, _run_cache_dit_blocks

    class Layout:
        signature = (2, 3, 4, 5, 6)
        segments = ((0, 2, "text"), (2, 14, "audio"), (14, 74, "video"))

    config = {
        "model_identity": "receipt-invalidation",
        "policy": "quality",
        "Fn_compute_blocks": 1,
        "max_warmup_steps": 0,
        "residual_diff_threshold": 0.04,
        "max_continuous_cached_steps": 1,
    }
    runtime = {
        "key": None,
        "key_fields": None,
        "state": None,
        "hits": 0,
        "misses": 0,
        "invalidations": 0,
        "events": [],
    }
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    context = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    first_key, first_fields = _cache_dit_key(config, sigmas, context, Layout(), 3)
    changed_key, changed_fields = _cache_dit_key(config, sigmas, context + 1, Layout(), 3)

    def run_block(index: int, hidden: Any) -> Any:
        return hidden + float(index + 1)

    value = torch.zeros(2)
    _run_cache_dit_blocks(value, run_block, 3, config, runtime, first_key, first_fields, 0)
    _run_cache_dit_blocks(value, run_block, 3, config, runtime, first_key, first_fields, 1)
    _run_cache_dit_blocks(value, run_block, 3, config, runtime, changed_key, changed_fields, 2)
    passed = runtime["invalidations"] == 1 and runtime["events"][-1]["cache_hit"] is False
    return {
        "status": "PASS" if passed else "FAIL",
        "change": "conditioning bytes",
        "invalidations": runtime["invalidations"],
        "events": runtime["events"],
    }


def run_worker(args: argparse.Namespace) -> int:
    torch = cast("Any", importlib.import_module("torch"))
    if not torch.cuda.is_available():
        raise ReceiptError("MiniMax H3 Cache-DiT receipt worker requires CUDA")
    args.cache_receipts = []
    args.evidence_directory.mkdir(parents=True, exist_ok=False)
    worker_started = time.perf_counter()
    load_started = time.perf_counter()
    model, clip, video_vae, audio_vae = _load_runtime(args)
    load_seconds = time.perf_counter() - load_started
    cases = [_run_case(args, case, model, clip, video_vae, audio_vae) for case in CASES]
    properties = torch.cuda.get_device_properties(0)
    result = {
        "status": "PASS",
        "arm": args.arm,
        "cache_policy": args.cache_policy if args.arm == "cache" else None,
        "workload": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": STEPS,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "cfg": 1.0,
            "cases": len(CASES),
        },
        "timing": {
            "worker_wall_seconds": time.perf_counter() - worker_started,
            "model_load_seconds": load_seconds,
            "cold_whole_workflow_seconds": load_seconds
            + float(cases[0]["timing"]["whole_job_seconds"]),
            "warm_whole_job_seconds": [case["timing"]["whole_job_seconds"] for case in cases[1:]],
        },
        "cases": cases,
        "cache_receipts": args.cache_receipts,
        "deliberate_invalidation": (_deliberate_invalidation() if args.arm == "cache" else None),
        "environment": {
            "hostname": os.uname().nodename,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": properties.name,
            "gpu_uuid": str(properties.uuid),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "inventory": _inventory(),
    }
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _tensor_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
    torch = cast("Any", importlib.import_module("torch"))
    if reference.shape != candidate.shape:
        return {
            "shape_exact": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "finite": False,
            "cosine_similarity": None,
            "relative_rmse": None,
        }
    left = reference.double().flatten()
    right = candidate.double().flatten()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {
            "shape_exact": True,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "finite": False,
            "cosine_similarity": None,
            "relative_rmse": None,
        }
    difference = right - left
    left_rms = float(left.square().mean().sqrt())
    rmse = float(difference.square().mean().sqrt())
    left_norm = float(left.square().sum().sqrt())
    right_norm = float(right.square().sum().sqrt())
    cosine = (
        float((left * right).sum()) / (left_norm * right_norm) if left_norm and right_norm else 1.0
    )
    return {
        "shape_exact": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "finite": True,
        "cosine_similarity": cosine,
        "rmse": rmse,
        "relative_rmse": rmse / left_rms if left_rms else (0.0 if rmse == 0.0 else None),
    }


def _ssim_windows(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 4 or left.shape[-1] != 3:
        raise ReceiptError("decoded video tensors must have equal FHWC RGB shapes")
    total = 0.0
    count = 0
    for left_frame, right_frame in zip(left, right, strict=True):
        height, width, channels = left_frame.shape
        if height % 8 or width % 8:
            raise ReceiptError("decoded video dimensions must be divisible by eight")
        rows, columns = height // 8, width // 8
        a = left_frame.reshape(rows, 8, columns, 8, channels).astype(np.float64) / 255.0
        b = right_frame.reshape(rows, 8, columns, 8, channels).astype(np.float64) / 255.0
        a_mean = a.mean(axis=(1, 3))
        b_mean = b.mean(axis=(1, 3))
        a_centered = a - a_mean[:, None, :, None, :]
        b_centered = b - b_mean[:, None, :, None, :]
        a_variance = np.mean(a_centered * a_centered, axis=(1, 3))
        b_variance = np.mean(b_centered * b_centered, axis=(1, 3))
        covariance = np.mean(a_centered * b_centered, axis=(1, 3))
        values = ((2.0 * a_mean * b_mean + 0.01**2) * (2.0 * covariance + 0.03**2)) / (
            (a_mean * a_mean + b_mean * b_mean + 0.01**2) * (a_variance + b_variance + 0.03**2)
        )
        total += float(values.sum(dtype=np.float64))
        count += int(values.size)
    return total / count


def _adjacent_duplicate_frames(video: np.ndarray) -> list[int]:
    return [
        index
        for index in range(1, video.shape[0])
        if np.array_equal(video[index - 1], video[index])
    ]


def _write_video(path: Path, reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    av = cast("Any", importlib.import_module("av"))
    combined = np.concatenate((reference, candidate), axis=2)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=FPS)
        stream.width = int(combined.shape[2])
        stream.height = int(combined.shape[1])
        stream.pix_fmt = "yuv420p"
        for pixels in combined:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    decoded_frames = []
    with av.open(str(path), mode="r") as source:
        for frame in source.decode(video=0):
            decoded_frames.append(frame.to_ndarray(format="rgb24"))
    decoded = np.stack(decoded_frames)
    return {
        **_file_receipt(path),
        "fps": FPS,
        "frames": int(decoded.shape[0]),
        "expected_frames": int(combined.shape[0]),
        "dropped_frames": max(0, int(combined.shape[0] - decoded.shape[0])),
        "duplicated_frames": _adjacent_duplicate_frames(decoded),
    }


def _audio_array(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ReceiptError("decoded audio must have shape [1, channels, samples]")
    return array[0]


def _write_wav(path: Path, channels: np.ndarray, sample_rate: int) -> dict[str, Any]:
    clipped = np.clip(channels, -1.0, 1.0)
    pcm = np.round(clipped.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(int(channels.shape[0]))
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return {**_file_receipt(path), "sample_rate": sample_rate, "samples": channels.shape[1]}


def _spectrogram(samples: np.ndarray) -> np.ndarray:
    mono = samples.mean(axis=0)
    window = 1024
    hop = 256
    if mono.size < window:
        mono = np.pad(mono, (0, window - mono.size))
    frames = np.lib.stride_tricks.sliding_window_view(mono, window)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(window), axis=1)).T
    db = 20.0 * np.log10(np.maximum(spectrum, 1e-7))
    db = np.clip((db - db.max() + 80.0) / 80.0, 0.0, 1.0)
    return np.flipud(np.round(db * 255.0).astype(np.uint8))


def _write_spectrogram(path: Path, reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = _spectrogram(reference)
    right = _spectrogram(candidate)
    width = max(left.shape[1], right.shape[1])
    canvas = Image.new("L", (width, left.shape[0] + right.shape[0] + 24))
    canvas.paste(Image.fromarray(left), (0, 0))
    canvas.paste(Image.fromarray(right), (0, left.shape[0] + 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, left.shape[0] + 4), "dense above / cache below", fill=255)
    canvas.save(path)
    return _file_receipt(path)


def _metric_floor_failures(
    video_latent: Mapping[str, Any],
    audio_latent: Mapping[str, Any],
    video_ssim: float | None,
) -> list[str]:
    failures = []
    if not video_latent["shape_exact"] or not video_latent["finite"]:
        failures.append("video latent must be finite and shape-exact")
    elif video_latent["cosine_similarity"] is None or (
        float(video_latent["cosine_similarity"]) < QUALITY_FLOORS["video_latent_cosine"]
    ):
        failures.append("video latent cosine below 0.95")
    elif video_latent["relative_rmse"] is None or (
        float(video_latent["relative_rmse"]) > QUALITY_FLOORS["video_latent_relative_rmse"]
    ):
        failures.append("video latent relative RMSE above 0.30")
    if not audio_latent["shape_exact"] or not audio_latent["finite"]:
        failures.append("audio latent must be finite and shape-exact")
    elif audio_latent["cosine_similarity"] is None or (
        float(audio_latent["cosine_similarity"]) < QUALITY_FLOORS["audio_latent_cosine"]
    ):
        failures.append("audio latent cosine below 0.98")
    elif audio_latent["relative_rmse"] is None or (
        float(audio_latent["relative_rmse"]) > QUALITY_FLOORS["audio_latent_relative_rmse"]
    ):
        failures.append("audio latent relative RMSE above 0.20")
    if video_ssim is None or video_ssim < QUALITY_FLOORS["decoded_video_ssim"]:
        failures.append("decoded video SSIM below 0.90")
    return failures


def _case_quality(
    case_name: str,
    reference_path: Path,
    candidate_path: Path,
    artifact_directory: Path,
    expected_frames: int,
) -> dict[str, Any]:
    torch = cast("Any", importlib.import_module("torch"))
    reference = torch.load(reference_path, map_location="cpu", weights_only=True, mmap=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True, mmap=True)
    video_latent = _tensor_metrics(reference["latents"]["video"], candidate["latents"]["video"])
    audio_latent = _tensor_metrics(reference["latents"]["audio"], candidate["latents"]["audio"])
    reference_video = reference["decoded_video"].numpy()
    candidate_video = candidate["decoded_video"].numpy()
    video_shape_exact = reference_video.shape == candidate_video.shape
    video_finite = bool(np.isfinite(reference_video).all() and np.isfinite(candidate_video).all())
    video_ssim = (
        _ssim_windows(reference_video, candidate_video)
        if video_shape_exact and video_finite
        else None
    )
    reference_audio = _audio_array(reference["decoded_audio"])
    candidate_audio = _audio_array(candidate["decoded_audio"])
    reference_rate = int(reference["audio_sample_rate"])
    candidate_rate = int(candidate["audio_sample_rate"])
    audio_shape_exact = reference_audio.shape == candidate_audio.shape
    audio_finite = bool(np.isfinite(reference_audio).all() and np.isfinite(candidate_audio).all())
    clipping = int(np.count_nonzero(np.abs(candidate_audio) >= AUDIO_CLIP_LEVEL))
    video_artifact = _write_video(
        artifact_directory / f"{case_name}-side-by-side.mp4",
        reference_video,
        candidate_video,
    )
    reference_wav = _write_wav(
        artifact_directory / f"{case_name}-dense.wav", reference_audio, reference_rate
    )
    candidate_wav = _write_wav(
        artifact_directory / f"{case_name}-cache.wav", candidate_audio, candidate_rate
    )
    ab_channels = np.stack((reference_audio.mean(axis=0), candidate_audio.mean(axis=0)))
    ab_wav = _write_wav(artifact_directory / f"{case_name}-ab.wav", ab_channels, reference_rate)
    spectrogram = _write_spectrogram(
        artifact_directory / f"{case_name}-spectrogram.png", reference_audio, candidate_audio
    )
    failures = _metric_floor_failures(video_latent, audio_latent, video_ssim)
    if not video_shape_exact or not video_finite:
        failures.append("decoded video must be finite and shape-exact")
    if reference_video.shape[0] != expected_frames or candidate_video.shape[0] != expected_frames:
        failures.append("decoded video frame count differs from the requested workload")
    if reference_video.shape[0] != video_artifact["frames"] or video_artifact["dropped_frames"]:
        failures.append("side-by-side video dropped frames")
    if _adjacent_duplicate_frames(candidate_video) or video_artifact["duplicated_frames"]:
        failures.append("decoded video contains duplicated adjacent frames")
    if not audio_shape_exact or not audio_finite:
        failures.append("decoded audio must be finite and duration-exact")
    if reference_rate != candidate_rate:
        failures.append("decoded audio sample rates differ")
    if clipping:
        failures.append("decoded audio contains clipped samples")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "video_latent": video_latent,
        "audio_latent": audio_latent,
        "decoded_video": {
            "shape_exact": video_shape_exact,
            "finite": video_finite,
            "ssim_8x8_data_range_1": video_ssim,
            "candidate_adjacent_duplicate_frames": _adjacent_duplicate_frames(candidate_video),
        },
        "decoded_audio": {
            "shape_exact": audio_shape_exact,
            "finite": audio_finite,
            "duration_samples": [reference_audio.shape[1], candidate_audio.shape[1]],
            "sample_rates": [reference_rate, candidate_rate],
            "candidate_clipped_samples": clipping,
        },
        "artifacts": {
            "side_by_side_video": video_artifact,
            "dense_audio": reference_wav,
            "cache_audio": candidate_wav,
            "synchronized_ab_audio": ab_wav,
            "spectrogram": spectrogram,
        },
    }


def _cache_summary(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hits = sum(int(receipt["hits"]) for receipt in receipts)
    misses = sum(int(receipt["misses"]) for receipt in receipts)
    invalidations = sum(int(receipt["invalidations"]) for receipt in receipts)
    computed = Counter()
    skipped = Counter()
    steps = []
    for case_index, receipt in enumerate(receipts):
        for event in cast("Sequence[Mapping[str, Any]]", receipt["events"]):
            for block in cast("Sequence[int]", event["computed_blocks"]):
                computed[block] += 1
            for block in cast("Sequence[int]", event["skipped_blocks"]):
                skipped[block] += 1
            steps.append({"case_index": case_index, **event})
    return {
        "hits": hits,
        "misses": misses,
        "invalidations": invalidations,
        "computed_by_block": dict(sorted(computed.items())),
        "skipped_by_block": dict(sorted(skipped.items())),
        "steps": steps,
    }


def _quality_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "video_latent_cosine": [case["video_latent"]["cosine_similarity"] for case in cases],
        "video_latent_relative_rmse": [case["video_latent"]["relative_rmse"] for case in cases],
        "audio_latent_cosine": [case["audio_latent"]["cosine_similarity"] for case in cases],
        "audio_latent_relative_rmse": [case["audio_latent"]["relative_rmse"] for case in cases],
        "decoded_video_ssim": [case["decoded_video"]["ssim_8x8_data_range_1"] for case in cases],
    }
    return {
        name: {
            "median": statistics.median(cast("Sequence[float]", values))
            if all(value is not None for value in values)
            else None,
            "worst": (min(values) if "cosine" in name or "ssim" in name else max(values))
            if all(value is not None for value in values)
            else None,
        }
        for name, values in fields.items()
    }


def _spawn_arm(
    args: argparse.Namespace,
    arm: str,
    result: Path,
    evidence_directory: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--arm",
        arm,
        "--cache-policy",
        args.cache_policy,
        "--diffusion",
        str(args.diffusion),
        "--diffusion-digest",
        args.diffusion_digest,
        "--text-encoder",
        str(args.text_encoder),
        "--text-encoder-digest",
        args.text_encoder_digest,
        "--video-vae",
        str(args.video_vae),
        "--video-vae-digest",
        args.video_vae_digest,
        "--audio-vae",
        str(args.audio_vae),
        "--audio-vae-digest",
        args.audio_vae_digest,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--frames",
        str(args.frames),
        "--result",
        str(result),
        "--evidence-directory",
        str(evidence_directory),
    ]
    environment = _source_environment(args.dinkster_root, args.fork_root)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    log = result.with_suffix(".log")
    with log.open("wb") as output:
        completed = subprocess.run(
            command,
            cwd=args.dinkster_root,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
    if completed.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
        raise ReceiptError(f"{arm} worker exited {completed.returncode}:\n{tail}")
    loaded = cast("dict[str, Any]", json.loads(result.read_text()))
    loaded["worker_log"] = _file_receipt(log)
    return loaded


def run_mint(args: argparse.Namespace) -> int:
    from dinkster_assets import digest_file

    args.dinkster_root = args.dinkster_root.resolve()
    args.fork_root = args.fork_root.resolve()
    paths = ("diffusion", "text_encoder", "video_vae", "audio_vae")
    for name in paths:
        path = cast("Path", getattr(args, name)).resolve()
        if not path.is_file():
            raise ReceiptError(f"{name.replace('_', ' ')} does not exist: {path}")
        setattr(args, name, path)
        setattr(args, f"{name}_digest", digest_file(path))
    if (args.width, args.height, args.frames) not in SHAPES:
        raise ReceiptError(f"workload must be one of {SHAPES}")
    for root in (args.dinkster_root, args.fork_root):
        if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ReceiptError(f"source checkout is dirty: {root}")
    args.output = args.output.resolve()
    args.artifact_directory = args.artifact_directory.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_directory.mkdir(parents=True, exist_ok=False)
    dense_result = args.artifact_directory / "dense-worker.json"
    cache_result = args.artifact_directory / "cache-worker.json"
    arm_order = ("cache", "dense") if args.candidate_first else ("dense", "cache")
    arms = {}
    for arm in arm_order:
        arms[arm] = _spawn_arm(
            args,
            arm,
            cache_result if arm == "cache" else dense_result,
            args.artifact_directory / f"{arm}-evidence",
        )
    dense = cast("dict[str, Any]", arms["dense"])
    cache = cast("dict[str, Any]", arms["cache"])
    dense_cases = {
        case["name"]: case for case in cast("Sequence[Mapping[str, Any]]", dense["cases"])
    }
    cache_cases = {
        case["name"]: case for case in cast("Sequence[Mapping[str, Any]]", cache["cases"])
    }
    quality_cases = []
    for case in CASES:
        name = cast("str", case["name"])
        quality = _case_quality(
            name,
            Path(cast("Mapping[str, Any]", dense_cases[name]["evidence"])["path"]),
            Path(cast("Mapping[str, Any]", cache_cases[name]["evidence"])["path"]),
            args.artifact_directory,
            args.frames,
        )
        quality_cases.append({"name": name, "category": case["category"], **quality})
    dense_times = [
        float(case["timing"]["whole_job_seconds"])
        for case in cast("Sequence[Mapping[str, Any]]", dense["cases"])
    ]
    cache_times = [
        float(case["timing"]["whole_job_seconds"])
        for case in cast("Sequence[Mapping[str, Any]]", cache["cases"])
    ]
    median_dense = statistics.median(dense_times)
    median_cache = statistics.median(cache_times)
    speedup = median_dense / median_cache
    performance_floor = 2.0 if (args.width, args.height, args.frames) == SHAPES[1] else None
    quality_pass = all(case["status"] == "PASS" for case in quality_cases)
    invalidation_pass = cache["deliberate_invalidation"]["status"] == "PASS"
    performance_pass = performance_floor is None or speedup >= performance_floor
    status = "PASS" if quality_pass and invalidation_pass and performance_pass else "FAIL"
    receipt = {
        "schema": "dinkster.minimax-h3-cache-dit-receipt.v1",
        "status": status,
        "profile": args.cache_policy,
        "source": {
            "dinkster_head": _git(args.dinkster_root, "rev-parse", "HEAD"),
            "fork_head": _git(args.fork_root, "rev-parse", "HEAD"),
            "harness_sha256": _sha256(Path(__file__)),
        },
        "invocation": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": os.getcwd(),
            "arm_order": list(arm_order),
        },
        "workload": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": STEPS,
            "fps": FPS,
            "cases": CASES,
        },
        "quality_contract": {
            "floors": QUALITY_FLOORS,
            "audio_clip_level": AUDIO_CLIP_LEVEL,
            "threshold_misses_fail_profile": True,
            "cases": quality_cases,
            "summary": _quality_summary(quality_cases),
        },
        "performance": {
            "measurement": "whole job from prompt conditioning through dual VAE decode",
            "dense_seconds": dense_times,
            "cache_seconds": cache_times,
            "dense_median_seconds": median_dense,
            "cache_median_seconds": median_cache,
            "speedup": speedup,
            "production_speedup_floor": performance_floor,
            "status": "PASS" if performance_pass else "FAIL",
        },
        "cache": {
            "policy": args.cache_policy,
            "summary": _cache_summary(cast("Sequence[Mapping[str, Any]]", cache["cache_receipts"])),
            "deliberate_invalidation": cache["deliberate_invalidation"],
        },
        "memory": {
            "dense_peak_allocated_bytes": max(
                int(case["peak_allocated_bytes"])
                for case in cast("Sequence[Mapping[str, Any]]", dense["cases"])
            ),
            "dense_peak_reserved_bytes": max(
                int(case["peak_reserved_bytes"])
                for case in cast("Sequence[Mapping[str, Any]]", dense["cases"])
            ),
            "cache_peak_allocated_bytes": max(
                int(case["peak_allocated_bytes"])
                for case in cast("Sequence[Mapping[str, Any]]", cache["cases"])
            ),
            "cache_peak_reserved_bytes": max(
                int(case["peak_reserved_bytes"])
                for case in cast("Sequence[Mapping[str, Any]]", cache["cases"])
            ),
        },
        "model_artifacts": {
            name: _file_receipt(cast("Path", getattr(args, name))) for name in paths
        },
        "dense": dense,
        "candidate": cache,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{status}: {args.output}")
    return 0 if status == "PASS" else 1


def _artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--audio-vae", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--cache-policy", choices=("quality", "speed"), required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    _artifact_arguments(worker)
    worker.add_argument("--arm", choices=("dense", "cache"), required=True)
    for name in ("diffusion", "text_encoder", "video_vae", "audio_vae"):
        worker.add_argument(f"--{name.replace('_', '-')}-digest", required=True)
    worker.add_argument("--result", type=Path, required=True)
    worker.add_argument("--evidence-directory", type=Path, required=True)
    worker.set_defaults(function=run_worker)
    mint = commands.add_parser("mint")
    _artifact_arguments(mint)
    mint.add_argument("--dinkster-root", type=Path, required=True)
    mint.add_argument("--fork-root", type=Path, required=True)
    mint.add_argument("--gpu", required=True)
    mint.add_argument("--output", type=Path, required=True)
    mint.add_argument("--artifact-directory", type=Path, required=True)
    mint.add_argument("--candidate-first", action="store_true")
    mint.add_argument("--timeout", type=int, default=21_600)
    mint.set_defaults(function=run_mint)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
