"""Generate audio editing-op goldens from ComfyUI e20d433a.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies (torch and torchaudio):

    COMFYUI_ROOT=/path/to/ComfyUI \
      /path/to/python tools/gen_audio_ops_goldens.py

The ComfyUI checkout must be clean and pinned to the commit below. Run the
generator twice and compare the printed sha256 before committing the fixture.

Input fixtures are minted here once and stored as exact float32 bytes, so
the Dinkster-side tests replay identical inputs with no cross-version RNG or
libm dependence. Pure value ops (trim, split, join, concat, merge, volume)
store the sha256 of the pinned output bytes: the native port must match
bit-exactly. Equalizer cases store full pinned outputs for a value-close
comparison against the float64 biquad port. Mixed-sample-rate cases are not
goldened: the native resampler is libswresample, a declared gap.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

BASELINE = "e20d433a4966dcc88fa5abbae6ace824cb78b263"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "audio_ops_e20d433a.json"

FIXTURE_SAMPLES = 4096
FIXTURE_SHORT_SAMPLES = 3000


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def build_fixtures() -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(20260828)

    def _mint(channels: int, samples: int, rate: int, amplitude: float) -> dict[str, object]:
        time = np.arange(samples, dtype=np.float64) / rate
        tones = (
            0.5 * np.sin(2.0 * np.pi * 110.0 * time)
            + 0.3 * np.sin(2.0 * np.pi * 1000.0 * time)
            + 0.2 * np.sin(2.0 * np.pi * 7000.0 * time)
        )
        waveform = np.stack([tones + 0.05 * rng.standard_normal(samples) for _ in range(channels)])
        peak = np.abs(waveform).max()
        waveform = np.asarray(waveform / peak * amplitude, dtype=np.float32)[None, ...]
        return {"waveform": waveform, "sample_rate": rate}

    return {
        "stereo_a": _mint(2, FIXTURE_SAMPLES, 44100, 0.7),
        "stereo_hot": _mint(2, FIXTURE_SAMPLES, 44100, 0.999),
        "mono_a": _mint(1, FIXTURE_SAMPLES, 44100, 0.6),
        "mono_b": _mint(1, FIXTURE_SHORT_SAMPLES, 44100, 0.8),
        "stereo_48k": _mint(2, FIXTURE_SAMPLES, 48000, 0.7),
    }


TRIM_CASES = [
    ("trim_mid", "stereo_a", 0.02, 0.05),
    ("trim_negative_start", "stereo_a", -0.03, 0.02),
    ("trim_clamp_end", "stereo_a", 0.05, 60.0),
    ("trim_rounding", "mono_a", 0.0123, 0.0456),
]

CONCAT_CASES = [
    ("concat_after", "stereo_a", "stereo_hot", "after"),
    ("concat_before", "stereo_a", "stereo_hot", "before"),
    ("concat_mono_dup", "mono_a", "stereo_a", "after"),
]

MERGE_CASES = [
    ("merge_add_normalize", "stereo_hot", "stereo_hot", "add"),
    ("merge_subtract_pad", "stereo_a", "mono_b", "subtract"),
    ("merge_multiply_trim", "mono_b", "stereo_a", "multiply"),
    ("merge_mean", "stereo_a", "stereo_hot", "mean"),
]

VOLUME_CASES = [
    ("volume_plus6", "stereo_a", 6),
    ("volume_minus6", "stereo_a", -6),
    ("volume_plus3", "mono_a", 3),
    ("volume_minus100", "stereo_a", -100),
]

EQ_CASES = [
    ("eq_low_boost", "stereo_a", 6.0, 100, 0.0, 1000, 0.707, 0.0, 5000),
    ("eq_mid_cut_high_q", "stereo_a", 0.0, 100, -4.5, 1000, 2.0, 0.0, 5000),
    ("eq_mid_boost_low_q", "mono_a", 0.0, 100, 3.0, 2500, 0.5, 0.0, 5000),
    ("eq_high_boost", "stereo_a", 0.0, 100, 0.0, 1000, 0.707, 6.0, 5000),
    ("eq_all_bands_48k", "stereo_48k", 4.0, 200, -3.0, 1500, 1.2, 5.0, 8000),
    ("eq_clamp_hot", "stereo_hot", 12.0, 250, 0.0, 1000, 0.707, 12.0, 9000),
    ("eq_extreme", "mono_a", 24.0, 20, -24.0, 4000, 10.0, 24.0, 15000),
]


def build_goldens(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    # nodes_audio imports comfy.model_management, which parses argv at
    # import; a CPU-only torch install needs --cpu to survive that import.
    import comfy.options  # pyright: ignore[reportMissingImports]

    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]

    import torch
    from comfy_extras import nodes_audio  # pyright: ignore[reportMissingImports]

    fixtures = build_fixtures()

    def _torch_audio(name: str) -> dict[str, object]:
        entry = fixtures[name]
        return {
            "waveform": torch.from_numpy(np.asarray(entry["waveform"]).copy()),
            "sample_rate": entry["sample_rate"],
        }

    def _output_array(value: Any) -> tuple[np.ndarray, int]:
        waveform = value["waveform"].detach().cpu().numpy()
        if waveform.dtype != np.float32:
            raise RuntimeError(f"pinned output dtype {waveform.dtype} is not float32")
        return np.ascontiguousarray(waveform), int(value["sample_rate"])

    def _hashed(value: Any) -> dict[str, object]:
        array, rate = _output_array(value)
        return {
            "shape": list(array.shape),
            "sampleRate": rate,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    def _full(value: Any) -> dict[str, object]:
        array, rate = _output_array(value)
        return {
            "shape": list(array.shape),
            "sampleRate": rate,
            "dataB64": _b64(array.tobytes()),
        }

    cases: dict[str, object] = {}
    for name, source, start_index, duration in TRIM_CASES:
        out = nodes_audio.TrimAudioDuration.execute(_torch_audio(source), start_index, duration)
        cases[name] = {
            "op": "trim",
            "source": source,
            "start": start_index,
            "duration": duration,
            "output": _hashed(out.result[0]),
        }

    split = nodes_audio.SplitAudioChannels.execute(_torch_audio("stereo_a"))
    cases["split_stereo"] = {
        "op": "split",
        "source": "stereo_a",
        "left": _hashed(split.result[0]),
        "right": _hashed(split.result[1]),
    }

    join = nodes_audio.JoinAudioChannels.execute(_torch_audio("mono_a"), _torch_audio("mono_b"))
    cases["join_trims_to_shorter"] = {
        "op": "join",
        "left": "mono_a",
        "right": "mono_b",
        "output": _hashed(join.result[0]),
    }

    for name, first, second, direction in CONCAT_CASES:
        out = nodes_audio.AudioConcat.execute(_torch_audio(first), _torch_audio(second), direction)
        cases[name] = {
            "op": "concat",
            "audio1": first,
            "audio2": second,
            "direction": direction,
            "output": _hashed(out.result[0]),
        }

    for name, first, second, method in MERGE_CASES:
        out = nodes_audio.AudioMerge.execute(_torch_audio(first), _torch_audio(second), method)
        cases[name] = {
            "op": "merge",
            "audio1": first,
            "audio2": second,
            "method": method,
            "output": _hashed(out.result[0]),
        }

    for name, source, volume in VOLUME_CASES:
        out = nodes_audio.AudioAdjustVolume.execute(_torch_audio(source), volume)
        cases[name] = {
            "op": "volume",
            "source": source,
            "volume": volume,
            "output": _hashed(out.result[0]),
        }

    for (
        name,
        source,
        low_gain,
        low_freq,
        mid_gain,
        mid_freq,
        mid_q,
        high_gain,
        high_freq,
    ) in EQ_CASES:
        out = nodes_audio.AudioEqualizer3Band.execute(
            _torch_audio(source),
            low_gain,
            low_freq,
            mid_gain,
            mid_freq,
            mid_q,
            high_gain,
            high_freq,
        )
        cases[name] = {
            "op": "equalizer",
            "source": source,
            "lowGainDb": low_gain,
            "lowFreq": low_freq,
            "midGainDb": mid_gain,
            "midFreq": mid_freq,
            "midQ": mid_q,
            "highGainDb": high_gain,
            "highFreq": high_freq,
            "output": _full(out.result[0]),
        }

    stored_fixtures = {
        name: {
            "shape": list(np.asarray(entry["waveform"]).shape),
            "sampleRate": entry["sample_rate"],
            "dataB64": _b64(np.ascontiguousarray(np.asarray(entry["waveform"])).tobytes()),
        }
        for name, entry in fixtures.items()
    }
    return {
        "revision": BASELINE,
        "generator": "tools/gen_audio_ops_goldens.py",
        "fixtures": stored_fixtures,
        "cases": cases,
    }


def main() -> None:
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    content = (json.dumps(build_goldens(comfy_root.resolve()), indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"wrote {OUT} sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
