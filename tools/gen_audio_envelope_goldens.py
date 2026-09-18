"""Generate audio envelope-extraction goldens from the native node itself.

Usage from the Dinkster repository root:

    .venv/bin/python tools/gen_audio_envelope_goldens.py

ExtractAudioEnvelope has no core ComfyUI equivalent (the audio-reactive
idiom lives in third-party packs), so these goldens pin the native node's
own output on deterministic fixtures: they prove bit-stability across
platforms and guard the frame slicing, band selection, aggregation, and
normalization behavior against regression rather than proving parity with
an external reference. Input fixtures are stored as exact float32 bytes;
outputs are full float lists compared value-close (1e-9) by the replay
test so ulp-level FFT implementation drift never masks a real change.

Run the generator twice and compare the printed sha256 before committing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "audio_envelope.json"

for entry in sorted(REPO.glob("packages/*/src")):
    sys.path.insert(0, str(entry))

from dinkster_nodes_media_io import ExtractAudioEnvelope  # noqa: E402


def _tone(
    samples: int, sample_rate: int, frequency: float, amplitude: float, *, channels: int = 1
) -> np.ndarray:
    timeline = np.arange(samples, dtype=np.float64) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * frequency * timeline)
    return np.broadcast_to(wave.astype(np.float32), (1, channels, samples)).copy()


def _noise(samples: int, *, channels: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wave = rng.standard_normal((1, channels, samples)) * 0.3
    return np.clip(wave, -1.0, 1.0).astype(np.float32)


def _fixtures() -> dict[str, dict[str, Any]]:
    return {
        "noise_mono_44k": {"waveform": _noise(44_100, seed=7), "sampleRate": 44_100},
        "noise_stereo_48k": {
            "waveform": _noise(48_000, channels=2, seed=11),
            "sampleRate": 48_000,
        },
        "tones_mixed": {
            "waveform": _tone(44_100, 44_100, 996.0, 0.5) + _tone(44_100, 44_100, 6000.0, 0.2),
            "sampleRate": 44_100,
        },
        "tone_short_tail": {"waveform": _tone(40_000, 44_100, 996.0, 0.5), "sampleRate": 44_100},
    }


CASES: tuple[dict[str, Any], ...] = (
    {"name": "defaults_max", "fixture": "noise_mono_44k", "args": {}},
    {
        "name": "avg_full_band",
        "fixture": "noise_mono_44k",
        "args": {"operation": "avg", "band_low_hz": 0.0, "band_high_hz": 100_000.0},
    },
    {
        "name": "sum_narrow_band",
        "fixture": "noise_mono_44k",
        "args": {"operation": "sum", "band_low_hz": 200.0, "band_high_hz": 800.0},
    },
    {
        "name": "raw_no_normalize",
        "fixture": "tones_mixed",
        "args": {"normalize": False},
    },
    {
        "name": "stereo_inverted",
        "fixture": "noise_stereo_48k",
        "args": {"invert": True},
    },
    {
        "name": "fractional_fps",
        "fixture": "noise_mono_44k",
        "args": {"frames_per_second": 23.976},
    },
    {
        "name": "short_tail_frame",
        "fixture": "tone_short_tail",
        "args": {"normalize": False},
    },
    {
        "name": "high_fps_stereo",
        "fixture": "noise_stereo_48k",
        "args": {"frames_per_second": 60.0, "operation": "avg"},
    },
)


def main() -> None:
    fixtures = _fixtures()
    fixture_entries = {
        name: {
            "shape": list(entry["waveform"].shape),
            "sampleRate": entry["sampleRate"],
            "dataB64": base64.b64encode(np.ascontiguousarray(entry["waveform"]).tobytes()).decode(
                "ascii"
            ),
        }
        for name, entry in fixtures.items()
    }
    case_entries = []
    for case in CASES:
        fixture = fixtures[case["fixture"]]
        result = ExtractAudioEnvelope.execute(
            audio={
                "waveform": fixture["waveform"],
                "sample_rate": fixture["sampleRate"],
            },
            **case["args"],
        )
        envelope = [float(value) for value in result["envelope"]]
        case_entries.append(
            {
                "name": case["name"],
                "fixture": case["fixture"],
                "args": case["args"],
                "envelope": envelope,
                "frames": int(result["frames"]),
            }
        )
    document = {
        "generator": "tools/gen_audio_envelope_goldens.py",
        "fixtures": fixture_entries,
        "cases": case_entries,
    }
    payload = json.dumps(document, indent=1, sort_keys=True) + "\n"
    OUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"sha256 {hashlib.sha256(payload.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
