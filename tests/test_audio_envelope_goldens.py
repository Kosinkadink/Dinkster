"""Replay the committed envelope-extraction goldens against ExtractAudioEnvelope.

The fixture (tests/goldens/audio_envelope.json) pins the node's own output
on deterministic inputs stored as exact float32 bytes: it proves
bit-stability of the frame slicing, band selection, aggregation, and
normalization behavior rather than parity with an external reference (the
audio-reactive idiom has no core ComfyUI equivalent). Values are compared
at 1e-9 so ulp-level FFT implementation drift never masks a real change;
see tools/gen_audio_envelope_goldens.py.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dinkster_nodes_media_io import ExtractAudioEnvelope

_GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "audio_envelope.json").read_text("utf-8")
)


def _fixture(name: str) -> dict[str, object]:
    entry = _GOLDENS["fixtures"][name]
    array = np.frombuffer(base64.b64decode(entry["dataB64"]), dtype=np.float32)
    return {
        "waveform": np.ascontiguousarray(array.reshape(entry["shape"])),
        "sample_rate": entry["sampleRate"],
    }


@pytest.mark.parametrize("case", _GOLDENS["cases"], ids=lambda case: case["name"])
def test_envelope_matches_pinned_golden(case: dict[str, Any]) -> None:
    result = ExtractAudioEnvelope.execute(audio=_fixture(str(case["fixture"])), **case["args"])
    pinned = np.asarray(case["envelope"], dtype=np.float64)
    envelope = np.asarray(result["envelope"], dtype=np.float64)
    assert result["frames"] == case["frames"]
    assert envelope.shape == pinned.shape
    np.testing.assert_allclose(envelope, pinned, rtol=0.0, atol=1e-9)


def test_goldens_cover_every_operation_and_normalization_mode() -> None:
    operations = {case["args"].get("operation", "max") for case in _GOLDENS["cases"]}
    assert operations == {"avg", "max", "sum"}
    assert any(case["args"].get("normalize") is False for case in _GOLDENS["cases"])
    assert any(case["args"].get("invert") is True for case in _GOLDENS["cases"])
    assert any(case["args"].get("frames_per_second") == 23.976 for case in _GOLDENS["cases"])
