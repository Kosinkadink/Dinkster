"""Replay pinned ComfyUI e20d433a audio editing-op goldens against the native ops.

The fixture (tests/goldens/audio_ops_e20d433a.json) stores exact float32
input bytes plus, per case, either the sha256 of the pinned output bytes
(pure value ops, compared bit-exactly) or the full pinned output (equalizer
cases, compared value-close); see tools/gen_audio_ops_goldens.py.

The equalizer tolerances below are per-case against the pinned float32
torchaudio chain. The native port is float64 end to end and agrees with the
same chain run in float64 to under 1e-12 for every case, so the residual is
the reference's own float32 arithmetic: coefficient rounding feeds an IIR
whose sensitivity grows as the shelf pole approaches DC (low corner
frequency) and with gain. Each bound is the observed drift with about 4x
headroom; do not widen them to absorb port changes.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from dinkster_nodes_media_io import (
    AdjustAudioVolume,
    ConcatAudio,
    EqualizeAudio,
    JoinAudioChannels,
    MergeAudio,
    SplitAudioChannels,
    TrimAudio,
)

_GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "audio_ops_e20d433a.json").read_text("utf-8")
)

# name -> (observed max abs drift vs the pinned float32 chain, pinned bound)
_EQ_ATOL = {
    "eq_low_boost": 2e-3,  # observed 4.1e-4 (+6 dB shelf at 100 Hz)
    "eq_mid_cut_high_q": 2e-5,  # observed 2.9e-6
    "eq_mid_boost_low_q": 2e-6,  # observed 3.6e-7
    "eq_high_boost": 1e-6,  # observed 1.8e-7
    "eq_all_bands_48k": 4e-4,  # observed 7.5e-5
    "eq_clamp_hot": 7e-4,  # observed 1.4e-4 (per-stage clamp engaged)
    "eq_extreme": 4e-2,  # observed 9.5e-3 (+24 dB shelf at 20 Hz, pole near DC)
}


def _fixture(name: str) -> dict[str, object]:
    entry = _GOLDENS["fixtures"][name]
    array = np.frombuffer(base64.b64decode(entry["dataB64"]), dtype=np.float32)
    return {
        "waveform": np.ascontiguousarray(array.reshape(entry["shape"])),
        "sampleRate": entry["sampleRate"],
        "sample_rate": entry["sampleRate"],
    }


def _cases(op: str) -> list[str]:
    return sorted(name for name, case in _GOLDENS["cases"].items() if case["op"] == op)


def _assert_bit_exact(native: object, reference: dict[str, object]) -> None:
    value = cast("dict[str, object]", native)
    waveform = np.ascontiguousarray(cast("np.ndarray", value["waveform"]))
    assert list(waveform.shape) == reference["shape"]
    assert value["sample_rate"] == reference["sampleRate"]
    assert hashlib.sha256(waveform.tobytes()).hexdigest() == reference["sha256"]


def test_goldens_are_pinned_to_the_audited_revision() -> None:
    assert _GOLDENS["revision"] == "e20d433a4966dcc88fa5abbae6ace824cb78b263"


@pytest.mark.parametrize("name", _cases("trim"))
def test_trim_matches_pinned_bytes_exactly(name: str) -> None:
    case = _GOLDENS["cases"][name]
    result = TrimAudio.execute(
        audio=_fixture(case["source"]), start=case["start"], duration=case["duration"]
    )
    _assert_bit_exact(result["audio"], case["output"])


def test_split_matches_pinned_bytes_exactly() -> None:
    case = _GOLDENS["cases"]["split_stereo"]
    result = SplitAudioChannels.execute(audio=_fixture(case["source"]))
    _assert_bit_exact(result["left"], case["left"])
    _assert_bit_exact(result["right"], case["right"])


def test_join_matches_pinned_bytes_exactly() -> None:
    case = _GOLDENS["cases"]["join_trims_to_shorter"]
    result = JoinAudioChannels.execute(
        audio_left=_fixture(case["left"]), audio_right=_fixture(case["right"])
    )
    _assert_bit_exact(result["audio"], case["output"])


@pytest.mark.parametrize("name", _cases("concat"))
def test_concat_matches_pinned_bytes_exactly(name: str) -> None:
    case = _GOLDENS["cases"][name]
    result = ConcatAudio.execute(
        audio1=_fixture(case["audio1"]),
        audio2=_fixture(case["audio2"]),
        direction=case["direction"],
    )
    _assert_bit_exact(result["audio"], case["output"])


@pytest.mark.parametrize("name", _cases("merge"))
def test_merge_matches_pinned_bytes_exactly(name: str) -> None:
    case = _GOLDENS["cases"][name]
    result = MergeAudio.execute(
        audio1=_fixture(case["audio1"]),
        audio2=_fixture(case["audio2"]),
        merge_method=case["method"],
    )
    _assert_bit_exact(result["audio"], case["output"])


@pytest.mark.parametrize("name", _cases("volume"))
def test_volume_matches_pinned_bytes_exactly(name: str) -> None:
    case = _GOLDENS["cases"][name]
    result = AdjustAudioVolume.execute(audio=_fixture(case["source"]), gain_db=case["volume"])
    _assert_bit_exact(result["audio"], case["output"])


@pytest.mark.parametrize("name", _cases("equalizer"))
def test_equalizer_is_value_close_to_the_pinned_chain(name: str) -> None:
    case = _GOLDENS["cases"][name]
    result = EqualizeAudio.execute(
        audio=_fixture(case["source"]),
        low_gain_db=case["lowGainDb"],
        low_freq=case["lowFreq"],
        mid_gain_db=case["midGainDb"],
        mid_freq=case["midFreq"],
        mid_q=case["midQ"],
        high_gain_db=case["highGainDb"],
        high_freq=case["highFreq"],
    )
    value = cast("dict[str, object]", result["audio"])
    waveform = cast("np.ndarray", value["waveform"])
    reference = np.frombuffer(base64.b64decode(case["output"]["dataB64"]), dtype=np.float32)
    reference = reference.reshape(case["output"]["shape"])
    assert list(waveform.shape) == case["output"]["shape"]
    assert value["sample_rate"] == case["output"]["sampleRate"]
    drift = float(np.abs(waveform.astype(np.float64) - reference.astype(np.float64)).max())
    assert drift <= _EQ_ATOL[name], f"{name}: drift {drift:.3e} exceeds {_EQ_ATOL[name]:.1e}"


def test_every_equalizer_case_has_a_pinned_tolerance() -> None:
    assert set(_EQ_ATOL) == set(_cases("equalizer"))
