"""CPU coverage for the public AUDIO tensor materializer."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from dinkster_api.v1 import materialize_audio
from dinkster_values.audio_codec import (
    AudioSourceUnavailableError,
    audio_from_source,
    audio_window,
    render_audio_wav,
)
from dinkster_values.audio_lazy import AUDIO_WINDOW_LIMIT

torch = pytest.importorskip("torch")


def test_materialize_audio_keeps_live_float_tensor_mapping_by_identity() -> None:
    waveform = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    audio = {"waveform": waveform, "sample_rate": 48000, "metadata": "kept"}
    assert materialize_audio(audio) is audio
    assert materialize_audio(audio)["waveform"] is waveform


def test_materialize_audio_converts_portable_pcm_and_preserves_metadata() -> None:
    floats = np.array([[[0.25, -0.5]], [[0.75, 1.0]]], dtype=np.float32)
    portable = {"waveform": floats, "sample_rate": 44100, "metadata": {"name": "portable"}}
    materialized = materialize_audio(portable)
    assert materialized["sample_rate"] == 44100
    assert materialized["metadata"] == {"name": "portable"}
    waveform = cast(Any, materialized["waveform"])
    assert waveform.dtype == torch.float32
    torch.testing.assert_close(waveform, torch.from_numpy(floats))

    pcm = {"waveform": np.array([[[-32768, 32767]]], dtype=np.int16), "sample_rate": 8000}
    normalized = materialize_audio(pcm)
    torch.testing.assert_close(
        normalized["waveform"], torch.tensor([[[-1.0, 32767 / 32768]]], dtype=torch.float32)
    )


def test_materialize_audio_reads_known_wav_extent_and_refuses_unbounded_whole_audio() -> None:
    samples = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(1, 1, 16)
    lazy = audio_from_source(render_audio_wav({"waveform": samples, "sample_rate": 8000}))
    materialized = materialize_audio(lazy)
    torch.testing.assert_close(
        materialized["waveform"], torch.from_numpy(samples), atol=1 / 32768, rtol=0
    )
    assert materialized["sample_rate"] == 8000
    assert audio_window(lazy, 0, 16)["waveform"].shape == (1, 1, 16)

    probe = cast(dict[str, object], lazy["probe"])
    unknown = {**lazy, "probe": {**probe, "frames": None, "duration": None}}
    with pytest.raises(ValueError, match="unknown extent"):
        materialize_audio(unknown)

    frames = (AUDIO_WINDOW_LIMIT // 4) + 1
    over_budget = {
        **lazy,
        "probe": {**probe, "frames": frames, "duration": frames / 8000},
    }
    with pytest.raises(ValueError, match="32 MiB"):
        materialize_audio(over_budget)


def test_materialize_audio_refuses_unavailable_lazy_source() -> None:
    samples = np.zeros((1, 1, 8), dtype=np.float32)
    lazy = audio_from_source(render_audio_wav({"waveform": samples, "sample_rate": 8000}))
    portable = {
        "source": {"digest": f"blake3:{'0' * 64}", "size": 1},
        "probe": lazy["probe"],
        "edits": [],
    }
    with pytest.raises(AudioSourceUnavailableError, match="no local asset binding"):
        materialize_audio(portable)
