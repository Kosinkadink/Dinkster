"""Silent audio retains duration rounding and channel count."""

import pytest
import torch
from dinkster_audio.audio_generators import EmptyAudio


@pytest.mark.parametrize(
    ("duration", "rate", "channels", "length"),
    ((0.0, 11025, 1, 0), (0.3125, 8, 2, 2), (0.4375, 8, 1, 4), (0.01, 11025, 2, 110)),
)
def test_empty_audio_sample_rounding(
    duration: float, rate: int, channels: int, length: int
) -> None:
    audio = EmptyAudio.execute(duration=duration, sample_rate=rate, channels=channels)["AUDIO"]
    assert set(audio) == {"waveform", "sample_rate"}
    assert audio["sample_rate"] == rate
    waveform = audio["waveform"]
    assert waveform.shape == (1, channels, length)
    assert waveform.dtype is torch.float32
    assert waveform.device.type == "cpu"
    assert not waveform.count_nonzero()
