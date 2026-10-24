"""CPU behavior for native ComfyUI audio combine nodes."""

from __future__ import annotations

import pytest
import torch
from dinkster_nodes_comfy.audio_combine import (
    AudioConcat,
    AudioMerge,
    JoinAudioChannels,
)


def _audio(values: list[list[list[float]]], rate: int = 8_000) -> dict[str, object]:
    return {"waveform": torch.tensor(values, dtype=torch.float32), "sample_rate": rate}


def test_null_audio_returns_the_original_object_without_materializing() -> None:
    opaque = object()
    assert JoinAudioChannels.execute(audio_left=opaque, audio_right=None)["audio"] is opaque
    assert AudioConcat.execute(audio1=None, audio2=opaque, direction="after")["AUDIO"] is opaque
    assert AudioMerge.execute(audio1=opaque, audio2=None, merge_method="add")["AUDIO"] is opaque


def test_join_rejects_non_mono_and_trims_with_left_right_order() -> None:
    left = _audio([[[1, 2, 3, 4]]])
    right = _audio([[[5, 6]]])
    joined = JoinAudioChannels.execute(audio_left=left, audio_right=right)["audio"]
    assert torch.equal(joined["waveform"], torch.tensor([[[1, 2], [5, 6]]], dtype=torch.float32))
    with pytest.raises(ValueError, match="Both input audios must be mono"):
        JoinAudioChannels.execute(audio_left=_audio([[[1], [2]]]), audio_right=right)


def test_concat_duplicates_mono_and_preserves_direction() -> None:
    first = _audio([[[1, 2]]])
    second = _audio([[[3]]])
    after = AudioConcat.execute(audio1=first, audio2=second, direction="after")["AUDIO"]
    before = AudioConcat.execute(audio1=first, audio2=second, direction="before")["AUDIO"]
    assert torch.equal(
        after["waveform"], torch.tensor([[[1, 2, 3], [1, 2, 3]]], dtype=torch.float32)
    )
    assert torch.equal(
        before["waveform"], torch.tensor([[[3, 1, 2], [3, 1, 2]]], dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("add", [0.75, 0.5, 0.5]),
        ("mean", [0.375, 0.25, 0.25]),
        ("subtract", [0.25, 0.5, 0.5]),
        ("multiply", [0.125, 0.0, 0.0]),
    ),
)
def test_merge_methods_pad_to_first_input_extent(method: str, expected: list[float]) -> None:
    first = _audio([[[0.5, 0.5, 0.5]]])
    second = _audio([[[0.25]]])
    result = AudioMerge.execute(audio1=first, audio2=second, merge_method=method)["AUDIO"]
    assert torch.equal(result["waveform"], torch.tensor([[expected]], dtype=torch.float32))


def test_merge_normalizes_only_above_one_and_retains_dtype_device() -> None:
    first = _audio([[[0.75, 0.75]]])
    result = AudioMerge.execute(audio1=first, audio2=first, merge_method="add")["AUDIO"]
    assert torch.equal(result["waveform"], torch.ones((1, 1, 2)))
    assert result["waveform"].dtype is torch.float32
    assert result["waveform"].device.type == "cpu"
