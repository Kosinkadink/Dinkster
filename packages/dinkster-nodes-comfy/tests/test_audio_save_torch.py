"""CPU behavior for native ComfyUI audio output nodes."""

from __future__ import annotations

import json
from pathlib import Path

import av
import pytest
import torch
from dinkster_api.v1 import MountWidget
from dinkster_nodes_comfy.audio_save import (
    SaveAudio,
    SaveAudioAdvanced,
    SaveAudioMP3,
    SaveAudioOpus,
)


def _audio() -> dict[str, object]:
    return {
        "waveform": torch.tensor(
            [
                [[0.0, 0.25, -0.5, 0.75], [0.75, -0.5, 0.25, 0.0]],
                [[-0.25, 0.5, -0.75, 1.0], [1.0, -0.75, 0.5, -0.25]],
            ],
            dtype=torch.float32,
        ),
        "sample_rate": 22_050,
    }


def _snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "output"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "outputMount": "output",
                "mounts": [{"id": "output", "root": str(root), "mode": "readwrite"}],
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root


@pytest.mark.parametrize(
    ("node", "kwargs", "suffix", "rate"),
    (
        (SaveAudio, {}, ".flac", 22_050),
        (SaveAudioMP3, {"quality": "320k"}, ".mp3", 22_050),
        (SaveAudioOpus, {"quality": "96k"}, ".opus", 48_000),
        (SaveAudioAdvanced, {"format": "opus", "quality": "192k"}, ".opus", 48_000),
    ),
)
def test_save_audio_writes_each_asymmetric_stereo_batch_and_passes_through_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node: type[SaveAudio],
    kwargs: dict[str, str],
    suffix: str,
    rate: int,
) -> None:
    root = _snapshot(tmp_path, monkeypatch)
    audio = _audio()
    original = audio["waveform"].clone()
    result = node.execute(
        audio=audio, filename_prefix="audio/%batch_num%/clip", mount="output", **kwargs
    )
    assert result["audio"] is audio
    assert torch.equal(audio["waveform"], original)
    paths = sorted(root.glob("audio/*/clip_*." + suffix.removeprefix(".")))
    assert len(paths) == 2
    assert [path.parent.name for path in paths] == ["0", "1"]
    for path in paths:
        source = path.read_bytes()
        assert source
        with av.open(path) as container:
            stream = container.streams.audio[0]
            assert stream.codec_context.sample_rate == rate
            assert stream.channels == 2
        assert path.read_bytes() == source
    assert paths[0].resolve().is_relative_to(root.resolve())


def test_save_audio_schema_preserves_upstream_inputs_with_a_trailing_mount() -> None:
    legacy = (SaveAudio, SaveAudioMP3, SaveAudioOpus)
    assert [input.id for input in SaveAudio.schema().inputs] == [
        "audio",
        "filename_prefix",
        "mount",
    ]
    for node in legacy:
        schema = node.schema()
        assert schema.outputs[0].id == "audio"
        assert schema.outputs[0].preview
        assert schema.deprecation is not None
        assert schema.inputs[-1].id == "mount"
        assert schema.inputs[-1].default is None
        assert isinstance(schema.inputs[-1].widget, MountWidget)
        assert schema.inputs[-1].adjacent_to == "filename_prefix"
    advanced = SaveAudioAdvanced.schema()
    assert [input.id for input in advanced.inputs] == ["audio", "filename_prefix", "mount"]
    assert advanced.combos[0].id == "format"
    assert tuple(option.key for option in advanced.combos[0].options) == ("flac", "mp3", "opus")


@pytest.mark.parametrize(
    ("node", "kwargs"),
    (
        (SaveAudioMP3, {"quality": "bad"}),
        (SaveAudioOpus, {"quality": "bad"}),
        (SaveAudioAdvanced, {"format": "wav"}),
        (SaveAudioAdvanced, {"format": "mp3", "quality": "bad"}),
    ),
)
def test_save_audio_rejects_untrusted_format_and_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node: type[SaveAudio], kwargs: dict[str, str]
) -> None:
    root = _snapshot(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        node.execute(audio=_audio(), filename_prefix="audio/clip", mount="output", **kwargs)
    assert not list(root.rglob("*"))


def test_save_audio_refuses_mounts_outside_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _snapshot(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="mount"):
        SaveAudio.execute(audio=_audio(), filename_prefix="audio/clip", mount="outside")
    assert not list(root.rglob("*"))
