"""Native asset-backed audio I/O contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_assets import AssetRef, AssetVault, AssetWriter, MountSnapshotResolver, digest_bytes
from dinkster_nodes_media_io import (
    EmptyAudio,
    LoadAudio,
    PreviewAudio,
    SaveAudio,
    SaveAudioMP3,
    SaveAudioOpus,
)
from dinkster_nodes_media_io.audio import MAX_ENCODED_AUDIO_BYTES
from dinkster_schema import AssetWidget
from dinkster_values import render_audio_wav


def _sine(
    samples: int = 44_100, sample_rate: int = 44_100, channels: int = 2, batch: int = 1
) -> dict[str, object]:
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    tone = np.sin(2 * np.pi * 440 * timeline, dtype=np.float32)
    clip = np.stack([tone * (1.0 - 0.5 * index) for index in range(channels)])
    waveform = np.stack([clip * (index + 1) / batch for index in range(batch)])
    return {"waveform": np.ascontiguousarray(waveform), "sample_rate": sample_rate}


def _mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "out"
    root.mkdir()
    index = root / ".dinkster-asset-index.json"
    index.write_text("{}", "utf-8")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "comfy-output",
                        "root": str(root),
                        "index": str(index),
                        "mode": "readwrite",
                    }
                ]
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root, snapshot


def _bound(ref: AssetRef, snapshot: Path) -> AssetRef:
    return AssetRef(
        ref.digest,
        ref.name,
        ref.size,
        ref.media_type,
        ref.virtual_path,
        MountSnapshotResolver(snapshot),
    )


@pytest.mark.parametrize(
    ("node", "kwargs", "suffix", "media_type", "codec"),
    [
        (SaveAudio, {}, ".flac", "audio/flac", "flac"),
        (SaveAudioMP3, {"quality": "V0"}, ".mp3", "audio/mpeg", "mp3float"),
        (SaveAudioMP3, {"quality": "320k"}, ".mp3", "audio/mpeg", "mp3float"),
        (SaveAudioOpus, {"quality": "128k"}, ".opus", "audio/ogg", "opus"),
    ],
)
def test_save_advertised_formats_publish_decodable_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node: type[Any],
    kwargs: dict[str, str],
    suffix: str,
    media_type: str,
    codec: str,
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    result = node.execute(
        audio=_sine(),
        target={"mount": "comfy-output", "prefix": "audio/clip"},
        **kwargs,
    )
    refs = cast("list[AssetRef]", result["audios"])
    assert len(refs) == 1
    ref = refs[0]
    assert ref.name.endswith(suffix)
    assert ref.media_type == media_type
    saved = root / "audio" / ref.name
    assert ref.digest == digest_bytes(saved.read_bytes())
    with av.open(str(saved)) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.name == codec
        assert stream.channels == 2


def test_save_publishes_one_ordered_asset_per_batch_element(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, snapshot = _mount(tmp_path, monkeypatch)
    result = SaveAudio.execute(audio=_sine(samples=4_800, batch=3))
    refs = cast("list[AssetRef]", result["audios"])
    assert [ref.name for ref in refs] == [
        "ComfyUI_00001.flac",
        "ComfyUI_00002.flac",
        "ComfyUI_00003.flac",
    ]
    peaks = []
    for ref in refs:
        loaded = cast("dict[str, Any]", LoadAudio.execute(audio=_bound(ref, snapshot))["audio"])
        peaks.append(float(np.abs(loaded["waveform"]).max()))
    assert peaks == sorted(peaks)
    assert peaks[0] < peaks[2]


def test_flac_round_trip_preserves_samples_within_s16_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, snapshot = _mount(tmp_path, monkeypatch)
    audio = _sine(samples=22_050, sample_rate=22_050)
    ref = cast("list[AssetRef]", SaveAudio.execute(audio=audio)["audios"])[0]
    loaded = cast("dict[str, Any]", LoadAudio.execute(audio=_bound(ref, snapshot))["audio"])
    waveform = cast(np.ndarray, loaded["waveform"])
    assert loaded["sample_rate"] == 22_050
    assert waveform.shape == (1, 2, 22_050)
    assert waveform.dtype == np.float32
    source = cast(np.ndarray, audio["waveform"])
    assert float(np.abs(waveform - source).max()) <= 1.0 / 32_768.0


def test_opus_coerces_unsupported_rates_and_preserves_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, snapshot = _mount(tmp_path, monkeypatch)
    ref = cast(
        "list[AssetRef]",
        SaveAudioOpus.execute(audio=_sine(samples=44_100, sample_rate=44_100))["audios"],
    )[0]
    loaded = cast("dict[str, Any]", LoadAudio.execute(audio=_bound(ref, snapshot))["audio"])
    waveform = cast(np.ndarray, loaded["waveform"])
    assert loaded["sample_rate"] == 48_000
    assert waveform.shape[2] == pytest.approx(48_000, abs=4_800)

    # Opus bitstreams always decode at 48 kHz (libopus contract), so even a
    # supported encode rate reads back as 48 kHz; duration is the invariant.
    ref = cast(
        "list[AssetRef]",
        SaveAudioOpus.execute(audio=_sine(samples=16_000, sample_rate=16_000))["audios"],
    )[0]
    loaded = cast("dict[str, Any]", LoadAudio.execute(audio=_bound(ref, snapshot))["audio"])
    assert loaded["sample_rate"] == 48_000
    waveform = cast(np.ndarray, loaded["waveform"])
    assert waveform.shape[2] == pytest.approx(48_000, abs=4_800)


def test_opus_quality_selects_declared_bit_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    sizes = {}
    for quality in ("64k", "320k"):
        ref = cast(
            "list[AssetRef]",
            SaveAudioOpus.execute(
                audio=_sine(),
                target={"mount": "comfy-output", "prefix": f"audio/{quality}"},
                quality=quality,
            )["audios"],
        )[0]
        sizes[quality] = (root / "audio" / ref.name).stat().st_size
    assert sizes["64k"] < sizes["320k"]


def test_mp3_rejects_unsupported_sample_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mount(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="MP3 encoding supports sample rates"):
        SaveAudioMP3.execute(audio=_sine(samples=1_000, sample_rate=44_123))


def test_save_quality_domains_are_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mount(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="quality"):
        SaveAudioMP3.execute(audio=_sine(samples=100), quality="192k")
    with pytest.raises(ValueError, match="quality"):
        SaveAudioOpus.execute(audio=_sine(samples=100), quality="V0")


@pytest.mark.parametrize(
    ("audio", "match"),
    [
        ({"waveform": np.zeros((2, 100), dtype=np.float32), "sample_rate": 44_100}, "layout"),
        ({"waveform": np.zeros((1, 0, 100), dtype=np.float32), "sample_rate": 44_100}, "channels"),
        ({"waveform": np.zeros((1, 1, 0), dtype=np.float32), "sample_rate": 44_100}, "samples"),
        (
            {"waveform": np.full((1, 1, 8), np.nan, dtype=np.float32), "sample_rate": 44_100},
            "non-finite",
        ),
        ({"waveform": np.zeros((1, 1, 8), dtype=np.float32), "sample_rate": 0}, "positive"),
        ({"waveform": [0.0], "sample_rate": 44_100}, "numpy"),
        (None, "waveform"),
    ],
)
def test_save_refuses_invalid_waveforms(audio: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SaveAudio.execute(audio=audio)


def test_save_input_budget_is_enforced_before_encoding() -> None:
    # as_strided reports nbytes from the shape (1*2*70e6*4 bytes ~= 534 MiB)
    # without allocating that much memory, so the budget check must trip.
    big = np.lib.stride_tricks.as_strided(
        np.zeros(8, dtype=np.float32), shape=(1, 2, 70_000_000), strides=(0, 0, 0)
    )
    with pytest.raises(ValueError, match="256 MiB"):
        SaveAudio.execute(audio={"waveform": big, "sample_rate": 44_100})


def test_empty_audio_matches_declared_sample_count_and_layout() -> None:
    result = cast(
        "dict[str, Any]",
        EmptyAudio.execute(duration=1.5, sample_rate=8_000, channels=1)["audio"],
    )
    waveform = cast(np.ndarray, result["waveform"])
    assert waveform.shape == (1, 1, 12_000)
    assert waveform.dtype == np.float32
    assert not waveform.any()
    assert result["sample_rate"] == 8_000
    half_sample = cast(
        "dict[str, Any]", EmptyAudio.execute(duration=0.25, sample_rate=2, channels=2)["audio"]
    )
    assert cast(np.ndarray, half_sample["waveform"]).shape == (1, 2, 0)
    surround = cast(
        "dict[str, Any]", EmptyAudio.execute(duration=0.5, sample_rate=8000, channels=6)["audio"]
    )
    assert cast(np.ndarray, surround["waveform"]).shape == (1, 6, 4000)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"duration": -1.0}, "duration"),
        ({"duration": float("nan")}, "duration"),
        ({"duration": 100_000.0}, "duration"),
        ({"sample_rate": 0}, "sample_rate"),
        ({"sample_rate": 400_000}, "sample_rate"),
        ({"sample_rate": 44_100.5}, "sample_rate must be an integer"),
        ({"sample_rate": 44_100.0}, "sample_rate must be an integer"),
        ({"sample_rate": "44100"}, "sample_rate must be an integer"),
        ({"sample_rate": True}, "sample_rate must be an integer"),
        ({"channels": 0}, "channels"),
        ({"channels": 1.0}, "channels must be a positive integer"),
        ({"channels": "2"}, "channels must be a positive integer"),
        ({"channels": True}, "channels must be a positive integer"),
        ({"duration": 3_600.0, "sample_rate": 192_000, "channels": 2}, "256 MiB"),
    ],
)
def test_empty_audio_rejects_out_of_domain_requests(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        EmptyAudio.execute(**kwargs)


def test_load_decodes_wav_bytes_exactly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _root, snapshot = _mount(tmp_path, monkeypatch)
    audio = _sine(samples=8_000, sample_rate=16_000)
    payload = render_audio_wav(audio)
    from dinkster_assets import MountSnapshotWriter

    ref = AssetWriter(MountSnapshotWriter(str(snapshot))).save_bytes(
        {"mount": "comfy-output", "prefix": "audio/tone"},
        payload,
        suffix=".wav",
        media_type="audio/wav",
    )
    loaded = cast("dict[str, Any]", LoadAudio.execute(audio=_bound(ref, snapshot))["audio"])
    waveform = cast(np.ndarray, loaded["waveform"])
    assert loaded["sample_rate"] == 16_000
    assert waveform.shape == (1, 2, 8_000)
    source = cast(np.ndarray, audio["waveform"])
    # WAV rendition scales positives by 32767 while s16 decode divides by
    # 32768, so worst-case error is scale skew (1) plus rounding (0.5).
    assert float(np.abs(waveform - source).max()) <= 1.5 / 32_768.0


def test_load_refuses_malformed_and_audioless_assets(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    payload = b"not audio"
    digest = digest_bytes(payload)
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    ref = AssetRef(digest, "bad.flac", len(payload), "audio/flac", resolver=vault)
    with pytest.raises(Exception, match="Invalid|no audio"):
        LoadAudio.execute(audio=ref)


def test_preview_validates_and_passes_audio_through() -> None:
    audio = _sine(samples=64)
    result = PreviewAudio.execute(audio=audio)
    assert result["audio"] is audio
    with pytest.raises((TypeError, ValueError)):
        PreviewAudio.execute(audio={"waveform": "wrong", "sample_rate": 8})


def test_audio_schemas_preserve_preview_asset_and_bound_contracts() -> None:
    load = LoadAudio.schema()
    assert load.node_type == "dinkster.load_audio"
    assert load.inputs[0].type.kind == "asset"
    assert load.inputs[0].type.element is not None
    assert load.inputs[0].type.element.types == ("comfy.AUDIO",)
    assert load.inputs[0].source_filename is not None
    assert load.inputs[0].source_filename.kind == "media/audio"
    assert isinstance(load.inputs[0].widget, AssetWidget)
    assert load.inputs[0].widget.accept == (
        "audio/wav",
        "audio/flac",
        "audio/mpeg",
        "audio/ogg",
        "audio/webm",
        "audio/mp4",
        "video/mp4",
        "video/webm",
    )
    assert [(output.id, output.preview) for output in load.outputs] == [("audio", True)]

    for node, quality_options in (
        (SaveAudio, None),
        (SaveAudioMP3, ("V0", "128k", "320k")),
        (SaveAudioOpus, ("64k", "96k", "128k", "192k", "320k")),
    ):
        schema = node.schema()
        assert schema.output_node
        assert not schema.idempotent
        by_id = {spec.id: spec for spec in schema.inputs}
        assert by_id["audio"].on_absent == "fail"
        assert by_id["target"].default == {"mount": "comfy-output", "prefix": "audio/ComfyUI"}
        output = schema.outputs[0]
        assert output.id == "audios"
        assert output.preview
        assert output.type.kind == "list"
        assert output.type.element is not None
        assert output.type.element.kind == "asset"
        if quality_options is None:
            assert "quality" not in by_id
        else:
            widget = by_id["quality"].widget
            assert widget is not None
            assert getattr(widget, "options", None) == quality_options

    preview = PreviewAudio.schema()
    assert preview.output_node
    assert preview.inputs[0].on_absent == "fail"
    assert [(output.id, output.preview) for output in preview.outputs] == [("audio", True)]

    empty = EmptyAudio.schema()
    assert not empty.output_node
    defaults = {spec.id: spec.default for spec in empty.inputs}
    assert defaults == {"duration": 60.0, "sample_rate": 44_100, "channels": 2}

    encoded_limit = MAX_ENCODED_AUDIO_BYTES
    assert encoded_limit == 256 * 1024 * 1024
