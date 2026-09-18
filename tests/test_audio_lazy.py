"""Sample-accurate lazy AUDIO, transport admission, surround, and bounded consumers."""

from __future__ import annotations

import asyncio
import copy
import io
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import GeneratorType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dinkster_assets import AssetError, AssetIntegrityError, AssetRef, AssetVault, digest_bytes
from dinkster_assets.audio import bind_audio_value, register_audio_value_type
from dinkster_nodes_foundation.types import Curve
from dinkster_nodes_media_io.audio import LoadAudio, SaveAudio, SaveAudioMP3, SaveAudioOpus
from dinkster_nodes_media_io.audio_ops import (
    AdjustAudioVolume,
    AudioOnsets,
    ConcatAudio,
    DownmixAudio,
    EqualizeAudio,
    ExtractAudioEnvelope,
    FadeAudio,
    JoinAudioChannels,
    MergeAudio,
    ResampleAudio,
    SplitAudioChannels,
    TrimAudio,
)
from dinkster_values import TypeRegistry
from dinkster_values.audio_codec import (
    AUDIO_INLINE_LIMIT,
    AudioRangeError,
    AudioSourceUnavailableError,
    AudioWindowReader,
    append_audio_edit,
    audio_encoded_meta,
    audio_fingerprint,
    audio_from_source,
    audio_meta,
    audio_window,
    bind_audio_sources,
    coerce_audio,
    decode_audio,
    effective_audio_facts,
    encode_audio,
    normalize_audio_waveform_request,
    normalize_audio_window_request,
    render_audio_wav,
    render_audio_waveform,
    render_audio_window,
    validate_audio_encoded,
)
from PIL import Image

from tests.test_audio_io import _bound, _mount


@pytest.fixture
def source(tmp_path: Path) -> tuple[AssetRef, np.ndarray]:
    pcm = np.arange(2 * 16_000, dtype=np.int16).reshape(1, 2, 16_000)
    buffer = io.BytesIO()
    buffer.write(struct.pack("<Q", 8000))
    np.save(buffer, pcm, allow_pickle=False)
    data = buffer.getvalue()
    vault = AssetVault(tmp_path / "vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    return AssetRef(digest, "source.pcm", len(data), resolver=vault), pcm.astype(np.float32) / 32768


def test_unknown_duration_probe_counts_frames_without_retaining_decode() -> None:
    import weakref

    import dinkster_values.audio_lazy as runtime

    class Frame:
        sample_rate = 48_000
        samples = 960
        layout = type("Layout", (), {"channels": (0, 1)})()

    class Frames:
        def __init__(self) -> None:
            self.index = 0
            self.references: list[weakref.ReferenceType[Frame]] = []
            self.max_live = 0

        def __iter__(self):
            return self

        def __next__(self) -> Frame:
            live = sum(reference() is not None for reference in self.references)
            self.max_live = max(self.max_live, live + 1)
            assert live <= 1, "decoded frames were retained"
            if self.index == 100:
                raise StopIteration
            self.index += 1
            frame = Frame()
            self.references.append(weakref.ref(frame))
            return frame

    frames = Frames()
    assert runtime._decoded_frame_count(frames, 48_000, 2) == 96_000
    assert frames.max_live <= 2


@pytest.mark.parametrize(
    ("format_name", "codec_name", "video_streams"),
    [
        ("ogg", "opus", []),
        ("webm", "flac", []),
        ("webm", "opus", [object()]),
    ],
)
def test_unknown_duration_probe_does_not_decode_outside_audio_webm_fallback(
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    codec_name: str,
    video_streams: list[object],
) -> None:
    import dinkster_values.audio_lazy as runtime

    stream = SimpleNamespace(
        duration=None,
        time_base=None,
        codec_context=SimpleNamespace(
            sample_rate=48_000,
            channels=2,
            name=codec_name,
            layout=SimpleNamespace(name="stereo"),
        ),
    )

    class Container:
        duration = None
        format = SimpleNamespace(name=format_name)
        streams = SimpleNamespace(audio=[stream], video=video_streams)

        def decode(self, *_args: object) -> object:
            raise AssertionError("ineligible unknown-duration input was decoded")

        def demux(self, *_args: object) -> object:
            raise AssertionError("ineligible unknown-duration input was demuxed")

        def close(self) -> None:
            pass

    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: Container())
    metadata = runtime.probe_audio(b"not-pcm")
    assert metadata["duration"] is None
    assert metadata["frames"] is None


def test_load_trim_gain_and_metadata_do_not_decode(source, monkeypatch: pytest.MonkeyPatch) -> None:
    import dinkster_values.audio_lazy as runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("unconsumed audio decoded")

    monkeypatch.setattr(runtime, "_decode_source_window", forbidden)
    asset, _ = source
    value = LoadAudio.execute(audio=asset, start_time=0.5, duration=1)["audio"]
    value = AdjustAudioVolume.execute(audio=value, gain_db=-6)["audio"]
    assert audio_meta(value)["shape"] == (1, 2, 8000)
    wire = encode_audio(value)
    assert len(wire) < 1024
    assert b"vault" not in wire
    assert audio_meta(decode_audio(wire))["duration"] == 1
    validate_audio_encoded(memoryview(wire), audio_meta(value))
    assert audio_meta(value)["asset_refs"] == [asset.to_wire()]
    assert audio_meta(value)["cost"] == {"ram": 0}
    with pytest.raises(ValueError, match="asset_refs"):
        validate_audio_encoded(memoryview(wire), {**audio_meta(value), "asset_refs": []})


def test_sample_windows_and_order_survive_portable_round_trip(source) -> None:
    asset, pcm = source
    value = audio_from_source(asset)
    value = append_audio_edit(value, {"trim": {"start_sample": 113, "sample_count": 4000}})
    value = append_audio_edit(value, {"gain": 0.5})
    value = append_audio_edit(value, {"trim": {"start_sample": 67, "sample_count": 1000}})
    before = encode_audio(value)
    rebound = bind_audio_value(decode_audio(before), asset.resolver)
    assert encode_audio(rebound) == before
    assert audio_fingerprint("comfy.AUDIO")(value) == audio_fingerprint("comfy.AUDIO")(rebound)
    np.testing.assert_array_equal(
        audio_window(rebound, 19, 311)["waveform"], pcm[..., 199:510] * 0.5
    )
    assert audio_window(rebound, 999, 900)["waveform"].shape == (1, 2, 1)
    assert audio_window(rebound, 1000, 1)["waveform"].shape == (1, 2, 0)


def test_audio_window_errors_are_public_value_errors() -> None:
    import dinkster_values

    assert dinkster_values.AudioRangeError is AudioRangeError
    assert dinkster_values.AudioSourceUnavailableError is AudioSourceUnavailableError
    assert issubclass(AudioRangeError, ValueError)
    assert issubclass(AudioSourceUnavailableError, ValueError)


@pytest.mark.parametrize("batch_index", [1, 2**64])
@pytest.mark.parametrize("sample_count", [0, 1])
def test_audio_batch_range_error_precedes_source_decode(
    source, monkeypatch, batch_index, sample_count
) -> None:
    import dinkster_values.audio_lazy as runtime

    value = audio_from_source(source[0])
    monkeypatch.setattr(
        runtime, "_decode_source_window", lambda *args: pytest.fail("invalid batch decoded source")
    )
    with pytest.raises(AudioRangeError, match="batch_index is out of range"):
        audio_window(value, 0, sample_count, batch_index=batch_index)


def test_unbound_wire_source_has_distinct_window_error(source) -> None:
    value = decode_audio(encode_audio(audio_from_source(source[0])))
    with pytest.raises(AudioSourceUnavailableError, match="no local asset binding"):
        audio_window(value, 0, 1)
    with pytest.raises(AudioRangeError):
        audio_window(value, 0, 1, batch_index=1)
    for start, count in ((0, 0), (16000, 1), (16001, 1)):
        waveform = audio_window(value, start, count)["waveform"]
        assert waveform.shape == (1, 2, 0)
        assert waveform.dtype == np.float32


@pytest.mark.parametrize("argument", ["start_sample", "sample_count", "batch_index"])
@pytest.mark.parametrize("invalid", [-1, True, 1.0, "1"])
def test_invalid_window_arguments_remain_plain_value_errors(source, argument, invalid) -> None:
    value = decode_audio(encode_audio(audio_from_source(source[0])))
    kwargs = {"start_sample": 0, "sample_count": 1, "batch_index": 0, argument: invalid}
    with pytest.raises(ValueError, match=argument) as error:
        audio_window(value, **kwargs)
    assert type(error.value) is ValueError


def test_audio_window_allocation_error_remains_plain_value_error(source, monkeypatch) -> None:
    import dinkster_values.audio_lazy as runtime

    value = decode_audio(encode_audio(audio_from_source(source[0])))
    monkeypatch.setattr(runtime, "AUDIO_WINDOW_LIMIT", 4)
    with pytest.raises(ValueError, match="allocation budget") as error:
        audio_window(value, 0, 1)
    assert type(error.value) is ValueError


@pytest.mark.parametrize("failure", ["range", "unbound"])
def test_typed_window_error_closes_active_reader(source, inline_encoded_audio, failure) -> None:
    value = audio_from_source(inline_encoded_audio[0])
    end = effective_audio_facts(value)["frames"]
    child = decode_audio(encode_audio(audio_from_source(source[0])))
    value = append_audio_edit(value, {"concat": [child]})
    with AudioWindowReader(value) as reader:
        reader.read(0, 1)
        active = next(iter(reader._sources.values()))
        frames = active.frames
        assert isinstance(frames, GeneratorType) and frames.gi_frame is not None
        assert not active.cache.closed
        expected = AudioRangeError if failure == "range" else AudioSourceUnavailableError
        with pytest.raises(expected):
            reader.read(end, 1, batch_index=1 if failure == "range" else None)
        assert frames.gi_frame is None
        assert active.cache.closed
        assert not reader._sources
        with pytest.raises(ValueError, match="reader is closed") as error:
            reader.read(0, 1)
        assert type(error.value) is ValueError


@pytest.mark.parametrize("failure", ["no_resolver", "missing", "integrity"])
def test_canonical_audio_binding_preserves_asset_errors(source, tmp_path, failure) -> None:
    asset, _ = source
    value = decode_audio(encode_audio(audio_from_source(asset)))

    class CorruptResolver:
        def resolve(self, digest: str) -> Path:
            return tmp_path / "corrupt.pcm"

    if failure == "integrity":
        (tmp_path / "corrupt.pcm").write_bytes(b"incorrect content")
        resolver = CorruptResolver()
    else:
        resolver = AssetVault(tmp_path / "empty") if failure == "missing" else None
    bound = bind_audio_value(value, resolver)
    assert isinstance(bound["source"], AssetRef)
    expected = AssetIntegrityError if failure == "integrity" else AssetError
    with pytest.raises(expected) as error:
        audio_window(bound, 0, 1)
    assert type(error.value) is expected
    assert not isinstance(error.value, (AudioRangeError, AudioSourceUnavailableError))


@pytest.mark.parametrize("target_rate", [4000, 11025, 48000])
def test_resampled_windows_are_exact_slices_not_restarted_filters(source, target_rate) -> None:
    asset, _ = source
    value = ResampleAudio.execute(audio=audio_from_source(asset), sample_rate=target_rate)["audio"]
    full = audio_window(value, 0, effective_audio_facts(value)["frames"])["waveform"]
    windows = [
        audio_window(value, start, min(311, full.shape[-1] - start))["waveform"]
        for start in range(0, full.shape[-1], 311)
    ]
    np.testing.assert_array_equal(np.concatenate(windows, axis=-1), full)


def test_small_int16_pcm_preserves_storage_and_float_consumers() -> None:
    pcm = np.array([-32768, -1, 0, 1, 32767, 10, 20, 30], dtype=np.int16).reshape(1, 1, 8)
    pcm = np.repeat(pcm, 2, axis=1)
    value = coerce_audio({"waveform": pcm, "sample_rate": 8000})
    assert cast(dict[str, Any], value["source"])["pcm"] is pcm
    assert effective_audio_facts(value)["codec"] == "pcm_s16le"
    wire = encode_audio(value)
    decoded = cast(Any, decode_audio(wire))
    assert decoded["source"]["pcm"].dtype == np.int16
    np.testing.assert_array_equal(
        audio_window(decoded, 0, 8)["waveform"], pcm.astype(np.float32) / 32768
    )
    assert audio_meta(decoded)["asset_refs"] == []
    assert audio_meta(decoded)["cost"] == {"ram": pcm.nbytes}
    validate_audio_encoded(wire, audio_meta(decoded))


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("residency", ["ram", "ram@receiver", "vram:cuda:0@producer"])
def test_audio_admission_separates_semantics_from_local_storage(source, inline, residency) -> None:
    asset, pcm = source
    value = (
        coerce_audio({"waveform": pcm[..., :8], "sample_rate": 8000})
        if inline
        else audio_from_source(asset)
    )
    wire = encode_audio(value)
    metadata = dict(audio_meta(value))
    for retained_bytes in (len(wire), pcm.nbytes):
        assert retained_bytes != cast(dict[str, int], metadata["cost"])["ram"]
        received = {**metadata, "cost": {residency: retained_bytes}}
        validate_audio_encoded(memoryview(wire), received)
        with pytest.raises(ValueError, match="sample_rate"):
            validate_audio_encoded(wire, {**received, "sample_rate": 1})
        with pytest.raises(ValueError, match="shape"):
            validate_audio_encoded(wire, {**received, "shape": (1, 2, 1)})
        with pytest.raises(ValueError):
            validate_audio_encoded(wire[:-1], received)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64, np.int32])
def test_audio_registration_canonicalizes_storage_before_metadata(dtype) -> None:
    pcm = np.arange(16, dtype=dtype).reshape(1, 2, 8)
    registry = TypeRegistry()
    register_audio_value_type(registry, "comfy.AUDIO")
    value = registry.wrap("comfy.AUDIO", {"waveform": pcm, "sample_rate": 8000})
    expected = pcm if dtype in (np.int16, np.float32) else pcm.astype(np.float32)
    encoded = encode_audio(value.resolve())
    decoded = cast(Any, decode_audio(encoded))
    assert decoded["source"]["pcm"].dtype == expected.dtype
    np.testing.assert_array_equal(decoded["source"]["pcm"], expected)
    assert value.meta.get("cost") == {"ram": expected.nbytes}
    validate_audio_encoded(encoded, value.meta.entries)
    samples = expected.astype(np.float32) / 32768 if dtype == np.int16 else expected
    np.testing.assert_array_equal(audio_window(decoded, 0, 8)["waveform"], samples)
    assert render_audio_wav({"waveform": pcm, "sample_rate": 8000}) == render_audio_wav(
        {"waveform": samples, "sample_rate": 8000}
    )


@pytest.mark.parametrize("samples", [10, 100_000])
def test_pcm_publication_preserves_channel_layout(tmp_path, monkeypatch, samples) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    value = bind_audio_value(
        {"waveform": np.zeros((1, 3, samples), np.float32), "sample_rate": 48000, "layout": "3.0"}
    )
    assert effective_audio_facts(value)["layout"] == "3.0"
    decoded = bind_audio_value(decode_audio(encode_audio(value)), AssetVault(tmp_path))
    assert effective_audio_facts(decoded)["layout"] == "3.0"
    np.testing.assert_array_equal(audio_window(decoded, 0, 10)["waveform"], 0)


def test_pcm_layout_declaration_does_not_override_source_facts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    value = bind_audio_value(
        {"waveform": np.zeros((1, 3, 100_000), np.float32), "sample_rate": 48000, "layout": "3.0"}
    )
    probe = {**cast(dict[str, Any], value["probe"]), "codec": "pcm_f32le"}
    with pytest.raises(ValueError, match="probe does not match"):
        audio_window({**value, "probe": probe}, 0, 10)


def test_large_batched_pcm_publishes_without_losing_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = AssetVault(tmp_path / "vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    pcm = np.arange(100_000, dtype=np.float32).reshape(2, 2, 25_000) / 100_000
    value = bind_audio_value({"waveform": pcm, "sample_rate": 8000}, vault)
    assert effective_audio_facts(value)["batch"] == 2
    assert len(encode_audio(value)) < 1024
    decoded = bind_audio_value(decode_audio(encode_audio(value)), vault)
    np.testing.assert_array_equal(
        audio_window(decoded, 123, 41, batch_index=1)["waveform"], pcm[1:2, :, 123:164]
    )


def test_legacy_persisted_v1_and_oversized_inline_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcm = np.zeros((1, 2, 64), dtype=np.float32)
    buffer = io.BytesIO()
    np.save(buffer, pcm, allow_pickle=False)
    wire = struct.pack("<Q", 48000) + buffer.getvalue()
    np.testing.assert_array_equal(cast(Any, decode_audio(wire))["waveform"], pcm)
    validate_audio_encoded(memoryview(wire), {"sample_rate": 48000, "shape": pcm.shape})
    with pytest.raises(ValueError, match="256 KiB"):
        encode_audio(
            {
                "waveform": np.zeros((1, 1, AUDIO_INLINE_LIMIT // 4 + 1), np.float32),
                "sample_rate": 48000,
            }
        )
    forged = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        forged, {"descr": "<f4", "fortran_order": False, "shape": (1, 2, 2**50)}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed wire reached numpy allocation")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="payload length"):
        decode_audio(struct.pack("<Q", 48000) + forged.getvalue())


@pytest.mark.parametrize(
    "mutation",
    ["path", "negative", "nan", "unknown_edit", "matrix", "duration", "trailing", "header"],
)
def test_malformed_v2_wire_refused_before_pcm_allocation(source, monkeypatch, mutation) -> None:
    asset, _ = source
    value = audio_from_source(asset)
    wire = encode_audio(value)
    offset = len(b"DINKSTER-AUDIO\x02")
    header = json.loads(wire[offset + 4 :])
    if mutation == "path":
        header["source"]["asset"]["path"] = "/private/producer.wav"
    elif mutation == "negative":
        header["edits"] = [{"trim": {"start_sample": -1, "sample_count": 1}}]
    elif mutation == "nan":
        header["edits"] = [{"gain": float("nan")}]
    elif mutation == "unknown_edit":
        header["edits"] = [{"pitch": 2}]
    elif mutation == "matrix":
        header["edits"] = [{"channel_map": {"matrix": [[1]], "layout": "mono"}}]
    elif mutation == "duration":
        header["probe"]["duration"] = 1e20
    body = json.dumps(header).encode()
    wire = b"DINKSTER-AUDIO\x02" + struct.pack("<I", len(body)) + body
    if mutation == "trailing":
        wire += b"extra"
    elif mutation == "header":
        wire = b"DINKSTER-AUDIO\x02" + struct.pack("<I", 2**30)

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed descriptor allocated PCM")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError):
        validate_audio_encoded(memoryview(wire), {})
    with pytest.raises(ValueError):
        decode_audio(wire)


def test_window_allocation_guard_fires_before_source_decode(source, monkeypatch) -> None:
    import dinkster_values.audio_lazy as runtime

    value = audio_from_source(source[0])
    value["probe"] = {
        **cast(dict, value["probe"]),
        "frames": 48000 * 7200,
        "duration": 48000 * 7200 / 8000,
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized consumer reached decoder")

    monkeypatch.setattr(runtime, "_decode_source_window", forbidden)
    with pytest.raises(ValueError, match="allocation budget"):
        audio_window(value, 0, 48000 * 7200)


def test_surround_wav_volume_flac_and_channel_ops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, snapshot = _mount(tmp_path, monkeypatch)
    waveform = np.zeros((1, 6, 24000), dtype=np.float32)
    frame_samples = 2400
    for frame in range(10):
        channel = 2 + frame % 4
        amplitude = (-1 if frame % 2 else 1) * (frame + 1) / 20
        waveform[0, channel, frame * frame_samples : (frame + 1) * frame_samples] = amplitude
    # libav assigns the conventional layout to the six discrete interleaved WAV channels.
    data = render_audio_wav({"waveform": waveform, "sample_rate": 48000})
    vault = AssetVault(tmp_path / "wav-vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    original = audio_from_source(AssetRef(digest, "surround.wav", len(data), resolver=vault))
    assert effective_audio_facts(original)["channels"] == 6
    scaled = AdjustAudioVolume.execute(audio=original, gain_db=-6.020599913279624)["audio"]
    saved = cast(list[AssetRef], SaveAudio.execute(audio=scaled)["audios"])[0]
    loaded = LoadAudio.execute(audio=_bound(saved, snapshot))["audio"]
    assert effective_audio_facts(loaded)["channels"] == 6
    expected = audio_window(original, 0, 24000)["waveform"] * 0.5
    actual = audio_window(loaded, 0, 24000)["waveform"]
    assert float(np.abs(actual - expected).max()) <= 1 / 32768
    for node in (FadeAudio, EqualizeAudio):
        assert effective_audio_facts(node.execute(audio=loaded)["audio"])["channels"] == 6
    assert (
        effective_audio_facts(ConcatAudio.execute(audio1=loaded, audio2=loaded)["audio"])[
            "channels"
        ]
        == 6
    )
    assert (
        effective_audio_facts(
            JoinAudioChannels.execute(audio_left=loaded, audio_right=loaded)["audio"]
        )["channels"]
        == 12
    )
    split = SplitAudioChannels.execute(audio=loaded, left_index=4, right_index=5)
    np.testing.assert_array_equal(audio_window(split["left"], 0, 24000)["waveform"], actual[:, 4:5])
    downmixed = DownmixAudio.execute(audio=loaded, layout="mono")["audio"]
    np.testing.assert_allclose(
        audio_window(downmixed, 0, 24000)["waveform"],
        actual.mean(axis=1, keepdims=True),
        atol=3e-8,
        rtol=0,
    )
    trimmed = TrimAudio.execute(audio=loaded, start=0.013, duration=0.101)["audio"]
    assert effective_audio_facts(trimmed)["channels"] == 6
    np.testing.assert_array_equal(audio_window(trimmed, 0, 4848)["waveform"], actual[..., 624:5472])
    resampled = ResampleAudio.execute(audio=loaded, sample_rate=32000)["audio"]
    assert effective_audio_facts(resampled)["sample_rate"] == 32000
    assert effective_audio_facts(resampled)["channels"] == 6
    assert effective_audio_facts(resampled)["frames"] == 16000
    resampled_window = audio_window(resampled, 0, 16000)["waveform"]
    from scipy.signal import resample_poly

    expected_resampled = resample_poly(actual, 2, 3, axis=-1)
    np.testing.assert_allclose(resampled_window, expected_resampled, atol=3e-8, rtol=0)
    quieter = AdjustAudioVolume.execute(audio=loaded, gain_db=-6.020599913279624)["audio"]
    merged = MergeAudio.execute(audio1=loaded, audio2=quieter, merge_method="subtract")["audio"]
    np.testing.assert_allclose(
        audio_window(merged, 0, 24000)["waveform"], actual * 0.5, atol=3e-8, rtol=0
    )
    envelope = ExtractAudioEnvelope.execute(
        audio=loaded,
        frames_per_second=20,
        band_low_hz=0,
        band_high_hz=4000,
        normalize=False,
    )
    assert envelope["frames"] == 10
    envelope_values = cast(list[float], envelope["envelope"])
    expected_envelope = []
    envelope_mono = actual[0].astype(np.float64).mean(axis=0)
    for frame in range(10):
        samples = envelope_mono[frame * frame_samples : (frame + 1) * frame_samples]
        spectrum = (2.0 / frame_samples) * np.abs(np.fft.rfft(samples))
        expected_envelope.append(float(np.max(spectrum[:200])))
    np.testing.assert_allclose(envelope_values, expected_envelope, atol=0, rtol=0)
    onsets = cast(Curve, AudioOnsets.execute(audio=loaded, frames_per_second=20)["curve"])
    expected_flux = []
    previous = None
    onset_mono = actual[0].mean(axis=0)
    for frame in range(10):
        samples = onset_mono[frame * frame_samples : (frame + 1) * frame_samples]
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(frame_samples), n=frame_samples))
        expected_flux.append(
            float(spectrum.sum())
            if previous is None
            else float(np.maximum(spectrum - previous, 0).sum())
        )
        previous = spectrum
    peak = max(expected_flux)
    expected_points = tuple((frame / 20, value / peak) for frame, value in enumerate(expected_flux))
    assert onsets.points == expected_points


def test_batched_wav_preview_selects_first_without_decoding_other_batches() -> None:
    first = np.array([[[0.0, 0.25, -0.5]]], dtype=np.float32)
    second = np.full_like(first, np.nan)
    batched = {"waveform": np.concatenate((first, second)), "sample_rate": 3}
    assert render_audio_wav(batched) == render_audio_wav({"waveform": first, "sample_rate": 3})
    assert render_audio_wav(decode_audio(encode_audio(batched))) == render_audio_wav(batched)


def test_window_wav_rendition_selects_exact_batch_and_time_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_values.audio_codec as codec

    waveform = np.array(
        [
            [[-1.0, -0.5, 0.0, 0.5]],
            [[0.25, 0.5, 0.75, 1.0]],
        ],
        dtype=np.float32,
    )
    audio = {"waveform": waveform, "sample_rate": 4}
    parameters = normalize_audio_window_request(
        {"batch": "01", "window": "0.25,0.50"}, audio_meta(audio)
    )
    assert parameters == {"batch": "1", "window": "0.25,0.5"}
    calls: list[tuple[int, int, int | None]] = []
    original = codec.audio_window

    def tracked(
        obj: object, start_sample: int, sample_count: int, *, batch_index: int | None = None
    ) -> dict[str, Any]:
        calls.append((start_sample, sample_count, batch_index))
        return original(obj, start_sample, sample_count, batch_index=batch_index)

    monkeypatch.setattr(codec, "audio_window", tracked)
    data = render_audio_window(audio, parameters)
    assert calls == [(1, 2, 1)]
    assert data[:12] == b"RIFF" + struct.pack("<I", 40) + b"WAVE"
    assert struct.unpack("<2h", data[44:]) == (16384, 24576)


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        (1, (-32768, 32767)),
        (2, (-32768, -32768, 32767, 32767)),
    ],
)
def test_window_wav_rendition_preserves_pcm16_endpoints(
    channels: int, expected: tuple[int, ...]
) -> None:
    waveform = np.tile(np.array([-32768, 32767], dtype=np.int16), (1, channels, 1))
    audio = {"waveform": waveform, "sample_rate": 2}
    parameters = normalize_audio_window_request({"batch": "0", "window": "0,1"}, audio_meta(audio))
    data = render_audio_window(audio, parameters)
    with wave.open(io.BytesIO(data), "rb") as decoded:
        assert decoded.getnchannels() == channels
        assert decoded.getframerate() == 2
        assert struct.unpack(f"<{channels * 2}h", decoded.readframes(2)) == expected


def test_window_wav_rendition_clips_stereo_samples_to_pcm16_endpoints() -> None:
    waveform = np.array([[[-1.5, -1.0, 1.0, 1.5], [1.5, 0.5, -0.5, -1.5]]], dtype=np.float32)
    audio = {"waveform": waveform, "sample_rate": 4}
    parameters = normalize_audio_window_request({"batch": "0", "window": "0,1"}, audio_meta(audio))
    data = render_audio_window(audio, parameters)
    with wave.open(io.BytesIO(data), "rb") as decoded:
        assert decoded.getnchannels() == 2
        assert decoded.getframerate() == 4
        assert struct.unpack("<8h", decoded.readframes(4)) == (
            -32768,
            32767,
            -32768,
            16384,
            32767,
            -16384,
            32767,
            -32768,
        )


def test_waveform_rendition_reads_only_each_column_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_values.audio_codec as codec

    waveform = np.array(
        [
            [[0.0] * 8],
            [[-1.0, -0.5, 0.0, 0.0, 0.5, 1.0, -1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    audio = {"waveform": waveform, "sample_rate": 8}
    calls: list[tuple[int, int, int | None]] = []

    class TrackingReader:
        def __init__(self, obj: object) -> None:
            assert obj is audio

        def __enter__(self) -> TrackingReader:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(
            self, start_sample: int, sample_count: int, *, batch_index: int | None = None
        ) -> dict[str, Any]:
            calls.append((start_sample, sample_count, batch_index))
            return audio_window(audio, start_sample, sample_count, batch_index=batch_index)

    monkeypatch.setattr(codec, "AudioWindowReader", TrackingReader)
    parameters = normalize_audio_waveform_request(
        {"batch": "1", "waveform": "4x5"}, audio_meta(audio)
    )
    with Image.open(io.BytesIO(render_audio_waveform(audio, parameters))) as image:
        pixels = np.asarray(image.convert("RGBA"))
    assert calls == [(0, 2, 1), (2, 2, 1), (4, 2, 1), (6, 2, 1)]
    background = (0x11, 0x18, 0x27, 0xFF)
    cyan = (0x22, 0xD3, 0xEE, 0xFF)
    center = (0x37, 0x41, 0x51, 0xFF)
    expected = np.full((5, 4, 4), background, dtype=np.uint8)
    expected[3:5, 0] = cyan
    expected[2, 1] = center
    expected[0:2, 2] = cyan
    expected[:, 3] = cyan
    np.testing.assert_array_equal(pixels, expected)


@pytest.mark.parametrize(
    ("normalizer", "parameters", "message"),
    [
        (normalize_audio_waveform_request, {"waveform": "0x10"}, "width"),
        (normalize_audio_waveform_request, {"waveform": "2048x512"}, "262144"),
        (normalize_audio_window_request, {"window": "0,31"}, "at most 30"),
        (normalize_audio_window_request, {"window": "0,1e2"}, "decimal notation"),
        (normalize_audio_window_request, {"batch": "2", "window": "0,1"}, "out of range"),
        (
            normalize_audio_waveform_request,
            {"waveform": f"{'1' * 5000}x1"},
            "WIDTHxHEIGHT",
        ),
        (normalize_audio_window_request, {"batch": "1" * 5000, "window": "0,1"}, "batch"),
        (
            normalize_audio_window_request,
            {"window": f"0,{'1' * 5000}"},
            "decimal notation",
        ),
    ],
)
def test_audio_rendition_selector_limits(normalizer, parameters, message) -> None:
    metadata = audio_meta({"waveform": np.zeros((2, 1, 16), np.float32), "sample_rate": 8})
    with pytest.raises(ValueError, match=message):
        normalizer({"batch": "0", **parameters}, metadata)


def test_downmix_accepts_discrete_channels_without_an_ffmpeg_layout() -> None:
    pcm = np.ones((1, 9, 16), dtype=np.float32)
    audio = {"waveform": pcm, "sample_rate": 8000}
    mono = DownmixAudio.execute(audio=audio, layout="mono")["audio"]
    np.testing.assert_array_equal(audio_window(mono, 0, 16)["waveform"], pcm[:, :1])
    with pytest.warns(UserWarning, match="distributing channels equally"):
        stereo = DownmixAudio.execute(audio=audio)["audio"]
    np.testing.assert_array_equal(audio_window(stereo, 0, 16)["waveform"], pcm[:, :2])


def test_onset_curve_tracks_impulses_and_silent_audio() -> None:
    waveform = np.zeros((1, 6, 8000), dtype=np.float32)
    waveform[..., 2100:2200] = 0.75
    waveform[..., 6100:6200] = 0.5
    value = AudioOnsets.execute(
        audio={"waveform": waveform, "sample_rate": 8000}, frames_per_second=20
    )["curve"]
    assert isinstance(value, Curve)
    curve = value
    from dinkster_nodes_media_io import register_media_types

    registry = TypeRegistry()
    register_media_types(registry)
    encoded = registry.spec("dinkster.curve").encode(curve)
    assert registry.spec("dinkster.curve").decode(encoded) == curve
    assert json.loads(encoded)["points"][0] == {"position": 0.0, "value": 0.0}
    assert max(curve.points, key=lambda point: point[1])[0] == 0.25
    assert curve.evaluate(0.75) > 0.5
    silent = AudioOnsets.execute(audio={"waveform": waveform * 0, "sample_rate": 8000})["curve"]
    assert isinstance(silent, Curve)
    assert all(point[1] == 0 for point in silent.points)


def test_long_onsets_respect_curve_point_budget() -> None:
    waveform = np.zeros((1, 1, 5000), dtype=np.float32)
    with pytest.warns(UserWarning, match="peak-aggregated"):
        value = AudioOnsets.execute(audio={"waveform": waveform, "sample_rate": 1})["curve"]
    assert isinstance(value, Curve)
    assert len(value.points) == 2500


def test_flac_seek_windows_are_sample_exact_and_decoder_bounded(
    source, tmp_path, monkeypatch
) -> None:
    import av

    asset, expected = source
    _, snapshot = _mount(tmp_path, monkeypatch)
    saved = cast(list[AssetRef], SaveAudio.execute(audio=audio_from_source(asset))["audios"])[0]
    value = LoadAudio.execute(audio=_bound(saved, snapshot))["audio"]
    opened = av.open
    decoded_samples: list[int] = []

    class CountedPacket:
        def __init__(self, packet):
            self.packet = packet

        def __getattr__(self, name):
            return getattr(self.packet, name)

        def decode(self):
            frames = self.packet.decode()
            decoded_samples.extend(frame.samples for frame in frames)
            return frames

    class CountedContainer:
        def __init__(self, container):
            self.container = container

        def __getattr__(self, name):
            return getattr(self.container, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.container.close()

        def demux(self, stream):
            for packet in self.container.demux(stream):
                yield CountedPacket(packet)

    monkeypatch.setattr(
        av, "open", lambda *args, **kwargs: CountedContainer(opened(*args, **kwargs))
    )
    for start, count in ((113, 251), (4001, 177), (15999, 1)):
        decoded_samples.clear()
        np.testing.assert_array_equal(
            audio_window(value, start, count)["waveform"], expected[..., start : start + count]
        )
        assert sum(decoded_samples) <= count + 2 * max(decoded_samples)
    assert coerce_audio(value).get("sample_rate") == 8000


@pytest.mark.parametrize("saver", [SaveAudioMP3, SaveAudioOpus])
def test_compressed_seek_windows_preserve_decoder_preroll(source, tmp_path, monkeypatch, saver):
    import dinkster_values.audio_lazy as runtime

    _, snapshot = _mount(tmp_path, monkeypatch)
    saved = saver.execute(audio=audio_from_source(source[0]))["audios"][0]
    value = LoadAudio.execute(audio=_bound(saved, snapshot))["audio"]
    facts = effective_audio_facts(value)
    full = audio_window(value, 0, facts["frames"])["waveform"]
    decode = runtime._decode_frames
    samples: list[int] = []

    def counted(*args):
        for frame in decode(*args):
            samples.append(frame.samples)
            yield frame

    monkeypatch.setattr(runtime, "_decode_frames", counted)
    for start in (113, facts["frames"] // 3, facts["frames"] - 2001):
        samples.clear()
        np.testing.assert_array_equal(
            audio_window(value, start, 2001)["waveform"], full[..., start : start + 2001]
        )
        assert sum(samples) <= 2001 + 2 * facts["sample_rate"] + 2 * max(samples)
    samples.clear()
    with AudioWindowReader(value) as reader:
        for start in range(0, facts["frames"], 311):
            np.testing.assert_array_equal(
                reader.read(start, 311)["waveform"], full[..., start : start + 311]
            )
    assert sum(samples) <= facts["frames"] + 2 * max(samples)


def test_resample_filter_budget_is_checked_before_scipy_allocation(source, monkeypatch) -> None:
    import scipy.signal

    value = append_audio_edit(audio_from_source(source[0]), {"resample": 1_000_000_007})

    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded filter allocation")

    monkeypatch.setattr(scipy.signal, "resample_poly", forbidden)
    with pytest.raises(ValueError, match="resample filter"):
        audio_window(value, 0, 1)


def test_resample_intermediate_is_bounded_before_scipy_allocation(monkeypatch) -> None:
    import scipy.signal

    value = append_audio_edit(
        {"waveform": np.zeros((1, 1, 10000), np.float32), "sample_rate": 1},
        {"resample": 1000},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded intermediate allocation")

    monkeypatch.setattr(scipy.signal, "resample_poly", forbidden)
    with pytest.raises(ValueError, match="resample intermediate"):
        audio_window(value, 0, 1)


def test_registry_audio_wire_admission_and_receiving_host_binding(source) -> None:
    asset, expected = source
    sender = TypeRegistry()
    receiver = TypeRegistry()
    register_audio_value_type(sender, "comfy.AUDIO", asset.resolver)
    register_audio_value_type(receiver, "comfy.AUDIO", asset.resolver)
    value = sender.wrap("comfy.AUDIO", audio_from_source(asset))
    wire = sender.spec("comfy.AUDIO").encode(value.resolve())
    validator = receiver.spec("comfy.AUDIO").validate_encoded_buffer
    assert validator is not None
    validator(memoryview(wire), value.meta.entries)
    received = receiver.spec("comfy.AUDIO").decode(wire)
    np.testing.assert_array_equal(
        audio_window(received, 712, 29)["waveform"], expected[..., 712:741]
    )


@pytest.mark.parametrize("concat", [False, True])
def test_lazy_audio_chain_matches_across_worker_shared_memory(source, concat) -> None:
    from dinkster_nodes_media_io import MEDIA_IO_NODES, register_media_types
    from dinkster_protocol import Invocation
    from dinkster_schema import build_node_types
    from dinkster_values import register_core_types
    from dinkster_workers import BoundaryDiagnostic, InProcessWorker, IsolatedWorker

    asset, pcm = source
    value = audio_from_source(asset)
    if concat:
        value = append_audio_edit(value, {"trim": {"start_sample": 0, "sample_count": 1100}})
        value = append_audio_edit(value, {"concat": [{"waveform": pcm, "sample_rate": 8000}]})

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_audio_value_type(registry, "comfy.AUDIO", asset.resolver)
        register_media_types(registry)

        async def run(worker: InProcessWorker | IsolatedWorker) -> object:
            result = await worker.invoke(
                Invocation(
                    invocation_id="trim",
                    node_id="trim",
                    node_type="dinkster.audio.trim",
                    inputs={
                        "audio": registry.wrap("comfy.AUDIO", value),
                        "start": registry.wrap("core.float", 0.113),
                        "duration": registry.wrap("core.float", 0.751),
                    },
                    effective_schema=worker.schemas["dinkster.audio.trim"],
                )
            )
            assert result.error is None
            assert result.outputs is not None
            return result.outputs["audio"].resolve()

        expected = await run(InProcessWorker(build_node_types(MEDIA_IO_NODES), registry))
        diagnostics: list[BoundaryDiagnostic] = []
        manifest = (
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-media-io/dinkster-pack.toml"
        )
        worker = IsolatedWorker(
            manifest, registry, shm_threshold=64, on_diagnostic=diagnostics.append
        )
        await worker.start()
        try:
            actual = await run(worker)
            assert encode_audio(actual) == encode_audio(expected)
            np.testing.assert_array_equal(
                audio_window(actual, 77, 317)["waveform"],
                audio_window(expected, 77, 317)["waveform"],
            )
            assert [
                crossing.transport
                for diagnostic in diagnostics
                for crossing in diagnostic.inputs
                if crossing.type_id == "comfy.AUDIO"
            ] == ["shm"]
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_two_hour_flac_trim_save_rss(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "tools/audio_conformance.py", "--rss", "--directory", str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["rss_growth_bytes"] < 64 * 1024 * 1024
    assert report["saved_frames"] == 480000
    assert report["source_frames"] == 345600000


def _audio_header(wire: bytes) -> tuple[dict[str, Any], bytes]:
    prefix = len(b"DINKSTER-AUDIO\x02")
    length = struct.unpack_from("<I", wire, prefix)[0]
    offset = prefix + 4
    return json.loads(wire[offset : offset + length]), wire[offset + length :]


def _audio_wire(header: dict[str, Any], payload: bytes = b"") -> bytes:
    body = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return b"DINKSTER-AUDIO\x02" + struct.pack("<I", len(body)) + body + payload


@pytest.fixture(params=["flac", "matroska"])
def inline_encoded_audio(request):
    import av

    pcm = np.arange(2 * 8000, dtype=np.int16).reshape(2, 8000)
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=request.param) as container:
        video = None
        if request.param == "matroska":
            video = cast(av.VideoStream, container.add_stream("ffv1", rate=1))
            video.width = video.height = 16
            video.pix_fmt = "yuv420p"
        audio = cast(av.AudioStream, container.add_stream("flac", rate=8000))
        audio.layout = "stereo"
        if video is not None:
            image = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")
            image.pts = 0
            for packet in video.encode(image):
                container.mux(packet)
            for packet in video.encode(None):
                container.mux(packet)
        frame = av.AudioFrame.from_ndarray(pcm.T.reshape(1, -1), format="s16", layout="stereo")
        frame.sample_rate = 8000
        frame.pts = 0
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)
    return buffer.getvalue(), pcm[np.newaxis].astype(np.float32) / 32768


def test_inline_encoded_audio_probes_without_decode_and_admits_without_probe(
    inline_encoded_audio, monkeypatch
) -> None:
    import av
    import dinkster_values.audio_lazy as runtime

    data, expected = inline_encoded_audio
    opened = av.open

    class ProbeOnly:
        def __init__(self, container):
            self.container = container

        def __getattr__(self, name):
            assert name not in ("decode", "demux"), "probing decoded media"
            return getattr(self.container, name)

        def close(self):
            self.container.close()

    with monkeypatch.context() as patch:
        patch.setattr(av, "open", lambda *a, **kw: ProbeOnly(opened(*a, **kw)))
        value = audio_from_source(data)
    assert value["source"] is data
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "probe_audio", lambda *a: pytest.fail("codec probed media"))
        patch.setattr(av, "open", lambda *a, **kw: pytest.fail("codec opened media"))
        patch.setattr(np, "load", lambda *a, **kw: pytest.fail("codec allocated PCM"))
        wire = encode_audio(value)
        header, payload = _audio_header(wire)
        assert header["source"] == {"inline_encoded": len(data)}
        assert payload == data
        assert audio_encoded_meta(wire) == audio_meta(value)
        assert audio_meta(value)["cost"] == {"ram": len(data)}
        validate_audio_encoded(memoryview(wire), audio_meta(value))
        decoded = decode_audio(wire)
        assert encode_audio(decoded) == wire
    np.testing.assert_array_equal(
        audio_window(decoded, 113, 411)["waveform"], expected[..., 113:524]
    )


@pytest.mark.parametrize("field", ["sample_rate", "layout"])
def test_inline_encoded_source_rechecks_real_probe(inline_encoded_audio, field) -> None:
    value = audio_from_source(inline_encoded_audio[0])
    probe = cast(dict[str, Any], value["probe"])
    if field == "sample_rate":
        probe[field] *= 2
        probe["duration"] = probe["frames"] / probe[field]
    else:
        probe[field] = "2c"
    decoded = decode_audio(encode_audio(value))
    with pytest.raises(ValueError, match="probe does not match"):
        audio_window(decoded, 0, 1)


@pytest.mark.parametrize("inline", [False, True])
def test_existing_v2_header_and_payload_remain_identical(source, inline) -> None:
    asset, pcm = source
    value = (
        coerce_audio({"waveform": pcm, "sample_rate": 8000}) if inline else audio_from_source(asset)
    )
    value = append_audio_edit(value, {"gain": 0.5})
    payload = b""
    if inline:
        buffer = io.BytesIO()
        np.save(buffer, pcm, allow_pickle=False)
        payload = buffer.getvalue()
        wire_source = {"inline_pcm": len(payload)}
    else:
        wire_source = {"asset": asset.to_wire()}
    expected = _audio_wire(
        {"source": wire_source, "probe": value["probe"], "edits": [{"gain": 0.5}]}, payload
    )
    assert encode_audio(value) == expected
    assert encode_audio(bind_audio_value(decode_audio(expected), asset.resolver)) == expected


@pytest.mark.parametrize("target_rate", [4000, 11025, 48000])
def test_concat_edited_bct_windows_equal_eager_composition(target_rate) -> None:
    from scipy.signal import resample_poly

    pcm = np.arange(2 * 3 * 1000, dtype=np.int16).reshape(2, 3, 1000)
    root = append_audio_edit({"waveform": pcm, "sample_rate": 8000, "layout": "3.0"}, {"gain": 0.5})
    root = append_audio_edit(root, {"trim": {"start_sample": 73, "sample_count": 601}})
    child = append_audio_edit(
        {"waveform": -pcm, "sample_rate": 4000, "layout": "3.0"}, {"resample": 8000}
    )
    child = append_audio_edit(child, {"trim": {"start_sample": 113, "sample_count": 907}})
    tail = {"waveform": pcm[..., :311], "sample_rate": 8000, "layout": "3.0"}
    child = append_audio_edit(child, {"concat": [tail]})
    value = append_audio_edit(root, {"concat": [child]})
    assert value["probe"] == root["probe"]
    assert effective_audio_facts(value)["frames"] == 601 + 907 + 311
    expected = np.concatenate(
        (
            pcm[..., 73:674].astype(np.float32) / 32768 * 0.5,
            resample_poly(-pcm.astype(np.float32) / 32768, 2, 1, axis=-1)[..., 113:1020],
            pcm[..., :311].astype(np.float32) / 32768,
        ),
        axis=-1,
    )
    value = append_audio_edit(value, {"trim": {"start_sample": 37, "sample_count": 1501}})
    expected = expected[..., 37:1538]
    value = append_audio_edit(value, {"resample": target_rate})
    expected = resample_poly(expected, target_rate, 8000, axis=-1)
    value = decode_audio(encode_audio(value))
    np.testing.assert_array_equal(audio_window(value, 0, expected.shape[-1])["waveform"], expected)
    for start in range(0, expected.shape[-1], 197):
        np.testing.assert_array_equal(
            audio_window(value, start, 197)["waveform"], expected[..., start : start + 197]
        )
        np.testing.assert_array_equal(
            audio_window(value, start, 197, batch_index=1)["waveform"],
            expected[1:2, :, start : start + 197],
        )


def test_concat_selects_only_intersecting_sources(source, monkeypatch) -> None:
    import dinkster_values.audio_lazy as runtime

    root = audio_from_source(source[0])
    selected = coerce_audio({"waveform": source[1][..., :17], "sample_rate": 8000})
    value = append_audio_edit(root, {"concat": [selected, root]})
    with monkeypatch.context() as patch:
        patch.setattr(AssetRef, "open", lambda self: pytest.fail("unselected source opened"))
        patch.setattr(
            runtime, "_decode_source_window", lambda *a: pytest.fail("unselected source decoded")
        )
        np.testing.assert_array_equal(
            audio_window(value, 16000, 17)["waveform"], source[1][..., :17]
        )
        assert audio_window(value, 32017, 5)["waveform"].shape == (1, 2, 0)
        assert audio_window(value, 16000, 0)["waveform"].shape == (1, 2, 0)


@pytest.mark.parametrize("field", ["sample_rate", "channels", "layout", "batch"])
def test_concat_requires_matching_effective_facts(source, field) -> None:
    value = audio_from_source(source[0])
    child = copy.deepcopy(value)
    probe = cast(dict[str, Any], child["probe"])
    probe[field] = "2c" if field == "layout" else probe[field] * 2
    if field == "sample_rate":
        probe["duration"] = probe["frames"] / probe[field]
    with pytest.raises(ValueError, match="matching"):
        append_audio_edit(value, {"concat": [child]})


def test_concat_unknown_duration_requires_trim_on_root_and_child(source) -> None:
    value = audio_from_source(source[0])
    unknown = {**value, "probe": {**cast(dict, value["probe"]), "frames": None, "duration": None}}
    for root, child in ((unknown, value), (value, unknown)):
        with pytest.raises(ValueError, match="known frames"):
            append_audio_edit(root, {"concat": [child]})
    trimmed = append_audio_edit(unknown, {"trim": {"start_sample": 5, "sample_count": 17}})
    assert effective_audio_facts(append_audio_edit(trimmed, {"concat": [trimmed]}))["frames"] == 34


def test_nested_inline_preorder_metadata_and_asset_binding(
    source, inline_encoded_audio, tmp_path, monkeypatch
) -> None:
    asset, expected = source
    data, encoded_pcm = inline_encoded_audio
    inline = audio_from_source(data)
    child = append_audio_edit(inline, {"concat": [audio_from_source(asset)]})
    value = append_audio_edit(
        {"waveform": expected[..., :11], "sample_rate": 8000},
        {"concat": [child, audio_from_source(asset)]},
    )
    wire = encode_audio(value)
    header, payload = _audio_header(wire)
    pcm_size = header["source"]["inline_pcm"]
    np.testing.assert_array_equal(np.load(io.BytesIO(payload[:pcm_size])), expected[..., :11])
    assert payload[pcm_size:] == data
    meta = audio_meta(value)
    assert meta["asset_refs"] == [asset.to_wire()]
    assert meta["cost"] == {"ram": expected[..., :11].nbytes + len(data)}
    assert audio_encoded_meta(memoryview(wire)) == meta
    validate_audio_encoded(
        wire, {**meta, "cost": {"ram@consumer": expected[..., :11].nbytes + len(data)}}
    )
    destination = AssetVault(tmp_path / "consumer")
    for ref in cast(list[dict], meta["asset_refs"]):
        with asset.open() as file, destination.writer(ref["digest"]) as writer:
            writer.write(file.read())
            writer.commit()
    seen = []

    def factory(ref):
        seen.append(ref)
        return AssetRef.from_wire(ref, destination)

    with monkeypatch.context() as patch:
        patch.setattr(AssetRef, "open", lambda self: pytest.fail("binding opened source"))
        rebound = bind_audio_sources(decode_audio(wire), factory)
        assert seen == [asset.to_wire(), asset.to_wire()]
        assert encode_audio(rebound) == wire
    np.testing.assert_array_equal(
        audio_window(rebound, 11 + encoded_pcm.shape[-1] - 3, 9)["waveform"],
        np.concatenate((encoded_pcm[..., -3:], expected[..., :6]), axis=-1),
    )


def test_nested_large_pcm_publication_preserves_child_edits_and_layout(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "producer"))
    pcm = np.arange(2 * 3 * 25000, dtype=np.float32).reshape(2, 3, 25000) / 150000
    child = append_audio_edit(
        {"waveform": pcm, "sample_rate": 8000, "layout": "3.0"},
        {"trim": {"start_sample": 123, "sample_count": 47}},
    )
    child = append_audio_edit(child, {"gain": 0.5})
    root = {"waveform": pcm[..., :5], "sample_rate": 8000, "layout": "3.0"}
    nested = append_audio_edit(root, {"concat": [child]})
    value = bind_audio_value(append_audio_edit(root, {"concat": [nested]}))
    assert audio_meta(value)["cost"] == {"ram": 2 * pcm[..., :5].nbytes}
    refs = cast(list[dict], audio_meta(value)["asset_refs"])
    assert len(refs) == 1
    receiver = AssetVault(tmp_path / "receiver")
    with (
        AssetRef.from_wire(refs[0], AssetVault(tmp_path / "producer")).open() as file,
        receiver.writer(refs[0]["digest"]) as writer,
    ):
        writer.write(file.read())
        writer.commit()
    decoded = bind_audio_value(decode_audio(encode_audio(value)), receiver)
    np.testing.assert_array_equal(
        audio_window(decoded, 10, 47, batch_index=1)["waveform"], pcm[1:2, :, 123:170] * 0.5
    )
    assert effective_audio_facts(decoded)["layout"] == "3.0"


@pytest.mark.parametrize(
    "mutation",
    [
        "depth",
        "records",
        "edits",
        "short",
        "trailing",
        "negative",
        "boolean",
        "oversized",
        "child_fields",
        "child_source",
        "child_shape",
        "header",
    ],
)
def test_nested_wire_mutations_fail_before_allocation(source, monkeypatch, mutation) -> None:
    value = append_audio_edit(
        {"waveform": source[1][..., :11], "sample_rate": 8000},
        {"concat": [audio_from_source(source[0])]},
    )
    header, payload = _audio_header(encode_audio(value))
    child = header["edits"][0]["concat"][0]
    if mutation == "depth":
        for _ in range(16):
            child = {**child, "edits": [{"concat": [child]}]}
        header["edits"] = [{"concat": [child]}]
    elif mutation == "records":
        header["edits"] = [{"concat": [child] * 64}]
    elif mutation == "edits":
        header["edits"] += [{"gain": 1}] * 128
        child["edits"] = [{"gain": 1}] * 128
    elif mutation == "short":
        payload = payload[:-1]
    elif mutation == "trailing":
        payload += b"extra"
    elif mutation in ("negative", "boolean", "oversized"):
        child["source"] = {
            "inline_encoded": {
                "negative": -1,
                "boolean": True,
                "oversized": AUDIO_INLINE_LIMIT + 1,
            }[mutation]
        }
    elif mutation == "child_fields":
        child["waveform"] = [1, 2, 3]
    elif mutation == "child_source":
        child["source"]["asset"]["path"] = "/private/child.wav"
    elif mutation == "child_shape":
        child["source"] = header["source"]
        payload += payload
    else:
        child["probe"]["layout"] = "x" * (1024 * 1024)
    wire = _audio_wire(header, payload)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("malformed wire allocated PCM"))
    monkeypatch.setattr(AssetRef, "open", lambda self: pytest.fail("admission opened source"))
    for admit in (decode_audio, audio_encoded_meta):
        with pytest.raises(ValueError):
            admit(wire)


@pytest.mark.parametrize("limit", ["depth", "records", "edits"])
def test_audio_tree_exact_budgets_are_accepted(source, limit) -> None:
    leaf = audio_from_source(source[0])
    value = leaf
    if limit == "depth":
        for _ in range(15):
            value = append_audio_edit(leaf, {"concat": [value]})
    elif limit == "records":
        value = append_audio_edit(leaf, {"concat": [leaf] * 63})
    else:
        value = {**leaf, "edits": [{"gain": 1}] * 128}
        value = {
            **value,
            "edits": [*value["edits"], {"concat": [{**leaf, "edits": [{"gain": 1}] * 127}]}],
        }
    wire = encode_audio(value)
    validate_audio_encoded(wire, audio_meta(value))
    assert encode_audio(decode_audio(wire)) == wire


def test_inline_encoded_size_limit_and_opaque_admission(source, monkeypatch) -> None:
    import dinkster_values.audio_lazy as runtime

    template = audio_from_source(source[0])
    monkeypatch.setattr(runtime, "probe_audio", lambda *a: pytest.fail("admission probed media"))
    value = {**template, "source": b"x" * AUDIO_INLINE_LIMIT}
    wire = encode_audio(value)
    assert encode_audio(decode_audio(wire)) == wire
    assert audio_encoded_meta(wire)["cost"] == {"ram": AUDIO_INLINE_LIMIT}
    oversized = b"x" * (AUDIO_INLINE_LIMIT + 1)
    with pytest.raises(ValueError, match="256 KiB"):
        audio_from_source(oversized)
    with pytest.raises(ValueError, match="256 KiB"):
        encode_audio({**value, "source": oversized})


def test_concat_assets_are_deduplicated_in_first_encounter_order(source, monkeypatch) -> None:
    asset = source[0]
    root = audio_from_source(asset)
    second_ref = {**asset.to_wire(), "digest": "blake3:" + "1" * 64}
    second = {**root, "source": second_ref}
    alias = {**root, "source": {**asset.to_wire(), "name": "alias.pcm"}}
    value = append_audio_edit(
        second, {"concat": [append_audio_edit(root, {"concat": [second, alias]})]}
    )
    monkeypatch.setattr(AssetRef, "open", lambda self: pytest.fail("metadata opened asset"))
    assert audio_meta(value)["asset_refs"] == [second_ref, asset.to_wire()]
    assert audio_encoded_meta(encode_audio(value)) == audio_meta(value)
    unbound = append_audio_edit(
        root, {"concat": [{**root, "source": AssetRef.from_wire(asset.to_wire())}]}
    )
    bound = bind_audio_value(unbound, asset.resolver)
    child = cast(Any, bound["edits"])[0]["concat"][0]
    assert child["source"].resolver is asset.resolver


def test_repeated_concat_edits_and_channel_maps_preserve_order() -> None:
    pcm = np.arange(24, dtype=np.float32).reshape(1, 2, 12)
    root = {"waveform": pcm, "sample_rate": 8000}
    child = {"waveform": pcm[..., :3], "sample_rate": 8000}
    value = append_audio_edit(root, {"concat": [child]})
    value = append_audio_edit(value, {"trim": {"start_sample": 10, "sample_count": 4}})
    value = append_audio_edit(value, {"concat": [root, child]})
    value = append_audio_edit(value, {"channel_map": {"matrix": [[0, 1]], "layout": "mono"}})
    value = append_audio_edit(
        value, {"concat": [{"waveform": pcm[:, :1, :2], "sample_rate": 8000}]}
    )
    expected = np.concatenate(
        (pcm[:, 1:, 10:], pcm[:, 1:, :2], pcm[:, 1:], pcm[:, 1:, :3], pcm[:, :1, :2]), axis=-1
    )
    np.testing.assert_array_equal(
        audio_window(decode_audio(encode_audio(value)), 0, 99)["waveform"], expected
    )
