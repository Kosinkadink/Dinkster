"""Nested AUDIO uses its canonical wire and receiving-host source binding."""

from __future__ import annotations

import asyncio
import io
import sys
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, cast

import av
import numpy as np
import pytest
from aiohttp import web
from dinkster_api.v1 import ABSENT
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    digest_bytes,
    register_video_value_type,
)
from dinkster_assets.audio import bind_audio_value, register_audio_value_type
from dinkster_assets.value import bind_video_value
from dinkster_compat_comfy.video import _VideoValue
from dinkster_nodes_media_io.video_ops import DisassembleVideo
from dinkster_protocol import Invocation
from dinkster_values import (
    TypeRegistry,
    decode_video,
    edit_video,
    encode_video,
    video_from_source,
    video_meta,
)
from dinkster_values.audio_codec import (
    append_audio_edit,
    audio_from_source,
    audio_window,
    decode_audio,
    effective_audio_facts,
    encode_audio,
)
from dinkster_values.audio_lazy import LazyAudio
from dinkster_values.video_codec import validate_video_encoded
from dinkster_video import assemble_video, disassemble_video
from dinkster_workers import RemoteWorker
from dinkster_workers.devices import DeviceMap
from test_av_codec import FakeTensor
from test_remote import TOKEN, start_service, stop_service
from test_remote_asset_staging import ENDPOINT_TOKEN, AssetHost


def _encoded_video_audio(
    *,
    video_start: Fraction = Fraction(2),
    audio_start: Fraction = Fraction(2),
    codec: str = "pcm_s16le",
    samples: int = 96000,
    extra_audio: bool = False,
    seed: int = 0,
) -> tuple[bytes, np.ndarray]:
    pcm = np.random.default_rng(seed).integers(-16384, 16384, samples, dtype=np.int16)
    output = io.BytesIO()
    with av.open(output, "w", format="matroska" if codec == "libopus" else "mov") as opened:
        mux = cast(Any, opened)
        video = mux.add_stream("libx264", rate=10)
        video.width, video.height, video.pix_fmt = 16, 16, "yuv420p"
        video.codec_context.thread_count = 1
        video.codec_context.max_b_frames = 0
        audio_streams = [mux.add_stream(codec, rate=48000) for _ in range(1 + int(extra_audio))]
        for stream in audio_streams:
            stream.layout = "mono"
            stream.codec_context.thread_count = 1
        for index in range(samples // 4800):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = int(video_start * 10) + index, Fraction(1, 10)
            for packet in video.encode(frame):
                mux.mux(packet)
        for packet in video.encode():
            mux.mux(packet)
        for index, stream in enumerate(audio_streams):
            signal = pcm if index == len(audio_streams) - 1 else np.zeros_like(pcm)
            for start in range(0, samples, 1024):
                array = signal[None, start : start + 1024]
                frame = av.AudioFrame.from_ndarray(
                    array if codec == "pcm_s16le" else array.astype(np.float32) / 32768,
                    format="s16" if codec == "pcm_s16le" else "fltp",
                    layout="mono",
                )
                frame.sample_rate = 48000
                frame.pts, frame.time_base = int(audio_start * 48000) + start, Fraction(1, 48000)
                for packet in stream.encode(frame):
                    mux.mux(packet)
            for packet in stream.encode():
                mux.mux(packet)
    return output.getvalue(), pcm[None, None].astype(np.float32) / 32768


def _video_asset(data: bytes, vault: AssetVault) -> AssetRef:
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    return AssetRef(digest, "source.mov", len(data), resolver=vault)


@pytest.fixture
def decoded_audio_samples(monkeypatch, tmp_path):
    import dinkster_values.audio_lazy as lazy
    import dinkster_video.runtime as runtime

    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    samples = []
    decode = lazy._decode_frames

    def counted(*args):
        for frame in decode(*args):
            samples.append(frame.samples)
            yield frame

    monkeypatch.setattr(lazy, "_decode_frames", counted)
    monkeypatch.setattr(
        runtime, "_audio_frames", lambda *args: pytest.fail("audio extraction must not decode")
    )
    return samples


@pytest.fixture(params=["native", "comfy"])
def extract_audio(request, monkeypatch):
    if request.param == "native":
        return lambda video: DisassembleVideo.execute(video=video)["audio"]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=FakeTensor))
    monkeypatch.setitem(
        sys.modules,
        "comfy_api.latest",
        SimpleNamespace(Types=SimpleNamespace(VideoComponents=SimpleNamespace)),
    )
    return lambda video: cast(Any, _VideoValue(video).get_components()).audio


@pytest.mark.parametrize("concat", [False, True])
def test_extraction_callers_publish_inline_video_before_lazy_audio(
    tmp_path, monkeypatch, extract_audio, decoded_audio_samples, concat
) -> None:
    import dinkster_video.runtime as runtime

    first_data, first_pcm = _encoded_video_audio(
        samples=48000, extra_audio=True, audio_start=Fraction(9, 4), seed=1
    )
    second_data, second_pcm = _encoded_video_audio(
        samples=48000, extra_audio=True, audio_start=Fraction(9, 4), seed=2
    )
    assert max(len(first_data), len(second_data)) <= 256 * 1024
    first, second = (video_from_source(data) for data in (first_data, second_data))
    video = edit_video(first, {"concat": [second]}) if concat else first
    video = edit_video(video, {"trim": {"start_time": 0.75, "duration": 0.75}})
    assert bind_video_value(video)["source"] is first_data
    expected_sources = [first_data, second_data] if concat else [first_data]
    sources = []

    def published_audio(source, *, stream_index):
        assert isinstance(source, AssetRef)
        data = expected_sources[len(sources)]
        assert source.digest == digest_bytes(data)
        assert source.local_path().read_bytes() == data
        assert stream_index == 1
        sources.append(source)
        return audio_from_source(source, stream_index=stream_index)

    monkeypatch.setattr(runtime, "audio_from_source", published_audio)
    audio = extract_audio(video)
    assert isinstance(audio, LazyAudio)
    assert "waveform" not in audio and "pcm" not in audio
    assert len(sources) == len(expected_sources)
    assert audio["source"] is sources[0]
    assert first["source"] is first_data
    facts = effective_audio_facts(audio)
    original = audio_from_source(first_data, stream_index=1)
    assert audio["probe"] == original["probe"]
    assert (facts["stream_index"], facts["sample_rate"], facts["layout"]) == (
        1,
        48000,
        effective_audio_facts(original)["layout"],
    )
    expected = first_pcm[..., 24000:36000]
    if concat:
        expected = np.concatenate([expected, second_pcm[..., :12000]], axis=-1)
    assert facts["frames"] == expected.shape[-1]
    receiver = AssetVault(tmp_path / "receiver")
    for source in sources:
        _video_asset(source.local_path().read_bytes(), receiver)
    encoded = encode_audio(audio)
    assert len(encoded) < 4096
    rebound = bind_audio_value(decode_audio(encoded), receiver)
    rebound_source = rebound["source"]
    assert isinstance(rebound_source, AssetRef) and rebound_source.resolver is receiver
    assert decoded_audio_samples == []
    np.testing.assert_array_equal(
        audio_window(rebound, 0, expected.shape[-1])["waveform"], expected
    )
    waveform = audio["waveform"]
    if isinstance(waveform, FakeTensor):
        waveform = waveform.array
    np.testing.assert_array_equal(waveform, expected)


@pytest.mark.parametrize("components", [False, True])
def test_silent_video_extraction_needs_no_vault(
    monkeypatch, extract_audio, decoded_audio_samples, components
) -> None:
    from test_video_runtime import _source

    monkeypatch.delenv("DINKSTER_ASSET_VAULT")
    video = (
        assemble_video(np.zeros((2, 2, 2, 3), np.float32), fps=2)
        if components
        else video_from_source(_source())
    )
    audio = extract_audio(video)
    assert audio is None or audio is ABSENT
    assert disassemble_video(video)["audio"] is None
    assert decoded_audio_samples == []


def test_extraction_binding_preserves_standalone_silent_video_above_inline_limit(
    monkeypatch, decoded_audio_samples
) -> None:
    import dinkster_assets.value as assets
    from test_video_runtime import _source

    monkeypatch.delenv("DINKSTER_ASSET_VAULT")
    data = _source()
    video = video_from_source(data)
    monkeypatch.setattr(assets, "VIDEO_INLINE_LIMIT", len(data) - 1)
    with pytest.raises(AssetError, match="DINKSTER_ASSET_VAULT"):
        bind_video_value(video)
    assert bind_video_value(video, for_audio_extraction=True)["source"] is data
    assert DisassembleVideo.execute(video=video)["audio"] is ABSENT
    assert decoded_audio_samples == []


def test_inline_audio_extraction_requires_explicit_publication(
    monkeypatch, extract_audio, decoded_audio_samples
) -> None:
    monkeypatch.delenv("DINKSTER_ASSET_VAULT")
    data, _ = _encoded_video_audio(samples=48000)
    video = video_from_source(data)
    assert bind_video_value(video)["source"] is data
    with pytest.raises(ValueError, match="for_audio_extraction=True"):
        disassemble_video(video)
    with pytest.raises(AssetError, match="publishing VIDEO.*DINKSTER_ASSET_VAULT"):
        extract_audio(video)
    assert isinstance(audio_from_source(data), LazyAudio)
    assert decoded_audio_samples == []


@pytest.mark.parametrize("storage", ["asset", "inline"])
@pytest.mark.parametrize(
    "video_start,audio_start,trim_start,first,last",
    [
        (Fraction(2), Fraction(9, 4), 0.125, 0, 42000),
        (Fraction(5, 2), Fraction(2), 0.125, 30000, 78000),
        (Fraction(2), Fraction(2), 1.75, 84000, 96000),
        (Fraction(2), Fraction(1), 1.75, 96000, 96000),
    ],
)
def test_source_video_audio_is_lazy_and_aligned_to_stream_origins(
    tmp_path, storage, video_start, audio_start, trim_start, first, last, decoded_audio_samples
) -> None:
    data, pcm = _encoded_video_audio(video_start=video_start, audio_start=audio_start)
    vault = AssetVault(tmp_path)
    source = _video_asset(data, vault) if storage == "asset" else data
    assert len(data) <= 256 * 1024
    video = video_from_source(source)
    probe = cast(dict[str, Any], video["probe"])
    assert probe["start_time"] == video_start
    assert probe["audio"][0]["start_time"] == audio_start
    assert probe["audio"][0]["index"] == 1
    video = edit_video(video, {"trim": {"start_time": trim_start, "duration": 1}})
    video = bind_video_value(video, for_audio_extraction=True)
    audio = disassemble_video(video)["audio"]
    if first == last:
        assert audio is None
        assert decoded_audio_samples == []
        return
    assert isinstance(audio, LazyAudio)
    extracted_source = audio["source"]
    assert isinstance(extracted_source, AssetRef)
    assert extracted_source.digest == digest_bytes(data)
    assert extracted_source is video["source"]
    assert audio["probe"] == audio_from_source(source)["probe"]
    assert effective_audio_facts(audio)["frames"] == last - first
    rebound = bind_audio_value(decode_audio(encode_audio(audio)), vault)
    assert decoded_audio_samples == []
    np.testing.assert_array_equal(
        audio_window(rebound, 0, last - first)["waveform"], pcm[..., first:last]
    )
    assert decoded_audio_samples


@pytest.mark.parametrize("storage", ["asset", "inline"])
def test_source_video_audio_uses_audio_list_position(
    tmp_path, storage, decoded_audio_samples
) -> None:
    data, pcm = _encoded_video_audio(samples=48000, extra_audio=True)
    source = _video_asset(data, AssetVault(tmp_path)) if storage == "asset" else data
    assert len(data) <= 256 * 1024
    video = video_from_source(source)
    assert [stream["index"] for stream in cast(Any, video["probe"])["audio"]] == [1, 2]
    video = bind_video_value(video, for_audio_extraction=True)
    audio = disassemble_video(video)["audio"]
    assert isinstance(audio, LazyAudio)
    assert effective_audio_facts(audio)["stream_index"] == 1
    assert decoded_audio_samples == []
    np.testing.assert_array_equal(audio_window(audio, 113, 2001)["waveform"], pcm[..., 113:2114])


@pytest.mark.parametrize("storage", ["asset", "inline"])
@pytest.mark.parametrize("codec", ["aac", "libopus"])
def test_source_video_audio_preserves_compressed_preroll(
    tmp_path, storage, codec, decoded_audio_samples
) -> None:
    data, _ = _encoded_video_audio(codec=codec)
    source = _video_asset(data, AssetVault(tmp_path)) if storage == "asset" else data
    root = audio_from_source(source)
    full = audio_window(root, 0, effective_audio_facts(root)["frames"])["waveform"]
    decoded_audio_samples.clear()
    video = video_from_source(source)
    video = edit_video(video, {"trim": {"start_time": 1.123, "duration": 0.5}})
    video = bind_video_value(video, for_audio_extraction=True)
    audio = disassemble_video(video)["audio"]
    assert isinstance(audio, LazyAudio)
    assert audio["probe"] == root["probe"]
    probe = cast(dict[str, Any], video["probe"])
    shift = probe["start_time"] - probe["audio"][0]["start_time"]
    first = int(np.ceil((shift + Fraction(1123, 1000)) * 48000))
    assert decoded_audio_samples == []
    for start in (0, 113, 21999):
        decoded_audio_samples.clear()
        np.testing.assert_array_equal(
            audio_window(audio, start, 2001)["waveform"],
            full[..., first + start : first + start + 2001],
        )
        assert sum(decoded_audio_samples) <= 2001 + 96000 + 2 * max(decoded_audio_samples)


@pytest.mark.parametrize("storage", ["asset", "inline"])
def test_source_video_concat_audio_is_lazy_across_selected_boundary(
    tmp_path, storage, decoded_audio_samples
) -> None:
    vault = AssetVault(tmp_path)
    first_data, first_pcm = _encoded_video_audio(seed=1)
    second_data, second_pcm = _encoded_video_audio(seed=2)
    sources = [
        _video_asset(data, vault) if storage == "asset" else data
        for data in (first_data, second_data)
    ]
    first, second = (video_from_source(source) for source in sources)
    joined = edit_video(first, {"concat": [second]})
    joined = edit_video(joined, {"trim": {"start_time": 1.75, "duration": 0.5}})
    joined = bind_video_value(joined, for_audio_extraction=True)
    audio = disassemble_video(joined)["audio"]
    assert isinstance(audio, LazyAudio)
    assert audio["probe"] == audio_from_source(sources[0])["probe"]
    assert effective_audio_facts(audio)["frames"] == 24000
    rebound = bind_audio_value(decode_audio(encode_audio(audio)), vault)
    assert decoded_audio_samples == []
    expected = np.concatenate([first_pcm[..., 84000:], second_pcm[..., :12000]], axis=-1)
    np.testing.assert_array_equal(
        audio_window(rebound, 11000, 2001)["waveform"], expected[..., 11000:13001]
    )


def test_source_video_unknown_duration_uses_measured_video_end(
    tmp_path, decoded_audio_samples
) -> None:
    from dinkster_video.runtime import _extract_audio, _Segment

    data, pcm = _encoded_video_audio()
    source = _video_asset(data, AssetVault(tmp_path))
    video = video_from_source(source)
    audio = _extract_audio([_Segment(video, Fraction(1, 4), None)], Fraction(1, 2))
    assert isinstance(audio, LazyAudio)
    assert effective_audio_facts(audio)["frames"] == 24000
    assert decoded_audio_samples == []
    np.testing.assert_array_equal(audio_window(audio, 0, 24000)["waveform"], pcm[..., 12000:36000])


@pytest.mark.parametrize(
    "channels,samples,layout",
    [(1, 9_000_000, "mono"), (3, 48_000, "3.0"), (6, 48_000, "5.1"), (9, 1_000_000, "9c")],
)
def test_video_returns_lazy_audio_consumed_in_bounded_windows(
    tmp_path, monkeypatch, channels, samples, layout
) -> None:
    import dinkster_values.audio_lazy as lazy

    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    pcm = np.full((1, channels, samples), 16384, np.int16)
    value = bind_video_value(
        assemble_video(
            np.zeros((2, 2, 2, 3), np.float32),
            fps=2,
            audio={"waveform": pcm, "sample_rate": 48000, "layout": layout},
        )
    )
    audio = cast(dict[str, Any], value["components"])["audio"]
    assert isinstance(audio["source"], AssetRef)
    windows = []
    decode = lazy._decode_source_window

    def bounded_window(source, probe, start, count, batch_index=None):
        assert count <= 1024
        assert 0 <= start < 48000 and start + count <= 48000
        windows.append((start, count))
        return decode(source, probe, start, count, batch_index)

    monkeypatch.setattr(lazy, "_decode_source_window", bounded_window)
    with monkeypatch.context() as patch:
        patch.setattr(AssetRef, "open", lambda self: pytest.fail("unused audio source opened"))
        result = DisassembleVideo.execute(video=value)["audio"]
        assert isinstance(result, LazyAudio)
        assert result["source"] is audio["source"]
        facts = effective_audio_facts(result)
        assert facts["frames"] == 48000
        assert facts["layout"] == layout
        assert facts["sample_rate"] == 48000
        assert len(encode_audio(result)) < 4096
    assert windows == []
    for start in range(0, 48000, 1024):
        count = min(1024, 48000 - start)
        waveform = audio_window(result, start, count)["waveform"]
        assert waveform.shape == (1, channels, count)
        np.testing.assert_array_equal(waveform, 0.5)
    assert sum(count for _, count in windows) == 48000


@pytest.mark.parametrize("channels", [9, 10, 17])
@pytest.mark.parametrize("dtype", [np.int16, np.float32])
def test_discrete_component_audio_roundtrips_without_speaker_assignment(channels, dtype) -> None:
    samples = 2051
    pcm = np.arange(channels * samples, dtype=dtype).reshape(1, channels, samples)
    if dtype == np.float32:
        pcm /= 65536
    images = np.zeros((2, 2, 2, 3), np.float32)
    value = assemble_video(images, fps=2, audio={"waveform": pcm, "sample_rate": samples})
    value = decode_video(encode_video(value))
    result = disassemble_video(value)
    audio = cast(dict[str, Any], result["audio"])
    assert isinstance(audio, LazyAudio)
    expected = pcm.astype(np.float32) / 32768 if dtype == np.int16 else pcm
    np.testing.assert_array_equal(audio["waveform"], expected)
    np.testing.assert_array_equal(result["images"], images)
    assert audio["sample_rate"] == samples
    assert effective_audio_facts(audio)["layout"] == f"{channels}c"


@pytest.mark.parametrize("channels,layout", [(3, "3.0"), (6, "5.1"), (9, "9c")])
def test_video_trim_consumes_edited_asset_audio_exactly(
    tmp_path, monkeypatch, channels, layout
) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    pcm = np.arange(channels * 100_000, dtype=np.int16).reshape(1, channels, 100_000)
    audio = bind_audio_value({"waveform": pcm, "sample_rate": 48000, "layout": layout})
    assert isinstance(audio["source"], AssetRef)
    audio = append_audio_edit(audio, {"trim": {"start_sample": 113, "sample_count": 64000}})
    audio = append_audio_edit(audio, {"gain": 0.5})
    video = assemble_video(np.zeros((2, 2, 2, 3), np.float32), fps=2, audio=audio)
    video = edit_video(video, {"trim": {"start_time": 0.1, "duration": 0.5}})
    rebound = bind_video_value(decode_video(encode_video(video)), AssetVault(tmp_path))
    with monkeypatch.context() as patch:
        patch.setattr(AssetRef, "open", lambda self: pytest.fail("unused edited audio opened"))
        result = disassemble_video(rebound)["audio"]
        assert isinstance(result, LazyAudio)
        assert effective_audio_facts(result)["layout"] == layout
    np.testing.assert_array_equal(
        audio_window(result, 0, 24000)["waveform"], pcm[..., 4913:28913].astype(np.float32) / 65536
    )
    assert result["sample_rate"] == 48000


@pytest.mark.parametrize(
    "trims,first,last",
    [
        ([(Fraction(1, 96000), 1)], 1, 48000),
        ([(0.25, 1.5), (0.25, 0.5)], 24000, 48000),
        ([(1.5, 0.5)], 48000, 48000),
        ([(-0.5, 0.5)], 48000, 48000),
    ],
)
def test_lazy_component_audio_preserves_sample_rounding_edits_and_eof(trims, first, last) -> None:
    pcm = np.arange(9 * 48000, dtype=np.float32).reshape(1, 9, 48000) / 524288
    video = assemble_video(
        np.zeros((4, 2, 2, 3), np.float32),
        fps=2,
        audio={"waveform": pcm, "sample_rate": 48000},
    )
    for start, duration in trims:
        video = edit_video(video, {"trim": {"start_time": start, "duration": duration}})
    audio = disassemble_video(video)["audio"]
    if first == last:
        assert audio is None
    else:
        assert isinstance(audio, LazyAudio)
        np.testing.assert_array_equal(
            audio_window(audio, 0, last - first)["waveform"], pcm[..., first:last]
        )


def test_concat_trim_selects_component_audio_without_eager_concat() -> None:
    images = np.zeros((2, 2, 2, 3), np.float32)
    pcm = np.ones((1, 9, 48000), np.float32)
    silent = assemble_video(images, fps=2)
    audible = assemble_video(images, fps=2, audio={"waveform": pcm, "sample_rate": 48000})
    joined = edit_video(silent, {"concat": [audible]})
    with pytest.raises(ValueError, match="identical-codec concat requires encoded source clips"):
        disassemble_video(joined)
    selected = edit_video(joined, {"trim": {"start_time": 1, "duration": 1}})
    audio = disassemble_video(selected)["audio"]
    assert isinstance(audio, LazyAudio)
    np.testing.assert_array_equal(audio_window(audio, 0, 48000)["waveform"], pcm)


@pytest.mark.parametrize("samples", [100, 100_000])
def test_nested_audio_roundtrip_binds_without_decoding(tmp_path, monkeypatch, samples) -> None:
    pcm = np.linspace(-1, 1, samples, dtype=np.float32).reshape(1, 1, -1)
    images = np.ones((2, 4, 4, 4), np.float32)
    value = assemble_video(images, audio={"waveform": pcm, "sample_rate": 8000}, fps=8)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "producer"))
    producer = TypeRegistry()
    register_video_value_type(producer, "comfy.VIDEO")
    wrapped = producer.wrap("comfy.VIDEO", value)
    bound = wrapped.resolve()
    encoded = encode_video(bound)
    refs = cast(list[dict[str, Any]], wrapped.meta.get("asset_refs"))
    assert len(refs) == (1 if samples == 100_000 else 0)
    destination = AssetVault(tmp_path / "consumer")
    for ref in refs:
        data = AssetRef.from_wire(ref, AssetVault(tmp_path / "producer")).open()
        with data, destination.writer(ref["digest"]) as writer:
            writer.write(data.read())
            writer.commit()
    with monkeypatch.context() as patch:
        patch.setattr(AssetRef, "open", lambda self: pytest.fail("codec opened audio"))
        validate_video_encoded(memoryview(encoded), wrapped.meta.entries)
        qualified = DeviceMap(mapping={}, qualifier="video-consumer").value(wrapped)
        validate_video_encoded(memoryview(encoded), qualified.meta.entries)
        with pytest.raises(ValueError, match="cost"):
            validate_video_encoded(
                encoded, {**qualified.meta.entries, "cost": {"ram@video-consumer": 0}}
            )
        rebound = bind_video_value(decode_video(encoded), destination)
        assert encode_video(rebound) == encoded
        assert video_meta(rebound) == wrapped.meta.entries
    audio = cast(dict[str, Any], rebound["components"])["audio"]
    np.testing.assert_array_equal(audio_window(audio, 13, 50)["waveform"], pcm[..., 13:63])
    assert len(encoded) < 4096


def test_remote_audio_stages_once_and_decodes_exact_edited_window(tmp_path, monkeypatch) -> None:
    from dinkster_nodes_media_io import register_media_types
    from dinkster_nodes_media_io.audio_ops import TrimAudio
    from dinkster_values import register_core_types

    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "producer"))
    pcm = np.linspace(-0.5, 0.5, 200_000, dtype=np.float32).reshape(1, 2, -1)
    source = bind_audio_value({"waveform": pcm, "sample_rate": 8000})
    asset = cast(AssetRef, source["source"])
    with asset.open() as file:
        data = file.read()
    expected = TrimAudio.execute(audio=source, start=0.113, duration=0.751)["audio"]
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "audio-contract"\nnamespaces = ["dinkster"]\n'
        '[pack.entry]\nnodes = "dinkster_nodes_media_io:AUDIO_OPS_NODES"\n'
        'types = "dinkster_nodes_media_io:register_media_types"\n'
    )

    class AudioHost(AssetHost):
        async def _handle(self, request: web.Request) -> web.StreamResponse:
            self.requests += 1
            assert request.headers.get("Authorization") == f"Bearer {ENDPOINT_TOKEN}"
            assert request.match_info["digest"] == asset.digest
            return web.Response(body=data)

    async def scenario() -> None:
        proc, host, port = await start_service(
            manifest, tmp_path, "--asset-vault", str(tmp_path / "consumer")
        )
        try:
            registry = TypeRegistry()
            register_core_types(registry)
            register_audio_value_type(registry, "comfy.AUDIO", asset.resolver)
            register_media_types(registry)
            async with AudioHost() as endpoint:
                worker = RemoteWorker(
                    host,
                    port,
                    TOKEN,
                    registry,
                    name="audio-consumer",
                    asset_endpoint=endpoint.endpoint,
                    asset_endpoint_token=ENDPOINT_TOKEN,
                )
                await worker.start()
                try:
                    trimmed = await worker.invoke(
                        Invocation(
                            invocation_id="trim",
                            node_id="trim",
                            node_type="dinkster.audio.trim",
                            inputs={
                                "audio": registry.wrap("comfy.AUDIO", source),
                                "start": registry.wrap("core.float", 0.113),
                                "duration": registry.wrap("core.float", 0.751),
                            },
                            effective_schema=worker.schemas["dinkster.audio.trim"],
                        )
                    )
                    assert trimmed.error is None and trimmed.outputs is not None
                    assert encode_audio(trimmed.outputs["audio"].resolve()) == encode_audio(
                        expected
                    )
                    decoded = await worker.invoke(
                        Invocation(
                            invocation_id="decode",
                            node_id="decode",
                            node_type="dinkster.audio.fade",
                            inputs={
                                "audio": trimmed.outputs["audio"],
                                "fade_in_seconds": registry.wrap("core.float", 0.0),
                                "fade_out_seconds": registry.wrap("core.float", 0.0),
                            },
                            effective_schema=worker.schemas["dinkster.audio.fade"],
                        )
                    )
                    assert decoded.error is None and decoded.outputs is not None
                    np.testing.assert_array_equal(
                        audio_window(decoded.outputs["audio"].resolve(), 0, 6008)["waveform"],
                        pcm[..., 904:6912],
                    )
                    assert endpoint.requests == 1
                finally:
                    await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())
