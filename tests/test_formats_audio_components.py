import io
from dataclasses import replace
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest
from av.audio.stream import AudioStream
from dinkster_assets import AssetRef
from dinkster_assets.audio import bind_audio_value
from dinkster_values import (
    AudioWindowReader,
    append_audio_edit,
    audio_from_source,
    edit_video,
    effective_video_facts,
)
from dinkster_values.audio_lazy import LazyAudio
from dinkster_video import assemble_video, save_video_stream


def test_declared_six_channel_component_audio_encodes_through_shared_windows() -> None:
    expected = np.broadcast_to(np.arange(1, 7, dtype=np.float32)[:, None] / 10, (6, 14400))
    value = assemble_video(
        np.full((3, 64, 64, 3), 0.4, np.float32),
        fps=10,
        audio={"waveform": expected[None], "sample_rate": 48000, "layout": "5.1"},
    )
    output = io.BytesIO()
    save_video_stream(value, output, codec="ffv1", container="mkv")
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert opened.streams.audio[0].layout.name == "5.1"
        resampler = av.AudioResampler(format="fltp", layout="5.1", rate=48000)
        actual = np.concatenate(
            [
                frame.to_ndarray()
                for decoded in opened.decode(audio=0)
                for frame in resampler.resample(decoded)
            ],
            axis=1,
        )
        np.testing.assert_allclose(actual, expected, atol=1 / 32768)


@pytest.mark.parametrize("channels,layout", [(1, "mono"), (6, "5.1")])
def test_asset_audio_saver_reads_only_selected_canonical_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: int, layout: str
) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    pcm = np.arange(channels * 200_000, dtype=np.int16).reshape(1, channels, -1)
    published = bind_audio_value({"waveform": pcm, "sample_rate": 48000, "layout": layout})
    source = published["source"]
    assert isinstance(source, AssetRef)
    audio = audio_from_source(source, stream_index=0)
    value = assemble_video(np.full((10, 64, 64, 3), 0.4, np.float32), fps=10, audio=audio)
    value = edit_video(value, {"trim": {"start_time": 0.1, "duration": 0.5}})
    windows: list[tuple[int, int]] = []
    read = AudioWindowReader.read

    def bounded_window(self, start, count, *, batch_index=None):
        assert batch_index == 0
        assert 0 < count <= 1024
        assert 4800 <= start < start + count <= 28800
        windows.append((start, count))
        return read(self, start, count, batch_index=batch_index)

    monkeypatch.setattr(AudioWindowReader, "read", bounded_window)
    output = io.BytesIO()
    save_video_stream(value, output, codec="ffv1", container="mkv")
    assert windows == [(start, min(1024, 28800 - start)) for start in range(4800, 28800, 1024)]
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert opened.streams.audio[0].layout.name == layout
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=48000)
        actual = np.concatenate(
            [
                frame.to_ndarray()
                for decoded in opened.decode(audio=0)
                for frame in resampler.resample(decoded)
            ],
            axis=1,
        )
        np.testing.assert_array_equal(actual, pcm[0, :, 4800:28800].astype(np.float32) / 32768)


@pytest.mark.parametrize("channels", [9, 10, 17])
@pytest.mark.parametrize("audio_layout", ["preserve", "stereo"])
def test_discrete_audio_fails_before_unsafe_pyav_frame_conversion(
    monkeypatch: pytest.MonkeyPatch, channels: int, audio_layout: str
) -> None:
    import dinkster_video.runtime as runtime

    value = assemble_video(
        np.full((3, 64, 64, 3), 0.4, np.float32),
        fps=10,
        audio={"waveform": np.zeros((1, channels, 14400), np.float32), "sample_rate": 48000},
    )

    def unsafe(*args):
        pytest.fail("discrete AUDIO reached the unsafe PyAV conversion")

    monkeypatch.setattr(runtime, "_audio_frames", unsafe)
    output = io.BytesIO()
    with pytest.raises(ValueError, match="explicit canonical AUDIO channel_map matrix"):
        save_video_stream(value, output, codec="ffv1", container="mkv", audio_layout=audio_layout)
    assert output.getvalue() == b""


@pytest.mark.parametrize("channels", [9, 10, 17])
def test_explicit_channel_map_makes_discrete_audio_encodable(channels: int) -> None:
    pcm = np.broadcast_to(
        np.arange(channels, dtype=np.float32)[None, :, None] / 32, (1, channels, 14400)
    )
    matrix = np.zeros((1, channels), np.float32)
    matrix[0, -1] = 1
    audio = append_audio_edit(
        {"waveform": pcm, "sample_rate": 48000},
        {"channel_map": {"matrix": matrix.tolist(), "layout": "mono"}},
    )
    value = assemble_video(np.full((3, 64, 64, 3), 0.4, np.float32), fps=10, audio=audio)
    output = io.BytesIO()
    save_video_stream(value, output, codec="ffv1", container="mkv")
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert opened.streams.audio[0].layout.name == "mono"
        decoded = np.concatenate([frame.to_ndarray() for frame in opened.decode(audio=0)], axis=1)
        np.testing.assert_array_equal(decoded.astype(np.float32) / 32768, pcm[0, -1:])


def test_trim_to_empty_audio_does_not_save_the_full_video() -> None:
    value = assemble_video(
        np.full((3, 64, 64, 3), 0.4, np.float32),
        fps=10,
        audio={"waveform": np.zeros((1, 1, 0), np.float32), "sample_rate": 48000},
    )
    output = io.BytesIO()
    with pytest.raises(ValueError, match="audio selection contains no samples"):
        save_video_stream(value, output, codec="ffv1", container="mkv", trim_to_audio=True)
    assert output.getvalue() == b""


def _wav_audio(samples: int) -> LazyAudio:
    output = io.BytesIO()
    with av.open(output, "w", format="wav") as writer:
        stream = cast(AudioStream, writer.add_stream("pcm_s32le", rate=8000))
        stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(
            np.full((1, samples * 2), 536870912, np.int32), format="s32", layout="stereo"
        )
        frame.sample_rate = 8000
        writer.mux(stream.encode(frame))
        writer.mux(stream.encode())
    audio = audio_from_source(output.getvalue())
    assert cast(dict[str, object], audio["probe"])["layout"] == "stereo"
    return audio


def _stereo_samples(opened):
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=8000)
    return np.concatenate(
        [
            frame.to_ndarray()
            for decoded in opened.decode(audio=0)
            for frame in resampler.resample(decoded)
        ],
        axis=1,
    )


@pytest.fixture
def estimated_audio():
    return _wav_audio(1000)


@pytest.mark.parametrize("bound", [None, 500, 1000, 4000])
@pytest.mark.parametrize("edited", [False, True])
def test_unproven_component_endpoint_preserves_video_and_reports_fallback(
    estimated_audio, monkeypatch, bound, edited
) -> None:
    import dinkster_values.audio_lazy as audio_runtime

    audio = estimated_audio
    audio["probe"] = {
        **audio["probe"],
        "frames": bound,
        "duration": None if bound is None else bound / 8000,
    }
    probe = audio_runtime.probe_audio

    def estimated_probe(source, stream_index=0):
        return {
            **probe(source, stream_index),
            "frames": bound,
            "duration": None if bound is None else bound / 8000,
        }

    monkeypatch.setattr(audio_runtime, "probe_audio", estimated_probe)
    if edited:
        audio = append_audio_edit(audio, {"trim": {"start_sample": 0, "sample_count": 4000}})
        audio = append_audio_edit(audio, {"concat": [audio]})
    value = assemble_video(np.zeros((10, 64, 64, 3), np.float32), fps=10, audio=audio)
    reads = []
    read, getitem = AudioWindowReader.read, LazyAudio.__getitem__
    decode = audio_runtime._decode_frames
    sessions = []

    def decoded(*args):
        session = []
        sessions.append(session)
        for frame in decode(*args):
            session.append(frame.samples)
            yield frame

    def bounded(self, start, count, *, batch_index=None):
        assert 0 < count <= 1024 and batch_index == 0
        assert 0 <= start < start + count <= 8000
        reads.append((start, count))
        return read(self, start, count, batch_index=batch_index)

    def no_waveform(self, key):
        assert key != "waveform"
        return getitem(self, key)

    monkeypatch.setattr(AudioWindowReader, "read", bounded)
    monkeypatch.setattr(LazyAudio, "__getitem__", no_waveform)
    monkeypatch.setattr(audio_runtime, "_decode_frames", decoded)
    output, diagnostics = io.BytesIO(), []
    save_video_stream(
        value,
        output,
        codec="ffv1",
        container="mkv",
        trim_to_audio=True,
        on_diagnostic=diagnostics.append,
    )
    assert reads and sum(count for _, count in reads) <= 8000
    assert len(sessions) == (2 if edited else 1)
    assert all(sum(session) <= 1000 for session in sessions)
    if bound is None:
        assert all(sum(session) == 1000 for session in sessions)
    (diagnostic,) = diagnostics
    assert diagnostic["code"] == "media_format_fallback"
    assert diagnostic["reason"] == "component_audio_endpoint_unproven"
    assert diagnostic["substitutions"] == [
        {
            "code": "media_format_fallback",
            "requested": {"trim_to_audio": True},
            "effective": {"trim_to_audio": False},
            "reason": "component_audio_endpoint_unproven",
        }
    ]
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert len(list(opened.decode(video=0))) == 10
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        samples = _stereo_samples(opened)
        assert samples.shape == (2, 8000)
        covered = min(1000, bound) if bound is not None else 1000
        expected = np.zeros((2, 8000), np.float32)
        expected[:, :covered] = 0.25
        if edited:
            offset = min(bound, 4000) if bound is not None else 4000
            expected[:, offset : offset + covered] = 0.25
        np.testing.assert_array_equal(samples, expected)


@pytest.mark.parametrize("published", [False, True])
def test_exact_pcm_edits_retain_shortest_audio(published, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path))
    audio = {"waveform": np.full((1, 2, 80000), 0.25, np.float32), "sample_rate": 8000}
    if published:
        audio = bind_audio_value(audio)
        assert cast(dict[str, object], audio["probe"])["codec"] == "pcm_npy"
    audio = append_audio_edit(audio, {"trim": {"start_sample": 3000, "sample_count": 1000}})
    audio = append_audio_edit(audio, {"concat": [audio]})
    audio = append_audio_edit(audio, {"resample": 16000})
    value = assemble_video(np.zeros((16, 64, 64, 3), np.float32), fps=16, audio=audio)
    output, diagnostics = io.BytesIO(), []
    save_video_stream(
        value,
        output,
        codec="ffv1",
        container="mkv",
        trim_to_audio=True,
        on_diagnostic=diagnostics.append,
    )
    assert diagnostics == []
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert len(list(opened.decode(video=0))) == 4
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert sum(frame.samples for frame in opened.decode(audio=0)) == 4000


@pytest.mark.parametrize("trim,bound", [(False, None), (True, None), (True, 1000)])
def test_unbounded_component_audio_and_video_reject_before_reader(
    estimated_audio, monkeypatch, trim, bound
) -> None:
    import dinkster_video.runtime as runtime

    estimated_audio["probe"] = {
        **estimated_audio["probe"],
        "frames": bound,
        "duration": None if bound is None else bound / 8000,
    }
    value = assemble_video(np.zeros((10, 64, 64, 3), np.float32), fps=10, audio=estimated_audio)
    segment = replace(runtime._plan(value)[0], duration=None)
    facts = {**effective_video_facts(value), "duration": None}
    monkeypatch.setattr(runtime, "_plan", lambda _: [segment])
    monkeypatch.setattr(runtime, "effective_video_facts", lambda _: facts)

    def forbidden(*args):
        pytest.fail("unbounded component audio reached decoder admission")

    monkeypatch.setattr(runtime, "_audio_frames", forbidden)
    output = io.BytesIO()
    message = "finite AUDIO or VIDEO read bound" if bound is None else "requires finite VIDEO"
    with pytest.raises(ValueError, match=message):
        save_video_stream(value, output, codec="ffv1", container="mkv", trim_to_audio=trim)
    assert output.getvalue() == b""


def test_pcm_root_does_not_certify_encoded_concat_child(estimated_audio) -> None:
    audio = append_audio_edit(
        {"waveform": np.full((1, 2, 1000), 0.25, np.float32), "sample_rate": 8000},
        {"concat": [estimated_audio]},
    )
    value = assemble_video(np.zeros((10, 64, 64, 3), np.float32), fps=10, audio=audio)
    output, diagnostics = io.BytesIO(), []
    save_video_stream(
        value,
        output,
        codec="ffv1",
        container="mkv",
        trim_to_audio=True,
        on_diagnostic=diagnostics.append,
    )
    assert diagnostics[0]["reason"] == "component_audio_endpoint_unproven"
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        assert len(list(opened.decode(video=0))) == 10
    with av.open(io.BytesIO(output.getvalue()), "r") as opened:
        samples = _stereo_samples(opened)
        assert samples.shape == (2, 8000)
        np.testing.assert_array_equal(samples[:, :2000], 0.25)
        np.testing.assert_array_equal(samples[:, 2000:], 0)
