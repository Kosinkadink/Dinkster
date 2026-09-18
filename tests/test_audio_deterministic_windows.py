"""AAC PNS fixtures generated with PyAV's native AAC encoder, seed 1353, 32 kbps mono."""

from __future__ import annotations

import asyncio
import io
import wave
from contextlib import closing
from fractions import Fraction
from tempfile import SpooledTemporaryFile
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_api.v1 import AudioWindowReader, Curve
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_nodes_media_io.audio import SaveAudio
from dinkster_nodes_media_io.audio_ops import AudioOnsets
from dinkster_values import (
    AudioRangeError,
    AudioSourceUnavailableError,
    append_audio_edit,
    audio_from_source,
    audio_window,
    effective_audio_facts,
    encode_audio,
    iter_audio_chunks,
)
from dinkster_values.audio_codec import coerce_audio, decode_audio
from dinkster_video import assemble_video, save_video_stream

from tests.test_audio_io import _bound, _mount


def _encoded_aac(offset, seeds=(1353,)):
    buffer = io.BytesIO()
    rate = 48000
    with av.open(buffer, "w", format="mov") as container:
        streams = []
        for seed in seeds:
            stream = cast(Any, container.add_stream("aac", rate=rate, layout="mono"))
            stream.bit_rate = 32000
            streams.append((stream, np.random.default_rng(seed)))
        for start in range(0, 8 * rate, 1024):
            count = min(1024, 8 * rate - start)
            for stream, rng in streams:
                pcm = rng.uniform(-0.5, 0.5, (1, count)).astype(np.float32)
                frame = av.AudioFrame.from_ndarray(pcm, format="fltp", layout="mono")
                frame.sample_rate = rate
                frame.pts = offset + start
                container.mux(stream.encode(frame))
        for stream, _ in streams:
            container.mux(stream.encode(None))
    data = buffer.getvalue()
    with av.open(io.BytesIO(data)) as container:
        assert bool(container.streams.audio[0].start_time) == bool(offset)
    return data


@pytest.fixture(params=[0, 96000], ids=["origin", "offset-mov"])
def aac_audio(tmp_path, request):
    data = _encoded_aac(request.param)
    vault = AssetVault(tmp_path / "vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    return audio_from_source(AssetRef(digest, "noise.mov", len(data), resolver=vault))


@pytest.fixture
def decode_counts(monkeypatch):
    import dinkster_values.audio_lazy as lazy

    counts = {"samples": 0, "sessions": 0}
    decode = lazy._decode_frames
    opened = AssetRef.open
    handles = []
    caches = []
    spooled = SpooledTemporaryFile

    def counted(*args):
        counts["sessions"] += 1
        for frame in decode(*args):
            counts["samples"] += frame.samples
            yield frame

    def tracked_open(self):
        handle = opened(self)
        handles.append(handle)
        return handle

    def tracked_bytes(data):
        handle = io.BytesIO(data)
        handles.append(handle)
        return handle

    def tracked_cache(*args, **kwargs):
        cache = spooled(*args, **kwargs)
        caches.append(cache)
        return cache

    monkeypatch.setattr(lazy, "_decode_frames", counted)
    monkeypatch.setattr(AssetRef, "open", tracked_open)
    monkeypatch.setattr(lazy, "BytesIO", tracked_bytes)
    monkeypatch.setattr(lazy, "SpooledTemporaryFile", tracked_cache)
    yield counts
    assert handles and all(handle.closed for handle in handles)
    assert caches and all(cache.closed for cache in caches)


def _edited(audio, rates):
    audio = append_audio_edit(audio, {"trim": {"start_sample": 54960, "sample_count": 290003}})
    audio = append_audio_edit(audio, {"gain": 0.375})
    audio = append_audio_edit(audio, {"channel_map": {"matrix": [[1], [-0.5]], "layout": "stereo"}})
    for rate in rates:
        audio = append_audio_edit(audio, {"resample": rate})
    return audio


@pytest.mark.parametrize("rates", [None, [44100], [32000, 44100, 48000]])
def test_aac_full_random_and_sequential_windows_are_bit_exact(aac_audio, decode_counts, rates):
    audio = aac_audio if rates is None else _edited(aac_audio, rates)
    facts = effective_audio_facts(audio)
    total = facts["frames"]
    full = audio_window(audio, 0, total)["waveform"]
    for start in (54960, total - 2001, 113, total // 2):
        np.testing.assert_array_equal(
            audio_window(audio, start, 2001)["waveform"], full[..., start : start + 2001]
        )
    decode_counts.update(samples=0, sessions=0)
    with AudioWindowReader(audio) as reader:
        for start in range(0, total, 1024):
            np.testing.assert_array_equal(
                reader.read(start, 1024)["waveform"], full[..., start : start + 1024]
            )
        assert reader.read(total, 1)["waveform"].shape[-1] == 0
    assert decode_counts["sessions"] == 1
    assert decode_counts["samples"] <= effective_audio_facts(aac_audio)["frames"] + 1024
    print("sequential", rates, decode_counts)


def test_aac_overlap_and_backward_reopen_do_not_modify_returned_windows(aac_audio, decode_counts):
    full = audio_window(aac_audio, 0, 150000)["waveform"]
    decode_counts.update(samples=0, sessions=0)
    with AudioWindowReader(aac_audio) as reader:
        first = reader.read(54960, 65536)["waveform"]
        overlap = reader.read(60000, 70000)["waveform"]
        assert decode_counts["sessions"] == 1
        # Modifying a returned window must not poison the retained source buffer.
        overlap[:] = 0
        np.testing.assert_array_equal(reader.read(60000, 2001)["waveform"], full[..., 60000:62001])
        np.testing.assert_array_equal(reader.read(17, 2001)["waveform"], full[..., 17:2018])
        assert decode_counts["sessions"] == 2
        np.testing.assert_array_equal(first, full[..., 54960:120496])
    with pytest.raises(ValueError, match="closed"):
        reader.read(0, 1)
    reader.close()


@pytest.mark.parametrize("consumer", ["save-audio", "video-mux"])
def test_aac_saving_decodes_source_once(aac_audio, tmp_path, monkeypatch, decode_counts, consumer):
    audio = _edited(aac_audio, [44100])
    facts = effective_audio_facts(audio)
    total = effective_audio_facts(aac_audio)["frames"]
    loaded = None
    if consumer == "save-audio":
        _, snapshot = _mount(tmp_path, monkeypatch)
        saved = cast(list[AssetRef], SaveAudio.execute(audio=audio)["audios"])[0]
        loaded = audio_from_source(_bound(saved, snapshot))
    else:
        duration = Fraction(facts["frames"], facts["sample_rate"])
        video = assemble_video(np.zeros((8, 16, 16, 3), np.float32), fps=8 / duration, audio=audio)
        output = io.BytesIO()
        save_video_stream(video, output, container="mp4", codec="h264")
        with av.open(io.BytesIO(output.getvalue())) as container:
            assert len(container.streams.audio) == 1
    assert decode_counts["sessions"] == 1
    assert total - 48000 <= decode_counts["samples"] <= total + 1024
    print(consumer, "source_frames", total, decode_counts)
    if loaded is not None:
        assert effective_audio_facts(loaded)["frames"] == facts["frames"]
        expected = audio_window(audio, 65531, 2001)["waveform"]
        actual = audio_window(loaded, 65531, 2001)["waveform"]
        assert np.max(np.abs(expected - actual)) <= 1 / 32768


def test_aac_onsets_decode_once_and_match_the_materialized_curve(aac_audio, decode_counts):
    audio = _edited(aac_audio, [44100])
    facts = effective_audio_facts(audio)
    full = audio_window(audio, 0, facts["frames"])
    expected = AudioOnsets.execute(audio=full)["curve"]
    decode_counts.update(samples=0, sessions=0)
    actual = AudioOnsets.execute(audio=audio)["curve"]
    assert isinstance(actual, Curve)
    assert actual == expected
    assert decode_counts["sessions"] == 1
    assert decode_counts["samples"] <= effective_audio_facts(aac_audio)["frames"] + 1024
    print("onsets", "points", len(actual.points), decode_counts)


@pytest.mark.parametrize(
    "consumer", ["reader", "save-audio", "video-generator", "video-mux", "onsets"]
)
def test_aac_consumers_close_decoder_on_failure_or_cancellation(
    aac_audio, tmp_path, monkeypatch, decode_counts, consumer
):
    import dinkster_nodes_media_io.audio as audio_io
    import dinkster_nodes_media_io.audio_ops as audio_ops
    import dinkster_video.runtime as video_runtime

    class Cancelled(BaseException):
        pass

    if consumer == "reader":
        reader = AudioWindowReader(aac_audio)
        reader.read(54960, 2001)
        with pytest.raises(ValueError, match="sample_count"):
            reader.read(0, -1)
        with pytest.raises(ValueError, match="closed"):
            reader.read(0, 1)
        return
    video = assemble_video(np.zeros((8, 16, 16, 3), np.float32), fps=1, audio=aac_audio)
    if consumer == "video-generator":
        iterator = video_runtime._audio_frames(video_runtime._plan(video), 0)
        next(iterator)
        iterator.close()
        return

    class FailingReader(AudioWindowReader):
        def read(self, *args, **kwargs):
            result = super().read(*args, **kwargs)
            if decode_counts["samples"] > 65536:
                raise Cancelled
            return result

    with pytest.raises(Cancelled):
        if consumer == "save-audio":
            _mount(tmp_path, monkeypatch)
            monkeypatch.setattr(audio_io, "AudioWindowReader", FailingReader)
            SaveAudio.execute(audio=aac_audio)
        elif consumer == "onsets":
            monkeypatch.setattr(audio_ops, "AudioWindowReader", FailingReader)
            AudioOnsets.execute(audio=aac_audio)
        else:
            monkeypatch.setattr(video_runtime, "AudioWindowReader", FailingReader)
            save_video_stream(video, io.BytesIO(), container="mp4", codec="h264")


@pytest.fixture
def concat_audio(aac_audio):
    inline = audio_from_source(_encoded_aac(48000, (11, 17)), stream_index=1)
    root = append_audio_edit(aac_audio, {"trim": {"start_sample": 54960, "sample_count": 70003}})
    child = append_audio_edit(inline, {"trim": {"start_sample": 12000, "sample_count": 80017}})
    child = append_audio_edit(child, {"gain": -0.25})
    tail = append_audio_edit(aac_audio, {"trim": {"start_sample": 24000, "sample_count": 60019}})
    nested = append_audio_edit(child, {"concat": [tail]})
    joined = append_audio_edit(root, {"concat": [nested, root]})
    pcm = np.concatenate(
        [
            audio_window(leaf, 0, effective_audio_facts(leaf)["frames"])["waveform"]
            for leaf in (root, child, tail, root)
        ],
        axis=-1,
    )
    eager = {"waveform": pcm, "sample_rate": 48000}
    edits = [
        {"trim": {"start_sample": 123, "sample_count": pcm.shape[-1] - 200}},
        {"gain": 0.375},
        {"channel_map": {"matrix": [[1], [-0.5]], "layout": "stereo"}},
        {"resample": 44100},
        {"resample": 32000},
    ]
    for edit in edits:
        joined = append_audio_edit(joined, edit)
        eager = append_audio_edit(eager, edit)
    return joined, audio_window(eager, 0, effective_audio_facts(eager)["frames"])


@pytest.mark.parametrize("consumer", ["windows", "save", "mux", "onsets"])
def test_concat_aac_consumers_are_exact_and_decode_each_occurrence_once(
    concat_audio, decode_counts, monkeypatch, tmp_path, consumer
):
    audio, expected = concat_audio
    waveform = expected["waveform"]
    total = waveform.shape[-1]
    for start in (46000, 99000, 139000):
        np.testing.assert_array_equal(
            audio_window(audio, start, 2501)["waveform"], waveform[..., start : start + 2501]
        )
    read = AudioWindowReader.read
    windows = []
    active = []

    def checked(self, start, count, **kwargs):
        result = read(self, start, count, **kwargs)
        np.testing.assert_array_equal(result["waveform"], waveform[..., start : start + count])
        windows.append((start, count))
        active.append(len(self._sources))
        assert sum(source.frames is not None for source in self._sources.values()) <= 1
        return result

    decode_counts.update(samples=0, sessions=0)
    monkeypatch.setattr(AudioWindowReader, "read", checked)
    if consumer == "windows":
        with AudioWindowReader(audio) as reader:
            for start in range(0, total, 1024):
                reader.read(start, min(1024, total - start))
    elif consumer == "save":
        _mount(tmp_path, monkeypatch)
        SaveAudio.execute(audio=audio)
    elif consumer == "mux":
        video = assemble_video(
            np.zeros((8, 16, 16, 3), np.float32),
            fps=Fraction(8 * expected["sample_rate"], total),
            audio=audio,
        )
        save_video_stream(video, io.BytesIO(), container="mp4", codec="h264")
    else:
        curve = AudioOnsets.execute(audio=audio)["curve"]
        assert isinstance(curve, Curve)
        # The eager reference must not pass through the instrumented lazy consumer.
        monkeypatch.setattr(AudioWindowReader, "read", read)
        assert curve == AudioOnsets.execute(audio=expected)["curve"]
    assert decode_counts["sessions"] == 4
    assert decode_counts["samples"] <= 2 * 125952 + 92160 + 84992
    assert len(windows) > 2
    # The final 65k save chunk spans the last two occurrences; small hops touch only one.
    assert active[-1] == (2 if consumer == "save" else 1)
    print("concat", consumer, len(windows), "max_active", max(active), decode_counts)


def test_reader_spills_large_overlaps_and_retires_concat_occurrences(aac_audio, decode_counts):
    leaf = append_audio_edit(aac_audio, {"trim": {"start_sample": 0, "sample_count": 150000}})
    leaf = append_audio_edit(leaf, {"resample": 8})
    expected = audio_window(leaf, 0, 25)["waveform"]
    audio = append_audio_edit(leaf, {"concat": [leaf] * 63})
    decode_counts.update(samples=0, sessions=0)
    with AudioWindowReader(audio) as reader:
        np.testing.assert_array_equal(reader.read(0, 1600)["waveform"], np.tile(expected, 64))
        sources = list(reader._sources.values())
        assert len(sources) == 64
        assert all(cast(Any, source.cache)._rolled for source in sources)
        assert sum(source.frames is not None for source in sources) == 1
        assert decode_counts["sessions"] == 64
        np.testing.assert_array_equal(reader.read(1590, 10)["waveform"], expected[..., -10:])
        assert len(reader._sources) == 1
        assert sum(source.cache.closed for source in sources) == 63
        assert decode_counts["sessions"] == 64
        np.testing.assert_array_equal(reader.read(0, 1)["waveform"], expected[..., :1])
        assert decode_counts["sessions"] == 65
    assert all(source.cache.closed for source in sources)


def test_concat_backward_window_reopens_suspended_sources_with_partial_overlap(
    aac_audio, decode_counts
):
    inline = audio_from_source(_encoded_aac(48000, (41,)))
    audio = append_audio_edit(aac_audio, {"concat": [inline]})
    boundary = effective_audio_facts(aac_audio)["frames"]
    expected = audio_window(audio, boundary - 2000, 4000)["waveform"]
    decode_counts.update(samples=0, sessions=0)
    with AudioWindowReader(audio) as reader:
        np.testing.assert_array_equal(
            reader.read(boundary - 1000, 2000)["waveform"], expected[..., 1000:3000]
        )
        assert decode_counts["sessions"] == 2
        np.testing.assert_array_equal(reader.read(boundary - 2000, 4000)["waveform"], expected)
        assert decode_counts["sessions"] == 4


@pytest.mark.parametrize("failure", ["invalid-child-probe", "cancel", "close"])
def test_concat_reader_closes_every_source_and_spill_on_exit(aac_audio, decode_counts, failure):
    inline = audio_from_source(_encoded_aac(48000, (41,)))
    if failure == "invalid-child-probe":
        inline["probe"] = {**cast(dict, inline["probe"]), "codec": "flac"}
    audio = append_audio_edit(aac_audio, {"concat": [inline]})
    boundary = effective_audio_facts(aac_audio)["frames"]
    reader = AudioWindowReader(audio)
    if failure == "invalid-child-probe":
        with pytest.raises(ValueError, match="probe does not match"):
            reader.read(boundary - 150000, 300000)
    else:
        with reader:
            reader.read(boundary - 150000, 300000)
            assert len(reader._sources) == 2
            if failure == "cancel":
                with pytest.raises(KeyboardInterrupt):
                    with reader:
                        raise KeyboardInterrupt
    assert reader._sources == {}
    with pytest.raises(ValueError, match="closed"):
        reader.read(0, 1)


@pytest.mark.parametrize("chunk_samples", [257, 4096])
@pytest.mark.parametrize("rates", [None, [32000, 44100, 48000]])
def test_aac_chunk_iterator_is_exact_with_one_decoder(
    aac_audio, decode_counts, monkeypatch, chunk_samples, rates
):
    import dinkster_values.audio_codec as codec
    import dinkster_values.audio_lazy as lazy

    assert codec.iter_audio_chunks is iter_audio_chunks
    audio = aac_audio if rates is None else _edited(aac_audio, rates)
    facts = effective_audio_facts(audio)
    expected = audio_window(audio, 0, facts["frames"])["waveform"]
    wire = encode_audio(audio)
    decode_counts.update(samples=0, sessions=0)
    monkeypatch.setattr(lazy, "audio_window", lambda *a, **kw: pytest.fail("independent window"))
    chunks = list(iter_audio_chunks(audio, chunk_samples=chunk_samples))
    assert all(0 < chunk["waveform"].shape[-1] <= chunk_samples for chunk in chunks)
    assert all(chunk["sample_rate"] == facts["sample_rate"] for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate([c["waveform"] for c in chunks], -1), expected)
    assert encode_audio(audio) == wire
    assert decode_counts["sessions"] == 1
    assert decode_counts["samples"] <= effective_audio_facts(aac_audio)["frames"] + 1024


@pytest.mark.parametrize("chunk_samples", [1024, 8191])
def test_concat_chunk_iterator_reuses_each_source_occurrence(
    concat_audio, decode_counts, chunk_samples
):
    audio, expected = concat_audio
    decode_counts.update(samples=0, sessions=0)
    chunks = list(iter_audio_chunks(audio, chunk_samples=chunk_samples))
    np.testing.assert_array_equal(
        np.concatenate([c["waveform"] for c in chunks], -1), expected["waveform"]
    )
    assert decode_counts["sessions"] == 4
    assert decode_counts["samples"] <= 2 * 125952 + 92160 + 84992


@pytest.mark.parametrize("batch_index", [None, 1])
@pytest.mark.parametrize("start,count", [(13, 37), (13, 999), (257, None), (0, 0)])
def test_pcm_chunks_preserve_storage_layout_batch_and_finite_ranges(batch_index, start, count):
    pcm = np.arange(2 * 9 * 257, dtype=np.int16).reshape(2, 9, 257)
    audio = coerce_audio({"waveform": pcm, "sample_rate": 8000})
    facts = effective_audio_facts(audio)
    wire = encode_audio(audio)
    chunks = list(
        iter_audio_chunks(
            audio, start_sample=start, sample_count=count, chunk_samples=17, batch_index=batch_index
        )
    )
    expected = audio_window(audio, start, 257 if count is None else count, batch_index=batch_index)
    if expected["waveform"].shape[-1]:
        np.testing.assert_array_equal(
            np.concatenate([c["waveform"] for c in chunks], -1), expected["waveform"]
        )
    else:
        assert chunks == []
    assert cast(Any, audio["source"])["pcm"] is pcm
    assert pcm.dtype == np.int16 and facts["layout"] == "9c"
    assert effective_audio_facts(audio) == facts and encode_audio(audio) == wire


@pytest.fixture(params=[0, 1003], ids=["empty-wav", "wav"])
def unknown_audio(request, monkeypatch):
    import dinkster_values.audio_lazy as lazy

    pcm = np.arange(request.param * 2, dtype=np.int16).reshape(request.param, 2)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(pcm.tobytes())
    probe = lazy.probe_audio
    if request.param == 0:
        assert probe(buffer.getvalue())["frames"] is None
    decode = lazy._decode_frames
    counts = {"sessions": 0, "samples": 0}

    def unknown_probe(*args):
        return {**probe(*args), "frames": None, "duration": None}

    def counted(*args):
        counts["sessions"] += 1
        for frame in decode(*args):
            counts["samples"] += frame.samples
            yield frame

    monkeypatch.setattr(lazy, "probe_audio", unknown_probe)
    monkeypatch.setattr(lazy, "_decode_frames", counted)
    audio = audio_from_source(buffer.getvalue())
    expected = {"waveform": pcm.T[None].astype(np.float32) / 32768, "sample_rate": 8000}
    return audio, expected, counts


@pytest.mark.parametrize("chunk_samples", [137, 509])
@pytest.mark.parametrize("rates", [[], [11025, 12000, 8000]])
@pytest.mark.parametrize(
    "start,count", [(0, None), (0, 0), (113, None), (2000, None), (0, 5000), (71, 251)]
)
def test_unknown_chunks_terminate_at_decoded_edited_eof(
    unknown_audio, chunk_samples, rates, start, count
):
    audio, expected, counts = unknown_audio
    for edit in [
        {"gain": 0.375},
        {"channel_map": {"matrix": [[0.5, 0], [0, 1], [1, -1]], "layout": "3c"}},
        *({"resample": rate} for rate in rates),
    ]:
        audio, expected = append_audio_edit(audio, edit), append_audio_edit(expected, edit)
    wire = encode_audio(audio)
    facts = effective_audio_facts(audio)
    assert facts["frames"] is None
    total = effective_audio_facts(expected)["frames"]
    full = audio_window(expected, 0, total)["waveform"]
    end = total if count is None else min(total, start + count)
    wanted = full[..., start:end]
    chunks = []
    with closing(
        iter_audio_chunks(
            audio, start_sample=start, sample_count=count, chunk_samples=chunk_samples
        )
    ) as iterator:
        for index, chunk in enumerate(iterator):
            assert index < (wanted.shape[-1] + chunk_samples - 1) // chunk_samples
            assert 0 < chunk["waveform"].shape[-1] <= chunk_samples
            chunks.append(chunk["waveform"])
    if chunks:
        np.testing.assert_array_equal(np.concatenate(chunks, -1), wanted)
    else:
        assert wanted.shape[-1] == 0
    assert effective_audio_facts(audio) == facts and encode_audio(audio) == wire
    assert counts["sessions"] == (0 if count == 0 else 1)
    assert counts["samples"] <= 1003


def test_trimmed_unknown_chunks_keep_declared_finite_padding(unknown_audio):
    audio, _, _ = unknown_audio
    audio = append_audio_edit(audio, {"trim": {"start_sample": 995, "sample_count": 51}})
    audio = append_audio_edit(audio, {"resample": 11025})
    total = effective_audio_facts(audio)["frames"]
    assert total == 71
    expected = audio_window(audio, 0, total)["waveform"]
    chunks = list(iter_audio_chunks(audio, chunk_samples=17))
    np.testing.assert_array_equal(np.concatenate([c["waveform"] for c in chunks], -1), expected)


def test_ordinary_unknown_windows_keep_zero_padding_after_iterator_eof(unknown_audio):
    audio, _, _ = unknown_audio
    assert list(iter_audio_chunks(audio, start_sample=2000, chunk_samples=17)) == []
    expected = np.zeros((1, 2, 17), np.float32)
    np.testing.assert_array_equal(audio_window(audio, 2000, 17)["waveform"], expected)
    with AudioWindowReader(audio) as reader:
        np.testing.assert_array_equal(reader.read(2000, 17)["waveform"], expected)
    assert effective_audio_facts(audio)["frames"] is None


@pytest.mark.parametrize("field", ["codec", "layout", "sample_rate"])
def test_unknown_iterator_still_rejects_forged_probes(unknown_audio, field):
    audio, _, _ = unknown_audio
    probe = dict(cast(dict, audio["probe"]))
    probe[field] = 16000 if field == "sample_rate" else "forged"
    with pytest.raises(ValueError, match="probe does not match"):
        list(iter_audio_chunks({**audio, "probe": probe}, chunk_samples=10))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"chunk_samples": 0}, ValueError),
        ({"chunk_samples": -1}, ValueError),
        ({"chunk_samples": True}, ValueError),
        ({"chunk_samples": 1.5}, ValueError),
        ({"chunk_samples": "1"}, ValueError),
        ({"chunk_samples": 2**64}, ValueError),
        ({"start_sample": -1}, ValueError),
        ({"start_sample": None}, ValueError),
        ({"sample_count": -1}, ValueError),
        ({"sample_count": True}, ValueError),
        ({"batch_index": 1}, AudioRangeError),
        ({"batch_index": -1}, ValueError),
        ({"batch_index": False}, ValueError),
    ],
)
def test_chunk_arguments_are_checked_even_for_empty_ranges_before_allocation(
    monkeypatch, kwargs, error
):
    audio = coerce_audio({"waveform": np.zeros((1, 1, 0), np.int16), "sample_rate": 8000})
    monkeypatch.setattr(np, "empty", lambda *a, **kw: pytest.fail("invalid chunk allocated"))
    monkeypatch.setattr(np, "zeros", lambda *a, **kw: pytest.fail("invalid chunk allocated"))
    with pytest.raises(error) as raised:
        list(iter_audio_chunks(audio, **{"chunk_samples": 1, "sample_count": 0, **kwargs}))
    assert type(raised.value) is error


def test_chunk_iterator_preserves_unbound_source_error(aac_audio):
    audio = decode_audio(encode_audio(aac_audio))
    assert list(iter_audio_chunks(audio, sample_count=0, chunk_samples=1)) == []
    with pytest.raises(AudioSourceUnavailableError):
        list(iter_audio_chunks(audio, chunk_samples=1))


def test_chunk_iterator_closes_on_invalid_concat_source(aac_audio, decode_counts):
    child = {**aac_audio, "probe": {**cast(dict, aac_audio["probe"]), "codec": "forged"}}
    audio = append_audio_edit(aac_audio, {"concat": [child]})
    with pytest.raises(ValueError, match="probe does not match"):
        list(iter_audio_chunks(audio, chunk_samples=150000))
    assert decode_counts["sessions"] == 1


@pytest.mark.parametrize("action", ["close", "break", "throw", "cancel", "read-error"])
def test_chunk_iterator_releases_decoder_and_spool_on_early_exit(
    aac_audio, decode_counts, monkeypatch, action
):
    iterator = iter_audio_chunks(aac_audio, chunk_samples=150000)
    with closing(iterator):
        if action == "break":
            for _ in iterator:
                break
        else:
            next(iterator)
            if action == "close":
                iterator.close()
            elif action == "read-error":

                def fail(*args, **kwargs):
                    raise ValueError("injected read failure")

                monkeypatch.setattr(AudioWindowReader, "read", fail)
                with pytest.raises(ValueError, match="injected read failure"):
                    next(iterator)
            else:
                error = asyncio.CancelledError if action == "cancel" else RuntimeError
                with pytest.raises(error):
                    iterator.throw(error())
    with pytest.raises(StopIteration):
        next(iterator)
    assert decode_counts["sessions"] == 1
