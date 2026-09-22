"""Portable audio sources and sample-indexed lazy edits, with no asset-store dependency."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import closing
from fractions import Fraction
from io import BytesIO
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO, Protocol, cast, runtime_checkable

AUDIO_INLINE_LIMIT = 256 * 1024
AUDIO_WINDOW_LIMIT = 32 * 1024 * 1024


class AudioRangeError(ValueError):
    """The selected AUDIO batch element is outside the value's batch."""


class AudioSourceUnavailableError(ValueError):
    """A portable AUDIO source has no local binding for sample reads."""


@runtime_checkable
class AudioSource(Protocol):
    def open(self) -> BinaryIO: ...

    def to_wire(self) -> dict[str, object]: ...


class _PCM(Protocol):
    shape: tuple[int, ...]


def mapping(obj: object, name: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast("Mapping[str, Any]", obj)


def integer(obj: object, name: str, minimum: int = 0) -> int:
    if isinstance(obj, bool) or not isinstance(obj, int) or obj < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return obj


def number(obj: object, name: str) -> float:
    if isinstance(obj, bool) or not isinstance(obj, (int, float)) or not math.isfinite(obj):
        raise ValueError(f"{name} must be finite")
    return float(obj)


def asset_wire(source: object) -> dict[str, object]:
    wire = source.to_wire() if isinstance(source, AudioSource) else mapping(source, "AUDIO source")
    digest = wire.get("digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 71
        or not digest.startswith("blake3:")
        or any(c not in "0123456789abcdef" for c in digest[7:])
    ):
        raise ValueError("AUDIO asset requires a canonical blake3 digest")
    if set(wire) - {"digest", "size", "name", "mediaType", "virtualPath"}:
        raise ValueError("AUDIO source contains unknown fields; host paths are not portable")
    integer(wire.get("size"), "AUDIO source size")
    for key in ("name", "mediaType", "virtualPath"):
        if key in wire and not isinstance(wire[key], str):
            raise ValueError(f"AUDIO source {key} must be a string")
    return dict(wire)


def channel_layout(channels: int) -> str:
    """Use FFmpeg's conventional layouts, preserving discrete unknown channel counts."""
    return {
        1: "mono",
        2: "stereo",
        3: "2.1",
        4: "quad",
        5: "5.0",
        6: "5.1",
        7: "6.1",
        8: "7.1",
    }.get(channels, f"{channels}c")


def _decoded_frame_count(frames: Iterable[Any], rate: int, channels: int) -> int:
    decoded_frames = 0
    for frame in frames:
        if frame.sample_rate != rate or len(frame.layout.channels) != channels:
            raise ValueError("audio stream changes sample rate or channel count")
        decoded_frames += frame.samples
    return decoded_frames


def probe_audio(source: bytes | AudioSource, stream_index: int = 0) -> dict[str, object]:
    import av

    integer(stream_index, "audio stream index")
    with BytesIO(source) if isinstance(source, bytes) else source.open() as handle:
        prefix = handle.read(14)
        if prefix[8:] == b"\x93NUMPY":
            rate = struct.unpack("<Q", prefix[:8])[0]
            shape, _, _ = pcm_asset_header(handle)
            integer(rate, "audio sample_rate", 1)
            if stream_index:
                raise ValueError("PCM assets have one audio stream")
            return {
                "sample_rate": rate,
                "channels": shape[1],
                "layout": channel_layout(shape[1]),
                "duration": shape[2] / rate,
                "codec": "pcm_npy",
                "frames": shape[2],
                "batch": shape[0],
                "stream_index": 0,
            }
    with (
        BytesIO(source) if isinstance(source, bytes) else source.open() as handle,
        closing(av.open(handle, mode="r")) as container,
    ):
        if stream_index >= len(container.streams.audio):
            raise ValueError("audio asset contains no selected audio stream")
        stream = cast(Any, container.streams.audio[stream_index])
        rate = int(stream.codec_context.sample_rate or 0)
        channels = int(stream.codec_context.channels or 0)
        integer(rate, "audio sample_rate", 1)
        integer(channels, "audio channels", 1)
        duration = (
            Fraction(stream.duration) * stream.time_base
            if stream.duration is not None and stream.time_base is not None
            else Fraction(container.duration, av.time_base)
            if container.duration is not None
            else None
        )
        # Ogg Opus granule duration includes pre-skip; decoded timestamps exclude it.
        if (
            duration is not None
            and container.format.name == "ogg"
            and stream.codec_context.name == "opus"
        ):
            duration = max(Fraction(0), duration - Fraction(stream.codec_context.delay, rate))
        if (
            duration is None
            and "webm" in container.format.name.split(",")
            and stream.codec_context.name in {"opus", "vorbis"}
            and not container.streams.video
        ):
            duration = Fraction(
                _decoded_frame_count(
                    cast("Iterable[Any]", container.decode(stream)), rate, channels
                ),
                rate,
            )
        result: dict[str, object] = {
            "sample_rate": rate,
            "channels": channels,
            "layout": stream.codec_context.layout.name or channel_layout(channels),
            "duration": float(duration) if duration is not None else None,
            "codec": stream.codec_context.name,
            "frames": round(duration * rate) if duration is not None else None,
            "batch": 1,
            "stream_index": stream_index,
        }
    return result


def pcm_asset_header(handle: BinaryIO) -> tuple[tuple[int, ...], Any, int]:
    import numpy as np

    handle.seek(8)
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError("unsupported PCM asset npy version")
    offset = handle.tell()
    handle.seek(0, 2)
    if (
        len(shape) != 3
        or min(shape) < 0
        or fortran
        or dtype not in (np.dtype("float32"), np.dtype("int16"))
        or offset + math.prod(shape) * dtype.itemsize != handle.tell()
    ):
        raise ValueError("invalid PCM asset shape, dtype, or length")
    return shape, dtype, offset


class LazyAudio(dict[str, object]):
    """Legacy indexed waveform access is an explicit, bounded materialization boundary."""

    def get(self, key: str, default: object = None) -> object:
        return self[key] if key in ("waveform", "sample_rate") else super().get(key, default)

    def __missing__(self, key: str) -> object:
        if key == "sample_rate":
            return effective_audio_facts(self)["sample_rate"]
        if key == "waveform":
            facts = effective_audio_facts(self)
            if facts["frames"] is None:
                raise ValueError("unknown-duration audio requires an explicit sample window")
            return audio_window(self, 0, facts["frames"])["waveform"]
        raise KeyError(key)


def audio_from_source(source: bytes | AudioSource, *, stream_index: int = 0) -> LazyAudio:
    if isinstance(source, bytes):
        if len(source) > AUDIO_INLINE_LIMIT:
            raise ValueError("inline encoded AUDIO exceeds 256 KiB")
    else:
        asset_wire(source)
    return LazyAudio(source=source, probe=probe_audio(source, stream_index), edits=[])


def _probe(obj: object) -> dict[str, Any]:
    probe = dict(mapping(obj, "AUDIO probe"))
    if set(probe) != {
        "sample_rate",
        "channels",
        "layout",
        "duration",
        "codec",
        "frames",
        "batch",
        "stream_index",
    }:
        raise ValueError("invalid AUDIO probe fields")
    for key in ("sample_rate", "channels", "batch"):
        integer(probe[key], f"AUDIO {key}", 1)
    integer(probe["stream_index"], "AUDIO stream_index")
    for key in ("layout", "codec"):
        if not isinstance(probe[key], str) or not probe[key]:
            raise ValueError(f"AUDIO {key} must be a nonempty string")
    if probe["frames"] is not None:
        integer(probe["frames"], "AUDIO frames")
        duration = number(probe["duration"], "AUDIO duration")
        if duration < 0 or round(duration * probe["sample_rate"]) != probe["frames"]:
            raise ValueError("AUDIO duration does not match frames")
    elif probe["duration"] is not None:
        raise ValueError("unknown AUDIO frames require unknown duration")
    return probe


def _edit_facts(
    facts: dict[str, Any], raw: object, *, depth: int = 1, budget: list[int] | None = None
) -> dict[str, Any]:
    edit = mapping(raw, "AUDIO edit")
    if len(edit) != 1:
        raise ValueError("AUDIO edit must contain exactly one operation")
    if "trim" in edit:
        trim = mapping(edit["trim"], "AUDIO trim")
        if set(trim) != {"start_sample", "sample_count"}:
            raise ValueError("invalid AUDIO trim fields")
        start = integer(trim["start_sample"], "trim start_sample")
        count = integer(trim["sample_count"], "trim sample_count")
        facts["frames"] = (
            count if facts["frames"] is None else min(count, max(0, facts["frames"] - start))
        )
    elif "gain" in edit:
        number(edit["gain"], "AUDIO gain")
    elif "concat" in edit:
        children = edit["concat"]
        if not isinstance(children, list) or not children:
            raise ValueError("AUDIO concat must contain child values")
        if facts["frames"] is None:
            raise ValueError("AUDIO concat requires known frames; trim unknown durations first")
        for child in cast("list[object]", children):
            other = _effective_audio_facts(
                child, depth + 1, budget if budget is not None else [0, 0]
            )
            if other["frames"] is None:
                raise ValueError("AUDIO concat requires known frames; trim unknown durations first")
            if any(
                facts[key] != other[key] for key in ("sample_rate", "channels", "layout", "batch")
            ):
                raise ValueError("AUDIO concat requires matching sample_rate/channels/layout/batch")
            facts["frames"] += other["frames"]
    elif "resample" in edit:
        rate = integer(edit["resample"], "resample rate", 1)
        if facts["frames"] is not None:
            facts["frames"] = (facts["frames"] * rate + facts["sample_rate"] - 1) // facts[
                "sample_rate"
            ]
        facts["sample_rate"] = rate
    elif "channel_map" in edit:
        channel_map = mapping(edit["channel_map"], "AUDIO channel_map")
        if set(channel_map) != {"matrix", "layout"}:
            raise ValueError("invalid AUDIO channel_map fields")
        matrix = channel_map["matrix"]
        if not isinstance(matrix, list) or not matrix:
            raise ValueError("AUDIO channel_map matrix must have output rows")
        for row in cast("list[object]", matrix):
            if not isinstance(row, list) or len(cast("list[object]", row)) != facts["channels"]:
                raise ValueError("AUDIO channel_map row must match input channels")
            for weight in cast("list[object]", row):
                number(weight, "channel weight")
        if not isinstance(channel_map["layout"], str) or not channel_map["layout"]:
            raise ValueError("channel_map layout must be nonempty")
        facts["channels"], facts["layout"] = (
            len(cast("list[object]", matrix)),
            channel_map["layout"],
        )
    else:
        raise ValueError("unknown AUDIO edit")
    facts["duration"] = None if facts["frames"] is None else facts["frames"] / facts["sample_rate"]
    return facts


def effective_audio_facts(obj: object) -> dict[str, Any]:
    return _effective_audio_facts(obj, 1, [0, 0])


def _effective_audio_facts(obj: object, depth: int, budget: list[int]) -> dict[str, Any]:
    budget[0] += 1
    if depth > 16 or budget[0] > 64:
        raise ValueError("AUDIO tree exceeds depth 16 or 64 records")
    value = mapping(obj, "AUDIO")
    if "source" not in value:
        from .audio_codec import audio_parts

        waveform, rate = audio_parts(obj)
        return {
            "sample_rate": rate,
            "channels": int(waveform.shape[1]),
            "layout": value.get("layout", channel_layout(int(waveform.shape[1]))),
            "duration": int(waveform.shape[2]) / rate,
            "codec": "pcm_s16le" if str(waveform.dtype) == "int16" else "pcm_f32le",
            "frames": int(waveform.shape[2]),
            "batch": int(waveform.shape[0]),
            "stream_index": 0,
        }
    facts = _probe(value.get("probe"))
    edits = value.get("edits")
    if not isinstance(edits, list):
        raise ValueError("AUDIO edits must be a list")
    budget[1] += len(cast("list[object]", edits))
    if budget[1] > 256:
        raise ValueError("AUDIO edits exceed the 256-operation wire budget")
    for edit in cast("list[object]", edits):
        facts = _edit_facts(facts, edit, depth=depth, budget=budget)
    return facts


def coerce_audio(obj: object) -> LazyAudio:
    value = mapping(obj, "AUDIO")
    if "source" not in value:
        from .audio_codec import audio_parts

        waveform, rate = audio_parts(obj)
        probe = effective_audio_facts({**value, "waveform": waveform})
        return LazyAudio(source={"pcm": waveform, "sample_rate": rate}, probe=probe, edits=[])
    if set(value) != {"source", "probe", "edits"}:
        raise ValueError("invalid AUDIO value fields")
    source = value["source"]
    probe = _probe(value["probe"])
    if isinstance(source, Mapping) and "pcm" in source:
        import numpy as np

        pcm_source = mapping(cast(object, source), "AUDIO PCM source")
        if set(pcm_source) != {"pcm", "sample_rate"}:
            raise ValueError("invalid AUDIO PCM source fields")
        pcm, rate = pcm_source["pcm"], pcm_source["sample_rate"]
        dtype = str(getattr(pcm, "dtype", "")).removeprefix("torch.")
        if not isinstance(pcm, np.ndarray) and not hasattr(pcm, "detach"):
            raise ValueError("inline AUDIO requires int16 or float32 PCM")
        if dtype not in ("int16", "float32"):
            raise ValueError("inline AUDIO requires int16 or float32 PCM")
        shape = tuple(int(n) for n in cast(_PCM, pcm).shape)
        if (probe["batch"], probe["channels"], probe["frames"]) != shape or rate != probe[
            "sample_rate"
        ]:
            raise ValueError("AUDIO probe does not match PCM shape/rate")
    else:
        if isinstance(source, bytes):
            if len(source) > AUDIO_INLINE_LIMIT:
                raise ValueError("inline encoded AUDIO exceeds 256 KiB")
        else:
            asset_wire(cast(object, source))
        if probe["batch"] != 1 and probe["codec"] != "pcm_npy":
            raise ValueError("asset AUDIO must have one batch")
    effective_audio_facts(value)
    edits = [
        {"concat": [coerce_audio(child) for child in edit["concat"]]}
        if "concat" in edit
        else dict(edit)
        for edit in value["edits"]
    ]
    return LazyAudio(source=cast(object, source), probe=probe, edits=edits)


def append_audio_edit(obj: object, edit: Mapping[str, object]) -> LazyAudio:
    value = coerce_audio(obj)
    value["edits"] = [*cast("list[object]", value["edits"]), dict(edit)]
    return coerce_audio(value)


def bind_audio_sources(
    obj: object, factory: Callable[[Mapping[str, object]], AudioSource]
) -> LazyAudio:
    value = coerce_audio(obj)
    source = value["source"]
    if not isinstance(source, (bytes, AudioSource)) and "pcm" not in mapping(
        source, "AUDIO source"
    ):
        value["source"] = factory(asset_wire(source))
    value["edits"] = [
        {"concat": [bind_audio_sources(child, factory) for child in edit["concat"]]}
        if "concat" in edit
        else edit
        for edit in cast("list[dict[str, Any]]", value["edits"])
    ]
    return value


def _independent_packets(codec: str) -> bool:
    return codec == "flac" or codec.startswith("pcm_")


def _decode_frames(container: Any, stream: Any, start: int, rate: int) -> Iterator[Any]:
    independent = _independent_packets(stream.codec_context.name)
    origin = stream.start_time or 0
    for packet in container.demux(stream):
        # FLAC/PCM packets need no codec pre-roll, unlike MP3's bit reservoir or Opus overlap.
        if (
            independent
            and packet.pts is not None
            and packet.duration
            and stream.time_base is not None
        ):
            end = (packet.pts + packet.duration - origin) * stream.time_base * rate
            if end <= start:
                continue
        yield from packet.decode()


def _decode_source_window(
    source: bytes | AudioSource,
    probe: Mapping[str, Any],
    start: int,
    count: int,
    batch_index: int | None = None,
) -> Any:
    import numpy as np

    if probe["codec"] != "pcm_npy":
        with closing(_SourceWindowReader(source, probe)) as reader:
            return reader.read(start, count, retain=False)
    measured = probe_audio(source, probe["stream_index"])
    if measured["codec"] == "pcm_npy":
        # PCM headers store channel count, not speaker assignments; the value declares layout.
        measured["layout"] = probe["layout"]
    if measured != probe:
        raise ValueError("AUDIO probe does not match source stream")
    channels = probe["channels"]
    batches = range(probe["batch"]) if batch_index is None else range(batch_index, batch_index + 1)
    result = np.empty((len(batches), channels, count), dtype=np.float32)
    with BytesIO(source) if isinstance(source, bytes) else source.open() as handle:
        shape, dtype, offset = pcm_asset_header(handle)
        for output_batch, batch in enumerate(batches):
            for channel in range(channels):
                handle.seek(
                    offset + ((batch * channels + channel) * shape[2] + start) * dtype.itemsize
                )
                data = handle.read(count * dtype.itemsize)
                if len(data) != count * dtype.itemsize:
                    raise ValueError("truncated PCM asset window")
                result[output_batch, channel] = np.frombuffer(data, dtype=dtype)
    if dtype == np.int16:
        result /= 32768.0
    return result


def _source_frames(
    source: bytes | AudioSource, probe: Mapping[str, Any], start: int
) -> Generator[tuple[int, Any], None, int]:
    import av
    import numpy as np

    if probe_audio(source, probe["stream_index"]) != probe:
        raise ValueError("AUDIO probe does not match source stream")
    rate, channels = probe["sample_rate"], probe["channels"]
    cursor = extent = 0
    with (
        BytesIO(source) if isinstance(source, bytes) else source.open() as handle,
        av.open(handle, mode="r") as container,
    ):
        stream = cast(Any, container.streams.audio[probe["stream_index"]])
        if stream.codec_context.sample_rate != rate or stream.codec_context.channels != channels:
            raise ValueError("AUDIO probe does not match source stream")
        origin = stream.start_time or 0
        # Seek to a preceding packet; PTS, not decoder packet size, determines sample placement.
        seek_start = max(0, start - (0 if _independent_packets(probe["codec"]) else rate))
        # FFmpeg AAC PNS advances random_state across all prior frames (aacdec_proc_template.c,
        # NOISE_BT); decoder flush does not reset it. Only a fresh origin decode is exact.
        if probe["codec"] == "aac" or probe["frames"] is None:
            seek_start = 0
        if seek_start and stream.time_base is not None:
            container.seek(
                origin + int(Fraction(seek_start, rate) / stream.time_base),
                stream=stream,
                backward=True,
            )
        # Unknown sources must observe skipped prefixes too, including requests past EOF.
        decode_start = 0 if probe["frames"] is None else start
        for frame in _decode_frames(container, stream, decode_start, rate):
            if frame.sample_rate != rate or len(frame.layout.channels) != channels:
                raise ValueError("audio stream changes sample rate or channel count")
            position = (
                round((Fraction(frame.pts) * frame.time_base - origin * stream.time_base) * rate)
                if frame.pts is not None
                and frame.time_base is not None
                and stream.time_base is not None
                else cursor
            )
            end = position + frame.samples
            cursor = end
            extent = max(extent, end)
            if end <= start:
                continue
            data = frame.to_ndarray()
            if not frame.format.is_planar:
                data = data.reshape(-1, channels).T
            if data.dtype.kind in "iu":
                info = np.iinfo(data.dtype)
                data = (data.astype(np.float32) - (128 if data.dtype.kind == "u" else 0)) / float(
                    128 if data.dtype.kind == "u" else max(abs(info.min), info.max)
                )
            yield position, np.asarray(data, dtype=np.float32)
    return extent


class _SourceWindowReader:
    def __init__(self, source: bytes | AudioSource, probe: Mapping[str, Any]) -> None:
        self.source = source
        self.probe = probe
        self.frames: Generator[tuple[int, Any], None, int] | None = None
        # At most 64 source occurrences share the 32 MiB in-memory overlap budget.
        self.cache = SpooledTemporaryFile(max_size=AUDIO_WINDOW_LIMIT // 64)
        self.pending: tuple[int, Any] | None = None
        self.first = 0
        self.stop = 0
        self.cursor = 0
        self.eof: int | None = None

    def suspend(self) -> None:
        if self.frames is not None:
            self.frames.close()
            self.frames = None
        self.pending = None

    def close(self) -> None:
        self.suspend()
        self.cache.close()

    def contains(self, start: int, count: int) -> bool:
        return self.first <= start and start + count <= self.stop

    def read(self, start: int, count: int, *, retain: bool = True) -> Any:
        import numpy as np

        result = np.zeros((1, self.probe["channels"], count), dtype=np.float32)
        if not count:
            return result
        if start < self.first:
            self.suspend()
            self.first = self.stop = start
        stop = start + count
        overlap = max(0, min(self.stop, stop) - start)
        if overlap:
            for channel in range(self.probe["channels"]):
                self.cache.seek(((self.stop - self.first) * channel + start - self.first) * 4)
                self.cache.readinto(memoryview(result[0, channel, :overlap]).cast("B"))
        if stop <= self.stop:
            return result
        if self.frames is None:
            self.frames = _source_frames(self.source, self.probe, start + overlap)
            self.cursor = 0
            self.eof = None

        def copy_frame(item: tuple[int, Any]) -> None:
            position, data = item
            lo, hi = max(position, start + overlap), min(position + data.shape[-1], stop)
            if lo < hi:
                result[0, :, lo - start : hi - start] = data[:, lo - position : hi - position]

        if self.pending is not None:
            copy_frame(self.pending)
        while self.eof is None and self.cursor < stop:
            try:
                item = next(self.frames)
            except StopIteration as end:
                self.eof = cast(int, end.value)
                break
            position, data = item
            self.cursor = position + data.shape[-1]
            copy_frame(item)
            self.pending = item
        if not retain:
            return result
        # Roll over before writing, not after allocating an oversized BytesIO backing buffer.
        if result.nbytes > AUDIO_WINDOW_LIMIT // 64:
            self.cache.rollover()
        self.cache.seek(0)
        self.cache.truncate()
        self.cache.write(memoryview(result).cast("B"))
        self.first, self.stop = start, stop
        return result


class AudioWindowReader:
    """Consumer-owned AUDIO decoder; use with a context manager or close explicitly.

    Sequential reads reuse codec state and a bounded overlap buffer. Reads behind that buffer
    reopen deterministically. A reader is not thread-safe and never mutates its AUDIO value.
    """

    def __init__(self, obj: object, *, _clip_at_eof: bool = False) -> None:
        self._value = coerce_audio(obj)
        self._sources: dict[tuple[int, ...], _SourceWindowReader] = {}
        self._used: set[tuple[int, ...]] = set()
        self._active: _SourceWindowReader | None = None
        self._closed = False
        # Only iterator-owned reads clip an originally unknown root at observed EOF.
        self._clip_at_eof = _clip_at_eof and effective_audio_facts(self._value)["frames"] is None
        self._source_extent: int | None = None

    def __enter__(self) -> AudioWindowReader:
        if self._closed:
            raise ValueError("audio window reader is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True
        for source in self._sources.values():
            source.close()
        self._sources.clear()
        self._used.clear()
        self._active = None

    def _decode(
        self,
        path: tuple[int, ...],
        source: bytes | AudioSource,
        probe: Mapping[str, Any],
        start: int,
        count: int,
        batch_index: int | None = None,
    ) -> Any:
        if probe["codec"] == "pcm_npy":
            return _decode_source_window(source, probe, start, count, batch_index)
        self._used.add(path)
        if path not in self._sources:
            self._sources[path] = _SourceWindowReader(source, probe)
        reader = self._sources[path]
        if not reader.contains(start, count) and reader is not self._active:
            # Concat visits sources in timeline order. Earlier occurrences need only cached
            # halos; keep one live decoder so large container indexes cannot accumulate.
            if self._active is not None:
                self._active.suspend()
            self._active = reader
        return reader.read(start, count)

    def read(
        self, start_sample: int, sample_count: int, *, batch_index: int | None = None
    ) -> dict[str, Any]:
        if self._closed:
            raise ValueError("audio window reader is closed")
        self._used.clear()
        try:
            result = _audio_window(
                self._value,
                start_sample,
                sample_count,
                batch_index,
                self._decode,
                source_extent=self._source_extent,
            )
            source = self._sources.get(())
            if self._clip_at_eof and self._source_extent is None and source is not None:
                if source.eof is not None:
                    self._source_extent = source.eof
                    # Reapply EOF at every resample stage before exposing the final windows.
                    result = self.read(start_sample, sample_count, batch_index=batch_index)
            # Only sources intersecting the current window/halos can be needed by the next
            # forward window. Backward requests reopen retired occurrences deterministically.
            for path in self._sources.keys() - self._used:
                source = self._sources.pop(path)
                source.close()
                if source is self._active:
                    self._active = None
            return result
        except BaseException:
            self.close()
            raise


def audio_window(
    obj: object, start_sample: int, sample_count: int, *, batch_index: int | None = None
) -> dict[str, Any]:
    """Decode only a bounded consumer window in the effective timeline, in float32 BCT form."""
    return _audio_window(obj, start_sample, sample_count, batch_index)


def _batch_size(batch: int, batch_index: int | None) -> int:
    if batch_index is None:
        return batch
    integer(batch_index, "batch_index")
    if batch_index >= batch:
        raise AudioRangeError("audio batch_index is out of range")
    return 1


def iter_audio_chunks(
    obj: object,
    *,
    start_sample: int = 0,
    sample_count: int | None = None,
    chunk_samples: int,
    batch_index: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Yield ordered windows with one decoder owner; close explicitly on early termination."""
    start = integer(start_sample, "start_sample")
    chunk = integer(chunk_samples, "chunk_samples", 1)
    end = None if sample_count is None else start + integer(sample_count, "sample_count")
    facts = effective_audio_facts(obj)
    with AudioWindowReader(obj, _clip_at_eof=True) as reader:
        batch = _batch_size(facts["batch"], batch_index)
        if chunk * facts["channels"] * batch * 4 > AUDIO_WINDOW_LIMIT:
            raise ValueError("audio chunk exceeds the 32 MiB allocation budget")
        if facts["frames"] is not None:
            end = facts["frames"] if end is None else min(end, facts["frames"])
        while end is None or start < end:
            count = chunk if end is None else min(chunk, end - start)
            result = reader.read(start, count, batch_index=batch_index)
            size = result["waveform"].shape[-1]
            if not size:
                break
            yield result
            start += size


def _audio_window(
    obj: object,
    start_sample: int,
    sample_count: int,
    batch_index: int | None,
    decode: Callable[
        [tuple[int, ...], bytes | AudioSource, Mapping[str, Any], int, int, int | None], Any
    ]
    | None = None,
    path: tuple[int, ...] = (),
    *,
    source_extent: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    start = integer(start_sample, "start_sample")
    count = integer(sample_count, "sample_count")
    value = coerce_audio(obj)
    edits = cast("list[object]", value["edits"])
    source_probe = _probe(value["probe"])
    stages = [dict(source_probe)]
    if source_extent is not None:
        stages[0].update(frames=source_extent, duration=source_extent / source_probe["sample_rate"])
    for edit in edits:
        stages.append(_edit_facts(dict(stages[-1]), edit))
    facts = stages[-1]
    batch = _batch_size(facts["batch"], batch_index)
    if facts["frames"] is not None:
        count = min(count, max(0, facts["frames"] - start))

    def read(stage: int, begin: int, length: int) -> Any:
        info = stages[stage]
        if length * info["channels"] * batch * 4 > AUDIO_WINDOW_LIMIT:
            raise ValueError(
                "audio window exceeds the 32 MiB allocation budget; request smaller chunks"
            )
        if not length:
            return np.empty((batch, info["channels"], 0), dtype=np.float32)
        if stage == 0:
            source = value["source"]
            if isinstance(source, Mapping) and "pcm" in source:
                pcm = mapping(cast(object, source), "PCM")["pcm"]
                if batch_index is not None:
                    pcm = pcm[batch_index : batch_index + 1]
                pcm = pcm[..., begin : begin + length]
                if str(pcm.dtype).removeprefix("torch.") == "int16":
                    return (
                        pcm.float() / 32768.0
                        if hasattr(pcm, "detach")
                        else pcm.astype(np.float32) / 32768.0
                    )
                return pcm
            if not isinstance(source, (bytes, AudioSource)):
                raise AudioSourceUnavailableError("AUDIO source has no local asset binding")
            return (
                _decode_source_window(source, source_probe, begin, length, batch_index)
                if decode is None
                else decode(path, source, source_probe, begin, length, batch_index)
            )
        edit = mapping(edits[stage - 1], "edit")
        if "trim" in edit:
            return read(stage - 1, begin + edit["trim"]["start_sample"], length)
        if "gain" in edit:
            return read(stage - 1, begin, length) * np.float32(edit["gain"])
        if "concat" in edit:
            result = np.empty((batch, info["channels"], length), dtype=np.float32)
            offset = 0
            for child_index, child in enumerate([None, *edit["concat"]]):
                frames = (
                    stages[stage - 1]["frames"]
                    if child is None
                    else effective_audio_facts(child)["frames"]
                )
                lo, hi = max(begin, offset), min(begin + length, offset + frames)
                if lo < hi:
                    result[..., lo - begin : hi - begin] = (
                        read(stage - 1, lo - offset, hi - lo)
                        if child is None
                        else _audio_window(
                            child,
                            lo - offset,
                            hi - lo,
                            batch_index,
                            decode,
                            (*path, stage - 1, child_index),
                        )["waveform"]
                    )
                offset += frames
                if offset >= begin + length:
                    break
            return result
        if "channel_map" in edit:
            return np.einsum(
                "oc,bct->bot",
                np.asarray(edit["channel_map"]["matrix"], dtype=np.float32),
                read(stage - 1, begin, length),
            )
        from scipy.signal import resample_poly

        previous = stages[stage - 1]
        ratio = Fraction(info["sample_rate"], previous["sample_rate"])
        up, down = ratio.numerator, ratio.denominator
        if (20 * max(up, down) + 1) * 8 > AUDIO_WINDOW_LIMIT:
            raise ValueError("audio resample filter exceeds the 32 MiB allocation budget")
        # Aligned source origins preserve the global polyphase phase across arbitrary windows.
        halo = 12 * max(up, down)
        first = max(0, ((begin * down // up - halo) // down) * down)
        last = ((begin + length) * down + up - 1) // up + halo
        if previous["frames"] is not None:
            last = min(last, previous["frames"])
        converted_frames = (max(0, last - first) * up + down - 1) // down
        if converted_frames * info["channels"] * batch * 4 > AUDIO_WINDOW_LIMIT:
            raise ValueError("audio resample intermediate exceeds the 32 MiB allocation budget")
        converted = resample_poly(read(stage - 1, first, max(0, last - first)), up, down, axis=-1)
        offset = begin - first * up // down
        return converted[..., offset : offset + length]

    waveform = np.ascontiguousarray(read(len(edits), start, count), dtype=np.float32)
    if not bool(np.isfinite(waveform).all()):
        raise ValueError("audio window contains non-finite samples")
    return {"waveform": waveform, "sample_rate": facts["sample_rate"]}
