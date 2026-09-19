"""Portable AUDIO v2 source/probe/edits, with persisted v1 PCM decoding."""

from __future__ import annotations

import io
import json
import math
import re
import struct
from collections.abc import Callable, Iterator, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from .audio_lazy import (
    AUDIO_INLINE_LIMIT,
    AudioRangeError,
    AudioSourceUnavailableError,
    AudioWindowReader,
    append_audio_edit,
    asset_wire,
    audio_from_source,
    audio_window,
    bind_audio_sources,
    coerce_audio,
    effective_audio_facts,
    integer,
    iter_audio_chunks,
    mapping,
)
from .image_codec import decode_image_array_buffer, encode_canonical_png
from .model import stable_hash
from .registry import InvalidRenditionRequest, RenditionUnavailable
from .storage import array_storage_meta, storage_dtype

AUDIO_WAVEFORM_VERSION = "rgba-v1"
AUDIO_WINDOW_VERSION = "pcm16-v1"
AUDIO_WAVEFORM_LIMITS = {"width": 2048, "height": 512, "pixels": 262_144}
AUDIO_WINDOW_LIMITS = {"durationSeconds": 30, "sampleValues": 8_388_608}
_AUDIO_SELECTOR_NUMBER_CHARS = 64

__all__ = [
    "AUDIO_INLINE_LIMIT",
    "AudioRangeError",
    "AudioSourceUnavailableError",
    "AudioWindowReader",
    "append_audio_edit",
    "audio_from_source",
    "audio_window",
    "bind_audio_sources",
    "coerce_audio",
    "effective_audio_facts",
    "audio_encoded_meta",
    "audio_fingerprint",
    "audio_meta",
    "normalize_audio_waveform_request",
    "normalize_audio_window_request",
    "decode_audio",
    "decode_audio_buffer",
    "encode_audio",
    "iter_audio_chunks",
    "render_audio_wav",
    "render_audio_waveform",
    "render_audio_window",
    "validate_audio_encoded",
]

_MAGIC = b"DINKSTER-AUDIO\x02"
_HEADER_LIMIT = 1024 * 1024


def _json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - audio venvs ship numpy
        raise RuntimeError("the audio codec needs numpy in this interpreter") from exc
    return numpy


def _as_array(obj: object) -> Any:
    if hasattr(obj, "detach") and hasattr(obj, "dtype"):
        tensor = cast("Any", obj)
        kind = str(tensor.dtype).removeprefix("torch.")
        return tensor if kind in ("int16", "float32") else tensor.detach().float()
    if hasattr(obj, "detach"):
        obj = cast("Any", obj).detach().cpu().numpy()
    np = _numpy()
    array = np.asarray(obj)
    return (
        array
        if array.dtype in (np.dtype("int16"), np.dtype("float32"))
        else array.astype(np.float32)
    )


def audio_parts(obj: object) -> tuple[Any, int]:
    if not isinstance(obj, Mapping):
        raise TypeError("audio value must be a mapping with waveform and sample_rate")
    value = cast("Mapping[str, object]", obj)
    try:
        waveform = value["waveform"]
        sample_rate = value["sample_rate"]
    except KeyError as exc:
        raise ValueError(f"audio value is missing {exc.args[0]}") from exc
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("audio sample_rate must be an int")
    if not 0 < sample_rate < 1 << 64:
        raise ValueError("audio sample_rate must fit a positive uint64")
    array = _as_array(waveform)
    if array.ndim != 3:
        raise ValueError(
            f"audio waveform must have [B, C, T] layout, got shape {tuple(array.shape)}"
        )
    return array, sample_rate


def encode_audio(obj: object) -> bytes:
    """Encode portable sources without materializing PCM or serializing local bindings."""
    np = _numpy()
    value = coerce_audio(obj)
    payloads: list[bytes] = []

    def encode_record(record: Mapping[str, Any]) -> dict[str, object]:
        source = record["source"]
        if isinstance(source, bytes):
            payloads.append(source)
            source = {"inline_encoded": len(source)}
        elif isinstance(source, Mapping) and "pcm" in source:
            pcm = mapping(cast(object, source), "PCM")["pcm"]
            if pcm.nbytes > AUDIO_INLINE_LIMIT:
                raise ValueError("inline AUDIO exceeds 256 KiB; publish PCM to an asset store")
            buffer = io.BytesIO()
            np.save(buffer, pcm, allow_pickle=False)
            payloads.append(buffer.getvalue())
            source = {"inline_pcm": len(payloads[-1])}
        else:
            source = {"asset": asset_wire(cast(object, source))}
        edits = [
            {"concat": [encode_record(child) for child in edit["concat"]]}
            if "concat" in edit
            else edit
            for edit in record["edits"]
        ]
        return {"source": source, "probe": record["probe"], "edits": edits}

    header = _json(encode_record(value))
    if len(header) > _HEADER_LIMIT:
        raise ValueError("AUDIO header exceeds 1 MiB")
    return _MAGIC + struct.pack("<I", len(header)) + header + b"".join(payloads)


def _records(
    obj: object, depth: int = 1, budget: list[int] | None = None
) -> Iterator[dict[str, Any]]:
    budget = [0, 0] if budget is None else budget
    budget[0] += 1
    if depth > 16 or budget[0] > 64:
        raise ValueError("AUDIO tree exceeds depth 16 or 64 records")
    record = mapping(obj, "AUDIO header")
    if set(record) != {"source", "probe", "edits"}:
        raise ValueError("invalid AUDIO header fields")
    edits = record["edits"]
    if not isinstance(edits, list):
        raise ValueError("AUDIO edits must be a list")
    budget[1] += len(cast("list[object]", edits))
    if budget[1] > 256:
        raise ValueError("AUDIO edits exceed the 256-operation wire budget")
    yield cast("dict[str, Any]", record)
    for raw in cast("list[object]", edits):
        edit = mapping(raw, "AUDIO edit")
        if "concat" in edit:
            children = edit["concat"]
            if not isinstance(children, list):
                raise ValueError("AUDIO concat must contain child values")
            for child in cast("list[object]", children):
                yield from _records(child, depth + 1, budget)


def _wire_records(
    header: object, payload: bytes | memoryview
) -> Iterator[tuple[dict[str, Any], Mapping[str, Any], memoryview]]:
    offset = 0
    for record in _records(header):
        source = mapping(record["source"], "AUDIO wire source")
        if set(source) == {"asset"}:
            length = 0
        elif set(source) in ({"inline_pcm"}, {"inline_encoded"}):
            length = integer(next(iter(source.values())), "AUDIO inline payload length")
            if "inline_encoded" in source and length > AUDIO_INLINE_LIMIT:
                raise ValueError("inline encoded AUDIO exceeds 256 KiB")
            if "inline_pcm" in source and length > AUDIO_INLINE_LIMIT + _HEADER_LIMIT:
                raise ValueError("inline AUDIO exceeds 256 KiB plus npy header")
        else:
            raise ValueError("invalid AUDIO wire source")
        if offset + length > len(payload):
            raise ValueError("invalid AUDIO inline payload length")
        yield record, source, memoryview(payload)[offset : offset + length]
        offset += length
    if offset != len(payload):
        raise ValueError("AUDIO payload length does not match inline sources")


def _npy_header(data: bytes | memoryview, *, inline: bool) -> tuple[tuple[int, ...], Any]:
    np = _numpy()
    # Only the bounded header is copied before the shape and exact payload length are checked.
    buffer = io.BytesIO(bytes(data[:_HEADER_LIMIT]))
    try:
        version = np.lib.format.read_magic(buffer)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(buffer)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(buffer)
        else:
            raise ValueError("unsupported npy version")
    except Exception as exc:
        raise ValueError(f"invalid audio npy payload: {exc}") from exc
    if (
        dtype not in (np.dtype("float32"), np.dtype("int16"))
        or (not inline and dtype != np.dtype("float32"))
        or len(shape) != 3
    ):
        raise ValueError(
            "audio npy payload must contain a float32 [B, C, T] waveform or v2 int16 PCM"
        )
    size = math.prod(shape) * dtype.itemsize
    if size + buffer.tell() != len(data):
        raise ValueError("audio npy shape does not match payload length")
    if inline and size > AUDIO_INLINE_LIMIT:
        raise ValueError("inline AUDIO exceeds 256 KiB")
    return tuple(shape), dtype


def _wire(data: bytes | memoryview) -> tuple[dict[str, Any] | None, bytes | memoryview, int]:
    data = memoryview(data)
    if bytes(data[: len(_MAGIC)]) != _MAGIC:
        if len(data) < 8:
            raise ValueError("audio payload is shorter than its sample-rate header")
        rate = struct.unpack_from("<Q", data)[0]
        if rate == 0:
            raise ValueError("audio sample_rate must be positive")
        _npy_header(data[8:], inline=False)
        return None, data[8:], rate
    offset = len(_MAGIC) + 4
    if len(data) < offset:
        raise ValueError("truncated AUDIO header")
    length = struct.unpack_from("<I", data, len(_MAGIC))[0]
    if length > _HEADER_LIMIT or offset + length > len(data):
        raise ValueError("invalid AUDIO header length")
    try:
        header = dict(mapping(json.loads(bytes(data[offset : offset + length])), "AUDIO header"))
    except RecursionError as exc:
        raise ValueError("AUDIO header nesting exceeds the tree budget") from exc
    payload = data[offset + length :]
    # Validate the entire descriptor tree before inspecting or allocating inline samples.
    records = list(_wire_records(header, payload))
    facts = effective_audio_facts(header)
    for record, source, chunk in records:
        probe = mapping(record["probe"], "probe")
        if "inline_pcm" in source:
            shape, _ = _npy_header(chunk, inline=True)
            if shape != (probe["batch"], probe["channels"], probe["frames"]):
                raise ValueError("AUDIO probe does not match inline PCM")
        else:
            if "asset" in source:
                asset_wire(source["asset"])
            if probe["batch"] != 1 and probe["codec"] != "pcm_npy":
                raise ValueError("asset AUDIO must have one batch")
    return header, payload, facts["sample_rate"]


def decode_audio(data: bytes) -> object:
    """Validate allocation sizes before decoding small inline PCM; assets stay lazy."""
    header, payload, rate = _wire(data)
    if header is None:
        return {
            "waveform": _numpy().load(io.BytesIO(payload), allow_pickle=False),
            "sample_rate": rate,
        }
    for record, source, chunk in _wire_records(header, payload):
        if "inline_pcm" in source:
            record["source"] = {
                "pcm": _numpy().load(io.BytesIO(chunk), allow_pickle=False),
                "sample_rate": record["probe"]["sample_rate"],
            }
        elif "inline_encoded" in source:
            record["source"] = bytes(chunk)
        else:
            record["source"] = asset_wire(source["asset"])
    return coerce_audio(header)


def decode_audio_buffer(data: memoryview, release: Callable[[], None]) -> object:
    header, payload, rate = _wire(data)
    if header is None:
        return {
            "waveform": decode_image_array_buffer(memoryview(payload), release),
            "sample_rate": rate,
        }
    records = list(_wire_records(header, payload))
    if len(records) == 1 and "inline_pcm" in records[0][1]:
        record, _, chunk = records[0]
        record["source"] = {
            "pcm": decode_image_array_buffer(chunk, release),
            "sample_rate": record["probe"]["sample_rate"],
        }
        return coerce_audio(header)
    copied = bytes(data)
    data.release()
    release()
    return decode_audio(copied)


def audio_encoded_meta(data: bytes | memoryview) -> Mapping[str, object]:
    """Validate and describe encoded AUDIO without allocating samples."""
    header, payload, rate = _wire(data)
    if header is None:
        shape, _ = _npy_header(payload, inline=False)
        expected = {"sample_rate": rate, "shape": shape, "duration": shape[-1] / rate}
    else:
        facts = effective_audio_facts(header)
        expected = {**facts, "shape": (facts["batch"], facts["channels"], facts["frames"])}
        resident = 0
        storage_kinds: set[str] = set()
        refs: dict[str, dict[str, object]] = {}
        for _, source, chunk in _wire_records(header, payload):
            if "inline_pcm" in source:
                shape, dtype = _npy_header(chunk, inline=True)
                resident += math.prod(shape) * dtype.itemsize
                storage_kinds.add(storage_dtype(_numpy().empty(0, dtype=dtype)))
            elif "inline_encoded" in source:
                resident += len(chunk)
            else:
                ref = asset_wire(source["asset"])
                refs.setdefault(cast(str, ref["digest"]), ref)
        expected.update(
            codec_version=2,
            asset_refs=list(refs.values()),
            cost={"ram": resident},
        )
        if len(storage_kinds) == 1:
            expected["storage_dtype"] = storage_kinds.pop()
    return expected


def validate_audio_encoded(data: bytes | memoryview, metadata: Mapping[str, object]) -> None:
    """Wire admission without numpy sample allocation, suitable for shared memory."""
    expected = audio_encoded_meta(data)
    for key in (
        "sample_rate",
        "shape",
        "duration",
        "channels",
        "layout",
        "codec_version",
        "asset_refs",
    ):
        if key in metadata:
            got = tuple(cast(Any, metadata[key])) if key == "shape" else metadata[key]
            if got != expected.get(key):
                raise ValueError(f"AUDIO metadata {key} does not match bytes")


def audio_fingerprint(type_id: str) -> Callable[[object], str]:
    """A form-independent fingerprint over the exact encoded bytes."""

    def fingerprint(obj: object) -> str:
        return stable_hash([type_id.encode("utf-8"), encode_audio(obj)])

    return fingerprint


def audio_meta(obj: object) -> Mapping[str, object]:
    """Report the sample rate, waveform shape, and duration in seconds."""
    if isinstance(obj, Mapping) and "source" in obj:
        value = coerce_audio(cast(object, obj))
        facts = effective_audio_facts(value)
        resident = 0
        storage_kinds: set[str] = set()
        refs: dict[str, dict[str, object]] = {}
        for record in _records(value):
            source = record["source"]
            if isinstance(source, bytes):
                resident += len(source)
            elif isinstance(source, Mapping) and "pcm" in source:
                pcm = mapping(cast(object, source), "source")["pcm"]
                resident += pcm.nbytes
                storage_kinds.add(storage_dtype(pcm))
            else:
                ref = asset_wire(cast(object, source))
                refs.setdefault(cast(str, ref["digest"]), ref)
        result = {
            **facts,
            "shape": (facts["batch"], facts["channels"], facts["frames"]),
            "codec_version": 2,
            "asset_refs": list(refs.values()),
            "cost": {"ram": resident},
        }
        if len(storage_kinds) == 1:
            result["storage_dtype"] = storage_kinds.pop()
        return result
    waveform, sample_rate = audio_parts(cast(object, obj))
    return {
        "sample_rate": sample_rate,
        "shape": tuple(int(n) for n in waveform.shape),
        "duration": int(waveform.shape[2]) / sample_rate,
        **array_storage_meta(waveform),
    }


def render_audio_wav(obj: object) -> bytes:
    """Render the first batch; lazy sources use a bounded ten-second preview window."""
    np = _numpy()
    if isinstance(obj, Mapping) and "source" in obj:
        facts = effective_audio_facts(cast(object, obj))
        obj = audio_window(cast(object, obj), 0, facts["sample_rate"] * 10, batch_index=0)
    waveform, sample_rate = audio_parts(cast(object, obj))
    batch, channels, frames = (int(n) for n in waveform.shape)
    if batch < 1:
        raise ValueError("audio WAV rendition requires a nonempty batch")
    if channels == 0 or frames == 0:
        raise ValueError("audio WAV rendition requires nonempty channels and frames")

    uint16_max = (1 << 16) - 1
    uint32_max = (1 << 32) - 1
    block_align = channels * 2
    byte_rate = sample_rate * block_align
    data_size = frames * block_align
    riff_size = 36 + data_size
    if channels > uint16_max or block_align > uint16_max:
        raise ValueError("audio WAV channel count or block alignment exceeds uint16")
    if sample_rate > uint32_max or byte_rate > uint32_max:
        raise ValueError("audio WAV sample rate or byte rate exceeds uint32")
    if data_size > uint32_max or riff_size > uint32_max:
        raise ValueError("audio WAV data or RIFF size exceeds uint32")
    if not bool(np.isfinite(waveform[0]).all()):
        raise ValueError("audio WAV rendition requires finite samples")

    samples = waveform[0]
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    scaled = np.rint(np.clip(samples, -1.0, 1.0) * 32768.0)
    pcm = np.clip(scaled, -32768, 32767).astype("<i2").T.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header + pcm


def _metadata_audio_facts(metadata: Mapping[str, object]) -> tuple[int, int, int, int | None]:
    shape = metadata.get("shape")
    batch = metadata.get("batch")
    channels = metadata.get("channels")
    frames = metadata.get("frames")
    sample_rate = metadata.get("sample_rate")
    shape_values: list[object] | tuple[object, ...] | None = None
    if isinstance(shape, list):
        shape_values = cast(list[object], shape)
    elif isinstance(shape, tuple):
        shape_values = cast(tuple[object, ...], shape)
    if shape_values is not None and len(shape_values) == 3:
        batch = shape_values[0] if batch is None else batch
        channels = shape_values[1] if channels is None else channels
        frames = shape_values[2] if frames is None else frames
    if not all(type(value) is int and value > 0 for value in (batch, channels, sample_rate)):
        raise InvalidRenditionRequest("audio metadata is unavailable for rendition selection")
    if frames is not None and (type(frames) is not int or frames < 0):
        raise InvalidRenditionRequest("audio frame metadata is invalid")
    return cast(int, batch), cast(int, channels), cast(int, sample_rate), frames


def _batch_parameter(parameters: Mapping[str, str], batch: int) -> str:
    raw = parameters.get("batch", "0")
    if len(raw) > _AUDIO_SELECTOR_NUMBER_CHARS or not re.fullmatch(r"[0-9]+", raw):
        raise InvalidRenditionRequest("batch must be an unsigned decimal integer")
    selected = int(raw)
    if selected >= batch:
        raise InvalidRenditionRequest(f"batch {selected} is out of range for batch size {batch}")
    return str(selected)


def _decimal_parameter(raw: str, name: str) -> tuple[Decimal, str]:
    if len(raw) > _AUDIO_SELECTOR_NUMBER_CHARS or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw):
        raise InvalidRenditionRequest(f"{name} must use nonnegative decimal notation")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise InvalidRenditionRequest(f"{name} must use nonnegative decimal notation") from None
    normalized = format(value, "f").rstrip("0").rstrip(".") if "." in raw else str(int(value))
    return value, normalized or "0"


def normalize_audio_waveform_request(
    parameters: Mapping[str, str], metadata: Mapping[str, object]
) -> Mapping[str, str]:
    batch, _, _, frames = _metadata_audio_facts(metadata)
    dimensions = parameters.get("waveform")
    match = None if dimensions is None else re.fullmatch(r"([0-9]+)x([0-9]+)", dimensions)
    if match is None or any(len(value) > _AUDIO_SELECTOR_NUMBER_CHARS for value in match.groups()):
        raise InvalidRenditionRequest("waveform must have WIDTHxHEIGHT syntax")
    width, height = (int(value) for value in match.groups())
    if not 0 < width <= AUDIO_WAVEFORM_LIMITS["width"]:
        raise InvalidRenditionRequest("waveform width must be between 1 and 2048")
    if not 0 < height <= AUDIO_WAVEFORM_LIMITS["height"]:
        raise InvalidRenditionRequest("waveform height must be between 1 and 512")
    if width * height > AUDIO_WAVEFORM_LIMITS["pixels"]:
        raise InvalidRenditionRequest("waveform dimensions exceed 262144 pixels")
    if frames is None:
        raise RenditionUnavailable("waveform requires a known effective audio duration")
    return {"batch": _batch_parameter(parameters, batch), "waveform": f"{width}x{height}"}


def normalize_audio_window_request(
    parameters: Mapping[str, str], metadata: Mapping[str, object]
) -> Mapping[str, str]:
    batch, channels, sample_rate, frames = _metadata_audio_facts(metadata)
    window = parameters.get("window")
    parts = [] if window is None else window.split(",")
    if len(parts) != 2:
        raise InvalidRenditionRequest("window must have START,DURATION syntax")
    start, normalized_start = _decimal_parameter(parts[0], "window start")
    duration, normalized_duration = _decimal_parameter(parts[1], "window duration")
    if duration <= 0 or duration > AUDIO_WINDOW_LIMITS["durationSeconds"]:
        raise InvalidRenditionRequest(
            "window duration must be greater than 0 and at most 30 seconds"
        )
    start_sample = int(start * sample_rate)
    sample_count = int(duration * sample_rate)
    if sample_count < 1:
        raise InvalidRenditionRequest("window duration is shorter than one sample")
    if sample_count * channels > AUDIO_WINDOW_LIMITS["sampleValues"]:
        raise InvalidRenditionRequest("window exceeds 8388608 sample values")
    if frames is not None and (start_sample >= frames or start_sample + sample_count > frames):
        raise InvalidRenditionRequest("window is outside the effective audio timeline")
    return {
        "batch": _batch_parameter(parameters, batch),
        "window": f"{normalized_start},{normalized_duration}",
    }


def _selected_audio_window(obj: object, parameters: Mapping[str, str]) -> dict[str, Any]:
    facts = effective_audio_facts(obj)
    start, duration = (Decimal(value) for value in parameters["window"].split(","))
    start_sample = int(start * facts["sample_rate"])
    sample_count = int(duration * facts["sample_rate"])
    try:
        result = audio_window(obj, start_sample, sample_count, batch_index=int(parameters["batch"]))
    except AudioSourceUnavailableError as error:
        raise RenditionUnavailable(str(error)) from error
    if result["waveform"].shape[-1] != sample_count:
        raise InvalidRenditionRequest("window is outside the effective audio timeline")
    return result


def render_audio_window(obj: object, parameters: Mapping[str, str]) -> bytes:
    return render_audio_wav(_selected_audio_window(obj, parameters))


def render_audio_waveform(obj: object, parameters: Mapping[str, str]) -> bytes:
    np = _numpy()
    width, height = (int(value) for value in parameters["waveform"].split("x"))
    batch_index = int(parameters["batch"])
    facts = effective_audio_facts(obj)
    frames = facts["frames"]
    if frames is None:
        raise RenditionUnavailable("waveform requires a known effective audio duration")
    background = np.array((0x11, 0x18, 0x27, 0xFF), dtype=np.uint8)
    waveform_color = np.array((0x22, 0xD3, 0xEE, 0xFF), dtype=np.uint8)
    center_color = np.array((0x37, 0x41, 0x51, 0xFF), dtype=np.uint8)
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    pixels[:] = background
    chunk_samples = max(1, AUDIO_WINDOW_LIMITS["sampleValues"] // facts["channels"])
    try:
        with AudioWindowReader(obj) as reader:
            for x in range(width):
                start = frames * x // width
                stop = frames * (x + 1) // width
                minimum = maximum = 0.0
                has_samples = False
                for offset in range(start, stop, chunk_samples):
                    count = min(chunk_samples, stop - offset)
                    samples = reader.read(offset, count, batch_index=batch_index)["waveform"][0]
                    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
                    samples = np.clip(samples, -1.0, 1.0)
                    if samples.size:
                        current_min = float(samples.min())
                        current_max = float(samples.max())
                        minimum = min(minimum, current_min) if has_samples else current_min
                        maximum = max(maximum, current_max) if has_samples else current_max
                        has_samples = True
                if not has_samples or (minimum == 0.0 and maximum == 0.0):
                    center = math.floor((height - 1) / 2 + 0.5)
                    pixels[center, x] = center_color
                    continue
                top = math.floor((1.0 - maximum) * (height - 1) / 2 + 0.5)
                bottom = math.floor((1.0 - minimum) * (height - 1) / 2 + 0.5)
                pixels[top : bottom + 1, x] = waveform_color
    except AudioSourceUnavailableError as error:
        raise RenditionUnavailable(str(error)) from error
    return encode_canonical_png(pixels.tobytes(), width=width, height=height, color_type=6)
