"""Portable VIDEO v2: admitted source, source probe, and ordered lazy edits."""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any, Protocol, cast, runtime_checkable

from .audio_codec import (
    audio_encoded_meta,
    audio_meta,
    bind_audio_sources,
    coerce_audio,
    decode_audio,
    effective_audio_facts,
    encode_audio,
)
from .image_codec import (
    decode_image_array,
    encode_image_array,
    image_array_meta,
    image_encoded_meta,
)
from .model import stable_hash
from .resources import COST_META_KEY
from .storage import BFLOAT16_FIELD, array_storage_meta, storage_dtype
from .video_edits import effective_video_facts, integer, mapping, seconds
from .video_probe import VideoSource, color_space_label, open_video_source, probe_video

VIDEO_CONTAINERS = frozenset({"mp4", "mkv", "mov", "webm", "avi", "gif"})
VIDEO_INLINE_LIMIT = 256 * 1024
VIDEO_HEADER_LIMIT = 1024 * 1024
VIDEO_BYTE_LIMIT = 1024 * 1024 * 1024
_MAGIC = b"DINKSTER-VIDEO\x02"
_MIMES = {
    "mp4": "video/mp4",
    "mkv": "video/x-matroska",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
    "gif": "image/gif",
}
_RATIONAL_FIELDS = ("fps", "time_base", "start_time", "duration")


@runtime_checkable
class VideoAssetSource(VideoSource, Protocol):
    def to_wire(self) -> dict[str, object]: ...


def _json(value: object) -> bytes:
    def rational(obj: object) -> list[int]:
        if isinstance(obj, Fraction):
            return [obj.numerator, obj.denominator]
        raise TypeError(f"VIDEO metadata cannot contain {type(obj).__name__}")

    return json.dumps(
        value, default=rational, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(data: bytes) -> str:
    from blake3 import blake3

    return "blake3:" + blake3(data).hexdigest()


def _asset_wire(source: object) -> dict[str, object]:
    value = (
        source.to_wire() if isinstance(source, VideoAssetSource) else mapping(source, "VIDEO asset")
    )
    digest = value.get("digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 71
        or not digest.startswith("blake3:")
        or any(c not in "0123456789abcdef" for c in digest[7:])
    ):
        raise ValueError("VIDEO asset requires a canonical blake3 digest")
    integer(value.get("size"), "VIDEO asset size", 0)
    keys = {"digest", "name", "size", "mediaType", "virtualPath"}
    if set(value) - keys:
        raise ValueError("VIDEO asset contains unknown fields; host paths are not portable")
    for key in ("name", "mediaType", "virtualPath"):
        if key in value and not isinstance(value[key], str):
            raise ValueError(f"VIDEO asset {key} must be a string")
    return dict(value)


def asset_reference(source: object) -> dict[str, object]:
    """Validate a portable content-addressed source without resolving it."""
    return _asset_wire(source)


def video_reference(value: object) -> dict[str, object]:
    """Snapshot a reference-only v2 VIDEO including its ordered base edits."""
    chunks: list[bytes] = []
    wire = _pack_video(value, chunks)
    if chunks:
        raise ValueError("publish VIDEO source before binding timeline")
    return {"type": "comfy.VIDEO", "video": json.loads(_json(wire))}


def source_video(reference: Mapping[str, object]) -> dict[str, object]:
    """Admit a v2 VIDEO binding without accepting inline payload chunks."""
    return coerce_video(_unpack_video(reference["video"], [], set(), [0, 0]))


def video_from_source(source: bytes | VideoAssetSource) -> dict[str, object]:
    """Admit media once; descendants retain this probe without reopening the source."""
    if not isinstance(source, bytes):
        _asset_wire(source)
    return {"source": source, "probe": _SourceProbe(probe_video(source), source), "edits": []}


class _SourceProbe(dict[str, object]):
    """Process-local evidence; serialized probe fields never convey this binding."""

    def __init__(
        self, facts: Mapping[str, object], source: bytes | VideoAssetSource | None = None
    ) -> None:
        super().__init__(facts)
        self._identity = self._source_identity(source) if source is not None else None
        self._facts = _json(facts)

    @staticmethod
    def _source_identity(source: bytes | VideoAssetSource) -> str:
        return _digest(source) if isinstance(source, bytes) else str(_asset_wire(source)["digest"])

    def verify(self, source: bytes | VideoAssetSource) -> None:
        identity = self._source_identity(source)
        facts = _json(self)
        if self._identity == identity and self._facts == facts:
            return
        if _json(probe_video(source)) != facts:
            raise ValueError("VIDEO probe does not match source bytes")
        self._identity, self._facts = identity, facts


def _probe(obj: object, *, components: bool) -> dict[str, object]:
    probe = dict(mapping(obj, "VIDEO probe"))
    if set(probe) != {
        "container",
        "video_codec",
        "pix_fmt",
        "bit_depth",
        "alpha",
        "color_space",
        "primaries",
        "transfer",
        "matrix",
        "range",
        "width",
        "height",
        "rotation",
        "frame_count",
        "frame_count_kind",
        "duration_kind",
        "audio",
        *_RATIONAL_FIELDS,
    }:
        raise ValueError("invalid VIDEO probe fields")
    for key in ("container", "video_codec", "pix_fmt"):
        if not ((components or key == "pix_fmt") and probe.get(key) is None) and not isinstance(
            probe.get(key), str
        ):
            raise ValueError(f"VIDEO probe {key} must be a string")
    if not components and probe["container"] not in VIDEO_CONTAINERS:
        raise ValueError("unsupported VIDEO source container")
    for key in ("primaries", "transfer", "matrix", "range"):
        integer(probe.get(key), f"probe {key}", 0)
    if probe.get("bit_depth") is not None:
        integer(probe["bit_depth"], "probe bit_depth", 1)
    if not isinstance(probe.get("alpha"), bool):
        raise ValueError("VIDEO probe alpha must be a boolean")
    if probe.get("color_space") not in ("sRGB", "HDR", "HDR PQ", "unknown"):
        raise ValueError("invalid VIDEO probe color_space")
    if probe.get("frame_count_kind") not in ("header", "estimated", "unknown"):
        raise ValueError("invalid VIDEO frame_count_kind")
    if probe.get("duration_kind") not in ("stream", "frames", "container", "unknown"):
        raise ValueError("invalid VIDEO duration_kind")
    for key in _RATIONAL_FIELDS:
        value = probe.get(key)
        probe[key] = seconds(value, f"probe {key}") if value is not None else None
    if probe["time_base"] is not None and cast(Fraction, probe["time_base"]) <= 0:
        raise ValueError("VIDEO time_base must be positive")
    streams = probe.get("audio")
    if not isinstance(streams, list):
        raise ValueError("VIDEO audio probe must be a list")
    audio: list[object] = []
    indexes: set[int] = set()
    for item in cast("list[object]", streams):
        stream = dict(mapping(item, "audio probe"))
        if set(stream) != {
            "index",
            "codec",
            "sample_rate",
            "channels",
            "layout",
            "time_base",
            "start_time",
            "duration",
        }:
            raise ValueError("invalid VIDEO audio probe fields")
        index = integer(stream.get("index"), "audio index", 0)
        if index in indexes:
            raise ValueError("duplicate VIDEO audio stream index")
        indexes.add(index)
        for key in ("sample_rate", "channels"):
            integer(stream.get(key), f"audio {key}", 1)
        for key in ("codec", "layout"):
            if not isinstance(stream.get(key), str):
                raise ValueError(f"audio {key} must be a string")
        for key in ("time_base", "start_time", "duration"):
            value = stream.get(key)
            stream[key] = seconds(value, f"audio {key}") if value is not None else None
        audio.append(stream)
    probe["audio"] = audio
    return probe


def _component_properties(obj: object) -> dict[str, object]:
    components = dict(mapping(obj, "VIDEO components"))
    if set(components) != {"images", "audio", "fps", "bit_depth", "color_space", "color"}:
        raise ValueError("invalid VIDEO components fields")
    rate = seconds(components["fps"], "components fps")
    if rate <= 0:
        raise ValueError("components fps must be positive")
    components["fps"] = rate
    integer(components["bit_depth"], "components bit_depth", 1)
    color = mapping(components["color"], "components color")
    if set(color) != {"primaries", "transfer", "matrix", "range"}:
        raise ValueError("invalid components color fields")
    for key, value in color.items():
        integer(value, key, 0)
    if components["color_space"] != color_space_label(cast(int, color["transfer"])):
        raise ValueError("components color_space does not match transfer")
    components["color"] = dict(color)
    return components


def _component_probe(components: Mapping[str, object], shape: tuple[int, ...]) -> dict[str, object]:
    count, height, width, channels = shape
    return {
        "container": None,
        "video_codec": None,
        "pix_fmt": None,
        "width": width,
        "height": height,
        "rotation": 0,
        "alpha": channels == 4,
        "fps": components["fps"],
        "bit_depth": components["bit_depth"],
        "color_space": components["color_space"],
        **mapping(components["color"], "color"),
        "frame_count": count,
        "frame_count_kind": "header",
        "duration": Fraction(count) / cast(Fraction, components["fps"]),
        "duration_kind": "frames",
        "start_time": Fraction(0),
        "time_base": None,
        "audio": [],
    }


def _components(obj: object) -> dict[str, object]:
    import numpy as np

    components = _component_properties(obj)
    images = components["images"]
    if not isinstance(images, np.ndarray):
        raise ValueError("VIDEO images must be a numpy array")
    images = cast(np.ndarray, images)
    if (
        images.ndim != 4
        or min(images.shape) < 1
        or images.shape[-1] not in (3, 4)
        or (images.dtype.kind not in "fiu" and storage_dtype(images) != "bf16")
    ):
        raise ValueError("VIDEO images require numeric [B,H,W,3|4] layout")
    finite = (
        (images[BFLOAT16_FIELD] & 0x7F80) != 0x7F80
        if storage_dtype(images) == "bf16"
        else np.isfinite(images)
    )
    if not bool(finite.all()):
        raise ValueError("VIDEO images contain non-finite pixels")
    if components["audio"] is not None:
        audio = coerce_audio(components["audio"])
        if effective_audio_facts(audio)["batch"] != 1:
            raise ValueError("VIDEO audio requires a single batch")
        components["audio"] = audio
    return components


def coerce_video(obj: object) -> dict[str, object]:
    value = mapping(obj, "VIDEO")
    if "timeline" in value:
        from .timeline_video import coerce_timeline_video

        return coerce_timeline_video(obj)
    if "bytes" in value:
        data = value["bytes"]
        if not isinstance(data, bytes):
            raise ValueError("legacy video bytes must be bytes")
        result = video_from_source(data)
        if mapping(result["probe"], "probe")["container"] != value.get("container"):
            raise ValueError("legacy video container does not match its bytes")
        return result
    if ("source" in value) == ("components" in value):
        raise ValueError("VIDEO requires exactly one of source and components")
    if set(value) - {"source", "components", "probe", "edits"}:
        raise ValueError("VIDEO contains unknown fields")
    result = dict(value)
    if "source" in value:
        facts = _probe(value.get("probe"), components=False)
        previous = value.get("probe")
        result["probe"] = (
            previous
            if isinstance(previous, _SourceProbe) and previous == facts
            else _SourceProbe(facts)
        )
        if not isinstance(value["source"], bytes):
            wire = _asset_wire(value["source"])
            if not isinstance(value["source"], VideoAssetSource):
                result["source"] = wire
    else:
        components = _components(value["components"])
        result["components"] = components
        expected = _component_probe(components, cast(Any, components["images"]).shape)
        result["probe"] = _probe(value.get("probe", expected), components=True)
        if result["probe"] != expected:
            raise ValueError("component VIDEO probe does not match its components")
    effective_video_facts(result)
    edits: list[object] = []
    for raw in cast("list[object]", value["edits"]):
        edit = dict(mapping(raw, "VIDEO edit"))
        if "concat" in edit:
            edit["concat"] = [coerce_video(child) for child in cast("list[object]", edit["concat"])]
        else:
            key = next(key for key in ("trim", "crop", "scale") if key in edit)
            params = dict(mapping(edit[key], key))
            if key == "trim":
                params = {k: seconds(params.get(k, 0), k) for k in ("start_time", "duration")}
                edit["strict_duration"] = edit.get("strict_duration", False)
            elif key == "scale" and "pad_color" in params:
                params["pad_color"] = list(cast("list[object]", params["pad_color"]))
            edit[key] = params
        edits.append(edit)
    result["edits"] = edits
    if isinstance(result.get("source"), bytes):
        cast(_SourceProbe, result["probe"]).verify(cast(bytes, result["source"]))
    return result


def video_source(obj: object) -> bytes | VideoAssetSource:
    value = coerce_video(obj)
    if "source" not in value:
        raise ValueError("component-backed VIDEO has no original encoded source")
    source = value["source"]
    if isinstance(source, (bytes, VideoAssetSource)):
        cast(_SourceProbe, value["probe"]).verify(source)
        return source
    raise ValueError(f"VIDEO source {_asset_wire(source)['digest']} has no local asset binding")


def bind_video_sources(
    obj: object, factory: Callable[[Mapping[str, object]], VideoAssetSource]
) -> dict[str, object]:
    """Attach the receiving host's asset resolver, including nested concat sources."""
    value = coerce_video(obj)
    if "timeline" in value:
        from .timeline_video import TimelineVideo

        return TimelineVideo(value["timeline"], factory)
    if "source" in value and not isinstance(value["source"], (bytes, VideoAssetSource)):
        value["source"] = factory(_asset_wire(value["source"]))
    if "components" in value:
        components = dict(mapping(value["components"], "VIDEO components"))
        if components["audio"] is not None:
            components["audio"] = bind_audio_sources(components["audio"], factory)
        value["components"] = components
    edits: list[object] = []
    for raw in cast("list[object]", value["edits"]):
        edit = dict(mapping(raw, "VIDEO edit"))
        if "concat" in edit:
            edit["concat"] = [
                bind_video_sources(child, factory) for child in cast("list[object]", edit["concat"])
            ]
        edits.append(edit)
    value["edits"] = edits
    return value


def edit_video(obj: object, edit: Mapping[str, object]) -> dict[str, object]:
    value = coerce_video(obj)
    if "timeline" in value:
        raise ValueError("VIDEO v3 edits must use video_document nodes")
    value["edits"] = [*cast("list[object]", value["edits"]), dict(edit)]
    return coerce_video(value)


def _pack_video(
    obj: object,
    chunks: list[bytes],
    *,
    identity: bool = False,
    budget: list[int] | None = None,
    depth: int = 0,
) -> dict[str, object]:
    budget = budget if budget is not None else [0, 0]
    value = coerce_video(obj)
    edits = cast("list[object]", value["edits"])
    budget[0] += len(edits)
    budget[1] += 1
    if depth > 16 or budget[0] > 256 or budget[1] > 64:
        raise ValueError("VIDEO edit tree exceeds its operation, clip, or nesting limit")
    wire: dict[str, object] = {"probe": value["probe"]}
    if "source" in value:
        source = value["source"]
        if isinstance(source, bytes):
            if identity:
                wire["source"] = {"digest": _digest(source)}
            else:
                if len(source) > VIDEO_INLINE_LIMIT:
                    raise ValueError(
                        "VIDEO inline source exceeds 256 KiB; publish it to an asset store"
                    )
                wire["source"] = {"chunk": len(chunks)}
                chunks.append(source)
        else:
            asset = _asset_wire(source)
            wire["source"] = {"digest": asset["digest"]} if identity else {"asset": asset}
    else:
        components = dict(mapping(value["components"], "VIDEO components"))
        for key, encoder, meta in (
            ("images", encode_image_array, image_array_meta),
            ("audio", encode_audio, audio_meta),
        ):
            component = components.get(key)
            if component is None:
                continue
            data = encoder(component)
            if key == "audio" and len(data) > 256 * 1024 * 1024:
                raise ValueError(f"VIDEO {key} component exceeds its encoded size limit")
            metadata = dict(meta(component))
            metadata.pop(COST_META_KEY, None)
            if identity:
                metadata.pop("storage_dtype", None)
            descriptor: dict[str, object] = {"meta": metadata, "codec": key}
            if identity:
                descriptor["digest"] = _digest(data)
            else:
                descriptor["chunk"] = len(chunks)
                chunks.append(data)
            components[key] = descriptor
        wire["components"] = components
    packed_edits: list[object] = []
    for raw in edits:
        edit = dict(mapping(raw, "VIDEO edit"))
        if "concat" in edit:
            edit["concat"] = [
                _pack_video(child, chunks, identity=identity, budget=budget, depth=depth + 1)
                for child in cast("list[object]", edit["concat"])
            ]
        packed_edits.append(edit)
    wire["edits"] = packed_edits
    return wire


def encode_video(obj: object) -> bytes:
    if "timeline" in mapping(obj, "VIDEO"):
        from .timeline_video import TIMELINE_MAGIC, coerce_timeline_video
        from .video_document import encode_document

        return TIMELINE_MAGIC + encode_document(coerce_timeline_video(obj)["timeline"])
    chunks: list[bytes] = []
    video = _pack_video(obj, chunks)
    header = _json({"video": video, "chunks": [len(chunk) for chunk in chunks]})
    if len(header) > VIDEO_HEADER_LIMIT or sum(map(len, chunks)) > VIDEO_BYTE_LIMIT:
        raise ValueError("VIDEO payload exceeds its header or media size limit")
    return _MAGIC + struct.pack("<Q", len(header)) + header + b"".join(chunks)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate VIDEO JSON key {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite VIDEO JSON constant {value}")


def _validate_array_chunk(data: memoryview, metadata: object, *, audio: bool) -> tuple[int, ...]:
    if not audio:
        import numpy as np

        expected_image = image_encoded_meta(data)
        shape = cast("tuple[int, ...]", expected_image["shape"])
        kind = expected_image["storage_dtype"]
        if len(shape) != 4 or min(shape) < 1 or shape[-1] not in (3, 4):
            raise ValueError("VIDEO images require numeric [B,H,W,3|4] layout")
        if kind != "bf16" and np.dtype(cast(str, expected_image["dtype"])).kind not in "fiu":
            raise ValueError("VIDEO component requires numeric samples")
        declared = mapping(metadata, "VIDEO image metadata")
        if not {"shape", "dtype"} <= set(declared) <= set(expected_image):
            raise ValueError("invalid VIDEO image metadata fields")
        if any(_json(value) != _json(expected_image[key]) for key, value in declared.items()):
            raise ValueError("VIDEO component metadata does not match payload")
        return shape

    expected = audio_encoded_meta(data)
    shape = cast("tuple[int, ...]", expected["shape"])
    if shape[0] != 1:
        raise ValueError("VIDEO audio requires a single batch")
    declared = mapping(metadata, "VIDEO audio metadata")
    if not {"shape", "sample_rate"} <= set(declared) <= set(expected):
        raise ValueError("invalid VIDEO audio metadata fields")
    if any(_json(value) != _json(expected[key]) for key, value in declared.items()):
        raise ValueError("VIDEO component metadata does not match payload")
    return shape


def _read_rational(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, list) or len(cast("list[object]", value)) != 2:
        raise ValueError("VIDEO rational requires a numerator and denominator")
    n, d = cast("list[object]", value)
    numerator = integer(n, "rational numerator")
    denominator = integer(d, "rational denominator", 1)
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise ValueError("VIDEO rational must be reduced")
    return result


def _unpack_video(
    obj: object,
    chunks: list[memoryview],
    used: set[int],
    budget: list[int],
    depth: int = 0,
) -> dict[str, object]:
    wire = dict(mapping(obj, "VIDEO header value"))
    if ("source" in wire) == ("components" in wire) or set(wire) - {
        "source",
        "components",
        "probe",
        "edits",
    }:
        raise ValueError("invalid VIDEO header value fields")
    budget[1] += 1
    if depth > 16 or budget[1] > 64:
        raise ValueError("VIDEO edit tree exceeds its clip or nesting limit")
    probe = dict(mapping(wire.get("probe"), "VIDEO probe"))
    for key in _RATIONAL_FIELDS:
        probe[key] = _read_rational(probe.get(key))
    audio: list[object] = []
    if not isinstance(probe.get("audio"), list):
        raise ValueError("VIDEO audio probe must be a list")
    for raw in cast("list[object]", probe["audio"]):
        stream = dict(mapping(raw, "audio probe"))
        for key in ("time_base", "start_time", "duration"):
            stream[key] = _read_rational(stream.get(key))
        audio.append(stream)
    probe["audio"] = audio
    wire["probe"] = _probe(probe, components="components" in wire)

    def take(descriptor: Mapping[str, object]) -> memoryview:
        index = integer(descriptor.get("chunk"), "VIDEO chunk", 0)
        if index >= len(chunks) or index in used:
            raise ValueError("VIDEO chunk is missing or referenced more than once")
        used.add(index)
        return chunks[index]

    if "source" in wire:
        source = mapping(wire["source"], "VIDEO source descriptor")
        if set(source) == {"chunk"}:
            data = take(source)
            if len(data) > VIDEO_INLINE_LIMIT:
                raise ValueError("VIDEO inline source exceeds 256 KiB")
            wire["source"] = data
        elif set(source) == {"asset"}:
            wire["source"] = _asset_wire(source["asset"])
        else:
            raise ValueError("invalid VIDEO source descriptor")
    elif "components" in wire:
        components = dict(mapping(wire["components"], "VIDEO components"))
        components["fps"] = _read_rational(components.get("fps"))
        components = _component_properties(components)
        for key in ("images", "audio"):
            if key == "audio" and components[key] is None:
                continue
            descriptor = mapping(components[key], "VIDEO component descriptor")
            if set(descriptor) != {"codec", "meta", "chunk"}:
                raise ValueError("invalid VIDEO component descriptor fields")
            if descriptor.get("codec") != key:
                raise ValueError("VIDEO component codec mismatch")
            data = take(descriptor)
            if len(data) > (512 if key == "images" else 256) * 1024 * 1024:
                raise ValueError("VIDEO component exceeds its size limit")
            shape = _validate_array_chunk(data, descriptor["meta"], audio=key == "audio")
            if key == "images" and _component_probe(components, shape) != wire["probe"]:
                raise ValueError("component VIDEO probe does not match its components")
            components[key] = data
        wire["components"] = components
    edits = wire.get("edits")
    if not isinstance(edits, list):
        raise ValueError("VIDEO edits must be a list")
    budget[0] += len(cast("list[object]", edits))
    if budget[0] > 256:
        raise ValueError("VIDEO edit tree exceeds 256 operations")
    decoded_edits: list[object] = []
    for raw in cast("list[object]", edits):
        edit = dict(mapping(raw, "VIDEO edit"))
        if "concat" in edit:
            if not isinstance(edit["concat"], list):
                raise ValueError("concat must be a list")
            edit["concat"] = [
                _unpack_video(child, chunks, used, budget, depth + 1)
                for child in cast("list[object]", edit["concat"])
            ]
        if "trim" in edit:
            trim = dict(mapping(edit["trim"], "trim"))
            for key in ("start_time", "duration"):
                if isinstance(trim.get(key), list):
                    trim[key] = _read_rational(trim[key])
            edit["trim"] = trim
        decoded_edits.append(edit)
    wire["edits"] = decoded_edits
    effective_video_facts(wire)
    return wire


def _materialize_video(wire: dict[str, object]) -> dict[str, object]:
    source = wire.get("source")
    if isinstance(source, memoryview):
        wire["source"] = bytes(cast(memoryview, source))
    if "components" in wire:
        components = cast("dict[str, object]", wire["components"])
        for key, decoder in (("images", decode_image_array), ("audio", decode_audio)):
            if components[key] is not None:
                components[key] = decoder(bytes(cast(memoryview, components[key])))
    for edit in cast("list[dict[str, object]]", wire["edits"]):
        if "concat" in edit:
            edit["concat"] = [
                _materialize_video(child)
                for child in cast("list[dict[str, object]]", edit["concat"])
            ]
    return coerce_video(wire)


def decode_video(data: bytes) -> dict[str, object]:
    from .timeline_video import TIMELINE_MAGIC, TimelineVideo
    from .video_document import decode_document

    if data.startswith(TIMELINE_MAGIC):
        return TimelineVideo(decode_document(data[len(TIMELINE_MAGIC) :]))
    if not data.startswith(b"DINKSTER-VIDEO"):
        if len(data) > VIDEO_BYTE_LIMIT:
            raise ValueError("legacy VIDEO exceeds its byte limit")
        return video_from_source(data)
    if not data.startswith(_MAGIC) or len(data) < len(_MAGIC) + 8:
        raise ValueError("unsupported or truncated VIDEO version")
    length = struct.unpack_from("<Q", data, len(_MAGIC))[0]
    start = len(_MAGIC) + 8
    end = start + length
    if length > VIDEO_HEADER_LIMIT or end > len(data):
        raise ValueError("VIDEO header exceeds its bound or payload")
    header = mapping(
        json.loads(data[start:end], object_pairs_hook=_pairs, parse_constant=_reject_constant),
        "VIDEO header",
    )
    if set(header) != {"video", "chunks"} or not isinstance(header["chunks"], list):
        raise ValueError("invalid VIDEO header fields")
    if len(data) - end > VIDEO_BYTE_LIMIT:
        raise ValueError("VIDEO payload exceeds its limit")
    chunks: list[memoryview] = []
    for size in cast("list[object]", header["chunks"]):
        count = integer(size, "VIDEO chunk size", 0)
        if count > VIDEO_BYTE_LIMIT or end + count > len(data):
            raise ValueError("VIDEO chunk exceeds its payload")
        chunks.append(memoryview(data)[end : end + count])
        end += count
    if end != len(data) or len(data) - start - length > VIDEO_BYTE_LIMIT:
        raise ValueError("VIDEO payload has trailing bytes or exceeds its limit")
    used: set[int] = set()
    video = _unpack_video(header["video"], chunks, used, [0, 0])
    if len(used) != len(chunks):
        raise ValueError("VIDEO payload has undeclared chunks")
    return _materialize_video(video)


def video_fingerprint(type_id: str) -> Callable[[object], str]:
    def fingerprint(obj: object) -> str:
        if "timeline" in mapping(obj, "VIDEO"):
            return stable_hash([type_id.encode(), encode_video(obj)])
        return stable_hash([type_id.encode(), _MAGIC, _json(_pack_video(obj, [], identity=True))])

    return fingerprint


def video_meta(obj: object) -> Mapping[str, object]:
    value = coerce_video(obj)
    if "timeline" in value:
        from .timeline_video import timeline_meta

        return timeline_meta(value)
    refs: dict[str, dict[str, object]] = {}
    resident: dict[str, int] = {"ram": 0}
    byte_size = 0

    def visit(video: Mapping[str, object]) -> None:
        nonlocal resident, byte_size
        if "source" in video:
            source = video["source"]
            if isinstance(source, bytes):
                resident["ram"] += len(source)
                byte_size += len(source)
            else:
                ref = _asset_wire(source)
                refs[str(ref["digest"])] = ref
                byte_size += cast("int", ref["size"])
        else:
            components = mapping(video["components"], "components")
            image_cost = cast(
                "Mapping[str, int]", array_storage_meta(components["images"])[COST_META_KEY]
            )
            for residency, size in image_cost.items():
                resident[residency] = resident.get(residency, 0) + size
            if components["audio"] is not None:
                audio = audio_meta(components["audio"])
                for residency, size in mapping(audio["cost"], "audio cost").items():
                    resident[residency] = resident.get(residency, 0) + cast(int, size)
                for raw in cast("list[object]", audio["asset_refs"]):
                    ref = dict(mapping(raw, "audio asset reference"))
                    refs[str(ref["digest"])] = ref
                    byte_size += cast("int", ref["size"])
        for raw in cast("list[object]", video["edits"]):
            edit = mapping(raw, "edit")
            for child in cast("list[object]", edit.get("concat", [])):
                visit(mapping(child, "concat clip"))

    visit(value)
    return {
        "codec_version": 2,
        "storage_dtype": (
            "encoded"
            if "source" in value
            else storage_dtype(mapping(value["components"], "components")["images"])
        ),
        "container": mapping(value["probe"], "probe").get("container"),
        "byte_size": byte_size,
        "probe": json.loads(_json(value["probe"])),
        "effective": json.loads(_json(effective_video_facts(value))),
        "asset_refs": [refs[key] for key in sorted(refs)],
        COST_META_KEY: resident,
    }


def video_container(data: bytes | memoryview) -> str:
    return str(probe_video(bytes(data))["container"])


def video_rendition_mime(metadata: Mapping[str, object]) -> str:
    container = metadata.get("container")
    if not isinstance(container, str) or container not in _MIMES:
        raise ValueError("video rendition metadata container has no original encoded source")
    return _MIMES[container]


def render_video_original(obj: object) -> bytes:
    with open_video_source(video_source(obj)) as handle:
        data = handle.read(VIDEO_BYTE_LIMIT + 1)
    if len(data) > VIDEO_BYTE_LIMIT:
        raise ValueError("VIDEO original exceeds its byte limit")
    return data


def validate_video_encoded(data: bytes | memoryview, metadata: Mapping[str, object]) -> None:
    raw = bytes(data)
    if not raw.startswith(b"DINKSTER-VIDEO"):
        if metadata.get("container") != video_container(raw):
            raise ValueError("video metadata container does not match encoded bytes")
        size = metadata.get("byte_size")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError("video metadata byte_size must be an int")
        if size != len(raw):
            raise ValueError("video metadata byte_size does not match encoded bytes")
        return
    expected = video_meta(decode_video(raw))
    for key, value in expected.items():
        if key == COST_META_KEY:
            cost = metadata.get(key)
            if not isinstance(cost, Mapping):
                raise ValueError("VIDEO metadata cost must contain one RAM residency")
            typed_cost = cast("Mapping[object, object]", cost)
            if len(typed_cost) != 1:
                raise ValueError("VIDEO metadata cost must contain one RAM residency")
            residency, amount = next(iter(typed_cost.items()))
            expected_cost = cast("Mapping[str, int]", value)
            if (
                not isinstance(residency, str)
                or (residency != "ram" and not residency.startswith("ram@"))
                or residency == "ram@"
                or isinstance(amount, bool)
                or not isinstance(amount, int)
                or amount < 0
            ):
                raise ValueError("VIDEO metadata cost must contain one RAM residency")
            if amount not in (next(iter(expected_cost.values())), len(raw)):
                raise ValueError("VIDEO metadata cost does not match encoded payload")
            continue
        if key == "storage_dtype" and key not in metadata:
            continue
        if _json(metadata.get(key)) != _json(value):
            raise ValueError(f"VIDEO metadata {key} does not match encoded payload")
