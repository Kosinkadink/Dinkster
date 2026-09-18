"""Source-only OTIO-shaped timeline documents; importing never resolves media."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import Any, cast

from .model import stable_hash
from .video_codec import asset_reference
from .video_codec import source_video as source_video
from .video_codec import video_reference as video_reference
from .video_edits import effective_video_facts, integer, mapping, seconds, trim_window

DOCUMENT_TYPE = "dinkster.video_document"
DOCUMENT_LIMIT = 1024 * 1024
MAX_ITEMS = 4096
MAX_DEPTH = 16
MEDIA_TYPES = frozenset({"comfy.VIDEO", "dinkster.image", "comfy.AUDIO", "dinkster.layers"})


class TimelineError(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        self.code, self.path = code, path
        super().__init__(f"{code} at {path}: {message}")


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TimelineError("invalid_document", key, "duplicate JSON key")
        result[key] = value
    return result


def json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def parse_json(data: str | bytes) -> Any:
    if len(data) >= DOCUMENT_LIMIT:
        raise TimelineError("document_limit", "$", "document must be below 1 MiB")
    try:
        result = json.loads(data, object_pairs_hook=_pairs)
    except (RecursionError, UnicodeError, json.JSONDecodeError) as exc:
        raise TimelineError("invalid_document", "$", "invalid JSON") from exc
    _data(result)
    return result


def _data(value: object, depth: int = 0) -> None:
    if depth > 64:
        raise TimelineError("document_limit", "$", "JSON nesting exceeds 64")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for child in cast(list[object], value):
            _data(child, depth + 1)
        return
    if isinstance(value, dict) and all(
        isinstance(k, str) for k in cast(dict[object, object], value)
    ):
        for child in cast(dict[str, object], value).values():
            _data(child, depth + 1)
        return
    raise TimelineError("invalid_document", "$", "only finite JSON data is allowed")


def rational_time(value: object, rate: object = 1) -> dict[str, object]:
    return {"OTIO_SCHEMA": "RationalTime.1", "value": value, "rate": rate}


def time_range(start: object, duration: object, rate: object = 1) -> dict[str, object]:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "start_time": rational_time(start, rate),
        "duration": rational_time(duration, rate),
    }


def time_seconds(value: object) -> Fraction:
    obj = mapping(value, "RationalTime")
    if obj.get("OTIO_SCHEMA") != "RationalTime.1":
        raise TimelineError("invalid_time", "$", "expected RationalTime.1")
    rate = seconds(obj.get("rate"), "rate")
    if rate <= 0:
        raise TimelineError("invalid_time", "$", "rate must be positive")
    return seconds(obj.get("value"), "value") / rate


def range_seconds(value: object) -> tuple[Fraction, Fraction]:
    obj = mapping(value, "TimeRange")
    if obj.get("OTIO_SCHEMA") != "TimeRange.1":
        raise TimelineError("invalid_time", "$", "expected TimeRange.1")
    start, duration = time_seconds(obj.get("start_time")), time_seconds(obj.get("duration"))
    if duration < 0:
        raise TimelineError("invalid_time", "$", "negative duration")
    return start, duration


def schema(item: Mapping[str, Any]) -> str:
    return str(item.get("OTIO_SCHEMA", "")).split(".")[0]


def extension(item: Mapping[str, Any]) -> dict[str, Any]:
    return dict(
        mapping(
            mapping(item.get("metadata", {}), "metadata").get("dinkster", {}), "metadata.dinkster"
        )
    )


def set_extension(item: dict[str, Any], **fields: object) -> None:
    item.setdefault("metadata", {})["dinkster"] = {**extension(item), **fields}


def walk(item: dict[str, Any], path: str = "tracks") -> Iterator[tuple[str, dict[str, Any]]]:
    yield path, item
    for index, child in enumerate(item.get("children", [])):
        yield from walk(child, f"{path}.children[{index}]")


def source_origin(item: Mapping[str, Any]) -> Fraction:
    refs = item.get("media_references", {})
    reference = refs.get(
        item.get("active_media_reference_key", "DEFAULT_MEDIA"), item.get("media_reference", {})
    )
    available = reference.get("available_range")
    return range_seconds(available)[0] if available is not None else Fraction(0)


def item_duration(item: Mapping[str, Any]) -> Fraction:
    if schema(item) == "Transition":
        return Fraction(0)
    if item.get("source_range") is not None:
        return range_seconds(item["source_range"])[1]
    durations = [item_duration(c) for c in item.get("children", [])]
    if schema(item) == "Stack":
        return max(durations, default=Fraction(0))
    if schema(item) == "Track":
        return sum(durations, Fraction(0))
    if schema(item) == "Clip":
        refs = item.get("media_references", {})
        reference = refs.get(item.get("active_media_reference_key", "DEFAULT_MEDIA"), {})
        available = reference.get("available_range")
        if available is not None:
            return range_seconds(available)[1]
    raise TimelineError("unknown_duration", "$", "item needs source_range or available_range")


def _tree(item: object, depth: int, budget: list[int]) -> None:
    obj = mapping(item, "timeline item")
    budget[0] += 1
    if depth > MAX_DEPTH or budget[0] > MAX_ITEMS:
        raise TimelineError("document_limit", "$", "too many items or nested compositions")
    kind = schema(obj)
    allowed_versions = {
        "Timeline.1",
        "Stack.1",
        "Track.1",
        "Clip.1",
        "Clip.2",
        "Gap.1",
        "Transition.1",
    }
    if obj.get("OTIO_SCHEMA") not in allowed_versions:
        raise TimelineError("unsupported_schema", "$", str(obj.get("OTIO_SCHEMA")))
    if obj.get("source_range") is not None:
        range_seconds(obj["source_range"])
    ext = extension(obj)
    if "video_edit" in ext:
        mapping(ext["video_edit"], "VIDEO_EDIT")
    if "strict_duration" in ext and not isinstance(ext["strict_duration"], bool):
        raise TimelineError("invalid_edit", "$", "strict_duration must be boolean")
    if kind == "Timeline":
        if schema(mapping(obj.get("tracks"), "tracks")) != "Stack":
            raise TimelineError("invalid_document", "tracks", "Timeline requires a Stack")
        _tree(obj["tracks"], depth + 1, budget)
    elif kind in ("Stack", "Track"):
        if kind == "Track" and obj.get("kind") not in ("Video", "Audio"):
            raise TimelineError("invalid_document", "$", "Track kind must be Video or Audio")
        children = obj.get("children")
        if not isinstance(children, list):
            raise TimelineError("invalid_document", "$", "children must be a list")
        for child in cast(list[object], children):
            child_kind = schema(mapping(child, "child"))
            if child_kind == "Timeline" or (kind == "Stack" and child_kind == "Transition"):
                raise TimelineError("invalid_document", "$", "illegal child composition")
            _tree(child, depth + 1, budget)
    elif kind == "Transition":
        for key in ("in_offset", "out_offset"):
            if time_seconds(obj.get(key)) < 0:
                raise TimelineError("invalid_time", "$", "negative transition offset")


def document(value: object) -> dict[str, Any]:
    """Return a validated detached document; no caller-owned mutable state survives."""
    _data(value)
    data = json_bytes(value)
    if len(data) >= DOCUMENT_LIMIT:
        raise TimelineError("document_limit", "$", "document must be below 1 MiB")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise TimelineError("invalid_document", "$", "expected a document object")
    obj = cast(dict[str, Any], parsed)
    if set(obj) != {"version", "timeline", "sources", "settings"}:
        raise TimelineError("invalid_document", "$", "expected version/timeline/sources/settings")
    if type(obj["version"]) is not int or obj["version"] != 1:
        raise TimelineError("unsupported_version", "$", "expected document version 1")
    _tree(obj["timeline"], 0, [0])
    if obj["timeline"]["OTIO_SCHEMA"] != "Timeline.1":
        raise TimelineError("invalid_document", "timeline", "expected Timeline.1 root")
    sources = mapping(obj["sources"], "sources")
    if len(sources) > 256:
        raise TimelineError("document_limit", "sources", "at most 256 declared sources")
    for key, source in sources.items():
        ref = mapping(source, key)
        if ref.get("type") == "comfy.VIDEO" and set(ref) == {"type", "video"}:
            source_video(ref)
        elif set(ref) == {"type", "asset"} and ref["type"] in MEDIA_TYPES:
            asset_reference(ref["asset"])
        elif set(ref) == {"type", "asset", "resources"} and ref["type"] == "dinkster.layers":
            asset_reference(ref["asset"])
            if not isinstance(ref["resources"], list):
                raise TimelineError("invalid_source", key, "resources must be declared assets")
            for resource in cast(list[object], ref["resources"]):
                asset_reference(resource)
        else:
            raise TimelineError("invalid_source", key, "expected declared media type and asset")
    settings = mapping(obj["settings"], "settings")
    if set(settings) != {"width", "height", "rate"}:
        raise TimelineError("invalid_document", "settings", "expected width/height/rate")
    for key in ("width", "height"):
        size = integer(settings[key], key, 1)
        if size > 8192:
            raise TimelineError("document_limit", key, "dimension exceeds 8192")
    rate = time_seconds(rational_time(settings["rate"]))
    if not 0 < rate <= 240:
        raise TimelineError("invalid_time", "settings.rate", "rate must be in (0,240]")
    return obj


def effective_document(value: object) -> dict[str, Any]:
    """Project widget selections at execution/export, never rewriting authored widget data."""
    obj = document(value)
    sources = obj["sources"]
    for _, item in walk(obj["timeline"]["tracks"]):
        ext = extension(item)
        if schema(item) != "Clip" or "video_edit" not in ext:
            continue
        reference = sources.get(ext.get("source", ""))
        length = None
        if isinstance(reference, Mapping) and "video" in reference:
            length = effective_video_facts(source_video(cast(Mapping[str, Any], reference)))[
                "duration"
            ]
        if length is None:
            refs = item.get("media_references", {})
            available = refs.get(item.get("active_media_reference_key", "DEFAULT_MEDIA"), {}).get(
                "available_range"
            )
            if available is not None:
                length = range_seconds(available)[1]
        if length is None:
            raise TimelineError("unknown_duration", "$", "widget clip requires base media duration")
        trim_value = ext["video_edit"].get("trim")
        trim = mapping({} if trim_value is None else trim_value, "VIDEO_EDIT trim")
        start, selected = trim_window(
            cast(Fraction, length),
            trim.get("start_time", 0),
            trim.get("duration", 0),
            ext.get("strict_duration", False),
        )
        assert selected is not None
        start += source_origin(item)
        ticks = math.lcm(start.denominator, selected.denominator)
        item["source_range"] = time_range(int(start * ticks), int(selected * ticks), ticks)
    return obj


def encode_document(value: object) -> bytes:
    return json_bytes(document(value))


def decode_document(data: bytes) -> dict[str, Any]:
    return document(parse_json(data))


def document_meta(value: object) -> dict[str, object]:
    from .video_codec import video_meta

    obj = document(value)
    refs: dict[str, Any] = {}
    for ref in obj["sources"].values():
        assets = (
            video_meta(source_video(ref))["asset_refs"]
            if "video" in ref
            else [ref["asset"], *ref.get("resources", [])]
        )
        for asset in cast(list[dict[str, Any]], assets):
            refs[asset["digest"]] = asset
    return {"version": 1, "asset_refs": [refs[k] for k in sorted(refs)]}


def document_fingerprint(value: object) -> str:
    return stable_hash([DOCUMENT_TYPE.encode(), encode_document(value)])
