"""Pure timeline mutations and lossless OTIO JSON interchange."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

from dinkster_values.video_document import (
    TimelineError,
    document,
    effective_document,
    extension,
    item_duration,
    json_bytes,
    parse_json,
    range_seconds,
    rational_time,
    schema,
    set_extension,
    source_origin,
    time_range,
)
from dinkster_values.video_edits import integer, seconds, trim_window

EDITOR_COMMANDS = (
    "make",
    "add_clip",
    "add_track",
    "set_effect",
    "transition",
    "retime",
    "mix_audio",
    "split",
    "move",
    "trim",
    "ripple",
    "roll",
    "bind_source",
    "render",
    "import_otio",
    "export_otio",
)
COMMAND_NODES = {command: f"dinkster.video_document.{command}" for command in EDITOR_COMMANDS}


def make(
    *,
    width: int = 1280,
    height: int = 720,
    rate: float = 24,
    name: str = "Timeline",
    clips: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return document(
        {
            "version": 1,
            "sources": {},
            "settings": {"width": width, "height": height, "rate": rate},
            "timeline": {
                "OTIO_SCHEMA": "Timeline.1",
                "name": name,
                "metadata": {},
                "global_start_time": None,
                "tracks": {
                    "OTIO_SCHEMA": "Stack.1",
                    "name": "tracks",
                    "metadata": {},
                    "source_range": None,
                    "effects": [],
                    "markers": [],
                    "children": [_track("Video", "Video", clips or [])],
                },
            },
        }
    )


def _track(kind: str, name: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "Track.1",
        "name": name,
        "kind": kind,
        "children": children,
        "source_range": None,
        "effects": [],
        "markers": [],
        "metadata": {},
    }


def clip(
    source: str,
    *,
    start: float = 0,
    duration: float,
    name: str = "Clip",
    video_edit: dict[str, Any] | None = None,
    strict_duration: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "OTIO_SCHEMA": "Clip.2",
        "name": name,
        "source_range": time_range(start, duration),
        "effects": [],
        "markers": [],
        "metadata": {},
        "active_media_reference_key": "DEFAULT_MEDIA",
        "media_references": {
            "DEFAULT_MEDIA": {
                "OTIO_SCHEMA": "MissingReference.1",
                "name": "",
                "metadata": {},
                "available_range": time_range(0, duration),
                "available_image_bounds": None,
            }
        },
    }
    set_extension(result, source=source)
    if video_edit is not None:
        set_extension(result, video_edit=video_edit, strict_duration=strict_duration)
    return result


def import_otio(
    text: str, *, width: int = 1280, height: int = 720, rate: float = 24
) -> dict[str, Any]:
    timeline = parse_json(text)
    envelope = extension(timeline).get("document", {})
    if not envelope:
        import importlib

        otio: Any = importlib.import_module("opentimelineio")
        # OTIO's decimal parser differs from Python for some Resolve float spellings.
        timeline = parse_json(
            otio.core.serialize_json_to_string(otio.core.deserialize_json_from_string(text))
        )
    if envelope:
        metadata = timeline["metadata"]["dinkster"]
        del metadata["document"]
        if not metadata:
            del timeline["metadata"]["dinkster"]
    return document(
        {
            "version": 1,
            "timeline": timeline,
            "sources": envelope.get("sources", {}),
            "settings": envelope.get("settings", {"width": width, "height": height, "rate": rate}),
        }
    )


def export_otio(value: object) -> str:
    obj = effective_document(value)
    timeline = obj["timeline"]
    # Foreign documents with no bindings are returned without adding Dinkster metadata.
    if obj["sources"]:
        set_extension(
            timeline,
            document={"version": 1, "sources": obj["sources"], "settings": obj["settings"]},
        )
    return json_bytes(timeline).decode()


def _select(obj: dict[str, Any], params: dict[str, Any]) -> tuple[dict[str, Any], int]:
    index = integer(params.get("track", 0), "track", 0)
    tracks = obj["timeline"]["tracks"]["children"]
    if index >= len(tracks) or schema(tracks[index]) != "Track":
        raise TimelineError("invalid_selection", "track", "not a top-level track")
    return tracks[index], integer(params.get("clip", 0), "clip", 0)


def _set_range(item: dict[str, Any], start: Fraction, duration: Fraction) -> None:
    if start < source_origin(item) or duration <= 0:
        raise TimelineError("invalid_edit", "source_range", "empty or negative source selection")
    # Choose an integer tick rate that represents both rationals exactly.
    import math

    rate = math.lcm(start.denominator, duration.denominator)
    item["source_range"] = time_range(int(start * rate), int(duration * rate), rate)
    ext = extension(item)
    if "video_edit" in ext:
        widget = ext["video_edit"]
        widget["trim"] = {
            **(widget.get("trim") or {}),
            "start_time": float(start - source_origin(item)),
            "duration": float(duration),
        }
        set_extension(item, video_edit=widget)


def playback_speed(item: dict[str, Any]) -> Fraction:
    speed = Fraction(1)
    for effect in item.get("effects", []):
        if schema(effect) == "FreezeFrame":
            speed = Fraction(0)
        elif schema(effect) == "LinearTimeWarp":
            speed *= seconds(effect["time_scalar"], "time_scalar")
    return speed


def _gap(length: Fraction) -> dict[str, Any]:
    result: dict[str, Any] = {
        "OTIO_SCHEMA": "Gap.1",
        "name": "",
        "metadata": {},
        "effects": [],
        "markers": [],
    }
    _set_range(result, Fraction(0), length)
    return result


def mutate(value: object, command: str, params: dict[str, Any]) -> dict[str, Any]:
    obj = (
        effective_document(value)
        if command in ("split", "roll", "ripple")
        or (command == "trim" and "video_edit" not in params)
        else document(value)
    )
    if command == "add_track":
        track = _track(params.get("kind", "Video"), params.get("name", "Track"), [])
        set_extension(track, blend=params.get("blend", "normal"), opacity=params.get("opacity", 1))
        obj["timeline"]["tracks"]["children"].append(track)
        return document(obj)
    if command == "bind_source":
        obj["sources"][params["source"]] = params["reference"]
        if "track" in params:
            track, index = _select(obj, params)
            set_extension(track["children"][index], source=params["source"])
        return document(obj)
    track, index = _select(obj, params)
    children = track["children"]
    if command == "add_clip":
        child = params["item"]
        at = integer(params.get("index", len(children)), "index", 0)
        if at > len(children):
            raise TimelineError("invalid_selection", "index", "destination out of range")
        children.insert(at, child)
    elif command == "mix_audio":
        set_extension(track, audio_mix=params["audio_mix"])
    else:
        if index >= len(children):
            raise TimelineError("invalid_selection", "clip", "child index out of range")
        item = children[index]
        if command == "set_effect":
            effects = extension(item).get("effects", [])
            at = integer(params.get("index", len(effects)), "effect index", 0)
            if at > len(effects):
                raise TimelineError("invalid_selection", "effect", "index out of range")
            if params.get("remove", False):
                if at == len(effects):
                    raise TimelineError("invalid_selection", "effect", "index out of range")
                del effects[at]
            elif at == len(effects):
                effects.append(params["effect"])
            else:
                effects[at] = params["effect"]
            set_extension(item, effects=effects)
        elif command == "retime":
            scalar = seconds(params["scalar"], "scalar")
            effect: dict[str, Any] = {
                "OTIO_SCHEMA": "FreezeFrame.1" if scalar == 0 else "LinearTimeWarp.1",
                "name": "",
                "effect_name": "FreezeFrame" if scalar == 0 else "LinearTimeWarp",
                "time_scalar": float(scalar),
                "metadata": {},
            }
            item["effects"] = [
                e
                for e in item.get("effects", [])
                if schema(e) not in ("FreezeFrame", "LinearTimeWarp")
            ] + [effect]
        elif command == "transition":
            if index + 1 >= len(children) or any(
                schema(c) != "Clip" for c in children[index : index + 2]
            ):
                raise TimelineError("invalid_transition", "clip", "requires two adjacent clips")
            transition = {
                "OTIO_SCHEMA": "Transition.1",
                "name": "Dissolve",
                "metadata": {},
                "transition_type": "SMPTE_Dissolve",
                "in_offset": rational_time(params.get("in_offset", 0.5)),
                "out_offset": rational_time(params.get("out_offset", 0.5)),
            }
            children.insert(index + 1, transition)
        elif command == "move":
            destination, _ = _select(obj, {"track": params.get("to_track", params.get("track", 0))})
            at = integer(params["index"], "index", 0)
            if at > len(destination["children"]):
                raise TimelineError("invalid_selection", "index", "destination out of range")
            if any(schema(c) == "Transition" for c in children[max(0, index - 1) : index + 2]):
                raise TimelineError(
                    "invalid_transition", "clip", "remove adjacent transitions first"
                )
            destination["children"].insert(at, children.pop(index))
        elif command == "trim" and "video_edit" in params:
            if "video_edit" not in extension(item) and item.get("source_range") is not None:
                start, length = range_seconds(item["source_range"])
                start -= source_origin(item)
                widget = {"trim": {"start_time": float(start), "duration": float(length)}}
                widget.update(params["video_edit"])
            else:
                widget = params["video_edit"]
            set_extension(
                item, video_edit=widget, strict_duration=params.get("strict_duration", False)
            )
        elif command in ("trim", "ripple", "roll", "split"):
            if schema(item) not in ("Clip", "Gap"):
                raise TimelineError("invalid_selection", "clip", "operation requires Clip or Gap")
            start, duration = range_seconds(item["source_range"])
            speed = playback_speed(item)
            if command == "split":
                at = seconds(params["position"], "position")
                if not 0 < at < duration:
                    raise TimelineError("invalid_edit", "position", "split must be inside clip")
                other = deepcopy(item)
                _set_range(item, start, at)
                _set_range(other, start + at * speed, duration - at)
                children.insert(index + 1, other)
            elif command == "roll":
                if index + 1 >= len(children) or schema(children[index + 1]) != "Clip":
                    raise TimelineError("invalid_selection", "clip", "roll needs adjacent Clip")
                delta = seconds(params["delta"], "delta")
                other = children[index + 1]
                other_start, other_duration = range_seconds(other["source_range"])
                _set_range(item, start, duration + delta)
                _set_range(
                    other, other_start + delta * playback_speed(other), other_duration - delta
                )
            else:
                offset, selected = trim_window(
                    duration,
                    params.get("start_time", 0),
                    params.get("duration", 0),
                    params.get("strict_duration", False),
                )
                assert selected is not None
                _set_range(item, start + offset * speed, selected)
                if command == "trim" and selected < duration:
                    if duration - offset - selected > 0:
                        children.insert(index + 1, _gap(duration - offset - selected))
                    if offset > 0:
                        children.insert(index, _gap(offset))
        else:
            raise TimelineError("unknown_command", "$", command)
    return document(obj)


def clip_duration(item: dict[str, Any]) -> Fraction:
    return item_duration(item)
