"""VIDEO v3 document variant; no source probe claims or inline media chunks."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

from .video_document import (
    TimelineError,
    document,
    document_meta,
    effective_document,
    encode_document,
    item_duration,
)
from .video_edits import mapping, seconds

TIMELINE_MAGIC = b"DINKSTER-VIDEO\x03"


class TimelineVideo(dict[str, object]):
    """The local resolver is not a wire field and grants no authority until used."""

    def __init__(self, value: object, factory: Callable[[Mapping[str, object]], Any] | None = None):
        super().__init__(timeline=document(value))
        self.factory = factory


def coerce_timeline_video(value: object) -> TimelineVideo:
    obj = mapping(value, "VIDEO v3")
    if set(obj) != {"timeline"}:
        raise TimelineError("invalid_document", "$", "VIDEO v3 has exactly one timeline field")
    factory = value.factory if isinstance(value, TimelineVideo) else None
    return TimelineVideo(obj["timeline"], factory)


def timeline_facts(value: object) -> dict[str, object]:
    obj = effective_document(mapping(value, "VIDEO v3")["timeline"])
    settings = obj["settings"]
    length = item_duration(obj["timeline"]["tracks"])
    rate = seconds(settings["rate"], "rate")
    return {
        "width": settings["width"],
        "height": settings["height"],
        "duration": length,
        "fps": rate,
        "frame_count": math.ceil(length * rate),
        "frame_count_kind": "estimated",
    }


def timeline_meta(value: object) -> dict[str, object]:
    obj = coerce_timeline_video(value)
    facts = {
        key: [fact.numerator, fact.denominator] if isinstance(fact, Fraction) else fact
        for key, fact in timeline_facts(obj).items()
    }
    return {
        "codec_version": 3,
        "representation": "timeline",
        "container": None,
        "byte_size": 0,
        "effective": facts,
        "asset_refs": document_meta(obj["timeline"])["asset_refs"],
        "cost": {"ram": len(encode_document(obj["timeline"]))},
    }
