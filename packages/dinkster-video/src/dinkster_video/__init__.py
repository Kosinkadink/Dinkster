"""Materialize portable VIDEO values without a full decoded frame batch."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .formats import (
    DITHERS,
    FRAME_FORMATS,
    read_video_metadata,
    save_frame_records,
    save_video_frames,
)
from .runtime import assemble_video, disassemble_video, save_video_stream

if TYPE_CHECKING:
    from . import document as timeline_document
    from . import image_math, timeline_runtime
    from . import timeline as timeline_render

_MODULES = {
    "image_math": "image_math",
    "timeline_document": "document",
    "timeline_render": "timeline",
    "timeline_runtime": "timeline_runtime",
}


def __getattr__(name: str) -> ModuleType:
    if name not in _MODULES:
        raise AttributeError(name)
    value = import_module(f".{_MODULES[name]}", __name__)
    globals()[name] = value
    return value


__all__ = [
    "DITHERS",
    "FRAME_FORMATS",
    "assemble_video",
    "disassemble_video",
    "read_video_metadata",
    "save_frame_records",
    "save_video_frames",
    "save_video_stream",
    "image_math",
    "timeline_document",
    "timeline_render",
    "timeline_runtime",
]
