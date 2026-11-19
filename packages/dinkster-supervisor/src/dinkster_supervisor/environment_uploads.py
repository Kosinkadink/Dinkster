"""Managed environment upload policy, independent of engine media dependencies."""

from __future__ import annotations

from typing import cast

from .environment_state import EnvironmentError


def read_upload_limits(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise EnvironmentError("uploadLimits must be an object")
    fields = cast("dict[str, object]", value)
    if set(fields) - {"generic", "media/image", "media/audio", "media/video", "media/model3d"}:
        raise EnvironmentError("unknown uploadLimits setting")
    for kind, limit in fields.items():
        if type(limit) is not int or limit <= 0:
            raise EnvironmentError(f"uploadLimits {kind} must be a positive integer byte count")
    return cast("dict[str, int]", fields.copy())
