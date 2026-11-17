"""Saved managed launch selection, separate from retained software and grants."""

from __future__ import annotations

import re
from typing import TypedDict, cast

from .environment_state import EnvironmentError


class PackSelection(TypedDict):
    allow: list[str] | None
    excludePartners: bool


def read_pack_selection(value: object) -> PackSelection:
    if not isinstance(value, dict):
        raise EnvironmentError("packSelection must be an object")
    fields = cast("dict[str, object]", value)
    if set(fields) - {"allow", "excludePartners"}:
        raise EnvironmentError("unknown packSelection setting")
    allowed = fields.get("allow")
    if allowed is not None:
        if not isinstance(allowed, list) or any(
            not isinstance(pack, str) or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", pack) is None
            for pack in allowed
        ):
            raise EnvironmentError(
                "packSelection allow must be null or an array of canonical pack ids"
            )
        if len(set(allowed)) != len(allowed):
            raise EnvironmentError("packSelection allow must contain distinct pack ids")
    excluded = fields.get("excludePartners", False)
    if type(excluded) is not bool:
        raise EnvironmentError("packSelection excludePartners must be a boolean")
    return {
        "allow": list(cast("list[str]", allowed)) if allowed is not None else None,
        "excludePartners": cast("bool", excluded),
    }
