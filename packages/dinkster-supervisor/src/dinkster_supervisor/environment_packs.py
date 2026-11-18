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
    names: list[str] | None = None
    if allowed is not None:
        if not isinstance(allowed, list):
            raise EnvironmentError(
                "packSelection allow must be null or an array of canonical pack ids"
            )
        values = cast("list[object]", allowed)
        if any(
            not isinstance(pack, str) or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", pack) is None
            for pack in values
        ):
            raise EnvironmentError("packSelection allow must contain canonical pack ids")
        names = list(cast("list[str]", values))
        if len(set(names)) != len(names):
            raise EnvironmentError("packSelection allow must contain distinct pack ids")
    excluded = fields.get("excludePartners", False)
    if type(excluded) is not bool:
        raise EnvironmentError("packSelection excludePartners must be a boolean")
    return {"allow": names, "excludePartners": excluded}
