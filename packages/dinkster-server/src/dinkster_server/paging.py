"""Query-bound keyset cursors, shared by every paged listing surface.

The frontend collection contract: listings are query-first, and a cursor
BINDS the query that produced it - silently paging a different query
would return wrong results, so a mismatch is a loud 400 and the client
restarts from page one. The cursor is opaque to clients; its encoding is
free to change between server versions.

``bound`` is the flattened query (every parameter that shapes the result
set), ``after`` the keyset position: (sort key, tiebreak id) of the
previous page's last row, for ORDER BY sort DESC, id DESC pagination.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import cast

from aiohttp import web


def encode_cursor(bound: Mapping[str, str], after: tuple[float, str]) -> str:
    payload = json.dumps({"b": dict(bound), "m": after[0], "i": after[1]})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode_cursor(cursor: str, bound: Mapping[str, str]) -> tuple[float, str]:
    """The keyset position a cursor carries - if it was minted for exactly
    this query. Tampered/malformed cursors and query mismatches are 400."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded))
        after = (float(data["m"]), str(data["i"]))
        cursor_bound = cast(object, data["b"])
        if not isinstance(cursor_bound, dict):
            raise TypeError("cursor bound must be an object")
        entries = cast("dict[object, object]", cursor_bound).items()
        matches = {str(k): str(v) for k, v in entries} == dict(bound)
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "malformed cursor"}),
            content_type="application/json",
        ) from exc
    if not matches:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "cursor does not match this query"}),
            content_type="application/json",
        )
    return after
