"""Path redaction for outbound wire payloads.

Wire text (error messages, tracebacks, pack failure reports, node-reported
event data) must not publish the host machine's filesystem layout. One
redactor, built from the roots the server knows at construction time,
rewrites absolute paths in outbound text to stable tokens:

- a known root prefix becomes its token: ``<install>/...``, ``<comfy>/...``,
  ``<library>/...``, ``<mount:models>/...``, ``<tmp>/...``, ``<home>/...``;
- any other absolute path that exists on this machine collapses to
  ``<path>/<basename>``;
- absolute-looking strings that are NOT local filesystem paths (URL routes
  like ``/api/nodes``, virtual paths) pass through untouched.

Redaction never raises: it is a text transform on data already destined for
the wire, and a redaction bug must not turn a reportable error into a crash.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

__all__ = ["PathRedactor"]

# The left boundary shared by every path pattern: not mid-word (URL hosts,
# unrelated longer paths like /var/tmp), not after ``:`` (schemes, drive
# letters), ``.``/``~`` (relative or home-relative spellings), and not
# after a separator (repeated separators like ``/var//tmp`` and URL ``://``
# never start a new path). Text already rewritten to a token
# (``<home>/tmp/x``) is protected by the substitution callbacks instead,
# so a path after an unrelated ``>`` (shell redirects) still redacts.
_LEFT_BOUNDARY = r"(?<![\w:.~/\\])"

# An absolute path candidate in prose. Segments are conservative (no
# spaces): known-root prefix rewriting above this fallback handles roots
# containing spaces.
_POSIX_PATH = re.compile(_LEFT_BOUNDARY + r"/(?:[\w.+@%-]+/)+[\w.+@%-]+/?")
_WINDOWS_PATH = re.compile(_LEFT_BOUNDARY + r"[A-Za-z]:[\\/](?:[\w.+@%\- ]+[\\/])*[\w.+@%-]+[\\/]?")

_WINDOWS_ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]")


def _variants(root: Path) -> set[str]:
    """The textual spellings a root can leak under: as given (only when
    absolute - a relative spelling like ``.`` or ``models`` would rewrite
    ordinary prose) and resolved (symlinked temp dirs, ``/var`` vs
    ``/private/var``)."""
    spellings: set[str] = set()
    raw = str(root)
    if root.is_absolute() or _WINDOWS_ABSOLUTE.match(raw):
        spellings.add(raw)
    try:
        spellings.add(str(root.resolve()))
    except OSError:
        pass
    return {s.rstrip("/\\") for s in spellings if s not in ("", "/", os.sep)}


class PathRedactor:
    """Rewrites absolute filesystem paths in outbound text to stable tokens."""

    def __init__(self, roots: Iterable[tuple[str, Path]] = ()) -> None:
        expanded: list[tuple[str, str]] = []
        for token, root in roots:
            for spelling in _variants(root):
                expanded.append((token, spelling))
        for token, default_root in (
            ("<tmp>", Path(tempfile.gettempdir())),
            ("<home>", Path.home()),
        ):
            for spelling in _variants(default_root):
                expanded.append((token, spelling))
        # Longest spelling first so a nested root (library under home) maps
        # to its most specific token.
        expanded.sort(key=lambda pair: len(pair[1]), reverse=True)
        self._prefixes: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
            # The prefix must start at a path boundary (so /tmp never
            # rewrites inside /var/tmp or a URL) and end at a separator or
            # end-of-path (so /home/user never rewrites inside
            # /home/username).
            (
                token,
                re.compile(_LEFT_BOUNDARY + re.escape(spelling) + r"(?=[/\\]|$|[^\w.+@%-])"),
            )
            for token, spelling in expanded
        )
        # The tokens this instance can emit. A match directly after one is
        # the tail of an already-redacted path, never a fresh path.
        self._tokens: tuple[str, ...] = tuple({token for token, _ in expanded} | {"<path>"})

    def _follows_token(self, match: re.Match[str]) -> bool:
        return match.string.endswith(self._tokens, 0, match.start())

    def redact_text(self, text: str) -> str:
        if "/" not in text and "\\" not in text:
            return text
        for token, pattern in self._prefixes:
            text = pattern.sub(lambda m, t=token: m.group(0) if self._follows_token(m) else t, text)
        text = _POSIX_PATH.sub(self._fallback, text)
        return _WINDOWS_PATH.sub(self._fallback, text)

    def redact_value(self, value: object) -> object:
        """Recursively redact every string inside a JSON-shaped value.
        Non-string scalars (and binary blobs) pass through unchanged."""
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            mapping = cast("Mapping[object, object]", value)
            return {key: self.redact_value(item) for key, item in mapping.items()}
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            sequence = cast("Sequence[object]", value)
            return [self.redact_value(item) for item in sequence]
        return value

    def _fallback(self, match: re.Match[str]) -> str:
        # Only rewrite strings that are demonstrably local filesystem paths;
        # route-like strings (/api/nodes) do not exist on disk and pass.
        if self._follows_token(match):
            return match.group(0)
        candidate = match.group(0)
        stripped = candidate.rstrip("/\\")
        try:
            if not (os.path.exists(stripped) or os.path.exists(os.path.dirname(stripped))):
                return candidate
        except OSError:
            return candidate
        return "<path>/" + os.path.basename(stripped)
