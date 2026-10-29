"""UAX #29 extended grapheme cluster segmentation, torch-free.

The T5 tokenizer's Precompiled normalizer iterates extended grapheme
clusters (t5_spm). Hugging Face tokenizers 0.22.2 does this through
the unicode-segmentation 1.12.0 crate (``graphemes(true)``), whose
tables come from Unicode 16.0.0 - so this module implements the UAX
#29 extended rules (GB1-GB13 + GB9c/InCB + GB11/emoji, Unicode 16)
over vendored tables generated from the authoritative UCD files by
tools/gen_grapheme_tables.py (provenance hashes recorded there).

Like the crate, surrogates classify as Any (Rust ``char`` cannot
represent them, so the crate's tables never contain them); Python
strings can carry lone surrogates, and they segment as Any here too.

Conformance is pinned against the full UCD GraphemeBreakTest.txt
(tests/goldens/grapheme_break_test.txt.gz, every test line), not a
sample.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from functools import lru_cache

from .vendored import read_vendored

#: sha256 of the UNCOMPRESSED vendored table JSON (printed by
#: tools/gen_grapheme_tables.py when regenerated).
GRAPHEME_BREAK_SHA256 = "dbcc2617a4a84113292047998c50ef5fe34569d6e2027ca4c39ce9be79fca48f"

# Category codes (module-internal; Any = no table entry).
_ANY = 0
_CR = 1
_LF = 2
_CONTROL = 3
_EXTEND = 4
_ZWJ = 5
_RI = 6
_PREPEND = 7
_SPACINGMARK = 8
_L = 9
_V = 10
_T = 11
_LV = 12
_LVT = 13
_EXT_PICT = 14
_INCB_CONSONANT = 15

_CATEGORY_CODES = {
    "CR": _CR,
    "LF": _LF,
    "Control": _CONTROL,
    "Extend": _EXTEND,
    "ZWJ": _ZWJ,
    "Regional_Indicator": _RI,
    "Prepend": _PREPEND,
    "SpacingMark": _SPACINGMARK,
    "L": _L,
    "V": _V,
    "T": _T,
    "LV": _LV,
    "LVT": _LVT,
    "Extended_Pictographic": _EXT_PICT,
    "InCB_Consonant": _INCB_CONSONANT,
}


class _RangeTable:
    """Sorted disjoint codepoint ranges -> small-int values, by
    binary search."""

    def __init__(self, ranges: list[tuple[int, int, int]]) -> None:
        ranges.sort()
        self._starts = [lo for lo, _, _ in ranges]
        self._ends = [hi for _, hi, _ in ranges]
        self._values = [value for _, _, value in ranges]

    def get(self, cp: int) -> int:
        i = bisect_right(self._starts, cp) - 1
        if i >= 0 and cp <= self._ends[i]:
            return self._values[i]
        return _ANY


class _RangeSet:
    """Sorted disjoint codepoint ranges membership, by binary search."""

    def __init__(self, ranges: list[list[int]]) -> None:
        ranges.sort()
        self._starts = [lo for lo, _ in ranges]
        self._ends = [hi for _, hi in ranges]

    def __contains__(self, cp: int) -> bool:
        i = bisect_right(self._starts, cp) - 1
        return i >= 0 and cp <= self._ends[i]


class GraphemeTables:
    """The vendored Unicode 16.0.0 tables; share one instance via
    :func:`load_grapheme_tables`."""

    def __init__(
        self,
        categories: _RangeTable,
        incb_extend: _RangeSet,
        incb_linker: _RangeSet,
        unicode_version: str,
    ) -> None:
        self.categories = categories
        self.incb_extend = incb_extend
        self.incb_linker = incb_linker
        self.unicode_version = unicode_version

    @classmethod
    def from_vendored_data(cls) -> GraphemeTables:
        data = json.loads(read_vendored("grapheme_break.json.gz", GRAPHEME_BREAK_SHA256))
        flat: list[tuple[int, int, int]] = []
        for name, ranges in data["categories"].items():
            code = _CATEGORY_CODES[name]
            flat.extend((lo, hi, code) for lo, hi in ranges)
        return cls(
            _RangeTable(flat),
            _RangeSet(data["incb_extend"]),
            _RangeSet(data["incb_linker"]),
            data["unicode_version"],
        )


@lru_cache(maxsize=1)
def load_grapheme_tables() -> GraphemeTables:
    """The shared vendored-table instance (parsing is done once)."""
    return GraphemeTables.from_vendored_data()


def grapheme_boundaries(text: str) -> list[int]:
    """The extended-grapheme-cluster boundary indices of ``text``:
    every index where a cluster starts, plus ``len(text)``. Empty
    text has the single boundary 0."""
    tables = load_grapheme_tables()
    cat = tables.categories.get
    incb_extend = tables.incb_extend
    incb_linker = tables.incb_linker

    boundaries = [0]
    if not text:
        return boundaries

    prev = cat(ord(text[0]))
    # GB12/13: consecutive Regional_Indicator run length ending at the
    # previous character.
    ri_run = 1 if prev == _RI else 0
    # GB11: does the text ending at the previous character match
    # Extended_Pictographic Extend* (ZWJ)?
    ep_state = 1 if prev == _EXT_PICT else 0  # 0 none, 1 EP Extend*, 2 +ZWJ
    # GB9c: does it match InCB=Consonant [Extend Linker]* with at
    # least one Linker?
    incb_active = prev == _INCB_CONSONANT
    incb_linked = False

    for i in range(1, len(text)):
        cp = ord(text[i])
        cur = cat(cp)

        if prev == _CR and cur == _LF:
            breaks = False  # GB3
        elif prev in (_CONTROL, _CR, _LF):
            breaks = True  # GB4
        elif cur in (_CONTROL, _CR, _LF):
            breaks = True  # GB5
        elif prev == _L and cur in (_L, _V, _LV, _LVT):
            breaks = False  # GB6
        elif prev in (_LV, _V) and cur in (_V, _T):
            breaks = False  # GB7
        elif prev in (_LVT, _T) and cur == _T:
            breaks = False  # GB8
        elif cur in (_EXTEND, _ZWJ):
            breaks = False  # GB9
        elif cur == _SPACINGMARK:
            breaks = False  # GB9a
        elif prev == _PREPEND:
            breaks = False  # GB9b
        elif cur == _INCB_CONSONANT and incb_active and incb_linked:
            breaks = False  # GB9c
        elif prev == _ZWJ and cur == _EXT_PICT and ep_state == 2:
            breaks = False  # GB11
        elif prev == _RI and cur == _RI:
            breaks = ri_run % 2 == 0  # GB12/GB13
        else:
            breaks = True  # GB999

        if breaks:
            boundaries.append(i)

        # Advance the run states over the current character.
        ri_run = ri_run + 1 if cur == _RI else 0
        if cur == _EXT_PICT:
            ep_state = 1
        elif ep_state == 1 and cur == _EXTEND:
            ep_state = 1
        elif ep_state == 1 and cur == _ZWJ:
            ep_state = 2
        else:
            ep_state = 0
        if cur == _INCB_CONSONANT:
            incb_active = True
            incb_linked = False
        elif incb_active and cp in incb_linker:
            incb_linked = True
        elif incb_active and cp in incb_extend:
            pass
        else:
            incb_active = False
            incb_linked = False

        prev = cur

    boundaries.append(len(text))
    return boundaries


def grapheme_clusters(text: str) -> list[str]:
    """``text`` split into extended grapheme clusters, in order."""
    bounds = grapheme_boundaries(text)
    return [text[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]


__all__ = [
    "GRAPHEME_BREAK_SHA256",
    "GraphemeTables",
    "grapheme_boundaries",
    "grapheme_clusters",
    "load_grapheme_tables",
]
