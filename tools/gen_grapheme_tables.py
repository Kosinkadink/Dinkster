"""Generate the vendored UAX #29 grapheme-break tables.

The T5 tokenizer's Precompiled normalizer iterates extended grapheme
clusters exactly as Hugging Face tokenizers 0.22.2 does - through the
unicode-segmentation 1.12.0 crate, whose tables are generated from
Unicode 16.0.0 (src/tables.rs UNICODE_VERSION). This script reproduces
that crate's table generation (scripts/unicode.py @ v1.12.0) from the
authoritative UCD files and writes the compact JSON consumed by
dinkster_inference.graphemes:

- Grapheme_Cluster_Break categories from auxiliary/GraphemeBreakProperty.txt
- Extended_Pictographic from emoji/emoji-data.txt
- InCB Consonant/Extend/Linker from DerivedCoreProperties.txt

Like the crate, surrogates (U+D800..U+DFFF) are REMOVED from Control:
Rust char cannot represent them, so the crate's tables never contain
them and they classify as Any. Extended_Pictographic and InCB=Consonant
are merged into the category table after asserting no overlap with the
GCB categories, exactly as the crate's generator does.

Usage:

    python tools/gen_grapheme_tables.py <ucd_dir>

where <ucd_dir> contains GraphemeBreakProperty.txt, emoji-data.txt,
DerivedCoreProperties.txt, and GraphemeBreakTest.txt, all from
https://www.unicode.org/Public/16.0.0/ucd/. Writes:

- packages/dinkster-inference/src/dinkster_inference/data/grapheme_break.json.gz
- tests/goldens/grapheme_break_test.txt.gz (the UCD conformance file)

and prints the sha256 hashes to pin in the loading module.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

UNICODE_VERSION = "16.0.0"

_LINE = re.compile(
    r"^([0-9A-F]+)(?:\.\.([0-9A-F]+))?\s*;\s*([A-Za-z_]+)"
    r"(?:\s*;\s*([A-Za-z_]+))?\s*(?:#.*)?$"
)


def load_properties(
    path: Path, wanted: set[str | tuple[str, str]]
) -> dict[str | tuple[str, str], list[tuple[int, int]]]:
    props: dict[str | tuple[str, str], list[tuple[int, int]]] = {key: [] for key in wanted}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            continue
        lo = int(match.group(1), 16)
        hi = int(match.group(2), 16) if match.group(2) else lo
        prop = match.group(3)
        sub = match.group(4)
        key: str | tuple[str, str] = (prop, sub) if sub else prop
        if key in props:
            props[key].append((lo, hi))
    return props


def merge_ranges(ranges: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def remove_surrogates(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in ranges:
        if hi < 0xD800 or lo > 0xDFFF:
            out.append((lo, hi))
            continue
        if lo < 0xD800:
            out.append((lo, 0xD7FF))
        if hi > 0xDFFF:
            out.append((0xE000, hi))
    return out


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    ucd = Path(sys.argv[1])
    repo = Path(__file__).resolve().parent.parent

    gcb = load_properties(
        ucd / "GraphemeBreakProperty.txt",
        {
            "CR",
            "LF",
            "Control",
            "Extend",
            "ZWJ",
            "Regional_Indicator",
            "Prepend",
            "SpacingMark",
            "L",
            "V",
            "T",
            "LV",
            "LVT",
        },
    )
    gcb["Control"] = remove_surrogates(gcb["Control"])
    emoji = load_properties(ucd / "emoji-data.txt", {"Extended_Pictographic"})
    derived = load_properties(
        ucd / "DerivedCoreProperties.txt",
        {("InCB", "Consonant"), ("InCB", "Extend"), ("InCB", "Linker")},
    )

    cats: dict[str, list[list[int]]] = {
        str(name): merge_ranges(ranges) for name, ranges in gcb.items()
    }
    cats["Extended_Pictographic"] = merge_ranges(emoji["Extended_Pictographic"])
    cats["InCB_Consonant"] = merge_ranges(derived[("InCB", "Consonant")])

    # The crate's generator asserts category tables are disjoint before
    # merging EP and InCB=Consonant in (scripts/unicode.py @ v1.12.0).
    covered: set[int] = set()
    for name, ranges in cats.items():
        for lo, hi in ranges:
            for cp in range(lo, hi + 1):
                if cp in covered:
                    raise SystemExit(f"overlapping category at U+{cp:04X} ({name})")
                covered.add(cp)

    data = {
        "unicode_version": UNICODE_VERSION,
        "categories": cats,
        "incb_extend": merge_ranges(derived[("InCB", "Extend")]),
        "incb_linker": merge_ranges(derived[("InCB", "Linker")]),
    }
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    out = repo / "packages/dinkster-inference/src/dinkster_inference/data/grapheme_break.json.gz"
    out.write_bytes(gzip.compress(raw, mtime=0))
    print(f"{out}: sha256 {hashlib.sha256(raw).hexdigest()}")

    test_raw = (ucd / "GraphemeBreakTest.txt").read_bytes()
    test_out = repo / "tests/goldens/grapheme_break_test.txt.gz"
    test_out.write_bytes(gzip.compress(test_raw, mtime=0))
    print(f"{test_out}: sha256 {hashlib.sha256(test_raw).hexdigest()}")


if __name__ == "__main__":
    main()
