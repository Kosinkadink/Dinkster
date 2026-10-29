"""Proving tests for UAX #29 extended grapheme segmentation.

Conformance is the FULL UCD GraphemeBreakTest.txt for Unicode 16.0.0
(tests/goldens/grapheme_break_test.txt.gz, byte-identical to the
authoritative file - provenance hashes in
tools/gen_grapheme_tables.py), every test line, not a sample. This is
the same table generation the reference tokenizer's
unicode-segmentation 1.12.0 crate uses, so passing it pins the
grapheme iteration underneath the T5 Precompiled normalizer.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from dinkster_inference.graphemes import (
    grapheme_boundaries,
    grapheme_clusters,
    load_grapheme_tables,
)

BREAK = "\u00f7"  # UCD test notation: division sign = break allowed
KEEP = "\u00d7"  # multiplication sign = no break

GOLDEN = Path(__file__).parent / "goldens" / "grapheme_break_test.txt.gz"


def load_conformance_cases() -> list[tuple[str, list[int]]]:
    """Each UCD test line -> (text, expected boundary indices)."""
    cases: list[tuple[str, list[int]]] = []
    for line in gzip.decompress(GOLDEN.read_bytes()).decode().splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        fields = body.split()
        assert fields[0] == BREAK and fields[-1] == BREAK
        text = ""
        boundaries = [0]
        for i in range(1, len(fields) - 1, 2):
            text += chr(int(fields[i], 16))
            if fields[i + 1] == BREAK:
                boundaries.append(len(text))
        cases.append((text, boundaries))
    return cases


CASES = load_conformance_cases()


def test_conformance_corpus_is_the_full_ucd_file() -> None:
    # Unicode 16.0.0 ships 1093 test lines; a shrunk vendored file
    # must fail loudly rather than weaken the conformance claim.
    assert len(CASES) == 1093


def test_full_ucd_conformance() -> None:
    failures = [
        (text, got, want) for text, want in CASES if (got := grapheme_boundaries(text)) != want
    ]
    assert failures == [], failures[:10]


def test_clusters_partition_the_text() -> None:
    for text, want in CASES:
        clusters = grapheme_clusters(text)
        assert "".join(clusters) == text
        assert len(clusters) == len(want) - 1


def test_empty_text_has_single_boundary() -> None:
    assert grapheme_boundaries("") == [0]
    assert grapheme_clusters("") == []


def test_lone_surrogates_segment_as_any() -> None:
    # Python strings can carry lone surrogates; the reference crate's
    # tables never contain them (Rust char cannot), so they are Any.
    assert grapheme_clusters("a\ud800b") == ["a", "\ud800", "b"]


def test_vendored_tables_declare_unicode_16() -> None:
    assert load_grapheme_tables().unicode_version == "16.0.0"


def test_vendored_table_hash_mismatch_is_loud() -> None:
    from dinkster_inference.vendored import read_vendored

    with pytest.raises(ValueError, match="hash mismatch"):
        read_vendored("grapheme_break.json.gz", "0" * 64)
