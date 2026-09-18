"""Replay pinned ComfyUI e20d433a SaveText goldens against the native saver.

The fixture (tests/goldens/text_io_e20d433a.json) stores the exact text fed
to the pinned SaveTextNode plus the bytes and name it saved; see
tools/gen_text_io_goldens.py. Text output is deterministic, so every case
asserts bit-exact bytes.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from dinkster_nodes_media_io.text import _TEXT_FORMATS, render_text

_GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "text_io_e20d433a.json").read_text("utf-8")
)

_FILE_CASES = [name for name in _GOLDENS["cases"] if name != "counter_second_save"]


def test_goldens_are_pinned_to_the_audited_revision() -> None:
    assert _GOLDENS["revision"] == "e20d433a4966dcc88fa5abbae6ace824cb78b263"


def test_every_pinned_case_confirms_the_passthrough_output() -> None:
    """The pinned SaveTextNode returned its input text alongside the file;
    that passthrough output is a declared gap on the alias record, so every
    golden must attest the pinned run actually produced it."""
    assert all(_GOLDENS["cases"][name]["passthrough"] is True for name in _FILE_CASES)


@pytest.mark.parametrize("name", _FILE_CASES)
def test_rendered_bytes_match_pinned_save_exactly(name: str) -> None:
    case = _GOLDENS["cases"][name]
    text = base64.b64decode(case["textB64"]).decode("utf-8")
    assert render_text(text, case["format"]) == base64.b64decode(case["fileB64"])


@pytest.mark.parametrize("name", _FILE_CASES)
def test_pinned_suffixes_match_the_native_format_table(name: str) -> None:
    case = _GOLDENS["cases"][name]
    suffix, _media_type = _TEXT_FORMATS[case["format"]]
    assert case["filename"].endswith(suffix)


def test_pinned_counter_naming_matches_the_native_writer_convention() -> None:
    """Both sides name saves {prefix}_{counter:05}{suffix}; the pinned second
    save under one prefix lands on _00002 exactly like the native writer
    (see test_text_io.py deterministic-name test)."""
    assert _GOLDENS["cases"]["counter_second_save"]["filename"] == "counter_00002.txt"
