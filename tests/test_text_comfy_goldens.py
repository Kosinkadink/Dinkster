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
from dinkster_text.save_text import SaveText

from tests.test_save_text import _snapshot

pytestmark = pytest.mark.usefixtures("saved_artifact_context")

_GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "text_io_e20d433a.json").read_text("utf-8")
)

_FILE_CASES = [name for name in _GOLDENS["cases"] if name != "counter_second_save"]


@pytest.mark.parametrize("name", _FILE_CASES)
def test_saved_bytes_name_and_passthrough_match_pinned_save_exactly(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)
    case = _GOLDENS["cases"][name]
    text = base64.b64decode(case["textB64"]).decode("utf-8")
    assert SaveText.execute(text=text, filename_prefix=name, format=case["format"]) == {
        "text": text
    }
    assert (roots["default"] / case["filename"]).read_bytes() == base64.b64decode(case["fileB64"])


def test_pinned_counter_naming_matches_the_native_writer_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)
    SaveText.execute(text="first", filename_prefix="counter")
    SaveText.execute(text="second", filename_prefix="counter")
    filename = _GOLDENS["cases"]["counter_second_save"]["filename"]
    assert (roots["default"] / filename).read_bytes() == b"second"
