"""Native bounded UTF-8 text saving contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from dinkster_assets import AssetRef, digest_bytes
from dinkster_nodes_media_io import SaveText
from dinkster_nodes_media_io import text as text_module
from dinkster_nodes_media_io.text import MAX_TEXT_BYTES, render_text


def _mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "out"
    root.mkdir()
    index = root / ".dinkster-asset-index.json"
    index.write_text("{}", "utf-8")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "comfy-output",
                        "root": str(root),
                        "index": str(index),
                        "mode": "readwrite",
                    }
                ]
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root


def _single_ref(result: object) -> AssetRef:
    refs = cast("list[AssetRef]", cast("dict[str, object]", result)["texts"])
    assert len(refs) == 1
    return refs[0]


def test_save_writes_utf8_bytes_verbatim_with_deterministic_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _mount(tmp_path, monkeypatch)
    content = "first line\nsecond line with e-acute \u00e9\n"
    first = _single_ref(SaveText.execute(text=content))
    second = _single_ref(SaveText.execute(text=content))
    assert first.name == "ComfyUI_00001.txt"
    assert second.name == "ComfyUI_00002.txt"
    saved = (root / "text" / first.name).read_bytes()
    assert saved == content.encode("utf-8")
    assert b"\r" not in saved
    assert first.digest == digest_bytes(saved)
    assert first.media_type == "text/plain"


@pytest.mark.parametrize(
    ("format_name", "suffix", "media_type"),
    [
        ("txt", ".txt", "text/plain"),
        ("csv", ".csv", "text/csv"),
        ("md", ".md", "text/markdown"),
        ("json", ".json", "application/json"),
    ],
)
def test_save_advertised_formats_publish_typed_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    suffix: str,
    media_type: str,
) -> None:
    _mount(tmp_path, monkeypatch)
    ref = _single_ref(
        SaveText.execute(
            text='{"a": 1}',
            format=format_name,
            target={"mount": "comfy-output", "prefix": "notes/note"},
        )
    )
    assert ref.name.endswith(suffix)
    assert ref.media_type == media_type


def test_render_json_pretty_prints_and_falls_back() -> None:
    assert (
        render_text('{"b":2,"a":[1,2],"s":"caf\u00e9"}', "json")
        == b'{\n  "b": 2,\n  "a": [\n    1,\n    2\n  ],\n  "s": "caf\xc3\xa9"\n}'
    )
    assert render_text("not json {", "json") == b"not json {"
    assert render_text("42", "json") == b"42"


def test_render_non_json_formats_never_transform() -> None:
    content = '{"a": 1}\nline'
    for format_name in ("txt", "csv", "md"):
        assert render_text(content, format_name) == content.encode("utf-8")


def test_unknown_format_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mount(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="format must be one of"):
        SaveText.execute(text="body", format="../evil")


def test_non_string_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="text must be a string"):
        SaveText.execute(text=42)


def test_size_bound_is_enforced_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _mount(tmp_path, monkeypatch)
    monkeypatch.setattr(text_module, "MAX_TEXT_BYTES", 16)
    with pytest.raises(ValueError, match="above the 16 byte save limit"):
        SaveText.execute(text="x" * 17)
    assert not (root / "text").exists() or not list((root / "text").iterdir())


def test_multibyte_content_is_measured_in_bytes() -> None:
    with pytest.raises(ValueError, match="save limit"):
        render_text("\u00e9" * (MAX_TEXT_BYTES // 2 + 1), "txt")


def test_save_without_mounts_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    with pytest.raises(Exception, match="DINKSTER_MOUNTS_SNAPSHOT"):
        SaveText.execute(text="body")


def test_schema_declares_output_node_and_bounded_combo() -> None:
    schema = SaveText.define_schema()
    assert schema.output_node
    assert not schema.idempotent
    widgets = {spec.id: spec.widget for spec in schema.inputs}
    combo = widgets["format"]
    assert getattr(combo, "options", None) == ("txt", "csv", "md", "json")
