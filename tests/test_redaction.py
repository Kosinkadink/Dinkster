"""Path redaction at the wire boundaries.

Outbound payloads (error reports, node-reported event data, pack failure
rows) must not publish the host's filesystem layout: known roots become
stable tokens, other real paths collapse to <path>/<basename>, and
route-like strings that are not filesystem paths pass through untouched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from dinkster_engine import EngineEvent
from dinkster_server import PathRedactor
from dinkster_server.events import engine_event_to_wire

# -- PathRedactor unit behavior -------------------------------------------------


def test_known_roots_become_tokens(tmp_path: Path) -> None:
    redactor = PathRedactor([("<library>", tmp_path)])
    text = f"failed to read {tmp_path}/vault/model.safetensors"
    assert redactor.redact_text(text) == "failed to read <library>/vault/model.safetensors"


def test_nested_root_maps_to_most_specific_token(tmp_path: Path) -> None:
    library = tmp_path / "library"
    redactor = PathRedactor([("<install>", tmp_path), ("<library>", library)])
    text = f"{library}/vault/x.bin and {tmp_path}/other.txt"
    assert redactor.redact_text(text) == "<library>/vault/x.bin and <install>/other.txt"


def test_root_prefix_requires_a_segment_boundary(tmp_path: Path) -> None:
    root = tmp_path / "user"
    sibling = f"{tmp_path}/username/file.txt"
    redactor = PathRedactor([("<home>", root)])
    redacted = redactor.redact_text(sibling)
    assert "<home>" not in redacted


def test_tmp_and_home_are_always_roots() -> None:
    redactor = PathRedactor()
    tmp = tempfile.gettempdir()
    assert redactor.redact_text(f"wrote {tmp}/scratch.dat") == "wrote <tmp>/scratch.dat"
    home = str(Path.home())
    assert redactor.redact_text(f"read {home}/notes.txt") == "read <home>/notes.txt"


def test_root_never_rewrites_inside_a_longer_unrelated_path() -> None:
    redactor = PathRedactor()
    text = "wrote /var/tmp/missing-dir-xyz/file.bin"
    redacted = redactor.redact_text(text)
    assert "<tmp>" not in redacted
    assert redacted == text


def test_root_never_rewrites_inside_a_url() -> None:
    redactor = PathRedactor()
    text = "fetch https://example.com/tmp/resource failed"
    assert redactor.redact_text(text) == text


def test_relative_roots_only_match_their_resolved_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A root given relatively (--install-root .) must not rewrite ordinary
    # prose; only its resolved absolute spelling is redacted.
    monkeypatch.chdir(tmp_path)
    redactor = PathRedactor([("<install>", Path(".")), ("<mount:models>", Path("models"))])
    # The "/" defeats redact_text's early-return prefilter so the prose
    # actually runs through the patterns.
    prose = "The models loaded. See /api/nodes for the surface."
    assert redactor.redact_text(prose) == prose
    resolved = str(Path(".").resolve())
    assert redactor.redact_text(f"read {resolved}/pack.toml") == "read <install>/pack.toml"


def test_generated_tokens_are_not_rewritten_again() -> None:
    redactor = PathRedactor()
    text = f"cached {Path.home()}/tmp/x"
    assert redactor.redact_text(text) == "cached <home>/tmp/x"


def test_repeated_separators_do_not_start_a_new_path() -> None:
    redactor = PathRedactor()
    text = "wrote /var//tmp/x then file:///tmp/y"
    redacted = redactor.redact_text(text)
    assert "<tmp>" not in redacted


def test_path_after_shell_redirect_is_redacted() -> None:
    redactor = PathRedactor()
    text = f"log redirected 2>{Path.home()}/secret.txt"
    assert redactor.redact_text(text) == "log redirected 2><home>/secret.txt"


def test_url_routes_pass_through() -> None:
    redactor = PathRedactor()
    for text in (
        "GET /api/nodes returned 404",
        "see https://example.com/docs/errors for details",
        "route /api/packs/foo/reload conflicted",
    ):
        assert redactor.redact_text(text) == text


def test_existing_unknown_path_collapses_to_basename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A path outside every known root that exists on this machine falls
    # back to <path>/<basename>. Existence is faked so the test does not
    # depend on the host's filesystem layout.
    fake = "/srv/models/weights.ckpt"
    monkeypatch.setattr(
        "dinkster_server.redaction.os.path.exists",
        lambda p: str(p).startswith("/srv"),
    )
    redactor = PathRedactor()
    assert redactor.redact_text(f"cached at {fake}") == "cached at <path>/weights.ckpt"


def test_nonexistent_absolute_string_passes_through() -> None:
    redactor = PathRedactor()
    text = "virtual key /definitely/not/a/real/dir/entry.bin missing"
    assert redactor.redact_text(text) == text


def test_traceback_frames_are_rewritten() -> None:
    redactor = PathRedactor()
    frame = f'  File "{Path(__file__).resolve()}", line 10, in boom'
    redacted = redactor.redact_text(frame)
    assert str(Path(__file__).resolve()) not in redacted
    assert "line 10, in boom" in redacted


def test_windows_paths_are_rewritten() -> None:
    redactor = PathRedactor([("<install>", Path("C:\\dinkster\\install"))])
    text = "failed on C:\\dinkster\\install\\packs\\core.toml"
    assert redactor.redact_text(text) == "failed on <install>\\packs\\core.toml"


def test_redact_value_recurses_and_preserves_non_strings(tmp_path: Path) -> None:
    redactor = PathRedactor([("<library>", tmp_path)])
    value = {
        "message": f"missing {tmp_path}/a.bin",
        "count": 3,
        "flag": True,
        "nested": [f"{tmp_path}/b.bin", 1.5, None],
        "blob": b"\x00\x01",
    }
    redacted = redactor.redact_value(value)
    assert redacted == {
        "message": "missing <library>/a.bin",
        "count": 3,
        "flag": True,
        "nested": ["<library>/b.bin", 1.5, None],
        "blob": b"\x00\x01",
    }


def test_redaction_never_raises_on_odd_shapes() -> None:
    redactor = PathRedactor()
    assert redactor.redact_value(object()) is not None
    assert redactor.redact_text("") == ""


# -- wire boundary: engine events ------------------------------------------------


def test_node_event_data_is_redacted(tmp_path: Path) -> None:
    redactor = PathRedactor([("<library>", tmp_path)])
    event = EngineEvent(
        kind="node_event",
        run_id="r1",
        node_id="n1",
        detail={
            "name": "log",
            "data": {"level": "info", "message": f"saved to {tmp_path}/out.png"},
            "attentionDiagnostic": f"fallback for {tmp_path}/worker",
        },
    )
    wire = engine_event_to_wire(event, client_id="c", job_id="j", redactor=redactor)
    data = wire["data"]
    assert isinstance(data, dict)
    assert data["message"] == "saved to <library>/out.png"
    assert wire["attentionDiagnostic"] == "fallback for <library>/worker"


def test_lifecycle_detail_is_redacted(tmp_path: Path) -> None:
    redactor = PathRedactor([("<library>", tmp_path)])
    event = EngineEvent(
        kind="node_failed",
        run_id="r1",
        node_id="n1",
        detail={"message": f"cannot open {tmp_path}/missing.bin"},
    )
    wire = engine_event_to_wire(event, client_id="c", job_id="j", redactor=redactor)
    detail = wire["detail"]
    assert isinstance(detail, dict)
    assert detail["message"] == "cannot open <library>/missing.bin"


def test_wire_unredacted_without_redactor(tmp_path: Path) -> None:
    event = EngineEvent(
        kind="node_event",
        run_id="r1",
        node_id="n1",
        detail={
            "name": "log",
            "data": {"message": f"{tmp_path}/raw.txt"},
            "attentionDiagnostic": f"fallback for {tmp_path}/worker",
        },
    )
    wire = engine_event_to_wire(event, client_id="c", job_id="j")
    data = wire["data"]
    assert isinstance(data, dict)
    assert data["message"] == f"{tmp_path}/raw.txt"
    assert wire["attentionDiagnostic"] == f"fallback for {tmp_path}/worker"
