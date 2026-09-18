"""The pack template stays healthy (templates/pack, DESIGN 3.6/3.9).

The template is documentation that executes: docs/pack-authoring.md points
authors at it, so this repo's own suite holds it to the same bar it
advertises - structural doctor health, no extra import dependencies, nodes
executable over plain values, declared codec and rendition working. Template
drift fails here, not in a copied pack's CI."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from dinkster_api.v1 import TypeRegistry
from dinkster_workers.doctor import diagnose, render_text
from dinkster_workers.manifest import load_manifest

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "pack"


def test_template_is_doctor_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def current_interpreter_version(selected: Path | str) -> tuple[int, int]:
        assert str(selected) == sys.executable
        assert sys.version_info[:2] >= (3, 12)
        return sys.version_info[:2]

    # Template health is independent of version-probe startup latency. Interpreter
    # admission has its own real-process and timeout tests; the pack probe stays real.
    monkeypatch.setattr(
        "dinkster_workers.doctor.preflight_interpreter", current_interpreter_version
    )
    report = diagnose(TEMPLATE / "dinkster-pack.toml")
    assert report.ok, render_text(report)
    assert not report.findings, render_text(report)


def test_template_imports_only_public_api_and_stdlib() -> None:
    source = ast.parse((TEMPLATE / "my_pack_nodes.py").read_text(encoding="utf-8"))
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            modules = [node.module or ""]
        else:
            continue
        assert all(
            name == "dinkster_api.v1" or name.split(".")[0] in sys.stdlib_module_names
            for name in modules
        ), modules

    # Method bodies run later; decorators, defaults and class bodies run during import.
    for node in ast.walk(source):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = []
    for node in ast.walk(source):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert all(
                isinstance(decorator, ast.Name) and decorator.id in {"classmethod", "staticmethod"}
                for decorator in node.decorator_list
            )
        elif isinstance(node, ast.ClassDef):
            assert not node.decorator_list and not node.keywords
            assert all(isinstance(base, ast.Name) and base.id == "Node" for base in node.bases)
    forbidden = (
        ast.Call,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.comprehension,
        ast.BinOp,
    )
    assert not any(isinstance(node, forbidden) for node in ast.walk(source))

    script = """
import importlib
import sys
import dinkster_api.v1

baseline = set(sys.modules)
sys.path.insert(0, sys.argv[1])
importlib.import_module("my_pack_nodes")
added = {name.split(".")[0] for name in set(sys.modules) - baseline}
unexpected = added - sys.stdlib_module_names - {"my_pack_nodes"}
assert not unexpected, sorted(unexpected)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(TEMPLATE)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_template_manifest_declares_presentation_and_icon() -> None:
    manifest = load_manifest(TEMPLATE / "dinkster-pack.toml")
    assert manifest.name == "my-pack"
    assert manifest.presentation is not None
    assert manifest.presentation.display_name == "My Pack"
    assert manifest.presentation.icon is not None
    assert manifest.presentation.icon.media_type == "image/png"


def test_template_nodes_execute_and_types_register() -> None:
    sys.path.insert(0, str(TEMPLATE))
    try:
        # Resolved off sys.path at runtime, as the worker host resolves
        # manifest entries; invisible to static analysis by design.
        from my_pack_nodes import (  # pyright: ignore[reportMissingImports]
            NODES,
            Shout,
            Tally,
            register_types,
        )
    finally:
        sys.path.remove(str(TEMPLATE))

    assert [node.define_schema().node_type for node in NODES] == [
        "my-pack.shout",
        "my-pack.tally",
    ]
    assert Shout.execute(text="hey", times=2) == {"shouted": "HEY!HEY!"}
    assert Tally.execute(text="a b a") == {"tally": {"a": 2, "b": 1}}

    registry = TypeRegistry()
    register_types(registry)
    spec = registry.spec("my-pack.tally")
    assert spec.declared_codec
    assert spec.decode(spec.encode({"a": 1})) == {"a": 1}
    rendition = registry.render(registry.wrap("my-pack.tally", {"a": 1}))
    assert rendition.mime == "text/plain"
