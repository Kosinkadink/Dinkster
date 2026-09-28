from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from dinkster_compat_comfy import CompatTranslation, bootstrap
from dinkster_native.native import NATIVE_NODES
from dinkster_workers import load_manifest


def test_compat_arm_entry_matches_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "load_comfyui_nodes", lambda: CompatTranslation())
    sys.modules.pop("dinkster_compat_comfy.entry", None)
    try:
        entry = importlib.import_module("dinkster_compat_comfy.entry")
        manifest = load_manifest(Path("packages/dinkster-compat-comfy/dinkster-pack.toml"))
        expected = dict(manifest.arms)["native"]
        actual = tuple(node.schema().node_type for node in entry.ARM_NODES["native"])
        assert actual == expected
        assert actual == manifest.executes
        default_types = {node.schema().node_type for node in entry.COMFY_NODES}
        native_types = {node.schema().node_type for node in NATIVE_NODES}
        assert set(actual).issubset(default_types)
        assert default_types & native_types == set(actual)
    finally:
        sys.modules.pop("dinkster_compat_comfy.entry", None)
