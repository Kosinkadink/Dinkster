from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from dinkster_compat_comfy import CompatTranslation, bootstrap
from dinkster_compat_comfy import legacy as legacy_loader
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
    finally:
        sys.modules.pop("dinkster_compat_comfy.entry", None)


def test_legacy_compat_arm_entry_matches_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "load_comfyui_nodes", lambda: CompatTranslation())
    monkeypatch.setattr(
        legacy_loader,
        "load_legacy_packs",
        lambda: (CompatTranslation(), []),
    )
    sys.modules.pop("dinkster_compat_comfy.entry", None)
    sys.modules.pop("dinkster_compat_comfy.legacy_entry", None)
    try:
        legacy_entry = importlib.import_module("dinkster_compat_comfy.legacy_entry")
        manifest = load_manifest(Path("packages/dinkster-compat-comfy/dinkster-legacy-pack.toml"))
        expected = dict(manifest.arms)["native"]
        actual = tuple(node.schema().node_type for node in legacy_entry.ARM_NODES["native"])
        assert actual == expected
    finally:
        sys.modules.pop("dinkster_compat_comfy.legacy_entry", None)
        sys.modules.pop("dinkster_compat_comfy.entry", None)
