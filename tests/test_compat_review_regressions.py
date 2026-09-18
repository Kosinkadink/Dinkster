from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from dinkster_compat_comfy import (
    CompatError,
    CompatTranslation,
    bootstrap,
    load_legacy_pack,
    translate_mappings,
)


class BadNode:
    RETURN_TYPES = ("INT",)


class GoodNode:
    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {"required": {}}

    def run(self) -> tuple[int]:
        return (1,)


def test_namespaced_skips_do_not_collide_and_success_clears_stale() -> None:
    translation = CompatTranslation()
    translate_mappings({"Shared": BadNode}, namespace="pack_a", translation=translation)
    translate_mappings({"Shared": BadNode}, namespace="pack_b", translation=translation)
    assert set(translation.skipped) == {"pack_a.Shared", "pack_b.Shared"}

    translate_mappings({"Shared": GoodNode}, namespace="pack_b", translation=translation)
    assert set(translation.skipped) == {"pack_a.Shared"}


def test_colliding_pack_skips_are_reported_per_pack(tmp_path: Path) -> None:
    source = "class Bad:\n    RETURN_TYPES = ('INT',)\nNODE_CLASS_MAPPINGS = {'Shared': Bad}\n"
    translation = CompatTranslation()
    reports = []
    for pack_id in ("pack_a", "pack_b"):
        pack = tmp_path / pack_id
        pack.mkdir()
        (pack / "__init__.py").write_text(source, encoding="utf-8")
        reports.append(load_legacy_pack(pack, translation, server_instance=None))
    assert [set(report.nodes_skipped) for report in reports] == [
        {"Shared"},
        {"Shared"},
    ]


@pytest.mark.parametrize("new_style", [False, True])
def test_bootstrap_calls_old_and_new_init_signatures_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_style: bool
) -> None:
    root = tmp_path
    (root / "nodes.py").write_text("", encoding="utf-8")
    (root / "folder_paths.py").write_text("", encoding="utf-8")
    calls: list[dict[str, bool]] = []
    if new_style:

        def new_init(*, init_custom_nodes: bool, init_api_nodes: bool) -> None:
            calls.append({"custom": init_custom_nodes, "api": init_api_nodes})

        init_extra_nodes = new_init
    else:

        def old_init(*, init_custom_nodes: bool) -> None:
            calls.append({"custom": init_custom_nodes})

        init_extra_nodes = old_init
    nodes = SimpleNamespace(init_extra_nodes=init_extra_nodes)
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(root))
    monkeypatch.setattr(bootstrap, "initialize_comfy_paths", lambda: object())
    monkeypatch.setattr(bootstrap, "_initialize_comfy_device", lambda: None)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: nodes if name == "nodes" else ModuleType(name),
    )
    monkeypatch.setattr(bootstrap, "ensure_prompt_server", lambda: None)
    assert bootstrap.bootstrap_comfyui() is nodes
    assert len(calls) == 1
    assert calls[0] == ({"custom": False, "api": False} if new_style else {"custom": False})


def test_bootstrap_propagates_body_type_error_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "nodes.py").write_text("", encoding="utf-8")
    (root / "folder_paths.py").write_text("", encoding="utf-8")
    calls = 0

    def init_extra_nodes(*, init_custom_nodes: bool, init_api_nodes: bool) -> None:
        nonlocal calls
        calls += 1
        raise TypeError("from body")

    nodes = SimpleNamespace(init_extra_nodes=init_extra_nodes)
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(root))
    monkeypatch.setattr(bootstrap, "initialize_comfy_paths", lambda: object())
    monkeypatch.setattr(bootstrap, "_initialize_comfy_device", lambda: None)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: nodes if name == "nodes" else ModuleType(name),
    )
    monkeypatch.setattr(bootstrap, "ensure_prompt_server", lambda: None)
    with pytest.raises(TypeError, match="from body"):
        bootstrap.bootstrap_comfyui()
    assert calls == 1


@pytest.mark.parametrize(
    ("selection", "cuda", "mps", "cpu"),
    [
        ("auto", False, False, True),
        ("auto", True, False, False),
        ("auto", False, True, False),
        ("cpu", True, False, True),
        ("cpu", False, True, True),
        ("mps", False, True, False),
    ],
)
def test_bootstrap_selects_worker_device_before_importing_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    cuda: bool,
    mps: bool,
    cpu: bool,
) -> None:
    (tmp_path / "nodes.py").touch()
    (tmp_path / "folder_paths.py").touch()
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(tmp_path))
    monkeypatch.setenv("DINKSTER_ACCELERATOR", selection)
    cli_args = SimpleNamespace(args=SimpleNamespace(cpu=False))
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        device=lambda name: SimpleNamespace(type=name.split(":")[0]),
    )
    nodes = ModuleType("nodes")

    def import_module(name: str) -> object:
        if name == "torch":
            return torch
        if name == "nodes":
            assert cli_args.args.cpu is cpu
            return nodes
        return ModuleType(name)

    monkeypatch.setattr(bootstrap, "initialize_comfy_paths", lambda: object())
    monkeypatch.setattr(bootstrap, "initialize_comfy_args", lambda: cli_args)
    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(bootstrap, "ensure_prompt_server", lambda: None)
    assert bootstrap.bootstrap_comfyui() is nodes


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ((), 512),
        (("--preview-size", "321"), 321),
    ],
)
def test_comfy_cli_args_initialize_from_exact_supplied_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supplied: tuple[str, ...],
    expected: int,
) -> None:
    comfy = tmp_path / "comfy"
    comfy.mkdir()
    (comfy / "__init__.py").write_text("", encoding="utf-8")
    (comfy / "options.py").write_text(
        "args_parsing = False\n"
        "def enable_args_parsing(enable=True):\n"
        "    global args_parsing\n"
        "    args_parsing = enable\n",
        encoding="utf-8",
    )
    (comfy / "cli_args.py").write_text(
        "import argparse\n"
        "import comfy.options\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--preview-size', type=int, default=512)\n"
        "args = parser.parse_args() if comfy.options.args_parsing else parser.parse_args([])\n",
        encoding="utf-8",
    )
    for name in ("comfy.cli_args", "comfy.options", "comfy"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bootstrap, "_comfy_args_initialized", None)
    monkeypatch.setattr(sys, "argv", ["worker", *supplied])
    try:
        cli_args = bootstrap.initialize_comfy_args()
        assert cli_args.args.preview_size == expected  # type: ignore[attr-defined]
        assert bootstrap._comfy_args_initialized == supplied
    finally:
        for name in ("comfy.cli_args", "comfy.options", "comfy"):
            sys.modules.pop(name, None)


def test_comfy_cli_args_rejection_names_supplied_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comfy = tmp_path / "comfy"
    comfy.mkdir()
    (comfy / "__init__.py").write_text("", encoding="utf-8")
    (comfy / "options.py").write_text(
        "def enable_args_parsing(enable=True):\n    pass\n", encoding="utf-8"
    )
    (comfy / "cli_args.py").write_text(
        "import argparse\nargparse.ArgumentParser().parse_args()\n", encoding="utf-8"
    )
    for name in ("comfy.cli_args", "comfy.options", "comfy"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bootstrap, "_comfy_args_initialized", None)
    monkeypatch.setattr(sys, "argv", ["worker", "--not-a-real-argument"])
    try:
        with pytest.raises(CompatError, match="--not-a-real-argument"):
            bootstrap.initialize_comfy_args()
    finally:
        for name in ("comfy.cli_args", "comfy.options", "comfy"):
            sys.modules.pop(name, None)


def test_missing_sampler_vocabulary_serves_empty_choices_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # entry has child-only import side effects, so neutralize its bootstrap.
    monkeypatch.setattr(
        bootstrap,
        "load_comfyui_nodes",
        lambda *, required=(): CompatTranslation(),
    )
    sys.modules.pop("dinkster_compat_comfy.entry", None)
    entry = importlib.import_module("dinkster_compat_comfy.entry")
    real_import = entry.importlib.import_module
    monkeypatch.setattr(
        entry.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(KSampler=object())
            if name == "comfy.samplers"
            else SimpleNamespace(get_filename_list=lambda category: ())
            if name == "folder_paths"
            else real_import(name)
        ),
    )
    assert entry.combo_choices() == {
        "comfy.samplers": (),
        "comfy.schedulers": (),
        "comfy.files.embeddings": (),
        "comfy.files.loras": (),
    }
    assert "DINKSTER_COMPAT_SAMPLER_CHOICES_UNAVAILABLE" in caplog.text
