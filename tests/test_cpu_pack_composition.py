from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dinkster_workers import load_manifest
from dinkster_workers.host import load_pack
from dinkster_workers.manifest import add_pack_root_to_import_path

torch = pytest.importorskip("torch")


def _write_fake_comfyui(root: Path) -> None:
    (root / "comfy").mkdir(parents=True)
    (root / "comfy" / "options.py").write_text(
        "def enable_args_parsing(value):\n    pass\n",
        encoding="utf-8",
    )
    (root / "comfy" / "cli_args.py").write_text(
        "from types import SimpleNamespace\n"
        "args = SimpleNamespace(\n"
        "    cpu=False, extra_model_paths_config=None, output_directory=None,\n"
        "    input_directory=None, temp_directory=None, user_directory=None,\n"
        ")\n",
        encoding="utf-8",
    )
    (root / "folder_paths.py").write_text(
        "from pathlib import Path\n"
        "root = Path(__file__).parent\n"
        "def get_output_directory(): return str(root / 'output')\n"
        "def add_model_folder_path(category, path): pass\n"
        "def get_filename_list(category): return []\n"
        "def get_folder_paths(category): return [str(root / 'models' / category)]\n",
        encoding="utf-8",
    )
    (root / "nodes.py").write_text(
        "class TestNode:\n"
        "    @classmethod\n"
        "    def INPUT_TYPES(cls):\n"
        "        return {'required': {'value': ('INT', {'default': 0})}}\n"
        "    RETURN_TYPES = ('INT',)\n"
        "    FUNCTION = 'run'\n"
        "    CATEGORY = 'test'\n"
        "    def run(self, value): return (value,)\n"
        "NODE_CLASS_MAPPINGS = {'TestNode': TestNode}\n"
        "NODE_DISPLAY_NAME_MAPPINGS = {}\n"
        "def init_extra_nodes(*, init_custom_nodes, init_api_nodes): pass\n",
        encoding="utf-8",
    )


def test_generation_and_comfy_compat_compose_with_cpu_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert torch.version.cuda is None
    assert not torch.cuda.is_available()

    comfy_root = tmp_path / "ComfyUI"
    _write_fake_comfyui(comfy_root)
    monkeypatch.setenv("DINKSTER_COMFYUI_ROOT", str(comfy_root))
    monkeypatch.setenv("DINKSTER_COMFY_NODES", "TestNode")
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")

    for pack in ("dinkster-nodes-generation", "dinkster-compat-comfy"):
        manifest = load_manifest(Path("packages") / pack / "dinkster-pack.toml")
        add_pack_root_to_import_path(manifest, sys.path)
        _worker, _registry, nodes, _arms = load_pack(manifest)
        assert nodes
