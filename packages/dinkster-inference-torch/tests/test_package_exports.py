"""Package exports preserve the eager API without eager model initialization."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "dinkster_inference_torch"
TRAINING_EXPORTS = frozenset(
    {
        "AttentionGuidanceContext",
        "MiniMaxH3AudioVAE",
        "MiniMaxH3ConditionerModel",
        "MiniMaxH3DiTConditioning",
        "MiniMaxH3KeyframeLatent",
        "MiniMaxH3ReferenceKind",
        "MiniMaxH3ReferenceLatents",
        "MiniMaxH3VideoVAE",
        "MiniMaxH3VideoVAEConfig",
        "MiniMaxMusic3TextModel",
        "QwenImageLanguageModel",
        "QwenImageTextModel",
        "SD15AttentionExecutionContext",
        "Wan21Model",
        "Wan21MultiTalkExecution",
        "Wan21TextRuntime",
        "Wan22VAE",
        "WanAttentionBlock",
        "WanVAE",
        "WanVAEConfig",
        "assemble_minimax_h3_dit",
        "bind_fp8_matmul_layer",
        "load_tensors",
        "minimax_h3_audio_vae_runtime_identity",
        "minimax_h3_conditioner_runtime_identity",
        "minimax_h3_video_vae_runtime_identity",
        "plan_minimax_h3_model_assembly",
        "select_attention",
        "simple_schedule",
        "tokenize_music_prompt",
        "worker_planning_context",
    }
)


def _run(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _typed_bindings() -> dict[str, tuple[str, str]]:
    tree = ast.parse((PACKAGE / "__init__.py").read_text())
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_TYPE_CHECKING"
    )
    bindings: dict[str, tuple[str, str]] = {}
    for node in block.body:
        assert isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
        for alias in node.names:
            # Assignment order, rather than a set, preserves the last binding.
            bindings[alias.asname or alias.name] = (node.module, alias.name)
    return bindings


def test_binding_table_matches_static_imports_and_eager_baseline() -> None:
    import dinkster_inference_torch as package

    bindings = _typed_bindings()
    exports = vars(package)["_EXPORTS"]
    assert {name: target for name, target in exports.items() if target[1] is not None} == bindings
    # Pin the complete lazy public surface, including the solvers.euler alias.
    assert len(bindings) == 881
    assert hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest() == (
        "a1d4e82516d5d3d1024461776bf7820dbaba303b7fa14877faf73409246a5831"
    )
    assert len(package.__all__) == 884
    assert hashlib.sha256(json.dumps(package.__all__).encode()).hexdigest() == (
        "f6836ae7c8dbef80051e57355adcb46ef5e32895894ca9adac78b3428ccdc402"
    )
    assert set(package.__all__) == set(bindings)
    modules = {
        name: f"dinkster_inference_torch.{target[0]}"
        for name, target in exports.items()
        if target[1] is None
    }
    assert len(modules) == 167
    assert hashlib.sha256(json.dumps(modules, sort_keys=True).encode()).hexdigest() == (
        "3a604ccc1c84104a6ddac2d78aa078fd1dbebcca24066d480c48ff5775a0e6cf"
    )
    package_entries = {entry.name for entry in PACKAGE.iterdir()}
    collisions = {name for name in bindings if f"{name}.py" in package_entries}
    assert collisions == {"component_publisher"}


def test_training_contract_is_public_from_package_root() -> None:
    import dinkster_inference_torch as package

    assert TRAINING_EXPORTS <= set(package.__all__)
    assert all(hasattr(package, name) for name in TRAINING_EXPORTS)


def test_cold_package_and_dir_do_not_import_execution_dependencies() -> None:
    _run("""
        import importlib.abc
        import sys

        class BlockExecution(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {
                    'torch', 'dinkster_kitchen', 'tokenizers', 'sentencepiece'
                }:
                    raise AssertionError('eager import: ' + fullname)

        sys.meta_path.insert(0, BlockExecution())
        import dinkster_inference_torch as package
        names = dir(package)
        assert set(package.__all__) <= set(names)
        assert len([name for name in names if not name.startswith('_')]) == 1048
        assert names == sorted(set(names))
        assert not any(name.startswith('dinkster_inference_torch.') for name in sys.modules)
        assert not hasattr(package, 'unknown_export')
        try:
            from dinkster_inference_torch import unknown_export
        except ImportError:
            pass
        else:
            raise AssertionError('unknown from-import succeeded')
        try:
            package.unknown_export
        except AttributeError as error:
            assert str(error) == (
                "module 'dinkster_inference_torch' has no attribute 'unknown_export'"
            )
        else:
            raise AssertionError('unknown attribute succeeded')
    """)


def test_package_reload_refreshes_exports_without_loading_unrelated_dependencies() -> None:
    _run("""
        import importlib.abc
        import sys

        blocked = {'torch', 'dinkster_aimdo', 'dinkster_kitchen', 'tokenizers', 'sentencepiece'}

        class BlockExecution(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    raise AssertionError('unrelated import: ' + fullname)

        sys.meta_path.insert(0, BlockExecution())
        import dinkster_inference_torch as package
        before = package.AimdoStatus
        owner = importlib.import_module('dinkster_inference_torch.aimdo')
        assert package.aimdo is owner
        importlib.reload(owner)
        refreshed = owner.AimdoStatus
        assert refreshed is not before
        loaded = set(sys.modules)

        for _ in range(2):
            assert importlib.reload(package) is package
            assert vars(package).keys().isdisjoint(vars(package)['_EXPORTS'])
            assert set(sys.modules) == loaded
            from dinkster_inference_torch import AimdoStatus
            assert AimdoStatus is package.AimdoStatus is owner.AimdoStatus is refreshed
            assert package.aimdo is owner

        assert blocked.isdisjoint(sys.modules)
        assert {
            name for name in sys.modules if name.startswith('dinkster_inference_torch.')
        } == {'dinkster_inference_torch.aimdo'}
    """)


@pytest.mark.parametrize("root_exports", [False, True])
def test_leaf_helpers_run_without_unrelated_optional_dependencies(root_exports: bool) -> None:
    _run(f"""
        import importlib.abc
        import sys

        blocked = {{'dinkster_kitchen', 'tokenizers', 'sentencepiece'}}
        assert blocked.isdisjoint(sys.modules)

        class MissingOptionalDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    raise ModuleNotFoundError(fullname, name=fullname)

        sys.meta_path.insert(0, MissingOptionalDependency())
        import torch
        from dinkster_inference import LatentStream, MultiStreamLatent
        if {root_exports!r}:
            from dinkster_inference_torch import (
                INITLESS,
                pack_latent_streams,
                unpack_latent_streams,
            )
        else:
            from dinkster_inference_torch.operations import INITLESS
            from dinkster_inference_torch.latent_streams import (
                pack_latent_streams, unpack_latent_streams
            )
        layer = INITLESS.conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(2)
        assert torch.equal(layer(torch.ones(1, 1, 2, 2)), torch.full((1, 1, 2, 2), 2.))
        tensor = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        value = MultiStreamLatent((LatentStream('video', tensor),))
        packed, layout = pack_latent_streams(value)
        assert torch.equal(unpack_latent_streams(packed, layout).streams[0].payload, tensor)
        assert blocked.isdisjoint(sys.modules)
        for name in (
            'anima_model', 'anima_component', 'attention', 'gemma_tokenizer',
            'minimax_h3_assembly', 'component_publisher', 'wan21_component', 'wiring'
        ):
            assert 'dinkster_inference_torch.' + name not in sys.modules, name
    """)


@pytest.mark.parametrize("order", ["exports", "reverse", "modules", "star"])
def test_all_exports_have_authoritative_identity_in_clean_process(order: str) -> None:
    bindings = _typed_bindings()
    _run(f"""
        import importlib
        import dinkster_inference_torch as package

        bindings = {bindings!r}
        order = {order!r}
        if order == 'modules':
            for module in sorted({{module for module, _ in bindings.values()}}):
                importlib.import_module('dinkster_inference_torch.' + module)
        elif order == 'star':
            from dinkster_inference_torch import *
            for name in package.__all__:
                assert globals()[name] is getattr(package, name), name

        names = list(bindings)
        if order == 'reverse':
            names.reverse()
        for name in names:
            value = getattr(package, name)
            module, attribute = bindings[name]
            owner = importlib.import_module('dinkster_inference_torch.' + module)
            assert value is getattr(owner, attribute), name
            assert vars(package)[name] is value, name
            imported = __import__('dinkster_inference_torch', fromlist=[name])
            assert getattr(imported, name) is value, name
        for name, (module, attribute) in vars(package)['_EXPORTS'].items():
            if attribute is None:
                assert getattr(package, name) is importlib.import_module(
                    'dinkster_inference_torch.' + module
                ), name
        assert set(bindings) <= set(dir(package))
    """)


@pytest.mark.parametrize(
    "first_import",
    [
        "from dinkster_inference_torch import component_publisher",
        "import dinkster_inference_torch.component_publisher",
        "from dinkster_inference_torch import ComponentPublisher",
        "import dinkster_inference_torch.minimax_h3_assembly",
    ],
)
def test_function_submodule_collision_and_shared_state(first_import: str) -> None:
    _run(f"""
        import importlib
        {first_import}
        import dinkster_inference_torch as package
        owner = importlib.import_module('dinkster_inference_torch.component_publisher')
        from dinkster_inference_torch import component_publisher, use_component_publisher
        assert component_publisher is owner.component_publisher
        assert package.component_publisher is owner.component_publisher
        assert use_component_publisher is owner.use_component_publisher
        publisher = object()
        with use_component_publisher(publisher):
            assert owner.component_publisher() is publisher
            assert package.component_publisher() is publisher
            importlib.reload(package)
            assert package.component_publisher is owner.component_publisher
            assert package.component_publisher() is publisher
        try:
            component_publisher()
        except RuntimeError as error:
            assert str(error) == 'no component publisher is available in this pack context'
        else:
            raise AssertionError('publisher context leaked')
        from dinkster_inference_torch import torch_euler
        from dinkster_inference_torch.solvers import euler
        assert torch_euler is euler
    """)


def test_requested_feature_import_failure_is_not_hidden_or_cached() -> None:
    _run("""
        import importlib.abc
        import sys
        import dinkster_inference_torch as package

        class MissingTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'torch':
                    raise ModuleNotFoundError('torch unavailable', name=fullname)

        blocker = MissingTorch()
        sys.meta_path.insert(0, blocker)
        try:
            package.AnimaModel
        except ModuleNotFoundError as error:
            assert error.name == 'torch'
        else:
            raise AssertionError('requested dependency error was hidden')
        assert 'AnimaModel' not in vars(package)
        sys.meta_path.remove(blocker)
        from dinkster_inference_torch import AnimaModel
        from dinkster_inference_torch.anima_model import AnimaModel as canonical
        assert AnimaModel is canonical
    """)
