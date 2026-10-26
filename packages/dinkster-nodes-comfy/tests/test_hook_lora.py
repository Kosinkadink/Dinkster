"""LoRA hook behavior with real CPU safetensors and controlled hook ownership."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    HOOKS_TYPE,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import AssetError, MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.hook_lora import HOOK_LORA_NODES, CreateHookLora, CreateHookLoraModelOnly

from dinkster.compose import PackSpec, ServingComposer


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _asset(path: Path, verified: bool) -> AssetRef | LocalAssetRef:
    if verified:
        return AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=_Resolver(path))
    mounts = MountTable()
    mounts.add(MountDef(id="loras", path=path.parent), kind="model/lora")
    ref = mounts.bind_name(path.name, "model/lora")
    assert isinstance(ref, LocalAssetRef)
    return ref


def _write_lora(path: Path) -> None:
    from safetensors.torch import save_file

    save_file(
        {
            "lora_unet.linear.lora_down.weight": torch.tensor([[2.0, -7.0]]),
            "lora_unet.linear.lora_up.weight": torch.tensor([[11.0], [0.5]]),
        },
        str(path),
    )


def test_lora_hook_schemas_are_torch_free_and_registered() -> None:
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

from dinkster_api.v1 import HOOKS_TYPE, TypeExpr
from dinkster_nodes_comfy.hook_lora import HOOK_LORA_NODES

expected = (
    (("lora_name", "strength_model", "strength_clip", "prev_hooks"), "Create Hook LoRA"),
    (("lora_name", "strength_model", "prev_hooks"), "Create Hook LoRA (MO)"),
)
for node, (names, display) in zip(HOOK_LORA_NODES, expected, strict=True):
    schema = node.schema()
    assert tuple(item.id for item in schema.inputs) == names
    assert schema.display_name == display
    assert schema.inputs[0].widget.kind == "model/lora"
    assert [(item.id, item.type) for item in schema.outputs] == [
        ("HOOKS", TypeExpr.concrete(HOOKS_TYPE))
    ]
assert "torch" not in sys.modules
assert not any(name.startswith("dinkster_inference.") for name in sys.modules)
""",
        ],
        check=True,
        cwd=root,
    )
    from dinkster_nodes_comfy import COMFY_NODES

    assert all(node in COMFY_NODES for node in HOOK_LORA_NODES)


@pytest.mark.parametrize("verified", [False, True])
def test_lora_hooks_load_safetensors_safely_preserve_sources_and_retain_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verified: bool
) -> None:
    initialize_inference_process()
    from dinkster_inference import hooks, utils

    checkpoint = tmp_path / "tiny.safetensors"
    _write_lora(checkpoint)
    asset = _asset(checkpoint, verified)
    original_load = utils.load_torch_file
    calls: list[tuple[str, dict[str, object]]] = []

    def load(path: str, **options: object) -> dict[str, Any]:
        calls.append((path, options))
        return original_load(path, **options)

    monkeypatch.setattr(utils, "load_torch_file", load)
    previous = hooks.HookGroup()
    prior = hooks.WeightHook(strength_model=-0.25, strength_clip=0.75)
    previous.add(prior)
    owner = object()
    old_keyframes = hooks.HookKeyframeGroup()
    previous._dinkster_resident_refs = (owner, old_keyframes, owner)
    previous._dinkster_hook_keyframe_refs = (old_keyframes,)

    full = CreateHookLora.execute(
        lora_name=asset, strength_model=-0.7, strength_clip=1.3, prev_hooks=previous
    )["HOOKS"]
    model_only = CreateHookLoraModelOnly.execute(
        lora_name=asset, strength_model=0.8, prev_hooks=full
    )["HOOKS"]

    assert calls == [(str(checkpoint), {"safe_load": True})]
    assert len(previous.hooks) == 1 and previous.hooks[0] is prior
    assert len(full.hooks) == 2 and full.hooks[0] is not prior and full.hooks[0] == prior
    assert (full.hooks[1].strength_model, full.hooks[1].strength_clip) == (-0.7, 1.3)
    assert (model_only.hooks[-1].strength_model, model_only.hooks[-1].strength_clip) == (0.8, 0.0)
    expected = {
        "lora_unet.linear.lora_down.weight": torch.tensor([[2.0, -7.0]]),
        "lora_unet.linear.lora_up.weight": torch.tensor([[11.0], [0.5]]),
    }
    for hook in (full.hooks[1], model_only.hooks[-1]):
        assert hook.weights.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(hook.weights[key], value, rtol=0, atol=0)
    assert full._dinkster_resident_refs[0] is owner
    assert not any(item is old_keyframes for item in full._dinkster_resident_refs)
    assert model_only._dinkster_resident_refs[0] is owner
    assert any(item is full.hooks[1].weights for item in model_only._dinkster_resident_refs)
    assert (
        CreateHookLoraModelOnly.execute(lora_name=asset, strength_model=0.0, prev_hooks=previous)[
            "HOOKS"
        ]
        is previous
    )


def test_lora_hook_revalidates_local_path_before_resident_lookup(tmp_path: Path) -> None:
    initialize_inference_process()
    checkpoint = tmp_path / "tiny.safetensors"
    _write_lora(checkpoint)
    asset = _asset(checkpoint, verified=False)
    CreateHookLoraModelOnly.execute(lora_name=asset, strength_model=1.0)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(AssetError, match="unavailable|revoked|changed"):
        CreateHookLoraModelOnly.execute(lora_name=asset, strength_model=1.0)


class LoraHookSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-hooks-fixture.source",
            inputs=(),
            outputs=(OutputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        hooks = __import__("dinkster_inference.hooks", fromlist=["HookGroup"])
        previous = hooks.HookGroup()
        previous.add(hooks.WeightHook(strength_model=-0.25, strength_clip=0.75))
        return cls.outputs(hooks=previous)


class LoraHookInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-hooks-fixture.inspect",
            inputs=(
                InputSpec("source", TypeExpr.concrete(HOOKS_TYPE)),
                InputSpec("full", TypeExpr.concrete(HOOKS_TYPE)),
                InputSpec("model_only", TypeExpr.concrete(HOOKS_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, source: Any, full: Any, model_only: Any) -> dict[str, object]:
        assert len(source.hooks) == 1
        assert len(full.hooks) == 2 and len(model_only.hooks) == 2
        assert full.hooks[0] == source.hooks[0] and full.hooks[0] is not source.hooks[0]
        assert model_only.hooks[0] == source.hooks[0] and model_only.hooks[0] is not source.hooks[0]
        assert (full.hooks[-1].strength_model, full.hooks[-1].strength_clip) == (-0.7, 1.3)
        assert (
            model_only.hooks[-1].strength_model,
            model_only.hooks[-1].strength_clip,
        ) == (0.8, 0.0)
        assert (source.hooks[0].strength_model, source.hooks[0].strength_clip) == (-0.25, 0.75)
        expected = {
            "lora_unet.linear.lora_down.weight": torch.tensor([[2.0, -7.0]]),
            "lora_unet.linear.lora_up.weight": torch.tensor([[11.0], [0.5]]),
        }
        for hook in (full.hooks[-1], model_only.hooks[-1]):
            assert hook.weights.keys() == expected.keys()
            for key, value in expected.items():
                assert torch.equal(hook.weights[key], value)
        return cls.outputs(summary="lora weights and sources preserved")


LORA_HOOK_FIXTURES = (LoraHookSource, LoraHookInspect)


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("verified", (False, True))
def test_lora_hooks_execute_through_fixture_pack_on_both_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, verified: bool
) -> None:
    checkpoint = tmp_path / "tiny.safetensors"
    _write_lora(checkpoint)
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    table.add(MountDef(id="loras", path=tmp_path), kind="model/lora")
    if verified:
        table.scan("loras")
    asset = table.bind_name(checkpoint.name, "model/lora")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "tests/fixtures/lora_hooks/dinkster-pack.toml"
        native = root / "dinkster-pack.toml"
        manifests = (native, manifest)
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src"))),
            }
        )
        try:
            for entry in manifests:
                pack = PackSpec(manifest=entry, in_process=True)
                if not in_process:
                    pack = PackSpec(
                        manifest=entry,
                        worker_group="lora-hooks-fixture",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            graph = Graph(
                nodes={
                    "source": GraphNode("lora-hooks-fixture.source", {}),
                    "full": GraphNode(
                        "CreateHookLora",
                        {
                            "lora_name": asset.to_wire(),
                            "strength_model": -0.7,
                            "strength_clip": 1.3,
                            "prev_hooks": Link("source", "hooks"),
                        },
                    ),
                    "model_only": GraphNode(
                        "CreateHookLoraModelOnly",
                        {
                            "lora_name": asset.to_wire(),
                            "strength_model": 0.8,
                            "prev_hooks": Link("source", "hooks"),
                        },
                    ),
                    "inspect": GraphNode(
                        "lora-hooks-fixture.inspect",
                        {
                            "source": Link("source", "hooks"),
                            "full": Link("full", "HOOKS"),
                            "model_only": Link("model_only", "HOOKS"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "lora weights and sources preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
