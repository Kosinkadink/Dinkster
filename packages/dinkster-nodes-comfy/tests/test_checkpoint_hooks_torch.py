"""Checkpoint hook behavior with actual CPU patchers and controlled loader I/O."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api import v1
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
from dinkster_nodes_comfy.hook_checkpoint import (
    CreateHookModelAsLora,
    CreateHookModelAsLoraModelOnly,
)

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
    mounts.add(MountDef(id="checkpoints", path=path.parent), kind="model/checkpoint")
    ref = mounts.bind_name(path.name, "model/checkpoint")
    assert isinstance(ref, LocalAssetRef)
    return ref


def _patcher(values: list[list[float]]) -> Any:
    from dinkster_inference.model_patcher import ModelPatcher

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor(values))
    model.model_sampling = torch.nn.Linear(1, 1, bias=False)
    return ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("has_clip", [False, True])
def test_checkpoint_hooks_preserve_weights_prior_hooks_and_strength_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, verified: bool, has_clip: bool
) -> None:
    initialize_inference_process()
    from dinkster_inference import hooks, sd

    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"cpu checkpoint loader seam")
    ref = _asset(checkpoint, verified)
    roots = (tmp_path / "embedding-b", tmp_path / "embedding-a")

    def mount_roots(kind: str) -> tuple[Path, ...]:
        assert kind == "model/embedding"
        return roots

    monkeypatch.setattr(v1, "mount_roots_for_kind", mount_roots, raising=False)
    model = _patcher([[2.0, -7.0], [11.0, 0.5]])
    clip = _patcher([[-3.0, 13.0]])
    calls = []

    def load(path: str, **options: Any) -> tuple[Any, Any, object, None]:
        assert path == str(checkpoint)
        assert options == {
            "output_vae": True,
            "output_clip": True,
            "embedding_directory": [str(root) for root in roots],
        }
        calls.append(path)
        return model, SimpleNamespace(patcher=clip) if has_clip else None, object(), None

    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    previous = hooks.HookGroup()
    prior = hooks.WeightHook(strength_model=-0.25, strength_clip=0.75)
    previous.add(prior)
    explicit_owner = object()
    old_keyframes = hooks.HookKeyframeGroup()
    previous._dinkster_resident_refs = (explicit_owner, old_keyframes, explicit_owner)
    previous._dinkster_hook_keyframe_refs = (old_keyframes,)
    result = CreateHookModelAsLora.execute(
        ckpt_name=ref, strength_model=-0.7, strength_clip=1.3, prev_hooks=previous
    )["HOOKS"]
    assert isinstance(result, hooks.HookGroup)
    assert result._dinkster_resident_refs[0] is explicit_owner
    assert len(result._dinkster_resident_refs) == 2
    assert not any(owner is old_keyframes for owner in result._dinkster_resident_refs)
    assert len(previous.hooks) == 1 and previous.hooks[0] is prior
    assert len(result.hooks) == 2
    assert result.hooks[0] is not prior
    assert result.hooks[0].strength_model == -0.25
    added = result.hooks[1]
    assert added.strength_model == -0.7 and added.strength_clip == 1.3
    assert not added.need_weight_init
    assert set(added.weights) == {"weight"}
    assert added.weights["weight"][0] == "model_as_lora"
    assert torch.equal(added.weights["weight"][1][0], torch.tensor([[2.0, -7.0], [11.0, 0.5]]))
    if has_clip:
        assert set(added.weights_clip) == {"weight"}
        assert torch.equal(added.weights_clip["weight"][1][0], torch.tensor([[-3.0, 13.0]]))
    else:
        assert added.weights_clip is None
    assert "model_sampling.weight" in model.model.state_dict()
    assert "model_sampling.weight" in clip.model.state_dict()

    model_only = CreateHookModelAsLoraModelOnly.execute(ckpt_name=ref, strength_model=0.0)["HOOKS"]
    assert isinstance(model_only, hooks.HookGroup)
    assert len(model_only.hooks) == 1
    assert model_only.hooks[0].strength_model == 0.0
    assert model_only.hooks[0].strength_clip == 0.0
    assert torch.equal(model_only.hooks[0].weights["weight"][1][0], model.model.weight)
    assert len(calls) == 1
    assert added.strength_model == -0.7 and added.strength_clip == 1.3


def test_checkpoint_hook_refreshes_embedding_authority_and_revalidates_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    initialize_inference_process()
    from dinkster_inference import hooks, sd

    checkpoint = tmp_path / "refresh.bin"
    checkpoint.write_bytes(b"first revision")
    ref = _asset(checkpoint, False)
    roots = (tmp_path / "first",)
    monkeypatch.setattr(v1, "mount_roots_for_kind", lambda kind: roots, raising=False)

    def load(path: str, **options: Any) -> tuple[Any, None, None, None]:
        assert options["embedding_directory"] == [str(root) for root in roots]
        assert options["output_vae"] is True and options["output_clip"] is True
        return _patcher([[3.0 if roots else -17.0]]), None, None, None

    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    first = CreateHookModelAsLoraModelOnly.execute(ckpt_name=ref, strength_model=0.8)["HOOKS"]
    roots = ()
    revoked = CreateHookModelAsLoraModelOnly.execute(ckpt_name=ref, strength_model=0.8)["HOOKS"]
    assert isinstance(first, hooks.HookGroup) and isinstance(revoked, hooks.HookGroup)
    assert first.hooks[0].weights["weight"][1][0].item() == 3.0
    assert revoked.hooks[0].weights["weight"][1][0].item() == -17.0
    checkpoint.write_bytes(b"different checkpoint revision")
    with pytest.raises(AssetError, match="unavailable|revoked|changed"):
        CreateHookModelAsLoraModelOnly.execute(ckpt_name=ref, strength_model=0.8)


def test_checkpoint_weights_share_residency_across_assets_nodes_and_strengths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    initialize_inference_process()
    from dinkster_inference import sd
    from dinkster_memory.resident import default_pool

    paths = (tmp_path / "first.bin", tmp_path / "second.bin", tmp_path / "same-first.bin")
    paths[0].write_bytes(b"first checkpoint")
    paths[1].write_bytes(b"second checkpoint")
    paths[2].write_bytes(paths[0].read_bytes())
    assets = [_asset(path, True) for path in paths]
    monkeypatch.setattr(v1, "mount_roots_for_kind", lambda kind: (), raising=False)
    model_a = _patcher([[2.0, -7.0], [11.0, 0.5]])
    model_b = _patcher([[-13.0, 3.0, 8.0]])
    calls: list[str] = []

    def load(path: str, **options: Any) -> tuple[Any, Any, None, None]:
        assert options == {"output_vae": True, "output_clip": True, "embedding_directory": []}
        calls.append(path)
        model = model_b if path == str(paths[1]) else model_a
        return model, SimpleNamespace(patcher=model), None, None

    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    baseline = default_pool().footprint("ram")
    retained = []
    previous = None
    for index, strength in ((0, -0.75), (1, 1.25), (2, 0.0), (1, -2.5), (0, 3.0)):
        result = CreateHookModelAsLora.execute(
            ckpt_name=assets[index],
            strength_model=strength,
            strength_clip=-strength,
            prev_hooks=previous,
        )["HOOKS"]
        added = result.hooks[-1]
        assert added.strength_model == strength and added.strength_clip == -strength
        expected = model_b.model.weight if index == 1 else model_a.model.weight
        torch.testing.assert_close(added.weights["weight"][1][0], expected, rtol=0, atol=0)
        assert added.weights_clip["weight"][1][0].data_ptr() == expected.data_ptr()
        owners = [
            owner
            for owner in result._dinkster_resident_refs
            if owner[0]["weight"].data_ptr() == expected.data_ptr()
        ]
        assert len(owners) == 1
        retained.append(owners[0])
        previous = result
    assert retained[0] is retained[2] is retained[4]
    assert retained[1] is retained[3] and retained[0] is not retained[1]
    assert calls == [str(paths[0]), str(paths[1])]
    # Each pair aliases model/CLIP storage, so charge each retained storage once.
    assert default_pool().footprint("ram") - baseline == 28


def test_checkpoint_hook_rejects_unbound_filename() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        CreateHookModelAsLoraModelOnly.execute(ckpt_name="unbound.ckpt", strength_model=1.0)


class CheckpointHookSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="checkpoint-hooks-fixture.source",
            inputs=(InputSpec("embedding_root", TypeExpr.concrete(CORE_STRING)),),
            outputs=(OutputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),),
        )

    @classmethod
    def execute(cls, *, embedding_root: str) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks, sd, utils

        calls: list[str] = []

        # Keep checkpoint architecture construction controlled, but read real
        # safe tensor bytes through the bound path and the production roots API.
        def load(path: str, **options: Any) -> tuple[Any, Any, object, None]:
            assert options == {
                "output_vae": True,
                "output_clip": True,
                "embedding_directory": [embedding_root],
            }, repr(options)
            actual_roots = v1.mount_roots_for_kind("model/embedding")
            assert actual_roots == (Path(embedding_root),), repr(actual_roots)
            state = utils.load_torch_file(path, safe_load=True)
            model = _patcher(state["model"].tolist())
            clip = _patcher(state["clip"].tolist())
            calls.append(path)
            return model, SimpleNamespace(patcher=clip), object(), None

        load.calls = calls  # type: ignore[attr-defined]
        sd.load_checkpoint_guess_config = load
        previous = hooks.HookGroup()
        previous.add(hooks.WeightHook(strength_model=-0.25, strength_clip=0.75))
        return cls.outputs(hooks=previous)


class CheckpointHookInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="checkpoint-hooks-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(HOOKS_TYPE))
                for name in ("source", "full", "model_only")
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, source: Any, full: Any, model_only: Any) -> dict[str, object]:
        from dinkster_inference import sd

        assert len(source.hooks) == 1
        assert source.hooks[0].strength_model == -0.25
        owners = []
        for group, strengths in ((full, (-0.7, 1.3)), (model_only, (0.0, 0.0))):
            assert len(group.hooks) == 2
            assert group.hooks[0] is not source.hooks[0]
            assert group.hooks[0] == source.hooks[0]
            added = group.hooks[1]
            assert (added.strength_model, added.strength_clip) == strengths
            assert set(added.weights) == {"weight"}
            assert set(added.weights_clip) == {"weight"}
            assert added.weights["weight"][0] == "model_as_lora"
            assert torch.equal(
                added.weights["weight"][1][0], torch.tensor([[2.0, -7.0], [11.0, 0.5]])
            )
            assert torch.equal(added.weights_clip["weight"][1][0], torch.tensor([[-3.0, 13.0]]))
            explicit = [
                owner
                for owner in group._dinkster_resident_refs
                if not any(owner is keyframes for keyframes in group._dinkster_hook_keyframe_refs)
            ]
            assert len(explicit) == 1, repr(explicit)
            owners.append(explicit[0])
        assert owners[0] is owners[1]
        assert len(sd.load_checkpoint_guess_config.calls) == 1
        return cls.outputs(summary="checkpoint weights and roots preserved")


CHECKPOINT_HOOK_FIXTURES = (CheckpointHookSource, CheckpointHookInspect)


@pytest.mark.parametrize("verified,in_process", ((False, True), (True, True), (True, False)))
def test_checkpoint_hooks_ordinary_hosts_use_snapshot_roots_and_resident_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, verified: bool
) -> None:
    initialize_inference_process()
    from dinkster_inference import sd
    from safetensors.torch import save_file

    original_loader = sd.load_checkpoint_guess_config
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file(
        {"model": torch.tensor([[2.0, -7.0], [11.0, 0.5]]), "clip": torch.tensor([[-3.0, 13.0]])},
        str(checkpoint),
    )
    embedding_root = tmp_path / "embeddings"
    embedding_root.mkdir()
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    table.add(MountDef(id="checkpoints", path=tmp_path), kind="model/checkpoint")
    table.add(MountDef(id="embeddings", path=embedding_root), kind="model/embedding")
    table.scan("embeddings")
    if verified:
        table.scan("checkpoints")
    asset = table.bind_name(checkpoint.name, "model/checkpoint")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            root / "dinkster-pack.toml",
            root / "tests/fixtures/checkpoint_hooks/dinkster-pack.toml",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src"))),
            }
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="checkpoint-hooks-fixture",
                        group_manifests=manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode(
                        "checkpoint-hooks-fixture.source", {"embedding_root": str(embedding_root)}
                    ),
                    "full": GraphNode(
                        "CreateHookModelAsLora",
                        {
                            "ckpt_name": asset.to_wire(),
                            "strength_model": -0.7,
                            "strength_clip": 1.3,
                            "prev_hooks": Link("source", "hooks"),
                        },
                    ),
                    "model_only": GraphNode(
                        "CreateHookModelAsLoraModelOnly",
                        {
                            "ckpt_name": asset.to_wire(),
                            "strength_model": 0.0,
                            "prev_hooks": Link("source", "hooks"),
                        },
                    ),
                    "inspect": GraphNode(
                        "checkpoint-hooks-fixture.inspect",
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
                    == "checkpoint weights and roots preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()
            sd.load_checkpoint_guess_config = original_loader

    asyncio.run(scenario())
