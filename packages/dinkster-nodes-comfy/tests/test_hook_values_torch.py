"""CPU transport checks for structural hook strengths."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    FLOATS_TYPE,
    HOOK_KEYFRAMES_TYPE,
    HOOKS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_inference_wire import register_inference_types
from dinkster_values import TypeRegistry

from dinkster.compose import PackSpec, ServingComposer


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


class FloatStrengthSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-values-source.source",
            outputs=(OutputSpec("strengths", TypeExpr.concrete(FLOATS_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(strengths=torch.tensor([0.5, -3.25], dtype=torch.bfloat16))


class FloatStrengthConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-values-consumer.consumer",
            inputs=(InputSpec("strengths", TypeExpr.concrete(FLOATS_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, strengths: Any) -> dict[str, object]:
        tensor = materialize_latent(strengths)
        return cls.outputs(
            summary=f"{tensor.dtype};{tuple(tensor.shape)};{hashlib.sha256(_raw_bytes(tensor)).hexdigest()}"
        )


class HookSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-objects.hook-source",
            outputs=(
                OutputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),
                OutputSpec("keyframes", TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP

        keyframes = hooks.HookKeyframeGroup()
        keyframes.add(hooks.HookKeyframe(strength=0.625, start_percent=0.25))
        hook = hooks.WeightHook(strength_model=0.75, strength_clip=-1.25)
        hook.hook_keyframe = keyframes
        group = hooks.HookGroup()
        group.add(hook)
        clip = CLIP(no_init=True)
        clip.patcher = ModelPatcher(torch.nn.Linear(2, 3), "cpu", "cpu")
        clip.cond_stage_model = clip.patcher.model
        clip.tokenizer = object()
        clip.layer_idx = None
        clip.tokenizer_options = {}
        clip.use_clip_schedule = True
        clip.apply_hooks_to_conds = group
        clip.patcher.forced_hooks = group
        return cls.outputs(hooks=group, keyframes=keyframes, clip=clip)


class HookConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-objects.hook-consumer",
            inputs=(
                InputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),
                InputSpec("keyframes", TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, hooks: Any, keyframes: Any, clip: Any) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks as inference_hooks

        assert isinstance(hooks, inference_hooks.HookGroup)
        assert isinstance(keyframes, inference_hooks.HookKeyframeGroup)
        assert len(hooks.hooks) == 1
        hook = hooks.hooks[0]
        assert hook.hook_keyframe is keyframes
        clone = hooks.clone()
        assert clone is not hooks
        assert clone.hooks[0] is not hook
        assert clone.hooks[0].hook_keyframe is keyframes
        assert clip.apply_hooks_to_conds is hooks
        assert clip.patcher.forced_hooks is hooks
        clip_clone = clip.clone(disable_dynamic=True)
        assert clip_clone.apply_hooks_to_conds is hooks
        assert clip_clone.patcher.forced_hooks is not hooks
        assert clip_clone.patcher.forced_hooks.hooks[0].hook_keyframe is keyframes
        keyframe = keyframes.keyframes[0]
        return cls.outputs(
            summary=(
                f"{len(hooks.hooks)};{len(keyframes.keyframes)};"
                f"{keyframe.strength};{keyframe.start_percent}"
            )
        )


class HookOwnershipConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-ownership-fixture.inspect",
            inputs=(
                InputSpec("original", TypeExpr.concrete(HOOKS_TYPE)),
                InputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, original: Any, hooks: Any, clip: Any, conditioning: Any
    ) -> dict[str, object]:
        initialize_inference_process()

        assert len(hooks.hooks) == 1
        hook = hooks.hooks[0]
        assert hook._strength_model == 0.75
        assert hook._strength_clip == -1.25
        hook_frames = [
            (frame.strength, frame.start_percent) for frame in hook.hook_keyframe.keyframes
        ]
        assert hook_frames == [(0.625, 0.25)]
        assert clip.apply_hooks_to_conds is None
        assert clip.use_clip_schedule is True
        forced = clip.patcher.forced_hooks
        assert forced is not hooks
        assert len(forced.hooks) == 1
        forced_hook = forced.hooks[0]
        assert forced_hook._strength_model == 0.75
        assert forced_hook._strength_clip == -1.25
        forced_frames = [
            (frame.strength, frame.start_percent) for frame in forced_hook.hook_keyframe.keyframes
        ]
        assert forced_frames == [(0.625, 0.25)]
        pair = original._dinkster_resident_refs[0]
        conditioned = conditioning[0][1]["hooks"]
        for group in (hooks, forced, conditioned):
            assert sum(owner is pair for owner in group._dinkster_resident_refs) == 1
            assert group.hooks[0].weights["weight"][1][0] is pair[0]["weight"]
            assert group.hooks[0].weights_clip["weight"][1][0] is pair[1]["weight"]
        assert torch.equal(conditioning[0][0], torch.tensor([[2.0, -7.0]]))
        return cls.outputs(summary="0.75;-1.25;0.625;0.25")


class HookOwnershipSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-ownership-fixture.source",
            outputs=(
                OutputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks

        return cls.outputs(
            hooks=_declared_weight_group()[0],
            clip=_clip(),
            conditioning=[[torch.tensor([[2.0, -7.0]]), {"hooks": hooks.HookGroup()}]],
        )


def _clip() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    clip = CLIP(no_init=True)
    clip.patcher = ModelPatcher(torch.nn.Linear(2, 3), "cpu", "cpu")
    clip.cond_stage_model = clip.patcher.model
    clip.tokenizer = object()
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    clip.patcher.forced_hooks = None
    return clip


def _declared_weight_group() -> tuple[Any, object, Any]:
    initialize_inference_process()
    from dinkster_inference import hooks

    weight_pair = (
        {"weight": torch.tensor([[0.25, -1.0], [2.0, 0.5], [-0.75, 3.0]])},
        {"weight": torch.tensor([[-0.5, 1.25], [0.75, -2.0], [3.5, 0.25]])},
    )
    keyframes = hooks.HookKeyframeGroup()
    keyframes.add(hooks.HookKeyframe(strength=-0.375, start_percent=0.1))
    group = hooks.create_hook_model_as_lora(*weight_pair, strength_model=0.75, strength_clip=-1.25)
    hook = group.hooks[0]
    hook.hook_keyframe = keyframes
    group._dinkster_resident_refs = (weight_pair,)
    return group, weight_pair, keyframes


def test_floats_tensor_codec_preserves_dtype_and_bytes() -> None:
    source = torch.tensor([0.5, -3.25], dtype=torch.bfloat16)
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(FLOATS_TYPE)
    decoded = spec.decode(spec.encode(source))
    restored = materialize_latent(decoded)

    assert restored.dtype == source.dtype
    assert restored.shape == source.shape
    assert _raw_bytes(restored) == _raw_bytes(source)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_transports_float_strengths_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            module = "hook_values_fixture_" + str(in_process)
            (tmp_path / f"{module}.py").write_text(
                "from test_hook_values_torch import (\n"
                "    FloatStrengthConsumer, FloatStrengthSource, HookConsumer, HookSource,\n"
                ")\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "SOURCE_NODES = [FloatStrengthSource]\n"
                "CONSUMER_NODES = [FloatStrengthConsumer]\n"
                "HOOK_NODES = [HookSource, HookConsumer]\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            consumer_manifest = tmp_path / "consumer-dinkster-pack.toml"
            hook_manifest = tmp_path / "hook-dinkster-pack.toml"
            source_manifest.write_text(
                '[pack]\nname = "hook-values-source"\npublisher = "dinkster"\n'
                'namespaces = ["hook-values-source"]\n'
                f'[pack.entry]\nnodes = "{module}:SOURCE_NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            consumer_manifest.write_text(
                '[pack]\nname = "hook-values-consumer"\npublisher = "dinkster"\n'
                'namespaces = ["hook-values-consumer"]\n'
                f'[pack.entry]\nnodes = "{module}:CONSUMER_NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            hook_manifest.write_text(
                '[pack]\nname = "hook-objects"\npublisher = "dinkster"\n'
                'namespaces = ["hook-objects"]\n'
                f'[pack.entry]\nnodes = "{module}:HOOK_NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            await composer.add_pack(PackSpec(manifest=consumer_manifest, in_process=in_process))
            await composer.add_pack(PackSpec(manifest=hook_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("hook-values-source.source", {}),
                    "consumer": GraphNode(
                        "hook-values-consumer.consumer", {"strengths": Link("source", "strengths")}
                    ),
                    "hook-source": GraphNode("hook-objects.hook-source", {}),
                    "hook-consumer": GraphNode(
                        "hook-objects.hook-consumer",
                        {
                            "hooks": Link("hook-source", "hooks"),
                            "keyframes": Link("hook-source", "keyframes"),
                            "clip": Link("hook-source", "clip"),
                        },
                    ),
                }
            )
            expected_tensor = torch.tensor([0.5, -3.25], dtype=torch.bfloat16)
            expected = (
                f"{expected_tensor.dtype};{tuple(expected_tensor.shape)};"
                f"{hashlib.sha256(_raw_bytes(expected_tensor)).hexdigest()}"
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer", "hook-consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == expected
                assert result.outputs["hook-consumer"]["summary"].resolve() == "1;1;0.625;0.25"
                if attempt:
                    assert set(result.cached) == {
                        "source",
                        "consumer",
                        "hook-source",
                        "hook-consumer",
                    }
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_hosts_retain_weight_owners_through_public_hook_nodes(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = root / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src")))}
        )
        try:
            (tmp_path / "hook_ownership_fixture.py").write_text(
                "from test_hook_values_torch import HookOwnershipConsumer, HookOwnershipSource\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "NODES = (HookOwnershipSource, HookOwnershipConsumer)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "hook-ownership-fixture-dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "hook-ownership-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["hook-ownership-fixture"]\n'
                '[pack.entry]\nnodes = "hook_ownership_fixture:NODES"\n'
                'types = "hook_ownership_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            manifests = (manifest, fixture_manifest)
            for pack_manifest in manifests:
                await composer.add_pack(
                    PackSpec(manifest=pack_manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=pack_manifest,
                        worker_group="hook-ownership",
                        group_manifests=manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("hook-ownership-fixture.source", {}),
                    "keyframes": GraphNode(
                        "CreateHookKeyframe", {"strength_mult": 0.625, "start_percent": 0.25}
                    ),
                    "set": GraphNode(
                        "SetHookKeyframes",
                        {
                            "hooks": Link("source", "hooks"),
                            "hook_kf": Link("keyframes", "HOOK_KF"),
                        },
                    ),
                    "combined-two": GraphNode(
                        "CombineHooks2",
                        {"hooks_A": Link("set", "HOOKS"), "hooks_B": Link("set", "HOOKS")},
                    ),
                    "combined-four": GraphNode(
                        "CombineHooks4",
                        {
                            "hooks_A": Link("combined-two", "HOOKS"),
                            "hooks_D": Link("combined-two", "HOOKS"),
                        },
                    ),
                    "combined-eight": GraphNode(
                        "CombineHooks8",
                        {
                            "hooks_A": Link("combined-four", "HOOKS"),
                            "hooks_H": Link("combined-four", "HOOKS"),
                        },
                    ),
                    "clip": GraphNode(
                        "SetClipHooks",
                        {
                            "clip": Link("source", "clip"),
                            "hooks": Link("combined-eight", "HOOKS"),
                            "apply_to_conds": False,
                            "schedule_clip": True,
                        },
                    ),
                    "conditioning": GraphNode(
                        "ConditioningSetProperties",
                        {
                            "cond_NEW": Link("source", "conditioning"),
                            "hooks": Link("combined-eight", "HOOKS"),
                        },
                    ),
                    "inspect": GraphNode(
                        "hook-ownership-fixture.inspect",
                        {
                            "original": Link("source", "hooks"),
                            "hooks": Link("combined-eight", "HOOKS"),
                            "clip": Link("clip", "clip"),
                            "conditioning": Link("conditioning", "conditioning"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "0.75;-1.25;0.625;0.25"
                if attempt:
                    assert set(result.cached) == set(graph.nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("native_first", (False, True))
@pytest.mark.parametrize("field", ("conditioning", "schedule", "both"))
@pytest.mark.parametrize("release", ("group", "keyframes", "group-owner", "group-ref"))
def test_clip_retained_hooks_invalidate_cached_outputs(
    monkeypatch: pytest.MonkeyPatch, native_first: bool, field: str, release: str
) -> None:
    from dinkster_api.v1 import CLIP_TYPE
    from dinkster_caches import MemoryLRUCache
    from dinkster_memory.resident import ResidentPool
    from dinkster_native.native import register_native_types
    from dinkster_protocol import CacheKey
    from dinkster_values import ResidentLookupError

    initialize_inference_process()
    from dinkster_inference import hooks
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    async def scenario() -> None:
        pool = ResidentPool(terminal_release=lambda _: True)
        monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
        monkeypatch.setattr("dinkster_native.native.default_pool", lambda: pool)
        registry = TypeRegistry()
        first, second = (
            (register_native_types, register_inference_types)
            if native_first
            else (register_inference_types, register_native_types)
        )
        first(registry)
        spec = registry.spec(CLIP_TYPE)
        second(registry)
        assert registry.spec(CLIP_TYPE) is spec
        cache = MemoryLRUCache(max_bytes=16384)
        pool.register_invalidator(cache.drop_referencing)
        keyframes = hooks.HookKeyframeGroup()
        keyframes.add(hooks.HookKeyframe(strength=-0.375, start_percent=0.25))
        hook = hooks.WeightHook()
        hook.hook_keyframe = keyframes
        group = hooks.HookGroup()
        group.add(hook)
        group_owner, group_ref = object(), object()
        if release == "group-owner":
            group._dinkster_resident_owner = group_owner
        group._dinkster_resident_refs = (group_ref,)

        def clip_with(retained: Any) -> Any:
            clip = CLIP(no_init=True)
            clip.patcher = ModelPatcher(torch.nn.Linear(2, 3), "cpu", "cpu")
            clip.cond_stage_model = clip.patcher.model
            clip.tokenizer = object()
            clip.layer_idx = None
            clip.tokenizer_options = {}
            clip.use_clip_schedule = field != "conditioning"
            clip.apply_hooks_to_conds = retained if field != "schedule" else None
            clip.patcher.forced_hooks = retained if field != "conditioning" else None
            return clip

        clips = {"first": clip_with(group), "second": clip_with(group), "other": clip_with(None)}
        encoded = {}
        for name, clip in clips.items():
            value = registry.wrap(CLIP_TYPE, clip)
            assert registry.wrap(CLIP_TYPE, clip).meta == value.meta
            encoded[name] = spec.encode(value.resolve())
            await cache.put(CacheKey(name), {"clip": value})
        registry.wrap(HOOKS_TYPE, group)
        registry.wrap(HOOK_KEYFRAMES_TYPE, keyframes)
        released = {
            "group": group,
            "keyframes": keyframes,
            "group-owner": group_owner,
            "group-ref": group_ref,
        }[release]
        assert pool.release_resident(released)
        for name in ("first", "second"):
            assert await cache.get(CacheKey(name)) is None
            with pytest.raises(ResidentLookupError):
                spec.decode(encoded[name])
        assert await cache.get(CacheKey("other")) is not None
        assert spec.decode(encoded["other"]) is clips["other"]
        assert hook.hook_keyframe is keyframes
        assert keyframes.keyframes[0].strength == -0.375

        foreign = TypeRegistry()
        monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: ResidentPool())
        register_inference_types(foreign)
        with pytest.raises(ResidentLookupError):
            foreign.spec(CLIP_TYPE).decode(encoded["other"])

    asyncio.run(scenario())


@pytest.mark.parametrize("native_first", (False, True))
def test_clip_clone_retains_actual_schedule_clone_not_source_group(
    monkeypatch: pytest.MonkeyPatch, native_first: bool
) -> None:
    from dinkster_memory.resident import ResidentPool
    from dinkster_native.native import register_native_types
    from dinkster_values import RESOURCE_REFS_META_KEY, ResidentLookupError, resident_resource_id

    initialize_inference_process()
    from dinkster_inference import hooks
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    pool = ResidentPool(terminal_release=lambda _: True)
    monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
    monkeypatch.setattr("dinkster_native.native.default_pool", lambda: pool)
    registry = TypeRegistry()
    if native_first:
        register_native_types(registry)
    register_inference_types(registry)
    spec = registry.spec(CLIP_TYPE)
    coerce = spec.coerce
    assert coerce is not None
    group = hooks.HookGroup()
    group.add(hooks.WeightHook())
    keyframes = group.hooks[0].hook_keyframe
    clip = CLIP(no_init=True)
    clip.patcher = ModelPatcher(torch.nn.Linear(2, 3), "cpu", "cpu")
    clip.cond_stage_model = clip.patcher.model
    clip.tokenizer = object()
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = True
    clip.apply_hooks_to_conds = None
    clip.patcher.forced_hooks = group
    clone = clip.clone(disable_dynamic=True)
    retained = clone.patcher.forced_hooks
    assert retained is not group
    assert retained.hooks[0].hook_keyframe is keyframes
    coerce(clone)
    cleared = clip.clone(disable_dynamic=True)
    coerce(cleared)
    cleared.patcher.forced_hooks = None
    cleared.use_clip_schedule = False
    cleared_value = registry.wrap(CLIP_TYPE, cleared)
    assert cleared_value.meta.get(RESOURCE_REFS_META_KEY) is None
    clone_value = registry.wrap(CLIP_TYPE, clone)
    assert clone_value.meta.get(RESOURCE_REFS_META_KEY) == (
        resident_resource_id(pool.rid_for(retained)),
        resident_resource_id(pool.rid_for(keyframes)),
    )
    encoded = spec.encode(clone_value.resolve())
    registry.wrap(CLIP_TYPE, clip)
    assert pool.release_resident(group)
    assert spec.decode(encoded) is clone
    assert pool.release_resident(keyframes)
    with pytest.raises(ResidentLookupError):
        spec.decode(encoded)
    assert spec.decode(spec.encode(cleared_value.resolve())) is cleared


def test_retained_keyframe_release_invalidates_groups_and_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_api.v1 import CONDITIONING_TYPE
    from dinkster_caches import MemoryLRUCache
    from dinkster_memory.resident import ResidentPool
    from dinkster_protocol import CacheKey
    from dinkster_values import (
        RESOURCE_ID_META_KEY,
        RESOURCE_REFS_META_KEY,
        ResidentLookupError,
        resident_resource_id,
    )

    initialize_inference_process()
    from dinkster_inference import hooks

    async def scenario() -> None:
        pool = ResidentPool(terminal_release=lambda _: True)
        monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
        registry = TypeRegistry()
        register_inference_types(registry)
        cache = MemoryLRUCache(max_bytes=16384)
        pool.register_invalidator(cache.drop_referencing)
        keyframes = hooks.HookKeyframeGroup()
        keyframes.add(hooks.HookKeyframe(strength=0.625, start_percent=0.25))
        group = hooks.HookGroup()
        hook = hooks.WeightHook()
        hook.hook_keyframe = keyframes
        group.add(hook)
        primary, explicit = object(), object()
        group._dinkster_resident_owner = primary
        group._dinkster_resident_refs = (explicit,)
        group_value = registry.wrap(HOOKS_TYPE, group)
        for _ in range(2):
            assert registry.wrap(HOOKS_TYPE, group).meta == group_value.meta
        assert group_value.resolve() is group
        assert group_value.meta.get(RESOURCE_ID_META_KEY) == resident_resource_id(
            pool.rid_for(primary)
        )
        assert group_value.meta.get(RESOURCE_REFS_META_KEY) == (
            resident_resource_id(pool.rid_for(explicit)),
            resident_resource_id(pool.rid_for(keyframes)),
        )

        clone = group.clone()
        assert clone.hooks[0].hook_keyframe is keyframes
        clone_value = registry.wrap(HOOKS_TYPE, clone)
        replacement = group.clone()
        # Normalize before replacement to detect stale automatically derived refs.
        coerce = registry.spec(HOOKS_TYPE).coerce
        assert coerce is not None
        coerce(replacement)
        supplied = hooks.HookKeyframeGroup()
        supplied.add(hooks.HookKeyframe(strength=-1.25, start_percent=0.75))
        replacement.set_keyframes_on_hooks(supplied)
        retained = replacement.hooks[0].hook_keyframe
        assert retained is not supplied and retained is not keyframes
        replacement_value = registry.wrap(HOOKS_TYPE, replacement)
        assert replacement_value.meta.get(RESOURCE_REFS_META_KEY) == (
            resident_resource_id(pool.rid_for(retained)),
        )
        assert keyframes.keyframes[0].strength == 0.625
        assert supplied.keyframes[0].strength == -1.25

        embedding = torch.tensor([[2.0, -7.0]], dtype=torch.float64)
        raw = [[embedding, {"hooks": group, "strength": 0.3}]]
        cloned_raw = [[embedding, {"hooks": clone, "strength": 0.3}]]
        replacement_raw = [[embedding, {"hooks": replacement, "strength": 1.7}]]
        values = {
            "group": (HOOKS_TYPE, group_value),
            "clone": (HOOKS_TYPE, clone_value),
            "conditioning": (CONDITIONING_TYPE, registry.wrap(CONDITIONING_TYPE, raw)),
            "cloned-conditioning": (
                CONDITIONING_TYPE,
                registry.wrap(CONDITIONING_TYPE, cloned_raw),
            ),
            "replacement": (HOOKS_TYPE, replacement_value),
            "replacement-conditioning": (
                CONDITIONING_TYPE,
                registry.wrap(CONDITIONING_TYPE, replacement_raw),
            ),
        }
        encoded = {}
        for name, (type_id, value) in values.items():
            encoded[name] = registry.spec(type_id).encode(value.resolve())
            await cache.put(CacheKey(name), {"value": value})
        # A separately resident keyframe is the release boundary, not the hook group.
        keyframe_value = registry.wrap(HOOK_KEYFRAMES_TYPE, keyframes)
        assert keyframe_value.resolve() is keyframes
        registry.wrap(HOOK_KEYFRAMES_TYPE, supplied)
        assert pool.release_resident(supplied)
        assert await cache.get(CacheKey("replacement")) is not None
        assert pool.release_resident(keyframes)
        for name in ("group", "clone", "conditioning", "cloned-conditioning"):
            assert await cache.get(CacheKey(name)) is None
            with pytest.raises(ResidentLookupError):
                registry.spec(values[name][0]).decode(encoded[name])
        for name in ("replacement", "replacement-conditioning"):
            assert await cache.get(CacheKey(name)) is not None
            assert registry.spec(values[name][0]).decode(encoded[name]) is values[name][1].resolve()
        assert pool.get(pool.rid_for(primary)) is primary
        assert pool.get(pool.rid_for(explicit)) is explicit
        assert raw[0][1] == {"hooks": group, "strength": 0.3}
        assert torch.equal(raw[0][0], embedding)
        assert pool.release_resident(retained)
        assert await cache.get(CacheKey("replacement-conditioning")) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("native_first", (False, True))
@pytest.mark.parametrize("schedule_clip", (False, True))
def test_weight_pair_ownership_survives_hook_clones_and_invalidates_dependents(
    monkeypatch: pytest.MonkeyPatch, native_first: bool, schedule_clip: bool
) -> None:
    from dinkster_api.v1 import CONDITIONING_TYPE
    from dinkster_caches import MemoryLRUCache
    from dinkster_memory.resident import ResidentPool
    from dinkster_native.native import register_native_types
    from dinkster_nodes_comfy.hook_conditioning import ConditioningSetProperties, SetClipHooks
    from dinkster_nodes_comfy.hook_scheduling import (
        CombineHooks2,
        CombineHooks4,
        CombineHooks8,
        SetHookKeyframes,
    )
    from dinkster_protocol import CacheKey
    from dinkster_values import ResidentLookupError

    initialize_inference_process()
    from dinkster_inference import hooks

    async def scenario() -> None:
        pool = ResidentPool(terminal_release=lambda _: True)
        monkeypatch.setattr("dinkster_inference_wire.types.default_pool", lambda: pool)
        monkeypatch.setattr("dinkster_native.native.default_pool", lambda: pool)
        registry = TypeRegistry()
        first, second = (
            (register_native_types, register_inference_types)
            if native_first
            else (register_inference_types, register_native_types)
        )
        first(registry)
        second(registry)
        cache = MemoryLRUCache(max_bytes=16384)
        pool.register_invalidator(cache.drop_referencing)

        source, weight_pair, old_keyframes = _declared_weight_group()
        registry.wrap(HOOKS_TYPE, source)
        replacement = hooks.HookKeyframeGroup()
        replacement.add(hooks.HookKeyframe(strength=0.625, start_percent=0.25))
        scheduled = SetHookKeyframes.execute(hooks=source, hook_kf=replacement)["HOOKS"]
        retained_keyframes = scheduled.hooks[0].hook_keyframe
        assert retained_keyframes is not replacement and retained_keyframes is not old_keyframes
        assert [(frame.strength, frame.start_percent) for frame in old_keyframes.keyframes] == [
            (-0.375, 0.1)
        ]
        retained_frames = [
            (frame.strength, frame.start_percent) for frame in retained_keyframes.keyframes
        ]
        assert retained_frames == [(0.625, 0.25)]

        shared = hooks.HookGroup()
        shared_hook = hooks.WeightHook(strength_model=0.75, strength_clip=-1.25)
        shared_hook.need_weight_init = False
        shared_hook.weights_clip = scheduled.hooks[0].weights_clip
        shared_hook.hook_keyframe = retained_keyframes
        shared.add(shared_hook)
        shared._dinkster_resident_refs = (weight_pair,)
        replaced = scheduled.clone()
        replaced_owner = object()
        replaced.hooks[0].weights_clip = object()
        replaced._dinkster_resident_refs = (replaced_owner,)
        empty = hooks.HookGroup()
        empty_owner = object()
        empty._dinkster_resident_refs = (empty_owner,)

        combined = CombineHooks2.execute(hooks_A=scheduled, hooks_B=shared)["HOOKS"]
        assert len(combined.hooks) == 2
        assert sum(owner is weight_pair for owner in combined._dinkster_resident_refs) == 1
        combined = CombineHooks4.execute(hooks_A=combined, hooks_B=replaced, hooks_C=empty)["HOOKS"]
        combined = CombineHooks8.execute(hooks_A=combined, hooks_H=combined)["HOOKS"]
        assert len(combined.hooks) == 2
        assert combined.hooks[0]._strength_model == 0.75
        assert combined.hooks[0]._strength_clip == -1.25
        assert sum(owner is weight_pair for owner in combined._dinkster_resident_refs) == 1
        assert not any(owner is old_keyframes for owner in combined._dinkster_resident_refs)
        assert not any(owner is replaced_owner for owner in combined._dinkster_resident_refs)
        assert not any(owner is empty_owner for owner in combined._dinkster_resident_refs)

        clip = SetClipHooks.execute(
            clip=_clip(), hooks=combined, apply_to_conds=False, schedule_clip=schedule_clip
        )["clip"]
        forced = clip.patcher.forced_hooks
        assert clip.apply_hooks_to_conds is None
        assert forced is not combined
        if schedule_clip:
            assert all(hook.hook_keyframe is retained_keyframes for hook in forced.hooks)
        else:
            assert all(not hook.hook_keyframe.keyframes for hook in forced.hooks)
        conditioning_input = [[torch.tensor([[2.0, -7.0]], dtype=torch.float64), {"hooks": shared}]]
        conditioning = ConditioningSetProperties.execute(
            cond_NEW=conditioning_input,
            hooks=combined,
        )["conditioning"]
        conditioned_group = conditioning[0][1]["hooks"]
        assert conditioned_group is not combined
        assert conditioning_input[0][1]["hooks"] is shared
        assert conditioned_group.hooks[0].hook_keyframe is retained_keyframes

        clip_value = registry.wrap(CLIP_TYPE, clip)
        conditioning_value = registry.wrap(CONDITIONING_TYPE, conditioning)
        assert sum(owner is weight_pair for owner in clip._dinkster_resident_refs) == 1
        assert sum(owner is weight_pair for owner in conditioned_group._dinkster_resident_refs) == 1
        assert not any(owner is old_keyframes for owner in clip._dinkster_resident_refs)
        assert not any(
            owner is old_keyframes for owner in conditioned_group._dinkster_resident_refs
        )
        encoded = {
            "clip": registry.spec(CLIP_TYPE).encode(clip_value.resolve()),
            "conditioning": registry.spec(CONDITIONING_TYPE).encode(conditioning_value.resolve()),
        }
        await cache.put(CacheKey("clip"), {"value": clip_value})
        await cache.put(CacheKey("conditioning"), {"value": conditioning_value})

        assert pool.release_resident(old_keyframes)
        assert await cache.get(CacheKey("clip")) is not None
        assert await cache.get(CacheKey("conditioning")) is not None
        assert registry.spec(CLIP_TYPE).decode(encoded["clip"]) is clip
        assert (
            registry.spec(CONDITIONING_TYPE).decode(encoded["conditioning"])
            is conditioning_value.resolve()
        )

        unrelated = (torch.nn.Linear(2, 3), torch.nn.Linear(2, 3))
        pool.rid_for(unrelated)
        assert pool.release_resident(unrelated)
        assert await cache.get(CacheKey("clip")) is not None
        assert await cache.get(CacheKey("conditioning")) is not None
        assert registry.spec(CLIP_TYPE).decode(encoded["clip"]) is clip

        assert pool.release_resident(weight_pair)
        for name, (type_id, _) in {
            "clip": (CLIP_TYPE, clip_value),
            "conditioning": (CONDITIONING_TYPE, conditioning_value),
        }.items():
            assert await cache.get(CacheKey(name)) is None
            with pytest.raises(ResidentLookupError):
                registry.spec(type_id).decode(encoded[name])

    asyncio.run(scenario())
