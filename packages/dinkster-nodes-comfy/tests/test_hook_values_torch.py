"""CPU transport checks for structural hook strengths."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
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
