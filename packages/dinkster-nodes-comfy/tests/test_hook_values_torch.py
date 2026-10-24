"""CPU transport checks for structural hook strengths."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
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
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import hooks

        keyframes = hooks.HookKeyframeGroup()
        keyframes.add(hooks.HookKeyframe(strength=0.625, start_percent=0.25))
        hook = hooks.WeightHook(strength_model=0.75, strength_clip=-1.25)
        hook.hook_keyframe = keyframes
        group = hooks.HookGroup()
        group.add(hook)
        return cls.outputs(hooks=group, keyframes=keyframes)


class HookConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-objects.hook-consumer",
            inputs=(
                InputSpec("hooks", TypeExpr.concrete(HOOKS_TYPE)),
                InputSpec("keyframes", TypeExpr.concrete(HOOK_KEYFRAMES_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, hooks: Any, keyframes: Any) -> dict[str, object]:
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
