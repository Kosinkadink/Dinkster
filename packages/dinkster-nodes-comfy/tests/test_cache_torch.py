"""CPU behavior for native EasyCache and LazyCache holders."""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from dinkster_api.v1 import (
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.cache import (
    EasyCacheHolder,
    LazyCacheHolder,
    easycache_forward_wrapper,
    lazycache_predict_noise_wrapper,
)

from dinkster.compose import PackSpec, ServingComposer


def _holder(holder_type: type[Any]) -> Any:
    holder = holder_type(0.2, 0.0, 1.0, 2, False, False, 2)
    holder.start_t = torch.tensor(1.0)
    holder.end_t = torch.tensor(-1.0)
    return holder


def test_easycache_reuses_per_uuid_diffs_and_resets_for_metadata_change() -> None:
    holder = _holder(EasyCacheHolder)
    uuids = ["positive", "negative"]
    options = {"easycache": holder, "sigmas": torch.tensor([0.5]), "uuids": uuids}
    calls = 0

    def executor(x: torch.Tensor, _options: dict[str, Any]) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return x + 2

    source = torch.zeros((2, 2, 4, 4))
    assert torch.equal(easycache_forward_wrapper(executor, source.clone(), options), source + 2)
    holder.relative_transformation_rate = torch.tensor(0.0)
    reused = easycache_forward_wrapper(executor, source.clone(), options)
    assert torch.equal(reused, source + 2)
    assert calls == 1
    assert holder.total_steps_skipped == 1
    assert not holder.check_metadata(torch.zeros((2, 2, 5, 4)))
    assert holder.uuid_cache_diffs == {}
    assert holder.first_cond_uuid is None


def test_easycache_preserves_audio_cache_diffs() -> None:
    holder = _holder(EasyCacheHolder)
    options = {"easycache": holder, "sigmas": torch.tensor([0.5]), "uuids": ["cond"]}
    video = torch.zeros((1, 2, 4, 4))
    audio = torch.zeros((1, 2, 2, 4))

    def executor(value: list[torch.Tensor], _options: dict[str, Any]) -> list[torch.Tensor]:
        return [value[0] + 3, value[1] + 5]

    result = easycache_forward_wrapper(executor, [video.clone(), audio.clone()], options)
    assert torch.equal(result[0], video + 3)
    assert torch.equal(result[1], audio + 5)
    holder.skip_current_step = True
    reused = easycache_forward_wrapper(executor, [video.clone(), audio.clone()], options)
    assert torch.equal(reused[0], video + 3)
    assert torch.equal(reused[1], audio + 5)


def test_lazycache_reuses_diff_and_metadata_reset_restarts_state() -> None:
    holder = _holder(LazyCacheHolder)
    options = {"transformer_options": {"easycache": holder}}
    calls = 0

    def executor(
        x: torch.Tensor, _timestep: torch.Tensor, _options: dict[str, Any]
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return x + 4

    source = torch.zeros((1, 2, 4, 4))
    assert torch.equal(
        lazycache_predict_noise_wrapper(executor, source.clone(), torch.tensor([0.5]), options),
        source + 4,
    )
    holder.relative_transformation_rate = torch.tensor(0.0)
    assert torch.equal(
        lazycache_predict_noise_wrapper(executor, source.clone(), torch.tensor([0.5]), options),
        source + 4,
    )
    assert calls == 1
    assert not holder.check_metadata(torch.zeros((1, 2, 5, 4)))
    assert holder.cache_diff is None


def test_cache_nodes_install_the_inference_wrapper_points() -> None:
    from dinkster_nodes_comfy.cache import EasyCache, LazyCache

    initialize_inference_process()

    class Model:
        class Inner:
            class LatentFormat:
                latent_channels = 4

            latent_format = LatentFormat()

        model = Inner()

        def __init__(self) -> None:
            self.model_options = {"transformer_options": {}}
            self.wrappers: list[tuple[str, str]] = []

        def clone(self) -> Model:
            return Model()

        def add_wrapper_with_key(self, point: str, key: str, _wrapper: object) -> None:
            self.wrappers.append((point, key))

    for node, expected in (
        (
            EasyCache,
            {
                ("outer_sample", "easycache"),
                ("calc_cond_batch", "easycache"),
                ("diffusion_model", "easycache"),
            },
        ),
        (LazyCache, {("outer_sample", "lazycache"), ("predict_noise", "lazycache")}),
    ):
        patched = node.execute(
            model=Model(), reuse_threshold=0.2, start_percent=0.15, end_percent=0.95, verbose=False
        )["MODEL"]
        assert set(patched.wrappers) == expected
        assert type(patched.model_options["transformer_options"]["easycache"]) is node.holder_type


_STRING = TypeExpr.concrete(CORE_STRING)


class CacheSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cache-source.source",
            outputs=(OutputSpec("value", _STRING),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        holder = _holder(LazyCacheHolder)
        source = torch.zeros((1, 2, 4, 4))
        result = lazycache_predict_noise_wrapper(
            lambda value, _timestep, _options: value + 4,
            source,
            torch.tensor([0.5]),
            {"transformer_options": {"easycache": holder}},
        )
        return cls.outputs(value=str(result.sum().item()))


class CacheConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cache-consumer.consumer",
            inputs=(InputSpec("value", _STRING),),
            outputs=(OutputSpec("value", _STRING),),
        )

    @classmethod
    def execute(cls, *, value: str) -> dict[str, object]:
        return cls.outputs(value=value)


def test_cache_callback_executes_twice_in_both_ordinary_hosts(tmp_path: Any) -> None:
    async def scenario(in_process: bool) -> None:
        composer = ServingComposer(
            worker_env={"PYTHONPATH": str(Path(__file__).parent), "DINKSTER_ACCELERATOR": "cpu"}
        )
        try:
            module = f"cache_fixture_{in_process}"
            (tmp_path / f"{module}.py").write_text(
                "from test_cache_torch import CacheConsumer, CacheSource\n"
                "SOURCE_NODES = (CacheSource,)\nCONSUMER_NODES = (CacheConsumer,)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "source.toml"
            consumer_manifest = tmp_path / "consumer.toml"
            for manifest, nodes, namespace in (
                (source_manifest, "SOURCE_NODES", "cache-source"),
                (consumer_manifest, "CONSUMER_NODES", "cache-consumer"),
            ):
                manifest.write_text(
                    f'[pack]\nname = "{namespace}"\npublisher = "dinkster"\n'
                    f'namespaces = ["{namespace}"]\n[pack.entry]\nnodes = "{module}:{nodes}"\n',
                    encoding="ascii",
                )
            await composer.add_pack(PackSpec(source_manifest, in_process=in_process))
            await composer.add_pack(PackSpec(consumer_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("cache-source.source", {}),
                    "consumer": GraphNode(
                        "cache-consumer.consumer", {"value": Link("source", "value")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["value"].resolve() == "128.0"
        finally:
            await composer.close()

    from pathlib import Path

    for in_process in (True, False):
        asyncio.run(scenario(in_process))
