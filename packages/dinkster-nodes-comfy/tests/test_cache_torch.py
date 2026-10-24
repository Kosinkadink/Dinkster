"""CPU behavior for native EasyCache and LazyCache holders."""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
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

    source = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1).expand(2, 2, 4, 4).clone()
    assert torch.equal(easycache_forward_wrapper(executor, source.clone(), options), source + 2)
    warmup = source + 0.5
    assert torch.equal(easycache_forward_wrapper(executor, warmup.clone(), options), warmup + 2)
    reuse_input = warmup + 0.05
    reused = easycache_forward_wrapper(executor, reuse_input.clone(), options)
    assert torch.equal(reused, reuse_input + 2)
    assert calls == 2
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

    source = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    assert torch.equal(
        lazycache_predict_noise_wrapper(executor, source.clone(), torch.tensor([0.5]), options),
        source + 4,
    )
    warmup = source + 0.5
    assert torch.equal(
        lazycache_predict_noise_wrapper(executor, warmup.clone(), torch.tensor([0.5]), options),
        warmup + 4,
    )
    reuse_input = warmup + 0.05
    assert torch.equal(
        lazycache_predict_noise_wrapper(
            executor, reuse_input.clone(), torch.tensor([0.5]), options
        ),
        reuse_input + 4,
    )
    assert calls == 2
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


_MODEL = TypeExpr.concrete(MODEL_TYPE)


class CacheModel:
    class Inner:
        class LatentFormat:
            latent_channels = 2

        latent_format = LatentFormat()

    model = Inner()

    def __init__(self, model_options: dict[str, Any] | None = None) -> None:
        self.model_options = model_options or {"transformer_options": {}}
        self.wrappers: list[tuple[str, str, object]] = []

    def clone(self) -> CacheModel:
        cloned = CacheModel(
            {"transformer_options": self.model_options["transformer_options"].copy()}
        )
        cloned.wrappers = self.wrappers.copy()
        return cloned

    def add_wrapper_with_key(self, point: str, key: str, wrapper: object) -> None:
        self.wrappers.append((point, key, wrapper))


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cache-fixture.source",
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(MODEL=CacheModel())


class ModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cache-fixture.consumer",
            inputs=(InputSpec("model", _MODEL), InputSpec("source", _MODEL)),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: CacheModel, source: CacheModel) -> dict[str, object]:
        assert source.wrappers == []
        assert [(point, key) for point, key, _ in model.wrappers] == [
            ("outer_sample", "easycache"),
            ("calc_cond_batch", "easycache"),
            ("diffusion_model", "easycache"),
            ("outer_sample", "lazycache"),
            ("predict_noise", "lazycache"),
        ]
        return cls.outputs(value=float(len(model.wrappers)))


def test_cache_nodes_execute_twice_in_both_ordinary_hosts(tmp_path: Any) -> None:
    async def scenario(in_process: bool) -> None:
        composer = ServingComposer(
            worker_env={"PYTHONPATH": str(Path(__file__).parent), "DINKSTER_ACCELERATOR": "cpu"}
        )
        try:
            module = f"cache_fixture_{in_process}"
            (tmp_path / f"{module}.py").write_text(
                "from test_cache_torch import ModelConsumer, ModelSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ModelSource, ModelConsumer)\n",
                encoding="ascii",
            )
            manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["cache-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("cache-fixture.source", {}),
                    "easy": GraphNode("EasyCache", {"model": Link("source", "MODEL")}),
                    "lazy": GraphNode("LazyCache", {"model": Link("easy", "MODEL")}),
                    "consumer": GraphNode(
                        "cache-fixture.consumer",
                        {"model": Link("lazy", "MODEL"), "source": Link("source", "MODEL")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["value"].resolve() == 5.0
        finally:
            await composer.close()

    from pathlib import Path

    for in_process in (True, False):
        asyncio.run(scenario(in_process))
