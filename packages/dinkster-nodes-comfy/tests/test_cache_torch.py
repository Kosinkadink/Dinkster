"""CPU behavior for native EasyCache and LazyCache holders."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
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
    FluxKVCache,
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
        model_sampling = type(
            "Sampling", (), {"percent_to_sigma": staticmethod(lambda value: value)}
        )()

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
        callbacks = {point: wrapper for point, _key, wrapper in model.wrappers}
        easy_options = model.model_options["transformer_options"].copy()
        easy_options["easycache"] = EasyCacheHolder(0.2, 0.0, 1.0, 2, False, False, 2)
        easy_options["easycache"].start_t = torch.tensor(1.0)
        easy_options["easycache"].end_t = torch.tensor(-1.0)
        easy_options["sigmas"] = torch.tensor([0.5])
        easy_options["uuids"] = ["positive", "negative"]
        video = torch.arange(64, dtype=torch.float32).reshape(2, 2, 4, 4)
        audio = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
        calls = 0

        def diffusion(value: list[torch.Tensor], _options: dict[str, Any]) -> list[torch.Tensor]:
            nonlocal calls
            calls += 1
            return [value[0] + 2, value[1] + 3]

        first = callbacks["diffusion_model"](
            diffusion, [video.clone(), audio.clone()], easy_options
        )
        warmup = callbacks["diffusion_model"](diffusion, [video + 0.5, audio + 0.5], easy_options)
        reused = callbacks["diffusion_model"](diffusion, [video + 0.55, audio + 0.55], easy_options)
        assert calls == 2
        assert torch.equal(first[0], video + 2)
        assert torch.equal(warmup[1], audio + 3.5)
        assert torch.equal(reused[0], video + 2.55)
        lazy_options = {
            "transformer_options": {"easycache": LazyCacheHolder(0.2, 0.0, 1.0, 2, False, False, 2)}
        }
        lazy_holder = lazy_options["transformer_options"]["easycache"]
        lazy_holder.start_t = torch.tensor(1.0)
        lazy_holder.end_t = torch.tensor(-1.0)
        lazy_calls = 0

        def predict(
            value: torch.Tensor, _timestep: torch.Tensor, _options: dict[str, Any]
        ) -> torch.Tensor:
            nonlocal lazy_calls
            lazy_calls += 1
            return value + 4

        callbacks["predict_noise"](predict, video.clone(), torch.tensor([0.5]), lazy_options)
        callbacks["predict_noise"](predict, video + 0.5, torch.tensor([0.5]), lazy_options)
        assert torch.equal(
            callbacks["predict_noise"](predict, video + 0.55, torch.tensor([0.5]), lazy_options),
            video + 4.55,
        )
        assert lazy_calls == 2
        original_options = model.model_options

        class Guider:
            model_options = original_options
            model_patcher = type("ModelPatcher", (), {"model": model.model})()

        class Outer:
            class_obj = Guider()

            def __call__(self, *_args: object, **_kwargs: object) -> str:
                return "sampled"

        assert callbacks["outer_sample"](Outer(), None, None, None, [1, 0]) == "sampled"
        assert Outer.class_obj.model_options is original_options
        return cls.outputs(value=float(len(model.wrappers)))


def test_cache_nodes_execute_twice_in_both_ordinary_hosts(tmp_path: Any) -> None:
    async def scenario(in_process: bool) -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                ),
                "DINKSTER_ACCELERATOR": "cpu",
            }
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


def test_flux_reference_cache_keeps_blocks_independent_and_repeats_changed_batch() -> None:
    class Model:
        def __init__(self, params: bool = True) -> None:
            self.model = SimpleNamespace(diffusion_model=SimpleNamespace())
            if params:
                self.model.diffusion_model.params = SimpleNamespace()
            self.attention: Any = None
            self.post_input: Any = None
            self.objects: dict[str, object] = {}

        def clone(self) -> Any:
            return Model(hasattr(self.model.diffusion_model, "params"))

        def set_model_attn1_patch(self, callback: Any) -> None:
            self.attention = callback

        def set_model_post_input_patch(self, callback: Any) -> None:
            self.post_input = callback

        def add_object_patch(self, path: str, value: object) -> None:
            self.objects[path] = value

    q = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    k, v = q + 10, q + 30
    image = q.reshape(1, 4, 2)
    for params in (True, False):
        source = Model(params)
        patched = FluxKVCache.execute(model=source)["MODEL"]
        path = "diffusion_model." + ("params." if params else "") + "default_ref_method"
        assert patched.objects == {path: "index_timestep_zero"}
        options = {"reference_image_num_tokens": [1, 1], "block_type": "double", "block_index": 3}
        assert patched.attention(q, k, v, {}) == {}
        assert patched.post_input({"img": image, "transformer_options": options})["img"] is image
        first = patched.attention(q, k, v, options)
        assert torch.equal(first["k"], k) and torch.equal(first["v"], v)
        assert torch.equal(
            patched.post_input({"img": image, "transformer_options": options})["img"],
            image[:, :-2],
        )
        next_q = q[:, :, :1].repeat(3, 1, 1, 1)
        next_k, next_v = next_q + 100, next_q + 200
        second = patched.attention(next_q, next_k, next_v, options)
        assert torch.equal(second["k"], torch.cat((next_k, k[:, :, -2:].repeat(3, 1, 1, 1)), 2))
        assert torch.equal(second["v"], torch.cat((next_v, v[:, :, -2:].repeat(3, 1, 1, 1)), 2))
        other = patched.attention(next_q, next_k, next_v, {**options, "block_index": 4})
        assert torch.equal(other["k"], next_k)
        patched.attention.cleanup()
        reset = patched.attention(next_q, next_k, next_v, options)
        assert torch.equal(reset["k"], next_k)
        assert source.attention is None and source.objects == {}
