"""CPU contracts for native latent upscale consumers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.latent_upscale import (
    HunyuanVideo15LatentUpscaleWithModel,
    LTXVLatentUpsampler,
)

from dinkster.compose import PackSpec, ServingComposer

_LATENT = TypeExpr.concrete(LATENT_TYPE)
_UPSCALE_MODEL = TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


class _Statistics:
    def un_normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value + 3

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value - 1


class _LTXModel(torch.nn.Conv2d):
    def get_dtype(self) -> torch.dtype:
        return self.weight.dtype


def _hunyuan_model():
    initialize_inference_process()
    from dinkster_inference.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel

    model = HunyuanVideo15SRModel(
        "720p", {"in_channels": 1, "out_channels": 1, "hidden_channels": 1, "num_blocks": 0}
    )
    with torch.no_grad():
        model.model.in_conv.conv.weight.zero_()
        model.model.in_conv.conv.weight[0, 0, 2, 1, 1] = 2
        model.model.in_conv.conv.bias.fill_(3)
        model.model.out_conv.conv.weight.zero_()
        model.model.out_conv.conv.weight[0, 0, 2, 1, 1] = -0.5
        model.model.out_conv.conv.bias.fill_(7)
    return model


class LatentUpscaleSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="latent-upscale-fixture.source",
            outputs=(
                OutputSpec("hunyuan_model", _UPSCALE_MODEL),
                OutputSpec("ltx_model", _UPSCALE_MODEL),
                OutputSpec("vae", _VAE),
                OutputSpec("hunyuan_latent", _LATENT),
                OutputSpec("ltx_latent", _LATENT),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference.model_patcher import CoreModelPatcher

        ltx = _LTXModel(1, 1, 1)
        with torch.no_grad():
            ltx.weight.fill_(2)
            ltx.bias.fill_(1)
        return cls.outputs(
            hunyuan_model=_hunyuan_model(),
            ltx_model=CoreModelPatcher(ltx, torch.device("cpu"), torch.device("cpu")),
            vae=SimpleNamespace(
                first_stage_model=SimpleNamespace(per_channel_statistics=_Statistics())
            ),
            hunyuan_latent={
                "samples": torch.arange(30, dtype=torch.float32).reshape(1, 1, 2, 3, 5),
                "metadata": "hunyuan-source",
                "noise_mask": "drop",
            },
            ltx_latent={
                "samples": torch.tensor(
                    [[[[2.0, -3.0, 7.0], [0.5, 4.0, -8.0]]]], dtype=torch.float64
                ),
                "noise_mask": "drop",
                "metadata": "keep",
            },
        )


class LatentUpscaleConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="latent-upscale-fixture.consumer",
            inputs=(
                InputSpec("hunyuan", _LATENT),
                InputSpec("ltx", _LATENT),
                InputSpec("source_hunyuan", _LATENT),
                InputSpec("source_ltx", _LATENT),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, hunyuan: object, ltx: object, source_hunyuan: object, source_ltx: object
    ) -> dict[str, object]:
        hunyuan_value = materialize_latent(hunyuan)
        ltx_value = materialize_latent(ltx)
        original = torch.arange(30, dtype=torch.float32).reshape(1, 1, 2, 3, 5)
        resized = original[..., (0, 1, 1, 2), :][..., (0, 1, 2, 2, 3, 4)]
        torch.testing.assert_close(hunyuan_value["samples"], 5.5 - resized, rtol=0, atol=0)
        assert set(hunyuan_value) == {"samples"}
        assert hunyuan_value["samples"].shape == (1, 1, 2, 4, 6)
        assert hunyuan_value["samples"].dtype is torch.float32
        assert hunyuan_value["samples"].device.type == "cpu"
        assert ltx_value["samples"].dtype is torch.float64
        assert ltx_value["samples"].device.type == "cpu"
        torch.testing.assert_close(
            ltx_value["samples"],
            torch.tensor([[[[10.0, 0.0, 20.0], [7.0, 14.0, -10.0]]]], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        assert ltx_value["metadata"] == "keep" and "noise_mask" not in ltx_value
        hunyuan_source = materialize_latent(source_hunyuan)
        ltx_source = materialize_latent(source_ltx)
        torch.testing.assert_close(hunyuan_source["samples"], original, rtol=0, atol=0)
        torch.testing.assert_close(
            ltx_source["samples"],
            torch.tensor([[[[2.0, -3.0, 7.0], [0.5, 4.0, -8.0]]]], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        assert hunyuan_source["metadata"] == "hunyuan-source"
        assert ltx_source["metadata"] == "keep"
        assert hunyuan_source["noise_mask"] == ltx_source["noise_mask"] == "drop"
        return cls.outputs(summary="latent-upscale")


@pytest.mark.parametrize("in_process", (True, False))
def test_latent_upscale_nodes_execute_twice_in_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        source_root = Path(__file__).resolve().parents[1] / "src"
        fixture_module = "latent_upscale_consumer_fixture"
        (tmp_path / f"{fixture_module}.py").write_text(
            "from test_latent_upscale_torch import LatentUpscaleConsumer, LatentUpscaleSource\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (LatentUpscaleSource, LatentUpscaleConsumer)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            '[pack]\nname = "latent-upscale-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["latent-upscale-fixture"]\n\n[pack.entry]\n'
            f'nodes = "{fixture_module}:NODES"\n'
            f'types = "{fixture_module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(source_root))
                ),
            }
        )
        try:
            if in_process:
                await composer.add_pack(PackSpec(native_manifest, in_process=True))
                await composer.add_pack(PackSpec(fixture_manifest, in_process=True))
            else:
                manifests = (native_manifest, fixture_manifest)
                await composer.add_pack(
                    PackSpec(
                        native_manifest, worker_group="latent-upscale", group_manifests=manifests
                    )
                )
                await composer.add_pack(
                    PackSpec(
                        fixture_manifest, worker_group="latent-upscale", group_manifests=manifests
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("latent-upscale-fixture.source"),
                    "hunyuan": GraphNode(
                        "HunyuanVideo15LatentUpscaleWithModel",
                        {
                            "model": Link("source", "hunyuan_model"),
                            "samples": Link("source", "hunyuan_latent"),
                            "upscale_method": "nearest-exact",
                            "width": 96,
                            "height": 64,
                            "crop": "disabled",
                        },
                    ),
                    "ltx": GraphNode(
                        "LTXVLatentUpsampler",
                        {
                            "samples": Link("source", "ltx_latent"),
                            "upscale_model": Link("source", "ltx_model"),
                            "vae": Link("source", "vae"),
                        },
                    ),
                    "consumer": GraphNode(
                        "latent-upscale-fixture.consumer",
                        {
                            "hunyuan": Link("hunyuan", "LATENT"),
                            "ltx": Link("ltx", "LATENT"),
                            "source_hunyuan": Link("source", "hunyuan_latent"),
                            "source_ltx": Link("source", "ltx_latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "latent-upscale"
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_native_runtime_models_execute_cpu_latent_upscale_contracts() -> None:
    initialize_inference_process()
    from dinkster_inference.model_patcher import CoreModelPatcher

    hunyuan = _hunyuan_model()
    assert type(hunyuan.patcher) is CoreModelPatcher
    source = {
        "samples": torch.arange(1 * 1 * 2 * 3 * 5, dtype=torch.float32).reshape(1, 1, 2, 3, 5)
    }
    result = HunyuanVideo15LatentUpscaleWithModel.execute(
        model=hunyuan,
        samples=source,
        upscale_method="nearest-exact",
        width=96,
        height=64,
        crop="disabled",
    )["LATENT"]
    assert result["samples"].shape == (1, 1, 2, 4, 6)
    assert result["samples"].dtype is torch.float32
    resized = source["samples"][..., (0, 1, 1, 2), :][..., (0, 1, 2, 2, 3, 4)]
    torch.testing.assert_close(result["samples"], 5.5 - resized, rtol=0, atol=0)

    class Statistics:
        def un_normalize(self, value: torch.Tensor) -> torch.Tensor:
            return value + 3

        def normalize(self, value: torch.Tensor) -> torch.Tensor:
            return value - 1

    ltx = CoreModelPatcher(torch.nn.Identity(), torch.device("cpu"), torch.device("cpu"))
    assert type(ltx) is CoreModelPatcher
    input_latent = {"samples": torch.tensor([[[[2.0]]]], dtype=torch.float64), "noise_mask": "drop"}
    vae = SimpleNamespace(first_stage_model=SimpleNamespace(per_channel_statistics=Statistics()))
    upsampled = LTXVLatentUpsampler.execute(samples=input_latent, upscale_model=ltx, vae=vae)[
        "LATENT"
    ]
    assert upsampled["samples"].dtype is torch.float64
    torch.testing.assert_close(upsampled["samples"], torch.tensor([[[[4.0]]]], dtype=torch.float64))
    assert "noise_mask" not in upsampled


def test_registry_and_schemas_are_core_compatible_without_torch_runtime() -> None:
    assert [
        node.schema().node_type
        for node in COMFY_NODES
        if node
        in {
            HunyuanVideo15LatentUpscaleWithModel,
            LTXVLatentUpsampler,
        }
    ] == ["HunyuanVideo15LatentUpscaleWithModel", "LTXVLatentUpsampler"]
    hunyuan = HunyuanVideo15LatentUpscaleWithModel.schema()
    assert [input.id for input in hunyuan.inputs] == [
        "model",
        "samples",
        "upscale_method",
        "width",
        "height",
        "crop",
    ]
    assert hunyuan.inputs[2].default == "bilinear"
    assert [input.id for input in LTXVLatentUpsampler.schema().inputs] == [
        "samples",
        "upscale_model",
        "vae",
    ]


def test_hunyuan_zero_dimensions_returns_the_original_public_latent() -> None:
    source = {"samples": torch.ones((1, 2, 3, 5)), "metadata": "kept"}
    result = HunyuanVideo15LatentUpscaleWithModel.execute(
        model=object(),
        samples=source,
        upscale_method="bilinear",
        width=0,
        height=0,
        crop="disabled",
    )["LATENT"]
    assert result is source


def test_hunyuan_one_zero_dimension_uses_width_height_in_the_core_order(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    class Model:
        def resample_latent(self, latent: torch.Tensor) -> torch.Tensor:
            return latent + 3

    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale.importlib.import_module",
        lambda name: SimpleNamespace(
            common_upscale=lambda tensor, width, height, _method, _crop: (
                calls.append((width, height)) or tensor.new_ones((1, 2, height, width))
            )
        ),
    )
    source = {"samples": torch.zeros((1, 2, 3, 5), dtype=torch.float64), "noise_mask": object()}
    output = HunyuanVideo15LatentUpscaleWithModel.execute(
        model=Model(), samples=source, upscale_method="bilinear", width=0, height=96, crop="center"
    )["LATENT"]
    assert calls == [(10, 6)]
    assert set(output) == {"samples"}
    assert output["samples"].shape == (1, 2, 6, 10)
    assert output["samples"].dtype is torch.float32


def test_ltx_normalizes_around_the_model_and_removes_noise_mask(monkeypatch) -> None:
    calls: list[object] = []

    class Statistics:
        def un_normalize(self, value: torch.Tensor) -> torch.Tensor:
            calls.append("un")
            return value + 10

        def normalize(self, value: torch.Tensor) -> torch.Tensor:
            calls.append("norm")
            return value - 4

    class Model:
        load_device = torch.device("cpu")
        model = staticmethod(lambda value: calls.append("model") or value * 2)

        @staticmethod
        def model_dtype() -> torch.dtype:
            return torch.float64

    management = SimpleNamespace(
        load_models_gpu=lambda models, memory_required: calls.append((models, memory_required)),
        intermediate_device=lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale.importlib.import_module", lambda _name: management
    )
    source = {
        "samples": torch.tensor([[[[2.0]]]], dtype=torch.float32),
        "noise_mask": object(),
        "keep": 7,
    }
    vae = SimpleNamespace(first_stage_model=SimpleNamespace(per_channel_statistics=Statistics()))
    output = LTXVLatentUpsampler.execute(samples=source, upscale_model=Model(), vae=vae)["LATENT"]
    assert calls[0][1] == 3000.0
    assert calls[1:] == ["un", "model", "norm"]
    assert output["samples"].dtype is torch.float32
    assert torch.equal(output["samples"], torch.tensor([[[[20.0]]]]))
    assert output["keep"] == 7 and "noise_mask" not in output
