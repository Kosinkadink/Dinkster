"""CPU behavior for Comfy model, CLIP, and VAE device routing."""
# pyright: standard

from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_STRING,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.device_routing import (
    DEVICE_ROUTING_NODES,
    MultiGPUWorkUnits,
    SelectCLIPDevice,
    SelectModelDevice,
    SelectVAEDevice,
)

from dinkster.compose import PackSpec, ServingComposer

_MODEL = TypeExpr.concrete(MODEL_TYPE)
_FLOAT32 = object()


class Device:
    def __init__(self, kind: str) -> None:
        self.type = kind

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and self.type == other.type


class Model:
    pass


class Patcher:
    def __init__(
        self, load_device: object, offload_device: object, *, dynamic: bool = False
    ) -> None:
        self.model = Model()
        self.load_device = load_device
        self.offload_device = offload_device
        self.dynamic = dynamic
        self.additional: dict[str, list[Patcher]] = {}
        self.compute_dtype: object | None = None
        self.deepclone_calls = 0

    def clone(self, disable_dynamic: bool = False) -> Patcher:
        cloned = copy.copy(self)
        cloned.model = copy.copy(self.model)
        cloned.additional = {key: list(value) for key, value in self.additional.items()}
        cloned.dynamic = self.dynamic and not disable_dynamic
        return cloned

    def deepclone_multigpu(self, new_load_device: object) -> Patcher:
        self.deepclone_calls += 1
        routed = self.clone()
        routed.load_device = new_load_device
        return routed

    def is_dynamic(self) -> bool:
        return self.dynamic

    def model_dtype(self) -> object:
        return _FLOAT32

    def set_model_compute_dtype(self, dtype: object) -> None:
        self.compute_dtype = dtype

    def get_additional_models_with_key(self, key: str) -> list[Patcher]:
        return self.additional.get(key, [])

    def set_additional_models(self, key: str, models: list[Patcher]) -> None:
        self.additional[key] = models

    def match_multigpu_clones(self) -> None:
        return None


class Clip:
    def __init__(self, patcher: Patcher) -> None:
        self.patcher = patcher

    def clone(self) -> Clip:
        return Clip(self.patcher.clone())


class VAE:
    def __init__(self, patcher: Patcher, device: object) -> None:
        self.patcher = patcher
        self.device = device
        self.first_stage_model = patcher.model


class DeviceManagement:
    @staticmethod
    def resolve_gpu_device_option(option: str) -> object | None:
        return {"default": None, "cpu": Device("cpu"), "gpu:1": Device("meta")}.get(option)

    @staticmethod
    def unet_manual_cast(_dtype: object, _device: object) -> None:
        return None

    @staticmethod
    def vae_offload_device() -> object:
        return Device("cpu")


def test_schemas_are_torch_free_and_accept_portable_gpu_values() -> None:
    assert tuple(node.schema().node_type for node in DEVICE_ROUTING_NODES) == (
        "MultiGPU_WorkUnits",
        "SelectModelDevice",
        "SelectCLIPDevice",
        "SelectVAEDevice",
    )
    model, clip, vae = (node.schema() for node in DEVICE_ROUTING_NODES[1:])
    assert tuple(option.value for option in cast(Any, model.inputs[1].widget).options) == (
        "default",
        "cpu",
    )
    assert tuple(option.value for option in cast(Any, clip.inputs[1].widget).options) == (
        "default",
        "cpu",
    )
    assert tuple(option.value for option in cast(Any, vae.inputs[1].widget).options) == ("default",)


def test_device_routing_preserves_sources_restores_defaults_and_syncs_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_nodes_comfy.device_routing as routing

    monkeypatch.setattr(routing, "_model_management", DeviceManagement)
    gpu0 = Device("meta")
    cpu = Device("cpu")
    model = Patcher(gpu0, cpu, dynamic=True)
    collision = Patcher(gpu0, cpu)
    other = Patcher(cpu, cpu)
    model.additional["multigpu"] = [collision, other]

    cpu_model = cast(Patcher, SelectModelDevice.execute(model=model, device="cpu")["MODEL"])
    assert model.load_device is gpu0 and model.offload_device is cpu
    assert cpu_model.load_device == cpu and cpu_model.offload_device == cpu
    assert not cpu_model.is_dynamic()
    assert cpu_model.additional["multigpu"] == [collision]

    gpu_model = cast(Patcher, SelectModelDevice.execute(model=model, device="gpu:1")["MODEL"])
    assert gpu_model.load_device == gpu0
    assert gpu_model.offload_device == cpu
    assert gpu_model.additional["multigpu"] == [other]
    restored = cast(Patcher, SelectModelDevice.execute(model=gpu_model, device="default")["MODEL"])
    assert restored.load_device == gpu0 and restored.offload_device == cpu

    clip = Clip(Patcher(gpu0, cpu))
    routed_clip = cast(Clip, SelectCLIPDevice.execute(clip=clip, device="cpu")["CLIP"])
    assert clip.patcher.load_device is gpu0
    assert routed_clip.patcher.load_device == cpu and routed_clip.patcher.offload_device == cpu

    vae = VAE(Patcher(gpu0, cpu), gpu0)
    routed_vae = cast(VAE, SelectVAEDevice.execute(vae=vae, device="gpu:1")["VAE"])
    assert vae.patcher.load_device is gpu0
    assert routed_vae.device == gpu0
    assert routed_vae.first_stage_model is routed_vae.patcher.model
    cpu_vae = cast(VAE, SelectVAEDevice.execute(vae=vae, device="cpu")["VAE"])
    assert cpu_vae.patcher.load_device is gpu0


def test_unavailable_devices_and_multigpu_work_units_use_runtime_seams(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import dinkster_nodes_comfy.device_routing as routing

    class MultiGPU:
        def create_multigpu_deepclones(
            self, model: object, max_gpus: int, *, reuse_loaded: bool
        ) -> object:
            assert reuse_loaded and max_gpus == 3
            return (model, max_gpus)

    monkeypatch.setattr(routing, "_model_management", DeviceManagement)
    monkeypatch.setattr(routing, "_multigpu", lambda: MultiGPU())
    caplog.set_level("INFO")
    model = Patcher(Device("meta"), Device("cpu"))
    assert MultiGPUWorkUnits.execute(model=model, max_gpus=3)["MODEL"] == (model, 3)
    unchanged = cast(Patcher, SelectModelDevice.execute(model=model, device="gpu:99")["MODEL"])
    assert unchanged.load_device is model.load_device
    assert "not available" in caplog.text


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="device-routing-fixture.source",
            outputs=(
                OutputSpec("model", _MODEL),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        import torch
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP, VAE

        def patcher() -> Any:
            model = torch.nn.Linear(2, 1)
            model.weight = torch.nn.Parameter(torch.tensor([[2.0, -3.0]]))
            model.bias = torch.nn.Parameter(torch.tensor([0.25]))
            return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))

        clip = CLIP(no_init=True)
        clip.patcher = patcher()
        clip.cond_stage_model = clip.patcher.model
        clip.tokenizer = None
        clip.layer_idx = None
        clip.tokenizer_options = {}
        clip.use_clip_schedule = False
        clip.apply_hooks_to_conds = None
        vae = VAE.__new__(VAE)
        vae.patcher = patcher()
        vae.device = torch.device("cpu")
        vae.first_stage_model = vae.patcher.model
        return dict(cls.outputs(model=patcher(), clip=clip, vae=vae))


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="device-routing-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("model", MODEL_TYPE),
                    ("original_model", MODEL_TYPE),
                    ("clip", CLIP_TYPE),
                    ("original_clip", CLIP_TYPE),
                    ("vae", VAE_TYPE),
                    ("original_vae", VAE_TYPE),
                )
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        original_model: Any,
        clip: Any,
        original_clip: Any,
        vae: Any,
        original_vae: Any,
    ) -> dict[str, object]:
        import torch
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP, VAE

        assert type(model) is ModelPatcher
        assert type(clip) is CLIP
        assert type(vae) is VAE
        assert model is not original_model
        assert clip is not original_clip and clip.patcher is not original_clip.patcher
        assert vae is not original_vae and vae.patcher is not original_vae.patcher
        assert vae.first_stage_model is vae.patcher.model
        assert vae.device.type == original_vae.device.type == "cpu"
        for routed, source in (
            (model, original_model),
            (clip.patcher, original_clip.patcher),
            (vae.patcher, original_vae.patcher),
        ):
            assert routed.load_device.type == routed.offload_device.type == "cpu"
            assert source.load_device.type == source.offload_device.type == "cpu"
            assert torch.equal(source.model.weight, torch.tensor([[2.0, -3.0]]))
            assert routed.model(torch.tensor([[1.5, -2.0]])).tolist() == [[9.25]]
        assert model.get_additional_models_with_key("multigpu") == []
        return dict(cls.outputs(result="routed"))


@pytest.mark.parametrize("in_process", (True, False))
def test_all_device_nodes_run_twice_through_both_production_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = "device_routing_fixture_" + str(int(in_process))
        (tmp_path / f"{module}.py").write_text(
            "from test_device_routing_torch import Consumer, Source\n"
            "NODES = (Source, Consumer)\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "\ndef register_types(registry):\n"
            "    register_inference_types(registry)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            "[pack]\n"
            'name = "device-routing-fixture"\n'
            'publisher = "dinkster"\n'
            'namespaces = ["device-routing-fixture"]\n\n'
            "[pack.entry]\n"
            f'nodes = "{module}:NODES"\n'
            f'types = "{module}:register_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifests = (manifest, fixture)
            worker_group = "device-routing" if not in_process else None
            await composer.add_pack(
                PackSpec(
                    manifest=manifest,
                    in_process=in_process,
                    worker_group=worker_group,
                    group_manifests=manifests if worker_group else (),
                )
            )
            await composer.add_pack(
                PackSpec(
                    manifest=fixture,
                    in_process=in_process,
                    worker_group=worker_group,
                    group_manifests=manifests if worker_group else (),
                )
            )
            graph = Graph(
                nodes={
                    "source": GraphNode("device-routing-fixture.source"),
                    "split": GraphNode(
                        "MultiGPU_WorkUnits", {"model": Link("source", "model"), "max_gpus": 3}
                    ),
                    "route": GraphNode(
                        "SelectModelDevice", {"model": Link("split", "MODEL"), "device": "cpu"}
                    ),
                    "clip": GraphNode(
                        "SelectCLIPDevice", {"clip": Link("source", "clip"), "device": "cpu"}
                    ),
                    "vae": GraphNode(
                        "SelectVAEDevice", {"vae": Link("source", "vae"), "device": "default"}
                    ),
                    "consumer": GraphNode(
                        "device-routing-fixture.consumer",
                        {
                            "model": Link("route", "MODEL"),
                            "original_model": Link("source", "model"),
                            "clip": Link("clip", "CLIP"),
                            "original_clip": Link("source", "clip"),
                            "vae": Link("vae", "VAE"),
                            "original_vae": Link("source", "vae"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["consumer"])
            assert first.outputs["consumer"]["result"].resolve() == "routed"
            second = await engine.run(graph, ["consumer"])
            assert second.outputs["consumer"]["result"].resolve() == "routed"
            assert set(second.cached) == set(graph.nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())
