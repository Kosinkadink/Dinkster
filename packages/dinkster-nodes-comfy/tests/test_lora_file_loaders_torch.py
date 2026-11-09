"""CPU coverage for real LoRA file-loader tensor application and host wiring."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DINKSTER_ACCELERATOR"] = "cpu"

import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_STRING,
    MODEL_TYPE,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import MountDef, MountTable
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.lora_file_loaders import (
    LoadLoraModel,
    LoadLoraTextEncoder,
    LoraLoader,
    LoraLoaderBypass,
    LoraLoaderBypassModelOnly,
    LoraLoaderModelOnly,
)

from dinkster.compose import PackSpec, ServingComposer


class _RawModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.linear = torch.nn.Linear(2, 2, bias=False)
        self.diffusion_model.linear.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.model_config = type("Config", (), {"unet_config": {}})()


class _RawClip(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = torch.nn.Linear(2, 2, bias=False)
        self.text_encoder.weight.data.copy_(torch.tensor([[2.0, -3.0], [4.0, 5.0]]))


def _model() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(_RawModel(), torch.device("cpu"), torch.device("cpu"))


def _clip() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    raw = _RawClip()
    clip = CLIP(no_init=True)
    clip.patcher = ModelPatcher(raw, torch.device("cpu"), torch.device("cpu"))
    clip.cond_stage_model = raw
    clip.tokenizer = object()
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = True
    clip.apply_hooks_to_conds = None
    return clip


def _asset(tmp_path: Path, *, verified: bool) -> AssetRef | LocalAssetRef:
    from safetensors.torch import save_file

    checkpoint = tmp_path / "tiny.safetensors"
    save_file(
        {
            "lora_unet_linear.lora_down.weight": torch.tensor([[2.0, -1.0]]),
            "lora_unet_linear.lora_up.weight": torch.tensor([[3.0], [5.0]]),
            "text_encoders.text_encoder.lora_down.weight": torch.tensor([[7.0, -2.0]]),
            "text_encoders.text_encoder.lora_up.weight": torch.tensor([[11.0], [13.0]]),
        },
        str(checkpoint),
        metadata={"purpose": "tiny-real-lora", "rank": "1"},
    )
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="loras", path=tmp_path), kind="model/lora")
    if verified:
        table.scan("loras")
    asset = table.bind_name(checkpoint.name, "model/lora")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    return asset


def _assert_normal_patches(model: Any, clip: Any) -> None:
    from dinkster_inference.lora import calculate_weight

    model_weight = calculate_weight(
        model.patches["diffusion_model.linear.weight"],
        model.model.diffusion_model.linear.weight.detach(),
        "diffusion_model.linear.weight",
    )
    clip_weight = calculate_weight(
        clip.patcher.patches["text_encoder.weight"],
        clip.cond_stage_model.text_encoder.weight.detach(),
        "text_encoder.weight",
    )
    torch.testing.assert_close(
        model_weight, torch.tensor([[-2.0, 3.5], [-2.0, 6.5]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        clip_weight, torch.tensor([[98.25, -30.5], [117.75, -27.5]]), rtol=0, atol=0
    )
    assert model.get_attachment("lora_metadata") == {"purpose": "tiny-real-lora", "rank": "1"}
    assert clip.patcher.get_attachment("lora_metadata") == {
        "purpose": "tiny-real-lora",
        "rank": "1",
    }


@pytest.mark.parametrize("verified", (False, True))
def test_real_safetensors_apply_through_all_lora_file_loaders(
    tmp_path: Path, verified: bool
) -> None:
    asset = _asset(tmp_path, verified=verified)
    full = LoraLoader.execute(
        model=_model(), clip=_clip(), lora_name=asset, strength_model=-0.5, strength_clip=1.25
    )
    _assert_normal_patches(full["MODEL"], full["CLIP"])

    model_only = LoraLoaderModelOnly.execute(model=_model(), lora_name=asset, strength_model=-0.5)
    assert set(model_only["MODEL"].patches) == {"diffusion_model.linear.weight"}

    model_stack = LoadLoraModel.execute(
        model=_model(),
        loras={
            "first.lora_name": asset,
            "first.strength": -0.5,
            "first.enabled": True,
            "disabled.lora_name": asset,
            "disabled.strength": 1.25,
            "disabled.enabled": False,
            "zero.lora_name": asset,
            "zero.strength": 0.0,
            "zero.enabled": True,
            "last.lora_name": asset,
            "last.strength": 1.25,
            "last.enabled": True,
        },
    )["MODEL"]
    assert [patch[0] for patch in model_stack.patches["diffusion_model.linear.weight"]] == [
        -0.5,
        1.25,
    ]

    clip_stack = LoadLoraTextEncoder.execute(
        clip=_clip(),
        loras={
            "first.lora_name": asset,
            "first.strength": -0.5,
            "first.enabled": True,
            "disabled.lora_name": asset,
            "disabled.strength": 1.25,
            "disabled.enabled": False,
            "zero.lora_name": asset,
            "zero.strength": 0.0,
            "zero.enabled": True,
            "last.lora_name": asset,
            "last.strength": 1.25,
            "last.enabled": True,
        },
    )["CLIP"]
    assert [patch[0] for patch in clip_stack.patcher.patches["text_encoder.weight"]] == [-0.5, 1.25]

    bypass = LoraLoaderBypass.execute(
        model=_model(), clip=_clip(), lora_name=asset, strength_model=-0.5, strength_clip=1.25
    )
    assert set(bypass["MODEL"].injections) == {"bypass_lora"}
    assert set(bypass["CLIP"].patcher.injections) == {"bypass_lora"}
    bypass["MODEL"].inject_model()
    torch.testing.assert_close(
        bypass["MODEL"].model.diffusion_model.linear(torch.tensor([[1.0, 0.0]])),
        torch.tensor([[-2.0, -2.0]]),
        rtol=0,
        atol=0,
    )
    bypass["MODEL"].eject_model()

    bypass_model_only = LoraLoaderBypassModelOnly.execute(
        model=_model(), lora_name=asset, strength_model=-0.5
    )["MODEL"]
    assert set(bypass_model_only.injections) == {"bypass_lora"}


class _HostSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-file-loaders-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=_model(), clip=_clip())


class _HostInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-file-loaders-fixture.inspect",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any, clip: Any) -> dict[str, object]:
        assert [patch[0] for patch in model.patches["diffusion_model.linear.weight"]] == [
            -0.5,
            1.25,
        ]
        assert [patch[0] for patch in clip.patcher.patches["text_encoder.weight"]] == [-0.5, 1.25]
        return cls.outputs(summary="ordered lora rows applied")


_HOST_NODES = (_HostSource, _HostInspect)


@pytest.mark.parametrize("in_process", (True,))
@pytest.mark.parametrize("verified", (False, True))
def test_lora_file_loaders_execute_twice_through_both_ordinary_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, verified: bool
) -> None:
    _asset(tmp_path, verified=verified)
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    table.add(MountDef(id="loras", path=tmp_path), kind="model/lora")
    if verified:
        table.scan("loras")
    asset = table.bind_name("tiny.safetensors", "model/lora")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        native_manifest = package / "dinkster-pack.toml"
        source_paths = (
            checkout / "src",
            *(sorted(checkout.glob("packages/*/src"))),
        )
        runtime_paths = tuple(Path(path) for path in sys.path if path and Path(path).is_dir())
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(package / "src"),
                        *(str(source) for source in source_paths),
                        *(str(path) for path in runtime_paths),
                    )
                ),
            }
        )
        try:
            module = f"lora_file_loaders_fixture_{tmp_path.name.replace('-', '_')}"
            (tmp_path / f"{module}.py").write_text(
                "from test_lora_file_loaders_torch import _HOST_NODES\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = _HOST_NODES\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "lora-file-loaders-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["lora-file-loaders-fixture"]\n\n[pack.entry]\n'
                f'nodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            manifests = (native_manifest, fixture_manifest)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "lora-file-loaders-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("lora-file-loaders-fixture.source"),
                    "model": GraphNode(
                        "LoadLoraModel",
                        {
                            "model": Link("source", "model"),
                            "loras.first.lora_name": asset.to_wire(),
                            "loras.first.strength": -0.5,
                            "loras.first.enabled": True,
                            "loras.disabled.lora_name": asset.to_wire(),
                            "loras.disabled.strength": 1.25,
                            "loras.disabled.enabled": False,
                            "loras.zero.lora_name": asset.to_wire(),
                            "loras.zero.strength": 0.0,
                            "loras.zero.enabled": True,
                            "loras.last.lora_name": asset.to_wire(),
                            "loras.last.strength": 1.25,
                            "loras.last.enabled": True,
                        },
                    ),
                    "clip": GraphNode(
                        "LoadLoraTextEncoder",
                        {
                            "clip": Link("source", "clip"),
                            "loras.first.lora_name": asset.to_wire(),
                            "loras.first.strength": -0.5,
                            "loras.first.enabled": True,
                            "loras.disabled.lora_name": asset.to_wire(),
                            "loras.disabled.strength": 1.25,
                            "loras.disabled.enabled": False,
                            "loras.zero.lora_name": asset.to_wire(),
                            "loras.zero.strength": 0.0,
                            "loras.zero.enabled": True,
                            "loras.last.lora_name": asset.to_wire(),
                            "loras.last.strength": 1.25,
                            "loras.last.enabled": True,
                        },
                    ),
                    "inspect": GraphNode(
                        "lora-file-loaders-fixture.inspect",
                        {"model": Link("model", "MODEL"), "clip": Link("clip", "CLIP")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "ordered lora rows applied"
        finally:
            await composer.close()

    asyncio.run(scenario())
