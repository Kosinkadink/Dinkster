"""Controlled CPU checkpoint export through real safetensors and mounted output.

Metadata expectations follow ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f,
comfy_extras/nodes_model_merging.py:170-227 and nodes_video_model.py:110-123.
Tensor values and metadata are hand-derived, not fixed-reference captures.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    AssetError,
    MountWidget,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_nodes_comfy.checkpoint_save import ImageOnlyCheckpointSave
from safetensors import safe_open
from safetensors.torch import load_file

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


class _Weights:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values

    def get_sd(self) -> dict[str, torch.Tensor]:
        return self.values.copy()


class _Model:
    def __init__(self, model: Any, sampling: Any) -> None:
        self.model = model
        self.sampling = sampling
        self.weight = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]).T

    def get_model_object(self, name: str) -> Any:
        assert name == "model_sampling"
        return self.sampling

    def state_dict_for_saving(
        self,
        clip: dict[str, torch.Tensor] | None,
        vae: dict[str, torch.Tensor],
        vision: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert clip is None
        return {"diffusion.weight": self.weight, **vae, **vision}


@pytest.fixture
def checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, _Model, _Weights, _Weights]:
    initialize_inference_process()
    base = importlib.import_module("dinkster_inference.model_base")
    management = importlib.import_module("dinkster_inference.model_management")
    args = importlib.import_module("dinkster_inference.cli_args").args
    monkeypatch.setattr(args, "disable_metadata", False)
    monkeypatch.setattr(management, "load_models_gpu", lambda models: None)
    model = base.SVD_img2vid.__new__(base.SVD_img2vid)
    torch.nn.Module.__init__(model)
    model.model_type = base.ModelType.EPS
    wrapped = _Model(model, SimpleNamespace())
    root = tmp_path / "output"
    root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "outputMount": "out",
                "mounts": [{"id": "out", "root": str(root), "mode": "readwrite"}],
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return (
        root,
        snapshot,
        wrapped,
        _Weights({"vision.weight": torch.tensor([7.0, -8.0])}),
        _Weights({"vae.weight": torch.tensor([[9.0, -10.0]])}),
    )


def test_checkpoint_bytes_metadata_and_reserved_counter(
    checkpoint: tuple[Path, Path, _Model, _Weights, _Weights],
) -> None:
    root, _, model, vision, vae = checkpoint
    original = model.weight.clone()
    for index, mount in enumerate((None, "out"), start=1):
        assert (
            ImageOnlyCheckpointSave.execute(
                model=model,
                clip_vision=vision,
                vae=vae,
                filename_prefix="checkpoints/image",
                mount=mount,
                prompt={"node": 2},
                extra_pnginfo={"workflow": {"id": "test"}},
            )
            == {}
        )
        path = root / "checkpoints" / f"image_{index:05}_.safetensors"
        values = load_file(path)
        assert set(values) == {"diffusion.weight", "vision.weight", "vae.weight"}
        assert torch.equal(values["diffusion.weight"], original)
        assert torch.equal(values["vision.weight"], torch.tensor([7.0, -8.0]))
        assert torch.equal(values["vae.weight"], torch.tensor([[9.0, -10.0]]))
        with safe_open(path, framework="pt") as saved:
            assert saved.metadata() == {
                "modelspec.architecture": "stable-video-diffusion-img2vid-v1",
                "modelspec.sai_model_spec": "1.0.0",
                "modelspec.implementation": "sgm",
                "modelspec.title": f"image {index}",
                "modelspec.predict_key": "epsilon",
                "prompt": '{"node": 2}',
                "workflow": '{"id": "test"}',
            }
    assert torch.equal(model.weight, original)
    assert not model.weight.is_contiguous()
    assert not list(root.rglob(".dinkster-save-*"))


def test_edm_prediction_keys_survive_disabled_workflow_metadata(
    checkpoint: tuple[Path, Path, _Model, _Weights, _Weights], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, model, vision, vae = checkpoint
    sampling = importlib.import_module("dinkster_inference.model_sampling")
    base = importlib.import_module("dinkster_inference.model_base")
    args = importlib.import_module("dinkster_inference.cli_args").args
    edm = type("EDMV", (sampling.ModelSamplingContinuousEDM, sampling.V_PREDICTION), {})()
    edm.set_parameters(0.125, 8.0, 1.0)
    edm.zsnr = True
    model.sampling = edm
    model.model = SimpleNamespace(model_type=base.ModelType.V_PREDICTION)
    monkeypatch.setattr(args, "disable_metadata", True)
    ImageOnlyCheckpointSave.execute(
        model=model,
        clip_vision=vision,
        vae=vae,
        filename_prefix="edm",
        prompt={"not": "saved"},
        extra_pnginfo={"workflow": "not saved"},
    )
    path = root / "edm_00001_.safetensors"
    values = load_file(path)
    assert values["edm_vpred.sigma_max"].item() == 8.0
    assert values["edm_vpred.sigma_min"].item() == 0.125
    assert values["v_pred"].numel() == values["ztsnr"].numel() == 0
    with safe_open(path, framework="pt") as saved:
        assert saved.metadata() == {"modelspec.predict_key": "v"}


@pytest.mark.parametrize("failure", ("traversal", "revoked", "readonly", "missing_snapshot"))
def test_checkpoint_requires_current_writable_mount(
    checkpoint: tuple[Path, Path, _Model, _Weights, _Weights],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root, snapshot, model, vision, vae = checkpoint
    prefix = "../escape" if failure == "traversal" else "safe"
    if failure == "revoked":
        snapshot.write_text(json.dumps({"mounts": []}))
    elif failure == "readonly":
        data = json.loads(snapshot.read_text())
        data["mounts"][0]["mode"] = "read"
        snapshot.write_text(json.dumps(data))
    elif failure == "missing_snapshot":
        monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT")
    with pytest.raises((AssetError, ValueError)):
        ImageOnlyCheckpointSave.execute(
            model=model, clip_vision=vision, vae=vae, filename_prefix=prefix, mount="out"
        )
    assert not list(root.rglob("*.safetensors"))


def test_checkpoint_schema_and_cold_discovery() -> None:
    schema = ImageOnlyCheckpointSave.schema()
    assert [input.id for input in schema.inputs] == [
        "model",
        "clip_vision",
        "vae",
        "filename_prefix",
        "mount",
    ]
    assert schema.inputs[3].default == "checkpoints/ComfyUI"
    assert isinstance(schema.inputs[4].widget, MountWidget)
    assert schema.inputs[4].adjacent_to == "filename_prefix"
    assert schema.output_node and not schema.idempotent and not schema.outputs
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster_nodes_comfy.checkpoint_save; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stderr


class _CheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.diffusion_model.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
        self.model_sampling = SimpleNamespace()
        self.model_type = importlib.import_module("dinkster_inference.model_base").ModelType.EPS

    def state_dict_for_saving(
        self,
        unet: dict[str, torch.Tensor],
        *,
        clip_state_dict: object,
        vae_state_dict: dict[str, torch.Tensor],
        clip_vision_state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert clip_state_dict is None
        return {**unet, **vae_state_dict, **clip_vision_state_dict}


class _CheckpointWeights:
    def __init__(self, name: str, values: torch.Tensor) -> None:
        self.name = name
        self.values = values

    def get_sd(self) -> dict[str, torch.Tensor]:
        return {self.name: self.values}


class CheckpointSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="checkpoint-fixture.source",
            idempotent=True,
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("model", MODEL_TYPE),
                    ("vision", CLIP_VISION_TYPE),
                    ("vae", VAE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference.model_patcher import ModelPatcher

        model = ModelPatcher(_CheckpointModel(), torch.device("cpu"), torch.device("cpu"))
        return cls.outputs(
            model=model,
            vision=_CheckpointWeights("vision.weight", torch.tensor([7.0, -8.0])),
            vae=_CheckpointWeights("vae.weight", torch.tensor([[9.0, -10.0]])),
        )


HOST_NODES = (CheckpointSource,)


@pytest.mark.parametrize("in_process", [True, False])
def test_checkpoint_save_graph_writes_mounted_safetensors_and_only_caches_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        root = tmp_path / "output"
        root.mkdir()
        snapshot = tmp_path / "mounts.json"
        snapshot.write_text(
            json.dumps(
                {
                    "outputMount": "out",
                    "mounts": [{"id": "out", "root": str(root), "mode": "readwrite"}],
                }
            ),
            encoding="ascii",
        )
        monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
        (tmp_path / "checkpoint_entry.py").write_text(
            "from test_checkpoint_save_torch import HOST_NODES, register_inference_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="checkpoint-fixture"\npublisher="dinkster"\nnamespaces=["checkpoint-fixture"]\n'
            '[pack.entry]\nnodes="checkpoint_entry:HOST_NODES"\ntypes="checkpoint_entry:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        from dinkster_graph import Graph, GraphNode, Link

        from dinkster.compose import PackSpec, ServingComposer

        manifests = (native, fixture)
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "checkpoint-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("checkpoint-fixture.source"),
                    "save": GraphNode(
                        "ImageOnlyCheckpointSave",
                        {
                            "model": Link("source", "model"),
                            "clip_vision": Link("source", "vision"),
                            "vae": Link("source", "vae"),
                            "filename_prefix": "checkpoints/image",
                            "mount": "out",
                        },
                    ),
                }
            )
            from dinkster_assets import digest_file

            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["save"])
                assert set(result.cached) == ({"source"} if attempt else set())
                saved = root / "checkpoints" / f"image_{attempt + 1:05}_.safetensors"
                assert saved.is_relative_to(root)
                assert len(result.artifacts) == 1
                assert result.artifacts[0].digest == digest_file(saved)
                assert result.artifacts[0].node_id == "save"
                assert result.artifacts[0].virtual_path == f"mounts/out/checkpoints/{saved.name}"
                assert torch.equal(
                    load_file(saved)["weight"],
                    torch.tensor([[1.0, -2.0], [3.0, 4.0]]),
                )
                assert torch.equal(load_file(saved)["vision.weight"], torch.tensor([7.0, -8.0]))
                assert torch.equal(load_file(saved)["vae.weight"], torch.tensor([[9.0, -10.0]]))
                with safe_open(saved, framework="pt") as exported:
                    assert exported.metadata() == {"modelspec.predict_key": "epsilon", "prompt": ""}
        finally:
            await composer.close()

    asyncio.run(scenario())
