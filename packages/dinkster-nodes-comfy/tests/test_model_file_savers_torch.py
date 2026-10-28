"""Encoder export writes actual safetensors through mounted output authority."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import MountDef, MountTable
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.model_file_savers import CheckpointSave, CLIPSave, ModelSave, VAESave
from safetensors import safe_open

from dinkster.compose import PackSpec, ServingComposer


class _Clip:
    def load_model(self) -> None:
        pass

    def state_dict_for_saving(self) -> dict[str, torch.Tensor]:
        return {
            "clip_l.transformer.left": torch.tensor([1.25, -3.5]),
            "clip_g.right": torch.tensor([[8, -2]], dtype=torch.int32),
            "transformer.other": torch.tensor([9.0], dtype=torch.float64),
        }


class _Vae:
    def get_sd(self) -> dict[str, torch.Tensor]:
        return {"decoder.weight": torch.tensor([[1.25, -3.5]], dtype=torch.float64)}


def test_encoder_exports_partition_clip_keys_and_preserve_vae_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "output", "root": str(output), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                ],
                "outputMount": "output",
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    assert CLIPSave.execute(clip=_Clip(), mount="alternate") == {}
    assert VAESave.execute(vae=_Vae()) == {}
    expected = (
        (alternate / "clip/ComfyUI_clip_l_00001.safetensors", "left", torch.tensor([1.25, -3.5])),
        (
            alternate / "clip/ComfyUI_clip_g_00001.safetensors",
            "right",
            torch.tensor([[8, -2]], dtype=torch.int32),
        ),
        (
            alternate / "clip/ComfyUI_00001.safetensors",
            "other",
            torch.tensor([9.0], dtype=torch.float64),
        ),
        (
            output / "vae/ComfyUI_vae_00001.safetensors",
            "decoder.weight",
            torch.tensor([[1.25, -3.5]], dtype=torch.float64),
        ),
    )
    for path, key, tensor in expected:
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert list(handle.keys()) == [key]
            assert handle.metadata() == (
                {"prompt": ""} if key == "decoder.weight" else {"format": "pt", "prompt": ""}
            )
            saved = handle.get_tensor(key)
            assert saved.dtype == tensor.dtype
            torch.testing.assert_close(saved, tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("model_class", "architecture", "checkpoint", "vpred", "disable_metadata"),
    (
        ("SDXL", "stable-diffusion-xl-v1-base", True, False, False),
        ("SDXL_instructpix2pix", "stable-diffusion-xl-v1-edit", False, False, False),
        ("SDXLRefiner", "stable-diffusion-xl-v1-refiner", False, False, False),
        ("SVD_img2vid", "stable-video-diffusion-img2vid-v1", True, False, False),
        ("SD3", "stable-diffusion-v3-medium", False, False, True),
        (None, None, False, False, False),
        (None, None, True, True, False),
    ),
)
def test_checkpoint_exports_preserve_metadata_counters_and_tensors(
    tmp_path: Path, monkeypatch, model_class, architecture, checkpoint, vpred, disable_metadata
) -> None:
    initialize_inference_process()
    from dinkster_inference import cli_args, model_base, model_management, model_sampling

    monkeypatch.setattr(cli_args.args, "disable_metadata", disable_metadata)
    monkeypatch.setattr(model_management, "load_models_gpu", lambda _models: None)
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "output", "root": str(output), "mode": "readwrite"},
                    {"id": "alternate", "root": str(alternate), "mode": "readwrite"},
                ],
                "outputMount": "output",
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    body = object.__new__(getattr(model_base, model_class)) if model_class else SimpleNamespace()
    if model_class:
        torch.nn.Module.__init__(body)
    body.model_type = model_base.ModelType.V_PREDICTION if vpred else model_base.ModelType.EPS
    if vpred:

        class Sampling(model_sampling.ModelSamplingContinuousEDM, model_sampling.V_PREDICTION):
            pass

        sampling = Sampling()
        sampling.zsnr = True
        sampling.sigmas = torch.tensor([0.125, 37.5])
    else:
        sampling = SimpleNamespace()
    weight = torch.tensor([[1.25, -3.5, 9.75], [-0.5, 7.0, 2.0]], dtype=torch.float64).T

    def state_dict(clip_sd, vae_sd, vision_sd):
        assert vision_sd is None
        state = {"model.weight": weight}
        if clip_sd is not None:
            state.update(clip_sd)
        if vae_sd is not None:
            state.update(vae_sd)
        return state

    model = SimpleNamespace(
        model=body, get_model_object=lambda _name: sampling, state_dict_for_saving=state_dict
    )
    prefix = "checkpoints/model" if checkpoint else "diffusion_models/model"
    root = alternate if checkpoint else output
    folder = root / prefix.split("/")[0]
    folder.mkdir()
    existing = folder / "model_00004_.safetensors"
    existing.write_bytes(b"existing")
    node = CheckpointSave if checkpoint else ModelSave
    options = {"clip": _Clip(), "vae": _Vae(), "mount": "alternate"} if checkpoint else {}
    expected = {"model.weight": weight}
    if checkpoint:
        expected.update(_Clip().state_dict_for_saving())
        expected.update(_Vae().get_sd())
    if vpred:
        expected.update(
            {
                "edm_vpred.sigma_min": torch.tensor(0.125),
                "edm_vpred.sigma_max": torch.tensor(37.5),
                "v_pred": torch.tensor([]),
                "ztsnr": torch.tensor([]),
            }
        )
    for counter in (5, 6):
        assert node.execute(model=model, filename_prefix=prefix, **options) == {}
        with safe_open(folder / f"model_{counter:05d}_.safetensors", framework="pt") as saved:
            metadata = {"modelspec.predict_key": "v" if vpred else "epsilon"}
            if architecture:
                metadata.update(
                    {
                        "modelspec.architecture": architecture,
                        "modelspec.sai_model_spec": "1.0.0",
                        "modelspec.implementation": "sgm",
                        "modelspec.title": f"model {counter}",
                    }
                )
            if not disable_metadata:
                metadata["prompt"] = ""
            assert saved.metadata() == metadata
            assert set(saved.keys()) == set(expected)
            for key, value in expected.items():
                actual = saved.get_tensor(key)
                assert actual.dtype == value.dtype
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
    assert existing.read_bytes() == b"existing"
    assert len(list(folder.glob("*.safetensors"))) == 3
    assert not list((output if checkpoint else alternate).rglob("*.safetensors"))
    assert not list(folder.glob(".dinkster-save-*"))


class CheckpointSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="checkpoint-save-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import model_base
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP, VAE

        class TinyModel(model_base.SDXL):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.diffusion_model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
                self.diffusion_model.weight.data.copy_(
                    torch.tensor([[1.25, -3.5, 9.75], [-0.5, 7.0, 2.0]])
                )
                self.model_type = model_base.ModelType.EPS
                self.model_sampling = SimpleNamespace()

            def get_dtype(self):
                return torch.float64

            def state_dict_for_saving(
                self,
                unet_state_dict,
                clip_state_dict=None,
                vae_state_dict=None,
                clip_vision_state_dict=None,
            ):
                assert clip_vision_state_dict is None
                state = {f"model.{key}": value for key, value in unet_state_dict.items()}
                state.update(
                    {f"clip.{key}": value for key, value in (clip_state_dict or {}).items()}
                )
                state.update({f"vae.{key}": value for key, value in (vae_state_dict or {}).items()})
                return state

        device = torch.device("cpu")
        encoder = torch.nn.Linear(2, 1, bias=False)
        encoder.weight.data.copy_(torch.tensor([[-7.0, 1.5]]))
        clip = CLIP(no_init=True)
        clip.patcher = ModelPatcher(encoder, device, device)
        clip.cond_stage_model = encoder
        clip.tokenizer = SimpleNamespace(state_dict=lambda: {})
        clip.layer_idx = None
        clip.tokenizer_options = {}
        clip.use_clip_schedule = True
        clip.apply_hooks_to_conds = None
        vae = object.__new__(VAE)
        vae.first_stage_model = torch.nn.Linear(1, 2, bias=False, dtype=torch.float64)
        vae.first_stage_model.weight.data.copy_(torch.tensor([[0.125], [-2.5]]))
        return cls.outputs(model=ModelPatcher(TinyModel(), device, device), clip=clip, vae=vae)


def test_checkpoint_exports_write_exact_files_through_native_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    for name, root in (("output", output), ("alternate", alternate)):
        table.add(MountDef(name, root, mode="readwrite"))
        table.scan(name)
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    async def scenario():
        package = Path(__file__).resolve().parents[1]
        module = f"checkpoint_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_model_file_savers_torch import CheckpointSource\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (CheckpointSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="checkpoint-save-fixture"\npublisher="dinkster"\n'
            'namespaces=["checkpoint-save-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer()
        try:
            for manifest in (package / "dinkster-pack.toml", fixture):
                await composer.add_pack(PackSpec(manifest=manifest, in_process=True))
            graph = Graph(
                nodes={
                    "source": GraphNode("checkpoint-save-fixture.source"),
                    "model": GraphNode("ModelSave", {"model": Link("source", "model")}),
                    "checkpoint": GraphNode(
                        "CheckpointSave",
                        {
                            "model": Link("source", "model"),
                            "clip": Link("source", "clip"),
                            "vae": Link("source", "vae"),
                            "mount": "alternate",
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["model", "checkpoint"])
                if attempt:
                    assert "source" in result.cached
                    assert not {"model", "checkpoint"} & set(result.cached)
        finally:
            await composer.close()

    asyncio.run(scenario())
    for checkpoint, root, prefix in (
        (False, output, "diffusion_models/ComfyUI"),
        (True, alternate, "checkpoints/ComfyUI"),
    ):
        expected = {
            "model.weight": torch.tensor(
                [[1.25, -3.5, 9.75], [-0.5, 7.0, 2.0]],
                dtype=torch.float64,
            )
        }
        if checkpoint:
            expected.update(
                {
                    "clip.weight": torch.tensor([[-7.0, 1.5]]),
                    "vae.weight": torch.tensor([[0.125], [-2.5]], dtype=torch.float64),
                }
            )
        files = sorted(root.glob(f"{prefix}_*.safetensors"))
        assert len(files) == 2
        for counter, path in enumerate(files, 1):
            assert path.name == f"ComfyUI_{counter:05d}_.safetensors"
            with safe_open(path, framework="pt") as saved:
                assert saved.metadata() == {
                    "modelspec.architecture": "stable-diffusion-xl-v1-base",
                    "modelspec.sai_model_spec": "1.0.0",
                    "modelspec.implementation": "sgm",
                    "modelspec.title": f"ComfyUI {counter}",
                    "modelspec.predict_key": "epsilon",
                    "prompt": "",
                }
                assert set(saved.keys()) == set(expected)
                for key, tensor in expected.items():
                    actual = saved.get_tensor(key)
                    assert actual.dtype == tensor.dtype
                    torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
    assert not list(output.glob("checkpoints/*"))
    assert not list(alternate.glob("diffusion_models/*"))
