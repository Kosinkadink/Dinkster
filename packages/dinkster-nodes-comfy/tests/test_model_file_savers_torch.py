"""Encoder export writes actual safetensors through mounted output authority."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    ExportSnapshot,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.model_file_savers import CheckpointSave, CLIPSave, ModelSave, VAESave
from dinkster_schema.reporting import capture_saved_artifacts
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
    with capture_saved_artifacts():
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
        with capture_saved_artifacts():
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


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("disabled", (False, True))
@pytest.mark.parametrize("prompt", ({"source": {"inputs": {"seed": 71}}}, {}, None))
def test_model_and_encoder_exports_consume_submission_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_process: bool,
    disabled: bool,
    prompt: dict | None,
) -> None:
    initialize_inference_process()
    from dinkster_inference.cli_args import args

    monkeypatch.setattr(args, "disable_metadata", disabled)
    output = tmp_path / "output"
    output.mkdir()
    mounts = tmp_path / "mounts.json"
    table = MountTable(mounts, index_root=tmp_path / "index")
    table.add(MountDef("output", output, mode="readwrite"))
    table.scan("output")
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(mounts))
    extra = {
        "workflow": {"nodes": [{"id": 41, "title": "model: caf\u00e9"}]},
        "seed": 149,
        "format": "submitted format",
        "modelspec.title": "submitted title",
    }

    async def scenario():
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"model_metadata_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_model_file_savers_torch import CheckpointSource\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "from dinkster_api.v1 import initialize_inference_process\n"
            "initialize_inference_process()\n"
            "from dinkster_inference.cli_args import args\n"
            f"args.disable_metadata = {disabled!r}\n"
            "NODES = (CheckpointSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="model-metadata-fixture"\npublisher="dinkster"\n'
            'namespaces=["checkpoint-save-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (str(tmp_path), str(Path(__file__).parent), *(p for p in sys.path if p))
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(mounts),
                "PYTHONPATH": os.pathsep.join(paths),
            }
        )
        try:
            manifests = (native, fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "model-metadata-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            nodes = {"source": GraphNode("checkpoint-save-fixture.source")}
            for kind, node_type, inputs in (
                ("model", "ModelSave", ("model",)),
                ("checkpoint", "CheckpointSave", ("model", "clip", "vae")),
                ("clip", "CLIPSave", ("clip",)),
                ("vae", "VAESave", ("vae",)),
            ):
                nodes[kind] = GraphNode(
                    node_type,
                    {"filename_prefix": kind, **{key: Link("source", key) for key in inputs}},
                )
            engine = composer.composition.make_engine(lambda _event: None)
            await engine.run(
                Graph(nodes=nodes),
                ["model", "checkpoint", "clip", "vae"],
                export_snapshot=ExportSnapshot(prompt=prompt, extra_pnginfo=extra),
            )
        finally:
            await composer.close()

    asyncio.run(scenario())
    for kind in ("model", "checkpoint", "clip", "vae"):
        paths = list(output.glob(f"{kind}_*.safetensors"))
        assert len(paths) == 1
        expected = {}
        if kind in ("model", "checkpoint"):
            expected.update(
                {
                    "modelspec.architecture": "stable-diffusion-xl-v1-base",
                    "modelspec.sai_model_spec": "1.0.0",
                    "modelspec.implementation": "sgm",
                    "modelspec.title": f"{kind} 1",
                    "modelspec.predict_key": "epsilon",
                }
            )
        if not disabled:
            if kind == "clip":
                expected["format"] = "pt"
            expected["prompt"] = json.dumps(prompt) if prompt is not None else ""
            expected.update({key: json.dumps(value) for key, value in extra.items()})
        tensors = {
            "model.weight": torch.tensor(
                [[1.25, -3.5, 9.75], [-0.5, 7.0, 2.0]], dtype=torch.float64
            )
        }
        if kind == "checkpoint":
            tensors.update(
                {
                    "clip.weight": torch.tensor([[-7.0, 1.5]]),
                    "vae.weight": torch.tensor([[0.125], [-2.5]], dtype=torch.float64),
                }
            )
        elif kind in ("clip", "vae"):
            tensors = {
                "weight": torch.tensor([[-7.0, 1.5]])
                if kind == "clip"
                else torch.tensor([[0.125], [-2.5]], dtype=torch.float64)
            }
        with safe_open(paths[0], framework="pt", device="cpu") as handle:
            assert handle.metadata() == expected
            assert set(handle.keys()) == set(tensors)
            for key, tensor in tensors.items():
                actual = handle.get_tensor(key)
                assert actual.dtype == tensor.dtype
                torch.testing.assert_close(actual, tensor, rtol=0, atol=0)


@pytest.mark.parametrize("in_process", (True, False))
def test_model_exports_attach_exact_mounted_files_on_each_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
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

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"model_attachment_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_model_file_savers_torch import CheckpointSource\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (CheckpointSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="model-attachment-fixture"\npublisher="dinkster"\n'
            'namespaces=["checkpoint-save-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (str(tmp_path), str(Path(__file__).parent), *(p for p in sys.path if p))
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(paths),
            }
        )
        try:
            manifests = (native, fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "model-attachment-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            nodes = {"source": GraphNode("checkpoint-save-fixture.source")}
            for kind, node_type, inputs in (
                ("model", "ModelSave", ("model",)),
                ("checkpoint", "CheckpointSave", ("model", "clip", "vae")),
                ("clip", "CLIPSave", ("clip",)),
                ("vae", "VAESave", ("vae",)),
            ):
                nodes[kind] = GraphNode(
                    node_type,
                    {
                        "filename_prefix": f"exports/{kind}",
                        **{key: Link("source", key) for key in inputs},
                        **({"mount": "alternate"} if kind in ("checkpoint", "clip") else {}),
                    },
                )
            engine = composer.composition.make_engine(lambda _event: None)
            targets = ["model", "checkpoint", "clip", "vae"]
            for attempt in range(2):
                result = await engine.run(Graph(nodes=nodes), targets)
                assert len(result.artifacts) == 4
                assert {artifact.node_id for artifact in result.artifacts} == set(targets)
                for kind in targets:
                    mount, root = (
                        ("alternate", alternate)
                        if kind in ("checkpoint", "clip")
                        else ("output", output)
                    )
                    tail = "_" if kind in ("model", "checkpoint") else ""
                    relative = f"exports/{kind}_{attempt + 1:05d}{tail}.safetensors"
                    path = root / relative
                    artifact = next(item for item in result.artifacts if item.node_id == kind)
                    assert artifact.virtual_path == f"mounts/{mount}/{relative}"
                    assert artifact.name == path.name
                    assert artifact.digest == digest_file(path)
                    assert artifact.size == path.stat().st_size
                    assert artifact.media_type == "application/octet-stream"
                    tensors = {
                        "model.weight": torch.tensor(
                            [[1.25, -3.5, 9.75], [-0.5, 7.0, 2.0]], dtype=torch.float64
                        )
                    }
                    if kind == "checkpoint":
                        tensors.update(
                            {
                                "clip.weight": torch.tensor([[-7.0, 1.5]]),
                                "vae.weight": torch.tensor([[0.125], [-2.5]], dtype=torch.float64),
                            }
                        )
                    elif kind in ("clip", "vae"):
                        tensors = {
                            "weight": torch.tensor([[-7.0, 1.5]])
                            if kind == "clip"
                            else torch.tensor([[0.125], [-2.5]], dtype=torch.float64)
                        }
                    with safe_open(path, framework="pt", device="cpu") as saved:
                        assert set(saved.keys()) == set(tensors)
                        for key, tensor in tensors.items():
                            actual = saved.get_tensor(key)
                            assert actual.dtype == tensor.dtype
                            torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
                if attempt:
                    assert "source" in result.cached
                    assert not set(targets) & set(result.cached)
            assert len(list(output.rglob("*.safetensors"))) == 4
            assert len(list(alternate.rglob("*.safetensors"))) == 4
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_clip_export_attaches_every_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    table.add(MountDef("output", output, mode="readwrite"))
    table.scan("output")
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    with capture_saved_artifacts() as claims:
        assert CLIPSave.execute(clip=_Clip(), filename_prefix="split/encoder") == {}
    relative_paths = (
        "split/encoder_clip_l_00001.safetensors",
        "split/encoder_clip_g_00001.safetensors",
        "split/encoder_00001.safetensors",
    )
    assert len(claims) == 3
    for claim, relative in zip(claims, relative_paths, strict=True):
        path = output / relative
        assert dict(claim) == {
            "digest": digest_file(path),
            "name": path.name,
            "size": path.stat().st_size,
            "mediaType": "application/octet-stream",
            "virtualPath": f"mounts/output/{relative}",
        }
