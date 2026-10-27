"""CPU ordinary-host regression coverage for bound model-file loaders."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Literal

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DINKSTER_ACCELERATOR"] = "cpu"

import torch
from dinkster_api.v1 import (
    CLIP_TYPE,
    CLIP_VISION_TYPE,
    CONTROL_NET_TYPE,
    CORE_STRING,
    GLIGEN_TYPE,
    MODEL_TYPE,
    STYLE_MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import MountDef, MountTable
from dinkster_graph import Graph, GraphNode, Link
from safetensors.torch import load_file, save_file

from dinkster.compose import PackSpec, ServingComposer


def _model(value: float) -> Any:
    from dinkster_inference.model_patcher import ModelPatcher

    layer = torch.nn.Linear(2, 1, bias=False)
    layer.weight.data.copy_(torch.tensor([[value, value + 0.5]]))
    return ModelPatcher(layer, torch.device("cpu"), torch.device("cpu"))


def _clip(value: float) -> Any:
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    layer = torch.nn.Linear(2, 1, bias=False)
    layer.weight.data.copy_(torch.tensor([[value, value + 0.5]]))
    clip = CLIP(no_init=True)
    clip.patcher = ModelPatcher(layer, torch.device("cpu"), torch.device("cpu"))
    clip.cond_stage_model = layer
    clip.tokenizer = object()
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = True
    clip.apply_hooks_to_conds = None
    return clip


def _runtime(path: str) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Use tiny safetensors as a real file boundary while controlling huge constructors."""
    state = load_file(path, device="cpu")
    assert torch.equal(state["marker"], torch.tensor([3.0, -7.0]))
    model = _model(float(state["marker"][0]))
    clip = _clip(float(state["marker"][1]))
    vae = {"kind": "vae", "marker": state["marker"].clone()}
    vision = {"kind": "vision", "marker": state["marker"].clone()}
    control = _model(float(state["marker"][0]) + 10.0)
    style = torch.nn.Linear(2, 1, bias=False)
    style.weight.data.copy_(torch.tensor([[13.0, -17.0]]))
    gligen = _model(19.0)
    return model, clip, vae, vision, control, style, gligen


_ORIGINALS: dict[tuple[object, str], object] = {}


def _install_controlled_runtime() -> None:
    initialize_inference_process()
    from dinkster_inference import clip_vision, controlnet, sd

    if _ORIGINALS:
        return

    def remember(module: object, name: str, replacement: object) -> None:
        _ORIGINALS[module, name] = getattr(module, name)
        setattr(module, name, replacement)

    def checkpoint(
        _config_path: str, checkpoint_path: str, **_kwargs: object
    ) -> tuple[Any, Any, Any]:
        return _runtime(checkpoint_path)[:3]

    def checkpoint_guess(path: str, **kwargs: object) -> tuple[Any, ...]:
        values = _runtime(path)
        return (*values[:3], values[3]) if kwargs.get("output_clipvision") else values[:3]

    remember(sd, "load_checkpoint", checkpoint)
    remember(sd, "load_checkpoint_guess_config", checkpoint_guess)
    remember(sd, "load_diffusion_model", lambda path, **_kwargs: _runtime(path)[0])
    remember(sd, "load_clip", lambda *, ckpt_paths, **_kwargs: _runtime(ckpt_paths[0])[1])
    remember(sd, "load_style_model", lambda path: _runtime(path)[5])
    remember(sd, "load_gligen", lambda path: _runtime(path)[6])
    remember(controlnet, "load_controlnet", lambda path, *_args: _runtime(path)[4])
    remember(clip_vision, "load", lambda path: _runtime(path)[3])


def _restore_controlled_runtime() -> None:
    for (module, name), value in _ORIGINALS.items():
        setattr(module, name, value)
    _ORIGINALS.clear()


class LoaderSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-file-loaders-fixture.source",
            outputs=(OutputSpec("ready", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        _install_controlled_runtime()
        return cls.outputs(ready="ready")


class LoaderInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-file-loaders-fixture.inspect",
            inputs=(InputSpec("ready", TypeExpr.concrete(CORE_STRING)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, ready: str) -> dict[str, object]:
        assert ready == "ready"
        return cls.outputs(summary=f"loader-runtime-pid-{os.getpid()}")


class LoaderOutputInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-file-loaders-fixture.output-inspect",
            inputs=(
                InputSpec("checkpoint_model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("checkpoint_clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("checkpoint_vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("configured_model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("unclip_vision", TypeExpr.concrete(CLIP_VISION_TYPE)),
                InputSpec("control", TypeExpr.concrete(CONTROL_NET_TYPE)),
                InputSpec("diff_control", TypeExpr.concrete(CONTROL_NET_TYPE)),
                InputSpec("unet", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("dual", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("vision", TypeExpr.concrete(CLIP_VISION_TYPE)),
                InputSpec("style", TypeExpr.concrete(STYLE_MODEL_TYPE)),
                InputSpec("gligen", TypeExpr.concrete(GLIGEN_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **outputs: Any) -> dict[str, object]:
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP

        assert type(outputs["checkpoint_model"]) is ModelPatcher
        assert type(outputs["configured_model"]) is ModelPatcher
        assert type(outputs["control"]) is ModelPatcher
        assert type(outputs["diff_control"]) is ModelPatcher
        assert type(outputs["unet"]) is ModelPatcher
        assert type(outputs["gligen"]) is ModelPatcher
        assert type(outputs["checkpoint_clip"]) is CLIP
        assert type(outputs["clip"]) is CLIP
        assert type(outputs["dual"]) is CLIP
        torch.testing.assert_close(
            outputs["checkpoint_model"].model.weight, torch.tensor([[3.0, 3.5]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["configured_model"].model.weight, torch.tensor([[3.0, 3.5]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["control"].model.weight, torch.tensor([[13.0, 13.5]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["diff_control"].model.weight, torch.tensor([[13.0, 13.5]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["unet"].model.weight, torch.tensor([[3.0, 3.5]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["checkpoint_clip"].cond_stage_model.weight,
            torch.tensor([[-7.0, -6.5]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            outputs["clip"].cond_stage_model.weight, torch.tensor([[-7.0, -6.5]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["dual"].cond_stage_model.weight, torch.tensor([[-7.0, -6.5]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["style"].weight, torch.tensor([[13.0, -17.0]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["gligen"].model.weight, torch.tensor([[19.0, 19.5]]), rtol=0, atol=0
        )
        for name in ("checkpoint_vae", "unclip_vision", "vision"):
            assert outputs[name]["marker"].tolist() == [3.0, -7.0]
        return cls.outputs(summary="all eleven output values reached observer")


class CheckpointVariantInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="model-file-loaders-fixture.checkpoint-variant-inspect",
            inputs=(
                InputSpec("configured_clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("configured_vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("unclip_model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("unclip_clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("unclip_vae", TypeExpr.concrete(VAE_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **outputs: Any) -> dict[str, object]:
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.sd import CLIP

        assert type(outputs["unclip_model"]) is ModelPatcher
        torch.testing.assert_close(
            outputs["unclip_model"].model.weight, torch.tensor([[3.0, 3.5]]), rtol=0, atol=0
        )
        for name in ("configured_clip", "unclip_clip"):
            assert type(outputs[name]) is CLIP
            torch.testing.assert_close(
                outputs[name].cond_stage_model.weight,
                torch.tensor([[-7.0, -6.5]]),
                rtol=0,
                atol=0,
            )
        for name in ("configured_vae", "unclip_vae"):
            assert outputs[name]["kind"] == "vae"
            assert outputs[name]["marker"].tolist() == [3.0, -7.0]
        return cls.outputs(summary="five checkpoint variant outputs reached observer")


HOST_NODES = (LoaderSource, LoaderInspect, LoaderOutputInspect, CheckpointVariantInspect)


@pytest.mark.parametrize("verified", (False, True))
@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("observe_outputs", (False, True, "checkpoint-variants"))
def test_model_file_loaders_bind_tiny_safetensors_through_ordinary_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: bool,
    in_process: bool,
    observe_outputs: bool | Literal["checkpoint-variants"],
) -> None:
    checkpoint = tmp_path / "loader.safetensors"
    save_file({"marker": torch.tensor([3.0, -7.0])}, checkpoint, metadata={"loader": "host"})
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    for kind in (
        "model/checkpoint",
        "model/config",
        "model/controlnet",
        "model/diffusion",
        "model/text-encoder",
        "model/clip-vision",
        "model/style",
        "model/gligen",
        "model/embedding",
    ):
        table.add(MountDef(id=kind.replace("/", "-"), path=tmp_path), kind=kind)
        if verified:
            table.scan(kind.replace("/", "-"))
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    asset = table.bind_name(checkpoint.name, "model/checkpoint")
    assert type(asset).__name__ == ("AssetRef" if verified else "LocalAssetRef")

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        module = f"model_file_loaders_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_model_file_loaders_torch import HOST_NODES, _install_controlled_runtime\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "_install_controlled_runtime()\n"
            "NODES = HOST_NODES\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname = "model-file-loaders-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["model-file-loaders-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        paths = (checkout / "src", *(sorted(checkout.glob("packages/*/src"))))
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts.json"),
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        *(str(p) for p in paths),
                        *(p for p in sys.path if p),
                    )
                ),
            }
        )
        try:
            manifests = (native, fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "model-file-loaders-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("model-file-loaders-fixture.source"),
                    "checkpoint": GraphNode(
                        "CheckpointLoaderSimple", {"ckpt_name": asset.to_wire()}
                    ),
                    "configured": GraphNode(
                        "CheckpointLoader",
                        {"config_name": asset.to_wire(), "ckpt_name": asset.to_wire()},
                    ),
                    "unclip": GraphNode("unCLIPCheckpointLoader", {"ckpt_name": asset.to_wire()}),
                    "control": GraphNode("ControlNetLoader", {"control_net_name": asset.to_wire()}),
                    "diff-control": GraphNode(
                        "DiffControlNetLoader",
                        {"model": Link("checkpoint", "MODEL"), "control_net_name": asset.to_wire()},
                    ),
                    "unet": GraphNode(
                        "UNETLoader", {"unet_name": asset.to_wire(), "weight_dtype": "default"}
                    ),
                    "clip": GraphNode(
                        "CLIPLoader",
                        {"clip_name": asset.to_wire(), "type": "stable_diffusion", "device": "cpu"},
                    ),
                    "dual": GraphNode(
                        "DualCLIPLoader",
                        {
                            "clip_name1": asset.to_wire(),
                            "clip_name2": asset.to_wire(),
                            "type": "sdxl",
                            "device": "cpu",
                        },
                    ),
                    "vision": GraphNode("CLIPVisionLoader", {"clip_name": asset.to_wire()}),
                    "style": GraphNode("StyleModelLoader", {"style_model_name": asset.to_wire()}),
                    "gligen": GraphNode("GLIGENLoader", {"gligen_name": asset.to_wire()}),
                    "inspect": GraphNode(
                        "model-file-loaders-fixture.inspect", {"ready": Link("source", "ready")}
                    ),
                }
            )
            targets = [
                "checkpoint",
                "configured",
                "unclip",
                "control",
                "diff-control",
                "unet",
                "clip",
                "dual",
                "vision",
                "style",
                "gligen",
                "inspect",
            ]
            if observe_outputs:
                graph = Graph(
                    nodes={
                        **graph.nodes,
                        "output-inspect": GraphNode(
                            "model-file-loaders-fixture.output-inspect",
                            {
                                "checkpoint_model": Link("checkpoint", "MODEL"),
                                "checkpoint_clip": Link("checkpoint", "CLIP"),
                                "checkpoint_vae": Link("checkpoint", "VAE"),
                                "configured_model": Link("configured", "MODEL"),
                                "unclip_vision": Link("unclip", "CLIP_VISION"),
                                "control": Link("control", "CONTROL_NET"),
                                "diff_control": Link("diff-control", "CONTROL_NET"),
                                "unet": Link("unet", "MODEL"),
                                "clip": Link("clip", "CLIP"),
                                "dual": Link("dual", "CLIP"),
                                "vision": Link("vision", "CLIP_VISION"),
                                "style": Link("style", "STYLE_MODEL"),
                                "gligen": Link("gligen", "GLIGEN"),
                            },
                        ),
                    }
                )
                targets = ["output-inspect"]
            if observe_outputs == "checkpoint-variants":
                graph = Graph(
                    nodes={
                        **graph.nodes,
                        "output-inspect": GraphNode(
                            "model-file-loaders-fixture.checkpoint-variant-inspect",
                            {
                                "configured_clip": Link("configured", "CLIP"),
                                "configured_vae": Link("configured", "VAE"),
                                "unclip_model": Link("unclip", "MODEL"),
                                "unclip_clip": Link("unclip", "CLIP"),
                                "unclip_vae": Link("unclip", "VAE"),
                            },
                        ),
                    }
                )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, targets)
                if observe_outputs == "checkpoint-variants":
                    assert result.outputs["output-inspect"]["summary"].resolve() == (
                        "five checkpoint variant outputs reached observer"
                    )
                elif observe_outputs:
                    assert result.outputs["output-inspect"]["summary"].resolve() == (
                        "all eleven output values reached observer"
                    )
                else:
                    assert (
                        result.outputs["inspect"]["summary"]
                        .resolve()
                        .startswith("loader-runtime-pid-")
                    )
                if attempt:
                    expected_cached = set(graph.nodes)
                    if observe_outputs == "checkpoint-variants":
                        expected_cached = {"configured", "unclip", "output-inspect"}
                    elif observe_outputs:
                        expected_cached -= {"source", "inspect"}
                    assert set(result.cached) == expected_cached
        finally:
            await composer.close()
            _restore_controlled_runtime()

    asyncio.run(scenario())


def test_image_only_checkpoint_loader_preserves_output_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_nodes_comfy import model_file_loaders as loaders

    initialize_inference_process()
    from dinkster_inference import sd

    checkpoint = tmp_path / "image.safetensors"
    save_file({"marker": torch.tensor([3.0, -7.0])}, checkpoint)
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="checkpoint", path=tmp_path), kind="model/checkpoint")
    asset = table.bind_name(checkpoint.name, "model/checkpoint")

    def load(path: str, **options: Any) -> tuple[object, ...]:
        assert options["output_clip"] is False
        assert options["output_clipvision"] is True
        assert options["output_vae"] is True
        return _runtime(path)[:4]

    monkeypatch.setattr(sd, "load_checkpoint_guess_config", load)
    result = loaders.ImageOnlyCheckpointLoader.execute(ckpt_name=asset)
    assert result["CLIP_VISION"]["kind"] == "vision"
    assert result["VAE"]["kind"] == "vae"
    torch.testing.assert_close(
        result["MODEL"].model.weight, torch.tensor([[3.0, 3.5]]), rtol=0, atol=0
    )


@pytest.mark.parametrize("invalid", (False, True))
def test_ltx_audio_vae_loader_filters_checkpoint_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: bool
) -> None:
    from dinkster_nodes_comfy import model_file_loaders as loaders

    initialize_inference_process()
    from dinkster_inference import sd

    checkpoint = tmp_path / "audio.safetensors"
    save_file(
        {
            "audio_vae.encoder.weight": torch.tensor([2.0, -3.0]),
            "vocoder.weight": torch.tensor([11.0]),
            "diffusion.weight": torch.tensor([97.0]),
        },
        checkpoint,
        metadata={"model": "audio"},
    )
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="checkpoint", path=tmp_path), kind="model/checkpoint")
    asset = table.bind_name(checkpoint.name, "model/checkpoint")

    class VAE:
        def __init__(self, *, sd: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
            self.state = sd
            self.metadata = metadata
            self.validated = False

        def throw_exception_if_invalid(self) -> None:
            if invalid:
                raise ValueError("invalid audio checkpoint")
            self.validated = True

    monkeypatch.setattr(sd, "VAE", VAE)
    if invalid:
        with pytest.raises(ValueError, match="invalid audio checkpoint"):
            loaders.LTXVAudioVAELoader.execute(ckpt_name=asset)
        return
    vae = loaders.LTXVAudioVAELoader.execute(ckpt_name=asset)["VAE"]
    assert vae.validated
    assert vae.metadata == {"model": "audio"}
    assert set(vae.state) == {"autoencoder.encoder.weight", "vocoder.weight"}
    assert vae.state["autoencoder.encoder.weight"].tolist() == [2.0, -3.0]
    assert vae.state["vocoder.weight"].tolist() == [11.0]


@pytest.mark.parametrize("device", ("default", "cpu", "other"))
def test_ltx_text_encoder_loader_preserves_file_order_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    from dinkster_nodes_comfy import model_file_loaders as loaders

    initialize_inference_process()
    from dinkster_inference import sd

    encoder, checkpoint = tmp_path / "encoder.safetensors", tmp_path / "checkpoint.safetensors"
    save_file({"marker": torch.tensor([3.0])}, encoder)
    save_file({"marker": torch.tensor([-7.0])}, checkpoint)
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="checkpoint", path=tmp_path), kind="model/checkpoint")
    table.add(MountDef(id="encoder", path=tmp_path), kind="model/text-encoder")

    def load(*, ckpt_paths: list[str], clip_type: object, **options: Any) -> tuple[float, ...]:
        assert clip_type == sd.CLIPType.LTXV
        expected = (
            {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}
            if device == "cpu"
            else {}
        )
        assert options["model_options"] == expected
        return tuple(float(load_file(path)["marker"][0]) for path in ckpt_paths)

    monkeypatch.setattr(sd, "load_clip", load)
    result = loaders.LTXAVTextEncoderLoader.execute(
        text_encoder=table.bind_name(encoder.name, "model/text-encoder"),
        ckpt_name=table.bind_name(checkpoint.name, "model/checkpoint"),
        device=device,
    )
    assert result == {"CLIP": (3.0, -7.0)}
