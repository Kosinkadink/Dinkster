"""Torch-free schemas for native bound model-file loaders."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_api.v1 import ASSET_TYPE, AssetRef, AssetWidget, ComboWidget, TypeExpr, digest_bytes
from dinkster_nodes_comfy import model_file_loaders
from dinkster_nodes_comfy.model_file_loaders import MODEL_FILE_LOADER_NODES


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        return self.path


def _asset(path: Path) -> AssetRef:
    data = path.read_bytes()
    return AssetRef(digest_bytes(data), path.name, len(data), resolver=_Resolver(path))


def test_model_file_loaders_use_bound_asset_inputs_and_expected_outputs() -> None:
    schemas = {node.schema().node_type: node.schema() for node in MODEL_FILE_LOADER_NODES}

    assert tuple(schemas) == (
        "CheckpointLoaderSimple",
        "CheckpointLoader",
        "unCLIPCheckpointLoader",
        "VAELoader",
        "ControlNetLoader",
        "DiffControlNetLoader",
        "UNETLoader",
        "CLIPLoader",
        "DualCLIPLoader",
        "CLIPVisionLoader",
        "StyleModelLoader",
        "GLIGENLoader",
    )
    assert schemas["CheckpointLoader"].display_name == "Load Checkpoint With Config (DEPRECATED)"
    assert schemas["CheckpointLoader"].deprecation is not None
    assert tuple(output.id for output in schemas["CheckpointLoaderSimple"].outputs) == (
        "MODEL",
        "CLIP",
        "VAE",
    )
    assert tuple(output.id for output in schemas["unCLIPCheckpointLoader"].outputs) == (
        "MODEL",
        "CLIP",
        "VAE",
        "CLIP_VISION",
    )
    assert tuple(input.id for input in schemas["DualCLIPLoader"].inputs) == (
        "clip_name1",
        "clip_name2",
        "type",
        "device",
    )
    assert not schemas["CLIPLoader"].inputs[2].required
    assert not schemas["DualCLIPLoader"].inputs[3].required
    for schema in schemas.values():
        for input in schema.inputs:
            if input.id.endswith("name"):
                assert input.type == TypeExpr.concrete(ASSET_TYPE)
                assert isinstance(input.widget, AssetWidget)


def test_model_file_loader_defaults_match_comfyui() -> None:
    schemas = {node.schema().node_type: node.schema() for node in MODEL_FILE_LOADER_NODES}
    assert schemas["UNETLoader"].inputs[1].default == "default"
    assert schemas["CLIPLoader"].inputs[1].default == "stable_diffusion"
    clip_type_widget = schemas["CLIPLoader"].inputs[1].widget
    assert isinstance(clip_type_widget, ComboWidget)
    assert len(clip_type_widget.options) == 32
    assert schemas["CLIPLoader"].inputs[2].default == "default"
    assert schemas["DualCLIPLoader"].inputs[2].default == "sdxl"
    assert schemas["DualCLIPLoader"].inputs[3].default == "default"


def test_model_file_loader_schema_discovery_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
from dinkster_nodes_comfy.model_file_loaders import MODEL_FILE_LOADER_NODES
assert len(MODEL_FILE_LOADER_NODES) == 12
assert sys.modules['torch'] is None
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_model_file_loader_execution_adapts_bound_assets_to_pinned_loader_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "fixture.safetensors"
    path.write_bytes(b"fixture")
    asset = _asset(path)
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    model, clip, vae, vision = object(), object(), object(), object()
    controlnet, style_model, gligen = object(), object(), object()
    state_dict = {"weight": object()}

    class LoadedVAE:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("VAE", (), kwargs))

        def throw_exception_if_invalid(self) -> None:
            calls.append(("validate", (), {}))

    def record(name: str, result: object):
        def call(*args: object, **kwargs: object) -> object:
            calls.append((name, args, kwargs))
            return result

        return call

    sd = SimpleNamespace(
        CLIPType=SimpleNamespace(STABLE_DIFFUSION="stable-diffusion", SDXL="sdxl"),
        VAE=LoadedVAE,
        load_checkpoint=record("checkpoint", (model, clip, vae)),
        load_checkpoint_guess_config=record("checkpoint-guess", (model, clip, vae, vision)),
        load_diffusion_model=record("diffusion", model),
        load_clip=record("clip", clip),
        load_style_model=record("style-model", style_model),
        load_gligen=record("gligen", gligen),
    )
    modules = {
        "dinkster_inference.sd": sd,
        "dinkster_inference.utils": SimpleNamespace(
            load_torch_file=record("torch-file", (state_dict, {"format": "test"}))
        ),
        "dinkster_inference.clip_vision": SimpleNamespace(load=record("vision", vision)),
        "dinkster_inference.controlnet": SimpleNamespace(
            load_controlnet=record("controlnet", controlnet)
        ),
        "torch": SimpleNamespace(float8_e4m3fn="fp8-e4m3fn", float8_e5m2="fp8-e5m2"),
    }
    monkeypatch.setattr(model_file_loaders, "initialize_inference_process", lambda: None)
    monkeypatch.setattr(model_file_loaders, "mount_roots_for_kind", lambda kind: (tmp_path,))
    real_import = model_file_loaders.importlib.import_module
    monkeypatch.setattr(
        model_file_loaders.importlib,
        "import_module",
        lambda name: modules.get(name) or real_import(name),
    )

    assert model_file_loaders.CheckpointLoaderSimple.execute(ckpt_name=asset) == {
        "MODEL": model,
        "CLIP": clip,
        "VAE": vae,
    }
    assert model_file_loaders.CheckpointLoader.execute(config_name=asset, ckpt_name=asset) == {
        "MODEL": model,
        "CLIP": clip,
        "VAE": vae,
    }
    assert model_file_loaders.unCLIPCheckpointLoader.execute(ckpt_name=asset) == {
        "MODEL": model,
        "CLIP": clip,
        "VAE": vae,
        "CLIP_VISION": vision,
    }
    assert model_file_loaders.VAELoader.execute(vae_name=asset)["VAE"].__class__ is LoadedVAE
    assert model_file_loaders.ControlNetLoader.execute(control_net_name=asset) == {
        "CONTROL_NET": controlnet
    }
    assert model_file_loaders.DiffControlNetLoader.execute(model=model, control_net_name=asset) == {
        "CONTROL_NET": controlnet
    }
    assert model_file_loaders.UNETLoader.execute(unet_name=asset, weight_dtype="fp8_e4m3fn") == {
        "MODEL": model
    }
    assert model_file_loaders.CLIPLoader.execute(
        clip_name=asset, type="stable_diffusion", device="default"
    ) == {"CLIP": clip}
    assert model_file_loaders.DualCLIPLoader.execute(
        clip_name1=asset, clip_name2=asset, type="sdxl", device="default"
    ) == {"CLIP": clip}
    assert model_file_loaders.CLIPVisionLoader.execute(clip_name=asset) == {"CLIP_VISION": vision}
    assert model_file_loaders.StyleModelLoader.execute(style_model_name=asset) == {
        "STYLE_MODEL": style_model
    }
    assert model_file_loaders.GLIGENLoader.execute(gligen_name=asset) == {"GLIGEN": gligen}

    root = str(tmp_path)
    assert calls == [
        (
            "checkpoint-guess",
            (str(path),),
            {"output_vae": True, "output_clip": True, "embedding_directory": [root]},
        ),
        (
            "checkpoint",
            (str(path), str(path)),
            {"output_vae": True, "output_clip": True, "embedding_directory": [root]},
        ),
        (
            "checkpoint-guess",
            (str(path),),
            {
                "output_vae": True,
                "output_clip": True,
                "output_clipvision": True,
                "embedding_directory": [root],
            },
        ),
        ("torch-file", (str(path),), {"safe_load": True, "return_metadata": True}),
        ("VAE", (), {"sd": state_dict, "metadata": {"format": "test"}}),
        ("validate", (), {}),
        ("controlnet", (str(path),), {}),
        ("controlnet", (str(path), model), {}),
        ("diffusion", (str(path),), {"model_options": {"dtype": "fp8-e4m3fn"}}),
        (
            "clip",
            (),
            {
                "ckpt_paths": [str(path)],
                "embedding_directory": [root],
                "clip_type": "stable-diffusion",
                "model_options": {},
            },
        ),
        (
            "clip",
            (),
            {
                "ckpt_paths": [str(path), str(path)],
                "embedding_directory": [root],
                "clip_type": "sdxl",
                "model_options": {},
            },
        ),
        ("vision", (str(path),), {}),
        ("style-model", (str(path),), {}),
        ("gligen", (str(path),), {}),
    ]
