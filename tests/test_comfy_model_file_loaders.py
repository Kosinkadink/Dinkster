"""Torch-free schemas for native bound model-file loaders."""

from pathlib import Path

from dinkster_api.v1 import AssetRef, digest_bytes
from dinkster_ltx import model_file_loaders as ltx_loaders
from dinkster_models import model_file_loaders
from dinkster_sd import model_file_loaders as sd_loaders

MODEL_FILE_LOADER_NODES = (
    model_file_loaders.CheckpointLoaderSimple,
    model_file_loaders.CheckpointLoader,
    sd_loaders.unCLIPCheckpointLoader,
    model_file_loaders.ControlNetLoader,
    model_file_loaders.DiffControlNetLoader,
    model_file_loaders.UNETLoader,
    model_file_loaders.CLIPLoader,
    model_file_loaders.DualCLIPLoader,
    model_file_loaders.CLIPVisionLoader,
    model_file_loaders.StyleModelLoader,
    sd_loaders.GLIGENLoader,
    sd_loaders.ImageOnlyCheckpointLoader,
    ltx_loaders.LTXVAudioVAELoader,
    ltx_loaders.LTXAVTextEncoderLoader,
)


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        return self.path


def _asset(path: Path) -> AssetRef:
    data = path.read_bytes()
    return AssetRef(digest_bytes(data), path.name, len(data), resolver=_Resolver(path))


def test_ltx_text_encoder_admits_omitted_device_with_default(tmp_path: Path) -> None:
    from dinkster_graph import Graph, GraphNode
    from dinkster_graph.validate import validate

    path = tmp_path / "fixture.safetensors"
    path.write_bytes(b"fixture")
    asset = _asset(path).to_wire()
    schema = ltx_loaders.LTXAVTextEncoderLoader.schema()
    graph = Graph(
        nodes={
            "loader": GraphNode(
                "dinkster.ltx.ltxav_text_encoder_loader",
                {"text_encoder": asset, "ckpt_name": asset},
            )
        }
    )
    assert validate(graph, {schema.node_type: schema}, ["loader"]) == []
