"""DA3 public results, bound files and ordinary pack-host transport on CPU."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    DA3_GEOMETRY_TYPE,
    DA3_MODEL_TYPE,
    IMAGE_TYPE,
    MESH_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_mesh,
    materialize_vision_packet,
)
from dinkster_assets import MountDef, MountTable
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.depth_anything_3 import (
    DA3GeometryToMesh,
    DA3Inference,
    DA3Render,
    LoadDA3Model,
)
from safetensors.torch import save_file

from dinkster.compose import PackSpec, ServingComposer


class ControlledDiffusion(torch.nn.Module):
    def __init__(self, dtype: Any = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype
        self.cam_dec = True
        self.head_type = "dualdpt"
        self.marker = torch.nn.Parameter(torch.tensor([2.0], dtype=dtype))

    def forward(self, image: Any, **options: Any) -> dict[str, Any]:
        assert not torch.is_grad_enabled()
        flat = image[0] if image.ndim == 5 else image
        assert image.dtype == self.dtype and image.device.type == "cpu"
        # Constant red channels survive the actual Lanczos resize. Decode their
        # ImageNet normalization to distinguish images and scene dimensions.
        ids = torch.round((flat[:, 0, 0, 0].float() * 0.229 + 0.485) / 0.5)
        depth = (self.marker.float() + 2 * ids)[:, None, None].expand(-1, 2, 3)
        result = {"depth": depth, "depth_conf": depth + 1, "sky": depth * 0}
        if image.ndim == 5:
            assert options["ref_view_strategy"] == "middle"
            assert options["use_ray_pose"] is True
            result["intrinsics"] = torch.tensor([[2.0, 0, 1], [0, 4, 1], [0, 0, 1]])[
                None, None
            ].repeat(1, len(ids), 1, 1)
            result["extrinsics"] = torch.eye(4)[None, None, :3].repeat(1, len(ids), 1, 1)
        return result


def _controlled_load(path: str, *, model_options: dict[str, Any]) -> Any:
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.utils import load_torch_file

    state = load_torch_file(path, safe_load=True)
    assert state["marker"].tolist() == [2.0, -7.0]
    model = torch.nn.Module()
    model.diffusion_model = ControlledDiffusion(model_options.get("dtype", torch.float32))
    return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


def _pixels() -> Any:
    return (
        torch.tensor([[0.0, 0.2, 0.4], [0.5, 0.7, 0.9]])[:, None, None].expand(2, 3, 5, 3).clone()
    )


def _packet() -> dict[str, Any]:
    return {
        "depth": torch.full((2, 3, 5), 4.0),
        "image": _pixels(),
        "intrinsics": torch.tensor([[2.0, 0, 1], [0, 4, 1], [0, 0, 1]])[None, None].repeat(
            1, 2, 1, 1
        ),
        "confidence": torch.full((2, 3, 5), 7.0),
        "sky": torch.zeros(2, 3, 5),
        "mode": "multiview",
    }


@pytest.mark.parametrize("dtype", ("default", "fp16", "bf16", "fp32"))
@pytest.mark.parametrize("mode", ("mono", "multiview"))
def test_bound_loader_and_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: str, mode: str
) -> None:
    initialize_inference_process()
    sd = importlib.import_module("dinkster_inference.sd")
    monkeypatch.setattr(sd, "load_diffusion_model", _controlled_load)
    save_file({"marker": torch.tensor([2.0, -7.0])}, tmp_path / "model.safetensors")
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="geometry", path=tmp_path), kind="model/geometry-estimation")
    asset = table.bind_name("model.safetensors", "model/geometry-estimation")
    loaded = LoadDA3Model.execute(model_name=asset, weight_dtype=dtype)["DA3_MODEL"]
    assert (
        loaded.model.diffusion_model.dtype
        == {
            "default": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[dtype]
    )
    pixels = _pixels()
    before = pixels.clone()
    result = DA3Inference.execute(
        da3_model=loaded,
        image=pixels,
        resolution=140,
        mode=mode,
        **{"mode.pose_method": "ray_pose", "mode.ref_view_strategy": "middle"},
    )["da3_geometry"]
    assert result["mode"] == mode and result["depth"].shape == (2, 3, 5)
    torch.testing.assert_close(
        result["depth"], torch.tensor([2.0, 4.0])[:, None, None].expand(2, 3, 5)
    )
    assert ("intrinsics" in result) == (mode == "multiview")
    torch.testing.assert_close(result["image"], before, rtol=0, atol=0)
    torch.testing.assert_close(pixels, before, rtol=0, atol=0)
    with pytest.raises(TypeError, match="AssetRef"):
        LoadDA3Model.execute(model_name=str(tmp_path / "model.safetensors"), weight_dtype=dtype)


def test_render_and_mesh_values_preserve_source() -> None:
    packet = _packet()
    before = {
        key: value.clone() for key, value in packet.items() if isinstance(value, torch.Tensor)
    }
    raw = DA3Render.execute(
        da3_geometry=packet,
        output="depth",
        **{
            "output.normalization": "raw",
            "output.apply_sky_clip": False,
        },
    )["IMAGE"]
    torch.testing.assert_close(raw, torch.full((2, 3, 5, 3), 4.0), rtol=0, atol=0)
    confidence = DA3Render.execute(
        da3_geometry=packet,
        output="confidence",
        **{
            "output.colored": False,
        },
    )["IMAGE"]
    torch.testing.assert_close(confidence, torch.ones(2, 3, 5, 3), rtol=0, atol=0)
    mesh = DA3GeometryToMesh.execute(
        da3_geometry=packet,
        batch_index=1,
        decimation=2,
        discontinuity_threshold=0,
    )["MESH"]
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor(
            [
                [
                    [-2.0, 1.0, -4.0],
                    [2.0, 1.0, -4.0],
                    [6.0, 1.0, -4.0],
                    [-2.0, -1.0, -4.0],
                    [2.0, -1.0, -4.0],
                    [6.0, -1.0, -4.0],
                ],
            ]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        mesh.faces, torch.tensor([[[0, 4, 1], [1, 5, 2], [0, 3, 4], [1, 4, 5]]])
    )
    torch.testing.assert_close(mesh.texture, before["image"][1:2], rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(packet[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="empty mesh"):
        DA3GeometryToMesh.execute(da3_geometry=packet | {"sky": torch.ones(2, 3, 5)})


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "da3-fixture.source", outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(image=_pixels())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "da3-fixture.inspect",
            inputs=(
                InputSpec("model", TypeExpr.concrete(DA3_MODEL_TYPE)),
                InputSpec("geometry", TypeExpr.concrete(DA3_GEOMETRY_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any, geometry: Any, image: Any, mesh: Any) -> dict[str, object]:
        packet = materialize_vision_packet(geometry)
        assert model.model.diffusion_model.marker.item() == 2.0
        torch.testing.assert_close(
            packet["depth"], torch.tensor([2.0, 4.0])[:, None, None].expand(2, 3, 5)
        )
        torch.testing.assert_close(packet["image"], _pixels(), rtol=0, atol=0)
        torch.testing.assert_close(materialize_image(image), torch.ones(2, 3, 5, 3), rtol=0, atol=0)
        value = materialize_mesh(mesh)
        torch.testing.assert_close(
            value.vertices[0, 0], torch.tensor([-2.0, 1.0, -4.0]), rtol=0, atol=0
        )
        assert value.faces.shape == (1, 16, 3)
        torch.testing.assert_close(value.texture, _pixels()[1:2], rtol=0, atol=0)
        return cls.outputs(result="DA3 model geometry image mesh preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_da3_ordinary_hosts_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    initialize_inference_process()
    sd = importlib.import_module("dinkster_inference.sd")
    original = sd.load_diffusion_model
    monkeypatch.setattr(sd, "load_diffusion_model", _controlled_load)
    save_file({"marker": torch.tensor([2.0, -7.0])}, tmp_path / "model.safetensors")
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="geometry", path=tmp_path), kind="model/geometry-estimation")
    table.scan("geometry")
    asset = table.bind_name("model.safetensors", "model/geometry-estimation")
    assert asset is not None
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    (tmp_path / "da3_fixture.py").write_text(
        "from test_da3_torch import Source, Inspect, _controlled_load\n"
        "from dinkster_api.v1 import initialize_inference_process, register_inference_types\n"
        "initialize_inference_process()\n"
        "from dinkster_inference import sd\n"
        "sd.load_diffusion_model = _controlled_load\n"
        "NODES = (Source, Inspect)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="da3-fixture"\npublisher="dinkster"\nnamespaces=["da3-fixture"]\n'
        '[pack.entry]\nnodes="da3_fixture:NODES"\ntypes="da3_fixture:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts.json"),
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="da3", group_manifests=(native, fixture)
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("da3-fixture.source"),
                    "load": GraphNode(
                        "LoadDA3Model", {"model_name": asset.to_wire(), "weight_dtype": "fp32"}
                    ),
                    "infer": GraphNode(
                        "DA3Inference",
                        {
                            "da3_model": Link("load", "DA3_MODEL"),
                            "image": Link("source", "image"),
                            "resolution": 140,
                            "resize_method": "lower_bound_resize",
                            "mode.pose_method": "ray_pose",
                            "mode.ref_view_strategy": "middle",
                        },
                        slot_variants={"mode": "multiview"},
                    ),
                    "render": GraphNode(
                        "DA3Render",
                        {
                            "da3_geometry": Link("infer", "da3_geometry"),
                            "output.colored": False,
                        },
                        slot_variants={"output": "confidence"},
                    ),
                    "mesh": GraphNode(
                        "DA3GeometryToMesh",
                        {
                            "da3_geometry": Link("infer", "da3_geometry"),
                            "batch_index": 1,
                            "discontinuity_threshold": 0.0,
                        },
                    ),
                    "inspect": GraphNode(
                        "da3-fixture.inspect",
                        {
                            "model": Link("load", "DA3_MODEL"),
                            "geometry": Link("infer", "da3_geometry"),
                            "image": Link("render", "IMAGE"),
                            "mesh": Link("mesh", "MESH"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["result"].resolve()
                    == "DA3 model geometry image mesh preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        sd.load_diffusion_model = original
