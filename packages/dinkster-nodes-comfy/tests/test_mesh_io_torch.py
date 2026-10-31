"""Mesh import exercises real file bytes, transforms and ordinary asset transport."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    FILE_3D_GLTF_TYPE,
    AssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_mesh,
)
from dinkster_assets import AssetVault, MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.mesh_io import Get3DComponents
from PIL import Image

from dinkster.compose import PackSpec, ServingComposer


def scene() -> tuple[dict[str, Any], bytes]:
    positions = struct.pack("<9f", -1, 0, 2, 1, 0, 2, 0, 2, -1)
    normals = struct.pack("<9f", *([1, 1, 0] * 3))
    tangents = struct.pack("<12f", *([1, 0, 0, 1] * 3))
    uvs = struct.pack("<6f", 0.1, 0.2, 0.8, 0.3, 0.4, 0.9)
    colors = bytes([255, 0, 128, 64, 0, 255, 128, 192, 128, 64, 255, 255])
    payload = positions + normals + tangents + uvs + colors
    sizes = [len(value) for value in (positions, normals, tangents, uvs, colors)]
    views, offset = [], 0
    for size in sizes:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": size})
        offset += size
    accessors = [
        {"bufferView": index, "componentType": 5126, "count": 3, "type": kind}
        for index, kind in enumerate(("VEC3", "VEC3", "VEC4", "VEC2", "VEC4"))
    ]
    accessors[4].update(componentType=5121, normalized=True)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(payload), "uri": "geometry.bin"}],
        "bufferViews": views,
        "accessors": accessors,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TANGENT": 2,
                            "TEXCOORD_0": 3,
                            "COLOR_0": 4,
                        },
                        "material": 0,
                    }
                ]
            }
        ],
        "nodes": [{"translation": [3, 5, 7], "children": [1]}, {"mesh": 0, "scale": [-2, 3, 1]}],
        "scenes": [{"nodes": [0]}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.2, 0.4, 0.6, 0.8],
                    "metallicFactor": 0.3,
                    "roughnessFactor": 0.7,
                },
                "doubleSided": True,
                "extensions": {"KHR_materials_unlit": {}},
            }
        ],
    }
    return gltf, payload


def embedded_scene() -> bytes:
    gltf, payload = scene()
    gltf["buffers"][0]["uri"] = (
        "data:application/octet-stream;base64," + base64.b64encode(payload).decode()
    )
    return json.dumps(gltf).encode()


def local_file(
    tmp_path: Path, data: bytes, format_id: str, resources: dict[str, bytes] | None = None
) -> dict[str, Any]:
    mounts = MountTable()
    mounts.add(MountDef(id="mesh-inputs", path=tmp_path), kind="model/mesh")
    source = tmp_path / ("model." + format_id)
    source.write_bytes(data)
    refs = {}
    for name, value in (resources or {}).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        refs[name] = mounts.bind_name(path.name, "model/mesh")
    return {
        "format": format_id,
        "source": mounts.bind_name(source.name, "model/mesh"),
        "resources": refs,
    }


def assert_scene(value: object) -> None:
    mesh = materialize_mesh(value)
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor([[[5, 5, 9], [1, 5, 9], [3, 11, 6]]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(mesh.faces, torch.tensor([[[2, 1, 0]]]), rtol=0, atol=0)
    expected_normal = torch.tensor([-0.5, 1 / 3, 0.0])
    expected_normal /= expected_normal.norm()
    torch.testing.assert_close(mesh.normals, expected_normal.expand(1, 3, 3), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        mesh.tangents, torch.tensor([[[-1, 0, 0, -1]]] * 3).transpose(0, 1).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        mesh.uvs, torch.tensor([[[0.1, 0.2], [0.8, 0.3], [0.4, 0.9]]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        mesh.vertex_colors,
        torch.tensor([[[255, 0, 128, 64], [0, 255, 128, 192], [128, 64, 255, 255]]]).float() / 255,
        rtol=0,
        atol=0,
    )
    assert mesh.unlit and mesh.material == {
        "base_color_factor": [0.2, 0.4, 0.6, 0.8],
        "metallic_factor": 0.3,
        "roughness_factor": 0.7,
        "double_sided": True,
    }


@pytest.mark.parametrize("format_id", ["gltf", "glb", "embedded"])
def test_transformed_scene_material_and_attributes(tmp_path: Path, format_id: str) -> None:
    gltf, payload = scene()
    if format_id == "embedded":
        data, resources = embedded_scene(), {}
        format_id = "gltf"
    elif format_id == "glb":
        del gltf["buffers"][0]["uri"]
        encoded = json.dumps(gltf).encode()
        encoded += b" " * (-len(encoded) % 4)
        payload += b"\x00" * (-len(payload) % 4)
        data = struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(payload))
        data += struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
        data += struct.pack("<II", len(payload), 0x004E4942) + payload
        resources = {}
    else:
        data, resources = json.dumps(gltf).encode(), {"geometry.bin": payload}
    source = local_file(tmp_path, data, format_id, resources)
    assert_scene(Get3DComponents.execute(model_3d=source)["mesh"])
    assert source["source"].read_bytes() == data


def test_merge_missing_attributes_and_first_material(tmp_path: Path) -> None:
    gltf, payload = scene()
    second = {"attributes": {"POSITION": 0}, "material": 1}
    gltf["meshes"][0]["primitives"].append(second)
    gltf["materials"].append({"pbrMetallicRoughness": {"metallicFactor": 0.99}})
    source = local_file(tmp_path, json.dumps(gltf).encode(), "gltf", {"geometry.bin": payload})
    mesh = Get3DComponents.execute(model_3d=source)["mesh"]
    torch.testing.assert_close(mesh.faces, torch.tensor([[[2, 1, 0], [5, 4, 3]]]), rtol=0, atol=0)
    assert mesh.normals is None and mesh.tangents is None
    torch.testing.assert_close(mesh.uvs[:, 3:], torch.zeros(1, 3, 2), rtol=0, atol=0)
    torch.testing.assert_close(mesh.vertex_colors[:, 3:], torch.ones(1, 3, 4), rtol=0, atol=0)
    assert mesh.material["metallic_factor"] == 0.3


def test_strided_sparse_positions_and_instance_transforms(tmp_path: Path) -> None:
    gltf, payload = scene()
    interleaved = struct.pack("<12f", -1, 0, 2, 91, 1, 0, 2, 92, 0, 2, -1, 93)
    payload = interleaved + payload[36:]
    gltf["bufferViews"][0].update(byteLength=48, byteStride=16)
    for view in gltf["bufferViews"][1:]:
        view["byteOffset"] += 12
    index_offset = len(payload)
    payload += b"\x01\x00\x00\x00" + struct.pack("<3f", 4, 1, -2)
    gltf["bufferViews"].extend(
        [
            {"buffer": 0, "byteOffset": index_offset, "byteLength": 1},
            {"buffer": 0, "byteOffset": index_offset + 4, "byteLength": 12},
        ]
    )
    gltf["accessors"][0]["sparse"] = {
        "count": 1,
        "indices": {"bufferView": 5, "componentType": 5121},
        "values": {"bufferView": 6},
    }
    instance_offset = len(payload)
    payload += struct.pack("<6f", 0, 0, 0, 1, -1, 2)
    gltf["bufferViews"].append({"buffer": 0, "byteOffset": instance_offset, "byteLength": 24})
    gltf["accessors"].append({"bufferView": 7, "componentType": 5126, "count": 2, "type": "VEC3"})
    gltf["nodes"][1]["extensions"] = {"EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": 5}}}
    gltf["extensionsRequired"] = ["EXT_mesh_gpu_instancing"]
    gltf["buffers"][0]["byteLength"] = len(payload)
    source = local_file(tmp_path, json.dumps(gltf).encode(), "gltf", {"geometry.bin": payload})
    mesh = Get3DComponents.execute(model_3d=source)["mesh"]
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor(
            [[[5, 5, 9], [-5, 8, 5], [3, 11, 6], [3, 2, 11], [-7, 5, 7], [1, 8, 8]]],
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(mesh.faces, torch.tensor([[[2, 1, 0], [5, 4, 3]]]), rtol=0, atol=0)


def test_external_texture_maps_and_material_factors(tmp_path: Path) -> None:
    gltf, payload = scene()
    image_path = tmp_path / "texture.png"
    Image.new("RGBA", (2, 1), (51, 102, 204, 17)).save(image_path)
    gltf["images"] = [{"uri": "textures/base%20color.png"}]
    gltf["textures"] = [{"source": 0}]
    material = gltf["materials"][0]
    material["pbrMetallicRoughness"].update(
        baseColorTexture={"index": 0}, metallicRoughnessTexture={"index": 0}
    )
    material.update(
        normalTexture={"index": 0, "scale": 0.6},
        emissiveTexture={"index": 0},
        emissiveFactor=[0.1, 0.3, 0.5],
        occlusionTexture={"index": 0, "strength": 0.4},
    )
    material["extensions"]["KHR_materials_emissive_strength"] = {"emissiveStrength": 2.5}
    source = local_file(
        tmp_path,
        json.dumps(gltf).encode(),
        "gltf",
        {
            "geometry.bin": payload,
            "textures/base color.png": image_path.read_bytes(),
        },
    )
    mesh = Get3DComponents.execute(model_3d=source)["mesh"]
    expected = torch.tensor([[[[0.2, 0.4, 0.8], [0.2, 0.4, 0.8]]]])
    for value in (mesh.texture, mesh.metallic_roughness, mesh.normal_map, mesh.emissive):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert mesh.occlusion_in_mr and mesh.unlit
    assert mesh.material["normal_scale"] == 0.6
    assert mesh.material["occlusion_strength"] == 0.4
    assert mesh.material["emissive_factor"] == [0.1, 0.3, 0.5]
    assert mesh.material["emissive_strength"] == 2.5


@pytest.mark.parametrize(
    "uri", ["geometry.bin", "../geometry.bin", "/geometry.bin", "https://example.invalid/model.bin"]
)
def test_gltf_cannot_read_undeclared_resources(tmp_path: Path, uri: str) -> None:
    gltf, payload = scene()
    gltf["buffers"][0]["uri"] = uri
    source = local_file(tmp_path, json.dumps(gltf).encode(), "gltf")
    (tmp_path / "geometry.bin").write_bytes(payload)
    with pytest.raises(ValueError, match="not granted|URI scheme"):
        Get3DComponents.execute(model_3d=source)


def test_obj_and_ascii_binary_stl_geometry(tmp_path: Path) -> None:
    obj = b"v 0 0 0\nv 2 0 0\nv 0 3 1\nvt .2 .8\nvt .7 .3\nf -3/1 -2/2 -1/1\n"
    mesh = Get3DComponents.execute(model_3d=local_file(tmp_path, obj, "obj"))["mesh"]
    expected = torch.tensor([[[0, 0, 0], [2, 0, 0], [0, 3, 1]]], dtype=torch.float32)
    torch.testing.assert_close(mesh.vertices, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        mesh.uvs, torch.tensor([[[0.2, 0.2], [0.7, 0.7], [0.2, 0.2]]]), rtol=0, atol=1e-7
    )
    ascii_stl = (
        b"solid mesh\nfacet normal 0 0 2\nouter loop\nvertex 0 0 0\n"
        b"vertex 2 0 0\nvertex 0 3 1\nendloop\nendfacet\nendsolid"
    )
    binary = b"\x00" * 80 + struct.pack("<I12fH", 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 1, 0)
    for data in (ascii_stl, binary):
        mesh = Get3DComponents.execute(model_3d=local_file(tmp_path, data, "stl"))["mesh"]
        torch.testing.assert_close(mesh.vertices, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            mesh.normals, torch.tensor([[[0, 0, 1]]] * 3).transpose(0, 1).float(), rtol=0, atol=0
        )


class File3DSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "mesh-io-fixture.source",
            outputs=(OutputSpec("file", TypeExpr.concrete(FILE_3D_GLTF_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            file={
                "format": "gltf",
                "source": AssetRef(
                    os.environ["MESH_IO_DIGEST"], "scene.gltf", int(os.environ["MESH_IO_SIZE"])
                ).to_wire(),
                "resources": json.loads(os.environ["MESH_IO_RESOURCES"]),
            }
        )


@pytest.mark.parametrize("in_process", [True, False])
def test_ordinary_mesh_io_hosts_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, in_process: bool
) -> None:
    path = tmp_path / "scene.gltf"
    gltf, payload = scene()
    path.write_bytes(json.dumps(gltf).encode())
    sidecar = tmp_path / "geometry.bin"
    sidecar.write_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    digest = digest_file(path)
    for item in (path, sidecar):
        with vault.writer(digest_file(item)) as writer:
            writer.write(item.read_bytes())
            writer.commit()
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "DINKSTER_ACCELERATOR": "cpu",
        "DINKSTER_ASSET_VAULT": str(vault.root),
        "MESH_IO_DIGEST": digest,
        "MESH_IO_SIZE": str(path.stat().st_size),
        "MESH_IO_RESOURCES": json.dumps(
            {"geometry.bin": AssetRef(digest_file(sidecar), sidecar.name, len(payload)).to_wire()}
        ),
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    async def scenario() -> None:
        (tmp_path / "mesh_io_fixture.py").write_text(
            "from dinkster_nodes_comfy import register_inference_types\n"
            "from test_mesh_io_torch import File3DSource\nNODES=(File3DSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="mesh-io-fixture"\npublisher="dinkster"\nnamespaces=["mesh-io-fixture"]\n'
            '[pack.entry]\nnodes="mesh_io_fixture:NODES"\ntypes="mesh_io_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(worker_env=environment)
        try:
            for manifest in (Path(__file__).resolve().parents[1] / "dinkster-pack.toml", fixture):
                await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                {
                    "source": GraphNode("mesh-io-fixture.source"),
                    "read": GraphNode("Get3DComponents", {"model_3d": Link("source", "file")}),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["read"])
                assert_scene(result.outputs["read"]["mesh"].resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
