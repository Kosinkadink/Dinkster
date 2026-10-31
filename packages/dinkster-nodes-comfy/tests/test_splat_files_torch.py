"""Splat file exports preserve valid counts and standard file layouts."""

from __future__ import annotations

import asyncio
import gzip
import os
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    FILE_3D_TYPE,
    SPLAT_TYPE,
    AssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    digest_bytes,
    materialize_splat,
    register_file3d_types,
    register_inference_types,
)
from dinkster_assets import AssetVault
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral
from dinkster_nodes_comfy.splat_files import File3DToSplat, SplatToFile3D
from dinkster_values import register_core_types

from dinkster.compose import PackSpec, ServingComposer


def export_splat() -> SimpleNamespace:
    return SimpleNamespace(
        positions=torch.tensor([[[-1.25, 2.5, 0.125], [7.0, -9.0, 3.0]]] * 2),
        scales=torch.tensor([[[1.0, 0.5, 0.25], [8.0, 8.0, 8.0]]] * 2),
        rotations=torch.tensor([[[-2.0, 0, 0, 0], [0.0, 1.0, 0, 0]]] * 2),
        opacities=torch.tensor([[[0.25], [0.75]]] * 2),
        sh=torch.tensor([[[[0.0, 1.0, -2.0], [3.0, 4.0, 5.0]], [[0.0] * 3] * 2]] * 2),
        counts=torch.tensor([1, 2]),
    )


@pytest.mark.parametrize("format", ["ply", "ksplat", "spz"])
@pytest.mark.parametrize("portable", [False, True])
def test_splat_export_preserves_first_item_and_file_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    format: str,
    portable: bool,
) -> None:
    monkeypatch.delenv("DINKSTER_PACK_SCRATCH", raising=False)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    source = export_splat()
    before = {name: tensor.clone() for name, tensor in vars(source).items()}
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SPLAT_TYPE)
    value = spec.decode(spec.encode(source)) if portable else source
    result = SplatToFile3D.execute(splat=value, format=format)["model_3d"]
    assert isinstance(result, dict) and result["format"] == format
    data = result["source"].read_bytes()
    assert "Got 2; using first" in caplog.text
    if format == "ply":
        header, body = data.split(b"end_header\n", 1)
        assert b"element vertex 1\n" in header
        columns = np.frombuffer(body, "<f4")
        np.testing.assert_array_equal(columns[:3], [-1.25, 2.5, 0.125])
        np.testing.assert_array_equal(columns[6:12], [0, 1, -2, 3, 4, 5])
        np.testing.assert_allclose(columns[12], np.log(1 / 3), rtol=1e-7)
        np.testing.assert_allclose(columns[13:16], np.log([1, 0.5, 0.25]), rtol=1e-7)
        np.testing.assert_array_equal(columns[16:], [-2, 0, 0, 0])
    elif format == "ksplat":
        assert len(data) == 4096 + 1024 + 44
        assert data[:2] == b"\0\1"
        assert struct.unpack_from("<IIII", data, 4) == (1, 1, 1, 1)
        assert struct.unpack_from("<II", data, 4096) == (1, 1)
        assert struct.unpack_from("<I", data, 4096 + 28) == (44,)
        assert struct.unpack_from("<10f", data, 5120) == (
            -1.25,
            2.5,
            0.125,
            1.0,
            0.5,
            0.25,
            -1.0,
            0.0,
            0.0,
            0.0,
        )
        assert data[-4:] == bytes([128, 199, 0, 64])
    else:
        raw = gzip.decompress(data)
        assert len(raw) == 16 + 9 + 1 + 3 + 3 + 3
        assert struct.unpack_from("<IIIBBBB", raw) == (0x5053474E, 2, 1, 0, 12, 0, 0)
        signed_positions = [
            int.from_bytes(raw[i : i + 3], "little", signed=True) for i in (16, 19, 22)
        ]
        assert signed_positions == [-5120, 10240, 512]
        assert raw[25] == 64
        assert raw[26:29] == bytes([128, 166, 51])
        assert raw[29:32] == bytes([160, 149, 138])
        assert raw[32:] == bytes([128, 128, 128])
    for name, tensor in before.items():
        torch.testing.assert_close(getattr(source, name), tensor, rtol=0, atol=0)


def test_splat_export_quantization_limits_and_quaternion_sign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DINKSTER_PACK_SCRATCH", raising=False)
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    source = export_splat()
    source.positions[0, 0] = torch.tensor([3000.0, -3000.0, 0.5 / 4096])
    source.rotations[0, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    source.scales[0, 0] = torch.tensor([0.0, 1.0, np.exp(10)])
    source.opacities[0, 0, 0] = 1.1
    source.sh[0, 0, 0] = torch.tensor([-10.0, 0.0, 10.0])
    output = SplatToFile3D.execute(splat=source, format="spz")["model_3d"]
    assert isinstance(output, dict)
    raw = gzip.decompress(output["source"].read_bytes())
    assert raw[16:25] == bytes([255, 255, 127, 0, 0, 128, 0, 0, 0])
    assert raw[25:29] == bytes([255, 0, 128, 255])
    assert raw[29:32] == bytes([0, 160, 255])
    assert raw[32:] == bytes([64, 191, 64])
    output = SplatToFile3D.execute(splat=source, format="ksplat")["model_3d"]
    assert isinstance(output, dict)
    data = output["source"].read_bytes()
    assert struct.unpack_from("<4f", data, 5120 + 24) == (-0.5, 0.5, -0.5, 0.5)
    assert data[-4:] == bytes([0, 128, 255, 255])
    assert source.rotations[0, 0].tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_splat_export_refuses_empty_unknown_format_and_missing_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DINKSTER_PACK_SCRATCH", raising=False)
    monkeypatch.delenv("DINKSTER_ASSET_VAULT", raising=False)
    source = export_splat()
    with pytest.raises(ValueError, match="Unsupported splat format: 'splat'"):
        SplatToFile3D.execute(splat=source, format="splat")
    with pytest.raises(RuntimeError, match="produced-output store"):
        SplatToFile3D.execute(splat=source)
    source.counts[0] = 0
    for format in ("ply", "ksplat", "spz"):
        with pytest.raises(ValueError, match="SplatToFile3D: gaussian is empty"):
            SplatToFile3D.execute(splat=source, format=format)


class SplatFileSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-file-source.make",
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(splat=export_splat())


@pytest.mark.parametrize("in_process", [True, False])
@pytest.mark.parametrize("format", ["ply", "ksplat", "spz"])
def test_ordinary_splat_export_hosts_with_restart_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, format: str
) -> None:
    vault = AssetVault(tmp_path / "vault")
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "DINKSTER_ACCELERATOR": "cpu",
        "DINKSTER_ASSET_VAULT": str(vault.root),
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    module = f"splat_file_source_{format}"
    (tmp_path / f"{module}.py").write_text(
        "from dinkster_nodes_comfy import register_inference_types\n"
        "from test_splat_files_torch import SplatFileSource\nNODES=(SplatFileSource,)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="splat-file-source"\npublisher="dinkster"\n'
        'namespaces=["splat-file-source"]\n[pack.entry]\n'
        f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        data = b""
        for replay in (False, True):
            registry = TypeRegistry()
            register_core_types(registry)
            register_file3d_types(registry, vault)
            composer = ServingComposer(
                registry=registry,
                worker_env=environment,
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
                pack_scratch_root=tmp_path / "scratch",
            )
            try:
                for manifest in (
                    Path(__file__).resolve().parents[1] / "dinkster-pack.toml",
                    fixture,
                ):
                    await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
                graph = Graph(
                    {
                        "source": GraphNode("splat-file-source.make"),
                        "export": GraphNode(
                            "SplatToFile3D", {"splat": Link("source", "splat"), "format": format}
                        ),
                        "import": GraphNode(
                            "File3DToSplat", {"model_3d": Link("export", "model_3d")}
                        ),
                    }
                )
                result = await composer.composition.make_engine(lambda _event: None).run(
                    graph, ["export", "import"]
                )
                assert set(result.cached if replay else result.executed) == {
                    "source",
                    "export",
                    "import",
                }
                output = result.outputs["export"]["model_3d"].resolve()
                assert output["format"] == format
                current = output["source"].read_bytes()
                assert current and (not data or data == current)
                data = current
                imported = result.outputs["import"]["splat"].resolve()
                tensor = materialize_splat(imported)
                torch.testing.assert_close(tensor.positions, torch.tensor([[[-1.25, 2.5, 0.125]]]))
                if format == "ply":
                    torch.testing.assert_close(tensor.sh, export_splat().sh[:1, :1])
                else:
                    assert tensor.sh.shape == (1, 1, 1, 3)
            finally:
                await composer.close()
            shutil.rmtree(tmp_path / "scratch")
            assert output["source"].read_bytes() == data

    asyncio.run(scenario())


def import_file(tmp_path: Path, data: bytes, format: str) -> SimpleNamespace:
    vault = AssetVault(tmp_path / "input-vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    source = AssetRef(digest, f"input.{format}", len(data), resolver=vault)
    return File3DToSplat.execute(model_3d={"source": source, "format": format})["splat"]


def test_splat_import_binary_records_and_detection(tmp_path: Path) -> None:
    data = struct.pack("<6f8B", -1.25, 2.5, 0.125, 1, 0.5, 0.25, 64, 128, 255, 17, 0, 128, 128, 128)
    for format in ("splat", "obj"):
        value = import_file(tmp_path, data, format)
        torch.testing.assert_close(value.positions, torch.tensor([[[-1.25, 2.5, 0.125]]]))
        torch.testing.assert_close(value.scales, torch.tensor([[[1.0, 0.5, 0.25]]]))
        torch.testing.assert_close(value.rotations, torch.tensor([[[-1.0, 0.0, 0.0, 0.0]]]))
        torch.testing.assert_close(value.opacities, torch.tensor([[[17 / 255]]]))
        torch.testing.assert_close(
            value.sh,
            torch.tensor(
                [
                    [
                        [
                            [
                                (64 / 255 - 0.5) / 0.28209479177387814,
                                (128 / 255 - 0.5) / 0.28209479177387814,
                                0.5 / 0.28209479177387814,
                            ]
                        ]
                    ]
                ]
            ),
        )
        assert value.counts is None and value.positions.dtype == torch.float32
    empty = import_file(tmp_path, b"", "splat")
    assert empty.positions.shape == (1, 0, 3) and empty.opacities.shape == (1, 0, 1)
    with pytest.raises(ValueError, match="multiple of 32"):
        import_file(tmp_path, data[:-1], "splat")


@pytest.mark.parametrize("level", [0, 1, 2])
def test_ksplat_import_sections_capacity_buckets_and_harmonic_stride(
    tmp_path: Path, level: int
) -> None:
    header = bytearray(4096)
    header[1] = 1
    struct.pack_into("<I", header, 4, 2)
    struct.pack_into("<H", header, 20, level)
    sections, bodies = [], []
    for index, count in enumerate((3, 1)):
        capacity = count + 1
        center_type, float_type = ("<f4", "<f4") if level == 0 else ("<u2", "<f2")
        dtype = np.dtype(
            [
                ("center", center_type, 3),
                ("scale", float_type, 3),
                ("rotation", float_type, 4),
                ("color", "u1", 4),
                ("sh", ("<f4", "<f2", "u1")[level], 9),
            ]
        )
        records = np.zeros(capacity, dtype=dtype)
        records["center"][:count] = [11, 9, 10] if level else [-1.25, 2.5, 0.125]
        records["scale"][:count] = [0.5, 2.0, 0.25]
        records["rotation"][:count] = [0.0, -2.0, 0.0, 0.0]
        records["color"][:count] = [32, 64, 128, 200]
        records["sh"] = 77
        section = bytearray(1024)
        if level:
            full = 1 if count == 3 else 0
            centers = np.array([[2 + index, 3, -4], [-7, 5, 9]][: full + 1], dtype="<f4")
            bucket_data = struct.pack("<I", 1) + centers.tobytes()
            struct.pack_into("<IIIIfH", section, 0, count, capacity, 2, full + 1, 4.0, 12)
            struct.pack_into("<III", section, 24, 10, len(records.tobytes()), full)
            struct.pack_into("<I", section, 36, 1)
        else:
            bucket_data = b""
            struct.pack_into("<II", section, 0, count, capacity)
        struct.pack_into("<H", section, 40, 1)
        sections.append(section)
        bodies.append(bucket_data + records.tobytes())
    data = bytes(header) + b"".join(sections) + b"".join(bodies)
    value = import_file(tmp_path, data, "ksplat")
    expected = (
        [[2.2, 2.8, -4], [2.2, 2.8, -4], [-6.8, 4.8, 9], [3.2, 2.8, -4]]
        if level
        else [[-1.25, 2.5, 0.125]] * 4
    )
    torch.testing.assert_close(value.positions, torch.tensor([expected]))
    torch.testing.assert_close(value.scales, torch.tensor([[[0.5, 2, 0.25]] * 4]))
    torch.testing.assert_close(value.rotations, torch.tensor([[[0.0, -1, 0, 0]] * 4]))
    torch.testing.assert_close(value.opacities, torch.full((1, 4, 1), 200 / 255))
    assert value.sh.shape == (1, 4, 1, 3)
    with pytest.raises(ValueError, match="truncated"):
        import_file(tmp_path, data[:-1], "ksplat")


@pytest.mark.parametrize("version", [1, 2, 3])
def test_spz_import_signed_coordinates_and_quaternion_layout(tmp_path: Path, version: int) -> None:
    positions = (
        struct.pack("<6e", -1.25, 2.5, 0.125, 3, -4, 0.5)
        if version == 1
        else b"".join(
            value.to_bytes(3, "little", signed=True)
            for value in (-5120, 10240, 512, 12288, -16384, 2048)
        )
    )
    rotation = (
        struct.pack("<II", (3 << 30) | 511, (1 << 30) | (1 << 9) | 255)
        if version == 3
        else bytes([128, 128, 255, 0, 128, 128])
    )
    raw = (
        struct.pack("<IIIBBBB", 0x5053474E, version, 2, 0, 12, 0, 0)
        + positions
        + bytes([17, 230])
        + bytes([32, 128, 240, 90, 12, 180])
        + bytes([160, 144, 128, 80, 120, 200])
        + rotation
    )
    value = import_file(tmp_path, gzip.compress(raw), "spz")
    torch.testing.assert_close(value.positions, torch.tensor([[[-1.25, 2.5, 0.125], [3, -4, 0.5]]]))
    torch.testing.assert_close(
        value.scales, torch.exp(torch.tensor([[[0, -1, -2], [-5, -2.5, 2.5]]]))
    )
    torch.testing.assert_close(value.opacities, torch.tensor([[[17 / 255], [230 / 255]]]))
    torch.testing.assert_close(
        value.sh, (torch.tensor([[[[32, 128, 240]], [[90, 12, 180]]]]) / 255 - 0.5) / 0.15
    )
    if version == 3:
        magnitude = 255 / (511 * np.sqrt(2))
        expected = torch.tensor(
            [[[np.sqrt(0.5), 0, 0, np.sqrt(0.5)], [-magnitude, 0, np.sqrt(1 - magnitude**2), 0]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(value.rotations, expected)
    else:
        vector = torch.tensor(
            [[128 / 127.5 - 1, 128 / 127.5 - 1, 1], [-1, 128 / 127.5 - 1, 128 / 127.5 - 1]]
        )
        expected = torch.cat([torch.zeros(2, 1), vector], dim=1)
        expected /= expected.norm(dim=1, keepdim=True)
        torch.testing.assert_close(value.rotations[0], expected)


def ply_import_bytes(colored_points: bool) -> bytes:
    properties = ["x", "y", "z"]
    if colored_points:
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        header += "".join(f"property float {name}\n" for name in properties)
        header += "".join(f"property uchar {name}\n" for name in ("red", "green", "blue"))
        return (header + "end_header\n").encode("ascii") + struct.pack(
            "<3f3B", -2.5, 1.25, 7, 32, 128, 240
        )
    properties += [f"f_dc_{index}" for index in range(3)]
    properties += [f"f_rest_{index}" for index in (5, 0, 3, 1, 4, 2)]
    properties += ["opacity"] + [f"scale_{index}" for index in range(3)]
    properties += [f"rot_{index}" for index in range(4)]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
    header += "".join(f"property float {name}\n" for name in properties)
    rows = [
        [
            -1.25,
            2.5,
            0.125,
            1,
            2,
            3,
            32,
            11,
            22,
            12,
            31,
            21,
            0,
            np.log(2),
            np.log(0.5),
            np.log(3),
            -2,
            0,
            0,
            0,
        ],
        [7, -4, 0.5, -1, 0.5, 2.5, -32, -11, -22, -12, -31, -21, np.log(3), 0, 0, 0, 0, 3, 4, 0],
    ]
    return (header + "end_header\n").encode("ascii") + b"".join(
        struct.pack("<20f", *row) for row in rows
    )


@pytest.mark.parametrize("colored_points", [False, True])
def test_ply_import_harmonics_order_activation_and_point_defaults(
    tmp_path: Path, colored_points: bool
) -> None:
    data = ply_import_bytes(colored_points)
    value = import_file(tmp_path, data, "ply")
    if colored_points:
        torch.testing.assert_close(value.positions, torch.tensor([[[-2.5, 1.25, 7]]]))
        torch.testing.assert_close(value.scales, torch.full((1, 1, 3), 0.01))
        torch.testing.assert_close(value.rotations, torch.tensor([[[1.0, 0, 0, 0]]]))
        torch.testing.assert_close(value.opacities, torch.ones(1, 1, 1))
        torch.testing.assert_close(
            value.sh, (torch.tensor([[[[32, 128, 240]]]]) / 255 - 0.5) / 0.28209479177387814
        )
    else:
        torch.testing.assert_close(
            value.positions, torch.tensor([[[-1.25, 2.5, 0.125], [7, -4, 0.5]]])
        )
        torch.testing.assert_close(value.scales, torch.tensor([[[2.0, 0.5, 3], [1, 1, 1]]]))
        torch.testing.assert_close(
            value.rotations, torch.tensor([[[-1.0, 0, 0, 0], [0, 0.6, 0.8, 0]]])
        )
        torch.testing.assert_close(value.opacities, torch.tensor([[[0.5], [0.75]]]))
        torch.testing.assert_close(
            value.sh,
            torch.tensor(
                [
                    [
                        [[1.0, 2, 3], [11, 21, 31], [12, 22, 32]],
                        [[-1, 0.5, 2.5], [-11, -21, -31], [-12, -22, -32]],
                    ]
                ]
            ),
        )
    for field in (value.positions, value.scales, value.rotations, value.opacities, value.sh):
        assert field.dtype == torch.float32 and field.is_contiguous()


def empty_ply_bytes(full_attributes: bool) -> bytes:
    if full_attributes:
        return (
            ply_import_bytes(False)
            .split(b"end_header\n", 1)[0]
            .replace(b"element vertex 2", b"element vertex 0")
            + b"end_header\n"
        )
    return (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )


@pytest.mark.parametrize("in_process", [False, True])
@pytest.mark.parametrize("full_attributes", [False, True])
def test_empty_ply_import_hosts_and_restart_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, full_attributes: bool
) -> None:
    data = empty_ply_bytes(full_attributes)
    vault = AssetVault(tmp_path / "vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "DINKSTER_ACCELERATOR": "cpu",
        "DINKSTER_ASSET_VAULT": str(vault.root),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    source = AssetRef(digest, "empty.ply", len(data), resolver=vault)

    async def scenario() -> None:
        for replay in (False, True):
            registry = TypeRegistry()
            register_core_types(registry)
            register_file3d_types(registry, vault)
            composer = ServingComposer(
                registry=registry,
                worker_env=environment,
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
                pack_scratch_root=tmp_path / "scratch",
            )
            try:
                await composer.add_pack(
                    PackSpec(
                        manifest=Path(__file__).resolve().parents[1] / "dinkster-pack.toml",
                        in_process=in_process,
                    )
                )
                graph = Graph(
                    {
                        "import": GraphNode(
                            "File3DToSplat",
                            {
                                "model_3d": TypedLiteral(
                                    FILE_3D_TYPE, {"format": "ply", "source": source.to_wire()}
                                )
                            },
                        )
                    }
                )
                result = await composer.composition.make_engine(lambda _event: None).run(
                    graph, ["import"]
                )
                assert set(result.cached if replay else result.executed) == {"import"}
                splat = materialize_splat(result.outputs["import"]["splat"].resolve())
                assert splat.positions.shape == splat.scales.shape == (1, 0, 3)
                assert splat.rotations.shape == (1, 0, 4)
                assert splat.opacities.shape == (1, 0, 1)
                assert splat.sh.shape == (1, 0, 3 if full_attributes else 1, 3)
                assert splat.counts is None
                for field in (
                    splat.positions,
                    splat.scales,
                    splat.rotations,
                    splat.opacities,
                    splat.sh,
                ):
                    assert field.dtype == torch.float32 and field.is_contiguous()
            finally:
                await composer.close()

    asyncio.run(scenario())
