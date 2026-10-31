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
    SPLAT_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    register_file3d_types,
    register_inference_types,
)
from dinkster_assets import AssetVault
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.splat_files import SplatToFile3D
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
                    }
                )
                result = await composer.composition.make_engine(lambda _event: None).run(
                    graph, ["export"]
                )
                assert set(result.cached if replay else result.executed) == {"source", "export"}
                output = result.outputs["export"]["model_3d"].resolve()
                assert output["format"] == format
                current = output["source"].read_bytes()
                assert current and (not data or data == current)
                data = current
            finally:
                await composer.close()
            shutil.rmtree(tmp_path / "scratch")
            assert output["source"].read_bytes() == data

    asyncio.run(scenario())
