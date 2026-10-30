"""Sparse cascade batching, quantization and conditioning preservation."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.trellis_upsample import Trellis2UpsampleStage
from dinkster_values import decode_latent, encode_latent

from dinkster.compose import PackSpec, ServingComposer


class ShapeVAE:
    vae_dtype = torch.float32

    def __init__(self) -> None:
        self.first_stage_model = self
        self.decoded: list[Any] = []
        self.prepared: Any = None

    def prepare_decode(self, shape: Any) -> None:
        self.prepared = shape

    def upsample_shape(self, slat: Any, *, upsample_times: int) -> Any:
        assert upsample_times == 4 and slat.feats.dtype == self.vae_dtype
        self.decoded.append(slat)
        if len(self.decoded) == 1:
            return torch.tensor([[0, 0, 4, 8], [0, 1, 5, 12], [0, 24, 28, 32]], dtype=torch.int32)
        return torch.tensor([[0, 16, 20, 24], [0, 16, 20, 24]], dtype=torch.int32)


@pytest.mark.parametrize("target", (1024, 2048))
@pytest.mark.parametrize("project", (False, True))
@pytest.mark.parametrize("portable", (False, True))
def test_batched_sparse_cascade_and_quantization(
    monkeypatch: pytest.MonkeyPatch, target: int, project: bool, portable: bool
) -> None:
    initialize_inference_process()
    # Hand-derived cell quantization: floor((c+.5)*grid/512) for Trellis,
    # round((c+.5)*(grid-1)/512) for Pixal. Duplicate coarse cells collapse.
    if target == 1024 and not project:
        expected = [[0, 0, 0, 1], [0, 3, 3, 4], [1, 2, 2, 3]]
        expected_counts = [2, 1]
    elif target == 1024:
        expected = [[0, 0, 1, 1], [0, 0, 1, 2], [0, 3, 4, 4], [1, 2, 3, 3]]
        expected_counts = [3, 1]
    else:
        expected = [[0, 0, 1, 2], [0, 0, 1, 3], [0, 6, 7, 8], [1, 4, 5, 6]]
        expected_counts = [3, 1]
    samples = torch.zeros(2, 32, 2, 1, dtype=torch.float64)
    samples[0, 0, :, 0] = torch.tensor([1.0, 2.0])
    samples[1, 0, :, 0] = torch.tensor([3.0, 999.0])
    coords = torch.tensor([[1, 9, 8, 7], [0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32)
    source: Any = {"samples": samples, "coords": coords, "coord_counts": torch.tensor([2, 1])}
    source_before = samples.clone(), coords.clone()
    if portable:
        source = decode_latent(encode_latent(source))
    pack = {"controlled projection": 7}
    positive = [
        [
            torch.tensor([11.0]),
            {"keep": "positive", **({"proj_feat_pack": pack} if project else {})},
        ]
    ]
    negative = [[torch.tensor([-11.0]), {"keep": "negative"}], "unchanged"]
    if project:
        model = importlib.import_module("dinkster_inference.ldm.trellis2.model")

        def projection(actual: Any, stage: str, *, coords: Any, coord_resolution: int) -> Any:
            assert actual is pack and stage == "shape_1024" and coord_resolution == target // 16
            return coords[:, 1:].float() + 100

        monkeypatch.setattr(model, "compute_stage_proj_feats", projection)
    vae = ShapeVAE()
    output: Any = Trellis2UpsampleStage.execute(
        positive=positive, negative=negative, shape_latent=source, vae=vae, target_resolution=target
    )
    result = output["LATENT"]
    torch.testing.assert_close(result["coords"], torch.tensor(expected, dtype=torch.int32))
    torch.testing.assert_close(result["coord_counts"], torch.tensor(expected_counts))
    assert result["coord_resolution"] == target // 16 and result["type"] == "trellis2"
    assert result["model_frame"] == ("y_up" if project else "z_up")
    assert result["samples"].shape == (2, 32, expected_counts[0], 1)
    assert torch.count_nonzero(result["samples"]) == 0
    assert vae.prepared == samples.shape and len(vae.decoded) == 2
    torch.testing.assert_close(vae.decoded[0].coords, coords[1:])
    torch.testing.assert_close(
        vae.decoded[1].coords, torch.tensor([[0, 9, 8, 7]], dtype=torch.int32)
    )
    # Fixed F Trellis2ShapeSLAT channel zero: mean .781296, std 5.972266.
    torch.testing.assert_close(vae.decoded[0].feats[:, 0], torch.tensor([6.753562, 12.725828]))
    torch.testing.assert_close(vae.decoded[1].feats[:, 0], torch.tensor([18.698094]))
    for name in ("positive", "negative"):
        meta = output[name][0][1]
        assert meta["keep"] == name and meta["trellis2_generation_mode"] == "shape_generation"
        torch.testing.assert_close(meta["trellis2_coords"], result["coords"])
        torch.testing.assert_close(meta["trellis2_coord_counts"], result["coord_counts"])
        if project:
            torch.testing.assert_close(
                meta["trellis2_proj_feats"], result["coords"][:, 1:].float() + 100
            )
        else:
            assert "trellis2_proj_feats" not in meta
    assert output["negative"][1] == "unchanged"
    assert "trellis2_coords" not in positive[0][1] and "trellis2_coords" not in negative[0][1]
    torch.testing.assert_close(samples, source_before[0])
    torch.testing.assert_close(coords, source_before[1])


def test_legacy_no_counts_and_explicit_frame() -> None:
    samples = torch.zeros(1, 32, 2, 1)
    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32)
    vae = ShapeVAE()
    result: Any = Trellis2UpsampleStage.execute(
        positive=[],
        negative=[],
        shape_latent={"samples": samples, "coords": coords, "model_frame": "custom"},
        vae=vae,
    )
    assert len(vae.decoded) == 1 and result["LATENT"]["model_frame"] == "custom"
    torch.testing.assert_close(vae.decoded[0].coords, coords)
    torch.testing.assert_close(result["LATENT"]["coord_counts"], torch.tensor([2]))


@pytest.mark.parametrize(
    "counts, message",
    [
        (torch.tensor([[2, 1]]), "must be 1D"),
        (torch.tensor([-1, 4]), "must be non-negative"),
        (torch.tensor([1, 1]), "does not match coords rows"),
        (torch.tensor([1, 2]), "rows for batch 1 expected"),
    ],
)
def test_invalid_sparse_counts(counts: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Trellis2UpsampleStage.execute(
            positive=[],
            negative=[],
            vae=ShapeVAE(),
            shape_latent={
                "samples": torch.zeros(2, 32, 2, 1),
                "coords": torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [1, 9, 8, 7]]),
                "coord_counts": counts,
            },
        )


class CascadeSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="trellis-cascade-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("shape_latent", LATENT_TYPE),
                    ("vae", VAE_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        conditioning = [[torch.tensor([[7.0]]), {"keep": "source"}]]
        return cls.outputs(
            positive=conditioning,
            negative=conditioning,
            shape_latent={
                "samples": torch.zeros(2, 32, 2, 1),
                "coords": torch.tensor(
                    [[0, 1, 2, 3], [0, 4, 5, 6], [1, 9, 8, 7]], dtype=torch.int32
                ),
                "coord_counts": torch.tensor([2, 1]),
            },
            vae=ShapeVAE(),
        )


class CascadeInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="trellis-cascade-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(type_id))
                for name, type_id in (
                    ("positive", CONDITIONING_TYPE),
                    ("negative", CONDITIONING_TYPE),
                    ("LATENT", LATENT_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any, LATENT: Any) -> Any:
        latent = materialize_latent(LATENT)
        expected = torch.tensor([[0, 0, 0, 1], [0, 3, 3, 4], [1, 2, 2, 3]], dtype=torch.int32)
        torch.testing.assert_close(latent["coords"], expected)
        torch.testing.assert_close(latent["coord_counts"], torch.tensor([2, 1]))
        assert latent["samples"].shape == (2, 32, 2, 1)
        assert torch.count_nonzero(latent["samples"]) == 0
        for value in (positive, negative):
            meta = materialize_latent(value)[0][1]
            torch.testing.assert_close(meta["trellis2_coords"], expected)
            assert (
                meta["keep"] == "source" and meta["trellis2_generation_mode"] == "shape_generation"
            )
        return cls.outputs(summary="sparse cascade transported")


@pytest.mark.parametrize("in_process", (False, True))
def test_unchanged_native_manifest_cascades_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "trellis_cascade_fixture.py").write_text(
            "from test_trellis_upsample_torch import CascadeSource, CascadeInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (CascadeSource, CascadeInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "cascade.toml"
        fixture.write_text(
            '[pack]\nname="trellis-cascade-fixture"\npublisher="dinkster"\n'
            'namespaces=["trellis-cascade-fixture"]\n[pack.entry]\n'
            'nodes="trellis_cascade_fixture:NODES"\n'
            'types="trellis_cascade_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "trellis-cascade-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("trellis-cascade-fixture.source"),
                    "cascade": GraphNode(
                        "Trellis2UpsampleStage",
                        {
                            name: Link("source", name)
                            for name in ("positive", "negative", "shape_latent", "vae")
                        },
                    ),
                    "inspect": GraphNode(
                        "trellis-cascade-fixture.inspect",
                        {
                            name: Link("cascade", name)
                            for name in ("positive", "negative", "LATENT")
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve() == "sparse cascade transported"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
