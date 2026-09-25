from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    PBR_CHANNELS,
    SUBDIVISION_CHANNELS,
    ConditioningCarrier,
    DenseVoxelGrid,
    ResidentPayloadBinding,
    SparseLatent,
    SparseSubdivisionGuides,
    SparseVolume,
    TriangleMesh,
    TriangleMeshBatch,
)
from dinkster_inference_torch import (
    Trellis2Conditioning,
    Trellis2ConditioningResource,
    make_sparse_support,
    make_trellis2_conditioning_resources,
    pack_sparse_latent,
)
from dinkster_inference_torch import trellis2_nodes as provider


def _resources() -> tuple[ConditioningCarrier, ConditioningCarrier]:
    positive, negative = make_trellis2_conditioning_resources(
        Trellis2Conditioning(
            torch.ones((1, 2, 4), dtype=torch.float32),
            torch.ones((1, 3, 4), dtype=torch.float32),
        ),
        vision_identity="native:dinkster.trellis2:" + "1" * 64,
        source_image_digest="sha256:" + "2" * 64,
        camera_angle_x=49.13,
    )
    return (
        provider.make_trellis2_conditioning_carrier(positive),
        provider.make_trellis2_conditioning_carrier(negative),
    )


def _resource(value: object) -> Trellis2ConditioningResource:
    assert type(value) is ConditioningCarrier
    assert len(value.bindings) == 1
    assert value.bindings[0].kind == "resident"
    payload = value.bindings[0].payload
    assert type(payload) is Trellis2ConditioningResource
    return payload


def test_resources_use_canonical_family_carriers_with_shared_owner() -> None:
    positive, negative = _resources()

    assert positive.conditioning.records[0].token_layout is not None
    assert positive.conditioning.records[0].token_layout.family_id == "dinkster.trellis2"
    positive_resource = _resource(positive)
    negative_resource = _resource(negative)
    assert positive_resource.shares_storage_with(negative_resource)
    positive_binding = positive.bindings[0]
    negative_binding = negative.bindings[0]
    assert isinstance(positive_binding, ResidentPayloadBinding)
    assert isinstance(negative_binding, ResidentPayloadBinding)
    assert positive_binding.fingerprint != negative_binding.fingerprint


def test_empty_structure_latent_matches_the_official_sampling_shape() -> None:
    result = provider.execute_empty_trellis2_latent_structure(batch_size=2)
    latent = cast("dict[str, object]", result["latent"])
    samples = latent["samples"]

    assert type(samples) is torch.Tensor
    assert samples.shape == (2, 32, 16, 16, 16)
    assert not torch.count_nonzero(samples)
    assert latent["trellis2_frame"] == "z_up"


def test_shape_and_texture_stages_preserve_resident_pair_and_sparse_support() -> None:
    positive, negative = _resources()
    occupancy = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)
    occupancy[0, 0, 1, 2, 3] = 1.0
    shape = provider.execute_trellis2_shape_stage(
        positive=positive,
        negative=negative,
        voxel=DenseVoxelGrid(occupancy, ("occupancy",), "z_up"),
    )
    shape_latent = cast("dict[str, object]", shape["latent"])
    samples = shape_latent["samples"]

    assert type(samples) is SparseLatent
    sparse = cast("SparseLatent[torch.Tensor]", samples)
    assert sparse.support.resolution == 8
    assert sparse.support.batch_counts == (1,)
    assert sparse.features.shape == (1, 32)
    assert _resource(shape["positive"]).stage == "shape-512"

    texture = provider.execute_trellis2_texture_stage(
        positive=shape["positive"],
        negative=shape["negative"],
        shape_latent=shape_latent,
    )
    texture_latent = cast("dict[str, object]", texture["latent"])
    texture_samples = cast("SparseLatent[torch.Tensor]", texture_latent["samples"])
    assert texture_samples.support.same_support(sparse.support)
    assert texture_samples.features.shape == sparse.features.shape
    assert _resource(texture["positive"]).stage == "texture"


def test_shape_stage_refuses_an_empty_batch() -> None:
    positive, negative = _resources()
    with pytest.raises(ValueError, match="every.*batch"):
        provider.execute_trellis2_shape_stage(
            positive=positive,
            negative=negative,
            voxel=DenseVoxelGrid(
                torch.zeros((1, 1, 8, 8, 8)),
                ("occupancy",),
                "z_up",
            ),
        )


def test_upsample_stage_matches_reference_per_sample_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShapeDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.empty(()))
            self.batch_columns: list[tuple[int, ...]] = []

        def upsample_shape(
            self, latent: SparseLatent[torch.Tensor], upsample_times: int
        ) -> torch.Tensor:
            assert upsample_times == 4
            coordinates = cast("torch.Tensor", latent.support.coordinates)
            self.batch_columns.append(tuple(int(item) for item in coordinates[:, 0].tolist()))
            return coordinates

    class Handle:
        def __init__(self, module: ShapeDecoder) -> None:
            self.component = module
            self.load_device = torch.device("cpu")

        @staticmethod
        def stage() -> nullcontext[None]:
            return nullcontext()

    module = ShapeDecoder()

    def component(_value: object, _name: str) -> Handle:
        return Handle(module)

    monkeypatch.setattr(provider, "_component", component)
    positive, negative = _resources()
    support = make_sparse_support(
        torch.tensor(((0, 1, 2, 3), (1, 4, 5, 6)), dtype=torch.int32),
        (1, 1),
        32,
        (-0.5, -0.5, -0.5),
        (1.0 / 32, 1.0 / 32, 1.0 / 32),
    )
    latent = {
        "samples": pack_sparse_latent(support, torch.zeros((2, 32))),
        "trellis2_frame": "z_up",
    }

    result = provider.execute_trellis2_upsample_stage(
        positive=positive,
        negative=negative,
        shape_latent=latent,
        vae=object(),
        target_resolution=1024,
    )

    output = cast("dict[str, object]", result["latent"])
    samples = cast("SparseLatent[torch.Tensor]", output["samples"])
    assert module.batch_columns == [(0,), (0,)]
    assert samples.support.batch_counts == (1, 1)
    coordinates = cast("torch.Tensor", samples.support.coordinates)
    assert coordinates[:, 0].tolist() == [0, 1]


def test_sparse_decode_memory_estimate_is_aligned_bounded_checked_math() -> None:
    estimate = cast("Any", provider)._sparse_decode_memory_required

    assert (
        estimate(
            input_points=1024,
            output_points=0,
            guide_points=0,
            element_size=2,
        )
        == 6656 * 1024**2
    )
    assert (
        estimate(
            input_points=1,
            output_points=1024 * 1024,
            guide_points=1024,
            element_size=4,
        )
        == 7936 * 1024**2
    )
    with pytest.raises(OverflowError, match="int64"):
        estimate(
            input_points=2**63,
            output_points=0,
            guide_points=0,
            element_size=2,
        )


def test_shape_and_texture_decode_pass_geometry_memory_to_the_component_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support = make_sparse_support(
        torch.tensor(((0, 1, 1, 1),), dtype=torch.int32),
        (1,),
        4,
        (-0.5, -0.5, -0.5),
        (0.25, 0.25, 0.25),
    )
    shape_latent = {
        "samples": pack_sparse_latent(support, torch.zeros((1, 32))),
        "trellis2_frame": "y_up",
    }
    guides = SparseSubdivisionGuides(
        (
            SparseVolume(
                support,
                torch.ones((1, 8)),
                SUBDIVISION_CHANNELS,
            ),
        )
    )

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.empty((), dtype=torch.float32))

        @staticmethod
        def decode_shape(
            _latent: SparseLatent[torch.Tensor], *, frame: str
        ) -> tuple[tuple[TriangleMesh[torch.Tensor], ...], SparseSubdivisionGuides[torch.Tensor]]:
            assert frame == "y_up"
            mesh = TriangleMesh(
                torch.zeros((1, 3)),
                torch.zeros((1, 3), dtype=torch.int64),
                "y_up",
            )
            return (mesh,), guides

        @staticmethod
        def decode_texture(
            _latent: SparseLatent[torch.Tensor],
            subdivisions: SparseSubdivisionGuides[torch.Tensor],
        ) -> SparseVolume[torch.Tensor]:
            assert subdivisions.levels[0].support.same_support(support)
            return SparseVolume(support, torch.zeros((1, 6)), PBR_CHANNELS)

    class Handle:
        def __init__(self) -> None:
            self.component = Decoder()
            self.load_device = torch.device("cpu")
            self.memory_required: list[int] = []

        @contextmanager
        def stage(self, *, memory_required: int = 0) -> Generator[None]:
            self.memory_required.append(memory_required)
            yield

    handle = Handle()

    def component(_value: object, _name: str) -> Handle:
        return handle

    monkeypatch.setattr(provider, "_component", component)

    shape_result = provider.execute_vae_decode_shape_trellis(
        samples=shape_latent,
        vae=object(),
    )
    texture_result = provider.execute_vae_decode_texture_trellis(
        samples=shape_latent,
        vae=object(),
        shape_subdivides=shape_result["shape_subdivides"],
    )

    assert handle.memory_required == [
        cast("Any", provider)._shape_decode_memory_required(
            cast("SparseLatent[torch.Tensor]", shape_latent["samples"]), torch.float32
        ),
        cast("Any", provider)._texture_decode_memory_required(
            cast("SparseLatent[torch.Tensor]", shape_latent["samples"]),
            cast("SparseSubdivisionGuides[torch.Tensor]", shape_result["shape_subdivides"]),
            torch.float32,
        ),
    ]
    assert type(texture_result["voxel_colors"]) is SparseVolume


def test_mesh_output_uses_the_generic_padded_mesh_contract() -> None:
    output = cast("Any", provider)._mesh_output(
        (
            TriangleMesh(
                torch.tensor(((1.0, 2.0, 3.0),), dtype=torch.float32),
                torch.tensor(((0, 0, 0),), dtype=torch.int64),
                "z_up",
            ),
            TriangleMesh(
                torch.tensor(((4.0, 5.0, 6.0), (7.0, 8.0, 9.0)), dtype=torch.float32),
                torch.tensor(((0, 1, 1), (1, 0, 0)), dtype=torch.int64),
                "y_up",
            ),
        )
    )

    assert type(output) is TriangleMeshBatch
    assert output.vertices.shape == (2, 2, 3)
    assert output.faces.shape == (2, 2, 3)
    assert output.vertex_counts is not None
    assert output.face_counts is not None
    assert output.vertex_counts.tolist() == [1, 2]
    assert output.face_counts.tolist() == [1, 2]
    assert output.vertices[0, 0].tolist() == [1.0, 3.0, -2.0]
    assert output.vertices[1, 0].tolist() == [4.0, 5.0, 6.0]
