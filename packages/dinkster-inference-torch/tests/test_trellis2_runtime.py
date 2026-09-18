# pyright: reportPrivateUsage=false

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import GuidanceRole, SparseSupport
from dinkster_inference_torch import (
    Trellis2Conditioning,
    Trellis2ConditioningResource,
    Trellis2DiffusionRuntime,
    Trellis2FlowBundle,
    Trellis2ProjectionMap,
    Trellis2ProjectionPack,
    make_sparse_support,
    make_trellis2_conditioning_resources,
    materialize_trellis2_resource,
    set_trellis2_conditioning_stage,
)
from dinkster_inference_torch.trellis2_runtime import _upsample_naf_features


def _resources() -> tuple[Trellis2ConditioningResource, Trellis2ConditioningResource]:
    return make_trellis2_conditioning_resources(
        Trellis2Conditioning(
            torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
            torch.arange(40, dtype=torch.float32).reshape(1, 5, 8),
        ),
        vision_identity="native:dinkster.trellis2:" + "1" * 64,
        source_image_digest="sha256:" + "2" * 64,
        camera_angle_x=49.13,
    )


def _support(x: int = 1) -> SparseSupport[torch.Tensor]:
    return make_sparse_support(
        torch.tensor([[0, x, 2, 3]], dtype=torch.int32),
        (1,),
        8,
        (-0.5, -0.5, -0.5),
        (0.125, 0.125, 0.125),
    )


class _TaggedFlow(torch.nn.Module):
    def __init__(self, tag: float) -> None:
        super().__init__()
        self.tag = tag

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        projected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, context, projected
        return latent + self.tag


class _ProfiledFlow(_TaggedFlow):
    def __init__(self, tag: float, profile: str) -> None:
        super().__init__(tag)
        self.config = SimpleNamespace(image_attention=profile)


def test_flow_bundle_dispatches_every_split_flow_variant() -> None:
    flows = [_TaggedFlow(tag) for tag in range(1, 6)]
    bundle = Trellis2FlowBundle(*cast("Any", flows))
    latent = torch.zeros((1, 1), dtype=torch.float32)
    timestep = torch.ones((1,), dtype=torch.float32)
    context = torch.zeros((1, 1, 1), dtype=torch.float32)

    assert torch.equal(bundle("structure", latent, timestep, context), latent + 1)
    assert torch.equal(bundle("shape", latent, timestep, context), latent + 2)
    assert torch.equal(
        bundle("shape", latent, timestep, context, first_shape_pass=True), latent + 3
    )
    assert torch.equal(bundle("texture", latent, timestep, context), latent + 4)
    assert torch.equal(
        bundle("texture", latent, timestep, context, low_resolution_texture=True), latent + 5
    )

    fused = Trellis2FlowBundle(*cast("Any", flows[:4]))
    assert torch.equal(
        fused("texture", latent, timestep, context, low_resolution_texture=True), latent + 4
    )


def test_conditioning_identity_binds_the_effective_texture_512_profile() -> None:
    common = [
        _ProfiledFlow(1, "global"),
        _ProfiledFlow(2, "local"),
        _ProfiledFlow(3, "local"),
        _ProfiledFlow(4, "projected"),
    ]
    first = Trellis2DiffusionRuntime(
        cast(
            "Any",
            SimpleNamespace(diffusion=Trellis2FlowBundle(*cast("Any", (*common, common[3])))),
        ),
        runtime_identity="native:first",
        compute_dtype=torch.float32,
    )
    second = Trellis2DiffusionRuntime(
        cast(
            "Any",
            SimpleNamespace(
                diffusion=Trellis2FlowBundle(
                    *cast("Any", (*common, _ProfiledFlow(5, "projected-local")))
                )
            ),
        ),
        runtime_identity="native:second",
        compute_dtype=torch.float32,
    )
    fused = Trellis2DiffusionRuntime(
        cast(
            "Any",
            SimpleNamespace(diffusion=Trellis2FlowBundle(*cast("Any", common))),
        ),
        runtime_identity="native:fused",
        compute_dtype=torch.float32,
    )

    assert first.conditioning_identity != second.conditioning_identity
    assert first.conditioning_identity == fused.conditioning_identity
    assert first.conditioning_identity == (
        "dinkster.trellis2.conditioning:v2:global:local:local:projected:projected"
    )


class _RecordingNaf(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((), dtype=torch.float32))
        self.calls: list[tuple[int, torch.device, torch.device]] = []

    def forward(
        self,
        image: torch.Tensor,
        features: torch.Tensor,
        output_size: tuple[int, int],
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert output_size == (4, 4)
        self.calls.append((image.shape[0], features.device, output.device))
        output.fill_(len(self.calls))
        return output


def test_naf_upsampling_processes_each_item_into_host_output() -> None:
    naf = _RecordingNaf()
    pixels = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    patches = torch.zeros((2, 8, 2, 2), dtype=torch.float32)

    output = _upsample_naf_features(naf, pixels, patches, 4, 4, torch.device("cpu"))

    assert naf.calls == [
        (1, torch.device("cpu"), torch.device("cpu")),
        (1, torch.device("cpu"), torch.device("cpu")),
    ]
    assert output.shape == (2, 8, 4, 4)
    assert torch.equal(output[:, 0, 0, 0], torch.tensor((1.0, 2.0)))


def test_conditioning_resources_are_compact_paired_resident_stubs() -> None:
    positive, negative = _resources()

    assert positive.guidance_role is GuidanceRole.CONDITIONAL
    assert negative.guidance_role is GuidanceRole.UNCONDITIONAL
    assert positive.shares_backing(negative)
    assert positive.stage == negative.stage == "structure"
    assert positive.frame == negative.frame == "z_up"
    resource = cast("Any", positive)
    owner = resource._dinkster_resident_owner
    assert owner._dinkster_resident_cost == {"ram": 256}
    assert len(resource._dinkster_resident_fingerprint) < 160


def test_conditioning_materializes_only_the_requested_guidance_lane() -> None:
    positive, negative = _resources()
    positive_value = materialize_trellis2_resource(positive, support=None, device="cpu")
    negative_value = materialize_trellis2_resource(negative, support=None, device="cpu")

    assert torch.count_nonzero(positive_value.global_512)
    assert torch.count_nonzero(positive_value.global_1024)
    assert not torch.count_nonzero(negative_value.global_512)
    assert not torch.count_nonzero(negative_value.global_1024)


def test_sparse_stage_binds_support_and_refuses_lane_or_support_mismatch() -> None:
    positive, negative = _resources()
    support = _support()
    shape_positive, shape_negative = set_trellis2_conditioning_stage(
        positive,
        negative,
        support,
        stage="shape-512",
    )

    assert shape_positive.support_id == support.support_id
    assert shape_positive.shares_backing(shape_negative)
    materialized = materialize_trellis2_resource(
        shape_positive,
        support=support,
        device="cpu",
    )
    assert materialized.stage == "shape-512"
    with pytest.raises(ValueError, match="lanes are reversed"):
        set_trellis2_conditioning_stage(negative, positive, support, stage="shape")
    with pytest.raises(ValueError, match="differs"):
        materialize_trellis2_resource(shape_positive, support=_support(2), device="cpu")


def test_conditioning_refuses_mutation_and_wrong_process(monkeypatch: pytest.MonkeyPatch) -> None:
    positive, _ = _resources()
    value = materialize_trellis2_resource(positive, support=None, device="cpu")
    value.global_512.zero_()
    fresh_value = materialize_trellis2_resource(positive, support=None, device="cpu")
    assert torch.count_nonzero(fresh_value.global_512)

    owner = cast("Any", positive)._dinkster_resident_owner
    owner._global_512.zero_()
    with pytest.raises(RuntimeError, match="mutated"):
        materialize_trellis2_resource(positive, support=None, device="cpu")

    fresh, _ = _resources()
    monkeypatch.setattr(os, "getpid", lambda: os.getppid())
    with pytest.raises(RuntimeError, match="another process"):
        materialize_trellis2_resource(fresh, support=None, device="cpu")


def test_pixal_negative_lane_zeros_projected_features() -> None:
    projection = Trellis2ProjectionMap(
        torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        None,
        4,
    )
    pack = Trellis2ProjectionPack(
        {"ss": projection},
        torch.eye(4, dtype=torch.float32).unsqueeze(0),
        torch.tensor((0.75,), dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
    )
    source = Trellis2Conditioning(
        torch.ones((1, 2, 4), dtype=torch.float32),
        torch.ones((1, 3, 4), dtype=torch.float32),
        projection_pack=pack,
        frame="y_up",
    )
    positive, negative = make_trellis2_conditioning_resources(
        source,
        vision_identity="native:dinkster.trellis2:" + "1" * 64,
        source_image_digest="sha256:" + "2" * 64,
        camera_angle_x=49.13,
    )

    positive_value = materialize_trellis2_resource(positive, support=None, device="cpu")
    negative_value = materialize_trellis2_resource(negative, support=None, device="cpu")

    positive_projected = positive_value.projected
    negative_projected = negative_value.projected
    assert positive_projected is not None
    assert negative_projected is not None
    assert torch.count_nonzero(positive_projected)
    assert not torch.count_nonzero(negative_projected)
    assert not torch.count_nonzero(negative_value.global_512)
    assert not torch.count_nonzero(negative_value.global_1024)


def test_texture_fingerprint_binds_private_shape_features() -> None:
    positive, negative = _resources()
    support = _support()
    first, _ = set_trellis2_conditioning_stage(
        positive,
        negative,
        support,
        stage="texture",
        shape_features=torch.zeros((1, 32), dtype=torch.float32),
    )
    second, _ = set_trellis2_conditioning_stage(
        positive,
        negative,
        support,
        stage="texture",
        shape_features=torch.ones((1, 32), dtype=torch.float32),
    )

    first_value = materialize_trellis2_resource(first, support=support, device="cpu")
    assert first_value.shape_features is not None
    first_value.shape_features.fill_(7.0)
    rematerialized = materialize_trellis2_resource(first, support=support, device="cpu")

    assert rematerialized.shape_features is not None
    assert not torch.count_nonzero(rematerialized.shape_features)
    assert (
        cast("Any", first)._dinkster_resident_fingerprint
        != cast("Any", second)._dinkster_resident_fingerprint
    )


def test_sparse_sampling_mask_preserves_features_and_requires_matching_support() -> None:
    from dinkster_inference import CustomSamplingRequest, PreparedMultiStreamConditioning
    from dinkster_inference_torch.solvers import torch_sampler_registry
    from dinkster_inference_torch.sparse import pack_sparse_latent, unpack_sparse_latent

    class Flow(_ProfiledFlow):
        def __init__(self) -> None:
            super().__init__(1.0, "global")
            self.input_layer = torch.nn.Linear(4, 4)

        def forward(self, latent: Any, *args: Any, **kwargs: Any) -> Any:
            support, features = unpack_sparse_latent(latent)
            return pack_sparse_latent(support, torch.full_like(features, self.tag))

    flows = Trellis2FlowBundle(*cast("Any", [Flow() for _ in range(5)]))
    runtime = Trellis2DiffusionRuntime(
        cast("Any", SimpleNamespace(diffusion=flows)),
        runtime_identity="test:sparse-mask",
        compute_dtype=torch.float32,
    )
    support = _support()
    latent = pack_sparse_latent(support, torch.ones((1, 4)))
    noise = pack_sparse_latent(support, torch.zeros((1, 4)))
    mask = pack_sparse_latent(support, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
    positive, negative = _resources()
    positive, _ = set_trellis2_conditioning_stage(positive, negative, support, stage="shape-512")
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    kwargs: dict[str, Any] = {
        "noise": noise,
        "cond": PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
        "request": CustomSamplingRequest(sampler, (), (1.0, 0.5, 0.0)),
    }
    result = runtime.sample_custom(latent, denoise_mask=mask, **kwargs)
    output_support, output = unpack_sparse_latent(result.output)
    assert output_support.same_support(support)
    assert torch.equal(output[:, ::2], torch.ones((1, 2)))
    assert not torch.equal(output[:, 1::2], torch.ones((1, 2)))
    with pytest.raises(ValueError, match="mask must use the latent support"):
        runtime.sample_custom(
            latent, denoise_mask=pack_sparse_latent(_support(2), torch.zeros((1, 4))), **kwargs
        )


def test_dense_conditioning_batch_matches_separate_evaluation() -> None:
    from dinkster_inference import (
        ConditioningBatching,
        ConditioningBatchingMode,
        CustomSamplingRequest,
        PreparedMultiStreamConditioning,
        SamplingGuidance,
    )
    from dinkster_inference_torch.solvers import torch_sampler_registry

    class Flow(_ProfiledFlow):
        def __init__(self) -> None:
            super().__init__(0.0, "global")
            self.input_layer = torch.nn.Linear(32, 32)
            self.batch_sizes: list[int] = []

        def forward(
            self,
            latent: torch.Tensor,
            timestep: torch.Tensor,
            context: torch.Tensor,
            *,
            projected: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del timestep, projected
            self.batch_sizes.append(latent.shape[0])
            scale = context.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
            return latent * 0.25 + scale

    def sample(mode: ConditioningBatchingMode) -> tuple[torch.Tensor, list[int]]:
        flows = [Flow() for _ in range(5)]
        runtime = Trellis2DiffusionRuntime(
            cast("Any", SimpleNamespace(diffusion=Trellis2FlowBundle(*cast("Any", flows)))),
            runtime_identity="test:dense-batching",
            compute_dtype=torch.float32,
        )
        positive = Trellis2Conditioning(torch.full((1, 3, 8), 1.0), torch.full((1, 5, 8), 2.0))
        negative = Trellis2Conditioning(torch.full((1, 3, 8), -1.0), torch.full((1, 5, 8), -2.0))

        def wrap(value: Trellis2Conditioning) -> PreparedMultiStreamConditioning:
            return PreparedMultiStreamConditioning(runtime.conditioning_identity, value)

        sampler = torch_sampler_registry().get("dinkster.euler")
        assert sampler is not None
        latent = torch.zeros((1, 32, 16, 16, 16), dtype=torch.float32)
        result = runtime.sample_custom(
            latent,
            noise=torch.zeros_like(latent),
            cond=wrap(positive),
            cfg=SamplingGuidance(
                wrap(negative),
                2.0,
                batching=ConditioningBatching(
                    mode,
                    max_fused_lanes=(
                        2 if mode is ConditioningBatchingMode.MAX_FUSED_LANES else None
                    ),
                ),
            ),
            request=CustomSamplingRequest(sampler, (), (1.0, 0.0)),
        )
        return cast("torch.Tensor", result.output), flows[0].batch_sizes

    fused, fused_batches = sample(ConditioningBatchingMode.MAX_FUSED_LANES)
    separate, separate_batches = sample(ConditioningBatchingMode.FORCE_SEPARATE)

    torch.testing.assert_close(fused, separate)
    assert fused_batches == [2]
    assert separate_batches == [1, 1]
