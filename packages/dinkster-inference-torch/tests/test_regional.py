"""Regional and masked materialization and evaluation proofs."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from dinkster_inference import (
    EMPTY_RANGE,
    AreaDescriptor,
    AreaUnits,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ConditionScaleVector,
    ExtensionInputValue,
    FlowSigmas,
    MaskDescriptor,
    Parameterization,
    PatchOverlay,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    SamplingTimelineSchedule,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    WeightSourceRef,
    encode_conditioning_carrier,
    make_conditioning_carrier,
    realize_sampling_timeline,
    scheduled_metadata,
)
from dinkster_inference.sampling_timeline import use_realized_sampling_timeline
from dinkster_inference_torch import (
    MASK_PAYLOAD_SPACE,
    SCALE_PAYLOAD_SPACE,
    FluxDenoiser,
    MaterializedRegion,
    RegionalConditioningError,
    RegionEvaluator,
    SDDenoiser,
    evaluate_regions,
    flux_region_evaluator,
    materialize_regions,
    sd_region_evaluator,
    tensor_to_payload_binding,
)
from dinkster_inference_torch import regional as regional_module
from dinkster_inference_torch.regional import realize_region_schedules
from golden_files import load_platform_golden, runtime_provenance
from test_denoise import tiny_cond as flux_cond
from test_denoise import tiny_flux
from test_denoise import tiny_latent as flux_latent
from test_sd_denoise import SPACE as SD15_SIGMAS
from test_sd_denoise import tiny_cond as sd_cond
from test_sd_denoise import tiny_latent as sd_latent
from test_sd_denoise import tiny_unet

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens/regional_goldens.json")
GROUPED_GOLDENS = json.loads(
    (Path(__file__).parent / "goldens/grouped_regional_goldens.json").read_text()
)
FLOW = FlowSigmas()


def _layout(family: str, tokens: int) -> TokenLayoutDescriptor:
    streams = {
        "dinkster.chroma": ("t5",),
        "dinkster.flux_dev": ("clip_l", "t5"),
        "dinkster.flux_schnell": ("clip_l", "t5"),
        "dinkster.qwen_image": ("qwen2_5_vl_7b",),
        "dinkster.sd15": ("clip_l",),
        "dinkster.sdxl": ("clip_l", "clip_g"),
        "dinkster.sdxl_refiner": ("clip_g",),
        "dinkster.wan21": ("umt5xxl",),
    }[family]
    segment_streams = ("t5",) if streams == ("clip_l", "t5") else streams
    return TokenLayoutDescriptor(
        family,
        1,
        streams,
        tuple(TokenSegmentDescriptor(stream, stream, 0, tokens) for stream in segment_streams),
    )


def _carrier(
    records: tuple[dict[str, object], ...],
    *,
    family: str = "dinkster.sd15",
    tokens: int = 3,
    features: int = 4,
):
    conditioning = []
    bindings = []
    for index, options in enumerate(records):
        text = cast(
            torch.Tensor,
            options.get(
                "text",
                torch.arange(tokens * features, dtype=torch.float32).reshape(1, tokens, features),
            ),
        )
        text_binding = tensor_to_payload_binding(f"text-{index}", text, space="conditioning-text")
        bindings.append(text_binding)
        channels: list[tuple[ConditioningChannel, PayloadDescriptor]] = [
            (
                ConditioningChannel.TEXT,
                PayloadDescriptor(
                    PayloadReference(text_binding.reference_id),
                    text_binding.shape,
                    text_binding.dtype,
                    text_binding.space,
                ),
            )
        ]
        pooled = cast(torch.Tensor | None, options.get("pooled"))
        if pooled is not None:
            pooled_binding = tensor_to_payload_binding(
                f"pooled-{index}", pooled, space="conditioning-pooled"
            )
            bindings.append(pooled_binding)
            channels.append(
                (
                    ConditioningChannel.POOLED,
                    PayloadDescriptor(
                        PayloadReference(pooled_binding.reference_id),
                        pooled_binding.shape,
                        pooled_binding.dtype,
                        pooled_binding.space,
                    ),
                )
            )
        mask_descriptor = None
        mask = cast(torch.Tensor | None, options.get("mask"))
        if mask is not None:
            mask_space = cast(str, options.get("mask_space", MASK_PAYLOAD_SPACE))
            mask_binding = tensor_to_payload_binding(f"mask-{index}", mask, space=mask_space)
            bindings.append(mask_binding)
            mask_descriptor = MaskDescriptor(
                PayloadReference(mask_binding.reference_id),
                cast(float, options.get("mask_strength", 1.0)),
                cast(bool, options.get("set_area_to_bounds", False)),
            )
        scale_vector = cast(
            ConditionScaleVector[PayloadDescriptor] | None,
            options.get("scale_vector"),
        )
        scale = cast(torch.Tensor | None, options.get("scale"))
        if scale is not None:
            scale_binding = tensor_to_payload_binding(
                f"scale-{index}", scale, space=SCALE_PAYLOAD_SPACE
            )
            bindings.append(scale_binding)
            scale_vector = ConditionScaleVector(
                PayloadDescriptor(
                    PayloadReference(scale_binding.reference_id),
                    scale_binding.shape,
                    scale_binding.dtype,
                    scale_binding.space,
                )
            )
        conditioning.append(
            ConditioningRecord(
                channels=tuple(channels),
                area=cast(AreaDescriptor | None, options.get("area")),
                mask=mask_descriptor,
                schedule=cast(
                    PercentRange,
                    options.get("schedule", PercentRange(0.0, 1.0)),
                ),
                scale_vector=scale_vector,
                token_layout=cast(
                    TokenLayoutDescriptor,
                    options.get("layout", _layout(family, int(text.shape[1]))),
                ),
                extension_metadata=cast(
                    "tuple[tuple[str, ExtensionInputValue], ...]",
                    options.get("extension_metadata", ()),
                ),
            )
        )
    carrier = make_conditioning_carrier(ConditioningSet(tuple(conditioning)), bindings)
    encode_conditioning_carrier(carrier)
    return carrier


def _tensor(golden: dict[str, object]) -> torch.Tensor:
    shape = tuple(cast("list[int]", golden["shape"]))
    return torch.tensor(golden["data"], dtype=torch.float32).reshape(shape)


def _scheduled_diffusion_metadata() -> tuple[tuple[str, ExtensionInputValue], ...]:
    overlay = PatchOverlay(
        WeightSourceRef("blake3:" + "a" * 64, "test.safetensors", 0),
        "lora",
        "identity",
        "1",
        "1",
        (),
    )
    return scheduled_metadata(
        target="dinkster.sd15",
        text_overlays=(),
        diffusion_overlays=(overlay,),
        transforms=(),
    )


def test_scale_payload_and_diffusion_digest_materialize_strictly() -> None:
    metadata = _scheduled_diffusion_metadata()
    digest = cast(str, dict(metadata)["dinkster.inference/diffusion-overlay-stack-digest"])
    carrier = _carrier(
        (
            {
                "scale": torch.tensor([0.25, 1.5]),
                "extension_metadata": metadata,
            },
        )
    )
    region = materialize_regions(carrier, "dinkster.sd15", 4, 4, "cpu")[0]
    assert region.patch_digest == digest
    assert region.scale_vector is not None
    assert torch.equal(region.scale_vector, torch.tensor([0.25, 1.5]))

    for key, value, code in (
        ("dinkster.inference/target", "dinkster.sdxl", "scheduled-metadata-target"),
        (
            "dinkster.inference/diffusion-overlay-digests",
            ("not-a-digest",),
            "scheduled-metadata-digest",
        ),
        (
            "dinkster.inference/diffusion-overlay-stack-digest",
            "b" * 64,
            "scheduled-metadata-diffusion",
        ),
        ("dinkster.inference/effective-patch-state", "b" * 64, "scheduled-metadata-effective"),
    ):
        changed = tuple((name, value if name == key else item) for name, item in metadata)
        with pytest.raises(RegionalConditioningError, match=code):
            materialize_regions(
                _carrier(({"scale": torch.ones(1), "extension_metadata": changed},)),
                "dinkster.sd15",
                4,
                4,
                "cpu",
            )

    with pytest.raises(RegionalConditioningError, match="scale-finite"):
        materialize_regions(
            _carrier(
                (
                    {
                        "scale": torch.tensor([float("nan")]),
                        "extension_metadata": metadata,
                    },
                )
            ),
            "dinkster.sd15",
            4,
            4,
            "cpu",
        )


def test_pinned_reference_provenance_percent_round_resize_and_aabb() -> None:
    expected_reference = {
        "commit": "b78cec879b9460d5cb25228a83a942fb78d2cd24",
        "device": "cpu",
        "paths": ["comfy/samplers.py", "comfy/conds.py"],
        "repo": "ComfyUI",
        "source_sha256": {
            "comfy/conds.py": "72058e9a22c972a9c875819e59d432d30d367fd2f7092ee6c6c45e5a60c959b0",
            "comfy/samplers.py": "a768a33b296c08055925f176e60e004a6f33fa95ecee3df6dd5583dccbc87a88",
        },
    }
    if "platform" in GOLDENS["reference"]:
        expected_reference.update(runtime_provenance())
    else:
        expected_reference.update({"python": "Python 3.12.3", "torch": "2.13.0+cu130"})
    assert GOLDENS["reference"] == expected_reference
    percent = AreaDescriptor(0.31, 0.49, 0.26, 0.51, AreaUnits.PERCENT)
    assert percent.materialize_percent(7, 9) == tuple(GOLDENS["percent_area"])
    assert GROUPED_GOLDENS["reference"] == {
        "commit": "b78cec879b9460d5cb25228a83a942fb78d2cd24",
        "device": "cpu",
        "paths": ["comfy/samplers.py", "comfy/conds.py", "comfy/model_base.py"],
        "python": "Python 3.12.3",
        "repo": "ComfyUI",
        "source_sha256": {
            "comfy/conds.py": "72058e9a22c972a9c875819e59d432d30d367fd2f7092ee6c6c45e5a60c959b0",
            "comfy/model_base.py": (
                "8a66ec5d6e441303f4a3c22b40d9c4f0d94e2edcdc97d073d1b1cea04c9f3de9"
            ),
            "comfy/samplers.py": "a768a33b296c08055925f176e60e004a6f33fa95ecee3df6dd5583dccbc87a88",
        },
        "torch": "2.13.0+cu130",
    }
    assert GROUPED_GOLDENS["facts"] == {
        "compatible_physical_order": ["2", "1", "0"],
        "forward_count": 1,
        "reciprocal_floor_counts_n4": [4, 2, 1],
        "incompatible_repeat_factor": 5,
        "lcm": 6,
        "mixed_role_outputs": [1.5, 3.0],
        "reciprocal_floor_boundary": [6, 3, 2],
        "reciprocal_floor_first_order": ["5", "4"],
        "repeat_factors": [2, 1, 2],
        "sd15_working_memory_bytes": 18119393.28,
        "serial_hook_forward_orders": [["0"], ["1"]],
    }

    resize_carrier = _carrier(({"mask": torch.tensor([[0.0, 1.0], [0.5, 0.25]])},))
    resized = materialize_regions(resize_carrier, "dinkster.sd15", 3, 5, "cpu")[0]
    assert resized.mask is not None
    assert torch.equal(resized.mask, _tensor(GOLDENS["mask_resize"]))

    aabb = torch.zeros((2, 12, 13), dtype=torch.float32)
    aabb[0, 4:6, 7:10] = -1.0
    bounded = materialize_regions(
        _carrier(({"mask": aabb, "set_area_to_bounds": True},)),
        "dinkster.sd15",
        12,
        13,
        "cpu",
    )[0]
    assert bounded.area == tuple(GOLDENS["mask_aabb_area"])
    zero = materialize_regions(
        _carrier(({"mask": torch.zeros((1, 3, 4)), "set_area_to_bounds": True},)),
        "dinkster.sd15",
        3,
        4,
        "cpu",
    )[0]
    assert zero.area == tuple(GOLDENS["mask_aabb_zero_area"])


@pytest.mark.parametrize("with_mask", (False, True))
def test_temporal_area_refuses_in_2d_regional_path(with_mask: bool) -> None:
    area = AreaDescriptor(
        0.25,
        0.5,
        0.0,
        0.125,
        AreaUnits.PERCENT,
        temporal=0.75,
        z=0.25,
    )
    options: dict[str, object] = {"area": area}
    if with_mask:
        options.update(mask=torch.ones((4, 4)), set_area_to_bounds=True)
    with pytest.raises(RegionalConditioningError) as captured:
        materialize_regions(_carrier((options,)), "dinkster.sd15", 4, 4, "cpu")
    assert captured.value.code == "temporal-area-unsupported"
    assert "supports only 2D latent areas" in str(captured.value)


def test_reference_mask_and_feather_multipliers_are_exact() -> None:
    x = torch.zeros((1, 1, 12, 12), dtype=torch.float32)
    area = AreaDescriptor(8, 8, 2, 2, AreaUnits.LATENT_CELLS, 0.75)
    region = materialize_regions(_carrier(({"area": area},)), "dinkster.sd15", 12, 12, "cpu")
    seen: list[torch.Tensor] = []
    crop, multiplier, _ = regional_module._crop_and_multiplier(region[0], x)
    assert crop._base is x
    assert torch.equal(multiplier, _tensor(GOLDENS["no_mask_multiplier"]))

    small_flush = materialize_regions(
        _carrier(({"area": AreaDescriptor(3, 4, 0, 8, AreaUnits.LATENT_CELLS, 0.5)},)),
        "dinkster.sd15",
        12,
        12,
        "cpu",
    )
    _, small_multiplier, _ = regional_module._crop_and_multiplier(small_flush[0], x)
    assert torch.equal(small_multiplier, _tensor(GOLDENS["small_flush_multiplier"]))

    def capture(_region: MaterializedRegion, crop: torch.Tensor, _sigma: float) -> torch.Tensor:
        seen.append(crop)
        return torch.ones_like(crop)

    actual = evaluate_regions(region, x, 0.5, FLOW, capture, lambda: False)
    expected_mult = _tensor(GOLDENS["no_mask_multiplier"])
    assert torch.equal(actual[:, :, 2:10, 2:10], torch.ones_like(expected_mult))
    assert seen[0]._base is x

    mask = torch.linspace(0.0, 1.0, 144).reshape(1, 12, 12)
    masked = materialize_regions(
        _carrier(
            (
                {
                    "area": AreaDescriptor(6, 7, 3, 2, AreaUnits.LATENT_CELLS, 0.8),
                    "mask": mask,
                    "mask_strength": 0.5,
                },
            )
        ),
        "dinkster.sd15",
        12,
        12,
        "cpu",
    )
    _, multiplier, _ = regional_module._crop_and_multiplier(masked[0], x)
    expected = _tensor(GOLDENS["mask_multiplier"])
    assert torch.equal(multiplier, expected)
    weighted = evaluate_regions(
        masked,
        x,
        0.5,
        FLOW,
        lambda _region, crop, _sigma: torch.ones_like(crop),
        lambda: False,
    )
    # A single active condition normalizes every covered nonzero mask cell to one.
    assert torch.equal(
        weighted[:, :, 3:9, 2:9][expected != 0], torch.ones_like(expected[expected != 0])
    )


def test_overlap_order_normalization_inactive_empty_and_uncovered_match_golden() -> None:
    carrier = _carrier(
        (
            {"area": AreaDescriptor(8, 8, 0, 0, AreaUnits.LATENT_CELLS)},
            {"area": AreaDescriptor(8, 8, 4, 4, AreaUnits.LATENT_CELLS)},
            {
                "schedule": PercentRange(0.0, 0.25),
                "area": AreaDescriptor(2, 2, 10, 0, AreaUnits.LATENT_CELLS),
            },
        )
    )
    regions = materialize_regions(carrier, "dinkster.sd15", 12, 12, "cpu")
    order: list[int] = []

    def evaluate(region: MaterializedRegion, crop: torch.Tensor, _sigma: float) -> torch.Tensor:
        index = next(i for i, candidate in enumerate(regions) if candidate is region)
        order.append(index)
        return torch.full_like(crop, 1.0 if index == 0 else 3.0)

    actual = evaluate_regions(
        regions, torch.zeros(1, 1, 12, 12), 0.5, FLOW, evaluate, lambda: False
    )
    assert order == [1, 0]
    assert GOLDENS["overlap"]["order"] == ["second", "first"]
    assert torch.equal(actual, _tensor(GOLDENS["overlap"]["output"]))
    assert GOLDENS["inactive_is_none"] is True
    assert GOLDENS["uncovered_is_zero"] is True
    assert torch.count_nonzero(actual[:, :, :4, 8:]) == 0
    assert torch.equal(
        evaluate_regions((), torch.ones(1, 1, 2, 2), 0.5, FLOW, evaluate, lambda: False),
        torch.zeros(1, 1, 2, 2),
    )
    empty = replace(regions[0], schedule=EMPTY_RANGE)
    assert (
        torch.count_nonzero(
            evaluate_regions((empty,), torch.ones(1, 1, 12, 12), 0.5, FLOW, evaluate, lambda: False)
        )
        == 0
    )


def test_pinned_concat_groups_preserve_group_order_and_float32_accumulation() -> None:
    # B2.1 pins _calc_cond_batch's sufficient-memory/max-concat path. Its
    # one-callback-per-region seam deliberately does not model memory-fit
    # subgroup splitting or a grouped forward; those remain B2.3.
    x = torch.zeros(1, 1, 12, 12)
    ones = torch.ones(1, 12, 12)
    heterogeneous = materialize_regions(
        _carrier(
            (
                {
                    "area": AreaDescriptor(10, 10, 0, 0, AreaUnits.LATENT_CELLS),
                    "mask": ones,
                },
                {
                    "area": AreaDescriptor(8, 8, 1, 1, AreaUnits.LATENT_CELLS),
                    "mask": ones,
                },
                {
                    "area": AreaDescriptor(6, 6, 2, 2, AreaUnits.LATENT_CELLS),
                    "mask": ones,
                },
            )
        ),
        "dinkster.sd15",
        12,
        12,
        "cpu",
    )
    heterogeneous_order: list[int] = []

    def heterogeneous_callback(
        region: MaterializedRegion, crop: torch.Tensor, _sigma: float
    ) -> torch.Tensor:
        index = next(i for i, candidate in enumerate(heterogeneous) if candidate is region)
        heterogeneous_order.append(index)
        return torch.full_like(crop, (1.0, 2.0, 4.0)[index])

    heterogeneous_output = evaluate_regions(
        heterogeneous, x, 0.5, FLOW, heterogeneous_callback, lambda: False
    )
    assert heterogeneous_order == [0, 1, 2]
    assert GOLDENS["heterogeneous_groups"]["order"] == [
        "heterogeneous-0",
        "heterogeneous-1",
        "heterogeneous-2",
    ]
    assert torch.equal(heterogeneous_output, _tensor(GOLDENS["heterogeneous_groups"]["output"]))

    mixed_text = (
        torch.zeros(1, 3, 4),
        torch.zeros(1, 15, 4),
        torch.zeros(1, 6, 4),
    )
    mixed = materialize_regions(
        _carrier(tuple({"text": text, "mask": ones} for text in mixed_text)),
        "dinkster.sd15",
        12,
        12,
        "cpu",
    )
    mixed_order: list[int] = []

    def mixed_callback(
        region: MaterializedRegion, crop: torch.Tensor, _sigma: float
    ) -> torch.Tensor:
        index = next(i for i, candidate in enumerate(mixed) if candidate is region)
        mixed_order.append(index)
        return torch.full_like(crop, (-1e20, 3.0, 1e20)[index])

    mixed_output = evaluate_regions(mixed, x, 0.5, FLOW, mixed_callback, lambda: False)
    assert mixed_order == [2, 0, 1]
    assert GOLDENS["mixed_groups"]["order"] == ["A1", "A0", "B"]
    assert torch.equal(mixed_output, _tensor(GOLDENS["mixed_groups"]["output"]))
    assert torch.equal(mixed_output, torch.ones_like(mixed_output))

    pooled_presence = materialize_regions(
        _carrier(
            (
                {"mask": ones},
                {"mask": ones, "pooled": torch.zeros(1, 2)},
                {"mask": ones},
            )
        ),
        "dinkster.sd15",
        12,
        12,
        "cpu",
    )
    pooled_order: list[int] = []

    def pooled_callback(
        region: MaterializedRegion, crop: torch.Tensor, _sigma: float
    ) -> torch.Tensor:
        pooled_order.append(
            next(i for i, candidate in enumerate(pooled_presence) if candidate is region)
        )
        return torch.zeros_like(crop)

    evaluate_regions(pooled_presence, x, 0.5, FLOW, pooled_callback, lambda: False)
    assert pooled_order == [2, 0, 1]

    pooled_shapes = materialize_regions(
        _carrier(
            (
                {"mask": ones, "pooled": torch.zeros(1, 2)},
                {"mask": ones, "pooled": torch.zeros(1, 3)},
            )
        ),
        "dinkster.sd15",
        12,
        12,
        "cpu",
    )
    pooled_shape_order: list[int] = []

    def pooled_shape_callback(
        region: MaterializedRegion, crop: torch.Tensor, _sigma: float
    ) -> torch.Tensor:
        pooled_shape_order.append(
            next(i for i, candidate in enumerate(pooled_shapes) if candidate is region)
        )
        return torch.zeros_like(crop)

    evaluate_regions(pooled_shapes, x, 0.5, FLOW, pooled_shape_callback, lambda: False)
    assert pooled_shape_order == [0, 1]


def test_materialization_interpolates_only_once_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    interpolate = torch.nn.functional.interpolate

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return interpolate(*args, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]

    monkeypatch.setattr(torch.nn.functional, "interpolate", counted)
    carrier = _carrier(
        (
            {
                "mask": torch.ones(2, 2),
                "schedule": PercentRange(0.25, 0.75),
                "extension_metadata": (("pack/key", "value"),),
            },
        )
    )
    regions = materialize_regions(carrier, "dinkster.sd15", 4, 4, "cpu")
    assert calls == 1
    assert regions[0].schedule == PercentRange(0.25, 0.75)
    assert regions[0].extension_metadata == (("pack/key", "value"),)
    evaluate_regions(
        regions,
        torch.zeros(1, 1, 4, 4),
        0.5,
        FLOW,
        lambda _region, crop, _sigma: torch.ones_like(crop),
        lambda: False,
    )
    assert calls == 1


def test_percent_range_consumes_the_engine_realized_step_row() -> None:
    regions = materialize_regions(
        _carrier(({"schedule": PercentRange(0.25, 0.75)},)),
        "dinkster.sd15",
        4,
        4,
        "cpu",
    )
    sigmas = tuple(FLOW.percent_to_sigma(percent) for percent in (0.0, 0.5, 1.0))
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule("sage", 0.0, 1.0),
        sigmas,
    )
    realized = realize_region_schedules(regions, timeline, FLOW)
    calls: list[int] = []

    with use_realized_sampling_timeline(timeline) as activate:
        for row in timeline.rows:
            activate(row.anchors.step_index)
            step_index = row.anchors.step_index

            def capture(
                _region: object,
                crop: torch.Tensor,
                _sigma: float,
                step_index: int = step_index,
            ) -> torch.Tensor:
                calls.append(step_index)
                return torch.ones_like(crop)

            evaluate_regions(
                realized,
                torch.zeros(1, 1, 4, 4),
                row.anchors.sigma,
                FLOW,
                capture,
                lambda: False,
            )

    assert calls == [
        row.anchors.step_index
        for row in timeline.rows
        if regions[0].schedule.is_active(row.anchors.sigma, FLOW)
    ]


def test_percent_range_holds_outer_step_decision_for_intermediate_sigma() -> None:
    regions = materialize_regions(
        _carrier(({"schedule": PercentRange(0.0, 0.25)},)),
        "dinkster.sd15",
        4,
        4,
        "cpu",
    )
    sigmas = tuple(FLOW.percent_to_sigma(percent) for percent in (0.0, 0.5, 1.0))
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule("sage", 0.0, 1.0),
        sigmas,
    )
    realized = realize_region_schedules(regions, timeline, FLOW)
    calls: list[float] = []

    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        evaluate_regions(
            realized,
            torch.zeros(1, 1, 4, 4),
            timeline.rows[1].anchors.sigma,
            FLOW,
            lambda _region, crop, sigma: calls.append(sigma) or torch.ones_like(crop),
            lambda: False,
        )

    assert calls == [timeline.rows[1].anchors.sigma]


@pytest.mark.parametrize(
    ("change", "code"),
    (
        ({"layout": None}, "missing-token-layout"),
        (
            {
                "layout": TokenLayoutDescriptor(
                    "dinkster.other",
                    1,
                    ("clip_l",),
                    (TokenSegmentDescriptor("clip_l", "clip_l", 0, 3),),
                ),
            },
            "unsupported-token-layout",
        ),
        (
            {
                "layout": TokenLayoutDescriptor(
                    "dinkster.sd15",
                    1,
                    ("clip_l",),
                    (TokenSegmentDescriptor("clip_l", "clip_l", 2, 2),),
                ),
            },
            "unsupported-token-layout",
        ),
        (
            {
                "layout": TokenLayoutDescriptor(
                    "dinkster.sd15",
                    1,
                    ("clip_l",),
                    (TokenSegmentDescriptor("empty", "clip_l", 3, None),),
                ),
            },
            "unsupported-token-layout",
        ),
        ({"mask": torch.ones(1, 2, 2, 2)}, "mask-shape"),
        ({"mask": torch.ones(2, 2, dtype=torch.float16)}, "mask-dtype"),
        ({"mask": torch.ones(2, 2), "mask_space": "wrong"}, "mask-space"),
    ),
)
def test_strict_family_layout_and_mask_refusals(change: dict[str, object], code: str) -> None:
    family = cast(str, change.pop("family", "dinkster.sd15"))
    with pytest.raises(RegionalConditioningError) as captured:
        materialize_regions(_carrier((change,)), family, 4, 4, "cpu")
    assert captured.value.code == code


def test_open_ended_token_segment_extends_to_text_end() -> None:
    layout = TokenLayoutDescriptor(
        "dinkster.sd15",
        1,
        ("clip_l",),
        (TokenSegmentDescriptor("prompt", "clip_l", 1, None),),
    )

    regions = materialize_regions(
        _carrier(({"layout": layout},)),
        "dinkster.sd15",
        4,
        4,
        "cpu",
    )

    assert len(regions) == 1


@pytest.mark.parametrize(
    ("family", "streams", "pooled"),
    (
        ("dinkster.chroma", ("t5",), False),
        ("dinkster.flux_dev", ("clip_l", "t5"), True),
        ("dinkster.flux_schnell", ("clip_l", "t5"), True),
        ("dinkster.flux_schnell", ("ovis_qwen3_2b",), False),
        ("dinkster.qwen_image", ("qwen2_5_vl_7b",), False),
        ("dinkster.sd15", ("clip_l",), False),
        ("dinkster.sd15", ("clip_l",), True),
        ("dinkster.sdxl", ("clip_l", "clip_g"), True),
        ("dinkster.sdxl_refiner", ("clip_g",), True),
        ("dinkster.wan21", ("umt5xxl",), False),
    ),
)
def test_family_descriptor_layouts_materialize_exactly(
    family: str, streams: tuple[str, ...], pooled: bool
) -> None:
    token_count = 3
    segment_streams = ("t5",) if streams == ("clip_l", "t5") else streams
    layout = TokenLayoutDescriptor(
        family,
        1,
        streams,
        tuple(TokenSegmentDescriptor(stream, stream, 0, token_count) for stream in segment_streams),
    )
    pooled_tensor = torch.ones(1, 2) if pooled else None
    region = materialize_regions(
        _carrier(
            ({"layout": layout, "pooled": pooled_tensor, "mask": torch.ones(2, 2)},),
            family=family,
        ),
        family,
        4,
        4,
        "cpu",
    )[0]
    assert region.conditioning.embeddings.shape == (1, 3, 4)
    assert (region.conditioning.pooled is not None) is pooled
    assert region.mask is not None and region.mask.shape == (1, 4, 4)


def test_missing_text_carried_only_condition_scale_and_malformed_shapes_refuse() -> None:
    base = _carrier(({},))
    record = base.conditioning.records[0]
    with pytest.raises(ValueError):
        replace(record, channels=())

    text_descriptor = record.channels[0][1]
    pooled_only = replace(
        record,
        channels=((ConditioningChannel.POOLED, text_descriptor),),
    )
    with pytest.raises(RegionalConditioningError, match="missing-text"):
        materialize_regions(
            replace(base, conditioning=ConditioningSet((pooled_only,))),
            "dinkster.sd15",
            4,
            4,
            "cpu",
        )
    carried = replace(
        record,
        channels=((ConditioningChannel.CONTROL_HINT, text_descriptor),),
    )
    with pytest.raises(RegionalConditioningError, match="carried-only-channel"):
        materialize_regions(
            replace(base, conditioning=ConditioningSet((carried,))), "dinkster.sd15", 4, 4, "cpu"
        )

    with pytest.raises(RegionalConditioningError, match="scale-without-patch"):
        materialize_regions(_carrier(({"scale": torch.ones(1)},)), "dinkster.sd15", 4, 4, "cpu")

    wrong_rank = _carrier(({"text": torch.ones(1, 3)},))
    with pytest.raises(RegionalConditioningError, match="payload-shape"):
        materialize_regions(wrong_rank, "dinkster.sd15", 4, 4, "cpu")
    wrong_dtype = _carrier(({"text": torch.ones(1, 3, 4, dtype=torch.int64)},))
    with pytest.raises(RegionalConditioningError, match="payload-dtype"):
        materialize_regions(wrong_dtype, "dinkster.sd15", 4, 4, "cpu")


_CALLBACK_REFUSALS: tuple[tuple[RegionEvaluator, str], ...] = (
    (lambda _r, crop, _s: crop[:, :, :-1], "callback-shape"),
    (lambda _r, crop, _s: crop.double(), "callback-dtype"),
    (lambda _r, crop, _s: torch.empty(crop.shape, device="meta"), "callback-device"),
    (lambda _r, _crop, _s: cast(torch.Tensor, "not-tensor"), "callback-type"),
)


@pytest.mark.parametrize(
    ("callback", "code"),
    _CALLBACK_REFUSALS,
)
def test_callback_contract_refuses(callback: RegionEvaluator, code: str) -> None:
    regions = materialize_regions(_carrier(({},)), "dinkster.sd15", 4, 4, "cpu")
    with pytest.raises(RegionalConditioningError) as captured:
        evaluate_regions(
            regions,
            torch.zeros(1, 1, 4, 4),
            0.5,
            FLOW,
            callback,
            lambda: False,
        )
    assert captured.value.code == code


def test_area_mask_batch_and_cancellation_boundaries_refuse_without_partial_result() -> None:
    out_of_bounds = materialize_regions(
        _carrier(({"area": AreaDescriptor(2, 2, 4, 0, AreaUnits.LATENT_CELLS)},)),
        "dinkster.sd15",
        4,
        4,
        "cpu",
    )
    with pytest.raises(RegionalConditioningError, match="area-out-of-bounds"):
        evaluate_regions(
            out_of_bounds,
            torch.zeros(1, 1, 4, 4),
            0.5,
            FLOW,
            lambda _region, crop, _sigma: crop,
            lambda: False,
        )
    mask_batch = materialize_regions(
        _carrier(({"mask": torch.ones(2, 4, 4)},)), "dinkster.sd15", 4, 4, "cpu"
    )
    with pytest.raises(RegionalConditioningError, match="mask-batch"):
        evaluate_regions(
            mask_batch,
            torch.zeros(3, 1, 4, 4),
            0.5,
            FLOW,
            lambda _region, crop, _sigma: crop,
            lambda: False,
        )
    geometry_regions = materialize_regions(_carrier(({},)), "dinkster.sd15", 4, 4, "cpu")
    callback_calls = 0

    def geometry_callback(
        _region: MaterializedRegion, crop: torch.Tensor, _sigma: float
    ) -> torch.Tensor:
        nonlocal callback_calls
        callback_calls += 1
        return crop

    with pytest.raises(RegionalConditioningError, match="materialized-shape"):
        evaluate_regions(
            geometry_regions,
            torch.zeros(1, 1, 5, 4),
            0.5,
            FLOW,
            geometry_callback,
            lambda: False,
        )
    with pytest.raises(RegionalConditioningError, match="materialized-device"):
        evaluate_regions(
            geometry_regions,
            torch.empty(1, 1, 4, 4, device="meta"),
            0.5,
            FLOW,
            geometry_callback,
            lambda: False,
        )
    with pytest.raises(RegionalConditioningError, match="latent-shape"):
        evaluate_regions(
            (),
            torch.empty(0, 1, 4, 4),
            0.5,
            FLOW,
            geometry_callback,
            lambda: False,
        )
    assert callback_calls == 0
    oversized_mask_batch = materialize_regions(
        _carrier(({"mask": torch.stack((torch.ones(4, 4), torch.zeros(4, 4)))},)),
        "dinkster.sd15",
        4,
        4,
        "cpu",
    )
    oversized = evaluate_regions(
        oversized_mask_batch,
        torch.zeros(1, 1, 4, 4),
        0.5,
        FLOW,
        lambda _region, crop, _sigma: torch.ones_like(crop),
        lambda: False,
    )
    assert torch.equal(oversized, torch.ones_like(oversized))

    calls = 0
    checks = 0

    def cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks == 3

    def callback(_region: MaterializedRegion, crop: torch.Tensor, _sigma: float) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.ones_like(crop)

    two = materialize_regions(_carrier(({}, {})), "dinkster.sd15", 4, 4, "cpu")
    with pytest.raises(RegionalConditioningError, match="cancelled"):
        evaluate_regions(two, torch.zeros(1, 1, 4, 4), 0.5, FLOW, callback, cancel)
    assert calls == 1
    assert checks == 3


def test_flux_and_sd_full_cover_factories_equal_ordinary_denoisers() -> None:
    flux_model = tiny_flux()
    flux_condition = flux_cond("regional")
    flux_carrier = _carrier(
        ({"text": flux_condition.embeddings, "pooled": flux_condition.pooled},),
        family="dinkster.flux_dev",
        tokens=flux_condition.embeddings.shape[1],
        features=flux_condition.embeddings.shape[2],
    )
    flux_regions = materialize_regions(
        flux_carrier,
        "dinkster.flux_dev",
        8,
        8,
        "cpu",
    )
    flux_x = flux_latent()
    expected_flux = FluxDenoiser(flux_model, flux_condition, compute_dtype=torch.float32)(
        flux_x, 0.5
    )
    actual_flux = evaluate_regions(
        flux_regions,
        flux_x,
        0.5,
        FLOW,
        flux_region_evaluator(flux_model, compute_dtype=torch.float32),
        lambda: False,
    )
    assert torch.equal(actual_flux, expected_flux)

    sd_model = tiny_unet()
    sd_condition = sd_cond("regional")
    sd_carrier = _carrier(
        ({"text": sd_condition.embeddings},),
        family="dinkster.sd15",
        tokens=sd_condition.embeddings.shape[1],
        features=sd_condition.embeddings.shape[2],
    )
    sd_regions = materialize_regions(sd_carrier, "dinkster.sd15", 8, 8, "cpu")
    sd_x = sd_latent()
    expected_sd = SDDenoiser(
        sd_model,
        SD15_SIGMAS,
        sd_condition,
        parameterization=Parameterization.EPS,
        compute_dtype=torch.float32,
    )(sd_x, 0.5)
    actual_sd = evaluate_regions(
        sd_regions,
        sd_x,
        0.5,
        SD15_SIGMAS,
        sd_region_evaluator(
            sd_model,
            SD15_SIGMAS,
            parameterization=Parameterization.EPS,
            compute_dtype=torch.float32,
        ),
        lambda: False,
    )
    assert torch.equal(actual_sd, expected_sd)


def test_legacy_evaluator_refuses_grouped_patch_rows_before_callback() -> None:
    calls = 0

    def callback(_region: MaterializedRegion, crop: torch.Tensor, _sigma: float) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return crop

    base = materialize_regions(_carrier(({},)), "dinkster.sd15", 4, 4, "cpu")[0]
    x = torch.zeros((1, 1, 4, 4))
    for region in (
        replace(base, patch_digest="a" * 64),
        replace(base, scale_vector=torch.ones(1), patch_digest="a" * 64),
    ):
        with pytest.raises(RegionalConditioningError, match="grouped-executor-required"):
            evaluate_regions((region,), x, 0.5, FLOW, callback, lambda: False)
    assert calls == 0


def test_tiny_sd_model_composes_two_regional_crops() -> None:
    model = tiny_unet()
    left_condition = sd_cond("regional-left")
    right_condition = sd_cond("regional-right")
    regions = materialize_regions(
        _carrier(
            (
                {
                    "text": left_condition.embeddings,
                    "area": AreaDescriptor(8, 4, 0, 0, AreaUnits.LATENT_CELLS),
                },
                {
                    "text": right_condition.embeddings,
                    "area": AreaDescriptor(8, 4, 0, 4, AreaUnits.LATENT_CELLS),
                },
            ),
            family="dinkster.sd15",
            tokens=left_condition.embeddings.shape[1],
            features=left_condition.embeddings.shape[2],
        ),
        "dinkster.sd15",
        8,
        8,
        "cpu",
    )
    x = sd_latent()
    actual = evaluate_regions(
        regions,
        x,
        0.5,
        SD15_SIGMAS,
        sd_region_evaluator(
            model,
            SD15_SIGMAS,
            parameterization=Parameterization.EPS,
            compute_dtype=torch.float32,
        ),
        lambda: False,
    )
    expected_left = SDDenoiser(
        model,
        SD15_SIGMAS,
        left_condition,
        parameterization=Parameterization.EPS,
        compute_dtype=torch.float32,
    )(x[:, :, :, :4], 0.5)
    expected_right = SDDenoiser(
        model,
        SD15_SIGMAS,
        right_condition,
        parameterization=Parameterization.EPS,
        compute_dtype=torch.float32,
    )(x[:, :, :, 4:], 0.5)
    assert torch.equal(actual[:, :, :, :4], expected_left)
    assert torch.equal(actual[:, :, :, 4:], expected_right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cuda_materialization_and_evaluation_use_checkout_source() -> None:
    source = Path(
        __import__("dinkster_inference_torch.regional", fromlist=["x"]).__file__
    ).resolve()
    assert (
        source == Path(__file__).resolve().parents[1] / "src/dinkster_inference_torch/regional.py"
    )
    carrier = _carrier(({"mask": torch.tensor([[0.0, 1.0], [1.0, 0.0]])},))
    regions = materialize_regions(carrier, "dinkster.sd15", 4, 4, "cuda:0")
    assert regions[0].conditioning.embeddings.device.type == "cuda"
    assert regions[0].mask is not None and regions[0].mask.device.type == "cuda"
    x = torch.zeros((1, 2, 4, 4), device="cuda")
    output = evaluate_regions(
        regions,
        x,
        0.5,
        FLOW,
        lambda _region, crop, _sigma: torch.ones_like(crop),
        lambda: False,
    )
    assert output.device.type == "cuda"
    assert torch.isfinite(output).all()
