# pyright: basic
"""Declared attention points execute in model forwards without model mutation."""

from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from dinkster_inference import (
    AttentionBackendDescriptor,
    AttentionContribution,
    AttentionOutputDescriptor,
    AttentionQKVDescriptor,
    AttentionSelector,
    AttentionWrapperDescriptor,
    BlockInjectionDescriptor,
    CancellationToken,
    Conditioning,
    FluxConfig,
    GuidanceCondition,
    GuidanceContractError,
    GuidanceExtensionError,
    GuidancePlanContext,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceRole,
    ProgressScope,
    SamplingCancelled,
    SamplingExecutionContext,
    UNetConfig,
)
from dinkster_inference_torch.attention_extensions import AttentionExecution, AttentionRegistry
from dinkster_inference_torch.flux import Flux
from dinkster_inference_torch.guidance import GuidanceExecutor, GuidanceRegistry
from dinkster_inference_torch.unet import UNetModel
from unet_fill import fill_state_dict, hashed_input


def contribution(**kwargs):
    return AttentionContribution(torch_version="2.13.0+cpu", aimdo_version="0.5.5.post2", **kwargs)


def execution(contributions, *, batch=1, cancelled=lambda: False):
    token = CancellationToken(cancelled)
    context = SamplingExecutionContext(
        (1.0, 0.0),
        0,
        0,
        1.0,
        23,
        token,
        ProgressScope(token),
        {owner: {} for owner, _ in contributions},
    )
    return AttentionExecution(
        AttentionRegistry(contributions),
        context,
        (
            GuidanceCondition("blue", GuidanceRole.CONDITIONAL, Conditioning(torch.ones(1), None)),
            GuidanceCondition(
                "empty", GuidanceRole.UNCONDITIONAL, Conditioning(torch.zeros(1), None)
            ),
        ),
        batch,
    )


def proof_pack(directory):
    path = Path(__file__).parents[3] / "tests" / "fixtures" / directory
    module = next(item for item in path.glob("*.py") if item.name != "__init__.py")
    return runpy.run_path(str(module))["make"]


def test_proof_runner_converts_chw_tensor_to_rgb_image():
    runner = Path(__file__).parents[3] / "tools" / "run_attention_pack_proofs.py"
    as_pil = runpy.run_path(str(runner))["as_pil"]
    image = torch.tensor([[[[0.0, 1.0]], [[0.5, 0.25]], [[1.0, 0.0]]]])

    converted = as_pil(image)

    assert converted.size == (2, 1)
    assert list(converted.getdata()) == [(0, 128, 255), (255, 64, 0)]


@pytest.mark.parametrize(
    ("family", "block", "kind"),
    (
        ("unet", "middle_block.1.transformer_blocks.0", "self"),
        ("flux", "double_blocks.0", "joint"),
    ),
)
@pytest.mark.parametrize("scale", (0.0, 0.5, 2.0))
def test_pag_proof_pack_uses_auxiliary_attention_and_scales_monotonically(
    family,
    block,
    kind,
    scale,
):
    contribution = proof_pack("pag-pack")(
        scale=scale,
        torch_version="2.13.0+cpu",
        aimdo_version="0.5.5.post2",
    )
    registry = GuidanceRegistry(
        (("proof_pag", contribution.guidance),),
        attention_contributions=(("proof_pag", contribution.attention),),
    )
    executor = GuidanceExecutor(registry)
    token = CancellationToken(lambda: False)
    sampling = SamplingExecutionContext(
        (1.0, 0.0),
        0,
        0,
        1.0,
        23,
        token,
        ProgressScope(token),
        {"proof_pag": {}},
    )
    lanes = (
        GuidanceCondition("positive", GuidanceRole.CONDITIONAL, Conditioning(torch.ones(1))),
        GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, Conditioning(torch.zeros(1))),
    )
    context = GuidancePlanContext(
        torch.zeros(1, 2),
        torch.tensor(1.0),
        2.0,
        lanes,
        False,
        sampling,
    )
    evaluations = 0

    def evaluate(request):
        nonlocal evaluations
        evaluations += 1
        lane_values = {"positive": (4.0, 2.0), "negative": (1.0, 0.0)}
        q = torch.tensor(
            [
                [[[lane_values[lane.id][0]], [lane_values[lane.id][0]]]]
                for lane in request.plan.lanes
            ]
        )
        v = torch.tensor(
            [
                [[[lane_values[lane.id][1]], [lane_values[lane.id][1]]]]
                for lane in request.plan.lanes
            ]
        )
        active = AttentionExecution(
            registry.attention_extensions,
            request.execution,
            request.plan.lanes,
            1,
        )
        call = active.context(
            family=family,
            block=block,
            kind=kind,
            heads=1,
            spatial_shape=(1, 2),
            query_tokens=2,
            key_tokens=2,
            text_tokens=0 if family == "unet" else 1,
        )
        output = active.attention(q, q, v, lambda query, _key, _value: query, call)
        return GuidancePredictions(
            tuple(
                GuidancePrediction(
                    lane.id,
                    torch.full_like(request.input, float(output[index].mean())),
                    GuidancePredictionSource.MODEL,
                )
                for index, lane in enumerate(request.plan.lanes)
            )
        )

    result = executor.execute(context, evaluate).denoised
    assert torch.equal(result, torch.full_like(result, 7.0 + 2.0 * scale))
    assert evaluations == (1 if scale == 0 else 2)


def test_flux_pag_uses_declared_block_and_scales_real_model_effect_monotonically():
    case = json.loads((Path(__file__).parent / "goldens" / "flux_goldens.json").read_text())[
        "cases"
    ]["dev_guidance"]
    config = dict(case["config"])
    config["axes_dim"] = tuple(config["axes_dim"])
    model = Flux(FluxConfig(**config))
    model.load_state_dict(fill_state_dict(case["state_dict"]), strict=True)
    x = hashed_input("pag-flux-x", (1, config["in_channels"], 4, 6))
    timestep = torch.tensor([0.5])
    guidance = torch.tensor([3.5])
    conditions = (
        GuidanceCondition(
            "positive",
            GuidanceRole.CONDITIONAL,
            Conditioning(
                hashed_input("pag-flux-positive", (1, 5, config["context_in_dim"])),
                hashed_input("pag-flux-positive-y", (1, config["vec_in_dim"])),
            ),
        ),
        GuidanceCondition(
            "negative",
            GuidanceRole.UNCONDITIONAL,
            Conditioning(
                hashed_input("pag-flux-negative", (1, 5, config["context_in_dim"])),
                hashed_input("pag-flux-negative-y", (1, config["vec_in_dim"])),
            ),
        ),
    )
    effects = []
    evaluations = []
    for scale in (0.0, 0.5, 2.0):
        contribution = proof_pack("pag-pack")(
            scale=scale,
            torch_version="2.13.0+cpu",
            aimdo_version="0.5.5.post2",
        )
        registry = GuidanceRegistry(
            (("proof_pag", contribution.guidance),),
            attention_contributions=(("proof_pag", contribution.attention),),
        )
        executor = GuidanceExecutor(registry)
        token = CancellationToken(lambda: False)
        sampling = SamplingExecutionContext(
            (1.0, 0.0),
            0,
            0,
            1.0,
            23,
            token,
            ProgressScope(token),
            {"proof_pag": {}},
        )
        context = GuidancePlanContext(
            x,
            timestep,
            2.0,
            conditions,
            False,
            sampling,
        )
        calls = 0

        def evaluate(request, registry=registry, sampling=sampling):
            nonlocal calls
            calls += 1
            lanes = request.plan.lanes
            active = AttentionExecution(registry.attention_extensions, sampling, lanes, 1)
            with torch.no_grad():
                values = model(
                    request.input.expand(len(lanes), -1, -1, -1),
                    request.sigma.expand(len(lanes)),
                    torch.cat([lane.conditioning.embeddings for lane in lanes]),
                    y=torch.cat([lane.conditioning.pooled for lane in lanes]),
                    guidance=guidance.expand(len(lanes)),
                    attention_extensions=active,
                )
            return GuidancePredictions(
                tuple(
                    GuidancePrediction(
                        lane.id,
                        values[index : index + 1],
                        GuidancePredictionSource.MODEL,
                    )
                    for index, lane in enumerate(lanes)
                )
            )

        result = executor.execute(context, evaluate).denoised
        baseline = GuidanceExecutor(GuidanceRegistry()).execute(context, evaluate).denoised
        if scale == 0:
            assert torch.equal(result, baseline)
        effects.append(float((result - baseline).square().mean().sqrt()))
        evaluations.append(calls)

    assert effects[0] == 0
    assert 0 < effects[1] < effects[2]
    assert evaluations == [2, 3, 3]


@pytest.mark.parametrize(
    ("family", "kind", "text_tokens"),
    (("unet", "cross", 0), ("flux", "joint", 1)),
)
@pytest.mark.parametrize("lane_ids", (("left", "right"), ("positive", "negative")))
def test_attention_couple_proof_pack_blends_asymmetric_regions(
    family,
    kind,
    text_tokens,
    lane_ids,
):
    contribution = proof_pack("attention-couple-pack")(
        split=0.5,
        torch_version="2.13.0+cpu",
        aimdo_version="0.5.5.post2",
    ).attention
    registry = AttentionRegistry((("proof_couple", contribution),))
    token = CancellationToken(lambda: False)
    sampling = SamplingExecutionContext(
        (1.0, 0.0),
        0,
        0,
        1.0,
        23,
        token,
        ProgressScope(token),
        {"proof_couple": {}},
    )
    lanes = (
        GuidanceCondition(lane_ids[0], GuidanceRole.CONDITIONAL, Conditioning(torch.ones(1))),
        GuidanceCondition(lane_ids[1], GuidanceRole.CONDITIONAL, Conditioning(torch.ones(1))),
    )
    active = AttentionExecution(registry, sampling, lanes, 1)
    tokens = 4 + text_tokens
    left = [10.0] * text_tokens + [1.0] * 4
    right = [20.0] * text_tokens + [3.0] * 4
    q = torch.tensor([[left], [right]]).unsqueeze(-1)
    call = active.context(
        family=family,
        block="middle_block.1.transformer_blocks.0" if family == "unet" else "double_blocks.0",
        kind=kind,
        heads=1,
        spatial_shape=(1, 4),
        query_tokens=tokens,
        key_tokens=tokens,
        text_tokens=text_tokens,
    )
    output = active.attention(q, q, q, lambda query, _key, _value: query, call)
    expected_left = [10.0] * text_tokens + [1.0, 1.0, 3.0, 3.0]
    expected_right = [20.0] * text_tokens + [1.0, 1.0, 3.0, 3.0]
    assert torch.equal(output, torch.tensor([[expected_left], [expected_right]]).unsqueeze(-1))


def test_attention_couple_proof_pack_reduces_prompt_lanes_by_region():
    contribution = proof_pack("attention-couple-pack")(
        split=0.5,
        torch_version="2.13.0+cpu",
        aimdo_version="0.5.5.post2",
    )
    registry = GuidanceRegistry((("proof_couple", contribution.guidance),))
    executor = GuidanceExecutor(registry)
    token = CancellationToken(lambda: False)
    sampling = SamplingExecutionContext(
        (1.0, 0.0),
        0,
        0,
        1.0,
        23,
        token,
        ProgressScope(token),
        {"proof_couple": {}},
    )
    lanes = (
        GuidanceCondition("positive", GuidanceRole.CONDITIONAL, Conditioning(torch.ones(1))),
        GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, Conditioning(torch.zeros(1))),
    )
    context = GuidancePlanContext(
        torch.zeros(1, 1, 2, 4),
        torch.tensor(1.0),
        1.0,
        lanes,
        False,
        sampling,
    )

    def evaluate(request):
        values = {"positive": 1.0, "negative": 3.0}
        return GuidancePredictions(
            tuple(
                GuidancePrediction(
                    lane.id,
                    torch.full_like(request.input, values[lane.id]),
                    GuidancePredictionSource.MODEL,
                )
                for lane in request.plan.lanes
            )
        )

    result = executor.execute(context, evaluate).denoised
    expected = torch.tensor([[[[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 3.0, 3.0]]]])
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    ("family", "kind", "text_tokens"),
    (("unet", "self", 0), ("flux", "joint", 1)),
)
def test_reference_attention_proof_pack_captures_then_injects_invocation_state(
    family,
    kind,
    text_tokens,
):
    contribution = proof_pack("reference-attention-pack")(
        strength=0.5,
        torch_version="2.13.0+cpu",
        aimdo_version="0.5.5.post2",
    ).attention
    registry = AttentionRegistry((("proof_reference", contribution),))
    token = CancellationToken(lambda: False)
    sampling = SamplingExecutionContext(
        (1.0, 0.0),
        0,
        0,
        1.0,
        23,
        token,
        ProgressScope(token),
        {"proof_reference": {}},
    )
    lanes = (GuidanceCondition("positive", GuidanceRole.CONDITIONAL, Conditioning(torch.ones(1))),)
    active = AttentionExecution(registry, sampling, lanes, 1)
    tokens = 4 + text_tokens
    call = active.context(
        family=family,
        block="middle_block.1.transformer_blocks.0" if family == "unet" else "double_blocks.0",
        kind=kind,
        heads=1,
        spatial_shape=(1, 4),
        query_tokens=tokens,
        key_tokens=tokens,
        text_tokens=text_tokens,
    )
    q = torch.zeros(1, 1, tokens, 1)
    reference = torch.full_like(q, 4.0)
    active.attention(q, reference, reference, lambda _q, _k, value: value, call)
    output = active.attention(q, q, q, lambda _q, _k, value: value, call)
    expected = torch.zeros_like(output)
    expected[:, :, text_tokens:] = 2.0
    assert torch.equal(output, expected)

    reference_state = sampling.extension_state["proof_reference"]["reference"]
    assert isinstance(reference_state, dict)
    captured = dict(reference_state)
    configured = proof_pack("reference-attention-pack")(
        strength=0.5,
        reference_state=captured,
        torch_version="2.13.0+cpu",
        aimdo_version="0.5.5.post2",
    ).attention
    captured.clear()
    configured_sampling = replace(
        sampling,
        extension_state={"proof_reference": {}},
    )
    configured_active = AttentionExecution(
        AttentionRegistry((("proof_reference", configured),)),
        configured_sampling,
        lanes,
        1,
    )
    configured_call = configured_active.context(
        family=family,
        block="middle_block.1.transformer_blocks.0" if family == "unet" else "double_blocks.0",
        kind=kind,
        heads=1,
        spatial_shape=(1, 4),
        query_tokens=tokens,
        key_tokens=tokens,
        text_tokens=text_tokens,
    )
    configured_output = configured_active.attention(
        q,
        q,
        q,
        lambda _q, _k, value: value,
        configured_call,
    )
    assert torch.equal(configured_output, expected)


def test_ordered_qkv_wrappers_outputs_and_condition_spans():
    events = []
    selector = AttentionSelector("unet", "middle", "cross")

    def qkv(q, k, v, context):
        events.append("qkv")
        assert [
            (s.axis, s.start, s.end, s.condition_id, s.batch_start, s.batch_end, s.stream)
            for s in context.spans
        ] == [
            ("query", 0, 3, "blue", 0, 2, "image"),
            ("key", 0, 5, "blue", 0, 2, "text"),
            ("query", 0, 3, "empty", 2, 4, "image"),
            ("key", 0, 5, "empty", 2, 4, "text"),
        ]
        context.state["value"] = 7
        return q + 2, k, v

    def wrap(q, k, v, context, next):
        events.append("enter")
        result = next(q, k, v)
        events.append("exit")
        return result * 3

    def backend(q, k, v, context):
        events.append("backend")
        return q + v.mean(2, keepdim=True)

    def out_a(output, context):
        events.append("a")
        assert context.state["value"] == 7
        return output + 5

    def out_b(output, context):
        events.append("b")
        assert "value" not in context.state
        return output * 2

    run = execution(
        (
            (
                "a",
                contribution(
                    qkv=(AttentionQKVDescriptor("a.qkv", selector, qkv),),
                    wrappers=(AttentionWrapperDescriptor("a.wrap", selector, wrap),),
                    outputs=(AttentionOutputDescriptor("a.output", selector, out_a, order=9),),
                    backends=(AttentionBackendDescriptor("a.backend", "unet", backend),),
                ),
            ),
            (
                "b",
                contribution(
                    outputs=(AttentionOutputDescriptor("b.output", selector, out_b, order=-2),)
                ),
            ),
        ),
        batch=2,
    )
    q = torch.arange(24, dtype=torch.float32).reshape(4, 1, 3, 2)
    k = torch.zeros(4, 1, 5, 2)
    v = torch.ones_like(k)
    context = run.context(
        family="unet",
        block="middle",
        kind="cross",
        heads=1,
        spatial_shape=(1, 3),
        query_tokens=3,
        key_tokens=5,
    )
    got = run.attention(q, k, v, lambda q, k, v: q + v.mean(2, keepdim=True), context)
    assert torch.equal(got, ((q + 3) * 3) * 2 + 5)
    assert events == ["qkv", "enter", "backend", "exit", "b", "a"]


def test_joint_attention_reports_each_reference_separately():
    run = replace(execution(()), lanes=execution(()).lanes[:1])
    context = run.context(
        family="flux",
        block="double_blocks.0",
        kind="joint",
        heads=2,
        spatial_shape=(2, 3),
        query_tokens=18,
        key_tokens=18,
        text_tokens=3,
        reference_tokens=(4, 5),
    )
    assert [
        (s.start, s.end, s.stream, s.condition_id) for s in context.spans if s.axis == "key"
    ] == [
        (0, 3, "text", "blue"),
        (3, 9, "image", "blue"),
        (9, 13, "reference", "blue:reference:0"),
        (13, 18, "reference", "blue:reference:1"),
    ]


def test_attention_next_cannot_escape_callback_and_preserves_cancellation():
    captured = []

    def wrapper(q, k, v, context, next):
        captured.append(next)
        return next(q, k, v)

    run = execution(
        (
            (
                "proof",
                contribution(
                    wrappers=(
                        AttentionWrapperDescriptor(
                            "proof.wrapper", AttentionSelector("flux"), wrapper
                        ),
                    )
                ),
            ),
        )
    )
    q = torch.ones(2, 1, 2, 3)
    context = run.context(
        family="flux",
        block="double_blocks.0",
        kind="joint",
        heads=1,
        spatial_shape=(1, 1),
        query_tokens=2,
        key_tokens=2,
        text_tokens=1,
    )
    run.attention(q, q, q, lambda q, k, v: v, context)
    with pytest.raises(GuidanceContractError, match="outside its callback"):
        captured[0](q, q, q)

    def cancelled_kernel(q, k, v):
        raise SamplingCancelled("explicit cancellation")

    with pytest.raises(SamplingCancelled, match="explicit cancellation"):
        run.attention(q, q, q, cancelled_kernel, context)


@pytest.mark.parametrize("terminal,calls", [(False, 0), (False, 2), (True, 2)])
def test_wrapper_must_honor_declared_termination(terminal, calls):
    def wrapper(q, k, v, context, next):
        for _ in range(calls):
            next(q, k, v)
        return v

    run = execution(
        (
            (
                "proof",
                contribution(
                    wrappers=(
                        AttentionWrapperDescriptor(
                            "proof.wrapper",
                            AttentionSelector("flux"),
                            wrapper,
                            terminal=terminal,
                        ),
                    )
                ),
            ),
        )
    )
    q = torch.ones(2, 1, 2, 3)
    context = run.context(
        family="flux",
        block="double_blocks.0",
        kind="joint",
        heads=1,
        spatial_shape=(1, 1),
        query_tokens=2,
        key_tokens=2,
        text_tokens=1,
    )
    with pytest.raises((GuidanceContractError, GuidanceExtensionError), match="next contract"):
        run.attention(q, q, q, lambda q, k, v: v, context)


@pytest.mark.parametrize("family", ["unet", "flux"])
def test_real_model_point_changes_output_and_unload_restores_baseline(family):
    fixtures = Path(__file__).parent / "goldens"
    data = json.loads((fixtures / f"{family}_goldens.json").read_text())
    name = "sd1_conv" if family == "unet" else "dev_guidance"
    case = data["cases"][name]
    config = dict(case["config"])
    for field in (
        "num_res_blocks",
        "channel_mult",
        "transformer_depth",
        "transformer_depth_output",
        "axes_dim",
    ):
        if field in config:
            config[field] = tuple(config[field])
    model = UNetModel(UNetConfig(**config)) if family == "unet" else Flux(FluxConfig(**config))
    model.load_state_dict(fill_state_dict(case["state_dict"]), strict=True)
    events = []
    selector = AttentionSelector(family)

    def output(output, context):
        events.append((context.block, context.kind))
        return output * 0.7

    def block(value, context):
        events.append((context.block, "block"))
        return value + 0.125

    run = execution(
        (
            (
                "proof",
                contribution(
                    outputs=(AttentionOutputDescriptor("proof.output", selector, output),),
                    blocks=(BlockInjectionDescriptor("proof.block", selector, block),),
                ),
            ),
        )
    )
    # One actual conditional row, with deliberately non-square spatial geometry.
    run = replace(run, lanes=run.lanes[:1])
    x = hashed_input(f"{name}:extension-x", (1, config["in_channels"], 4, 6))
    context_dim = config.get("context_dim", config.get("context_in_dim"))
    assert isinstance(context_dim, int)
    context = hashed_input(f"{name}:extension-text", (1, 5, context_dim))
    kwargs = {}
    if family == "unet":
        if config["adm_in_channels"] is not None:
            kwargs["y"] = hashed_input("extension-adm", (1, config["adm_in_channels"]))
    else:
        if config["vec_in_dim"] is not None:
            kwargs["y"] = hashed_input("extension-y", (1, config["vec_in_dim"]))
        if config["guidance_embed"]:
            kwargs["guidance"] = torch.tensor([3.5])
    with torch.no_grad():
        plain = model(x, torch.tensor([0.5]), context, **kwargs)
        changed = model(x, torch.tensor([0.5]), context, **kwargs, attention_extensions=run)
        restored = model(x, torch.tensor([0.5]), context, **kwargs)
    assert events
    assert any(kind == "block" for _, kind in events)
    assert not torch.equal(changed, plain)
    assert torch.equal(restored, plain)
    invalid = execution(
        (
            (
                "proof",
                contribution(
                    outputs=(
                        AttentionOutputDescriptor(
                            "proof.missing",
                            AttentionSelector(family, "nonexistent.block"),
                            output,
                        ),
                    )
                ),
            ),
        )
    )
    with pytest.raises(GuidanceContractError, match="matches no declared model point"):
        model(x, torch.tensor([0.5]), context, **kwargs, attention_extensions=invalid)
