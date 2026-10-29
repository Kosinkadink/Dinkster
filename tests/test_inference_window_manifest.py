"""Windowed-evaluation manifest slot binding tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from dinkster_inference import (
    WINDOWED_EVALUATION_SLOT,
    CompositeWindowPlan,
    LayerWindow,
    MediaAxis,
    MergeDeclaration,
    WindowIndexList,
    WindowPlanBinding,
    WindowPlanLayer,
    WindowSlotRefusal,
    WindowSlotRefusalCode,
    WindowWeightKind,
    WindowWeightProfile,
    build_canonical_manifest,
    build_windowed_evaluation_slot,
    compile_window_plan,
    prove_manifest_consensus,
)

RUNTIME_IDENTITY = f"native:dinkster.test:{hashlib.sha256(b'runtime').hexdigest()}"
PLAN_DIGEST = hashlib.sha256(b"rank plan").hexdigest()
DERIVATION_IDENTITY = "static-window-set.v1"
DERIVATION_FACTS = hashlib.sha256(b"derivation facts").hexdigest()


def _layer(axis: str, windows: tuple[tuple[int, ...], ...]) -> WindowPlanLayer:
    return WindowPlanLayer(
        (axis,),
        tuple(LayerWindow((WindowIndexList(indices, False),)) for indices in windows),
        (WindowWeightProfile(WindowWeightKind.FLAT, 0),),
        MergeDeclaration(),
        (),
    )


def _plan(*, temporal_windows: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3))):
    return compile_window_plan(
        axes=(MediaAxis("temporal", 4, False), MediaAxis("height", 2, False)),
        layers=(
            _layer("temporal", temporal_windows),
            _layer("height", ((0, 1),)),
        ),
    )


def _window_digests(plan: CompositeWindowPlan, salt: str) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(f"{salt}:{index}".encode()).hexdigest()
        for index in range(len(plan.joint_windows))
    )


def _binding(plan: CompositeWindowPlan | None = None, **overrides: tuple[str, ...]):
    plan = _plan() if plan is None else plan
    return WindowPlanBinding(
        plan,
        overrides.get("window_token_layout_digests", _window_digests(plan, "layout")),
        overrides.get("window_partition_plan_digests", _window_digests(plan, "partition")),
        overrides.get("window_structural_row_digests", _window_digests(plan, "structural")),
    )


def _slot(
    plan_bindings: tuple[WindowPlanBinding, ...] | None = None,
    *,
    derivation_identity: str = DERIVATION_IDENTITY,
    derivation_facts_digest: str = DERIVATION_FACTS,
):
    return build_windowed_evaluation_slot(
        derivation_identity=derivation_identity,
        derivation_facts_digest=derivation_facts_digest,
        plan_bindings=(_binding(),) if plan_bindings is None else plan_bindings,
    )


def test_slot_binds_derivation_plans_layers_and_per_window_digests() -> None:
    plan = _plan()
    slot = _slot((_binding(plan),))

    assert slot.slot == WINDOWED_EVALUATION_SLOT
    assert f"derivation={DERIVATION_IDENTITY}" in slot.facts
    assert f"derivation_facts={DERIVATION_FACTS}" in slot.facts
    assert "plan_count=1" in slot.facts
    assert f"plan[0].composite={plan.digest}" in slot.facts
    assert f"plan[0].layer_count={len(plan.layer_digests)}" in slot.facts
    for index, digest in enumerate(plan.layer_digests):
        assert f"plan[0].layer[{index}]={digest}" in slot.facts
    assert f"plan[0].joint_window_count={len(plan.joint_windows)}" in slot.facts
    layouts = _window_digests(plan, "layout")
    partitions = _window_digests(plan, "partition")
    structurals = _window_digests(plan, "structural")
    for index in range(len(plan.joint_windows)):
        assert f"plan[0].window[{index}].token_layout={layouts[index]}" in slot.facts
        assert f"plan[0].window[{index}].partition_plan={partitions[index]}" in slot.facts
        assert f"plan[0].window[{index}].structural_rows={structurals[index]}" in slot.facts


def test_slot_covers_every_realizable_plan_in_order() -> None:
    first = _plan()
    second = _plan(temporal_windows=((0, 1, 2), (3,)))
    slot = _slot((_binding(first), _binding(second)))

    assert "plan_count=2" in slot.facts
    assert f"plan[0].composite={first.digest}" in slot.facts
    assert f"plan[1].composite={second.digest}" in slot.facts

    reordered = _slot((_binding(second), _binding(first)))
    assert reordered.digest != slot.digest


def test_slot_digest_is_deterministic_and_binds_every_input() -> None:
    base = _slot()
    assert _slot().digest == base.digest

    plan = _plan()
    variants = (
        _slot(derivation_identity="per-step-window-set.v1"),
        _slot(derivation_facts_digest=hashlib.sha256(b"other derivation").hexdigest()),
        _slot((_binding(_plan(temporal_windows=((0, 1, 2), (3,)))),)),
        _slot((_binding(), _binding(_plan(temporal_windows=((0, 1, 2), (3,)))))),
        _slot((_binding(plan, window_token_layout_digests=_window_digests(plan, "other-layout")),)),
        _slot(
            (
                _binding(
                    plan, window_partition_plan_digests=_window_digests(plan, "other-partition")
                ),
            )
        ),
        _slot(
            (
                _binding(
                    plan, window_structural_row_digests=_window_digests(plan, "other-structural")
                ),
            )
        ),
    )
    digests = {base.digest, *(variant.digest for variant in variants)}
    assert len(digests) == len(variants) + 1


def test_swapping_per_window_digest_roles_changes_identity() -> None:
    plan = _plan()
    layouts = _window_digests(plan, "layout")
    partitions = _window_digests(plan, "partition")
    structurals = _window_digests(plan, "structural")
    forward = _slot((WindowPlanBinding(plan, layouts, partitions, structurals),))
    swapped = _slot((WindowPlanBinding(plan, partitions, layouts, structurals),))
    structural_swapped = _slot((WindowPlanBinding(plan, layouts, structurals, partitions),))
    assert forward.digest != swapped.digest
    assert forward.digest != structural_swapped.digest


def test_slot_flows_into_manifest_identity_and_consensus() -> None:
    plan = _plan()

    def manifest(slot_salt: str):
        return build_canonical_manifest(
            runtime_identity=RUNTIME_IDENTITY,
            invocation_facts=("sigma_table=1.0,0.5,0.0",),
            slots=(
                _slot(
                    (
                        _binding(
                            plan,
                            window_token_layout_digests=_window_digests(plan, slot_salt),
                        ),
                    )
                ),
            ),
            rank_plan_digests=((PLAN_DIGEST,),),
        )

    agreeing = manifest("layout")
    assert agreeing.digest == manifest("layout").digest
    assert agreeing.digest != manifest("other-layout").digest

    class EchoTransport:
        def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
            return (digest,)

    token = prove_manifest_consensus(agreeing, rank=0, transport=EchoTransport())
    assert token.manifest_digest == agreeing.digest


def test_refuses_non_plan_values() -> None:
    plan = _plan()

    class DuckPlan:
        digest = hashlib.sha256(b"duck").hexdigest()
        layer_digests = plan.layer_digests
        joint_windows = plan.joint_windows

    with pytest.raises(WindowSlotRefusal) as error:
        WindowPlanBinding(
            DuckPlan(),  # type: ignore[arg-type]
            _window_digests(plan, "layout"),
            _window_digests(plan, "partition"),
            _window_digests(plan, "structural"),
        )
    assert error.value.code is WindowSlotRefusalCode.INVALID_WINDOW_PLAN


def test_refuses_non_binding_values() -> None:
    binding = _binding()

    class DuckBinding:
        plan = binding.plan
        window_token_layout_digests = binding.window_token_layout_digests
        window_partition_plan_digests = binding.window_partition_plan_digests
        window_structural_row_digests = binding.window_structural_row_digests

    with pytest.raises(WindowSlotRefusal) as error:
        build_windowed_evaluation_slot(
            derivation_identity=DERIVATION_IDENTITY,
            derivation_facts_digest=DERIVATION_FACTS,
            plan_bindings=(DuckBinding(),),  # type: ignore[arg-type]
        )
    assert error.value.code is WindowSlotRefusalCode.INVALID_WINDOW_PLAN


def test_refuses_invalid_derivation_bindings() -> None:
    cases: tuple[dict[str, object], ...] = (
        {"derivation_identity": ""},
        {"derivation_identity": 7},
        {"derivation_identity": "line\nbreak"},
        {"derivation_facts_digest": "not-a-digest"},
        {"derivation_facts_digest": DERIVATION_FACTS.upper()},
        {"plan_bindings": ()},
        {"plan_bindings": [_binding()]},
        {"plan_bindings": (_binding(), _binding())},
    )
    for overrides in cases:
        arguments: dict[str, object] = {
            "derivation_identity": DERIVATION_IDENTITY,
            "derivation_facts_digest": DERIVATION_FACTS,
            "plan_bindings": (_binding(),),
            **overrides,
        }
        with pytest.raises(WindowSlotRefusal) as error:
            build_windowed_evaluation_slot(**arguments)  # type: ignore[arg-type]
        assert error.value.code is WindowSlotRefusalCode.INVALID_DERIVATION_BINDING


def test_refuses_mismatched_or_malformed_window_digests() -> None:
    plan = _plan()
    good = _window_digests(plan, "layout")
    cases: tuple[tuple[str, object], ...] = (
        ("window_token_layout_digests", good[:-1]),
        ("window_token_layout_digests", (*good[:-1], "not-a-digest")),
        ("window_token_layout_digests", list(good)),
        ("window_partition_plan_digests", ()),
        ("window_partition_plan_digests", (*good[:-1], good[-1].upper())),
        ("window_structural_row_digests", good[:-1]),
        ("window_structural_row_digests", (*good[:-1], 7)),
    )
    for name, value in cases:
        with pytest.raises(WindowSlotRefusal) as error:
            replace(_binding(plan), **{name: value})  # type: ignore[arg-type]
        assert error.value.code is WindowSlotRefusalCode.WINDOW_BINDING_MISMATCH
        assert name in str(error.value)
