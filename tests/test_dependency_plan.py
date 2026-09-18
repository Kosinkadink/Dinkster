"""Pure dependency planner ordering, selection, and boundary proofs."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import pytest
from dinkster_inference import (
    ActiveDependencyPlan,
    DependencyPlan,
    DependencyRef,
    PlanNode,
    ReconstructionRecipe,
    ResidencyGroupPlan,
    RuntimeKnobs,
    WeightSourceBinding,
    WeightSourceRef,
    plan_dependencies,
)


def recipe(
    name: str,
    dependencies: tuple[DependencyRef[ReconstructionRecipe], ...] = (),
) -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "checkpoint",
                WeightSourceRef(
                    digest="blake3:" + "1" * 64,
                    name=f"{name}.safetensors",
                    size=1,
                ),
            ),
        ),
        family_id=f"proof.{name}",
        component_identity=(f"family=proof.{name}",),
        knobs=RuntimeKnobs(
            diffusion_dtype="float32",
            text_dtype="float32",
            vae_dtype="float32",
            fp8_matmul=False,
        ),
        dependencies=dependencies,
    )


def edge(
    child_id: str,
    child: ReconstructionRecipe,
    *,
    group: str = "group/main",
    scope: str = "model",
    owner: str = "parent",
) -> DependencyRef[ReconstructionRecipe]:
    return DependencyRef(
        child_id=child_id,
        child=child,
        residency_group=group,
        scope=scope,  # type: ignore[arg-type]
        clone_mode="with-parent",
        accounting_owner=owner,
    )


def test_depth_first_nodes_and_children_before_parent_orders_are_stable() -> None:
    first = recipe(
        "first",
        (
            edge("grand-a", recipe("grand-a")),
            edge("grand-b", recipe("grand-b"), group="group/other"),
        ),
    )
    root = recipe(
        "root",
        (
            edge("first", first),
            edge("second", recipe("second"), group="group/second"),
        ),
    )

    plan = plan_dependencies(root)

    assert tuple(node.path for node in plan.nodes) == (
        (),
        ("first",),
        ("first", "grand-a"),
        ("first", "grand-b"),
        ("second",),
    )
    assert plan.load_order == (
        ("first", "grand-a"),
        ("first", "grand-b"),
        ("first",),
        ("second",),
        (),
    )
    assert plan.release_order == tuple(reversed(plan.load_order))
    assert plan == plan_dependencies(root)


def test_groups_are_local_ordered_and_resolve_parent_and_child_owners() -> None:
    nested = recipe(
        "nested",
        (edge("leaf", recipe("leaf"), group="shared"),),
    )
    root = recipe(
        "root",
        (
            edge("owner", nested, group="shared", owner="owner"),
            edge("peer", recipe("peer"), group="shared", owner="owner"),
            edge("solo", recipe("solo"), group="solo"),
        ),
    )

    groups = plan_dependencies(root).residency_groups

    assert groups == (
        ResidencyGroupPlan(
            declaring_path=(),
            name="shared",
            member_paths=(("owner",), ("peer",)),
            owner_path=("owner",),
        ),
        ResidencyGroupPlan(
            declaring_path=(),
            name="solo",
            member_paths=(("solo",),),
            owner_path=(),
        ),
        ResidencyGroupPlan(
            declaring_path=("owner",),
            name="shared",
            member_paths=(("owner", "leaf"),),
            owner_path=("owner",),
        ),
    )


def test_equal_children_remain_distinct_per_edge_nodes() -> None:
    child = recipe("same")
    plan = plan_dependencies(
        recipe(
            "root",
            (
                edge("left", child),
                edge("right", child),
            ),
        )
    )

    left, right = plan.nodes[1:]
    assert left.path == ("left",)
    assert right.path == ("right",)
    assert left.runtime_identity == right.runtime_identity
    assert len(plan.nodes) == 3


def test_conditional_selection_prunes_whole_subtrees_in_full_relative_order() -> None:
    conditional = recipe(
        "conditional",
        (edge("grandchild", recipe("grandchild")),),
    )
    plan = plan_dependencies(
        recipe(
            "root",
            (
                edge("optional", conditional, scope="conditional"),
                edge("always", recipe("always")),
            ),
        )
    )

    inactive = plan.select_active(frozenset())
    active = plan.select_active(frozenset({("optional",)}))

    assert inactive.load_order == (("always",), ())
    assert active.load_order == plan.load_order
    assert active.release_order == tuple(reversed(active.load_order))


def test_conditional_selection_refuses_unknown_and_nonconditional_paths() -> None:
    plan = plan_dependencies(
        recipe(
            "root",
            (
                edge("optional", recipe("optional"), scope="conditional"),
                edge("always", recipe("always")),
            ),
        )
    )

    with pytest.raises(ValueError, match="is unknown"):
        plan.select_active(frozenset({("missing",)}))
    with pytest.raises(ValueError, match="is not conditional"):
        plan.select_active(frozenset({("always",)}))


@pytest.mark.parametrize("scope", ("contribution", "invocation"))
def test_active_reserved_scopes_refuse_instead_of_behaving_like_model(
    scope: str,
) -> None:
    reserved = recipe(
        "reserved-parent",
        (edge("reserved", recipe("reserved"), scope=scope),),
    )
    plan = plan_dependencies(
        recipe(
            "root",
            (edge("optional", reserved, scope="conditional"),),
        )
    )

    assert plan.select_active(frozenset()).load_order == ((),)
    with pytest.raises(ValueError, match=f"reserved scope '{scope}'"):
        plan.select_active(frozenset({("optional",)}))


def test_planner_refuses_cycles_before_reading_recursive_identity() -> None:
    child = recipe("cycle")
    object.__setattr__(child, "dependencies", (edge("self", child),))

    with pytest.raises(ValueError, match="must be acyclic"):
        plan_dependencies(child)


def test_plan_records_and_active_view_are_frozen() -> None:
    plan = plan_dependencies(recipe("root", (edge("child", recipe("child")),)))
    active = plan.select_active(frozenset())

    values: tuple[DependencyPlan | PlanNode | ResidencyGroupPlan | ActiveDependencyPlan, ...] = (
        plan,
        plan.nodes[0],
        plan.residency_groups[0],
        active,
    )
    for value in values:
        with pytest.raises(FrozenInstanceError):
            value.load_order = ()  # type: ignore[union-attr,misc]
    with pytest.raises(TypeError, match="load order must be a tuple"):
        ActiveDependencyPlan(load_order=[()], release_order=((),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="release order must be a tuple"):
        ActiveDependencyPlan(load_order=((),), release_order=[()])  # type: ignore[arg-type]


def test_plan_data_is_rpc_clean_plain_data_without_backend_mechanisms() -> None:
    plan = plan_dependencies(recipe("root", (edge("child", recipe("child")),)))

    wire = asdict(plan)
    encoded = json.dumps(wire, sort_keys=True)

    assert json.loads(encoded)["load_order"] == [["child"], []]

    def assert_plain(value: object) -> None:
        if value is None or type(value) in (str, int, bool):
            return
        if isinstance(value, tuple):
            for item in value:
                assert_plain(item)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str)
                assert key not in {"callable", "handle", "lease", "mechanism", "pool"}
                assert_plain(item)
            return
        pytest.fail(f"non-plain plan value: {type(value).__name__}")

    assert_plain(wire)
