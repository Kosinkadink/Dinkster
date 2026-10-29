"""Pure declarative planning for reconstruction recipe dependencies.

Plans contain only frozen, RPC-clean facts. Runtime handles, materialization
mechanisms, pools, leases, filesystem paths, callables, size estimates, and
backend accounting are deliberately excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from .recipe import (
    DependencyCloneMode,
    DependencyScope,
    ReconstructionRecipe,
)

NodePath: TypeAlias = tuple[str, ...]


def _require_path(name: str, path: NodePath) -> None:
    path_obj = cast("object", path)
    if not isinstance(path_obj, tuple) or not all(
        isinstance(part, str) and part and part != "parent"
        for part in cast("tuple[object, ...]", path_obj)
    ):
        raise ValueError(f"{name} must be a tuple of child ids")


def _require_paths(name: str, paths: tuple[NodePath, ...]) -> None:
    if not isinstance(cast("object", paths), tuple):
        raise TypeError(f"{name} must be a tuple of node paths")
    for path in paths:
        _require_path(name, path)


@dataclass(frozen=True)
class PlanNode:
    """One root or dependency-edge occurrence in a dependency plan."""

    path: NodePath
    runtime_identity: str
    residency_group: str | None = None
    scope: DependencyScope | None = None
    clone_mode: DependencyCloneMode | None = None
    accounting_owner: str | None = None

    def __post_init__(self) -> None:
        _require_path("plan node path", self.path)
        if not isinstance(cast("object", self.runtime_identity), str):
            raise TypeError("plan node runtime_identity must be a string")
        edge_facts = (
            self.residency_group,
            self.scope,
            self.clone_mode,
            self.accounting_owner,
        )
        if not self.path:
            if any(value is not None for value in edge_facts):
                raise ValueError("root plan node must not have declaring edge facts")
            return
        if not all(isinstance(value, str) for value in edge_facts):
            raise ValueError("dependency plan node requires all declaring edge facts")
        if self.scope not in ("model", "contribution", "invocation", "conditional"):
            raise ValueError("plan node dependency scope is unsupported")
        if self.clone_mode not in ("with-parent", "shared"):
            raise ValueError("plan node dependency clone_mode is unsupported")


@dataclass(frozen=True)
class ResidencyGroupPlan:
    """One declaring-recipe-local residency group."""

    declaring_path: NodePath
    name: str
    member_paths: tuple[NodePath, ...]
    owner_path: NodePath

    def __post_init__(self) -> None:
        _require_path("residency group declaring_path", self.declaring_path)
        if not isinstance(cast("object", self.name), str) or not self.name:
            raise ValueError("residency group name must be non-empty")
        members_obj = cast("object", self.member_paths)
        if not isinstance(members_obj, tuple) or not self.member_paths:
            raise ValueError("residency group member_paths must be a non-empty tuple")
        for path in self.member_paths:
            _require_path("residency group member path", path)
            if path[:-1] != self.declaring_path:
                raise ValueError("residency group members must be direct children")
        if len(self.member_paths) != len(set(self.member_paths)):
            raise ValueError("residency group member paths must be unique")
        _require_path("residency group owner_path", self.owner_path)
        if self.owner_path != self.declaring_path and (self.owner_path not in self.member_paths):
            raise ValueError("residency group owner must be its parent or a member")


@dataclass(frozen=True)
class ActiveDependencyPlan:
    """The active load and release projection of a full dependency plan."""

    load_order: tuple[NodePath, ...]
    release_order: tuple[NodePath, ...]

    def __post_init__(self) -> None:
        _require_paths("active dependency load order", self.load_order)
        _require_paths("active dependency release order", self.release_order)
        if not self.load_order or self.load_order[-1] != ():
            raise ValueError("active dependency load order must end with the root")
        if self.release_order != tuple(reversed(self.load_order)):
            raise ValueError("active dependency release order must reverse load order")


@dataclass(frozen=True)
class DependencyPlan:
    """Complete declarative dependency graph and deterministic lifecycle order."""

    nodes: tuple[PlanNode, ...]
    load_order: tuple[NodePath, ...]
    release_order: tuple[NodePath, ...]
    residency_groups: tuple[ResidencyGroupPlan, ...]

    def __post_init__(self) -> None:
        values = (
            ("nodes", self.nodes, PlanNode),
            ("residency_groups", self.residency_groups, ResidencyGroupPlan),
        )
        for name, value, item_type in values:
            value_obj = cast("object", value)
            if not isinstance(value_obj, tuple) or not all(
                isinstance(item, item_type) for item in cast("tuple[object, ...]", value_obj)
            ):
                raise TypeError(f"dependency plan {name} must be a tuple of {item_type.__name__}")
        _require_paths("dependency plan load order", self.load_order)
        _require_paths("dependency plan release order", self.release_order)
        paths = tuple(node.path for node in self.nodes)
        if not paths or paths[0] != () or len(paths) != len(set(paths)):
            raise ValueError("dependency plan nodes must have one unique root-first path")
        if set(self.load_order) != set(paths) or len(self.load_order) != len(paths):
            raise ValueError("dependency plan load order must contain every node once")
        if self.load_order[-1] != ():
            raise ValueError("dependency plan load order must end with the root")
        if self.release_order != tuple(reversed(self.load_order)):
            raise ValueError("dependency plan release order must reverse load order")
        group_keys = tuple((group.declaring_path, group.name) for group in self.residency_groups)
        if len(group_keys) != len(set(group_keys)):
            raise ValueError("dependency plan residency group keys must be unique")
        known_paths = set(paths)
        for group in self.residency_groups:
            if group.declaring_path not in known_paths or any(
                member not in known_paths for member in group.member_paths
            ):
                raise ValueError("dependency plan residency group path is unknown")

    def select_active(self, active_conditional: frozenset[NodePath]) -> ActiveDependencyPlan:
        """Select conditional edges and return their model-scope lifecycle order."""
        selected_obj = cast("object", active_conditional)
        if not isinstance(selected_obj, frozenset):
            raise TypeError("active_conditional must be a frozenset of node paths")
        nodes_by_path = {node.path: node for node in self.nodes}
        for path in active_conditional:
            _require_path("active conditional path", path)
            node = nodes_by_path.get(path)
            if node is None:
                raise ValueError(f"active conditional path {path!r} is unknown")
            if node.scope != "conditional":
                raise ValueError(f"active conditional path {path!r} is not conditional")

        active_paths: set[NodePath] = {()}
        children: dict[NodePath, list[PlanNode]] = {}
        for node in self.nodes[1:]:
            children.setdefault(node.path[:-1], []).append(node)

        def activate(parent_path: NodePath) -> None:
            for node in children.get(parent_path, []):
                if node.scope in ("contribution", "invocation"):
                    raise ValueError(
                        f"active dependency path {node.path!r} has reserved scope {node.scope!r}"
                    )
                if node.scope == "conditional" and node.path not in active_conditional:
                    continue
                active_paths.add(node.path)
                activate(node.path)

        activate(())
        load_order = tuple(path for path in self.load_order if path in active_paths)
        return ActiveDependencyPlan(
            load_order=load_order,
            release_order=tuple(reversed(load_order)),
        )


def _refuse_dependency_cycles(recipe: ReconstructionRecipe, active: set[int]) -> None:
    recipe_id = id(recipe)
    if recipe_id in active:
        raise ValueError("recipe dependency graph must be acyclic")
    active.add(recipe_id)
    for dependency in recipe.dependencies:
        _refuse_dependency_cycles(dependency.child, active)
    active.remove(recipe_id)


def plan_dependencies(recipe: ReconstructionRecipe) -> DependencyPlan:
    """Plan every dependency edge in deterministic depth-first tuple order."""
    if not isinstance(cast("object", recipe), ReconstructionRecipe):
        raise TypeError("dependency planner requires a ReconstructionRecipe")
    _refuse_dependency_cycles(recipe, set())

    nodes: list[PlanNode] = []
    load_order: list[NodePath] = []
    residency_groups: list[ResidencyGroupPlan] = []

    def walk(current: ReconstructionRecipe, path: NodePath) -> None:
        if not path:
            nodes.append(PlanNode(path=(), runtime_identity=current.runtime_identity))

        group_members: dict[str, list[NodePath]] = {}
        group_owners: dict[str, str] = {}
        group_order: list[str] = []
        for dependency in current.dependencies:
            child_path = path + (dependency.child_id,)
            if dependency.residency_group not in group_members:
                group_members[dependency.residency_group] = []
                group_owners[dependency.residency_group] = dependency.accounting_owner
                group_order.append(dependency.residency_group)
            group_members[dependency.residency_group].append(child_path)

        for group_name in group_order:
            owner = group_owners[group_name]
            residency_groups.append(
                ResidencyGroupPlan(
                    declaring_path=path,
                    name=group_name,
                    member_paths=tuple(group_members[group_name]),
                    owner_path=path if owner == "parent" else path + (owner,),
                )
            )

        for dependency in current.dependencies:
            child_path = path + (dependency.child_id,)
            nodes.append(
                PlanNode(
                    path=child_path,
                    runtime_identity=dependency.child.runtime_identity,
                    residency_group=dependency.residency_group,
                    scope=dependency.scope,
                    clone_mode=dependency.clone_mode,
                    accounting_owner=dependency.accounting_owner,
                )
            )
            walk(dependency.child, child_path)
            load_order.append(child_path)

    walk(recipe, ())
    load_order.append(())
    frozen_load_order = tuple(load_order)
    return DependencyPlan(
        nodes=tuple(nodes),
        load_order=frozen_load_order,
        release_order=tuple(reversed(frozen_load_order)),
        residency_groups=tuple(residency_groups),
    )


__all__ = [
    "ActiveDependencyPlan",
    "DependencyPlan",
    "NodePath",
    "PlanNode",
    "ResidencyGroupPlan",
    "plan_dependencies",
]
