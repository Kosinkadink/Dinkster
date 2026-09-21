"""Lazy value routing nodes."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ABSENT,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    ComboWidget,
    InputFamilyOptionSource,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)
T = TypeExpr.variable("T")


class RouteSwitch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.route.switch",
            display_name="Route Switch",
            category="routing",
            inputs=(InputSpec("index", INT, default=0, widget=NumberWidget(min=0, max=511)),),
            input_families=(
                InputFamilySpec(
                    "values",
                    (InputSpec("value", T, lazy=True),),
                    min_members=1,
                    max_members=512,
                ),
            ),
            outputs=(OutputSpec("value", T),),
        )

    @staticmethod
    def _selected(index: int, values: Mapping[str, object | None]) -> tuple[str, object | None]:
        if type(index) is not int or not 0 <= index < len(values):
            raise ValueError(f"index must be in [0, {len(values) - 1}]")
        return tuple(values.items())[index]

    @classmethod
    def check_lazy_status(
        cls, *, index: int, values: Mapping[str, object | None]
    ) -> tuple[str, ...]:
        suffix, value = cls._selected(index, values)
        return (f"values.{suffix}",) if value is None else ()

    @classmethod
    def execute(cls, *, index: int, values: Mapping[str, object | None]) -> Mapping[str, object]:
        _suffix, value = cls._selected(index, values)
        if value is None:
            raise ValueError("selected value is unavailable")
        return cls.outputs(value=value)


class RouteSwitchByName(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.route.switch_by_name",
            editor_role="named-route-switch",
            display_name="Route Switch by Name",
            category="routing",
            inputs=(
                InputSpec(
                    "choice",
                    COMBO,
                    widget=ComboWidget(
                        option_source=InputFamilyOptionSource(input_family="values")
                    ),
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "values",
                    (InputSpec("value", T, lazy=True),),
                    min_members=1,
                    max_members=512,
                ),
            ),
            outputs=(OutputSpec("value", T),),
        )

    @staticmethod
    def _selected(choice: str, values: Mapping[str, object | None]) -> object | None:
        if type(choice) is not str or choice not in values:
            raise ValueError(f"choice must name a values member, got {choice!r}")
        return values[choice]

    @classmethod
    def check_lazy_status(
        cls, *, choice: str, values: Mapping[str, object | None]
    ) -> tuple[str, ...]:
        value = cls._selected(choice, values)
        return (f"values.{choice}",) if value is None else ()

    @classmethod
    def execute(cls, *, choice: str, values: Mapping[str, object | None]) -> Mapping[str, object]:
        value = cls._selected(choice, values)
        if value is None:
            raise ValueError("selected value is unavailable")
        return cls.outputs(value=value)


class RouteGate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.route.gate",
            display_name="Route Gate",
            category="routing",
            inputs=(
                InputSpec("condition", BOOLEAN),
                InputSpec("value", T, lazy=True),
            ),
            outputs=(OutputSpec("value", T, optional=True),),
        )

    @classmethod
    def check_lazy_status(cls, *, condition: bool, value: object | None) -> tuple[str, ...]:
        return ("value",) if condition and value is None else ()

    @classmethod
    def execute(cls, *, condition: bool, value: object | None) -> Mapping[str, object]:
        if condition and value is None:
            raise ValueError("gated value is unavailable")
        return cls.outputs(value=value if condition else ABSENT)


ROUTING_NODES: tuple[type[Node], ...] = (RouteSwitch, RouteSwitchByName, RouteGate)
