"""Native execution-semantics fixtures for the development pack.

These nodes intentionally use only the public Dinkster node, expansion, and
run-control contracts. They give the execution gate stable, small graphs
without importing ComfyUI's V1 API or its test pack.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping

from dinkster_api.v1 import (
    BLOCKED,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    MASK_TYPE,
    Expansion,
    ExpansionBuilder,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)

INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
FLOAT = TypeExpr.concrete(CORE_FLOAT)


class ExecutionValue(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.value",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(value=value)


class ExecutionAdd(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.add",
            inputs=(InputSpec("value", INT), InputSpec("amount", INT, default=1)),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, value: int, amount: int) -> Mapping[str, object]:
        return cls.outputs(value=value + amount)


class ExecutionVolatile(ExecutionValue):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.volatile",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("value", INT),),
            idempotent=False,
        )


class ExecutionOutput(ExecutionValue):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.output",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("value", INT),),
            output_node=True,
        )


class ExecutionLazySelect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.lazy_select",
            inputs=(
                InputSpec("choose_right", BOOLEAN),
                InputSpec("left", INT, lazy=True),
                InputSpec("right", INT, lazy=True),
            ),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def check_lazy_status(
        cls, *, choose_right: bool, left: object | None, right: object | None
    ) -> tuple[str, ...]:
        selected = "right" if choose_right else "left"
        value = right if choose_right else left
        return (selected,) if value is None else ()

    @classmethod
    def execute(
        cls, *, choose_right: bool, left: int | None, right: int | None
    ) -> Mapping[str, object]:
        value = right if choose_right else left
        if value is None:
            raise ValueError("selected lazy input is unavailable")
        return cls.outputs(value=value)


class ExecutionLazyMix(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.lazy_mix",
            inputs=(
                InputSpec("weight", FLOAT),
                InputSpec("black", INT, lazy=True),
                InputSpec("white", INT, lazy=True),
            ),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def check_lazy_status(
        cls, *, weight: float, black: object | None, white: object | None
    ) -> tuple[str, ...]:
        requested: list[str] = []
        if weight < 1.0 and black is None:
            requested.append("black")
        if weight > 0.0 and white is None:
            requested.append("white")
        return tuple(requested)

    @classmethod
    def execute(
        cls, *, weight: float, black: int | None, white: int | None
    ) -> Mapping[str, object]:
        if black is None and weight < 1.0:
            raise ValueError("black branch was not demanded")
        if white is None and weight > 0.0:
            raise ValueError("white branch was not demanded")
        return cls.outputs(value=math.floor((black or 0) * (1.0 - weight) + (white or 0) * weight))


class ExecutionBlock(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.block",
            inputs=(
                InputSpec("value", INT),
                InputSpec("block", BOOLEAN),
                InputSpec("message", TypeExpr.concrete(CORE_STRING), default="blocked"),
            ),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, value: int, block: bool, message: str) -> Mapping[str, object]:
        return cls.outputs(value=BLOCKED(message) if block else value)


class ExecutionExpand(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.expand",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("value", INT), OutputSpec("literal", INT)),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls, *, value: int) -> Expansion:
        graph = ExpansionBuilder()
        child = graph.node("dev.execution.add", "child", value=value, amount=1)
        return graph.finalize(value=child.out(0), literal=99)


class ExecutionDynamicCycle(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.dynamic_cycle",
            outputs=(OutputSpec("value", INT),),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls) -> Expansion:
        graph = ExpansionBuilder()
        child = graph.node("dev.execution.add", "child")
        child.set_input("value", child.out(0))
        child.set_input("amount", 1)
        return graph.finalize(value=child.out(0))


class ExecutionSleep(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.sleep",
            inputs=(InputSpec("value", INT), InputSpec("seconds", FLOAT, default=0.1)),
            outputs=(OutputSpec("value", INT),),
            io_bound=True,
        )

    @classmethod
    async def execute(cls, *, value: int, seconds: float) -> Mapping[str, object]:
        await asyncio.sleep(seconds)
        return cls.outputs(value=value)


class ExecutionSum(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.sum",
            inputs=(InputSpec("left", INT), InputSpec("middle", INT), InputSpec("right", INT)),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, left: int, middle: int, right: int) -> Mapping[str, object]:
        return cls.outputs(value=left + middle + right)


class ExecutionParallelSleep(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.parallel_sleep",
            inputs=(
                InputSpec("left", INT),
                InputSpec("middle", INT),
                InputSpec("right", INT),
                InputSpec("seconds", FLOAT, default=0.1),
            ),
            outputs=(OutputSpec("value", INT),),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls, *, left: int, middle: int, right: int, seconds: float) -> Expansion:
        graph = ExpansionBuilder()
        slept = [
            graph.node("dev.execution.sleep", name, value=value, seconds=seconds)
            for name, value in (("left", left), ("middle", middle), ("right", right))
        ]
        total = graph.node(
            "dev.execution.sum",
            "sum",
            left=slept[0].out(0),
            middle=slept[1].out(0),
            right=slept[2].out(0),
        )
        return graph.finalize(value=total.out(0))


class TestForLoopOpen(Node):
    """Expands a finite carried-value loop without raw links or V1 adapters."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.test_for_loop_open",
            inputs=(InputSpec("initial", INT), InputSpec("iterations", INT)),
            outputs=(OutputSpec("value", INT),),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls, *, initial: int, iterations: int) -> Expansion:
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        if iterations == 0:
            graph = ExpansionBuilder()
            terminal = graph.node("dev.execution.value", "terminal", value=initial)
            return graph.finalize(value=terminal.out(0))
        graph = ExpansionBuilder(cache_policy="rerun")
        next_iteration = graph.node(
            "dev.execution.test_for_loop_open",
            "next",
            initial=initial + 1,
            iterations=iterations - 1,
        )
        return graph.finalize(value=next_iteration.out(0))


class ExecutionFail(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.fail",
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        raise RuntimeError("execution fixture failure")


class ExecutionAdmissionSource(ExecutionValue):
    executed = False

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.admission_source",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        cls.executed = True
        return cls.outputs(value=value)


class ExecutionFloatValue(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.float_value",
            inputs=(InputSpec("value", FLOAT),),
            outputs=(OutputSpec("value", FLOAT),),
        )

    @classmethod
    def execute(cls, *, value: float) -> Mapping[str, object]:
        return cls.outputs(value=value)


class ExecutionMaskSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.mask_source",
            inputs=(InputSpec("value", FLOAT),),
            outputs=(OutputSpec("value", TypeExpr.concrete(MASK_TYPE)),),
        )

    @classmethod
    def execute(cls, *, value: float) -> Mapping[str, object]:
        raise AssertionError("MASK source must be rejected before execution")


class ExecutionAdmission(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.admission",
            inputs=(InputSpec("value", FLOAT),),
            outputs=(OutputSpec("value", FLOAT),),
            admission_validation=True,
        )

    @classmethod
    def validate_admission(
        cls, *, literal_inputs: Mapping[str, object], linked_input_types: Mapping[str, str]
    ) -> str | None:
        value = literal_inputs.get("value")
        if value is not None and type(value) is not float:
            return "value must be a float literal"
        if linked_input_types.get("value") not in (None, "FLOAT"):
            return "value must be linked from FLOAT"
        return None

    @classmethod
    def execute(cls, *, value: float) -> Mapping[str, object]:
        return cls.outputs(value=value)


class ExecutionAdmissionKwargs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.admission_kwargs",
            inputs=(InputSpec("input1", FLOAT), InputSpec("input2", FLOAT)),
            outputs=(OutputSpec("value", FLOAT),),
            admission_validation=True,
        )

    @classmethod
    def validate_admission(
        cls, *, literal_inputs: Mapping[str, object], linked_input_types: Mapping[str, str]
    ) -> str | None:
        if literal_inputs.get("input2") == 7.0:
            return "7s are not allowed. I've never liked 7s."
        return None

    @classmethod
    def execute(cls, *, input1: float, input2: float) -> Mapping[str, object]:
        return cls.outputs(value=input1 * input2)


EXECUTION_NODES: tuple[type[Node], ...] = (
    ExecutionValue,
    ExecutionAdd,
    ExecutionVolatile,
    ExecutionOutput,
    ExecutionLazySelect,
    ExecutionLazyMix,
    ExecutionBlock,
    ExecutionExpand,
    ExecutionDynamicCycle,
    ExecutionSleep,
    ExecutionSum,
    ExecutionParallelSleep,
    TestForLoopOpen,
    ExecutionFail,
    ExecutionAdmissionSource,
    ExecutionFloatValue,
    ExecutionMaskSource,
    ExecutionAdmission,
    ExecutionAdmissionKwargs,
)
