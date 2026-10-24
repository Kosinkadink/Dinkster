"""Native execution-semantics fixtures for the development pack.

These nodes intentionally use only the public Dinkster node, expansion, and
run-control contracts. They give the execution gate stable, small graphs
without importing ComfyUI's V1 API or its test pack.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from dinkster_api.v1 import (
    BLOCKED,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
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


class ExecutionBlock(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.execution.block",
            inputs=(
                InputSpec("value", INT),
                InputSpec("block", BOOLEAN),
                InputSpec("message", TypeExpr.concrete("STRING"), default="blocked"),
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
            outputs=(OutputSpec("value", INT),),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls, *, value: int) -> Expansion:
        graph = ExpansionBuilder()
        child = graph.node("dev.execution.add", "child", value=value, amount=1)
        return graph.finalize(value=child.out(0))


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


EXECUTION_NODES: tuple[type[Node], ...] = (
    ExecutionValue,
    ExecutionAdd,
    ExecutionVolatile,
    ExecutionOutput,
    ExecutionLazySelect,
    ExecutionBlock,
    ExecutionExpand,
    ExecutionDynamicCycle,
    ExecutionSleep,
    ExecutionSum,
    ExecutionParallelSleep,
    TestForLoopOpen,
    ExecutionFail,
)
