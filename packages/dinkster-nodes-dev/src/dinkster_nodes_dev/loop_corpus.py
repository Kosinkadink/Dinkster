"""Small observable nodes used by the Generic Loops corpus compiler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dinkster_api.v1 import (
    CORE_BOOLEAN,
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
VALUE = TypeExpr.concrete("core.value")
ANY = TypeExpr.wildcard()
LIST_INT = TypeExpr.list_of(INT)
LIST_VALUE = TypeExpr.list_of(VALUE)


class Observations:
    values: dict[str, list[object]] = {}

    @classmethod
    def reset(cls) -> None:
        cls.values = {
            "increment_calls": [],
            "outer_probe_calls": [],
            "index_independent_probe_calls": [],
            "capture_values": [],
            "passthrough_values": [],
            "first_values": [],
            "second_values": [],
            "record_carried_values": [],
            "append_values": [],
            "loop_state_values": [],
            "false_branch_calls": [],
            "true_branch_calls": [],
        }


def _schema(
    node_type: str, inputs: tuple[InputSpec, ...], output: TypeExpr | None, *, effect: bool = False
) -> NodeSchema:
    return NodeSchema(
        node_type,
        inputs=inputs,
        outputs=() if output is None else (OutputSpec("output_0", output),),
        idempotent=not effect,
    )


class Constant(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("Issue408Constant", (InputSpec("value", INT),), INT)

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(output_0=value)


class Increment(Node):
    observation = "increment_calls"
    node_type = "Issue408Increment"

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(cls.node_type, (InputSpec("value", INT),), INT)

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        result = value + 1
        Observations.values[cls.observation].append(result)
        return cls.outputs(output_0=result)


class OuterProbe(Increment):
    observation = "outer_probe_calls"
    node_type = "Issue408OuterProbe"


class ExpandIncrement(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "Issue408ExpandIncrement",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("output_0", INT),),
            may_expand_graph=True,
        )

    @classmethod
    def execute(cls, *, value: int) -> Expansion:
        builder = ExpansionBuilder(cache_policy="rerun")
        increment = builder.node("Issue408Increment", "increment", value=value)
        return builder.finalize(output_0=increment.out("output_0"))


class IndexIndependentProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "Issue408IndexIndependentProbe",
            (InputSpec("value", INT), InputSpec("dependency", INT)),
            INT,
        )

    @classmethod
    def execute(cls, *, value: int, dependency: int) -> Mapping[str, object]:
        del dependency
        Observations.values["index_independent_probe_calls"].append(value)
        return cls.outputs(output_0=value)


def _recorder(
    node_type: str,
    observation: str,
    inputs: tuple[InputSpec, ...],
    output: TypeExpr,
    *,
    effect: bool = True,
):
    class Recorder(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return _schema(node_type, inputs, output, effect=effect)

        @classmethod
        def execute(cls, **values: object) -> Mapping[str, object]:
            if observation == "loop_state_values":
                recorded: object = [
                    values["index"],
                    values["is_first"],
                    values["is_last"],
                    values["item"],
                ]
                result = values["index"]
            else:
                value = values["value"]
                recorded = value
                result = recorded
                if observation == "second_values":
                    recorded = [value, values["last"]]
                elif observation == "append_values":
                    if not isinstance(value, list):
                        raise TypeError("Issue408AppendIndex requires a list value")
                    recorded = [*cast(list[object], value), values["index"]]
                    result = recorded
            Observations.values[observation].append(recorded)
            return cls.outputs(output_0=result)

    Recorder.__name__ = node_type
    return Recorder


class Capture(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("Issue408Capture", (InputSpec("value", ANY),), None, effect=True)

    @classmethod
    def execute(cls, *, value: object) -> Mapping[str, object]:
        Observations.values["capture_values"].append(value)
        return {}


Passthrough = _recorder(
    "Issue408CapturePassthrough",
    "passthrough_values",
    (InputSpec("value", INT),),
    INT,
    effect=False,
)
First = _recorder(
    "Issue408CapturePassthroughFirst", "first_values", (InputSpec("value", INT),), INT
)
Second = _recorder(
    "Issue408CapturePassthroughSecond",
    "second_values",
    (InputSpec("value", INT), InputSpec("last", BOOLEAN)),
    INT,
)
FalseBranch = _recorder(
    "Issue408FalseBranch", "false_branch_calls", (InputSpec("value", INT),), INT
)
TrueBranch = _recorder("Issue408TrueBranch", "true_branch_calls", (InputSpec("value", INT),), INT)
LoopState = _recorder(
    "Issue408CaptureLoopState",
    "loop_state_values",
    (
        InputSpec("index", INT),
        InputSpec("is_first", BOOLEAN),
        InputSpec("is_last", BOOLEAN),
        InputSpec("item", VALUE),
    ),
    INT,
)
RecordCarried = _recorder(
    "Issue408RecordCarried", "record_carried_values", (InputSpec("value", VALUE),), VALUE
)
AppendIndex = _recorder(
    "Issue408AppendIndex",
    "append_values",
    (InputSpec("value", VALUE), InputSpec("index", INT)),
    VALUE,
)


class Pair(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("Issue408Pair", (InputSpec("value", INT),), LIST_VALUE)

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(output_0=[value, str(value)])


class ListBackedScalar(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("Issue408ListBackedScalar", (InputSpec("value", INT),), VALUE)

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(output_0=[[value]])


class EmptyList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("Issue408EmptyList", (), LIST_VALUE)

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(output_0=[])


class IntegerList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("Issue408IntegerList", (), LIST_INT)

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(output_0=[10, 20])


LOOP_CORPUS_NODES = (
    Constant,
    Increment,
    OuterProbe,
    ExpandIncrement,
    IndexIndependentProbe,
    Capture,
    Passthrough,
    First,
    Second,
    FalseBranch,
    TrueBranch,
    LoopState,
    RecordCarried,
    AppendIndex,
    Pair,
    ListBackedScalar,
    EmptyList,
    IntegerList,
)
