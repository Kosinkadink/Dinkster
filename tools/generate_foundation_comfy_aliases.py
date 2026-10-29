"""Generate maintained ComfyUI replacement data for foundation nodes."""

from __future__ import annotations

import json
import string
from itertools import combinations
from pathlib import Path

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    BooleanWidget,
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)
from dinkster_schema import (
    InputFamilyMapping,
    InputFamilyMember,
    MappingSource,
    ReplacementCase,
    ReplacementLink,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    ValueTransform,
    schema_to_wire,
)
from dinkster_schema.replace import rule_to_wire

OUTPUT_PATH = (
    Path(__file__).parents[1] / "packages" / "dinkster-nodes-foundation" / "comfy-aliases.json"
)
NUMERIC_EVIDENCE = (
    "tests/test_numeric_stdlib.py::test_foundation_comfy_alias_records",
    "tests/test_numeric_stdlib.py::test_foundation_comfy_alias_behavior",
)
STRING_EVIDENCE = (
    "tests/test_string_comfy_aliases.py::test_string_alias_registry_is_canonical_and_valid",
    "tests/test_string_comfy_aliases.py::test_string_alias_operation_mappings",
)
CONVERSION_EVIDENCE = (
    "tests/test_conversion_curve_stdlib.py::test_conversion_alias_registry_is_canonical_and_valid",
    "tests/test_conversion_curve_stdlib.py::test_conversion_alias_operation_mappings",
)
ROUTING_EVIDENCE = (
    "tests/test_routing_stdlib.py::test_routing_alias_registry_is_canonical_and_valid",
    "tests/test_routing_stdlib.py::test_routing_alias_operation_mappings",
)
TRELLIS2_WORKFLOW_EVIDENCE = ("tests/test_numeric_stdlib.py::test_foundation_comfy_alias_records",)

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)
STRING = TypeExpr.concrete(CORE_STRING)
ANY = TypeExpr.wildcard()
NUMBER = TypeExpr.concrete("comfy.NUMBER")
DICT = TypeExpr.concrete("comfy.DICT")
ARRAY = TypeExpr.concrete("comfy.ARRAY")

COMFY_REVISION = "b78cec87"
TRELLIS2_REVISION = "8a33128f"
ESSENTIALS_REVISION = "9d9f4bedfc9f0321c19faf71855e228c93bd0dc9"
EASY_USE_REVISION = "595e0738a9e3f8d0d9c4d875461b2d2c9e7559c7"
EASY_USE_STRING_REVISION = "4de1ab3b66e48da916b6f263bacd001df53a2720"
IMPACT_REVISION = "429d0159ad429e64d2b3916e6e7be9c22d025c3c"
WAS_REVISION = "ea935d1044ae5a26efa54ebeb18fe9020af49a45"
MTB_REVISION = "b35b5d8a17c0d59e80a8b3627b679c2c1003d04f"
KJ_REVISION = "3f20054214fec9f9234fd3841ae6f1e4287948f6"


def _switch_source_schema(
    node_type: str,
    display_name: str,
    condition: str,
    on_false: str,
    on_true: str,
    output: str,
) -> NodeSchema:
    value_type = TypeExpr.variable("T")
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="comfy/logic",
        inputs=(
            InputSpec(condition, BOOLEAN),
            InputSpec(on_false, value_type, lazy=True),
            InputSpec(on_true, value_type, lazy=True),
        ),
        outputs=(OutputSpec(output, value_type),),
        aliases=(display_name,),
    )


def _bool_logic_source_schema(
    node_class: str,
    display_name: str,
    description: str,
) -> NodeSchema:
    return NodeSchema(
        node_type=f"comfy.{node_class}",
        display_name=display_name,
        category="comfy/utilities/logic",
        description=description,
        input_families=(
            InputFamilySpec("values", ANY, min_members=1, max_members=10, member_prefix="value"),
        ),
        outputs=(OutputSpec("boolean", BOOLEAN),),
        aliases=(node_class,),
    )


SOURCE_SCHEMAS = (
    NodeSchema(
        node_type="comfy.PrimitiveBoolean",
        display_name="Boolean",
        category="comfy/utilities/primitive",
        inputs=(InputSpec("value", BOOLEAN, default=False),),
        outputs=(OutputSpec("boolean", BOOLEAN),),
        aliases=("PrimitiveBoolean",),
    ),
    NodeSchema(
        node_type="comfy.ComfyMathExpression",
        display_name="Math Expression",
        category="comfy/utilities",
        inputs=(
            InputSpec(
                "expression",
                STRING,
                required=False,
                default="a + b",
                widget=StringWidget(multiline=True),
            ),
        ),
        input_families=(
            InputFamilySpec(
                "values",
                (InputSpec("value", TypeExpr.union(CORE_FLOAT, CORE_INT, CORE_BOOLEAN)),),
                min_members=1,
                member_names=tuple(string.ascii_lowercase),
            ),
        ),
        outputs=(
            OutputSpec("FLOAT", FLOAT),
            OutputSpec("INT", INT),
            OutputSpec("BOOL", BOOLEAN),
        ),
        aliases=("ComfyMathExpression",),
    ),
    _bool_logic_source_schema(
        "ComfyAndNode",
        "And",
        "Logical AND operation. Returns true if all of the values are truthy. "
        "Uses Python's rules for truthiness.",
    ),
    _bool_logic_source_schema(
        "ComfyOrNode",
        "Or",
        "Logical OR operation. Returns true if any of the values are truthy. "
        "Uses Python's rules for truthiness.",
    ),
    _switch_source_schema(
        "comfy.ComfySwitchNode",
        "ComfySwitchNode",
        "switch",
        "on_false",
        "on_true",
        "output",
    ),
    _switch_source_schema(
        "comfy.comfyui-kjnodes.LazySwitchKJ",
        "LazySwitchKJ",
        "switch",
        "on_false",
        "on_true",
        "*",
    ),
    _switch_source_schema(
        "comfy.comfyui-easy-use.easy ifElse",
        "easy ifElse",
        "boolean",
        "on_false",
        "on_true",
        "*",
    ),
    _switch_source_schema(
        "comfy.comfyui-impact-pack.ImpactConditionalBranch",
        "ImpactConditionalBranch",
        "cond",
        "ff_value",
        "tt_value",
        "*",
    ),
    NodeSchema(
        node_type="comfy.comfyui_essentials.SimpleComparison+",
        display_name="\U0001f527 Simple Comparison",
        category="comfy/essentials/utilities",
        inputs=(
            InputSpec("a", ANY, default=0),
            InputSpec("b", ANY, default=0),
            InputSpec(
                "comparison",
                COMBO,
                default="==",
                widget=ComboWidget(options=("==", "!=", "<", "<=", ">", ">=")),
            ),
        ),
        outputs=(OutputSpec("boolean", BOOLEAN),),
        aliases=("SimpleComparison+",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-easy-use.easy compare",
        display_name="Compare",
        category="comfy/EasyUse/Logic/Math",
        inputs=(
            InputSpec("a", ANY, required=False),
            InputSpec("b", ANY, required=False),
            InputSpec(
                "comparison",
                COMBO,
                required=False,
                default="a == b",
                widget=ComboWidget(
                    options=(
                        "a == b",
                        "a != b",
                        "a < b",
                        "a > b",
                        "a <= b",
                        "a >= b",
                        "a > 0",
                        "a <= 0",
                        "b > 0",
                        "b <= 0",
                    )
                ),
            ),
        ),
        outputs=(OutputSpec("boolean", BOOLEAN),),
        aliases=("easy compare",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-impact-pack.ImpactCompare",
        display_name="ImpactCompare",
        category="comfy/ImpactPack/Logic",
        inputs=(
            InputSpec(
                "cmp",
                COMBO,
                default="a = b",
                widget=ComboWidget(
                    options=(
                        "a = b",
                        "a <> b",
                        "a > b",
                        "a < b",
                        "a >= b",
                        "a <= b",
                        "tt",
                        "ff",
                    )
                ),
            ),
            InputSpec("a", ANY),
            InputSpec("b", ANY),
        ),
        outputs=(OutputSpec("boolean", BOOLEAN),),
        aliases=("ImpactCompare",),
    ),
    NodeSchema(
        node_type="comfy.was-node-suite-comfyui.Number Input Switch",
        display_name="Number Input Switch",
        category="comfy/WAS Suite/Logic",
        inputs=(
            InputSpec("number_a", NUMBER),
            InputSpec("number_b", NUMBER),
            InputSpec("boolean", BOOLEAN, force_input=True),
        ),
        outputs=(
            OutputSpec("number", NUMBER),
            OutputSpec("float", FLOAT),
            OutputSpec("int", INT),
        ),
        aliases=("Number Input Switch",),
    ),
    NodeSchema(
        node_type="comfy.comfy-mtb.Fit Number (mtb)",
        display_name="Fit Number (mtb)",
        category="comfy/mtb/math",
        description="Fit the input float using a source and target range",
        inputs=(
            InputSpec("value", FLOAT, default=0, force_input=True),
            InputSpec("clamp", BOOLEAN, default=False),
            InputSpec("source_min", FLOAT, default=0.0, widget=NumberWidget(min=-1e5, step=0.01)),
            InputSpec("source_max", FLOAT, default=1.0, widget=NumberWidget(min=-1e5, step=0.01)),
            InputSpec("target_min", FLOAT, default=0.0, widget=NumberWidget(min=-1e5, step=0.01)),
            InputSpec("target_max", FLOAT, default=1.0, widget=NumberWidget(min=-1e5, step=0.01)),
            InputSpec(
                "easing",
                COMBO,
                default="Linear",
                widget=ComboWidget(
                    options=(
                        "Linear",
                        "Sine In",
                        "Sine Out",
                        "Sine In/Out",
                        "Quart In",
                        "Quart Out",
                        "Quart In/Out",
                        "Cubic In",
                        "Cubic Out",
                        "Cubic In/Out",
                        "Circ In",
                        "Circ Out",
                        "Circ In/Out",
                        "Back In",
                        "Back Out",
                        "Back In/Out",
                        "Elastic In",
                        "Elastic Out",
                        "Elastic In/Out",
                        "Bounce In",
                        "Bounce Out",
                        "Bounce In/Out",
                    )
                ),
            ),
        ),
        outputs=(OutputSpec("float", FLOAT),),
        aliases=("Fit Number (mtb)",),
    ),
    NodeSchema(
        node_type="comfy.ComfyNumberConvert",
        display_name="Convert Number",
        category="comfy/utilities",
        inputs=(
            InputSpec("value", TypeExpr.union(CORE_INT, CORE_FLOAT, CORE_STRING, CORE_BOOLEAN)),
        ),
        outputs=(OutputSpec("FLOAT", FLOAT), OutputSpec("INT", INT)),
        aliases=("ComfyNumberConvert",),
    ),
    NodeSchema(
        node_type="comfy.comfy-mtb.Int To Bool (mtb)",
        display_name="Int To Bool (mtb)",
        category="comfy/mtb/number",
        inputs=(InputSpec("int", INT, default=0),),
        outputs=(OutputSpec("BOOLEAN", BOOLEAN),),
        aliases=("Int To Bool (mtb)",),
    ),
)

SOURCE_SCHEMAS += (
    NodeSchema(
        node_type="comfy.PrimitiveInt",
        display_name="Primitive Int",
        category="comfy/utilities/primitive",
        inputs=(InputSpec("value", INT, default=0),),
        outputs=(OutputSpec("_0_INT_", INT),),
        aliases=("PrimitiveInt",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-kjnodes.StringConstantMultiline",
        display_name="String Constant Multiline",
        category="comfy/KJNodes/constants",
        inputs=(
            InputSpec("string", STRING, default="", widget=StringWidget(multiline=True)),
            InputSpec("strip_newlines", BOOLEAN, default=True),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("StringConstantMultiline",),
    ),
    NodeSchema(
        node_type="comfy.was-node-suite-comfyui.Text Concatenate",
        display_name="Text Concatenate",
        category="comfy/WAS Suite/Text",
        inputs=(
            InputSpec("delimiter", STRING, default=", "),
            InputSpec(
                "clean_whitespace",
                BOOLEAN,
                default=True,
                widget=BooleanWidget(label_on="true", label_off="false"),
            ),
            InputSpec("text_a", STRING, required=False, force_input=True),
            InputSpec("text_b", STRING, required=False, force_input=True),
            InputSpec("text_c", STRING, required=False, force_input=True),
            InputSpec("text_d", STRING, required=False, force_input=True),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("Text Concatenate",),
    ),
    NodeSchema(
        node_type="comfy.StringConcatenate",
        display_name="Concatenate Text",
        category="comfy/text",
        inputs=(
            InputSpec("string_a", STRING, widget=StringWidget(multiline=True)),
            InputSpec("string_b", STRING, widget=StringWidget(multiline=True)),
            InputSpec(
                "delimiter",
                STRING,
                required=False,
                default="",
                widget=StringWidget(multiline=False),
            ),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("StringConcatenate",),
    ),
    NodeSchema(
        node_type="comfy.StringSubstring",
        display_name="Substring",
        category="comfy/text",
        inputs=(
            InputSpec("string", STRING, widget=StringWidget(multiline=True)),
            InputSpec("start", INT),
            InputSpec("end", INT),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("StringSubstring",),
    ),
    NodeSchema(
        node_type="comfy.StringLength",
        display_name="Text Length",
        category="comfy/text",
        inputs=(InputSpec("string", STRING, widget=StringWidget(multiline=True)),),
        outputs=(OutputSpec("length", INT),),
        aliases=("StringLength",),
    ),
    NodeSchema(
        node_type="comfy.CaseConverter",
        display_name="Convert Text Case",
        category="comfy/text",
        inputs=(
            InputSpec("string", STRING, widget=StringWidget(multiline=True)),
            InputSpec(
                "mode",
                COMBO,
                required=False,
                default="UPPERCASE",
                widget=ComboWidget(options=("UPPERCASE", "lowercase", "Capitalize", "Title Case")),
            ),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("CaseConverter",),
    ),
    NodeSchema(
        node_type="comfy.StringTrim",
        display_name="Trim Text",
        category="comfy/text",
        inputs=(
            InputSpec("string", STRING, widget=StringWidget(multiline=True)),
            InputSpec(
                "mode",
                COMBO,
                required=False,
                default="Both",
                widget=ComboWidget(options=("Both", "Left", "Right")),
            ),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("StringTrim",),
    ),
    NodeSchema(
        node_type="comfy.StringReplace",
        display_name="Replace Text",
        category="comfy/text",
        inputs=(
            InputSpec("string", STRING, widget=StringWidget(multiline=True)),
            InputSpec("find", STRING, widget=StringWidget(multiline=True)),
            InputSpec("replace", STRING, widget=StringWidget(multiline=True)),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("StringReplace",),
    ),
    NodeSchema(
        node_type="comfy.StringContains",
        display_name="Contains Text",
        category="comfy/text",
        inputs=(
            InputSpec("string", STRING, widget=StringWidget(multiline=True)),
            InputSpec("substring", STRING, widget=StringWidget(multiline=True)),
            InputSpec("case_sensitive", BOOLEAN, required=False, default=True, advanced=True),
        ),
        outputs=(OutputSpec("contains", BOOLEAN),),
        aliases=("StringContains",),
    ),
    NodeSchema(
        node_type="comfy.StringCompare",
        display_name="Compare Text",
        category="comfy/text",
        inputs=(
            InputSpec("string_a", STRING, widget=StringWidget(multiline=True)),
            InputSpec("string_b", STRING, widget=StringWidget(multiline=True)),
            InputSpec(
                "mode",
                COMBO,
                required=False,
                default="Starts With",
                widget=ComboWidget(options=("Starts With", "Ends With", "Equal")),
            ),
            InputSpec("case_sensitive", BOOLEAN, required=False, default=True, advanced=True),
        ),
        outputs=(OutputSpec("boolean", BOOLEAN),),
        aliases=("StringCompare",),
    ),
    NodeSchema(
        node_type="comfy.RegexMatch",
        display_name="Match Text",
        category="comfy/text",
        inputs=(
            InputSpec("string", STRING, widget=StringWidget(multiline=True)),
            InputSpec("regex_pattern", STRING, widget=StringWidget(multiline=True)),
            InputSpec("case_insensitive", BOOLEAN, required=False, default=True, advanced=True),
            InputSpec("multiline", BOOLEAN, required=False, default=False, advanced=True),
            InputSpec("dotall", BOOLEAN, required=False, default=False, advanced=True),
        ),
        outputs=(OutputSpec("matches", BOOLEAN),),
        aliases=("RegexMatch",),
    ),
    NodeSchema(
        node_type="comfy.RegexReplace",
        display_name="Replace Text (Regex)",
        category="comfy/text",
        description="Find and replace text using regex patterns.",
        inputs=(
            InputSpec("string", STRING, widget=StringWidget(multiline=True)),
            InputSpec("regex_pattern", STRING, widget=StringWidget(multiline=True)),
            InputSpec("replace", STRING, widget=StringWidget(multiline=True)),
            InputSpec("case_insensitive", BOOLEAN, required=False, default=True, advanced=True),
            InputSpec("multiline", BOOLEAN, required=False, default=False, advanced=True),
            InputSpec("dotall", BOOLEAN, required=False, default=False, advanced=True),
            InputSpec(
                "count",
                INT,
                required=False,
                default=0,
                widget=NumberWidget(min=0, max=100),
                advanced=True,
            ),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("RegexReplace",),
    ),
    NodeSchema(
        node_type="comfy.JsonExtractString",
        display_name="Extract Text from JSON",
        category="comfy/text",
        inputs=(
            InputSpec("json_string", STRING, widget=StringWidget(multiline=True)),
            InputSpec("key", STRING, widget=StringWidget(multiline=False)),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("JsonExtractString",),
    ),
    NodeSchema(
        node_type="comfy.ConvertDictionaryToString",
        display_name="Convert Dictionary to String",
        category="comfy/text",
        inputs=(
            InputSpec("dictionary", DICT),
            InputSpec(
                "indent",
                INT,
                required=False,
                default=2,
                widget=NumberWidget(min=0, max=8),
            ),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("ConvertDictionaryToString",),
    ),
    NodeSchema(
        node_type="comfy.ConvertArrayToString",
        display_name="Convert Array to String",
        category="comfy/text",
        inputs=(
            InputSpec("array", ARRAY),
            InputSpec(
                "indent",
                INT,
                required=False,
                default=2,
                widget=NumberWidget(min=0, max=8),
            ),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("ConvertArrayToString",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-kjnodes.StringConstant",
        display_name="String Constant",
        category="comfy/KJNodes/constants",
        inputs=(InputSpec("string", STRING, default="", widget=StringWidget(multiline=False)),),
        outputs=(OutputSpec("string", STRING),),
        aliases=("StringConstant",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-kjnodes.JoinStrings",
        display_name="Join Strings",
        category="comfy/KJNodes/text",
        inputs=(
            InputSpec("delimiter", STRING, default=" ", widget=StringWidget(multiline=False)),
            InputSpec("string1", STRING, required=False, default="", force_input=True),
            InputSpec("string2", STRING, required=False, default="", force_input=True),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("JoinStrings",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-easy-use.easy string",
        display_name="String",
        category="comfy/EasyUse/Logic/Type",
        inputs=(
            InputSpec(
                "value",
                STRING,
                required=False,
                default="",
                widget=StringWidget(multiline=False),
            ),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("easy string",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-easy-use.easy mathString",
        display_name="Math String",
        category="comfy/EasyUse/Logic/Math",
        inputs=(
            InputSpec("a", STRING, widget=StringWidget(multiline=False)),
            InputSpec("b", STRING, widget=StringWidget(multiline=False)),
            InputSpec(
                "operation",
                COMBO,
                required=False,
                default="a == b",
                widget=ComboWidget(
                    options=(
                        "a == b",
                        "a != b",
                        "a IN b",
                        "a MATCH REGEX(b)",
                        "a BEGINSWITH b",
                        "a ENDSWITH b",
                    )
                ),
            ),
            InputSpec("case_sensitive", BOOLEAN, required=False, default=True),
        ),
        outputs=(OutputSpec("BOOLEAN", BOOLEAN),),
        aliases=("easy mathString",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-easy-use.easy stringJoinLines",
        display_name="String Join Lines",
        category="comfy/EasyUse/Logic",
        inputs=(
            InputSpec(
                "string",
                STRING,
                required=False,
                default="",
                widget=StringWidget(multiline=True),
            ),
            InputSpec(
                "delimiter",
                STRING,
                required=False,
                default=" | ",
                widget=StringWidget(multiline=False),
            ),
        ),
        outputs=(OutputSpec("STRING", STRING),),
        aliases=("easy stringJoinLines",),
    ),
    NodeSchema(
        node_type="comfy.comfyui-impact-pack.ImpactStringSelector",
        display_name="String Selector",
        category="comfy/ImpactPack/Util",
        inputs=(
            InputSpec("strings", STRING, widget=StringWidget(multiline=True)),
            InputSpec(
                "multiline",
                BOOLEAN,
                default=False,
                widget=BooleanWidget(label_on="enabled", label_off="disabled"),
            ),
            InputSpec(
                "select",
                INT,
                default=0,
                widget=NumberWidget(min=0, step=1),
            ),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("ImpactStringSelector",),
    ),
    NodeSchema(
        node_type="comfy.was-node-suite-comfyui.Text Contains",
        display_name="Text Contains",
        category="comfy/WAS Suite/Logic",
        inputs=(
            InputSpec("text", STRING, default="", widget=StringWidget(multiline=False)),
            InputSpec("sub_text", STRING, default="", widget=StringWidget(multiline=False)),
            InputSpec("case_insensitive", BOOLEAN, required=False, default=True),
        ),
        outputs=(OutputSpec("boolean", BOOLEAN),),
        aliases=("Text Contains",),
    ),
    NodeSchema(
        node_type="comfy.comfy-mtb.String Replace (mtb)",
        display_name="String Replace (mtb)",
        category="comfy/mtb/string",
        description="Basic string replacement with regex support.",
        inputs=(
            InputSpec("string", STRING, force_input=True),
            InputSpec("old", STRING, default=""),
            InputSpec("new", STRING, default=""),
            InputSpec("use_regex", BOOLEAN, default=False),
        ),
        outputs=(OutputSpec("string", STRING),),
        aliases=("String Replace (mtb)",),
    ),
)


def _comparison_case(
    source_operation: str,
    operations: dict[str, str],
    *,
    when: ReplacementPredicate | None = None,
    a: MappingSource | None = None,
    b: MappingSource | None = None,
) -> ReplacementCase:
    return ReplacementCase.build(
        "dinkster.value.compare",
        when=when,
        inputs={
            "a": a or MappingSource.copy("a"),
            "b": b or MappingSource.copy("b"),
            "operation": MappingSource.from_value(
                source_operation, ValueTransform.enum_rename(operations)
            ),
            "epsilon": MappingSource.constant(0.0),
        },
        outputs={"result": "boolean"},
    )


def _fixed_comparison_case(
    operation: str,
    *,
    when: ReplacementPredicate,
    a: MappingSource,
    b: MappingSource,
) -> ReplacementCase:
    return ReplacementCase.build(
        "dinkster.value.compare",
        when=when,
        inputs={
            "a": a,
            "b": b,
            "operation": MappingSource.constant(operation),
            "epsilon": MappingSource.constant(0.0),
        },
        outputs={"result": "boolean"},
    )


def _transform_case(
    operation: MappingSource,
    *,
    text: MappingSource,
    output: str = "string",
    when: ReplacementPredicate | None = None,
    value: MappingSource | None = None,
    replacement: MappingSource | None = None,
    start: MappingSource | None = None,
    end: MappingSource | None = None,
    count: MappingSource | None = None,
    fill: MappingSource | None = None,
    unit: MappingSource | None = None,
    side: MappingSource | None = None,
) -> ReplacementCase:
    return ReplacementCase.build(
        "dinkster.string.transform",
        when=when,
        inputs={
            "text": text,
            "operation": operation,
            "value": value or MappingSource.constant(""),
            "replacement": replacement or MappingSource.constant(""),
            "start": start or MappingSource.constant(0),
            "end": end or MappingSource.constant((1 << 53) - 1),
            "count": count or MappingSource.constant(0),
            "fill": fill or MappingSource.constant(" "),
            "unit": unit or MappingSource.constant("characters"),
            "side": side or MappingSource.constant("start"),
        },
        outputs={"text": output},
    )


def _test_case(
    operation: MappingSource,
    *,
    text: MappingSource,
    query: MappingSource,
    output: str,
    case_mode: str,
    when: ReplacementPredicate | None = None,
) -> ReplacementCase:
    return ReplacementCase.build(
        "dinkster.string.test",
        when=when,
        inputs={
            "text": text,
            "query": query,
            "operation": operation,
            "case_mode": MappingSource.constant(case_mode),
        },
        outputs={"result": output},
    )


def _regex_case(
    operation: MappingSource,
    *,
    text: MappingSource,
    pattern: MappingSource,
    outputs: dict[str, str],
    case_mode: str,
    when: ReplacementPredicate | None = None,
    replacement: MappingSource | None = None,
    group_index: MappingSource | None = None,
    count: MappingSource | None = None,
    multiline: MappingSource | None = None,
    dotall: MappingSource | None = None,
) -> ReplacementCase:
    return ReplacementCase.build(
        "dinkster.string.regex",
        when=when,
        inputs={
            "text": text,
            "pattern": pattern,
            "operation": operation,
            "replacement": replacement or MappingSource.constant(""),
            "group_index": group_index or MappingSource.constant(1),
            "count": count or MappingSource.constant(0),
            "case_mode": MappingSource.constant(case_mode),
            "multiline": multiline or MappingSource.constant(False),
            "dotall": dotall or MappingSource.constant(False),
        },
        outputs=outputs,
    )


def _record(
    *,
    source_pack: str,
    node_class: str,
    revision: str,
    carrier: str,
    rule: ReplacementRule,
    tier: str,
    evidence: tuple[str, ...] = NUMERIC_EVIDENCE,
) -> dict[str, object]:
    return {
        "id": f"comfy_alias:{source_pack}/{node_class}",
        "mappingKind": "op",
        "carrier": carrier,
        "source": {
            "pack": source_pack,
            "nodeClass": node_class,
            "nodeType": rule.from_type,
            "revision": revision,
        },
        "replacement": rule_to_wire(rule),
        "confidence": {"tier": tier, "evidence": list(evidence)},
    }


def _switch_record(
    *,
    source_pack: str,
    node_class: str,
    revision: str,
    source_type: str,
    condition: str,
    on_false: str,
    on_true: str,
    source_output: str,
) -> dict[str, object]:
    return _record(
        source_pack=source_pack,
        node_class=node_class,
        revision=revision,
        carrier="dinkster.value.select",
        rule=ReplacementRule(
            from_type=source_type,
            cases=(
                ReplacementCase.build(
                    "dinkster.value.select",
                    inputs={
                        "condition": MappingSource.copy(condition),
                        "on_false": MappingSource.copy(on_false),
                        "on_true": MappingSource.copy(on_true),
                    },
                    outputs={"value": source_output},
                ),
            ),
        ),
        tier="exact",
        evidence=ROUTING_EVIDENCE,
    )


def _bool_logic_record(node_class: str, operation: str) -> dict[str, object]:
    return _record(
        source_pack="comfy-core",
        node_class=node_class,
        revision=COMFY_REVISION,
        carrier="dinkster.bool.logic",
        rule=ReplacementRule(
            from_type=f"comfy.{node_class}",
            cases=(
                ReplacementCase.build(
                    "dinkster.bool.logic",
                    inputs={"operation": MappingSource.constant(operation)},
                    input_families={
                        "values": InputFamilyMapping.copy(
                            "values",
                            inputs={"value": MappingSource.copy("value")},
                        )
                    },
                    outputs={"result": "boolean"},
                ),
            ),
            note=(
                "Inputs must resolve to native booleans; legacy Python truthiness for other "
                "values is not preserved."
            ),
        ),
        tier="parametric",
    )


def _string_record(
    *,
    source_pack: str,
    node_class: str,
    revision: str,
    carrier: str,
    source_type: str,
    cases: tuple[ReplacementCase, ...],
    note: str = "",
) -> dict[str, object]:
    return _record(
        source_pack=source_pack,
        node_class=node_class,
        revision=revision,
        carrier=carrier,
        rule=ReplacementRule(from_type=source_type, cases=cases, note=note),
        tier="parametric",
        evidence=STRING_EVIDENCE,
    )


def _was_text_concatenate_cases() -> tuple[ReplacementCase, ...]:
    source_inputs = ("text_a", "text_b", "text_c", "text_d")
    member_combinations = tuple(
        members
        for count in range(len(source_inputs), -1, -1)
        for members in combinations(source_inputs, count)
    )
    cases: list[ReplacementCase] = []
    for members in member_combinations:
        family_mapping = (
            {
                "pieces": InputFamilyMapping.from_members(
                    *(
                        InputFamilyMember.build(
                            input_id,
                            inputs={"value": MappingSource.copy(input_id)},
                        )
                        for input_id in members
                    )
                )
            }
            if members
            else None
        )
        cases.append(
            ReplacementCase.build(
                "dinkster.string.join",
                when=(
                    ReplacementPredicate.all_of(
                        *(
                            ReplacementPredicate.any_of(
                                ReplacementPredicate.input_connected(input_id),
                                ReplacementPredicate.value_present(input_id),
                            )
                            for input_id in members
                        )
                    )
                    if members
                    else None
                ),
                inputs={
                    "separator": MappingSource.copy("delimiter"),
                    "trim": MappingSource.copy("clean_whitespace"),
                    "skip_empty": MappingSource.constant(True),
                    "delimiter_escape": MappingSource.constant("newline"),
                },
                input_families=family_mapping,
                outputs={"text": "string"},
            )
        )
    return tuple(cases)


def _string_records() -> list[dict[str, object]]:
    core_case_sensitive = ReplacementPredicate.value_equals("case_sensitive", False)
    core_regex_sensitive = ReplacementPredicate.value_equals("case_insensitive", False)

    core_string_compare_operation = MappingSource.from_value(
        "mode",
        ValueTransform.enum_rename(
            {
                "Starts With": "starts_with",
                "Ends With": "ends_with",
                "Equal": "equals",
            }
        ),
    )

    available_string1 = ReplacementPredicate.any_of(
        ReplacementPredicate.input_connected("string1"),
        ReplacementPredicate.value_present("string1"),
    )
    available_string2 = ReplacementPredicate.any_of(
        ReplacementPredicate.input_connected("string2"),
        ReplacementPredicate.value_present("string2"),
    )

    easy_regex = ReplacementPredicate.value_equals("operation", "a MATCH REGEX(b)")
    easy_contains = ReplacementPredicate.value_equals("operation", "a IN b")
    easy_case_insensitive = ReplacementPredicate.value_equals("case_sensitive", False)
    easy_test_operation = MappingSource.from_value(
        "operation",
        ValueTransform.enum_rename(
            {
                "a == b": "equals",
                "a != b": "not_equals",
                "a BEGINSWITH b": "starts_with",
                "a ENDSWITH b": "ends_with",
            }
        ),
    )

    return [
        _string_record(
            source_pack="comfy-core",
            node_class="StringConcatenate",
            revision=COMFY_REVISION,
            carrier="std.string.concat",
            source_type="comfy.StringConcatenate",
            cases=(
                ReplacementCase.build(
                    "std.string.concat",
                    inputs={
                        "a": MappingSource.copy("string_a"),
                        "b": MappingSource.copy("string_b"),
                        "separator": MappingSource.copy("delimiter"),
                    },
                    outputs={"text": "string"},
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="StringSubstring",
            revision=COMFY_REVISION,
            carrier="dinkster.string.transform",
            source_type="comfy.StringSubstring",
            cases=(
                _transform_case(
                    MappingSource.constant("slice"),
                    text=MappingSource.copy("string"),
                    start=MappingSource.copy("start"),
                    end=MappingSource.copy("end"),
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="StringLength",
            revision=COMFY_REVISION,
            carrier="dinkster.string.length",
            source_type="comfy.StringLength",
            cases=(
                ReplacementCase.build(
                    "dinkster.string.length",
                    inputs={"text": MappingSource.copy("string")},
                    outputs={"length": "length"},
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="CaseConverter",
            revision=COMFY_REVISION,
            carrier="dinkster.string.transform",
            source_type="comfy.CaseConverter",
            cases=(
                _transform_case(
                    MappingSource.from_value(
                        "mode",
                        ValueTransform.enum_rename(
                            {
                                "UPPERCASE": "upper",
                                "lowercase": "lower",
                                "Capitalize": "capitalize",
                                "Title Case": "title",
                            }
                        ),
                    ),
                    text=MappingSource.copy("string"),
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="StringTrim",
            revision=COMFY_REVISION,
            carrier="dinkster.string.transform",
            source_type="comfy.StringTrim",
            cases=(
                _transform_case(
                    MappingSource.from_value(
                        "mode",
                        ValueTransform.enum_rename(
                            {
                                "Both": "trim",
                                "Left": "trim_left",
                                "Right": "trim_right",
                            }
                        ),
                    ),
                    text=MappingSource.copy("string"),
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="StringReplace",
            revision=COMFY_REVISION,
            carrier="dinkster.string.transform",
            source_type="comfy.StringReplace",
            cases=(
                _transform_case(
                    MappingSource.constant("replace_literal"),
                    text=MappingSource.copy("string"),
                    value=MappingSource.copy("find"),
                    replacement=MappingSource.copy("replace"),
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="StringContains",
            revision=COMFY_REVISION,
            carrier="dinkster.string.test",
            source_type="comfy.StringContains",
            cases=(
                _test_case(
                    MappingSource.constant("contains"),
                    text=MappingSource.copy("string"),
                    query=MappingSource.copy("substring"),
                    output="contains",
                    case_mode="lower_both",
                    when=core_case_sensitive,
                ),
                _test_case(
                    MappingSource.constant("contains"),
                    text=MappingSource.copy("string"),
                    query=MappingSource.copy("substring"),
                    output="contains",
                    case_mode="sensitive",
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="StringCompare",
            revision=COMFY_REVISION,
            carrier="dinkster.string.test",
            source_type="comfy.StringCompare",
            cases=(
                _test_case(
                    core_string_compare_operation,
                    text=MappingSource.copy("string_a"),
                    query=MappingSource.copy("string_b"),
                    output="boolean",
                    case_mode="lower_both",
                    when=core_case_sensitive,
                ),
                _test_case(
                    core_string_compare_operation,
                    text=MappingSource.copy("string_a"),
                    query=MappingSource.copy("string_b"),
                    output="boolean",
                    case_mode="sensitive",
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="RegexMatch",
            revision=COMFY_REVISION,
            carrier="dinkster.string.regex",
            source_type="comfy.RegexMatch",
            cases=(
                _regex_case(
                    MappingSource.constant("search"),
                    text=MappingSource.copy("string"),
                    pattern=MappingSource.copy("regex_pattern"),
                    outputs={"matched": "matches"},
                    case_mode="sensitive",
                    when=core_regex_sensitive,
                    multiline=MappingSource.copy("multiline"),
                    dotall=MappingSource.copy("dotall"),
                ),
                _regex_case(
                    MappingSource.constant("search"),
                    text=MappingSource.copy("string"),
                    pattern=MappingSource.copy("regex_pattern"),
                    outputs={"matched": "matches"},
                    case_mode="unicode_ignorecase",
                    multiline=MappingSource.copy("multiline"),
                    dotall=MappingSource.copy("dotall"),
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="RegexReplace",
            revision=COMFY_REVISION,
            carrier="dinkster.string.regex",
            source_type="comfy.RegexReplace",
            cases=(
                _regex_case(
                    MappingSource.constant("replace"),
                    text=MappingSource.copy("string"),
                    pattern=MappingSource.copy("regex_pattern"),
                    outputs={"text": "string"},
                    case_mode="sensitive",
                    when=core_regex_sensitive,
                    replacement=MappingSource.copy("replace"),
                    count=MappingSource.copy("count"),
                    multiline=MappingSource.copy("multiline"),
                    dotall=MappingSource.copy("dotall"),
                ),
                _regex_case(
                    MappingSource.constant("replace"),
                    text=MappingSource.copy("string"),
                    pattern=MappingSource.copy("regex_pattern"),
                    outputs={"text": "string"},
                    case_mode="unicode_ignorecase",
                    replacement=MappingSource.copy("replace"),
                    count=MappingSource.copy("count"),
                    multiline=MappingSource.copy("multiline"),
                    dotall=MappingSource.copy("dotall"),
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="JsonExtractString",
            revision=COMFY_REVISION,
            carrier="dinkster.string.json",
            source_type="comfy.JsonExtractString",
            cases=(
                ReplacementCase.build(
                    "dinkster.string.json",
                    inputs={
                        "operation": MappingSource.constant("extract_string_legacy"),
                        "text": MappingSource.copy("json_string"),
                        "selector": MappingSource.copy("key"),
                        "indent": MappingSource.constant(2),
                        "key_order": MappingSource.constant("preserve"),
                    },
                    outputs={"text": "string"},
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="ConvertDictionaryToString",
            revision=COMFY_REVISION,
            carrier="dinkster.string.json_emit",
            source_type="comfy.ConvertDictionaryToString",
            cases=(
                ReplacementCase.build(
                    "dinkster.string.json_emit",
                    inputs={
                        "value": MappingSource.copy("dictionary"),
                        "operation": MappingSource.constant("legacy"),
                        "indent": MappingSource.copy("indent"),
                        "key_order": MappingSource.constant("preserve"),
                    },
                    outputs={"text": "string"},
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-core",
            node_class="ConvertArrayToString",
            revision=COMFY_REVISION,
            carrier="dinkster.string.json_emit",
            source_type="comfy.ConvertArrayToString",
            cases=(
                ReplacementCase.build(
                    "dinkster.string.json_emit",
                    inputs={
                        "value": MappingSource.copy("array"),
                        "operation": MappingSource.constant("legacy"),
                        "indent": MappingSource.copy("indent"),
                        "key_order": MappingSource.constant("preserve"),
                    },
                    outputs={"text": "string"},
                ),
            ),
        ),
        _string_record(
            source_pack="comfyui-kjnodes",
            node_class="StringConstant",
            revision=KJ_REVISION,
            carrier="dinkster.string",
            source_type="comfy.comfyui-kjnodes.StringConstant",
            cases=(
                ReplacementCase.build(
                    "dinkster.string",
                    inputs={"value": MappingSource.copy("string")},
                    outputs={"value": "string"},
                ),
            ),
        ),
        _string_record(
            source_pack="comfyui-kjnodes",
            node_class="StringConstantMultiline",
            revision=KJ_REVISION,
            carrier="dinkster.string.transform",
            source_type="comfy.comfyui-kjnodes.StringConstantMultiline",
            cases=(
                ReplacementCase.build(
                    "dinkster.string_multiline",
                    when=ReplacementPredicate.value_equals("strip_newlines", False),
                    inputs={"value": MappingSource.copy("string")},
                    outputs={"value": "string"},
                ),
                ReplacementCase.build(
                    "dinkster.string.transform",
                    nodes={
                        "without_newlines": ReplacementNode.build(
                            "dinkster.string.transform",
                            values={
                                "operation": "replace_literal",
                                "value": "\n",
                                "replacement": "",
                            },
                        )
                    },
                    inputs={
                        "operation": MappingSource.constant("trim"),
                        "without_newlines:text": MappingSource.copy("string"),
                    },
                    links=(ReplacementLink("without_newlines:text", "text"),),
                    outputs={"text": "string"},
                ),
            ),
            note="The enabled path removes LF characters before trimming surrounding whitespace.",
        ),
        _string_record(
            source_pack="comfyui-kjnodes",
            node_class="JoinStrings",
            revision=KJ_REVISION,
            carrier="std.string.concat",
            source_type="comfy.comfyui-kjnodes.JoinStrings",
            cases=(
                ReplacementCase.build(
                    "std.string.concat",
                    when=ReplacementPredicate.all_of(available_string1, available_string2),
                    inputs={
                        "a": MappingSource.copy("string1"),
                        "b": MappingSource.copy("string2"),
                        "separator": MappingSource.copy("delimiter"),
                    },
                    outputs={"text": "string"},
                ),
                ReplacementCase.build(
                    "std.string.concat",
                    when=available_string1,
                    inputs={
                        "a": MappingSource.copy("string1"),
                        "b": MappingSource.constant(""),
                        "separator": MappingSource.copy("delimiter"),
                    },
                    outputs={"text": "string"},
                ),
                ReplacementCase.build(
                    "std.string.concat",
                    when=available_string2,
                    inputs={
                        "a": MappingSource.constant(""),
                        "b": MappingSource.copy("string2"),
                        "separator": MappingSource.copy("delimiter"),
                    },
                    outputs={"text": "string"},
                ),
                ReplacementCase.build(
                    "std.string.concat",
                    inputs={
                        "a": MappingSource.constant(""),
                        "b": MappingSource.constant(""),
                        "separator": MappingSource.copy("delimiter"),
                    },
                    outputs={"text": "string"},
                ),
            ),
        ),
        _string_record(
            source_pack="comfyui-easy-use",
            node_class="easy string",
            revision=EASY_USE_STRING_REVISION,
            carrier="dinkster.string",
            source_type="comfy.comfyui-easy-use.easy string",
            cases=(
                ReplacementCase.build(
                    "dinkster.string",
                    inputs={"value": MappingSource.copy("value")},
                    outputs={"value": "string"},
                ),
            ),
        ),
        _string_record(
            source_pack="comfyui-easy-use",
            node_class="easy mathString",
            revision=EASY_USE_STRING_REVISION,
            carrier="dinkster.string.test",
            source_type="comfy.comfyui-easy-use.easy mathString",
            cases=(
                _regex_case(
                    MappingSource.constant("match"),
                    text=MappingSource.copy("a"),
                    pattern=MappingSource.copy("b"),
                    outputs={"matched": "BOOLEAN"},
                    case_mode="lower_both",
                    when=ReplacementPredicate.all_of(easy_regex, easy_case_insensitive),
                ),
                _regex_case(
                    MappingSource.constant("match"),
                    text=MappingSource.copy("a"),
                    pattern=MappingSource.copy("b"),
                    outputs={"matched": "BOOLEAN"},
                    case_mode="sensitive",
                    when=easy_regex,
                ),
                _test_case(
                    MappingSource.constant("contains"),
                    text=MappingSource.copy("b"),
                    query=MappingSource.copy("a"),
                    output="BOOLEAN",
                    case_mode="lower_both",
                    when=ReplacementPredicate.all_of(easy_contains, easy_case_insensitive),
                ),
                _test_case(
                    MappingSource.constant("contains"),
                    text=MappingSource.copy("b"),
                    query=MappingSource.copy("a"),
                    output="BOOLEAN",
                    case_mode="sensitive",
                    when=easy_contains,
                ),
                _test_case(
                    easy_test_operation,
                    text=MappingSource.copy("a"),
                    query=MappingSource.copy("b"),
                    output="BOOLEAN",
                    case_mode="lower_both",
                    when=easy_case_insensitive,
                ),
                _test_case(
                    easy_test_operation,
                    text=MappingSource.copy("a"),
                    query=MappingSource.copy("b"),
                    output="BOOLEAN",
                    case_mode="sensitive",
                ),
            ),
        ),
        _string_record(
            source_pack="comfyui-easy-use",
            node_class="easy stringJoinLines",
            revision=EASY_USE_STRING_REVISION,
            carrier="dinkster.string.transform",
            source_type="comfy.comfyui-easy-use.easy stringJoinLines",
            cases=(
                _transform_case(
                    MappingSource.constant("join_nonempty_lines"),
                    text=MappingSource.copy("string"),
                    value=MappingSource.copy("delimiter"),
                    output="STRING",
                ),
            ),
        ),
        _string_record(
            source_pack="comfyui-impact-pack",
            node_class="ImpactStringSelector",
            revision=IMPACT_REVISION,
            carrier="dinkster.string.transform",
            source_type="comfy.comfyui-impact-pack.ImpactStringSelector",
            cases=(
                _transform_case(
                    MappingSource.constant("select_hash_section"),
                    text=MappingSource.copy("strings"),
                    count=MappingSource.copy("select"),
                    when=ReplacementPredicate.value_equals("multiline", True),
                ),
                _transform_case(
                    MappingSource.constant("select_line"),
                    text=MappingSource.copy("strings"),
                    count=MappingSource.copy("select"),
                ),
            ),
        ),
        _string_record(
            source_pack="was-node-suite-comfyui",
            node_class="Text Concatenate",
            revision=WAS_REVISION,
            carrier="dinkster.string.join",
            source_type="comfy.was-node-suite-comfyui.Text Concatenate",
            cases=_was_text_concatenate_cases(),
            note=(
                "Cases preserve optional socket presence and text_a-through-text_d order; "
                "trim, empty filtering, and newline delimiter decoding match the source."
            ),
        ),
        _string_record(
            source_pack="was-node-suite-comfyui",
            node_class="Text Contains",
            revision=WAS_REVISION,
            carrier="dinkster.string.test",
            source_type="comfy.was-node-suite-comfyui.Text Contains",
            cases=(
                _test_case(
                    MappingSource.constant("contains"),
                    text=MappingSource.copy("text"),
                    query=MappingSource.copy("sub_text"),
                    output="boolean",
                    case_mode="sensitive",
                    when=ReplacementPredicate.value_equals("case_insensitive", False),
                ),
                _test_case(
                    MappingSource.constant("contains"),
                    text=MappingSource.copy("text"),
                    query=MappingSource.copy("sub_text"),
                    output="boolean",
                    case_mode="lower_both",
                ),
            ),
        ),
        _string_record(
            source_pack="comfy-mtb",
            node_class="String Replace (mtb)",
            revision=MTB_REVISION,
            carrier="dinkster.string.transform",
            source_type="comfy.comfy-mtb.String Replace (mtb)",
            cases=(
                _regex_case(
                    MappingSource.constant("replace"),
                    text=MappingSource.copy("string"),
                    pattern=MappingSource.copy("old"),
                    outputs={"text": "string"},
                    case_mode="sensitive",
                    when=ReplacementPredicate.value_equals("use_regex", True),
                    replacement=MappingSource.copy("new"),
                ),
                _transform_case(
                    MappingSource.constant("replace_literal"),
                    text=MappingSource.copy("string"),
                    value=MappingSource.copy("old"),
                    replacement=MappingSource.copy("new"),
                ),
            ),
        ),
    ]


def _records() -> list[dict[str, object]]:
    easy_operations = {
        "a == b": "eq",
        "a != b": "ne",
        "a < b": "lt",
        "a <= b": "le",
        "a > b": "gt",
        "a >= b": "ge",
    }
    impact_operations = {
        "a = b": "eq",
        "a <> b": "ne",
        "a < b": "lt",
        "a <= b": "le",
        "a > b": "gt",
        "a >= b": "ge",
    }
    simple_operations = {
        "==": "eq",
        "!=": "ne",
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
    }

    available_a = ReplacementPredicate.any_of(
        ReplacementPredicate.input_connected("a"),
        ReplacementPredicate.value_present("a"),
    )
    available_b = ReplacementPredicate.any_of(
        ReplacementPredicate.input_connected("b"),
        ReplacementPredicate.value_present("b"),
    )
    easy_unary_cases = tuple(
        case
        for source_operation, operation, operand, available in (
            ("a > 0", "gt", "a", available_a),
            ("a <= 0", "le", "a", available_a),
            ("b > 0", "gt", "b", available_b),
            ("b <= 0", "le", "b", available_b),
        )
        for case in (
            _fixed_comparison_case(
                operation,
                when=ReplacementPredicate.all_of(
                    ReplacementPredicate.value_equals("comparison", source_operation),
                    available,
                ),
                a=MappingSource.copy(operand),
                b=MappingSource.constant(0),
            ),
            _fixed_comparison_case(
                operation,
                when=ReplacementPredicate.value_equals("comparison", source_operation),
                a=MappingSource.constant(0),
                b=MappingSource.constant(0),
            ),
        )
    )
    easy_rule = ReplacementRule(
        from_type="comfy.comfyui-easy-use.easy compare",
        note=(
            "Omitted operands retain the legacy zero default. Inputs must resolve to a type "
            "accepted by dinkster.value.compare."
        ),
        cases=(
            *easy_unary_cases,
            _comparison_case(
                "comparison",
                easy_operations,
                when=ReplacementPredicate.all_of(available_a, available_b),
            ),
            _comparison_case(
                "comparison",
                easy_operations,
                when=available_a,
                b=MappingSource.constant(0),
            ),
            _comparison_case(
                "comparison",
                easy_operations,
                when=available_b,
                a=MappingSource.constant(0),
            ),
            _comparison_case(
                "comparison",
                easy_operations,
                a=MappingSource.constant(0),
                b=MappingSource.constant(0),
            ),
        ),
    )

    records = [
        _record(
            source_pack="comfy-core",
            node_class="PrimitiveInt",
            revision=TRELLIS2_REVISION,
            carrier="dinkster.int",
            rule=ReplacementRule(
                from_type="comfy.PrimitiveInt",
                cases=(
                    ReplacementCase.build(
                        "dinkster.int",
                        inputs={"value": MappingSource.copy("value")},
                        outputs={"value": "_0_INT_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=TRELLIS2_WORKFLOW_EVIDENCE,
        ),
        _record(
            source_pack="comfy-core",
            node_class="PrimitiveBoolean",
            revision="b78cec87",
            carrier="dinkster.boolean",
            rule=ReplacementRule(
                from_type="comfy.PrimitiveBoolean",
                cases=(
                    ReplacementCase.build(
                        "dinkster.boolean",
                        inputs={"value": MappingSource.copy("value")},
                        outputs={"value": "boolean"},
                    ),
                ),
            ),
            tier="exact",
        ),
        _record(
            source_pack="comfy-core",
            node_class="ComfyMathExpression",
            revision=COMFY_REVISION,
            carrier="dinkster.math.expression",
            rule=ReplacementRule(
                from_type="comfy.ComfyMathExpression",
                cases=(
                    ReplacementCase.build(
                        "dinkster.math.expression",
                        inputs={"expression": MappingSource.copy("expression")},
                        input_families={
                            "values": InputFamilyMapping.copy(
                                "values",
                                inputs={"value": MappingSource.copy("value")},
                            )
                        },
                        outputs={
                            "float": "FLOAT",
                            "int": "INT",
                            "boolean": "BOOL",
                        },
                    ),
                ),
            ),
            tier="parametric",
        ),
        _bool_logic_record("ComfyAndNode", "and"),
        _bool_logic_record("ComfyOrNode", "or"),
        _switch_record(
            source_pack="comfy-core",
            node_class="ComfySwitchNode",
            revision=COMFY_REVISION,
            source_type="comfy.ComfySwitchNode",
            condition="switch",
            on_false="on_false",
            on_true="on_true",
            source_output="output",
        ),
        _switch_record(
            source_pack="comfyui-kjnodes",
            node_class="LazySwitchKJ",
            revision=KJ_REVISION,
            source_type="comfy.comfyui-kjnodes.LazySwitchKJ",
            condition="switch",
            on_false="on_false",
            on_true="on_true",
            source_output="*",
        ),
        _switch_record(
            source_pack="comfyui-easy-use",
            node_class="easy ifElse",
            revision=EASY_USE_REVISION,
            source_type="comfy.comfyui-easy-use.easy ifElse",
            condition="boolean",
            on_false="on_false",
            on_true="on_true",
            source_output="*",
        ),
        _switch_record(
            source_pack="comfyui-impact-pack",
            node_class="ImpactConditionalBranch",
            revision=IMPACT_REVISION,
            source_type="comfy.comfyui-impact-pack.ImpactConditionalBranch",
            condition="cond",
            on_false="ff_value",
            on_true="tt_value",
            source_output="*",
        ),
        _record(
            source_pack="comfyui_essentials",
            node_class="SimpleComparison+",
            revision=ESSENTIALS_REVISION,
            carrier="dinkster.value.compare",
            rule=ReplacementRule(
                from_type="comfy.comfyui_essentials.SimpleComparison+",
                note="Inputs must resolve to a type accepted by dinkster.value.compare.",
                cases=(_comparison_case("comparison", simple_operations),),
            ),
            tier="parametric",
        ),
        _record(
            source_pack="comfyui-easy-use",
            node_class="easy compare",
            revision=EASY_USE_REVISION,
            carrier="dinkster.value.compare",
            rule=easy_rule,
            tier="parametric",
        ),
        _record(
            source_pack="comfyui-impact-pack",
            node_class="ImpactCompare",
            revision=IMPACT_REVISION,
            carrier="dinkster.value.compare",
            rule=ReplacementRule(
                from_type="comfy.comfyui-impact-pack.ImpactCompare",
                note="Inputs must resolve to a type accepted by dinkster.value.compare.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.boolean",
                        when=ReplacementPredicate.value_equals("cmp", "tt"),
                        inputs={"value": MappingSource.constant(True)},
                        outputs={"value": "boolean"},
                    ),
                    ReplacementCase.build(
                        "dinkster.boolean",
                        when=ReplacementPredicate.value_equals("cmp", "ff"),
                        inputs={"value": MappingSource.constant(False)},
                        outputs={"value": "boolean"},
                    ),
                    _comparison_case("cmp", impact_operations),
                ),
            ),
            tier="parametric",
        ),
        _record(
            source_pack="was-node-suite-comfyui",
            node_class="Number Input Switch",
            revision=WAS_REVISION,
            carrier="dinkster.value.select",
            rule=ReplacementRule(
                from_type="comfy.was-node-suite-comfyui.Number Input Switch",
                note=(
                    "Maps the primary NUMBER output; FLOAT and INT output links require "
                    "explicit conversions. Both arms must resolve to one native type."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.value.select",
                        inputs={
                            "condition": MappingSource.copy("boolean"),
                            "on_false": MappingSource.copy("number_b"),
                            "on_true": MappingSource.copy("number_a"),
                        },
                        outputs={"value": "number"},
                    ),
                ),
            ),
            tier="parametric",
        ),
        _record(
            source_pack="comfy-mtb",
            node_class="Fit Number (mtb)",
            revision=MTB_REVISION,
            carrier="dinkster.value.remap",
            rule=ReplacementRule(
                from_type="comfy.comfy-mtb.Fit Number (mtb)",
                note=(
                    "Linear easing only. Equal source endpoints returned target_min in the "
                    "source and are rejected by dinkster.value.remap."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.value.remap",
                        inputs={
                            "value": MappingSource.copy("value"),
                            "input_minimum": MappingSource.copy("source_min"),
                            "input_maximum": MappingSource.copy("source_max"),
                            "output_minimum": MappingSource.copy("target_min"),
                            "output_maximum": MappingSource.copy("target_max"),
                            "clamp": MappingSource.copy("clamp"),
                            "curve": MappingSource.from_value(
                                "easing", ValueTransform.enum_rename({"Linear": "linear"})
                            ),
                        },
                        outputs={"value": "float"},
                    ),
                ),
            ),
            tier="parametric",
        ),
        _record(
            source_pack="comfy-core",
            node_class="ComfyNumberConvert",
            revision=COMFY_REVISION,
            carrier="dinkster.value.convert",
            rule=ReplacementRule(
                from_type="comfy.ComfyNumberConvert",
                note=(
                    "Selects the numeric-pair conversion and enables lossy conversion to preserve "
                    "the source's numeric casts."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.value.convert",
                        inputs={
                            "value": MappingSource.copy("value"),
                            "target": MappingSource.constant("number"),
                            "force_lossy": MappingSource.constant(True),
                        },
                        outputs={"float": "FLOAT", "int": "INT"},
                    ),
                ),
            ),
            tier="exact",
            evidence=CONVERSION_EVIDENCE,
        ),
        _record(
            source_pack="comfy-mtb",
            node_class="Int To Bool (mtb)",
            revision=MTB_REVISION,
            carrier="dinkster.value.convert",
            rule=ReplacementRule(
                from_type="comfy.comfy-mtb.Int To Bool (mtb)",
                cases=(
                    ReplacementCase.build(
                        "dinkster.value.convert",
                        inputs={
                            "value": MappingSource.copy("int"),
                            "target": MappingSource.constant("boolean"),
                            "force_lossy": MappingSource.constant(True),
                        },
                        outputs={"boolean": "BOOLEAN"},
                    ),
                ),
            ),
            tier="exact",
            evidence=CONVERSION_EVIDENCE,
        ),
    ]
    return records + _string_records()


def build_aliases() -> dict[str, object]:
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(schema) for schema in SOURCE_SCHEMAS],
        "records": _records(),
    }


def main() -> None:
    OUTPUT_PATH.write_text(
        json.dumps(build_aliases(), ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
