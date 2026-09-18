"""Explicit bridges across the core.string/core.combo boundary.

``core.combo`` values remain plain strings at execution, in literals, and in
caches, but their envelope type is distinct from ``core.string``. Direct
links across that boundary are hard-invalid in either direction; these two
identity nodes are the one explicit bridge. Combo-to-combo links remain
legal regardless of their choice lists because vocabulary is presentation,
not type identity.

Vocabulary membership stays the frontend's diagnostic per the v9 combo
contract ("a stored value outside the current options is the frontend's
diagnostic to surface, not a schema error"): what a computed string that
names no current choice DOES at execution is the consuming node's own
behavior - typically a loud failure, but that node owns the call.
"""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)

STRING = TypeExpr.concrete(CORE_STRING)
COMBO = TypeExpr.concrete(CORE_COMBO)


class StringToCombo(Node):
    """Identity bridge: drive a combo-widgeted input from a computed
    string. Mirrors the upstream ConvertStringToComboNode prototype
    (never shipped there - no legacy alias to claim; its search prose
    rides as search terms instead)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.string_to_combo",
            display_name="Convert String to Combo",
            category="string",
            description=(
                "Passes a string through as a combo choice. Connect the "
                "output to a dropdown (combo) input to drive it from a "
                "computed value; a value that names no current choice is "
                "flagged in the editor, and how the consuming node treats "
                "it at execution is that node's own behavior."
            ),
            search_terms=("string to dropdown", "text to combo", "combo"),
            inputs=(InputSpec("string", STRING),),
            outputs=(OutputSpec("choice", COMBO),),
        )

    @classmethod
    def execute(cls, *, string: str) -> Mapping[str, object]:
        return cls.outputs(choice=string)


class ComboToString(Node):
    """Identity bridge: use a combo selection as a plain string."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.combo_to_string",
            display_name="Convert Combo to String",
            category="string",
            description=(
                "Passes a combo choice through as a plain string, for "
                "feeding a dropdown selection into ordinary string inputs."
            ),
            search_terms=("combo to string", "dropdown to text", "combo"),
            inputs=(InputSpec("choice", COMBO),),
            outputs=(OutputSpec("text", STRING),),
        )

    @classmethod
    def execute(cls, *, choice: str) -> Mapping[str, object]:
        return cls.outputs(text=choice)


COMBO_EDGE_NODES: list[type[Node]] = [StringToCombo, ComboToString]
