"""Native curve editor using the shared curve value and histogram event."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    CURVE_TYPE,
    HISTOGRAM_TYPE,
    Curve,
    CurveWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    report_event,
)


class CurveEditor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CurveEditor",
            display_name="Curve Editor",
            category="utilities",
            editor_role="curve",
            inputs=(
                InputSpec(
                    "curve",
                    TypeExpr.concrete(CURVE_TYPE),
                    default={
                        "points": [[0.0, 0.0], [1.0, 1.0]],
                        "interpolation": "monotone_cubic",
                    },
                    widget=CurveWidget(),
                ),
                InputSpec("histogram", TypeExpr.concrete(HISTOGRAM_TYPE), required=False),
            ),
            outputs=(OutputSpec("curve", TypeExpr.concrete(CURVE_TYPE)),),
            emits_previews=True,
        )

    @classmethod
    def execute(
        cls, *, curve: Curve, histogram: Sequence[int] | None = None
    ) -> Mapping[str, object]:
        if histogram is not None:
            report_event("dinkster.curve.histogram", {"histogram": list(histogram)})
        return cls.outputs(curve=curve)
