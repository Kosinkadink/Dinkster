"""Native Gaussian splat inspection."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_INT,
    SPLAT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_splat,
    report_progress,
)


class GetSplatCount(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="GetSplatCount",
            display_name="Get Splat Count",
            category="3d/splat",
            description="Returns the number of splats summed across the batch.",
            search_terms=("splat count", "gaussian count", "number of splats", "splat info"),
            inputs=(InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
            outputs=(
                OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),
                OutputSpec("count", TypeExpr.concrete(CORE_INT)),
            ),
        )

    @classmethod
    def execute(cls, *, splat: object) -> Mapping[str, object]:
        value = materialize_splat(splat)
        count = sum(
            int(value.counts[index].item())
            if value.counts is not None
            else value.positions.shape[1]
            for index in range(value.positions.shape[0])
        )
        report_progress(1, 1, text=f"{count:,} splats")
        return cls.outputs(splat=splat, count=count)
