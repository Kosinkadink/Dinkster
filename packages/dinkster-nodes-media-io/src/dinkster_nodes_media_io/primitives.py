"""Structured save-target construction for media outputs."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_STRING,
    SAVE_TARGET_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    SaveTarget,
    SaveTargetWidget,
    TypeExpr,
)

STRING = TypeExpr.concrete(CORE_STRING)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)


class SetSaveTargetPrefix(Node):
    """Replace only the relative prefix of a structured save target.

    The target input keeps mount selection explicit and portable. A linked
    string can compute the relative prefix, but can never supply mount
    authority or a host path. SaveTarget construction applies the same strict
    prefix validation as literals and save consumers.
    """

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.set_save_target_prefix",
            display_name="Set Save Target Prefix",
            category="utilities/save",
            description=(
                "Replaces a structured save target's relative prefix while "
                "preserving its explicit mount id. The prefix may be linked "
                "from a computed string; mount write authority is still "
                "checked by the save node at execution."
            ),
            inputs=(
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    widget=SaveTargetWidget(),
                    display_name="Target and Mount",
                ),
                InputSpec(
                    "prefix",
                    STRING,
                    required=False,
                    display_name="Computed Prefix",
                ),
            ),
            outputs=(OutputSpec("save_target", SAVE_TARGET),),
            search_terms=(
                "computed save prefix",
                "string to save target",
                "save destination",
                "output path",
            ),
        )

    @classmethod
    def execute(cls, *, target: SaveTarget, prefix: str | None = None) -> Mapping[str, object]:
        if prefix is None:
            return cls.outputs(save_target=target)
        return cls.outputs(save_target=SaveTarget(mount=target.mount, prefix=prefix))


SAVE_TARGET_NODES: tuple[type[Node], ...] = (SetSaveTargetPrefix,)
