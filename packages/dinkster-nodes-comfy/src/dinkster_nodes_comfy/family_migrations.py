"""Declarations for replacing historical family suffixes with canonical positions."""

from dinkster_api.v1 import (
    InputFamilyMapping,
    MappingSource,
    ReplacementCase,
    ReplacementMigration,
    ReplacementRule,
)


def ordered_family_migration(
    node_type: str,
    family: str,
    historical_input: str,
    outputs: tuple[str, ...],
    *,
    historical_member_port: str = "",
    inputs: tuple[str, ...] = (),
) -> ReplacementRule:
    return ReplacementRule(
        from_type=node_type,
        migration=ReplacementMigration(historical_inputs=(family,)),
        cases=(
            ReplacementCase(
                to=node_type,
                inputs=tuple((name, MappingSource.copy(name)) for name in inputs),
                input_families=(
                    (
                        family,
                        InputFamilyMapping.copy(
                            family,
                            source_template=(historical_input,),
                            source_member_port=historical_member_port,
                            member_ids="targetPositions",
                            inputs={historical_input: MappingSource.copy(historical_input)},
                        ),
                    ),
                ),
                outputs=tuple((name, name) for name in outputs),
            ),
        ),
    )
