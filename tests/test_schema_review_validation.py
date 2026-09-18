from typing import Any, cast

import pytest
from dinkster_schema import (
    AssetWidget,
    BooleanWidget,
    ComboWidget,
    InputSpec,
    NodeSchema,
    OutputSpec,
    SaveTargetWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)


def _wire() -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        schema_to_wire(
            NodeSchema(
                node_type="review.node",
                inputs=(InputSpec("value", TypeExpr.concrete("core.int")),),
                outputs=(OutputSpec("out", TypeExpr.concrete("core.int")),),
            )
        ),
    )


@pytest.mark.parametrize(
    ("widget", "type_id", "message"),
    [
        (BooleanWidget(label_on="yes"), "core.string", "boolean widget"),
        (ComboWidget(options=("a",)), "core.int", "combo widget"),
        (SaveTargetWidget(), "core.string", "save target widget"),
    ],
)
def test_new_widget_socket_bindings_are_enforced(
    widget: object, type_id: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        InputSpec("bad", TypeExpr.concrete(type_id), widget=widget)  # type: ignore[arg-type]


def test_asset_widget_binding_matrix_accepts_assets_and_concrete_scalars() -> None:
    """The picker binds to direct asset destinations and concrete scalar
    decode/merge destinations, but never list destinations or unresolved
    asset variables."""
    image_asset = TypeExpr.asset_of(TypeExpr.concrete("comfy.IMAGE"))
    InputSpec("bare", TypeExpr.concrete("dinkster.asset"), widget=AssetWidget())
    InputSpec("typed", image_asset, widget=AssetWidget())
    InputSpec("scalar", TypeExpr.concrete("comfy.IMAGE"), widget=AssetWidget())
    InputSpec(
        "sheet",
        TypeExpr.asset_of(TypeExpr.list_of(TypeExpr.concrete("comfy.IMAGE"))),
        widget=AssetWidget(),
    )
    with pytest.raises(ValueError, match="asset widget"):
        InputSpec("many", TypeExpr.list_of(image_asset), widget=AssetWidget())
    with pytest.raises(ValueError, match="asset widget"):
        InputSpec(
            "values",
            TypeExpr.list_of(TypeExpr.concrete("comfy.IMAGE")),
            widget=AssetWidget(),
        )
    with pytest.raises(ValueError, match="asset widget"):
        InputSpec("open", TypeExpr.asset_of(TypeExpr.variable("T")), widget=AssetWidget())


def test_unknown_interface_role_is_rejected() -> None:
    wire = _wire()
    wire["interface"] = [{"role": "bogus"}]
    with pytest.raises(ValueError, match="unsupported interface role"):
        schema_from_wire(wire)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda wire: wire.__setitem__("idempotent", "false"), "idempotent must be a boolean"),
        (lambda wire: wire.__setitem__("version", True), "version must be an integer"),
        (lambda wire: wire.__setitem__("ioBound", 1), "ioBound must be a boolean"),
        (
            lambda wire: wire["interface"][0].__setitem__(
                "widget", {"type": "NUMBER", "min": "zero"}
            ),
            "NUMBER widget min must be a number",
        ),
    ],
)
def test_schema_decoder_rejects_wrong_json_primitive_types(mutate: Any, message: str) -> None:
    wire = _wire()
    mutate(wire)
    with pytest.raises(ValueError, match=message):
        schema_from_wire(wire)
