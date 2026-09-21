from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dinkster_compat_comfy.native import NATIVE_NODES
from dinkster_compat_comfy.native_arm import GENERATION_PROVIDER_NODES
from dinkster_model_wan import WAN_MODEL_NODES
from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_nodes_generation import GENERATION_NODES
from dinkster_nodes_image import IMAGE_NODES
from dinkster_nodes_media_io import MEDIA_IO_NODES
from dinkster_schema import DynamicComboSpec, InputSpec, NodeSchema, build_schemas

BASELINE = Path(__file__).parent / "fixtures" / "input_prominence_baseline.json"


def _catalog() -> dict[str, NodeSchema]:
    nodes = (
        *FOUNDATION_NODES,
        *MEDIA_IO_NODES,
        *IMAGE_NODES,
        *GENERATION_NODES,
        *NATIVE_NODES,
        *WAN_MODEL_NODES,
    )
    return build_schemas(nodes)  # pyright: ignore[reportArgumentType]


def _input_paths(schema: NodeSchema) -> dict[str, InputSpec | DynamicComboSpec]:
    paths: dict[str, InputSpec | DynamicComboSpec] = {item.id: item for item in schema.inputs}

    def add_combos(combos: tuple[DynamicComboSpec, ...], prefix: str = "") -> None:
        for combo in combos:
            paths[f"{prefix}{combo.id}"] = combo
            for option in combo.options:
                branch = f"{prefix}{combo.id}[{option.key}]"
                for item in option.inputs:
                    if isinstance(item, DynamicComboSpec):
                        add_combos((item,), branch + ".")
                    elif isinstance(item, InputSpec):
                        paths[f"{branch}.{item.id}"] = item

    add_combos(schema.combos)
    for slot in schema.slots:
        if slot.variants is None:
            continue
        for variant in slot.variants:
            for item in variant.inputs:
                if isinstance(item, InputSpec):
                    paths[f"{slot.id}<{variant.key}>.{item.id}"] = item
    return paths


def test_approved_input_prominence_baseline() -> None:
    baseline = json.loads(BASELINE.read_text())
    baseline_keys = [(item["node"], item["path"]) for item in baseline]
    assert len(baseline_keys) == len(set(baseline_keys))
    expected_counts = {
        "P": 82,
        "CP": 120,
        "A": 128,
        "I": 17,
    }
    assert Counter(item["class"] for item in baseline) == expected_counts
    schemas = _catalog()
    resize_rows = [item for item in baseline if item["node"] == "dinkster.image.resize"]
    assert Counter(item["class"] for item in resize_rows) == {"P": 5, "CP": 21, "A": 6}
    resize_expected = {item["path"] for item in baseline if item["node"] == "dinkster.image.resize"}
    assert resize_expected == set(_input_paths(schemas["dinkster.image.resize"]))
    expected_advanced = {(item["node"], item["path"]) for item in baseline if item["class"] == "A"}
    live_advanced = {
        (node_type, path)
        for node_type, schema in schemas.items()
        for path, input_spec in _input_paths(schema).items()
        if isinstance(input_spec, InputSpec) and input_spec.advanced and not input_spec.hidden
    }
    assert live_advanced == expected_advanced

    live = Counter[str]()
    for expected in baseline:
        schema = schemas[expected["node"]]
        inputs = _input_paths(schema)
        input_spec = inputs.get(expected["path"])
        assert input_spec is not None, (expected["node"], expected["path"])
        classification = expected["class"]
        live[classification] += 1
        if classification == "A":
            assert isinstance(input_spec, InputSpec)
            assert input_spec.advanced and not input_spec.hidden
        elif classification == "I":
            assert isinstance(input_spec, InputSpec)
            assert input_spec.hidden and not input_spec.advanced
        else:
            assert not isinstance(input_spec, InputSpec) or (
                not input_spec.advanced and not input_spec.hidden
            )
    assert live == expected_counts


def test_generation_execution_providers_use_owner_schemas() -> None:
    assert build_schemas(GENERATION_PROVIDER_NODES) == build_schemas(GENERATION_NODES)


def test_conditional_primary_controls_follow_their_operation() -> None:
    schema = _catalog()["dinkster.string.transform"]
    paths = _input_paths(schema)
    assert "unit" in paths
    assert "side" in paths
    assert schema.widget_groups[0].input == "operation"
    assert schema.widget_groups[0].values == ("take",)
    assert schema.widget_groups[0].members == ("unit", "side")

    channels = _catalog()["dinkster.image.channels.split"]
    assert channels.widget_groups[0].input == "channel_layout"
    assert channels.widget_groups[0].values == ("single",)
    assert channels.widget_groups[0].members == ("single_channel_image",)

    stitch = _catalog()["dinkster.image.stitch"]
    assert stitch.widget_groups[0].input == "spacing_width"
    assert 0 not in stitch.widget_groups[0].values
    assert stitch.widget_groups[0].members == ("spacing_color",)
