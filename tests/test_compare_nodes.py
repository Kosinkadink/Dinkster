"""The comparison CLI rejects interface drift without alias matching."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize(
    "name,parameter,baseline,observed,passed",
    [
        (
            "ModelAttentionBackend",
            "attention",
            ["pytorch attention"],
            ["pytorch attention", "comfy kitchen attention"],
            True,
        ),
        (
            "ModelAttentionBackend",
            "attention",
            ["pytorch attention"],
            ["comfy kitchen attention"],
            False,
        ),
        (
            "ModelAttentionBackend",
            "attention",
            ["pytorch attention"],
            ["pytorch attention", "unknown"],
            False,
        ),
        (
            "SelectModelDevice",
            "device",
            ["default", "cpu"],
            ["default", "cpu", "gpu:0", "gpu:2"],
            True,
        ),
        ("SelectModelDevice", "device", ["default", "cpu"], ["default", "cpu", "gpu:x"], False),
        ("SelectVAEDevice", "device", ["default"], ["default", "cpu", "gpu:0"], False),
    ],
)
def test_environment_options_remain_conditional_without_hiding_static_drift(
    name: str, parameter: str, baseline: list[str], observed: list[str], passed: bool
) -> None:
    from dinkster_schema import ComboWidget, InputSpec, NodeSchema, TypeExpr, schema_to_wire

    from tools.compare_nodes import compare_registry
    from tools.dump_comfy_nodes import annotate_environment_options

    inputs: dict[str, Any] = {"required": {parameter: [baseline]}}
    annotate_environment_options(name, inputs)
    declaration = inputs["required"][parameter][1]["conditional_options"]
    assert declaration[0]["condition"]
    native = schema_to_wire(
        NodeSchema(
            node_type=name,
            inputs=(
                InputSpec(
                    parameter, TypeExpr.concrete("COMBO"), widget=ComboWidget(tuple(observed))
                ),
            ),
        )
    )
    result = compare_registry(
        {"nodes": [{"class_name": name, "input_types": inputs, "return_types": []}]},
        {"nodes": {name: native}},
    )
    assert result["passed"] is passed
    assert result["rows"][0]["status"] == ("conditional" if passed else "differs")
    assert result["eligible_node_count"] == 1
    assert result["conditional_options"] == {name: {parameter: declaration}}


@pytest.mark.parametrize(
    "declaration",
    [
        [{"value": "cpu", "condition": "anything"}],
        [{"pattern": ".*", "condition": "anything"}],
        ["bad"],
    ],
)
def test_conditional_metadata_cannot_remove_static_choices(declaration: object) -> None:
    from tools.compare_nodes import compare_registry

    result = compare_registry(
        {
            "nodes": [
                {
                    "class_name": "SelectModelDevice",
                    "input_types": {
                        "required": {
                            "device": [["default", "cpu"], {"conditional_options": declaration}]
                        }
                    },
                }
            ]
        },
        {
            "nodes": {
                "SelectModelDevice": {
                    "nodeType": "SelectModelDevice",
                    "interface": [
                        {
                            "role": "input",
                            "id": "device",
                            "type": {"kind": "concrete", "types": ["COMBO"]},
                            "widget": {"type": "COMBO", "options": ["default"]},
                        }
                    ],
                }
            }
        },
    )
    assert not result["passed"]
    assert "invalid conditional option declaration" in result["rows"][0]["differences"]


@pytest.mark.parametrize("status", ["differs", "absent", "excluded"])
@pytest.mark.parametrize(
    "inputs,message",
    [
        ({"required": {"device": [["default", "cpu"], None]}}, "input options must be an object"),
        ("invalid", "reference input schema unavailable"),
        ({"required": "invalid"}, "input group must be an object"),
    ],
)
def test_malformed_reference_options_are_reported_without_crashing(
    status: str, inputs: object, message: str
) -> None:
    from tools.compare_nodes import compare_registry

    result = compare_registry(
        {
            "nodes": [
                {
                    "class_name": "SelectModelDevice",
                    "api_node": status == "excluded",
                    "input_types": inputs,
                }
            ]
        },
        {
            "nodes": {}
            if status == "absent"
            else {"SelectModelDevice": {"nodeType": "SelectModelDevice", "interface": []}}
        },
    )
    assert result["rows"][0]["status"] == status
    assert result["conditional_options"] == {}
    if status == "differs":
        assert message in result["rows"][0]["differences"]
    assert result["passed"] is (status == "excluded")


def test_failed_reference_import_cannot_pass_an_incomplete_census() -> None:
    from tools.compare_nodes import compare_registry

    result = compare_registry(
        {
            "nodes": [{"class_name": "PresentNode", "input_types": {}}],
            "environment": {"device_policy": "cpu", "torch_version": "2.14.1+cpu"},
            "builtin_import_failures": ["nodes_glsl.py"],
        },
        {"nodes": {"PresentNode": {"nodeType": "PresentNode", "interface": []}}},
    )
    assert not result["passed"]
    assert result["registry_errors"] == ["reference builtin imports failed: nodes_glsl.py"]
    assert result["reference_environment"] == {
        "device_policy": "cpu",
        "torch_version": "2.14.1+cpu",
    }


@pytest.mark.parametrize("upload", ["image_upload", "audio_upload", "video_upload", "file_upload"])
def test_reference_upload_choices_are_file_inputs(upload: str) -> None:
    from tools.dump_comfy_nodes import annotate_files

    assert annotate_files({"required": {"file": (["none"], {upload: True})}}) == {
        "required": {"file": [["none"], {upload: True, "file_input": True}]}
    }
    assert annotate_files({"required": {"mode": (["none"], {})}}) == {
        "required": {"mode": [["none"], {}]}
    }


@pytest.mark.parametrize(
    "source,excluded",
    [
        ("comfy_extras/nodes_train.py", True),
        ("comfy_extras/nodes_dataset.py", True),
        ("custom_nodes/nodes_train.py", False),
        ("nodes.py", False),
    ],
)
def test_cli_training_exclusion_keeps_deletion_errors(
    tmp_path: Path, source: str, excluded: bool
) -> None:
    reference = tmp_path / "reference.json"
    registry = tmp_path / "registry.json"
    csv_output = tmp_path / "comparison.csv"
    summary_output = tmp_path / "summary.json"
    reference.write_text(
        json.dumps(
            {
                "nodes": [
                    {"class_name": "LoraModelLoader", "file": source, "input_types": {}},
                    {"class_name": "OrdinaryNode", "file": "nodes.py", "input_types": {}},
                    {"class_name": "ApiNode", "api_node": True},
                ]
            }
        )
    )
    served = {
        name: {"nodeType": name, "interface": []}
        for name in ("LoraModelLoader", "OrdinaryNode", "NativeOnly")
    }
    for present in (True, False):
        if not present:
            del served["LoraModelLoader"]
        registry.write_text(json.dumps({"nodes": served}))
        result = subprocess.run(
            [
                sys.executable,
                "tools/compare_nodes.py",
                str(reference),
                str(registry),
                str(csv_output),
                str(summary_output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.stderr == ""
        summary = json.loads(summary_output.read_text())
        assert summary["eligible_node_count"] == (1 if excluded else 2)
        assert summary["passed"] is (not present if excluded else present)
        assert result.returncode == (0 if summary["passed"] else 1)
        assert summary["registry_errors"] == (
            ["excluded training-origin node remains registered: LoraModelLoader"]
            if excluded and present
            else []
        )
        with csv_output.open() as handle:
            rows = {row["class_name"]: row for row in csv.DictReader(handle)}
        assert rows["LoraModelLoader"]["status"] == (
            "excluded" if excluded else "identical" if present else "absent"
        )
        assert rows["LoraModelLoader"]["exclusion"] == ("training-origin node" if excluded else "")
        assert rows["ApiNode"]["status"] == "excluded"
        assert rows["NativeOnly"]["status"] == "Dinkster-only"


@pytest.mark.parametrize(
    "name", ["SaveExample", "CLIPSave", "VAESave", "CheckpointSave", "ModelSave"]
)
@pytest.mark.parametrize(
    "output_node,prefix_type", [(True, "STRING"), (False, "STRING"), (True, "INT")]
)
@pytest.mark.parametrize("input_is_list", [False, True])
def test_save_mount_admission_uses_source_output_contract(
    name: str, output_node: bool, prefix_type: str, input_is_list: bool
) -> None:
    from dinkster_schema import InputSpec, MountWidget, NodeSchema, TypeExpr, schema_to_wire

    from tools.compare_nodes import compare_registry

    reference = {
        "nodes": [
            {
                "class_name": name,
                "output_node": output_node,
                "input_is_list": input_is_list,
                "input_types": {"required": {"filename_prefix": [prefix_type]}},
                "return_types": [],
            }
        ]
    }
    native = schema_to_wire(
        NodeSchema(
            node_type=name,
            output_node=output_node,
            inputs=(
                InputSpec(
                    "filename_prefix",
                    TypeExpr.list_of(TypeExpr.concrete(prefix_type))
                    if input_is_list
                    else TypeExpr.concrete(prefix_type),
                ),
                InputSpec(
                    "mount",
                    TypeExpr.concrete("STRING"),
                    required=False,
                    default=None,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
        )
    )
    result = compare_registry(reference, {"nodes": {name: native}})
    assert result["passed"] is (output_node and prefix_type == "STRING" and not input_is_list)
    if not result["passed"]:
        assert "inputs" in result["rows"][0]["differences"]


def test_reference_file_arguments_follow_declared_execution_source(tmp_path: Path) -> None:
    import importlib.util

    from tools.dump_comfy_nodes import annotated_file_arguments

    source = tmp_path / "source_nodes.py"
    source.write_text("""
class LoadExample:
    FUNCTION = "load"

    def load(self, latent, mode):
        path = folder_paths.get_annotated_filepath(latent)
        other = different_owner.get_annotated_filepath(mode)
        fixed = folder_paths.get_annotated_filepath("constant.latent")
        local = "local.latent"
        ignored = folder_paths.get_annotated_filepath(local)
        return path, other, fixed, ignored

    def validate(self, mode):
        return folder_paths.get_annotated_filepath(mode)

class NonFile:
    FUNCTION = "run"

    def run(self, mode):
        return mode
""")
    spec = importlib.util.spec_from_file_location("source_nodes", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert annotated_file_arguments(module.LoadExample) == {"latent"}
    assert annotated_file_arguments(module.NonFile) == set()


def test_comparison_cli(tmp_path: Path) -> None:
    from dinkster_schema import InputSpec, NodeSchema, StringWidget, TypeExpr, schema_to_wire

    wire = schema_to_wire(
        NodeSchema(
            node_type="SaveExample",
            inputs=(
                InputSpec(
                    "mount",
                    TypeExpr.concrete("STRING"),
                    required=False,
                    default=None,
                    widget=StringWidget(multiline=False),
                ),
            ),
        )
    )
    mount = json.loads(json.dumps(wire))["interface"][0]
    assert "default" not in mount
    reference = {
        "nodes": [
            {
                "class_name": "SaveExample",
                "file": "nodes.py",
                "category": "image",
                "display_name": "Save Example",
                "output_node": True,
                "input_types": {
                    "required": {
                        "filename_prefix": ["STRING", {"default": "Example"}],
                        "quality": [["fast", "best"], {"default": "best"}],
                    }
                },
                "return_types": ["IMAGE", "MASK"],
                "return_names": ["image", "mask"],
            },
            {"class_name": "ApiExample", "api_node": True},
        ]
    }
    native = {
        "nodeType": "SaveExample",
        "category": "image",
        "displayName": "Save Example",
        "outputNode": True,
        "interface": [
            {
                "role": "input",
                "id": "filename_prefix",
                "required": True,
                "type": {"kind": "concrete", "types": ["STRING"]},
                "default": "Example",
            },
            {
                "role": "input",
                "id": "quality",
                "required": True,
                "type": {"kind": "concrete", "types": ["COMBO"]},
                "default": "best",
                "widget": {
                    "type": "COMBO",
                    "options": [
                        {"value": "fast", "label": "Quick"},
                        "best",
                    ],
                },
            },
            mount,
            {"role": "output", "id": "image", "type": {"kind": "concrete", "types": ["IMAGE"]}},
            {"role": "output", "id": "mask", "type": {"kind": "concrete", "types": ["MASK"]}},
        ],
    }
    source = tmp_path / "reference.json"
    source.write_text(json.dumps(reference))
    registry = tmp_path / "registry.json"
    csv_path = tmp_path / "comparison.csv"
    summary_path = tmp_path / "summary.json"

    def run(schema: dict) -> tuple[int, dict, list[dict]]:
        registry.write_text(json.dumps({"nodes": {"SaveExample": schema}}))
        result = subprocess.run(
            [
                sys.executable,
                "tools/compare_nodes.py",
                str(source),
                str(registry),
                str(csv_path),
                str(summary_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.stderr == ""
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        return result.returncode, json.loads(summary_path.read_text()), rows

    code, summary, rows = run(native)
    assert code == 0
    assert summary["counts"] == {"identical": 1, "excluded": 1}
    assert [row["status"] for row in rows] == ["identical", "excluded"]

    native["interface"][2]["default"] = None
    assert run(native)[0] == 0
    del native["interface"][0]["default"]
    code, _, rows = run(native)
    assert code == 1 and "inputs[0].default" in rows[0]["differences"]
    native["interface"][0]["default"] = "Example"
    native["interface"][0]["widget"] = {"type": "STRING", "multiline": False}
    code, _, rows = run(native)
    assert code == 1 and "inputs[0].multiline" in rows[0]["differences"]
    del native["interface"][0]["widget"]

    native["interface"][1]["widget"]["options"].reverse()
    code, summary, rows = run(native)
    assert code == 1 and summary["counts"]["differs"] == 1
    assert "options[0]" in rows[0]["differences"]

    native["interface"][1]["widget"] = {"type": "ASSET"}
    code, _, rows = run(native)
    assert code == 1
    assert "asset picker on a non-file input" in rows[0]["differences"]

    native["nodeType"] = "dinkster.SaveExample"
    code, summary, _ = run(native)
    assert code == 1
    assert summary["registry_errors"]

    registry.write_text('{"nodes": {}, "nodes": {}}')
    duplicate = subprocess.run(
        [
            sys.executable,
            "tools/compare_nodes.py",
            str(source),
            str(registry),
            str(csv_path),
            str(summary_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "duplicate JSON key: nodes" in duplicate.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("required", True),
        ("default", "output"),
        ("default", "another-grant"),
        ("default", 0),
        ("default", False),
        ("type", {"kind": "concrete", "types": ["INT"]}),
        ("widget", {"type": "STRING", "multiline": True}),
        ("widget", {"type": "STRING", "multiline": 0}),
        ("widget", {"type": "STRING", "min": 0}),
    ],
)
def test_save_mount_is_not_an_arbitrary_extra_input(field: str, value: object) -> None:
    from tools.compare_nodes import compare_registry

    mount = {
        "role": "input",
        "id": "mount",
        "required": False,
        "default": None,
        "type": {"kind": "concrete", "types": ["STRING"]},
    }
    mount[field] = value
    reference = {
        "nodes": [
            {
                "class_name": "SaveExample",
                "output_node": True,
                "input_types": {"required": {"filename_prefix": ["STRING"]}},
                "return_types": [],
            }
        ]
    }
    registry = {
        "nodes": {
            "SaveExample": {
                "nodeType": "SaveExample",
                "outputNode": True,
                "interface": [
                    {
                        "role": "input",
                        "id": "filename_prefix",
                        "required": True,
                        "type": {"kind": "concrete", "types": ["STRING"]},
                    },
                    mount,
                ],
            }
        }
    }
    summary = compare_registry(reference, registry)
    assert not summary["passed"]
    assert "mount" in summary["rows"][0]["differences"]


@pytest.mark.parametrize("list_inputs", [False, True])
@pytest.mark.parametrize("native_type", ["COMBO", "ASSET"])
def test_exact_ids_and_file_widget_exemption(list_inputs: bool, native_type: str) -> None:
    from tools.compare_nodes import compare_registry

    reference = {
        "nodes": [
            {
                "class_name": "LoadExample",
                "file_inputs": ["file"],
                "input_is_list": list_inputs,
                "input_types": {"required": {"file": [["original.safetensors"]]}},
                "return_types": ["MODEL"],
                "return_names": ["model"],
                "output_is_list": [True],
            }
        ]
    }
    schema = {
        "nodeType": "LoadExample",
        "interface": [
            {
                "role": "input",
                "id": "file",
                "required": True,
                "type": {"kind": "concrete", "types": [native_type]},
                "widget": {"type": "ASSET"},
            },
            {
                "role": "output",
                "id": "model",
                "type": {"kind": "concrete", "types": ["MODEL"]},
            },
        ],
    }
    registry = {"nodes": {"LoadExample": schema, "Extra": {"nodeType": "Extra", "interface": []}}}
    summary = compare_registry(reference, registry)
    assert summary["passed"]
    assert summary["counts"] == {"identical": 1, "Dinkster-only": 1}

    file_input = schema["interface"][0]
    reference["nodes"][0]["file_inputs"] = []
    assert not compare_registry(reference, registry)["passed"]
    reference["nodes"][0]["file_inputs"] = ["file"]
    if native_type == "ASSET":
        from dinkster_schema import AssetWidget, InputSpec, NodeSchema, TypeExpr, schema_to_wire

        wire = schema_to_wire(
            NodeSchema(
                node_type="LoadExample",
                inputs=(
                    InputSpec(
                        "file",
                        TypeExpr.concrete("ASSET"),
                        widget=AssetWidget(kind="model/lora", accept=("application/octet-stream",)),
                    ),
                ),
            )
        )
        interface = wire["interface"]
        assert isinstance(interface, list)
        file_input = interface[0]
        assert isinstance(file_input, dict)
        schema["interface"][0] = file_input
        assert compare_registry(reference, registry)["passed"]
        widget = file_input.pop("widget")
        assert not compare_registry(reference, registry)["passed"]
        file_input["widget"] = widget
    file_input["id"] = "renamed"
    assert not compare_registry(reference, registry)["passed"]
    file_input["id"] = "file"
    file_input["type"] = {"kind": "concrete", "types": ["STRING"]}
    assert not compare_registry(reference, registry)["passed"]
    file_input["type"] = {"kind": "concrete", "types": [native_type]}
    file_input["required"] = False
    assert not compare_registry(reference, registry)["passed"]
    file_input["required"] = True
    file_input["type"] = {"kind": "list", "element": file_input["type"]}
    assert compare_registry(reference, registry)["passed"] is list_inputs
    original_spec = reference["nodes"][0]["input_types"]["required"]["file"]
    reference["nodes"][0]["input_types"]["required"]["file"] = [{"list": "COMBO"}]
    assert compare_registry(reference, registry)["passed"]
    reference["nodes"][0]["input_types"]["required"]["file"] = original_spec
    file_input["type"] = {"kind": "concrete", "types": [native_type]}
    file_input["default"] = "another-mount/file.safetensors"
    assert compare_registry(reference, registry)["passed"]
    file_input.pop("default")

    schema["interface"][1]["type"] = {
        "kind": "list",
        "element": {"kind": "concrete", "types": ["MODEL"]},
    }
    assert compare_registry(reference, registry)["passed"]
    reference["nodes"][0]["output_is_list"] = [False]
    assert not compare_registry(reference, registry)["passed"]

    del registry["nodes"]["LoadExample"]
    registry["nodes"]["Extra"]["aliases"] = ["LoadExample"]
    summary = compare_registry(reference, registry)
    assert not summary["passed"]
    assert summary["counts"] == {"absent": 1, "Dinkster-only": 1}


@pytest.fixture
def example_contract() -> tuple[dict, dict]:
    return (
        {
            "class_name": "Example",
            "input_types": {"required": {"value": ["INT"]}},
            "return_types": ["INT", "FLOAT"],
            "return_names": None,
        },
        {
            "nodeType": "Example",
            "interface": [
                {"role": "input", "id": "value", "type": {"kind": "concrete", "types": ["INT"]}},
                {"role": "output", "id": "first", "type": {"kind": "concrete", "types": ["INT"]}},
                {
                    "role": "output",
                    "id": "second",
                    "type": {"kind": "concrete", "types": ["FLOAT"]},
                },
            ],
        },
    )


def test_v3_input_order_preserves_interleaved_optional_inputs() -> None:
    from tools.compare_nodes import compare

    reference = {
        "class_name": "Example",
        "is_v3": True,
        "v3_inputs": [{"id": name} for name in ("first", "optional", "last")],
        "input_types": {
            "required": {"first": ["INT"], "last": ["FLOAT"]},
            "optional": {"optional": ["STRING"]},
        },
    }
    native = {
        "nodeType": "Example",
        "interface": [
            {
                "role": "input",
                "id": name,
                "required": required,
                "type": {"kind": "concrete", "types": [kind]},
            }
            for name, required, kind in (
                ("first", True, "INT"),
                ("optional", False, "STRING"),
                ("last", True, "FLOAT"),
            )
        ],
    }
    assert compare(reference, native) == []
    native["interface"][1:] = reversed(native["interface"][1:])
    assert compare(reference, native)
    del reference["is_v3"]
    del reference["v3_inputs"]
    assert compare(reference, native) == []
    native["interface"][0], native["interface"][1] = (
        native["interface"][1],
        native["interface"][0],
    )
    assert compare(reference, native)


@pytest.mark.parametrize("order", [None, [], [{"id": "other"}], [{"id": "value"}] * 2])
def test_v3_input_order_requires_exact_membership(
    example_contract: tuple[dict, dict], order: list[dict] | None
) -> None:
    from tools.compare_nodes import compare

    reference, native = example_contract
    reference.update(is_v3=True, v3_inputs=order)
    assert compare(reference, native)


def test_hidden_inputs_do_not_hide_public_input_drift(example_contract: tuple[dict, dict]) -> None:
    from tools.compare_nodes import compare

    reference, native = example_contract
    reference["input_types"]["hidden"] = {"dynprompt": ["DYNPROMPT"]}
    native["interface"].insert(
        0,
        {
            "role": "input",
            "id": "execution_context",
            "hidden": True,
            "type": {"kind": "wildcard"},
        },
    )
    assert compare(reference, native) == []
    native["interface"][1]["id"] = "renamed"
    assert any("name" in difference for difference in compare(reference, native))


def test_hidden_family_members_do_not_create_public_inputs(
    example_contract: tuple[dict, dict],
) -> None:
    from tools.compare_nodes import compare

    reference, native = example_contract
    family = {
        "role": "inputFamily",
        "id": "dependencies",
        "template": [
            {"role": "input", "id": "dependency", "hidden": True, "type": {"kind": "wildcard"}}
        ],
    }
    native["interface"].append(family)
    assert compare(reference, native) == []
    family["template"][0]["hidden"] = False
    assert any("inputs: length" in difference for difference in compare(reference, native))


def test_unnamed_outputs_still_compare_type_position_and_explicit_names(
    example_contract: tuple[dict, dict],
) -> None:
    from tools.compare_nodes import compare

    reference, native = example_contract
    assert compare(reference, native) == []
    native["interface"][1:3] = reversed(native["interface"][1:3])
    assert any("outputs[0].type" in difference for difference in compare(reference, native))
    native["interface"][1:3] = reversed(native["interface"][1:3])
    reference["return_names"] = ["INT", "FLOAT"]
    assert any("outputs[0].name" in difference for difference in compare(reference, native))


def test_v3_schema_dump_preserves_only_declared_names(
    example_contract: tuple[dict, dict],
) -> None:
    from tools.compare_nodes import compare
    from tools.dump_comfy_nodes import v3_contract

    reference, native = example_contract
    schema = SimpleNamespace(
        display_name="Model Attention Backend",
        category="model/patch/wan",
        inputs=[SimpleNamespace(id="value")],
        is_input_list=False,
        is_output_node=False,
        is_api_node=False,
        outputs=[
            SimpleNamespace(display_name=None, io_type="INT", is_output_list=False),
            SimpleNamespace(display_name="second", io_type="FLOAT", is_output_list=False),
        ],
    )
    reference["category"] = "model/patch"
    reference["return_types"] = ["WRONG"]
    reference["output_is_list"] = [True]
    reference.update(v3_contract(schema))
    assert reference["category"] == "model/patch/wan"
    native["category"] = "model/patch/wan"
    assert reference["display_name"] == "Model Attention Backend"
    native["displayName"] = "Model Attention Backend"
    assert reference["return_names"] == [None, "second"]
    assert reference["v3_inputs"] == [{"id": "value"}]
    assert reference["return_types"] == ["INT", "FLOAT"]
    assert reference["output_is_list"] == [False, False]
    assert compare(reference, native) == []
    native["interface"][1]["id"] = "another_unnamed_output"
    assert compare(reference, native) == []
    native["interface"][2]["id"] = "renamed"
    assert any("outputs[1].name" in difference for difference in compare(reference, native))
    schema.display_name = None
    assert "display_name" not in v3_contract(schema)


@pytest.mark.parametrize("list_inputs", [False, True])
def test_float_constraints_compare_after_exact_float_projection(list_inputs: bool) -> None:
    from tools.compare_nodes import compare

    constraints = {"min": -(2**63 - 1), "max": 2**63 - 1, "step": 2**53 + 1, "round": False}
    reference = {
        "class_name": "PrimitiveFloat",
        "input_is_list": list_inputs,
        "input_types": {"required": {"value": ["FLOAT", constraints]}},
    }
    atom = {"kind": "concrete", "types": ["FLOAT"]}
    native = {
        "nodeType": "PrimitiveFloat",
        "interface": [
            {
                "role": "input",
                "id": "value",
                "type": {"kind": "list", "element": atom} if list_inputs else atom,
                "widget": {
                    "type": "NUMBER",
                    "min": -9.223372036854776e18,
                    "max": 9.223372036854776e18,
                    "step": 9007199254740992.0,
                    "round": False,
                },
            }
        ],
    }
    assert compare(reference, native) == []
    widget = native["interface"][0]["widget"]
    widget["min"] = -1e20
    assert any(".min" in difference for difference in compare(reference, native))
    widget["min"] = -9.223372036854776e18
    del widget["max"]
    assert any(".max" in difference for difference in compare(reference, native))
    widget["max"] = 9.223372036854776e18
    widget["round"] = 0.0
    assert any(".round" in difference for difference in compare(reference, native))
    widget["round"] = False
    constraints["round"] = 2**53 + 1
    widget["round"] = 9007199254740992.0
    assert compare(reference, native) == []
    reference["input_types"]["required"]["value"][0] = "INT"
    atom["types"] = ["INT"]
    constraints.clear()
    constraints["min"] = 2**63 - 2
    widget.clear()
    widget.update(type="NUMBER", min=str(2**63 - 1))
    assert any(".min" in difference for difference in compare(reference, native))


@pytest.mark.parametrize("direction,index", [("input", 0), ("output", 1)])
def test_wildcard_accepts_template_but_not_concrete_or_restricted_type(
    example_contract: tuple[dict, dict], direction: str, index: int
) -> None:
    from tools.compare_nodes import compare

    reference, native = example_contract
    if direction == "input":
        reference["input_types"]["required"]["value"] = ["*"]
    else:
        reference["return_types"][0] = "*"
    native["interface"][index]["type"] = {"kind": "variable", "templateId": "carry"}
    assert compare(reference, native) == []
    native["interface"][index]["type"]["allowed"] = ["INT"]
    assert compare(reference, native)
    native["interface"][index]["type"] = {"kind": "concrete", "types": ["INT"]}
    assert compare(reference, native)


@pytest.mark.parametrize(
    "input_list,output_list", [(False, False), (False, True), (True, False), (True, True)]
)
def test_list_execution_flags_do_not_change_declared_socket_types(
    example_contract: tuple[dict, dict], input_list: bool, output_list: bool
) -> None:
    from tools.compare_nodes import compare

    reference, native = example_contract
    reference["input_is_list"] = input_list
    reference["output_is_list"] = [output_list, False]
    assert compare(reference, native) == []
    index = 0 if input_list else 1
    native["interface"][index]["type"] = {
        "kind": "list",
        "element": {"kind": "concrete", "types": ["INT"]},
    }
    assert bool(compare(reference, native)) is not (input_list or output_list)
    native["interface"][index]["type"]["element"] = {
        "kind": "list",
        "element": {"kind": "concrete", "types": ["INT"]},
    }
    assert compare(reference, native)


def test_list_flags_do_not_hide_element_type_or_unflagged_output_drift(
    example_contract: tuple[dict, dict],
) -> None:
    from tools.compare_nodes import compare

    reference, native = example_contract
    reference["output_is_list"] = [True, False]
    native["interface"][1]["type"] = {
        "kind": "list",
        "element": {"kind": "concrete", "types": ["FLOAT"]},
    }
    assert compare(reference, native)
    native["interface"][1]["type"]["element"]["types"] = ["INT"]
    assert compare(reference, native) == []
    native["interface"][2]["type"] = {
        "kind": "list",
        "element": {"kind": "concrete", "types": ["FLOAT"]},
    }
    assert compare(reference, native)


@pytest.mark.parametrize("kind", ["autogrow", "group", "combo"])
@pytest.mark.parametrize("native_type", ["COMBO", "ASSET"])
def test_nested_inputs_preserve_file_slots_and_detect_child_drift(
    kind: str, native_type: str
) -> None:
    from tools.compare_nodes import compare_registry

    reference_children = {
        "required": {
            "file": ["COMBO", {"options": [], "file_input": True}],
            "strength": ["FLOAT", {"default": 0.75}],
        }
    }
    native_children = [
        {
            "role": "input",
            "id": "file",
            "required": True,
            "type": {"kind": "concrete", "types": [native_type]},
            "widget": {"type": "ASSET"},
        },
        {
            "role": "input",
            "id": "strength",
            "required": True,
            "type": {"kind": "concrete", "types": ["FLOAT"]},
            "default": 0.75,
        },
    ]
    if kind == "combo":
        reference_input = [
            "COMFY_DYNAMICCOMBO_V3",
            {
                "options": [
                    {"key": "selected", "inputs": reference_children},
                    {"key": "none", "inputs": {"required": {}}},
                ]
            },
        ]
        native_input = {
            "role": "dynamicCombo",
            "id": "settings",
            "required": True,
            "options": [
                {"key": "selected", "inputs": native_children},
                {"key": "none", "inputs": []},
            ],
        }
    else:
        native_input = {
            "role": "inputFamily",
            "id": "settings",
            "required": True,
            "template": native_children,
            "minMembers": 1,
            "maxMembers": 3,
        }
        if kind == "autogrow":
            # Autogrow has one scalar template; groups have a row of fields.
            reference_children["required"].pop("file")
            native_children.pop(0)
            reference_input = [
                "COMFY_AUTOGROW_V3",
                {
                    "template": {
                        "input": reference_children,
                        "min": 1,
                        "max": 3,
                        "prefix": "input",
                    }
                },
            ]
            native_input["memberPrefix"] = "input"
        else:
            reference_input = [
                "COMFY_DYNAMICGROUP_V3",
                {
                    "template": reference_children,
                    "min": 1,
                    "max": 3,
                },
            ]
    reference = {
        "nodes": [
            {
                "class_name": "Example",
                "input_types": {"required": {"settings": reference_input}},
                "return_types": [],
            }
        ]
    }
    registry = {"nodes": {"Example": {"nodeType": "Example", "interface": [native_input]}}}
    assert compare_registry(reference, registry)["passed"]

    if kind != "autogrow":
        native_children.reverse()
        assert not compare_registry(reference, registry)["passed"]
        native_children.reverse()
    native_children[-1]["default"] = 0.5
    result = compare_registry(reference, registry)
    assert not result["passed"]
    assert "0.75 vs 0.5" in result["rows"][0]["differences"]

    native_children[-1].update(
        type={"kind": "concrete", "types": ["COMBO"]},
        default="default",
        widget={"type": "COMBO", "options": ["default"]},
    )
    for declaration in (
        [{"value": "cpu", "condition": "anything"}],
        [{"pattern": ".*", "condition": "anything"}],
        ["bad"],
    ):
        reference_children["required"]["strength"] = [
            ["default", "cpu"],
            {"conditional_options": declaration},
        ]
        result = compare_registry(reference, registry)
        assert not result["passed"]
        assert "invalid conditional option declaration" in result["rows"][0]["differences"]


def test_namespace_rejects_core_id_ownership_and_nested_type_prefixes() -> None:
    from tools.compare_nodes import compare_registry

    reference = {"nodes": [{"class_name": "Example", "input_types": {}, "return_types": []}]}
    registry = {
        "nodes": {
            "Example": {"nodeType": "Example", "pack": "custom-pack", "interface": []},
            "Extra": {
                "nodeType": "Extra",
                "interface": [
                    {
                        "role": "output",
                        "id": "result",
                        "type": {
                            "kind": "list",
                            "element": {"kind": "concrete", "types": ["comfy.IMAGE"]},
                        },
                    }
                ],
            },
        }
    }
    result = compare_registry(reference, registry)
    assert not result["passed"]
    assert result["registry_errors"] == [
        "Extra: prefixed type name: comfy.IMAGE",
        "core class id 'Example' is claimed by pack 'custom-pack'",
    ]


@pytest.mark.parametrize(
    "name",
    ["ComfySwitchNode", "StartLoop", "EndLoop", "LoopIteration", "LoopProgress", "LoopResult"],
)
def test_engine_nodes_are_compared_despite_historical_exclusion_metadata(name: str) -> None:
    from tools.compare_nodes import compare_registry

    reference = {
        "nodes": [{"class_name": name, "input_types": {}, "return_types": ["INT"]}],
        "engine_exclusions": {name: ["loop semantics"]},
    }
    schema = {
        "nodeType": name,
        "interface": [
            {"role": "output", "id": "value", "type": {"kind": "concrete", "types": ["INT"]}}
        ],
    }
    registry = {"nodes": {name: schema}}
    assert compare_registry(reference, registry)["counts"] == {"identical": 1}
    schema["interface"][0]["type"]["types"] = ["FLOAT"]
    result = compare_registry(reference, registry)
    assert not result["passed"]
    assert result["counts"] == {"differs": 1}
    assert result["engine_exclusions"] == {}
    assert compare_registry(reference, {"nodes": {}})["counts"] == {"absent": 1}


@pytest.mark.parametrize("v3", [False, True])
@pytest.mark.parametrize(
    "kind", ["model/vae", "model/diffusers", "directory/input", "directory/dataset"]
)
def test_source_marked_selections_require_exact_picker_kind_and_socket(kind: str, v3: bool) -> None:
    from dinkster_api.v1 import AssetWidget, InputSpec, NodeSchema, TypeExpr, schema_to_wire

    from tools.compare_nodes import compare
    from tools.dump_comfy_nodes import FileChoices, annotate_files

    options = FileChoices(["source-option"], kind)
    spec = ("COMBO", {"options": options}) if v3 else (options,)
    row = {
        "class_name": "SelectionExample",
        "input_types": annotate_files({"required": {"selection": spec}}),
        "return_types": [],
    }
    schema = schema_to_wire(
        NodeSchema(
            node_type="SelectionExample",
            inputs=(
                InputSpec(
                    "selection",
                    TypeExpr.union("COMBO", "ASSET")
                    if kind == "model/vae"
                    else TypeExpr.concrete("COMBO"),
                    widget=AssetWidget(kind=kind, selection=True),
                ),
            ),
        )
    )
    assert compare(row, schema) == []
    interface = schema["interface"]
    assert isinstance(interface, list)
    item = interface[0]
    assert isinstance(item, dict)
    widget = item["widget"]
    widget["kind"] = "directory/input" if kind != "directory/input" else "directory/dataset"
    assert compare(row, schema)
    widget["kind"] = kind
    widget.pop("selection")
    assert compare(row, schema)
    widget["selection"] = True
    original_type = item["type"]
    for invalid_type in (
        {"kind": "concrete", "types": ["ASSET"]},
        {"kind": "concrete", "types": ["STRING"]},
        {"kind": "union", "types": ["COMBO", "ASSET", "STRING"]},
        {"kind": "list", "element": original_type},
    ):
        item["type"] = invalid_type
        assert compare(row, schema)
    item["type"] = original_type
    unmarked = ["source-option"]
    row["input_types"] = annotate_files(
        {"required": {"selection": ("COMBO", {"options": unmarked}) if v3 else (unmarked,)}}
    )
    assert compare(row, schema)
