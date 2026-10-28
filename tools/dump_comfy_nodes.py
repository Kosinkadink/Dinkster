"""Dump the core node contract from a fixed ComfyUI checkout on CPU.

Run in the reference checkout's environment, with its dependencies installed:
CUDA_VISIBLE_DEVICES='' python dump_comfy_nodes.py CHECKOUT OUTPUT.json

This reuses the issue 504 audit's registered-class discovery and v1 projection.
The dump contains metadata only; it does not provide Dinkster implementations.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


class FileChoices(list):
    """Source provenance retained through v1 schema projection."""

    def __init__(self, values: list[str], selection_kind: str = "") -> None:
        super().__init__(values)
        self.selection_kind = selection_kind


def annotate_files(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: annotate_files(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        converted = [annotate_files(item) for item in value]
        if len(value) in (1, 2) and isinstance(value[0], (str, list)):
            options = value[1] if len(value) == 2 else {}
            if isinstance(options, dict):
                choices = value[0] if isinstance(value[0], FileChoices) else options.get("options")
                if isinstance(choices, FileChoices) or any(
                    options.get(key) for key in ("image_upload", "audio_upload", "video_upload")
                ):
                    if len(converted) == 1:
                        converted.append({})
                    converted[1]["file_input"] = True
                    if isinstance(choices, FileChoices) and choices.selection_kind:
                        converted[1]["asset_selection_kind"] = choices.selection_kind
        return converted
    return value


def annotated_file_arguments(cls: type) -> set[str]:
    """Recognize file choices resolved directly by the declared execution method."""
    function = getattr(cls, getattr(cls, "FUNCTION", ""), None)
    if function is None:
        return set()
    try:
        source = textwrap.dedent(inspect.getsource(function))
        parameters = inspect.signature(function).parameters
    except (OSError, TypeError):
        return set()
    return {
        call.args[0].id
        for call in ast.walk(ast.parse(source))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "folder_paths"
        and call.func.attr == "get_annotated_filepath"
        and call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id in parameters
    }


def engine_semantics(cls: type, source: str) -> list[str]:
    semantics = set()
    for item in ast.walk(ast.parse(inspect.getsource(cls))):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "check_lazy_status":
                semantics.add("lazy inputs")
        if not isinstance(item, ast.Call):
            continue
        call = ast.unparse(item.func)
        keywords = {keyword.arg: keyword.value for keyword in item.keywords}
        for keyword, semantic in (
            ("lazy", "lazy inputs"),
            ("enable_expand", "GraphBuilder expansion"),
        ):
            value = keywords.get(keyword)
            if isinstance(value, ast.Constant) and value.value is True:
                semantics.add(semantic)
        if call.endswith("ExecutionBlocker") or "block_execution" in keywords:
            semantics.add("ExecutionBlocker")
        if call.endswith("GraphBuilder") or (call.endswith("NodeOutput") and "expand" in keywords):
            semantics.add("GraphBuilder expansion")
    if source == "comfy_extras/nodes_loop.py":
        semantics.add("loop semantics")
    return sorted(semantics)


def v3_contract(schema: Any) -> dict[str, Any]:
    """Read finalized declarations instead of potentially inherited V1 caches."""
    return {
        **({"display_name": schema.display_name} if schema.display_name is not None else {}),
        "is_v3": True,
        "v3_inputs": [{"id": item.id} for item in schema.inputs],
        "category": schema.category,
        "input_is_list": schema.is_input_list,
        "output_node": schema.is_output_node,
        "api_node": schema.is_api_node,
        "return_types": [item.io_type for item in schema.outputs],
        "return_names": [item.display_name for item in schema.outputs],
        "output_is_list": [item.is_output_list for item in schema.outputs],
        "v3_outputs": [
            {"template": item.template.as_dict()} if hasattr(item, "template") else {}
            for item in schema.outputs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.checkout.resolve()
    output = args.output.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(root))
    os.chdir(root)
    sys.argv = ["main.py", "--cpu", "--disable-auto-launch"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import folder_paths
    import nodes

    get_filename_list = folder_paths.get_filename_list

    def file_choices(category: str) -> list[str]:
        return FileChoices(get_filename_list(category), "model/vae" if category == "vae" else "")

    folder_paths.get_filename_list = file_choices
    get_input_subfolders = folder_paths.get_input_subfolders
    folder_paths.get_input_subfolders = lambda: FileChoices(
        get_input_subfolders(), "directory/input"
    )
    model_paths = nodes.DiffusersLoader._model_paths
    nodes.DiffusersLoader._model_paths = classmethod(
        lambda cls: FileChoices(model_paths(), "model/diffusers")
    )

    async def dump() -> dict[str, Any]:
        base_names = set(nodes.NODE_CLASS_MAPPINGS)
        await nodes.init_builtin_extra_nodes()
        dataset_cls = nodes.NODE_CLASS_MAPPINGS.get("LoadTrainingDataset")
        if dataset_cls is not None:
            dataset_module = inspect.getmodule(dataset_cls)
            if dataset_module is not None:
                list_dataset_folders = dataset_module.list_dataset_folders
                dataset_module.list_dataset_folders = lambda: FileChoices(
                    list_dataset_folders(), "directory/dataset"
                )
        rows = []
        for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
            source = inspect.getsourcefile(cls)
            row: dict[str, Any] = {
                "class_name": name,
                "file": str(Path(source).relative_to(root)) if source else None,
                "source": "nodes.py" if name in base_names else "comfy_extras",
                "display_name": nodes.NODE_DISPLAY_NAME_MAPPINGS.get(name),
                "file_inputs": [],
            }
            row["engine_semantics"] = engine_semantics(cls, row["file"])
            try:
                inputs = annotate_files(cls.INPUT_TYPES())
                row["input_types"] = inputs
                source_files = annotated_file_arguments(cls)
                for group in ("required", "optional"):
                    for key, spec in inputs.get(group, {}).items():
                        options = spec[1] if len(spec) > 1 else {}
                        if options.get("file_input") or key in source_files:
                            row["file_inputs"].append(key)
            except Exception as exc:
                row["input_types"] = None
                row["input_types_error"] = f"{type(exc).__name__}: {exc}"
            for attribute in (
                "RETURN_TYPES",
                "RETURN_NAMES",
                "OUTPUT_IS_LIST",
                "INPUT_IS_LIST",
                "CATEGORY",
                "OUTPUT_NODE",
                "API_NODE",
            ):
                row[attribute.lower()] = getattr(cls, attribute, None)
            if hasattr(cls, "GET_SCHEMA"):
                try:
                    row.update(v3_contract(cls.GET_SCHEMA()))
                except Exception as exc:
                    row["v3_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        return {
            "source_commit": commit,
            "count_nodes_py": len(base_names),
            "count_total": len(rows),
            "engine_exclusions": {
                row["class_name"]: row["engine_semantics"]
                for row in rows
                if row["engine_semantics"]
            },
            "nodes": rows,
        }

    result = asyncio.run(dump())
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "nodes"}))


if __name__ == "__main__":
    main()
