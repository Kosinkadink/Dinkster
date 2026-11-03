"""Record new immutable-reference preview metadata and served-schema cells once."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer
from dinkster_nodes_comfy.preview3d import (
    Preview3DAdvanced,
    PreviewGaussianSplat,
    PreviewPointCloud,
)
from dinkster_server import create_app

from dinkster.compose import PackSpec, ServingComposer

REFERENCE = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
SOURCE_SHA256 = "414933ee218ab11bd25d4b87b51688a92aec69be2421d5d94dbddaade83c8e50"
NODES = (Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud)


class OriginalFile:
    format = "gltf"

    def save_to(self, destination: str) -> None:
        # The reference's temporary copy is replaced by the agreed original-asset boundary.
        assert destination.endswith(".gltf")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("reference_dump", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == SOURCE_SHA256
    receipt = (
        json.loads(args.receipt.read_text())
        if args.receipt.exists()
        else {"reference": REFERENCE, "source_sha256": SOURCE_SHA256, "cells": {}}
    )
    assert receipt["reference"] == REFERENCE and receipt["source_sha256"] == SOURCE_SHA256

    def record(name: str, value: object) -> None:
        receipt["cells"][name] = value
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(name, value, flush=True)

    namespace = dict(
        uuid=uuid,
        os=os,
        Types=SimpleNamespace(File3D=OriginalFile),
        folder_paths=SimpleNamespace(get_temp_directory=lambda: "reference-temp"),
        IO=SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values, **kwargs: values),
        UI=SimpleNamespace(Saved3DModels=lambda *args: None, SavedResult=lambda *args: None),
    )
    namespace["IO"].FolderType = SimpleNamespace(temp="temp")
    definitions = [
        n
        for n in ast.parse(args.source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name in {node.__name__ for node in NODES}
    ]
    assert len(definitions) == 3
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(args.source), "exec"), namespace
    )
    camera, placement = {"zoom": 2.5}, [{"position": {"x": 3}}]
    for node in NODES:
        for name, state, explicit_camera, explicit_placement in (
            ("viewport", {"camera_info": camera, "model_3d_info": placement}, None, None),
            ("explicit-empty", {"camera_info": camera, "model_3d_info": placement}, {}, []),
            ("null-placement", {"model_3d_info": None}, None, None),
            ("non-dict", None, None, None),
        ):
            key = f"{node.__name__}-{name}"
            if key in receipt["cells"]:
                continue
            source = OriginalFile()
            kwargs = dict(
                model_3d=source,
                viewport_state=state,
                width=371,
                height=209,
                camera_info=explicit_camera,
                model_3d_info=explicit_placement,
            )
            expected = namespace[node.__name__].execute(**kwargs)
            actual = node.execute(**kwargs)
            assert actual["model_3d"] is expected[0] is source
            assert (
                tuple(actual[k] for k in ("model_3d_info", "camera_info", "width", "height"))
                == expected[1:]
            )
            record(key, {"identical": True})

    pending = [node.__name__ for node in NODES if f"{node.__name__}-schema" not in receipt["cells"]]
    if not pending:
        return
    os.environ["DINKSTER_SERVING_PYTHON"] = sys.executable
    composer = ServingComposer()
    client = None
    try:
        await composer.add_pack(
            PackSpec(
                manifest=Path("packages/dinkster-nodes-comfy/dinkster-pack.toml").resolve(),
                in_process=False,
            )
        )
        c = composer.composition
        app = create_app(
            c.make_engine,
            c.schemas,
            packs=c.packs,
            node_packs=c.node_packs,
            choices=c.choices,
            lazy_choices=c.lazy_choices,
            schema_owners=c.schema_owners,
            choice_owners=c.choice_owners,
        )
        client = TestClient(TestServer(app, host="127.0.0.1", port=0))
        await client.start_server()
        async with client.get("/api/nodes") as response:
            assert response.status == 200
            registry = await response.json()
        reference = json.loads(args.reference_dump.read_text())
        spec = importlib.util.spec_from_file_location("compare_nodes", "tools/compare_nodes.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in pending:
            ref = {**reference, "nodes": [n for n in reference["nodes"] if n["class_name"] == name]}
            definition = next(n for n in definitions if n.name == name)
            schema_call = next(
                n
                for n in ast.walk(definition)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "Schema"
            )
            inputs = next(k.value for k in schema_call.keywords if k.arg == "inputs")
            assert isinstance(inputs, ast.List)
            declarations = []
            for item in inputs.elts:
                assert isinstance(item, ast.Call) and isinstance(item.args[0], ast.Constant)
                assert isinstance(item.args[0].value, str)
                declarations.append({"id": item.args[0].value})
            # Older dumps omit V3 ordering; recover it from the hash-verified fixed source.
            ref["nodes"] = [
                {**row, "is_v3": True, "v3_inputs": declarations} for row in ref["nodes"]
            ]
            served = {**registry, "nodes": {name: registry["nodes"][name]}}
            result = module.compare_registry(ref, served)
            assert result["passed"], result
            record(f"{name}-schema", result)
    finally:
        if client is not None:
            await client.close()
        await composer.close()


if __name__ == "__main__":
    asyncio.run(main())
