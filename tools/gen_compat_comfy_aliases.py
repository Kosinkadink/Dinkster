"""Generate maintained ComfyUI aliases owned by the compatibility pack."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from dinkster_compat_comfy import CompatTranslation, translate_node
from dinkster_schema import AssetWidget, MappingSource, ReplacementCase, ReplacementRule, TypeExpr
from dinkster_schema.model import NodeSchema
from dinkster_schema.replace import rule_to_wire
from dinkster_schema.wire import schema_to_wire

COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
COMFY_REVISION = "b78cec87"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "dinkster-compat-comfy" / "comfy-aliases.json"


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _asset_source(
    node_class: str,
    implementation: type[Any],
    *,
    input_id: str,
    asset_kind: str,
) -> NodeSchema:
    source = translate_node(node_class, implementation, CompatTranslation()).schema()
    if len(source.inputs) != 1 or source.inputs[0].id != input_id:
        raise RuntimeError(f"unexpected {node_class} source schema")
    return replace(
        source,
        inputs=(
            replace(
                source.inputs[0],
                type=TypeExpr.concrete("dinkster.asset"),
                default=None,
                widget=AssetWidget(
                    accept=("application/octet-stream",),
                    kind=asset_kind,
                ),
            ),
        ),
    )


def _record(
    node_class: str,
    carrier: str,
    source: NodeSchema,
    *,
    source_input: str,
    target_input: str,
    source_output: str,
    target_output: str,
    evidence: str,
) -> dict[str, object]:
    rule = ReplacementRule(
        from_type=source.node_type,
        cases=(
            ReplacementCase.build(
                carrier,
                inputs={target_input: MappingSource.copy(source_input)},
                outputs={target_output: source_output},
            ),
        ),
    )
    return {
        "id": f"comfy_alias:comfy-core/{node_class}",
        "mappingKind": "op",
        "carrier": carrier,
        "source": {
            "pack": "comfy-core",
            "nodeClass": node_class,
            "nodeType": source.node_type,
            "revision": COMFY_REVISION,
        },
        "replacement": rule_to_wire(rule),
        "confidence": {"tier": "exact", "evidence": [evidence]},
    }


def build_registry(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != COMFY_BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {COMFY_BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    comfy_args.cpu = True
    from nodes import CLIPVisionLoader, VAELoader  # pyright: ignore[reportMissingImports]

    vision = _asset_source(
        "CLIPVisionLoader",
        CLIPVisionLoader,
        input_id="clip_name",
        asset_kind="model/clip-vision",
    )
    vae = _asset_source(
        "VAELoader",
        VAELoader,
        input_id="vae_name",
        asset_kind="model/vae",
    )
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(vision), schema_to_wire(vae)],
        "records": [
            _record(
                "CLIPVisionLoader",
                "dinkster.load_vision",
                vision,
                source_input="clip_name",
                target_input="vision_encoder",
                source_output="clip_vision",
                target_output="vision",
                evidence=(
                    "tests/test_compat_comfy_aliases.py::test_compat_comfy_aliases_are_canonical"
                ),
            ),
            _record(
                "VAELoader",
                "dinkster.load_vae",
                vae,
                source_input="vae_name",
                target_input="vae",
                source_output="vae",
                target_output="vae",
                evidence=(
                    "tests/test_compat_comfy_aliases.py::"
                    "test_vae_loader_alias_is_asset_bound_and_executable"
                ),
            ),
        ],
    }


def main() -> None:
    root = Path(os.environ.get("COMFYUI_ROOT", REPO.parent / "ComfyUI"))
    registry = build_registry(root)
    encoded = json.dumps(registry, sort_keys=True, separators=(",", ":")) + "\n"
    OUT.write_text(encoded, encoding="utf-8", newline="\n")
    print(f"{OUT}: sha256={hashlib.sha256(encoded.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
