"""Maintained aliases owned by the native compatibility pack."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

from dinkster_compat_comfy import LoadVae, LoadVision
from dinkster_schema import (
    build_schemas,
    comfy_alias_registry_from_wire,
    comfy_alias_registry_problems,
    comfy_alias_registry_to_wire,
    schema_from_wire,
)

ROOT = Path(__file__).parent.parent
ALIAS_PATH = ROOT / "packages" / "dinkster-compat-comfy" / "comfy-aliases.json"


def _registry() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(ALIAS_PATH.read_text(encoding="utf-8")))


def test_compat_comfy_aliases_are_canonical() -> None:
    payload = _registry()
    registry = comfy_alias_registry_from_wire(payload)
    assert comfy_alias_registry_to_wire(registry) == payload
    assert comfy_alias_registry_problems(registry, build_schemas((LoadVae, LoadVision))) == ()

    records = {record["source"]["nodeClass"]: record for record in payload["records"]}
    assert set(records) == {"CLIPVisionLoader", "VAELoader"}
    assert {record["source"]["revision"] for record in records.values()} == {"b78cec87"}
    assert records["CLIPVisionLoader"]["replacement"]["cases"][0]["inputs"] == {
        "vision_encoder": {"kind": "copy", "input": "clip_name"}
    }
    assert records["VAELoader"]["replacement"]["cases"][0]["inputs"] == {
        "vae": {"kind": "copy", "input": "vae_name"}
    }


def test_vae_loader_alias_is_asset_bound_and_executable() -> None:
    registry = _registry()
    records = {record["source"]["nodeClass"]: record for record in registry["records"]}
    source_schemas = {
        source["nodeType"]: schema_from_wire(source) for source in registry["sourceSchemas"]
    }
    source = source_schemas["comfy.VAELoader"]
    assert source.inputs[0].id == "vae_name"
    assert source.inputs[0].type.types == ("dinkster.asset",)
    assert source.inputs[0].widget is not None
    assert cast(Any, source.inputs[0].widget).kind == "model/vae"
    assert records["VAELoader"]["source"] == {
        "pack": "comfy-core",
        "nodeClass": "VAELoader",
        "nodeType": "comfy.VAELoader",
        "revision": "b78cec87",
    }


def test_compat_pack_bundles_manifest_nodes_and_alias_registry() -> None:
    configuration = cast(
        "dict[str, Any]",
        tomllib.loads(
            (ROOT / "packages" / "dinkster-compat-comfy" / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        ),
    )
    included = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included == {
        "dinkster-pack.toml": "dinkster_compat_comfy_pack/dinkster-pack.toml",
        "comfy-aliases.json": "dinkster_compat_comfy_pack/comfy-aliases.json",
        "src/dinkster_compat_comfy": "dinkster_compat_comfy_pack/dinkster_compat_comfy",
    }
