from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from dinkster_nodes_image import (
    AnimeLineartPreprocessor,
    AnyLinePreprocessor,
    MangaLineartPreprocessor,
    MLSDPreprocessor,
    ModelEdgePreprocessor,
    RealisticLineartPreprocessor,
    TEEDPreprocessor,
)
from dinkster_schema import (
    ComboWidget,
    comfy_alias_registry_from_wire,
    validate_replacement_references,
)

ROOT = Path(__file__).parent.parent
ALIASES = ROOT / "packages" / "dinkster-nodes-image" / "comfy-aliases.json"

SUPPORTED = {
    "LineArtPreprocessor": "dinkster.preprocess.lineart_realistic",
    "AnimeLineArtPreprocessor": "dinkster.preprocess.lineart_anime",
    "Manga2Anime_LineArt_Preprocessor": "dinkster.preprocess.lineart_manga",
    "AnyLineArtPreprocessor_aux": "dinkster.preprocess.anyline",
    "HEDPreprocessor": "dinkster.preprocess.model_edges",
    "FakeScribblePreprocessor": "dinkster.preprocess.model_edges",
    "TEEDPreprocessor": "dinkster.preprocess.teed",
    "M-LSDPreprocessor": "dinkster.preprocess.mlsd",
}
REFUSED = {
    "PiDiNetPreprocessor",
    "Scribble_PiDiNet_Preprocessor",
    "DiffusionEdge_Preprocessor",
}


def _wire() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(ALIASES.read_text(encoding="utf-8")))


def _records() -> dict[str, dict[str, object]]:
    return {
        cast("str", cast("dict[str, object]", record["source"])["nodeClass"]): record
        for record in cast("list[dict[str, object]]", _wire()["records"])
        if cast("dict[str, object]", record["source"])["nodeClass"] in set(SUPPORTED) | REFUSED
    }


def _cases(record: dict[str, object]) -> list[dict[str, object]]:
    replacement = cast("dict[str, object]", record["replacement"])
    return cast("list[dict[str, object]]", replacement["cases"])


def test_line_edge_aliases_are_owned_by_image_schemas_and_reference_valid() -> None:
    registry = comfy_alias_registry_from_wire(_wire())
    records = _records()
    assert set(records) == set(SUPPORTED) | REFUSED
    assert set(SUPPORTED) | REFUSED <= {
        snapshot.schema.node_type.rsplit(".", 1)[-1] for snapshot in registry.source_schemas
    }

    targets = {
        schema.node_type: schema
        for schema in (
            ModelEdgePreprocessor.schema(),
            RealisticLineartPreprocessor.schema(),
            AnimeLineartPreprocessor.schema(),
            MangaLineartPreprocessor.schema(),
            AnyLinePreprocessor.schema(),
            TEEDPreprocessor.schema(),
            MLSDPreprocessor.schema(),
        )
    }
    sources = {snapshot.schema.node_type: snapshot.schema for snapshot in registry.source_schemas}
    schemas = {**sources, **targets}
    for record in registry.records:
        schemas[record.source.node_type] = replace(
            sources[record.source.node_type], replacements=(record.replacement,)
        )
    assert validate_replacement_references(schemas) == ()


def test_line_edge_aliases_preserve_parameters_and_refuse_excluded_models() -> None:
    records = _records()
    for node_class, target in SUPPORTED.items():
        assert all(case["to"] == target for case in _cases(records[node_class]))

    lineart_cases = _cases(records["LineArtPreprocessor"])
    assert cast("dict[str, object]", lineart_cases[0]["when"])["value"] == "enable"
    assert cast("dict[str, object]", lineart_cases[0]["inputs"])["coarse"] == {
        "kind": "constant",
        "value": True,
    }
    assert cast("dict[str, object]", lineart_cases[1]["inputs"])["coarse"] == {
        "kind": "constant",
        "value": False,
    }

    hed_cases = _cases(records["HEDPreprocessor"])
    fake_cases = _cases(records["FakeScribblePreprocessor"])
    assert cast("dict[str, object]", hed_cases[0]["inputs"])["scribble"] == {
        "kind": "constant",
        "value": False,
    }
    assert cast("dict[str, object]", fake_cases[0]["inputs"])["scribble"] == {
        "kind": "constant",
        "value": True,
    }
    assert cast("dict[str, object]", hed_cases[0]["inputs"])["safe"] == {
        "kind": "constant",
        "value": False,
    }
    assert cast("dict[str, object]", hed_cases[1]["inputs"])["safe"] == {
        "kind": "constant",
        "value": True,
    }

    anyline_inputs = cast(
        "dict[str, object]", _cases(records["AnyLineArtPreprocessor_aux"])[0]["inputs"]
    )
    assert set(anyline_inputs) == {
        "image",
        "provider",
        "merge_with_lineart",
        "resolution",
        "lineart_lower_bound",
        "lineart_upper_bound",
        "object_min_size",
        "object_connectivity",
    }
    for input_id in set(anyline_inputs) - {"provider"}:
        assert anyline_inputs[input_id] == {"kind": "copy", "input": input_id}

    mlsd_inputs = cast("dict[str, object]", _cases(records["M-LSDPreprocessor"])[0]["inputs"])
    assert mlsd_inputs["distance_threshold"] == {"kind": "copy", "input": "dist_threshold"}

    for node_class in REFUSED:
        replacement = cast("dict[str, object]", records[node_class]["replacement"])
        assert "not translated" in cast("str", replacement["note"])
        provider = cast("dict[str, object]", _cases(records[node_class])[0]["inputs"])["provider"]
        transform = cast("dict[str, object]", cast("dict[str, object]", provider)["transform"])
        assert transform == {"kind": "enumRename", "map": {}}


def test_line_edge_source_widgets_retain_upstream_defaults_and_spellings() -> None:
    registry = comfy_alias_registry_from_wire(_wire())
    sources = {
        snapshot.schema.node_type.rsplit(".", 1)[-1]: snapshot.schema
        for snapshot in registry.source_schemas
    }
    merge = sources["AnyLineArtPreprocessor_aux"].input("merge_with_lineart")
    assert merge is not None and merge.default == "lineart_standard"
    assert isinstance(merge.widget, ComboWidget)
    assert merge.widget.options == (
        "lineart_standard",
        "lineart_realisitic",
        "lineart_anime",
        "manga_line",
    )
    environment = sources["DiffusionEdge_Preprocessor"].input("environment")
    assert environment is not None and environment.default == "indoor"
    assert isinstance(environment.widget, ComboWidget)
    assert environment.widget.options == ("indoor", "urban", "natrual")
    distance = sources["M-LSDPreprocessor"].input("dist_threshold")
    assert distance is not None and distance.default == 0.1
