from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

CENSUS_PATH = (
    Path(__file__).parent.parent / "tools" / "data" / "loader_stack_census_2026-08-28.json"
)


def _census() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(CENSUS_PATH.read_text(encoding="utf-8")))


def test_loader_stack_census_is_complete_and_source_pinned() -> None:
    census = _census()
    assert set(census) == {"format", "selection", "registrySource", "sources", "records"}
    assert census["format"] == "dinkster-loader-stack-census/1"
    assert census["registrySource"] == {
        "repository": "Comfy-Org/ComfyUI-Manager",
        "revision": "3f159c5f651f6f3cf14ee0d51267bc433ade9a85",
        "file": "extension-node-map.json",
    }

    sources = census["sources"]
    assert len(sources) == 10
    assert all(set(source) == {"repository", "revision", "qualifyingCount"} for source in sources)
    assert len({source["repository"] for source in sources}) == len(sources)
    assert all(re.fullmatch(r"[0-9a-f]{40}", source["revision"]) for source in sources)
    assert all(
        type(source["qualifyingCount"]) is int and source["qualifyingCount"] >= 0
        for source in sources
    )

    records = census["records"]
    record_fields = {
        "repository",
        "nodeType",
        "symbol",
        "sourceFile",
        "signature",
        "shape",
        "disposition",
        "reason",
    }
    assert all(set(record) == record_fields for record in records)
    identities = [(record["repository"], record["nodeType"]) for record in records]
    assert len(records) == len(set(identities)) == 46
    assert {record["repository"] for record in records} <= {
        source["repository"] for source in sources
    }
    assert all(record["sourceFile"].endswith(".py") and record["symbol"] for record in records)
    assert all(record["signature"] and record["reason"] for record in records)

    expected_by_repository = {source["repository"]: source["qualifyingCount"] for source in sources}
    actual_by_repository = Counter(record["repository"] for record in records)
    assert {
        repository: actual_by_repository[repository] for repository in expected_by_repository
    } == expected_by_repository


def test_loader_stack_census_has_a_closed_disposition_for_every_record() -> None:
    records = _census()["records"]
    assert Counter(record["disposition"] for record in records) == {
        "native-shortcut": 7,
        "canonical-chain": 7,
        "quarantine": 32,
    }
    assert Counter(record["shape"] for record in records) == {
        "advanced-lora": 6,
        "cache-loader": 3,
        "checkpoint-stack": 6,
        "lora-stack": 7,
        "script-loader": 1,
        "sdxl-base-refiner-stack": 1,
        "single-loader": 9,
        "specialized-loader": 4,
        "specialized-pipeline": 7,
        "stable-cascade-stack": 2,
    }
