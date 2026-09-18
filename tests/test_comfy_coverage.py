from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import tools.comfy_coverage as coverage
from tools.comfy_confidence import create_receipt, write_receipt
from tools.comfy_coverage import (
    ComfyUISource,
    CoverageError,
    DownloadSnapshot,
    EvidenceSource,
    LowStepLoraPath,
    ModelEvidence,
    NodeOccurrence,
    RegistryCatalog,
    RegistryMapping,
    Workflow,
    build_evidence_ledger,
    build_report,
    build_template_features,
    canonical_json,
    lint_supported_claims,
    load_comfyui_source,
    load_download_snapshot,
    load_evidence_source,
    load_registry_catalog,
    load_source_parity_baseline,
    load_workflows,
    render_evidence_markdown,
    render_markdown,
)

REPO = Path(__file__).resolve().parents[1]


def snapshot() -> DownloadSnapshot:
    return DownloadSnapshot(
        captured_at="2026-08-26",
        source="https://registry.example.invalid/nodes",
        total_packs=901,
        total_downloads=1_000_000,
        rank_355_downloads=900_000,
        rank_900_downloads=950_000,
        ranks={
            "top-pack": (10, "top-pack", 1_000),
            "lower-pack": (500, "lower-pack", 500),
        },
    )


def mapping(
    source_pack: str,
    source_name: str,
    *,
    mapping_kind: str = "op",
    target_provider: str = "native-pack",
    tier: str = "exact",
    tolerances: tuple[tuple[str, str, float], ...] = (),
    refusal: bool = False,
) -> RegistryMapping:
    return RegistryMapping(
        registry_id=f"comfy_alias:{source_pack}/{source_name}",
        record_digest="sha256:" + source_name.casefold().encode().hex().ljust(64, "0")[:64],
        registry_kind="alias",
        mapping_kind=mapping_kind,
        source_pack=source_pack,
        source_name=source_name,
        revision="b78cec87" if source_pack == "comfy-core" else "static-revision",
        carrier=f"native.{source_name.lower()}",
        target_provider=target_provider,
        tier=tier,
        evidence=("tests/reference.json",),
        tolerances=tolerances,
        family_id="native.values" if mapping_kind == "family" else None,
        family_provider="native-pack" if mapping_kind == "family" else None,
        refusal=refusal,
    )


def catalog(
    *records: RegistryMapping, available: tuple[str, ...] = ("native-pack",)
) -> RegistryCatalog:
    return RegistryCatalog(
        mappings=tuple(records),
        available_providers=frozenset(available),
        sidecars=(("packages/native-pack/comfy-aliases.json", "sha256:" + "0" * 64),),
    )


def workflow(
    path: str,
    *nodes: tuple[str, str | None],
    subgraph_ids: tuple[str, ...] = (),
) -> Workflow:
    return Workflow(
        path=path,
        nodes=tuple(
            NodeOccurrence(node_type=node_type, explicit_pack=pack) for node_type, pack in nodes
        ),
        subgraph_ids=frozenset(subgraph_ids),
        subgraph_count=len(subgraph_ids),
    )


def test_workflow_loader_includes_root_and_subgraph_nodes_and_reports_other_json(
    tmp_path: Path,
) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "workflow.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"type": "Root", "properties": {"cnr_id": "comfy-core"}},
                    {"type": "subgraph-id"},
                ],
                "definitions": {
                    "subgraphs": [
                        {
                            "id": "subgraph-id",
                            "nodes": [
                                {
                                    "type": "Inside",
                                    "properties": {"cnr_id": "top-pack"},
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="ascii",
    )
    (templates / "index.json").write_text('{"templates":[]}', encoding="ascii")

    workflows, ignored = load_workflows(templates)

    assert ignored == ("index.json",)
    assert workflows == (
        Workflow(
            path="workflow.json",
            nodes=(
                NodeOccurrence("Root", "comfy-core"),
                NodeOccurrence("subgraph-id", None),
                NodeOccurrence("Inside", "top-pack"),
            ),
            subgraph_ids=frozenset({"subgraph-id"}),
            subgraph_count=1,
        ),
    )


@pytest.mark.parametrize(
    "document, message",
    (
        ('{"nodes":[],"nodes":[]}', "duplicate JSON object key"),
        ('{"nodes":[],"value":NaN}', "non-finite JSON number"),
    ),
)
def test_workflow_loader_rejects_noncanonical_json(
    tmp_path: Path, document: str, message: str
) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "hostile.json").write_text(document, encoding="ascii")

    with pytest.raises(CoverageError, match=message):
        load_workflows(templates)


def test_workflow_loader_rejects_links(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"nodes":[]}', encoding="ascii")
    try:
        (templates / "linked.json").symlink_to(outside)
    except OSError:
        pytest.skip("this host does not permit test symlinks")

    with pytest.raises(CoverageError, match="escapes its root|ordinary file"):
        load_workflows(templates)


def test_report_classifies_sources_and_reports_op_family_confidence_separately(
    tmp_path: Path,
) -> None:
    records = catalog(
        mapping("comfy-core", "Exact"),
        mapping(
            "top-pack",
            "Family",
            mapping_kind="family",
            tier="equivalent",
            tolerances=(("max_abs", "<=", 0.001),),
        ),
        mapping("lower-pack", "Unavailable", target_provider="missing-pack"),
        mapping("outside-pack", "RegistryOnly", tier="parametric"),
    )
    workflows = (
        workflow(
            "first.json",
            ("Exact", "comfy-core"),
            ("Exact", None),
            ("Family", "top-pack"),
            ("Unavailable", "lower-pack"),
            ("RegistryOnly", None),
            ("KnownButUnmapped", "top-pack"),
            ("Unknown", None),
            ("Note", None),
            ("local-subgraph", None),
            subgraph_ids=("local-subgraph",),
        ),
    )

    report = build_report(
        workflows=workflows,
        ignored_files=(),
        catalog=records,
        snapshot=snapshot(),
        receipts_root=tmp_path / "no-receipts",
    )
    result = cast("dict[str, Any]", report["coverage"])

    assert result["statusCounts"] == {
        "mapped": 4,
        "quarantine": 1,
        "unavailable": 1,
        "unsupported": 1,
        "structural": 2,
    }
    assert result["mappingKindOccurrences"] == {"op": 3, "family": 1}
    assert result["confidenceTierOccurrences"] == {
        "exact": 2,
        "parametric": 1,
        "equivalent": 1,
        "grouped": 0,
    }
    assert result["confidenceTierOccurrencesByMappingKind"] == {
        "op": {"exact": 2, "parametric": 1, "equivalent": 0, "grouped": 0},
        "family": {"exact": 0, "parametric": 0, "equivalent": 1, "grouped": 0},
    }
    assert result["rankBands"]["core"]["mapped"] == 2
    assert result["rankBands"]["top-355"]["mapped"] == 1
    assert result["rankBands"]["rank-356-900"]["unavailable"] == 1
    assert result["rankBands"]["outside-top-900"]["mapped"] == 1
    assert result["translationReadyWorkflows"] == 0
    assert result["supportedWorkflows"] == 0
    assert result["workflowEvidenceStates"]["refused"] == 1
    assert report["workflows"][0]["capabilityEvidence"]["state"] == "refused"  # type: ignore[index]
    assert report["registry"]["declarationsByKind"] == {"op": 3, "family": 1}  # type: ignore[index]


def test_alias_translation_is_t1_and_never_counts_as_support(tmp_path: Path) -> None:
    report = build_report(
        workflows=(workflow("workflow.json", ("Exact", "comfy-core")),),
        ignored_files=(),
        catalog=catalog(mapping("comfy-core", "Exact")),
        snapshot=snapshot(),
        receipts_root=tmp_path / "none",
    )

    evidence = report["workflows"][0]["capabilityEvidence"]  # type: ignore[index]
    disposition = report["workflows"][0]["dispositions"][0]  # type: ignore[index]
    assert evidence == {
        "highestProvenTier": "T1",
        "state": "proven",
        "supported": False,
        "weakestCapabilities": [
            {
                "evidenceState": "proven",
                "highestProvenTier": "T1",
                "nodeType": "Exact",
                "sourcePack": "comfy-core",
            }
        ],
    }
    assert disposition["supported"] is False
    assert report["coverage"]["supportedWorkflows"] == 0  # type: ignore[index]


def test_report_does_not_guess_ambiguous_source_ownership(tmp_path: Path) -> None:
    workflows = (
        workflow("a.json", ("Shared", "pack-a")),
        workflow("b.json", ("Shared", "pack-b")),
        workflow("unknown.json", ("Shared", None)),
    )

    report = build_report(
        workflows=workflows,
        ignored_files=(),
        catalog=catalog(),
        snapshot=snapshot(),
        receipts_root=tmp_path / "none",
    )
    coverage_result = cast("dict[str, Any]", report["coverage"])
    unknown = cast("dict[str, Any]", report["workflows"][2])  # type: ignore[index]

    assert coverage_result["statusCounts"]["quarantine"] == 2
    assert coverage_result["statusCounts"]["unsupported"] == 1
    assert unknown["dispositions"][0]["reason"] == "ambiguous-source-pack"
    assert unknown["dispositions"][0]["providers"] == ["pack-a", "pack-b"]


def test_report_classifies_an_explicit_refusal_as_unsupported(tmp_path: Path) -> None:
    refused = mapping("top-pack", "Refused", refusal=True)
    report = build_report(
        workflows=(workflow("refused.json", ("Refused", "top-pack")),),
        ignored_files=(),
        catalog=catalog(refused),
        snapshot=snapshot(),
        receipts_root=tmp_path / "none",
    )

    coverage_result = cast("dict[str, Any]", report["coverage"])
    disposition = cast("dict[str, Any]", report["workflows"][0])["dispositions"][0]  # type: ignore[index]
    registry_record = cast("dict[str, Any]", report["registry"])["records"][0]
    assert coverage_result["statusCounts"]["unsupported"] == 1
    assert disposition["reason"] == "maintained-native-refusal"
    assert disposition["evidenceState"] == "refused"
    assert disposition["registryId"] == refused.registry_id
    assert registry_record["refusal"] is True
    assert registry_record["target"]["available"] is False


def test_download_snapshot_validates_exact_cutlines_and_rank_order(tmp_path: Path) -> None:
    packs = [
        {"rank": rank, "id": f"pack-{rank:04d}", "downloads": 1_001 - rank}
        for rank in range(1, 901)
    ]
    rank_355 = sum(cast("int", item["downloads"]) for item in packs[:355])
    rank_900 = sum(cast("int", item["downloads"]) for item in packs)
    document = {
        "capturedAt": "2026-08-26",
        "cutlines": [
            {"rank": 355, "cumulativeDownloads": rank_355},
            {"rank": 900, "cumulativeDownloads": rank_900},
        ],
        "format": coverage.SNAPSHOT_FORMAT,
        "packs": packs,
        "source": "https://registry.example.invalid/nodes",
        "totalDownloads": rank_900,
        "totalPacks": 900,
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(document), encoding="ascii")

    loaded = load_download_snapshot(path)
    assert loaded.ranks["pack-0355"][0] == 355
    assert loaded.rank_355_downloads == rank_355

    document["packs"][10], document["packs"][11] = (  # type: ignore[index]
        document["packs"][11],  # type: ignore[index]
        document["packs"][10],  # type: ignore[index]
    )
    path.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(CoverageError, match="rank/downloads|canonical download order"):
        load_download_snapshot(path)


def _write_exact_receipt(
    root: Path,
    record: RegistryMapping,
    *,
    case_id: str = "coverage/exact",
    stem: str = "exact",
    equal: bool = True,
) -> None:
    reference_name = f"{stem}.reference.json"
    native_name = f"{stem}.native.json"
    (root / reference_name).write_text('{"result":1}', encoding="ascii")
    (root / native_name).write_text('{"result":1}' if equal else '{"result":2}', encoding="ascii")
    receipt = create_receipt(
        case_id=case_id,
        mapping={
            "registryId": record.registry_id,
            "mappingKind": record.mapping_kind,
            "tier": record.tier,
            "source": {
                "pack": record.source_pack,
                "name": record.source_name,
                "revision": record.revision,
                "referenceKind": "comfyui-pinned",
            },
            "target": {"kind": "node", "id": record.carrier},
        },
        parameters={"mappingDigest": record.record_digest},
        comparison={"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=root,
        reference_path=reference_name,
        native_path=native_name,
    )
    write_receipt(root / f"{stem}.receipt.json", receipt)


def test_report_verifies_and_associates_canonical_confidence_receipts(tmp_path: Path) -> None:
    record = mapping("comfy-core", "Exact")
    _write_exact_receipt(tmp_path, record)

    report = build_report(
        workflows=(workflow("workflow.json", ("Exact", "comfy-core")),),
        ignored_files=(),
        catalog=catalog(record),
        snapshot=snapshot(),
        receipts_root=tmp_path,
    )

    assert report["receipts"] == {  # type: ignore[comparison-overlap]
        "cases": [
            {
                "caseId": "coverage/exact",
                "mappingKind": "op",
                "pass": True,
                "path": "exact.receipt.json",
                "registryId": record.registry_id,
                "registryKind": "alias",
                "source": {
                    "name": "Exact",
                    "pack": "comfy-core",
                    "revision": "b78cec87",
                },
                "target": {"id": "native.exact", "kind": "node"},
                "tier": "exact",
                "tolerances": [],
            }
        ],
        "countsByMappingKind": {"op": 1, "family": 0},
        "countsByTier": {"exact": 1, "parametric": 0, "equivalent": 0, "grouped": 0},
        "countsByTierAndMappingKind": {
            "op": {"exact": 1, "parametric": 0, "equivalent": 0, "grouped": 0},
            "family": {"exact": 0, "parametric": 0, "equivalent": 0, "grouped": 0},
        },
        "failed": 0,
        "passing": 1,
        "recordsWithPassingReceiptsByMappingKind": {"op": 1, "family": 0},
        "recordsWithPassingReceipts": [record.registry_id],
        "total": 1,
    }

    changed = replace(record, carrier="native.changed")
    with pytest.raises(CoverageError, match="does not match maintained record"):
        build_report(
            workflows=(),
            ignored_files=(),
            catalog=catalog(changed),
            snapshot=snapshot(),
            receipts_root=tmp_path,
        )

    changed_digest = replace(record, record_digest="sha256:" + "f" * 64)
    with pytest.raises(CoverageError, match="stale mapping digest"):
        build_report(
            workflows=(),
            ignored_files=(),
            catalog=catalog(changed_digest),
            snapshot=snapshot(),
            receipts_root=tmp_path,
        )


def test_receipts_enforce_declared_tolerances_and_group_targets(tmp_path: Path) -> None:
    tolerance_root = tmp_path / "tolerance"
    tolerance_root.mkdir()
    (tolerance_root / "reference.json").write_text("1.0", encoding="ascii")
    (tolerance_root / "native.json").write_text("1.01", encoding="ascii")
    equivalent = mapping(
        "comfy-core",
        "Equivalent",
        tier="equivalent",
        tolerances=(("max_abs", "<=", 0.01),),
    )
    widened = create_receipt(
        case_id="coverage/widened",
        mapping={
            "registryId": equivalent.registry_id,
            "mappingKind": "op",
            "tier": "equivalent",
            "source": {
                "pack": "comfy-core",
                "name": "Equivalent",
                "revision": "b78cec87",
                "referenceKind": "comfyui-pinned",
            },
            "target": {"kind": "node", "id": equivalent.carrier},
        },
        parameters={"mappingDigest": equivalent.record_digest},
        comparison={
            "comparator": "numeric-value/1",
            "dataKind": "value",
            "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 0.02}],
        },
        artifact_root=tolerance_root,
        reference_path="reference.json",
        native_path="native.json",
    )
    write_receipt(tolerance_root / "widened.receipt.json", widened)

    with pytest.raises(CoverageError, match="widens a maintained tolerance"):
        build_report(
            workflows=(),
            ignored_files=(),
            catalog=catalog(equivalent),
            snapshot=snapshot(),
            receipts_root=tolerance_root,
        )

    group_root = tmp_path / "group"
    group_root.mkdir()
    (group_root / "reference.json").write_text('{"result":1}', encoding="ascii")
    (group_root / "native.json").write_text('{"result":1}', encoding="ascii")
    grouped = replace(
        mapping("custom-pack", "resize-stack", tier="grouped"),
        registry_id="comfy_group:custom-pack/resize-stack",
        registry_kind="group",
        carrier="native.resize-stack",
    )
    receipt = create_receipt(
        case_id="coverage/group",
        mapping={
            "registryId": grouped.registry_id,
            "mappingKind": "op",
            "tier": "grouped",
            "source": {
                "pack": "custom-pack",
                "name": "resize-stack",
                "revision": "static-revision",
                "referenceKind": "static",
            },
            "target": {"kind": "group", "id": grouped.carrier},
        },
        parameters={"mappingDigest": grouped.record_digest},
        comparison={"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=group_root,
        reference_path="reference.json",
        native_path="native.json",
    )
    write_receipt(group_root / "group.receipt.json", receipt)

    report = build_report(
        workflows=(),
        ignored_files=(),
        catalog=catalog(grouped),
        snapshot=snapshot(),
        receipts_root=group_root,
    )
    case = report["receipts"]["cases"][0]  # type: ignore[index]
    assert case["registryKind"] == "group"
    assert case["target"] == {"kind": "group", "id": "native.resize-stack"}


def _baseline(
    *,
    declarations: int,
    backed: int,
    cases: int,
    unreceipted: dict[str, str] | None = None,
) -> dict[str, object]:
    grandfathered = dict(unreceipted or {})
    assert len(grandfathered) == declarations - backed
    return {
        "format": coverage.SOURCE_PARITY_BASELINE_FORMAT,
        "scope": "maintained-comfy-mapping-records",
        "parityUnit": "unique-mapping-with-verified-passing-receipt",
        "excludedFromParityCounts": [
            "maintained-native-refusals",
            "mapped-workflow-occurrences",
            "translation-ready-workflows",
        ],
        "measured": {
            "translationDeclarations": declarations,
            "receiptBackedMappings": backed,
            "unreceiptedMappings": declarations - backed,
            "passingReceiptCases": cases,
        },
        "grandfatheredUnreceiptedDigest": "sha256:"
        + hashlib.sha256(canonical_json(grandfathered)).hexdigest(),
        "excludedNativePacks": [],
    }


def test_source_parity_counts_unique_mappings_and_rejects_debt_growth(tmp_path: Path) -> None:
    receipted = mapping("comfy-core", "Receipted")
    unreceipted = mapping("comfy-core", "Unreceipted")
    _write_exact_receipt(tmp_path, receipted, case_id="coverage/first", stem="first")
    _write_exact_receipt(tmp_path, receipted, case_id="coverage/second", stem="second")
    baseline = _baseline(
        declarations=2,
        backed=1,
        cases=2,
        unreceipted={unreceipted.registry_id: unreceipted.record_digest},
    )

    report = build_report(
        workflows=(),
        ignored_files=(),
        catalog=catalog(receipted, unreceipted),
        snapshot=snapshot(),
        receipts_root=tmp_path,
        source_parity_baseline=baseline,
    )

    assert report["sourceParity"] == {
        "baseline": baseline,
        "excludedFromParityCounts": [
            "maintained-native-refusals",
            "mapped-workflow-occurrences",
            "translation-ready-workflows",
        ],
        "failingReceiptCases": 0,
        "parityUnit": "unique-mapping-with-verified-passing-receipt",
        "passingReceiptCases": 2,
        "receiptBackedMappings": 1,
        "refusedMappingIds": [],
        "refusedMappings": 0,
        "scope": "maintained-comfy-mapping-records",
        "translationDeclarations": 2,
        "unreceiptedMappingDigest": baseline["grandfatheredUnreceiptedDigest"],
        "unreceiptedMappings": 1,
    }

    new_mapping = mapping("comfy-core", "NewDebt")
    with pytest.raises(CoverageError, match="debt exceeds"):
        build_report(
            workflows=(),
            ignored_files=(),
            catalog=catalog(receipted, unreceipted, new_mapping),
            snapshot=snapshot(),
            receipts_root=tmp_path,
            source_parity_baseline=baseline,
        )
    _write_exact_receipt(tmp_path, new_mapping, case_id="coverage/new", stem="new")
    improved = build_report(
        workflows=(),
        ignored_files=(),
        catalog=catalog(receipted, unreceipted, new_mapping),
        snapshot=snapshot(),
        receipts_root=tmp_path,
        source_parity_baseline=baseline,
    )
    assert improved["sourceParity"]["receiptBackedMappings"] == 2  # type: ignore[index]

    refused = replace(mapping("comfy-core", "Refused"), refusal=True)
    with_refusal = build_report(
        workflows=(),
        ignored_files=(),
        catalog=catalog(receipted, unreceipted, new_mapping, refused),
        snapshot=snapshot(),
        receipts_root=tmp_path,
        source_parity_baseline=baseline,
    )
    source_parity = with_refusal["sourceParity"]
    assert source_parity["translationDeclarations"] == 3  # type: ignore[index]
    assert source_parity["refusedMappings"] == 1  # type: ignore[index]
    assert source_parity["refusedMappingIds"] == [refused.registry_id]  # type: ignore[index]

    with pytest.raises(CoverageError, match="baseline does not match"):
        build_report(
            workflows=(),
            ignored_files=(),
            catalog=catalog(
                receipted,
                replace(unreceipted, record_digest="sha256:" + "f" * 64),
                new_mapping,
            ),
            snapshot=snapshot(),
            receipts_root=tmp_path,
            source_parity_baseline=baseline,
        )


def test_source_parity_rejects_failing_committed_receipts(tmp_path: Path) -> None:
    record = mapping("comfy-core", "Failed")
    _write_exact_receipt(tmp_path, record, equal=False)

    with pytest.raises(CoverageError, match="forbids failing"):
        build_report(
            workflows=(),
            ignored_files=(),
            catalog=catalog(record),
            snapshot=snapshot(),
            receipts_root=tmp_path,
            source_parity_baseline=_baseline(
                declarations=1,
                backed=0,
                cases=0,
                unreceipted={record.registry_id: record.record_digest},
            ),
        )


def test_source_parity_rejects_receipts_for_fail_closed_mappings(tmp_path: Path) -> None:
    refused = replace(mapping("comfy-core", "Refused"), refusal=True)
    _write_exact_receipt(tmp_path, refused)

    with pytest.raises(CoverageError, match="cannot claim parity for a refusal"):
        build_report(
            workflows=(),
            ignored_files=(),
            catalog=catalog(refused),
            snapshot=snapshot(),
            receipts_root=tmp_path,
        )


def test_checked_in_source_parity_baseline_has_explicit_non_parity_dispositions() -> None:
    baseline = load_source_parity_baseline(
        REPO / "docs" / "comfy-source-parity-baseline.json", REPO / "packages"
    )

    assert baseline["measured"] == {
        "passingReceiptCases": 17,
        "receiptBackedMappings": 17,
        "translationDeclarations": 230,
        "unreceiptedMappings": 213,
    }
    assert baseline["excludedFromParityCounts"] == [
        "maintained-native-refusals",
        "mapped-workflow-occurrences",
        "translation-ready-workflows",
    ]
    assert {
        item["package"]
        for item in baseline["excludedNativePacks"]  # type: ignore[union-attr]
    } == {"dinkster-nodes-dev", "dinkster-nodes-partner", "dinkster-nodes-training"}


def test_video_operation_evidence_is_not_counted_as_mapping_parity() -> None:
    report = json.loads((REPO / "docs" / "comfy-translation-coverage.json").read_text())
    backed = set(report["receipts"]["recordsWithPassingReceipts"])
    assert backed.isdisjoint(
        {
            "comfy_alias:comfy-core/CreateVideo",
            "comfy_alias:comfy-core/GetVideoComponents",
        }
    )

    direct = json.loads(
        (REPO / "tests" / "goldens" / "comfy_direct_operations_b78cec87.json").read_text()
    )
    video = next(
        operation
        for operation in direct["operations"]
        if operation["sourceNode"] == "CreateVideo + GetVideoComponents"
    )
    assert video["scope"] == "operation-only-not-mapping-parity"


def test_comfyui_static_source_scan_reads_v1_and_v3_node_ids(tmp_path: Path) -> None:
    (tmp_path / "comfy_extras").mkdir()
    (tmp_path / "comfy_api_nodes").mkdir()
    (tmp_path / "comfy").mkdir()
    (tmp_path / "nodes.py").write_text(
        'NODE_CLASS_MAPPINGS = {"Legacy": object}\n', encoding="ascii"
    )
    (tmp_path / "comfy" / "supported_models.py").write_text(
        "class SourceModel:\n    pass\n", encoding="ascii"
    )
    (tmp_path / "comfy_extras" / "nodes_v3.py").write_text(
        "\ufeffclass Declared:\n"
        '    node_id = "ClassNode"\n'
        "    def define_schema(self):\n"
        '        return IO.Schema(node_id="SchemaNode")\n',
        encoding="utf-8",
    )
    (tmp_path / "comfy_api_nodes" / "__init__.py").write_bytes(b"")

    source = load_comfyui_source(tmp_path)

    assert source.node_ids == frozenset({"Legacy", "ClassNode", "SchemaNode"})
    assert "SourceModel" in source.class_names


def test_evidence_source_enforces_native_drift_and_tier_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "test_family.py").write_text("def test_executes():\n    pass\n", encoding="ascii")
    monkeypatch.setattr(
        coverage,
        "builtin_families",
        lambda: (SimpleNamespace(id="dinkster.test", display_name="Test"),),
    )
    document: dict[str, Any] = {
        "format": coverage.EVIDENCE_SOURCE_FORMAT,
        "modelFamilies": [
            {
                "evidence": [
                    {
                        "kind": "focused-test",
                        "selector": "test_family.py::test_executes",
                    }
                ],
                "exposure": "supported",
                "id": "dinkster.test",
                "sourceIdentifiers": ["TestModel"],
                "state": "proven",
                "tier": "T2",
            }
        ],
        "templateFeatures": {
            "adapterNodeTypes": ["Adapter"],
            "loraNodeTypes": ["Lora"],
            "lowStepLoraPaths": [],
        },
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(document), encoding="ascii")

    loaded = load_evidence_source(path, tmp_path)
    assert loaded.model_families[0].tier == "T2"

    document["modelFamilies"][0]["evidence"][0]["kind"] = "invented-proof"  # type: ignore[index]
    path.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(CoverageError, match="is not a recognized evidence kind"):
        load_evidence_source(path, tmp_path)

    document["modelFamilies"][0]["evidence"][0] = {  # type: ignore[index]
        "kind": "focused-test",
        "selector": "test_family.py",
    }
    path.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(CoverageError, match="test-backed evidence must name a collected test"):
        load_evidence_source(path, tmp_path)

    document["modelFamilies"] = []
    path.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(CoverageError, match="does not match builtin_families"):
        load_evidence_source(path, tmp_path)

    model = {
        "evidence": [{"kind": "focused-test", "selector": "test_family.py::test_executes"}],
        "exposure": "supported",
        "id": "dinkster.test",
        "sourceIdentifiers": ["TestModel"],
        "state": "proven",
        "tier": "T4",
    }
    document["modelFamilies"] = [model]
    path.write_text(json.dumps(document), encoding="ascii")
    with pytest.raises(CoverageError, match="lacks evidence kinds required by T4"):
        load_evidence_source(path, tmp_path)


def test_supported_claim_lint_checks_tier_and_exposure(tmp_path: Path) -> None:
    source = EvidenceSource(
        model_families=(
            ModelEvidence("dinkster.native", "proven", "T2", "supported", ("Native",), ()),
            ModelEvidence("dinkster.hidden", "proven", "T2", "implemented", ("Hidden",), ()),
        ),
        lora_node_types=frozenset({"Lora"}),
        adapter_node_types=frozenset({"Adapter"}),
        low_step_lora_paths=(),
    )
    supported = tmp_path / "SUPPORTED.md"
    supported.write_text(
        "<!-- capability:dinkster.native -->\n"
        "## Implemented capabilities with limited or specialized exposure\n"
        "<!-- capability:dinkster.hidden -->\n",
        encoding="ascii",
    )
    lint_supported_claims(source, supported, tmp_path)

    supported.write_text(
        "## Implemented capabilities with limited or specialized exposure\n"
        "<!-- capability:dinkster.native -->\n"
        "<!-- capability:dinkster.hidden -->\n",
        encoding="ascii",
    )
    with pytest.raises(CoverageError, match="supported family is listed as not exposed"):
        lint_supported_claims(source, supported, tmp_path)


def test_supported_claim_lint_reads_index_and_area_documents(tmp_path: Path) -> None:
    source = EvidenceSource(
        model_families=(
            ModelEvidence("dinkster.native", "proven", "T2", "supported", ("Native",), ()),
            ModelEvidence("dinkster.hidden", "proven", "T2", "implemented", ("Hidden",), ()),
        ),
        lora_node_types=frozenset(),
        adapter_node_types=frozenset(),
        low_step_lora_paths=(),
    )
    supported = tmp_path / "SUPPORTED.md"
    supported.write_text("# Supported\n", encoding="ascii")
    areas = tmp_path / "docs" / "supported"
    areas.mkdir(parents=True)
    (areas / "model-families.md").write_text(
        "<!-- capability:dinkster.native -->\n", encoding="ascii"
    )
    (areas / "implemented-capabilities-with-limited-or-specialized-exposure.md").write_text(
        "## Implemented capabilities with limited or specialized exposure\n"
        "<!-- capability:dinkster.hidden -->\n",
        encoding="ascii",
    )

    lint_supported_claims(source, supported, tmp_path)
    lint_supported_claims(source, areas, tmp_path)


def test_template_feature_paths_validate_low_step_and_classification() -> None:
    source = EvidenceSource(
        model_families=(),
        lora_node_types=frozenset({"LoraLoader"}),
        adapter_node_types=frozenset({"ControlNetLoader"}),
        low_step_lora_paths=(
            LowStepLoraPath(
                path="low.json",
                lora_node_id=1,
                lora_node_type="LoraLoader",
                sampler_node_id=2,
                sampler_node_type="KSampler",
                step_widget_index=2,
                steps=4,
            ),
        ),
    )
    low = Workflow(
        path="low.json",
        nodes=(
            NodeOccurrence("LoraLoader", "comfy-core", 1),
            NodeOccurrence("KSampler", "comfy-core", 2, (1, "fixed", 4)),
            NodeOccurrence("ControlNetLoader", "comfy-core", 3),
        ),
        subgraph_ids=frozenset(),
        subgraph_count=0,
    )

    result = build_template_features((low,), source)
    assert result["lowStepLora"][0]["samplerNode"]["steps"] == 4  # type: ignore[index]
    assert result["lora"] == [{"nodeTypes": ["LoraLoader"], "path": "low.json"}]
    assert result["adapter"] == [{"nodeTypes": ["ControlNetLoader"], "path": "low.json"}]

    drifted = replace(
        source, low_step_lora_paths=(replace(source.low_step_lora_paths[0], steps=3),)
    )
    with pytest.raises(CoverageError, match="step evidence drifted"):
        build_template_features((low,), drifted)


def test_ledger_exposes_t1_aliases_and_absent_template_capabilities(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "reference.json").write_text("{}", encoding="ascii")
    (tests / "test_reference.py").write_text("def test_reference():\n    pass\n", encoding="ascii")
    record = replace(
        mapping("comfy-core", "Exact"),
        evidence=("tests/test_reference.py::test_reference",),
    )
    records = catalog(record)
    workflows = (
        workflow(
            "workflow.json",
            ("Exact", "comfy-core"),
            ("Missing", "comfy-core"),
        ),
    )
    report = build_report(
        workflows=workflows,
        ignored_files=(),
        catalog=records,
        snapshot=snapshot(),
        receipts_root=tmp_path / "none",
    )
    source = EvidenceSource((), frozenset({"Lora"}), frozenset({"Adapter"}), ())

    ledger = build_evidence_ledger(
        report=report,
        workflows=workflows,
        catalog=records,
        source=source,
        comfyui=ComfyUISource(frozenset({"Exact", "Missing"})),
        repo_root=tmp_path,
    )
    capabilities = {
        item["capabilityId"]: item for item in cast("list[dict[str, Any]]", ledger["capabilities"])
    }
    assert capabilities[record.registry_id]["highestProvenTier"] == "T1"
    assert capabilities[record.registry_id]["supported"] is False
    missing = capabilities["template-node:comfy-core/Missing"]
    assert missing["state"] == "absent"
    assert missing["highestProvenTier"] is None
    assert missing["source"]["registeredAtSource"] is True
    assert render_evidence_markdown(ledger) == render_evidence_markdown(ledger)

    with pytest.raises(
        CoverageError,
        match="maintained core aliases name nodes missing from pinned ComfyUI: Gone",
    ):
        build_evidence_ledger(
            report=report,
            workflows=workflows,
            catalog=catalog(replace(record, source_name="Gone")),
            source=source,
            comfyui=ComfyUISource(frozenset({"Exact", "Missing"})),
            repo_root=tmp_path,
        )

    with pytest.raises(
        CoverageError,
        match="test-backed evidence must name a collected test",
    ):
        build_evidence_ledger(
            report=report,
            workflows=workflows,
            catalog=catalog(replace(record, evidence=("tests/reference.json",))),
            source=source,
            comfyui=ComfyUISource(frozenset({"Exact", "Missing"})),
            repo_root=tmp_path,
        )

    invalid_model_source = replace(
        source,
        model_families=(
            ModelEvidence(
                "dinkster.sd15",
                "proven",
                "T2",
                "supported",
                ("StableDiffusion",),
                (),
            ),
        ),
    )
    with pytest.raises(
        CoverageError,
        match="names missing ComfyUI source identifiers: StableDiffusion",
    ):
        build_evidence_ledger(
            report=report,
            workflows=workflows,
            catalog=records,
            source=invalid_model_source,
            comfyui=ComfyUISource(frozenset({"Exact", "Missing"})),
            repo_root=tmp_path,
        )


def test_generated_outputs_are_deterministic_and_check_detects_staleness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "workflow.json").write_text('{"nodes":[]}', encoding="ascii")
    report = build_report(
        workflows=(workflow("workflow.json"),),
        ignored_files=(),
        catalog=catalog(),
        snapshot=snapshot(),
        receipts_root=tmp_path / "none",
    )
    assert canonical_json(report) == canonical_json(report)
    assert render_markdown(report) == render_markdown(report)

    monkeypatch.setattr(coverage, "verify_template_revision", lambda _root: None)
    monkeypatch.setattr(coverage, "verify_comfyui_revision", lambda _root: None)
    monkeypatch.setattr(
        coverage, "load_comfyui_source", lambda _root: ComfyUISource(frozenset({"Empty"}))
    )
    monkeypatch.setattr(coverage, "load_registry_catalog", lambda _root: catalog())
    monkeypatch.setattr(coverage, "load_download_snapshot", lambda _path: snapshot())
    monkeypatch.setattr(
        coverage,
        "load_evidence_source",
        lambda _path, _root: EvidenceSource((), frozenset({"Lora"}), frozenset({"Adapter"}), ()),
    )
    monkeypatch.setattr(coverage, "lint_supported_claims", lambda *_args: None)
    json_output = tmp_path / "coverage.json"
    markdown_output = tmp_path / "coverage.md"
    baseline_output = tmp_path / "source-parity-baseline.json"
    baseline_output.write_bytes(canonical_json(_baseline(declarations=0, backed=0, cases=0)))
    evidence_json_output = tmp_path / "evidence.json"
    evidence_markdown_output = tmp_path / "evidence.md"
    arguments = [
        "--templates",
        os.fspath(templates),
        "--comfyui",
        os.fspath(tmp_path),
        "--json-output",
        os.fspath(json_output),
        "--markdown-output",
        os.fspath(markdown_output),
        "--source-parity-baseline",
        os.fspath(baseline_output),
        "--receipts",
        os.fspath(tmp_path / "no-receipts"),
        "--evidence-json-output",
        os.fspath(evidence_json_output),
        "--evidence-markdown-output",
        os.fspath(evidence_markdown_output),
    ]

    assert coverage.main(arguments) == 0
    assert coverage.main([*arguments, "--check"]) == 0
    markdown_output.write_text("stale\n", encoding="ascii")
    assert coverage.main([*arguments, "--check"]) == 1


def test_checked_in_report_describes_the_pinned_corpus_and_loaded_registries() -> None:
    report = json.loads((REPO / "docs" / "comfy-translation-coverage.json").read_text())
    ledger = json.loads((REPO / "docs" / "comfy-capability-evidence.json").read_text())
    historical = (REPO / "docs" / "research" / "comfy-translation-coverage-aa3661d9.md").read_text()
    assert report["format"] == coverage.FORMAT
    assert report["inputs"]["templateRevision"] == coverage.TEMPLATE_REVISION
    assert report["inputs"]["comfyuiRevision"] == coverage.COMFYUI_REVISION
    assert "580 workflows" in historical
    assert "aa3661d9fc1a493f8de6b029f5b8af27da3c5d08" in historical
    assert {key: value for key, value in report["corpus"].items() if key != "ignoredJsonFiles"} == {
        "jsonFiles": 618,
        "nodes": 12_861,
        "subgraphs": 484,
        "workflows": 602,
    }
    assert len(report["corpus"]["ignoredJsonFiles"]) == 16
    assert sum(report["coverage"]["statusCounts"].values()) == 12_861
    assert report["coverage"]["workflowEvidenceStates"] == {
        "proven": 8,
        "unverified": 0,
        "absent": 594,
        "refused": 0,
    }

    loaded = load_registry_catalog(REPO / "packages")
    assert len(loaded.mappings) == len(report["registry"]["records"])
    assert loaded.sidecars
    refused = {
        mapping.source_name
        for mapping in loaded.mappings
        if mapping.source_pack == "comfyui_controlnet_aux" and mapping.refusal
    }
    assert refused == {
        "DiffusionEdge_Preprocessor",
        "PiDiNetPreprocessor",
        "Scribble_PiDiNet_Preprocessor",
    }
    report_records = {
        record["source"]["name"]: record
        for record in report["registry"]["records"]
        if record["source"]["pack"] == "comfyui_controlnet_aux"
    }
    assert all(report_records[node_class]["refusal"] for node_class in refused)
    assert all(not report_records[node_class]["target"]["available"] for node_class in refused)
    source_parity = report["sourceParity"]
    assert source_parity["receiptBackedMappings"] == 99
    assert source_parity["translationDeclarations"] == 312
    assert source_parity["unreceiptedMappings"] == 213
    assert source_parity["refusedMappings"] == 3
    backed = set(report["receipts"]["recordsWithPassingReceipts"])
    assert {
        "comfy_alias:comfy-core/ControlNetApply",
        "comfy_alias:comfy-core/ControlNetApplyAdvanced",
        "comfy_alias:comfy-core/ControlNetLoader",
        "comfy_alias:comfy-core/SetUnionControlNetType",
        "comfy_group:comfy-core/remove-background-birefnet",
        "comfy_group:comfy-core/rtdetr-detect-fp16",
        "comfy_group:comfy-core/sam3-text-detection",
        "comfy_group:comfy-core/sam3-video-track-initial-mask",
    } <= backed
    backed_control_aux = {
        registry_id.rsplit("/", 1)[-1]
        for registry_id in report["receipts"]["recordsWithPassingReceipts"]
        if registry_id.startswith("comfy_alias:comfyui_controlnet_aux/")
    }
    assert backed_control_aux == {
        "AnimeLineArtPreprocessor",
        "AnyLineArtPreprocessor_aux",
        "FakeScribblePreprocessor",
        "HEDPreprocessor",
        "LineArtPreprocessor",
        "M-LSDPreprocessor",
        "Manga2Anime_LineArt_Preprocessor",
        "TEEDPreprocessor",
    }
    source = load_evidence_source(REPO / "tools" / "data" / "comfy_capability_evidence.json", REPO)
    assert {item.family_id for item in source.model_families} == {
        family.id for family in coverage.builtin_families()
    }
    assert ledger["format"] == coverage.EVIDENCE_FORMAT
    expected_tiers = {
        tier: sum(item.tier == tier for item in source.model_families)
        for tier in coverage.EVIDENCE_TIERS
    }
    expected_tiers["T0"] += sum(
        mapping.registry_kind == "group" and not mapping.refusal for mapping in loaded.mappings
    )
    expected_tiers["T1"] += sum(
        mapping.registry_kind == "alias" and not mapping.refusal for mapping in loaded.mappings
    )
    assert ledger["summary"]["byTier"] == expected_tiers
    assert ledger["summary"]["byState"]["absent"] > 0
    assert ledger["summary"]["byState"]["refused"] >= len(refused)
    ledger_records = {record["capabilityId"]: record for record in ledger["capabilities"]}
    for node_class in refused:
        record = ledger_records[f"comfy_alias:comfyui_controlnet_aux/{node_class}"]
        assert record["state"] == "refused"
        assert record["highestProvenTier"] is None
        assert record["target"]["available"] is False
    assert len(ledger["workflows"]) == 602
    assert len(ledger["templateFeaturePaths"]["lowStepLora"]) == 4
    assert ledger["templateFeaturePaths"]["adapter"]
