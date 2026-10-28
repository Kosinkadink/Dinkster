"""Requirements come from used nodes and semantic assets, not ambient installs."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_assets import AssetRef, digest_bytes
from dinkster_workers import PackSandboxNeeds, VisionProvider, load_manifest

from dinkster.workflow_requirements import (
    PackRequirement,
    RequirementError,
    WorkflowRequirements,
    derive_workflow_requirements,
    manifest_unavailability,
    plan_workflow_requirements,
    project_requirements,
)


def workflow() -> dict[str, object]:
    first = AssetRef(digest_bytes(b"first"), "first.bin", 5).to_wire()
    second = AssetRef(digest_bytes(b"second"), "second.bin", 6).to_wire()
    return {
        "format": "dinkster-workflow",
        "formatVersion": 1,
        "root": "root",
        "graphs": {
            "root": {
                "nodes": {
                    "one": {"type": "a.load", "values": {"asset": first}},
                    "sub": {"type": "#nested", "values": {}},
                },
                "valueSources": {"shared": {"value": [first, {"nested": second}]}},
                "ext": {"unrelated": AssetRef(digest_bytes(b"ignored"), "ignore", 7).to_wire()},
            },
            "nested": {"nodes": {"two": {"type": "b.save", "values": {}}}},
        },
        "environment": {
            "nodes": {
                "a.load": {"pack": "tools-a", "signature": "a"},
                "b.save": {"pack": "tools-b", "signature": "b"},
                "unused": {"pack": "unused", "signature": "unused"},
            },
            "packs": {
                "tools-a": {"version": "1.2.3"},
                "tools-b": {"version": "4.5.6"},
                "unused": {"version": "9.9.9"},
            },
        },
        "view": {"digest": digest_bytes(b"view")},
    }


def test_derive_used_packs_versions_and_deduplicated_semantic_assets() -> None:
    document = workflow()
    before = json.dumps(document)
    requirements = derive_workflow_requirements(document)
    assert requirements.packs == (
        PackRequirement("tools-a", "1.2.3", ("a.load",)),
        PackRequirement("tools-b", "4.5.6", ("b.save",)),
    )
    assert {asset.digest for asset in requirements.assets} == {
        digest_bytes(b"first"),
        digest_bytes(b"second"),
    }
    assert requirements.unattributed_nodes == ()
    assert json.dumps(document) == before


def test_unstamped_workflow_does_not_guess_pack_namespace_or_versions() -> None:
    document = workflow()
    document.pop("environment")
    requirements = derive_workflow_requirements(document)
    assert requirements.packs == ()
    assert requirements.unattributed_nodes == ("a.load", "b.save")
    document["environment"] = {"nodes": {"a.load": {"pack": "not-the-namespace"}}}
    requirements = derive_workflow_requirements(document)
    assert requirements.packs == (PackRequirement("not-the-namespace", None, ("a.load",)),)
    assert requirements.unattributed_nodes == ("b.save",)


def test_project_union_preserves_workflows_and_explicit_additions() -> None:
    document = workflow()
    second = copy.deepcopy(document)
    before = json.dumps([document, second])
    extra = AssetRef(digest_bytes(b"user"), "user.bin", 4)
    requirements = project_requirements(
        [document, second],
        additions=WorkflowRequirements((PackRequirement("user-added", "7.0.0"),), (extra,)),
    )
    assert requirements.packs == (
        PackRequirement("tools-a", "1.2.3", ("a.load",)),
        PackRequirement("tools-b", "4.5.6", ("b.save",)),
        PackRequirement("user-added", "7.0.0"),
    )
    assert {asset.digest for asset in requirements.assets} == {
        digest_bytes(b"first"),
        digest_bytes(b"second"),
        extra.digest,
    }
    assert json.dumps([document, second]) == before
    with pytest.raises(RequirementError, match="conflicting saved versions for tools-b"):
        project_requirements(
            [document], additions=WorkflowRequirements((PackRequirement("tools-b", "0.1.0"),))
        )


def test_missing_plan_and_verified_present_control() -> None:
    requirements = derive_workflow_requirements(workflow())
    source = "https://models.example.test/first.bin"
    plan = plan_workflow_requirements(
        requirements,
        installed_packs={"unused": "9.9.9", "tools-b": "0.1.0"},
        asset_available=lambda asset: False,
        sources_for=lambda digest: [source, source] if digest == digest_bytes(b"first") else [],
        pack_unavailable=lambda pack: None,
    )
    assert plan.to_wire() == {
        "packs": [
            {"pack": "tools-a", "version": "1.2.3"},
            {"pack": "tools-b", "version": "4.5.6"},
        ],
        "assets": [
            {
                **asset.to_wire(),
                "sources": [source] if asset.digest == digest_bytes(b"first") else [],
            }
            for asset in requirements.assets
        ],
        "errors": [],
    }
    assert not plan.empty
    present = plan_workflow_requirements(
        requirements,
        installed_packs={"tools-a": "1.2.3", "tools-b": "4.5.6", "unused": "9.9.9"},
        asset_available=lambda asset: True,
        sources_for=lambda digest: pytest.fail("present assets must not need resolver leads"),
        pack_unavailable=lambda pack: None,
    )
    assert present.empty
    assert present.to_wire() == {"packs": [], "assets": [], "errors": []}


def test_missing_version_is_not_silently_satisfied_by_unknown_installed_version() -> None:
    requirements = WorkflowRequirements((PackRequirement("unknown-version", None),))
    plan = plan_workflow_requirements(
        requirements,
        installed_packs={"unknown-version": None},
        asset_available=lambda asset: True,
        sources_for=lambda digest: [],
        pack_unavailable=lambda pack: None,
    )
    assert not plan.empty
    assert plan.to_wire()["errors"] == ["pack unknown-version: no saved version"]


def test_variant_plan_uses_only_manifest_declarations_and_preserves_documents() -> None:
    document = workflow()
    before = json.dumps(document)
    requirements = project_requirements([document])
    manifest = load_manifest(Path("packages/dinkster-nodes-image/dinkster-pack.toml"))

    def plan(variant: str, gpu: bool = False, devices: tuple[str, ...] = ("cpu", "cuda")):
        declared = replace(
            manifest,
            sandbox=PackSandboxNeeds(gpu=gpu),
            vision_providers=(VisionProvider("provider", "a.load", devices, (), "single", ()),),
        )
        return plan_workflow_requirements(
            requirements,
            installed_packs={},
            asset_available=lambda asset: False,
            sources_for=lambda digest: [],
            pack_unavailable=lambda pack: manifest_unavailability(pack, declared, variant),
        )

    cpu = plan("linux-cpu").to_wire()
    assert cpu == plan("win-nvidia").to_wire()
    gpu_cpu = plan("linux-cpu", gpu=True).to_wire()
    assert gpu_cpu["packs"] == cpu["packs"]
    assert gpu_cpu["assets"] == cpu["assets"]
    assert gpu_cpu["errors"] == [
        "pack tools-a: pack.sandbox.gpu requires a GPU",
        "pack tools-b: pack.sandbox.gpu requires a GPU",
    ]
    assert plan("win-nvidia", gpu=True).to_wire() == cpu
    assert plan("win-cpu", devices=("cuda",)).to_wire()["errors"] == [
        "pack tools-a: node a.load declares devices cuda; cpu is not supported"
    ]
    assert json.dumps(document) == before
