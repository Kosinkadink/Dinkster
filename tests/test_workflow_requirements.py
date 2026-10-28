"""Requirements come from used nodes and semantic assets, not ambient installs."""

from __future__ import annotations

import copy
import json

import pytest
from dinkster_assets import AssetRef, digest_bytes

from dinkster.workflow_requirements import (
    PackRequirement,
    RequirementError,
    WorkflowRequirements,
    derive_workflow_requirements,
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
        PackRequirement("tools-a", "1.2.3"),
        PackRequirement("tools-b", "4.5.6"),
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
    assert requirements.packs == (PackRequirement("not-the-namespace", None),)
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
        PackRequirement("tools-a", "1.2.3"),
        PackRequirement("tools-b", "4.5.6"),
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
