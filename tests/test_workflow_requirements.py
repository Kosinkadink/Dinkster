"""Requirements come from used nodes and semantic assets, not ambient installs."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    LibraryStore,
    MountDef,
    MountTable,
    PackagedSource,
    ProvenanceRecord,
    ProvenanceStore,
    RemoteSource,
    digest_bytes,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_registry.artifact import build_artifact
from dinkster_registry.install import LockedPack, Lockfile
from dinkster_server import PackInfo, ServerLibrary, create_app
from dinkster_values import TypeRegistry
from dinkster_workers import InProcessWorker, PackSandboxNeeds, VisionProvider, load_manifest

from dinkster.mounts_api import MountService, add_mount_routes
from dinkster.project_requirements import load_project_requirements
from dinkster.workflow_requirements import (
    PackRequirement,
    RequirementError,
    WorkflowRequirements,
    derive_workflow_requirements,
    manifest_unavailability,
    plan_workflow_requirements,
    project_requirements,
)
from dinkster.workflow_requirements_api import add_workflow_requirement_routes


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
        sources_for=lambda digest: (
            [RemoteSource(source), RemoteSource(source), PackagedSource("starter", "first.bin")]
            if digest == digest_bytes(b"first")
            else []
        ),
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
                **({"packagedFrom": ["starter"]} if asset.digest == digest_bytes(b"first") else {}),
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


@pytest.mark.parametrize("engine_variant", [None, "linux-cpu", "win-nvidia"])
def test_public_plan_is_read_only_project_filtered_and_verified(
    tmp_path: Path, engine_variant: str | None
) -> None:
    store = LibraryStore(tmp_path / "library.sqlite")
    vault = AssetVault(tmp_path / "vault")
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    first_digest = digest_bytes(b"first")
    second_digest = digest_bytes(b"second")
    url = "https://models.example.test/first.bin"
    provenance.add(ProvenanceRecord(first_digest, (url,)))
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    first_path = mounted / "first.bin"
    first_path.write_bytes(b"first")
    (mounted / "second.bin").write_bytes(b"second")
    table = MountTable(tmp_path / "snapshot.json")
    table.add(MountDef("models", mounted))
    table.scan("models")
    mounts = MountService(table)
    mounts.projects.update("excluded", [], available=("models",), output=None)

    class ForbiddenFetchingResolver:
        def resolve(self, digest: str) -> Path | None:
            pytest.fail("read-only plans must not call a potentially fetching resolver")

    library = ServerLibrary(
        vault, store, resolver=ForbiddenFetchingResolver(), provenance=provenance
    )
    manifest = load_manifest(Path("packages/dinkster-nodes-image/dinkster-pack.toml"))
    declared = {"tools-a": manifest, "tools-b": manifest}
    snapshot = (tmp_path / "snapshot.json").read_bytes()

    async def scenario() -> None:
        registry = TypeRegistry()
        app = create_app(
            lambda listener: Engine(
                schemas={},
                registry=registry,
                worker=InProcessWorker({}, registry),
                cache=MemoryLRUCache(),
                on_event=listener,
            ),
            {},
            library=library,
            packs={
                "tools-a": PackInfo("A", version="1.2.3"),
                "tools-b": PackInfo("B", version="4.5.6"),
            },
        )
        add_mount_routes(app, mounts)
        add_workflow_requirement_routes(
            app,
            manifests=lambda: declared,
            engine_variant=engine_variant,
        )
        async with TestClient(TestServer(app)) as client:
            body = {"workflows": [workflow()], "variant": "linux-cpu", "project": "excluded"}
            response = await client.post(
                "/api/workflows/plan", json={"workflows": [workflow()], "project": "included"}
            )
            if engine_variant is None:
                assert response.status == 400
                assert "engine variant is undeclared" in (await response.json())["error"]
            else:
                assert response.status == 200
                assert await response.json() == {"packs": [], "assets": [], "errors": []}
            response = await client.post("/api/workflows/plan", json=body)
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            excluded = await response.json()
            assert excluded["packs"] == []
            assert excluded["errors"] == []
            assert {asset["digest"] for asset in excluded["assets"]} == {
                first_digest,
                second_digest,
            }
            assert next(asset for asset in excluded["assets"] if asset["digest"] == first_digest)[
                "sources"
            ] == [url]
            response = await client.post(
                "/api/workflows/plan", json={**body, "project": "included"}
            )
            assert await response.json() == {"packs": [], "assets": [], "errors": []}
            first_path.write_bytes(b"wrong")
            response = await client.post(
                "/api/workflows/plan", json={**body, "project": "included"}
            )
            corrupt = await response.json()
            assert [asset["digest"] for asset in corrupt["assets"]] == [first_digest]
            assert not vault.digests()
            outside = tmp_path / "outside.bin"
            outside.write_bytes(b"first")
            first_path.unlink()
            first_path.symlink_to(outside)
            response = await client.post(
                "/api/workflows/plan", json={**body, "project": "included"}
            )
            assert [asset["digest"] for asset in (await response.json())["assets"]] == [
                first_digest
            ]
            with vault.writer(first_digest) as writer:
                writer.write(b"first")
                vault_path = writer.commit()
            response = await client.post(
                "/api/workflows/plan", json={**body, "project": "included"}
            )
            assert await response.json() == {"packs": [], "assets": [], "errors": []}
            vault_path.write_bytes(b"wrong")
            response = await client.post(
                "/api/workflows/plan", json={**body, "project": "included"}
            )
            assert [asset["digest"] for asset in (await response.json())["assets"]] == [
                first_digest
            ]
            response = await client.post(
                "/api/workflows/plan", json={**body, "acquireAssets": [first_digest]}
            )
            assert response.status == 400
            response = await client.post(
                "/api/workflows/plan", json=body, headers={"X-Dinkster-Actor-Kind": "agent"}
            )
            assert response.status == 403
            assert (await response.json())["error"] == "local-operator-required"
            declared["tools-a"] = replace(manifest, sandbox=PackSandboxNeeds(gpu=True))
            response = await client.post("/api/workflows/plan", json={"workflows": [workflow()]})
            if engine_variant is not None:
                assert (await response.json())["errors"] == (
                    ["pack tools-a: pack.sandbox.gpu requires a GPU"]
                    if engine_variant == "linux-cpu"
                    else []
                )
            response = await client.post(
                "/api/workflows/plan", json={**body, "variant": "win-nvidia"}
            )
            assert (await response.json())["errors"] == []

    try:
        asyncio.run(scenario())
        assert (tmp_path / "snapshot.json").read_bytes() == snapshot
        assert provenance.sources(first_digest) == (url,)
    finally:
        store.close()


def test_project_registered_records_are_scoped_verified_and_read_only(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    scratch = tmp_path_factory.mktemp("workflow-plan-scratch")
    library = tmp_path / "library"
    vault = AssetVault(library / "vault")
    store = LibraryStore(library / "library.sqlite")
    data = json.dumps(workflow()).encode()
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        blob = writer.commit()
    try:
        record = store.create(
            "local", "Saved workflow", digest, "application/x-dinkster-workflow+json"
        )
        other = store.create("local", "Not a workflow", digest, "application/json")
    finally:
        store.close()
    project = {
        "format": "dinkster.project/1",
        "workflows": [{"scope": "local", "id": record.id}],
        "additions": {"packs": [{"pack": "user-added", "version": "7.0.0"}]},
    }
    path = tmp_path / "project.json"
    path.write_text(json.dumps(project))
    database = (library / "library.sqlite").read_bytes()
    before = path.read_bytes()
    requirements, inputs = load_project_requirements(tmp_path)
    assert requirements.packs == (
        PackRequirement("tools-a", "1.2.3", ("a.load",)),
        PackRequirement("tools-b", "4.5.6", ("b.save",)),
        PackRequirement("user-added", "7.0.0"),
    )
    assert {asset.digest for asset in requirements.assets} == {
        digest_bytes(b"first"),
        digest_bytes(b"second"),
    }
    assert inputs == (
        {"scope": "local", "id": record.id, "revision": record.revision, "digest": digest},
    )
    assert (library / "library.sqlite").read_bytes() == database
    assert path.read_bytes() == before
    assert blob.read_bytes() == data

    source = tmp_path / "source"
    source.mkdir()
    staging = tmp_path / "staged-release"
    staging.mkdir()
    artifacts = staging / "artifacts"
    artifacts.mkdir()
    catalog = staging / "official-packs.lock"

    def stage_artifacts(*, gpu: bool = False) -> list[LockedPack]:
        packs: list[LockedPack] = []
        for pack, version in (("tools-a", "1.2.3"), ("tools-b", "4.5.6"), ("user-added", "7.0.0")):
            manifest = tmp_path / "fixture-packs" / pack / "dinkster-pack.toml"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            name = "user_added" if pack == "user-added" else pack
            manifest.write_text(
                f'[pack]\nname = "{name}"\n[pack.entry]\nnodes = "never_import:NODES"\n'
                + ("\n[pack.sandbox]\ngpu = true\n" if gpu and pack == "tools-a" else "")
                + (
                    '\n[[pack.assets]]\nid = "first"\nname = "First asset"\n'
                    f'digest = "{digest_bytes(b"first")}"\n'
                    'urls = ["https://models.example.test/first.bin"]\n'
                    if pack == "tools-a"
                    else ""
                )
            )
            archive = artifacts / "building.zip"
            digest = build_artifact(manifest.parent, archive)
            archive.replace(artifacts / f"{digest.split(':')[1]}.zip")
            packs.append(LockedPack(pack, version, digest, "author", ()))
        catalog.write_text(Lockfile.of(packs).record_json())
        return packs

    target_packs = stage_artifacts()
    inventory = source / "dinkster.lock"
    inventory.write_text(
        Lockfile.of(
            [
                replace(pack, version="0.1.0") if pack.pack == "tools-b" else pack
                for pack in target_packs
            ]
        ).record_json()
    )

    def run_plan(variant: str, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "dinkster.workflow_requirements",
                "project-plan",
                "--project-root",
                str(tmp_path),
                "--source-generation",
                str(source),
                "--catalog-lock",
                str(catalog),
                "--artifacts",
                str(artifacts),
                "--variant",
                variant,
                "--json",
                *extra,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env={**os.environ, "TMPDIR": str(scratch)},
        )

    catalog_bytes = catalog.read_bytes()
    cached_bytes = {path.name: path.read_bytes() for path in artifacts.iterdir()}
    cpu = run_plan("linux-cpu")
    nvidia = run_plan("win-nvidia")
    assert cpu.returncode == nvidia.returncode == 0, (cpu.stderr, nvidia.stderr)
    assert cpu.stdout == nvidia.stdout
    result = json.loads(cpu.stdout)
    assert result["requirements"] == requirements.to_wire()
    assert result["inputs"] == list(inputs)
    assert result["plan"]["packs"] == [{"pack": "tools-b", "version": "4.5.6"}]
    assert {row["digest"] for row in result["plan"]["assets"]} == {
        digest_bytes(b"first"),
        digest_bytes(b"second"),
    }
    assert next(row for row in result["plan"]["assets"] if row["digest"] == digest_bytes(b"first"))[
        "sources"
    ] == ["https://models.example.test/first.bin"]
    assert catalog.read_bytes() == catalog_bytes
    assert {path.name: path.read_bytes() for path in artifacts.iterdir()} == cached_bytes

    for content in (b"first", b"second"):
        with vault.writer(digest_bytes(content)) as writer:
            writer.write(content)
            writer.commit()
    inventory.write_text(Lockfile.of(target_packs).record_json())
    present = run_plan("win-nvidia")
    assert present.returncode == 0, present.stderr
    assert json.loads(present.stdout)["plan"] == {"packs": [], "assets": [], "errors": []}
    archive = artifacts / f"{target_packs[0].artifact_digest.split(':')[1]}.zip"
    original = archive.read_bytes()
    archive.unlink()
    missing_archive = run_plan("win-nvidia")
    assert missing_archive.returncode == 1 and archive.name in missing_archive.stderr
    archive.write_bytes(original + b"changed")
    corrupted = run_plan("win-nvidia")
    assert corrupted.returncode == 1 and "digest mismatch" in corrupted.stderr
    assert corrupted.stdout == ""
    archive.write_bytes(original)
    manifest = tmp_path / "fixture-packs/tools-a/dinkster-pack.toml"
    manifest.write_text(manifest.read_text().replace('name = "tools-a"', 'name = "impostor"'))
    wrong_archive = artifacts / "building.zip"
    wrong_digest = build_artifact(manifest.parent, wrong_archive)
    wrong_archive.replace(artifacts / f"{wrong_digest.split(':')[1]}.zip")
    catalog.write_text(
        Lockfile.of(
            [
                replace(pack, artifact_digest=wrong_digest) if pack.pack == "tools-a" else pack
                for pack in target_packs
            ]
        ).record_json()
    )
    wrong_identity = run_plan("win-nvidia")
    assert (
        wrong_identity.returncode == 1
        and "identity does not match pack tools-a" in wrong_identity.stderr
    )
    assert wrong_identity.stdout == ""
    target_packs = stage_artifacts(gpu=True)
    refused = run_plan("linux-cpu")
    assert refused.returncode == 1
    assert json.loads(refused.stdout)["plan"]["errors"] == [
        "pack tools-a: pack.sandbox.gpu requires a GPU"
    ]
    assert "blocking errors" in refused.stderr
    assert run_plan("win-nvidia").returncode == 0
    absent_target = tmp_path / "not-created"
    absent = run_plan("win-nvidia", "--generation", str(absent_target))
    assert absent.returncode == 0 and absent.stdout == run_plan("win-nvidia").stdout
    assert not absent_target.exists()
    catalog.write_text(
        Lockfile.of(
            [
                replace(pack, version="9.9.9") if pack.pack == "tools-b" else pack
                for pack in target_packs
            ]
        ).record_json()
    )
    wrong_version = run_plan("win-nvidia")
    assert wrong_version.returncode == 1
    assert json.loads(wrong_version.stdout)["plan"]["errors"] == [
        "pack tools-b: saved version is unavailable in the selected target release"
    ]
    inventory.unlink()
    assert run_plan("win-nvidia").returncode == 1
    assert list(scratch.iterdir()) == []
    assert (library / "library.sqlite").read_bytes() == database
    assert path.read_bytes() == before
    assert blob.read_bytes() == data

    project["workflows"] = [{"scope": "other", "id": record.id}]
    path.write_text(json.dumps(project))
    with pytest.raises(RequirementError, match="workflow record other/.* is unavailable"):
        load_project_requirements(tmp_path)
    project["workflows"] = [{"scope": "local", "id": other.id}]
    path.write_text(json.dumps(project))
    with pytest.raises(RequirementError, match="is not a workflow"):
        load_project_requirements(tmp_path)
    path.write_bytes(before)
    blob.write_bytes(b"x" * len(data))
    with pytest.raises(AssetError):
        load_project_requirements(tmp_path)


def test_empty_project_requirements_do_not_create_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dinkster.workflow_requirements import main

    path = tmp_path / "project.json"
    path.write_text('{"format":"dinkster.project/1"}')
    requirements, inputs = load_project_requirements(tmp_path)
    assert requirements == WorkflowRequirements()
    assert inputs == ()
    args = ["project-plan", "--project-root", str(tmp_path), "--variant", "linux-cpu", "--json"]
    assert main(args) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {
        "requirements": {"packs": [], "assets": [], "unattributedNodes": []},
        "plan": {"packs": [], "assets": [], "errors": []},
        "inputs": [],
    }
    path.write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "additions": {"packs": [{"pack": "added", "version": "1.0.0"}]},
            }
        )
    )
    assert main(args) == 1
    output = capsys.readouterr()
    assert output.out == "" and "trusted target pack catalog is unavailable" in output.err
    assert sorted(child.name for child in tmp_path.iterdir()) == ["project.json"]
