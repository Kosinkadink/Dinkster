"""Managed preview placement clears temporary bytes but preserves saved originals."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import aiohttp
import pytest
from dinkster_assets import AssetVault, digest_bytes
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_registry import LockedPack, Lockfile, build_artifact
from dinkster_supervisor.environment_runtime import start_engine, stop_engine
from dinkster_workers.doctor import prepare_catalog
from test_environment_runtime import free_port, make_generation
from test_preview_artifacts import _tiny_mp4, _write_preview_pack
from test_serve import _job_finish

from dinkster.generation_packs import stage_generation_packs


@pytest.mark.parametrize("in_process", [False, True])
def test_managed_previews_use_selected_storage_across_restart_and_selection_changes(
    tmp_path: Path,
    in_process: bool,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    data = _tiny_mp4(23)
    manifest = _write_preview_pack(source, data)
    manifest.write_text(
        manifest.read_text().replace("[pack.entry]", "version='1.0.0'\nengine='>=0'\n[pack.entry]")
    )
    for module in source.glob("*.py"):
        module.write_text(module.read_text().replace("SaveTarget('renders'", "SaveTarget('output'"))
    project = tmp_path / "project"
    generation = project / "builds/1"
    generation.mkdir(parents=True)
    archive = tmp_path / "preview.zip"
    digest = build_artifact(source, archive)
    stage_generation_packs(
        generation,
        Lockfile.of(
            [LockedPack("preview-fixture", "1.0.0", digest, "dinkster", ("preview-fixture",))]
        ),
        {"preview-fixture": archive},
    )
    make_generation(project, fake_serve=False)
    installed = generation / "pack-sources/preview-fixture"
    assert prepare_catalog(
        installed / "dinkster-pack.toml", environment={"PYTHONPATH": str(installed)}
    ).ok
    if in_process:
        record = generation / "build.json"
        record.write_text(
            json.dumps(json.loads(record.read_text()) | {"inProcessPacks": ["preview-fixture"]})
        )
    else:
        venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=os.name != "nt").create(
            generation / "packs/preview-fixture"
        )
    port = free_port()
    settings = project / "environment.json"
    settings.write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "mirrors": [],
                "port": port,
            }
        )
    )
    directories = (tmp_path / "first", tmp_path / "second")
    for directory in directories:
        directory.mkdir()
    sibling = directories[0] / "not-a-preview"
    sibling.write_bytes(b"operator sibling is not granted")
    roots: list[Path] = []
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(project),
        "previews",
    ]

    def select(directory: Path | None) -> None:
        result = subprocess.run(
            [*command, *(("--directory", str(directory)) if directory else ()), "--yes", "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if directory is not None:
            roots.append(Path(json.loads(result.stdout)["previewRoot"]))

    async def probe(
        job_id: str | None,
        *,
        mode: str | None = None,
        before_publication: bool = False,
        available: bool = True,
    ) -> None:
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession() as client:
            if before_publication:
                async with client.post(base + "/api/projects", json={"id": "film"}) as response:
                    assert response.status == 200, await response.text()
                async with client.get(base + "/api/library/storage") as response:
                    assert response.status == 200
                    accounting = await response.json()
                    assert accounting["vaultBytes"] == 0, accounting
                    assert accounting["adoptedBytes"] == 0, accounting
                    assert accounting["blocked"] == []
                    assert accounting["collectionEnabled"] is True
                [temporary_vault] = roots[0].glob("engines/*/temporary/vault")
                for root in (project / "dinkster-library/vault", temporary_vault):
                    assert ProducedAssetInventory(root).entries() == ()
                return
            if job_id is not None:
                async with asyncio.timeout(60):
                    while True:
                        async with client.get(base + "/api/mounts") as response:
                            assert response.status == 200
                            mounts = (await response.json())["mounts"]
                        if any(row["id"] == "output" and row["state"] == "ready" for row in mounts):
                            break
                        await asyncio.sleep(0.05)
                async with client.post(
                    base + "/api/jobs",
                    json={
                        "clientId": "e2e",
                        "jobId": job_id,
                        "project": "film",
                        "graph": graph_to_wire(
                            Graph(
                                nodes={
                                    "save": GraphNode(
                                        "preview-fixture.emit",
                                        {"mode": mode or "mounted-" + job_id},
                                    )
                                }
                            )
                        ),
                        "targets": ["save"],
                    },
                ) as response:
                    assert response.status == 202, await response.text()
                result = await _job_finish(client, base, job_id)
                assert result["state"] == "completed", result
                artifacts = result["artifacts"]
                assert isinstance(artifacts, list) and len(artifacts) == 2
                if available:
                    assert all(
                        item["preview"]["digest"] == digest_bytes(data) for item in artifacts
                    )
                else:
                    assert all(item.get("preview") is None for item in artifacts)
            async with client.get(base + f"/api/assets/{digest_bytes(data)}") as response:
                assert response.status == (200 if available else 404), await response.text()
                if available:
                    assert await response.read() == data
            async with client.get(
                base + f"/api/assets/{digest_bytes(sibling.read_bytes())}"
            ) as response:
                assert response.status == 404
            async with client.get(
                base + "/api/assets/catalog?scope=local&project=film"
            ) as response:
                assert response.status == 200
                assert (await response.json())["assets"]
            async with client.get(base + "/api/mounts") as response:
                assert response.status == 200
                assert not any(
                    Path(row["path"]).is_relative_to(directory)
                    for row in (await response.json())["mounts"]
                    for directory in directories
                )
            if job_id == "second":
                current = AssetVault(next(roots[1].glob("engines/*/temporary/vault"))).resolve(
                    digest_bytes(data)
                )
                assert current is not None
                try:
                    current.write_bytes(b"corrupted preview")
                    async with client.get(base + f"/api/assets/{digest_bytes(data)}") as response:
                        assert response.status == 409
                        assert response.headers["Cache-Control"] == "no-store"
                finally:
                    current.write_bytes(data)

    try:
        select(directories[0])
        start_engine(project, 1, "main", port)
        asyncio.run(probe(None, before_publication=True))
        asyncio.run(probe("first"))
        original_vault = AssetVault(project / "dinkster-library/vault")
        assert original_vault.resolve(digest_bytes(data)) is None
        selected = AssetVault(next(roots[0].glob("engines/*/temporary/vault"))).resolve(
            digest_bytes(data)
        )
        assert selected is not None and selected.read_bytes() == data
        stop_engine(project, "main")
        assert not selected.exists()
        start_engine(project, 1, "main", port)
        asyncio.run(probe(None, available=False))
        stop_engine(project, "main")
        select(directories[1])
        start_engine(project, 1, "main", port)
        asyncio.run(probe("cached", mode="mounted-first", available=False))
        second_vault = AssetVault(next(roots[1].glob("engines/*/temporary/vault")))
        assert second_vault.resolve(digest_bytes(data)) is None
        asyncio.run(probe("second"))
        assert second_vault.resolve(digest_bytes(data)) is not None
        assert not selected.exists()
        assert sibling.read_bytes() == b"operator sibling is not granted"
        stop_engine(project, "main")
        assert second_vault.resolve(digest_bytes(data)) is None
        select(None)
        start_engine(project, 1, "main", port)
        asyncio.run(probe(None, available=False))
        assert original_vault.resolve(digest_bytes(data)) is None
        asyncio.run(probe("reset"))
        selection = json.loads((project / "engine-state/main/state.json").read_text())
        reset_preview = AssetVault(
            project / "engine-state/main" / selection["directory"] / "temporary/vault"
        ).resolve(digest_bytes(data))
        assert reset_preview is not None and reset_preview.read_bytes() == data
        assert original_vault.resolve(digest_bytes(data)) is None
        assert not selected.exists()
        originals = list((project / "dinkster-library").rglob("*.exr"))
        assert len(originals) == 6
        assert all(path.read_bytes() == b"duplicate saved bytes" for path in originals)
    finally:
        stop_engine(project, "main")


def test_preview_storage_changes_keep_previous_locations_and_originals(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    settings = project / "environment.json"
    settings.write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "mirrors": [],
                "port": free_port(),
            }
        )
    )
    original = project / "dinkster-library" / "outputs" / "original.txt"
    original.parent.mkdir(parents=True)
    original.write_text("retained original")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    sibling = first / "unrelated.txt"
    sibling.write_text("not a preview")
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(project),
        "previews",
    ]

    def configure(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([*command, *arguments, "--json"], capture_output=True, text=True)

    before = settings.read_bytes()
    plan = configure("--directory", str(first), "--plan")
    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert settings.read_bytes() == before and list(first.iterdir()) == [sibling]
    applied = configure("--directory", str(first), "--yes")
    assert applied.returncode == 0, applied.stdout + applied.stderr
    vault = Path(json.loads(applied.stdout)["previewRoot"])
    identity = json.loads(settings.read_text())["previewStorage"]["identity"]
    assert vault.is_dir()
    vault.rmdir()
    missing_vault = configure("--directory", str(first), "--yes")
    assert missing_vault.returncode != 0 and "vault is unavailable" in missing_vault.stdout
    assert not vault.exists()
    vault.mkdir()
    saved = vault / "kept-preview"
    saved.write_bytes(b"earlier preview bytes")
    unavailable = configure("--directory", str(tmp_path / "missing"), "--yes")
    assert unavailable.returncode != 0 and "unavailable" in unavailable.stdout
    assert json.loads(settings.read_text())["previewStorage"] == {
        "directory": str(first),
        "identity": identity,
    }
    changed = configure("--directory", str(second), "--yes")
    assert changed.returncode == 0, changed.stdout + changed.stderr
    assert json.loads(settings.read_text())["previewStorage"] == {
        "directory": str(second),
        "previousDirectories": [str(first)],
        "identity": identity,
    }
    cleared = configure("--yes")
    assert cleared.returncode == 0, cleared.stdout + cleared.stderr
    assert json.loads(settings.read_text())["previewStorage"] == {
        "previousDirectories": [str(first), str(second)],
        "identity": identity,
    }
    restored = configure("--directory", str(first), "--yes")
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert json.loads(settings.read_text())["previewStorage"] == {
        "directory": str(first),
        "previousDirectories": [str(second)],
        "identity": identity,
    }
    assert saved.read_bytes() == b"earlier preview bytes"
    assert original.read_text() == "retained original"
    assert sibling.read_text() == "not a preview"
    moved = tmp_path / "moved-project"
    project.rename(moved)
    project = moved
    settings = moved / "environment.json"
    command[command.index("--root") + 1] = str(moved)
    assert configure("--directory", str(first), "--yes").returncode == 0
    assert saved.read_bytes() == b"earlier preview bytes"
    before = settings.read_bytes()
    owner = vault.parent / "owner.json"
    owner.write_text("{}")
    refused = configure("--yes")
    assert refused.returncode != 0 and "not owned" in refused.stdout
    assert settings.read_bytes() == before
