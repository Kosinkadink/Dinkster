from __future__ import annotations

import os

import pytest

# Tests opt in to an isolated fake gateway explicitly. Collection and ordinary
# composition tests must never contact the production remote-node service.
os.environ["DINKSTER_REMOTE_CATALOG_BASE"] = ""
os.environ["DINKSTER_REMOTE_GATEWAY_BASE"] = ""


@pytest.fixture(autouse=True)
def _isolate_remote_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    # CLI startup removes service settings from the parent environment.
    monkeypatch.setenv("DINKSTER_REMOTE_CATALOG_BASE", "")
    monkeypatch.setenv("DINKSTER_REMOTE_GATEWAY_BASE", "")


@pytest.fixture
def unrestricted_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    # Synthetic replica indices must not inherit the runner's GPU visibility mask.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture(scope="session")
def installed_default_catalogs(tmp_path_factory: pytest.TempPathFactory) -> None:
    from dinkster_workers import load_manifest
    from dinkster_workers.catalog import read_catalog
    from dinkster_workers.doctor import prepare_catalog

    from dinkster.comfy_compose import comfy_compat_specs
    from dinkster.compose import default_pack_specs

    for spec in (*default_pack_specs(), *comfy_compat_specs()):
        report = prepare_catalog(
            spec.manifest, environment={**(spec.env or {}), "CUDA_VISIBLE_DEVICES": ""}
        )
        assert report.ok, report
        assert read_catalog(load_manifest(spec.manifest)) is not None, report
