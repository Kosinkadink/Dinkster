from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from dinkster.compose import Composition

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
    from dinkster_workers import diagnose, load_manifest
    from dinkster_workers.catalog import read_catalog

    from dinkster.compose import default_pack_specs, native_execution_spec

    for spec in (*default_pack_specs(), native_execution_spec()):
        report = diagnose(
            spec.manifest, environment={**(spec.env or {}), "CUDA_VISIBLE_DEVICES": ""}
        )
        assert read_catalog(load_manifest(spec.manifest)) is not None, report


@pytest.fixture(scope="session")
def shared_default_composition() -> Iterator[
    tuple[asyncio.Runner, AbstractAsyncContextManager[Composition]]
]:
    from dinkster.compose import compose_serving

    with asyncio.Runner() as runner:
        composition = runner.run(compose_serving())
        try:
            yield runner, nullcontext(composition)
        finally:
            runner.run(composition.close())
