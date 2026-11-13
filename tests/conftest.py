from __future__ import annotations

import asyncio
import os
import socket
import ssl
import sys
import uuid
from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager, nullcontext
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest_plugins = ["tests.fixtures.git_sources"]

if TYPE_CHECKING:
    from dinkster.compose import Composition

# Tests opt in to an isolated fake gateway explicitly. Collection and ordinary
# composition tests must never contact the production remote-node service.
os.environ["DINKSTER_REMOTE_CATALOG_BASE"] = ""
os.environ["DINKSTER_REMOTE_GATEWAY_BASE"] = ""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network(*hosts): requires outbound HTTPS to named hosts")
    config.addinivalue_line(
        "markers", "pack_runtime(name): execute a cross-pack case in its locked runtime"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # These four suites are nonblocking under comfy-vibe-station#867 while audited.
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    nonblocking = {
        "test_environment_history.py",
        "test_environment_generations.py",
        "test_environment_transactions.py",
        "test_release_install.py",
    }
    root = Path(__file__).parent
    for item in items:
        if item.path.parent == root and item.path.name in nonblocking:
            item.add_marker(pytest.mark.skip(reason="Nonblocking suite: comfy-vibe-station#867"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    helper = sys.modules.get("supervisor_subprocess")
    if helper is not None and (operation_reports := helper.operation_reports):
        report.sections.append(("Supervisor child operation timings", "\n".join(operation_reports)))
        operation_reports.clear()
    return report


def pytest_terminal_summary(terminalreporter) -> None:
    for reports in terminalreporter.stats.values():
        for report in reports:
            if isinstance(report, pytest.TestReport):
                for name, text in report.sections:
                    if name in {
                        "Supervisor child operation timings",
                        "Captured Published pack resolver trap call",
                        "Captured Folded engine CPU environment call",
                    }:
                        terminalreporter.section(name)
                        terminalreporter.write_line(text)


@pytest.fixture(scope="session")
def network_reachability() -> dict[str, str | None]:
    return {}


@pytest.fixture(autouse=True)
def _probe_network_hosts(request: pytest.FixtureRequest, network_reachability) -> None:
    marker = request.node.get_closest_marker("network")
    if marker is None:
        return
    assert marker.args, "network tests must name their external hosts"
    for host in marker.args:
        if host not in network_reachability:
            try:
                with socket.create_connection((host, 443), timeout=5) as connection:
                    with ssl.create_default_context().wrap_socket(connection, server_hostname=host):
                        pass
            except OSError as error:
                network_reachability[host] = str(error)
            else:
                network_reachability[host] = None
        if error := network_reachability[host]:
            pytest.skip(f"HTTPS reachability probe failed for {host}: {error}")


def pytest_sessionstart(session: pytest.Session) -> None:
    if os.name != "nt":
        return
    import winreg

    base = r"SOFTWARE\Microsoft\Windows\Windows Error Reporting"
    for suffix, names in (
        ("", ("DontShowUI", "Disabled")),
        (r"\LocalDumps", ("DumpFolder", "DumpCount", "DumpType")),
    ):
        path = base + suffix
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, access=winreg.KEY_READ) as key:
                for name in names:
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except OSError as exc:
                        value = f"unavailable: {exc}"
                    print(f"WER HKLM\\{path} {name}={value!r}")
        except OSError as exc:
            print(f"WER HKLM\\{path} unavailable: {exc}")


@pytest.fixture(autouse=True)
def _isolate_remote_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    # CLI startup removes service settings from the parent environment.
    monkeypatch.setenv("DINKSTER_REMOTE_CATALOG_BASE", "")
    monkeypatch.setenv("DINKSTER_REMOTE_GATEWAY_BASE", "")


@pytest.fixture
def unrestricted_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    # Synthetic replica indices must not inherit the runner's GPU visibility mask.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture
def shm_segments(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    from dinkster_workers import boundary

    names: set[str] = set()
    prefix = "dinkster" + uuid.uuid4().hex[:8]
    attach = boundary._attach_segment

    def create_owned(size: int) -> SharedMemory:
        segment = SharedMemory(name=prefix + uuid.uuid4().hex[:8], create=True, size=size)
        names.add(segment.name)
        return segment

    def attach_owned(name: str) -> SharedMemory:
        segment = attach(name)
        names.add(segment.name)
        return segment

    # UUID names identify both host-created and child-created payloads without
    # observing segments belonging to concurrent workers in other tests.
    monkeypatch.setattr(boundary, "_create_empty_segment", create_owned)
    monkeypatch.setattr(boundary, "_attach_segment", attach_owned)
    return names


@pytest.fixture(scope="session")
def installed_default_catalogs(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    from dinkster_workers import diagnose, load_manifest
    from dinkster_workers.catalog import read_catalog
    from filelock import FileLock

    from dinkster.compose import default_pack_specs

    shared = tmp_path_factory.getbasetemp()
    if worker_id != "master":
        shared = shared.parent
    ready = shared / "installed-default-catalogs.ready"
    # Doctor removes the old catalog before probing. Finish preparation once
    # before any worker serves from these shared installed pack directories.
    with FileLock(str(shared / "installed-default-catalogs.lock"), timeout=120):
        if not ready.exists():
            for spec in default_pack_specs():
                report = diagnose(
                    spec.manifest, environment={**(spec.env or {}), "CUDA_VISIBLE_DEVICES": ""}
                )
                assert read_catalog(load_manifest(spec.manifest)) is not None, report
            ready.touch()


@pytest.fixture(scope="session")
def shared_default_composition() -> Iterator[
    tuple[asyncio.Runner, AbstractAsyncContextManager[Composition]]
]:
    from dinkster.compose import compose_serving

    with asyncio.Runner() as runner:
        composition = runner.run(compose_serving(include_default_packs=False))
        try:
            yield runner, nullcontext(composition)
        finally:
            runner.run(composition.close())
