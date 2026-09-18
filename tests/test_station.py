"""The station: multi-install management (see packages/dinkster-supervisor/README.md).

Every configured install gets its own public port speaking the exact
single-engine supervisor surface - bound the moment the station starts,
so a stopped install answers an honest 503, never a connection refusal -
and one management port lists and drives the fleet. The management
surface is tested with stub engine subprocesses (fast, deterministic);
the final test runs TWO REAL dinkster-serve engines over one shared content
store to prove the Comfy Desktop role end to end.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import textwrap
import time
from pathlib import Path

import aiohttp
import pytest
from dinkster_registry import Lockfile
from dinkster_supervisor import (
    InstallDef,
    InstallsError,
    engine_command,
    start_station,
)

import dinkster.installer as installer_module
from dinkster.installer import Installer, lock_local_pack

# Ports below the kernel's ephemeral range (32768+) cannot be claimed by
# outgoing connections between this probe and the eventual server bind.
_port_candidates = iter(range(21100, 32000))


def free_port() -> int:
    for candidate in _port_candidates:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError("no free test port below the ephemeral range")


async def wait_for(predicate, timeout: float = 60.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.05)


# A stand-in engine that accepts the full station-injected argv
# (--install-root from engine_command, --host/--port from the supervisor
# protocol) and can prove which root it was given.
STUB_ENGINE = textwrap.dedent(
    """
    import argparse
    from aiohttp import web

    parser = argparse.ArgumentParser()
    parser.add_argument("--install-root", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    async def health(request):
        return web.json_response({"ok": True})

    async def root(request):
        return web.json_response({"root": args.install_root})

    app = web.Application()
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/root", root)
    web.run_app(app, host=args.host, port=args.port, print=None)
    """
)


def stub_install(tmp_path: Path, name: str, *, autostart: bool = False) -> InstallDef:
    script = tmp_path / "stub_engine.py"
    if not script.exists():
        script.write_text(STUB_ENGINE)
    root = tmp_path / f"root-{name}"
    root.mkdir(exist_ok=True)
    return InstallDef(
        name=name,
        root=root,
        port=free_port(),
        autostart=autostart,
        engine=(sys.executable, str(script)),
    )


def test_engine_command_shapes(tmp_path: Path) -> None:
    """The configured command (or the station default) always gains
    --install-root; --host/--port ride the supervisor protocol later."""
    root = tmp_path / "r"
    configured = InstallDef(
        name="a", root=root, port=8200, engine=("/venv/bin/python", "-m", "dinkster.serve")
    )
    assert engine_command(configured) == [
        "/venv/bin/python",
        "-m",
        "dinkster.serve",
        "--install-root",
        str(root),
    ]
    defaulted = InstallDef(name="b", root=root, port=8201)
    assert engine_command(defaulted) == [
        sys.executable,
        "-m",
        "dinkster.serve",
        "--install-root",
        str(root),
    ]
    assert engine_command(defaulted, ["/other/python", "-m", "engine"]) == [
        "/other/python",
        "-m",
        "engine",
        "--install-root",
        str(root),
    ]


def test_station_ports_bind_before_any_engine(tmp_path: Path) -> None:
    """Every port answers from the first request: the management port
    narrates the fleet, each install port speaks the ordinary supervisor
    surface (honest 503, never a refusal), and the management port
    proxies NOTHING."""

    async def scenario() -> None:
        a = stub_install(tmp_path, "alpha")
        b = stub_install(tmp_path, "beta")
        management = free_port()
        station = await start_station((a, b), port=management)
        try:
            async with aiohttp.ClientSession() as session:
                base = f"http://127.0.0.1:{management}"
                async with session.get(base + "/supervisor/status") as resp:
                    status = await resp.json()
                assert status == {
                    "protocol": 1,
                    "state": "ready",
                    "role": "station",
                    "installs": 2,
                }
                async with session.get(base + "/supervisor/installs") as resp:
                    listing = (await resp.json())["installs"]
                assert set(listing) == {"alpha", "beta"}
                assert listing["alpha"]["root"] == str(a.root)
                assert listing["alpha"]["port"] == a.port
                assert listing["alpha"]["autostart"] is False
                assert listing["alpha"]["status"]["state"] == "stopped"

                # install ports: supervisor surface, deliberately down
                for install in (a, b):
                    origin = f"http://127.0.0.1:{install.port}"
                    async with session.get(origin + "/supervisor/status") as resp:
                        assert (await resp.json())["state"] == "stopped"
                    async with session.get(origin + "/api/nodes") as resp:
                        assert resp.status == 503
                        assert (await resp.json())["error"] == "engine-not-ready"

                # the management port is not an engine port
                async with session.get(base + "/api/nodes") as resp:
                    assert resp.status == 404
                    assert (await resp.json())["error"] == "not-an-engine-port"
                async with session.get(base + "/supervisor/nonsense") as resp:
                    assert resp.status == 404
                    assert (await resp.json())["error"] == "unknown-supervisor-path"

                # unknown install and wrong-state transitions are crisp
                async with session.post(base + "/supervisor/installs/ghost/start") as resp:
                    assert resp.status == 404
                    assert (await resp.json())["error"] == "unknown-install"
                async with session.post(base + "/supervisor/installs/alpha/stop") as resp:
                    assert resp.status == 409
                    assert (await resp.json())["error"] == "not-running"
        finally:
            await station.close()

    asyncio.run(scenario())


def test_station_start_stop_restart_lifecycle(tmp_path: Path) -> None:
    """Runtime control end to end against a real subprocess: start makes
    the install port proxy (with the right --install-root), duplicate
    start is a 409, stop returns the port to an honest 503, and restart
    from stopped starts cleanly again."""

    async def scenario() -> None:
        install = stub_install(tmp_path, "solo")
        management = free_port()
        station = await start_station((install,), port=management)
        try:
            async with aiohttp.ClientSession() as session:
                base = f"http://127.0.0.1:{management}"
                origin = f"http://127.0.0.1:{install.port}"
                route = base + "/supervisor/installs/solo"

                async with session.post(route + "/start") as resp:
                    assert resp.status == 200

                async def is_ready() -> bool:
                    async with session.get(origin + "/supervisor/status") as resp:
                        return (await resp.json())["state"] == "ready"

                async with asyncio.timeout(60):
                    while not await is_ready():
                        await asyncio.sleep(0.05)

                # the proxy is live and the engine got the configured root
                async with session.get(origin + "/api/root") as resp:
                    assert (await resp.json())["root"] == str(install.root)

                async with session.post(route + "/start") as resp:
                    assert resp.status == 409
                    assert (await resp.json())["error"] == "already-running"

                async with session.post(route + "/stop") as resp:
                    assert resp.status == 200
                    assert (await resp.json())["state"] == "stopped"
                async with session.get(origin + "/api/root") as resp:
                    assert resp.status == 503

                async with session.post(route + "/restart") as resp:
                    assert resp.status == 200
                async with asyncio.timeout(60):
                    while not await is_ready():
                        await asyncio.sleep(0.05)
        finally:
            await station.close()

    asyncio.run(scenario())


def test_station_autostart_and_failure_isolation(tmp_path: Path) -> None:
    """Autostart starts exactly the installs that ask for it, and one
    install's crash is a queryable fact on the management surface while
    its neighbor keeps serving - never a dead station."""

    async def scenario() -> None:
        good = stub_install(tmp_path, "good", autostart=True)
        dies = tmp_path / "dies.py"
        dies.write_text("import sys; sys.exit(3)\n")
        bad_root = tmp_path / "root-bad"
        bad_root.mkdir()
        bad = InstallDef(
            name="bad",
            root=bad_root,
            port=free_port(),
            autostart=True,
            engine=(sys.executable, str(dies)),
        )
        idle = stub_install(tmp_path, "idle")  # no autostart
        management = free_port()
        station = await start_station((good, bad, idle), port=management)
        try:
            link_states = station.managed
            await wait_for(lambda: link_states["good"].link.state == "ready")
            await wait_for(lambda: link_states["bad"].link.state == "failed")
            assert link_states["idle"].link.state == "stopped"

            async with aiohttp.ClientSession() as session:
                base = f"http://127.0.0.1:{management}"
                async with session.get(base + "/supervisor/installs") as resp:
                    listing = (await resp.json())["installs"]
                assert listing["good"]["status"]["state"] == "ready"
                assert listing["bad"]["status"]["state"] == "failed"
                assert listing["bad"]["status"]["engine"]["exitCode"] == 3
                assert listing["idle"]["status"]["state"] == "stopped"

                # the survivor proxies; the corpse narrates
                async with session.get(f"http://127.0.0.1:{good.port}/api/root") as resp:
                    assert resp.status == 200
                async with session.get(f"http://127.0.0.1:{bad.port}/api/root") as resp:
                    assert resp.status == 503
        finally:
            await station.close()

    asyncio.run(scenario())


def test_station_refuses_management_port_collision(tmp_path: Path) -> None:
    async def scenario() -> None:
        install = stub_install(tmp_path, "clash")
        with pytest.raises(InstallsError, match="management port"):
            await start_station((install,), port=install.port)

    asyncio.run(scenario())


# -- two real engines over one shared store ----------------------------

MANIFEST_TEMPLATE = '[pack]\nname = "{name}"\n\n[pack.entry]\nnodes = "{name}_nodes:NODES"\n'


def write_pack(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dinkster-pack.toml").write_text(MANIFEST_TEMPLATE.format(name=name))
    (directory / f"{name}_nodes.py").write_text("NODES = []\n")
    return directory


def test_station_two_real_engines_share_one_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed_default_catalogs: None,
) -> None:
    """The Comfy Desktop role, live: two install roots on one shared
    content store, each generation different, one station managing both.
    Each engine serves exactly its own generation, the shared pack exists
    as one physical copy, and stopping one engine leaves the other
    serving - storage is shared, lifecycles are not."""
    monkeypatch.setattr(installer_module, "detect_runtime", lambda _accelerator: ())
    monkeypatch.setenv("DINKSTER_SERVING_PYTHON", sys.executable)
    shared = tmp_path / "shared"
    common = write_pack(tmp_path / "src-common", "common")
    only_a = write_pack(tmp_path / "src-only-a", "onlya")
    only_b = write_pack(tmp_path / "src-only-b", "onlyb")

    roots: dict[str, Path] = {}
    pack_dirs: list[Path] = []
    for name, packs in (("a", (common, only_a)), ("b", (common, only_b))):
        root = tmp_path / f"install-{name}"
        installer = Installer(root, shared_store=shared)
        target = Lockfile.of([lock_local_pack(pack, installer.artifacts_dir)[0] for pack in packs])
        installer.apply(target, venvs=False)  # host interpreter runs the packs
        roots[name] = root
        pack_dirs.extend(Path(spec.manifest).parent for spec in installer.packs_for_serving())

    # one physical copy: both roots point at the shared store, no local copies
    assert not (roots["a"] / "store").exists()
    assert not (roots["b"] / "store").exists()
    store_dirs = {path for path in pack_dirs if path.is_relative_to(shared / "store")}
    assert len(store_dirs) == 3  # common counted once, plus onlya and onlyb

    # no venvs were staged, so the engine workers import pack modules from
    # the shared store via PYTHONPATH (test wiring, inherited by children)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(str(path) for path in store_dirs))

    engine = (
        sys.executable,
        "-m",
        "dinkster.serve",
        "--library-root",
        "",
    )
    installs = tuple(
        InstallDef(name=name, root=roots[name], port=free_port(), autostart=True, engine=engine)
        for name in ("a", "b")
    )

    management = free_port()

    async def scenario() -> None:
        station = await start_station(installs, port=management)
        try:
            async with aiohttp.ClientSession() as session:

                async def packs_of(port: int) -> set[str] | None:
                    try:
                        async with session.get(f"http://127.0.0.1:{port}/api/nodes") as resp:
                            if resp.status != 200:
                                return None
                            data = await resp.json()
                    except aiohttp.ClientError:
                        return None
                    if "composing" in data:
                        return None  # still announcing; wait for the full surface
                    return set(data["packs"])

                served: list[set[str] | None] = [None, None]
                async with asyncio.timeout(120):
                    while not all(names is not None for names in served):
                        for entry in station.managed.values():
                            assert entry.link.state in ("starting", "ready"), (
                                f"{entry.install.name}: {entry.link.state} ({entry.link.detail})"
                            )
                        served = [await packs_of(install.port) for install in installs]
                        await asyncio.sleep(0.2)

                served_a, served_b = served
                assert served_a is not None and served_b is not None
                # each engine serves exactly its own generation
                assert {"common", "onlya"} <= served_a and "onlyb" not in served_a
                assert {"common", "onlyb"} <= served_b and "onlya" not in served_b

                # lifecycles are independent: stop a via the station API,
                # b keeps serving the exact same surface
                async with session.post(
                    f"http://127.0.0.1:{management}/supervisor/installs/a/stop"
                ) as resp:
                    assert resp.status == 200
                async with session.get(f"http://127.0.0.1:{installs[0].port}/api/nodes") as resp:
                    assert resp.status == 503
                assert await packs_of(installs[1].port) == served_b
        finally:
            await station.close()

    asyncio.run(scenario())
