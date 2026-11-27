from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_supervisor import home_setup
from dinkster_supervisor.environment_state import write_document
from dinkster_supervisor.home import Home, create_home_app
from test_environment_runtime import free_port

from tests.async_scenarios import run_scenario
from tests.fixtures.environments.build import KEYS, build


def test_retained_setup_job_discovery_does_not_wait_for_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked_plan(
            self: home_setup.SetupJobs, root: Path, channel: str, variant: str | None
        ) -> dict[str, object]:
            entered.set()
            await release.wait()
            return {"refusal": None}

        monkeypatch.setattr(home_setup.SetupJobs, "plan", blocked_plan)
        home = Home(installations=frozenset)
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            planning = asyncio.ensure_future(client.get("/supervisor/setup"))
            try:
                await entered.wait()
                response = await client.get("/supervisor/setup/active")
                assert response.status == 200
                assert await response.json() == {"activeJob": None}
                assert not planning.done()
            finally:
                release.set()
                response = await planning
                assert response.status == 200

    run_scenario(scenario())


@pytest.mark.usefixtures("signed_channels")
@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("operation", ["update", "restore"])
def test_installation_job_binds_review_and_cancels_real_cli_without_opening_stopped_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opened: bool, operation: str
) -> None:
    from dinkster_supervisor.environment_runtime import (
        engine_assignments,
        engine_status,
        stop_engine,
    )

    fixture = build(tmp_path / "fixture", "fixture-cpu", free_port())
    monkeypatch.setattr(home_setup, "RELEASE_MIRROR", fixture["mirror"])
    monkeypatch.setenv("DINKSTER_LIBRARY_ROOT", str(tmp_path / "library"))
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    root = tmp_path / "installation"
    cancel = True
    change_before_launch = False
    launch = asyncio.create_subprocess_exec

    async def signed_process(executable: str, *arguments: str, **kwargs: Any):
        if change_before_launch and operation in arguments and "--yes" in arguments:
            settings_path = root / "environment.json"
            changed = json.loads(settings_path.read_text())
            changed["responseCompression"] = True
            write_document(settings_path, changed)
        if arguments[:3] == ("-I", "-m", "dinkster_supervisor.environment"):
            script = (
                "from dinkster_supervisor import environment, environment_channel_signatures; "
                f"environment_channel_signatures.CHANNEL_KEYS = {KEYS!r}\n"
            )
            if cancel and operation in arguments and "--yes" in arguments:
                module = "environment_restore" if operation == "restore" else "environment"
                script += f"from dinkster_supervisor import {module} as observed\n"
                script += """
import sys, time
from pathlib import Path
observe = observed.preparation_progress
def progress(stage, completed=0, total=0):
    if stage == 'starting-engine':
        marker = Path(sys.argv[sys.argv.index('--progress') + 1])
        marker.with_suffix('.boundary').touch()
        deadline = time.monotonic() + 30
        while not marker.with_name(marker.name + '.cancel').exists():
            if time.monotonic() > deadline:
                raise ValueError('test did not cancel at activation boundary')
            time.sleep(0.01)
    observe(stage, completed, total)
observed.preparation_progress = progress
"""
            arguments = ("-I", "-c", script + "environment.main()", *arguments[3:])
        return await launch(executable, *arguments, **kwargs)

    monkeypatch.setattr(home_setup.asyncio, "create_subprocess_exec", signed_process)

    async def scenario() -> None:
        nonlocal cancel, change_before_launch
        installed = await home_setup.environment_command(
            root,
            "setup",
            "--channel",
            "nightly",
            "--variant",
            "fixture-cpu",
            "--data",
            str(tmp_path / "store"),
            "--port",
            str(free_port()),
            "--yes",
        )
        assert installed["current"] == 1
        channel = Path(fixture["mirror"]) / "channels/nightly.json"
        original_channel = channel.read_bytes()
        channel.write_bytes(Path(fixture["updatedChannel"]).read_bytes())
        if operation == "restore":
            assert (await home_setup.environment_command(root, "update", "--yes"))["current"] == 2
        await asyncio.to_thread(stop_engine, root, "main")
        current = 2 if operation == "restore" else 1
        settings = (root / "environment.json").read_bytes()
        previous = (root / "builds" / str(current) / "build.json").read_bytes()
        home = Home(watch_interval=0.01)
        path = "/supervisor/installations/" + operation
        params = {"root": str(root), **({"entry": "1"} if operation == "restore" else {})}
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            if opened:
                await home.open(root, "main")
                for _ in range(600):
                    if home.link.state == "ready":
                        break
                    await asyncio.sleep(0.05)
                assert home.link.state == "ready", home.link.detail
            for invalid_params, expected in (
                ({"root": str(tmp_path / "unknown")}, 404),
                ({"root": "relative"}, 400),
                ({"root": str(root), "command": "stop"}, 400),
            ):
                assert (await client.get(path, params=invalid_params)).status == expected
            if operation == "restore":
                for entry in ("", "0", "-1", "unknown"):
                    assert (
                        await client.get(path, params={"root": str(root), "entry": entry})
                    ).status == 400
            response = await client.get(path, params=params)
            assert response.status == 200, await response.text()
            plan = (await response.json())["plan"]
            if operation == "restore":
                assert plan["observed"]["current"] == current and plan["exact"] is True
                plan_id = plan["alternatives"][0]["planId"]
            else:
                assert plan["current"] == 1 and plan["version"] == "2.0"
                plan_id = plan["id"]
            assert (root / "environment.json").read_bytes() == settings
            assert (root / "builds" / str(current) / "build.json").read_bytes() == previous
            expected_state = "ready" if opened else "stopped"
            before_engines = engine_status(root)
            assert len(before_engines) == 1 and before_engines[0]["state"] == expected_state
            before_assignments = engine_assignments(root, current)
            if operation == "update":
                channel.write_bytes(original_channel)
                stale = await client.post(path, json={"planId": plan_id})
                assert stale.status == 409 and (await stale.json())["error"] == "plan-changed"
                channel.write_bytes(Path(fixture["updatedChannel"]).read_bytes())
            changed = json.loads(settings)
            changed["responseCompression"] = True
            write_document(root / "environment.json", changed)
            stale = await client.post(path, json={"planId": plan_id})
            assert stale.status == 409 and (await stale.json())["error"] == "plan-changed"
            (root / "environment.json").write_bytes(settings)
            assert (await client.post("/supervisor/setup", json={"planId": plan_id})).status == 404
            other_operation = "restore" if operation == "update" else "update"
            assert (
                await client.post(
                    "/supervisor/installations/" + other_operation, json={"planId": plan_id}
                )
            ).status == 404
            assert (
                await client.post(path, json={"planId": plan_id, "root": str(root)})
            ).status == 400
            change_before_launch = True
            response = await client.post(path, json={"planId": plan_id})
            assert response.status == 202, await response.text()
            raced = await response.json()
            race_path = "/supervisor/installations/jobs/" + raced["id"]
            for _ in range(600):
                raced = await (await client.get(race_path)).json()
                if raced["state"] not in {"running", "cancelling"}:
                    break
                await asyncio.sleep(0.05)
            assert raced["state"] == "failed", raced
            expected_error = "approval is stale" if operation == "restore" else "review a new plan"
            assert expected_error in raced["error"]["message"], raced
            assert (root / "current").read_text() == f"{current}\n"
            (root / "environment.json").write_bytes(settings)
            change_before_launch = False
            for cancelling in (True, False):
                cancel = cancelling
                response = await client.post(path, json={"planId": plan_id})
                assert response.status == 202, await response.text()
                job = await response.json()
                job_path = "/supervisor/installations/jobs/" + job["id"]
                progress = home_setup.dinkster_home() / "setup-jobs" / (job["id"] + ".json")
                requested = False
                for _ in range(600):
                    job = await (await client.get(job_path)).json()
                    if cancelling and not requested and progress.with_suffix(".boundary").exists():
                        assert job["progress"] is not None
                        for start_path in (path, "/supervisor/setup"):
                            assert (
                                await client.post(start_path, json={"planId": plan_id})
                            ).status == 409
                        assert (await client.delete(job_path)).status == 202
                        requested = True
                    if job["state"] not in {"running", "cancelling"}:
                        break
                    await asyncio.sleep(0.05)
                assert job["state"] == ("cancelled" if cancelling else "completed"), job
                assert (
                    not progress.exists()
                    and not progress.with_name(progress.name + ".cancel").exists()
                )
                if opened:
                    for _ in range(600):
                        if home.link.state == "ready":
                            break
                        await asyncio.sleep(0.05)
                    assert home.link.state == "ready", home.link.detail
                    assert home.target is not None and home.target.root == root
                else:
                    assert home.target is None
                if cancelling:
                    assert requested and (root / "current").read_text() == f"{current}\n"
                    assert (root / "environment.json").read_bytes() == settings
                    assert (root / "builds" / str(current) / "build.json").read_bytes() == previous
                    assert not (root / "builds" / str(current + 1)).exists()
                else:
                    assert (root / "current").read_text() == f"{current + 1}\n"
                    assert json.loads(
                        (root / "builds" / str(current + 1) / "build.json").read_text()
                    )["engine"]["version"] == ("1.0" if operation == "restore" else "2.0")
                engines = engine_status(root)
                assert len(engines) == 1 and engines[0]["state"] == expected_state
                assignments = engine_assignments(root, current if cancelling else current + 1)
                assert [
                    {key: value for key, value in row.items() if key != "port"}
                    for row in assignments
                ] == [
                    {key: value for key, value in row.items() if key != "port"}
                    for row in before_assignments
                ]
                if cancelling or not opened:
                    assert assignments[0]["port"] == before_assignments[0]["port"]
            stale = await client.post(path, json={"planId": plan_id})
            assert stale.status == 409 and (await stale.json())["error"] == "plan-changed"

    try:
        run_scenario(scenario())
    finally:
        stop_engine(root, "main")


@pytest.mark.parametrize(
    "outcome", ["completed", "cancelled", "opening-failed", "metadata-changed"]
)
def test_setup_job_is_shared_between_screens_and_finishes_cleanup_before_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    root = tmp_path / "installation"
    cancel = outcome == "cancelled"

    def facts(root: Path, channel: str, variant: str | None) -> dict[str, object]:
        return {
            "root": str(root),
            "channel": channel,
            "variant": "linux-cpu",
            "buildLabel": "CPU only",
            "engineRelease": "sha256:" + "a" * 64,
            "baseId": "sha256:" + "b" * 64,
            "detectedCards": [],
            "choices": [{"variant": "linux-cpu", "label": "CPU only"}],
            "gpus": [],
            "downloadBytes": None,
            "runtimeDownloadBytes": None,
            "freeBytes": 10000,
            "refusal": None,
        }

    monkeypatch.setattr(home_setup, "setup_facts", facts)

    async def scenario() -> None:
        started, finish, cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
        opened: list[Path] = []
        metadata_identity = "a" * 64

        async def completed(root: Path) -> None:
            if outcome == "opening-failed":
                raise LookupError("installation registration is unavailable")
            opened.append(root)

        async def command(root: Path, *arguments: str) -> dict[str, object]:
            if "--plan" in arguments:
                return {
                    "runtimeDownloadBytes": 4096,
                    "status": "planned",
                    "downloadEstimate": {
                        "kind": "upper-bound",
                        "bytes": 8273,
                        "reason": "Runtime and all locked alternatives.",
                    },
                    "packEstimateIdentity": metadata_identity,
                }
            progress = Path(arguments[arguments.index("--progress") + 1])
            write_document(progress, {"stage": "downloading", "completed": 17, "total": 83})
            started.set()
            await finish.wait()
            if cancel:
                assert progress.with_name(progress.name + ".cancel").exists()
                await cleanup.wait()
                return {"status": "cancelled"}
            return {"status": "applied"}

        monkeypatch.setattr(home_setup, "environment_command", command)
        home = Home(installations=frozenset)
        home.setup = home_setup.SetupJobs(completed)
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            plan = await (await client.get("/supervisor/setup", params={"root": str(root)})).json()
            assert plan["plan"]["downloadBytes"] is None
            assert plan["plan"]["runtimeDownloadBytes"] == 4096
            assert plan["plan"]["downloadEstimate"] == {
                "kind": "upper-bound",
                "bytes": 8273,
                "reason": "Runtime and all locked alternatives.",
            }
            if outcome == "metadata-changed":
                metadata_identity = "b" * 64
                response = await client.post(
                    "/supervisor/setup", json={"planId": plan["plan"]["id"]}
                )
                assert response.status == 409
                assert (await response.json())["error"] == "plan-changed"
                assert not started.is_set() and not opened and not root.exists()
                return
            responses = await asyncio.gather(
                client.post("/supervisor/setup", json={"planId": plan["plan"]["id"]}),
                client.post("/supervisor/setup", json={"planId": plan["plan"]["id"]}),
            )
            assert sorted(response.status for response in responses) == [202, 409]
            response = next(response for response in responses if response.status == 202)
            job = await response.json()
            await started.wait()
            path = "/supervisor/setup/" + job["id"]
            if outcome == "completed":
                reading, release = threading.Event(), threading.Event()
                original_read = home_setup.read_document

                def held_read(progress: Path) -> dict[str, object]:
                    reading.set()
                    if not release.wait(timeout=30):
                        raise AssertionError("Progress reading blocked the Home event loop")
                    return original_read(progress)

                with monkeypatch.context() as held:
                    held.setattr(home_setup, "read_document", held_read)
                    pending = asyncio.create_task(client.get(path))
                    try:
                        assert await asyncio.to_thread(reading.wait, 30)
                        status = await client.get("/supervisor/status")
                        assert status.status == 200
                        release.set()
                        assert (await pending).status == 200
                    except BaseException:
                        finish.set()
                        cleanup.set()
                        raise
                    finally:
                        release.set()
            assert (await (await client.get(path)).json())["progress"] == {
                "stage": "downloading",
                "completed": 17,
                "total": 83,
                "unit": "bytes",
            }
            active = await (await client.get("/supervisor/setup/active")).json()
            assert active["activeJob"] == await (await client.get(path)).json()
            assert active["activeJob"]["state"] == "running"
            reloaded = await (
                await client.get("/supervisor/setup", params={"root": str(root)})
            ).json()
            assert reloaded["activeJob"]["id"] == job["id"]
            busy = await client.post("/supervisor/setup", json={"planId": plan["plan"]["id"]})
            assert busy.status == 409 and (await busy.json())["error"] == "setup-active"
            if cancel:
                response = await client.delete(path)
                assert response.status == 202 and (await response.json())["state"] == "cancelling"
                active = await (await client.get("/supervisor/setup/active")).json()
                assert active["activeJob"]["id"] == job["id"]
                assert active["activeJob"]["state"] == "cancelling"
                finish.set()
                await asyncio.sleep(0)
                assert (await (await client.get(path)).json())["state"] == "cancelling"
                joining = await (
                    await client.get("/supervisor/setup", params={"root": str(root)})
                ).json()
                assert joining["activeJob"]["state"] == "cancelling"
                assert joining["plan"]["runtimeDownloadBytes"] == 4096
                assert joining["plan"]["downloadEstimate"] == plan["plan"]["downloadEstimate"]
                assert joining["plan"]["refusal"] is None
                assert opened == []
                cleanup.set()
            else:
                finish.set()
            final = await (await client.get(path)).json()
            for _ in range(100):
                final = await (await client.get(path)).json()
                if final["state"] in {"completed", "cancelled", "failed"}:
                    break
                await asyncio.sleep(0.01)
            assert final["state"] == ("failed" if outcome == "opening-failed" else outcome), final
            assert final["installation"] == (
                {"root": str(root), "engine": "main"} if outcome == "completed" else None
            )
            if outcome == "opening-failed":
                assert final["error"] == {
                    "code": "setup-failed",
                    "message": "installation registration is unavailable",
                }
            assert opened == ([root] if outcome == "completed" else [])
            assert await (await client.get("/supervisor/setup/active")).json() == {
                "activeJob": final
            }
            assert (await client.delete(path)).status == 200

    run_scenario(scenario())


@pytest.mark.parametrize(
    ("torch", "cuda"),
    [("2.14.1+cu132", "13.2"), ("2.9.1+cu126", "12.6"), ("2.14.1+cpu", None), (None, None)],
)
def test_setup_details_follow_the_selected_build_not_the_planning_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch: str | None, cuda: str | None
) -> None:
    from dinkster_supervisor import environment_hardware

    monkeypatch.setattr(environment_hardware, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(environment_hardware.platform, "machine", lambda: "x86_64")
    common = {"platform": {"os": "linux", "arch": "x86_64"}, "nvidia": None}
    catalog = SimpleNamespace(
        bases={
            "first": {
                **common,
                "python": "3.14.10",
                "wheels": [{"name": "torch", "version": "2.8.0+cpu"}],
            },
            "second": {
                **common,
                "python": "3.13.8",
                "wheels": [{"name": "torch", "version": torch}] if torch else [],
            },
        },
        channel=SimpleNamespace(engine_release="release", base_for=lambda variant: variant),
        engine={"version": "0.0.19.dev123"},
    )
    monkeypatch.setattr(home_setup, "discover", lambda *args: catalog)
    monkeypatch.setattr(home_setup, "detect_graphics_cards", list)
    monkeypatch.setattr(home_setup, "library_engine", lambda _: None)

    async def command(root: Path, *arguments: str) -> dict[str, object]:
        assert "--plan" in arguments and "--yes" not in arguments
        return {"runtimeDownloadBytes": 123}

    monkeypatch.setattr(home_setup, "environment_command", command)

    async def scenario() -> None:
        home = Home(installations=frozenset)
        root = tmp_path / "prospective"
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            first = await client.get(
                "/supervisor/setup", params={"root": str(root), "variant": "first"}
            )
            assert first.status == 200
            assert (await first.json())["plan"]["buildDetails"] == {
                "python": "3.14.10",
                "pytorch": "2.8.0+cpu",
                "cuda": None,
                "dinkster": "0.0.19.dev123",
            }
            second = await client.get(
                "/supervisor/setup", params={"root": str(root), "variant": "second"}
            )
            assert second.status == 200
            assert (await second.json())["plan"]["buildDetails"] == (
                {
                    "python": "3.13.8",
                    "pytorch": torch,
                    "cuda": cuda,
                    "dinkster": "0.0.19.dev123",
                }
                if torch
                else None
            )
        assert not root.exists()

    run_scenario(scenario())


@pytest.mark.usefixtures("signed_channels")
@pytest.mark.parametrize(
    ("cancel", "occupied"),
    [(None, False), (None, True), ("checking-setup", False), ("starting-engine", False)],
)
def test_setup_route_runs_real_cli_completion_or_cancellation_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: str | None, occupied: bool
) -> None:
    from dinkster_supervisor.environment_runtime import engine_status, stop_engine
    from dinkster_supervisor.environment_state import shared_library
    from dinkster_supervisor.home import known_installations

    fixture = build(tmp_path / "fixture", "fixture-cpu", free_port())
    monkeypatch.setattr(home_setup, "RELEASE_MIRROR", fixture["mirror"])
    monkeypatch.setattr(home_setup, "detect_graphics_cards", list)
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    launch = asyncio.create_subprocess_exec

    async def signed_process(
        executable: str, *arguments: str, **kwargs: Any
    ) -> asyncio.subprocess.Process:
        if arguments[:3] == ("-I", "-m", "dinkster_supervisor.environment"):
            script = (
                "from dinkster_supervisor import environment, environment_channel_signatures; "
                f"environment_channel_signatures.CHANNEL_KEYS = {KEYS!r}\n"
            )
            if "--yes" in arguments:
                script += """
from dinkster_library._migration import record_backup
from dinkster_supervisor import environment_transactions
from dinkster_supervisor.environment_state import dinkster_home, shared_library
start_engine = environment_transactions.start_engine
def migrate_and_start(root, *args, **kwargs):
    with record_backup(shared_library(), dinkster_home()):
        pass
    return start_engine(root, *args, **kwargs)
environment_transactions.start_engine = migrate_and_start
"""
            if cancel and "--yes" in arguments:
                script += (
                    f"cancel_stage = {cancel!r}\n"
                    + """
import sys
import time
from pathlib import Path
observe = environment.preparation_progress
def progress(stage, completed=0, total=0):
    if stage == cancel_stage and Path(sys.argv[sys.argv.index('--root') + 1]).exists():
        marker = Path(sys.argv[sys.argv.index('--progress') + 1])
        marker.with_suffix('.boundary').touch()
        deadline = time.monotonic() + 30
        while not marker.with_name(marker.name + '.cancel').exists():
            if time.monotonic() > deadline:
                raise ValueError('test did not request cancellation at engine boundary')
            time.sleep(0.01)
    observe(stage, completed, total)
environment.preparation_progress = progress
"""
                )
            arguments = (
                "-I",
                "-c",
                script + "environment.main()",
                *arguments[3:],
            )
        return await launch(executable, *arguments, **kwargs)

    monkeypatch.setattr(home_setup.asyncio, "create_subprocess_exec", signed_process)
    root = tmp_path / "installation"
    competitor = socket.socket()
    if sys.platform == "win32":
        competitor.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    command = home_setup.environment_command

    async def competing_command(root: Path, *arguments: str) -> dict[str, object]:
        if "--yes" in arguments:
            port = int(arguments[arguments.index("--port") + 1])
            competitor.bind(("127.0.0.1", port))
            competitor.listen()
        return await command(root, *arguments)

    if occupied:
        monkeypatch.setattr(home_setup, "environment_command", competing_command)

    async def scenario() -> None:
        nonlocal cancel
        home = Home(watch_interval=0.01)
        app = create_home_app(home, frontend=None, library=shared_library())
        async with TestClient(TestServer(app)) as client:
            created = await client.post("/api/projects", json={"id": "film", "name": "Film"})
            assert created.status == 200
            response = await client.get("/supervisor/setup", params={"root": str(root)})
            assert response.status == 200, await response.text()
            plan = (await response.json())["plan"]
            assert plan["refusal"] is None and plan["variant"] == "fixture-cpu"
            assert plan["buildReason"] == "Does not use the graphics card and is much slower."
            assert plan["choices"] == [
                {
                    "variant": "fixture-cpu",
                    "vendor": None,
                    "label": "fixture-cpu",
                    "reason": plan["buildReason"],
                    "gpus": [],
                }
            ]
            assert plan["runtimeDownloadBytes"] > 0 and plan["downloadBytes"] is None
            assert not root.exists()
            response = await client.post("/supervisor/setup", json={"planId": plan["id"]})
            assert response.status == 202, await response.text()
            job = await response.json()
            path = "/supervisor/setup/" + job["id"]
            boundary = home_setup.dinkster_home() / "setup-jobs" / (job["id"] + ".boundary")
            requested_cancel = False
            for _ in range(600):
                job = await (await client.get(path)).json()
                if cancel and not requested_cancel and boundary.exists():
                    assert (await client.get("/api/projects/film")).status == 503
                    joining = await (
                        await client.get("/supervisor/setup", params={"root": str(root)})
                    ).json()
                    assert joining["activeJob"]["id"] == job["id"]
                    for key in (
                        "variant",
                        "buildLabel",
                        "buildReason",
                        "choices",
                        "detectedCards",
                        "gpus",
                        "engineRelease",
                        "baseId",
                        "downloadBytes",
                        "runtimeDownloadBytes",
                        "downloadEstimate",
                        "refusal",
                    ):
                        assert joining["plan"][key] == plan[key], key
                    unrelated = tmp_path / "unrelated"
                    unrelated.mkdir()
                    (unrelated / "keep.txt").write_text("not setup state")
                    for params in (
                        {"root": str(root), "variant": "different-build"},
                        {"root": str(root), "channel": "stable"},
                        {"root": str(unrelated)},
                    ):
                        other = await (await client.get("/supervisor/setup", params=params)).json()
                        assert other["plan"]["refusal"]["code"] == "invalid-root"
                    assert (unrelated / "keep.txt").read_text() == "not setup state"
                    refused = await client.post(
                        "/supervisor/setup", json={"planId": joining["plan"]["id"]}
                    )
                    assert refused.status == 409
                    assert (await refused.json())["error"] == "setup-active"
                    response = await client.delete(path)
                    assert response.status == 202
                    assert (await response.json())["state"] == "cancelling"
                    requested_cancel = True
                if job["state"] not in {"running", "cancelling"}:
                    break
                await asyncio.sleep(0.05)
            if cancel:
                assert requested_cancel and job["state"] == "cancelled", job
                assert job["installation"] is None
                assert root not in known_installations()
                assert not (root / "current").exists()
                assert not (root / "environment.json").exists()
                assert not (root / "builds/1").exists()
                assert home.target is None
                project = await client.get("/api/projects/film")
                assert project.status == 200
                assert (await project.json())["project"]["name"] == "Film"
                journals = {path: path.read_bytes() for path in (root / "journal").glob("*")}
                retry = await client.get("/supervisor/setup", params={"root": str(root)})
                assert retry.status == 200
                assert (await retry.json())["plan"]["refusal"] is None
                kept = root / "keep.txt"
                kept.write_text("not setup metadata")
                response = await client.post("/supervisor/setup", json={"planId": plan["id"]})
                assert response.status == 409
                assert (await response.json())["error"] == "invalid-root"
                assert kept.read_text() == "not setup metadata"
                kept.unlink()
                lock = root / ".lock"
                original_lock = lock.read_bytes()
                lock.write_bytes(b"changed after cancellation")
                response = await client.post("/supervisor/setup", json={"planId": plan["id"]})
                assert response.status == 409
                assert (await response.json())["error"] == "invalid-root"
                lock.write_bytes(original_lock)
                outside = tmp_path / "outside.txt"
                outside.write_text("untouched")
                builds = root / "builds"
                had_builds = builds.exists()
                builds.mkdir(exist_ok=True)
                linked = builds / "escape"
                linked.symlink_to(outside)
                response = await client.post("/supervisor/setup", json={"planId": plan["id"]})
                assert response.status == 409
                assert (await response.json())["error"] == "invalid-root"
                assert outside.read_text() == "untouched"
                linked.unlink()
                if not had_builds:
                    builds.rmdir()
                for journal, original in journals.items():
                    journal.write_text('{"format":"invalid"}')
                    response = await client.post("/supervisor/setup", json={"planId": plan["id"]})
                    assert response.status == 409
                    assert (await response.json())["error"] == "invalid-root"
                    journal.write_bytes(original)
                for attempt in range(2):
                    if attempt == 1:
                        cancel = None
                    reviewed = await (
                        await client.get("/supervisor/setup", params={"root": str(root)})
                    ).json()
                    assert reviewed["plan"]["refusal"] is None
                    response = await client.post("/supervisor/setup", json={"planId": plan["id"]})
                    assert response.status == 202, await response.text()
                    job = await response.json()
                    path = "/supervisor/setup/" + job["id"]
                    boundary = home_setup.dinkster_home() / "setup-jobs" / (job["id"] + ".boundary")
                    requested_cancel = False
                    for _ in range(600):
                        job = await (await client.get(path)).json()
                        if cancel and not requested_cancel and boundary.exists():
                            joined = await (
                                await client.get("/supervisor/setup", params={"root": str(root)})
                            ).json()
                            assert joined["activeJob"]["id"] == job["id"]
                            for key in (
                                "variant",
                                "buildLabel",
                                "choices",
                                "gpus",
                                "engineRelease",
                                "baseId",
                                "runtimeDownloadBytes",
                                "downloadEstimate",
                                "refusal",
                            ):
                                assert joined["plan"][key] == reviewed["plan"][key], key
                            response = await client.delete(path)
                            assert response.status == 202
                            requested_cancel = True
                        if job["state"] not in {"running", "cancelling"}:
                            break
                        await asyncio.sleep(0.05)
                    if cancel:
                        assert requested_cancel and job["state"] == "cancelled", job
                        assert job["installation"] is None and root not in known_installations()
                assert all(path.read_bytes() == content for path, content in journals.items())
            assert job["state"] == "completed", job.get("error")
            assert job["installation"] == {"root": str(root), "engine": "main"}
            assert root in known_installations()
            assert (root / "current").read_text().strip() == "1"
            assert home.target is not None and home.target.root == root
            assert engine_status(root)[0]["state"] == "ready"
            ended = await (await client.get("/supervisor/setup", params={"root": str(root)})).json()
            assert ended["plan"]["refusal"]["code"] == "invalid-root"
            other = tmp_path / "another-installation"
            response = await client.get("/supervisor/setup", params={"root": str(other)})
            assert response.status == 200
            refused = await response.json()
            refused_plan = refused["plan"]
            assert refused_plan["refusal"]["code"] == "library-held"
            assert refused_plan["variant"] == "fixture-cpu"
            assert refused_plan["buildLabel"] == "fixture-cpu"
            assert refused_plan["runtimeDownloadBytes"] >= 0
            assert refused_plan["downloadBytes"] is None
            assert refused_plan["freeBytes"] > 0
            assert refused["activeJob"]["id"] == job["id"]
            assert refused["activeJob"]["state"] == "completed"
            assert not other.exists()
            assert engine_status(root)[0]["state"] == "ready"
            if occupied:
                assert engine_status(root)[0]["url"] != (
                    f"http://127.0.0.1:{competitor.getsockname()[1]}"
                )
            await home.shutdown()
            assert engine_status(root)[0]["state"] == "stopped"
            if occupied:
                competitor.setblocking(False)
                with pytest.raises(BlockingIOError):
                    competitor.accept()

    try:
        run_scenario(scenario())
    finally:
        try:
            if (root / "current").exists():
                stop_engine(root, "main")
        finally:
            competitor.close()


@pytest.mark.usefixtures("signed_channels")
@pytest.mark.parametrize("unavailable", [None, "build", "hardware", "download"])
def test_held_library_preserves_read_only_facts_and_refuses_start_without_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unavailable: str | None
) -> None:
    from test_environment_artifacts import _mirror

    mirror = tmp_path / "mirror"
    _mirror(mirror)
    monkeypatch.setattr(home_setup, "RELEASE_MIRROR", str(mirror))
    monkeypatch.setattr(
        home_setup, "detect_graphics_cards", lambda: None if unavailable == "hardware" else []
    )
    monkeypatch.setattr(home_setup, "library_engine", lambda _: "other installation/main")

    def no_build(*args: object, **kwargs: object) -> None:
        raise ValueError("metadata unavailable")

    async def metadata(root: Path, *arguments: str) -> dict[str, object]:
        assert "--plan" in arguments and "--yes" not in arguments
        if unavailable == "download":
            raise home_setup.SetupRefused("setup-failed", "download metadata unavailable")
        return {
            "runtimeDownloadBytes": 431,
            "downloadEstimate": {"kind": "upper-bound", "bytes": 827, "reason": "Bound."},
            "packEstimateIdentity": "a" * 64,
        }

    if unavailable == "build":
        monkeypatch.setattr(home_setup, "discover", no_build)
    monkeypatch.setattr(home_setup, "environment_command", metadata)

    async def scenario() -> None:
        home = Home(installations=frozenset)
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            response = await client.get("/supervisor/setup", params={"root": str(tmp_path / "new")})
            assert response.status == 200
            plan = (await response.json())["plan"]
            assert plan["refusal"]["code"] == "library-held"
            assert plan["root"] == str(tmp_path / "new") and plan["freeBytes"] > 0
            assert plan["downloadBytes"] is None
            if unavailable in {"build", "hardware"}:
                assert plan["variant"] is None and plan["buildLabel"] is None
                assert plan["detectedCards"] is None and plan["choices"] is None
            else:
                assert plan["variant"] == "fixture-cpu" and plan["buildLabel"] == "fixture-cpu"
                assert plan["detectedCards"] == [] and plan["choices"]
                assert plan["engineRelease"] and plan["baseId"]
            if unavailable:
                assert plan["runtimeDownloadBytes"] is None and plan["downloadEstimate"] is None
            else:
                assert plan["runtimeDownloadBytes"] == 431
                assert plan["downloadEstimate"] == {
                    "kind": "upper-bound",
                    "bytes": 827,
                    "reason": "Bound.",
                }
            assert "Stop that engine" in plan["refusal"]["message"]
            refused = await client.post("/supervisor/setup", json={"planId": plan["id"]})
            assert refused.status == 409 and (await refused.json())["error"] == "library-held"
            for method, path in (
                ("GET", "/supervisor/setup"),
                ("GET", "/supervisor/setup/active"),
                ("POST", "/supervisor/setup"),
                ("DELETE", "/supervisor/setup/job"),
            ):
                assert (
                    await client.request(
                        method, path, headers={"Origin": "https://untrusted.example"}
                    )
                ).status == 403
                assert (
                    await client.request(method, path, headers={"Host": "untrusted.example"})
                ).status == 421
            assert not (tmp_path / "new").exists()
            assert (await client.get("/supervisor/setup/missing")).status == 404
            assert (
                await client.post("/supervisor/setup", json={"planId": "missing"})
            ).status == 404
            assert (
                await client.post("/supervisor/setup", json={"root": "/unreviewed"})
            ).status == 400

    run_scenario(scenario())


def test_repairable_card_requires_a_build_choice_before_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_supervisor import environment_hardware

    monkeypatch.setattr(environment_hardware, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(environment_hardware.platform, "machine", lambda: "x86_64")
    cpu = {"platform": {"os": "linux", "arch": "x86_64"}, "nvidia": None}
    gpu = {
        **cpu,
        "nvidia": {
            "minimumDriver": "580",
            "minimumComputeCapability": "7.5",
            "maximumExclusiveComputeCapability": None,
        },
    }
    monkeypatch.setattr(
        home_setup,
        "discover",
        lambda *args: SimpleNamespace(
            bases={"linux-cpu": cpu, "linux-nvidia": gpu},
            channel=SimpleNamespace(engine_release="release", base_for=lambda _: "base"),
        ),
    )
    monkeypatch.setattr(
        home_setup,
        "detect_graphics_cards",
        lambda: [
            {
                "id": "0",
                "name": "NVIDIA card",
                "driver": "570",
                "computeCapability": "8.6",
                "memoryMiB": 8192,
            }
        ],
    )
    estimates = []

    async def command(root: Path, *arguments: str) -> dict[str, object]:
        assert "--plan" in arguments and "--variant" in arguments
        assert arguments[arguments.index("--variant") + 1] == "linux-cpu"
        estimates.append(root)
        return {"runtimeDownloadBytes": 123}

    monkeypatch.setattr(home_setup, "environment_command", command)
    root = tmp_path / "prospective"

    async def scenario() -> None:
        home = Home(installations=frozenset)
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            response = await client.get("/supervisor/setup", params={"root": str(root)})
            plan = (await response.json())["plan"]
            assert plan["variant"] is None and plan["refusal"] is None
            assert plan["buildLabel"] is None and plan["engineRelease"] is None
            assert [choice["variant"] for choice in plan["choices"]] == ["linux-cpu"]
            assert plan["unusableCards"][0]["reason"] == "driver-too-old"
            assert estimates == []
            response = await client.post("/supervisor/setup", json={"planId": plan["id"]})
            assert response.status == 409
            assert (await response.json())["error"] == "build-not-chosen"
            assert estimates == []
            response = await client.get(
                "/supervisor/setup", params={"root": str(root), "variant": "linux-cpu"}
            )
            chosen = (await response.json())["plan"]
            assert chosen["variant"] == "linux-cpu" and chosen["gpus"] == []
            assert chosen["refusal"] is None
            assert chosen["unusableCards"] == plan["unusableCards"]
            assert chosen["runtimeDownloadBytes"] == 123
            assert estimates == [root]
        assert not root.exists()

    run_scenario(scenario())


@pytest.mark.parametrize("refusal", ["invalid-root", "build-unavailable", "hardware-unavailable"])
def test_unavailable_build_and_invalid_folder_have_distinct_refusals_not_empty_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    root = tmp_path / "installation"
    if refusal == "invalid-root":
        root.mkdir()
        (root / "keep.txt").write_text("keep")

    def unavailable(*args: object, **kwargs: object) -> object:
        if refusal == "invalid-root":
            pytest.fail("invalid folder must be refused before build resolution")
        if refusal == "hardware-unavailable":
            return SimpleNamespace(bases={})
        raise ValueError("no mirror supplied a valid trusted channel")

    monkeypatch.setattr(home_setup, "discover", unavailable)
    monkeypatch.setattr(home_setup, "detect_graphics_cards", lambda: None)

    async def scenario() -> None:
        home = Home(installations=frozenset)
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            response = await client.get("/supervisor/setup", params={"root": str(root)})
            assert response.status == 200
            result = await response.json()
            plan = result["plan"]
            assert plan["refusal"]["code"] == refusal
            assert plan["refusal"]["message"]
            assert plan["buildReason"] is None
            assert plan["detectedCards"] is None and plan["choices"] is None
            assert plan["runtimeDownloadBytes"] is None and plan["downloadBytes"] is None
            assert plan["freeBytes"] >= 0
            assert result["activeJob"] is None
            response = await client.post("/supervisor/setup", json={"planId": plan["id"]})
            assert response.status == 409
            assert (await response.json())["error"] == plan["refusal"]["code"]
        if refusal == "invalid-root":
            assert (root / "keep.txt").read_text() == "keep"
        else:
            assert not root.exists()

    run_scenario(scenario())
