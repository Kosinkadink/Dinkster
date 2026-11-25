from __future__ import annotations

import asyncio
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


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "opening-failed"])
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
            "buildLabel": "Linux with CPU only",
            "engineRelease": "sha256:" + "a" * 64,
            "baseId": "sha256:" + "b" * 64,
            "detectedCards": [],
            "choices": [{"variant": "linux-cpu", "label": "Linux with CPU only"}],
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

        async def completed(root: Path) -> None:
            if outcome == "opening-failed":
                raise LookupError("installation registration is unavailable")
            opened.append(root)

        async def command(root: Path, *arguments: str) -> dict[str, object]:
            if "--plan" in arguments:
                return {"runtimeDownloadBytes": 4096, "status": "planned"}
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
            responses = await asyncio.gather(
                client.post("/supervisor/setup", json={"planId": plan["plan"]["id"]}),
                client.post("/supervisor/setup", json={"planId": plan["plan"]["id"]}),
            )
            assert sorted(response.status for response in responses) == [202, 409]
            response = next(response for response in responses if response.status == 202)
            job = await response.json()
            await started.wait()
            path = "/supervisor/setup/" + job["id"]
            assert (await (await client.get(path)).json())["progress"] == {
                "stage": "downloading",
                "completed": 17,
                "total": 83,
                "unit": "bytes",
            }
            reloaded = await (
                await client.get("/supervisor/setup", params={"root": str(root)})
            ).json()
            assert reloaded["activeJob"]["id"] == job["id"]
            busy = await client.post("/supervisor/setup", json={"planId": plan["plan"]["id"]})
            assert busy.status == 409 and (await busy.json())["error"] == "setup-active"
            if cancel:
                response = await client.delete(path)
                assert response.status == 202 and (await response.json())["state"] == "cancelling"
                finish.set()
                await asyncio.sleep(0)
                assert (await (await client.get(path)).json())["state"] == "cancelling"
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
            assert (await client.delete(path)).status == 200

    run_scenario(scenario())


@pytest.mark.usefixtures("signed_channels")
@pytest.mark.parametrize("cancel", [False, True])
def test_setup_route_runs_real_cli_completion_or_cancellation_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    from dinkster_supervisor.environment_runtime import engine_status, stop_engine
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
            if cancel and "--yes" in arguments:
                script += """
import sys
import time
from pathlib import Path
observe = environment.preparation_progress
def progress(stage, completed=0, total=0):
    if stage == 'starting-engine':
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
            arguments = (
                "-I",
                "-c",
                script + "environment.main()",
                *arguments[3:],
            )
        return await launch(executable, *arguments, **kwargs)

    monkeypatch.setattr(home_setup.asyncio, "create_subprocess_exec", signed_process)
    root = tmp_path / "installation"

    async def scenario() -> None:
        home = Home(watch_interval=0.01)
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            response = await client.get("/supervisor/setup", params={"root": str(root)})
            assert response.status == 200, await response.text()
            plan = (await response.json())["plan"]
            assert plan["refusal"] is None and plan["variant"] == "fixture-cpu"
            assert "No supported NVIDIA card" in plan["buildReason"]
            assert plan["choices"] == [
                {
                    "variant": "fixture-cpu",
                    "label": "fixture-cpu",
                    "reason": plan["buildReason"],
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
                return
            assert job["state"] == "completed", job
            assert job["installation"] == {"root": str(root), "engine": "main"}
            assert root in known_installations()
            assert (root / "current").read_text().strip() == "1"
            assert home.target is not None and home.target.root == root
            assert engine_status(root)[0]["state"] == "ready"
            other = tmp_path / "another-installation"
            response = await client.get("/supervisor/setup", params={"root": str(other)})
            assert response.status == 200
            refused = await response.json()
            refused_plan = refused["plan"]
            assert refused_plan["refusal"]["code"] == "library-held"
            assert refused_plan["runtimeDownloadBytes"] is None
            assert refused["activeJob"]["id"] == job["id"]
            assert refused["activeJob"]["state"] == "completed"
            assert not other.exists()
            assert engine_status(root)[0]["state"] == "ready"
            await home.shutdown()
            assert engine_status(root)[0]["state"] == "stopped"

    try:
        run_scenario(scenario())
    finally:
        if (root / "current").exists():
            stop_engine(root, "main")


def test_held_library_refuses_planning_without_acquisition_and_routes_enforce_browser_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(home_setup, "library_engine", lambda _: "other installation/main")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("held-library planning must not discover or download software")

    monkeypatch.setattr(home_setup, "discover", forbidden)

    async def scenario() -> None:
        home = Home(installations=frozenset)
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            response = await client.get("/supervisor/setup", params={"root": str(tmp_path / "new")})
            assert response.status == 200
            plan = (await response.json())["plan"]
            assert plan["refusal"]["code"] == "library-held"
            assert plan["detectedCards"] is None and plan["choices"] is None
            assert "Stop that engine" in plan["refusal"]["message"]
            refused = await client.post("/supervisor/setup", json={"planId": plan["id"]})
            assert refused.status == 409 and (await refused.json())["error"] == "library-held"
            for method, path in (
                ("GET", "/supervisor/setup"),
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
