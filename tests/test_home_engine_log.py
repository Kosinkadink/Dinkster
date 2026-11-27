from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_supervisor.home import Home, OpenTarget, create_home_app

from tests.async_scenarios import run_scenario


@pytest.mark.parametrize("state", ["ready", "stopped", "failed"])
def test_engine_log_is_available_independently_of_engine_state(tmp_path: Path, state: str) -> None:
    async def scenario() -> None:
        (tmp_path / "current").write_text("2\n", encoding="ascii")
        runtime = tmp_path / "runtime" / "2"
        runtime.mkdir(parents=True)
        (runtime / "main.log").write_bytes(b"engine output\nlast failure\n\xff")
        (runtime / "other.log").write_text("other engine secret", encoding="utf-8")
        home = Home(installations=frozenset)
        home.target = OpenTarget(tmp_path, "main")
        home.link.state = state
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            response = await client.get("/supervisor/engine/log?engine=other&root=/")
            assert response.status == 200
            assert await response.json() == {
                "available": True,
                "text": "engine output\nlast failure\n\ufffd",
                "truncated": False,
            }
            denied = await client.get(
                "/supervisor/engine/log", headers={"Origin": "https://untrusted.example"}
            )
            assert denied.status == 403
            denied = await client.get(
                "/supervisor/engine/log", headers={"Host": "untrusted.example"}
            )
            assert denied.status == 421

    run_scenario(scenario())


@pytest.mark.parametrize("size", [0, 65535, 65536, 65537, 2 * 1024 * 1024])
def test_engine_log_returns_only_the_fixed_size_tail(tmp_path: Path, size: int) -> None:
    async def scenario() -> None:
        (tmp_path / "current").write_text("3\n", encoding="ascii")
        runtime = tmp_path / "runtime" / "3"
        runtime.mkdir(parents=True)
        output = b"first\n" + b"x" * (size - 11) + b"last\n" if size else b""
        (runtime / "main.log").write_bytes(output)
        home = Home(installations=frozenset)
        home.target = OpenTarget(tmp_path, "main")
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            response = await client.get("/supervisor/engine/log")
            assert response.status == 200
            assert await response.json() == {
                "available": True,
                "text": output[-65536:].decode("ascii"),
                "truncated": size > 65536,
            }

    run_scenario(scenario())


def test_engine_log_without_a_log_or_open_installation(tmp_path: Path) -> None:
    async def scenario() -> None:
        home = Home(installations=frozenset)
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            assert (await client.get("/supervisor/engine/log")).status == 409
            home.target = OpenTarget(tmp_path, "main")
            for has_snapshot in (False, True):
                if has_snapshot:
                    (tmp_path / "current").write_text("1\n", encoding="ascii")
                response = await client.get("/supervisor/engine/log")
                assert response.status == 200
                assert await response.json() == {
                    "available": False,
                    "text": "",
                    "truncated": False,
                }

    run_scenario(scenario())


def test_engine_failure_remains_in_public_log_after_status_polling(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "current").write_text("1\n", encoding="ascii")
        runtime = tmp_path / "runtime" / "1"
        runtime.mkdir(parents=True)
        log_path = runtime / "main.log"
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        with log_path.open("wb") as output:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "dinkster_supervisor.supervisor",
                "--port",
                str(port),
                "--instance=log-test",
                "--shutdown-token=log-test-token",
                "--",
                sys.executable,
                "-c",
                "import logging; "
                "print('engine startup diagnostic', flush=True); "
                "logging.warning('engine warning diagnostic'); "
                "logging.error('engine failure diagnostic'); raise SystemExit(3)",
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                async with aiohttp.ClientSession() as session:
                    async with asyncio.timeout(30):
                        while True:
                            try:
                                async with session.get(url + "/supervisor/status") as response:
                                    if (await response.json())["state"] == "failed":
                                        break
                            except aiohttp.ClientConnectionError:
                                pass
                            await asyncio.sleep(0.01)
                    for poll in range(1000):
                        async with session.get(url + f"/supervisor/status?poll={poll}") as response:
                            assert response.status == 200
                            assert (await response.json())["state"] == "failed"
                    async with session.get(
                        url + "/supervisor/status?denied=1", headers={"Host": "untrusted.example"}
                    ) as response:
                        assert response.status == 421
                        await response.read()
                    async with session.get(url + "/supervisor/missing") as response:
                        assert response.status == 404
                        await response.read()
                    async with session.post(
                        url + "/supervisor/shutdown",
                        headers={
                            "X-Dinkster-Instance": "log-test",
                            "X-Dinkster-Shutdown-Token": "log-test-token",
                        },
                    ) as response:
                        assert response.status == 200
                        await response.read()
                assert await asyncio.wait_for(process.wait(), timeout=10) == 0
            finally:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
        home = Home(installations=frozenset)
        home.target = OpenTarget(tmp_path, "main")
        async with TestClient(TestServer(create_home_app(home, frontend=None))) as client:
            response = await client.get("/supervisor/engine/log")
            assert response.status == 200
            tail = await response.json()
        assert tail["available"] is True
        assert tail["truncated"] is False
        text = tail["text"]
        assert "engine startup diagnostic" in text.splitlines()
        assert "WARNING:root:engine warning diagnostic" in text
        assert "ERROR:root:engine failure diagnostic" in text
        assert "exit code 3" in text
        assert "?poll=" not in text
        assert 'GET /supervisor/status?denied=1 HTTP/1.1" 421' in text
        assert 'GET /supervisor/missing HTTP/1.1" 404' in text

    run_scenario(scenario())
