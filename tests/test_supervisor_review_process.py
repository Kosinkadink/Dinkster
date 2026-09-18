from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

from dinkster_supervisor import EngineLink, EngineProcess


def _write_engine(path: Path, *, answer_health: bool) -> None:
    response = (
        "conn.recv(65536); conn.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Type: "
        'application/json\\r\\nContent-Length: 12\\r\\n\\r\\n{"ok": true}\')'
        if answer_health
        else "conn.recv(65536); time.sleep(60)"
    )
    path.write_text(
        textwrap.dedent(
            f"""
            import argparse, socket, time
            parser = argparse.ArgumentParser()
            parser.add_argument('--host')
            parser.add_argument('--port', type=int)
            args = parser.parse_args()
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((args.host, args.port))
            sock.listen()
            while True:
                conn, _ = sock.accept()
                with conn:
                    {response}
            """
        )
    )


def test_startup_timeout_reaps_stalled_health_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        script = tmp_path / "stalled.py"
        _write_engine(script, answer_health=False)
        link = EngineLink()
        engine = EngineProcess(
            [sys.executable, str(script)], link, probe_interval=0.01, startup_timeout=0.05
        )
        await engine.start()
        process = engine._process
        watch_task = engine._watch_task
        assert process is not None
        assert watch_task is not None
        await asyncio.wait_for(watch_task, timeout=7)
        assert process.returncode is not None
        assert link.state == "failed"
        assert link.exit_code == process.returncode
        assert "did not become healthy" in link.detail
        await engine.close()

    asyncio.run(scenario())


def test_overlapping_restarts_leave_only_latest_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        script = tmp_path / "healthy.py"
        _write_engine(script, answer_health=True)
        link = EngineLink(state="stopped")
        engine = EngineProcess([sys.executable, str(script)], link, probe_interval=0.01)
        await engine.start()
        first_process = engine._process
        first_pid = link.pid
        await asyncio.gather(engine.restart(), engine.restart())
        final_process = engine._process
        assert final_process is not None
        assert final_process.returncode is None
        assert link.pid == final_process.pid
        assert link.pid != first_pid
        assert first_process is not None
        assert first_pid is not None
        assert first_process.returncode is not None
        await engine.close()

    asyncio.run(scenario())
