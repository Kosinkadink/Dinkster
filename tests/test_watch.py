"""Dev pack source watcher (DESIGN 3.9): sugar over hot reload.

What this proves: the settle cycle fires exactly one reload per change
burst (and only after the tree stops churning), first sighting baselines
without firing, editor droppings and bytecode never trigger, a failed
reload does not re-fire until the files change again, and the whole loop
drives the REAL composer end to end - file change on disk to new code
executing - with no sleeps anywhere (tests drive poll_once directly).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from dinkster_registry import LockedPack, Lockfile, build_artifact
from test_reload import run_echo, worker_env, write_pack

from dinkster.compose import ServingComposer
from dinkster.watch import PackWatcher
from tests.async_scenarios import run_scenario


def bump_mtime(path: Path, ns: int = 1_000_000_000) -> None:
    """Advance a file's mtime deterministically: content-size-neutral
    edits must still register on filesystems with coarse timestamps."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + ns))


class Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    async def __call__(self, name: str) -> None:
        self.calls.append(name)
        if self.fail:
            raise RuntimeError("boom")


def make_pack_dir(root: Path, name: str = "p") -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "mod.py").write_text("x = 1\n")
    return directory


# -- snapshot ------------------------------------------------------------------


# -- settle cycle --------------------------------------------------------------


def test_settle_cycle_fires_once(tmp_path: Path) -> None:
    """First sighting baselines silently; a change marks pending on one
    scan and fires on the next quiet one; a quiet tree never fires."""
    pack = make_pack_dir(tmp_path)
    reload_fn = Recorder()
    watcher = PackWatcher(lambda: {"p": pack}, reload_fn)

    async def scenario() -> None:
        assert await watcher.poll_once() == []  # baseline, no fire
        assert await watcher.poll_once() == []  # quiet, no fire
        bump_mtime(pack / "mod.py")
        assert await watcher.poll_once() == []  # diff seen: pending
        assert reload_fn.calls == []
        assert await watcher.poll_once() == ["p"]  # settled: fires
        assert reload_fn.calls == ["p"]
        assert await watcher.poll_once() == []  # once, not again

    run_scenario(scenario())


def test_churn_stays_pending_until_quiet(tmp_path: Path) -> None:
    """A burst of writes (editor save storms, multi-file changes)
    coalesces: no reload while the tree keeps changing, exactly one when
    it stops."""
    pack = make_pack_dir(tmp_path)
    reload_fn = Recorder()
    watcher = PackWatcher(lambda: {"p": pack}, reload_fn)

    async def scenario() -> None:
        await watcher.poll_once()
        bump_mtime(pack / "mod.py")
        assert await watcher.poll_once() == []
        (pack / "extra.py").write_text("y = 2\n")  # still churning
        assert await watcher.poll_once() == []
        bump_mtime(pack / "extra.py")  # and churning
        assert await watcher.poll_once() == []
        assert reload_fn.calls == []
        assert await watcher.poll_once() == ["p"]  # quiet at last
        assert reload_fn.calls == ["p"]

    run_scenario(scenario())


def test_ignored_churn_never_fires(tmp_path: Path) -> None:
    """__pycache__ writes (a reload itself produces them) and editor
    droppings cause no pending state at all."""
    pack = make_pack_dir(tmp_path)
    reload_fn = Recorder()
    watcher = PackWatcher(lambda: {"p": pack}, reload_fn)

    async def scenario() -> None:
        await watcher.poll_once()
        cache = pack / "__pycache__"
        cache.mkdir()
        (cache / "mod.cpython-312.pyc").write_text("bc")
        (pack / "mod.py.swp").write_text("swap")
        assert await watcher.poll_once() == []
        assert await watcher.poll_once() == []
        assert reload_fn.calls == []

    run_scenario(scenario())


def test_failed_reload_does_not_refire(tmp_path: Path) -> None:
    """The endpoint's failure contract holds here: the error is logged,
    the watcher keeps running, and the SAME pending change never retries
    - only a new change on disk fires again."""
    pack = make_pack_dir(tmp_path)
    reload_fn = Recorder(fail=True)
    watcher = PackWatcher(lambda: {"p": pack}, reload_fn)

    async def scenario() -> None:
        await watcher.poll_once()
        bump_mtime(pack / "mod.py")
        await watcher.poll_once()
        assert await watcher.poll_once() == []  # attempted, failed
        assert reload_fn.calls == ["p"]
        assert await watcher.poll_once() == []  # no retry loop
        assert reload_fn.calls == ["p"]
        bump_mtime(pack / "mod.py")  # a NEW change fires again
        await watcher.poll_once()
        await watcher.poll_once()
        assert reload_fn.calls == ["p", "p"]

    run_scenario(scenario())


# -- end to end ----------------------------------------------------------------


@pytest.mark.parametrize("generation_source", [False, True])
def test_watcher_reloads_composed_pack_end_to_end(tmp_path: Path, generation_source: bool) -> None:
    """The real thing: rewrite a composed pack on disk, drive the poll
    loop, and the live engine executes the NEW code - watch_targets from
    the composer's records, the settle cycle, and reload_pack all in one
    path, no HTTP and no sleeps."""

    async def scenario() -> None:
        write_pack(tmp_path, "v1")
        selected = tmp_path
        if generation_source:
            from dinkster.generation_packs import generation_pack_specs, stage_generation_packs

            generation = tmp_path.parent / "generation"
            generation.mkdir()
            archive = tmp_path.parent / "pack.zip"
            identity = build_artifact(tmp_path, archive)
            entry = LockedPack(
                "rlpack", "1.0", identity, "dinkster", ("rl",), "local-build:" + str(tmp_path)
            )
            stage_generation_packs(generation, Lockfile.of([entry]), {"rlpack": archive})
            (generation / "development-sources.json").write_text(
                json.dumps(
                    {
                        "format": "dinkster.development-sources/1",
                        "sources": [{"kind": "pack", "path": str(tmp_path), "live": True}],
                    }
                )
            )
            subprocess.run(
                [
                    "uv",
                    "venv",
                    "--python",
                    sys.executable,
                    "--system-site-packages",
                    str(generation / "packs/rlpack"),
                ],
                check=True,
                capture_output=True,
            )
            python = (
                generation
                / "packs/rlpack"
                / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            )
            site = Path(
                subprocess.check_output(
                    [
                        str(python),
                        "-I",
                        "-c",
                        "import sysconfig; print(sysconfig.get_path('purelib'))",
                    ],
                    text=True,
                ).strip()
            )
            (site / "fixture.pth").write_text(
                "\n".join(str(Path(path).resolve()) for path in sys.path if path) + "\n"
            )
            (selected,) = generation_pack_specs(generation)
        composer = ServingComposer(worker_env=worker_env(tmp_path))
        try:
            targets = composer.watch_targets()
            assert targets == {}  # nothing composed yet
            await composer.add_pack(selected)
            assert composer.watch_targets() == {"rlpack": tmp_path}
            engine = composer.composition.make_engine(lambda event: None)
            assert await run_echo(engine) == "hi-v1"

            async def do_reload(name: str) -> None:
                result = await composer.reload_pack(name)
                engine.replace_schemas(result.removed_types, result.delta.schemas)

            watcher = PackWatcher(composer.watch_targets, do_reload)
            assert await watcher.poll_once() == []  # baseline
            write_pack(tmp_path, "v2", extra_type="rl.extra")
            assert await watcher.poll_once() == []  # settling
            assert await watcher.poll_once() == ["rlpack"]
            # Fresh input (the raw composer does not clear the result
            # cache; apply_reload does): fresh process, new code.
            assert await run_echo(engine, "two") == "two-v2"
            assert "rl.extra" in composer.composition.schemas
            assert await watcher.poll_once() == []  # quiet again
        finally:
            await composer.close()

    run_scenario(scenario())
