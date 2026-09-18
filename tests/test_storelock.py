"""The store writer lock: one mutating process per content store.

What this proves, with real subprocesses: the lock excludes across
process boundaries, a held lock turns installer mutations into loud
timeouts instead of corruption, and two processes applying against one
shared store concurrently leave exactly one copy of the content and two
independently activated roots.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dinkster.installer import Installer, InstallError
from dinkster.storelock import StoreLockTimeout, hold_lock

MANIFEST_TEMPLATE = '[pack]\nname = "{name}"\n\n[pack.entry]\nnodes = "{name}_nodes:NODES"\n'

HOLDER = """
import sys
from pathlib import Path
from dinkster.storelock import hold_lock
with hold_lock(Path(sys.argv[1]), timeout=10):
    print("held", flush=True)
    sys.stdin.readline()  # hold until the parent closes stdin
"""

APPLIER = """
import sys
from pathlib import Path
from dinkster_registry import Lockfile
from dinkster.installer import Installer, lock_local_pack
root, shared, pack = (Path(arg) for arg in sys.argv[1:4])
installer = Installer(root, accelerator="cpu", shared_store=shared)
entry, _ = lock_local_pack(pack, installer.artifacts_dir)
installer.apply(Lockfile.of([entry]), venvs=False)
print("ok", flush=True)
"""


def write_pack(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dinkster-pack.toml").write_text(MANIFEST_TEMPLATE.format(name=name))
    (directory / f"{name}_nodes.py").write_text("NODES = []\n")
    return directory


def hold_in_subprocess(lock_path: Path) -> subprocess.Popen[str]:
    """A child process holding the lock; release by closing its stdin."""
    child = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "held"
    return child


def release(child: subprocess.Popen[str]) -> None:
    assert child.stdin is not None
    child.stdin.close()
    child.wait(timeout=10)


def test_lock_excludes_across_processes(tmp_path: Path) -> None:
    lock_path = tmp_path / ".lock"
    child = hold_in_subprocess(lock_path)
    try:
        with pytest.raises(StoreLockTimeout, match="held by another process"):
            with hold_lock(lock_path, timeout=0.3):
                pass
    finally:
        release(child)
    # released: acquisition succeeds immediately
    with hold_lock(lock_path, timeout=1.0):
        pass


def test_installer_mutation_times_out_loudly_while_lock_is_held(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    installer = Installer(
        tmp_path / "root", accelerator="cpu", shared_store=shared, lock_timeout=0.3
    )
    child = hold_in_subprocess(shared / ".lock")
    try:
        with pytest.raises(InstallError, match="store lock held"):
            installer.gc()
    finally:
        release(child)
    assert installer.gc() == ()  # lock released: gc runs (and finds nothing)


def test_concurrent_cross_process_applies_share_the_store(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    pack_dir = write_pack(tmp_path / "demo", "demo")
    children = [
        subprocess.Popen(
            [sys.executable, "-c", APPLIER, str(tmp_path / name), str(shared), str(pack_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        for name in ("root-a", "root-b")
    ]
    for child in children:
        out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err
        assert out.strip() == "ok"
    # exactly one copy of the content in the shared store
    store_dirs = [child for child in (shared / "store").iterdir() if child.is_dir()]
    assert len(store_dirs) == 1
    archives = list((shared / "artifacts").glob("*.zip"))
    assert len(archives) == 1
    assert not list((shared / "artifacts").glob("*.tmp"))  # no staging leftovers
    # both roots activated their own generation 1
    for name in ("root-a", "root-b"):
        root = tmp_path / name
        assert (root / "current").read_text().strip() == "1"
        assert (root / "generations" / "1.json").is_file()
