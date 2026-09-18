"""Process launchers: how a worker child comes to exist (DESIGN 3.11).

IsolatedWorker owns *what* runs - the host command, the endpoint, the
manifest; a Launcher owns *how* the OS process is spawned. Sandboxing is a
launcher concern, not a boundary concern: framing, codec, and session never
know whether the child is a plain subprocess or jailed, and node authors
never see any of it (hazard H9).

``SubprocessLauncher`` is the portable rung of the ladder and the default
on every platform: dependency isolation and crash containment, no security
boundary claimed. Hardened launchers (``sandbox.BubblewrapLauncher``) take
the same LaunchSpec, which carries the launch as *data* - structured enough
that a sandbox can grant exactly the filesystem the child needs without
parsing argv.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

HOST_OWNED_ENVIRONMENT: frozenset[str] = frozenset(
    {"DINKSTER_EGRESS_PROXY", "DINKSTER_PACK_SCRATCH"}
)


@dataclass(frozen=True)
class LaunchSpec:
    """One child launch, as data.

    ``command`` is the complete host argv (interpreter first). ``env`` is
    what the child *needs beyond* the parent's environment - extra_env plus
    the endpoint's child_env (which may carry a boundary secret; secrets
    ride the environment, never argv - see transport.py). Launchers decide
    what else the child inherits: the plain subprocess passes the parent's
    environment through except for host-owned keys, while a sandbox starts
    from empty and allowlists. Host-owned keys only enter through ``env``.

    The structured fields exist for sandboxing launchers: the endpoint
    directory must be writable inside the jail (the unix socket lives
    there), the pack root readable, and /dev/shm shared only while the shm
    transport is actually in use.
    """

    command: tuple[str, ...]
    env: Mapping[str, str]
    endpoint_dir: Path
    pack_root: Path
    python: str
    use_shm: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", dict(self.env))


class Launcher(Protocol):
    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process: ...


class SubprocessLauncher:
    """The default: a plain child inheriting the parent's environment."""

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        environment = {
            name: value for name, value in os.environ.items() if name not in HOST_OWNED_ENVIRONMENT
        }
        environment.update(spec.env)
        return await asyncio.create_subprocess_exec(*spec.command, env=environment)
