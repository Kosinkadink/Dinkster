"""A compiler and declared child-owned pool for maintenance exclusion tests."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from dinkster_api.v1 import CompilerEmission, GraphCompilerDescriptor, InferenceContribution
from dinkster_compat_comfy import ResidentPool
from dinkster_memory import FullReleaseCommitResult, ReleaseCandidate

ROOT = Path(os.environ["DINKSTER_FULL_FREE_COMPILER_ROOT"])


class BlockingPool(ResidentPool):
    async def release_full(self, candidates: Sequence[ReleaseCandidate]) -> FullReleaseCommitResult:
        (ROOT / "commit-entered").touch()
        async with asyncio.timeout(10):
            while not (ROOT / "finish").exists():
                await asyncio.sleep(0.005)
        result = await super().release_full(candidates)
        (ROOT / "commit-finished").touch()
        return result


POOL = BlockingPool(cost_of=lambda _: {"ram": 0})


def consumers() -> dict[str, ResidentPool]:
    return {"pool": POOL}


def compile_graph(_view: object) -> CompilerEmission:
    (ROOT / "compiled").touch()
    assert (ROOT / "commit-finished").exists()
    POOL.rid_for(bytearray(16))
    return CompilerEmission()


def register() -> InferenceContribution:
    return InferenceContribution(
        graph_compilers=(GraphCompilerDescriptor("maintenance_compiler.noop", 0, compile_graph),)
    )
