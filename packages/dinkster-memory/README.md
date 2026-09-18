# dinkster-memory

`dinkster-memory` owns Dinkster's process-usable system-memory view and its per-residency
memory governor. `system_memory_snapshot()` uses psutil for portable host total
and available memory. On Linux it resolves the process hierarchy through
`/proc/self/cgroup` and `/proc/self/mountinfo`, then clamps capacity and headroom
to finite cgroup-v1 or cgroup-v2 ancestor limits, including hybrid hierarchies.
RAM and swap remain separate; v1 `memsw` values are decomposed into swap-only
capacity before clamping host swap. Swap sampling is explicit so RAM-only
residency and pressure checks do not pay for extra mutable-file reads; benchmark
telemetry requests both. Immutable mount topology is cached while mutable cgroup
paths, limits, and usage are read for every snapshot. Residency, pinning,
benchmarks, workers, caches, and server memory endpoints share these contracts.

## Setup

This package is a uv workspace member. From the repository root, install the
whole workspace with:

```sh
uv sync --all-packages
```

The package is not published separately yet and provides no console script.

## Use

The public surface includes `SystemMemorySnapshot`, `MemoryGovernor`, `Shedder`,
`PressureSignal`, reservations and reservation services, `LeaseBroker`,
two-phase release contracts, and item-level consumer detail types. Reserve bytes
before an allocation and release them automatically when the context exits:

```python
from dinkster_memory import MemoryGovernor


async def allocate() -> None:
    governor = MemoryGovernor({"vram:cuda:0": 8_000_000_000})
    async with governor.reserve("vram:cuda:0", 2_000_000_000) as reservation:
        assert reservation.nbytes == 2_000_000_000
        # Perform the governed allocation here.

    assert governor.reserved("vram:cuda:0") == 0
```

Consumers implement `Shedder` and register with
`governor.register_shedder(...)`. Lower priorities shed first, allowing caches
to yield memory before more expensive resident resources. RAM reservations with
a nonzero sheddable footprint read one fresh effective-memory snapshot and ask
consumers for the immediate shortfall before allocation. This is pressure
handling, not a default RAM cap: an unbudgeted request still grants when
consumers cannot free enough or the memory query is unavailable.

## Learn more

See DESIGN 3.10 for parallel execution and memory governance, including
reservation, consumer detail, leases, pins, and two-phase release. See hazards
H2, H9, and H13 in `docs/hazards.md`.

Focused tests include `tests/test_system_memory.py`, `tests/test_memory.py`,
`tests/test_reservations.py`, `tests/test_leases.py`, `tests/test_device_map.py`, and
`tests/test_compat_pool.py`.
