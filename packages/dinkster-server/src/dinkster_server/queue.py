"""The job queue: policy above the already-concurrent engine (DESIGN 3.10).

The engine decides HOW things run (parallel scheduling, admission,
coalescing); the queue decides WHAT runs and WHEN: admission control,
priorities, cancellation, and job identity. Jobs are identified as
(clientId, jobId) - the frontend already treats that pair as the execution
identity, so multi-client/multi-backend is first-class from day one.

max_running_jobs bounds how many jobs execute concurrently (default 1: one
workflow at a time, the familiar posture; raise it deliberately - the
engine's admission lanes keep hardware safe either way).

Run control: pause() stops dispatching (running jobs finish; the queue only
holds new ones back), resume() releases it, clear() cancels what is queued -
all or one client's. Terminal jobs stay queryable for a bounded while
(history_limit) so a client that missed the job_state event can still poll
the result; the oldest terminal jobs are forgotten first.

Durability: the queue itself is in-memory, but an optional QueuePersistence
store records every accepted job before the submission is acknowledged and
retires the record when the job reaches a terminal state. A process that
dies mid-queue leaves those records behind for the store's owner to surface
as interrupted history - nothing is ever re-executed automatically.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import secrets
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from dinkster_engine import (
    CompiledGraph,
    Engine,
    ExecutionError,
    ExecutionRuntime,
    GraphCompileError,
    GraphValidationError,
    RunResult,
)
from dinkster_graph import Graph, graph_to_wire, lower_selectors
from dinkster_protocol import (
    AttentionPolicyConfig,
    ExportSnapshot,
    PreviewPolicy,
    attention_policy_config_to_wire,
)

from .redaction import PathRedactor

JobState = Literal["queued", "running", "completed", "failed", "cancelled"]

TERMINAL_STATES: frozenset[JobState] = frozenset({"completed", "failed", "cancelled"})
log = logging.getLogger("dinkster.server.queue")


def new_job_ref() -> str:
    """Mint a server-assigned, globally unique job reference (the platform
    plan's jobRef): 12 hex chars of millisecond timestamp + 80 random bits.
    Opaque to clients; lexical order approximates submission order. One
    identity, two wire names: this is the value historically exposed as
    runId, and jobRef is its canonical public name going forward."""
    return f"{int(time.time() * 1000):012x}{secrets.token_hex(10)}"


@dataclass(frozen=True)
class JobKey:
    client_id: str
    job_id: str

    def __str__(self) -> str:
        return f"{self.client_id}/{self.job_id}"


@dataclass
class Job:
    key: JobKey
    graph: Graph
    targets: tuple[str, ...]
    priority: int  # higher runs first; FIFO within a priority
    run_id: str
    fingerprint: str
    """Canonical submission-content digest. The (clientId, jobId) pair is
    an active-window idempotency key: matching content resolves to this job,
    while different content conflicts."""
    compiled_graph: CompiledGraph | None = field(default=None, kw_only=True)
    """Immutable admission-compiled artifact, separate from submitted graph."""
    scope: str = "local"
    principal_id: str = "local"
    principal_kind: str = "human"
    source_document: str = ""
    """Optional canonical asset digest ("blake3:<hex>") of the workflow
    document this job was compiled from - an execution-OPAQUE provenance
    record (contract agreed with the frontend 2026-07). Never part of
    graph execution or cache identity; it rides the job wire and history
    so any run can answer "what exact document produced this". Empty
    means unset (omitted on the wire, never null)."""
    export_snapshot: ExportSnapshot | None = None
    """Opaque compat export payload carried only to output-node invocations."""
    execution: ExecutionRuntime | None = None
    """Immutable extension/worker/schema generation pinned at admission."""
    preview_policy: PreviewPolicy | None = None
    """Effective sampling-preview policy for this job's run, resolved at
    submission from the global default and any workflow/node overrides.
    None means previews stay off."""
    attention_config: AttentionPolicyConfig | None = None
    """Effective job-scoped attention routing policy. None preserves the
    legacy automatic route for non-HTTP callers."""
    cache_enabled: bool = True
    """Whether native result reuse and single-flight coalescing apply."""
    attempt: int = 1
    """Engine-side execution attempt for this jobRef. Retry plumbing is not
    implemented yet, so every current job has exactly one attempt."""
    latest_seq: int = 0
    """Server event-publication watermark for this jobRef."""
    state: JobState = "queued"
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: RunResult | None = None
    error: dict[str, Any] | None = None  # heterogeneous JSON error payload
    node_receipts: dict[str, dict[str, str]] = field(default_factory=lambda: {})
    task: asyncio.Task[None] | None = None


JobListener = Callable[[Job], None]
"""Called on every job state transition (queued/running/terminal)."""


class QueuePersistence(Protocol):
    """Durable record of accepted work (HistoryStore implements this).

    All three methods are synchronous and called inline on the event loop,
    deliberately: job_accepted must land before the submission is
    acknowledged (a raise refuses the job), and job_terminal must atomically
    move the accepted record into terminal history - orderings a
    fire-and-forget thread write cannot provide. The writes are tiny
    WAL-backed rows at human submission rates."""

    def job_accepted(self, job: Job) -> None: ...

    def job_started(self, job: Job) -> None: ...

    def job_terminal(self, job: Job) -> None: ...


class JobGraphAdmissionError(ValueError):
    """The submitted graph is not already in an execution-safe lowered form."""


class JobQueue:
    def __init__(
        self,
        engine: Engine,
        *,
        max_running_jobs: int = 1,
        on_job_event: JobListener | None = None,
        history_limit: int = 256,
        debug_errors: bool = False,
        redactor: PathRedactor | None = None,
        store: QueuePersistence | None = None,
    ) -> None:
        if max_running_jobs < 1:
            raise ValueError("max_running_jobs must be >= 1")
        if history_limit < 1:
            raise ValueError("history_limit must be >= 1")
        self._engine = engine
        self._max_running = max_running_jobs
        self._on_job_event = on_job_event
        self._history_limit = history_limit
        # debug_errors=True is the local-debugging escape hatch: error
        # payloads (message, hints, traceback) go out raw. The default sends
        # them path-redacted; the server terminal always logs the raw form.
        self._debug_errors = debug_errors
        self._redactor = redactor if redactor is not None else PathRedactor()
        self._store = store
        self._jobs: dict[JobKey, Job] = {}
        self._pending: list[Job] = []  # sorted by (-priority, seq) at dispatch
        self._seq = itertools.count()
        self._order: dict[JobKey, int] = {}
        self._by_run: dict[str, JobKey] = {}
        self._history: deque[Job] = deque()  # terminal jobs, oldest first
        self._running: set[JobKey] = set()
        self._paused = False
        self._maintenance_active = False
        self._wakeup = asyncio.Event()
        self._dispatcher: asyncio.Task[None] | None = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._dispatcher is None:
            self._dispatcher = asyncio.get_running_loop().create_task(self._dispatch_loop())

    async def close(self) -> None:
        """Stop dispatching, cancel everything in flight, and drain."""
        self._closed = True
        self._wakeup.set()
        for job in list(self._pending):
            self._finish(job, "cancelled", error={"message": "server shutting down"})
        self._pending.clear()
        running = [self._jobs[key] for key in list(self._running)]
        for job in running:
            if job.task is not None:
                job.task.cancel()
        for job in running:
            if job.task is not None:
                try:
                    await job.task
                except asyncio.CancelledError:
                    pass
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            try:
                await self._dispatcher
            except asyncio.CancelledError:
                pass
            self._dispatcher = None

    # -- public API ---------------------------------------------------------

    def submit(
        self,
        client_id: str,
        job_id: str,
        graph: Graph,
        targets: Sequence[str],
        *,
        priority: int = 0,
        scope: str = "local",
        principal_id: str = "local",
        principal_kind: str = "human",
        source_document: str = "",
        fingerprint: str = "",
        execution: ExecutionRuntime | None = None,
        compiled_graph: CompiledGraph | None = None,
        export_snapshot: ExportSnapshot | None = None,
        previews: PreviewPolicy | None = None,
        attention_config: AttentionPolicyConfig | None = None,
        cache_enabled: bool = True,
    ) -> Job:
        """Queue a lowered graph under an optional pre-pinned execution."""
        if self._closed:
            raise RuntimeError("queue is closed")
        execution = execution or self._engine.pin_execution()
        assert execution.schemas is not None
        schemas = execution.schemas
        lowered = lower_selectors(graph, targets, schemas)
        if lowered.problems:
            problem = lowered.problems[0]
            raise JobGraphAdmissionError(
                f"selector admission failed: {problem.code} at node "
                f"{problem.node_id}: {problem.message}"
            )
        if lowered.graph != graph or lowered.targets != tuple(targets):
            raise JobGraphAdmissionError(
                "selector admission failed: stored selector residue would rewrite "
                "the submitted graph or targets"
            )
        if attention_config is not None and type(attention_config) is not AttentionPolicyConfig:
            raise TypeError("attention_config must be an AttentionPolicyConfig")
        if type(cache_enabled) is not bool:
            raise TypeError("cache_enabled must be a boolean")
        export_snapshot = deepcopy(export_snapshot)
        if not fingerprint:
            fingerprint_payload: dict[str, object] = {
                "graph": graph_to_wire(graph),
                "targets": list(targets),
                "priority": priority,
                "sourceDocument": source_document,
            }
            if export_snapshot is not None:
                fingerprint_payload["exportSnapshot"] = {
                    "prompt": export_snapshot.prompt,
                    "extraPnginfo": export_snapshot.extra_pnginfo,
                }
            if previews is not None:
                fingerprint_payload["previews"] = {
                    "mode": previews.mode,
                    "nodes": dict(previews.node_modes),
                }
            if attention_config is not None and attention_config != AttentionPolicyConfig():
                fingerprint_payload["attention"] = attention_policy_config_to_wire(attention_config)
            if not cache_enabled:
                fingerprint_payload["cacheEnabled"] = False
            fingerprint = hashlib.sha256(
                json.dumps(
                    fingerprint_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        key = JobKey(client_id, job_id)
        existing = self._jobs.get(key)
        if existing is not None and existing.state not in TERMINAL_STATES:
            if existing.fingerprint == fingerprint:
                return existing
            raise ValueError(
                f"job key {key} is in use by {existing.state} job with different content"
            )
        job = Job(
            key=key,
            graph=graph,
            targets=tuple(targets),
            priority=priority,
            run_id=new_job_ref(),
            fingerprint=fingerprint,
            compiled_graph=compiled_graph,
            scope=scope,
            principal_id=principal_id,
            principal_kind=principal_kind,
            source_document=source_document,
            export_snapshot=export_snapshot,
            execution=execution,
            preview_policy=previews,
            attention_config=attention_config,
            cache_enabled=cache_enabled,
        )
        if self._store is not None:
            # Durability before acknowledgment: if the accepted-job record
            # cannot be written, the raise propagates, the submission is
            # refused, and the queue is untouched - no in-memory state for
            # a job the disk never heard of.
            self._store.job_accepted(job)
        if existing is not None:
            # Resubmitting a key retires its terminal predecessor completely:
            # its run must not resolve to the new job, and its history slot
            # must not evict the new job later.
            self._by_run.pop(existing.run_id, None)
            try:
                self._history.remove(existing)
            except ValueError:
                pass
        self._jobs[key] = job
        self._order[key] = next(self._seq)
        self._by_run[job.run_id] = key
        self._pending.append(job)
        self._emit(job)
        self._wakeup.set()
        return job

    def get(self, client_id: str, job_id: str) -> Job | None:
        return self._jobs.get(JobKey(client_id, job_id))

    def job_for_run(self, run_id: str) -> Job | None:
        """Correlate an engine event's run_id back to its job."""
        key = self._by_run.get(run_id)
        return self._jobs.get(key) if key is not None else None

    def jobs(self, client_id: str | None = None) -> list[Job]:
        """All known jobs (bounded by history_limit for terminal ones) in
        submission order; client_id narrows to one client's."""
        jobs = sorted(self._jobs.values(), key=lambda j: self._order[j.key])
        if client_id is None:
            return jobs
        return [job for job in jobs if job.key.client_id == client_id]

    def pending(self) -> list[Job]:
        """Queued jobs in dispatch order (priority, then submission)."""
        return sorted(self._pending, key=lambda j: (-j.priority, self._order[j.key]))

    def running(self) -> list[Job]:
        """Running jobs in submission order."""
        return sorted(
            (self._jobs[key] for key in self._running),
            key=lambda j: self._order[j.key],
        )

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Stop dispatching. Running jobs finish; queued jobs wait."""
        self._paused = True

    def resume(self) -> None:
        if self._maintenance_active:
            raise RuntimeError("queue maintenance is active")
        self._paused = False
        self._wakeup.set()

    def begin_maintenance(self) -> str | None:
        """Freeze dispatch while a paused and idle queue is maintained."""
        if self._maintenance_active:
            return "queue maintenance is already active"
        if not self._paused:
            return "queue must be paused before memory maintenance"
        if self._running:
            return "queue must be idle before memory maintenance"
        self._maintenance_active = True
        return None

    def end_maintenance(self) -> None:
        self._maintenance_active = False

    def set_max_running(self, max_running_jobs: int) -> None:
        """Replace the concurrent-job limit and promptly fill new slots."""
        if max_running_jobs < 1:
            raise ValueError("max_running_jobs must be >= 1")
        raised = max_running_jobs > self._max_running
        self._max_running = max_running_jobs
        if raised:
            self._wakeup.set()

    def clear(self, client_id: str | None = None) -> list[Job]:
        """Cancel every queued job (or one client's). Running jobs are not
        touched - cancel them individually; clearing a queue should not
        yank work already on the hardware."""
        victims = [
            job for job in self.pending() if client_id is None or job.key.client_id == client_id
        ]
        for job in victims:
            self._pending.remove(job)
            self._finish(job, "cancelled")
        return victims

    def cancel(self, client_id: str, job_id: str) -> Job | None:
        """Cancel a queued or running job; terminal jobs are left alone."""
        job = self._jobs.get(JobKey(client_id, job_id))
        if job is None or job.state in TERMINAL_STATES:
            return job
        if job.state == "queued":
            self._pending.remove(job)
            self._finish(job, "cancelled")
        elif job.task is not None:
            job.task.cancel()  # _run_job's handler marks it cancelled
        return job

    def status(self) -> dict[str, object]:
        return {
            "queued": len(self._pending),
            "running": sorted(str(key) for key in self._running),
            "maxRunningJobs": self._max_running,
            "paused": self._paused,
        }

    # -- internals -----------------------------------------------------------

    def _emit(self, job: Job) -> None:
        if self._on_job_event is not None:
            self._on_job_event(job)

    def _clean(self, text: str) -> str:
        """Error-payload text bound for the wire: raw only under the
        debug_errors escape hatch, path-redacted otherwise."""
        return text if self._debug_errors else self._redactor.redact_text(text)

    def _finish(self, job: Job, state: JobState, *, error: dict[str, Any] | None = None) -> None:
        job.state = state
        job.error = error
        job.finished_at = time.time()
        if self._store is not None:
            try:
                self._store.job_terminal(job)
            except Exception:  # noqa: BLE001 - a failed write must not kill the server
                # The accepted record lingers, so the next restart
                # conservatively surfaces this job as interrupted.
                log.exception("durable terminal write failed for job %s", job.key)
        self._history.append(job)
        while len(self._history) > self._history_limit:
            evicted = self._history.popleft()
            # Guard on identity: a resubmitted key replaces its terminal
            # predecessor in _jobs, and the stale history entry must not
            # take the live job with it.
            if self._jobs.get(evicted.key) is evicted:
                del self._jobs[evicted.key]
                del self._order[evicted.key]
            self._by_run.pop(evicted.run_id, None)
        self._emit(job)

    async def _dispatch_loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            if self._closed:
                return
            while (
                not self._paused
                and not self._maintenance_active
                and self._pending
                and len(self._running) < self._max_running
            ):
                self._pending.sort(key=lambda j: (-j.priority, self._order[j.key]))
                job = self._pending.pop(0)
                job.state = "running"
                job.started_at = time.time()
                if self._store is not None:
                    try:
                        self._store.job_started(job)
                    except Exception:  # noqa: BLE001 - must not kill the dispatcher
                        log.exception("durable job-started write failed for job %s", job.key)
                self._running.add(job.key)
                job.task = asyncio.get_running_loop().create_task(self._run_job(job))
                self._emit(job)

    async def _run_job(self, job: Job) -> None:
        try:
            if job.compiled_graph is not None:
                result = await self._engine.run_compiled(
                    job.compiled_graph,
                    run_id=job.run_id,
                    attempt_id=job.attempt,
                    execution=job.execution,
                    export_snapshot=job.export_snapshot,
                    preview_policy=job.preview_policy,
                    attention_config=job.attention_config,
                    cache_enabled=job.cache_enabled,
                )
            else:
                result = await self._engine.run(
                    job.graph,
                    job.targets,
                    run_id=job.run_id,
                    attempt_id=job.attempt,
                    execution=job.execution,
                    export_snapshot=job.export_snapshot,
                    preview_policy=job.preview_policy,
                    attention_config=job.attention_config,
                    cache_enabled=job.cache_enabled,
                )
            job.result = result
            self._finish(job, "completed")
        except asyncio.CancelledError:
            self._finish(job, "cancelled")
            raise
        except GraphCompileError as exc:
            self._finish(
                job,
                "failed",
                error={"kind": "compile", "code": exc.code, "message": self._clean(str(exc))},
            )
        except GraphValidationError as exc:
            self._finish(
                job,
                "failed",
                error={
                    "kind": "validation",
                    "diagnostics": [
                        {
                            "severity": d.severity,
                            "code": d.code,
                            "nodeId": d.node_id,
                            "inputId": d.input_id,
                            "message": self._clean(d.message),
                        }
                        for d in exc.diagnostics
                    ],
                },
            )
        except ExecutionError as exc:
            log.exception("job %s failed with ExecutionError", job.key)
            if exc.error.traceback:
                # The node's own traceback (remote frames when the node ran
                # in an isolated worker) is not part of the local raise that
                # log.exception prints; keep the raw form in the terminal.
                log.error("job %s node traceback:\n%s", job.key, exc.error.traceback)
            error: dict[str, Any] = {
                "kind": "execution",
                "type": type(exc).__name__,
                "nodeId": exc.error.node_id,
                "nodeType": exc.error.node_type,
                "message": self._clean(exc.error.message),
                # The traceback always rides along so failures stay
                # debuggable; redaction (not omission) is the privacy
                # boundary, and the terminal log above keeps the raw form.
                "traceback": self._clean(exc.error.traceback),
            }
            if exc.error.hints:
                error["hints"] = [
                    {
                        "code": hint.code,
                        "message": self._clean(hint.message),
                        **(
                            {"suggestion": self._clean(hint.suggestion)}
                            if hint.suggestion is not None
                            else {}
                        ),
                    }
                    for hint in exc.error.hints
                ]
            self._finish(
                job,
                "failed",
                error=error,
            )
        except Exception as exc:  # noqa: BLE001 - a job must never kill the server
            log.exception("job %s failed with %s", job.key, type(exc).__name__)
            error = {
                "kind": "internal",
                "type": type(exc).__name__,
                "message": self._clean(str(exc)),
                "traceback": self._clean(traceback.format_exc()),
            }
            self._finish(job, "failed", error=error)
        finally:
            self._running.discard(job.key)
            self._wakeup.set()
