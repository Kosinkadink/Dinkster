"""First-class absence: a typed "no value here" with provenance.

The anti-ExecutionBlocker (DESIGN 3.15). ComfyUI's ExecutionBlocker collapsed
four distinct states - not demanded, value absent, node skipped, producer
error - into one sentinel that travels through data edges, list machinery,
and caches, with diagnostics attributed to the first downstream consumer.
Dinkster keeps them separate; this module owns exactly one of them: **absent**,
an ordinary envelope whose type is ``core.absent``.

Properties that make it a value and not a sentinel:

- It is interrogable anywhere any envelope is: origin (which node/output
  decided there is no value), reason, and the type it stands in for all live
  in meta - so diagnostics always name the producer, never a downstream
  victim.
- Node code never sees it unless the input *declares* it wants to
  (``on_absent="accept"``, arriving as plain ``None``). The engine applies
  input policies before invocation; propagation is an engine decision
  recorded in the run report, not a value smuggled through node returns.
- It cannot hide inside collections: workers wrap list outputs elementwise
  by declared element type, so a "list with a hole" is unrepresentable
  through any normal path.
- It crosses boundaries and caches like any value (registered core type),
  so a deliberate absence (a loader with no VAE) caches and replays exactly
  like the value it stands in for.
"""

from __future__ import annotations

from .model import PyObjPayload, Value, ValueMeta, stable_hash

CORE_ABSENT = "core.absent"
"""Type id of absent envelopes. Registered as a core type; payload is None."""

ABSENT_ORIGIN_META_KEY = "absent.origin"
"""Meta key: ``node_id/output_id`` of the producer that decided no value
exists. Propagated unchanged through engine-level skips, so the root cause
is always one hop away no matter how far the absence cascaded."""

ABSENT_REASON_META_KEY = "absent.reason"
"""Meta key: the producer's human-readable reason (may be empty)."""

ABSENT_STANDS_FOR_META_KEY = "absent.standsFor"
"""Meta key: the runtime type id this absence stands in for, when the
declared output type resolves to one (empty string otherwise)."""


def make_absent_value(*, origin: str, reason: str = "", stands_for: str | None = None) -> Value:
    """Build an absent envelope. Worker shims and the engine call this; node
    authors return the ABSENT marker (dinkster-schema) and never construct
    envelopes (hazard H9)."""
    stands = stands_for or ""
    fingerprint = stable_hash(
        [
            CORE_ABSENT.encode("utf-8"),
            origin.encode("utf-8"),
            reason.encode("utf-8"),
            stands.encode("utf-8"),
        ]
    )
    meta = ValueMeta(
        {
            ABSENT_ORIGIN_META_KEY: origin,
            ABSENT_REASON_META_KEY: reason,
            ABSENT_STANDS_FOR_META_KEY: stands,
        }
    )
    return Value(
        type_id=CORE_ABSENT,
        fingerprint=fingerprint,
        meta=meta,
        payload=PyObjPayload(None),
    )


def is_absent(value: Value) -> bool:
    return value.type_id == CORE_ABSENT
