"""Torch-free binding of the windowed-evaluation plan into the manifest.

The windowed-evaluation manifest slot binds the window-set derivation
(identity and canonical facts digest), the ordered vector of every
composite plan the invocation can realize, each plan's composite and
ordered per-layer digests, and per joint window the ``ModelTokenLayout``
digest, partition-plan digest, and structural-row accounting digest
contributed by their owning subsystems. Facts already inside a
composite digest's canonical preimage - the realized window schedule,
wrap topology, index-map profile identities and parameters, weights and
traversal, merge declaration, and lifted passthrough coordinates - are
bound transitively through that digest and are not repeated here.

The per-window digest vectors are indexed by joint-window position
within their plan; each composite digest fixes which joint window
holds each position, so the pairing is unambiguous without repeating
the window schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .manifest import CompiledPlanSlot
from .window_plan import CompositeWindowPlan

__all__ = [
    "WINDOWED_EVALUATION_SLOT",
    "WindowPlanBinding",
    "WindowSlotRefusal",
    "WindowSlotRefusalCode",
    "build_windowed_evaluation_slot",
]

WINDOWED_EVALUATION_SLOT = "windowed-evaluation"


class WindowSlotRefusalCode(StrEnum):
    INVALID_WINDOW_PLAN = "invalid-window-plan"
    WINDOW_BINDING_MISMATCH = "window-binding-mismatch"
    INVALID_DERIVATION_BINDING = "invalid-derivation-binding"


class WindowSlotRefusal(ValueError):
    def __init__(self, code: WindowSlotRefusalCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


def _is_sha256_hex(digest: object) -> bool:
    return (
        type(digest) is str
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )


def _require_window_digests(name: str, digests: tuple[str, ...], window_count: int) -> None:
    if type(digests) is not tuple:
        raise WindowSlotRefusal(
            WindowSlotRefusalCode.WINDOW_BINDING_MISMATCH,
            f"{name} must be a tuple of sha256 digests",
        )
    if len(digests) != window_count:
        raise WindowSlotRefusal(
            WindowSlotRefusalCode.WINDOW_BINDING_MISMATCH,
            f"{name} carries {len(digests)} digests for {window_count} joint windows",
        )
    for index, digest in enumerate(digests):
        if not _is_sha256_hex(digest):
            raise WindowSlotRefusal(
                WindowSlotRefusalCode.WINDOW_BINDING_MISMATCH,
                f"{name}[{index}] must be a lowercase sha256 hex digest",
            )


@dataclass(frozen=True, slots=True)
class WindowPlanBinding:
    """One realizable composite plan with its per-joint-window digests.

    The three digest vectors carry one digest per joint window, in
    ``plan.joint_windows`` order. ``window_token_layout_digests`` and
    ``window_partition_plan_digests`` are produced by the token-layout
    and partition subsystems. ``window_structural_row_digests`` are
    produced by the owning window/layout compilation boundary as sha256
    over each window's canonical structural-row accounting: which
    carried rows are structural (causal anchors, injected guide rows),
    the dropped-row mapping, and any declared passthrough restoration
    payload identity. This binding never derives any of them from the
    plan.
    """

    plan: CompositeWindowPlan
    window_token_layout_digests: tuple[str, ...]
    window_partition_plan_digests: tuple[str, ...]
    window_structural_row_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.plan) is not CompositeWindowPlan:
            raise WindowSlotRefusal(
                WindowSlotRefusalCode.INVALID_WINDOW_PLAN,
                "a plan binding is built only from an exact CompositeWindowPlan value",
            )
        window_count = len(self.plan.joint_windows)
        _require_window_digests(
            "window_token_layout_digests", self.window_token_layout_digests, window_count
        )
        _require_window_digests(
            "window_partition_plan_digests", self.window_partition_plan_digests, window_count
        )
        _require_window_digests(
            "window_structural_row_digests", self.window_structural_row_digests, window_count
        )


def build_windowed_evaluation_slot(
    *,
    derivation_identity: str,
    derivation_facts_digest: str,
    plan_bindings: tuple[WindowPlanBinding, ...],
) -> CompiledPlanSlot:
    """Freeze the windowed-evaluation plan facts as their manifest slot.

    ``derivation_identity`` names the versioned window-set derivation
    profile (the pure function of declared family geometry and timeline
    anchors that realizes window sets, chapter 9.2 of the typed
    execution contracts). ``derivation_facts_digest`` is a sha256 over
    that derivation's canonical parameter and anchor serialization,
    produced by the owning derivation boundary.

    ``plan_bindings`` is the ordered vector of every composite plan the
    invocation can realize across its denoising steps, one entry per
    distinct realized plan in the derivation's canonical order; a
    static derivation supplies exactly one. The outer consensus
    therefore proves every window set the invocation will use before
    any collective for any window is issued.
    """
    if type(derivation_identity) is not str or not derivation_identity:
        raise WindowSlotRefusal(
            WindowSlotRefusalCode.INVALID_DERIVATION_BINDING,
            "derivation_identity must be a non-empty string",
        )
    if "\n" in derivation_identity:
        raise WindowSlotRefusal(
            WindowSlotRefusalCode.INVALID_DERIVATION_BINDING,
            "derivation_identity must not contain newlines",
        )
    if not _is_sha256_hex(derivation_facts_digest):
        raise WindowSlotRefusal(
            WindowSlotRefusalCode.INVALID_DERIVATION_BINDING,
            "derivation_facts_digest must be a lowercase sha256 hex digest",
        )
    if type(plan_bindings) is not tuple or not plan_bindings:
        raise WindowSlotRefusal(
            WindowSlotRefusalCode.INVALID_DERIVATION_BINDING,
            "plan_bindings must be a non-empty tuple of WindowPlanBinding values",
        )
    for binding in plan_bindings:
        if type(binding) is not WindowPlanBinding:
            raise WindowSlotRefusal(
                WindowSlotRefusalCode.INVALID_WINDOW_PLAN,
                "plan_bindings must contain only exact WindowPlanBinding values",
            )
    composite_digests = tuple(binding.plan.digest for binding in plan_bindings)
    if len(composite_digests) != len(set(composite_digests)):
        raise WindowSlotRefusal(
            WindowSlotRefusalCode.INVALID_DERIVATION_BINDING,
            "plan_bindings must carry each realizable composite plan exactly once",
        )
    facts = [
        f"derivation={derivation_identity}",
        f"derivation_facts={derivation_facts_digest}",
        f"plan_count={len(plan_bindings)}",
    ]
    for position, binding in enumerate(plan_bindings):
        plan = binding.plan
        facts.append(f"plan[{position}].composite={plan.digest}")
        facts.append(f"plan[{position}].layer_count={len(plan.layer_digests)}")
        facts.extend(
            f"plan[{position}].layer[{index}]={digest}"
            for index, digest in enumerate(plan.layer_digests)
        )
        window_count = len(plan.joint_windows)
        facts.append(f"plan[{position}].joint_window_count={window_count}")
        for index in range(window_count):
            facts.append(
                f"plan[{position}].window[{index}].token_layout="
                f"{binding.window_token_layout_digests[index]}"
            )
            facts.append(
                f"plan[{position}].window[{index}].partition_plan="
                f"{binding.window_partition_plan_digests[index]}"
            )
            facts.append(
                f"plan[{position}].window[{index}].structural_rows="
                f"{binding.window_structural_row_digests[index]}"
            )
    return CompiledPlanSlot(slot=WINDOWED_EVALUATION_SLOT, facts=tuple(facts))
