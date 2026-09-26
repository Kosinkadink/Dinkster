"""Execution memory receipt contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
from dinkster_inference import (
    MEMORY_PAGE_CLASSES,
    ComponentMemorySnapshot,
    DeviceMemorySnapshot,
    ExecutionMemoryDecision,
    ExecutionMemoryReceipt,
    ExecutionMemorySnapshot,
    ExecutionObserverAttachment,
    MemoryPageClass,
)


def _classes(**overrides: int) -> dict[MemoryPageClass, int]:
    values: dict[MemoryPageClass, int] = {page_class: 0 for page_class in MEMORY_PAGE_CLASSES}
    values.update(cast("dict[MemoryPageClass, int]", overrides))
    return values


def _component(
    component_id: str,
    *,
    storage_id: str | None = None,
    loaded: int = 60,
    offloaded: int = 40,
    resident: int = 75,
) -> ComponentMemorySnapshot:
    return ComponentMemorySnapshot(
        component_id=component_id,
        component_role=component_id.split(".", 1)[0],
        storage_id=storage_id or component_id,
        device="cuda:3",
        total_bytes=loaded + offloaded,
        loaded_bytes=loaded,
        offloaded_bytes=offloaded,
        resident_bytes=resident,
        bytes_by_page_class=_classes(
            weights=loaded,
            **{"other-reclaimable": resident - loaded},
        ),
    )


def test_receipt_preserves_distinct_components_transitions_unknown_and_peak() -> None:
    receipt = ExecutionMemoryReceipt(frozenset({"diffusion.main", "text.encoder"}))
    attachment = ExecutionObserverAttachment(receipt.observe, "run-7")
    first = ExecutionMemorySnapshot(
        boundary="first-sampling-seam",
        components=(
            _component("diffusion.main", loaded=60, offloaded=40, resident=75),
            _component("text.encoder", loaded=25, offloaded=75, resident=25),
        ),
        devices=(
            DeviceMemorySnapshot(
                device="cuda:3",
                measured_bytes=112,
                reconciliation_bound_bytes=0,
                unknown_bytes=12,
            ),
        ),
        memory_compiler="disabled",
    )
    peak = ExecutionMemorySnapshot(
        boundary="peak-residency",
        components=(
            _component("diffusion.main", loaded=100, offloaded=0, resident=120),
            _component("text.encoder", loaded=0, offloaded=100, resident=0),
        ),
        devices=(
            DeviceMemorySnapshot(
                device="cuda:3",
                measured_bytes=128,
                reconciliation_bound_bytes=0,
                unknown_bytes=8,
            ),
        ),
        memory_compiler="disabled",
    )
    attachment.record_memory_snapshot("sample", "residency", first)
    attachment.record_memory_decision(
        "sample",
        "residency-policy",
        ExecutionMemoryDecision(
            component_id="text.encoder",
            component_role="text",
            device="cuda:3",
            source="residency-policy",
            action="offload",
            byte_count=25,
            reason="stage-lifecycle",
        ),
    )
    attachment.record_memory_snapshot("sample", "residency", peak)

    record = receipt.to_dict()
    assert record["schema"] == "dinkster.execution-memory.v1"
    assert record["invocationId"] == "run-7"
    assert record["peakResidentBytesByDevice"] == {"cuda:3": 128}
    snapshots = record["snapshots"]
    assert isinstance(snapshots, list)
    assert snapshots[0]["boundary"] == "first-sampling-seam"
    assert [component["componentId"] for component in snapshots[0]["components"]] == [
        "diffusion.main",
        "text.encoder",
    ]
    assert snapshots[0]["devices"][0]["unknownBytes"] == 12
    assert record["decisions"] == [
        {
            "componentId": "text.encoder",
            "componentRole": "text",
            "device": "cuda:3",
            "source": "residency-policy",
            "action": "offload",
            "byteCount": 25,
            "reason": "stage-lifecycle",
        }
    ]


def test_snapshot_deduplicates_shared_storage_for_reconciliation() -> None:
    shared_a = _component("diffusion.base", storage_id="shared-model")
    shared_b = _component("control.patch", storage_id="shared-model")

    snapshot = ExecutionMemorySnapshot(
        boundary="post-load",
        components=(shared_a, shared_b),
        devices=(
            DeviceMemorySnapshot(
                device="cuda:3",
                measured_bytes=80,
                reconciliation_bound_bytes=0,
                unknown_bytes=5,
            ),
        ),
        memory_compiler="unavailable",
    )

    assert snapshot.devices[0].unknown_bytes == 5


@pytest.mark.parametrize(
    ("components", "device", "message"),
    (
        (
            (_component("diffusion.main", resident=75),),
            DeviceMemorySnapshot("cuda:3", 70, 0, 0),
            "reconciliation bound",
        ),
        (
            (_component("diffusion.main", resident=75),),
            DeviceMemorySnapshot("cuda:3", 90, 14, 0),
            "reconciliation bound",
        ),
        (
            (
                _component("diffusion.main", storage_id="shared", resident=75),
                _component("control.patch", storage_id="shared", resident=80),
            ),
            DeviceMemorySnapshot("cuda:3", 80, 5, 5),
            "shared storage",
        ),
    ),
)
def test_snapshot_rejects_double_count_and_unreconciled_totals(
    components: tuple[ComponentMemorySnapshot, ...],
    device: DeviceMemorySnapshot,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ExecutionMemorySnapshot(
            boundary="stage-end",
            components=components,
            devices=(device,),
            memory_compiler="active",
        )


def test_receipt_rejects_missing_component() -> None:
    receipt = ExecutionMemoryReceipt(frozenset({"diffusion.main", "text.encoder"}))
    attachment = ExecutionObserverAttachment(receipt.observe)
    snapshot = ExecutionMemorySnapshot(
        boundary="post-load",
        components=(_component("diffusion.main"),),
        devices=(DeviceMemorySnapshot("cuda:3", 75, 0, 0),),
        memory_compiler="disabled",
    )

    attachment.record_memory_snapshot("load", "residency", snapshot)

    with pytest.raises(ValueError, match="missing components: text.encoder"):
        receipt.to_dict()


def test_receipt_rejects_missing_snapshots() -> None:
    receipt = ExecutionMemoryReceipt(frozenset({"diffusion.main"}))

    with pytest.raises(ValueError, match="no snapshots"):
        receipt.to_dict()


def test_component_requires_explicit_unknown_page_class() -> None:
    classes = _classes(weights=60)
    del classes["unknown"]

    with pytest.raises(ValueError, match="every page class"):
        ComponentMemorySnapshot(
            component_id="diffusion.main",
            component_role="diffusion",
            storage_id="diffusion.main",
            device="cuda:3",
            total_bytes=100,
            loaded_bytes=60,
            offloaded_bytes=40,
            resident_bytes=60,
            bytes_by_page_class=classes,
        )


def test_snapshot_requires_device_accounting_for_each_component() -> None:
    with pytest.raises(ValueError, match="every component device"):
        ExecutionMemorySnapshot(
            boundary="post-load",
            components=(_component("diffusion.main"),),
            devices=(),
            memory_compiler="unavailable",
        )


def test_memory_records_reject_unknown_closed_values() -> None:
    with pytest.raises(ValueError, match="snapshot boundary"):
        ExecutionMemorySnapshot(
            boundary=cast("Any", "during-load"),
            components=(),
            devices=(),
            memory_compiler="unavailable",
        )
    with pytest.raises(ValueError, match="decision action"):
        ExecutionMemoryDecision(
            component_id="diffusion.main",
            component_role="diffusion",
            device="cuda:3",
            source="residency-policy",
            action=cast("Any", "move"),
            byte_count=1,
            reason="test",
        )
