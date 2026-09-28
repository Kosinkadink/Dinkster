from __future__ import annotations

import pytest

from tools.workflow_benchmark import dinkster_execution_receipt


def test_dinkster_execution_receipt_binds_run_extensions_and_source() -> None:
    accepted = {
        "runId": "run-466",
        "extensionSnapshotDigest": "sha256:" + "a" * 64,
    }

    assert dinkster_execution_receipt(accepted, "b" * 40) == {
        "runId": "run-466",
        "extensionSnapshotDigest": "sha256:" + "a" * 64,
        "dinksterGitHead": "b" * 40,
    }


@pytest.mark.parametrize("missing", ["runId", "extensionSnapshotDigest"])
def test_dinkster_execution_receipt_rejects_missing_identity(missing: str) -> None:
    accepted = {
        "runId": "run-466",
        "extensionSnapshotDigest": "sha256:" + "a" * 64,
    }
    del accepted[missing]

    with pytest.raises(ValueError, match="missing execution provenance"):
        dinkster_execution_receipt(accepted, "b" * 40)
