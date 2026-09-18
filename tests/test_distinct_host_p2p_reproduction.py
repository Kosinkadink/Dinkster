from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_assets import derive_p2p_descriptor

from scripts import reproduce_distinct_host_p2p as reproduction


def test_fixture_is_identical_across_hosts(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "other" / "second.safetensors"
    second.parent.mkdir()

    reproduction._write_source(first, payload_size=257)
    reproduction._write_source(second, payload_size=257)

    assert first.read_bytes() == second.read_bytes()
    assert derive_p2p_descriptor(first) == derive_p2p_descriptor(second)


@pytest.mark.parametrize(
    ("result_exists", "max_downloaded", "saw_download", "expected"),
    [
        (True, 1, True, "complete"),
        (False, 0, False, "discovery-failed"),
        (False, 0, True, "zero-byte-stall"),
        (False, 1, True, "failed-after-progress"),
    ],
)
def test_result_classification_distinguishes_zero_byte_stalls(
    tmp_path: Path,
    result_exists: bool,
    max_downloaded: int,
    saw_download: bool,
    expected: str,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"expected")
    result = tmp_path / "result"
    if result_exists:
        result.write_bytes(source.read_bytes())

    assert (
        reproduction._classify_result(
            result if result_exists else None,
            source=source,
            max_downloaded=max_downloaded,
            saw_download=saw_download,
        )
        == expected
    )


def test_transfer_observation_reads_progress_and_peer_addresses() -> None:
    status: dict[str, object] = {
        "sidecar": {
            "totals": {"downloadedBytes": 8192},
            "leases": [{"kind": "download"}],
            "diagnostics": {
                "events": [
                    {"kind": "peer_connected", "address": "192.168.1.53"},
                    {"kind": "peer_connected", "address": "192.168.1.56"},
                    {"kind": "peer_disconnected", "address": "192.168.1.57"},
                ]
            },
        }
    }

    assert reproduction._transfer_observation(status) == (
        8192,
        True,
        {"192.168.1.53", "192.168.1.56"},
    )
