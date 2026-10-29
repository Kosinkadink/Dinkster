"""Platform-neutral parsing for Windows USN records."""

from __future__ import annotations


def usn_from_record(record: bytes, returned: int) -> int:
    major_version = int.from_bytes(record[4:6], "little")
    if major_version not in (2, 3):
        raise OSError(f"unsupported USN record version {major_version}")
    usn_offset = 24 if major_version == 2 else 40
    if returned < usn_offset + 8:
        raise OSError("incomplete USN record")
    return int.from_bytes(record[usn_offset : usn_offset + 8], "little", signed=True)
