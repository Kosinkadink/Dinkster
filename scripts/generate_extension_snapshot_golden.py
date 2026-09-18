"""Generate the canonical S0-A extension snapshot proof golden."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dinkster_protocol import (
    ActiveExtension,
    ExtensionSnapshot,
    canonical_extension_snapshot,
    extension_behavior_hash,
)

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "goldens" / "extensions" / "snapshot-v1.json"


def proof_snapshot() -> ExtensionSnapshot:
    return ExtensionSnapshot(
        (
            ActiveExtension(
                id="example.attention",
                version="2.4.1",
                package_digest="sha256:" + "1a" * 32,
                contribution_ids=(
                    "attention/qkv-normalize",
                    "attention/kernel-wrapper",
                ),
                selector_resolutions=(
                    (
                        "attention:cross",
                        ("double-block/0/attn", "single-block/0/attn"),
                    ),
                ),
                service_providers=(("attention.kernel", "example.attention/kernel-wrapper"),),
                capabilities=("model-family-registration",),
                behavior_configuration=(
                    ("enabled", True),
                    ("limit", 4),
                    ("mode", "strict"),
                    ("offset", -2),
                    ("override", None),
                ),
            ),
            ActiveExtension(
                id="example.monitor",
                version="1.0.0",
                package_digest="sha256:" + "b7" * 32,
                contribution_ids=("events/progress-observer",),
                capabilities=("background-jobs", "routes"),
                behavior_configuration=(
                    ("sampleEvery", 5),
                    ("unicode", "coffee \u2615"),
                ),
            ),
        )
    )


def build_golden() -> dict[str, object]:
    snapshot = proof_snapshot()
    canonical = canonical_extension_snapshot(snapshot)
    return {
        "snapshot": json.loads(canonical),
        "canonicalSerialization": canonical,
        "behaviorSha256": extension_behavior_hash(snapshot),
    }


def render_golden() -> bytes:
    rendered = json.dumps(build_golden(), indent=2, ensure_ascii=True) + "\n"
    return rendered.encode("utf-8")


def main() -> None:
    content = render_golden()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_bytes(content)
    print(f"golden file sha256: {hashlib.sha256(content).hexdigest()}")
    print(f"behavior sha256: {build_golden()['behaviorSha256']}")


if __name__ == "__main__":
    main()
