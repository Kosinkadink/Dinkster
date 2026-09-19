"""Repository-local tools package."""

from .evidence_paths import EVIDENCE_ROOT

# Parity tooling is maintained in the companion evidence checkout.
__path__.append(str(EVIDENCE_ROOT / "tools"))
