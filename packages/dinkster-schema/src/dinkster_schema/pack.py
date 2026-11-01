"""Pack-owned publication versions and engine compatibility declarations."""

from __future__ import annotations

from dataclasses import dataclass

from packaging.specifiers import SpecifierSet
from packaging.version import Version


@dataclass(frozen=True)
class PackCompatibility:
    """One pack version and its PEP 440 engine range, independent of engine releases."""

    version: str
    engine: str

    def __post_init__(self) -> None:
        version = Version(self.version)
        if version.local is not None or str(version) != self.version:
            raise ValueError("pack version must be a normalized public PEP 440 version")
        if not str(SpecifierSet(self.engine)):
            raise ValueError("pack engine must be a nonempty PEP 440 specifier set")

    def require_engine(self, pack: str, engine_version: str) -> None:
        """Refuse an incompatible target before installing any pack code."""
        if not SpecifierSet(self.engine).contains(engine_version, prereleases=True):
            raise ValueError(
                f"pack {pack}@{self.version} declares engine {self.engine}; "
                f"engine {engine_version} is incompatible"
            )
