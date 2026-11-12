"""Read sealed wheel provenance without executing installed interpreters or packs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from packaging.utils import canonicalize_name, parse_wheel_filename
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .documents import SecurityError
from .policy import Dependency


class _Wheel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    filename: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry: str = Field(min_length=1)

    @field_validator("registry")
    @classmethod
    def issuing_registry(cls, value: str) -> str:
        issuer = urlparse(value)
        if (
            issuer.scheme not in {"http", "https"}
            or not issuer.hostname
            or issuer.username
            or issuer.password
            or issuer.query
            or issuer.fragment
        ):
            raise ValueError("installed wheel requires an uncredentialed issuing registry")
        return value


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    format: Literal["dinkster.installed-wheels/1"]
    wheels: tuple[_Wheel, ...]
    inherited: tuple[str, ...]

    @model_validator(mode="after")
    def unique_distributions(self) -> _Receipt:
        if any(
            Path(wheel.filename).name != wheel.filename or "\\" in wheel.filename
            for wheel in self.wheels
        ):
            raise ValueError("installed wheel filenames must be basenames")
        names = [parse_wheel_filename(wheel.filename)[0] for wheel in self.wheels]
        if len(names) != len(set(names)) or len(self.inherited) != len(set(self.inherited)):
            raise ValueError("installed wheel receipt repeats a distribution")
        if any(canonicalize_name(name) != name for name in self.inherited):
            raise ValueError("inherited distribution names must be canonical")
        if set(names) & set(self.inherited):
            raise ValueError("installed wheel cannot also be inherited")
        return self


def generation_dependencies(
    generation: Path,
    files: Mapping[str, str],
    packs: tuple[str, ...],
    in_process_packs: tuple[str, ...],
) -> dict[str, tuple[Dependency, ...]]:
    """Include every library exposed by the parent and each pack's local venv.

    In-process packs add libraries to the shared interpreter; layered workers can
    import those libraries too, even when their own dependency lock omits them.
    """

    def read(name: str) -> _Receipt:
        path = generation / name
        if any(item.is_symlink() or item.is_junction() for item in (path, *path.parents)):
            raise SecurityError(f"security inventory refuses linked provenance: {name}")
        try:
            raw = path.read_bytes()
            if "sha256:" + hashlib.sha256(raw).hexdigest() != files.get(name):
                raise SecurityError(f"security inventory requires sealed provenance: {name}")
            return _Receipt.model_validate_json(raw)
        except (OSError, ValueError) as exc:
            raise SecurityError(f"security inventory cannot read {name}: {exc}") from exc

    if set(in_process_packs) - set(packs):
        raise SecurityError("security inventory hosting names an absent pack")
    if len(set(packs)) != len(packs) or any(
        re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name) is None for name in packs
    ):
        raise SecurityError("security inventory requires portable pack names")
    root = read("installed-wheels.json")
    if root.inherited:
        raise SecurityError("generation wheel provenance cannot inherit another environment")
    receipts = {name: read(f"packs/{name}/installed-wheels.json") for name in packs}

    def dependency(wheel: _Wheel) -> Dependency:
        name, version, _, _ = parse_wheel_filename(wheel.filename)
        return Dependency(wheel.registry, name, str(version), "sha256:" + wheel.sha256)

    shared: dict[str, Dependency] = {}
    for receipt in (root, *(receipts[name] for name in in_process_packs)):
        for wheel in receipt.wheels:
            name = parse_wheel_filename(wheel.filename)[0]
            if name in shared:
                raise SecurityError(f"security inventory repeats a shared library: {name}")
            shared[name] = dependency(wheel)
    result: dict[str, tuple[Dependency, ...]] = {}
    for name, receipt in receipts.items():
        if set(receipt.inherited) - shared.keys():
            raise SecurityError(f"security inventory cannot identify inherited libraries: {name}")
        result[name] = tuple(
            sorted(
                set(shared.values()) | {dependency(wheel) for wheel in receipt.wheels},
                key=lambda item: (item.registry, item.name, item.version, item.digest),
            )
        )
    return result
