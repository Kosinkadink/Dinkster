"""Verify the signed registry notice contract without trusting feed-provided keys."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

Digest = Annotated[str, Field(pattern=r"^(?:blake3|sha256):[0-9a-f]{64}$")]
KeyId = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1)]


class SecurityError(ValueError):
    """A document cannot be authenticated or accepted without losing known blocks."""


class _Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, extra="forbid", frozen=True, strict=True)


class Release(_Model):
    pack: str
    version: str
    artifact_digest: Digest

    @field_validator("pack")
    @classmethod
    def canonical_pack(cls, value: str) -> str:
        if len(value) > 64 or re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", value) is None:
            raise ValueError("release pack must be a canonical package name")
        return value

    @field_validator("version")
    @classmethod
    def public_version(cls, value: str) -> str:
        try:
            version = Version(value)
        except InvalidVersion as exc:
            raise ValueError("release version must be public") from exc
        if str(version) != value or version.local is not None:
            raise ValueError("release version must be canonical and public")
        return value


class Notice(_Model):
    id: UUID
    revision: int = Field(gt=0)
    previous_revision: int | None = Field(default=None, gt=0)
    release: Release
    state: Literal["malicious", "vulnerable", "affected"]
    status: Literal["active", "cleared"]
    actor: Text
    authority: Literal["security", "publisher"]
    security_confirmed: bool
    changed_at: datetime
    reason: Text
    message: Text
    incident: Text
    fixed_release: Release | None = None
    fixed_lock_digest: Digest | None = None
    causes: tuple[Digest, ...] = ()

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.changed_at.utcoffset() is None:
            raise ValueError("notice time must include a timezone")
        if self.previous_revision is not None and self.previous_revision >= self.revision:
            raise ValueError("previous revision must precede replacement")
        if bool(self.causes) != (self.state == "affected"):
            raise ValueError("only affected notices must identify malicious dependency digests")
        if len(set(self.causes)) != len(self.causes):
            raise ValueError("dependency causes must be distinct")
        if self.authority == "publisher" and self.security_confirmed:
            raise ValueError("a publisher cannot confirm a security classification")
        if self.fixed_release is not None and self.fixed_release.pack != self.release.pack:
            raise ValueError("fixed release must belong to the same pack")
        return self


class _Payload(_Model):
    registry: Text
    revision: int = Field(gt=0)
    issued_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def validity_period(self) -> Self:
        if self.issued_at.utcoffset() is None or self.expires_at.utcoffset() is None:
            raise ValueError("document times must include timezones")
        if self.expires_at <= self.issued_at:
            raise ValueError("expiry must follow issue time")
        return self


class OwnershipChange(_Model):
    kind: Literal["ownership_transferred"] = "ownership_transferred"
    id: UUID
    revision: int = Field(gt=0)
    pack_id: UUID
    pack: str
    previous_publisher: Text
    publisher: Text
    actor: Text
    changed_at: datetime
    reason: Text

    @model_validator(mode="after")
    def identity(self) -> Self:
        Release.canonical_pack(self.pack)
        if self.changed_at.utcoffset() is None:
            raise ValueError("ownership time must include a timezone")
        if self.publisher == self.previous_publisher:
            raise ValueError("ownership must change publisher")
        return self


class Document(_Payload):
    format: Literal["dinkster.security/1"]
    notices: tuple[Notice, ...]
    ownership_changes: tuple[OwnershipChange, ...] = ()

    @model_validator(mode="after")
    def identities(self) -> Self:
        for records in (self.notices, self.ownership_changes):
            if len({record.id for record in records}) != len(records):
                raise ValueError("document identities must be distinct")
            if any(record.revision > self.revision for record in records):
                raise ValueError("record revision exceeds document revision")
        return self


def _decode(value: str, size: int) -> bytes:
    decoded = base64.b64decode(value, validate=True)
    if len(decoded) != size or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError("invalid security base64 size or encoding")
    return decoded


class Key(_Model):
    key_id: KeyId
    public_key: str

    @model_validator(mode="after")
    def identity(self) -> Self:
        if "sha256:" + hashlib.sha256(_decode(self.public_key, 32)).hexdigest() != self.key_id:
            raise ValueError("key identifier differs from public bytes")
        return self


class Trust(_Payload):
    format: Literal["dinkster.security-trust/1"]
    keys: tuple[Key, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identities(self) -> Self:
        if len({key.key_id for key in self.keys}) != len(self.keys):
            raise ValueError("trusted keys must be distinct")
        return self


class _Signature(_Model):
    key_id: KeyId
    signature: str

    @field_validator("signature")
    @classmethod
    def signature_bytes(cls, value: str) -> str:
        _decode(value, 64)
        return value


class _Envelope(_Model):
    signed: Annotated[Document | Trust, Field(discriminator="format")]
    signatures: tuple[_Signature, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_signers(self) -> Self:
        if len({signature.key_id for signature in self.signatures}) != len(self.signatures):
            raise ValueError("signers must be distinct")
        return self


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SecurityError("duplicate security field")
        result[key] = value
    return result


def signed_payload(data: bytes) -> bytes:
    """Retain received signed fields, including omitted defaults, in revision checkpoints."""
    try:
        raw = json.loads(data, object_pairs_hook=_unique_fields)
        return canonical(raw["signed"])
    except (ValueError, TypeError, KeyError) as exc:
        raise SecurityError("invalid security document") from exc


def _authenticated(data: bytes, trust: Trust) -> Document | Trust:
    try:
        payload = signed_payload(data)
        envelope = _Envelope.model_validate_json(data)
        if envelope.signed.registry != trust.registry:
            raise SecurityError("security registry differs from independently trusted registry")
        keys = {key.key_id: _decode(key.public_key, 32) for key in trust.keys}
        for signature in envelope.signatures:
            if signature.key_id not in keys:
                continue
            try:
                Ed25519PublicKey.from_public_bytes(keys[signature.key_id]).verify(
                    _decode(signature.signature, 64), payload
                )
            except InvalidSignature:
                continue
            return envelope.signed
    except (ValueError, TypeError, KeyError) as exc:
        raise SecurityError("invalid security document") from exc
    raise SecurityError("no valid independently trusted security signature")


def verify_document(data: bytes, trust: Trust, *, previous: bytes | None = None) -> Document:
    document = _authenticated(data, trust)
    if not isinstance(document, Document):
        raise SecurityError("expected release notices, not trust")
    if previous is None:
        return document
    old_payload = signed_payload(previous)
    old = Document.model_validate_json(old_payload)
    if old.registry != trust.registry or document.revision < old.revision:
        raise SecurityError("security registry or revision rolled back")
    if document.revision == old.revision and signed_payload(data) != old_payload:
        raise SecurityError("security signed bytes changed without a new revision")
    current = {notice.id: notice for notice in document.notices}
    for prior in old.notices:
        updated = current.get(prior.id)
        if updated is None or updated.revision < prior.revision:
            raise SecurityError("security update dropped a notice or clearing record")
        if updated.release != prior.release:
            raise SecurityError("security update changed a release identity")
        if updated != prior and (
            updated.revision <= prior.revision
            or updated.previous_revision is None
            or updated.previous_revision < prior.revision
        ):
            raise SecurityError("notice replacement does not name its predecessor")
    ownership = {change.id: change for change in document.ownership_changes}
    if any(ownership.get(change.id) != change for change in old.ownership_changes):
        raise SecurityError("security update dropped or changed an ownership record")
    return document


def verify_trust(data: bytes, trust: Trust) -> Trust:
    replacement = _authenticated(data, trust)
    if not isinstance(replacement, Trust) or replacement.revision <= trust.revision:
        raise SecurityError("security trust update must increase its revision")
    return replacement
