"""Registry principals: users, publishers, memberships, tokens, audit.

The identity half of the registry model, pure like everything else in
this package - no server, storage, network, or clock (callers supply
ISO 8601 UTC timestamps, exactly like ``ReviewLog``). Each rule answers
a recorded Comfy-Org/registry-backend gap (publisher
identity):

- Users (authentication identities) and publishers (the org-like
  principals the grant table keys to) are SEPARATE entities, joined by
  real memberships whose roles gate operations in one place
  (``authorize``). Every mutation records the individual actor, never
  just the publisher. (Prior art: a ``member`` role that gates nothing.)
- Publisher ids ride the one closed name grammar
  (``dinkster_schema.validate_name`` / ``canonical_name``), so separator
  and case variants can never mint two publishers. User ids are the
  auth layer's business and stay opaque non-empty strings here.
- Publisher tokens are hashed at rest (sha256; the plaintext exists
  only in ``mint_token``'s return value), carry the leak-scannable
  prefix ``dinkster_pat_``, MUST expire (no expiry, no token), are scoped
  to publisher membership and optionally one pack, and record who minted
  them. Pack-scoped tokens remain pack-operation credentials; services
  require an unscoped token before considering publisher-wide or
  registry-wide role authorization.
  Removing a member invalidates every token they minted for that
  publisher, both eagerly (revocation on removal) and structurally
  (``verify_token`` re-checks the minter's membership). (Prior art:
  plaintext UUIDs, unscoped, non-expiring, identifying nobody.)
- Ids are immutable; display names are mutable but every change is an
  audit record naming the actor and both values, so attribution history
  cannot be silently rewritten.
- The audit log is append-only by construction: readers get tuples,
  writers only append.

Deliberately NOT here: password/OAuth machinery (the auth layer that
produces user ids), token transport (Authorization headers are server
wiring), review deadlines/escalation (server policy over ``ReviewLog``),
and persistence (the registry service serializes these tables the same
way the installer serializes lockfiles).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal, get_args

from dinkster_schema import canonical_name, validate_name

from .model import RegistryError

Role = Literal["owner", "member"]
"""Membership roles. ``owner`` administers the publisher; ``member``
publishes and mints their own publish tokens. Further roles are a
deliberate later addition, never a reinterpretation."""

Action = Literal["publish", "mint-token", "edit-metadata", "manage-members", "transfer"]
"""The operations roles gate. One table, one enforcement point."""

_ROLE_ACTIONS: dict[Role, frozenset[Action]] = {
    "owner": frozenset({"publish", "mint-token", "edit-metadata", "manage-members", "transfer"}),
    "member": frozenset({"publish", "mint-token"}),
}
_VALID_ROLES = frozenset(get_args(Role))

TOKEN_PREFIX = "dinkster_pat_"
"""Leak-scannable prefix on every publishing token."""

_TOKEN_SECRET_BYTES = 32
_TOKEN_ID_CHARS = 12


def _parse_at(at: str, what: str) -> datetime:
    """Parse a caller-supplied ISO 8601 timestamp; garbage is a caller bug."""
    try:
        return datetime.fromisoformat(at)
    except ValueError as exc:
        raise RegistryError(f"{what} must be an ISO 8601 timestamp, got {at!r}") from exc


@dataclass(frozen=True)
class AuditRecord:
    """One attributed mutation: who did what to which subject, when.

    ``details`` is a human-readable sentence (old/new values for edits,
    role for membership changes) - the record is evidence, not wire data.
    """

    actor: str
    action: str
    subject: str
    at: str
    details: str = ""


@dataclass(frozen=True)
class Membership:
    """One user's role in one publisher."""

    user: str
    publisher: str
    role: Role


@dataclass(frozen=True)
class TokenRecord:
    """A publisher token at rest: everything except the secret.

    ``token_id`` is the public handle (revocation, listing, audit) - a
    stable prefix of the secret's hash, never the secret. ``pack`` narrows
    pack operations to one pack; None permits publisher-wide role checks.
    """

    token_id: str
    publisher: str
    minted_by: str
    pack: str | None
    secret_hash: str
    created_at: str
    expires_at: str
    revoked: bool = False
    revoked_reason: str = ""


def _hash_secret(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


class PrincipalDirectory:
    """Users, publishers, memberships, and tokens - the one authority.

    Every mutation takes the acting user and a timestamp, authorizes the
    action through the single role table, and appends an audit record.
    Refusals raise ``RegistryError`` naming the missing role or broken
    invariant.
    """

    def __init__(self) -> None:
        self._users: set[str] = set()
        self._publishers: dict[str, str] = {}  # canonical id -> display name
        self._memberships: dict[tuple[str, str], Membership] = {}  # (publisher, user)
        self._tokens: dict[str, TokenRecord] = {}  # token_id -> record
        self._audit: list[AuditRecord] = []

    @classmethod
    def restore(
        cls,
        *,
        users: Iterable[str] = (),
        publishers: Mapping[str, str] | None = None,
        memberships: Iterable[Membership] = (),
        tokens: Iterable[TokenRecord] = (),
        audit: Iterable[AuditRecord] = (),
    ) -> PrincipalDirectory:
        """Rebuild a directory from persisted state, re-running every
        structural validation - a hand-edited or corrupt store fails at
        load, never at authorization time (the ``Lockfile.from_record_json``
        stance). Persisted forms are canonical: non-canonical publisher
        ids refuse rather than re-normalize, so storage can never hold a
        second spelling of one principal.
        """
        directory = cls()
        for user in users:
            directory.register_user(user)
        for publisher, display in (publishers or {}).items():
            problem = validate_name(publisher)
            if problem is not None:
                raise RegistryError(f"publisher id {publisher!r} {problem}")
            if canonical_name(publisher) != publisher:
                raise RegistryError(
                    f"publisher id {publisher!r} is not canonical (expected "
                    f"{canonical_name(publisher)!r}); the directory stores canonical forms only"
                )
            directory._publishers[publisher] = display
        owners: set[str] = set()
        for member in memberships:
            if member.publisher not in directory._publishers:
                raise RegistryError(
                    f"membership of {member.user!r} names unknown publisher {member.publisher!r}"
                )
            directory._require_user(member.user)
            if member.role not in get_args(Role):
                raise RegistryError(f"membership of {member.user!r} has unknown role")
            if (member.publisher, member.user) in directory._memberships:
                raise RegistryError(
                    f"duplicate membership of {member.user!r} in {member.publisher!r}"
                )
            directory._memberships[(member.publisher, member.user)] = member
            if member.role == "owner":
                owners.add(member.publisher)
        for publisher in directory._publishers:
            if publisher not in owners:
                raise RegistryError(
                    f"publisher {publisher!r} restored without an owner; a publisher "
                    f"always has at least one owner"
                )
        for token in tokens:
            if token.publisher not in directory._publishers:
                raise RegistryError(
                    f"token {token.token_id} names unknown publisher {token.publisher!r}"
                )
            if token.token_id != token.secret_hash[:_TOKEN_ID_CHARS]:
                raise RegistryError(f"token {token.token_id} does not match its secret hash")
            if token.token_id in directory._tokens:
                raise RegistryError(f"duplicate token {token.token_id}")
            if token.revoked and not token.revoked_reason:
                raise RegistryError(f"token {token.token_id} is revoked without a reason")
            created = _parse_at(token.created_at, f"token {token.token_id} created_at")
            expires = _parse_at(token.expires_at, f"token {token.token_id} expires_at")
            if expires <= created:
                raise RegistryError(f"token {token.token_id} expires before it was minted")
            directory._tokens[token.token_id] = token
        for record in audit:
            if not record.actor:
                raise RegistryError("audit records require an actor")
            _parse_at(record.at, "audit timestamp")
            directory._audit.append(record)
        return directory

    # -- reads ------------------------------------------------------------

    def audit(self) -> tuple[AuditRecord, ...]:
        return tuple(self._audit)

    def memberships(self, publisher: str) -> tuple[Membership, ...]:
        canonical = canonical_name(publisher)
        return tuple(m for (p, _), m in self._memberships.items() if p == canonical)

    def role_of(self, user: str, publisher: str) -> Role | None:
        member = self._memberships.get((canonical_name(publisher), user))
        return member.role if member is not None else None

    def display_name(self, publisher: str) -> str:
        return self._require_publisher(publisher)[1]

    def tokens(self, publisher: str) -> tuple[TokenRecord, ...]:
        canonical = canonical_name(publisher)
        return tuple(t for t in self._tokens.values() if t.publisher == canonical)

    def authorize(self, user: str, publisher: str, action: Action) -> None:
        """The one enforcement point: refuse unless ``user``'s role in
        ``publisher`` permits ``action``."""
        canonical = self._require_publisher(publisher)[0]
        member = self._memberships.get((canonical, user))
        if member is None:
            raise RegistryError(f"user {user!r} is not a member of publisher {canonical!r}")
        if action not in _ROLE_ACTIONS[member.role]:
            needed = sorted(role for role, actions in _ROLE_ACTIONS.items() if action in actions)
            raise RegistryError(
                f"user {user!r} has role {member.role!r} in publisher "
                f"{canonical!r}; {action!r} requires {' or '.join(needed)}"
            )

    def record(self, actor: str, action: str, subject: str, at: str, details: str = "") -> None:
        """Append an attributed event whose state lives outside this
        directory (publish, review resolution). The registry keeps ONE
        audit trail, not one per subsystem."""
        self._record(actor, action, subject, at, details)

    # -- principals -------------------------------------------------------

    def register_user(self, user: str) -> None:
        """Record an authentication identity. Ids are opaque and immutable;
        the auth layer that produces them lives outside this model."""
        if not user:
            raise RegistryError("user ids must not be empty")
        self._users.add(user)

    def register_publisher(self, publisher: str, owner: str, at: str) -> str:
        """Create a publisher with its first owner; returns the canonical id."""
        _parse_at(at, "timestamp")
        problem = validate_name(publisher)
        if problem is not None:
            raise RegistryError(f"publisher id {publisher!r} {problem}")
        canonical = canonical_name(publisher)
        if canonical in self._publishers:
            raise RegistryError(f"publisher {canonical!r} already exists")
        self._require_user(owner)
        self._publishers[canonical] = canonical
        self._memberships[(canonical, owner)] = Membership(owner, canonical, "owner")
        self._record(owner, "register-publisher", canonical, at)
        return canonical

    def set_display_name(self, publisher: str, name: str, actor: str, at: str) -> None:
        """Ids never change; display does, and every change is audited."""
        _parse_at(at, "timestamp")
        canonical = self._require_publisher(publisher)[0]
        self.authorize(actor, canonical, "edit-metadata")
        old = self._publishers[canonical]
        self._publishers[canonical] = name
        self._record(actor, "edit-metadata", canonical, at, f"display name {old!r} -> {name!r}")

    # -- membership -------------------------------------------------------

    def add_member(self, publisher: str, user: str, role: Role, actor: str, at: str) -> None:
        if role not in _VALID_ROLES:
            raise RegistryError(f"unknown membership role {role!r}")
        _parse_at(at, "timestamp")
        canonical = self._require_publisher(publisher)[0]
        self.authorize(actor, canonical, "manage-members")
        self._require_user(user)
        if (canonical, user) in self._memberships:
            raise RegistryError(
                f"user {user!r} is already a member of {canonical!r}; "
                f"use set_role to change their role"
            )
        self._memberships[(canonical, user)] = Membership(user, canonical, role)
        self._record(actor, "manage-members", canonical, at, f"added {user!r} as {role!r}")

    def set_role(self, publisher: str, user: str, role: Role, actor: str, at: str) -> None:
        if role not in _VALID_ROLES:
            raise RegistryError(f"unknown membership role {role!r}")
        _parse_at(at, "timestamp")
        canonical = self._require_publisher(publisher)[0]
        self.authorize(actor, canonical, "manage-members")
        member = self._memberships.get((canonical, user))
        if member is None:
            raise RegistryError(f"user {user!r} is not a member of publisher {canonical!r}")
        if member.role == "owner" and role != "owner":
            self._guard_last_owner(canonical, user)
        self._memberships[(canonical, user)] = replace(member, role=role)
        self._record(
            actor, "manage-members", canonical, at, f"{user!r} role {member.role!r} -> {role!r}"
        )

    def remove_member(self, publisher: str, user: str, actor: str, at: str) -> None:
        """Remove a member and revoke every token they minted for this
        publisher - membership revocation invalidates the member's tokens."""
        _parse_at(at, "timestamp")
        canonical = self._require_publisher(publisher)[0]
        self.authorize(actor, canonical, "manage-members")
        member = self._memberships.get((canonical, user))
        if member is None:
            raise RegistryError(f"user {user!r} is not a member of publisher {canonical!r}")
        if member.role == "owner":
            self._guard_last_owner(canonical, user)
        del self._memberships[(canonical, user)]
        self._record(actor, "manage-members", canonical, at, f"removed {user!r}")
        for token in list(self._tokens.values()):
            if token.publisher == canonical and token.minted_by == user and not token.revoked:
                self._revoke(token, actor, at, f"minter {user!r} removed from publisher")

    # -- tokens -----------------------------------------------------------

    def mint_token(
        self,
        publisher: str,
        minted_by: str,
        at: str,
        expires_at: str,
        pack: str | None = None,
        secret: str | None = None,
    ) -> tuple[str, TokenRecord]:
        """Mint a publish token; returns (plaintext, record). The plaintext
        exists only in this return value - at rest there is only the hash.

        ``secret`` is injectable for tests; production callers omit it and
        get ``secrets.token_hex``. Expiry is mandatory and must be after
        ``at`` - non-expiring tokens are unrepresentable.
        """
        canonical = self._require_publisher(publisher)[0]
        self.authorize(minted_by, canonical, "mint-token")
        if _parse_at(expires_at, "token expiry") <= _parse_at(at, "mint time"):
            raise RegistryError(
                f"token expiry {expires_at!r} must be after mint time {at!r}; "
                f"non-expiring tokens are not a thing"
            )
        entropy = secret if secret is not None else secrets.token_hex(_TOKEN_SECRET_BYTES)
        plaintext = TOKEN_PREFIX + entropy
        secret_hash = _hash_secret(plaintext)
        token_id = secret_hash[:_TOKEN_ID_CHARS]
        if token_id in self._tokens:
            raise RegistryError(f"token id {token_id!r} already exists; supply fresh entropy")
        record = TokenRecord(
            token_id=token_id,
            publisher=canonical,
            minted_by=minted_by,
            pack=canonical_name(pack) if pack is not None else None,
            secret_hash=secret_hash,
            created_at=at,
            expires_at=expires_at,
        )
        self._tokens[token_id] = record
        scope = f"pack {record.pack!r}" if record.pack is not None else "any pack"
        self._record(minted_by, "mint-token", canonical, at, f"token {token_id} (publish, {scope})")
        return plaintext, record

    def revoke_token(self, token_id: str, actor: str, at: str, reason: str) -> None:
        """Revoke one token. The minter may revoke their own; anything
        else requires manage-members. Reasons are mandatory - silent
        revocations are the opaque states this model exists to prevent."""
        _parse_at(at, "timestamp")
        token = self._tokens.get(token_id)
        if token is None:
            raise RegistryError(f"no token {token_id!r}")
        if not reason:
            raise RegistryError("revoking a token requires an explicit reason")
        if actor != token.minted_by:
            self.authorize(actor, token.publisher, "manage-members")
        if token.revoked:
            return
        self._revoke(token, actor, at, reason)

    def verify_token(self, presented: str, at: str) -> TokenRecord:
        """Resolve a presented plaintext token to its record, refusing
        revoked, expired, and orphaned (minter no longer a member) tokens.
        Hash comparison is constant-time. One refusal message for unknown
        and wrong-secret alike - verification never confirms token ids."""
        secret_hash = _hash_secret(presented)
        token = self._tokens.get(secret_hash[:_TOKEN_ID_CHARS])
        if (
            token is None
            or not presented.startswith(TOKEN_PREFIX)
            or not hmac.compare_digest(token.secret_hash, secret_hash)
        ):
            raise RegistryError("unknown token")
        if token.revoked:
            raise RegistryError(f"token {token.token_id} was revoked: {token.revoked_reason}")
        if _parse_at(at, "verification time") >= _parse_at(token.expires_at, "token expiry"):
            raise RegistryError(f"token {token.token_id} expired at {token.expires_at}")
        if (token.publisher, token.minted_by) not in self._memberships:
            raise RegistryError(
                f"token {token.token_id} was minted by {token.minted_by!r}, "
                f"who is no longer a member of {token.publisher!r}"
            )
        return token

    def authorize_token(self, token: TokenRecord, publisher: str, pack: str) -> None:
        """Refuse unless this verified token may act on ``pack`` under
        ``publisher``. This checks the publisher and optional per-pack
        narrowing; role authorization remains a separate required check."""
        canonical = canonical_name(publisher)
        if token.publisher != canonical:
            raise RegistryError(
                f"token {token.token_id} belongs to publisher {token.publisher!r}, "
                f"not {canonical!r}"
            )
        if token.pack is not None and token.pack != canonical_name(pack):
            raise RegistryError(
                f"token {token.token_id} is scoped to pack {token.pack!r}, "
                f"not {canonical_name(pack)!r}"
            )

    # -- internals --------------------------------------------------------

    def _require_user(self, user: str) -> None:
        if user not in self._users:
            raise RegistryError(f"unknown user {user!r}; register the account first")

    def _require_publisher(self, publisher: str) -> tuple[str, str]:
        canonical = canonical_name(publisher)
        display = self._publishers.get(canonical)
        if display is None:
            raise RegistryError(f"unknown publisher {publisher!r}")
        return canonical, display

    def _guard_last_owner(self, publisher: str, user: str) -> None:
        owners = [
            m
            for (p, _), m in self._memberships.items()
            if p == publisher and m.role == "owner" and m.user != user
        ]
        if not owners:
            raise RegistryError(
                f"user {user!r} is the last owner of {publisher!r}; a publisher "
                f"always has at least one owner - add another owner first"
            )

    def _revoke(self, token: TokenRecord, actor: str, at: str, reason: str) -> None:
        self._tokens[token.token_id] = replace(token, revoked=True, revoked_reason=reason)
        self._record(
            actor, "revoke-token", token.publisher, at, f"token {token.token_id}: {reason}"
        )

    def _record(self, actor: str, action: str, subject: str, at: str, details: str = "") -> None:
        _parse_at(at, "audit timestamp")
        if not actor:
            raise RegistryError("every registry mutation records its individual actor")
        self._audit.append(AuditRecord(actor, action, subject, at, details))


__all__ = [
    "TOKEN_PREFIX",
    "Action",
    "AuditRecord",
    "Membership",
    "PrincipalDirectory",
    "Role",
    "TokenRecord",
]
