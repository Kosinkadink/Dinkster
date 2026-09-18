"""Registry principals (DESIGN M8, publisher identity).

What this proves: each recorded Comfy-Org/registry-backend identity gap
is unrepresentable - roles that gate nothing, tokens that identify
nobody (plaintext, unscoped, non-expiring), separator-variant publisher
spoofing, orphaned tokens surviving membership revocation, and display
edits with no attribution trail.
"""

from __future__ import annotations

import pytest
from dinkster_registry import (
    TOKEN_PREFIX,
    PrincipalDirectory,
    RegistryError,
)

T0 = "2026-07-01T00:00:00+00:00"
T1 = "2026-07-02T00:00:00+00:00"
T2 = "2026-07-03T00:00:00+00:00"
EXPIRY = "2026-08-01T00:00:00+00:00"
AFTER_EXPIRY = "2026-09-01T00:00:00+00:00"


def directory() -> PrincipalDirectory:
    d = PrincipalDirectory()
    d.register_user("alice")
    d.register_user("bob")
    d.register_publisher("img-tools", owner="alice", at=T0)
    return d


# ---------------------------------------------------------------------------
# Principals and memberships
# ---------------------------------------------------------------------------


def test_users_and_publishers_are_separate_principals() -> None:
    d = directory()
    # "alice" the user exists; "alice" is not thereby a publisher.
    with pytest.raises(RegistryError, match="unknown publisher"):
        d.display_name("alice")
    assert d.role_of("alice", "img-tools") == "owner"


def test_publisher_ids_ride_the_one_grammar_and_canonicalize() -> None:
    d = directory()
    # Separator variants are ONE publisher - no spoof by re-registration.
    with pytest.raises(RegistryError, match="already exists"):
        d.register_publisher("img_tools", owner="bob", at=T1)
    with pytest.raises(RegistryError, match="grammar|lowercase|must"):
        d.register_publisher("Img-Tools!", owner="alice", at=T1)
    assert d.role_of("alice", "img.tools") == "owner"  # canonical lookup


def test_publisher_requires_registered_owner() -> None:
    d = PrincipalDirectory()
    with pytest.raises(RegistryError, match="unknown user"):
        d.register_publisher("img-tools", owner="ghost", at=T0)


def test_roles_gate_operations_at_one_enforcement_point() -> None:
    d = directory()
    d.add_member("img-tools", "bob", "member", actor="alice", at=T1)
    # A member publishes and mints, and nothing else.
    d.authorize("bob", "img-tools", "publish")
    d.authorize("bob", "img-tools", "mint-token")
    for action in ("edit-metadata", "manage-members", "transfer"):
        with pytest.raises(RegistryError, match="requires owner"):
            d.authorize("bob", "img-tools", action)  # type: ignore[arg-type]
    # A non-member does nothing at all.
    d.register_user("carol")
    with pytest.raises(RegistryError, match="not a member"):
        d.authorize("carol", "img-tools", "publish")


def test_membership_mutations_require_manage_members() -> None:
    d = directory()
    d.add_member("img-tools", "bob", "member", actor="alice", at=T1)
    d.register_user("carol")
    with pytest.raises(RegistryError, match="requires owner"):
        d.add_member("img-tools", "carol", "member", actor="bob", at=T2)
    with pytest.raises(RegistryError, match="requires owner"):
        d.remove_member("img-tools", "alice", actor="bob", at=T2)


def test_membership_mutations_reject_unknown_roles() -> None:
    d = directory()
    with pytest.raises(RegistryError, match="unknown membership role"):
        d.add_member("img-tools", "bob", "admin", actor="alice", at=T1)  # type: ignore[arg-type]
    d.add_member("img-tools", "bob", "member", actor="alice", at=T1)
    with pytest.raises(RegistryError, match="unknown membership role"):
        d.set_role("img-tools", "bob", "admin", actor="alice", at=T1)  # type: ignore[arg-type]
    assert d.role_of("bob", "img-tools") == "member"


def test_last_owner_cannot_be_removed_or_demoted() -> None:
    d = directory()
    with pytest.raises(RegistryError, match="last owner"):
        d.remove_member("img-tools", "alice", actor="alice", at=T1)
    with pytest.raises(RegistryError, match="last owner"):
        d.set_role("img-tools", "alice", "member", actor="alice", at=T1)
    # With a second owner both operations work.
    d.add_member("img-tools", "bob", "owner", actor="alice", at=T1)
    d.set_role("img-tools", "alice", "member", actor="alice", at=T2)
    assert d.role_of("alice", "img-tools") == "member"


def test_display_is_mutable_ids_are_not_and_edits_are_audited() -> None:
    d = directory()
    d.set_display_name("img-tools", "Image Tools", actor="alice", at=T1)
    assert d.display_name("img-tools") == "Image Tools"
    record = d.audit()[-1]
    assert record.actor == "alice"
    assert record.action == "edit-metadata"
    assert "'img-tools' -> 'Image Tools'" in record.details


def test_every_mutation_records_the_individual_actor() -> None:
    d = directory()
    d.add_member("img-tools", "bob", "member", actor="alice", at=T1)
    d.mint_token("img-tools", minted_by="bob", at=T1, expires_at=EXPIRY, secret="s1")
    actions = [(r.actor, r.action) for r in d.audit()]
    assert ("alice", "register-publisher") in actions
    assert ("alice", "manage-members") in actions
    assert ("bob", "mint-token") in actions
    d.register_user("carol")
    with pytest.raises(RegistryError, match="ISO 8601"):
        d.add_member("img-tools", "carol", "member", actor="alice", at="yesterday-ish")
    # The refused mutation left no membership behind.
    assert d.role_of("carol", "img-tools") is None


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_token_plaintext_exists_only_at_mint_and_hash_at_rest() -> None:
    d = directory()
    plaintext, record = d.mint_token("img-tools", minted_by="alice", at=T0, expires_at=EXPIRY)
    assert plaintext.startswith(TOKEN_PREFIX)
    assert plaintext not in repr(record)
    assert record.secret_hash != plaintext
    assert d.verify_token(plaintext, at=T1) == record


def test_tokens_must_expire() -> None:
    d = directory()
    with pytest.raises(RegistryError, match="must be after mint time"):
        d.mint_token("img-tools", minted_by="alice", at=T1, expires_at=T1)
    plaintext, _ = d.mint_token("img-tools", minted_by="alice", at=T0, expires_at=EXPIRY)
    with pytest.raises(RegistryError, match="expired"):
        d.verify_token(plaintext, at=AFTER_EXPIRY)


def test_unknown_and_wrong_secret_get_one_refusal() -> None:
    d = directory()
    d.mint_token("img-tools", minted_by="alice", at=T0, expires_at=EXPIRY, secret="s1")
    for presented in ("", "garbage", TOKEN_PREFIX + "wrong"):
        with pytest.raises(RegistryError, match="^unknown token$"):
            d.verify_token(presented, at=T1)


def test_token_scope_is_publish_only_and_optionally_per_pack() -> None:
    d = directory()
    plaintext, record = d.mint_token(
        "img-tools", minted_by="alice", at=T0, expires_at=EXPIRY, pack="img-tools-extra"
    )
    verified = d.verify_token(plaintext, at=T1)
    d.authorize_token(verified, "img-tools", "img_tools.extra")  # separator-blind
    with pytest.raises(RegistryError, match="scoped to pack"):
        d.authorize_token(verified, "img-tools", "other-pack")
    with pytest.raises(RegistryError, match="belongs to publisher"):
        d.authorize_token(verified, "someone-else", "img-tools-extra")
    # Unscoped token publishes any pack of its publisher.
    wide, _ = d.mint_token("img-tools", minted_by="alice", at=T0, expires_at=EXPIRY, secret="s2")
    d.authorize_token(d.verify_token(wide, at=T1), "img-tools", "anything")


def test_membership_revocation_invalidates_the_members_tokens() -> None:
    d = directory()
    d.add_member("img-tools", "bob", "member", actor="alice", at=T0)
    bob_token, _ = d.mint_token("img-tools", minted_by="bob", at=T0, expires_at=EXPIRY)
    alice_token, _ = d.mint_token("img-tools", minted_by="alice", at=T0, expires_at=EXPIRY)
    d.remove_member("img-tools", "bob", actor="alice", at=T1)
    with pytest.raises(RegistryError, match="revoked.*removed from publisher"):
        d.verify_token(bob_token, at=T2)
    assert d.verify_token(alice_token, at=T2).minted_by == "alice"
    # The revocation itself is on the audit trail, attributed to the actor.
    assert any(
        r.action == "revoke-token" and r.actor == "alice" and "removed" in r.details
        for r in d.audit()
    )


def test_token_revocation_requires_a_reason_and_gates_on_actor() -> None:
    d = directory()
    d.add_member("img-tools", "bob", "member", actor="alice", at=T0)
    plaintext, record = d.mint_token("img-tools", minted_by="alice", at=T0, expires_at=EXPIRY)
    with pytest.raises(RegistryError, match="explicit reason"):
        d.revoke_token(record.token_id, actor="alice", at=T1, reason="")
    # A member cannot revoke someone else's token...
    with pytest.raises(RegistryError, match="requires owner"):
        d.revoke_token(record.token_id, actor="bob", at=T1, reason="nope")
    # ...but can revoke their own.
    _, own = d.mint_token("img-tools", minted_by="bob", at=T0, expires_at=EXPIRY, secret="s3")
    d.revoke_token(own.token_id, actor="bob", at=T1, reason="rotating")
    d.revoke_token(record.token_id, actor="alice", at=T1, reason="leaked")
    with pytest.raises(RegistryError, match="revoked: leaked"):
        d.verify_token(plaintext, at=T2)


def test_minting_requires_membership() -> None:
    d = directory()
    d.register_user("carol")
    with pytest.raises(RegistryError, match="not a member"):
        d.mint_token("img-tools", minted_by="carol", at=T0, expires_at=EXPIRY)


def test_token_listing_never_exposes_plaintext() -> None:
    d = directory()
    plaintext, _ = d.mint_token("img-tools", minted_by="alice", at=T0, expires_at=EXPIRY)
    (record,) = d.tokens("img-tools")
    assert plaintext not in (record.token_id, record.secret_hash)
    assert record.minted_by == "alice"
