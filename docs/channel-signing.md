# Release channel signing

Channel signing authenticates Dinkster's choice of release artifacts, not the
mirror serving them. The channel private key belongs only to the channel
publication step in the Dinkster release workflow. Mirrors, Desktop,
installers, the registry service and Git source checkouts never receive it.

## Signed bytes and trust

A channel uses the same Ed25519 envelope as release security notices:
`{"signed": {...}, "signatures": [{"keyId": "sha256:...", "signature": "..."}]}`.
The key identifier is SHA-256 of the raw 32-byte public key. The signature is
canonical padded standard base64 of the 64 Ed25519 signature bytes. The
signature covers the complete `signed` object, including `format` and the
requested `channel` (`stable` or `nightly`). There is no `main` channel alias.
Git branch names and existing `main-...` release tags are not channel names.

The shared security-notice canonicalizer serializes ASCII JSON with sorted
object keys, compact separators and no nonfinite numbers. Channel transport
is that exact canonical envelope, with no trailing newline or extra
whitespace. This also makes duplicate-field encodings unambiguous in readers
without a duplicate-aware JSON parser. Channel numbers must be integers
between -9007199254740991 and 9007199254740991; unknown signed fields are
covered by the signature, not discarded before authentication.

Environment tools and Desktop embed independently provisioned public keys.
The pinned channel root has key ID
`sha256:4b5b2c78b32db1ebedc4702a92d6e4e252c17531f0e102ab82dd910be6dec7b9`.
Its raw public key is `m6rm7XY806I6eifUgiElbaK4nYkWQlDU+WjdsH8Eb9Q=`
in base64. Publication self-verifies the final envelope against the pinned
roots before uploading; a different workflow key refuses publication.
A mirror-provided key cannot authorize itself. The registry security-notice
key is a different authority: it cannot sign channels, and the channel key
cannot issue registry notices. Both use the same canonical Ed25519 envelope
and signature-verification rules. The notice schema and its incident,
revision, expiry and ownership rules are unchanged.

After authentication, every referenced manifest, wheel and archive still
requires its recorded digest and size. A signature does not mean that code is
safe to execute. It does not add a sandbox or authorize registry notices.
Unsigned channels and signed channels from unknown keys are refused even on
Dinkster-operated hosts. Credential-free HTTPS mirrors use exactly the same
signature check; loopback transports and local copies do not bypass it.

Only original authenticated channel bytes belong in the remote metadata
cache. An explicit local frontend/source selection is a user overlay, not a
new producer signature. Historical installed generations retain their
existing sealed exact provenance; they are not relabeled as newly signed or
silently replaced with today's channel during restoration. Git refs remain
explicit unpublished source choices, not release-channel statements.

## Generate and provision a key

Run these commands on a trusted machine with OpenSSL and an authenticated
GitHub CLI. OpenSSL prompts for the encryption/decryption passphrase without
echoing it. Decrypted private bytes go directly to the Actions secret through
stdin, never through command arguments, terminal output or the checkout.
`set -euo pipefail` stops after a failed generation or upload. Do not paste
private-key bytes, encrypted private files, passphrases or tokens into issues.

```bash
set -euo pipefail
umask 077
key_dir="$(mktemp -d "${TMPDIR:-/tmp}/dinkster-channel-key.XXXXXXXX")"
openssl genpkey -algorithm ED25519 -aes-256-cbc -out "$key_dir/channel-key.pem"
openssl pkey -in "$key_dir/channel-key.pem" -pubout -out "$key_dir/channel-public.pem"
openssl pkey -pubin -in "$key_dir/channel-public.pem" -text -noout
openssl pkey -in "$key_dir/channel-key.pem" \
  | gh secret set DINKSTER_CHANNEL_SIGNING_KEY --repo Kosinkadink/Dinkster
rm -- "$key_dir/channel-key.pem"
printf 'Stored DINKSTER_CHANNEL_SIGNING_KEY; encrypted temporary private file removed.\n'
cat -- "$key_dir/channel-public.pem"
```

Expected output contains only public key material and success messages.
After success, the Actions secret is the only retained private-key copy.
The user supplies the PUBLIC KEY PEM for independent review and pinning in
both environment tools and Desktop. Preserve that public file as the record
of the trust root. If the upload fails, retain only the encrypted temporary
private file until the trusted-machine retry succeeds, then remove it.
Do not request or print the secret's value to validate provisioning: validate
its public identity with an actual signed release and both readers instead.

## Normal rotation

1. Generate a replacement key on a trusted machine using the commands above,
   but STOP before the `openssl pkey ... | gh secret set` pipeline. Keep the
   replacement encrypted private file until the overlap releases are ready.
   Before changing the active secret, distribute tools and Desktop releases
   that pin both the current and replacement PUBLIC keys. Independently verify
   the replacement fingerprint and the pinned bytes in each release.
2. The user replaces `DINKSTER_CHANNEL_SIGNING_KEY` using hidden prompts and
   stdin, as above. Publish through the ordinary gated release workflow and
   read back the exact signed `stable` and `nightly` objects. Verify both with
   the independently pinned replacement public key in Linux and Windows
   environment tools and Desktop.
3. Release readers that remove the former public key. They refuse old-key
   remote channels and cached channels on their next verification. Confirm
   current channels and the channel-selected tools no longer require the old
   key before removing the overlap. An old reader that knows only the old
   key must be upgraded through an independently authenticated distribution;
   it must not accept a new key merely because a mirror supplies it.

This procedure rotates trust through the existing software distribution,
not a new mirror-controlled trust service. Do not pass channel authority to
the registry's signed-notice trust document or change notice authority to
solve a channel-key rotation.

## Lost or leaked key

If only the private key is lost, existing signatures remain verifiable but
no new channel may be signed with that key. Generate and independently pin
a replacement through a tools/Desktop update, then publish with the
replacement. A missing secret is not permission to publish an unsigned
channel or accept a mirror-provided replacement key.

If a key may be leaked, stop channel publication and have the user revoke or
replace its Actions secret. Treat attacker-created signatures by that key
as valid cryptographic signatures but untrusted release authority. Ship
independently authenticated tools/Desktop updates that REMOVE the leaked
key rather than trusting a rotation statement signed only by it. Check and
republish the intended channel objects with the replacement key. Reverify
or discard affected remote caches with the new pinned keys; do not weaken
signature, digest or size checks to recover. Preserve the public fingerprint
and affected channel/artifact identities for incident investigation.

Already-installed sealed generations do not change automatically when a key
is removed. Identify which artifacts were acquired during the compromise
and replace or explicitly refuse affected installations. Existing signed
release security notices can classify affected pack artifacts using their
current authority and exact-digest rules; they are not a general engine or
channel revocation format. Never invent a notice schema or auto-resume
blocked execution during recovery.

Signatures alone do not prove freshness. An isolated old reader can still
verify an old valid channel with a key it has not learned to distrust. A
mirror cannot be trusted to deliver its own revocation reliably; recovery
requires an independently authenticated reader update. Do not describe
cached/offline verification as automatic protection against a later key
compromise or replay of an older valid channel.
