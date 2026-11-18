# Release channel signing

Channel signing authenticates Dinkster's choice of release artifacts, not the
mirror serving them. The private key belongs only to the channel publication
steps in the Dinkster release workflows. Mirrors, Desktop, installers, the
registry service and Git source checkouts never receive it.

## Signed bytes and trust

Channels reuse the release security notices' Ed25519 envelope:
`{"signed": {...}, "signatures": [{"keyId": "sha256:...", "signature": "..."}]}`.
The key ID is SHA-256 of the raw 32-byte public key. Signatures use canonical
padded standard base64 of the 64 Ed25519 signature bytes and cover the complete
`signed` object, including its `format` and requested `channel` (`stable` or
`nightly`). There is no `main` channel alias. Git branches and existing
`main-...` release tags are not channel names.

The shared notice canonicalizer produces ASCII JSON with sorted object keys,
compact separators and no nonfinite numbers. Channel transport is that exact
canonical envelope, without trailing whitespace. Readers reject noncanonical
transport, including duplicate fields. Channel numbers must be integers in
[-9007199254740991, 9007199254740991]. Unknown signed fields remain covered.

Environment tools and Desktop pin independently provisioned public keys. The
channel root is `sha256:4b5b2c78b32db1ebedc4702a92d6e4e252c17531f0e102ab82dd910be6dec7b9`,
raw public base64 `m6rm7XY806I6eifUgiElbaK4nYkWQlDU+WjdsH8Eb9Q=`. Publication
self-verifies before upload; a different workflow key refuses publication.
Mirror-provided keys cannot authorize themselves. Notice and channel keys are
separate authorities, despite sharing cryptographic code: notice keys cannot
sign channels and channel keys cannot issue notices. Notice schema, incident,
revision, expiry and ownership rules are unchanged.

Unsigned, tampered, wrong-channel and unknown-key channels are refused even on
official hosts, loopback transports and local copies. Credential-free HTTPS
mirrors use the same signature check. Referenced manifests, wheels and archives
still require their recorded digests and sizes. Signatures do not sandbox code.

Only original authenticated envelopes enter remote metadata caches. User
frontend/source overlays are not producer signatures. Historical sealed exact
provenance remains historical; restoring it does not relabel it as newly signed
or silently substitute today's channel. Git refs remain explicit unpublished
source choices.

## Initial publication

The Stable workflow publishes the matching strict tools, signs and self-verifies
Stable, then verifies exact public readback. Missing or legacy unsigned Nightly
is initialized solely from authenticated Stable selections, bound to `nightly`,
signed and publicly verified under the same publication lock before strict
maintenance runs. No unsigned predecessor engine, tools, frontend or date field
is inherited. An existing valid signed Nightly is preserved; an invalid signed
one is an error, not permission to overwrite or downgrade it. Ordinary Nightly
writers verify the predecessor and sign final selections. Consumers have no
unsigned initialization bypass.

## Generate and provision a key

The user runs this on a trusted machine with OpenSSL and authenticated GitHub
CLI. OpenSSL asks for encryption/decryption passphrases without echo. Decrypted
private bytes flow directly through stdin, not arguments, terminal output or a
checkout. Stop on any failure; do not paste private bytes, encrypted private
files, passphrases or tokens into an issue or thread.

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

Expected output is public material and success messages only. After success,
the Actions secret is the only retained private-key copy. Preserve the public
file and independently verify its fingerprint and pinned bytes in environment
tools and Desktop. On upload failure retain only the encrypted private file
until trusted-machine retry succeeds, then remove it. Validate provisioning by
an actual signed release and both readers, never by reading the secret.

## Normal rotation

1. Generate a replacement with the commands above, but stop before the
   `openssl pkey ... | gh secret set` pipeline. Retain the encrypted private
   file. Distribute tools and Desktop that pin both old and replacement public
   keys; independently verify the fingerprint and bytes in each release.
2. Only then have the user replace `DINKSTER_CHANNEL_SIGNING_KEY` with the
   hidden-prompt/stdin pipeline. Publish through the ordinary gated workflow.
   Read back both public channels and verify them with the replacement root
   using Linux and Windows tools and Desktop.
3. Distribute readers removing the former root. Confirm channels and their
   selected tools no longer need it. Old-key remote and cached channels then
   fail verification. Old readers must upgrade through independently
   authenticated distribution, not accept a mirror-provided replacement key.

Rotation uses software distribution, not a second trust service. Do not give
channel authority to the registry's notice trust document or alter notice
ownership to rotate a channel key.

## Lost or leaked key

A lost private key leaves existing signatures verifiable but cannot sign new
channels. Generate and independently pin a replacement through tools/Desktop
updates, then publish with it. A missing secret never permits unsigned output.

For a suspected leak, stop publication and have the user revoke or replace its
Actions secret. Attacker signatures remain cryptographically valid, but are no
longer trusted release authority. Ship independently authenticated reader
updates that remove the leaked key; a rotation statement signed solely by that
key is insufficient. Republish intended selections with the replacement and
reverify or discard remote caches. Preserve the public fingerprint and affected
channel/artifact identities for investigation. Never weaken digest, size or
signature checks for recovery.

Installed sealed snapshots do not automatically change when a key is removed.
Identify acquisitions during compromise and replace or explicitly refuse
affected installations. Existing signed security notices can classify affected
pack artifacts under their existing authority and exact-digest rules; they are
not an engine or channel revocation schema. Do not invent a parallel notice
format or automatically resume blocked execution.

Signatures alone do not prove freshness. An isolated old reader can verify an
old channel with a root it has not learned to distrust. Mirrors cannot reliably
revoke themselves; recovery requires an independently authenticated reader
update. Cached/offline verification is not automatic protection against later
key compromise or replay.
