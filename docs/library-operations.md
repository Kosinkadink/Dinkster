# Library helper operations

## Client lifetime and recovery

Use a caller-owned `LibraryClient` context for synchronous work, or
`AsyncLibraryClient.start(library)` and `close()` for event-loop consumers.
Consumers given a client must not close it on the owner's behalf. Keep calls
on a `ProjectFolders` instance serialized. Managed launches select the tools
interpreter through `DINKSTER_LIBRARY_COMMAND`; do not select an older helper
merely because an engine snapshot is restored.

After a timeout or malformed reply, reopen the client and reread state before
another write. A missing reply does not prove the previous write failed.
Do not blindly replay a mutation against an uncertain revision or outcome.
For API changes and tools replacement, follow the existing
[API register and removal procedure](library-api.md).

## Trusted engine responsibilities

The private pipe trusts its owning engine, not a network caller. Before
publishing records, the engine must validate runtime schemas, operator
permissions, filesystem grants, resolver source/URL/provider policy and byte
identity at their respective boundaries. In particular:

- Verify upload/download bytes, including the strict latent parser, before
  persisting media grants. Stored metadata alone is not publication authority.
- Verify the public HTTPS chain, fetched digest/size and trusted declaration
  before recording an acquisition receipt. A receipt is evidence of past
  acquisition, not media authority or a request to retain bytes.
- Authorize operator routes using `assets.recorded`; global inventory
  membership is not permission to read a scoped asset. Inventory and verify
  vault files on the host before excluding already-recorded digests in bounded
  calls. Ingest history and provenance leads grant no byte access.
- Enforce delegation expiry, owner permissions, human-session freshness and
  principal administration outside the storage lookup.
- Read verified workflow bytes to compute current requirements rather than
  trusting cached attribution from a different document digest.
- Validate known settings and filter unknown fields before applying them to a
  runtime; preserve the raw stored extensions for newer engines. Folder
  preferences, model selections and stored paths do not themselves grant
  filesystem access.

## Reading extensible records

When using the strict `PublicAcquisitionReceiptV1.from_wire` parser, project
only `version`, `receiptId`, `digest`, `sizeBytes`, `sourceType`, `sourceId`,
`sourceRevision`, `listedUrl`, `finalUrl` and `fetchedAt`. Keep the full helper
record separately; do not overwrite it with the v1 parser's `to_wire()` result
or feed it to the strict legacy JSON store.

Likewise, project only `source`, `providerId` and `subscriptionId` into
`_OfficialResolverBootstrap.from_wire`, retaining the raw configuration rather
than replacing it with the parser's strict output. Fetching, resolver semantics
and regional lead derivation belong to the engine before snapshot publication.

## Produced-byte retention and collection

Hold `EngineLifetime(library, engine)` in the engine process until engine exit,
not merely until the helper exits, and pass it to `EngineReferences`. Producers
must mark the manifest incomplete before creating unreported references, then
publish the complete set before relinquishing that protection.

Supply the complete host-assigned produced-vault set, including retained
preview locations. Collection requires both the helper's reference reservation
and the vault claims that protect readers and publishers; neither replaces the
other. Never continue deleting from a saved snapshot after its helper exits.
A writer's vault claim alone proves liveness, not collection authority.

Pass only trusted host-selected absolute vault paths and that engine's assigned
scratch location, never browser-selected paths. Scratch retirement targets one
job attempt; it is not permission to discover or reset other engines' work.
Engine-local temporary cleanup is separate from durable library collection.

File identity must be reliable: nonzero device/inode identities must identify
files correctly. Arbitrary collisions from a faulty filesystem cannot be
detected from that metadata, either for vault alias handling or the account
hash index. Do not treat shipped model-index hints as content proofs or as a
verified filename binding; loader verification remains necessary.
