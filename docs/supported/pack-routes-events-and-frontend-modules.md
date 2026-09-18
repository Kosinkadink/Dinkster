## Pack routes, events, and frontend modules

- Packs can declare authenticated GET/POST JSON routes and typed execution
  events without importing server internals. Routes live in host-owned pack
  namespaces; events retain pack, worker, node, and native execution identity.
  Integer payload values use JavaScript's safe integer range; finite floating
  point values remain supported in number fields.
- Installed packs can publish snapshot-selected immutable JavaScript modules.
  Snapshot and module reads do not start execution workers. Frontend privileges
  and controls do not grant or remove backend node execution ownership.
- Packs can publish validated locale catalogs for node, blueprint, and guide
  text. Catalog descriptors are available on schema wire 44, and their exact
  digest-addressed JSON bytes are served as immutable pack resources.
- The opt-in `dinkster-video-preview` pack initializes bounded VHS-style VIDEO
  metadata, exposes its preview policy, and supplies a declared event consumer
  and host-rendered metadata status for compatible frontends.
