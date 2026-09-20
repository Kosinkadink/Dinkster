## Pack routes, events, and frontend modules

- Packs can declare authenticated GET/POST JSON routes and typed execution
  events without importing server internals. Routes live in host-owned pack
  namespaces; events retain pack, worker, node, and native execution identity.
  Integer payload values use JavaScript's safe integer range; finite floating
  point values remain supported in number fields.
- Installed packs can publish snapshot-selected immutable JavaScript modules.
  Snapshot and module reads do not start execution workers. Frontend privileges
  and controls do not grant or remove backend node execution ownership.
- Doctor warns when a pack declares a frontend contribution kind or extension
  capability for which no runtime consumer exists.
- Isolated packs can publish value renditions whose metadata, parameter
  normalization, MIME selection, and rendering remain pack-owned while being
  available through the host value API.
- Packs can publish validated locale catalogs for node, blueprint, and guide
  text. Their exact digest-addressed JSON bytes are served as immutable pack
  resources.
- The opt-in `dinkster-video-preview` pack initializes bounded VHS-style VIDEO
  metadata, exposes its preview policy, and supplies a declared event consumer
  and host-rendered metadata status for compatible frontends.
