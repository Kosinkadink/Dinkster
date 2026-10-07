## Peer-to-peer host support

- P2P is disabled by default and is not installed with the base package.
  The optional `p2p` extra supplies dinkster-p2p and libtorrent.
  `features.p2p.enabled` in the library's settings.json defaults to false.
  Without the extra or when disabled, no controller, API routes, LAN mapping
  routes or mDNS advertisement exists. The seed and diagnostics entry points
  exit with "P2P is disabled". Persisted download/seeding choices do not enable
  the feature. `--disable-p2p` also closes the feature boundary at startup.
- The following describes the retained optional implementation, not default
  installation behavior:
- The host can run one isolated libtorrent 2.1.1 sidecar per active
  installation and asset vault on Linux x86-64/AArch64, Windows AMD64, and
  macOS ARM64 with CPython 3.12 or 3.13. Downloading and seeding are separate
  persisted settings and both default off with LAN and Internet scope.
  Enabling transfers requires explicit feature opt-in and the extra; saved
  transfer choices alone never activate it.
- The retained headless seeder implementation operates on read-only existing model stores
  without starting the UI, editor, execution engine, or workers. It verifies
  provider-listed safe files in place, reuses persisted fingerprints after a
  restart, supports a configurable active-seed limit and upload rate, and
  exposes loopback health, readiness, status, and enable/disable controls.
- The sidecar exposes local status and host-only download/seed lease controls,
  uses private authenticated IPC, persists resume state, and pauses with no
  leases after corrupt state. LAN discovery stays local; DHT, PEX, and NAT
  mappings open only with current trusted global authority. Resolver transport
  is trackerless.
