## Peer-to-peer host support

- The host can run one isolated libtorrent 2.1.1 sidecar per active
  installation and asset vault on Linux x86-64/AArch64, Windows AMD64, and
  macOS ARM64 with CPython 3.12 or 3.13. Downloading and seeding are separate
  persisted settings and both default off with LAN and Internet scope.
  Enabling P2P starts both; saved choices remain authoritative, and
  `--disable-p2p` disables both at startup. The optional P2P plugin will make
  P2P default on once it can show and stop seeding explicitly.
- The sidecar exposes local status and host-only download/seed lease controls,
  uses private authenticated IPC, persists resume state, and pauses with no
  leases after corrupt state. LAN discovery stays local; DHT, PEX, and NAT
  mappings open only with current trusted global authority. Resolver transport
  is trackerless.
