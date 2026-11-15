# Production service addresses

- Production service resolution, explicit custom catalog/job endpoints and empty-catalog disabling: 2 tests pass.
- Service resolution, CLI and serving integration: 73 tests pass on the managed linux-cpu contributor generation.
- Extension-kind generation, extension factories, builtin type spelling, Ruff lint/format and Linux/Windows-target Pyright pass.
- Account/service/CLI/serving integration: 80 passed in 43.14 seconds on the managed linux-cpu contributor generation. Local identity HTTP fixtures cover PKCE/state/issuer checks, consent denial, paginated projects, selected-project mint, OAuth rotation, revocation, redirect refusal, private credentials outside projects and endpoint/install isolation.
- Full Ruff lint/format and Linux/Windows-target Pyright pass with account source included.
- Account/service/CLI/serving/composition/sandbox integration: 167 passed in 69.15 seconds. The real engine refreshes its selected install bearer at startup and periodically. A real bubblewrap worker observes atomic bearer replacement while OAuth state stays hidden. Lost OAuth-refresh replies invalidate local state and are never replayed automatically.
- These checks do not establish live identity bridge availability or authenticated catalog integration with the remote pack.

# Managed Git sources

- Public engine, pack, inference/helper and frontend Git lifecycles retain installed base/wheels, disclose exact source identities, switch refs and restore recorded commits. [Linux and Windows evidence and whole-backend verification](https://github.com/Kosinkadink/comfy-vibe-station/issues/297#issuecomment-6893348832).
- Native Desktop confirmation installs the verified published frontend wheel and refuses an incompatible API before creating project/store state. [Measured behavior and fixture boundaries](https://github.com/Kosinkadink/comfy-vibe-station/issues/297#issuecomment-6893330604).
- Seven source lifecycle and frontend API-boundary cases pass with the Nightly channel contract. Assertions remain unchanged; the cases exercise both API boundaries and both refusals, exact rollback, dependency acquisition and retained sources.
- Desktop bundled tools integration and its whole-unit acceptance are separate from backend verification.

# Contributor wheel acquisition

- Real linux-cpu resolver: 14 selected wheels retained, including 10 wheel rows without sizes; verified files carry measured byte counts.
- Regression against original source: missing-size wheel fails on the separate HEAD request. Changed source: acquisition succeeds, disconnected reuse works, and corrupt cached/remote bytes are rejected.
- Downloader suite: 32 passed. Downloader plus development inference generation cases: 36 passed, 33 deselected.
- Full contributor setup with explicit Python 3.12 and isolated port 15971 passed; its engine was stopped through the managed CLI.
- Full Ruff format/check passed. Managed contributor Pyright on Linux and Windows targets: zero errors.
- No tests removed, new skips/retries, relaxed deadlines or Actions reruns.
