# Production service addresses

- Production service resolution, explicit custom catalog/job endpoints and empty-catalog disabling: 2 tests pass.
- Service resolution, CLI and serving integration: 73 tests pass on the managed linux-cpu contributor generation.
- Extension-kind generation, extension factories, builtin type spelling, Ruff lint/format and Linux/Windows-target Pyright pass.
- Account/service/CLI/serving integration: 80 passed in 43.14 seconds on the managed linux-cpu contributor generation. Local identity HTTP fixtures cover PKCE/state/issuer checks, consent denial, paginated projects, selected-project mint, OAuth rotation, revocation, redirect refusal, private credentials outside projects and endpoint/install isolation.
- Full Ruff lint/format and Linux/Windows-target Pyright pass with account source included.
- Account/service/CLI/serving/composition/sandbox integration: 167 passed in 69.15 seconds. The real engine refreshes its selected install bearer at startup and periodically. A real bubblewrap worker observes atomic bearer replacement while OAuth state stays hidden. Lost OAuth-refresh replies invalidate local state and are never replayed automatically.
- Both default pack helper commands retain disabled discovery when endpoints are omitted or explicitly empty, and honor explicit custom endpoints: six public CLI cases pass against a real fixture pack. The omitted-endpoint cases fail on the previous helper implementation.
- These checks do not establish live identity bridge availability or authenticated catalog integration with the remote pack.

# Published remote catalog defaults

- Registry default resolution requires remote pack 0.0.7 or newer. The public static registry resolved and delivered 0.0.7 with verified wheel bytes and all six published native install locks.
- Installer, archive-admission and composition CPU suites: 273 passed, 2 existing skips. Full fast checks passed, including Ruff and Linux/Windows Pyright.
- Authenticated catalog/policy/cache behavior is covered by the remote pack's local and selected-wheel admission suites. This default selection does not claim deployed identity sign-in or provider/billing acceptance.

# Signed release channels

- CPU contributor setup passed. Generated extension checks, Ruff format/lint and Linux/Windows Pyright passed.
- Environment discovery, acquisition, CLI, collection, generations, history, tools, transactions and launch: 404 passed, 3 skipped in 711.08 seconds.
- Installed-wheel unavailable-update case reproduced on the unchanged base; its obsolete older-version fallback assertion now requires exact-target source refusal. No-mutation and resolver-trap checks remain. Corrected history plus eight registry controls: 9 passed.
- Independent tools publisher: signed selection preserved; unsigned and tampered channels refused, 3 passed. Stable/Nightly publication controls verify signatures, daily selection and readback.
- Live catalog bootstrap used the published Stable supervisor rather than the strict checkout decoder: 646 nodes and 37 packs, no composing state. Wrong digest/size, absent digest and prerelease bootstrap inputs were refused.
- Public signed release, Windows/Linux Desktop and final cross-repository acceptance remain unverified; this is not a publication receipt.

# Bundled environment tool compatibility

- Relocated native tools run their own uv through the Windows CPython preparation version guard, followed by offline setup, update and rollback. The regression refused the previous uv 0.11.29 bundle; all 10 tools cases passed with locked uv 0.13.6.
- Native tools publication checks the same shipped executable on Linux, Windows and macOS before uploading its archive. Installed Desktop Windows development acceptance remains a separate hosted gate.

# Managed Git sources

- Public engine, pack, inference/helper and frontend Git lifecycles retain installed base/wheels, disclose exact source identities, switch refs and restore recorded commits. [Linux and Windows evidence and whole-backend verification](https://github.com/Kosinkadink/comfy-vibe-station/issues/297#issuecomment-6893348832).
- Native Desktop confirmation installs the verified published frontend wheel and refuses an incompatible API before creating project/store state. [Measured behavior and fixture boundaries](https://github.com/Kosinkadink/comfy-vibe-station/issues/297#issuecomment-6893330604).
- Seven source lifecycle and frontend API-boundary cases pass with the Nightly channel contract. Assertions remain unchanged; the cases exercise both API boundaries and both refusals, exact rollback, dependency acquisition and retained sources.
- Desktop bundled tools integration and its whole-unit acceptance are separate from backend verification.

# Development Git frontend

- Public Git engine/frontend selection installed the verified CI frontend wheel without an interpreter override; served HTML matched generation assets and survived stop/start.
- Exact restore reused the retained frontend and development metadata without fetching Git or rebuilding the frontend, including a project with no packs.
- Public development CLI coverage checks selected frontend HTML for both verified CI and local builds, corrupt archive refusal and missing Node/pnpm refusal before activation. The local producer and CI download boundaries are controlled fixtures, not another real frontend build claim.
- Existing source-approval coverage checks changed Git refs, changed CI artifact identities, local-build selection changes and incompatible frontend API refusal. Existing shared-library mismatch assertions remain intact.

# Contributor wheel acquisition

- Real linux-cpu resolver: 14 selected wheels retained, including 10 wheel rows without sizes; verified files carry measured byte counts.
- Regression against original source: missing-size wheel fails on the separate HEAD request. Changed source: acquisition succeeds, disconnected reuse works, and corrupt cached/remote bytes are rejected.
- Downloader suite: 32 passed. Downloader plus development inference generation cases: 36 passed, 33 deselected.
- Full contributor setup with explicit Python 3.12 and isolated port 15971 passed; its engine was stopped through the managed CLI.
- Full Ruff format/check passed. Managed contributor Pyright on Linux and Windows targets: zero errors.
- No tests removed, new skips/retries, relaxed deadlines or Actions reruns.
