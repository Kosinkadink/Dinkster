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
