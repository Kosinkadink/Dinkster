# Stable recovery release versions

- Engine/API/workspace 0.0.9 and supervisor 0.0.7 pass isolated Linux CPU contributor setup, generated checks, Ruff format/lint and Linux/Windows Pyright.
- The existing release installation and channel signature suites pass: 20 tests. Tagged and development wheels retain exact workspace pins, complete packaged contents, provenance and unchanged-lock checks.
- The offline lock refresh changes only workspace versions. Public Stable publication and packaged Windows/Linux recovery are not established by these source checks.

# Published pack rollback

- Published snapshot restore with compatibility-range pack locks failed on unchanged main for isolated and shared in-process packs. The repaired six-case CLI regression passes, including incompatible-engine refusal without mutation.
- The complete environment history suite passed: 23 tests on Linux CPU. Generated extension checks, Ruff format/lint and Linux/Windows Pyright passed.
- Public pack lock bytes were verified against registry size/digest and reconstructed installed metadata matched the sealed Linux/Windows generation hashes. Packaged Desktop rollback after tools publication remains unverified.

# Published catalog bootstrap authentication

- The focused bootstrap check fails on unchanged main when a token is set; the exporter now authenticates that API request. Absent and empty tokens preserve unauthenticated behavior. Draft, prerelease, invalid-tag and missing-digest refusals remain intact.
- An authenticated public bootstrap downloaded supervisor 0.0.6 and verified its published SHA-256 and size (265380 bytes). This checks release discovery and wheel integrity, not complete catalog export or packaged Desktop recovery.

# Public Hugging Face model downloads

| Scenario | Host | Result |
| --- | --- | --- |
| Public SD 1.5 consent, cancel, verified named download and full native quickstart | X570 Linux CPU, Python 3.14.10, torch 2.14.1+cpu | [Real browser and native receipt](https://github.com/Kosinkadink/comfy-vibe-station/issues/496#issuecomment-6901157634): HTTP 409/409/202, no download before consent or after Cancel, all seven nodes executed with no cached/skipped nodes, real 512x512 RGB PNG. Provider SHA-256, declared BLAKE3 and persisted source/license verified. |
| Stock ComfyUI loads the same downloaded file from the same folder | X570 Linux CPU, inherited ComfyUI af89add63f71, Python 3.12.13, torch 2.15.0+cpu | [Same-file receipt](https://github.com/Kosinkadink/comfy-vibe-station/issues/496#issuecomment-6901213963): actual Run completed decode/save and a real 512x512 RGB PNG without copying or renaming the checkpoint. Not a numerical or performance-parity claim. |

- Whole Ruff check/format and pyright passed; 196 targeted acquisition/writer/model/mount/receipt tests passed.
- Managed folder-grant restart, runtime grant/browse/revoke and mounted save tests use isolated shared model settings and assert the complete configured model mount set. All four cases reproduce the obsolete two-mount expectation, then pass while retaining path/mode/source, restart, real saved bytes, digest-streaming and revocation checks.
- Gated credentials, Civitai, folder visibility management, byte-progress jobs and export/output model hashes are not accepted by these public-download receipts.

# Production and QA service addresses

- Public registry readback confirms all 36 pinned default versions are active, including remote 0.0.10 with six platform locks. Downloaded remote 0.0.9 bytes passed both hashes and four environment-match/mismatch serving cases. Installed remote 0.0.10 passed both hashes and 16 default/explicit-generation worker cases: unset controls, explicit real mode, all four fake-outcome values and QA outcome-only mode; submission-only headers and retry idempotency are preserved, with generation files unchanged. These local gateway checks do not certify hosted sign-in, provider outcome semantics or fresh published-channel installs.
- Real sandboxed remote workers receive process-local partner mode and fake outcome while preserving bearer rotation/logout and hiding private credentials; all four default-pack/explicit-generation and external-token/account-token cases pass. Without forwarding, both default-pack workers report missing values.
- QA routing and custom endpoint reporting: 46 targeted CLI/account/services/serving/composition checks pass on Linux CPU. Doctor/catalog preparation keeps discovery disabled with no selection, honors explicitly selected production/QA endpoints and preserves empty/custom overrides. Four explicit-service cases fail before the selection repair. Normal serving defaults to production.
- Stable/Nightly project-channel fixtures preserve every project/generation file byte-for-byte across QA then default-production restarts. Actual published fresh Stable/Nightly installation acceptance remains separate.

- Real sandboxed serving preserves explicit external bearer precedence over a selected account, observes atomic replacement and source removal without restarting the worker, and hides neighboring credentials. The same serving test covers account bearer rotation and logout. Before the repair, the explicit-file worker failed its protected-bind check; both cases pass after dedicated-directory binding.
- Production service resolution, explicit custom catalog/job endpoints and empty-catalog disabling: 2 tests pass.
- Service resolution, CLI and serving integration: 73 tests pass on the managed linux-cpu contributor generation.
- Extension-kind generation, extension factories, builtin type spelling, Ruff lint/format and Linux/Windows-target Pyright pass.
- Account/service/CLI/serving integration: 80 passed in 43.14 seconds on the managed linux-cpu contributor generation. Local identity HTTP fixtures cover PKCE/state/issuer checks, consent denial, paginated projects, selected-project mint, OAuth rotation, revocation, redirect refusal, private credentials outside projects and endpoint/install isolation.
- Full Ruff lint/format and Linux/Windows-target Pyright pass with account source included.
- Account/service/CLI/serving/composition/sandbox integration: 167 passed in 69.15 seconds. The real engine refreshes its selected install bearer at startup and periodically. A real bubblewrap worker observes atomic bearer replacement while OAuth state stays hidden. Lost OAuth-refresh replies invalidate local state and are never replayed automatically.
- Both default pack helper commands retain disabled discovery when endpoints are omitted or explicitly empty, and honor explicit custom endpoints: six public CLI cases pass against a real fixture pack. The omitted-endpoint cases fail on the previous helper implementation.
- These checks do not establish live identity bridge availability or authenticated catalog integration with the remote pack.

# Checkout engine version

- The checkout version guard failed on main (runtime 0.0.7 versus project 0.0.8) and passed after alignment; all eight CLI tests passed on Linux CPU.
- The same from-scratch Linux CPU setup refused inconsistent registry inputs before alignment and applied successfully afterward, passing health and activating generation 1. Generated extension checks, Ruff format/lint and Linux/Windows Pyright passed.
- Published Stable 0.0.8 and Nightly 0.0.9.dev1080 wheels have matching runtime and distribution versions.

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

# WSL2 pack GPU isolation

- [SidePC real-worker receipt](https://github.com/Kosinkadink/comfy-vibe-station/issues/968#issuecomment-6902825259): RTX 4090, driver 591.86, Python 3.14.10, torch 2.14.1+cu130. Three plain, GPU-granted and denied-GPU workers exercised CUDA, WSL-disk safetensors loading and filesystem/network/interop exclusions at the receipt's exact source head.
- Median float16 8192x8192 matmul: plain 145.836/s, sandbox 144.516/s. Warm safetensors-to-CUDA load median: plain 1.693s, sandbox 1.678s, 453 tensors each. Denied-GPU workers could not perform either CUDA operation. This is not inference or numerical-parity evidence.
- Driver filesystem was read-only in granted workers and its digest stayed unchanged. Windows-drive, live interop socket, /init, host-loopback and external-TCP exclusions had successful plain controls.
- Desktop private-distribution provisioning and sandbox-toggle integration remain separate validation; these receipts do not establish that experience.
