# Signed release channels

- CPU contributor setup passed. Generated extension checks, Ruff format/lint and Linux/Windows Pyright passed.
- Environment discovery, acquisition, CLI, collection, generations, history, tools, transactions and launch: 404 passed, 3 skipped in 711.08 seconds.
- Installed-wheel unavailable-update case reproduced on the unchanged base; its obsolete older-version fallback assertion now requires exact-target source refusal. No-mutation and resolver-trap checks remain. Corrected history plus eight registry controls: 9 passed.
- Independent tools publisher: signed selection preserved; unsigned and tampered channels refused, 3 passed. Stable/Nightly publication controls verify signatures, daily selection and readback.
- Live catalog bootstrap used the published Stable supervisor rather than the strict checkout decoder: 646 nodes and 37 packs, no composing state. Wrong digest/size, absent digest and prerelease bootstrap inputs were refused.
- Public signed release, Windows/Linux Desktop and final cross-repository acceptance remain unverified; this is not a publication receipt.

# Contributor wheel acquisition

- Real linux-cpu resolver: 14 selected wheels retained, including 10 wheel rows without sizes; verified files carry measured byte counts.
- Regression against original source: missing-size wheel fails on the separate HEAD request. Changed source: acquisition succeeds, disconnected reuse works, and corrupt cached/remote bytes are rejected.
- Downloader suite: 32 passed. Downloader plus development inference generation cases: 36 passed, 33 deselected.
- Full contributor setup with explicit Python 3.12 and isolated port 15971 passed; its engine was stopped through the managed CLI.
- Full Ruff format/check passed. Managed contributor Pyright on Linux and Windows targets: zero errors.
- No tests removed, new skips/retries, relaxed deadlines or Actions reruns.
