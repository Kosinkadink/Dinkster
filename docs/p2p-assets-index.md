# Preserved P2P asset code

This branch preserves the host and browser code before removal from main.
The external dinkster-p2p repository is not changed. Digest identity, vault
storage, consent and verified HTTP acquisition belong to the surviving core.

The index includes dedicated transfer files and shared files with transfer
references. A shared file's count is its whole-file size, not a deletion size.
Incidental encoded-fixture matches are identified separately. Binary assets
have no meaningful physical line count. Counts use newline bytes.

Source main: [adb9e2feafb2](https://github.com/Kosinkadink/Dinkster/commit/adb9e2feafb21322fa2f40fe4142a7dd800828de).

## Two-host transfer requirements

Both hosts need compatible plugin/runtime dependencies, reachable peer
endpoints, discovery or explicitly configured peers, authorization grants,
and a source file whose digest is available from a mount or vault. The
receiving host needs writable staging/storage, transfer activity reporting,
and final digest verification before publishing the bytes to the catalog.
Tests and workflow lanes are references, not evidence that this preserved
code works between real hosts. Restore components only after testing them.

| File | Physical lines | Purpose | Two-host responsibility |
| --- | ---: | --- | --- |
| `.github/workflows/full-validation.yml` | 434 | Define automated transfer validation or shared validation jobs | Recover dedicated transfer lanes without replacing surviving CI |
| `.github/workflows/p2p-validation.yml` | 55 | Define automated transfer validation or shared validation jobs | Recover dedicated transfer lanes without replacing surviving CI |
| `DESIGN.md` | 966 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `conftest.py` | 47 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `docs/auth.md` | 224 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `docs/multigpu-hosting.md` | 140 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `docs/quickstart.md` | 130 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `docs/seed-service.md` | 166 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `docs/serve-cli.md` | 875 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `docs/settings-api.md` | 266 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `docs/supported/peer-to-peer-host-support.md` | 27 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `docs/supported/server-api-and-execution.md` | 106 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `docs/typed-execution-contracts.md` | 1855 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `packages/dinkster-assets/README.md` | 244 | Document transfer behavior or shared host configuration | Recover the documented setup and security expectations |
| `packages/dinkster-assets/src/dinkster_assets/__init__.py` | 394 | Connect peer transfer behavior to the shared application surface | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/acquire.py` | 218 | Acquisition: turning a declared need into verified local bytes (DESIGN 3.12 + roadmap "templates/asset distribution"). | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_descriptor.py` | 357 | Open and validate shared file descriptors | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_fixture.py` | 103 | Build deterministic transfer fixtures | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_global.py` | 890 | Coordinate global peer discovery and transfers | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_grants.py` | 438 | Authorize peer access to published content | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_plugin.py` | 34 | Connect host services to the optional transfer plugin | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_settings.py` | 116 | Validate peer transfer policy and settings | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_storage.py` | 1745 | Map torrent pieces onto local asset storage | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_store.py` | 78 | Store and retrieve assets through the optional peer runtime | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_usn.py` | 13 | Observe Windows filesystem changes for shared files | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/p2p_windows.py` | 772 | Implement Windows-specific sharing operations | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/resolver_index.py` | 382 | Forward-compatible declarative resolver-index format. | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/resolver_subscription.py` | 1501 | Persistent subscriptions to local and hosted resolver indexes. | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/transport.py` | 347 | Rank and resolve local, peer and HTTP asset sources | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-assets/src/dinkster_assets/vault.py` | 371 | AssetVault: where fetched and transferred bytes land (DESIGN 3.12). | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/__init__.py` | 272 | Connect peer transfer behavior to the shared application surface | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/app.py` | 4690 | The native Dinkster HTTP/WS protocol (DESIGN 3.5). | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/auth.py` | 1072 | Inbound principals, capabilities, and agent permission categories. | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/lan_discovery.py` | 353 | Discover peer hosts on the local network | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/lan_mapping.py` | 488 | Map peer endpoints and local network addresses | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/library.py` | 1084 | Workflow persistence surface: client upload + scoped library (DESIGN roadmap: "Workflow persistence: workflows as assets" + the identity/scoping constraint). | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/p2p_activity.py` | 252 | Publish transfer activity to the host UI | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/p2p_plugin.py` | 23 | Connect host services to the optional transfer plugin | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/preflight.py` | 296 | Job asset preflight: no run starts with unresolvable assets, and no byte downloads without digest-exact consent (roadmap "templates/asset distribution"; the reproduce plan/apply principle applied to submission). | Restore only the transfer integration after proving the digest core still works |
| `packages/dinkster-server/src/dinkster_server/settings.py` | 530 | Runtime-editable server settings and their HTTP surface. | Restore only the transfer integration after proving the digest core still works |
| `pyproject.toml` | 150 | Declare and pin the optional transfer dependencies and commands | Install compatible runtimes on both hosts |
| `scripts/ci-fast.sh` | 23 | Connect peer transfer behavior to the shared application surface | Restore only the transfer integration after proving the digest core still works |
| `scripts/reproduce_distinct_host_p2p.py` | 379 | Reproduce a LAN P2P download while two hosts seed the same digest. | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/cli.py` | 127 | The umbrella `dinkster` CLI: one entry point fronting every console script. | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/lan_p2p.py` | 939 | Compose local-network discovery and transfer services | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/launch.py` | 91 | The no-argument local launch experience. | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/p2p_api.py` | 510 | Expose peer status and transfer HTTP routes | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/p2p_diagnostics.py` | 82 | Inspect optional runtime and host transfer readiness | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/p2p_plugin.py` | 42 | Connect host services to the optional transfer plugin | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/resolver_api.py` | 166 | Management API for persistent declarative resolver-index subscriptions. | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/seed.py` | 399 | Host the standalone seed service | Restore only the transfer integration after proving the digest core still works |
| `src/dinkster/serve.py` | 2516 | dinkster-serve: run a Dinkster server from a composed node-pack set. | Restore only the transfer integration after proving the digest core still works |
| `tests/goldens/audio_aliases_b78cec87.json` | 673 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/audio_envelope.json` | 281 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/audio_io_e20d433a.json` | 153 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/audio_ops_e20d433a.json` | 450 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/depth_anything_3_mono_large_e7051b0.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/depth_anything_v2_controlnet_aux_59b1fc4.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/detr_r50_29901c5.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/efficient_sam_vitt_d525f62.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/hint_resize_controlnet_aux_59b1fc4.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/line_edge_controlnet_aux_59b1fc4.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/preprocessors_controlnet_aux_59b1fc4.json` | 126 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/rtdetr_v4_x_hgnet_c67885b.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/sam31_detect_8dc3f3f.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/sam31_multiplex_8dc3f3f.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/upscale_realesrgan_a1079ba1.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/goldens/usdu_a5547db9.json` | 1 | Existing model fixture with an incidental lexical match | No transfer responsibility; preserve fixture bytes unchanged |
| `tests/p2p_global_fixtures.py` | 137 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_cli.py` | 149 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_dependency_rule.py` | 414 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_disabled_features.py` | 88 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_distinct_host_p2p_reproduction.py` | 75 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_durable_delegations.py` | 415 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_lan_p2p_discovery.py` | 284 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_lan_p2p_mapping.py` | 609 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_lan_p2p_runtime.py` | 1541 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_launch.py` | 242 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p.py` | 1740 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_activity.py` | 299 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_artifact_smoke.py` | 231 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_descriptor.py` | 361 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_global.py` | 507 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_global_dht.py` | 125 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_global_sidecar.py` | 1887 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_listeners.py` | 331 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_native_diagnostics.py` | 397 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_receipts_and_grants.py` | 1079 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_staging_budget.py` | 307 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_p2p_storage.py` | 852 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_resolver_indexes.py` | 2903 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_seed_service.py` | 572 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_seed_service_recovery.py` | 387 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_serve.py` | 3859 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_server.py` | 7092 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_settings.py` | 570 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tests/test_transport_resolver.py` | 523 | Test transfer behavior or a shared host integration | Recheck applicable assertions with two independent hosts and shared content |
| `tools/flux_window_multigpu_receipt.py` | 855 | Measure fork-backed Flux window scattering against serial window execution. | Restore only the transfer integration after proving the digest core still works |
| `tools/minimax_h3_multigpu_receipt.py` | 1352 | Measure fork-backed MiniMax H3 single-job multi-GPU execution. | Restore only the transfer integration after proving the digest core still works |
| `uv.lock` | 2567 | Declare and pin the optional transfer dependencies and commands | Install compatible runtimes on both hosts |
