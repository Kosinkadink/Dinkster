# ComfyUI compatibility layers

This branch preserves the alias, group, replacement, translation and carrier implementation.
The native-node contract uses one registry and one type namespace instead of these bridges.

Indexed source: `adb9e2feafb21322fa2f40fe4142a7dd800828de`. Counts are physical lines of tracked files at that source.
Dedicated layer files may be removed; shared callers retain their unrelated responsibilities.
Test and fixture entries identify the layer contract, not permission to delete surviving behavior.
The importer remains responsible for positional workflow inputs and versioned per-node upgrades.

| File | Lines | Purpose |
| --- | ---: | --- |
| `.github/workflows/full-validation.yml` | 434 | Implements full validation in the layer or its shared caller. |
| `docs/comfy-translation-coverage.md` | 88 | Documents comfy translation coverage; shared documentation is edited rather than wholly removed. |
| `docs/in-process-hosting-design.md` | 437 | Documents in process hosting design; shared documentation is edited rather than wholly removed. |
| `docs/lazy-input-m1-contract.md` | 522 | Documents lazy input m1 contract; shared documentation is edited rather than wholly removed. |
| `docs/lazy-switch-lowering.md` | 358 | Documents lazy switch lowering; shared documentation is edited rather than wholly removed. |
| `docs/list-map-lowering.md` | 186 | Documents list map lowering; shared documentation is edited rather than wholly removed. |
| `docs/replacement-1-to-n.md` | 199 | Documents replacement 1 to n; shared documentation is edited rather than wholly removed. |
| `docs/supported/comfyui-compatibility.md` | 24 | Documents comfyui compatibility; shared documentation is edited rather than wholly removed. |
| `docs/testing.md` | 251 | Documents testing; shared documentation is edited rather than wholly removed. |
| `goldens/replacements/README.md` | 28 | Reference input/output data for README.md; retain unrelated fixture bytes. |
| `goldens/replacements/chain.json` | 204 | Reference input/output data for chain.json; retain unrelated fixture bytes. |
| `goldens/replacements/combo.json` | 180 | Reference input/output data for combo.json; retain unrelated fixture bytes. |
| `goldens/replacements/dynamic.json` | 296 | Reference input/output data for dynamic.json; retain unrelated fixture bytes. |
| `goldens/replacements/vocabulary.json` | 352 | Reference input/output data for vocabulary.json; retain unrelated fixture bytes. |
| `packages/dinkster-compat-comfy/README.md` | 23 | Documents README; shared documentation is edited rather than wholly removed. |
| `packages/dinkster-compat-comfy/comfy-aliases.inactive.json` | 1 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-compat-comfy/dinkster-legacy-pack.toml` | 95 | Package/dependency metadata wiring the layer into installation or composition. |
| `packages/dinkster-compat-comfy/dinkster-pack.toml` | 101 | Package/dependency metadata wiring the layer into installation or composition. |
| `packages/dinkster-compat-comfy/pyproject.toml` | 44 | Package/dependency metadata wiring the layer into installation or composition. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/__init__.py` | 93 | ComfyUI application-node compatibility. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/adapters.py` | 890 | Input adapters for natively-ported nodes: legacy v1 input shapes -> |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/audio.py` | 12 | Implements audio in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/bootstrap.py` | 389 | Import a real ComfyUI installation and translate its nodes. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/comfy_execution.py` | 150 | ComfyUI sampling and device eviction for the compatibility execution arm. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/core_schemas.json` | 72248 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/devices.py` | 12 | Implements devices in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/entry.py` | 88 | Manifest entry points for ComfyUI application-node compatibility. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/image.py` | 12 | Implements image in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/latent.py` | 12 | Implements latent in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/legacy.py` | 480 | Load unmodified ComfyUI custom-node packs and translate them (DESIGN 3.8). |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/legacy_entry.py` | 73 | Manifest entry points for the legacy custom-pack quarantine worker. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/legacy_sources.py` | 12 | Implements legacy sources in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/listmap.py` | 150 | Lower ComfyUI's implicit list mapping into explicit Dinkster map regions. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/memory.py` | 12 | Implements memory in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/native.py` | 12 | Implements native in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/pool.py` | 16 | Implements pool in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/port_probe.py` | 247 | Probe a legacy ComfyUI pack for ``dinkster port`` (child process only). |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/prompt.py` | 526 | ComfyUI API prompt adapter: v1 prompt JSON -> native Dinkster Graph. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/py.typed` | 0 | Implements py in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/resident.py` | 12 | Implements resident in the layer or its shared caller. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/saved_results.py` | 59 | Guarded import of ComfyUI SavedResult UI records. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/schema_snapshot.py` | 11 | Pinned upstream schemas for import metadata, never executable classes. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/source_staging.py` | 565 | Invocation-owned staging of verified media for Comfy source filenames. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/translate.py` | 3341 | ComfyUI v1 -> Dinkster translation, pure and importable without ComfyUI. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/translate_v3.py` | 750 | ComfyUI V3 (comfy_entrypoint) -> Dinkster schema translation, pure. |
| `packages/dinkster-compat-comfy/src/dinkster_compat_comfy/video.py` | 12 | Implements video in the layer or its shared caller. |
| `packages/dinkster-nodes-foundation/comfy-aliases.json` | 1 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-nodes-foundation/pyproject.toml` | 22 | Package/dependency metadata wiring the layer into installation or composition. |
| `packages/dinkster-nodes-generation/README.md` | 9 | Documents README; shared documentation is edited rather than wholly removed. |
| `packages/dinkster-nodes-generation/comfy-aliases.inactive.json` | 27638 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-nodes-generation/comfy-groups.inactive.json` | 1728 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-nodes-generation/src/dinkster_nodes_generation/model3d.py` | 541 | Generic geometry-estimation and mesh-processing schemas. |
| `packages/dinkster-nodes-generation/src/dinkster_nodes_generation/nodes.py` | 6421 | Universal generation node schemas, independent of execution providers. |
| `packages/dinkster-nodes-generation/src/dinkster_nodes_generation/trellis2.py` | 240 | Universal TRELLIS.2 and Pixal3D generation schemas. |
| `packages/dinkster-nodes-image/README.md` | 112 | Documents README; shared documentation is edited rather than wholly removed. |
| `packages/dinkster-nodes-image/comfy-aliases.json` | 21374 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-nodes-image/comfy-groups.json` | 1539 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-nodes-image/pyproject.toml` | 31 | Package/dependency metadata wiring the layer into installation or composition. |
| `packages/dinkster-nodes-media-io/comfy-aliases.json` | 9235 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-nodes-media-io/pyproject.toml` | 31 | Package/dependency metadata wiring the layer into installation or composition. |
| `packages/dinkster-nodes-vision/dinkster_vision_depth_anything_v2_pack/comfy-aliases.inactive.json` | 297 | Committed schema, alias or group data consumed by the layer registry. |
| `packages/dinkster-schema/README.md` | 208 | Documents README; shared documentation is edited rather than wholly removed. |
| `packages/dinkster-schema/src/dinkster_schema/__init__.py` | 326 | Implements   init   in the layer or its shared caller. |
| `packages/dinkster-schema/src/dinkster_schema/comfy_alias.py` | 496 | Maintained ComfyUI-to-native translation registry data. |
| `packages/dinkster-schema/src/dinkster_schema/comfy_group.py` | 768 | Declarative ComfyUI node-group translation patterns. |
| `packages/dinkster-schema/src/dinkster_schema/replace.py` | 1744 | Declarative node-replacement rules: pack-declared migration data. |
| `packages/dinkster-server/src/dinkster_server/app.py` | 4690 | The native Dinkster HTTP/WS protocol (DESIGN 3.5). |
| `packages/dinkster-workers/README.md` | 190 | Documents README; shared documentation is edited rather than wholly removed. |
| `packages/dinkster-workers/src/dinkster_workers/doctor.py` | 1382 | dinkster doctor - the pack linter (DESIGN 3.9). |
| `packages/dinkster-workers/src/dinkster_workers/host.py` | 2937 | Isolated worker host: the child-process side of IsolatedWorker. |
| `packages/dinkster-workers/src/dinkster_workers/manifest.py` | 3280 | Pack manifests: what a pack is, declared as data (DESIGN 3.6). |
| `packages/dinkster-workers/src/dinkster_workers/remote.py` | 906 | RemoteWorker: the Worker protocol across a machine boundary. |
| `packages/dinkster-workers/src/dinkster_workers/service.py` | 593 | Remote worker service: a pack served over the network. |
| `packages/dinkster-workers/src/dinkster_workers/session.py` | 2347 | BoundarySession: the engine side of one boundary conversation. |
| `scripts/generate_replacement_goldens.py` | 426 | Reference input/output data for generate_replacement_goldens.py; retain unrelated fixture bytes. |
| `src/dinkster/comfy_compose.py` | 769 | ComfyUI compat workers as composable packs (DESIGN 3.8 meets 3.6). |
| `src/dinkster/compat_api.py` | 355 | ComfyUI compat HTTP surface: POST /api/compat/comfy/prompt. |
| `src/dinkster/compose.py` | 7806 | Multi-pack composition: pack manifests -> one serving engine. |
| `src/dinkster/mounts_api.py` | 464 | Filesystem mounts API: the operator's directory-grant surface. |
| `src/dinkster/packs.py` | 199 | Pack provenance glue: manifest presentation -> server pack table. |
| `src/dinkster/port.py` | 1097 | dinkster-port - generate a native Dinkster pack skeleton from a legacy ComfyUI pack. |
| `src/dinkster/reload_api.py` | 243 | Pack reload coordination and dev-mode mutation endpoints (DESIGN 3.9). |
| `src/dinkster/serve.py` | 2516 | dinkster-serve: run a Dinkster server from a composed node-pack set. |
| `tests/memorypack_nodes.py` | 265 | Tests memorypack nodes.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/respack_nodes.py` | 71 | Tests respack nodes.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_audio_alias_replay.py` | 414 | Tests audio alias replay.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_audio_video_contract.py` | 650 | Tests audio video contract.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_av_codec.py` | 454 | Tests av codec.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_choices.py` | 1338 | Tests choices.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_ckpt_conversion_tool.py` | 59 | Tests ckpt conversion tool.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_comfy_alias_registry.py` | 250 | Tests comfy alias registry.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_comfy_group_registry.py` | 377 | Tests comfy group registry.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_compat_arm_entry.py` | 54 | Tests compat arm entry.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_compat_comfy.py` | 4633 | Tests compat comfy.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_compat_legacy.py` | 519 | Tests compat legacy.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_compat_listing_combos.py` | 1169 | Tests compat listing combos.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_compat_review_regressions.py` | 359 | Tests compat review regressions.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_compose.py` | 3324 | Tests compose.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_compose_dispatch.py` | 3305 | Tests compose dispatch.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_conversion_curve_stdlib.py` | 407 | Tests conversion curve stdlib.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_dependency_rule.py` | 414 | Tests dependency rule.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_fork_nodes.py` | 949 | Tests fork nodes.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_image_codec.py` | 840 | Tests image codec.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_image_comfy_aliases.py` | 883 | Tests image comfy aliases.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_image_comfy_groups.py` | 163 | Tests image comfy groups.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_image_dynamic_schemas.py` | 518 | Tests image dynamic schemas.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_image_preprocessors.py` | 517 | Tests image preprocessors.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_isolated.py` | 4598 | Tests isolated.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_lazy_input_m1_contract.py` | 647 | Tests lazy input m1 contract.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_legacy_sources.py` | 520 | Tests legacy sources.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_loader_stack_census.py` | 123 | Tests loader stack census.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_manifest.py` | 859 | Tests manifest.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_media_comfy_aliases.py` | 543 | Tests media comfy aliases.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_media_io_comfy_aliases.py` | 191 | Tests media io comfy aliases.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_numeric_stdlib.py` | 676 | Tests numeric stdlib.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_pins.py` | 208 | Tests pins.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_port.py` | 675 | Tests port.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_primitives.py` | 400 | Tests primitives.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_relay.py` | 1789 | Tests relay.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_replace.py` | 2145 | Tests replace.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_reservations.py` | 803 | Tests reservations.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_routing_stdlib.py` | 241 | Tests routing stdlib.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_saved_artifacts.py` | 354 | Tests saved artifacts.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_serve_remote.py` | 1599 | Tests serve remote.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_server.py` | 7092 | Tests server.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_source_filename.py` | 392 | Tests source filename.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_source_staging.py` | 861 | Tests source staging.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_string_comfy_aliases.py` | 334 | Tests string comfy aliases.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_translate_v3.py` | 1159 | Tests translate v3.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_video_alias_templates.py` | 625 | Tests video alias templates.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_video_runtime.py` | 890 | Tests video runtime.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_video_vhs_live.py` | 327 | Tests video vhs live.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_vision_depth_anything_v2_schema.py` | 190 | Tests vision depth anything v2 schema.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_vision_hed_schema.py` | 158 | Tests vision hed schema.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/test_vision_line_edge_schema.py` | 171 | Tests vision line edge schema.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tests/type_equivalence_compat_nodes.py` | 107 | Tests type equivalence compat nodes.py; preserve assertions for surviving behavior when removing exclusive layer tests. |
| `tools/benchmark_schema_catalog.py` | 278 | Generates or checks benchmark schema catalog. |
| `tools/comfy-core-schema-requirements.txt` | 84 | Generates or checks comfy core schema requirements. |
| `tools/comfy_confidence.py` | 781 | Generates or checks comfy confidence. |
| `tools/convert_ckpt_to_safetensors.py` | 33 | Generates or checks convert ckpt to safetensors. |
| `tools/gen_comfy_core_schemas.py` | 71 | Generates or checks gen comfy core schemas. |
| `tools/gen_image_alias_goldens.py` | 58 | Reference input/output data for gen_image_alias_goldens.py; retain unrelated fixture bytes. |
| `tools/gen_image_comfy_aliases.py` | 5416 | Generates or checks gen image comfy aliases. |
| `tools/gen_image_comfy_groups.py` | 668 | Generates or checks gen image comfy groups. |
| `tools/gen_image_geometry_goldens.py` | 323 | Reference input/output data for gen_image_geometry_goldens.py; retain unrelated fixture bytes. |
| `tools/gen_image_load_goldens.py` | 90 | Reference input/output data for gen_image_load_goldens.py; retain unrelated fixture bytes. |
| `tools/gen_image_mask_goldens.py` | 169 | Reference input/output data for gen_image_mask_goldens.py; retain unrelated fixture bytes. |
| `tools/gen_image_media_comfy_aliases.py` | 303 | Generates or checks gen image media comfy aliases. |
| `tools/gen_media_comfy_aliases.py` | 1282 | Generates or checks gen media comfy aliases. |
| `tools/gen_video_media_comfy_aliases.py` | 455 | Generates or checks gen video media comfy aliases. |
| `tools/generate_foundation_comfy_aliases.py` | 1982 | Generates or checks generate foundation comfy aliases. |
| `tools/layer_comfy_aliases.py` | 179 | Generates or checks layer comfy aliases. |
| `tools/legacy_survey.py` | 184 | Generates or checks legacy survey. |

Total: 153 tracked files; whole-file counts include shared integrations.
