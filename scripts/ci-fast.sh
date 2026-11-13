#!/usr/bin/env bash
set -euo pipefail

# Dependency and build-backend provisioning belongs to setup, not the gate.
export UV_OFFLINE=1

if [ "${1:-all}" != tests ]; then
python scripts/develop.py run -- python tools/gen_extension_contribution_kinds.py --check
python scripts/develop.py run -- python scripts/check_extension_factories.py \
  --baseline-ref "origin/${GITHUB_BASE_REF:-main}"
python scripts/develop.py run -- python scripts/check_builtin_type_spellings.py
python scripts/develop.py run -- ruff format --check .
python scripts/develop.py run -- ruff check .
python scripts/develop.py run -- pyright
python scripts/develop.py run -- pyright --pythonplatform Windows
fi
if [ "${1:-all}" = checks ]; then exit 0; fi

python scripts/develop.py run -- python -m pytest -q -m 'not network' -p tools.ci_selection \
  tests/test_settings.py \
  tests/test_security_notices.py \
  tests/test_macos_workflow_receipt.py \
  tests/test_memory_review_regressions.py::test_public_keyed_residents_share_first_concurrent_load \
  tests/test_pack_environment.py \
  tests/test_environment_collection.py \
  tests/test_environment_state.py \
  tests/test_environment_generations.py::test_generation_sync_command_does_not_expand_wheel_sources \
  tests/test_environment_generations.py::test_checkout_pack_commands_do_not_expand_wheel_sources \
  tests/test_environment_generations.py::test_public_setup_acquires_pack_wheel_absent_from_release \
  tests/test_environment_generations.py::test_offline_generations_share_unchanged_wheels_and_failed_sync_preserves_active \
  tests/test_environment_generations.py::test_additive_registry_preparation_retains_sources_and_only_fetches_new_locks \
  'tests/test_environment_generations.py::test_generation_requires_dependency_locks_for_the_complete_admitted_catalog[restore]' \
  tests/test_environment_cli.py::test_update_workflow_plan_uses_core_adapter_without_mutating_project_or_cache \
  tests/test_environment_cli.py::test_confirmed_offline_setup_update_rollback_and_busy_refusal \
  tests/test_environment_cli.py::test_workflow_update_prepares_target_registry_inputs_before_journaling \
  tests/test_environment_cli.py::test_custom_cli_setup_retains_base_on_update_without_definition \
  tests/test_environment_history.py \
  tests/test_environment_runtime.py::test_start_status_idempotence_and_stop \
  tests/test_environment_runtime.py::test_start_failure_preserves_exit_code_and_current_output \
  'tests/test_environment_transactions.py::test_first_setup_publishes_only_a_healthy_engine_and_recovers_without_previous_generation[crlf-None]' \
  tests/test_environment_transactions.py::test_completed_activation_recovers_history_publication_once_after_process_exit \
  tests/test_workflow_requirements.py \
  tests/test_run_scripts.py \
  tests/test_launch.py \
  tests/test_environment_tools.py \
  tests/test_release_install.py \
  tests/test_environment_generations.py::test_generation_retains_verified_engine_manifest \
  tests/test_installer.py::test_exact_library_release_resolves_and_downloads_without_node_discovery \
  tests/test_installer.py::test_http_registry_fetcher_uses_verified_static_wheel \
  tests/test_installer.py::test_registry_download_retries_only_transport \
  tests/test_installer.py::test_http_registry_fetcher_rejects_invalid_wheel_size \
  tests/test_installer.py::test_cli_published_restore_routes_require_managed_generation \
  tests/test_installer.py::test_routing_fetcher_dispatches_on_recorded_provenance \
  tests/test_installer.py::test_apply_checks_pack_engine_range_before_staging_or_identical_lock_return \
  tests/test_installer.py::test_registry_resolves_newest_compatible_pack_across_all_pages \
  tests/test_installer.py::test_registry_compatible_selection_refuses_malformed_pages \
  tests/test_installer.py::test_split_registry_reads_only_selected_packs_and_refuses_substitution \
  tests/test_installer.py::test_registry_preparation_snapshot_isolates_endpoints_and_credentials \
  tests/test_installer.py::test_registry_source_identity_comes_from_verified_archive \
  tests/test_installer.py::test_cli_prepares_compatible_registry_sources_without_generation_mutation \
  tests/test_installer.py::test_cli_acquires_published_locks_for_exact_sources_and_base \
  tests/test_installer.py::test_cli_publishes_exact_archive_with_optional_published_locks \
  tests/test_installer.py::test_cli_retains_nonregistry_generation_sources_before_writing_candidate \
  tests/test_installer.py::test_cli_adds_exact_pack_without_reresolving_installed_sources \
  tests/test_installer.py::test_cli_reresolves_installed_sources_without_mutating_project \
  tests/test_installer.py::test_generation_source_staging_cli \
  tests/test_pack_archive.py::test_publication_client_preserves_archive_and_lock_bytes \
  tests/test_pack_archive.py::test_registry_archive_admission_requires_verified_wheel \
  tests/test_pack_wheel.py \
  tests/test_pack_archive.py::test_registry_archive_security_preserves_block_and_review \
  tests/test_pack_archive.py::test_registry_manifest_admission_preserves_engine_range \
  tests/test_pack_archive.py::test_registry_manifest_admission_refuses_invalid_metadata \
  tests/test_pack_archive.py::test_registry_executable_probe_bounds_and_sanitizes \
  tests/test_pack_archive.py::test_registry_probe_refuses_degraded_isolation \
  tests/test_archive_admission.py \
  tests/test_frontend_artifact.py \
  tests/test_schema.py \
  tests/test_schema_current_contracts.py \
  tests/test_names.py \
  tests/test_first_party_node_ids.py \
  tests/test_comfy_controlnet_nodes.py \
  tests/test_values.py \
  tests/test_graph.py \
  tests/test_graph_wire.py \
  tests/test_model3d_io.py::test_viewport_capture_survives_workers_and_persistent_cache \
  tests/test_manifest.py::test_pack_prefix_reservation_depends_on_publisher \
  tests/test_manifest.py::test_pack_declares_independent_version_and_engine_compatibility \
  tests/test_doctor.py::test_catalog_driven_node_source_is_reported_without_hiding_entry_errors \
  tests/test_compose.py::test_pack_namespace_compose_reload_execute_and_reject_second_owner \
  tests/test_compose.py::test_asset_literals_use_host_and_worker_resolvers \
  tests/test_serve.py::test_serve_plain_dev_pack_is_announced \
  tests/test_serve.py::test_serve_empty_generation_health_distinguishes_failed_selection
