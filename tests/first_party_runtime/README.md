# Comfy node tests

Node behavior runs through the in-process composer. Subprocess tests are
reserved for codec, resident ownership, worker environment and persistent
cache boundaries. Add a keeper when a new boundary needs coverage, not for
each node that uses an existing value type. In-process cases retain the
node-specific output assertions.

The following 42 subprocess cases are the boundary keepers. Paths are
relative to this directory; bracketed suffixes are pytest parameter IDs.
The three grouped fixtures use one worker per composition while retaining
the production encode/decode path.

| Test | Boundary property |
| --- | --- |
| `test_device_routing_torch.py::test_all_device_nodes_run_twice_through_both_production_hosts[False]` | Real MODEL, CLIP and VAE residents; worker accelerator environment |
| `test_audio_combine_torch.py::test_audio_combine_nodes_execute_twice_in_both_manifest_hosts[False]` | AUDIO media codec |
| `test_audio_encoder_torch.py::test_audio_encoder_ordinary_hosts[True-False]` | AUDIO_ENCODER ownership, AUDIO_ENCODER_OUTPUT and scanned ASSET mount |
| `test_audio_latent_torch.py::test_audio_size_production_manifest_both_hosts_twice[False]` | MultiStreamLatent samples inside LATENT |
| `test_latent_upscale_torch.py::test_latent_upscale_nodes_execute_twice_in_ordinary_hosts[False]` | LATENT noise mask, metadata and float64; VAE and LATENT_UPSCALE_MODEL consumption |
| `test_latent_upscale_loader.py::test_latent_upscale_loader_crosses_ordinary_hosts_twice[ltx-False]` | ASSET vault access and LATENT_UPSCALE_MODEL creation in worker |
| `test_cfg_guiders_torch.py::test_native_guider_nodes_connect_fixture_producer_and_consumer_twice[False]` | Native GUIDER residents, MODEL and CONDITIONING |
| `test_custom_sampling_torch.py::test_custom_sampling_on_ordinary_hosts[advanced-False]` | SAMPLER, SIGMAS, NOISE, GUIDER and LATENT inputs |
| `test_hook_scheduling_torch.py::test_ordinary_manifests_execute_hook_scheduling_chain_twice[False]` | HOOKS, HOOK_KEYFRAMES and FLOATS chain; cache replay |
| `test_hook_values_torch.py::test_ordinary_manifest_transports_float_strengths_twice[False]` | Exact bfloat16 FLOATS bytes, HOOK_KEYFRAMES identity and real CLIP in a shared worker |
| `test_hook_values_torch.py::test_ordinary_hosts_retain_weight_owners_through_public_hook_nodes[False]` | HOOKS resident weight owners, nested CONDITIONING hooks and CLIP |
| `test_hook_conditioning_torch.py::test_hook_conditioning_properties_cross_ordinary_hosts_twice[False]` | TIMESTEPS_RANGE, CONDITIONING mask, HOOKS and CLIP |
| `test_checkpoint_hooks_torch.py::test_checkpoint_hooks_ordinary_hosts_use_snapshot_roots_and_resident_pair[True-False]` | Verified AssetRef snapshot roots and HOOKS weight-pair residents |
| `test_hook_lora.py::test_lora_hooks_execute_through_fixture_pack_on_both_hosts[False-False]` | LocalAssetRef revalidation in worker and HOOKS LoRA weights |
| `test_image_control_patches_torch.py::test_image_controls_on_ordinary_hosts[zimage-False]` | MODEL_PATCH weight identity, MODEL, VAE and IMAGE |
| `test_controlnet_torch.py::test_ordinary_manifest_executes_controlnet_nodes[ControlNetApplyAdvanced-False]` | CONTROL_NET, CONDITIONING, IMAGE, MASK and VAE |
| `test_gligen_conditioning_torch.py::test_gligen_textbox_crosses_both_ordinary_hosts_twice[False]` | GLIGEN residents nested in CONDITIONING and CLIP |
| `test_hunyuan3d_torch.py::test_hunyuan_ordinary_manifest[False]` | Dictionary CLIP_VISION_OUTPUT and CONDITIONING |
| `test_compositing_torch.py::test_manifest_links_decoded_masks_to_compositing_nodes[False]` | Exact blended values from numpy float32 IMAGE and MASK |
| `test_image_color_conversion_torch.py::test_color_conversion_ordinary_manifest_twice[False]` | Numpy float16 IMAGE and dynamic slot variants |
| `test_flux_image_layout_torch.py::test_ordinary_manifest_executes_flux_image_layout_nodes[False]` | ARRAY and IMAGE |
| `test_image_crop_torch.py::test_crop_and_bounding_box_execute_twice_through_ordinary_manifest[False]` | BOUNDING_BOX and exact cropped IMAGE values in a shared worker |
| `test_image_tiles_torch.py::test_registered_image_and_list_tools_execute_twice_in_ordinary_hosts[False]` | Lists, STRING, MASK and float64-to-float32 IMAGE |
| `test_image_tiles_torch.py::test_native_mixed_lists_cross_hosts_and_persistent_cache[False]` | Heterogeneous lists and persistent layered cache across two shared-worker compositions |
| `test_standard_sampling_torch.py::test_standard_samplers_execute_on_both_ordinary_hosts[False]` | Real MODEL, CONDITIONING and LATENT; cached sampling result |
| `test_model_file_loaders_torch.py::test_model_file_loaders_bind_tiny_safetensors_through_ordinary_hosts[True-False-True]` | Worker safetensors loading of MODEL, CLIP, VAE, CLIP_VISION, CONTROL_NET, STYLE_MODEL and GLIGEN |
| `test_minimax_h3_guide_hosts_torch.py::test_guides_run_twice_through_both_ordinary_hosts[False]` | Torch AUDIO dictionary, multi-stream LATENT, VAE and IMAGE |
| `test_vision_output_torch.py::test_vision_output_ordinary_manifest[False]` | Byte-exact float16, bfloat16, float64, empty and nested CLIP_VISION_OUTPUT |
| `test_wan_image_conditioning_hosts_torch.py::test_wan_image_conditioning_runs_twice_in_ordinary_hosts[False]` | CONDITIONING nested metadata, VAE, IMAGE, MODEL and LATENT |
| `test_unclip_conditioning_torch.py::test_unclip_conditioning_ordinary_manifest_uses_vision_encoder_and_native_consumer[False]` | Real vision Output objects nested in CONDITIONING metadata |
| `test_wandancer_keyframes_torch.py::test_keyframes_run_twice_through_ordinary_hosts[False]` | IMAGE, MASK, AUDIO and lists of each |
| `test_perpneg_torch.py::test_perpneg_nodes_run_twice_in_both_ordinary_hosts_with_cache[False]` | Real PerpNegGuider resident and cache replay |
| `test_sampler_factories_torch.py::test_all_sampler_factories_cross_ordinary_manifest_hosts_twice[False]` | Twelve SAMPLER residents and cache replay |
| `test_sampling_wire_torch.py::test_runtime_sampler_and_noise_repeat_in_ordinary_hosts[False]` | Runtime NOISE and SAMPLER with executed/cached node assertions |
| `test_sigma_operations_torch.py::test_sigma_operations_cross_both_manifest_hosts_twice[False]` | Nine native SIGMAS links and MODEL resident |
| `test_supir_torch.py::test_supir_executes_on_both_ordinary_hosts[False]` | Real MODEL_PATCH weight identity, two MODEL residents, VAE and IMAGE |
| `test_style_model_wire_torch.py::test_model_patch_full_native_manifest_and_fixture_hosts[LATENT_UPSCALE_MODEL-False]` | LATENT_UPSCALE_MODEL resident identity and cache replay |
| `test_style_model_apply_torch.py::test_style_apply_runs_through_native_conditioning_consumer_twice[False]` | Real STYLE_MODEL, CLIP_VISION_OUTPUT and CONDITIONING; cache replay |
| `test_photomaker_torch.py::test_photo_identity_crosses_ordinary_resident_hosts[False]` | PHOTOMAKER resource identity, CLIP, MODEL and IMAGE |
| `test_rtdetr_torch.py::test_ordinary_manifest_detects_and_draws_twice[False]` | BOUNDING_BOX and IMAGE |
| `test_sam3_track_transport_torch.py::test_public_sam3_track_producer_consumer_repeats_through_ordinary_hosts[False-asymmetric-present-scores]` | SAM3_TRACK_DATA packed masks, scores and cache replay |
| `test_wan_portable_values_torch.py::test_wan_portable_values_cross_ordinary_hosts_twice_with_cache[False]` | Byte-exact WAN_CAMERA_EMBEDDING, TRACKS and AUDIO_ENCODER_OUTPUT through layered cache |
