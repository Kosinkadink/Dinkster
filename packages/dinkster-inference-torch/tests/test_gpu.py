"""CUDA/Triton validation of the stage-4c cast-at-use, fp8, and
residency paths.

Everything here is capability-gated: without a CUDA torch build the
whole module skips (so the CPU-only `.venv-torch` gate is unaffected),
and the kitchen-CUDA / triton / multi-GPU tests state exactly which
capability is missing instead of silently passing on a fallback path.
Run instructions for a GPU interpreter are in this package's README
("GPU validation"). First validated 2026-07 on 2x RTX 4090 (torch
2.9.1+cu130, triton 3.5.1, dinkster-kitchen 0.2.22 wheel with the
prebuilt CUDA backend).

Dispatch honesty matters: dinkster_kitchen routes per-tensor-device, so a
test that merely calls stochastic_rounding on a CPU tensor exercises
the eager backend even when CUDA is present. The tests below pin the
registry's backend CHOICE for CUDA tensors, not just the numerics.

Two upstream kitchen-CUDA quirks are pinned here (one .md each under
docs/comfyui-issues/): the stochastic_rounding_fp8 kernel mutates its
``rng`` argument in place (eager does not; Dinkster allocates rng fresh
per call, so it is immune), and the kernel's DLPack export fails for
tensors off the current CUDA device (Dinkster pins the device context in
rounding.py; validated on both GPUs below).
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import weakref
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference import Z_IMAGE_CONFIG
from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
from dinkster_inference_torch import (
    FP8_DTYPES,
    INITLESS,
    SAGE2_PROVIDER,
    SOL_ATTENTION_PROVIDER,
    AnimaModel,
    AutoencoderKL,
    CastOperations,
    DeferredPatch,
    DeviceMemory,
    Flux,
    FluxDenoiser,
    Fp8Linear,
    Fp8ScaledWeight,
    GgufEncodedLinear,
    MemoryPolicy,
    ModuleStateStore,
    PagedKVCache,
    PagedKVGeometry,
    PagedKVSessionBusyError,
    QwenImage,
    ResidencyManager,
    ResidentWeights,
    StoredWeight,
    TAESDDecoder,
    TAESDEncoder,
    ZImageAttention,
    assemble_flux,
    cast_weight,
    collect_partial_residency_timing,
    cosmos_predict2_rope_table,
    discover_attention_route_token,
    enroll_component,
    get_total_memory,
    ltxv_vae_max_chunk_bytes,
    move_stored,
    patch_wan_ati_motion,
    patch_weights,
    payload_binding_to_tensor,
    pinned_host,
    prepare_noise,
    prepare_wan_ati_tracks,
    quantize_fp8_scaled,
    requantize_fp8_scaled,
    restore_weights,
    run_denoise,
    select_attention,
    soft_empty_cache,
    stochastic_rounding,
    stored_nbytes,
    string_to_seed,
    supports_fp8_matmul,
    tensor_to_payload_binding,
    tiled_apply,
)
from dinkster_inference_torch import quant_linear as quant_linear_mod
from dinkster_inference_torch import rounding as rounding_mod
from dinkster_inference_torch._nvfp4_diagnostics import Nvfp4DiagnosticsRecorder
from dinkster_inference_torch.attention import attention_kernel_context
from dinkster_inference_torch.gguf_linear import (
    GGUF_BLOCK_DECODERS,
    decode_q8_0_blocks,
)
from dinkster_inference_torch.model_prefetch import make_prefetch_queue, prefetch_queue_pop
from dinkster_inference_torch.quant_linear import Nvfp4ExecutionError, Nvfp4Linear
from dinkster_inference_torch.wan21_vae import WanVAE, WanVAEConfig
from dinkster_memory import PressureSignal
from golden_files import load_platform_golden
from gpu_test_gate import require_gpu_tests_enabled
from test_gguf_linear import encode_q8_0, reference_quant_blocks
from test_ltx_video_vae import build_vae as build_ltx_video_vae
from test_qwen_image import fill_parameters as fill_qwen_image_parameters
from test_qwen_image import reduced_config as reduced_qwen_image_config
from test_qwen_text import GOLDENS as QWEN_GOLDENS
from test_qwen_text import build_tiny as build_tiny_qwen
from test_qwen_text import dec as decode_qwen_golden

INFERENCE_PARITY_RECORDS = Path(
    os.environ.get(
        "DINKSTER_INFERENCE_PARITY_RECORDS",
        Path(__file__).resolve().parents[3].parent
        / "dinkster-evidence"
        / "inference-parity"
        / "records",
    )
)

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA GPU required (see README)")


def test_ltx_video_vae_derived_chunk_budget_matches_unchunked_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_gpu_tests_enabled()
    vae = build_ltx_video_vae("vae_v0").to("cuda:0")
    content = torch.linspace(-1.0, 1.0, 1 * 3 * 17 * 768 * 768, device="cuda:0").reshape(
        1, 3, 17, 768, 768
    )
    budget = ltxv_vae_max_chunk_bytes(6 * 1024**3)

    with torch.inference_mode():
        unchunked_latent = vae.encode(content, max_chunk_bytes=128 * 1024**2)
        split = torch.split
        encoder_chunk_counts: list[int] = []

        def traced_split(
            tensor: torch.Tensor, split_size_or_sections: Any, dim: int = 0
        ) -> tuple[torch.Tensor, ...]:
            chunks = split(tensor, split_size_or_sections, dim)
            encoder_chunk_counts.append(len(chunks))
            return chunks

        monkeypatch.setattr(torch, "split", traced_split)
        chunked_latent = vae.encode(content, max_chunk_bytes=budget)
        monkeypatch.setattr(torch, "split", split)
        unchunked_pixels = vae.decode(unchunked_latent, max_chunk_bytes=128 * 1024**2)
        chunked_pixels = vae.decode(unchunked_latent, max_chunk_bytes=budget)

    assert budget == 32 * 1024**2
    assert encoder_chunk_counts[0] == 2
    assert torch.equal(chunked_latent, unchunked_latent)
    assert torch.equal(chunked_pixels, unchunked_pixels)


def test_paged_kv_host_tier_restores_exact_values_to_cuda() -> None:
    require_gpu_tests_enabled()
    geometry = PagedKVGeometry(
        block_tokens=4,
        layer_count=2,
        kv_heads=2,
        head_dim=3,
        dtype=torch.float16,
    )
    cache = PagedKVCache(
        "gpu-kv",
        "gpu-test.safetensors",
        geometry,
        load_device="cuda:0",
        max_device_blocks=1,
        max_host_blocks=1,
    )
    cache.create_session("session")
    key = torch.arange(36, dtype=torch.float16, device="cuda:0").reshape(2, 3, 2, 3)
    value = key + 100
    with cache.pin("session") as lease:
        lease.append(key, value)
        lease.commit()

    assert cache.partially_unload(cache.block_nbytes) == cache.block_nbytes
    assert cache.loaded_bytes() == 0
    assert cache.offloaded_bytes() == cache.block_nbytes
    assert cache.block_states()[0].tier == "host"
    assert cache.partially_load(None) == cache.block_nbytes
    assert cache.partially_unload(cache.block_nbytes) == cache.block_nbytes

    with cache.pin("session") as lease:
        assert cache.loaded_bytes() == cache.block_nbytes
        assert cache.offloaded_bytes() == 0
        assert torch.equal(lease.block_views(0)[0].key, key[0])
        assert torch.equal(lease.block_views(1)[0].value, value[1])
        assert lease.block_views(0)[0].key.device == torch.device("cuda:0")
        with pytest.raises(PagedKVSessionBusyError, match="active sessions"):
            cache.partially_unload(cache.block_nbytes)
        detail = cache.details()[0]
        assert detail.bytes_by_residency == {"vram:cuda:0": cache.block_nbytes, "ram": 0}
        assert detail.pages is not None
        assert detail.pages.flags == (2,)

    assert cache.partially_unload(cache.block_nbytes) == cache.block_nbytes
    for index, session_id in enumerate(("second", "third"), start=1):
        cache.create_session(session_id)
        with cache.pin(session_id) as lease:
            lease.append(key + index * 1_000, value + index * 1_000)
            lease.commit()
    states = {state.session_id: state for state in cache.session_states()}
    assert set(states) == {"session", "third"}
    assert states["session"].host_block_count == 1
    assert states["third"].device_block_count == 1
    assert cache.loaded_bytes() == cache.offloaded_bytes() == cache.block_nbytes


def test_paged_kv_partial_load_handles_session_eviction_during_iteration() -> None:
    require_gpu_tests_enabled()
    geometry = PagedKVGeometry(4, 1, 1, 2, torch.float16)
    cache = PagedKVCache(
        "gpu-kv",
        "gpu-test.safetensors",
        geometry,
        load_device="cuda:0",
        max_device_blocks=1,
        max_host_blocks=1,
    )
    key = torch.arange(6, dtype=torch.float16, device="cuda:0").reshape(1, 3, 1, 2)
    cache.create_session("host")
    with cache.pin("host") as lease:
        lease.append(key, key + 100)
        lease.commit()
    cache.partially_unload(cache.block_nbytes)

    cache.create_session("victim")
    with cache.pin("victim") as lease:
        lease.append(key + 200, key + 300)
        lease.commit()
    cache.fork_session("host", "newest")

    assert cache.partially_load(None) == 0
    assert {state.session_id for state in cache.session_states()} == {"host", "newest"}
    with cache.pin("newest") as lease:
        assert torch.equal(lease.block_views(0)[0].key, key[0])


def test_paged_kv_restore_keeps_shared_target_blocks_on_cuda() -> None:
    require_gpu_tests_enabled()
    geometry = PagedKVGeometry(4, 1, 1, 2, torch.float16)
    cache = PagedKVCache(
        "gpu-kv",
        "gpu-test.safetensors",
        geometry,
        load_device="cuda:0",
        max_device_blocks=2,
        max_host_blocks=2,
    )
    key = torch.arange(16, dtype=torch.float16, device="cuda:0").reshape(1, 8, 1, 2)
    cache.create_session("target")
    with cache.pin("target") as lease:
        lease.append(key, key + 100)
        lease.commit()
    cache.fork_session("target", "shared-prefix", token_count=4)

    with cache.pin("shared-prefix"):
        assert (
            asyncio.run(cache.shed(PressureSignal("vram:cuda:0", cache.block_nbytes)))
            == cache.block_nbytes
        )
    cache.create_session("newer")
    with cache.pin("newer") as lease:
        lease.append(key[:, :4] + 200, key[:, :4] + 300)
        lease.commit()

    with cache.pin("target") as lease:
        views = lease.block_views(0)
        assert all(view.key.device == torch.device("cuda:0") for view in views)
        assert torch.equal(torch.cat([view.key for view in views]), key[0])


@pytest.mark.parametrize(
    ("token_count", "max_host_blocks", "required_blocks", "preserves_session"),
    ((3, 1, 2, True), (3, 1, 3, True), (3, 0, 2, False), (5, 1, 2, False)),
)
def test_paged_kv_manager_full_eviction_preserves_only_fully_host_backed_sessions(
    token_count: int,
    max_host_blocks: int,
    required_blocks: int,
    preserves_session: bool,
) -> None:
    require_gpu_tests_enabled()
    geometry = PagedKVGeometry(4, 1, 1, 2, torch.float16)
    cache = PagedKVCache(
        "gpu-kv",
        "gpu-test.safetensors",
        geometry,
        load_device="cuda:0",
        max_device_blocks=2,
        max_host_blocks=max_host_blocks,
    )
    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            min_weight_memory_ratio=0.0,
            load_inflation=1.0,
        ),
        free_memory=lambda _device: DeviceMemory(
            free_total=2 * cache.block_nbytes - cache.loaded_bytes(),
            free_torch=0,
        ),
        total_memory=lambda _device: 2 * cache.block_nbytes,
        empty_cache=lambda _device: None,
    )
    manager.load((cache,))
    cache.create_session("session")
    key = torch.arange(2 * token_count, dtype=torch.float16, device="cuda:0").reshape(
        1, token_count, 1, 2
    )
    value = key + 100
    with cache.pin("session") as lease:
        lease.append(key, value)
        lease.commit()
    cache.fork_session("session", "shared")

    manager.free(required_blocks * cache.block_nbytes, torch.device("cuda:0"))

    if not preserves_session:
        assert manager.registered() == ()
        assert cache.session_states() == ()
        assert cache.total_bytes() == 0
        return

    assert manager.registered() == (cache,)
    assert cache.loaded_bytes() == 0
    assert cache.offloaded_bytes() == cache.block_nbytes
    assert all(state.host_block_count == 1 for state in cache.session_states())
    with cache.pin("session") as lease:
        view = lease.block_views(0)[0]
        assert torch.equal(view.key, key[0])
        assert torch.equal(view.value, value[0])
    manager.remove((cache,))
    assert manager.registered() == ()
    assert cache.session_states() == ()
    assert cache.total_bytes() == 0


def test_wan_ati_motion_projection_executes_on_cuda() -> None:
    require_gpu_tests_enabled()
    tracks = prepare_wan_ati_tracks(
        '[[{"x": 3, "y": 5}, {"x": 4, "y": 6}]]',
        width=16,
        height=16,
        length=5,
        batch_size=1,
    )
    video = torch.linspace(-1.0, 1.0, 128, device="cuda").reshape(1, 16, 2, 2, 2)

    mask, motion = patch_wan_ati_motion(tracks, video, temperature=7.5, topk=1)
    torch.cuda.synchronize()

    assert mask.shape == (1, 4, 2, 2, 2)
    assert motion.shape == video.shape
    assert mask.device.type == motion.device.type == "cuda"
    assert torch.equal(mask[:, :, :1], torch.ones_like(mask[:, :, :1]))
    assert torch.isfinite(mask).all()
    assert torch.isfinite(motion).all()


def test_official_sd15_t2i_adapter_executes_on_cuda() -> None:
    from test_t2i_adapter import CHECKPOINT, _plan  # pyright: ignore[reportPrivateUsage]

    if not CHECKPOINT.is_file():
        pytest.skip("official local T2I Adapter fixture is unavailable")
    from dinkster_inference_torch import assemble_sd15_t2i_adapter

    adapter = assemble_sd15_t2i_adapter(_plan(CHECKPOINT), adapter_dtype=torch.float32).adapter
    adapter.to("cuda:0")
    hint = torch.linspace(0.0, 1.0, 64 * 64, device="cuda:0").reshape(1, 1, 64, 64)
    with torch.no_grad():
        residuals = adapter(hint)
    torch.cuda.synchronize()
    assert all(
        value.device.type == "cuda" and bool(torch.isfinite(value).all())
        for value in (*residuals.down, residuals.middle)
    )


def test_cast_embedding_cuda_gathers_fp8_rows_before_compute_cast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = CastOperations(torch.float32).embedding(128, 16).to("cuda:0")
    weight = torch.randn((128, 16), device="cuda:0").to(torch.float8_e4m3fn)
    embedding.load_state_dict({"weight": weight}, assign=True)
    token_ids = torch.tensor([[7, 3, 7]], device="cuda:0")
    seen: list[torch.Tensor] = []
    original = torch.nn.functional.embedding

    def tracked(input: torch.Tensor, stored: torch.Tensor, *args: Any) -> torch.Tensor:
        seen.append(stored)
        return original(input, stored, *args)

    monkeypatch.setattr(torch.nn.functional, "embedding", tracked)
    with torch.no_grad():
        output = embedding(token_ids)

    assert seen == [embedding.weight]
    assert seen[0].dtype == torch.float8_e4m3fn
    assert output.dtype == torch.float32
    assert torch.equal(output, original(token_ids, weight).to(torch.float32))


def test_shared_attention_prioritizes_cudnn_when_flash_is_unavailable() -> None:
    q = torch.randn((1, 16, 1024, 128), device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    if torch.backends.cuda.can_use_flash_attention(params):
        pytest.skip("this GPU supports the higher-priority Flash backend")
    if not torch.backends.cuda.can_use_cudnn_attention(params):
        pytest.skip("this GPU cannot execute the cuDNN SDPA backend")

    kernel = select_attention("flux").kernel
    kernel(q, k, v)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as captured:
        kernel(q, k, v)
    torch.cuda.synchronize()

    operators = {event.key for event in captured.key_averages()}
    assert "aten::_scaled_dot_product_cudnn_attention" in operators


def test_shared_attention_priority_context_preserves_large_output() -> None:
    require_gpu_tests_enabled()
    generator = torch.Generator(device="cuda:0").manual_seed(841)
    q = torch.randn((1, 16, 1024, 128), generator=generator, device="cuda:0", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    kernel = select_attention("flux").kernel

    expected = kernel(q, k, v)
    with attention_kernel_context(kernel, q.numel(), device=q.device):
        actual = kernel(q, k, v)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("outer", (False, True))
@pytest.mark.parametrize("caller_mode", ("default", "disabled", "custom"))
def test_first_cuda_attention_preserves_priority_and_output(outer: bool, caller_mode: str) -> None:
    require_gpu_tests_enabled()
    script = textwrap.dedent(
        """
        import sys
        from contextlib import nullcontext
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from dinkster_inference_torch.attention import builtin_sdpa_kernel, attention_kernel_context

        assert not torch.cuda.is_initialized()
        generator = torch.Generator('cpu').manual_seed(220)
        q, k, v = [torch.randn((1, 56, 32, 128), generator=generator,
                              dtype=torch.bfloat16).cuda() for _ in range(3)]
        torch.use_deterministic_algorithms(True)
        original_choice = torch._fused_sdp_choice
        choice_calls = []
        choice_events = []
        choice_memory = []
        def recording_choice(*args, **kwargs):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
            ) as captured:
                result = original_choice(*args, **kwargs)
            torch.cuda.synchronize()
            choice_calls.append((args, kwargs))
            choice_events.extend(event.name for event in captured.events()
                                 if event.device_type.name == 'CUDA')
            choice_memory.extend((event.name, event.device_memory_usage)
                                 for event in captured.events()
                                 if event.device_memory_usage != 0)
            return result
        torch._fused_sdp_choice = recording_choice
        selected = builtin_sdpa_kernel()
        original = F.scaled_dot_product_attention
        orders = []
        def recording(*args, **kwargs):
            before = torch._C._get_sdp_priority_order()[:4]
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as captured:
                result = original(*args, **kwargs)
            torch.cuda.synchronize()
            backend = sorted(event.key for event in captured.key_averages()
                             if event.key.startswith('aten::_scaled_dot_product_'))
            orders.append((before, torch._C._get_sdp_priority_order()[:4]))
            return result, backend
        F.scaled_dot_product_attention = recording
        caller_priority = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
                           SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]
        caller_context = (sdpa_kernel([]) if sys.argv[2] == 'disabled'
                          else sdpa_kernel(caller_priority, set_priority=True)
                          if sys.argv[2] == 'custom' else nullcontext())
        with caller_context:
            enabled_before = [torch.backends.cuda.flash_sdp_enabled(),
                              torch.backends.cuda.cudnn_sdp_enabled(),
                              torch.backends.cuda.mem_efficient_sdp_enabled(),
                              torch.backends.cuda.math_sdp_enabled()]
            priority_before = torch._C._get_sdp_priority_order()
            results = []
            for _ in range(4):
                context = (attention_kernel_context(selected, q.numel(), device=q.device)
                           if sys.argv[1] == 'True' else nullcontext())
                with context:
                    results.append(selected(q, k, v))
                enabled_after = [torch.backends.cuda.flash_sdp_enabled(),
                                 torch.backends.cuda.cudnn_sdp_enabled(),
                                 torch.backends.cuda.mem_efficient_sdp_enabled(),
                                 torch.backends.cuda.math_sdp_enabled()]
                assert enabled_after == enabled_before
                assert torch._C._get_sdp_priority_order() == priority_before
        assert len(choice_calls) == 1, choice_calls
        assert choice_events == [], choice_events
        assert choice_memory == [], choice_memory
        assert orders == [([1, 3, 2, 0], [1, 3, 2, 0])] * 4, orders
        outputs = [result[0] for result in results]
        backends = [result[1] for result in results]
        assert all(backends[0] == backend for backend in backends[1:]), backends
        assert len(backends[0]) == 1, backends
        assert all(torch.equal(outputs[0], value) for value in outputs[1:])
        print('FIRST_CUDA_SDPA_PRIORITY_EXACT')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(outer), caller_mode],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIRST_CUDA_SDPA_PRIORITY_EXACT" in result.stdout


def test_z_image_qk_rmsnorm_uses_official_epsilon_on_cuda() -> None:
    with torch.device("meta"):
        attention = ZImageAttention(
            Z_IMAGE_CONFIG,
            operations=INITLESS,
            attention_kernel=select_attention("flux").kernel,
        )
    q_norm = attention.q_norm.to_empty(device="cuda:0").to(dtype=torch.bfloat16)
    with torch.no_grad():
        q_norm.weight.copy_(torch.linspace(0.5, 1.5, 128, device="cuda:0", dtype=torch.bfloat16))
    value = torch.linspace(-2.0, 2.0, 2 * 3 * 128, device="cuda:0", dtype=torch.bfloat16).view(
        2, 3, 128
    )

    actual = q_norm(value)
    official_epsilon = torch.nn.functional.rms_norm(value, (128,), q_norm.weight, eps=1e-5)
    dtype_default = torch.nn.functional.rms_norm(
        value, (128,), q_norm.weight, eps=torch.finfo(torch.bfloat16).eps
    )
    torch.testing.assert_close(actual, official_epsilon, rtol=0, atol=0)
    assert not torch.equal(actual, dtype_default)


def test_int8_convrot_embedding_matches_kitchen_on_cuda() -> None:
    pytest.importorskip("dinkster_kitchen")
    from dinkster_inference_torch.quant_linear import Int8Embedding

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260830)
    weight = torch.randint(
        -100, 101, (11, 256), generator=generator, device=device, dtype=torch.int8
    )
    scale = torch.rand((11, 1), generator=generator, device=device, dtype=torch.float32) / 100
    indices = torch.tensor([[1, 7, 4], [10, 0, 3]], device=device)
    layer = Int8Embedding(
        11,
        256,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=256,
    ).to(device)
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)

    expected = torch.ops.dinkster_kitchen.dequantize_int8_embedding(
        weight, scale, indices, 256, 2
    ).to(torch.bfloat16)
    torch.testing.assert_close(layer(indices), expected, rtol=0, atol=0)


def test_gemma4_rope_matches_cpu_precomputed_inverse_frequencies_on_cuda() -> None:
    from dinkster_inference_torch import gemma_text as gemma_text_mod

    device = torch.device("cuda:0")
    cosine, sine, negative_sine = gemma_text_mod._rope(  # pyright: ignore[reportPrivateUsage]
        512,
        1024,
        1_000_000.0,
        1.0,
        device=device,
        rotary_fraction=0.25,
        dtype=torch.bfloat16,
        precompute_inverse_on_cpu=True,
    )
    numerator = torch.arange(0, 128, 2).float()
    inverse = 1.0 / (1_000_000.0 ** (numerator / 512))
    inverse = torch.cat((inverse, torch.zeros(192))).to(device)
    positions = torch.arange(1024, device=device).float()
    frequencies = (inverse[:, None] @ positions[None, :]).transpose(0, 1)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    expected_cosine = embedding.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    expected_sine = embedding.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    assert torch.equal(cosine, expected_cosine)
    assert torch.equal(sine, expected_sine[..., :256])
    assert torch.equal(negative_sine, -expected_sine[..., 256:])


def test_int8_fused_training_forward_and_input_gradient_on_cuda() -> None:
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
    from dinkster_inference_torch.quant_linear import Int8Linear

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(543)
    weight = torch.randint(
        -100, 101, (384, 256), generator=generator, device=device, dtype=torch.int8
    )
    scale = torch.rand((384, 1), generator=generator, device=device, dtype=torch.float32) / 100
    bias = torch.randn(384, generator=generator, device=device, dtype=torch.bfloat16)
    input = torch.randn(
        (2, 3, 256), generator=generator, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    layer = Int8Linear(
        256,
        384,
        bias=True,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=64,
    ).to(device)
    layer.load_state_dict({"weight": weight, "weight_scale": scale, "bias": bias}, assign=True)
    layer.bind_fused_training(True)
    grad_output = torch.randn((2, 3, 384), generator=generator, device=device, dtype=torch.bfloat16)
    expected = dinkster_kitchen.int8_linear(
        input.detach(),
        weight,
        scale,
        bias,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=64,
    )
    dequantized = torch.ops.dinkster_kitchen.dequantize_int8_convrot_weight_dtype(
        weight, scale, 64, 2
    )
    expected_grad = grad_output.reshape(-1, 384).matmul(dequantized).reshape(input.shape)

    actual = layer(input)
    actual.backward(grad_output)

    assert torch.equal(actual, expected)
    assert input.grad is not None
    torch.testing.assert_close(input.grad, expected_grad, rtol=0, atol=0)


def test_int8_fused_training_accepts_unaligned_output_width_on_cuda() -> None:
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
    from dinkster_inference_torch.quant_linear import Int8Linear

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(545)
    weight = torch.randint(-100, 101, (6, 64), generator=generator, device=device, dtype=torch.int8)
    scale = torch.rand((), generator=generator, device=device, dtype=torch.float32) / 100
    input = torch.randn(
        (2, 64), generator=generator, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    layer = Int8Linear(
        64,
        6,
        bias=False,
        compute_dtype=torch.bfloat16,
        convrot=False,
        convrot_groupsize=64,
    ).to(device)
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)
    layer.bind_fused_training(True)
    grad_output = torch.randn((2, 6), generator=generator, device=device, dtype=torch.bfloat16)
    expected = dinkster_kitchen.int8_linear(
        input.detach(),
        weight,
        scale,
        None,
        out_dtype=torch.bfloat16,
    )
    expected_grad = grad_output.matmul(weight.to(dtype=torch.bfloat16) * scale)

    actual = layer(input)
    actual.backward(grad_output)

    assert torch.equal(actual, expected)
    assert input.grad is not None
    torch.testing.assert_close(input.grad, expected_grad, rtol=0, atol=0)


def test_int8_fused_training_casts_autocast_gradient_to_matmul_dtype() -> None:
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
    from dinkster_inference_torch.quant_linear import Int8Linear

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(544)
    weight = torch.randint(
        -100, 101, (384, 256), generator=generator, device=device, dtype=torch.int8
    )
    scale = torch.rand((384, 1), generator=generator, device=device, dtype=torch.float32) / 100
    input = torch.randn(
        (2, 3, 256), generator=generator, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    grad_output = torch.randn((2, 3, 384), generator=generator, device=device, dtype=torch.bfloat16)
    layer = Int8Linear(
        256,
        384,
        bias=False,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=64,
    ).to(device)
    layer.load_state_dict({"weight": weight, "weight_scale": scale}, assign=True)
    layer.bind_fused_training(True)
    expected = dinkster_kitchen.int8_linear(
        input.detach().to(torch.float16),
        weight,
        scale,
        None,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=64,
    )
    dequantized = torch.ops.dinkster_kitchen.dequantize_int8_convrot_weight_dtype(
        weight, scale, 64, 1
    )
    expected_grad = (
        grad_output.reshape(-1, 384)
        .to(torch.float16)
        .matmul(dequantized)
        .reshape(input.shape)
        .to(torch.bfloat16)
    )

    with torch.autocast("cuda", dtype=torch.float16):
        actual = layer(input)
    actual.backward(grad_output)

    assert torch.equal(actual, expected)
    assert input.grad is not None
    torch.testing.assert_close(input.grad, expected_grad, rtol=0, atol=0)


def test_int8_fused_training_bounds_real_h3_adaln_backward_peak() -> None:
    from dinkster_inference_torch.quant_linear import Int8Linear

    device = torch.device("cuda:0")
    output_features = 96_768
    input_features = 2_688
    layer = Int8Linear(
        input_features,
        output_features,
        bias=False,
        compute_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=64,
    ).to(device)
    layer.weight.data.zero_()
    layer.weight_scale.data.fill_(0.001)
    layer.bind_fused_training(True)
    warm = torch.zeros((3, input_features), device=device, dtype=torch.bfloat16)
    layer(warm)
    torch.cuda.synchronize(device)
    del warm
    gc.collect()
    torch.cuda.empty_cache()

    input = torch.zeros(
        (3, input_features), device=device, dtype=torch.bfloat16, requires_grad=True
    )
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    layer(input).sum().backward()
    torch.cuda.synchronize(device)
    incremental_peak = torch.cuda.max_memory_allocated(device) - baseline

    assert input.grad is not None
    assert incremental_peak <= 192 * 1024 * 1024
    del layer, input
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("input_act", ["swiglu", "gelu_tanh"])
def test_int8_input_activation_folding_matches_materialized_order(
    input_act: Literal["gelu_tanh", "swiglu"],
) -> None:
    require_gpu_tests_enabled()
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
    from dinkster_inference_torch.quant_linear import Int8Linear
    from dinkster_kitchen.tensor import (  # pyright: ignore[reportMissingTypeStubs]
        QuantizedTensor,
        TensorWiseINT8Layout,
    )

    torch.manual_seed(20260823)
    device = torch.device("cuda:0")
    width = 256
    output_width = 384
    input_width = width * 2 if input_act == "swiglu" else width
    input = torch.randn((1, 32, input_width), device=device, dtype=torch.float16)
    source_weight = torch.randn((output_width, width), device=device, dtype=torch.float16).mul_(
        0.02
    )
    weight, params = TensorWiseINT8Layout.quantize(
        source_weight,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=256,
    )
    wrapped = QuantizedTensor(
        weight,
        "TensorWiseINT8Layout",
        params,
    )
    layer = Int8Linear(
        width,
        output_width,
        bias=False,
        compute_dtype=torch.float16,
        convrot=True,
        convrot_groupsize=256,
    ).to(device)
    layer.load_state_dict({"weight": weight, "weight_scale": params.scale}, assign=True)

    if input_act == "swiglu":
        gate, up = input.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
    else:
        activated = torch.nn.functional.gelu(input, approximate="tanh")
    expected = torch.nn.functional.linear(activated, wrapped)
    actual = quant_linear_mod.linear_input_act(layer, input, input_act)
    fused = dinkster_kitchen.int8_linear(
        input,
        weight,
        params.scale,
        out_dtype=torch.float16,
        convrot=True,
        convrot_groupsize=256,
        input_act=input_act,
    )

    # Five seeds at this valid 256-wide geometry measured max drift
    # 0.00323486328125; 0.00390625 gives 20% headroom for fused
    # activation-kernel rounding without a relative tolerance.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.00390625)
    assert torch.equal(actual, fused)


def test_h3_int8_guidance_identity_is_available_on_gpu(tmp_path: Path) -> None:
    import test_minimax_h3_assembly as h3_assembly
    from dinkster_inference_torch.minimax_h3_assembly import minimax_h3_guidance_receipt_identity

    fl2va, _ref2va, _sources = h3_assembly._runtime_plans(tmp_path)  # pyright: ignore[reportPrivateUsage]
    int8_fl2va = h3_assembly._int8_convrot_plan(fl2va)  # pyright: ignore[reportPrivateUsage]
    identity = minimax_h3_guidance_receipt_identity(int8_fl2va)
    assert identity.startswith("distributed:dinkster.minimax_h3:")


_H3_VIDEO_VAE_FP16 = Path("/home/kosin/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors")
_H3_VIDEO_VAE_INT8 = Path(
    "/home/kosin/ComfyUI/models/vae/minimax_h3_video_vae_int8_convrot.safetensors"
)


@pytest.mark.skipif(
    not (_H3_VIDEO_VAE_FP16.exists() and _H3_VIDEO_VAE_INT8.exists()),
    reason="FP16 and INT8 ConvRot H3 video VAEs are required",
)
def test_real_h3_int8_video_vae_matches_fp16_and_reduces_peak_memory() -> None:
    from dinkster_assets import digest_file
    from dinkster_inference.sources import load_safetensors_header_from_file
    from dinkster_inference_torch import minimax_h3_assembly as assembly
    from dinkster_inference_torch.assemble import (
        _load_component,  # pyright: ignore[reportPrivateUsage]
    )
    from dinkster_inference_torch.minimax_h3_video_vae import MiniMaxH3VideoVAE
    from dinkster_inference_torch.quant_linear import Int8Linear

    def load(path: Path) -> MiniMaxH3VideoVAE:
        with path.open("rb") as handle:
            size = path.stat().st_size
            digest = digest_file(path)
            assembly._verify_artifact_file(  # pyright: ignore[reportPrivateUsage]
                "video-vae", path, handle, digest, size
            )
            source = load_safetensors_header_from_file(handle, path=path)
            with torch.device("meta"):
                layout = assembly._module_layout(  # pyright: ignore[reportPrivateUsage]
                    MiniMaxH3VideoVAE()
                )
            storage, quant, _ = assembly._extract_quantized_source(  # pyright: ignore[reportPrivateUsage]
                source, "video VAE", layout
            )
            plan = assembly._component_plan(  # pyright: ignore[reportPrivateUsage]
                component="video_vae",
                path=path,
                config=assembly.MiniMaxH3VideoVAEConfig(),
                storage=storage,
                quant=quant,
            )
            return _load_component(  # pyright: ignore[reportPrivateUsage]
                plan,
                MiniMaxH3VideoVAE,
                compute_dtype=torch.float16,
                fp8_matmul=False,
                source_file=handle,
                source=source,
            )

    def decode(path: Path, latent: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        model = load(path)
        int8_layers = sum(isinstance(module, Int8Linear) for module in model.modules())
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model.to(latent.device)
            output = model.decode(latent).float().cpu()
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return output, peak, int8_layers

    torch.manual_seed(123)
    latent = torch.randn((1, 24, 2, 2, 2), device="cuda:0", dtype=torch.float16)
    expected, fp16_peak, fp16_int8_layers = decode(_H3_VIDEO_VAE_FP16, latent)
    actual, int8_peak, int8_layers = decode(_H3_VIDEO_VAE_INT8, latent)

    assert fp16_int8_layers == 0
    assert int8_layers == 144
    assert actual.shape == expected.shape == (1, 3, 5, 32, 32)
    difference = (actual - expected).abs()
    # Temporal max/mean drift is 0.026797/0.002059; limits leave 12%/46% headroom.
    assert difference.max().item() <= 0.03
    assert difference.mean().item() <= 0.003
    # Peaks are 3,202,677,248/5,224,113,664 bytes (0.613057); limit leaves 6% headroom.
    assert int8_peak / fp16_peak <= 0.65


kitchen_available = (
    rounding_mod._probe_kitchen_fp8_kernel()  # pyright: ignore[reportPrivateUsage]
    is not None
)

CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cuda:0"]


def test_nvfp4_blackwell_acceleration_fallback_and_loud_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.major != 12:
        pytest.skip("NVFP4 Kitchen acceleration requires a Blackwell (SM12x) GPU")

    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]
    # Kitchen's CUDA scaled-mm contract requires packed K/2 divisible by 16.
    weight = torch.linspace(-2.0, 2.0, 512, device=device, dtype=torch.bfloat16).reshape(16, 32)
    tensor_scale = torch.amax(weight.abs()).to(torch.float32) / (448.0 * 6.0)
    qweight, block_scale = kitchen.quantize(weight, tensor_scale, pad_16x=False)
    layer = Nvfp4Linear(32, 16, bias=False, compute_dtype=torch.bfloat16, input_scale=False).to(
        device
    )
    diagnostics = Nvfp4DiagnosticsRecorder()
    layer._bind_diagnostics(diagnostics)  # pyright: ignore[reportPrivateUsage]
    layer.load_state_dict(
        {
            "weight": qweight,
            "weight_scale": block_scale,
            "weight_scale_2": tensor_scale,
        },
        strict=True,
        assign=True,
    )
    x = torch.randn(16, 32, device=device, dtype=torch.bfloat16)
    calls: list[str] = []

    def observed_quantize(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("quantize")
        return kitchen.quantize(*args, **kwargs)

    def observed_scaled_mm(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append("scaled_mm")
        return kitchen.scaled_mm(*args, **kwargs)

    monkeypatch.setattr(
        quant_linear_mod,
        "_nvfp4_kitchen",
        quant_linear_mod._Nvfp4Kitchen(  # pyright: ignore[reportPrivateUsage]
            observed_quantize,
            kitchen.dequantize,
            observed_scaled_mm,
            kitchen.registry,
        ),
    )
    with torch.no_grad():
        accelerated = layer(x)
    assert calls == ["quantize", "scaled_mm"]
    assert accelerated.shape == (16, 16)
    assert bool(torch.isfinite(accelerated).all())

    class UnsupportedRegistry:
        def get_capable_backend(self, _operation: str, _kwargs: object) -> str:
            return "eager"

    monkeypatch.setattr(
        quant_linear_mod,
        "_nvfp4_kitchen",
        quant_linear_mod._Nvfp4Kitchen(  # pyright: ignore[reportPrivateUsage]
            kitchen.quantize,
            kitchen.dequantize,
            kitchen.scaled_mm,
            UnsupportedRegistry(),
        ),
    )
    with torch.no_grad():
        fallback = layer(x)
    assert fallback.shape == accelerated.shape
    assert bool(torch.isfinite(fallback).all())
    dequantized = kitchen.dequantize(
        qweight,
        tensor_scale,
        block_scale,
        output_type=torch.bfloat16,
    )
    assert torch.equal(fallback, torch.nn.functional.linear(x, dequantized))

    def corrupt_kernel(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("injected native corruption")

    monkeypatch.setattr(
        quant_linear_mod,
        "_nvfp4_kitchen",
        quant_linear_mod._Nvfp4Kitchen(  # pyright: ignore[reportPrivateUsage]
            corrupt_kernel,
            kitchen.dequantize,
            kitchen.scaled_mm,
            kitchen.registry,
        ),
    )
    with torch.no_grad(), pytest.raises(Nvfp4ExecutionError, match="injected native corruption"):
        layer(x)
    assert diagnostics.snapshot().lifetime == {
        "quantize_success": 1,
        "scaled_mm_success": 1,
        "route_native": 1,
        "route_no_quantize_backend": 1,
        "route_backend_fallback": 1,
        "dequantize_success": 1,
        "quantize_error": 1,
        "observed_loaded": 3,
    }


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_conditioning_payload_adapter_normalizes_cuda_tensor(device: str) -> None:
    source = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4).T
    binding = tensor_to_payload_binding("cuda", source, space="proof")
    expected = bytes(source.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
    assert binding.data == expected
    decoded = payload_binding_to_tensor(binding)
    assert decoded.device.type == "cpu"
    assert torch.equal(decoded, source.cpu())


def test_gpu_sde_variants_native_reference_subprocess() -> None:
    """GPU-tree bindings and float-sensitive solvers match pinned ComfyUI."""
    require_gpu_tests_enabled()
    root = Path(__file__).resolve().parents[3]
    reference = Path(os.environ.get("COMFYUI_ROOT", root.parent / "ComfyUI"))
    script = textwrap.dedent(
        f"""
        import os, sys, types
        if os.environ.get("DINKSTER_ENABLE_GPU_TESTS") != "1":
            raise RuntimeError("GPU test worker requires DINKSTER_ENABLE_GPU_TESTS=1")
        import torch
        stub = types.ModuleType("comfy.model_patcher")
        sampling_stub = types.ModuleType("comfy.model_sampling")
        sampling_stub.CONST = type("CONST", (), {{}})
        memory_stub = types.ModuleType("comfy.memory_management")
        utils_stub = types.ModuleType("comfy.utils")
        utils_stub.model_trange = lambda count, **kwargs: range(count)
        sys.modules.update({{
            "comfy.model_patcher": stub,
            "comfy.model_sampling": sampling_stub,
            "comfy.memory_management": memory_stub,
            "comfy.utils": utils_stub,
        }})
        sys.path.insert(0, {str(reference)!r})
        import comfy
        comfy.model_patcher = stub
        comfy.model_sampling = sampling_stub
        comfy.memory_management = memory_stub
        from comfy.k_diffusion import sampling as ref
        from dinkster_inference import Parameterization, SamplerInfo
        from dinkster_inference_torch import BrownianTreeNoise
        from dinkster_inference_torch.solvers import torch_sampler_registry

        x = torch.linspace(-1, 1, 24, device="cuda").reshape(1, 3, 2, 4)
        sigmas = torch.tensor([1.0, 0.7, 0.3, 0.0], device="cuda")
        class Model:
            def __init__(self, model_sampling):
                self.inner_model = types.SimpleNamespace(
                    model_patcher=types.SimpleNamespace(
                        get_model_object=lambda name: model_sampling
                    )
                )
            def __call__(self, value, sigma, **kwargs):
                return value / (1 + sigma.reshape(-1, 1, 1, 1))
        def denoiser(value, sigma):
            sigma_tensor = torch.full(
                (value.shape[0],), sigma, dtype=torch.float32, device=value.device
            )
            return value / (1 + sigma_tensor.reshape(-1, 1, 1, 1))

        expected_dpmpp_2m = ref.sample_dpmpp_2m(
            Model(object()), x.clone(), sigmas, disable=True
        )
        dpmpp_2m = torch_sampler_registry().get("dpmpp_2m")
        assert dpmpp_2m is not None
        actual_dpmpp_2m = dpmpp_2m.build()(
            denoiser,
            x.clone(),
            tuple(float(value) for value in sigmas.cpu()),
            SamplerInfo(Parameterization.EPS),
        )
        assert torch.equal(actual_dpmpp_2m, expected_dpmpp_2m)

        flow_sigmas = torch.tensor(
            [0.9999857130611196, 0.9723517894744873, 0.875, 0.5839160680770874, 0.0],
            device="cuda",
        )
        fixed_noise_value = torch.linspace(-0.5, 0.5, 24, device="cuda").reshape_as(x)
        def fixed_noise(sigma, sigma_next):
            return fixed_noise_value
        expected_euler_ancestral = ref.sample_euler_ancestral_RF(
            Model(sampling_stub.CONST()),
            x.clone(),
            flow_sigmas,
            disable=True,
            noise_sampler=fixed_noise,
        )
        euler_ancestral = torch_sampler_registry().get("euler_ancestral")
        assert euler_ancestral is not None
        actual_euler_ancestral = euler_ancestral.build()(
            denoiser,
            x.clone(),
            tuple(float(value) for value in flow_sigmas.cpu()),
            SamplerInfo(Parameterization.FLOW),
            noise=fixed_noise,
        )
        assert torch.equal(actual_euler_ancestral, expected_euler_ancestral)

        expected = ref.sample_dpmpp_sde_gpu(
            Model(sampling_stub.CONST()),
            x.clone(),
            flow_sigmas,
            disable=True,
            noise_sampler=fixed_noise,
        )
        descriptor = torch_sampler_registry().get("dpmpp_sde_gpu")
        assert descriptor is not None
        actual = descriptor.build()(
            denoiser,
            x.clone(),
            tuple(float(value) for value in flow_sigmas.cpu()),
            SamplerInfo(Parameterization.FLOW),
            noise=fixed_noise,
        )
        assert torch.equal(actual, expected)

        production_from = torch.tensor(0.875, device="cuda")
        production_to = torch.tensor(0.8321351408958435, device="cuda")
        reference_cpu_noise = ref.BrownianTreeNoiseSampler(
            x, flow_sigmas[flow_sigmas > 0].min(), flow_sigmas.max(), seed=123, cpu=True
        )
        expected_noise = reference_cpu_noise(production_from, production_to)
        cpu_noise = BrownianTreeNoise(
            x,
            float(flow_sigmas[flow_sigmas > 0].min()),
            float(flow_sigmas.max()),
            seed=123,
            cpu=True,
        )
        actual_noise = cpu_noise(float(production_from), float(production_to))
        assert torch.equal(actual_noise, expected_noise)

        cases = (
            ("dpmpp_sde_gpu", ref.sample_dpmpp_sde_gpu),
            ("dpmpp_2m_sde_gpu", ref.sample_dpmpp_2m_sde_gpu),
            ("dpmpp_2m_sde_heun_gpu", ref.sample_dpmpp_2m_sde_heun_gpu),
            ("dpmpp_3m_sde_gpu", ref.sample_dpmpp_3m_sde_gpu),
        )
        for name, reference_fn in cases:
            model = Model(object())
            reference_noise = ref.BrownianTreeNoiseSampler(
                x, sigmas[sigmas > 0].min(), sigmas.max(), seed=123, cpu=False
            )
            expected = reference_fn(
                model, x.clone(), sigmas, disable=True, noise_sampler=reference_noise
            )
            descriptor = torch_sampler_registry().get(name)
            assert descriptor is not None
            noise = BrownianTreeNoise(x, 0.3, 1.0, seed=123, cpu=False)
            assert noise._tree.device.type == "cuda"  # pyright: ignore[reportPrivateUsage]
            actual = descriptor.build()(
                denoiser, x.clone(), tuple(float(value) for value in sigmas.cpu()),
                SamplerInfo(Parameterization.EPS), noise=noise,
            )
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            assert actual.device.type == "cuda"
        print("GPU_SOLVER_REFERENCE_OK=8")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GPU_SOLVER_REFERENCE_OK=8" in result.stdout


def test_aimdo_dual_dtype_warm_forwards_stay_bit_identical() -> None:
    """Warm forwards that flip one key between two dtypes re-transfer
    through the transfer streams every pass; the overwrites must be
    ordered after in-flight compute reading the previous representation
    or warm outputs corrupt (issue #857)."""
    require_gpu_tests_enabled()
    import importlib.util

    if importlib.util.find_spec("dinkster_aimdo") is None:
        pytest.skip("dinkster_aimdo is not installed")
    root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        """
        import os
        if os.environ.get("DINKSTER_ENABLE_GPU_TESTS") != "1":
            raise RuntimeError("GPU test worker requires DINKSTER_ENABLE_GPU_TESTS=1")
        import dinkster_aimdo.control as control
        assert control.init(simple_vram_headroom=1 << 28)
        import torch
        from dinkster_inference_torch import aimdo_residency
        from dinkster_inference_torch.aimdo_activation import ensure_aimdo_devices

        ready = ensure_aimdo_devices(
            tuple((index, (1 << 28) if index == 0 else 0)
                  for index in range(torch.cuda.device_count()))
        )
        assert ready is True, f"aimdo activation failed: {ready!r}"
        device = torch.device("cuda", 0)
        generator = torch.Generator().manual_seed(857)
        weights = {
            "w": torch.randn(4096, 4096, dtype=torch.float32, generator=generator),
            "prime_a": torch.randn(4096, 4096, dtype=torch.float32, generator=generator),
            "prime_b": torch.randn(4096, 4096, dtype=torch.float32, generator=generator),
        }
        mechanism = aimdo_residency.AimdoWeights(
            weights,
            load_device=device,
            offload_device=torch.device("cpu"),
        )
        x = torch.randn(4096, 4096, dtype=torch.float32, generator=generator)
        x = x.to(device=device, dtype=torch.bfloat16)

        # Fault one equal-sized priming key onto each transfer stream so
        # both arenas exist and largest_ref stays on a different key.
        # Otherwise the single-request arena re-rotation parks every "w"
        # transfer on one stream, whose rotation edges already order the
        # overwrites, and the warm passes never exercise the race.
        with mechanism.lease("prime_a") as lease:
            lease.get("prime_a", dtype=torch.float32)
        with mechanism.lease("prime_b") as lease:
            lease.get("prime_b", dtype=torch.float32)

        baseline = None
        for forward in range(6):
            handle = mechanism.prefetch((("w", torch.float32),))
            assert handle is not None, f"forward {forward} staged nothing"
            with mechanism.lease("w") as lease:
                w = lease.get("w", dtype=torch.bfloat16)
                y = x
                for _ in range(8):
                    y = (y @ w).clamp(-3, 3)
            if handle is not None:
                handle.close()
            out = y.float().cpu()
            assert bool(torch.isfinite(out).all()), f"forward {forward} not finite"
            if baseline is None:
                baseline = out
            else:
                assert torch.equal(out, baseline), f"forward {forward} diverged"
        mechanism.unload()
        print("AIMDO_WARM_FORWARD_OK=6")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AIMDO_WARM_FORWARD_OK=6" in result.stdout


def _kitchen_cuda_backend_available() -> bool:
    """True when dinkster_kitchen's compiled CUDA backend is importable
    AND advertises the stochastic-rounding kernel - the only state in
    which the accelerated GPU rounding path is actually proven."""
    if not kitchen_available:
        return False
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]

    info = dinkster_kitchen.list_backends().get("cuda")
    return bool(info and info["available"] and "stochastic_rounding_fp8" in info["capabilities"])


def _triton_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("triton") is not None
    except Exception:
        return False


requires_kitchen_cuda = pytest.mark.skipif(
    not _kitchen_cuda_backend_available(),
    reason=(
        "dinkster-kitchen CUDA backend unavailable - accelerated GPU"
        " rounding NOT proven (install the PyPI wheel, which ships"
        " the prebuilt _C.abi3.so; see README)"
    ),
)

requires_triton = pytest.mark.skipif(
    not _triton_available(),
    reason="triton unavailable - inductor CUDA compile NOT proven",
)

requires_two_gpus = pytest.mark.skipif(
    not cuda_available or torch.cuda.device_count() < 2,
    reason="needs two CUDA devices",
)


@requires_two_gpus
def test_multidevice_attention_preserves_owner_stream_and_sdpa_values() -> None:
    from dinkster_inference_torch.attention import builtin_sdpa_kernel
    from dinkster_inference_torch.multidevice_attention import MultiDeviceAttentionKernel

    owner = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=owner)
    inner = builtin_sdpa_kernel()
    generator = torch.Generator(device=owner).manual_seed(275)
    with torch.cuda.stream(stream):
        q = torch.randn(1, 8, 128, 64, device=owner, dtype=torch.bfloat16, generator=generator)
        k = torch.randn(1, 8, 128, 64, device=owner, dtype=torch.bfloat16, generator=generator)
        v = torch.randn(1, 8, 128, 64, device=owner, dtype=torch.bfloat16, generator=generator)
        expected = inner(q, k, v)
        with MultiDeviceAttentionKernel(
            inner,
            (owner, torch.device("cuda:1")),
        ) as kernel:
            actual = kernel(q, k, v)
            repeated = kernel(q, k, v)
            stressed = tuple(kernel(q, k, v) for _ in range(16))
    stream.synchronize()

    assert actual.device == owner
    assert torch.equal(actual, expected)
    assert torch.equal(repeated, expected)
    assert all(torch.equal(item, expected) for item in stressed)


@requires_two_gpus
def test_multidevice_attention_supports_cpu_owner_and_reuses_after_lane_failure() -> None:
    from dinkster_inference_torch.attention import builtin_sdpa_kernel
    from dinkster_inference_torch.multidevice_attention import (
        MultiDeviceAttentionError,
        MultiDeviceAttentionKernel,
    )

    inner = builtin_sdpa_kernel()
    generator = torch.Generator().manual_seed(276)
    q = torch.randn(1, 8, 128, 64, dtype=torch.float32, generator=generator)
    k = torch.randn(1, 8, 128, 64, dtype=torch.float32, generator=generator)
    v = torch.randn(1, 8, 128, 64, dtype=torch.float32, generator=generator)
    expected = inner(q, k, v)
    fail_owner_lane = True

    def fail_once(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        nonlocal fail_owner_lane
        if q.device.index == 0 and fail_owner_lane:
            fail_owner_lane = False
            raise MultiDeviceAttentionError("injected owner-lane failure")
        return inner(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    with MultiDeviceAttentionKernel(
        fail_once,
        (torch.device("cuda:1"), torch.device("cuda:0")),
    ) as kernel:
        with pytest.raises(MultiDeviceAttentionError, match="injected owner-lane failure"):
            kernel(q, k, v)
        actual = kernel(q, k, v)

    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected)


def _run_ulysses_transport_parity(rank: int, rendezvous: str, result_queue: Any) -> None:
    """Two-rank NCCL worker: the peer-copy Ulysses transport must produce
    bitwise-identical attention output to the NCCL all_to_all transport."""
    require_gpu_tests_enabled()
    import torch.distributed as dist
    from dinkster_inference import PlacementMap, SequenceLayout, UspMesh, plan_sequence_partition
    from dinkster_inference_torch.attention import builtin_sdpa_kernel
    from dinkster_inference_torch.sequence_exchange import UlyssesSequenceExchange
    from manifest_token import minted_consensus_token

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        device = torch.device("cuda", rank)
        mesh = UspMesh.build(guidance=1, ulysses=2, ring=1)
        placement = PlacementMap.identity(mesh.process_mesh)
        sequence_length = 64
        partition = plan_sequence_partition(sequence_length, 2)
        shard = partition.shards[rank]
        generator = torch.Generator(device="cpu").manual_seed(20260828)
        full = tuple(
            torch.randn(1, 8, sequence_length, 32, dtype=torch.float32, generator=generator).to(
                device=device, dtype=torch.bfloat16
            )
            for _ in range(3)
        )
        local = tuple(tensor[:, :, shard.start : shard.stop] for tensor in full)
        heads_per_rank = 8 // 2
        layout = SequenceLayout(
            sequence_length,
            shard,
            "fixed-seed-token-order:20260828",
            8,
            rank * heads_per_rank,
            (rank + 1) * heads_per_rank,
            shard.start,
            mesh.digest,
        )
        outputs: dict[str, torch.Tensor] = {}
        for transport in ("nccl", "peer-copy"):
            os.environ["DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT"] = transport
            backend = UlyssesSequenceExchange(
                mesh,
                placement,
                consensus_token=minted_consensus_token(2, shard.index),
            )
            output = backend.attend(local[0], local[1], local[2], layout, builtin_sdpa_kernel())
            # Repeat to exercise slot reuse across consecutive exchanges.
            repeated = backend.attend(local[0], local[1], local[2], layout, builtin_sdpa_kernel())
            assert torch.equal(output, repeated)
            outputs[transport] = output
        identical = torch.equal(outputs["nccl"], outputs["peer-copy"])
        result_queue.put((rank, bool(identical)))
        assert identical
    finally:
        dist.destroy_process_group()


@requires_two_gpus
def test_ulysses_peer_copy_transport_matches_nccl_bitwise() -> None:
    require_gpu_tests_enabled()
    import tempfile

    import torch.multiprocessing as mp
    from torch.multiprocessing.spawn import spawn

    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "transport-parity-rendezvous")
        spawn(
            _run_ulysses_transport_parity,
            args=(rendezvous, queue),
            nprocs=2,
            join=True,
        )
    results = dict([queue.get(), queue.get()])
    assert results == {0: True, 1: True}


def _run_rank_zero_sampling_broadcast(rank: int, rendezvous: str, result_queue: Any) -> None:
    require_gpu_tests_enabled()
    import torch.distributed as dist
    from dinkster_inference_torch import distributed

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        device = torch.device("cuda", rank)
        config = distributed.DistributedSamplingConfig(
            rank, 2, "guidance", f"file://{rendezvous}", "1" * 32
        )
        distributed.ensure_process_group = lambda: config  # type: ignore[assignment]
        template = torch.zeros(2, 3, device=device)

        def action() -> torch.Tensor:
            assert rank == 0
            return torch.full_like(template, 17.0)

        output = distributed.run_rank_zero_sampling(action, template, config)
        passed = bool(torch.equal(output, torch.full_like(template, 17.0)))
        result_queue.put((rank, passed))
        assert passed
    finally:
        dist.destroy_process_group()


@requires_two_gpus
def test_rank_zero_sampling_broadcasts_between_cuda_ranks() -> None:
    require_gpu_tests_enabled()
    import tempfile

    import torch.multiprocessing as mp
    from torch.multiprocessing.spawn import spawn

    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "rank-zero-sampling-rendezvous")
        spawn(
            _run_rank_zero_sampling_broadcast,
            args=(rendezvous, queue),
            nprocs=2,
            join=True,
        )
    results = dict([queue.get(), queue.get()])
    assert results == {0: True, 1: True}


def make_fp8_cuda(
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float8_e4m3fn,
    orig: torch.dtype = torch.float16,
) -> Fp8ScaledWeight:
    source = torch.randn(8, 6, generator=torch.Generator().manual_seed(77)) * 0.5
    return quantize_fp8_scaled(source.to(orig).to(device), dtype)


def diff_entry(
    shape: tuple[int, ...],
    fill: float,
    device: str,
    strength: float = 1.0,
) -> PatchEntry[torch.Tensor]:
    return PatchEntry(
        DiffPatch(torch.full(shape, fill, device=device)),
        strength=strength,
    )


@requires_kitchen_cuda
def test_h3_norm_rope_dispatches_kitchen_cuda_backend_in_place() -> None:
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
    from dinkster_inference_torch.minimax_h3_dit import (
        _fused_h3_norm_rope,  # pyright: ignore[reportPrivateUsage]
    )
    from dinkster_kitchen import registry  # pyright: ignore[reportMissingTypeStubs]

    # Input is SEEDED: the tail bit-exactness pinned below is not universal.
    # Draws landing within half a bf16 ulp of a rounding boundary make the
    # fused kernel's normalized tail differ from torch rms_norm by one ulp
    # (observed 0.001953125 on one element of an unseeded draw).
    generator = torch.Generator(device="cuda:0").manual_seed(1234)
    query = torch.randn(1, 64, 2, 128, device="cuda:0", dtype=torch.bfloat16, generator=generator)
    key = torch.randn(1, 64, 2, 128, device="cuda:0", dtype=torch.bfloat16, generator=generator)
    angles = torch.randn(1, 64, 1, 48, device="cuda:0", dtype=torch.bfloat16, generator=generator)
    cosine, sine = angles.cos(), angles.sin()
    table = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(1, 64, 1, 48, 2, 2)
    query_norm = INITLESS.rms_norm(128, eps=1e-5).to(device="cuda:0", dtype=torch.bfloat16)
    key_norm = INITLESS.rms_norm(128, eps=1e-5).to(device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        query_norm.weight.fill_(1.0)
        key_norm.weight.fill_(1.0)
    query_snapshot = query.clone()
    key_snapshot = key.clone()
    expected_query = torch.nn.functional.rms_norm(query_snapshot, (128,), query_norm.weight, 1e-5)
    expected_key = torch.nn.functional.rms_norm(key_snapshot, (128,), key_norm.weight, 1e-5)
    for expected in (expected_query, expected_key):
        prefix = expected[..., :96]
        pairs = prefix.reshape(*prefix.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        rotated = table[..., 0] * pairs[..., 0] + table[..., 1] * pairs[..., 1]
        prefix.copy_(rotated.movedim(-1, -2).reshape_as(prefix))
    query_ptr, key_ptr = query.data_ptr(), key.data_ptr()
    impl = registry.get_implementation(
        "rms_rope_split_half_",
        kwargs={
            "q": query,
            "k": key,
            "freqs_cis": table,
            "q_scale": query_norm.weight,
            "k_scale": key_norm.weight,
            "epsilon": 1e-5,
            "rot_dim": 96,
        },
    )
    assert impl.__module__ == "dinkster_kitchen.backends.cuda"
    assert dinkster_kitchen.rms_rope_split_half_ is not None

    with torch.no_grad():
        actual_query, actual_key = _fused_h3_norm_rope(
            query,
            key,
            table,
            query_norm,
            key_norm,
            1e-5,
            96,
        )
    assert actual_query.data_ptr() == query_ptr
    assert actual_key.data_ptr() == key_ptr
    # Ten seeded BF16 CUDA probes observed at most 0.00245 RMS and 0.03125
    # max prefix drift from separate RMSNorm and RoPE kernels. The tail is
    # only normalized and is bit-exact for this seeded input (see the
    # generator comment above for why that claim needs the seed).
    for actual, expected in (
        (actual_query, expected_query),
        (actual_key, expected_key),
    ):
        delta = (actual - expected).detach().float()
        assert float(delta.square().mean().sqrt()) <= 0.003
        assert float(delta.abs().max()) <= 0.04
        assert torch.equal(actual[..., 96:], expected[..., 96:])


# ------------------------------------------------------ cast_weight


def test_cast_weight_moves_plain_tensor_to_cuda() -> None:
    """CPU-stored weight, CUDA compute: the move + cast pipeline lands
    on the device at the compute dtype, and the stored tensor is
    untouched."""
    stored = torch.randn(4, 4)
    snapshot = stored.clone()
    out = cast_weight(stored, dtype=torch.float16, device=torch.device("cuda:0"))
    assert out.device.type == "cuda" and out.device.index == 0
    assert out.dtype == torch.float16
    assert torch.equal(out.cpu(), stored.to(torch.float16))
    assert torch.equal(stored, snapshot)


def test_cast_weight_functions_owned_buffer_on_cuda() -> None:
    """The copy=True-when-functions rule holds across a device move:
    an in-place-mutating weight function never reaches back into the
    CUDA-stored source."""
    stored = torch.randn(4, 4, device="cuda:0")
    snapshot = stored.clone()

    def mutate(w: torch.Tensor) -> torch.Tensor:
        return w.add_(1.0)

    out = cast_weight(stored, dtype=torch.float32, functions=[mutate])
    assert torch.equal(stored, snapshot)
    assert torch.equal(out, snapshot + 1.0)


def test_fp8_cast_dequantizes_on_cuda() -> None:
    """fp8 storage moves at STORAGE dtype (the small transfer) and
    dequantizes on the device, matching the CPU dequantize product
    bit for bit."""
    source = torch.randn(8, 6, generator=torch.Generator().manual_seed(7))
    stored_cpu = quantize_fp8_scaled(source.clone(), torch.float8_e4m3fn)
    out = cast_weight(stored_cpu, dtype=torch.float16, device=torch.device("cuda:0"))
    assert out.device.type == "cuda"
    assert torch.equal(out.cpu(), stored_cpu.dequantize(torch.float16))


@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=["e4m3fn", "e5m2"])
def test_fp8_seeded_quantize_deterministic_on_cuda(
    dtype: torch.dtype,
) -> None:
    """The seed>0 stochastic writeback path (whatever kernel serves
    it on this build) is a pure function of (tensor, seed) on CUDA:
    same seed replays bit-exact, a different seed draws differently."""
    source = torch.randn(16, 12, generator=torch.Generator().manual_seed(3), device="cpu").to(
        "cuda:0"
    )
    a = quantize_fp8_scaled(source.clone(), dtype, seed=9)
    b = quantize_fp8_scaled(source.clone(), dtype, seed=9)
    c = quantize_fp8_scaled(source.clone(), dtype, seed=10)
    assert torch.equal(a.qdata.view(torch.uint8), b.qdata.view(torch.uint8))
    assert not torch.equal(a.qdata.view(torch.uint8), c.qdata.view(torch.uint8))


def test_gguf_block_decoders_match_cpu_on_cuda() -> None:
    """Every vectorized block decoder is elementwise IEEE float32
    arithmetic, so CUDA must reproduce the CPU decode bit for bit."""
    from dinkster_inference import Q4_0, Q4_K, Q5_K, Q6_K, Q8_0

    for ggml_type in (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0):
        decoder = GGUF_BLOCK_DECODERS[ggml_type.name]
        blocks = reference_quant_blocks(ggml_type, 18, seed=ggml_type.code)
        shape = (18, ggml_type.block_elements)
        decoded_cpu = decoder(blocks, shape)
        decoded_gpu = decoder(blocks.to("cuda:0"), shape)
        assert decoded_gpu.device.type == "cuda"
        assert torch.equal(decoded_gpu.cpu().view(torch.int32), decoded_cpu.view(torch.int32)), (
            ggml_type.name
        )


def test_gguf_q8_linear_decodes_and_forwards_on_cuda() -> None:
    """Q8_0 block decode is elementwise IEEE arithmetic (int8 quants
    times float16 scale in float32), so CUDA must reproduce the CPU
    decode bit for bit, and the encoded-resident forward must equal
    F.linear over the on-device decoded weight."""
    torch.manual_seed(11)
    blocks = encode_q8_0(torch.randn(8, 64))
    bias = torch.randn(8)

    decoded_cpu = decode_q8_0_blocks(blocks, (8, 64))
    decoded_gpu = decode_q8_0_blocks(blocks.to("cuda:0"), (8, 64))
    assert decoded_gpu.device.type == "cuda"
    assert torch.equal(decoded_gpu.cpu(), decoded_cpu)

    module = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})
    module.to("cuda:0")
    assert module.weight_blocks.dtype == torch.uint8
    assert module.weight_blocks.device.type == "cuda"

    x = torch.randn(3, 64, device="cuda:0")
    expected = torch.nn.functional.linear(x, decoded_gpu, bias.to("cuda:0"))
    output = module(x)
    assert output.device.type == "cuda"
    assert torch.equal(output, expected)


def test_gguf_encoded_linear_streams_offloaded_blocks_to_cuda() -> None:
    """An enrolled encoded linear leases its uint8 blocks to the load
    device and decodes there, so the offloaded forward must equal the
    fully loaded one bit for bit."""
    torch.manual_seed(13)
    blocks = encode_q8_0(torch.randn(8, 64))
    bias = torch.randn(8)
    module = GgufEncodedLinear(64, 8, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})

    decoded_gpu = decode_q8_0_blocks(blocks.to("cuda:0"), (8, 64))
    x = torch.randn(3, 64, device="cuda:0")
    expected = torch.nn.functional.linear(x, decoded_gpu, bias.to("cuda:0"))

    mechanism = enroll_component(module, load_device="cuda:0", offload_device="cpu")
    assert mechanism.loaded_unit_names() == frozenset()
    assert module.weight_blocks.device.type == "cpu"
    offloaded = module(x)
    assert offloaded.device.type == "cuda"
    assert torch.equal(offloaded, expected)

    mechanism.partially_load(None)
    assert module.weight_blocks.device.type == "cuda"
    assert torch.equal(module(x), expected)

    mechanism.unload()
    assert module.weight_blocks.device.type == "cpu"
    assert torch.equal(module(x), expected)


def test_timing_receipts_report_cuda_leased_phases() -> None:
    """Receipts for an offloaded CUDA forward attribute producer-stream
    copy time to transfer, the consumer stream's wait to exposed stall,
    and the on-device decode and matmul to dequant and compute, without
    changing the output."""
    torch.manual_seed(19)
    blocks = encode_q8_0(torch.randn(64, 1024))
    bias = torch.randn(64)
    module = GgufEncodedLinear(1024, 64, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})

    decoded_gpu = decode_q8_0_blocks(blocks.to("cuda:0"), (64, 1024))
    x = torch.randn(3, 1024, device="cuda:0")
    expected = torch.nn.functional.linear(x, decoded_gpu, bias.to("cuda:0"))

    torch.cuda.synchronize("cuda:0")
    enroll_component(module, load_device="cuda:0", offload_device="cpu")
    with collect_partial_residency_timing() as timing:
        offloaded = module(x)
    report = timing.report()

    assert torch.equal(offloaded, expected)
    assert report.leased_forwards == 1
    assert report.leased_transfers == 2
    assert report.transfer_bytes == blocks.nbytes + bias.nbytes
    assert report.transfer_ms > 0.0
    assert report.dequant_ms > 0.0
    assert report.compute_ms > 0.0
    # The consumer wait cannot exceed the producer copies by more than
    # event-timing noise; small slack absorbs the ~microsecond
    # resolution of CUDA event pairs.
    assert 0.0 <= report.exposed_stall_ms <= report.transfer_ms + 1.0


def test_prefetch_queue_streams_offloaded_gguf_weights_to_cuda() -> None:
    """A CUDA forward under the block prefetch queue consumes staged
    producer-stream copies: output bit-identical to the eager decode,
    both moves attributed to prefetch, no lease-started transfer, and
    the consumer's exposed wait bounded by the producer copy time."""
    torch.manual_seed(23)
    blocks = encode_q8_0(torch.randn(64, 1024))
    bias = torch.randn(64)
    module = GgufEncodedLinear(1024, 64, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks, "bias": bias})

    decoded_gpu = decode_q8_0_blocks(blocks.to("cuda:0"), (64, 1024))
    x = torch.randn(3, 1024, device="cuda:0")
    expected = torch.nn.functional.linear(x, decoded_gpu, bias.to("cuda:0"))

    torch.cuda.synchronize("cuda:0")
    mechanism = enroll_component(module, load_device="cuda:0", offload_device="cpu")
    assert mechanism.loaded_unit_names() == frozenset()
    with collect_partial_residency_timing() as timing:
        queue = make_prefetch_queue([module])
        assert queue is not None
        prefetch_queue_pop(queue, module)
        offloaded = module(x)
        prefetch_queue_pop(queue, None)
    report = timing.report()

    assert torch.equal(offloaded, expected)
    assert report.leased_forwards == 1
    assert report.prefetched_transfers == 2
    assert report.prefetch_bytes == blocks.nbytes + bias.nbytes
    assert report.transfer_bytes == report.prefetch_bytes
    assert report.leased_transfers == 0
    assert report.transfer_ms > 0.0
    assert report.dequant_ms > 0.0
    assert report.compute_ms > 0.0
    assert 0.0 <= report.exposed_stall_ms <= report.transfer_ms + 1.0

    # Uncollected, the queued forward is the same bits.
    queue = make_prefetch_queue([module])
    assert queue is not None
    prefetch_queue_pop(queue, module)
    assert torch.equal(module(x), expected)
    prefetch_queue_pop(queue, None)


# -------------------------------------------- fused GGUF matmul route


# --------------------------------------------- patches on the device


def test_patch_weights_fp8_roundtrip_on_cuda() -> None:
    """The full fp8 store cycle on the device: dequantize -> patch ->
    seeded requantize, deterministic per key-seed, and restore puts
    the exact original storage back."""
    stored = make_fp8_cuda()
    weights = {"model.w": stored}
    diff = torch.full((8, 6), 0.125, device="cuda:0")
    patch_set = PatchSet({"model.w": (PatchEntry(DiffPatch(diff)),)})

    backup = patch_weights(weights, patch_set)
    patched = weights["model.w"]
    assert isinstance(patched, Fp8ScaledWeight)
    assert patched.qdata.device.type == "cuda"
    assert patched.orig_dtype == stored.orig_dtype
    # exact replay of the pipeline's own steps on the device:
    # dequantize to the fp32 intermediate, add the diff, requantize
    # with the key-derived seed - bit-identical to what patch_weights
    # wrote back (both draws come from the same seeded device stream).
    expected = requantize_fp8_scaled(
        stored,
        stored.dequantize(torch.float32) + 0.125,
        seed=string_to_seed("model.w"),
    )
    assert torch.equal(
        patched.qdata.view(torch.uint8),
        expected.qdata.view(torch.uint8),
    )
    assert torch.equal(patched.scale, expected.scale)
    # deterministic writeback: replay from a fresh copy of the store
    weights2 = {"model.w": stored}
    patch_weights(
        weights2,
        PatchSet({"model.w": (PatchEntry(DiffPatch(diff)),)}),
    )
    patched2 = weights2["model.w"]
    assert isinstance(patched2, Fp8ScaledWeight)
    assert torch.equal(
        patched.qdata.view(torch.uint8),
        patched2.qdata.view(torch.uint8),
    )

    restore_weights(weights, backup)
    assert weights["model.w"] is stored


def test_patch_weights_backup_device_offloads_and_restores() -> None:
    """backup_device moves each backup off the patching device as it is
    taken (the patch_weight_to_device @ b78cec87 backup offload, so the
    device never holds original plus patched for every key) and
    restore_weights moves it back to the device of the weight it
    replaces, bit-identical, with Parameter registration preserved."""
    device = torch.device("cuda:0")
    plain = torch.nn.Parameter(
        torch.randn(8, 6, generator=torch.Generator().manual_seed(11)).to(device),
        requires_grad=False,
    )
    packed = make_fp8_cuda()
    weights: dict[str, StoredWeight] = {"model.a": plain, "model.q": packed}
    plain_snapshot = plain.detach().clone()
    packed_qdata_snapshot = packed.qdata.view(torch.uint8).clone()
    packed_scale_snapshot = packed.scale.clone()
    diff = torch.full((8, 6), 0.125, device=device)
    patch_set = PatchSet(
        {
            "model.a": (PatchEntry(DiffPatch(diff)),),
            "model.q": (PatchEntry(DiffPatch(diff)),),
        }
    )

    backup = patch_weights(weights, patch_set, backup_device=torch.device("cpu"))

    backed_plain = backup["model.a"]
    assert isinstance(backed_plain, torch.nn.Parameter)
    assert backed_plain.device.type == "cpu"
    backed_packed = backup["model.q"]
    assert isinstance(backed_packed, Fp8ScaledWeight)
    assert backed_packed.qdata.device.type == "cpu"
    assert backed_packed.scale.device.type == "cpu"
    patched_plain = weights["model.a"]
    assert isinstance(patched_plain, torch.Tensor)
    assert patched_plain.device == device
    assert torch.equal(patched_plain, plain_snapshot + 0.125)

    restore_weights(weights, backup)

    restored_plain = weights["model.a"]
    assert isinstance(restored_plain, torch.nn.Parameter)
    assert restored_plain.device == device
    assert torch.equal(restored_plain.detach(), plain_snapshot)
    restored_packed = weights["model.q"]
    assert isinstance(restored_packed, Fp8ScaledWeight)
    assert restored_packed.qdata.device.type == "cuda"
    assert restored_packed.scale.device.type == "cuda"
    assert torch.equal(restored_packed.qdata.view(torch.uint8), packed_qdata_snapshot)
    assert torch.equal(restored_packed.scale, packed_scale_snapshot)


def test_patch_weights_backup_device_preserves_tied_identity() -> None:
    """A patched key whose stored tensor is aliased by another store key
    keeps the original itself as backup (no offload - the alias keeps it
    alive anyway) so restore reassigns the identical object and the tie
    survives the round trip."""
    device = torch.device("cuda:0")
    tied = torch.randn(8, 6, generator=torch.Generator().manual_seed(17)).to(device)
    lone = torch.randn(8, 6, generator=torch.Generator().manual_seed(19)).to(device)
    weights = {"model.a": tied, "model.b": tied, "model.c": lone}
    tied_snapshot = tied.clone()
    diff = torch.full((8, 6), 0.125, device=device)
    patch_set = PatchSet(
        {
            "model.a": (PatchEntry(DiffPatch(diff)),),
            "model.c": (PatchEntry(DiffPatch(diff)),),
        }
    )

    backup = patch_weights(weights, patch_set, backup_device=torch.device("cpu"))

    assert backup["model.a"] is tied
    assert backup["model.a"].device == device
    assert weights["model.b"] is tied
    backed_lone = backup["model.c"]
    assert backed_lone.device.type == "cpu"

    restore_weights(weights, backup)

    assert weights["model.a"] is tied
    assert weights["model.a"] is weights["model.b"]
    assert torch.equal(weights["model.a"], tied_snapshot)
    restored_lone = weights["model.c"]
    assert restored_lone.device == device
    assert torch.equal(restored_lone, lone)


def test_patch_weights_backup_device_preserves_packed_tied_identity() -> None:
    """Aliasing between packed store keys lives in the module-owned
    component tensors, not the per-access wrapper objects
    (ModuleStateStore builds a fresh wrapper in ``__getitem__``). A
    patched packed key whose qdata Parameter is tied to another key's
    must keep its on-device backup so restore reassigns the identical
    Parameter and the tie survives."""
    device = torch.device("cuda:0")
    first = Fp8Linear(6, 8, bias=False, compute_dtype=torch.float16)
    second = Fp8Linear(6, 8, bias=False, compute_dtype=torch.float16)
    source = torch.randn(8, 6, generator=torch.Generator().manual_seed(23)) * 0.5
    tied_param = torch.nn.Parameter(
        source.to(torch.float16).to(device).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    for layer in (first, second):
        layer.weight = tied_param
        layer.weight_scale = torch.tensor(0.625, device=device)
        layer.input_scale = torch.tensor(1.0, device=device)
    module = torch.nn.Module()
    module.first = first
    module.second = second
    weights = ModuleStateStore(module)
    qdata_snapshot = tied_param.detach().view(torch.uint8).clone()
    diff = torch.full((8, 6), 0.125, device=device)
    patch_set = PatchSet({"first.weight": (PatchEntry(DiffPatch(diff)),)})

    backup = patch_weights(weights, patch_set, backup_device=torch.device("cpu"))

    backed = backup["first.weight"]
    assert isinstance(backed, Fp8ScaledWeight)
    assert backed.qdata is tied_param
    assert backed.qdata.device == device
    assert second.weight is tied_param
    assert first.weight is not tied_param

    restore_weights(weights, backup)

    assert first.weight is tied_param
    assert first.weight is second.weight
    assert torch.equal(tied_param.detach().view(torch.uint8), qdata_snapshot)


def test_patch_weights_backup_device_rollback_restores_on_device() -> None:
    """A mid-set failure with backup_device set rolls already-patched
    keys back onto the store's device, values bit-identical."""
    from dinkster_inference_torch import PatchApplyError

    device = torch.device("cuda:0")
    original = torch.randn(8, 6, generator=torch.Generator().manual_seed(13)).to(device)
    weights = {"model.a": original.clone()}
    diff = torch.full((8, 6), 0.125, device=device)
    patch_set = PatchSet(
        {
            "model.a": (PatchEntry(DiffPatch(diff)),),
            "model.missing": (PatchEntry(DiffPatch(diff)),),
        }
    )

    with pytest.raises(PatchApplyError):
        patch_weights(weights, patch_set, backup_device=torch.device("cpu"))

    rolled_back = weights["model.a"]
    assert rolled_back.device == device
    assert torch.equal(rolled_back, original)


# --------------------------------------------- stochastic rounding


@requires_kitchen_cuda
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=["e4m3fn", "e5m2"])
def test_rounding_dispatches_kitchen_cuda_backend(device: str, dtype: torch.dtype) -> None:
    """The registry must pick the compiled CUDA backend for CUDA
    tensors (not silently fall back to eager), and the kernel's output
    must be a valid adjacent-grid rounding, seed-deterministic through
    Dinkster's wrapper - on every GPU in the machine."""
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
    from dinkster_kitchen import (  # pyright: ignore[reportMissingTypeStubs]
        registry,
    )

    value = torch.randn(64, 48, device=device) * 2.0
    rng = torch.randint(0, 256, value.shape, dtype=torch.uint8, device=device)
    impl = registry.get_implementation(
        "stochastic_rounding_fp8",
        kwargs={"x": value, "rng": rng, "output_type": dtype},
    )
    assert impl.__module__ == "dinkster_kitchen.backends.cuda"
    assert dinkster_kitchen.stochastic_rounding_fp8 is not None

    # Imported here: test_patches loads a platform golden at import, and an
    # unminted tuple must skip only the tests that need it.
    from test_patches import assert_on_adjacent_grid

    out = stochastic_rounding(value, dtype, seed=5)
    assert out.dtype == dtype
    assert out.device == value.device
    assert_on_adjacent_grid(out.cpu(), value.cpu(), dtype)
    again = stochastic_rounding(value, dtype, seed=5)
    assert torch.equal(out.view(torch.uint8), again.view(torch.uint8))
    other = stochastic_rounding(value, dtype, seed=6)
    assert not torch.equal(out.view(torch.uint8), other.view(torch.uint8))


@requires_kitchen_cuda
@requires_two_gpus
def test_kitchen_cuda_stochastic_rounding_wrong_device_canary() -> None:
    """Pin the 0.2.31 DLPack current-device defect until upstream fixes it."""
    from dinkster_kitchen.backends.cuda import (  # pyright: ignore[reportMissingTypeStubs]
        stochastic_rounding_fp8 as kitchen_round,
    )

    with torch.cuda.device(0):
        value = torch.randn(64, 48, device="cuda:1")
        rng = torch.randint(0, 256, value.shape, dtype=torch.uint8, device="cuda:1")
        with pytest.raises(BufferError, match="different CUDA device index"):
            kitchen_round(value, rng, output_type=torch.float8_e4m3fn)


@requires_kitchen_cuda
@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=["e4m3fn", "e5m2"])
def test_kitchen_cuda_kernel_matches_eager_bitwise(
    dtype: torch.dtype,
) -> None:
    """Given the SAME (x, rng), the compiled CUDA kernel and the eager
    reference produce identical bits - so the CPU-generated kitchen
    goldens pin the CUDA kernel's semantics too. rng is cloned per
    call because the CUDA kernel mutates it in place (upstream quirk,
    see module docstring). Input is SEEDED: upstream's two backends
    genuinely disagree for values just below a power of two (the
    eager fp16-log2 defect, docs/comfyui-issues/
    stochastic-rounding-fp16-log2-boundary.md, pinned by the canary
    below), so an unseeded draw flakes whenever it lands within half
    an fp16 ulp below a boundary."""
    import dinkster_kitchen as ck  # pyright: ignore[reportMissingTypeStubs]

    generator = torch.Generator(device="cuda:0").manual_seed(1234)
    x = torch.randn(128, 96, device="cuda:0", generator=generator) * 3.0
    rng = torch.randint(
        0,
        256,
        x.shape,
        dtype=torch.uint8,
        device="cuda:0",
        generator=generator,
    )
    out_cuda = ck.stochastic_rounding_fp8(x, rng.clone(), output_type=dtype)
    ck.disable_backend("cuda")
    try:
        out_eager = ck.stochastic_rounding_fp8(x, rng.clone(), output_type=dtype)
    finally:
        ck.enable_backend("cuda")
    assert torch.equal(out_cuda.view(torch.uint8), out_eager.view(torch.uint8))


@requires_kitchen_cuda
def test_kitchen_cuda_rng_mutation_canary() -> None:
    """Pins the upstream quirk: kitchen's CUDA kernel mutates its rng
    argument in place (the eager backend does not). Dinkster is immune -
    rounding.py allocates rng fresh per call - but if this test ever
    FAILS, upstream fixed the kernel: update the status line in
    docs/comfyui-issues/comfy-kitchen-cuda-stochastic-rounding-mutates-rng.md
    and drop the defensive clones in test_kitchen_cuda_kernel_matches_
    eager_bitwise."""
    import dinkster_kitchen as ck  # pyright: ignore[reportMissingTypeStubs]

    x = torch.randn(64, 64, device="cuda:0")
    rng = torch.randint(0, 256, x.shape, dtype=torch.uint8, device="cuda:0")
    rng_snapshot = rng.clone()
    ck.stochastic_rounding_fp8(x, rng, output_type=torch.float8_e4m3fn)
    assert not torch.equal(rng, rng_snapshot)


@requires_kitchen_cuda
def test_kitchen_eager_log2_boundary_divergence_canary() -> None:
    """Pins the upstream defect (docs/comfyui-issues/
    stochastic-rounding-fp16-log2-boundary.md): for inputs just below
    a power of two with rng byte 0, kitchen's EAGER backend computes
    the fp8 exponent from fp16 log2 (which rounds up to the integer)
    and lands one fp8 ulp below the correct lower neighbor -
    off-grid, and bitwise different from the CUDA kernel, which gets
    these inputs right (truncation to the lower neighbor). If this
    test ever FAILS, upstream fixed the eager path: update the issue
    file's status line and fold these inputs back into the seeded
    bitwise test above."""
    import dinkster_kitchen as ck  # pyright: ignore[reportMissingTypeStubs]

    x = torch.tensor([7.99614334, -0.249910235, 0.0624428205], device="cuda:0")
    rng = torch.zeros(x.shape, dtype=torch.uint8, device="cuda:0")
    out_cuda = ck.stochastic_rounding_fp8(x, rng.clone(), output_type=torch.float8_e4m3fn)
    ck.disable_backend("cuda")
    try:
        out_eager = ck.stochastic_rounding_fp8(x, rng.clone(), output_type=torch.float8_e4m3fn)
    finally:
        ck.enable_backend("cuda")
    # CUDA truncates onto the grid (correct for rng byte 0)...
    torch.testing.assert_close(out_cuda.float().cpu(), torch.tensor([7.5, -0.234375, 0.05859375]))
    # ...eager lands one ulp below the lower neighbor (the defect)
    torch.testing.assert_close(out_eager.float().cpu(), torch.tensor([7.0, -0.21875, 0.0546875]))


@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=["e4m3fn", "e5m2"])
def test_rounding_manual_fallback_on_cuda(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manual torch path (kitchen kernel forced away) works on
    CUDA tensors: adjacent-grid results, seed-deterministic on the
    device generator."""
    from test_patches import assert_on_adjacent_grid

    monkeypatch.setattr(rounding_mod, "_ck_stochastic_rounding_fp8", None)
    value = torch.randn(32, 24, device="cuda:0") * 2.0
    out = stochastic_rounding(value, dtype, seed=11)
    assert out.dtype == dtype
    assert out.device == value.device
    assert_on_adjacent_grid(out.cpu(), value.cpu(), dtype)
    again = stochastic_rounding(value, dtype, seed=11)
    assert torch.equal(out.view(torch.uint8), again.view(torch.uint8))


# ------------------------------------------- torch.compile / triton


@requires_triton
def test_compiled_consumer_inductor_cuda() -> None:
    """The declared compile boundary on real hardware: cast_weight
    (patching, storage mutation) runs eager, and the compute consuming
    its output compiles through inductor (fullgraph - any graph break
    raises) with triton-generated pointwise kernels on CUDA. Two
    invocations prove compiled execution, not just tracing."""

    def consumer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.nn.functional.linear(x, w) * 2.0 + 1.0)

    compiled = torch.compile(consumer, fullgraph=True)

    stored = make_fp8_cuda(orig=torch.float32)
    entries = (diff_entry((8, 6), 0.125, "cuda:0", strength=0.5),)
    weight = cast_weight(
        stored,
        dtype=torch.float32,
        functions=[DeferredPatch("k", entries)],
    )
    for seed in (1, 2):
        x = torch.randn(
            3,
            6,
            device="cuda:0",
            generator=torch.Generator("cuda:0").manual_seed(seed),
        )
        torch.testing.assert_close(compiled(x, weight), consumer(x, weight))


@requires_triton
def test_compiled_consumer_inductor_cuda_from_worker_thread() -> None:
    """Triton compilation from a non-main thread (Dinkster's engine and
    workers are async + threaded, unlike the reference's executor;
    triton has a history of threading bugs - user flag, ROADMAP
    'torch.compile / triton compatibility gate')."""

    def consumer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ w.T + 0.5)

    weight = cast_weight(make_fp8_cuda(orig=torch.float32), dtype=torch.float32)
    x = torch.randn(4, 6, device="cuda:0")
    failures: list[BaseException] = []
    results: list[torch.Tensor] = []

    def run() -> None:
        try:
            compiled = torch.compile(consumer, fullgraph=True)
            results.append(compiled(x, weight))
            results.append(compiled(x * 2.0, weight))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=300)
    assert not thread.is_alive(), "compile thread hung"
    if failures:
        raise failures[0]
    torch.testing.assert_close(results[0], consumer(x, weight))
    torch.testing.assert_close(results[1], consumer(x * 2.0, weight))


# ------------------------------------------------------- multi-GPU


@requires_two_gpus
def test_cast_weight_cross_device_move() -> None:
    """cuda:0-stored weights (plain and fp8-scaled) cast for compute
    on cuda:1: the result lands on the second GPU with identical
    values, the source stays put."""
    stored = torch.randn(4, 4, device="cuda:0")
    out = cast_weight(stored, dtype=torch.float16, device=torch.device("cuda:1"))
    assert out.device.index == 1
    assert stored.device.index == 0
    assert torch.equal(out.cpu(), stored.to(torch.float16).cpu())

    fp8 = make_fp8_cuda(device="cuda:0")
    out8 = cast_weight(fp8, dtype=torch.float16, device=torch.device("cuda:1"))
    assert out8.device.index == 1
    assert fp8.qdata.device.index == 0
    assert torch.equal(out8.cpu(), fp8.dequantize(torch.float16).cpu())


# ------------------------------------------- residency (stage 4c s2)


def test_cuda_residency_producer_marks_stager_active() -> None:
    from dinkster_inference_torch import residency as residency_mod

    device = torch.device("cuda", 0)
    stager = residency_mod._cuda_transfer_stager(device)  # pyright: ignore[reportPrivateUsage]
    hooks = residency_mod._CudaTransferHooks(device)  # pyright: ignore[reportPrivateUsage]
    assert not stager.pin_active
    with hooks.producer_context():
        assert stager.pin_active
    assert not stager.pin_active


def test_cuda_residency_shedding_skips_a_busy_foreign_stager() -> None:
    from dinkster_inference_torch import residency as residency_mod

    stager = residency_mod._CudaTransferStager(  # pyright: ignore[reportPrivateUsage]
        torch.device("cuda", 0), slot_bytes=(4 * 1024**2,)
    )
    acquired = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    results: list[tuple[int, int]] = []

    def hold_stager() -> None:
        with stager.lock:
            acquired.set()
            release.wait()

    def shed_stager() -> None:
        results.append(
            (
                stager.free_pins(stager.capacity_bytes),
                stager.free_registrations(stager.capacity_bytes),
            )
        )
        completed.set()

    holder = threading.Thread(target=hold_stager)
    holder.start()
    assert acquired.wait(timeout=1)
    shedder = threading.Thread(target=shed_stager)
    shedder.start()
    try:
        assert completed.wait(timeout=1)
        assert results == [(0, 0)]
    finally:
        release.set()
        holder.join(timeout=1)
        shedder.join(timeout=1)
    assert not holder.is_alive()
    assert not shedder.is_alive()


def test_cuda_residency_staging_reuses_a_bounded_pinned_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import residency as residency_mod
    from dinkster_inference_torch.tensor_ops import transfer_to_device, use_transfer_stager

    # Pin host memory readings: under real host memory pressure the pin budget
    # evicts inactive stagers mid-test, breaking the counter deltas below.
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (1 << 60, 1 << 60))
    device = torch.device("cuda", 0)
    initial_pinned = pinned_host.TOTAL_PINNED_MEMORY
    initial_storage = pinned_host.TOTAL_PINNED_STORAGE
    assert sum(residency_mod._CUDA_STAGING_SLOT_BYTES) == 128 * 1024**2  # pyright: ignore[reportPrivateUsage]
    stager = residency_mod._CudaTransferStager(  # pyright: ignore[reportPrivateUsage]
        device, slot_bytes=(16 * 1024**2, 16 * 1024**2)
    )
    producer = torch.cuda.Stream(device=device)
    sources = [torch.full((4 * 1024**2,), index, dtype=torch.float32) for index in range(12)]

    with stager.lock, torch.cuda.stream(producer), use_transfer_stager(stager):
        torch.cuda._sleep(10_000_000)  # pyright: ignore[reportPrivateUsage]
        outputs = [transfer_to_device(source, device, non_blocking=True) for source in sources]
        strided = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
        converted = transfer_to_device(strided, device, dtype=torch.float16, non_blocking=True)
    torch.cuda.current_stream(device).wait_stream(producer)

    assert stager.allocated_bytes == stager.capacity_bytes
    assert pinned_host.TOTAL_PINNED_MEMORY - initial_pinned == stager.capacity_bytes
    assert pinned_host.TOTAL_PINNED_STORAGE - initial_storage == stager.capacity_bytes
    assert all(
        torch.equal(output.cpu(), source) for output, source in zip(outputs, sources, strict=True)
    )
    assert converted.dtype == torch.float16
    assert torch.equal(converted.cpu(), strided.to(torch.float16))
    assert stager.free_pins(stager.capacity_bytes) == stager.capacity_bytes
    assert stager.allocated_bytes == 0
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned
    assert pinned_host.TOTAL_PINNED_STORAGE == initial_storage
    assert stager not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]


def test_cuda_residency_staging_can_unregister_and_reregister_a_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import residency as residency_mod
    from dinkster_inference_torch.tensor_ops import transfer_to_device, use_transfer_stager

    # Pin host memory readings: under real host memory pressure the pin budget
    # evicts inactive stagers mid-test, breaking the counter deltas below.
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (1 << 60, 1 << 60))
    device = torch.device("cuda", 0)
    initial_pinned = pinned_host.TOTAL_PINNED_MEMORY
    initial_storage = pinned_host.TOTAL_PINNED_STORAGE
    stager = residency_mod._CudaTransferStager(  # pyright: ignore[reportPrivateUsage]
        device, slot_bytes=(4 * 1024**2,)
    )
    source = torch.arange(512 * 1024, dtype=torch.float32)

    with use_transfer_stager(stager):
        first = transfer_to_device(source, device, non_blocking=True)
    torch.cuda.synchronize(device)
    assert pinned_host.TOTAL_PINNED_MEMORY - initial_pinned == stager.capacity_bytes
    assert pinned_host.TOTAL_PINNED_STORAGE - initial_storage == stager.capacity_bytes

    assert stager.free_registrations(stager.capacity_bytes) == stager.capacity_bytes
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned
    assert pinned_host.TOTAL_PINNED_STORAGE - initial_storage == stager.capacity_bytes

    with use_transfer_stager(stager):
        second = transfer_to_device(source, device, non_blocking=True)
    torch.cuda.synchronize(device)
    assert torch.equal(first.cpu(), source)
    assert torch.equal(second.cpu(), source)
    assert pinned_host.TOTAL_PINNED_MEMORY - initial_pinned == stager.capacity_bytes

    assert stager.free_pins(stager.capacity_bytes) == stager.capacity_bytes
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned
    assert pinned_host.TOTAL_PINNED_STORAGE == initial_storage


def test_cuda_residency_staging_can_shed_and_rematerialize_a_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import residency as residency_mod
    from dinkster_inference_torch.tensor_ops import transfer_to_device, use_transfer_stager

    # Pin host memory readings: under real host memory pressure the pin budget
    # evicts inactive stagers mid-test, breaking the counter deltas below.
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (1 << 60, 1 << 60))
    device = torch.device("cuda", 0)
    initial_pinned = pinned_host.TOTAL_PINNED_MEMORY
    initial_storage = pinned_host.TOTAL_PINNED_STORAGE
    stager = residency_mod._CudaTransferStager(  # pyright: ignore[reportPrivateUsage]
        device, slot_bytes=(4 * 1024**2,)
    )
    source = torch.arange(512 * 1024, dtype=torch.float32)

    with use_transfer_stager(stager):
        first = transfer_to_device(source, device, non_blocking=True)
    assert stager.free_pins(stager.capacity_bytes) == stager.capacity_bytes
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned
    assert pinned_host.TOTAL_PINNED_STORAGE == initial_storage

    with use_transfer_stager(stager):
        second = transfer_to_device(source, device, non_blocking=True)
    torch.cuda.synchronize(device)
    assert torch.equal(first.cpu(), source)
    assert torch.equal(second.cpu(), source)
    assert pinned_host.TOTAL_PINNED_MEMORY - initial_pinned == stager.capacity_bytes
    assert pinned_host.TOTAL_PINNED_STORAGE - initial_storage == stager.capacity_bytes

    assert stager.free_pins(stager.capacity_bytes) == stager.capacity_bytes
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned
    assert pinned_host.TOTAL_PINNED_STORAGE == initial_storage


@pytest.mark.parametrize("refusal", ("disabled", "storage", "headroom", "register"))
def test_cuda_residency_staging_refusal_uses_pageable_fallback_without_accounting(
    monkeypatch: pytest.MonkeyPatch,
    refusal: str,
) -> None:
    from dinkster_inference_torch import residency as residency_mod
    from dinkster_inference_torch.tensor_ops import transfer_to_device, use_transfer_stager

    # Pin host memory readings: under real host memory pressure the pin budget
    # evicts inactive stagers mid-test, breaking the counter deltas below.
    monkeypatch.setattr(pinned_host, "memory_status", lambda: (1 << 60, 1 << 60))
    device = torch.device("cuda", 0)
    initial_pinned = pinned_host.TOTAL_PINNED_MEMORY
    initial_storage = pinned_host.TOTAL_PINNED_STORAGE
    if refusal == "disabled":
        monkeypatch.setattr(pinned_host, "DISABLED", True)
    elif refusal == "storage":

        def refuse_storage(_owner: object, _size: int) -> bool:
            return False

        monkeypatch.setattr(pinned_host, "reserve_storage", refuse_storage)
    elif refusal == "headroom":

        def refuse_headroom(_size: int) -> bool:
            return False

        monkeypatch.setattr(pinned_host, "ensure_pin_budget", refuse_headroom)
    else:

        def fail_register() -> object:
            raise RuntimeError("injected CUDA host registration failure")

        monkeypatch.setattr(torch.cuda, "cudart", fail_register)
    stager = residency_mod._CudaTransferStager(  # pyright: ignore[reportPrivateUsage]
        device, slot_bytes=(4 * 1024**2,)
    )
    source = torch.arange(512 * 1024, dtype=torch.float32)

    with use_transfer_stager(stager):
        output = transfer_to_device(source, device, non_blocking=True)

    assert torch.equal(output.cpu(), source)
    assert stager.allocated_bytes == 0
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned
    assert pinned_host.TOTAL_PINNED_STORAGE == initial_storage
    assert stager not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]


def test_cuda_residency_staging_falls_back_above_its_memory_ceiling() -> None:
    from dinkster_inference_torch import residency as residency_mod
    from dinkster_inference_torch.tensor_ops import transfer_to_device, use_transfer_stager

    device = torch.device("cuda", 0)
    stager = residency_mod._CudaTransferStager(  # pyright: ignore[reportPrivateUsage]
        device, slot_bytes=(1024, 1024)
    )
    source = torch.arange(1024, dtype=torch.float32)

    with stager.lock, use_transfer_stager(stager):
        output = transfer_to_device(source, device, non_blocking=True)

    assert stager.allocated_bytes == 0
    assert torch.equal(output.cpu(), source)


def _residency_store(*sizes: tuple[str, int]) -> dict[str, StoredWeight]:
    gen = torch.Generator().manual_seed(41)
    return {name: torch.randn(n, n, generator=gen) for name, n in sizes}


def _plain(stored: StoredWeight) -> torch.Tensor:
    """Narrow a store value to a plain tensor."""
    assert isinstance(stored, torch.Tensor)
    return stored


def test_cuda_residency_lease_pins_offloaded_sources_in_place() -> None:
    """The first offloaded lease read host-registers the stored CPU
    tensor where it sits: values are unchanged, repeated leases account
    nothing new, and unload returns the registration budget."""
    store = _residency_store(("a", 64))
    source = _plain(store["a"])
    reference = source.clone()
    initial_pinned = pinned_host.TOTAL_PINNED_MEMORY
    resident = ResidentWeights(store, load_device="cuda:0", offload_device="cpu")

    with resident.lease("a") as lease:
        out = lease.get("a", dtype=torch.float16)
    assert source.is_pinned()
    assert torch.equal(source, reference)
    assert torch.equal(out.cpu(), reference.to(torch.float16))
    assert pinned_host.TOTAL_PINNED_MEMORY - initial_pinned == source.nbytes

    with resident.lease("a") as lease:
        lease.get("a", dtype=torch.float16)
    assert pinned_host.TOTAL_PINNED_MEMORY - initial_pinned == source.nbytes

    resident.unload()
    assert not source.is_pinned()
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned
    pins = resident._source_pins  # pyright: ignore[reportPrivateUsage]
    assert pins is not None
    assert pins not in pinned_host._owners  # pyright: ignore[reportPrivateUsage]


def test_cuda_residency_source_pins_release_on_load_and_repin_after_unload() -> None:
    """Loading a unit unpins its CPU source; after the unit is unloaded
    the next lease pins the restored offload tensor lazily."""
    store = _residency_store(("a", 32))
    source = _plain(store["a"])
    initial_pinned = pinned_host.TOTAL_PINNED_MEMORY
    resident = ResidentWeights(store, load_device="cuda:0", offload_device="cpu")

    with resident.lease("a") as lease:
        lease.get("a", dtype=torch.float16)
    assert source.is_pinned()

    resident.partially_load(None)
    assert not source.is_pinned()
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned

    resident.partially_unload(resident.total_bytes())
    restored = _plain(store["a"])
    with resident.lease("a") as lease:
        out = lease.get("a", dtype=torch.float16)
    assert restored.is_pinned()
    assert pinned_host.TOTAL_PINNED_MEMORY - initial_pinned == restored.nbytes
    assert torch.equal(out.cpu(), source.to(torch.float16))

    resident.unload()
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned


def test_cuda_residency_source_pinning_refusal_stays_pageable() -> None:
    """A zero registration ceiling refuses every pin; leases still
    produce correct values from pageable sources with no accounting."""
    store = _residency_store(("a", 32))
    source = _plain(store["a"])
    initial_pinned = pinned_host.TOTAL_PINNED_MEMORY
    resident = ResidentWeights(store, load_device="cuda:0", offload_device="cpu")
    previous_maximum = pinned_host.MAX_PINNED_MEMORY
    previous_storage = pinned_host.MAX_PINNED_STORAGE
    pinned_host.configure(maximum=0)
    try:
        with resident.lease("a") as lease:
            out = lease.get("a", dtype=torch.float16)
    finally:
        pinned_host.configure(maximum=previous_maximum, storage_maximum=previous_storage)
    assert not source.is_pinned()
    assert pinned_host.TOTAL_PINNED_MEMORY == initial_pinned
    assert torch.equal(out.cpu(), source.to(torch.float16))
    resident.unload()


def test_cuda_residency_source_pins_shed_under_pressure_and_repin() -> None:
    """Registration pressure unregisters source pins in place with the
    data intact; the next lease re-pins and stays bit-identical."""
    store = _residency_store(("a", 32))
    source = _plain(store["a"])
    reference = source.clone()
    resident = ResidentWeights(store, load_device="cuda:0", offload_device="cpu")

    with resident.lease("a") as lease:
        lease.get("a", dtype=torch.float16)
    pins = resident._source_pins  # pyright: ignore[reportPrivateUsage]
    assert pins is not None
    assert source.is_pinned()

    assert pins.free_registrations(source.nbytes) == source.nbytes
    assert not source.is_pinned()
    assert torch.equal(source, reference)

    with resident.lease("a") as lease:
        out = lease.get("a", dtype=torch.float16)
    assert source.is_pinned()
    assert torch.equal(out.cpu(), reference.to(torch.float16))
    resident.unload()
    assert not source.is_pinned()


def test_cuda_residency_leases_and_prefetch_mark_source_pins_active() -> None:
    store = _residency_store(("a", 16))
    resident = ResidentWeights(store, load_device="cuda:0", offload_device="cpu")
    pins = resident._source_pins  # pyright: ignore[reportPrivateUsage]
    assert pins is not None
    assert not pins.pin_active
    with resident.lease("a"):
        assert pins.pin_active
    assert not pins.pin_active
    handle = resident.prefetch((("a", torch.float16),))
    assert handle is not None
    assert pins.pin_active
    handle.close()
    assert not pins.pin_active
    handle.close()
    assert not pins.pin_active
    resident.unload()


def test_resident_weights_moves_storage_to_cuda_and_back() -> None:
    """Full load moves storage (patched in place) onto the GPU; unload
    restores the patched key's EXACT original object and moves
    everything back to the offload device bitwise-intact."""
    store = _residency_store(("a", 16), ("b", 16))
    original_a = _plain(store["a"])
    original_b_clone = _plain(store["b"]).clone()
    delta = torch.full((16, 16), 0.25)
    resident = ResidentWeights(
        store,
        load_device="cuda:0",
        offload_device="cpu",
        patch_set=PatchSet({"a": (PatchEntry(DiffPatch(delta)),)}),
    )

    resident.partially_load(None)
    loaded_a = store["a"]
    loaded_b = store["b"]
    assert isinstance(loaded_a, torch.Tensor)
    assert isinstance(loaded_b, torch.Tensor)
    assert loaded_a.device.type == "cuda"
    assert loaded_b.device.type == "cuda"
    assert torch.allclose(loaded_a.cpu(), original_a + delta)
    # resident patched storage needs no cast-time functions
    assert resident.weight_functions("a") == ()
    used = resident.use("a", dtype=torch.float32)
    assert used.device.type == "cuda"
    assert torch.equal(used, loaded_a)

    resident.unload()
    assert store["a"] is original_a  # exact original object restored
    restored_b = store["b"]
    assert isinstance(restored_b, torch.Tensor)
    assert restored_b.device.type == "cpu"
    assert torch.equal(restored_b, original_b_clone)


def test_resident_weights_partial_budget_defers_offloaded_cuda() -> None:
    """Under a partial budget the big unit claims GPU residency and
    the small patched unit stays on CPU - but use() still produces the
    patched weight on the GPU via DeferredPatch, matching the fully
    resident value."""
    store = _residency_store(("big", 32), ("small", 8))
    small_original = _plain(store["small"]).clone()
    delta = torch.full((8, 8), 0.5)
    resident = ResidentWeights(
        store,
        load_device="cuda:0",
        offload_device="cpu",
        patch_set=PatchSet({"small": (PatchEntry(DiffPatch(delta)),)}),
    )

    big_bytes = stored_nbytes(store["big"])
    resident.partially_load(big_bytes + 1)
    assert resident.loaded_unit_names() == frozenset({"big"})
    assert _plain(store["big"]).device.type == "cuda"
    small_stored = _plain(store["small"])
    assert small_stored.device.type == "cpu"
    assert torch.equal(small_stored, small_original)  # pristine

    used = resident.use("small", dtype=torch.float32)
    assert used.device.type == "cuda"
    assert torch.allclose(used.cpu(), small_original + delta)

    resident.unload()
    assert _plain(store["big"]).device.type == "cpu"


def test_failed_unit_load_rolls_back_across_devices() -> None:
    """A patch failure mid-unit rolls already-moved keys back to the
    offload device - the store never leaks CUDA storage for an
    unloaded unit."""
    from dinkster_inference_torch import PatchApplyError, ResidencyUnit

    store = _residency_store(("w", 4), ("b", 4))
    original_w = store["w"]
    resident = ResidentWeights(
        store,
        load_device="cuda:0",
        offload_device="cpu",
        # wrong-shape diff: patching "b" raises after "w" already moved
        patch_set=PatchSet({"b": (PatchEntry(DiffPatch(torch.zeros(2, 2))),)}),
        units=(ResidencyUnit("unit", ("w", "b")),),
    )
    with pytest.raises(PatchApplyError):
        resident.partially_load(None)
    assert resident.loaded_bytes() == 0
    w_restored = _plain(store["w"])
    assert w_restored.device.type == "cpu"
    assert torch.equal(w_restored, _plain(original_w))
    assert _plain(store["b"]).device.type == "cpu"


def test_fp8_residency_roundtrip_cuda() -> None:
    """A patched fp8-scaled weight requantizes into GPU-resident
    storage and unload restores the exact original object."""
    source = torch.randn(8, 8, generator=torch.Generator().manual_seed(5))
    fp8 = quantize_fp8_scaled(source, torch.float8_e4m3fn)
    store: dict[str, StoredWeight] = {"q": fp8}
    delta = torch.full((8, 8), 0.125)
    resident = ResidentWeights(
        store,
        load_device="cuda:0",
        offload_device="cpu",
        patch_set=PatchSet({"q": (PatchEntry(DiffPatch(delta)),)}),
    )

    resident.partially_load(None)
    loaded = store["q"]
    assert isinstance(loaded, Fp8ScaledWeight)
    assert loaded.qdata.device.type == "cuda"
    # exact replay of the load-time pipeline: move to the device, then
    # patch_stored_weight's seeded requantize - bit-identical because
    # both draws come from the same key-seeded device stream.
    moved = move_stored(fp8, torch.device("cuda", 0))
    assert isinstance(moved, Fp8ScaledWeight)
    expected = requantize_fp8_scaled(
        moved,
        moved.dequantize(torch.float32) + 0.125,
        seed=string_to_seed("q"),
    )
    assert torch.equal(
        loaded.qdata.view(torch.uint8),
        expected.qdata.view(torch.uint8),
    )
    assert torch.equal(loaded.scale, expected.scale)
    used = resident.use("q", dtype=torch.float32)
    assert torch.equal(used, expected.dequantize(torch.float32))

    resident.unload()
    assert store["q"] is fp8


def test_residency_manager_loads_and_frees_on_real_cuda_memory() -> None:
    """The manager drives real ResidentWeights mechanisms against real
    CUDA free-memory measurements: tiny models fully load (a 4090 has
    orders of magnitude more free than the reserve), and an impossible
    free() demand detaches them back to CPU."""
    device = torch.device("cuda", 0)
    store_a = _residency_store(("w", 64))
    store_b = _residency_store(("w", 64))
    mech_a = ResidentWeights(store_a, load_device=device, offload_device="cpu")
    mech_b = ResidentWeights(store_b, load_device=device, offload_device="cpu")
    manager = ResidencyManager(policy=MemoryPolicy())

    hooks_a = cast(Any, mech_a._transfer_hooks)  # pyright: ignore[reportPrivateUsage]
    hooks_b = cast(Any, mech_b._transfer_hooks)  # pyright: ignore[reportPrivateUsage]
    assert hooks_a._stager is hooks_b._stager
    manager.load([mech_a])
    manager.load([mech_b])
    assert mech_a.offloaded_bytes() == 0
    assert mech_b.offloaded_bytes() == 0
    assert manager.registered() == (mech_b, mech_a)
    assert _plain(store_a["w"]).device.type == "cuda"

    # demand more than the card can ever free: everything detaches
    manager.free(get_total_memory(device) * 2, device)
    assert manager.registered() == ()
    assert mech_a.loaded_bytes() == 0
    assert _plain(store_a["w"]).device.type == "cpu"
    assert _plain(store_b["w"]).device.type == "cpu"


def test_fully_resident_loading_defers_the_producer_stream() -> None:
    require_gpu_tests_enabled()
    script = textwrap.dedent(
        """
        import torch
        from dinkster_inference_torch import ResidentWeights

        expected = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        store = {"weight": expected.clone()}
        resident = ResidentWeights(store, load_device="cuda:0", offload_device="cpu")
        hooks = resident._transfer_hooks
        assert hooks._stager._stream is None
        resident.partially_load(None)
        assert hooks._stager._stream is None
        resident.partially_unload(resident.total_bytes())
        with resident.lease("weight") as lease:
            assert torch.equal(lease.get("weight", dtype=torch.float32).cpu(), expected)
        assert hooks._stager._stream is not None
        resident.unload()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_memory_introspection_reports_reclaimable_reserve() -> None:
    """get_free_memory @ 947c2749 semantics on real CUDA: freeing a
    tensor moves its bytes into the allocator's reclaimable reserve
    (free_torch), and soft_empty_cache hands them back to the
    driver."""
    require_gpu_tests_enabled()
    script = textwrap.dedent(
        """
        import os
        if os.environ.get("DINKSTER_ENABLE_GPU_TESTS") != "1":
            raise RuntimeError("GPU test worker requires DINKSTER_ENABLE_GPU_TESTS=1")
        import torch
        from dinkster_inference_torch import get_free_memory, get_total_memory, soft_empty_cache

        device = torch.device("cuda", 0)
        assert get_total_memory(device) > 0
        soft_empty_cache(device)
        payload = torch.empty(64 * 1024 * 1024 // 4, device=device)
        nbytes = payload.nbytes
        del payload
        torch.cuda.synchronize(device)
        measured = get_free_memory(device)
        assert measured.free_torch >= nbytes
        assert measured.free_total >= measured.free_torch
        soft_empty_cache(device)
        after = get_free_memory(device)
        assert after.free_torch < measured.free_torch
        print("CUDA_MEMORY_INTROSPECTION_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA_MEMORY_INTROSPECTION_OK" in result.stdout


@requires_two_gpus
def test_residency_manager_isolates_devices() -> None:
    """free() on one device never evicts models resident on the
    other."""
    store_0 = _residency_store(("w", 32))
    store_1 = _residency_store(("w", 32))
    mech_0 = ResidentWeights(store_0, load_device="cuda:0", offload_device="cpu")
    mech_1 = ResidentWeights(store_1, load_device="cuda:1", offload_device="cpu")
    manager = ResidencyManager(policy=MemoryPolicy())
    manager.load([mech_0, mech_1])
    assert mech_0.offloaded_bytes() == 0
    assert mech_1.offloaded_bytes() == 0

    manager.free(
        get_total_memory(torch.device("cuda", 0)) * 2,
        torch.device("cuda", 0),
    )
    assert mech_0.loaded_bytes() == 0
    assert mech_1.offloaded_bytes() == 0
    assert _plain(store_1["w"]).device.index == 1
    assert manager.registered() == (mech_1,)


def _routed_cuda_module() -> torch.nn.Sequential:
    module = torch.nn.Sequential(
        INITLESS.linear(8, 16),
        INITLESS.layer_norm(16),
        INITLESS.linear(16, 4),
    )
    generator = torch.Generator().manual_seed(83)
    module.load_state_dict(
        {
            key: torch.randn(tuple(value.shape), generator=generator)
            for key, value in module.state_dict().items()
        },
        strict=True,
        assign=True,
    )
    return module


def test_module_residency_partial_cuda_forward_is_bitwise() -> None:
    """A mixed CPU/CUDA module runs through routed offloaded units
    without changing its fully resident CUDA forward."""
    module = _routed_cuda_module()
    mechanism = enroll_component(module, load_device="cuda:0", offload_device="cpu")
    input = torch.randn(3, 8, device="cuda:0")
    mechanism.partially_load(None)
    resident = module(input)
    mechanism.partially_unload(1)
    assert 0 < mechanism.loaded_bytes() < mechanism.total_bytes()
    assert torch.equal(module(input), resident)


def test_module_residency_manager_budget_keeps_partial_cuda_placement() -> None:
    """ResidencyManager may settle only one unit under a byte budget;
    the remaining CPU units still execute through cast-at-use."""
    module = _routed_cuda_module()
    mechanism = enroll_component(module, load_device="cuda:0", offload_device="cpu")
    input = torch.randn(3, 8, device="cuda:0")
    mechanism.partially_load(None)
    expected = module(input)
    mechanism.unload()

    # Module 0 is 576 bytes and is the largest unit. Strict-< settling
    # under 577 bytes keeps exactly it resident.
    manager = ResidencyManager(
        policy=MemoryPolicy(
            inference_reserve=0,
            physical_headroom=0,
            min_weight_memory_ratio=0.0,
        ),
        free_memory=lambda _device: DeviceMemory(577, 0),
        empty_cache=lambda _device: None,
    )
    manager.load([mechanism])
    assert 0 < mechanism.loaded_bytes() < mechanism.total_bytes()
    assert torch.equal(module(input), expected)


# -------------------------------------- aimdo demand-paged residency


def _run_aimdo_proof(
    body: str,
    *,
    device_entries: str = "(0,)",
    pre_activation: str = "",
) -> None:
    """Run one aimdo proof in a fresh pre-torch-init child process."""
    require_gpu_tests_enabled()
    bootstrap = """
import os
if os.environ.get("DINKSTER_ENABLE_GPU_TESTS") != "1":
    raise RuntimeError("GPU test worker requires DINKSTER_ENABLE_GPU_TESTS=1")
try:
    from dinkster_aimdo import control
    initialized = control.init()
except Exception:
    raise SystemExit(77)
if not initialized:
    raise SystemExit(77)

import threading
import torch
from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
from dinkster_inference_torch import (
    INITLESS,
    AimdoWeights,
    Fp8ScaledWeight,
    ResidencyUnit,
    ResidentWeights,
    aimdo_resident_bytes,
    aimdo_memory_status,
    dynamic_cuda_memory_snapshot,
    enroll_component,
    ensure_aimdo_devices,
    set_simple_vram_headroom,
)

__PRE_ACTIVATION__
entries = list(__DEVICE_ENTRIES__)
specified = {entry[0] if isinstance(entry, tuple) else entry for entry in entries}
entries.extend(index for index in range(torch.cuda.device_count()) if index not in specified)
if not torch.cuda.is_available() or not ensure_aimdo_devices(entries):
    raise SystemExit(77)

device = torch.device("cuda:0")

def assert_bytes_equal(left, right):
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    left_bytes = left.reshape(-1).view(torch.uint8).cpu()
    right_bytes = right.reshape(-1).view(torch.uint8).cpu()
    assert torch.equal(left_bytes, right_bytes)

def assert_aimdo_resident(mechanism, *keys):
    assert mechanism.loaded_bytes() > 0
    assert all(
        key in mechanism._cache or mechanism.is_loaded(mechanism._unit_of[key])
        for key in keys
    )

def make_fp8():
    return Fp8ScaledWeight(
        torch.tensor([[0.0, 1.0], [2.0, -3.0]]).to(torch.float8_e4m3fn),
        torch.tensor(0.625),
        torch.float32,
    )
"""
    bootstrap = bootstrap.replace("__DEVICE_ENTRIES__", device_entries)
    bootstrap = bootstrap.replace("__PRE_ACTIVATION__", pre_activation)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(bootstrap + "\n" + body)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("dinkster-aimdo native init/init_devices unavailable in child process")
    assert result.returncode == 0, (
        f"aimdo child failed ({result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "storage_dtype,compute_dtype",
    [
        ("bfloat16", "float32"),
        ("float16", "float32"),
        ("bfloat16", "bfloat16"),
        ("float32", "float32"),
    ],
)
def test_aimdo_direct_table_cast_preserves_raw_storage_allocation(
    storage_dtype: str, compute_dtype: str
) -> None:
    _run_aimdo_proof(
        f"""
import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from test_ltx_model import assert_direct_table_cast_preserves_aimdo_allocation
assert_direct_table_cast_preserves_aimdo_allocation(
    torch.{storage_dtype}, torch.{compute_dtype}, "cuda:0"
)
"""
    )


def test_aimdo_direct_first_user_admits_all_visible_devices() -> None:
    _run_aimdo_proof(
        """
from dinkster_inference_torch.aimdo_activation import ensure_visible_aimdo_devices
assert ensure_visible_aimdo_devices()
for index in reversed(range(torch.cuda.device_count())):
    target = torch.device("cuda", index)
    weights = {"weight": torch.arange(8192, dtype=torch.float32)}
    mechanism = AimdoWeights(weights, load_device=target, offload_device="cpu")
    with mechanism.lease("weight") as lease:
        value = lease.get("weight", dtype=torch.float16)
        assert torch.equal(value.cpu(), weights["weight"].to(torch.float16))
    mechanism.unload()
""",
        pre_activation="""
first = AimdoWeights(
    {"weight": torch.ones(1)},
    load_device=f"cuda:{torch.cuda.device_count() - 1}", offload_device="cpu"
)
first.unload()
""",
    )


def test_aimdo_unpatched_plain_and_raw_fp8_match_both_eager_routes() -> None:
    _run_aimdo_proof(
        """
plain = torch.arange(12, dtype=torch.float32).reshape(3, 4)
fp8 = make_fp8()

resident_store = {"plain": plain.clone(), "fp8": make_fp8()}
resident = ResidentWeights(
    resident_store, load_device=device, offload_device="cpu"
)
resident.partially_load(None)
with resident.lease("plain") as lease:
    resident_plain = lease.get("plain", dtype=torch.float32)
    resident_fp8 = lease.get_stored("fp8")

offloaded_store = {"plain": plain.clone(), "fp8": make_fp8()}
offloaded = ResidentWeights(
    offloaded_store, load_device=device, offload_device="cpu"
)
with offloaded.lease("plain") as lease:
    offloaded_plain = lease.get("plain", dtype=torch.float32)
    offloaded_fp8 = lease.get_stored("fp8")

aimdo_store = {"plain": plain.clone(), "fp8": fp8}
aimdo = AimdoWeights(aimdo_store, load_device=device, offload_device="cpu")
with aimdo.lease("plain") as lease:
    aimdo_plain = lease.get("plain", dtype=torch.float32)
    aimdo_fp8 = lease.get_stored("fp8")
assert_aimdo_resident(aimdo, "plain", "fp8")

assert_bytes_equal(aimdo_plain, resident_plain)
assert_bytes_equal(aimdo_plain, offloaded_plain)
assert isinstance(aimdo_fp8, Fp8ScaledWeight)
assert isinstance(resident_fp8, Fp8ScaledWeight)
assert isinstance(offloaded_fp8, Fp8ScaledWeight)
assert_bytes_equal(aimdo_fp8.qdata, resident_fp8.qdata)
assert_bytes_equal(aimdo_fp8.qdata, offloaded_fp8.qdata)
assert_bytes_equal(aimdo_fp8.scale, resident_fp8.scale)
assert_bytes_equal(aimdo_fp8.scale, offloaded_fp8.scale)
"""
    )


def test_aimdo_patched_plain_and_fp8_match_conventional_eager_bytes() -> None:
    _run_aimdo_proof(
        """
patch_set = PatchSet(
    {
        "plain": (PatchEntry(DiffPatch(torch.full((2, 2), 0.125))),),
        "fp8": (PatchEntry(DiffPatch(torch.full((2, 2), -0.25))),),
    }
)
plain = torch.arange(4, dtype=torch.float32).reshape(2, 2)

offloaded = ResidentWeights(
    {"plain": plain.clone(), "fp8": make_fp8()},
    load_device=device,
    offload_device="cpu",
    patch_set=patch_set,
)
offloaded.partially_load(None)
with offloaded.lease("plain") as lease:
    expected_plain = lease.get("plain", dtype=torch.float32)
    expected_fp8 = lease.get("fp8", dtype=torch.float32)

aimdo = AimdoWeights(
    {"plain": plain.clone(), "fp8": make_fp8()},
    load_device=device,
    offload_device="cpu",
    patch_set=patch_set,
)
with aimdo.lease("plain") as lease:
    actual_plain = lease.get("plain", dtype=torch.float32)
    actual_fp8 = lease.get("fp8", dtype=torch.float32)
assert_aimdo_resident(aimdo, "plain", "fp8")

assert_bytes_equal(actual_plain, expected_plain)
# Compare the complete patched-fp8 result as raw bytes, not only numerics.
assert_bytes_equal(actual_fp8, expected_fp8)
"""
    )


def test_aimdo_eviction_then_refault_recomputes_with_fresh_signature() -> None:
    _run_aimdo_proof(
        """
stored = torch.arange(5000, dtype=torch.float32)
aimdo = AimdoWeights(
    {"weight": stored}, load_device=device, offload_device="cpu"
)
with aimdo.lease("weight") as lease:
    first = lease.get("weight", dtype=torch.float32).clone()
assert_aimdo_resident(aimdo, "weight")
first_signature = next(iter(aimdo._cache["weight"]))[0]
aimdo._reap_unpins(wait=True)
freed = aimdo.partially_unload(1)
assert freed > 0
aimdo.partially_load(0)
with aimdo.lease("weight") as lease:
    second = lease.get("weight", dtype=torch.float32).clone()
assert_aimdo_resident(aimdo, "weight")
second_signature = next(iter(aimdo._cache["weight"]))[0]
assert first_signature != second_signature
assert_bytes_equal(first, stored.to(device))
assert_bytes_equal(second, stored.to(device))
"""
    )


def test_aimdo_oom_fault_falls_back_without_caching() -> None:
    _run_aimdo_proof(
        """
stored = torch.arange(5000, dtype=torch.float32)
aimdo = AimdoWeights(
    {"weight": stored}, load_device=device, offload_device="cpu"
)
with aimdo.lease("weight") as lease:
    lease.get("weight", dtype=torch.float32)
assert_aimdo_resident(aimdo, "weight")
aimdo.unload()
assert not aimdo._cache
assert aimdo.loaded_bytes() == 0
aimdo._vbar.set_watermark(0)
with aimdo.lease("weight") as lease:
    actual = lease.get("weight", dtype=torch.float32)
assert "weight" not in aimdo._cache
assert aimdo.loaded_bytes() == 0
assert_bytes_equal(actual, stored.to(device))
"""
    )


def test_aimdo_unpin_after_close_allows_eviction() -> None:
    _run_aimdo_proof(
        """
aimdo = AimdoWeights(
    {"weight": torch.ones(5000)}, load_device=device, offload_device="cpu"
)
with aimdo.lease("weight") as lease:
    lease.get("weight", dtype=torch.float32)
    assert_aimdo_resident(aimdo, "weight")
    assert aimdo.partially_unload(1) == 0
# The page remains pinned until the lease's consumer-stream event completes.
aimdo._reap_unpins(wait=True)
assert aimdo.loaded_bytes() == 0
"""
    )


def test_aimdo_end_to_end_routed_forward_matches_offloaded_eager() -> None:
    _run_aimdo_proof(
        """
def make_layer():
    layer = INITLESS.linear(4, 3)
    layer.load_state_dict(
        {
            "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8,
            "bias": torch.tensor([0.25, -0.5, 1.0]),
        },
        strict=True,
        assign=True,
    )
    return layer

eager_layer = make_layer()
enroll_component(eager_layer, load_device=device, offload_device="cpu")
aimdo_layer = make_layer()
aimdo = enroll_component(
    aimdo_layer,
    load_device=device,
    offload_device="cpu",
    mechanism_factory=AimdoWeights,
)
assert aimdo.partially_load(0) > 0
input = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4) / 4
with torch.inference_mode():
    expected = eager_layer(input)
    actual = aimdo_layer(input)
assert_aimdo_resident(aimdo, "weight", "bias")
assert not aimdo._cache
assert_bytes_equal(actual, expected)
"""
    )


def test_aimdo_gguf_encoded_forward_is_bitwise_and_reports_receipts() -> None:
    _run_aimdo_proof(
        """
from dinkster_inference_torch import collect_partial_residency_timing
from dinkster_inference_torch.gguf_linear import GgufEncodedLinear

torch.manual_seed(11)
qs = torch.randint(-127, 128, (2048, 32), dtype=torch.int8)
scales = torch.full((2048, 1), 0.25, dtype=torch.float16)
blocks = torch.cat((scales.view(torch.uint8), qs.view(torch.uint8)), dim=1)
bias = torch.randn(64)

def make_module():
    module = GgufEncodedLinear(1024, 64, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks.clone(), "bias": bias.clone()})
    return module

eager_module = make_module()
enroll_component(eager_module, load_device=device, offload_device="cpu")
aimdo_module = make_module()
aimdo = enroll_component(
    aimdo_module,
    load_device=device,
    offload_device="cpu",
    mechanism_factory=AimdoWeights,
)
x = torch.randn(3, 1024, device=device)
with torch.inference_mode():
    expected = eager_module(x)
    with collect_partial_residency_timing() as timing:
        actual = aimdo_module(x)
report = timing.report()
assert_aimdo_resident(aimdo, "weight_blocks", "bias")
assert_bytes_equal(actual, expected)
assert report.leased_forwards == 1
assert report.leased_transfers == 2
assert report.transfer_bytes == blocks.nbytes + bias.nbytes
assert report.transfer_ms > 0.0
assert report.dequant_ms > 0.0
assert report.compute_ms > 0.0
assert 0.0 <= report.exposed_stall_ms <= report.transfer_ms + 1.0

# Uncollected, the aimdo forward is the same bits.
with torch.inference_mode():
    assert_bytes_equal(aimdo_module(x), expected)
"""
    )


def test_aimdo_gguf_prefetched_forward_is_bitwise_with_prefetch_receipts() -> None:
    _run_aimdo_proof(
        """
from dinkster_inference_torch import collect_partial_residency_timing
from dinkster_inference_torch.gguf_linear import GgufEncodedLinear

torch.manual_seed(23)
qs = torch.randint(-127, 128, (2048, 32), dtype=torch.int8)
scales = torch.full((2048, 1), 0.25, dtype=torch.float16)
blocks = torch.cat((scales.view(torch.uint8), qs.view(torch.uint8)), dim=1)
bias = torch.randn(64)

def make_module():
    module = GgufEncodedLinear(1024, 64, bias=True, compute_dtype=torch.float32)
    module.load_state_dict({"weight_blocks": blocks.clone(), "bias": bias.clone()})
    return module

eager_module = make_module()
enroll_component(eager_module, load_device=device, offload_device="cpu")
aimdo_module = make_module()
aimdo = enroll_component(
    aimdo_module,
    load_device=device,
    offload_device="cpu",
    mechanism_factory=AimdoWeights,
)
route = aimdo_module.residency_prefetch()
assert route is not None
mechanism, requests = route
assert requests[0] == ("weight_blocks", None)
x = torch.randn(3, 1024, device=device)
with torch.inference_mode():
    expected = eager_module(x)
    with collect_partial_residency_timing() as timing:
        handle = mechanism.prefetch(requests)
        assert handle is not None
        actual = aimdo_module(x)
        handle.close()
report = timing.report()
assert_bytes_equal(actual, expected)
assert report.leased_forwards == 1
assert report.prefetched_transfers == 2
assert report.prefetch_bytes == blocks.nbytes + bias.nbytes
assert report.transfer_bytes == report.prefetch_bytes
assert report.leased_transfers == 0
assert report.transfer_ms > 0.0
assert report.dequant_ms > 0.0
assert report.compute_ms > 0.0
"""
    )


def test_aimdo_fresh_construction_and_fresh_thread_fault_resident() -> None:
    _run_aimdo_proof(
        """
# No torch CUDA operation precedes construction. AimdoWeights must establish
# the thread-current CUDA context before creating or faulting its VBAR.
stored = torch.arange(5000, dtype=torch.float32)
aimdo = AimdoWeights(
    {"weight": stored}, load_device=device, offload_device="cpu"
)
with aimdo.lease("weight") as lease:
    first = lease.get("weight", dtype=torch.float32).clone()
assert_aimdo_resident(aimdo, "weight")
first_signature = next(iter(aimdo._cache["weight"]))[0]
assert_bytes_equal(first, stored.to(device))

assert aimdo.partially_unload(1) > 0
aimdo.partially_load(0)
results = []
errors = []

def fault_on_fresh_thread():
    try:
        with aimdo.lease("weight") as lease:
            results.append(lease.get("weight", dtype=torch.float32).cpu())
    except BaseException as error:
        errors.append(error)

thread = threading.Thread(target=fault_on_fresh_thread)
thread.start()
thread.join()
assert not errors, errors
assert len(results) == 1
assert_aimdo_resident(aimdo, "weight")
second_signature = next(iter(aimdo._cache["weight"]))[0]
assert second_signature != first_signature
assert_bytes_equal(results[0], stored)
"""
    )


def test_aimdo_global_resident_bytes_tracks_fault_and_unload() -> None:
    _run_aimdo_proof(
        """
page_size = 32 * 1024**2
stored = torch.arange(5000, dtype=torch.float32)
aimdo = AimdoWeights(
    {"weight": stored}, load_device=device, offload_device="cpu"
)
before = aimdo_resident_bytes(device)
with aimdo.lease("weight") as lease:
    lease.get("weight", dtype=torch.float32)
resident = aimdo_resident_bytes(device)
assert resident > before
assert resident % page_size == 0
assert resident >= stored.nbytes
aimdo.unload()
after = aimdo_resident_bytes(device)
assert after < resident
assert after == before
"""
    )


def test_aimdo_dynamic_memory_excludes_active_pins_then_normal_path_reaps() -> None:
    _run_aimdo_proof(
        """
for index in range(torch.cuda.device_count()):
    test_device = torch.device("cuda", index)
    aimdo = AimdoWeights(
        {"weight": torch.ones(5000)}, load_device=test_device, offload_device="cpu"
    )
    with aimdo.lease("weight") as lease:
        lease.get("weight", dtype=torch.float32)
        active = aimdo_memory_status(test_device)
        active_snapshot = dynamic_cuda_memory_snapshot(test_device)
        assert active.evictable_bytes == 0
        assert active.pinned_bytes > 0
        assert active_snapshot.dynamic_evictable_bytes == 0
        assert active_snapshot.dynamic_pinned_bytes == active.pinned_bytes
        assert active_snapshot.free_bytes == min(
            active_snapshot.total_bytes,
            active_snapshot.driver_free_bytes + active_snapshot.allocator_reclaimable_bytes,
        )
    torch.cuda.synchronize(test_device)
    deferred = aimdo_memory_status(test_device)
    assert deferred.resident_bytes == active.pinned_bytes
    assert aimdo.partially_unload(0) == 0
    released = aimdo_memory_status(test_device)
    released_snapshot = dynamic_cuda_memory_snapshot(test_device)
    assert released.pinned_bytes == 0
    assert released.evictable_bytes == active.pinned_bytes
    assert released_snapshot.dynamic_evictable_bytes == released.evictable_bytes
    assert released_snapshot.dynamic_pinned_bytes == 0
    assert released_snapshot.free_bytes == min(
        released_snapshot.total_bytes,
        released_snapshot.driver_free_bytes
        + released_snapshot.allocator_reclaimable_bytes
        + released.evictable_bytes,
    )
    aimdo.unload()
""",
        device_entries="tuple(range(torch.cuda.device_count()))",
    )


def test_aimdo_activation_with_per_device_headroom_can_fault() -> None:
    _run_aimdo_proof(
        """
aimdo = AimdoWeights(
    {"weight": torch.ones(5000)}, load_device=device, offload_device="cpu"
)
with aimdo.lease("weight") as lease:
    lease.get("weight", dtype=torch.float32)
assert_aimdo_resident(aimdo, "weight")
""",
        device_entries="((0, 64 * 1024**2),)",
    )


def test_aimdo_simple_headroom_after_activation_preserves_faulting() -> None:
    _run_aimdo_proof(
        """
assert set_simple_vram_headroom(256 * 1024**2)
aimdo = AimdoWeights(
    {"weight": torch.ones(5000)}, load_device=device, offload_device="cpu"
)
with aimdo.lease("weight") as lease:
    lease.get("weight", dtype=torch.float32)
assert_aimdo_resident(aimdo, "weight")
""",
        pre_activation="assert not set_simple_vram_headroom(256 * 1024**2)",
    )


def test_aimdo_tiny_unit_is_conventionally_resident_without_faults() -> None:
    _run_aimdo_proof(
        """
stored = torch.arange(16, dtype=torch.float32).reshape(4, 4)
expected_store = {"weight": stored.clone()}
expected = ResidentWeights(
    expected_store, load_device=device, offload_device="cpu"
)
expected.partially_load(None)

before = aimdo_resident_bytes(device)
aimdo_store = {"weight": stored.clone()}
aimdo = AimdoWeights(aimdo_store, load_device=device, offload_device="cpu")
assert aimdo.partially_load(0) == stored.nbytes
with aimdo.lease("weight") as lease, expected.lease("weight") as expected_lease:
    actual = lease.get("weight", dtype=torch.float32)
    eager = expected_lease.get("weight", dtype=torch.float32)
assert aimdo.is_loaded("weight")
assert not aimdo._cache
assert aimdo_resident_bytes(device) == before
assert_bytes_equal(actual, eager)
"""
    )


def test_aimdo_mixed_tiers_fault_only_large_and_compose_loaded_bytes() -> None:
    _run_aimdo_proof(
        """
tiny = torch.arange(16, dtype=torch.float32).reshape(4, 4)
large = torch.arange(5000, dtype=torch.float32)
aimdo = AimdoWeights(
    {"tiny": tiny, "large": large},
    load_device=device,
    offload_device="cpu",
)
assert aimdo.partially_load(0) == tiny.nbytes
before_fault = aimdo_resident_bytes(device)
with aimdo.lease("tiny") as lease:
    actual_tiny = lease.get("tiny", dtype=torch.float32)
assert aimdo_resident_bytes(device) == before_fault
with aimdo.lease("large") as lease:
    actual_large = lease.get("large", dtype=torch.float32)
assert aimdo_resident_bytes(device) > before_fault
assert aimdo.is_loaded("tiny")
assert "tiny" not in aimdo._cache
assert "large" in aimdo._cache
assert aimdo.loaded_bytes() == tiny.nbytes + large.nbytes
assert_bytes_equal(actual_tiny, tiny.to(device))
assert_bytes_equal(actual_large, large.to(device))
"""
    )


def test_aimdo_streamed_plain_and_fp8_match_synchronous_bytes() -> None:
    _run_aimdo_proof(
        """
plain = torch.arange(5000, dtype=torch.float32)
fp8 = Fp8ScaledWeight(
    torch.arange(17000, dtype=torch.float32).to(torch.float8_e4m3fn),
    torch.tensor(0.625),
    torch.float32,
)
patch_set = PatchSet(
    {
        "plain": (PatchEntry(DiffPatch(torch.full((5000,), 0.125))),),
        "fp8": (PatchEntry(DiffPatch(torch.full((17000,), -0.25))),),
    }
)

for patches in (None, patch_set):
    sync = AimdoWeights(
        {"plain": plain.clone(), "fp8": fp8},
        load_device=device,
        offload_device="cpu",
        patch_set=patches,
        stream_count=0,
    )
    streamed = AimdoWeights(
        {"plain": plain.clone(), "fp8": fp8},
        load_device=device,
        offload_device="cpu",
        patch_set=patches,
        stream_count=2,
    )
    with sync.lease("plain") as sync_lease, streamed.lease("plain") as stream_lease:
        sync_plain = sync_lease.get("plain", dtype=torch.float32)
        stream_plain = stream_lease.get("plain", dtype=torch.float32)
        sync_fp8 = sync_lease.get("fp8", dtype=torch.float32)
        stream_fp8 = stream_lease.get("fp8", dtype=torch.float32)
    assert_bytes_equal(stream_plain, sync_plain)
    assert_bytes_equal(stream_fp8, sync_fp8)
"""
    )


def test_aimdo_prepared_patch_payloads_pin_reuse_and_mixed_accounting() -> None:
    _run_aimdo_proof(
        """
from dinkster_inference.patches import AdapterPatch, NestedPatch
from dinkster_inference_torch import LoRAAdapter, pinned_host

plain = torch.arange(5000, dtype=torch.float32)
fp8 = Fp8ScaledWeight(
    torch.arange(17000, dtype=torch.float32).to(torch.float8_e4m3fn),
    torch.tensor(0.625),
    torch.float32,
)
plain_entries = (
    PatchEntry(DiffPatch(torch.full((5000,), 0.125))),
    PatchEntry(
        AdapterPatch(
            LoRAAdapter(
                torch.full((5000, 1), 0.25),
                torch.full((1, 1), 0.5),
            )
        ),
        strength=0.75,
    ),
    PatchEntry(
        NestedPatch(
            torch.zeros(5000),
            (PatchEntry(DiffPatch(torch.full((5000,), -0.375))),),
        )
    ),
)
patch_set = PatchSet(
    {
        "plain": plain_entries,
        "fp8": (PatchEntry(DiffPatch(torch.full((17000,), -0.25))),),
    }
)
sync = AimdoWeights(
    {"plain": plain.clone(), "fp8": fp8},
    load_device=device,
    offload_device="cpu",
    patch_set=patch_set,
    stream_count=0,
)
streamed = AimdoWeights(
    {"plain": plain.clone(), "fp8": fp8},
    load_device=device,
    offload_device="cpu",
    patch_set=patch_set,
    stream_count=2,
    pin_all_sources=True,
)
with sync.lease("plain") as left, streamed.lease("plain") as right:
    expected_plain = left.get("plain", dtype=torch.float32)
    actual_plain = right.get("plain", dtype=torch.float32)
    expected_fp8 = left.get("fp8", dtype=torch.float32)
    actual_fp8 = right.get("fp8", dtype=torch.float32)
assert_bytes_equal(actual_plain, expected_plain)
assert_bytes_equal(actual_fp8, expected_fp8)
patch_pins = {
    identity: pin for identity, pin in streamed._pins.items() if identity[0] == "patches"
}
weight_pins = {
    identity: pin for identity, pin in streamed._pins.items() if identity[0] == "weights"
}
assert len(patch_pins) == 6 and len(weight_pins) == 2
assert all(
    pin.tensor.is_pinned() and streamed._backend.is_pinned(pin.tensor)
    for pin in patch_pins.values()
)
assert pinned_host.TOTAL_PINNED_MEMORY == sum(pin.tensor.nbytes for pin in streamed._pins.values())
first_patch_ptrs = {identity: pin.tensor.data_ptr() for identity, pin in patch_pins.items()}

assert streamed.partially_unload(1 << 30) > 0
streamed.partially_load(0)
with streamed.lease("plain") as lease:
    again_plain = lease.get("plain", dtype=torch.float32)
    again_fp8 = lease.get("fp8", dtype=torch.float32)
assert_bytes_equal(again_plain, expected_plain)
assert_bytes_equal(again_fp8, expected_fp8)
assert first_patch_ptrs == {
    identity: streamed._pins[identity].tensor.data_ptr() for identity in first_patch_ptrs
}
streamed.unload()
assert pinned_host.TOTAL_PINNED_MEMORY == 0
streamed.partially_load(0)
with streamed.lease("plain") as lease:
    healthy = lease.get("plain", dtype=torch.float32)
assert_bytes_equal(healthy, expected_plain)
"""
    )


def test_aimdo_prefetch_full_forward_parity_consumption_and_abort_cleanup() -> None:
    _run_aimdo_proof(
        """
from dinkster_inference.patches import AdapterPatch, NestedPatch
from dinkster_inference_torch import Fp8Linear, LoRAAdapter, pinned_host
from dinkster_inference_torch.model_prefetch import (
    close_prefetch_queue,
    make_prefetch_queue,
    prefetch_queue_pop,
)

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.plain = INITLESS.linear(128, 128)
        self.fp8 = Fp8Linear(128, 128, compute_dtype=torch.float32)

    def forward(self, x):
        return self.fp8(torch.nn.functional.silu(self.plain(x)))

class Model(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList((Block(), Block()))
        self.fail = fail

    def forward(self, x):
        queue = make_prefetch_queue(self.blocks)
        try:
            for index, block in enumerate(self.blocks):
                prefetch_queue_pop(queue, block)
                x = block(x)
                if self.fail and index == 0:
                    raise RuntimeError("injected block failure")
            prefetch_queue_pop(queue, None)
            return x
        finally:
            close_prefetch_queue(queue)

def build(fail=False):
    model = Model(fail)
    state = {}
    for index in range(2):
        prefix = f"blocks.{index}"
        state[f"{prefix}.plain.weight"] = (
            torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128) % 31
        ) / 64
        state[f"{prefix}.plain.bias"] = torch.arange(128, dtype=torch.float32) / 128
        state[f"{prefix}.fp8.weight"] = (
            torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128) % 17
        ).to(torch.float8_e4m3fn)
        state[f"{prefix}.fp8.weight_scale"] = torch.tensor(0.03125)
        state[f"{prefix}.fp8.input_scale"] = torch.tensor(1.0)
        state[f"{prefix}.fp8.bias"] = torch.arange(128, dtype=torch.float32) / 256
    model.load_state_dict(state, strict=True, assign=True)
    return model

def patches():
    up = torch.full((128, 2), 0.03125)
    down = torch.full((2, 128), -0.0625)
    nested = NestedPatch(
        torch.zeros(128, 128),
        (PatchEntry(DiffPatch(torch.full((128, 128), 0.0078125))),),
    )
    return PatchSet(
        {
            "blocks.0.plain.weight": (
                PatchEntry(AdapterPatch(LoRAAdapter(up, down))),
                PatchEntry(nested),
            ),
            "blocks.1.fp8.weight": (
                PatchEntry(AdapterPatch(LoRAAdapter(up, down))),
                PatchEntry(nested),
            ),
        }
    )

def enroll(model, stream_count, patch_set=None):
    def factory(weights, **kwargs):
        return AimdoWeights(
            weights,
            stream_count=stream_count,
            pin_all_sources=True,
            **kwargs,
        )
    return enroll_component(
        model,
        load_device=device,
        offload_device="cpu",
        patch_set=patch_set,
        mechanism_factory=factory,
    )

x = torch.arange(4 * 128, dtype=torch.float32, device=device).reshape(4, 128) / 512
for patch_set in (None, patches()):
    sync_model = build()
    streamed_model = build()
    sync = enroll(sync_model, 0, patch_set)
    streamed = enroll(streamed_model, 2, patch_set)
    faulted = []
    original_fault = streamed._fault
    def record_fault(lease, key):
        faulted.append(key)
        return original_fault(lease, key)
    streamed._fault = record_fault
    with torch.inference_mode():
        expected = sync_model(x)
        actual = streamed_model(x)
    assert_bytes_equal(actual, expected)
    assert len(faulted) == len(set(faulted))
    assert set(faulted) == {
        key for key in streamed._allocations if not key.endswith("input_scale")
    }
    assert not streamed._prefetched
    assert any(identity[0] == "weights" for identity in streamed._pins)
    if patch_set is not None:
        assert any(identity[0] == "patches" for identity in streamed._pins)
        assert all(source._prepared is None for source in streamed._prepared_sources.values())
    sync.unload()
    streamed.unload()
    assert pinned_host.TOTAL_PINNED_MEMORY == 0

failed_model = build(fail=True)
failed = enroll(failed_model, 2, patches())
try:
    with torch.inference_mode():
        failed_model(x)
except RuntimeError as error:
    assert str(error) == "injected block failure"
else:
    raise AssertionError("injected block failure did not escape")
assert not failed._prefetched
assert all(source._prepared is None for source in failed._prepared_sources.values())
failed._reap_unpins(wait=True)
assert failed.partially_unload(1 << 60) > 0
failed.unload()
assert pinned_host.TOTAL_PINNED_MEMORY == 0
with torch.inference_mode():
    healthy_model = build()
    healthy = enroll(healthy_model, 2)
    healthy_model(x)
healthy.unload()
"""
    )


def test_aimdo_prefetch_native_flux_forward_is_bitwise() -> None:
    _run_aimdo_proof(
        """
from dinkster_inference import FluxConfig
from dinkster_inference_torch import Flux

config = FluxConfig(
    in_channels=16,
    out_channels=16,
    vec_in_dim=64,
    context_in_dim=64,
    hidden_size=128,
    depth=2,
    depth_single_blocks=2,
    num_heads=4,
    axes_dim=(8, 12, 12),
    guidance_embed=False,
)

def build():
    model = Flux(config)
    state = {}
    for index, (name, value) in enumerate(model.state_dict().items()):
        data = torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
        state[name] = ((data + index + 1) % 29 / 128).to(value.dtype)
    model.load_state_dict(state, strict=True, assign=True)
    return model

def enroll(model, stream_count):
    def factory(weights, **kwargs):
        return AimdoWeights(weights, stream_count=stream_count, **kwargs)
    return enroll_component(
        model,
        load_device=device,
        offload_device="cpu",
        mechanism_factory=factory,
    )

sync_model = build()
streamed_model = build()
sync = enroll(sync_model, 0)
streamed = enroll(streamed_model, 2)
x = torch.arange(16 * 4 * 4, dtype=torch.float32, device=device).reshape(1, 16, 4, 4) / 256
timesteps = torch.tensor([0.5], device=device)
context = torch.arange(3 * 64, dtype=torch.float32, device=device).reshape(1, 3, 64) / 192
y = torch.arange(64, dtype=torch.float32, device=device).reshape(1, 64) / 64
with torch.inference_mode():
    expected = sync_model(x, timesteps, context, y)
    actual = streamed_model(x, timesteps, context, y)
assert_bytes_equal(actual, expected)
assert not streamed._prefetched
sync.unload()
streamed.unload()
"""
    )


def test_aimdo_rotated_operations_reuse_arenas_without_corruption() -> None:
    _run_aimdo_proof(
        """
first = torch.arange(5000, dtype=torch.float32)
second = torch.arange(5001, dtype=torch.float32) * 0.5
units = (
    ResidencyUnit("first-unit", ("first",)),
    ResidencyUnit("second-unit", ("second",)),
)
aimdo = AimdoWeights(
    {"first": first, "second": second},
    load_device=device,
    offload_device="cpu",
    units=units,
    stream_count=2,
)
for iteration in range(6):
    key = "first" if iteration % 2 == 0 else "second"
    unit = f"{key}-unit"
    expected = first if key == "first" else second
    with aimdo.lease(unit) as lease:
        actual = lease.get(key, dtype=torch.float16).clone()
    assert_bytes_equal(actual, expected.to(device, dtype=torch.float16))
    assert aimdo.partially_unload(1 << 30) > 0
    aimdo.partially_load(0)
assert len(aimdo._stream_state.arenas) == 2
"""
    )


def test_aimdo_batch_multi_unit_fault_matches_synchronous() -> None:
    _run_aimdo_proof(
        """
first = torch.arange(5000, dtype=torch.float32)
second = torch.arange(6000, dtype=torch.float32) * 0.25
sync = AimdoWeights(
    {"first": first.clone(), "second": second.clone()},
    load_device=device,
    offload_device="cpu",
    stream_count=0,
)
streamed = AimdoWeights(
    {"first": first.clone(), "second": second.clone()},
    load_device=device,
    offload_device="cpu",
    stream_count=2,
)
requests = (("first", torch.float32), ("second", torch.float32))
with sync.lease("first") as sync_lease, streamed.lease("first") as stream_lease:
    expected = sync_lease.get_many(requests)
    actual = stream_lease.get_many(requests)
for left, right in zip(actual, expected):
    assert_bytes_equal(left, right)
"""
    )


def test_aimdo_stream_teardown_allows_fresh_lease() -> None:
    _run_aimdo_proof(
        """
stored = torch.arange(5000, dtype=torch.float32)
aimdo = AimdoWeights(
    {"weight": stored},
    load_device=device,
    offload_device="cpu",
    stream_count=2,
)
with aimdo.lease("weight") as lease:
    first = lease.get("weight", dtype=torch.float16).clone()
assert aimdo._stream_state.arenas
aimdo.unload()
assert not aimdo._stream_state.arenas
assert aimdo.loaded_bytes() == 0
aimdo.partially_load(0)
with aimdo.lease("weight") as lease:
    second = lease.get("weight", dtype=torch.float16).clone()
assert aimdo._stream_state.arenas
assert_bytes_equal(first, stored.to(device, dtype=torch.float16))
assert_bytes_equal(second, stored.to(device, dtype=torch.float16))
"""
    )


def test_aimdo_pinned_source_is_page_locked_and_stream_bitwise() -> None:
    _run_aimdo_proof(
        """
stored = torch.arange(5000, dtype=torch.float32)
sync = AimdoWeights(
    {"weight": stored.clone()}, load_device=device, offload_device="cpu", stream_count=0
)
pinned = AimdoWeights(
    {"weight": stored.clone()}, load_device=device, offload_device="cpu", pin_all_sources=True
)
with sync.lease("weight") as left, pinned.lease("weight") as right:
    expected = left.get("weight", dtype=torch.float32)
    actual = right.get("weight", dtype=torch.float32)
pin = pinned._pins[("weights", "weight")]
assert pinned._backend.is_pinned(pin.tensor) and pin.tensor.is_pinned()
assert_bytes_equal(actual, expected)
"""
    )


def test_aimdo_oom_streaming_pin_created_then_reused() -> None:
    _run_aimdo_proof(
        """
stored = torch.arange(5000, dtype=torch.float32)
sync = AimdoWeights(
    {"weight": stored.clone()}, load_device=device, offload_device="cpu", stream_count=0
)
with sync.lease("weight") as lease:
    expected = lease.get("weight", dtype=torch.float32)
aimdo = AimdoWeights({"weight": stored}, load_device=device, offload_device="cpu")
aimdo._vbar.set_watermark(0)
with aimdo.lease("weight") as lease:
    first = lease.get("weight", dtype=torch.float32)
pin = aimdo._pins[("weights", "weight")]
with aimdo.lease("weight") as lease:
    second = lease.get("weight", dtype=torch.float32)
assert aimdo._pins[("weights", "weight")] is pin and "weight" not in aimdo._cache
assert aimdo._stream_state.started and aimdo._stream_state.streams
assert_bytes_equal(first, expected)
assert_bytes_equal(second, expected)
"""
    )


def test_aimdo_pin_all_sources_first_miss_and_refault_reuse() -> None:
    _run_aimdo_proof(
        """
stored = torch.arange(5000, dtype=torch.float32)
aimdo = AimdoWeights(
    {"weight": stored}, load_device=device, offload_device="cpu", pin_all_sources=True
)
with aimdo.lease("weight") as lease:
    first = lease.get("weight", dtype=torch.float32).clone()
pin = aimdo._pins[("weights", "weight")]
aimdo._reap_unpins(wait=True)
assert aimdo.partially_unload(1) > 0
aimdo.partially_load(0)
with aimdo.lease("weight") as lease:
    second = lease.get("weight", dtype=torch.float32).clone()
assert aimdo._pins[("weights", "weight")] is pin
assert_bytes_equal(first, second)
"""
    )


def test_aimdo_unregistered_pin_reregisters_with_intact_data() -> None:
    _run_aimdo_proof(
        """
stored = torch.arange(5000, dtype=torch.float32)
aimdo = AimdoWeights(
    {"weight": stored}, load_device=device, offload_device="cpu", pin_all_sources=True
)
with aimdo.lease("weight") as lease:
    first = lease.get("weight", dtype=torch.float32).clone()
pin = aimdo._pins[("weights", "weight")]
snapshot = pin.tensor.clone()
assert aimdo.free_registrations(1) == stored.nbytes
assert not pin.registered and not pin.tensor.is_pinned()
assert torch.equal(pin.tensor, snapshot)
assert aimdo.partially_unload(1) > 0
aimdo.partially_load(0)
with aimdo.lease("weight") as lease:
    second = lease.get("weight", dtype=torch.float32).clone()
assert pin.registered and pin.tensor.is_pinned()
assert_bytes_equal(first, second)
"""
    )


def test_aimdo_unload_lifo_clears_pin_accounting_and_stays_healthy() -> None:
    _run_aimdo_proof(
        """
from dinkster_inference_torch import pinned_host
aimdo = AimdoWeights(
    {"first": torch.ones(5000), "second": torch.ones(5001)},
    load_device=device, offload_device="cpu", pin_all_sources=True,
)
with aimdo.lease("first") as lease:
    lease.get("first", dtype=torch.float32)
    lease.get("second", dtype=torch.float32)
hostbuf = aimdo._pin_state["weights"][0]
assert pinned_host.TOTAL_PINNED_MEMORY == 40004
aimdo.unload()
assert hostbuf.size == 0 and pinned_host.TOTAL_PINNED_MEMORY == 0
aimdo.partially_load(0)
with aimdo.lease("first") as lease:
    actual = lease.get("first", dtype=torch.float32)
assert_bytes_equal(actual, torch.ones(5000, device=device))
"""
    )


def test_aimdo_budget_exhaustion_steals_equal_size_pin() -> None:
    _run_aimdo_proof(
        """
from dinkster_inference_torch import pinned_host
aimdo = AimdoWeights(
    {"first": torch.ones(5000), "second": torch.full((5000,), 2.0)},
    load_device=device, offload_device="cpu", pin_all_sources=True,
)
with aimdo.lease("first") as lease:
    lease.get("first", dtype=torch.float32)
victim = aimdo._pins[("weights", "first")]
pinned_host.configure(maximum=0)
with aimdo.lease("second") as lease:
    actual = lease.get("second", dtype=torch.float32)
replacement = aimdo._pins[("weights", "second")]
assert ("weights", "first") not in aimdo._pins
assert replacement.tensor.data_ptr() == victim.tensor.data_ptr()
assert_bytes_equal(actual, torch.full((5000,), 2.0, device=device))
"""
    )


# -------------------------------------------- tiled codec execution


def _golden_case(name: str) -> dict[str, Any]:
    # Imported here: test_tiling loads a platform golden at import, and an
    # unminted tuple must skip only the tests that need it.
    import test_tiling as tiling_goldens

    for case in tiling_goldens.GOLDENS["cases"]:
        if case["name"] == name:
            return case
    raise KeyError(name)


@pytest.mark.parametrize("name", ["up2_2d", "causal_up_video"])
def test_tiled_apply_on_cuda_matches_reference_golden(name: str) -> None:
    """The tiler's arithmetic must not drift across devices: running
    the golden case with the function on cuda:0 (output accumulated
    on the GPU, then compared on CPU) reproduces the executed
    reference within float tolerance."""
    import test_tiling as tiling_goldens
    from dinkster_inference.tiling import plan_tiles

    case = _golden_case(name)
    samples = tiling_goldens.dec(case["samples"])
    plan = plan_tiles(
        tuple(samples.shape[2:]),
        tuple(case["tile"]),
        overlap=tuple(case["overlap"]),
        scale=tiling_goldens.build_scales(case["scales"]),
        downscale=case["downscale"],
    )
    fn = tiling_goldens.FUNCTIONS[case["function"]]

    def cuda_fn(a: torch.Tensor) -> torch.Tensor:
        return fn(a.to("cuda:0"))

    out = tiled_apply(
        samples,
        cuda_fn,
        plan,
        out_channels=case["out_channels"],
        output_device="cuda:0",
    )
    assert out.device.type == "cuda"
    expected = tiling_goldens.dec(case["expected"])
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_tiled_apply_gradients_flow_on_cuda() -> None:
    from dinkster_inference.tiling import LinearScale, plan_tiles

    x = torch.randn(1, 2, 12, 12, device="cuda:0", requires_grad=True)
    weight = torch.randn(2, 2, 1, 1, device="cuda:0", requires_grad=True)
    plan = plan_tiles((12, 12), (8, 8), overlap=4, scale=LinearScale(1))

    def conv(a: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(a, weight)

    out = tiled_apply(x, conv, plan, out_channels=2, output_device="cuda:0")
    out.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert weight.grad is not None


@requires_triton
def test_tiled_apply_compiled_tile_function_cuda() -> None:
    """The per-tile codec function is the compile boundary: an
    inductor-compiled (fullgraph) tile function must run inside the
    tiler's accumulation loop with varying edge-tile shapes."""
    from dinkster_inference.tiling import LinearScale, plan_tiles

    def tile_fn(a: torch.Tensor) -> torch.Tensor:
        up = a.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        return torch.tanh(up) * 2.0

    compiled = torch.compile(tile_fn, fullgraph=True, dynamic=True)
    x = torch.randn(1, 3, 21, 17, device="cuda:0")
    plan = plan_tiles((21, 17), (8, 8), overlap=4, scale=LinearScale(2))
    out = tiled_apply(x, compiled, plan, out_channels=3, output_device="cuda:0")
    expected = tiled_apply(x, tile_fn, plan, out_channels=3, output_device="cuda:0")
    torch.testing.assert_close(out, expected)


@requires_two_gpus
def test_tiled_apply_accumulates_across_devices() -> None:
    """Function on cuda:1, accumulation on cuda:0 - the reference's
    output_device contract (function results are .to()'d onto the
    output device before feathering)."""
    from dinkster_inference.tiling import LinearScale, plan_tiles

    x = torch.randn(1, 1, 20, device="cuda:1")

    def fn(a: torch.Tensor) -> torch.Tensor:
        return a * 3.0

    plan = plan_tiles((20,), (8,), overlap=2, scale=LinearScale(1))
    out = tiled_apply(x, fn, plan, out_channels=1, output_device="cuda:0")
    assert out.device == torch.device("cuda", 0)
    torch.testing.assert_close(out.cpu(), (x * 3.0).cpu())


# ------------------------------------- KL autoencoder (stage 5 s2)
#
# The KL goldens (test_autoencoder_kl) pin the executed reference on
# CPU; these tests prove the same model does not drift on real CUDA
# devices, in reduced precision, under torch.compile, and from worker
# threads. Ampere+ GPUs default conv/matmul to TF32 (a 10-bit
# mantissa), which is precision POLICY, not porting fidelity - golden
# comparisons run under strict fp32 (TF32 off) so the tolerance stays
# tight enough to catch real arithmetic drift.


@contextmanager
def _strict_fp32() -> Generator[None]:
    conv = torch.backends.cudnn.allow_tf32
    matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = conv
        torch.backends.cuda.matmul.allow_tf32 = matmul


def _kl_case(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import test_autoencoder_kl as kl

    case = kl.GOLDENS["cases"][name]
    return kl.dec(case["input"]), kl.dec(case["latent"]), kl.dec(case["decoded"])


def _kl_model(name: str) -> AutoencoderKL:
    import test_autoencoder_kl as kl

    return kl.build_model(name)


_KL_X4_CUDA_DECODE_GOLDEN_ATOL = 3e-5
_KL_X4_CUDA_DECODE_COMPILE_ATOL = 3e-5
# Pinned ComfyUI and Dinkster are bit-identical on SM 12.0 CUDA. Both differ
# from the CPU golden by at most 2.635e-5 from CPU/CUDA kernel accumulation.
# Their compiled outputs are also bit-identical and differ from eager by at
# most 2.444e-5 from Inductor/eager kernel accumulation.


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_kl_encode_decode_on_cuda_matches_golden(device: str) -> None:
    content, latent, decoded = _kl_case("x4")
    model = _kl_model("x4").to(device)
    with _strict_fp32():
        got_latent = model.encode(content.to(device))
        got_decoded = model.decode(latent.to(device))
    assert got_latent.device == torch.device(device)
    torch.testing.assert_close(got_latent.cpu(), latent, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        got_decoded.cpu(), decoded, rtol=1e-4, atol=_KL_X4_CUDA_DECODE_GOLDEN_ATOL
    )


def test_kl_tf32_default_stays_within_policy_envelope() -> None:
    """The flip side of the strict-fp32 pin: under the DEVICE DEFAULT
    precision policy (TF32 on for Ampere+), outputs may drift from
    the fp32 golden only within TF32's ~1e-2 relative envelope -
    catching accidental half-precision or wrong-kernel regressions
    without dictating the policy."""
    content, latent, _ = _kl_case("x4")
    model = _kl_model("x4").to("cuda:0")
    got = model.encode(content.to("cuda:0"))
    torch.testing.assert_close(got.cpu(), latent, rtol=0.05, atol=0.01)


def test_kl_bfloat16_tracks_float32_on_cuda() -> None:
    """bf16 is a declared working dtype (kl_descriptor); the halved
    mantissa costs precision but must track the fp32 result."""
    content, latent, _ = _kl_case("x4")
    model = _kl_model("x4").to("cuda:0", torch.bfloat16)
    got = model.encode(content.to("cuda:0", torch.bfloat16))
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float().cpu(), latent, rtol=0.05, atol=0.05)


def test_kl_float16_forward_is_finite_on_cuda() -> None:
    """fp16 is not a declared KL working dtype (the reference also
    routes SD VAEs to bf16/fp32), but the module must still run
    mechanically without NaN/Inf at these magnitudes."""
    content, _, _ = _kl_case("x4")
    model = _kl_model("x4").to("cuda:0", torch.float16)
    latent = model.encode(content.to("cuda:0", torch.float16))
    decoded = model.decode(latent)
    assert bool(torch.isfinite(latent).all())
    assert bool(torch.isfinite(decoded).all())


def test_kl_gradients_flow_on_cuda() -> None:
    model = _kl_model("x4").to("cuda:0")
    content, _, _ = _kl_case("x4")
    content = content.to("cuda:0").requires_grad_(True)
    loss = model.decode(model.encode(content)).square().mean()
    loss.backward()
    assert content.grad is not None
    assert bool(torch.isfinite(content.grad).all())
    conv_grad = model.encoder.conv_in.weight.grad
    assert conv_grad is not None
    assert bool(torch.isfinite(conv_grad).all())


def test_kl_plugin_tiled_decode_on_cuda() -> None:
    """The full plugin path (content transforms + sweep + average)
    with the model resident on the GPU and accumulation on the GPU;
    single-covering tiles make tiled == direct the exact oracle."""
    import test_autoencoder_kl as kl
    from dinkster_inference_torch import kl_codec_plugin

    model = kl.build_model("x4").to("cuda:0")
    plugin = kl_codec_plugin(model)
    _, latent, _ = _kl_case("x4")
    latent = latent.to("cuda:0")
    tiled = plugin.decode_tiled(latent, tile=(8, 12), overlap=(2, 2), output_device="cuda:0")
    assert tiled.device == torch.device("cuda", 0)
    torch.testing.assert_close(tiled, plugin.decode(latent), rtol=1e-4, atol=1e-5)


@requires_triton
def test_kl_encode_decode_compile_fullgraph_cuda() -> None:
    """Inductor must swallow the whole encode and decode graphs
    (fullgraph=True): the architecture holds no per-forward Python
    policy to graph-break on (operations.py compile discipline).
    Compiled output is pinned against same-device eager (isolating
    compile drift from device precision policy) AND against the
    executed-reference golden under strict fp32."""
    content, latent, decoded = _kl_case("x4")
    model = _kl_model("x4").to("cuda:0")
    encode = torch.compile(model.encode, fullgraph=True)
    decode = torch.compile(model.decode, fullgraph=True)
    with _strict_fp32():
        got_latent = encode(content.to("cuda:0"))
        eager_latent = model.encode(content.to("cuda:0"))
        got_decoded = decode(latent.to("cuda:0"))
        eager_decoded = model.decode(latent.to("cuda:0"))
    torch.testing.assert_close(got_latent, eager_latent, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        got_decoded,
        eager_decoded,
        rtol=1e-4,
        atol=_KL_X4_CUDA_DECODE_COMPILE_ATOL,
    )
    torch.testing.assert_close(got_latent.cpu(), latent, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        got_decoded.cpu(), decoded, rtol=1e-4, atol=_KL_X4_CUDA_DECODE_GOLDEN_ATOL
    )


def test_kl_runs_from_worker_threads_per_device() -> None:
    """One model per CUDA device, each encoded on its own worker
    thread concurrently - the engine executes jobs off the main
    thread, and device context must not leak between threads."""
    content, latent, _ = _kl_case("x4")
    results: dict[str, torch.Tensor] = {}
    errors: list[BaseException] = []

    def run(device: str) -> None:
        try:
            model = _kl_model("x4").to(device)
            results[device] = model.encode(content.to(device)).cpu()
        except BaseException as exc:  # noqa: BLE001 - reraised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(device,)) for device in CUDA_DEVICES]
    with _strict_fp32():  # the flags are process-global
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert not errors, errors
    for device in CUDA_DEVICES:
        torch.testing.assert_close(results[device], latent, rtol=1e-4, atol=1e-5)


# ----------------------------------------- CLIP text model on CUDA
#
# Stage 5 slice 4. Same discipline as the KL block above: golden
# comparisons run under strict fp32, and the model must hold on real
# devices, in reduced precision, under torch.compile, and from worker
# threads.


def _clip_encoder(case: str):  # noqa: ANN202 - test-local helper
    import test_clip_text as ct

    return ct.build_encoder(case)


def _clip_golden(case: str) -> tuple[torch.Tensor, torch.Tensor]:
    import test_clip_text as ct

    spec = ct.GOLDENS["cases"][case]
    return ct.dec(spec["cond"]), ct.dec(spec["pooled"])


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_clip_encode_on_cuda_matches_golden(device: str) -> None:
    import test_clip_text as ct

    cond, pooled = _clip_golden("l_weighted")
    encoder = _clip_encoder("l_weighted")
    encoder.model.to(device)
    with _strict_fp32():
        got = encoder.encode_chunks(ct.case_chunks("l_weighted"))
    assert got.embeddings.device == torch.device(device)
    torch.testing.assert_close(got.embeddings.cpu(), cond, rtol=1e-4, atol=1e-5)
    assert got.pooled is not None
    torch.testing.assert_close(got.pooled.cpu(), pooled, rtol=1e-4, atol=1e-5)


def test_clip_encode_with_offloaded_embedding_is_bitwise() -> None:
    import test_clip_text as ct

    encoder = _clip_encoder("l_weighted")
    mechanism = enroll_component(encoder.model, load_device="cuda:0", offload_device="cpu")
    chunks = ct.case_chunks("l_weighted")
    mechanism.partially_load(None)
    position_embedding = encoder.model.text_model.embeddings.position_embedding
    positions = torch.arange(len(chunks[0]), device="cuda:0")
    assert torch.equal(position_embedding(positions), position_embedding.weight[: len(chunks[0])])
    resident = encoder.encode_chunks(chunks)
    mechanism.unload()
    mechanism.partially_load(0)
    embedding_unit = "text_model.embeddings.token_embedding"
    assert not mechanism.is_loaded(embedding_unit)

    offloaded = encoder.encode_chunks(chunks)
    assert torch.equal(offloaded.embeddings, resident.embeddings)
    assert offloaded.pooled is not None
    assert resident.pooled is not None
    assert torch.equal(offloaded.pooled, resident.pooled)


def test_clip_bfloat16_tracks_float32_on_cuda() -> None:
    """bf16 parameters, float32 conditioning out (the encoder floats
    its outputs like the reference); the halved mantissa costs
    precision but must track the fp32 golden."""
    import test_clip_text as ct

    cond, pooled = _clip_golden("l_weighted")
    encoder = _clip_encoder("l_weighted")
    encoder.model.to("cuda:0", torch.bfloat16)
    got = encoder.encode_chunks(ct.case_chunks("l_weighted"))
    assert got.embeddings.dtype == torch.float32
    torch.testing.assert_close(got.embeddings.cpu(), cond, rtol=0.05, atol=0.05)
    assert got.pooled is not None
    torch.testing.assert_close(got.pooled.cpu(), pooled, rtol=0.05, atol=0.05)


def test_clip_float16_forward_is_finite_on_cuda() -> None:
    """fp16 is how SD1 checkpoints commonly ship their text encoder;
    the forward must run mechanically without NaN/Inf."""
    import test_clip_text as ct

    encoder = _clip_encoder("l_weighted")
    encoder.model.to("cuda:0", torch.float16)
    got = encoder.encode_chunks(ct.case_chunks("l_weighted"))
    assert bool(torch.isfinite(got.embeddings).all())
    assert got.pooled is not None
    assert bool(torch.isfinite(got.pooled).all())


def test_clip_gradients_flow_on_cuda() -> None:
    import test_clip_text as ct

    encoder = _clip_encoder("l_weighted")
    encoder.model.to("cuda:0")
    got = encoder.encode_chunks(ct.case_chunks("l_weighted"))
    got.embeddings.square().mean().backward()
    grad = encoder.model.text_model.embeddings.token_embedding.weight.grad
    assert grad is not None
    assert bool(torch.isfinite(grad).all())


@requires_triton
def test_clip_forward_compile_fullgraph_cuda() -> None:
    """Inductor must swallow the whole text-model forward
    (fullgraph=True): hidden-layer selection and the activation are
    bind-time constants, so there is no per-forward Python policy to
    graph-break on. Compiled output is pinned against same-device
    eager under strict fp32."""
    import test_clip_text as ct

    model = ct.build_model("l_weighted").to("cuda:0")
    compiled = torch.compile(model, fullgraph=True)
    ids = torch.arange(77, device="cuda:0").unsqueeze(0) % 400
    embeds = model.embed_tokens(ids)
    eos = torch.tensor([5], device="cuda:0")
    with _strict_fp32():
        got = compiled(embeds, eos, hidden_layer=-2)
        eager = model(embeds, eos, hidden_layer=-2)
    torch.testing.assert_close(got.last_hidden, eager.last_hidden, rtol=1e-4, atol=1e-5)
    assert got.hidden is not None
    assert eager.hidden is not None
    torch.testing.assert_close(got.hidden, eager.hidden, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got.pooled, eager.pooled, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got.projected, eager.projected, rtol=1e-4, atol=1e-5)


def test_clip_runs_from_worker_threads_per_device() -> None:
    """One model per CUDA device, each encoding on its own worker
    thread concurrently - the engine executes jobs off the main
    thread, and device context must not leak between threads."""
    import test_clip_text as ct

    cond, _ = _clip_golden("l_weighted")
    results: dict[str, torch.Tensor] = {}
    errors: list[BaseException] = []

    def run(device: str) -> None:
        try:
            encoder = _clip_encoder("l_weighted")
            encoder.model.to(device)
            results[device] = encoder.encode_chunks(ct.case_chunks("l_weighted")).embeddings.cpu()
        except BaseException as exc:  # noqa: BLE001 - reraised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(device,)) for device in CUDA_DEVICES]
    with _strict_fp32():  # the flags are process-global
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert not errors, errors
    for device in CUDA_DEVICES:
        torch.testing.assert_close(results[device], cond, rtol=1e-4, atol=1e-5)


# ---------------------------------------- Ovis Qwen3-2B text model


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_ovis_qwen_text_on_cuda_matches_executed_reference(device: str) -> None:
    spec = QWEN_GOLDENS["model"]
    model = build_tiny_qwen().to(device)
    ids = torch.tensor(spec["ids"], dtype=torch.long, device=device)
    mask = torch.tensor(spec["attention_mask"], dtype=torch.long, device=device)
    with torch.no_grad():
        actual = model(ids, mask).cpu()
    torch.testing.assert_close(actual, decode_qwen_golden(spec["output"]), rtol=1e-5, atol=2e-6)


# ---------------------------------------- T5 text model (stage 5 s7)
#
# Stage 5 slice 7. Same discipline as the CLIP block above: golden
# comparisons run under strict fp32, and the model must hold on real
# devices, in reduced precision, under torch.compile, and from worker
# threads. The real-checkpoint test additionally proves the FULL-SIZE
# T5-XXL (9.1 GiB fp16) loads and encodes on one 4090 without OOM.


def _t5_encoder(case: str):  # noqa: ANN202 - test-local helper
    import test_t5_text as tt

    return tt.build_encoder(case)


def _t5_golden(case: str) -> torch.Tensor:
    import test_t5_text as tt

    return tt.dec(tt.GOLDENS["cases"][case]["cond"])


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_t5_encode_on_cuda_matches_golden(device: str) -> None:
    import test_t5_text as tt

    cond = _t5_golden("t5_weighted")
    encoder = _t5_encoder("t5_weighted")
    encoder.model.to(device)
    with _strict_fp32():
        got = encoder.encode_chunks(tt.case_chunks("t5_weighted"))
    assert got.embeddings.device == torch.device(device)
    assert got.pooled is None
    torch.testing.assert_close(got.embeddings.cpu(), cond, rtol=1e-4, atol=1e-5)


def test_t5_encode_with_offloaded_embedding_is_bitwise() -> None:
    import test_t5_text as tt

    encoder = _t5_encoder("t5_weighted")
    mechanism = enroll_component(encoder.model, load_device="cuda:0", offload_device="cpu")
    chunks = tt.case_chunks("t5_weighted")
    mechanism.partially_load(None)
    resident = encoder.encode_chunks(chunks)
    mechanism.unload()
    mechanism.partially_load(0)
    assert not mechanism.is_loaded("shared")

    offloaded = encoder.encode_chunks(chunks)
    assert torch.equal(offloaded.embeddings, resident.embeddings)
    assert offloaded.pooled is None
    assert resident.pooled is None


def test_t5_bfloat16_tracks_float32_on_cuda() -> None:
    """bf16 parameters, float32 conditioning out (the encoder floats
    its outputs like the reference); the halved mantissa costs
    precision but must track the fp32 golden."""
    import test_t5_text as tt

    cond = _t5_golden("t5_weighted")
    encoder = _t5_encoder("t5_weighted")
    encoder.model.to("cuda:0", torch.bfloat16)
    got = encoder.encode_chunks(tt.case_chunks("t5_weighted"))
    assert got.embeddings.dtype == torch.float32
    torch.testing.assert_close(got.embeddings.cpu(), cond, rtol=0.05, atol=0.05)


def test_t5_float16_forward_is_finite_on_cuda() -> None:
    """Pure-fp16 compute runs mechanically on the tiny model (whose
    activations stay in fp16 range). It is NOT the supported policy
    for the full-size model - T5-XXL activations overflow fp16 and
    the encoding collapses; fp16 CHECKPOINTS run as fp16 storage at
    fp32 compute (CastOperations; see the real-checkpoint test)."""
    import test_t5_text as tt

    encoder = _t5_encoder("t5_weighted")
    encoder.model.to("cuda:0", torch.float16)
    got = encoder.encode_chunks(tt.case_chunks("t5_weighted"))
    assert bool(torch.isfinite(got.embeddings).all())


def test_umt5_fp16_collapse_regression_on_cuda() -> None:
    """The documented Wan UMT5 fp16 failure executed on CUDA kernels:
    a tiny UMT5 whose embedding magnitudes exceed sqrt(fp16 max)
    encodes finite and non-degenerate at fp32, and collapses to exact
    zeros at fp16 (T5LayerNorm squares in the input dtype, the RMS
    variance overflows to inf, rsqrt(inf) = 0). Locks the arithmetic
    behind default_text_dtype pinning Wan text encode to float32
    against later dtype or kernel changes, and executes the
    production boundary itself: fp16 storage computed at the policy
    dtype through the assemble_wan21 cast decision."""
    import test_t5_text as tt

    chunks = tt.umt5_overflow_chunks()

    fp32_encoder = tt.T5TextEncoder(tt.build_umt5_overflow_model(torch.float32))
    fp32_encoder.model.to("cuda:0")
    fp32 = fp32_encoder.encode_chunks(chunks)
    assert fp32.embeddings.dtype == torch.float32
    assert fp32.embeddings.shape == (1, 10, 48)
    assert bool(torch.isfinite(fp32.embeddings).all())
    assert bool(fp32.embeddings.abs().max() > 0)

    fp16_encoder = tt.T5TextEncoder(tt.build_umt5_overflow_model(torch.float16))
    fp16_encoder.model.to("cuda:0")
    fp16 = fp16_encoder.encode_chunks(chunks)
    assert fp16.embeddings.dtype == torch.float32
    assert fp16.embeddings.shape == (1, 10, 48)
    assert bool((fp16.embeddings == 0).all())

    policy_encoder = tt.T5TextEncoder(tt.build_umt5_overflow_model_via_wan_policy())
    policy_encoder.model.to("cuda:0")
    policy = policy_encoder.encode_chunks(chunks)
    assert policy.embeddings.dtype == torch.float32
    assert policy.embeddings.shape == (1, 10, 48)
    assert bool(torch.isfinite(policy.embeddings).all())
    assert bool(policy.embeddings.abs().max() > 0)


def test_t5_gradients_flow_on_cuda() -> None:
    import test_t5_text as tt

    encoder = _t5_encoder("t5_weighted")
    encoder.model.to("cuda:0")
    got = encoder.encode_chunks(tt.case_chunks("t5_weighted"))
    got.embeddings.square().mean().backward()
    grads = dict(encoder.model.named_parameters())
    bias_grad = grads["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].grad
    assert bias_grad is not None
    assert bool(torch.isfinite(bias_grad).all())
    shared = grads["shared.weight"].grad
    assert shared is not None
    assert bool(torch.isfinite(shared).all())


@requires_triton
def test_t5_forward_compile_fullgraph_cuda() -> None:
    """Inductor must swallow the whole T5 forward (fullgraph=True):
    activation and bias ownership are bind-time constants, and the
    past_bias threading is plain dataflow, so there is no per-forward
    Python policy to graph-break on. Compiled output is pinned
    against same-device eager under strict fp32."""
    import test_t5_text as tt

    model = tt.build_model("t5_unweighted").to("cuda:0")
    compiled = torch.compile(model, fullgraph=True)
    ids = torch.arange(16, device="cuda:0").unsqueeze(0) % 500
    embeds = model.embed_tokens(ids)
    with _strict_fp32():
        got = compiled(embeds)
        eager = model(embeds)
    torch.testing.assert_close(got, eager, rtol=1e-4, atol=1e-5)


def test_t5_runs_from_worker_threads_per_device() -> None:
    """One model per CUDA device, each encoding on its own worker
    thread concurrently - the engine executes jobs off the main
    thread, and device context must not leak between threads."""
    import test_t5_text as tt

    cond = _t5_golden("t5_weighted")
    results: dict[str, torch.Tensor] = {}
    errors: list[BaseException] = []

    def run(device: str) -> None:
        try:
            encoder = _t5_encoder("t5_weighted")
            encoder.model.to(device)
            results[device] = encoder.encode_chunks(tt.case_chunks("t5_weighted")).embeddings.cpu()
        except BaseException as exc:  # noqa: BLE001 - reraised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(device,)) for device in CUDA_DEVICES]
    with _strict_fp32():  # the flags are process-global
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert not errors, errors
    for device in CUDA_DEVICES:
        torch.testing.assert_close(results[device], cond, rtol=1e-4, atol=1e-5)


_T5_XXL_FP16 = Path("/home/kosin/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors")


@pytest.mark.skipif(
    not _T5_XXL_FP16.exists(),
    reason="real T5-XXL checkpoint not present",
)
def test_t5_real_checkpoint_encodes_on_cuda_without_oom() -> None:
    """The full-size fp16 T5-XXL under the reference execution
    policy: detect from the real header, keep the 9.1 GiB of weights
    at fp16 STORAGE on cuda:0, run at fp32 COMPUTE through
    CastOperations (sd1_clip.SDClipModel @ 947c2749 = manual_cast
    ops + a hardcoded float32 forward), tokenize a real prompt with
    the native SentencePiece tokenizer, and encode. fp16 compute is
    NOT a valid shape for this model - its activations overflow fp16
    and the encoding collapses to zeros - so this test also proves
    the storage/compute decoupling end to end on real weights: shape
    (1, 256, 4096) per the Flux packing profile, finite, and
    non-degenerate."""
    from dinkster_inference import (
        detect_t5_config,
        load_safetensors_header,
        load_t5_spm,
        tokenize_prompt,
    )
    from dinkster_inference_torch import (
        CastOperations,
        T5TextEncoder,
        T5TextModel,
        load_tensors,
    )

    source = load_safetensors_header(_T5_XXL_FP16)
    geometries = {key: entry.geometry for key, entry in source.entries.items()}
    config = detect_t5_config(geometries)
    with torch.device("meta"):
        model = T5TextModel(config, operations=CastOperations(torch.float32))
    wanted = set(model.state_dict())
    loaded = load_tensors(_T5_XXL_FP16, wanted)
    model.load_state_dict(
        {key: value.to("cuda:0") for key, value in loaded.items()},
        strict=True,
        assign=True,
    )

    spans = tokenize_prompt(
        "a photo of a cat sitting on a windowsill",
        encode_word=load_t5_spm().encode,
    ).spans
    # Grad mode is caller policy (the library never disables it);
    # with autograd on, every per-layer fp32 weight cast would be
    # retained for backward - ~18 GiB on top of the fp16 storage.
    # Inference callers run no_grad; training parity is proven on
    # the tiny model (test_t5_gradients_flow_on_cuda).
    with torch.no_grad():
        got = T5TextEncoder(model).encode(spans)
    assert got.embeddings.shape == (1, 256, 4096)
    assert got.pooled is None
    assert bool(torch.isfinite(got.embeddings).all())
    assert float(got.embeddings.std()) > 0.01


# ------------------------------------------- diffusion UNet (stage 5 s5)
#
# Stage 5 slice 5. Same discipline as the KL and CLIP blocks above:
# golden comparisons run under strict fp32, and the model must hold
# on real devices, in reduced precision, under torch.compile, and
# from worker threads.


def _unet_model(case: str):  # noqa: ANN202 - test-local helper
    import test_unet as tu

    return tu.build_model(case)


def _unet_case(
    case: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    import test_unet as tu

    x, timesteps, context, y = tu.case_inputs(case)
    golden = tu.dec(tu.GOLDENS["cases"][case]["output"])
    return x, timesteps, context, y, golden


_UNET_CUDA_GOLDEN_ATOL = {
    "sd1_conv": 3e-5,
    "xl_linear_adm": 3e-5,
}
_UNET_CUDA_COMPILE_ATOL = {
    "sd1_conv": 3e-5,
    "xl_linear_adm": 3e-5,
}
# Pinned ComfyUI and Dinkster are bit-identical on SM 12.0 CUDA. Their maximum
# CPU-golden drift is 2.408e-5 and 2.551e-5 respectively from kernel accumulation.
# Their compiled outputs are also bit-identical and differ from eager by at
# most 2.569e-5 and 2.742e-5 respectively from Inductor/eager kernel accumulation.


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("case", ["sd1_conv", "xl_linear_adm"])
def test_unet_forward_on_cuda_matches_golden(device: str, case: str) -> None:
    x, timesteps, context, y, golden = _unet_case(case)
    model = _unet_model(case).to(device)
    with _strict_fp32():
        got = model(
            x.to(device),
            timesteps.to(device),
            context.to(device),
            y.to(device) if y is not None else None,
        )
    assert got.device == torch.device(device)
    torch.testing.assert_close(got.cpu(), golden, rtol=1e-4, atol=_UNET_CUDA_GOLDEN_ATOL[case])


def test_unet_odd_spatial_on_cuda_matches_golden() -> None:
    """The Upsample output_shape leg on a real device: odd extents
    must round-trip through the strided downsample."""
    x, timesteps, context, y, golden = _unet_case("sd1_odd_spatial")
    assert y is None
    model = _unet_model("sd1_odd_spatial").to("cuda:0")
    with _strict_fp32():
        got = model(x.to("cuda:0"), timesteps.to("cuda:0"), context.to("cuda:0"))
    assert tuple(got.shape) == tuple(golden.shape)
    torch.testing.assert_close(got.cpu(), golden, rtol=1e-4, atol=1e-5)


def test_unet_bfloat16_tracks_float32_on_cuda() -> None:
    """bf16 parameters and activations; the halved mantissa costs
    precision but must track the fp32 golden. The denoiser stacks far
    more residual adds than the KL/CLIP models above, so bf16
    rounding accumulates further - hence the wider atol (measured
    max |diff| ~0.09 on outputs of unit magnitude)."""
    x, timesteps, context, y, golden = _unet_case("xl_linear_adm")
    assert y is not None
    model = _unet_model("xl_linear_adm").to("cuda:0", torch.bfloat16)
    got = model(
        x.to("cuda:0", torch.bfloat16),
        timesteps.to("cuda:0"),
        context.to("cuda:0", torch.bfloat16),
        y.to("cuda:0", torch.bfloat16),
    )
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float().cpu(), golden, rtol=0.05, atol=0.15)


def test_unet_float16_forward_is_finite_on_cuda() -> None:
    """fp16 is how SD1/SDXL checkpoints commonly ship; the forward
    must run mechanically without NaN/Inf at these magnitudes."""
    x, timesteps, context, y, _ = _unet_case("xl_linear_adm")
    assert y is not None
    model = _unet_model("xl_linear_adm").to("cuda:0", torch.float16)
    got = model(
        x.to("cuda:0", torch.float16),
        timesteps.to("cuda:0"),
        context.to("cuda:0", torch.float16),
        y.to("cuda:0", torch.float16),
    )
    assert bool(torch.isfinite(got).all())


def test_unet_gradients_flow_on_cuda() -> None:
    x, timesteps, context, y, _ = _unet_case("xl_linear_adm")
    assert y is not None
    model = _unet_model("xl_linear_adm").to("cuda:0")
    out = model(
        x.to("cuda:0"),
        timesteps.to("cuda:0"),
        context.to("cuda:0"),
        y.to("cuda:0"),
    )
    out.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


@requires_triton
@pytest.mark.parametrize("case", ["sd1_conv", "xl_linear_adm"])
def test_unet_forward_compile_fullgraph_cuda(case: str) -> None:
    """Inductor must swallow the whole denoiser forward
    (fullgraph=True): _Layers dispatch and the ADM gate are resolved
    at trace time, so there is no per-forward Python policy to
    graph-break on. Compiled output is pinned against same-device
    eager AND the executed-reference golden under strict fp32."""
    x, timesteps, context, y, golden = _unet_case(case)
    model = _unet_model(case).to("cuda:0")
    compiled = torch.compile(model, fullgraph=True)
    args = (
        x.to("cuda:0"),
        timesteps.to("cuda:0"),
        context.to("cuda:0"),
        y.to("cuda:0") if y is not None else None,
    )
    with _strict_fp32():
        got = compiled(*args)
        eager = model(*args)
    torch.testing.assert_close(got, eager, rtol=1e-4, atol=_UNET_CUDA_COMPILE_ATOL[case])
    torch.testing.assert_close(got.cpu(), golden, rtol=1e-4, atol=_UNET_CUDA_GOLDEN_ATOL[case])


def test_unet_runs_from_worker_threads_per_device() -> None:
    """One model per CUDA device, each denoising on its own worker
    thread concurrently - the engine executes jobs off the main
    thread, and device context must not leak between threads."""
    x, timesteps, context, y, golden = _unet_case("xl_linear_adm")
    assert y is not None
    results: dict[str, torch.Tensor] = {}
    errors: list[BaseException] = []

    def run(device: str) -> None:
        try:
            model = _unet_model("xl_linear_adm").to(device)
            results[device] = model(
                x.to(device),
                timesteps.to(device),
                context.to(device),
                y.to(device),
            ).cpu()
        except BaseException as exc:  # noqa: BLE001 - reraised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(device,)) for device in CUDA_DEVICES]
    with _strict_fp32():  # the flags are process-global
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert not errors, errors
    for device in CUDA_DEVICES:
        torch.testing.assert_close(
            results[device],
            golden,
            rtol=1e-4,
            atol=_UNET_CUDA_GOLDEN_ATOL["xl_linear_adm"],
        )


# ------------------------------------------------- Flux
#
# Same discipline as the UNet block above, plus the
# Flux-specific seam: apply_rope routes through the Dinkster-owned fused
# rotation on CUDA (bit-identical to the pure-torch reference), with
# dinkster-kitchen's combined operation as the fallback tier, when no
# input needs gradients. Every no-grad forward here proves the owned
# route; the kitchen tier is pinned separately with the owned probe
# disabled. Golden comparisons use atol=1e-4 because CUDA SDPA
# accumulates differently from the CPU reference.


def _flux_model(case: str):  # noqa: ANN202 - test-local helper
    import test_flux as tf

    return tf.build_model(case)


def _flux_case(
    case: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    import test_flux as tf

    x, timesteps, context, y, guidance = tf.case_inputs(case)
    golden = tf.dec(tf.GOLDENS["cases"][case]["output"])
    return x, timesteps, context, y, guidance, golden


def _vector_free_flux_case() -> tuple[
    Flux,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    import test_flux as tf

    x, timesteps, context, _, _ = tf.vector_free_inputs()
    golden = tf.dec(tf.VECTOR_FREE_GOLDEN["case"]["output"])
    return tf.build_vector_free_model(), x, timesteps, context, golden


def _flux_kitchen_rope_available() -> bool:
    from dinkster_inference_torch import flux as flux_module

    return flux_module._kitchen_apply_rope() is not None  # pyright: ignore[reportPrivateUsage]


requires_kitchen_rope = pytest.mark.skipif(
    not _flux_kitchen_rope_available(),
    reason=(
        "dinkster_kitchen apply_rope unavailable - the fused RoPE"
        " path is NOT proven (install the PyPI wheel; see README)"
    ),
)


@requires_kitchen_rope
@requires_triton
@requires_two_gpus
def test_kitchen_triton_rope_wrong_device_canary() -> None:
    """Pin the 0.2.31 Triton current-device defect until upstream fixes it."""
    require_gpu_tests_enabled()
    script = """
import os
if os.environ.get("DINKSTER_ENABLE_GPU_TESTS") != "1":
    raise RuntimeError("GPU test worker requires DINKSTER_ENABLE_GPU_TESTS=1")
import torch
from dinkster_kitchen.backends.triton.rope import apply_rope1

def inputs(device):
    value = torch.randn(2, 3, 17, 64, device=device)
    freqs = torch.randn(1, 1, 17, 32, 2, 2, device=device)
    return value, freqs

torch.cuda.set_device(0)
apply_rope1(*inputs("cuda:0"))
try:
    apply_rope1(*inputs("cuda:1"))
except ValueError as error:
    assert "Pointer argument" in str(error), error
else:
    raise AssertionError("Kitchen Triton accepted tensors from the non-current device")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("case", ["dev_guidance", "schnell_plain"])
def test_flux_forward_on_cuda_matches_golden(device: str, case: str) -> None:
    x, timesteps, context, y, guidance, golden = _flux_case(case)
    model = _flux_model(case).to(device)
    with _strict_fp32(), torch.no_grad():
        got = model(
            x.to(device),
            timesteps.to(device),
            context.to(device),
            y.to(device),
            guidance.to(device) if guidance is not None else None,
        )
    assert got.device == torch.device(device)
    torch.testing.assert_close(got.cpu(), golden, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_vector_free_flux_absent_y_forward_on_cuda_matches_golden(
    device: str,
) -> None:
    model, x, timesteps, context, golden = _vector_free_flux_case()
    model.to(device)
    with _strict_fp32(), torch.no_grad():
        got = model(
            x.to(device),
            timesteps.to(device),
            context.to(device),
        )
    assert got.device == torch.device(device)
    assert model.vector_in is None
    assert model.guidance_in is None
    torch.testing.assert_close(got.cpu(), golden, rtol=1e-4, atol=1e-4)


def test_flux_odd_spatial_on_cuda_matches_golden() -> None:
    """Odd latent extents exercise the ceil-div patch padding and the
    unpatchify crop on a real device."""
    x, timesteps, context, y, guidance, golden = _flux_case("dev_odd_spatial")
    model = _flux_model("dev_odd_spatial").to("cuda:0")
    assert guidance is not None
    with _strict_fp32(), torch.no_grad():
        got = model(
            x.to("cuda:0"),
            timesteps.to("cuda:0"),
            context.to("cuda:0"),
            y.to("cuda:0"),
            guidance.to("cuda:0"),
        )
    assert tuple(got.shape) == tuple(golden.shape)
    torch.testing.assert_close(got.cpu(), golden, rtol=1e-4, atol=1e-4)


def _rope_case(device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from dinkster_inference_torch.flux import rope

    positions = torch.arange(4352, dtype=torch.float32, device=device)
    freqs = rope(positions[None], 128, 10000).unsqueeze(1)
    generator = torch.Generator(device=device).manual_seed(99)
    q = torch.randn(2, 24, 4352, 128, dtype=dtype, device=device, generator=generator)
    k = torch.randn(2, 24, 4352, 128, dtype=dtype, device=device, generator=generator)
    return q, k, freqs


@requires_kitchen_rope
def test_flux_kitchen_rope_tier_matches_direct_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the owned kernel unavailable, Dinkster dispatches the same
    combined operation as pinned ComfyUI."""
    import dinkster_kitchen  # pyright: ignore[reportMissingTypeStubs]
    from dinkster_inference_torch import flux as flux_module
    from dinkster_inference_torch.flux import apply_rope

    monkeypatch.setattr(flux_module, "_dk_rope", None)
    monkeypatch.setattr(flux_module, "_dk_rope_probed", True)
    for device in CUDA_DEVICES:
        q, k, freqs = _rope_case(device, torch.bfloat16)
        with torch.cuda.device(q.device), torch.no_grad():
            kitchen_q, kitchen_k = apply_rope(q, k, freqs)
            direct_q, direct_k = dinkster_kitchen.apply_rope(q, k, freqs)
        assert torch.equal(kitchen_q, direct_q)
        assert torch.equal(kitchen_k, direct_k)


def test_flux_bfloat16_tracks_float32_on_cuda() -> None:
    """bf16 parameters and activations must track the fp32 golden.
    Flux runs 2x19 double + 38 single blocks of accumulating adds;
    the tolerance matches the UNet block above (measured max |diff|
    well under it on outputs of unit magnitude)."""
    x, timesteps, context, y, guidance, golden = _flux_case("dev_guidance")
    assert guidance is not None
    model = _flux_model("dev_guidance").to("cuda:0", torch.bfloat16)
    with torch.no_grad():
        got = model(
            x.to("cuda:0", torch.bfloat16),
            timesteps.to("cuda:0"),
            context.to("cuda:0", torch.bfloat16),
            y.to("cuda:0", torch.bfloat16),
            guidance.to("cuda:0"),
        )
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float().cpu(), golden, rtol=0.05, atol=0.15)


def test_flux_float16_forward_is_finite_on_cuda() -> None:
    """fp16 is how Flux checkpoints commonly ship; the forward must
    run mechanically without NaN/Inf at these magnitudes."""
    x, timesteps, context, y, guidance, _ = _flux_case("schnell_plain")
    assert guidance is None
    model = _flux_model("schnell_plain").to("cuda:0", torch.float16)
    with torch.no_grad():
        got = model(
            x.to("cuda:0", torch.float16),
            timesteps.to("cuda:0"),
            context.to("cuda:0", torch.float16),
            y.to("cuda:0", torch.float16),
        )
    assert bool(torch.isfinite(got).all())


def test_flux_gradients_flow_on_cuda() -> None:
    """Training viability on the device that will train: every
    parameter (including the QKNorm scales behind the RoPE seam)
    receives a finite gradient."""
    x, timesteps, context, y, guidance, _ = _flux_case("dev_guidance")
    assert guidance is not None
    model = _flux_model("dev_guidance").to("cuda:0")
    out = model(
        x.to("cuda:0"),
        timesteps.to("cuda:0"),
        context.to("cuda:0"),
        y.to("cuda:0"),
        guidance.to("cuda:0"),
    )
    out.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


@requires_triton
@pytest.mark.parametrize("case", ["dev_guidance", "schnell_plain"])
def test_flux_forward_compile_fullgraph_cuda(case: str) -> None:
    """Inductor must swallow the whole Flux forward (fullgraph=True)
    on the kitchen-equipped interpreter: apply_rope detects the
    compile and routes to the pure-torch math (the kitchen launcher
    would graph-break), which inductor fuses. Compiled output is
    pinned against same-device eager AND the executed-reference
    golden under strict fp32."""
    x, timesteps, context, y, guidance, golden = _flux_case(case)
    model = _flux_model(case).to("cuda:0")
    compiled = torch.compile(model, fullgraph=True)
    args = (
        x.to("cuda:0"),
        timesteps.to("cuda:0"),
        context.to("cuda:0"),
        y.to("cuda:0"),
        guidance.to("cuda:0") if guidance is not None else None,
    )
    with _strict_fp32(), torch.no_grad():
        got = compiled(*args)
        eager = model(*args)
    torch.testing.assert_close(got, eager, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got.cpu(), golden, rtol=1e-4, atol=1e-4)


def test_flux_runs_from_worker_threads_per_device() -> None:
    """One model per CUDA device, each denoising on its own worker
    thread concurrently - device context (and the kitchen RoPE
    dispatch) must not leak between threads."""
    x, timesteps, context, y, guidance, golden = _flux_case("dev_guidance")
    assert guidance is not None
    results: dict[str, torch.Tensor] = {}
    errors: list[BaseException] = []

    def run(device: str) -> None:
        try:
            model = _flux_model("dev_guidance").to(device)
            with torch.no_grad():
                results[device] = model(
                    x.to(device),
                    timesteps.to(device),
                    context.to(device),
                    y.to(device),
                    guidance.to(device),
                ).cpu()
        except BaseException as exc:  # noqa: BLE001 - reraised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(device,)) for device in CUDA_DEVICES]
    with _strict_fp32():  # the flags are process-global
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert not errors, errors
    for device in CUDA_DEVICES:
        torch.testing.assert_close(results[device], golden, rtol=1e-4, atol=1e-4)


# ---------------------------- fp8 linear + flux assembly (stage 5 s8)
#
# Stage 5 slice 8: Fp8Linear's hardware matmul path (torch._scaled_mm
# needs Ada/Hopper; the dequant path's numerics are CPU-proven in
# test_quant_linear.py) and real installed-checkpoint assembly, both
# plain-fp8 (combined flux1-dev-fp8) and scaled-fp8 (split
# flux1-dev-fp8-new): 266 swapped layers, bounded forwards on both
# GPUs, dequant-vs-matmul agreement on real weights.

_FLUX_MODELS = Path("/home/kosin/ComfyUI/models")
_REAL_SPLIT_SCALED = {
    "diffusion": _FLUX_MODELS / "diffusion_models/flux1-dev-fp8-new.safetensors",
    "clip_l": _FLUX_MODELS / "text_encoders/clip_l.safetensors",
    "t5xxl": _FLUX_MODELS / "text_encoders/t5xxl_fp16.safetensors",
    "vae": _FLUX_MODELS / "vae/ae.safetensors",
}
_REAL_COMBINED_PLAIN = _FLUX_MODELS / "diffusion_models/flux1-dev-fp8.safetensors"

requires_fp8_matmul = pytest.mark.skipif(
    not (cuda_available and supports_fp8_matmul()),
    reason=(
        "hardware fp8 matmul unavailable (needs CUDA compute"
        " capability 8.9+) - the torch._scaled_mm path is NOT proven"
    ),
)


def _run_fp8_matmul_proof(body: str) -> None:
    """Run the backend-neutrality and real-layer proof in a child."""
    require_gpu_tests_enabled()
    environment = dict(os.environ)
    environment["DINKSTER_SCALED_FP8_PROOF"] = str(_REAL_SPLIT_SCALED["diffusion"])
    environment["DINKSTER_PLAIN_FP8_PROOF"] = str(_REAL_COMBINED_PLAIN)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import os
                if os.environ.get("DINKSTER_ENABLE_GPU_TESTS") != "1":
                    raise RuntimeError(
                        "GPU test worker requires DINKSTER_ENABLE_GPU_TESTS=1"
                    )
                """
            )
            + body,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert result.returncode == 0, (
        f"fp8 matmul child failed ({result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "FP8_MATMUL_NATIVE_PROOF_OK" in result.stdout


@pytest.mark.skipif(
    not _REAL_SPLIT_SCALED["diffusion"].exists() or not _REAL_COMBINED_PLAIN.exists(),
    reason="scaled and plain fp8 proof checkpoints are both required",
)
@requires_fp8_matmul
def test_fp8_matmul_native_subprocess_proves_real_scaled_and_plain_layers() -> None:
    """The installed kitchen and eager torch routes are bitwise identical
    per real layer; scaled and plain checkpoints both
    execute their routed and route-off reference forwards in a fresh
    CUDA process."""
    _run_fp8_matmul_proof(
        """
import os
from pathlib import Path

import torch
from dinkster_inference_torch import CastOperations, Fp8Linear, load_tensors
from dinkster_inference_torch.quant_linear import (
    _probe_kitchen_scaled_mm_v2,
    fp8_matmul_forward,
)

assert _probe_kitchen_scaled_mm_v2() is not None

device = torch.device("cuda:0")
scaled_path = Path(os.environ["DINKSTER_SCALED_FP8_PROOF"])
plain_path = Path(os.environ["DINKSTER_PLAIN_FP8_PROOF"])
assert scaled_path.exists() and plain_path.exists()

def quantized_reference(layer, input, input_scale, weight_scale):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scaled = input.float() / input_scale.float()
    qdata = torch.clamp(scaled, -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    output = (qdata.float() * input_scale) @ (
        layer.weight.float() * weight_scale
    ).T
    if layer.bias is not None:
        output = output + layer.bias.float()
    return output.to(torch.bfloat16)

scaled_keys = (
    "double_blocks.0.img_attn.qkv.weight",
    "double_blocks.0.img_attn.qkv.bias",
    "double_blocks.0.img_attn.qkv.weight_scale",
    "double_blocks.0.img_attn.qkv.input_scale",
)
scaled = load_tensors(scaled_path, scaled_keys)
scaled_layer = Fp8Linear(3072, 9216, compute_dtype=torch.bfloat16)
scaled_layer.load_state_dict(
    {
        "weight": scaled[scaled_keys[0]],
        "bias": scaled[scaled_keys[1]],
        "weight_scale": scaled[scaled_keys[2]],
        "input_scale": scaled[scaled_keys[3]],
    },
    assign=True,
)
scaled_layer.to(device)
scaled_input = torch.randn(
    2,
    3072,
    device=device,
    dtype=torch.bfloat16,
    generator=torch.Generator(device).manual_seed(101),
) * 0.1
assert not scaled_layer.fp8_matmul
with torch.no_grad():
    scaled_off = scaled_layer(scaled_input)
    scaled_off_reference = torch.nn.functional.linear(
        scaled_input,
        scaled_layer.weight.to(torch.bfloat16)
        * scaled_layer.weight_scale.to(torch.bfloat16),
        scaled_layer.bias,
    )
assert torch.equal(scaled_off, scaled_off_reference)
scaled_layer.bind_fp8_matmul(True)
assert scaled_layer._fp8_matmul_backend == "kitchen"
with torch.no_grad():
    scaled_kitchen = scaled_layer(scaled_input)
    scaled_layer._fp8_matmul_backend = "torch"
    scaled_eager = scaled_layer(scaled_input)
if not torch.equal(scaled_kitchen, scaled_eager):
    raise RuntimeError(
        "KITCHEN_SCALED_MM_V2_BITWISE_DIVERGENCE: real scaled-fp8 layer"
    )
torch.testing.assert_close(
    scaled_kitchen,
    quantized_reference(
        scaled_layer,
        scaled_input,
        scaled_layer.input_scale,
        scaled_layer.weight_scale,
    ),
    rtol=0.05,
    atol=0.05,
)
assert (scaled_kitchen - scaled_off).abs().amax() <= 0.15 * scaled_off.abs().amax()

plain_keys = (
    "model.diffusion_model.img_in.weight",
    "model.diffusion_model.img_in.bias",
)
plain = load_tensors(plain_path, plain_keys)
plain_layer = CastOperations(torch.bfloat16).linear(64, 3072)
plain_layer.load_state_dict(
    {"weight": plain[plain_keys[0]], "bias": plain[plain_keys[1]]},
    assign=True,
)
plain_layer.to(device)
plain_input = torch.randn(
    2,
    64,
    device=device,
    dtype=torch.bfloat16,
    generator=torch.Generator(device).manual_seed(202),
) * 0.1
assert not plain_layer.fp8_matmul
with torch.no_grad():
    plain_off = plain_layer(plain_input)
    plain_off_reference = torch.nn.functional.linear(
        plain_input,
        plain_layer.weight.to(torch.bfloat16),
        plain_layer.bias.to(torch.bfloat16),
    )
assert torch.equal(plain_off, plain_off_reference)
plain_layer.bind_fp8_matmul(True)
assert plain_layer._fp8_matmul_backend == "kitchen"
with torch.no_grad():
    plain_kitchen = plain_layer(plain_input)
    plain_layer._fp8_matmul_backend = "torch"
    plain_eager = plain_layer(plain_input)
if not torch.equal(plain_kitchen, plain_eager):
    raise RuntimeError(
        "KITCHEN_SCALED_MM_V2_BITWISE_DIVERGENCE: real plain-fp8 layer"
    )
one_a = torch.ones((), device=device, dtype=torch.float32)
one_b = torch.ones((), device=device, dtype=torch.float32)
torch.testing.assert_close(
    plain_kitchen,
    quantized_reference(plain_layer, plain_input, one_a, one_b),
    rtol=0.05,
    atol=0.05,
)
assert (plain_kitchen - plain_off).abs().amax() <= 0.15 * plain_off.abs().amax()

def assert_backend_neutral(label, input, weight, bias, out_dtype):
    input_scale = torch.tensor(0.75, device=device, dtype=torch.float32)
    weight_scale = torch.tensor(1.25, device=device, dtype=torch.float32)
    outputs = {}
    with torch.no_grad():
        for backend in ("kitchen", "torch"):
            outputs[backend] = fp8_matmul_forward(
                input,
                weight,
                input_scale=input_scale,
                weight_scale=weight_scale,
                bias=bias,
                out_dtype=out_dtype,
                backend=backend,
            )
    if not torch.equal(outputs["kitchen"], outputs["torch"]):
        raise RuntimeError(f"KITCHEN_SCALED_MM_V2_BITWISE_DIVERGENCE: {label}")

generator = torch.Generator(device).manual_seed(303)
synthetic_weight = torch.randn(
    32, 32, device=device, dtype=torch.bfloat16, generator=generator
).to(torch.float8_e4m3fn)
assert_backend_neutral(
    "float32 unfused bias",
    torch.randn(5, 32, device=device, dtype=torch.float32, generator=generator),
    synthetic_weight,
    torch.randn(32, device=device, dtype=torch.float32, generator=generator),
    torch.float32,
)
assert_backend_neutral(
    "3d no bias",
    torch.randn(2, 3, 32, device=device, dtype=torch.bfloat16, generator=generator),
    synthetic_weight,
    None,
    torch.bfloat16,
)
print("FP8_MATMUL_NATIVE_PROOF_OK")
"""
    )


def make_fp8_linear_cuda(
    device: str,
    in_features: int = 32,
    out_features: int = 16,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    input_scale: float = 1.0,
) -> Fp8Linear:
    layer = Fp8Linear(
        in_features,
        out_features,
        fp8_dtype=torch.float8_e4m3fn,
        compute_dtype=compute_dtype,
    )
    generator = torch.Generator().manual_seed(3)
    layer.load_state_dict(
        {
            "weight": torch.randn(out_features, in_features, generator=generator).to(
                torch.float8_e4m3fn
            ),
            "weight_scale": torch.tensor(0.5, dtype=torch.float32),
            "input_scale": torch.tensor(input_scale, dtype=torch.float32),
            "bias": torch.randn(out_features, generator=generator).to(compute_dtype),
        },
        assign=True,
    )
    return layer.to(device)


def quantized_input_reference(layer: Fp8Linear, x: torch.Tensor) -> torch.Tensor:
    """What the fp8 matmul path computes, replayed at fp32: quantize
    the input per-tensor exactly like the module, then matmul the
    dequantized operands."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scaled = x.float() / layer.input_scale.float()
    q = torch.clamp(scaled, -fp8_max, fp8_max).to(torch.float8_e4m3fn).float()
    out = (q * layer.input_scale) @ (layer.weight.float() * layer.weight_scale).T
    if layer.bias is not None:
        out = out + layer.bias.float()
    return out.to(layer.compute_dtype)


def test_fp8_module_residency_dequant_path_is_bitwise() -> None:
    layer = make_fp8_linear_cuda("cpu")
    mechanism = enroll_component(layer, load_device="cuda:0", offload_device="cpu")
    input = torch.randn(4, 32, device="cuda:0", dtype=torch.bfloat16)
    mechanism.partially_load(None)
    resident = layer(input)
    mechanism.unload()
    assert torch.equal(layer(input), resident)


@requires_fp8_matmul
def test_fp8_module_residency_hardware_path_is_bitwise() -> None:
    """Unpatched offloaded fp8 keeps qdata + scales intact and uses
    the same hardware matmul numerics as a resident unit."""
    layer = make_fp8_linear_cuda("cpu")
    layer.bind_fp8_matmul(True)
    mechanism = enroll_component(layer, load_device="cuda:0", offload_device="cpu")
    input = torch.randn(4, 32, device="cuda:0", dtype=torch.bfloat16)
    mechanism.partially_load(None)
    resident = layer(input)
    mechanism.unload()
    assert torch.equal(layer(input), resident)


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_supports_fp8_matmul_reports_capability(device: str) -> None:
    major, minor = torch.cuda.get_device_capability(torch.device(device))
    expected = major >= 9 or (major == 8 and minor >= 9)
    assert supports_fp8_matmul(torch.device(device)) is expected


@requires_fp8_matmul
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("input_scale", [1.0, 0.5])
def test_fp8_matmul_matches_quantized_reference(device: str, input_scale: float) -> None:
    layer = make_fp8_linear_cuda(device, input_scale=input_scale)
    layer.bind_fp8_matmul(True)
    x = torch.randn(
        4,
        32,
        device=device,
        dtype=torch.bfloat16,
        generator=torch.Generator(device).manual_seed(11),
    )
    got = layer(x)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, quantized_input_reference(layer, x), rtol=0.05, atol=0.05)


@pytest.mark.skipif(
    not cuda_available
    or torch.cuda.device_count() < 2
    or not supports_fp8_matmul(torch.device("cuda:1")),
    reason="hardware fp8 matmul unavailable on cuda:1 (needs compute capability 8.9+)",
)
@requires_two_gpus
def test_fp8_matmul_kitchen_quantizes_on_non_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.cuda.set_device("cuda:0")
    layer = make_fp8_linear_cuda("cuda:1")
    layer.bind_fp8_matmul(True)
    x = torch.randn(
        4,
        32,
        device="cuda:1",
        dtype=torch.bfloat16,
        generator=torch.Generator("cuda:1").manual_seed(17),
    )
    kitchen = quant_linear_mod._kitchen_quantize_per_tensor_fp8  # pyright: ignore[reportPrivateUsage]
    assert kitchen is not None
    quantize_calls: list[tuple[torch.Tensor, int]] = []

    def recording_kitchen(
        input: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        quantize_calls.append((input, torch.cuda.current_device()))
        return kitchen(input, scale, dtype)

    monkeypatch.setattr(
        quant_linear_mod,
        "_kitchen_quantize_per_tensor_fp8",
        recording_kitchen,
    )
    with torch.no_grad():
        got = layer(x)

    assert quantize_calls == [(x, 1)]
    assert torch.cuda.current_device() == 0
    assert got.device == torch.device("cuda:1")
    torch.testing.assert_close(got, quantized_input_reference(layer, x), rtol=0.05, atol=0.05)


@requires_fp8_matmul
def test_fp8_matmul_uses_kitchen_input_quantizer_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = make_fp8_linear_cuda("cuda:0")
    layer.bind_fp8_matmul(True)
    x = torch.randn(
        8,
        32,
        device="cuda:0",
        dtype=torch.bfloat16,
        generator=torch.Generator("cuda:0").manual_seed(19),
    )
    kitchen = quant_linear_mod._kitchen_quantize_per_tensor_fp8  # pyright: ignore[reportPrivateUsage]
    assert kitchen is not None
    quantize_calls: list[torch.Tensor] = []

    def recording_kitchen(
        input: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        quantize_calls.append(input)
        return kitchen(input, scale, dtype)

    monkeypatch.setattr(
        quant_linear_mod,
        "_kitchen_quantize_per_tensor_fp8",
        recording_kitchen,
    )
    with torch.no_grad():
        got = layer(x)
    assert quantize_calls == [x]
    torch.testing.assert_close(got, quantized_input_reference(layer, x), rtol=0.05, atol=0.05)


@requires_fp8_matmul
def test_fp8_matmul_3d_input_reshapes_and_rank4_falls_back() -> None:
    layer = make_fp8_linear_cuda("cuda:0")
    layer.bind_fp8_matmul(True)
    x3 = torch.randn(2, 5, 32, device="cuda:0", dtype=torch.bfloat16)
    got = layer(x3)
    assert got.shape == (2, 5, 16)
    torch.testing.assert_close(
        got,
        quantized_input_reference(layer, x3.reshape(-1, 32)).reshape(2, 5, 16),
        rtol=0.05,
        atol=0.05,
    )
    x4 = torch.randn(2, 2, 3, 32, device="cuda:0", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(
        x4,
        layer.weight.to(torch.bfloat16) * layer.weight_scale.to(torch.bfloat16),
        layer.bias,
    )
    assert torch.equal(layer(x4), expected)


@requires_fp8_matmul
def test_fp8_matmul_agrees_with_dequant_path() -> None:
    """The two routes are the same math up to input quantization; on
    unit-scale activations they must agree within e4m3's per-tensor
    quantization envelope."""
    layer = make_fp8_linear_cuda("cuda:0")
    x = torch.randn(8, 32, device="cuda:0", dtype=torch.bfloat16)
    dequant = layer(x)
    layer.bind_fp8_matmul(True)
    matmul = layer(x)
    reference_span = dequant.abs().amax()
    assert (matmul - dequant).abs().amax() <= 0.15 * reference_span


@requires_fp8_matmul
def test_fp8_matmul_float32_compute_adds_bias_unfused() -> None:
    """cuBLASLt's fused bias epilogue is fp16/bf16-output-only; at
    float32 compute the route must pass bias=None to _scaled_mm and
    add the bias afterwards - same numbers, no backend error."""
    layer = make_fp8_linear_cuda("cuda:0", compute_dtype=torch.float32)
    layer.bind_fp8_matmul(True)
    x = torch.randn(
        4,
        32,
        device="cuda:0",
        dtype=torch.float32,
        generator=torch.Generator("cuda:0").manual_seed(17),
    )
    got = layer(x)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, quantized_input_reference(layer, x), rtol=0.05, atol=0.05)


@requires_fp8_matmul
def test_fp8_matmul_route_refuses_gradient_input_on_cuda() -> None:
    """torch._scaled_mm has no backward; the bound route must refuse a
    gradient-requiring input loudly instead of silently cutting the
    graph (the dequant route keeps autograd, proven below)."""
    layer = make_fp8_linear_cuda("cuda:0")
    layer.bind_fp8_matmul(True)
    x = torch.randn(2, 32, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        layer(x)
    with torch.no_grad():
        assert bool(torch.isfinite(layer(x).float()).all())


@requires_fp8_matmul
def test_fp8_matmul_compile_fullgraph_inductor_cuda() -> None:
    """The compile discipline holds on the hardware path: the route
    is bind-time module state, so fullgraph compilation through
    inductor must not graph-break on torch._scaled_mm."""
    layer = make_fp8_linear_cuda("cuda:0")
    layer.bind_fp8_matmul(True)
    assert quant_linear_mod._kitchen_quantize_per_tensor_fp8 is not None  # pyright: ignore[reportPrivateUsage]
    compiled = torch.compile(layer, fullgraph=True)
    for seed in (1, 2):
        x = torch.randn(
            8,
            32,
            device="cuda:0",
            dtype=torch.bfloat16,
            generator=torch.Generator("cuda:0").manual_seed(seed),
        )
        torch.testing.assert_close(compiled(x), layer(x), rtol=1e-2, atol=1e-2)


@requires_fp8_matmul
def test_fp8_matmul_compile_first_forward_in_fresh_process_cuda() -> None:
    require_gpu_tests_enabled()
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    source = str(root / "packages" / "dinkster-inference-torch" / "src")
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if inherited_path is None else os.pathsep.join((source, inherited_path))
    )
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        import torch
        from dinkster_inference_torch import Fp8Linear
        from dinkster_inference_torch import quant_linear

        assert os.environ.get("DINKSTER_ENABLE_GPU_TESTS") == "1"
        assert Path(quant_linear.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())

        layer = Fp8Linear(
            32,
            16,
            fp8_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        generator = torch.Generator().manual_seed(3)
        layer.load_state_dict(
            {
                "weight": torch.randn(16, 32, generator=generator).to(torch.float8_e4m3fn),
                "weight_scale": torch.tensor(0.5, dtype=torch.float32),
                "input_scale": torch.tensor(1.0, dtype=torch.float32),
                "bias": torch.randn(16, generator=generator).to(torch.bfloat16),
            },
            assign=True,
        )
        layer = layer.to("cuda:0")
        layer.bind_fp8_matmul(True)
        compiled = torch.compile(layer, fullgraph=True)
        x = torch.randn(
            8,
            32,
            device="cuda:0",
            dtype=torch.bfloat16,
            generator=torch.Generator("cuda:0").manual_seed(7),
        )
        compiled_output = compiled(x)
        eager_output = layer(x)
        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-2, atol=1e-2)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, source],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
        env=environment,
        cwd=root,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@requires_fp8_matmul
def test_fp8_linear_worker_thread_per_device() -> None:
    layers = {device: make_fp8_linear_cuda(device) for device in CUDA_DEVICES}
    for layer in layers.values():
        layer.bind_fp8_matmul(True)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    expected = {device: layers[device](x.to(device)).cpu() for device in CUDA_DEVICES}
    errors: list[BaseException] = []
    results: dict[str, torch.Tensor] = {}

    def run(device: str) -> None:
        try:
            results[device] = layers[device](x.to(device)).cpu()
        except BaseException as exc:  # noqa: BLE001 - reraised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(device,)) for device in CUDA_DEVICES]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for device in CUDA_DEVICES:
        torch.testing.assert_close(results[device], expected[device])


def test_fp8_linear_autograd_input_grad_on_cuda() -> None:
    layer = make_fp8_linear_cuda("cuda:0", compute_dtype=torch.float32)
    x = torch.randn(2, 32, device="cuda:0", requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


# ----------------------------------------- real checkpoint assembly

#: The real-checkpoint tests spread across both 4090s when both are
#: visible; under a CUDA_VISIBLE_DEVICES=<one> run everything shares
#: the single device.
_SECONDARY = "cuda:1" if cuda_available and torch.cuda.device_count() >= 2 else "cuda:0"


def _real_flux_plan(**sources: Path):  # noqa: ANN202 - test-local helper
    from dinkster_inference import load_safetensors_header, plan_flux_assembly

    return plan_flux_assembly(
        **{name: load_safetensors_header(path) for name, path in sources.items()}
    )


def _bounded_flux_inputs(device: str, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device).manual_seed(5)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    return (
        randn(1, 16, 8, 8),
        torch.tensor([0.5], device=device, dtype=dtype),
        randn(1, 256, 4096),
        randn(1, 768),
        torch.tensor([3.5], device=device, dtype=dtype),
    )


@requires_fp8_matmul
@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_SPLIT_SCALED.values()),
    reason="split scaled-fp8 checkpoint set not present",
)
def test_real_scaled_fp8_split_assembles_and_runs_on_both_gpus() -> None:
    """The full slice on real weights: plan the split scaled-fp8 set,
    assemble (266 Fp8Linear swaps, mixed storage preserved), run the
    DiT on cuda:0 through BOTH matmul routes, and run the VAE and the
    identity-filled CLIP-L on cuda:1."""
    from dinkster_inference import FLUX_DEV

    plan = _real_flux_plan(**_REAL_SPLIT_SCALED)
    assert len(plan.diffusion.quant) == 266
    assembled = assemble_flux(plan)
    assert assembled.clip_l is not None and assembled.t5xxl is not None
    assert assembled.family is FLUX_DEV
    fp8_layers = [
        module for module in assembled.diffusion.modules() if isinstance(module, Fp8Linear)
    ]
    assert len(fp8_layers) == len(plan.diffusion.quant)
    assert all(layer.weight.dtype == torch.float8_e4m3fn for layer in fp8_layers)
    # T5 storage stays exactly as shipped (fp16), per parameter.
    t5_dtypes = {tensor.dtype for tensor in assembled.t5xxl.state_dict().values()}
    assert t5_dtypes == {torch.float16}

    diffusion = assembled.diffusion.to("cuda:0")
    inputs = _bounded_flux_inputs("cuda:0", torch.bfloat16)
    with torch.no_grad():
        dequant_out = diffusion(*inputs)
    assert bool(torch.isfinite(dequant_out).all())
    for layer in fp8_layers:
        if not layer.full_precision_matmul:
            layer.bind_fp8_matmul(True)
    with torch.no_grad():
        matmul_out = diffusion(*inputs)
    assert bool(torch.isfinite(matmul_out).all())
    # Whole-model dequant-vs-matmul agreement is a structural gate,
    # not a numeric one: the checkpoint's input_scale calibration is
    # for real activation distributions, so random inputs compound
    # e4m3 input-quantization error over all 57 blocks (measured
    # rel-RMS ~0.32, cosine ~0.95 on this seed). A scale/orientation
    # defect would send cosine toward 0; per-layer numerics are
    # pinned tightly in test_fp8_matmul_matches_quantized_reference.
    cosine = torch.nn.functional.cosine_similarity(
        dequant_out.float().flatten(), matmul_out.float().flatten(), dim=0
    )
    assert float(cosine) > 0.9
    rel_rms = (matmul_out.float() - dequant_out.float()).norm() / dequant_out.float().norm()
    assert float(rel_rms) < 0.5
    diffusion.to("cpu")
    soft_empty_cache(torch.device("cuda:0"))

    vae = assembled.vae.to(_SECONDARY)
    with torch.no_grad():
        image = vae.decode(torch.randn(1, 16, 32, 32, device=_SECONDARY))
    assert image.shape == (1, 3, 256, 256)
    assert bool(torch.isfinite(image).all())

    clip = assembled.clip_l.to(_SECONDARY)
    tokens = torch.tensor([[49406, 320, 2368, 49407]], device=_SECONDARY)
    with torch.no_grad():
        clip_out = clip(clip.embed_tokens(tokens), torch.tensor([3], device=_SECONDARY))
    assert clip_out.pooled.shape == (1, 768)
    assert bool(torch.isfinite(clip_out.pooled).all())


@pytest.mark.skipif(
    not _REAL_COMBINED_PLAIN.exists(),
    reason="combined plain-fp8 checkpoint not present",
)
@requires_fp8_matmul
def test_real_combined_plain_fp8_assembles_and_encodes() -> None:
    """flux1-dev-fp8: plain fp8 storage, NO scales - assembly must
    keep fp8 per-parameter storage with cast-at-use (no Fp8Linear
    anywhere), bind its scale-1 matmul route, and keep the fp8 T5's
    NaN scrub gate live on the real weights (finite, non-degenerate
    routed encode on cuda:1)."""
    plan = _real_flux_plan(checkpoint=_REAL_COMBINED_PLAIN)
    assembled = assemble_flux(plan, fp8_matmul=True)
    assert assembled.t5xxl is not None
    for module in (assembled.diffusion, assembled.t5xxl):
        assert not any(isinstance(child, Fp8Linear) for child in module.modules())
        linears = [child for child in module.modules() if isinstance(child, torch.nn.Linear)]
        assert linears
        assert all(child.__dict__.get("fp8_matmul") is True for child in linears)
    dit_dtypes = {tensor.dtype for tensor in assembled.diffusion.state_dict().values()}
    assert dit_dtypes == {torch.float8_e4m3fn}
    t5_dtypes = {tensor.dtype for tensor in assembled.t5xxl.state_dict().values()}
    assert t5_dtypes == {torch.float8_e4m3fn}

    t5 = assembled.t5xxl.to(_SECONDARY)
    tokens = torch.tensor([[71, 1712, 13, 3, 9, 1712, 1]], device=_SECONDARY)
    embeds = t5.embed_tokens(tokens)
    assert embeds.dtype == torch.float32
    assert bool(torch.isfinite(embeds).all())
    with torch.no_grad():
        encoded = t5(embeds)
    assert encoded.shape == (1, 7, 4096)
    assert bool(torch.isfinite(encoded).all())
    assert float(encoded.std()) > 0.01
    t5.to("cpu")
    soft_empty_cache(torch.device(_SECONDARY))


@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_SPLIT_SCALED.values()),
    reason="split scaled-fp8 checkpoint set not present",
)
def test_real_flux_denoise_runs_on_cuda() -> None:
    """The denoise bridge over real assembled weights: native step
    planning (sampling_sigmas), FluxDenoiser with the distilled
    guidance default, and run_denoise driving euler for two real
    steps on the fp8 flux1-dev DiT. Minimal by design - the latent is
    tiny and the schedule short; this proves the bridge end to end on
    real weights, not image quality."""
    from dinkster_inference import (
        FLUX_DEV,
        Conditioning,
        FluxFlowSigmas,
        sampling_sigmas,
    )
    from dinkster_inference.schedules import DINKSTER_NORMAL
    from dinkster_inference.solvers import DINKSTER_EULER

    plan = _real_flux_plan(**_REAL_SPLIT_SCALED)
    assembled = assemble_flux(plan)
    diffusion = assembled.diffusion.to("cuda:0")
    generator = torch.Generator("cuda:0").manual_seed(3)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda:0", dtype=torch.float32, generator=generator)

    cond = Conditioning(embeddings=randn(1, 256, 4096), pooled=randn(1, 768))
    denoiser = FluxDenoiser(diffusion, cond)
    assert denoiser.guidance == 3.5

    sigmas = sampling_sigmas(DINKSTER_NORMAL, FluxFlowSigmas(), 2)
    assert len(sigmas) == 3 and sigmas[-1] == 0.0
    latent = torch.zeros(1, 16, 8, 8)
    with torch.no_grad():
        out = run_denoise(
            denoiser,
            DINKSTER_EULER.build(),
            latent=latent,
            noise=prepare_noise(latent, 7),
            sigmas=sigmas,
            family=FLUX_DEV,
            seed=7,
            device="cuda:0",
        )
    assert out.shape == latent.shape
    assert out.device.type == "cuda"
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())
    assert float(out.std()) > 0.01
    diffusion.to("cpu")
    soft_empty_cache(torch.device("cuda:0"))


_Z_IMAGE_MODELS = Path(
    os.environ.get("DINKSTER_Z_IMAGE_MODELS", "C:/Users/kosin/Documents/ComfyUI/models")
)
_REAL_Z_IMAGE_COMMON = {"vae": _Z_IMAGE_MODELS / "vae/ae.safetensors"}
_REAL_Z_IMAGE_QWEN = _Z_IMAGE_MODELS / "text_encoders/qwen_3_4b.safetensors"
_REAL_Z_IMAGE_QWEN_FP8 = Path(
    "C:/Users/kosin/Documents/ComfyUI/models/text_encoders/qwen_3_4b_fp8_mixed.safetensors"
)
_REAL_Z_IMAGE_QWEN_FP4 = Path(
    "C:/Users/kosin/Documents/ComfyUI/models/text_encoders/qwen_3_4b_fp4_mixed.safetensors"
)
_REAL_Z_IMAGE_CONTROL = Path(
    "C:/Users/kosin/ComfyUI-Shared/models/model_patches/"
    "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"
)
_REAL_Z_IMAGE_PIXEL = Path(
    "C:/Users/kosin/ComfyUI-Shared/models/diffusion_models/"
    "zeta-chroma-base-x0-pixel-no-dino-1024.safetensors"
)


@pytest.mark.parametrize(
    ("diffusion_path", "qwen_path"),
    (
        pytest.param(
            _Z_IMAGE_MODELS / "diffusion_models/z_image_turbo_bf16.safetensors",
            _REAL_Z_IMAGE_QWEN,
            id="turbo-bf16",
        ),
        pytest.param(
            Path("C:/Users/kosin/ComfyUI-Shared/models/diffusion_models/z_image_bf16.safetensors"),
            _REAL_Z_IMAGE_QWEN,
            id="base-bf16",
        ),
        pytest.param(
            Path(
                "C:/Users/kosin/Documents/ComfyUI/models/diffusion_models/"
                "z_image_turbo_bf16.safetensors"
            ),
            _REAL_Z_IMAGE_QWEN_FP8,
            id="turbo-qwen-fp8-mixed",
        ),
        pytest.param(
            Path(
                "C:/Users/kosin/Documents/ComfyUI/models/diffusion_models/"
                "z_image_turbo_bf16.safetensors"
            ),
            _REAL_Z_IMAGE_QWEN_FP4,
            id="turbo-qwen-fp4-mixed",
        ),
    ),
)
def test_real_z_image_runtime_encodes_and_samples_on_cuda(
    diffusion_path: Path, qwen_path: Path
) -> None:
    from dinkster_inference import load_safetensors_header, probe_native
    from dinkster_inference_torch import ZImageRuntime, enroll_assembled, load_runtime

    required = (diffusion_path, qwen_path, *_REAL_Z_IMAGE_COMMON.values())
    if not all(path.exists() for path in required):
        pytest.skip("real Z-Image checkpoint set not present")
    diffusion = load_safetensors_header(diffusion_path)
    qwen3_4b = load_safetensors_header(qwen_path)
    vae = load_safetensors_header(_REAL_Z_IMAGE_COMMON["vae"])
    capability = probe_native(diffusion=diffusion, qwen3_4b=qwen3_4b, vae=vae)
    assert capability.native and capability.family_id == "dinkster.z_image"
    runtime = load_runtime(
        diffusion=diffusion,
        qwen3_4b=qwen3_4b,
        vae=vae,
        diffusion_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
    )
    assert isinstance(runtime, ZImageRuntime)
    assert runtime.runtime_identity.startswith("native:dinkster.z_image:")
    assert runtime.assembled.compute_dtype("qwen3_4b") is torch.float32
    assert not hasattr(runtime, "streamed_residency_components")

    enrolled = enroll_assembled(
        runtime.assembled,
        load_device="cuda:0",
        offload_device="cpu",
    )
    enrolled["qwen3_4b"].partially_load(None)
    with torch.no_grad():
        resident_cond = runtime.encode_text("a photo of a cat")
    enrolled["qwen3_4b"].unload()
    soft_empty_cache(torch.device("cuda:0"))
    with torch.no_grad():
        cond = runtime.encode_text("a photo of a cat")
    torch.testing.assert_close(cond.embeddings, resident_cond.embeddings, rtol=0, atol=0)
    assert cond.embeddings.shape[0] == 1 and cond.embeddings.shape[2] == 2560
    assert bool(torch.isfinite(cond.embeddings).all())

    if diffusion_path.name == "z_image_turbo_bf16.safetensors" and qwen_path == _REAL_Z_IMAGE_QWEN:
        content = torch.linspace(0.0, 1.0, 3 * 32 * 32, device="cuda:0").reshape(1, 3, 32, 32)
        enrolled["vae"].partially_load(None)
        assert enrolled["vae"].loaded_bytes() == enrolled["vae"].total_bytes()
        with torch.no_grad():
            resident_latent = runtime.encode_content(content)
            resident_decoded = runtime.decode_latent(resident_latent)
        enrolled["vae"].unload()
        assert enrolled["vae"].loaded_bytes() == 0
        soft_empty_cache(torch.device("cuda:0"))
        enrolled["vae"].partially_load(None)
        with torch.no_grad():
            reloaded_latent = runtime.encode_content(content)
            reloaded_decoded = runtime.decode_latent(reloaded_latent)
        assert resident_latent.shape == (1, 16, 4, 4)
        assert resident_decoded.shape == content.shape
        assert resident_latent.device == resident_decoded.device == torch.device("cuda:0")
        assert resident_latent.dtype == torch.bfloat16
        assert resident_decoded.dtype == torch.float32
        assert reloaded_latent.device == reloaded_decoded.device == torch.device("cuda:0")
        assert reloaded_latent.dtype == torch.bfloat16
        assert reloaded_decoded.dtype == torch.float32
        assert bool(torch.isfinite(resident_latent).all())
        assert bool(torch.isfinite(resident_decoded).all())
        torch.testing.assert_close(reloaded_latent, resident_latent, rtol=0, atol=0)
        torch.testing.assert_close(reloaded_decoded, resident_decoded, rtol=0, atol=0)
        enrolled["vae"].unload()
        soft_empty_cache(torch.device("cuda:0"))

    enrolled["diffusion"].partially_load(None)
    latent = torch.zeros(1, 16, 8, 8)
    with torch.no_grad():
        out = runtime.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            seed=7,
            device="cuda:0",
        )
    assert out.shape == latent.shape
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())
    assert float(out.std()) > 0.01

    if (
        "turbo" in diffusion_path.name
        and qwen_path == _REAL_Z_IMAGE_QWEN
        and _REAL_Z_IMAGE_CONTROL.exists()
    ):
        from dinkster_inference import (
            ControlApplication,
            PayloadReference,
            PercentRange,
            plan_z_image_control,
        )
        from dinkster_inference_torch import (
            ZImageControlConditioning,
            assemble_z_image_control,
            enroll_component,
            z_image_control_hint_digest,
        )

        control_plan = plan_z_image_control(
            load_safetensors_header(_REAL_Z_IMAGE_CONTROL),
            asset_digest=(
                "blake3:50534075327608a2ac2a3bd1bcdc49a8a220b86a06374ca138c6a13b3784460e"
            ),
        )
        assembled_control = assemble_z_image_control(control_plan)
        control = assembled_control.control
        control_mechanism = enroll_component(
            control,
            load_device="cuda:0",
            offload_device="cpu",
        )
        control_mechanism.partially_load(None)
        control_hint = torch.zeros_like(latent, device="cuda:0", dtype=torch.bfloat16)
        hint_digest = z_image_control_hint_digest(control_hint)
        control_conditioning = ZImageControlConditioning(
            ControlApplication(
                "z-image-fun",
                PayloadReference(hint_digest),
                1.0,
                PercentRange(0.0, 1.0),
            ),
            control,
            control_hint,
            assembled_control.resource_digest,
            hint_digest,
        )
        with torch.no_grad():
            controlled = runtime.sample(
                latent,
                cond=cond,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=1,
                seed=7,
                device="cuda:0",
                control=control_conditioning,
            )
        assert controlled.shape == latent.shape
        assert bool(torch.isfinite(controlled).all())
        assert float(controlled.std()) > 0.01
        assert not torch.equal(controlled, out)
        control_mechanism.unload()

    for mechanism in enrolled.values():
        mechanism.unload()
    soft_empty_cache(torch.device("cuda:0"))


@pytest.mark.skipif(
    not _REAL_Z_IMAGE_PIXEL.exists() or not _REAL_Z_IMAGE_QWEN.exists(),
    reason="real Zeta-Chroma checkpoint set not present",
)
def test_real_z_image_pixel_runtime_encodes_and_samples_on_cuda() -> None:
    from dinkster_inference import load_safetensors_header, probe_native
    from dinkster_inference_torch import (
        ZImagePixelCodec,
        ZImageRuntime,
        enroll_assembled,
        load_runtime,
    )

    diffusion = load_safetensors_header(_REAL_Z_IMAGE_PIXEL)
    qwen3_4b = load_safetensors_header(_REAL_Z_IMAGE_QWEN)
    capability = probe_native(diffusion=diffusion, qwen3_4b=qwen3_4b)
    assert capability.native and capability.family_id == "dinkster.z_image_pixel_space"
    runtime = load_runtime(
        diffusion=diffusion,
        qwen3_4b=qwen3_4b,
        diffusion_dtype=torch.bfloat16,
        text_dtype=torch.bfloat16,
    )
    assert isinstance(runtime, ZImageRuntime)
    assert isinstance(runtime.assembled.vae, ZImagePixelCodec)
    assert runtime.runtime_identity.startswith("native:dinkster.z_image_pixel_space:")

    enrolled = enroll_assembled(
        runtime.assembled,
        load_device="cuda:0",
        offload_device="cpu",
    )
    with torch.inference_mode():
        cond = runtime.encode_text("a photo of a cat")
    assert cond.embeddings.shape[0] == 1 and cond.embeddings.shape[2] == 2560
    assert bool(torch.isfinite(cond.embeddings).all())

    enrolled["diffusion"].partially_load(None)
    pixels = torch.zeros(1, 3, 256, 256)
    with torch.inference_mode():
        output = runtime.sample(
            pixels,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            seed=294,
            device="cuda:0",
        )
        decoded = runtime.decode_latent(output)
    assert output.shape == pixels.shape
    assert output.dtype == torch.float32
    assert bool(torch.isfinite(output).all())
    assert float(output.std()) > 0.01
    torch.testing.assert_close(decoded, output, rtol=0, atol=0)

    for mechanism in enrolled.values():
        mechanism.unload()
    soft_empty_cache(torch.device("cuda:0"))


_REAL_SD1_CHECKPOINT = _FLUX_MODELS / "checkpoints/v1-5-pruned-emaonly-fp16.safetensors"

_SDXL_CONTROL_MODEL_ROOT = Path(
    os.environ.get("DINKSTER_SDXL_CONTROL_MODELS", "/home/kosin/ComfyUI/models")
)
_REAL_SDXL_BASE = _SDXL_CONTROL_MODEL_ROOT / "checkpoints/sd_xl_base_1.0.safetensors"
_REAL_SDXL_CONTROL_LORA = (
    _SDXL_CONTROL_MODEL_ROOT / "controlnet/control-lora-canny-rank128.safetensors"
)
_REAL_SDXL_CONTROLNET_UNION = (
    _SDXL_CONTROL_MODEL_ROOT / "controlnet/xinsir-controlnet-union-sdxl-1.0-promax.safetensors"
)
_REAL_SDXL_CONTROLNET = (
    _SDXL_CONTROL_MODEL_ROOT / "controlnet/diffusers-controlnet-canny-sdxl-1.0-fp16.safetensors"
)


@pytest.mark.skipif(
    not (_REAL_SDXL_BASE.exists() and _REAL_SDXL_CONTROL_LORA.exists()),
    reason="official SDXL base and Control-LoRA artifacts not present",
)
def test_real_sdxl_control_lora_end_to_end_on_cuda() -> None:
    from dinkster_inference import (
        load_safetensors_header,
        plan_sd_assembly,
        plan_sdxl_control_lora,
    )
    from dinkster_inference_torch import assemble_sd, assemble_sdxl_control_lora

    golden = load_platform_golden(
        Path(__file__).parent / "goldens" / "sdxl_control_lora_acceptance.json"
    )
    base_artifact = golden["artifacts"]["base"]
    control_artifact = golden["artifacts"]["control_lora"]
    base_digest = "blake3:" + base_artifact["blake3"]
    control_digest = "blake3:" + control_artifact["blake3"]
    base_plan = plan_sd_assembly(checkpoint=load_safetensors_header(_REAL_SDXL_BASE))
    control_plan = plan_sdxl_control_lora(
        load_safetensors_header(_REAL_SDXL_CONTROL_LORA),
        asset_digest=control_digest,
        base_asset_digest=base_digest,
    )
    base = assemble_sd(base_plan, diffusion_asset_digest=base_digest)
    control = assemble_sdxl_control_lora(control_plan, base)
    device = torch.device("cuda:0")
    base.diffusion.to(device)
    control.control_lora.to(device)
    generator = torch.Generator(device=device).manual_seed(golden["workload"]["seed"])
    x = torch.randn((1, 4, 8, 8), generator=generator, device=device, dtype=torch.float16)
    hint = torch.rand((1, 3, 64, 64), generator=generator, device=device, dtype=torch.float16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.float16)
    context = torch.randn((1, 77, 2048), generator=generator, device=device, dtype=torch.float16)
    adm = torch.randn((1, 2816), generator=generator, device=device, dtype=torch.float16)
    with torch.inference_mode():
        residuals = control.control_lora(x, hint, timestep, context, adm)
        controlled = base.diffusion(x, timestep, context=context, y=adm, control=residuals).float()
        plain = base.diffusion(x, timestep, context=context, y=adm).float()
    output_bytes = controlled.cpu().contiguous().numpy().tobytes()
    assert hashlib.sha256(output_bytes).hexdigest() == golden["result"]["output_sha256"]
    assert (
        float((controlled - plain).abs().max()) == golden["result"]["controlled_vs_plain_max_abs"]
    )
    assert (
        float((controlled - plain).abs().mean()) == golden["result"]["controlled_vs_plain_mean_abs"]
    )
    base.diffusion.to("cpu")
    control.control_lora.to("cpu")
    soft_empty_cache(device)


@pytest.mark.skipif(
    not _REAL_SDXL_CONTROLNET_UNION.exists(),
    reason="official xinsir SDXL ControlNet Union artifact not present",
)
def test_real_sdxl_controlnet_union_matches_acceptance_golden_on_cuda() -> None:
    from dinkster_inference import load_safetensors_header, plan_sdxl_controlnet_union
    from dinkster_inference_torch import assemble_sdxl_controlnet_union

    golden = load_platform_golden(
        Path(__file__).parent / "goldens" / "sdxl_controlnet_union_acceptance.json"
    )
    artifact = golden["artifact"]
    digest = "blake3:" + artifact["blake3"]
    plan = plan_sdxl_controlnet_union(
        load_safetensors_header(_REAL_SDXL_CONTROLNET_UNION), asset_digest=digest
    )
    assembled = assemble_sdxl_controlnet_union(plan)
    device = torch.device("cuda:0")
    assembled.controlnet_union.to(device)
    generator = torch.Generator(device=device).manual_seed(golden["workload"]["seed"])
    x = torch.randn((1, 4, 8, 8), generator=generator, device=device, dtype=torch.float16)
    hint = torch.rand((1, 3, 64, 64), generator=generator, device=device, dtype=torch.float16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.float16)
    context = torch.randn((1, 77, 2048), generator=generator, device=device, dtype=torch.float16)
    adm = torch.randn((1, 2816), generator=generator, device=device, dtype=torch.float16)
    with torch.inference_mode():
        residuals = assembled.controlnet_union(
            x, hint, timestep, context, adm, golden["workload"]["mode_index"]
        )
    values = (*residuals.down, residuals.middle)
    output_bytes = b"".join(value.float().cpu().contiguous().numpy().tobytes() for value in values)
    assert hashlib.sha256(output_bytes).hexdigest() == golden["result"]["residuals_sha256"]
    assert [list(value.shape) for value in values] == golden["result"]["residual_shapes"]
    assembled.controlnet_union.to("cpu")
    soft_empty_cache(device)


@pytest.mark.skipif(
    not _REAL_SDXL_CONTROLNET.exists(),
    reason="official classic SDXL ControlNet artifact not present",
)
def test_real_classic_sdxl_controlnet_matches_acceptance_golden_on_cuda() -> None:
    from dinkster_inference import load_safetensors_header, plan_sdxl_controlnet
    from dinkster_inference_torch import assemble_sdxl_controlnet

    golden = load_platform_golden(
        Path(__file__).parent / "goldens" / "sdxl_controlnet_acceptance.json"
    )
    artifact = golden["artifact"]
    digest = "blake3:" + artifact["blake3"]
    plan = plan_sdxl_controlnet(load_safetensors_header(_REAL_SDXL_CONTROLNET), asset_digest=digest)
    assembled = assemble_sdxl_controlnet(plan)
    device = torch.device("cuda:0")
    assembled.controlnet.to(device)
    generator = torch.Generator(device=device).manual_seed(golden["workload"]["seed"])
    x = torch.randn((1, 4, 8, 8), generator=generator, device=device, dtype=torch.float16)
    hint = torch.rand((1, 3, 64, 64), generator=generator, device=device, dtype=torch.float16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.float16)
    context = torch.randn((1, 77, 2048), generator=generator, device=device, dtype=torch.float16)
    adm = torch.randn((1, 2816), generator=generator, device=device, dtype=torch.float16)
    with torch.inference_mode():
        residuals = assembled.controlnet(x, hint, timestep, context, adm)
    values = (*residuals.down, residuals.middle)
    output_bytes = b"".join(value.float().cpu().contiguous().numpy().tobytes() for value in values)
    assert hashlib.sha256(output_bytes).hexdigest() == golden["result"]["residuals_sha256"]
    assert [list(value.shape) for value in values] == golden["result"]["residual_shapes"]
    assembled.controlnet.to("cpu")
    soft_empty_cache(device)


@pytest.mark.skipif(
    not _REAL_SD1_CHECKPOINT.exists(),
    reason="real SD1 checkpoint not present",
)
def test_real_sd1_runtime_end_to_end_on_cuda() -> None:
    """The full SD seam on real weights: probe -> load_runtime over
    the combined fp16 SD 1.5 checkpoint -> encode_text (CLIP-L, final
    hidden, raw pooled) -> a two-step euler sample on the fp16 UNet
    (the family's reference compute dtype) -> a KL decode. Minimal by
    design - a tiny latent and short schedule prove the seam end to
    end on real weights, not image quality."""
    from dinkster_inference import SamplingGuidance, load_safetensors_header, probe_native
    from dinkster_inference_torch import SDRuntime, load_runtime

    source = load_safetensors_header(_REAL_SD1_CHECKPOINT)
    capability = probe_native(source)
    assert capability.native and capability.family_id == "dinkster.sd15"
    runtime = load_runtime(source)
    assert isinstance(runtime, SDRuntime)
    assert runtime.family.id == "dinkster.sd15"
    assert runtime.runtime_identity.startswith("native:dinkster.sd15:")
    cond = runtime.encode_text("a photo of a cat")
    uncond = runtime.encode_text("")
    assert cond.embeddings.shape == (1, 77, 768)
    assert cond.pooled is not None and cond.pooled.shape == (1, 768)
    runtime.assembled.diffusion.to("cuda:0")  # placement is caller business
    latent = torch.zeros(1, 4, 8, 8)
    with torch.no_grad():
        out = runtime.sample(
            latent,
            cond=cond,
            cfg=SamplingGuidance(uncond, 7.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=2,
            seed=7,
            device="cuda:0",
        )
    assert out.shape == latent.shape
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())
    assert float(out.std()) > 0.01
    with torch.no_grad():
        image = runtime.decode_latent(out.to("cpu"))
    assert image.shape == (1, 3, 64, 64)
    assert bool(torch.isfinite(image).all())
    runtime.assembled.diffusion.to("cpu")
    soft_empty_cache(torch.device("cuda:0"))


@pytest.mark.skipif(
    not _REAL_SD1_CHECKPOINT.exists(),
    reason="real SD1 checkpoint not present",
)
@pytest.mark.parametrize(("size", "steps"), ((512, 20), (1024, 30)))
def test_real_sd1_single_device_fused_split_signature_on_cuda(size: int, steps: int) -> None:
    from dinkster_inference import (
        DiscreteSigmas,
        SamplingGuidance,
        cfg_combine,
        load_safetensors_header,
        sampling_sigmas,
    )
    from dinkster_inference_torch import SDRuntime, load_runtime
    from dinkster_inference_torch.denoise import prepare_noise, run_denoise
    from dinkster_inference_torch.schedules import torch_scheduler_registry
    from dinkster_inference_torch.sd_denoise import SDDenoiser
    from dinkster_inference_torch.solvers import torch_sampler_registry

    attention_route_token = discover_attention_route_token("sdpa")
    runtime = load_runtime(
        load_safetensors_header(_REAL_SD1_CHECKPOINT),
        attention_policy="sdpa",
        attention_route_token=attention_route_token,
        # The split-reference records pin SDPA and float32 text encoding so
        # provider or conditioning changes cannot redefine the topology records.
        text_dtype=torch.float32,
    )
    assert isinstance(runtime, SDRuntime)
    assert runtime.runtime_identity.startswith("native:dinkster.sd15:")
    assert runtime.receipt_identity is None
    runtime.assembled.diffusion.to("cuda:0")  # type: ignore[attr-defined]
    cond = runtime.encode_text("a photo of a cat")
    uncond = runtime.encode_text("")
    latent = torch.zeros(1, 4, size // 8, size // 8)
    space = DiscreteSigmas.linear_beta()
    scheduler = torch_scheduler_registry().get("dinkster.normal")
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert scheduler is not None and sampler is not None
    sigmas = sampling_sigmas(scheduler, space, steps)

    positive = SDDenoiser(
        runtime.assembled.diffusion,  # type: ignore[attr-defined]
        space,
        cond,
        compute_dtype=torch.float16,
    )
    negative = SDDenoiser(
        runtime.assembled.diffusion,  # type: ignore[attr-defined]
        space,
        uncond,
        compute_dtype=torch.float16,
    )

    def split(x: torch.Tensor, sigma: float) -> torch.Tensor:
        return cfg_combine(positive(x, sigma), negative(x, sigma), 7.0)

    with torch.no_grad():
        seed = 264
        noise = prepare_noise(latent, seed)
        runtime_result = runtime.sample(
            latent=latent,
            cond=cond,
            cfg=SamplingGuidance(uncond, 7.0),
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=steps,
            seed=seed,
            device="cuda:0",
        )
        split_result = run_denoise(
            split,
            sampler.build(),
            latent=latent,
            noise=noise,
            sigmas=sigmas,
            family=runtime.family,
            seed=seed,
            device="cuda:0",
        )
        assert not torch.equal(runtime_result, split_result)
        record_dir = {
            (8, 9): "distributed-sd15-fp16-guidance-d1",
            (12, 0): "distributed-sd15-fp16-guidance-sm120-d1",
        }.get(torch.cuda.get_device_capability(0))
        if record_dir is None:
            pytest.skip("no SD1.5 split-reference record for this device capability")
        reference = json.loads(
            (INFERENCE_PARITY_RECORDS / record_dir / "split-reference.json").read_text()
        )
        expected = {(case["width"], case["steps"]): case for case in reference["cases"]}

        def samples_sha256(value: torch.Tensor) -> str:
            payload = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            return hashlib.sha256(payload).hexdigest()

        case = expected[(size, steps)]
        assert samples_sha256(runtime_result) == case["direct_fused_samples_sha256"]
        assert samples_sha256(split_result) == case["direct_split_samples_sha256"]

    runtime.assembled.diffusion.to("cpu")  # type: ignore[attr-defined]
    soft_empty_cache(torch.device("cuda:0"))


@pytest.mark.parametrize("device_index", range(torch.cuda.device_count()))
def test_taesd_strict_load_and_basic_codec_on_each_gpu(device_index: int) -> None:
    weights = Path("/tmp/dinkster-taesd-weights")
    if not weights.exists():
        pytest.skip("official TAESD weights are not installed")
    device = torch.device(f"cuda:{device_index}")
    encoder, decoder = TAESDEncoder(), TAESDDecoder()
    encoder.load_state_dict(
        torch.load(weights / "taesd_encoder.pth", map_location="cpu", weights_only=True),
        strict=True,
        assign=True,
    )
    decoder.load_state_dict(
        torch.load(weights / "taesd_decoder.pth", map_location="cpu", weights_only=True),
        strict=True,
        assign=True,
    )
    encoder.to(device)
    decoder.to(device)
    with torch.no_grad():
        latent = encoder.encode(torch.zeros(1, 3, 16, 16, device=device))
        decoded = decoder.decode(latent)
    assert latent.shape == (1, 4, 2, 2)
    assert decoded.shape == (1, 3, 16, 16)
    assert bool(torch.isfinite(decoded).all())


def _fill_wan21_parameters(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("gamma"):
                parameter.fill_(1.0)
            elif name.endswith("bias"):
                parameter.zero_()
            else:
                parameter.fill_(0.01)


def _reduced_wan21_config() -> WanVAEConfig:
    return WanVAEConfig(
        dim=2,
        z_dim=2,
        dim_mult=(1, 1, 1),
        num_res_blocks=1,
        temporal_downsample=(True, True),
    )


def _release_cuda_allocations() -> int:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated()


_WAN21_MODELS = Path(os.environ.get("DINKSTER_WAN21_MODELS", str(_FLUX_MODELS)))
_REAL_WAN21_SPLIT = {
    "diffusion": _WAN21_MODELS / "diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
    "t5xxl": _WAN21_MODELS / "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": _WAN21_MODELS / "vae/wan_2.1_vae.safetensors",
}
_WAN21_REVISION = "617a7633e636506f850e043bc4605f290a466a8e"
_WAN21_OFFICIAL_GOLDENS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "goldens" / "wan21_official_goldens.json").read_text()
)
_REAL_WAN21_ARTIFACTS = {
    "diffusion": (
        2_838_303_560,
        "be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
        f"{_WAN21_REVISION}/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
    ),
    "t5xxl": (
        6_735_906_897,
        "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
        f"{_WAN21_REVISION}/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    "vae": (
        253_815_318,
        "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
        f"{_WAN21_REVISION}/split_files/vae/wan_2.1_vae.safetensors",
    ),
}

_REAL_WAN21_CAUSAL = Path(
    os.environ.get("DINKSTER_WAN21_CAUSAL", "F:/models/causal_forcing-framewise.safetensors")
)
_WAN21_CAUSAL_REVISION = "f208499adcca1bbcc91597c7cbbbb74da5abf20c"
_REAL_WAN21_CAUSAL_ARTIFACT = (
    5_676_070_464,
    "408c67a8c6725756f5be2c5cf2d5c584c15dd147f5a0c458be62dcb3efb78477",
    "blake3:2dd24d865c730a39e20bf0f9dbe2a89562708322547f5ce9792ffe7dfce5ce44",
    "https://huggingface.co/Comfy-Org/causal_forcing_framewise_ComfyUI_repackaged/resolve/"
    f"{_WAN21_CAUSAL_REVISION}/split_files/diffusion_models/"
    "causal_forcing-framewise.safetensors",
)
_WAN21_CAUSAL_RUNTIME_IDENTITY = (
    "native:dinkster.wan21:7bbf58d92772d632cd8075c8e76b845fc5f717b4227eb73ac6694d09494b2fac"
)

_WAN_FLOW_RVS_MODELS = Path(os.environ.get("DINKSTER_WAN_FLOW_RVS_MODELS", str(_WAN21_MODELS)))
_REAL_WAN_FLOW_RVS_SPLIT = {
    "diffusion": (_WAN_FLOW_RVS_MODELS / "diffusion_models/wan21_1.3b_flow_rvs_bf16.safetensors"),
    "t5xxl": _WAN21_MODELS / "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": _WAN_FLOW_RVS_MODELS / "vae/wan21_flow_rvs_mask_vae_bf16.safetensors",
}
_WAN_FLOW_RVS_REVISION = "9de69f74aa43cef3c05e37206aff062f4f31e06d"
_REAL_WAN_FLOW_RVS_ARTIFACTS = {
    "diffusion": (
        2_838_077_376,
        "7dd4f7afff4e25ba123f7fa8d581dedbacbc7d9a043eda7285c7e0a35cb3b80f",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/"
        f"{_WAN_FLOW_RVS_REVISION}/FlowRVS/wan21_1.3b_flow_rvs_bf16.safetensors?download=true",
    ),
    "vae": (
        253_795_874,
        "0ba9ca5e9f9929572ae484144a4b49d7dbec92429d98d1d850c4163246a75ae7",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/"
        f"{_WAN_FLOW_RVS_REVISION}/wan21_flow_rvs_mask_vae_bf16.safetensors?download=true",
    ),
}

_REAL_WAN21_UNI3C = Path(
    os.environ.get(
        "DINKSTER_WAN21_UNI3C",
        "F:/models/model_patches/Wan21_Uni3C_controlnet_fp16.safetensors",
    )
)
_WAN21_UNI3C_REVISION = "8260d429d19fd7a72304cad059160b95d843913f"
_REAL_WAN21_UNI3C_ARTIFACT = (
    1_997_314_376,
    "d7dd1bfe5f34dea607a749d72f206908e76a908b9999cb28c53c3eb167ddd709",
    "blake3:9ca3a09e88ad657b4341762ed862fa9eed8f04945132e9a75a4c288658c98ec8",
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/"
    f"{_WAN21_UNI3C_REVISION}/Wan21_Uni3C_controlnet_fp16.safetensors?download=true",
)
_REAL_WAN22_BERNINI_SPLIT = {
    "diffusion": Path(
        os.environ.get(
            "DINKSTER_WAN22_BERNINI",
            "F:/models/diffusion_models/wan2.2_bernini_r_high_noise_fp8_scaled.safetensors",
        )
    ),
    "t5xxl": Path(os.environ.get("DINKSTER_WAN22_BERNINI_TEXT", str(_REAL_WAN21_SPLIT["t5xxl"]))),
    "vae": Path(os.environ.get("DINKSTER_WAN22_BERNINI_VAE", str(_REAL_WAN21_SPLIT["vae"]))),
}
_WAN22_BERNINI_REVISION = "fc371005c90d24177f3658cfacd78b44a41bbd8e"
_REAL_WAN22_BERNINI_ARTIFACT = (
    15_574_833_216,
    "9ff3d7369da98f8eaf71045f7d0e99d5e344eea5e7dc934a930608786fe73f52",
    "https://huggingface.co/Comfy-Org/Bernini-R/resolve/"
    f"{_WAN22_BERNINI_REVISION}/diffusion_models/"
    "wan2.2_bernini_r_high_noise_fp8_scaled.safetensors",
)
_REAL_WAN22_S2V = Path(
    os.environ.get(
        "DINKSTER_WAN22_S2V",
        "F:/models/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors",
    )
)
_WAN22_S2V_REVISION = "c4f60d30c55a624e35427060fdd217579a6c1d77"
_REAL_WAN22_S2V_ARTIFACT = (
    16_394_832_474,
    "140e75af5534ac3d91e710d9df756f7032addd64b341ba2c1c70e3e6da9aa216",
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/"
    f"{_WAN22_S2V_REVISION}/split_files/diffusion_models/"
    "wan2.2_s2v_14B_fp8_scaled.safetensors",
)
_WAN22_DANCER_REVISION = "e3a8d63a176f8e4709000856453f508cfe39c2a0"
_REAL_WAN22_DANCER = {
    branch: Path(
        os.environ.get(
            f"DINKSTER_WAN22_DANCER_{branch.upper()}",
            f"F:/models/diffusion_models/wan2.2_dancer_14b_{branch}_fp8_scaled.safetensors",
        )
    )
    for branch in ("local", "global")
}
_REAL_WAN22_DANCER_ARTIFACTS = {
    "local": (
        18_342_596_856,
        "744b93871a4ca8d9645790dd9a52fe8b2e6876557aac94c930906270d3d9bd60",
        "https://huggingface.co/Comfy-Org/Wan-Dancer/resolve/"
        f"{_WAN22_DANCER_REVISION}/diffusion_models/"
        "wan2.2_dancer_14b_local_fp8_scaled.safetensors",
    ),
    "global": (
        18_342_596_856,
        "e6590802f488979209cb416dee711e55d44f1e651d4dbf7e92b2358fe74a2143",
        "https://huggingface.co/Comfy-Org/Wan-Dancer/resolve/"
        f"{_WAN22_DANCER_REVISION}/diffusion_models/"
        "wan2.2_dancer_14b_global_fp8_scaled.safetensors",
    ),
}
_HUMO_REVISION = "2e746dc158c41696fd168accc7a3f19a6593fed6"
_REAL_WAN21_HUMO = Path(
    os.environ.get(
        "DINKSTER_WAN21_HUMO",
        "F:/models/diffusion_models/humo_17B_fp8_e4m3fn.safetensors",
    )
)
_REAL_WAN21_HUMO_ARTIFACT = (
    17_058_372_152,
    "222ddeac4dea6b78363cb5be78c47660c92963a69386026cd6dc0de4d3094f66",
    "blake3:5e8f77743388d67e21243877097edfa9878eafd0a1a4e8dde0f7bd1277a14e85",
    "https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/"
    f"{_HUMO_REVISION}/split_files/diffusion_models/humo_17B_fp8_e4m3fn.safetensors",
)
_WAN21_HUMO_RUNTIME_IDENTITY = (
    "native:dinkster.wan21:5ab182fd6336de10a1e7d80d3fb36dfad425c61a1cadd8038e03dda6fc13b64c"
)
_REAL_WHISPER_LARGE_V3 = Path(
    os.environ.get(
        "DINKSTER_WHISPER_LARGE_V3",
        "F:/models/audio_encoders/whisper_large_v3_fp16.safetensors",
    )
)
_REAL_WHISPER_LARGE_V3_ARTIFACT = (
    3_087_130_976,
    "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
    "blake3:d89cb4436c15d5671ed0faa8950ca256e15c1a2a16246379742d5fb7ab7a6d07",
    "https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/"
    f"{_HUMO_REVISION}/split_files/audio_encoders/whisper_large_v3_fp16.safetensors",
)
_WHISPER_LARGE_V3_RUNTIME_IDENTITY = (
    "native:dinkster.whisper-large-v3:"
    "971b34d3b5ab09b001f9516dc3116e90be55872ec491bf2586388d612fd0c025"
)
_REAL_WAV2VEC2_CHINESE_BASE = Path(
    os.environ.get(
        "DINKSTER_WAV2VEC2_CHINESE_BASE",
        "F:/models/audio_encoders/wav2vec2-chinese-base_fp16.safetensors",
    )
)
_WAV2VEC2_CHINESE_BASE_REVISION = "87847d3bc53702afda44078249e7c33e867827c4"
_REAL_WAV2VEC2_CHINESE_BASE_ARTIFACT = (
    190_115_368,
    "000813e441020f18cff844c969d2d5d4adc2a5ce46b2db1f23950b05d88805b4",
    "blake3:6b3eed6f786174a6a5faf51e0e314e0e6934636dd05e52f59638db2247d3d005",
    "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/"
    f"{_WAV2VEC2_CHINESE_BASE_REVISION}/wav2vec2-chinese-base_fp16.safetensors",
)
_WAV2VEC2_CHINESE_BASE_RUNTIME_IDENTITY = (
    "native:dinkster.wav2vec2:68aea51c07e05cb5e83aec58a1c67167238e2a11223320cceb93c2e8c6673314"
)
_REAL_WAN21_I2V = Path(
    os.environ.get(
        "DINKSTER_WAN21_I2V",
        "F:/workspaces/station1/stations/station14/installs/wan21-i2v-artifacts/"
        "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors",
    )
)
_REAL_WAN21_I2V_ARTIFACT = (
    16_401_356_938,
    "b2de21b99b2e72cb0ff15253b07e926f26e7cf1b7e229efc32f94ad1f1ed9395",
    "blake3:b9bc11ce11e84b35988bacb9d754afdc67fc47082d453771cf33fb7d9f1ef318",
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
    f"{_WAN21_REVISION}/split_files/diffusion_models/"
    "wan2.1_i2v_480p_14B_fp8_scaled.safetensors",
)
_REAL_WAN21_I2V_CLIP = Path(
    os.environ.get(
        "DINKSTER_WAN21_I2V_CLIP",
        "F:/workspaces/station1/stations/station14/installs/wan21-i2v-artifacts/"
        "split_files/clip_vision/clip_vision_h.safetensors",
    )
)
_REAL_WAN21_I2V_CLIP_ARTIFACT = (
    1_264_219_396,
    "64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161",
    "blake3:de4037bc9d3aed3fcc081fd95ec398c04d54b06bc79a6ae51fa9bc574bf6d63b",
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
    f"{_WAN21_REVISION}/split_files/clip_vision/clip_vision_h.safetensors",
)
_REAL_WAN21_MULTITALK = Path(
    os.environ.get(
        "DINKSTER_WAN21_MULTITALK",
        "F:/models/model_patches/wan2.1_infiniteTalk_multi_fp16.safetensors",
    )
)
_REAL_WAN21_MULTITALK_ARTIFACT = (
    5_124_439_112,
    "4c2486cdfb6ff9a9f27408e98e11e20619136933b20411e0c365b1e84075d195",
    "blake3:4d65fad48e3beb4c6baecbe8e44e451814d050a4294b0ff53464c2925d86f56b",
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
    f"{_WAN21_REVISION}/split_files/model_patches/"
    "wan2.1_infiniteTalk_multi_fp16.safetensors",
)
_WAN21_MULTITALK_RESOURCE_IDENTITY = (
    "c9c3e4f5cde6dbe7857975f2040970da3c513799f3ca78dfe5138c1763f1266a"
)


def _wan21_official_tensor(section: str, name: str) -> torch.Tensor:
    payload = _WAN21_OFFICIAL_GOLDENS[section][name]
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


_WAN21_OFFICIAL_ATOLS = {
    # All four outputs are bit-exact against the golden: the reference and this
    # validation runtime share the same host, GPU, and torch 2.13 kernels, and
    # the bf16 VAE default matches ComfyUI's model-management selection.
    "sampled": 0.0,
    "encoded": 0.0,
    "decoded": 0.0,
    "pipeline_decoded": 0.0,
}


@pytest.mark.skipif(
    not _REAL_WAN21_CAUSAL.exists(),
    reason="official Wan 2.1 CausalAR artifact not present (set DINKSTER_WAN21_CAUSAL)",
)
def test_real_wan21_causalar_loads_and_matches_first_block_on_cuda() -> None:
    from dinkster_assets import AssetRef
    from dinkster_inference import WAN21_CAUSAL_AR_1_3B
    from dinkster_inference_torch import Wan21CausalModel, load_wan21_component
    from dinkster_inference_torch.wan21_model import Wan21Model

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return _REAL_WAN21_CAUSAL if digest == asset_digest else None

    baseline = _release_cuda_allocations()
    expected_size, expected_sha256, asset_digest, source_url = _REAL_WAN21_CAUSAL_ARTIFACT
    assert source_url.startswith(
        "https://huggingface.co/Comfy-Org/causal_forcing_framewise_ComfyUI_repackaged/"
        f"resolve/{_WAN21_CAUSAL_REVISION}/"
    )
    assert _REAL_WAN21_CAUSAL.stat().st_size == expected_size
    with _REAL_WAN21_CAUSAL.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256

    asset = AssetRef(
        asset_digest,
        _REAL_WAN21_CAUSAL.name,
        expected_size,
        resolver=Resolver(),
    )
    loaded = load_wan21_component(
        _REAL_WAN21_CAUSAL,
        asset=asset,
        expected_role="diffusion",
        expected_identity=_WAN21_CAUSAL_RUNTIME_IDENTITY,
        compute_dtype=torch.bfloat16,
    )
    assert loaded.runtime_identity == _WAN21_CAUSAL_RUNTIME_IDENTITY
    model = loaded.module
    assert isinstance(model, Wan21CausalModel)
    assert model.config is WAN21_CAUSAL_AR_1_3B

    mechanism = enroll_component(model, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    latent = torch.linspace(-0.25, 0.25, 64, device="cuda:0", dtype=torch.bfloat16).reshape(
        1, 16, 1, 2, 2
    )
    timestep = torch.tensor([500.0], device="cuda:0")
    context = torch.linspace(
        -0.5,
        0.5,
        2 * 4096,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 2, 4096)
    caches = model.create_caches(
        batch_size=1,
        max_tokens=1,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        expected = Wan21Model.forward(model, latent, timestep, context)
        actual = model.forward_block(latent, timestep, context, time_start=0, caches=caches)
    assert torch.equal(actual, expected)
    assert actual.shape == latent.shape and bool(torch.isfinite(actual).all())
    assert all(cache.end == 1 for cache in caches.self_attention)
    assert all(
        cache.key is not None and cache.value is not None for cache in caches.cross_attention
    )

    mechanism.unload()
    del actual, caches, context, expected, latent, loaded, mechanism, model
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_WAN21_SPLIT.values()),
    reason="official Wan 2.1 split artifacts not present (set DINKSTER_WAN21_MODELS)",
)
def test_real_wan21_comfyui_parity_and_residency_on_cuda() -> None:
    from dinkster_inference import (
        MultiStreamLatent,
        SamplingGuidance,
        load_safetensors_header,
        probe_native,
    )
    from dinkster_inference_torch import (
        Umt5SentencePieceTokenizer,
        Wan21Runtime,
        load_runtime,
    )

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    reference = _WAN21_OFFICIAL_GOLDENS["reference"]
    # Bit-exact torch.equal parity against a host-pinned golden only holds on
    # the generator GPU; other hosts drift ~1e-6 relative in float32 UMT5
    # embeddings (same cross-host drift class as Dinkster #636).
    live_device = torch.cuda.get_device_name()
    assert reference["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert reference["comfy_kitchen"] == "v0.2.31"
    assert reference["comfy_aimdo"] == "v0.4.13"
    assert reference["diffusion_dtype"] == "bfloat16"
    assert reference["text_dtype"] == "float32"
    assert reference["vae_dtype"] == "bfloat16"
    for name, path in _REAL_WAN21_SPLIT.items():
        expected_size, expected_digest, source_url = _REAL_WAN21_ARTIFACTS[name]
        receipt_name = "text_encoder" if name == "t5xxl" else name
        receipt = _WAN21_OFFICIAL_GOLDENS["artifacts"][receipt_name]
        assert receipt == {
            "url": source_url,
            "byte_size": expected_size,
            "sha256": expected_digest,
        }
        assert source_url.startswith("https://huggingface.co/")
        assert path.stat().st_size == expected_size
        with path.open("rb") as artifact:
            assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_digest

    sources = {name: load_safetensors_header(path) for name, path in _REAL_WAN21_SPLIT.items()}
    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["t5xxl"],
        vae=sources["vae"],
    )
    assert capability.native and capability.family_id == "dinkster.wan21"
    runtime = load_runtime(
        diffusion=sources["diffusion"],
        t5xxl=sources["t5xxl"],
        vae=sources["vae"],
        fp8_matmul=False,
    )
    assert isinstance(runtime, Wan21Runtime)
    assert runtime.assembled.compute_dtype("diffusion") is torch.bfloat16
    assert runtime.assembled.compute_dtype("umt5xxl") is torch.float32
    assert runtime.assembled.compute_dtype("vae") is torch.bfloat16

    conditioning = _WAN21_OFFICIAL_GOLDENS["conditioning"]
    tokenizer = Umt5SentencePieceTokenizer(runtime.assembled.tokenizer_model)
    prompt_ids = [*tokenizer.encode(conditioning["prompt"]), 1]
    negative_ids = [*tokenizer.encode(conditioning["negative_prompt"]), 1]
    prompt_ids.extend([0] * (512 - len(prompt_ids)))
    negative_ids.extend([0] * (512 - len(negative_ids)))
    assert prompt_ids == conditioning["prompt_token_ids"]
    assert negative_ids == conditioning["negative_token_ids"]

    runtime.assembled.umt5xxl.to("cuda:0")
    with torch.no_grad():
        positive = runtime.encode_text(conditioning["prompt"])
        negative = runtime.encode_text(conditioning["negative_prompt"])
    assert positive.embeddings.shape == negative.embeddings.shape == (1, 512, 4096)
    assert positive.pooled is None and negative.pooled is None
    assert bool(torch.isfinite(positive.embeddings).all())
    assert bool(torch.isfinite(negative.embeddings).all())
    if live_device != reference["device"]:
        runtime.assembled.umt5xxl.to("cpu")
        del negative, positive, runtime
        assert _release_cuda_allocations() <= baseline + 64 * 1024**2
        pytest.skip(
            f"scaled-FP8 UMT5 CUDA execution passed on {live_device!r}; "
            f"bit-exact parity requires generator host {reference['host']} "
            f"({reference['device']}, driver {reference['nvidia_driver']})"
        )
    positive_active = _wan21_official_tensor("conditioning", "positive_active")
    negative_active = _wan21_official_tensor("conditioning", "negative_active")
    assert torch.equal(positive.embeddings[:, : positive_active.shape[1]].cpu(), positive_active)
    assert torch.equal(negative.embeddings[:, : negative_active.shape[1]].cpu(), negative_active)
    assert not torch.count_nonzero(positive.embeddings[:, positive_active.shape[1] :])
    assert not torch.count_nonzero(negative.embeddings[:, negative_active.shape[1] :])
    runtime.assembled.umt5xxl.to("cpu")
    soft_empty_cache(torch.device("cuda:0"))

    diffusion = runtime.assembled.diffusion
    mechanism = enroll_component(diffusion, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    assert mechanism.loaded_bytes() == mechanism.total_bytes()
    pipeline = _WAN21_OFFICIAL_GOLDENS["pipeline"]
    latent = _wan21_official_tensor("pipeline", "initial_latent")
    assert torch.equal(
        prepare_noise(latent, pipeline["seed"]),
        _wan21_official_tensor("pipeline", "noise"),
    )
    assert pipeline["sigmas"] == [1.0, 0.0]
    with torch.no_grad():
        sampled_streams = runtime.sample_multistream(
            MultiStreamLatent.from_pairs((("video", latent),)),
            conditioning=runtime.prepare_text_conditioning(positive),
            cfg=SamplingGuidance(runtime.prepare_text_conditioning(negative), pipeline["cfg"]),
            sampler_id=pipeline["sampler"],
            scheduler_id=pipeline["scheduler"],
            steps=pipeline["steps"],
            denoise=pipeline["denoise"],
            seed=pipeline["seed"],
            device="cuda:0",
        )
        sampled = sampled_streams.by_role("video")
    assert sampled.shape == latent.shape
    assert sampled.device.type == "cuda" and bool(torch.isfinite(sampled).all())

    mechanism.partially_unload(max(1, mechanism.total_bytes() // 3))
    assert 0 < mechanism.loaded_bytes() < mechanism.total_bytes()
    with torch.no_grad():
        partially_offloaded = diffusion(
            torch.zeros((1, 16, 1, 2, 2), device="cuda:0", dtype=torch.bfloat16),
            torch.tensor([500.0], device="cuda:0"),
            positive.embeddings.to(device="cuda:0", dtype=torch.bfloat16),
        )
    assert partially_offloaded.shape == latent.shape
    assert bool(torch.isfinite(partially_offloaded).all())
    mechanism.unload()

    vae = runtime.assembled.vae.to("cuda:0")
    content = _wan21_official_tensor("causal_vae", "content").to("cuda:0")
    with torch.no_grad():
        encoded = runtime.encode_content(content)
        decoded = runtime.decode_latent(encoded)
        pipeline_decoded = runtime.decode_latent(sampled)
    assert encoded.shape == (1, 16, 2, 2, 2)
    assert decoded.shape == content.shape and bool(torch.isfinite(decoded).all())
    assert pipeline_decoded.shape == (1, 3, 1, 16, 16)

    official_outputs = {
        "sampled": _wan21_official_tensor("pipeline", "sampled_latent"),
        "encoded": _wan21_official_tensor("causal_vae", "encoded"),
        "decoded": _wan21_official_tensor("causal_vae", "decoded"),
        "pipeline_decoded": _wan21_official_tensor("pipeline", "decoded_content"),
    }
    observed_outputs = {
        "sampled": sampled.float().cpu(),
        "encoded": encoded.float().cpu(),
        "decoded": decoded.float().cpu(),
        "pipeline_decoded": pipeline_decoded.float().cpu(),
    }
    for name, expected in official_outputs.items():
        torch.testing.assert_close(
            observed_outputs[name],
            expected,
            rtol=0,
            atol=_WAN21_OFFICIAL_ATOLS[name],
        )
    vae.to("cpu")

    assert torch.cuda.max_memory_allocated() - baseline < 12 * 1024**3
    del content, decoded, encoded, latent, mechanism, negative, partially_offloaded
    del pipeline_decoded
    del positive, runtime, sampled, vae, diffusion, observed_outputs, official_outputs
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


_TRIPOSPLAT_REVISION = "56a96e603204ec410c4da60c13ea4fa09a2169a9"
_TRIPOSPLAT_RELATIVE_PATHS = {
    "diffusion": "diffusion_models/triposplat_fp16.safetensors",
    "vision": "clip_vision/dino_v3_vit_h.safetensors",
    "reference_vae": "vae/flux2-vae.safetensors",
    "gaussian_decoder": "vae/triposplat_vae_decoder_fp16.safetensors",
}
_TRIPOSPLAT_MODEL_CANDIDATES = (
    *(
        (Path(configured).expanduser(),)
        if (configured := os.environ.get("DINKSTER_TRIPOSPLAT_MODELS"))
        else ()
    ),
    Path("/home/kosin/model-artifacts/dinkster-triposplat-e2e"),
    Path("/home/kosin/ComfyUI-Shared/models"),
    Path("/mnt/data/comfy-models"),
)
_TRIPOSPLAT_MODELS = next(
    (
        root
        for root in _TRIPOSPLAT_MODEL_CANDIDATES
        if all((root / relative).is_file() for relative in _TRIPOSPLAT_RELATIVE_PATHS.values())
    ),
    _TRIPOSPLAT_MODEL_CANDIDATES[0],
)
_REAL_TRIPOSPLAT_SPLIT = {
    name: _TRIPOSPLAT_MODELS / relative for name, relative in _TRIPOSPLAT_RELATIVE_PATHS.items()
}
_REAL_TRIPOSPLAT_ARTIFACTS = {
    "diffusion": (
        741_106_994,
        "c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20",
        "blake3:2798db16111877461c45ee6e6d9048e55bff75be819ad5e6048bab649b7d81d5",
        "https://huggingface.co/VAST-AI/TripoSplat/resolve/"
        f"{_TRIPOSPLAT_REVISION}/diffusion_models/triposplat_fp16.safetensors",
    ),
    "vision": (
        1_681_247_696,
        "a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53",
        "blake3:d299baef1fa96df4a61f0b414ba3e83e0ac373be2262f1645c20d186a1b120a3",
        "https://huggingface.co/VAST-AI/TripoSplat/resolve/"
        f"{_TRIPOSPLAT_REVISION}/clip_vision/dino_v3_vit_h.safetensors",
    ),
    "reference_vae": (
        336_213_556,
        "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
        "blake3:fcb1d172993424c66d325d139863ccbaadf64a920073b2d005d73a31fa5a851d",
        "https://huggingface.co/VAST-AI/TripoSplat/resolve/"
        f"{_TRIPOSPLAT_REVISION}/vae/flux2-vae.safetensors",
    ),
    "gaussian_decoder": (
        576_148_442,
        "ed0d0c3d43b599e326845d0ec70f3cf77be9a55e2d97627ac3b34d2830763cc8",
        "blake3:343ed688a3dfe4b5bfd2225cc2aae0227c6bbfa524d300b021530e5a3febd478",
        "https://huggingface.co/VAST-AI/TripoSplat/resolve/"
        f"{_TRIPOSPLAT_REVISION}/vae/triposplat_vae_decoder_fp16.safetensors",
    ),
}
_TRIPOSPLAT_OFFICIAL_GOLDENS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "goldens" / "triposplat_official_goldens.json").read_text()
)


def _triposplat_official_tensor(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


def _triposplat_summary_max_abs(
    actual: torch.Tensor, payload: dict[str, Any]
) -> tuple[float, float]:
    actual = actual.detach().float().cpu().contiguous()
    assert list(actual.shape) == payload["shape"]
    flat = actual.flatten()
    stride = payload["slice"]["stride"]
    expected_slice = torch.tensor(payload["slice"]["data"], dtype=torch.float32)
    actual_slice = flat[::stride][: expected_slice.numel()]
    observed_moments = torch.tensor(
        [flat.mean(), flat.std(unbiased=False), flat.min(), flat.max()], dtype=torch.float32
    )
    expected_moments = torch.tensor(
        [payload["moments"][name] for name in ("mean", "std", "min", "max")],
        dtype=torch.float32,
    )
    return (
        (actual_slice - expected_slice).abs().max().item(),
        (observed_moments - expected_moments).abs().max().item(),
    )


# Every tensor was bit-exact against the pinned torch 2.10 reference run, so
# the limits intentionally leave no room for a loading or numerical drift.
_TRIPOSPLAT_OFFICIAL_ATOLS = {
    "preprocessed_slice": 0.0,
    "preprocessed_moments": 0.0,
    "features_slice": 0.0,
    "features_moments": 0.0,
    "reference_latent_slice": 0.0,
    "reference_latent_moments": 0.0,
    "sampled_latent": 0.0,
    "sampled_camera": 0.0,
    "positions_first_64": 0.0,
    "positions_moments": 0.0,
    "scales_first_64": 0.0,
    "scales_moments": 0.0,
    "rotations_first_64": 0.0,
    "rotations_moments": 0.0,
    "opacities_first_64": 0.0,
    "opacities_moments": 0.0,
    "sh_first_64": 0.0,
    "sh_moments": 0.0,
}


@pytest.mark.skipif(
    not all(path.is_file() for path in _REAL_TRIPOSPLAT_SPLIT.values()),
    reason="official TripoSplat artifacts not present (set DINKSTER_TRIPOSPLAT_MODELS)",
)
def test_real_triposplat_comfyui_parity_on_cuda() -> None:
    from dinkster_assets import AssetRef
    from dinkster_inference import (
        BFLOAT16,
        FLOAT32,
        ConditioningCarrier,
        SamplingGuidance,
        TripoSplatComponentRole,
        flux2_component_runtime_identity,
        load_safetensors_header,
        plan_flux2_split_component,
        plan_triposplat_split_component,
        split_component_conditioning,
    )
    from dinkster_inference_torch import (
        TripoSplatDiffusionRuntime,
        load_flux2_component,
        load_triposplat_component,
    )
    from dinkster_model_triposplat import provider as triposplat_provider

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return paths_by_digest.get(digest)

    class ComponentHandle:
        def __init__(
            self,
            module: torch.nn.Module,
            mechanism: Any,
            resource_identity: str,
        ) -> None:
            self.module = module
            self.mechanism = mechanism
            self.resource_identity = resource_identity
            self.load_device = torch.device("cuda:0")
            self._staged = False

        @property
        def component(self) -> object:
            if not self._staged:
                raise RuntimeError("component read outside its residency lease")
            return self.module

        def require_active(self) -> None:
            pass

        @contextmanager
        def stage(self) -> Generator[None, None, None]:
            self.mechanism.partially_load(None)
            self._staged = True
            try:
                yield
            finally:
                self._staged = False

        @contextmanager
        def stage_with(self, _runtime_handle: object, _role: str) -> Generator[None, None, None]:
            with self.stage():
                yield

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    reference = _TRIPOSPLAT_OFFICIAL_GOLDENS["reference"]
    live_device = torch.cuda.get_device_name()
    if live_device != reference["device"]:
        pytest.skip(
            f"golden generated on host {reference['host']} ({reference['device']}, "
            f"driver {reference['nvidia_driver']}); live GPU is {live_device!r}"
        )
    if torch.__version__ != reference["torch"]:
        pytest.skip(
            f"bit-exact golden uses torch {reference['torch']}; live torch is {torch.__version__}"
        )
    assert reference["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert reference["comfy_kitchen"] == "v0.2.31"
    assert reference["comfy_aimdo"] == "v0.4.13"
    assert reference["attention"] == "pytorch SDPA"
    assert reference["diffusion_dtype"] == "bfloat16"
    assert reference["vision_dtype"] == "float32"
    assert reference["reference_vae_dtype"] == "float32"
    assert reference["gaussian_decoder_dtype"] == "float32"

    paths_by_digest: dict[str, Path] = {}
    assets: dict[str, AssetRef] = {}
    for name, path in _REAL_TRIPOSPLAT_SPLIT.items():
        expected_size, expected_sha256, asset_digest, source_url = _REAL_TRIPOSPLAT_ARTIFACTS[name]
        assert _TRIPOSPLAT_OFFICIAL_GOLDENS["artifacts"][name] == {
            "url": source_url,
            "byte_size": expected_size,
            "sha256": expected_sha256,
        }
        assert source_url.startswith(
            f"https://huggingface.co/VAST-AI/TripoSplat/resolve/{_TRIPOSPLAT_REVISION}/"
        )
        assert path.stat().st_size == expected_size
        with path.open("rb") as artifact:
            assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256
        paths_by_digest[asset_digest] = path
        assets[name] = AssetRef(asset_digest, path.name, expected_size, resolver=Resolver())

    # These bodies bind the artifact digest, strict loading recipe, role, and
    # dtype. A body change means the loading recipe changed and needs review.
    triposplat_identities = {
        "dit": (
            "native:dinkster.triposplat:"
            "e8098f1092608fc73b0bfa73c6fde79b45158cb14dbd49da455ced7da8a247fe"
        ),
        "dinov3-vision-conditioner": (
            "native:dinkster.triposplat:"
            "90d9652e2f4f669a3dd61912984a6ffd6ebac24d9ca69a64a43de7a3b14d497a"
        ),
        "gaussian-decoder": (
            "native:dinkster.triposplat:"
            "8d6cbe4ea94aeab364b51a65c3251facfc63e96616651d740c85a2b88aa6b14c"
        ),
    }
    loaded_components: dict[str, Any] = {}
    for role, artifact_name, dtype, identity in (
        ("dit", "diffusion", torch.bfloat16, triposplat_identities["dit"]),
        (
            "dinov3-vision-conditioner",
            "vision",
            torch.float32,
            triposplat_identities["dinov3-vision-conditioner"],
        ),
        (
            "gaussian-decoder",
            "gaussian_decoder",
            torch.float32,
            triposplat_identities["gaussian-decoder"],
        ),
    ):
        path = _REAL_TRIPOSPLAT_SPLIT[artifact_name]
        asset = assets[artifact_name]
        role_value = cast("TripoSplatComponentRole", role)
        source = load_safetensors_header(path, asset_digest=asset.digest, asset_size=asset.size)
        planned = plan_triposplat_split_component(source, role=role_value, path=path)
        loaded = load_triposplat_component(
            path,
            asset=asset,
            expected_role=role_value,
            expected_identity=identity,
            compute_dtype=dtype,
        )
        assert loaded.plan == planned.plan
        assert loaded.runtime_identity == identity
        loaded_components[role] = loaded

    flux_path = _REAL_TRIPOSPLAT_SPLIT["reference_vae"]
    flux_asset = assets["reference_vae"]
    flux_source = load_safetensors_header(
        flux_path, asset_digest=flux_asset.digest, asset_size=flux_asset.size
    )
    flux_plan = plan_flux2_split_component(flux_source, role="vae", path=flux_path)
    flux_identity = flux2_component_runtime_identity(flux_plan, FLOAT32)
    assert flux_identity == (
        "native:dinkster.flux2:9a6d12bbcdccc815225e3cba179351f2f79ecd436b4e4035aaa1c1f43de53753"
    )
    loaded_flux = load_flux2_component(
        flux_path,
        asset=flux_asset,
        expected_role="vae",
        expected_identity=flux_identity,
        compute_dtype=torch.float32,
    )
    assert loaded_flux.plan == flux_plan.plan
    assert loaded_flux.runtime_identity == flux_identity

    assert BFLOAT16.name == "bfloat16" and FLOAT32.name == "float32"
    input_config = _TRIPOSPLAT_OFFICIAL_GOLDENS["input"]
    generator = torch.Generator(device="cpu").manual_seed(input_config["image_seed"])
    height, width = input_config["image_height"], input_config["image_width"]
    y = torch.linspace(0.0, 1.0, height).view(height, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, width)
    image = torch.stack(
        (
            x.expand(height, -1),
            y.expand(-1, width),
            (x + y).expand(height, width) * 0.5,
        ),
        dim=-1,
    )
    image = (image + torch.rand(image.shape, generator=generator) * 0.01).clamp(0.0, 1.0)
    yy = torch.arange(height).view(height, 1) - (height - 1) / 2
    xx = torch.arange(width).view(1, width) - (width - 1) / 2
    radius = min(height, width) * 0.32
    mask = ((xx.square() + yy.square()) <= radius**2).to(torch.float32)
    prepared_result = triposplat_provider.execute_triposplat_preprocess_image(
        image=image.unsqueeze(0),
        mask=mask.unsqueeze(0),
        erode_radius=input_config["erode_radius"],
        size=input_config["preprocess_size"],
    )
    prepared = cast("torch.Tensor", prepared_result["image"])
    conditioning_golden = _TRIPOSPLAT_OFFICIAL_GOLDENS["conditioning"]
    preprocessed_slice, preprocessed_moments = _triposplat_summary_max_abs(
        prepared, conditioning_golden["preprocessed"]
    )

    vision_module = loaded_components["dinov3-vision-conditioner"].module
    flux_module = loaded_flux.module
    vision_mechanism = enroll_component(vision_module, load_device="cuda:0", offload_device="cpu")
    flux_mechanism = enroll_component(flux_module, load_device="cuda:0", offload_device="cpu")
    vision_handle = ComponentHandle(
        vision_module,
        vision_mechanism,
        triposplat_identities["dinov3-vision-conditioner"],
    )
    flux_handle = ComponentHandle(flux_module, flux_mechanism, flux_identity)
    conditioning_result = triposplat_provider.execute_triposplat_conditioning(
        vision=vision_handle,
        vae=flux_handle,
        image=prepared,
    )
    positive_bound = cast("ConditioningCarrier", conditioning_result["positive"])
    negative_bound = cast("ConditioningCarrier", conditioning_result["negative"])
    positive_carrier, _ = split_component_conditioning(positive_bound)
    negative_carrier, _ = split_component_conditioning(negative_bound)

    diffusion_module = loaded_components["dit"].module
    runtime = TripoSplatDiffusionRuntime(
        diffusion_module,
        runtime_identity=triposplat_identities["dit"],
        compute_dtype=torch.bfloat16,
    )
    positive = runtime.prepare_conditioning(positive_carrier)
    negative = runtime.prepare_conditioning(negative_carrier)
    features_slice, features_moments = _triposplat_summary_max_abs(
        positive.features, conditioning_golden["features"]
    )
    assert positive.reference_latent is not None
    reference_slice, reference_moments = _triposplat_summary_max_abs(
        positive.reference_latent, conditioning_golden["reference_latent"]
    )
    assert torch.equal(negative.features, torch.zeros_like(positive.features))
    assert negative.reference_latent is not None
    assert torch.equal(negative.reference_latent, torch.zeros_like(positive.reference_latent))

    for mechanism in (vision_mechanism, flux_mechanism):
        mechanism.partially_unload(max(1, mechanism.total_bytes() // 3))
        assert 0 < mechanism.loaded_bytes() < mechanism.total_bytes()
        mechanism.unload()
    soft_empty_cache(torch.device("cuda:0"))

    latent_mapping = cast("dict[str, object]", conditioning_result["latent"])
    latent = cast("Any", latent_mapping["samples"])
    diffusion_mechanism = enroll_component(
        diffusion_module, load_device="cuda:0", offload_device="cpu"
    )
    diffusion_mechanism.partially_load(None)
    pipeline = _TRIPOSPLAT_OFFICIAL_GOLDENS["pipeline"]
    with torch.no_grad():
        sampled_streams = runtime.sample_multistream(
            latent,
            conditioning=positive,
            cfg=SamplingGuidance(negative, pipeline["cfg"]),
            sampler_id=pipeline["sampler"],
            scheduler_id=pipeline["scheduler"],
            steps=pipeline["steps"],
            denoise=pipeline["denoise"],
            seed=pipeline["seed"],
        )
    sampled_latent = sampled_streams.by_role("latent").float().cpu()
    sampled_camera = sampled_streams.by_role("camera").float().cpu()
    expected_latent = _triposplat_official_tensor(pipeline["sampled_latent"])
    expected_camera = _triposplat_official_tensor(pipeline["sampled_camera"])
    sampled_latent_max = (sampled_latent - expected_latent).abs().max().item()
    sampled_camera_max = (sampled_camera - expected_camera).abs().max().item()
    diffusion_mechanism.partially_unload(max(1, diffusion_mechanism.total_bytes() // 3))
    assert 0 < diffusion_mechanism.loaded_bytes() < diffusion_mechanism.total_bytes()
    diffusion_mechanism.unload()
    soft_empty_cache(torch.device("cuda:0"))

    decoder_module = loaded_components["gaussian-decoder"].module
    decoder_mechanism = enroll_component(decoder_module, load_device="cuda:0", offload_device="cpu")
    decoder_handle = ComponentHandle(
        decoder_module,
        decoder_mechanism,
        triposplat_identities["gaussian-decoder"],
    )
    sampled_cpu = type(sampled_streams).from_pairs(
        (("latent", sampled_latent), ("camera", sampled_camera))
    )
    decode_config = _TRIPOSPLAT_OFFICIAL_GOLDENS["decode"]
    decoded_result = triposplat_provider.execute_triposplat_decode(
        samples={"samples": sampled_cpu},
        decoder=decoder_handle,
        num_gaussians=decode_config["num_gaussians"],
        seed=decode_config["seed"],
    )
    decoded = cast("dict[str, torch.Tensor]", decoded_result["splat"])
    observed: dict[str, float] = {
        "preprocessed_slice": preprocessed_slice,
        "preprocessed_moments": preprocessed_moments,
        "features_slice": features_slice,
        "features_moments": features_moments,
        "reference_latent_slice": reference_slice,
        "reference_latent_moments": reference_moments,
        "sampled_latent": sampled_latent_max,
        "sampled_camera": sampled_camera_max,
    }
    for name, actual in decoded.items():
        payload = decode_config["tensors"][name]
        expected_first = _triposplat_official_tensor(payload["first_64"])
        observed[f"{name}_first_64"] = (
            (actual[:, :64].float().cpu() - expected_first).abs().max().item()
        )
        flat = actual.float().cpu().flatten()
        actual_moments = torch.tensor(
            [flat.mean(), flat.std(unbiased=False), flat.min(), flat.max()], dtype=torch.float32
        )
        expected_moments = torch.tensor(
            [payload["moments"][key] for key in ("mean", "std", "min", "max")],
            dtype=torch.float32,
        )
        observed[f"{name}_moments"] = (actual_moments - expected_moments).abs().max().item()
    for name, difference in observed.items():
        assert difference <= _TRIPOSPLAT_OFFICIAL_ATOLS[name], observed

    decoder_mechanism.partially_unload(max(1, decoder_mechanism.total_bytes() // 3))
    assert 0 < decoder_mechanism.loaded_bytes() < decoder_mechanism.total_bytes()
    decoder_mechanism.unload()
    soft_empty_cache(torch.device("cuda:0"))
    assert torch.cuda.max_memory_allocated() - baseline < 12 * 1024**3

    del decoded, decoded_result, decoder_handle, decoder_mechanism, decoder_module
    del diffusion_mechanism, diffusion_module, expected_camera, expected_latent, flux_handle
    del flux_mechanism, flux_module, image, latent, loaded_components, loaded_flux, mask
    del negative, negative_bound, negative_carrier, positive, positive_bound, positive_carrier
    del prepared, prepared_result, runtime, sampled_camera, sampled_cpu, sampled_latent
    del sampled_streams, vision_handle, vision_mechanism, vision_module
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


_WAN21_UMT5_GGUF = _WAN21_MODELS / "text_encoders/umt5-xxl-encoder-Q8_0.gguf"
_WAN21_UMT5_GGUF_ARTIFACT = (
    6_043_068_256,
    "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc",
    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/"
    "b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf",
)
_WAN21_UMT5_GGUF_FACTS = (
    "gguf.artifact.file_sha256=2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc",
    "gguf.artifact.manifest_sha256=d75380e32dbcb48a57c967edf02fb408c8b3444d0f3ca1f690099e62b1abe62f",
    "gguf.artifact.mapper_id=dinkster.gguf.text.v1",
    "gguf.artifact.architecture=t5encoder",
    "gguf.artifact.family_id=dinkster.text.umt5xxl",
    "gguf.artifact.component=umt5xxl",
    # The city96 encoder is all-Q8_0, so the auto default resolves to the
    # balanced route.
    "gguf.route.provider_key=dinkster-gguf-torch-onuse",
    "gguf.route.implementation_version=v1",
    "gguf.route.kind=cached-decode",
    "gguf.route.device_kind=any",
    "gguf.route.device_capability=generic",
    "gguf.route.compute_dtype=float32",
    "gguf.route.accumulation_dtype=float32",
    "gguf.route.decoded_cache=auto",
)
# Extended identity of the balanced route for the city96 UMT5-XXL Q8_0
# artifact.
_WAN21_UMT5_GGUF_BALANCED_IDENTITY = (
    "native:dinkster.wan21:56ad5f271b2f59ff555a42fb22fd7768d26d5c15a60560f541a0283b12f9399a"
)
# The GGUF is city96's Q8_0 requantization and the goldens come from the
# official fp8-e4m3fn-scaled checkpoint at float32 compute, so the residual
# is the difference between two quantizations of the same fp16 master.
# Observed on an RTX 5060 Ti with torch 2.13.0+cu130: positive max_abs
# 0.05513 / min row cosine 0.98502, negative max_abs 0.00481 / min row
# cosine 0.99980. The limits leave at least 1.9x max_abs headroom and
# cosine slack for cross-host float32 accumulation differences.
_WAN21_GGUF_TEXT_LIMITS = {
    "positive": (0.11, 0.970),
    "negative": (0.011, 0.9995),
}


@pytest.mark.skipif(
    not all(path.exists() for path in (*_REAL_WAN21_SPLIT.values(), _WAN21_UMT5_GGUF)),
    reason="Wan 2.1 artifacts with the city96 umt5 GGUF absent (set DINKSTER_WAN21_MODELS)",
)
def test_real_wan21_gguf_text_encoder_default_dtype_conditioning_on_cuda() -> None:
    from dinkster_inference import (
        MultiStreamLatent,
        SamplingGuidance,
        load_gguf_weight_source,
        load_safetensors_header,
        load_umt5_spiece,
        probe_native,
    )
    from dinkster_inference_torch import (
        Umt5SentencePieceTokenizer,
        Wan21Runtime,
        load_runtime,
    )

    baseline = _release_cuda_allocations()
    expected_size, expected_digest, source_url = _WAN21_UMT5_GGUF_ARTIFACT
    assert source_url.startswith("https://huggingface.co/")
    assert _WAN21_UMT5_GGUF.stat().st_size == expected_size
    with _WAN21_UMT5_GGUF.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_digest

    text_source = load_gguf_weight_source(_WAN21_UMT5_GGUF)
    assert text_source.runtime_facts == _WAN21_UMT5_GGUF_FACTS
    diffusion = load_safetensors_header(_REAL_WAN21_SPLIT["diffusion"])
    vae = load_safetensors_header(_REAL_WAN21_SPLIT["vae"])
    capability = probe_native(diffusion=diffusion, t5xxl=text_source, vae=vae)
    assert capability.native and capability.family_id == "dinkster.wan21"

    # No text_dtype pin and no residency_mode pin: this proves the shipped
    # default path with a GGUF text source (float32 on CUDA hosts, auto
    # residency resolving to balanced for this all-Q8_0 artifact).
    runtime = load_runtime(diffusion=diffusion, t5xxl=text_source, vae=vae)
    assert isinstance(runtime, Wan21Runtime)
    assert runtime.assembled.compute_dtype("umt5xxl") is torch.float32
    # Verified on RTX PRO 6000 Blackwell with torch 2.13.0+cu130.
    assert runtime.runtime_identity == _WAN21_UMT5_GGUF_BALANCED_IDENTITY

    # GGUF text sources carry no spiece_model tensor; assembly must have
    # substituted the vendored tokenizer, which reproduces the official
    # checkpoint's token stream exactly.
    assert runtime.assembled.tokenizer_model == load_umt5_spiece()
    conditioning = _WAN21_OFFICIAL_GOLDENS["conditioning"]
    tokenizer = Umt5SentencePieceTokenizer(runtime.assembled.tokenizer_model)
    prompt_ids = [*tokenizer.encode(conditioning["prompt"]), 1]
    prompt_ids.extend([0] * (512 - len(prompt_ids)))
    assert prompt_ids == conditioning["prompt_token_ids"]

    runtime.assembled.umt5xxl.to("cuda:0")
    with torch.no_grad():
        positive = runtime.encode_text(conditioning["prompt"])
        negative = runtime.encode_text(conditioning["negative_prompt"])
    assert positive.embeddings.shape == negative.embeddings.shape == (1, 512, 4096)
    assert positive.pooled is None and negative.pooled is None
    for name, embeddings in (("positive", positive), ("negative", negative)):
        expected = _wan21_official_tensor("conditioning", f"{name}_active")
        active = embeddings.embeddings[:, : expected.shape[1]].float().cpu()
        assert bool(torch.isfinite(embeddings.embeddings).all())
        assert not torch.count_nonzero(embeddings.embeddings[:, expected.shape[1] :])
        max_abs_limit, cosine_floor = _WAN21_GGUF_TEXT_LIMITS[name]
        assert (active - expected).abs().max().item() <= max_abs_limit
        cosine = torch.nn.functional.cosine_similarity(
            active.reshape(-1, active.shape[-1]),
            expected.reshape(-1, expected.shape[-1]),
            dim=-1,
        )
        assert cosine.min().item() >= cosine_floor
    runtime.assembled.umt5xxl.to("cpu")
    soft_empty_cache(torch.device("cuda:0"))

    pipeline = _WAN21_OFFICIAL_GOLDENS["pipeline"]
    latent = _wan21_official_tensor("pipeline", "initial_latent")
    runtime.assembled.diffusion.to("cuda:0")
    with torch.no_grad():
        sampled_streams = runtime.sample_multistream(
            MultiStreamLatent.from_pairs((("video", latent),)),
            conditioning=runtime.prepare_text_conditioning(positive),
            cfg=SamplingGuidance(runtime.prepare_text_conditioning(negative), pipeline["cfg"]),
            sampler_id=pipeline["sampler"],
            scheduler_id=pipeline["scheduler"],
            steps=pipeline["steps"],
            denoise=pipeline["denoise"],
            seed=pipeline["seed"],
            device="cuda:0",
        )
        sampled = sampled_streams.by_role("video")
    assert sampled.shape == latent.shape
    assert bool(torch.isfinite(sampled).all())
    runtime.assembled.diffusion.to("cpu")

    del latent, negative, positive, runtime, sampled, sampled_streams, text_source
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not all(path.exists() for path in (*_REAL_WAN21_SPLIT.values(), _WAN21_UMT5_GGUF)),
    reason="Wan 2.1 artifacts with the city96 umt5 GGUF absent (set DINKSTER_WAN21_MODELS)",
)
def test_real_wan21_gguf_text_encoded_residency_modes_on_cuda() -> None:
    from dinkster_inference import load_gguf_weight_source, load_safetensors_header
    from dinkster_inference_torch import GgufEncodedLinear, Wan21Runtime, load_runtime

    baseline = _release_cuda_allocations()
    diffusion = load_safetensors_header(_REAL_WAN21_SPLIT["diffusion"])
    vae = load_safetensors_header(_REAL_WAN21_SPLIT["vae"])
    conditioning = _WAN21_OFFICIAL_GOLDENS["conditioning"]
    prompts = (conditioning["prompt"], conditioning["negative_prompt"])

    encoded: dict[str, list[torch.Tensor]] = {}
    identities: dict[str, str] = {}
    swapped: dict[str, int] = {}
    linears: dict[str, int] = {}
    peaks: dict[str, int] = {}
    cache_stats: dict[str, int] = {}
    speed_required = 0
    speed_available = 0
    for mode in ("speed", "memory", "balanced"):
        text_source = load_gguf_weight_source(_WAN21_UMT5_GGUF, residency_mode=mode)
        assert text_source.runtime_facts[:6] == _WAN21_UMT5_GGUF_FACTS[:6]
        runtime = load_runtime(diffusion=diffusion, t5xxl=text_source, vae=vae)
        assert isinstance(runtime, Wan21Runtime)
        assert runtime.assembled.compute_dtype("umt5xxl") is torch.float32
        identities[mode] = runtime.runtime_identity
        modules = list(runtime.assembled.umt5xxl.modules())
        swapped[mode] = sum(isinstance(module, GgufEncodedLinear) for module in modules)
        linears[mode] = sum(isinstance(module, torch.nn.Linear) for module in modules)
        if mode == "speed":
            speed_required = sum(
                parameter.numel() * parameter.element_size()
                for parameter in runtime.assembled.umt5xxl.parameters()
            )
            speed_available, _ = torch.cuda.mem_get_info("cuda:0")
            if speed_required > speed_available:
                del runtime, text_source, modules
                continue
        runtime.assembled.umt5xxl.to("cuda:0")
        torch.cuda.reset_peak_memory_stats("cuda:0")
        with torch.no_grad():
            encoded[mode] = [runtime.encode_text(prompt).embeddings.cpu() for prompt in prompts]
        peaks[mode] = torch.cuda.max_memory_allocated("cuda:0")
        if mode == "balanced":
            caches = {
                module.decoded_cache for module in modules if isinstance(module, GgufEncodedLinear)
            }
            (cache,) = caches
            assert cache is not None
            assert cache.budget_bytes is None
            cache_stats[mode] = cache.used_bytes
        runtime.assembled.umt5xxl.to("cpu")
        del runtime, text_source, modules
        soft_empty_cache(torch.device("cuda:0"))

    # Every one of UMT5-XXL's 24 x 7 projection Linears divides into
    # 32-element blocks, so encoded residency swaps them all.
    assert swapped == {"speed": 0, "memory": 168, "balanced": 168}
    assert linears["speed"] == 168 and linears["memory"] == 0 and linears["balanced"] == 0
    # Verified on RTX PRO 6000 Blackwell with torch 2.13.0+cu130.
    assert identities["speed"] == (
        "native:dinkster.wan21:a8dfe7b595157c0272bd75b1f804ff9d5492edb081a338eee21eb178cee76b7d"
    )
    assert identities["memory"] == (
        "native:dinkster.wan21:87551abf82205a831ac95bde95a2382f980e6610b2589ad4e1b73efedd06db08"
    )
    assert identities["balanced"] == _WAN21_UMT5_GGUF_BALANCED_IDENTITY
    for ours, reference in zip(encoded["balanced"], encoded["memory"], strict=True):
        assert torch.equal(ours, reference)
    if "speed" in encoded:
        for mode in ("memory", "balanced"):
            for ours, reference in zip(encoded[mode], encoded["speed"], strict=True):
                assert torch.equal(ours, reference)
        assert peaks["memory"] * 2 < peaks["speed"]
    else:
        assert speed_required > speed_available
        assert peaks["memory"] * 2 < speed_required
    for mode in ("memory", "balanced"):
        for ours in encoded[mode]:
            assert bool(torch.isfinite(ours).all())
    # Balanced live auto admission cached decoded weights on first use.
    assert cache_stats["balanced"] > 0
    del encoded
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_WAN_FLOW_RVS_SPLIT.values()),
    reason=(
        "official Wan FlowRVS artifacts not present "
        "(set DINKSTER_WAN_FLOW_RVS_MODELS and DINKSTER_WAN21_MODELS)"
    ),
)
def test_real_wan21_flow_rvs_executes_mask_components_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import (
        WAN21_FLOW_RVS_CODEC,
        load_safetensors_header,
        plan_wan21_assembly,
        probe_native,
    )
    from dinkster_inference_torch import assemble as assemble_mod
    from dinkster_inference_torch import assemble_wan21

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    for name, (expected_size, expected_digest, source_url) in _REAL_WAN_FLOW_RVS_ARTIFACTS.items():
        path = _REAL_WAN_FLOW_RVS_SPLIT[name]
        assert source_url.startswith(
            f"https://huggingface.co/Kijai/WanVideo_comfy/resolve/{_WAN_FLOW_RVS_REVISION}/"
        )
        assert path.stat().st_size == expected_size
        with path.open("rb") as artifact:
            assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_digest

    sources = {
        name: load_safetensors_header(path) for name, path in _REAL_WAN_FLOW_RVS_SPLIT.items()
    }
    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["t5xxl"],
        vae=sources["vae"],
    )
    assert capability.native and capability.family_id == "dinkster.wan21"
    plan = plan_wan21_assembly(
        diffusion=sources["diffusion"],
        umt5xxl=sources["t5xxl"],
        vae=sources["vae"],
    )
    real_load_component = assemble_mod._load_component  # pyright: ignore[reportPrivateUsage]
    real_load_tensors = assemble_mod.load_tensors

    def load_component(component: Any, factory: Any, **kwargs: Any) -> torch.nn.Module:
        if component.component == "umt5xxl":
            return torch.nn.Identity()
        return real_load_component(component, factory, **kwargs)

    def load_tokenizer(path: Path, keys: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        if tuple(keys or ()) == (plan.tokenizer_source_key,):
            return {plan.tokenizer_source_key: torch.tensor([1], dtype=torch.uint8)}
        return real_load_tensors(path, keys)

    monkeypatch.setattr(assemble_mod, "_load_component", load_component)
    monkeypatch.setattr(assemble_mod, "load_tensors", load_tokenizer)
    assembled = assemble_wan21(plan, fp8_matmul=False)
    assert assembled.diffusion.config.model_variant == "flow_rvs"
    assert assembled.vae.config.image_channels == 3
    assert assembled.vae.config.conv_out_channels == 1
    assert WAN21_FLOW_RVS_CODEC.content_channels == 1

    diffusion = assembled.diffusion.to("cuda:0")
    latent = torch.zeros((1, 16, 1, 2, 2), device="cuda:0", dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device="cuda:0")
    context = torch.zeros((1, 1, 4096), device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        output = diffusion(latent, timestep, context)
    assert output.shape == latent.shape and bool(torch.isfinite(output).all())
    diffusion.to("cpu")
    soft_empty_cache(torch.device("cuda:0"))

    vae = assembled.vae.to("cuda:0")
    content = torch.linspace(0.0, 1.0, 3 * 16 * 16, device="cuda:0").reshape(1, 3, 1, 16, 16)
    with torch.no_grad():
        encoded = vae.encode(content * 2.0 - 1.0)
        decoded = ((vae.decode(encoded) + 1.0) / 2.0).clamp_(0.0, 1.0)
    assert encoded.shape == (1, 16, 1, 2, 2) and bool(torch.isfinite(encoded).all())
    assert decoded.shape == (1, 1, 1, 16, 16) and bool(torch.isfinite(decoded).all())
    vae.to("cpu")

    assert torch.cuda.max_memory_allocated() - baseline < 6 * 1024**3
    del assembled, content, decoded, diffusion, encoded, latent, output, vae
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not _REAL_WAN21_UNI3C.exists(),
    reason="real Wan 2.1 Uni3C artifact not present (set DINKSTER_WAN21_UNI3C)",
)
def test_real_wan21_uni3c_assembles_and_executes_all_blocks_on_cuda() -> None:
    from dinkster_inference import load_safetensors_header, plan_wan21_uni3c
    from dinkster_inference_torch import (
        assemble_wan21_uni3c,
        validate_wan21_uni3c_resource,
    )

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    expected_size, expected_sha256, asset_digest, source_url = _REAL_WAN21_UNI3C_ARTIFACT
    assert source_url.startswith(
        f"https://huggingface.co/Kijai/WanVideo_comfy/resolve/{_WAN21_UNI3C_REVISION}/"
    )
    assert _REAL_WAN21_UNI3C.stat().st_size == expected_size
    with _REAL_WAN21_UNI3C.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256

    source = load_safetensors_header(
        _REAL_WAN21_UNI3C,
        asset_digest=asset_digest,
        asset_size=expected_size,
    )
    plan = plan_wan21_uni3c(source, asset_digest=asset_digest)
    assembled = assemble_wan21_uni3c(plan)
    model = assembled.patch
    assert len(model.state_dict()) == 490
    assert assembled.compute_dtype is torch.bfloat16
    validate_wan21_uni3c_resource(model, assembled.resource_digest)

    mechanism = enroll_component(model, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    validate_wan21_uni3c_resource(model, assembled.resource_digest)
    control_input = torch.linspace(
        -1.0,
        1.0,
        36 * 4 * 4,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 36, 1, 4, 4)
    additional = torch.linspace(
        0.0,
        1.0,
        7 * 32 * 32,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 7, 1, 32, 32)
    temb = torch.linspace(
        -0.5,
        0.5,
        5120,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 5120)
    residuals: list[torch.Tensor] = []
    with torch.no_grad():
        hidden, freqs = model.process_input(control_input, additional)
        for block_index in range(20):
            hidden, residual = model.forward_block(block_index, hidden, temb, freqs)
            residuals.append(residual)
    torch.cuda.synchronize()

    assert hidden.shape == (1, 4, 1024)
    assert freqs.shape == (1, 4, 1, 32, 2, 2)
    assert len(residuals) == 20
    assert all(residual.shape == (1, 4, 5120) for residual in residuals)
    assert all(bool(torch.isfinite(residual).all()) for residual in residuals)
    assert any(bool(torch.count_nonzero(residual)) for residual in residuals)
    assert torch.cuda.max_memory_allocated() - baseline < 5 * 1024**3

    mechanism.unload()
    validate_wan21_uni3c_resource(model, assembled.resource_digest)
    del additional, assembled, control_input, freqs, hidden, mechanism, model, residuals, temb
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_WAN22_BERNINI_SPLIT.values()),
    reason=(
        "official Wan 2.2 Bernini artifacts not present "
        "(set DINKSTER_WAN22_BERNINI, DINKSTER_WAN22_BERNINI_TEXT, and DINKSTER_WAN22_BERNINI_VAE)"
    ),
)
def test_real_wan22_bernini_loads_fp8_and_executes_context_on_cuda() -> None:
    from dinkster_inference import (
        WAN22_BERNINI_14B,
        load_safetensors_header,
        plan_wan21_assembly,
        probe_native,
    )
    from dinkster_inference_torch import assemble as assemble_mod
    from dinkster_inference_torch.wan21_model import Wan21Model

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    expected_size, expected_sha256, source_url = _REAL_WAN22_BERNINI_ARTIFACT
    assert source_url == (
        f"https://huggingface.co/Comfy-Org/Bernini-R/resolve/{_WAN22_BERNINI_REVISION}/"
        "diffusion_models/wan2.2_bernini_r_high_noise_fp8_scaled.safetensors"
    )
    diffusion_path = _REAL_WAN22_BERNINI_SPLIT["diffusion"]
    assert diffusion_path.stat().st_size == expected_size
    with diffusion_path.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256

    sources = {
        name: load_safetensors_header(path) for name, path in _REAL_WAN22_BERNINI_SPLIT.items()
    }
    assert sources["diffusion"].metadata() == {"format": "pt", "model_type": "bernini_high"}
    capability = probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["t5xxl"],
        vae=sources["vae"],
    )
    assert capability.native and capability.family_id == "dinkster.wan21"
    plan = plan_wan21_assembly(
        diffusion=sources["diffusion"],
        umt5xxl=sources["t5xxl"],
        vae=sources["vae"],
    )
    assert plan.diffusion.config is WAN22_BERNINI_14B
    assert len(plan.diffusion.keys) == 1095
    assert len(plan.diffusion.quant) == 360

    model = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        plan.diffusion,
        Wan21Model,
        compute_dtype=torch.bfloat16,
        fp8_matmul=True,
    ).to("cuda:0")
    latent = torch.linspace(
        -1.0,
        1.0,
        16 * 2 * 2,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 16, 1, 2, 2)
    context_latent = torch.linspace(
        0.0,
        1.0,
        16 * 2 * 4,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 16, 1, 2, 4)
    timestep = torch.tensor([500.0], device="cuda:0")
    text = torch.zeros((1, 1, 4096), device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        output = model(latent, timestep, text, context_latents=(context_latent,))
    torch.cuda.synchronize()

    assert output.shape == latent.shape
    assert bool(torch.isfinite(output).all())
    assert torch.cuda.max_memory_allocated() - baseline < 24 * 1024**3
    del context_latent, latent, model, output, text, timestep
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not _REAL_WAN22_S2V.exists(),
    reason="official Wan 2.2 S2V artifact not present (set DINKSTER_WAN22_S2V)",
)
def test_real_wan22_s2v_loads_fp8_and_executes_every_control_path_on_cuda() -> None:
    from dinkster_inference import (
        WAN22_S2V_14B,
        load_safetensors_header,
        plan_wan21_standalone_component,
    )
    from dinkster_inference_torch import assemble as assemble_mod
    from dinkster_inference_torch.wan22_s2v import Wan22S2VModel

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    expected_size, expected_sha256, source_url = _REAL_WAN22_S2V_ARTIFACT
    assert source_url == (
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/"
        f"{_WAN22_S2V_REVISION}/split_files/diffusion_models/"
        "wan2.2_s2v_14B_fp8_scaled.safetensors"
    )
    assert _REAL_WAN22_S2V.stat().st_size == expected_size
    with _REAL_WAN22_S2V.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256

    source = load_safetensors_header(_REAL_WAN22_S2V)
    plan = plan_wan21_standalone_component(source, "diffusion").component
    assert plan.config is WAN22_S2V_14B
    assert len(plan.keys) == 1260
    assert len(plan.quant) == 467
    model = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        plan,
        Wan22S2VModel,
        compute_dtype=torch.bfloat16,
        fp8_matmul=True,
    ).to("cuda:0")
    target = torch.linspace(
        -1.0,
        1.0,
        16 * 8 * 8,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 16, 1, 8, 8)
    timestep = torch.tensor([500.0], device="cuda:0")
    text = torch.zeros((1, 1, 4096), device="cuda:0", dtype=torch.bfloat16)
    audio = torch.linspace(
        -0.25,
        0.25,
        25 * 1024 * 4,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 25, 1024, 4)
    reference = torch.zeros((1, 16, 1, 8, 8), device="cuda:0", dtype=torch.bfloat16)
    control = torch.full_like(target, 0.125)
    motion = torch.zeros((1, 16, 19, 8, 8), device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        output = model(
            target,
            timestep,
            text,
            audio_embed=audio,
            reference_latent=reference,
            control_video=control,
            reference_motion=motion,
        )
    torch.cuda.synchronize()

    assert output.shape == target.shape
    assert bool(torch.isfinite(output).all())
    assert torch.cuda.max_memory_allocated() - baseline < 24 * 1024**3
    del audio, control, model, motion, output, reference, target, text, timestep
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_WAN22_DANCER.values()),
    reason=(
        "official WanDancer artifacts not present "
        "(set DINKSTER_WAN22_DANCER_LOCAL and DINKSTER_WAN22_DANCER_GLOBAL)"
    ),
)
def test_real_wan22_dancer_loads_fp8_and_executes_every_conditioning_path_on_cuda() -> None:
    from collections import Counter

    from dinkster_inference import (
        WAN22_WANDANCER_14B,
        load_safetensors_header,
        plan_wan21_standalone_component,
    )
    from dinkster_inference_torch import assemble as assemble_mod
    from dinkster_inference_torch.wan22_dancer import Wan22DancerModel

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    sources = {}
    for branch, path in _REAL_WAN22_DANCER.items():
        expected_size, expected_sha256, source_url = _REAL_WAN22_DANCER_ARTIFACTS[branch]
        assert source_url == (
            "https://huggingface.co/Comfy-Org/Wan-Dancer/resolve/"
            f"{_WAN22_DANCER_REVISION}/diffusion_models/"
            f"wan2.2_dancer_14b_{branch}_fp8_scaled.safetensors"
        )
        assert path.stat().st_size == expected_size
        with path.open("rb") as artifact:
            assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256
        source = load_safetensors_header(path)
        assert source.metadata() == {"model_type": f"wanvideo_wantodance_{branch}"}
        assert Counter(source.entry(key).geometry.dtype.name for key in source.keys()) == {
            "bfloat16": 944,
            "float32": 480,
            "float8_e4m3fn": 480,
            "uint8": 480,
        }
        plan = plan_wan21_standalone_component(source, "diffusion").component
        assert plan.config is WAN22_WANDANCER_14B
        assert len(plan.keys) == 1432
        assert len(plan.quant) == 480
        assert len(plan.transforms) == 12
        sources[branch] = (source, plan)

    target = torch.linspace(
        -1.0,
        1.0,
        36 * 3 * 7,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 36, 1, 3, 7)
    timestep = torch.tensor([500.0], device="cuda:0")
    text = torch.zeros((1, 1, 4096), device="cuda:0", dtype=torch.bfloat16)
    vision = torch.zeros((1, 1, 1280), device="cuda:0", dtype=torch.bfloat16)
    reference_vision = torch.ones((1, 1, 1280), device="cuda:0", dtype=torch.bfloat16)
    audio = torch.linspace(
        -0.25,
        0.25,
        3 * 35,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 3, 35)

    local_model = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        sources["local"][1],
        Wan22DancerModel,
        compute_dtype=torch.bfloat16,
        fp8_matmul=True,
    )
    assert type(local_model) is Wan22DancerModel
    assert local_model.config is WAN22_WANDANCER_14B
    local_mechanism = enroll_component(local_model, load_device="cuda:0", offload_device="cpu")
    local_mechanism.partially_load(12 * 1024**3)
    assert 0 < local_mechanism.loaded_bytes() < local_mechanism.total_bytes()
    with torch.no_grad():
        local = local_model(
            target,
            timestep,
            text,
            vision,
            reference_vision=reference_vision,
            audio_embed=audio,
            fps=30.0,
            audio_inject_scale=1.0,
        )
        scale_zero = local_model(
            target,
            timestep,
            text,
            vision,
            reference_vision=reference_vision,
            audio_embed=audio,
            fps=30.0,
            audio_inject_scale=0.0,
        )
        no_audio = local_model(
            target,
            timestep,
            text,
            vision,
            reference_vision=reference_vision,
            fps=30.0,
        )
    local_mechanism.unload()
    del local_mechanism, local_model
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2

    global_model = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        sources["global"][1],
        Wan22DancerModel,
        compute_dtype=torch.bfloat16,
        fp8_matmul=True,
    )
    assert type(global_model) is Wan22DancerModel
    assert global_model.config is WAN22_WANDANCER_14B
    global_mechanism = enroll_component(global_model, load_device="cuda:0", offload_device="cpu")
    global_mechanism.partially_load(12 * 1024**3)
    assert 0 < global_mechanism.loaded_bytes() < global_mechanism.total_bytes()
    with torch.no_grad():
        global_output = global_model(
            target,
            timestep,
            text,
            vision,
            reference_vision=reference_vision,
            audio_embed=audio,
            fps=24.0,
            audio_inject_scale=1.0,
        )
    torch.cuda.synchronize()

    assert (
        local.shape
        == global_output.shape
        == scale_zero.shape
        == no_audio.shape
        == target[:, :16].shape
    )
    assert all(bool(torch.isfinite(output).all()) for output in (local, global_output, scale_zero))
    assert torch.equal(scale_zero, no_audio)
    assert not torch.equal(local, global_output)
    assert torch.cuda.max_memory_allocated() - baseline < 24 * 1024**3
    global_mechanism.unload()
    del audio, global_mechanism, global_model, global_output, local, no_audio, reference_vision
    del scale_zero, sources, target, text, timestep, vision
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not _REAL_WHISPER_LARGE_V3.exists(),
    reason="official Whisper Large v3 artifact not present (set DINKSTER_WHISPER_LARGE_V3)",
)
def test_real_whisper_large_v3_loads_and_encodes_audio_on_cuda() -> None:
    from dinkster_assets import AssetRef
    from dinkster_inference import WHISPER_LARGE_V3
    from dinkster_inference_torch import WhisperLargeV3Model, load_whisper_large_v3_component

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return _REAL_WHISPER_LARGE_V3 if digest == asset_digest else None

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    expected_size, expected_sha256, asset_digest, source_url = _REAL_WHISPER_LARGE_V3_ARTIFACT
    assert source_url == (
        f"https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/{_HUMO_REVISION}/"
        "split_files/audio_encoders/whisper_large_v3_fp16.safetensors"
    )
    assert _REAL_WHISPER_LARGE_V3.stat().st_size == expected_size
    with _REAL_WHISPER_LARGE_V3.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256

    asset = AssetRef(
        asset_digest,
        _REAL_WHISPER_LARGE_V3.name,
        expected_size,
        resolver=Resolver(),
    )
    loaded = load_whisper_large_v3_component(
        _REAL_WHISPER_LARGE_V3,
        asset=asset,
        expected_identity=_WHISPER_LARGE_V3_RUNTIME_IDENTITY,
        compute_dtype=torch.float32,
    )
    assert loaded.runtime_identity == _WHISPER_LARGE_V3_RUNTIME_IDENTITY
    assert loaded.plan.config is WHISPER_LARGE_V3
    assert len(loaded.plan.keys) == 487
    assert len(loaded.plan.ignored) == 772
    model = loaded.module
    assert isinstance(model, WhisperLargeV3Model)

    mechanism = enroll_component(model, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    audio = torch.linspace(-0.25, 0.25, 16_000, device="cuda:0").reshape(1, 1, -1)
    with torch.no_grad():
        encoded, layers = model(audio)
    torch.cuda.synchronize()

    assert encoded.shape == (1, 1500, 1280)
    assert len(layers) == 33
    assert all(layer.shape == encoded.shape for layer in layers)
    assert bool(torch.isfinite(encoded).all())
    assert all(bool(torch.isfinite(layer).all()) for layer in layers)
    assert torch.cuda.max_memory_allocated() - baseline < 12 * 1024**3
    mechanism.unload()
    del audio, encoded, layers, loaded, mechanism, model
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not _REAL_WAN21_HUMO.exists(),
    reason="official Wan HuMo artifact not present (set DINKSTER_WAN21_HUMO)",
)
def test_real_wan21_humo_loads_fp8_and_executes_audio_reference_on_cuda() -> None:
    from dinkster_assets import AssetRef
    from dinkster_inference import WAN21_HUMO_17B
    from dinkster_inference_torch import Wan21HumoModel, load_wan21_component

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return _REAL_WAN21_HUMO if digest == asset_digest else None

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    expected_size, expected_sha256, asset_digest, source_url = _REAL_WAN21_HUMO_ARTIFACT
    assert source_url == (
        f"https://huggingface.co/Comfy-Org/HuMo_ComfyUI/resolve/{_HUMO_REVISION}/"
        "split_files/diffusion_models/humo_17B_fp8_e4m3fn.safetensors"
    )
    assert _REAL_WAN21_HUMO.stat().st_size == expected_size
    with _REAL_WAN21_HUMO.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256

    asset = AssetRef(
        asset_digest,
        _REAL_WAN21_HUMO.name,
        expected_size,
        resolver=Resolver(),
    )
    loaded = load_wan21_component(
        _REAL_WAN21_HUMO,
        asset=asset,
        expected_role="diffusion",
        expected_identity=_WAN21_HUMO_RUNTIME_IDENTITY,
        compute_dtype=torch.bfloat16,
    )
    assert loaded.runtime_identity == _WAN21_HUMO_RUNTIME_IDENTITY
    assert loaded.plan.config is WAN21_HUMO_17B
    assert len(loaded.plan.keys) == 1583
    assert not loaded.plan.quant
    model = loaded.module
    assert isinstance(model, Wan21HumoModel)

    mechanism = enroll_component(model, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    target = torch.linspace(
        -1.0,
        1.0,
        36 * 2 * 2,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 36, 1, 2, 2)
    timestep = torch.tensor([500.0], device="cuda:0")
    text = torch.zeros((1, 1, 4096), device="cuda:0", dtype=torch.bfloat16)
    audio = torch.linspace(
        -0.25,
        0.25,
        8 * 5 * 1280,
        device="cuda:0",
        dtype=torch.bfloat16,
    ).reshape(1, 1, 8, 5, 1280)
    reference = torch.zeros_like(target)
    with torch.no_grad():
        output = model(
            target,
            timestep,
            text,
            audio_embed=audio,
            reference_latent=reference,
        )
    torch.cuda.synchronize()

    assert output.shape == (1, 16, 1, 2, 2)
    assert bool(torch.isfinite(output).all())
    assert torch.cuda.max_memory_allocated() - baseline < 28 * 1024**3
    mechanism.unload()
    del audio, loaded, mechanism, model, output, reference, target, text, timestep
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not _REAL_WAV2VEC2_CHINESE_BASE.exists(),
    reason=(
        "official Chinese Wav2Vec2 base artifact not present (set DINKSTER_WAV2VEC2_CHINESE_BASE)"
    ),
)
def test_real_wav2vec2_chinese_base_loads_and_encodes_audio_on_cuda() -> None:
    from dinkster_assets import AssetRef
    from dinkster_inference import WAV2VEC2_CHINESE_BASE
    from dinkster_inference_torch import Wav2Vec2Model, load_wav2vec2_component
    from dinkster_model_wan import provider as wan_provider

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return _REAL_WAV2VEC2_CHINESE_BASE if digest == asset_digest else None

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    expected_size, expected_sha256, asset_digest, source_url = _REAL_WAV2VEC2_CHINESE_BASE_ARTIFACT
    assert source_url == (
        "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/"
        f"{_WAV2VEC2_CHINESE_BASE_REVISION}/wav2vec2-chinese-base_fp16.safetensors"
    )
    assert _REAL_WAV2VEC2_CHINESE_BASE.stat().st_size == expected_size
    with _REAL_WAV2VEC2_CHINESE_BASE.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256

    asset = AssetRef(
        asset_digest,
        _REAL_WAV2VEC2_CHINESE_BASE.name,
        expected_size,
        resolver=Resolver(),
    )
    loaded = load_wav2vec2_component(
        _REAL_WAV2VEC2_CHINESE_BASE,
        asset=asset,
        expected_identity=_WAV2VEC2_CHINESE_BASE_RUNTIME_IDENTITY,
        compute_dtype=torch.float16,
    )
    assert loaded.runtime_identity == _WAV2VEC2_CHINESE_BASE_RUNTIME_IDENTITY
    assert loaded.plan.config is WAV2VEC2_CHINESE_BASE
    assert len(loaded.plan.keys) == 211
    model = loaded.module
    assert isinstance(model, Wav2Vec2Model)

    mechanism = enroll_component(model, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    audio = torch.linspace(
        -0.25,
        0.25,
        16_000,
        device="cuda:0",
        dtype=torch.float16,
    ).reshape(1, 1, -1)
    with torch.no_grad():
        encoded, layers = model(audio)
    torch.cuda.synchronize()

    assert len(layers) == 13
    assert encoded.ndim == 3 and encoded.shape[0] == 1 and encoded.shape[2] == 768
    assert encoded.shape[1] > 0
    assert all(layer.shape == encoded.shape for layer in layers)
    assert bool(torch.isfinite(encoded).all())
    assert all(bool(torch.isfinite(layer).all()) for layer in layers)
    stacked_layers = torch.stack(layers).squeeze(1)[1:]
    expected_audio = (
        torch.nn.functional.interpolate(
            stacked_layers.transpose(1, 2),
            size=int(stacked_layers.shape[1] / 2),
            mode="linear",
            align_corners=True,
        )
        .transpose(1, 2)
        .movedim(0, 1)
        .contiguous()
    )
    owned_audio = wan_provider.WanInfiniteTalkAudioOutput(
        tuple(layer.detach().to(device="cpu").contiguous() for layer in layers),
        16_000,
    )
    projected_input = wan_provider._infinite_talk_audio_features(  # pyright: ignore[reportPrivateUsage]
        owned_audio,
        "audio_encoder_output_1",
        torch.device("cuda:0"),
    )
    torch.testing.assert_close(projected_input, expected_audio, rtol=0.0, atol=0.0)
    assert torch.cuda.max_memory_allocated() - baseline < 2 * 1024**3
    mechanism.unload()
    del audio, encoded, expected_audio, layers, loaded, mechanism, model, owned_audio
    del projected_input, stacked_layers
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


@pytest.mark.skipif(
    not (
        _REAL_WAN21_I2V.exists()
        and _REAL_WAN21_I2V_CLIP.exists()
        and _REAL_WAN21_MULTITALK.exists()
    ),
    reason=(
        "official Wan 2.1 I2V, CLIP vision, and MultiTalk artifacts not present "
        "(set DINKSTER_WAN21_I2V, DINKSTER_WAN21_I2V_CLIP, and DINKSTER_WAN21_MULTITALK)"
    ),
)
def test_real_wan21_multitalk_loads_and_executes_single_and_two_speakers_on_cuda() -> None:
    from dinkster_inference import (
        WAN21_I2V_14B,
        load_gguf_weight_source,
        load_safetensors_header,
        plan_wan21_assembly,
        plan_wan21_multitalk,
    )
    from dinkster_inference_torch import (
        Wan21InfiniteTalkExecution,
        Wan21MultiTalkExecution,
        assemble_wan21_multitalk,
        validate_wan21_multitalk_resource,
        wan21_multitalk_resource_digest,
        wan21_multitalk_tensor_digest,
    )
    from dinkster_inference_torch import assemble as assemble_mod
    from dinkster_inference_torch.wan21_model import Wan21Model

    baseline = _release_cuda_allocations()
    torch.cuda.reset_peak_memory_stats()
    for path, artifact_details in (
        (_REAL_WAN21_I2V, _REAL_WAN21_I2V_ARTIFACT),
        (_REAL_WAN21_I2V_CLIP, _REAL_WAN21_I2V_CLIP_ARTIFACT),
        (_REAL_WAN21_MULTITALK, _REAL_WAN21_MULTITALK_ARTIFACT),
    ):
        expected_size, expected_sha256, _asset_digest, source_url = artifact_details
        assert source_url.startswith(
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
            f"{_WAN21_REVISION}/split_files/"
        )
        assert path.stat().st_size == expected_size
        with path.open("rb") as artifact:
            assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256

    patch_size, _patch_sha256, patch_asset_digest, _patch_url = _REAL_WAN21_MULTITALK_ARTIFACT
    patch_source = load_safetensors_header(
        _REAL_WAN21_MULTITALK,
        asset_digest=patch_asset_digest,
        asset_size=patch_size,
    )
    patch_plan = plan_wan21_multitalk(patch_source, asset_digest=patch_asset_digest)
    assembled_patch = assemble_wan21_multitalk(patch_plan)
    patch = assembled_patch.patch
    assert len(patch.state_dict()) == 330
    assert assembled_patch.resource_digest == _WAN21_MULTITALK_RESOURCE_IDENTITY
    assert assembled_patch.resource_digest == wan21_multitalk_resource_digest(
        patch_asset_digest, torch.bfloat16
    )

    model_size, _model_sha256, model_asset_digest, _model_url = _REAL_WAN21_I2V_ARTIFACT
    diffusion_source = load_safetensors_header(
        _REAL_WAN21_I2V,
        asset_digest=model_asset_digest,
        asset_size=model_size,
    )
    plan = plan_wan21_assembly(
        diffusion=diffusion_source,
        umt5xxl=load_gguf_weight_source(_WAN21_UMT5_GGUF),
        clip_vision=load_safetensors_header(_REAL_WAN21_I2V_CLIP),
        vae=load_safetensors_header(_REAL_WAN21_SPLIT["vae"]),
    )
    assert plan.diffusion.config is WAN21_I2V_14B
    assert len(plan.diffusion.keys) == 1303
    assert len(plan.diffusion.quant) == 488
    model = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        plan.diffusion,
        Wan21Model,
        compute_dtype=torch.bfloat16,
        fp8_matmul=True,
    )

    patch_mechanism = enroll_component(patch, load_device="cuda:0", offload_device="cpu")
    model_mechanism = enroll_component(model, load_device="cuda:0", offload_device="cpu")
    patch_mechanism.partially_load(None)
    model_mechanism.partially_load(None)
    validate_wan21_multitalk_resource(patch, assembled_patch.resource_digest)

    first_audio = torch.linspace(
        -0.25,
        0.25,
        12 * 768,
        device="cuda:0",
        dtype=torch.float16,
    ).reshape(1, 12, 768)
    second_audio = first_audio.flip(2) + 0.125
    single_context = patch.project_audio((first_audio,), 0, 1)
    two_context = patch.project_audio((first_audio, second_audio), 0, 1)
    single_execution = Wan21MultiTalkExecution(
        patch,
        single_context,
        None,
        1.0,
        assembled_patch.resource_digest,
        wan21_multitalk_tensor_digest(single_context),
        None,
    )
    target_masks = torch.tensor(
        [[True, True, False, False], [False, False, True, True]],
        device="cuda:0",
    )
    two_execution = Wan21MultiTalkExecution(
        patch,
        two_context,
        target_masks,
        1.0,
        assembled_patch.resource_digest,
        wan21_multitalk_tensor_digest(two_context),
        wan21_multitalk_tensor_digest(target_masks),
    )
    motion = torch.zeros((1, 16, 1, 4, 4), device="cuda:0", dtype=torch.bfloat16)
    runtime_execution = Wan21InfiniteTalkExecution(
        two_execution,
        motion,
        True,
        wan21_multitalk_tensor_digest(motion),
    )
    assert runtime_execution.patch is two_execution

    latent = torch.zeros((1, 36, 1, 4, 4), device="cuda:0", dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device="cuda:0")
    text = torch.zeros((1, 1, 4096), device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        single_output = model(latent, timestep, text, multitalk=single_execution)
        two_output = model(latent, timestep, text, multitalk=two_execution)
    torch.cuda.synchronize()

    assert single_output.shape == two_output.shape == (1, 16, 1, 4, 4)
    assert bool(torch.isfinite(single_output).all())
    assert bool(torch.isfinite(two_output).all())
    assert torch.cuda.max_memory_allocated() - baseline < 30 * 1024**3
    model_mechanism.unload()
    patch_mechanism.unload()
    validate_wan21_multitalk_resource(patch, assembled_patch.resource_digest)
    del assembled_patch, diffusion_source, first_audio, latent, model, model_mechanism
    del motion, patch, patch_mechanism, patch_plan, patch_source, plan, runtime_execution
    del second_audio, single_context, single_execution, single_output, target_masks, text
    del timestep, two_context, two_execution, two_output
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


def _qwen_image_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = torch.linspace(-0.5, 0.5, 8).reshape(1, 2, 1, 2, 2)
    timestep = torch.tensor([0.25])
    context = torch.linspace(-0.25, 0.25, 12).reshape(1, 2, 6)
    return latent, timestep, context


def test_qwen_image_reduced_cuda_matches_cpu_and_routes_injected_attention() -> None:
    latent, timestep, context = _qwen_image_inputs()
    cpu_model = QwenImage(reduced_qwen_image_config())
    fill_qwen_image_parameters(cpu_model)
    spy = CallableModuleKernel(select_attention("qwen", "sdpa").kernel)
    cuda_model = QwenImage(reduced_qwen_image_config(), attention_kernel=spy).to("cuda:0")
    cuda_model.load_state_dict(cpu_model.state_dict(), strict=True)
    with torch.no_grad():
        expected = cpu_model(latent, timestep, context)
        actual = cuda_model(
            latent.to("cuda:0"),
            timestep.to("cuda:0"),
            context.to("cuda:0"),
        )
    assert actual.device == torch.device("cuda:0")
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    assert len(spy.calls) == 1
    assert spy.calls[0]["q_shape"] == (1, 2, 3, 6)
    assert_kernel_is_not_model_state(cuda_model, spy)


@pytest.mark.parametrize(
    ("dtype", "atol"),
    (
        # Dual-Blackwell max errors were 3.16e-4 (FP16) and 2.41e-3 (BF16).
        (torch.float16, 1e-3),
        pytest.param(
            torch.bfloat16,
            6e-3,
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="CUDA BF16 required"
            ),
        ),
    ),
)
def test_qwen_image_reduced_low_precision_tracks_float32_on_cuda(
    dtype: torch.dtype, atol: float
) -> None:
    latent, timestep, context = _qwen_image_inputs()
    reference = QwenImage(reduced_qwen_image_config())
    fill_qwen_image_parameters(reference)
    model = QwenImage(reduced_qwen_image_config())
    model.load_state_dict(reference.state_dict(), strict=True)
    model.to("cuda:0", dtype=dtype)
    with torch.no_grad():
        expected = reference(latent, timestep, context)
        actual = model(
            latent.to("cuda:0", dtype=dtype),
            timestep.to("cuda:0"),
            context.to("cuda:0", dtype=dtype),
        )
    assert actual.dtype == dtype
    assert bool(torch.isfinite(actual).all())
    assert float(expected.abs().max()) > 10 * atol
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0, atol=atol)


def test_qwen_image_reduced_cuda_autograd_repeats_and_exception_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent, timestep, context = _qwen_image_inputs()
    model = QwenImage(reduced_qwen_image_config()).to("cuda:0")
    fill_qwen_image_parameters(model)
    cuda_latent = latent.to("cuda:0").requires_grad_()
    cuda_timestep = timestep.to("cuda:0")
    cuda_context = context.to("cuda:0").requires_grad_()
    output = model(cuda_latent, cuda_timestep, cuda_context)
    output.square().mean().backward()
    assert cuda_latent.grad is not None and bool(torch.isfinite(cuda_latent.grad).all())
    assert cuda_context.grad is not None and bool(torch.isfinite(cuda_context.grad).all())
    assert bool(torch.count_nonzero(cuda_latent.grad))
    assert bool(torch.count_nonzero(cuda_context.grad))
    parameters = dict(model.named_parameters())
    expected_gradient_paths = {
        "img_in.weight",
        "txt_in.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.attn.add_k_proj.weight",
        "transformer_blocks.0.attn.add_v_proj.weight",
        "transformer_blocks.0.attn.to_out.0.weight",
        "proj_out.weight",
    }
    for name in expected_gradient_paths:
        parameter = parameters[name]
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert bool(torch.count_nonzero(parameter.grad)), name
    assert parameters["transformer_blocks.0.attn.to_add_out.weight"].grad is None
    assert parameters["transformer_blocks.0.txt_mlp.net.0.proj.weight"].grad is None
    model.zero_grad(set_to_none=True)
    del output, cuda_latent, cuda_context
    baseline = _release_cuda_allocations()
    with torch.no_grad():
        first = model(latent.to("cuda:0"), cuda_timestep, context.to("cuda:0"))
        expected = first.cpu()
        del first
    assert _release_cuda_allocations() == baseline
    with torch.no_grad():
        second = model(latent.to("cuda:0"), cuda_timestep, context.to("cuda:0"))
        torch.testing.assert_close(second.cpu(), expected)
        del second
    assert _release_cuda_allocations() == baseline

    block = model.transformer_blocks[0]
    from dinkster_inference_torch import qwen_image as qwen_image_module

    class PrefetchHandle:
        def __init__(self) -> None:
            self.tensor: torch.Tensor | None = torch.ones(1024, device="cuda:0")
            self.closed = False

    prefetch = PrefetchHandle()
    assert prefetch.tensor is not None
    prefetch_tensor = weakref.ref(prefetch.tensor)

    def close_prefetch(handle: object) -> None:
        assert handle is prefetch
        prefetch.tensor = None
        prefetch.closed = True

    def make_prefetch(_blocks: torch.nn.ModuleList) -> PrefetchHandle:
        return prefetch

    def pop_prefetch(_handle: object, _block: torch.nn.Module | None) -> None:
        return None

    monkeypatch.setattr(qwen_image_module, "make_prefetch_queue", make_prefetch)
    monkeypatch.setattr(qwen_image_module, "prefetch_queue_pop", pop_prefetch)
    monkeypatch.setattr(qwen_image_module, "close_prefetch_queue", close_prefetch)

    def raise_after_allocating(*_args: object, **_kwargs: object) -> None:
        temporary = torch.ones(1024, device="cuda:0")
        assert temporary.is_cuda
        raise RuntimeError("injected Qwen block failure")

    monkeypatch.setattr(block, "forward", raise_after_allocating)
    with pytest.raises(RuntimeError, match="injected Qwen block failure"):
        model(latent.to("cuda:0"), cuda_timestep, context.to("cuda:0"))
    assert prefetch.closed
    assert prefetch_tensor() is None
    assert _release_cuda_allocations() == baseline


def _wan21_content() -> torch.Tensor:
    return torch.linspace(-0.5, 0.5, 1 * 3 * 5 * 8 * 8).reshape(1, 3, 5, 8, 8)


def test_wan21_vae_reduced_cuda_matches_cpu_and_routes_injected_attention() -> None:
    content = _wan21_content()
    cpu_model = WanVAE(_reduced_wan21_config())
    _fill_wan21_parameters(cpu_model)
    spy = CallableModuleKernel(select_attention("vae").kernel)
    cuda_model = WanVAE(_reduced_wan21_config(), attention_kernel=spy).to("cuda:0")
    cuda_model.load_state_dict(cpu_model.state_dict(), strict=True)
    with torch.no_grad(), _strict_fp32():
        expected_latent = cpu_model.encode(content)
        expected_content = cpu_model.decode(expected_latent)
        actual_latent = cuda_model.encode(content.to("cuda:0"))
        actual_content = cuda_model.decode(actual_latent)
    torch.testing.assert_close(actual_latent.cpu(), expected_latent, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual_content.cpu(), expected_content, rtol=1e-4, atol=1e-5)
    assert len(spy.calls) > 0
    assert all(call["mask"] is None and call["causal"] is False for call in spy.calls)
    assert_kernel_is_not_model_state(cuda_model, spy)


@pytest.mark.parametrize(
    ("dtype", "latent_atol", "output_atol"),
    (
        # Dual-Blackwell latent/output max errors: FP16 1.12e-6/4.39e-5,
        # BF16 1.07e-5/3.96e-4.
        (torch.float16, 5e-6, 2e-4),
        pytest.param(
            torch.bfloat16,
            5e-5,
            2e-3,
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="CUDA BF16 required"
            ),
        ),
    ),
)
def test_wan21_vae_reduced_low_precision_tracks_float32_on_cuda(
    dtype: torch.dtype, latent_atol: float, output_atol: float
) -> None:
    content = _wan21_content()
    reference = WanVAE(_reduced_wan21_config())
    _fill_wan21_parameters(reference)
    model = WanVAE(_reduced_wan21_config())
    model.load_state_dict(reference.state_dict(), strict=True)
    model.to("cuda:0", dtype=dtype)
    with torch.no_grad():
        expected_latent = reference.encode(content)
        expected = reference.decode(expected_latent)
        actual_latent = model.encode(content.to("cuda:0", dtype=dtype))
        actual = model.decode(actual_latent)
    assert actual.dtype == dtype
    assert bool(torch.isfinite(actual).all())
    assert float(expected_latent.abs().max()) > 10 * latent_atol
    assert float(expected.abs().max()) > 10 * output_atol
    torch.testing.assert_close(
        actual_latent.float().cpu(), expected_latent, rtol=0, atol=latent_atol
    )
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0, atol=output_atol)


def test_wan21_vae_reduced_cuda_autograd_repeats_and_exception_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _wan21_content()
    model = WanVAE(_reduced_wan21_config()).to("cuda:0")
    _fill_wan21_parameters(model)
    # Break channel symmetry so gradient checks do not depend on SDPA rounding noise.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("weight"):
                parameter.copy_(
                    torch.linspace(
                        -0.02, 0.02, parameter.numel(), device=parameter.device
                    ).reshape_as(parameter)
                )
    cuda_content = content.to("cuda:0").requires_grad_()
    latent = model.encode(cuda_content)
    decoded = model.decode(latent)
    decoded.square().mean().backward()
    assert cuda_content.grad is not None and bool(torch.isfinite(cuda_content.grad).all())
    assert bool(torch.count_nonzero(cuda_content.grad))
    parameters = dict(model.named_parameters())
    expected_gradient_paths = {
        "encoder.conv1.weight",
        "encoder.middle.1.to_qkv.weight",
        "conv1.weight",
        "conv2.weight",
        "decoder.middle.1.to_qkv.weight",
        "decoder.head.2.weight",
    }
    for name in expected_gradient_paths:
        parameter = parameters[name]
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert bool(torch.count_nonzero(parameter.grad)), name
    model.zero_grad(set_to_none=True)
    del decoded, latent, cuda_content
    baseline = _release_cuda_allocations()
    with torch.no_grad():
        first = model.decode(model.encode(content.to("cuda:0")))
        expected = first.cpu()
        del first
    assert _release_cuda_allocations() == baseline
    with torch.no_grad():
        second = model.decode(model.encode(content.to("cuda:0")))
        torch.testing.assert_close(second.cpu(), expected)
        del second
    assert _release_cuda_allocations() == baseline

    original = model.decoder.forward
    calls = 0

    def fail_second_chunk(
        x: torch.Tensor,
        feat_cache: Any = None,
        feat_idx: list[int] | None = None,
    ) -> list[torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == 2:
            temporary = torch.ones(1024, device="cuda:0")
            assert temporary.is_cuda
            raise RuntimeError("injected Wan21 decoder failure")
        return original(x, feat_cache, feat_idx)

    monkeypatch.setattr(model.decoder, "forward", fail_second_chunk)
    with pytest.raises(RuntimeError, match="injected Wan21 decoder failure"):
        model.decode(torch.zeros(1, 2, 2, 2, 2, device="cuda:0"))
    assert calls == 2
    assert _release_cuda_allocations() == baseline


def test_cosmos_predict2_rope_table_on_cuda_matches_cpu() -> None:
    """The rope table is generated directly on the model device, so CUDA
    trigonometry must agree with the CPU tables the goldens executed,
    at the full Anima head dimension and 4x spatial extrapolation."""
    with torch.device("meta"):
        geometry = AnimaModel().geometry
    cpu_table = cosmos_predict2_rope_table(geometry, 1, 48, 32)
    cuda_table = cosmos_predict2_rope_table(geometry, 1, 48, 32, torch.device("cuda:0"))
    assert cuda_table.device.type == "cuda"
    assert cuda_table.dtype == torch.float32
    torch.testing.assert_close(cuda_table.cpu(), cpu_table)


_VAE_DTYPE_MODELS = Path(os.environ.get("DINKSTER_VAE_DTYPE_MODELS", "/nonexistent"))
_REAL_VAE_DTYPE_ARTIFACTS = {
    "sdxl": _VAE_DTYPE_MODELS / "sdxl/sd_xl_base_1.0.safetensors",
    "z_image": _VAE_DTYPE_MODELS / "zimage/ae.safetensors",
    "wan21": _VAE_DTYPE_MODELS / "wan21/wan_2.1_vae.safetensors",
}
_VAE_DTYPE_ARTIFACT_FACTS = {
    "sdxl": (
        6_938_078_334,
        "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/"
        "462165984030d82259a11f4367a4eed129e94a7b/sd_xl_base_1.0.safetensors",
    ),
    "z_image": (
        335_304_388,
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/"
        "08d04455279082882deaabc8d0d09fc914c071e1/split_files/vae/ae.safetensors",
    ),
    "wan21": (
        253_815_318,
        "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
        "617a7633e636506f850e043bc4605f290a466a8e/split_files/vae/"
        "wan_2.1_vae.safetensors",
    ),
}
_VAE_DTYPE_GOLDENS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "goldens" / "vae_dtype_goldens.json").read_text()
)


def _vae_dtype_tensor(case: str, name: str) -> torch.Tensor:
    payload = _VAE_DTYPE_GOLDENS["cases"][case][name]
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


@pytest.mark.parametrize("family", ("sdxl", "z_image", "wan21"))
@pytest.mark.skipif(
    not all(path.is_file() for path in _REAL_VAE_DTYPE_ARTIFACTS.values()),
    reason="official VAE dtype artifacts not present (set DINKSTER_VAE_DTYPE_MODELS)",
)
def test_real_bfloat16_vae_decode_matches_comfyui_on_cuda(family: str) -> None:
    from dinkster_inference import (
        ComponentPlan,
        Wan21VAEConfig,
        detect_kl_config,
        load_safetensors_header,
    )
    from dinkster_inference import assembly as assembly_mod
    from dinkster_inference_torch import Operations, kl_codec_plugin
    from dinkster_inference_torch import assemble as torch_assembly
    from dinkster_inference_torch.wan21_vae import WanVAE
    from dinkster_inference_torch.wan21_vae import WanVAEConfig as TorchWanVAEConfig

    reference = _VAE_DTYPE_GOLDENS["reference"]
    live_device = torch.cuda.get_device_name()
    if live_device != reference["device"]:
        pytest.skip(f"golden generated on {reference['device']}; live GPU is {live_device!r}")
    if torch.__version__ != reference["torch"]:
        pytest.skip(
            f"golden generated with torch {reference['torch']}; live torch is {torch.__version__}"
        )
    assert reference["commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert reference["vae_dtype"] == "bfloat16"
    assert reference["output_dtype"] == "float32"

    path = _REAL_VAE_DTYPE_ARTIFACTS[family]
    size, digest, url = _VAE_DTYPE_ARTIFACT_FACTS[family]
    assert _VAE_DTYPE_GOLDENS["artifacts"][family] == {
        "url": url,
        "byte_size": size,
        "sha256": digest,
    }
    assert path.stat().st_size == size
    with path.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == digest

    source = load_safetensors_header(path)
    if family == "sdxl":
        plan = assembly_mod.plan_sd_assembly(checkpoint=source).vae
        assert isinstance(plan, ComponentPlan)
        model = torch_assembly._load_component(  # pyright: ignore[reportPrivateUsage]
            plan,
            AutoencoderKL,
            compute_dtype=torch.bfloat16,
            fp8_matmul=False,
        )
    elif family == "z_image":
        extracted = assembly_mod._component_source(  # pyright: ignore[reportPrivateUsage]
            "vae", source, None, assembly_mod.FLUX_VAE_PREFIX
        )
        config = detect_kl_config(extracted.geometries)
        renames, transforms = assembly_mod._kl_conversion(  # pyright: ignore[reportPrivateUsage]
            extracted.geometries
        )
        plan = assembly_mod._plan(  # pyright: ignore[reportPrivateUsage]
            "vae", extracted, config, renames=renames, transforms=transforms
        )
        model = torch_assembly._load_component(  # pyright: ignore[reportPrivateUsage]
            plan,
            AutoencoderKL,
            compute_dtype=torch.bfloat16,
            fp8_matmul=False,
        )
    else:
        plan = assembly_mod.plan_wan21_standalone_component(source, "vae").component

        def build_wan_vae(config: Wan21VAEConfig, *, operations: Operations) -> WanVAE:
            return WanVAE(
                TorchWanVAEConfig(
                    dim=config.dim,
                    z_dim=config.z_dim,
                    dim_mult=config.dim_mult,
                    num_res_blocks=config.num_res_blocks,
                    attn_scales=config.attn_scales,
                    temporal_downsample=config.temporal_downsample,
                    image_channels=config.image_channels,
                    conv_out_channels=config.conv_out_channels,
                    dropout=config.dropout,
                ),
                operations=operations,
            )

        model = torch_assembly._load_component(  # pyright: ignore[reportPrivateUsage]
            plan,
            build_wan_vae,
            compute_dtype=torch.bfloat16,
            fp8_matmul=False,
        )

    baseline = _release_cuda_allocations()
    mechanism = enroll_component(model, load_device="cuda:0", offload_device="cpu")
    mechanism.partially_load(None)
    assert mechanism.loaded_bytes() == mechanism.total_bytes()
    latent = _vae_dtype_tensor(family, "latent").to("cuda:0", torch.bfloat16)
    with torch.no_grad():
        if isinstance(model, AutoencoderKL):
            decoded = kl_codec_plugin(model).decode(latent)
        else:
            decoded = ((model.decode(latent).float() + 1.0) / 2.0).clamp_(0.0, 1.0)
    assert decoded.dtype is torch.float32
    assert torch.equal(decoded.cpu(), _vae_dtype_tensor(family, "decoded"))

    mechanism.unload()
    del decoded, latent, mechanism, model
    assert _release_cuda_allocations() <= baseline + 64 * 1024**2


def _kitchen_int8_attention_available() -> bool:
    if not cuda_available:
        return False
    from dinkster_inference_torch import dinkster_kitchen_int8_available

    return dinkster_kitchen_int8_available()


requires_kitchen_int8_attention = pytest.mark.skipif(
    not _kitchen_int8_attention_available(),
    reason=(
        "dinkster-kitchen INT8 attention unavailable - the optimized H3"
        " attention route is NOT proven (needs the dinkster-kitchen CUDA"
        " backend; see README)"
    ),
)


def _sol_attention_available() -> bool:
    if not cuda_available:
        return False
    from dinkster_inference_torch import sol_attention_available

    return sol_attention_available()


requires_sol_attention = pytest.mark.skipif(
    not _sol_attention_available(),
    reason=(
        "dinkster-kitchen Sol attention unavailable - the sparse attention route is NOT proven"
        " (needs dinkster-kitchen 0.2.35.post1 CUDA on NVIDIA SM80+)"
    ),
)


def _sage2_attention_available() -> bool:
    if not cuda_available:
        return False
    from dinkster_inference_torch import sage2_attention_available

    return sage2_attention_available()


requires_sage2_attention = pytest.mark.skipif(
    not _sage2_attention_available(),
    reason=(
        "managed SageAttention unavailable - the quantized attention route is NOT proven"
        " (needs the dinkster-kitchen CUDA wheel; see README)"
    ),
)

_H3_ATTENTION_HEADS = 56
_H3_ATTENTION_HEAD_DIM = 128


def _h3_rope_table(sequence: int, rotary_dim: int) -> torch.Tensor:
    angles = torch.linspace(0.1, 0.7, sequence * (rotary_dim // 2)).reshape(
        1, sequence, 1, rotary_dim // 2
    )
    cosine = angles.cos()
    sine = angles.sin()
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, sequence, 1, rotary_dim // 2, 2, 2
    )


@requires_sol_attention
def test_sol_attention_executes_fused_backend_and_preserves_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import attention as attention_module
    from dinkster_inference_torch.attention import (
        attention_provider_identity,
        builtin_sdpa_kernel,
    )

    require_gpu_tests_enabled()
    executed = 0
    real_sol = attention_module._KITCHEN_SOL_ATTENTION  # pyright: ignore[reportPrivateUsage]

    def counting_sol(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal executed
        executed += 1
        return real_sol(*args, **kwargs)

    monkeypatch.setattr(attention_module, "_KITCHEN_SOL_ATTENTION", counting_sol)
    selection = select_attention("flux", "sol")
    assert selection.status.primary == "sol"
    assert attention_provider_identity(selection.status) == (
        SOL_ATTENTION_PROVIDER,
        "0.2.35.post1",
    )
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20_260_902)
    shape = (1, 8, 4096, 128)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    with torch.no_grad():
        actual = selection.kernel(q, k, v)
        provider = real_sol(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            tau=1.0,
            scale=None,
            tail=True,
        ).transpose(1, 2)
        reference = builtin_sdpa_kernel()(q, k, v)
    assert executed == 1
    assert actual.shape == q.shape
    assert actual.dtype is torch.bfloat16
    assert bool(torch.isfinite(actual).all())
    assert torch.equal(actual, provider)
    difference = actual.float() - reference.float()
    assert bool(torch.isfinite(difference).all())
    assert difference.abs().max().item() > 0.0

    small = tuple(tensor[:, :, :4] for tensor in (q, k, v))
    with torch.no_grad():
        fallback = selection.kernel(*small, causal=True)
        expected = builtin_sdpa_kernel()(*small, causal=True)
    assert executed == 1
    assert torch.equal(fallback, expected)
    del q, k, v, small, actual, provider, reference, difference, fallback, expected
    gc.collect()
    torch.cuda.empty_cache()


@requires_sol_attention
def test_scheduled_sol_h3_conditioning_sink_executes_exact_kv_range() -> None:
    from dinkster_inference import (
        AttentionModifierSchedule,
        SamplingTimelineSchedule,
        realize_sampling_timeline,
    )
    from dinkster_inference.sampling_timeline import use_realized_sampling_timeline
    from dinkster_inference_torch import MiniMaxH3PackedSequenceFacts
    from dinkster_inference_torch import attention as attention_module

    require_gpu_tests_enabled()
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20_260_904)
    shape = (1, 4, 1024, 128)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    facts = MiniMaxH3PackedSequenceFacts(
        1024,
        ((0, 32, "text"), (32, 64, "audio"), (64, 1024, "video")),
    )
    scheduled = attention_module.schedule_aware_attention_kernel(
        "sol", select_attention("flux", "sol").kernel
    )
    schedule = SamplingTimelineSchedule(
        "sol",
        0.0,
        1.0,
        attention_modifiers=(AttentionModifierSchedule("sol_conditioning_exact_kv", 0.0, 1.0),),
    )
    timeline = realize_sampling_timeline(schedule, (1.0, 0.0))

    with torch.no_grad(), use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        bound = attention_module.bind_packed_attention_kernel(scheduled, facts)
        actual = bound(q, k, v)
        provider = attention_module._KITCHEN_SOL_ATTENTION(  # pyright: ignore[reportPrivateUsage]
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            tau=1.0,
            scale=None,
            sink_blocks=[0, 1],
            tail=True,
        ).transpose(1, 2)

    assert torch.equal(actual, provider)
    del q, k, v, actual, provider
    gc.collect()
    torch.cuda.empty_cache()


def test_auto_vae_cuda_oom_uses_bounded_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_inference import AttentionCapabilityEvidence
    from dinkster_inference_torch import attention as attention_module

    require_gpu_tests_enabled()
    version = str(torch.__version__)
    evidence = AttentionCapabilityEvidence(
        version=1,
        device_kind="cuda",
        device_sm=torch.cuda.get_device_capability(0)[0] * 10
        + torch.cuda.get_device_capability(0)[1],
        sdpa_torch_runtime=version.split("+")[0],
        adapter_contract_revision=attention_module.ATTENTION_ADAPTER_CONTRACT,
        available_policies=("sdpa",),
        provider_versions=(("torch", version),),
    )

    def capabilities_probe(**_kwargs: object) -> AttentionCapabilityEvidence:
        return evidence

    monkeypatch.setattr(attention_module, "discover_attention_capabilities", capabilities_probe)
    token = attention_module.discover_attention_route_token("auto")
    assert token.version == 4
    selection = attention_module.resolve_role_attention("vae", "auto", token)
    assert selection.status.authenticated
    assert selection.status.primary == "sdpa"
    assert selection.status.fallback == "bounded"

    direct = attention_module._SDPA  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def oom_once(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise torch.OutOfMemoryError("injected SDPA OOM")
        return direct(*args, **kwargs)

    monkeypatch.setattr(attention_module, "_SDPA", oom_once)
    generator = torch.Generator(device="cuda:0").manual_seed(20_260_911)
    q = torch.randn((1, 2, 7, 16), device="cuda:0", generator=generator)
    k = torch.randn((1, 2, 9, 16), device="cuda:0", generator=generator)
    v = torch.randn((1, 2, 9, 12), device="cuda:0", generator=generator)
    with torch.no_grad():
        expected = direct(q, k, v)
        actual = selection.kernel(q, k, v)
    assert calls == 2
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@requires_sage2_attention
def test_managed_sageattention_executes_and_preserves_sdpa_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import version as distribution_version

    import sageattention  # pyright: ignore[reportMissingImports]
    from dinkster_inference_torch import attention as attention_module
    from dinkster_inference_torch.attention import attention_provider_identity, builtin_sdpa_kernel

    require_gpu_tests_enabled()
    assert sageattention.__distribution__ == "dinkster-kitchen"
    assert distribution_version("dinkster-kitchen") == "2.2.0.post1"

    executed = 0
    real_sage = attention_module._SAGE_ATTENTION  # pyright: ignore[reportPrivateUsage]

    def counting_sage(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal executed
        executed += 1
        return real_sage(*args, **kwargs)

    monkeypatch.setattr(attention_module, "_SAGE_ATTENTION", counting_sage)
    selection = select_attention("flux", "auto")
    assert selection.status.requested_policy == "auto"
    assert selection.status.primary == "sage"
    assert attention_provider_identity(selection.status) == (SAGE2_PROVIDER, "2.2.0.post1")

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20_260_901)
    shape = (2, 24, 4096, 128)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    with torch.no_grad():
        actual = selection.kernel(q, k, v)
        reference = builtin_sdpa_kernel()(q, k, v)
    assert executed == 1
    assert actual.shape == q.shape
    assert actual.dtype is torch.bfloat16
    assert bool(torch.isfinite(actual).all())
    difference = actual.float() - reference.float()
    relative_rms = difference.square().mean().sqrt() / reference.float().square().mean().sqrt()
    # The managed SM120 wheel measured 0.038670 relative RMS for this seed and
    # geometry. The cap leaves 55% headroom while rejecting SDPA fallback at 0.
    assert 0.0 < relative_rms.item() <= 0.06

    mask = torch.zeros((1, 1, shape[2], shape[2]), device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        fallback = selection.kernel(q, k, v, mask=mask)
        expected = builtin_sdpa_kernel()(q, k, v, mask=mask)
    assert executed == 1
    assert torch.equal(fallback, expected)
    del q, k, v, mask, actual, reference, difference, relative_rms, fallback, expected
    gc.collect()
    torch.cuda.empty_cache()


@requires_kitchen_int8_attention
def test_kitchen_int8_attention_matches_sdpa_on_h3_geometry() -> None:
    from dinkster_inference_torch.attention import builtin_sdpa_kernel

    selection = select_attention("flux", "dinkster_kitchen_int8")
    assert selection.status.primary == "dinkster_kitchen_int8"
    sdpa = builtin_sdpa_kernel()
    device = torch.device("cuda:0")
    # The per-seed MEAN drift is the primary contract: measured bf16 mean
    # drift on N(0,1) inputs is 0.000603 at S=1024 and 0.000329 at S=4096,
    # stable to ~1e-6 across seeds and SKUs (RTX PRO 6000 Blackwell and
    # RTX 5090, torch 2.13.0+cu130, dinkster-kitchen 0.2.31); the mean limits
    # keep >=32% headroom. The MAX drift is a per-element extreme value and
    # is seed- and SKU-sensitive: 20-seed sweeps observed worst-case
    # 0.019531 at S=1024 and 0.008789 at S=4096 (issue #845), so the max
    # limits are gross-defect caps sized >=28% above those extremes, not
    # tight parity bounds - a broken quantization path still trips them by
    # orders of magnitude while healthy extreme values never do.
    for sequence, max_cap, mean_bound in ((1024, 0.025, 0.0008), (4096, 0.012, 0.00045)):
        shape = (1, _H3_ATTENTION_HEADS, sequence, _H3_ATTENTION_HEAD_DIM)
        for seed in (sequence, sequence + 1, sequence + 2, sequence + 3):
            generator = torch.Generator(device=device).manual_seed(seed)
            q = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
            k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
            v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
            with torch.no_grad():
                expected = sdpa(q, k, v)
                actual = selection.kernel(q, k, v)
            assert actual.dtype is torch.bfloat16
            assert actual.shape == expected.shape
            difference = (actual.float() - expected.float()).abs()
            assert difference.max().item() <= max_cap, (sequence, seed)
            assert difference.mean().item() <= mean_bound, (sequence, seed)
            del q, k, v, expected, actual, difference
    gc.collect()
    torch.cuda.empty_cache()


@requires_kitchen_int8_attention
def test_kitchen_int8_attention_consume_matches_borrow_and_frees_fused_qkv() -> None:
    from dinkster_inference_torch.attention import (
        AttentionTensorLease,
        QkvConsumingAttentionKernel,
    )

    kernel = select_attention("flux", "dinkster_kitchen_int8").kernel
    assert isinstance(kernel, QkvConsumingAttentionKernel)
    device = torch.device("cuda:0")
    sequence = 4096
    inner = _H3_ATTENTION_HEADS * _H3_ATTENTION_HEAD_DIM
    generator = torch.Generator(device=device).manual_seed(sequence + 1)
    with torch.no_grad():
        fused = torch.randn(
            1, sequence, 3 * inner, device=device, dtype=torch.bfloat16, generator=generator
        )
        query, key, value = (
            part.view(1, sequence, _H3_ATTENTION_HEADS, _H3_ATTENTION_HEAD_DIM).transpose(1, 2)
            for part in fused.chunk(3, dim=-1)
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        expected = kernel(query, key, value)
        torch.cuda.synchronize()
        borrow_peak = torch.cuda.max_memory_allocated() - base

        leases = (
            AttentionTensorLease(query),
            AttentionTensorLease(key),
            AttentionTensorLease(value),
        )
        del query, key, value, fused
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        actual = kernel.consume(*leases)
        torch.cuda.synchronize()
        consume_peak = torch.cuda.max_memory_allocated() - base

    assert torch.equal(actual, expected)
    # Consuming prequantizes q/k/v to INT8 and frees the fused bf16 buffer
    # before attention runs. Peaks measured 88,367,616/147,087,872 bytes
    # (0.6008); the limit leaves 8% headroom.
    assert consume_peak / borrow_peak <= 0.65
    del actual, expected
    gc.collect()
    torch.cuda.empty_cache()


@requires_kitchen_int8_attention
def test_h3_attention_module_kitchen_route_matches_sdpa_and_lowers_peak() -> None:
    from dinkster_inference_torch import InitlessOperations
    from dinkster_inference_torch.attention import builtin_sdpa_kernel
    from dinkster_inference_torch.minimax_h3_dit import (
        BUILTIN_SDPA_PROVIDER,
        MiniMaxH3Attention,
        MiniMaxH3AttentionGeometry,
        MiniMaxH3AttentionProviderEvidence,
    )

    device = torch.device("cuda:0")
    geometry = MiniMaxH3AttentionGeometry(5376, _H3_ATTENTION_HEADS, _H3_ATTENTION_HEAD_DIM, 96)
    evidence = MiniMaxH3AttentionProviderEvidence(BUILTIN_SDPA_PROVIDER, str(torch.__version__))
    torch.manual_seed(20260827)
    model = MiniMaxH3Attention(
        geometry, builtin_sdpa_kernel(), evidence, operations=InitlessOperations()
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
    model = model.to(device=device, dtype=torch.bfloat16)
    kitchen = select_attention("flux", "dinkster_kitchen_int8").kernel

    sequence = 1024
    generator = torch.Generator(device=device).manual_seed(sequence)
    hidden = torch.randn(
        1, sequence, geometry.hidden_width, device=device, dtype=torch.bfloat16, generator=generator
    )
    table = _h3_rope_table(sequence, geometry.rotary_dim).to(device=device, dtype=torch.bfloat16)

    # Warm both kernels first so one-time CUDA workspace allocations are not
    # attributed to either measured pass.
    with torch.no_grad():
        for kernel in (None, kitchen):
            model(hidden, table.clone(), attention_kernel=kernel)
    torch.cuda.synchronize()

    peaks: dict[str, int] = {}
    outputs: dict[str, torch.Tensor] = {}
    for name, kernel in (("sdpa", None), ("kitchen", kitchen)):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            output = model(hidden, table.clone(), attention_kernel=kernel)
            torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - base
        outputs[name] = output.float().cpu()
        del output

    difference = (outputs["kitchen"] - outputs["sdpa"]).abs()
    # Full 56x128 H3 attention through the fused norm+rope inference path.
    # Measured max/mean drift is 0.003296/0.000534 against an SDPA output
    # abs-mean of 0.061; limits leave >=33% headroom.
    assert difference.max().item() <= 0.005
    assert difference.mean().item() <= 0.0008
    # The consuming kernel frees the fused QKV projection buffer before
    # attention output allocation, so its peak stays strictly below SDPA's.
    # Warm peaks measured 66,554,368/70,123,520 bytes (0.9491) on RTX PRO
    # 6000 Blackwell; under full-suite allocator state on RTX 4090/Windows
    # the kitchen pass gains a one-time 1,835,008-byte scratch allocation
    # (0.9753, issue #886) while the SDPA peak stays byte-identical. A
    # regression that keeps the 44,040,192-byte fused buffer alive across
    # the output allocation raises the ratio well above 1.0, so 0.99 keeps
    # the defect signal while absorbing that environmental scratch.
    assert peaks["kitchen"] / peaks["sdpa"] <= 0.99
    del model, hidden, table, outputs
    gc.collect()
    torch.cuda.empty_cache()


@requires_kitchen_int8_attention
def test_kitchen_int8_discovery_route_authenticates_flux_selection() -> None:
    from importlib.metadata import version as distribution_version

    from dinkster_inference_torch import resolve_role_attention

    token = discover_attention_route_token("dinkster_kitchen_int8")
    routes = {route.role: route.primary for route in token.routes}
    assert all(primary == "dinkster_kitchen_int8" for primary in routes.values())
    assert all(route.fallback == "sdpa" for route in token.routes)
    assert token.requested_policy == "dinkster_kitchen_int8"
    assert ("dinkster-kitchen", distribution_version("dinkster-kitchen")) in token.provider_versions
    assert token.device_kind == "cuda"

    selection = resolve_role_attention("flux", "dinkster_kitchen_int8", token)
    assert selection.status.authenticated is True
    assert selection.status.primary == "dinkster_kitchen_int8"
    assert selection.status.device_kind == "cuda"
    assert selection.status.provider_versions == token.provider_versions


@requires_kitchen_int8_attention
def test_kitchen_int8_attention_matches_sdpa_on_each_permitted_role_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import attention as attention_module
    from dinkster_inference_torch.attention import AttentionRole, builtin_sdpa_kernel

    executed = 0
    real_kitchen = attention_module._KITCHEN_ATTENTION  # pyright: ignore[reportPrivateUsage]

    def counting_kitchen(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal executed
        executed += 1
        return real_kitchen(*args, **kwargs)

    monkeypatch.setattr(attention_module, "_KITCHEN_ATTENTION", counting_kitchen)
    sdpa = builtin_sdpa_kernel()
    device = torch.device("cuda:0")
    # One representative INT8-executing geometry per non-diffusion role:
    # vae is the MiniMax H3 video VAE ViT3D decoder, clip is the CLIP ViT-L
    # vision tower (256 patches + class token), t5 is T5-XXL self-attention
    # with its relative position bias as an additive mask and scale=1.0
    # (T5 folds 1/sqrt(head_dim) into trained projections, so q/k are scaled
    # by head_dim**-0.25 for order-1 logits), and qwen is the Qwen Image DiT
    # joint attention over 77 text + 1024 image tokens.
    #
    # Per-seed MEAN drift is the primary contract; MAX is a gross-defect cap.
    # Bounds are minted from 20-seed sweeps (seeds 10000-10019) on N(0,1)
    # bf16 inputs (RTX PRO 6000 Blackwell, torch 2.13.0+cu130, dinkster-kitchen
    # 0.2.31), worst observed mean/max per geometry, >=28% headroom:
    #   vae  (32h x 64d,  S=1024): mean 0.000587 / max 0.019531
    #   clip (16h x 64d,  S=257):  mean 0.001101 / max 0.017578
    #   t5   (64h x 64d,  S=512):  mean 0.001268 / max 0.031250
    #   qwen (24h x 128d, S=1101): mean 0.000612 / max 0.011719
    cases: tuple[
        tuple[AttentionRole, tuple[int, int, int, int], bool, float | None, float, float], ...
    ] = (
        ("vae", (1, 32, 1024, 64), False, None, 0.0008, 0.025),
        ("clip", (1, 16, 257, 64), False, None, 0.0015, 0.023),
        ("t5", (1, 64, 512, 64), True, 1.0, 0.0017, 0.040),
        ("qwen", (1, 24, 1101, 128), False, None, 0.0008, 0.015),
    )
    for role, shape, masked, scale, mean_bound, max_cap in cases:
        selection = select_attention(role, "dinkster_kitchen_int8")
        assert selection.status.primary == "dinkster_kitchen_int8", role
        _, heads, sequence, head_dim = shape
        factor = head_dim**-0.25 if scale is not None else 1.0
        before = executed
        for seed in (10_000, 10_001, 10_002, 10_003):
            generator = torch.Generator(device=device).manual_seed(seed)
            q = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
            k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
            v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
            if factor != 1.0:
                q *= factor
                k *= factor
            mask = None
            if masked:
                mask = torch.randn(
                    (1, heads, sequence, sequence),
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
            with torch.no_grad():
                expected = sdpa(q, k, v, mask=mask, scale=scale)
                actual = selection.kernel(q, k, v, mask=mask, scale=scale)
            assert actual.dtype is torch.bfloat16
            difference = (actual.float() - expected.float()).abs()
            assert difference.mean().item() <= mean_bound, (role, seed)
            assert difference.max().item() <= max_cap, (role, seed)
            del q, k, v, mask, expected, actual, difference
        # Every seed must have executed the real INT8 kernel, not a fallback.
        assert executed - before == 4, role
    gc.collect()
    torch.cuda.empty_cache()


@requires_kitchen_int8_attention
def test_kitchen_int8_serves_kl_vae_wide_head_attention_through_sdpa_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import attention as attention_module
    from dinkster_inference_torch.attention import (
        AttentionTensorLease,
        QkvConsumingAttentionKernel,
        builtin_sdpa_kernel,
    )

    def refusing_kitchen(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("wide-head attention must not reach the INT8 kernel")

    monkeypatch.setattr(attention_module, "_KITCHEN_ATTENTION", refusing_kitchen)
    monkeypatch.setattr(attention_module, "_KITCHEN_PREQUANTIZE", refusing_kitchen)
    kernel = select_attention("vae", "dinkster_kitchen_int8").kernel
    sdpa = builtin_sdpa_kernel()
    device = torch.device("cuda:0")
    # The KL VAE mid-block runs single-head attention whose head dim is the
    # channel axis (512), above dinkster-kitchen's 256 head-dim limit.
    shape = (1, 1, 4096, 512)
    generator = torch.Generator(device=device).manual_seed(512)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    with torch.no_grad():
        expected = sdpa(q, k, v)
        borrowed = kernel(q, k, v)
        consuming = cast(QkvConsumingAttentionKernel, kernel)
        consumed = consuming.consume(
            AttentionTensorLease(q.clone()),
            AttentionTensorLease(k.clone()),
            AttentionTensorLease(v.clone()),
        )
    assert torch.equal(borrowed, expected)
    assert torch.equal(consumed, expected)
    del q, k, v, expected, borrowed, consumed
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not os.environ.get("DINKSTER_MOGE_MODEL"),
    reason="CUDA MoGe test requires DINKSTER_MOGE_MODEL",
)
def test_moge_v2_real_checkpoint_executes_on_cuda() -> None:
    from dinkster_inference_torch.checkpoint import load_checkpoint
    from dinkster_inference_torch.moge import MoGeModelV2, build_from_state_dict

    state = cast(dict[str, Any], load_checkpoint(Path(os.environ["DINKSTER_MOGE_MODEL"])))
    with torch.device("meta"):
        model = build_from_state_dict(state, operations=CastOperations(torch.float32)).eval()
    assert type(model) is MoGeModelV2
    model.to(device="cuda:0")
    image = torch.rand(
        (1, 3, 224, 280),
        generator=torch.Generator(device="cuda:0").manual_seed(179),
        device="cuda:0",
    )
    with torch.inference_mode():
        output = model(image, num_tokens=1200)
    assert {name: tuple(value.shape) for name, value in output.items()} == {
        "points": (1, 224, 280, 3),
        "mask": (1, 224, 280),
        "metric_scale": (1,),
        "normal": (1, 224, 280, 3),
    }
    assert all(value.device.type == "cuda" for value in output.values())
    assert all(torch.isfinite(value).all() for value in output.values())
    del state, model, image, output
    gc.collect()
    torch.cuda.empty_cache()
