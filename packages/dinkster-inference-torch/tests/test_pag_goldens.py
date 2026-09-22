# pyright: basic
"""Replay the pinned ComfyUI PAG cases through Dinkster's sampling seam."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    SD15,
    ClipTextConfig,
    Conditioning,
    CustomSamplingRequest,
    KLConfig,
    SamplingGuidance,
    UNetConfig,
    builtin_sampler_registry,
    use_sampling_environment,
)
from dinkster_inference_torch import (
    AssembledSD,
    AutoencoderKL,
    ClipTextModel,
    GuidanceExecutor,
    GuidanceRegistry,
    SDRuntime,
    UNetModel,
    prepare_noise,
)
from golden_files import load_platform_golden
from unet_fill import fill_state_dict

PAG_GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "pag_goldens_e638023d.json")
PROOF_PACK = Path(__file__).parents[3] / "tests" / "fixtures" / "pag-pack" / "pag_pack.py"


def golden_tensor(case: dict[str, Any], name: str) -> torch.Tensor:
    value = cast("dict[str, Any]", case[name])
    return torch.tensor(value["data"], dtype=torch.float32).reshape(value["shape"])


@pytest.fixture(scope="module")
def pag_assembled_sd() -> AssembledSD:
    first_case = cast("dict[str, Any]", next(iter(PAG_GOLDENS["cases"].values())))
    config = dict(first_case["config"])
    for name in (
        "image_size",
        "use_checkpoint",
        "use_spatial_transformer",
        "legacy",
        "use_temporal_attention",
        "use_temporal_resblock",
        "num_head_channels",
    ):
        config.pop(name)
    for name in (
        "num_res_blocks",
        "channel_mult",
        "transformer_depth",
        "transformer_depth_output",
    ):
        config[name] = tuple(config[name])
    diffusion = UNetModel(UNetConfig(**config))
    diffusion.load_state_dict(fill_state_dict(first_case["state_dict"]), strict=True)
    clip = ClipTextModel(
        ClipTextConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            hidden_act="quick_gelu",
            vocab_size=1,
            eos_token_id=0,
        )
    )
    vae = AutoencoderKL(
        KLConfig(
            in_channels=3,
            out_channels=3,
            ch=32,
            decoder_ch=32,
            ch_mult=(1, 2),
            num_res_blocks=1,
            z_channels=4,
            embed_dim=4,
        )
    )
    return AssembledSD(SD15, diffusion, clip, None, vae)


@pytest.mark.parametrize("case_name", tuple(PAG_GOLDENS["cases"]))
def test_pag_proof_pack_matches_comfyui_golden(
    case_name: str,
    pag_assembled_sd: AssembledSD,
) -> None:
    case = cast("dict[str, Any]", PAG_GOLDENS["cases"][case_name])
    params = cast("dict[str, Any]", case["params"])
    make = runpy.run_path(str(PROOF_PACK))["make"]
    contribution = make(
        scale=params["pag_scale"],
        torch_version="2.13.0+cpu",
        aimdo_version="0.5.5.post2",
    )
    executor = GuidanceExecutor(
        GuidanceRegistry(
            (("proof_pag", contribution.guidance),),
            attention_contributions=(("proof_pag", contribution.attention),),
        )
    )
    runtime = SDRuntime(
        pag_assembled_sd,
        runtime_identity="native:test:pag-golden",
        guidance_executor=executor,
    )
    sampler = builtin_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    latent = golden_tensor(case, "latent_image")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(PAG_GOLDENS["reference"]["torch_num_threads"])
    try:
        with use_sampling_environment(("proof_pag",), lambda: False):
            result = runtime.sample_custom(
                latent,
                noise=prepare_noise(latent, params["seed"]),
                cond=Conditioning(golden_tensor(case, "positive_context")),
                cfg=SamplingGuidance(
                    Conditioning(golden_tensor(case, "negative_context")),
                    params["cfg"],
                ),
                request=CustomSamplingRequest(sampler, (), tuple(case["sigmas"])),
                seed=params["seed"],
                compute_dtype=torch.float32,
                device="cpu",
            ).output
    finally:
        torch.set_num_threads(previous_threads)

    assert torch.equal(result, golden_tensor(case, "output"))
    if params["pag_scale"] == 0:
        assert torch.equal(result, golden_tensor(case, "output_baseline"))
        assert torch.equal(result, golden_tensor(case, "output_scale_zero"))
