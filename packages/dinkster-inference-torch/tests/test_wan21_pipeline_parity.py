"""Wan 2.1 cross-implementation acceptance against pinned ComfyUI.

``wan21_pipeline_goldens.json`` comes from executing ComfyUI commit b78cec87
through ``tools/gen_wan21_pipeline_goldens.py``. The replay uses identical
hash-filled tiny UMT5, Wan T2V/I2V, and causal Wan VAE state so it can compare the
public native boundaries without checking large model weights into the repo:

- real UMT5 SentencePiece token IDs -> 512-row text conditioning;
- five-frame causal VAE encode and two-frame latent decode;
- one T2V and one masked-reference I2V Euler FLOW step at Wan's shift 8.0,
  then VAE decode.

The full tensors are pinned, not probes. Torch-version-keyed payloads preserve
exact RNG references while the tolerance covers only CPU float32 accumulation
differences between equivalent attention and tensor-layout kernels.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
import torch
from clip_fill import fill_state_dict as clip_fill_state_dict
from dinkster_inference import WAN21, WAN21_SIGMAS, MultiStreamLatent, T5Config, Wan21Config
from dinkster_inference_torch import (
    AssembledWan21,
    T5TextModel,
    Wan21ClipVisionEncoder,
    Wan21Runtime,
    torch_sampler_registry,
    torch_scheduler_registry,
    umt5_tokenizer,
)
from dinkster_inference_torch.denoise import prepare_noise
from dinkster_inference_torch.sampling_execution import build_sampling_schedule
from dinkster_inference_torch.wan21_model import Wan21Model
from dinkster_inference_torch.wan21_vae import WanVAE, WanVAEConfig
from golden_files import platform_golden_path
from kl_fill import fill_state_dict as kl_fill_state_dict
from unet_fill import fill_state_dict as unet_fill_state_dict
from unet_fill import hashed_input

_GOLDEN_PATH = Path(__file__).parent / "goldens" / "wan21_pipeline_goldens.json"
_LEGACY_DEFAULT_PLATFORM = "win32"


def _golden_key(python_version: str, torch_version: str, platform_id: str) -> str:
    runtime = f"py{python_version}-torch{torch_version}"
    return runtime if platform_id.startswith("linux") else f"{platform_id}-{runtime}"


def _load_goldens() -> dict[str, Any]:
    python_version = platform.python_version()
    torch_version = str(torch.__version__)
    default: dict[str, Any] = json.loads(_GOLDEN_PATH.read_text())
    if (
        sys.platform == _LEGACY_DEFAULT_PLATFORM
        and str(default["source"]["python"]) == python_version
        and str(default["source"]["torch"]) == torch_version
    ):
        return default
    key = _golden_key(python_version, torch_version, sys.platform)
    versioned_path = platform_golden_path(_GOLDEN_PATH, key=key)
    versioned: dict[str, Any] = json.loads(versioned_path.read_text())
    assert str(versioned["source"]["python"]) == python_version
    assert str(versioned["source"]["torch"]) == torch_version
    if not sys.platform.startswith("linux"):
        assert str(versioned["source"]["platform"]) == sys.platform
        assert str(versioned["source"]["os"]) == platform.platform()
    return versioned


GOLDENS = _load_goldens()
RTOL = GOLDENS["tolerances"]["rtol"]
ATOL = GOLDENS["tolerances"]["atol"]
# ComfyUI torch 2.10 versus Dinkster torch 2.13 measured maximum absolute
# differences of 4.77e-7 (UMT5), 8.95e-8 (VAE), and 1.79e-6 (sampled
# latent). The 1e-5 absolute tolerance leaves 5.5x headroom at the widest seam.

UNDERSTOOD_DIFFERENCES = {
    "official_vae_torch_version": (
        "Float32 VAE encode/decode reach 0.001788/0.000456 max error under torch "
        "2.13 and are bit-exact under the reference torch 2.10 runtime."
    ),
    "official_pipeline_decode": (
        "The bit-exact sampled latent reaches 0.000159 max error after causal VAE "
        "decode under torch 2.13."
    ),
}

ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)
FillState = Callable[
    [Sequence[tuple[str, Sequence[int]]]],
    Mapping[str, torch.Tensor],
]


@pytest.mark.parametrize(
    ("platform_id", "expected"),
    (
        ("linux", "py3.12.3-torch2.13.0+cpu"),
        ("win32", "win32-py3.12.3-torch2.13.0+cpu"),
    ),
)
def test_wan21_pipeline_golden_key_includes_non_linux_platform(
    platform_id: str, expected: str
) -> None:
    assert _golden_key("3.12.3", "2.13.0+cpu", platform_id) == expected


def _tensor(payload: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


def _filled(module: ModuleT, fill_state: FillState, component: str) -> ModuleT:
    entries = sorted((key, list(value.shape)) for key, value in module.state_dict().items())
    assert [[key, shape] for key, shape in entries] == GOLDENS["state"][component]
    module.load_state_dict(dict(fill_state(entries)), strict=True)
    return module


class _SentencePiece:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def get_piece_size(self) -> int:
        return 256000

    def pad_id(self) -> int:
        return 0

    def eos_id(self) -> int:
        return 1

    def bos_id(self) -> int:
        return 2

    def unk_id(self) -> int:
        return 3

    def encode(
        self,
        text: str,
        *,
        out_type: type[int],
        add_bos: bool,
        add_eos: bool,
    ) -> list[int]:
        assert text == GOLDENS["conditioning"]["prompt"]
        assert out_type is int and not add_bos and not add_eos
        return list(GOLDENS["conditioning"]["prompt_ids"])


def _runtime(monkeypatch: pytest.MonkeyPatch, *, model_type: str = "t2v") -> Wan21Runtime:
    def import_sentencepiece(name: str) -> object:
        assert name == "sentencepiece"
        return SimpleNamespace(SentencePieceProcessor=_SentencePiece)

    monkeypatch.setattr(
        umt5_tokenizer.importlib,
        "import_module",
        import_sentencepiece,
    )
    architecture = GOLDENS["architecture"]
    wan = architecture["wan_i2v" if model_type == "i2v" else "wan"]
    vae_arch = architecture["vae"]
    diffusion = _filled(
        Wan21Model(
            Wan21Config(
                model_type=wan["model_type"],
                in_channels=wan["in_dim"],
                hidden_size=wan["dim"],
                ffn_hidden_size=wan["ffn_dim"],
                num_heads=wan["num_heads"],
                num_layers=wan["num_layers"],
                text_dim=wan["text_dim"],
                time_freq_dim=wan["freq_dim"],
                out_channels=wan["out_dim"],
                patch_size=tuple(wan["patch_size"]),
                qk_norm=wan["qk_norm"],
                cross_attn_norm=wan["cross_attn_norm"],
                eps=wan["eps"],
            )
        ),
        unet_fill_state_dict,
        "diffusion_i2v" if model_type == "i2v" else "diffusion",
    )
    text_model = _filled(
        T5TextModel(T5Config(**architecture["umt5"])),
        clip_fill_state_dict,
        "umt5",
    )
    vae = _filled(
        WanVAE(
            WanVAEConfig(
                dim=vae_arch["dim"],
                z_dim=vae_arch["z_dim"],
                dim_mult=tuple(vae_arch["dim_mult"]),
                num_res_blocks=vae_arch["num_res_blocks"],
                attn_scales=tuple(vae_arch["attn_scales"]),
                temporal_downsample=tuple(vae_arch["temporal_downsample"]),
                image_channels=vae_arch["image_channels"],
                conv_out_channels=vae_arch["conv_out_channels"],
                dropout=vae_arch["dropout"],
            )
        ),
        kl_fill_state_dict,
        "vae",
    )
    assembled = AssembledWan21(
        WAN21,
        diffusion,
        text_model,
        vae,
        b"pinned tokenizer supplied by test double",
        clip_vision=(
            cast("Wan21ClipVisionEncoder", torch.nn.Identity()) if model_type == "i2v" else None
        ),
        _component_compute_dtypes={
            "diffusion": torch.float32,
            "umt5xxl": torch.float32,
            "vae": torch.float32,
            **({"clip_vision": torch.float32} if model_type == "i2v" else {}),
        },
    )
    return Wan21Runtime(assembled, runtime_identity="native:test:wan21-pipeline")


def test_wan21_conditioning_matches_executed_comfyui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch)

    conditioning = runtime.encode_text(GOLDENS["conditioning"]["prompt"])

    assert conditioning.pooled is None
    torch.testing.assert_close(
        conditioning.embeddings,
        _tensor(GOLDENS["conditioning"]["embedding"]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_wan_runtime_family_labels_do_not_change_neural_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _runtime(monkeypatch)
    pipeline = GOLDENS["t2v_pipeline"]
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for family_id in ("dinkster.wan21", "dinkster.wan22", "example.renamed-video"):
        runtime = Wan21Runtime(
            replace(original.assembled, family=replace(WAN21, id=family_id)),
            runtime_identity=original.runtime_identity,
        )
        conditioning = runtime.encode_text(GOLDENS["conditioning"]["prompt"])
        sampled = runtime.sample_multistream(
            MultiStreamLatent.from_pairs((("video", _tensor(pipeline["initial_latent"])),)),
            conditioning=runtime.prepare_text_conditioning(conditioning),
            sampler_id=pipeline["sampler"],
            scheduler_id=pipeline["scheduler"],
            steps=pipeline["steps"],
            denoise=1.0,
            seed=pipeline["seed"],
            compute_dtype=torch.float32,
        ).by_role("video")
        actual = (conditioning.embeddings, sampled, runtime.decode_latent(sampled))
        if expected is None:
            expected = actual
        else:
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_wan21_multiframe_causal_vae_matches_executed_comfyui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch)
    golden = GOLDENS["causal_vae"]

    encoded = runtime.encode_content(_tensor(golden["content"]))
    decoded = runtime.decode_latent(_tensor(golden["latent"]))

    assert encoded.shape[2] == 2
    assert decoded.shape[2] == 5
    torch.testing.assert_close(encoded, _tensor(golden["encoded"]), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(decoded, _tensor(golden["decoded"]), rtol=RTOL, atol=ATOL)


def test_wan21_t2v_pipeline_matches_executed_comfyui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch)
    pipeline = GOLDENS["t2v_pipeline"]
    conditioning = runtime.encode_text(GOLDENS["conditioning"]["prompt"])
    initial = _tensor(pipeline["initial_latent"])
    sampler = torch_sampler_registry().get(pipeline["sampler"])
    scheduler = torch_scheduler_registry().get(pipeline["scheduler"])
    assert sampler is not None and scheduler is not None
    schedule = build_sampling_schedule(
        scheduler,
        WAN21_SIGMAS,
        sampler,
        pipeline["steps"],
        denoise=1.0,
        flow=True,
    )

    sampled_streams = runtime.sample_multistream(
        MultiStreamLatent.from_pairs((("video", initial),)),
        conditioning=runtime.prepare_text_conditioning(conditioning),
        sampler_id=pipeline["sampler"],
        scheduler_id=pipeline["scheduler"],
        steps=pipeline["steps"],
        denoise=1.0,
        seed=pipeline["seed"],
        compute_dtype=torch.float32,
    )
    sampled = sampled_streams.by_role("video")
    decoded = runtime.decode_latent(sampled)

    assert pipeline["flow_shift"] == 8.0
    assert list(schedule.sigmas) == pipeline["sigmas"] == [1.0, 0.0]
    assert torch.equal(prepare_noise(initial, pipeline["seed"]), _tensor(pipeline["noise"]))
    torch.testing.assert_close(
        sampled,
        _tensor(pipeline["sampled_latent"]),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        decoded,
        _tensor(pipeline["decoded_content"]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_wan21_i2v_pipeline_matches_executed_comfyui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch, model_type="i2v")
    pipeline = GOLDENS["i2v_pipeline"]
    conditioning = runtime.encode_text(GOLDENS["conditioning"]["prompt"])
    initial = _tensor(pipeline["initial_latent"])
    reference = runtime.encode_content(_tensor(pipeline["padded_content"]))
    torch.testing.assert_close(
        reference,
        _tensor(pipeline["reference_latent"]),
        rtol=RTOL,
        atol=ATOL,
    )
    concat_mask = _tensor(pipeline["concat_mask"])
    prepared_concat = torch.cat(((1.0 - concat_mask).repeat(1, 4, 1, 1, 1), reference), dim=1)
    vision = hashed_input("i2v_pipeline:vision", pipeline["vision_shape"])
    prepared = runtime.prepare_i2v_conditioning(
        runtime.prepare_text_conditioning(conditioning),
        prepared_concat,
        vision,
    )

    sampled_streams = runtime.sample_multistream(
        MultiStreamLatent.from_pairs((("video", initial),)),
        conditioning=prepared,
        sampler_id=pipeline["sampler"],
        scheduler_id=pipeline["scheduler"],
        steps=pipeline["steps"],
        denoise=1.0,
        seed=pipeline["seed"],
        compute_dtype=torch.float32,
    )
    sampled = sampled_streams.by_role("video")
    decoded = runtime.decode_latent(sampled)

    assert pipeline["flow_shift"] == 8.0
    assert pipeline["sigmas"] == [1.0, 0.0]
    assert torch.equal(prepare_noise(initial, pipeline["seed"]), _tensor(pipeline["noise"]))
    torch.testing.assert_close(
        sampled,
        _tensor(pipeline["sampled_latent"]),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        decoded,
        _tensor(pipeline["decoded_content"]),
        rtol=RTOL,
        atol=ATOL,
    )
