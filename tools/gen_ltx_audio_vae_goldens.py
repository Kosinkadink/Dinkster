"""Generate LTX audio VAE and vocoder goldens from ComfyUI.

Runs the REFERENCE AudioVAE (comfy/ldm/lightricks/vae/audio_vae.py,
causal_audio_autoencoder.py, and vocoders/vocoder.py @ the audited
baseline) and writes
packages/dinkster-inference-torch/tests/goldens/ltx_audio_vae_goldens.json.
dinkster_inference.ltx_audio (ltx_audio_vae_layout, ltx_vocoder_layout,
ltx_vocoder_bwe_layout) and dinkster_inference_torch.ltx_audio_vae are
pinned against these outputs; the oracle is the executed reference,
never a re-derivation.

Payload:

- "layouts": the sorted (key, shape) state-dict listing of the
  FULL-SIZE 19B audio VAE + vocoder, built on the meta device from the
  checkpoint's verbatim metadata config dicts (safetensors header of
  Lightricks/LTX-2 ltx-2-19b-dev.safetensors), exactly as the reference
  loads them.
- "properties": the reference wrapper's derived facts on that 19B
  model (sample rates, latent geometry, latent-count math).
- "transforms": the torchaudio-backed front end executed alone -
  windowed-sinc resampling and the log-mel spectrogram - so the
  torch-only Dinkster reimplementation can prove bit-equality.
- "cases": tiny architectures executed with deterministic hash-filled
  weights (ltx_vae_fill.py) covering encode (stereo and mono-expanded),
  decode to mel, the vocoder handoff, all three residual block kinds,
  both snake activations, and the bandwidth-extended vocoder.

The reference encode path needs torchaudio, so this generator pins the
torchaudio build alongside torch; both are recorded for provenance.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_ltx_audio_vae_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = (REPO.parent / "ComfyUI").resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

# In front of any ambient PYTHONPATH: the reference must come from the
# pinned sibling checkout, not an installed or stray comfy package.
sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))

# comfy.model_management probes CUDA at import; goldens execute on
# CPU float32 either way, so force ComfyUI's CPU state and keep the
# generator runnable from a CPU-only torch venv. The reference only
# reads argv when args parsing is explicitly enabled.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import torchaudio  # noqa: E402

#: Bit-stability holds only for one interpreter per platform: generation
#: REFUSES any other torch or torchaudio build so a regeneration cannot
#: silently rotate the payload hash through CPU-kernel drift. Linux uses
#: the +cpu wheels; macOS arm64 wheels carry no local version tag.
if sys.platform == "darwin":
    GENERATOR_TORCH = "2.13.0"
    GENERATOR_TORCHAUDIO = "2.11.0"
else:
    GENERATOR_TORCH = "2.13.0+cpu"
    GENERATOR_TORCHAUDIO = "2.11.0+cpu"
if torch.__version__ != GENERATOR_TORCH:
    raise SystemExit(
        f"goldens are pinned to torch {GENERATOR_TORCH}; this interpreter has"
        f" {torch.__version__}. Regenerating on another build rotates the payload"
        " hash - update the pin deliberately and re-prove bit-stability."
    )
if torchaudio.__version__ != GENERATOR_TORCHAUDIO:
    raise SystemExit(
        f"goldens are pinned to torchaudio {GENERATOR_TORCHAUDIO}; this interpreter"
        f" has {torchaudio.__version__}. Regenerating on another build rotates the"
        " payload hash - update the pin deliberately and re-prove bit-stability."
    )

from comfy.ldm.lightricks.vae import audio_vae as ref_audio  # noqa: E402
from comfy.ldm.lightricks.vocoders import vocoder as ref_vocoder  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402
from ltx_vae_fill import fill_vae_state_dict  # noqa: E402
from unet_fill import hashed_input  # noqa: E402

_GOLDENS_DIR = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens"
OUT = platform_golden_path(_GOLDENS_DIR / "ltx_audio_vae_goldens.json", torch.__version__)

#: The verbatim metadata config dicts of the published 19B checkpoint
#: (safetensors header of Lightricks/LTX-2 ltx-2-19b-dev.safetensors,
#: "audio_vae" and "vocoder" entries).
_19B_AUDIO_VAE_METADATA = {
    "preprocessing": {
        "audio": {
            "sampling_rate": 16000,
            "max_wav_value": 32768.0,
            "duration": 5.12,
            "stereo": True,
            "causal_padding": 3,
        },
        "stft": {
            "filter_length": 1024,
            "hop_length": 160,
            "win_length": 1024,
            "causal": True,
        },
        "mel": {"n_mel_channels": 64, "mel_fmin": 0, "mel_fmax": 8000},
    },
    "model": {
        "params": {
            "sampling_rate": 16000,
            "embed_dim": 8,
            "ddconfig": {
                "double_z": True,
                "mel_bins": 64,
                "z_channels": 8,
                "resolution": 256,
                "downsample_time": False,
                "in_channels": 2,
                "out_ch": 2,
                "ch": 128,
                "ch_mult": [1, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
                "mid_block_add_attention": False,
                "norm_type": "pixel",
                "causality_axis": "height",
            },
        }
    },
}

_19B_VOCODER_METADATA = {
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "stereo": True,
    "upsample_rates": [6, 5, 2, 2, 2],
    "resblock_kernel_sizes": [3, 7, 11],
    "upsample_kernel_sizes": [16, 15, 8, 4, 4],
    "resblock": "1",
    "upsample_initial_channel": 1024,
}


def _tiny_audio_vae_metadata(mel_bins: int, z_channels: int, ch: int) -> dict:
    """A reference config dict at tiny width; every ddconfig key the 19B
    checkpoint carries, so the reference constructor walks the same
    parsing path."""
    return {
        "preprocessing": {
            "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
        },
        "model": {
            "params": {
                "sampling_rate": 16000,
                "embed_dim": z_channels,
                "ddconfig": {
                    "double_z": True,
                    "mel_bins": mel_bins,
                    "z_channels": z_channels,
                    "resolution": 256,
                    "downsample_time": False,
                    "in_channels": 2,
                    "out_ch": 2,
                    "ch": ch,
                    "ch_mult": [1, 2, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                    "mid_block_add_attention": False,
                    "norm_type": "pixel",
                    "causality_axis": "height",
                },
            }
        },
    }


#: Tiny vocoder attached to the encode case only for construction; the
#: decode case actually runs its vocoder.
_TINY_VOCODER = {
    "resblock": "1",
    "stereo": True,
    "upsample_rates": [2, 2],
    "upsample_kernel_sizes": [4, 4],
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3, 5]],
    "upsample_initial_channel": 32,
}

#: Dinkster-vocabulary geometry of the tiny audio VAEs (the replay tests
#: rebuild LTXAudioVAEConfig from "config").
_TINY_ENCODE_DINKSTER = {
    "in_channels": 2,
    "out_channels": 2,
    "base_channels": 8,
    "ch_mult": [1, 2, 4],
    "num_res_blocks": 2,
    "z_channels": 4,
    "mel_bins": 16,
    "sampling_rate": 16000,
    "mel_hop_length": 160,
    "n_fft": 1024,
}

_TINY_DECODE_DINKSTER = {**_TINY_ENCODE_DINKSTER, "mel_bins": 64}

#: name -> (reference vocoder config, input shape). The Dinkster
#: vocabulary carries the same keys as tuples, rebuilt in the tests.
VOCODER_CASES = {
    "vocoder_resblock1_stereo": (
        {
            "resblock": "1",
            "stereo": True,
            "upsample_rates": [2, 2],
            "upsample_kernel_sizes": [4, 4],
            "resblock_kernel_sizes": [3, 5],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
            "upsample_initial_channel": 32,
        },
        (1, 2, 64, 7),
    ),
    "vocoder_resblock2_mono": (
        {
            "resblock": "2",
            "stereo": False,
            "upsample_rates": [2],
            "upsample_kernel_sizes": [4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3]],
            "upsample_initial_channel": 16,
            "use_tanh_at_final": False,
        },
        (1, 64, 9),
    ),
    "vocoder_amp1_snake_stereo": (
        {
            "resblock": "AMP1",
            "activation": "snake",
            "stereo": True,
            "upsample_rates": [2],
            "upsample_kernel_sizes": [4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3, 5]],
            "upsample_initial_channel": 16,
        },
        (1, 2, 64, 5),
    ),
    "vocoder_amp1_snakebeta_mono": (
        {
            "resblock": "AMP1",
            "activation": "snakebeta",
            "stereo": False,
            "upsample_rates": [2, 2],
            "upsample_kernel_sizes": [4, 4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3, 5]],
            "upsample_initial_channel": 16,
            "use_bias_at_final": False,
            "use_tanh_at_final": False,
        },
        (1, 64, 6),
    ),
}

_BWE_CONFIG = {
    "vocoder": {
        "resblock": "1",
        "stereo": True,
        "upsample_rates": [2, 2],
        "upsample_kernel_sizes": [4, 4],
        "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1, 3, 5]],
        "upsample_initial_channel": 32,
    },
    "bwe": {
        "resblock": "1",
        "stereo": True,
        "upsample_rates": [4, 4, 2],
        "upsample_kernel_sizes": [8, 8, 4],
        "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1, 3, 5]],
        "upsample_initial_channel": 32,
        "input_sampling_rate": 8000,
        "output_sampling_rate": 16000,
        "hop_length": 16,
        "n_fft": 64,
        "num_mels": 64,
    },
}


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


def build_reference(
    audio_vae_metadata: dict, vocoder_metadata: dict, device: str
) -> ref_audio.AudioVAE:
    metadata = {"config": {"audio_vae": audio_vae_metadata, "vocoder": vocoder_metadata}}
    if device == "meta":
        with torch.device("meta"):
            return ref_audio.AudioVAE(metadata=metadata)
    model = ref_audio.AudioVAE(metadata=metadata)
    # The reference processor hardcodes the published statistics width
    # (128); tiny cases re-register the buffers at their patch width.
    ddconfig = audio_vae_metadata["model"]["params"]["ddconfig"]
    channels = ddconfig["z_channels"] * (ddconfig["mel_bins"] // ref_audio.LATENT_DOWNSAMPLE_FACTOR)
    if channels != 128:
        statistics = model.autoencoder.per_channel_statistics
        statistics.register_buffer("std-of-means", torch.empty(channels))
        statistics.register_buffer("mean-of-means", torch.empty(channels))
    return model


def entries_of(module: torch.nn.Module) -> list[tuple[str, list[int]]]:
    return sorted((key, list(value.shape)) for key, value in module.state_dict().items())


def fill(module: torch.nn.Module) -> list[tuple[str, list[int]]]:
    entries = entries_of(module)
    module.load_state_dict(fill_vae_state_dict(entries), strict=True)
    return entries


def audio_vae_cases(payload: dict) -> None:
    encode_metadata = _tiny_audio_vae_metadata(mel_bins=16, z_channels=4, ch=8)
    model = build_reference(encode_metadata, _TINY_VOCODER, "cpu")
    entries = fill(model)
    x = hashed_input("audio_vae_tiny_encode:x", (1, 2, 8820))
    x_mono = hashed_input("audio_vae_tiny_encode:x_mono", (1, 1, 8820))
    with torch.no_grad():
        latent = model.encode(x, sample_rate=44100)
        latent_mono = model.encode(x_mono, sample_rate=44100)
    latent_input = hashed_input("audio_vae_tiny_encode:latent", list(latent.shape))
    with torch.no_grad():
        mel = model.autoencoder.decode(
            model.normalizer.denormalize(latent_input),
            target_shape=model.target_shape_from_latents(latent_input.shape),
        )
    payload["cases"]["audio_vae_tiny_encode"] = {
        "config": _TINY_ENCODE_DINKSTER,
        "reference_config": encode_metadata,
        "state_dict": entries,
        "input_shape": [1, 2, 8820],
        "input_sample_rate": 44100,
        "encode": enc(latent),
        "encode_mono": enc(latent_mono),
        "latent_shape": list(latent.shape),
        "decode_mel": enc(mel),
    }

    decode_metadata = _tiny_audio_vae_metadata(mel_bins=64, z_channels=4, ch=8)
    model = build_reference(decode_metadata, _TINY_VOCODER, "cpu")
    entries = fill(model)
    x = hashed_input("audio_vae_tiny_decode:x", (1, 2, 8820))
    latent_input = hashed_input("audio_vae_tiny_decode:latent", (1, 4, 5, 16))
    with torch.no_grad():
        latent = model.encode(x, sample_rate=44100)
        mel = model.autoencoder.decode(
            model.normalizer.denormalize(latent_input),
            target_shape=model.target_shape_from_latents(latent_input.shape),
        )
        waveform = model.run_vocoder(mel)
    payload["cases"]["audio_vae_tiny_decode"] = {
        "config": _TINY_DECODE_DINKSTER,
        "reference_config": decode_metadata,
        "vocoder_reference_config": _TINY_VOCODER,
        "state_dict": entries,
        "input_shape": [1, 2, 8820],
        "input_sample_rate": 44100,
        "encode": enc(latent),
        "latent_shape": [1, 4, 5, 16],
        "decode_mel": enc(mel),
        "decode_waveform": enc(waveform),
    }


def vocoder_cases(payload: dict) -> None:
    for name, (config, input_shape) in sorted(VOCODER_CASES.items()):
        model = ref_vocoder.Vocoder(config=config)
        computed_filters = {
            key: enc(value) for key, value in model.state_dict().items() if key.endswith(".filter")
        }
        entries = fill(model)
        x = hashed_input(f"{name}:x", input_shape)
        with torch.no_grad():
            waveform = model(x)
        payload["cases"][name] = {
            "reference_config": config,
            "state_dict": entries,
            "input_shape": list(input_shape),
            "computed_filters": computed_filters,
            "output": enc(waveform),
        }

    model = ref_vocoder.VocoderWithBWE(config=_BWE_CONFIG)
    # The skip resampler's filter is non-persistent (never in a state
    # dict or checkpoint); record the constructed value directly.
    computed_filters = {"resampler.filter": enc(model.resampler.filter)}
    entries = fill(model)
    x = hashed_input("vocoder_bwe:x", (1, 2, 64, 8))
    with torch.no_grad():
        waveform = model(x)
    payload["cases"]["vocoder_bwe"] = {
        "reference_config": _BWE_CONFIG,
        "state_dict": entries,
        "input_shape": [1, 2, 64, 8],
        "computed_filters": computed_filters,
        "output": enc(waveform),
    }


def transforms(payload: dict) -> None:
    for name, (input_shape, source_rate, target_rate) in sorted(
        {
            "resample_44100_to_16000": ((1, 2, 4410), 44100, 16000),
            "resample_24000_to_16000": ((1, 2, 2400), 24000, 16000),
        }.items()
    ):
        x = hashed_input(f"{name}:x", input_shape)
        with torch.no_grad():
            resampled = torchaudio.functional.resample(x, source_rate, target_rate)
        payload["transforms"][name] = {
            "input_shape": list(input_shape),
            "source_rate": source_rate,
            "target_rate": target_rate,
            "output": enc(resampled),
        }

    preprocessor = ref_audio.AudioPreprocessor(
        target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024
    )
    for name, (input_shape, source_rate) in sorted(
        {
            "mel_16000": ((1, 2, 3360), 16000),
            "mel_44100": ((1, 2, 8820), 44100),
        }.items()
    ):
        x = hashed_input(f"{name}:x", input_shape)
        with torch.no_grad():
            mel = preprocessor.waveform_to_mel(x, source_rate, device="cpu")
        payload["transforms"][name] = {
            "input_shape": list(input_shape),
            "source_rate": source_rate,
            "output": enc(mel),
        }


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != REFERENCE_COMMIT:
        raise SystemExit(
            f"{COMFY_ROOT} is at {commit}; goldens must be generated"
            f" from the audited baseline {REFERENCE_COMMIT}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"{COMFY_ROOT} has uncommitted changes; a clean checkout"
            f" of {REFERENCE_COMMIT} is required:\n{dirty}"
        )
    for module in (ref_audio, ref_vocoder):
        module_file = Path(module.__file__ or "").resolve()
        if not module_file.is_relative_to(COMFY_ROOT):
            raise SystemExit(
                "the reference audio VAE was imported from"
                f" {module_file}, not the pinned checkout {COMFY_ROOT}"
            )

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            **tuple_provenance(torch.__version__),
        },
        "layouts": {},
        "properties": {},
        "transforms": {},
        "cases": {},
    }

    model = build_reference(_19B_AUDIO_VAE_METADATA, _19B_VOCODER_METADATA, "meta")
    payload["layouts"]["ltxav_19b_audio_vae"] = entries_of(model)
    payload["properties"] = {
        "latents_per_second": model.latents_per_second,
        "output_sample_rate": model.output_sample_rate,
        "latent_channels": model.latent_channels,
        "mel_bins": model.mel_bins,
        "latent_frequency_bins": model.latent_frequency_bins,
        "sample_rate": model.sample_rate,
        "mel_hop_length": model.mel_hop_length,
        "num_of_latents": [
            [frames, rate, model.num_of_latents_from_frames(frames, rate)]
            for frames, rate in ((97, 24), (121, 24), (48, 12), (1, 25))
        ],
    }

    transforms(payload)
    audio_vae_cases(payload)
    vocoder_cases(payload)

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
