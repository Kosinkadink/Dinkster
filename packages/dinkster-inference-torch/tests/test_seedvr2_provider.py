from __future__ import annotations

import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference_torch import SeedVR2DiffusionRuntime, seedvr2_conditioning
from dinkster_inference_torch.checkpoint_runtime import (
    ComponentAssembly,
    ComponentCheckpointRuntime,
)

PACKAGES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGES / "dinkster-nodes-generation" / "src"))
sys.path.insert(0, str(PACKAGES / "dinkster-compat-comfy" / "src"))
arm = cast("Any", importlib.import_module("dinkster_compat_comfy.native_arm"))
GOLDENS = json.loads((Path(__file__).parent / "goldens" / "seedvr2_goldens.json").read_text())


def _decode(payload: dict[str, object]) -> torch.Tensor:
    data = cast("list[float]", payload["data"])
    shape = cast("list[int]", payload["shape"])
    return torch.tensor(data, dtype=torch.float32).reshape(shape)


@pytest.fixture(autouse=True)
def _torch_runtime(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arm, "_torch", lambda: torch)


def test_preprocess_pads_video_geometry_and_refuses_zero_frames() -> None:
    images = torch.linspace(0.0, 1.0, 2 * 17 * 18 * 4).reshape(2, 17, 18, 4)

    output = cast(
        "torch.Tensor",
        arm.GenerationSeedVR2Preprocess.execute(resized_images=images)["images"],
    )

    assert output.shape == (1, 5, 32, 32, 3)
    assert torch.equal(output[0, :2, :17, :18], images[..., :3])
    assert torch.equal(output[:, 2:], output[:, 1:2].expand(-1, 3, -1, -1, -1))
    with pytest.raises(ValueError, match="at least one frame"):
        arm.GenerationSeedVR2Preprocess.execute(resized_images=torch.empty((0, 16, 16, 3)))


def test_preprocessed_video_reaches_seedvr2_vae_in_native_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Codec:
        descriptor = SimpleNamespace(kind="video")
        load_device = torch.device("cpu")
        accepts_batched_video = True
        manages_input_device = True

        def __init__(self) -> None:
            self.encoded: torch.Tensor | None = None

        def stage(self):
            return torch.inference_mode()

        def encode_content(self, content: torch.Tensor) -> torch.Tensor:
            self.encoded = content
            return torch.zeros((1, 16, 2, 4, 4))

    codec = Codec()

    def component_codec(_value: object) -> Codec:
        return codec

    monkeypatch.setattr(arm, "NativeComponentHandle", SimpleNamespace)
    monkeypatch.setattr(arm, "_native_component_codec", component_codec)
    images = torch.rand((5, 17, 18, 3), generator=torch.Generator().manual_seed(4))
    preprocessed = arm.GenerationSeedVR2Preprocess.execute(resized_images=images)["images"]

    latent = cast(
        "dict[str, torch.Tensor]",
        arm.GenerationVAEEncode.execute(pixels=preprocessed, vae=SimpleNamespace())["latent"],
    )

    assert codec.encoded is not None
    assert codec.encoded.shape == (1, 3, 5, 32, 32)
    assert latent["samples"].shape == (1, 16, 2, 4, 4)


def test_ordinary_images_round_trip_through_seedvr2_codec_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import SEEDVR2_CODEC

    class Handle:
        pass

    class Codec:
        descriptor = SEEDVR2_CODEC
        load_device = torch.device("cuda")
        accepts_batched_video = True
        accepts_image_batch_latent = True
        manages_input_device = True

        def stage(self):
            return torch.inference_mode()

        @staticmethod
        def encode_content(content: torch.Tensor) -> torch.Tensor:
            assert content.shape == (2, 3, 16, 24)
            return torch.zeros((2, 16, 2, 3))

        @staticmethod
        def decode_latent(latent: torch.Tensor) -> torch.Tensor:
            assert latent.shape == (2, 16, 2, 3)
            return torch.zeros((2, 3, 16, 24))

    codec = Codec()

    def component_codec(_value: object) -> Codec:
        return codec

    monkeypatch.setattr(arm, "NativeComponentHandle", Handle)
    monkeypatch.setattr(arm, "_native_component_codec", component_codec)
    encoded = arm.GenerationVAEEncode.execute(
        pixels=torch.zeros((2, 16, 24, 3)),
        vae=Handle(),
    )
    decoded = arm.GenerationVAEDecode.execute(samples=encoded["latent"], vae=Handle())

    assert cast("torch.Tensor", cast("dict[str, object]", encoded["latent"])["samples"]).shape == (
        2,
        16,
        2,
        3,
    )
    assert cast("torch.Tensor", decoded["image"]).shape == (2, 16, 24, 3)


def test_postprocess_crops_even_geometry_and_restores_alpha() -> None:
    decoded = torch.rand((3, 19, 21, 3), generator=torch.Generator().manual_seed(1))
    reference = torch.rand((2, 17, 18, 4), generator=torch.Generator().manual_seed(2))

    output = cast(
        "torch.Tensor",
        arm.GenerationSeedVR2PostProcessing.execute(
            images=decoded,
            original_resized_images=reference,
            color_correction_method="none",
        )["images"],
    )

    assert output.shape == (2, 16, 18, 4)
    assert torch.equal(output[..., :3], decoded[:2, :16, :18])
    assert torch.equal(output[..., 3:], reference[:2, :16, :18, 3:4])


def test_color_transfer_chunks_copy_into_one_preallocated_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
    reference = torch.ones_like(decoded)

    def transfer(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return content + style

    fake_inference = SimpleNamespace(
        lab_color_transfer=transfer,
        wavelet_color_transfer=transfer,
        adain_color_transfer=transfer,
    )

    def select_device(_torch: object) -> torch.device:
        return torch.device("cpu")

    def chunk_size(*_args: object) -> int:
        return 2

    def fail_cat(*_args: object, **_kwargs: object) -> None:
        pytest.fail("must not concatenate")

    monkeypatch.setattr(arm, "select_load_device", select_device)
    monkeypatch.setattr(arm, "_seedvr2_color_chunk_size", chunk_size)
    monkeypatch.setattr(torch, "cat", fail_cat)

    output = arm._seedvr2_color_transfer(  # pyright: ignore[reportPrivateUsage]
        decoded, reference, "adain", torch, fake_inference
    )

    assert torch.equal(output, decoded + reference)


def _checkpoint_runtime(diffusion: SeedVR2DiffusionRuntime) -> ComponentCheckpointRuntime:
    assembled = ComponentAssembly(
        diffusion.family,
        "diffusion",
        {"diffusion": diffusion.assembled.diffusion},
        {"diffusion": torch.float32},
        {},
    )
    return ComponentCheckpointRuntime(diffusion, assembled)


@pytest.mark.parametrize("image_batch", [False, True], ids=["video", "images"])
@pytest.mark.parametrize("tiled", [False, True])
def test_checkpoint_codec_nodes_preserve_video_batch_and_defer_device_transfer(
    tiled: bool, image_batch: bool
) -> None:
    from dinkster_inference import (
        ReconstructionRecipe,
        RuntimeCodecAdapter,
        RuntimeKnobs,
        WeightSourceBinding,
        WeightSourceRef,
    )
    from dinkster_inference_torch.seedvr2_runtime import checkpoint_codec
    from test_seedvr2_runtime import RecordingVAE

    recipe = ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "checkpoint",
                WeightSourceRef(
                    digest="blake3:" + "0" * 64,
                    name="seedvr2.safetensors",
                    size=1,
                ),
            ),
        ),
        family_id="dinkster.seedvr2",
        component_identity=("family=dinkster.seedvr2",),
        knobs=RuntimeKnobs(
            diffusion_dtype="float32", text_dtype="float32", vae_dtype="float16", fp8_matmul=False
        ),
    )
    diffusion = SeedVR2DiffusionRuntime(
        cast("Any", torch.nn.Identity()),
        runtime_identity=recipe.runtime_identity,
        compute_dtype=torch.float32,
    )
    vae = RecordingVAE()
    assembled = ComponentAssembly(
        diffusion.family,
        "diffusion",
        {"diffusion": diffusion.assembled.diffusion, "vae": vae},
        {"diffusion": torch.float32, "vae": torch.float16},
        {},
    )
    vae.anchor = torch.nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)
    codec = checkpoint_codec(assembled)
    assert codec is not None
    runtime = ComponentCheckpointRuntime(diffusion, assembled, codec=codec)
    stages: list[str] = []

    def stage(role: str) -> nullcontext[None]:
        stages.append(role)
        return nullcontext()

    handle: Any = SimpleNamespace(
        recipe=recipe,
        runtime=runtime,
        load_device=torch.device("meta"),
        require_active=lambda: None,
        stage=stage,
    )
    adapter: RuntimeCodecAdapter[torch.Tensor] = RuntimeCodecAdapter(
        handle, descriptor=codec.descriptor
    )
    pixels = torch.full((2, 16, 16, 3) if image_batch else (2, 1, 16, 16, 3), 0.75)
    if tiled:
        encoded = arm.GenerationVAEEncodeTiled.execute(
            pixels=pixels,
            vae=adapter,
            tile_size=32,
            overlap=8,
            temporal_size=8,
            temporal_overlap=2,
        )
    else:
        encoded = arm.GenerationVAEEncode.execute(pixels=pixels, vae=adapter)
    latent = encoded["latent"]
    assert latent["samples"].shape == ((2, 16, 2, 2) if image_batch else (2, 16, 1, 2, 2))
    assert vae.encode_inputs[0].shape == (2, 3, 1, 16, 16)
    assert vae.encode_inputs[0].dtype is torch.float16
    assert vae.encode_inputs[0].device.type == "cpu"
    if tiled:
        decoded = arm.GenerationVAEDecodeTiled.execute(
            samples=latent,
            vae=adapter,
            tile_size=32,
            overlap=8,
            temporal_size=8,
            temporal_overlap=2,
        )
    else:
        decoded = arm.GenerationVAEDecode.execute(samples=latent, vae=adapter)
    assert vae.decode_inputs[0][0].shape == (2, 16, 1, 2, 2)
    assert vae.decode_inputs[0][0].dtype is torch.float16
    assert vae.decode_inputs[0][0].device.type == "cpu"
    assert decoded["image"].shape == (2, 16, 16, 3)
    torch.testing.assert_close(decoded["image"], torch.full((2, 16, 16, 3), 0.375), rtol=0, atol=0)
    assert stages == ["vae", "vae"]


@pytest.mark.parametrize("combined", [False, True])
def test_conditioning_provider_binds_the_loaded_diffusion_identity(
    monkeypatch: pytest.MonkeyPatch,
    combined: bool,
) -> None:
    identity = "native:dinkster.seedvr2:" + "1" * 64

    class ExtendedSeedVR2Runtime(SeedVR2DiffusionRuntime):
        pass

    runtime = ExtendedSeedVR2Runtime(
        cast("Any", torch.nn.Identity()),
        runtime_identity=identity,
        compute_dtype=torch.float32,
    )
    handle = SimpleNamespace(
        runtime=_checkpoint_runtime(runtime) if combined else runtime,
        load_device=torch.device("cpu"),
        recipe=SimpleNamespace(runtime_identity=identity),
    )

    def application_chain(model: object, _name: str) -> tuple[object, tuple[()]]:
        return model, ()

    def native_model(_model: object, _name: str) -> tuple[object, ...]:
        return handle, (), {}, None, None, (), None, {}

    monkeypatch.setattr(arm, "_application_chain_model", application_chain)
    monkeypatch.setattr(arm, "_native_model", native_model)
    latent = torch.rand((1, 16, 2, 3, 4), generator=torch.Generator().manual_seed(3))

    output = arm.GenerationSeedVR2Conditioning.execute(
        model=handle,
        vae_conditioning={"samples": latent},
    )

    positive = cast("list[list[object]]", output["positive"])[0]
    negative = cast("list[list[object]]", output["negative"])[0]
    assert torch.equal(cast("torch.Tensor", positive[0])[:, :16], latent)
    for row, branch in ((positive, "positive"), (negative, "negative")):
        prepared = cast("Any", next(iter(cast("dict[str, object]", row[1]).values())))
        assert prepared.component_identity == identity
        assert prepared.branch == branch


def test_custom_sampling_route_uses_matching_wrapped_seedvr2_runtime() -> None:
    inference = importlib.import_module("dinkster_inference")
    identity = "native:dinkster.seedvr2:" + "2" * 64

    class DescriptorEquivalentRuntime:
        def __init__(self, runtime_identity: str) -> None:
            self.runtime_identity = runtime_identity

    class DescriptorEquivalentWrapper:
        def __init__(self, component_sampling_runtime: object) -> None:
            self.component_sampling_runtime = component_sampling_runtime

    runtime = DescriptorEquivalentRuntime(identity)
    wrapper = DescriptorEquivalentWrapper(runtime)
    handle = SimpleNamespace(
        runtime=wrapper,
        recipe=SimpleNamespace(
            family_id="dinkster.seedvr2",
            sources=(SimpleNamespace(role="diffusion"),),
            runtime_identity=identity,
        ),
    )
    latent = torch.zeros((1, 16, 1, 2, 2))
    positive, negative = seedvr2_conditioning(latent, component_identity=identity)

    def rows(conditioning: object) -> list[list[object]]:
        typed = cast("Any", conditioning)
        return [
            [
                typed.embeddings,
                {arm._NATIVE_PREPARED_CONDITIONING_KEY: conditioning},
            ]
        ]

    positive_rows = rows(positive)
    negative_rows = rows(negative)
    selected = arm.resolve_seedvr2_component_execution(
        handle, positive_rows, negative_rows, inference
    )
    assert selected is not None
    assert selected[0] is runtime
    assert selected[1] is positive_rows
    assert selected[2] is negative_rows

    wrapper.component_sampling_runtime = DescriptorEquivalentRuntime(
        "native:dinkster.seedvr2:other"
    )
    with pytest.raises(TypeError, match="sampling runtime identity"):
        arm.resolve_seedvr2_component_execution(handle, positive_rows, negative_rows, inference)

    wrapper.component_sampling_runtime = runtime
    wrong_positive, wrong_negative = seedvr2_conditioning(
        latent, component_identity="native:dinkster.seedvr2:other"
    )
    with pytest.raises(ValueError, match="different model"):
        arm.resolve_seedvr2_component_execution(
            handle, rows(wrong_positive), negative_rows, inference
        )
    with pytest.raises(ValueError, match="different model"):
        arm.resolve_seedvr2_component_execution(
            handle, positive_rows, rows(wrong_negative), inference
        )


def test_temporal_chunk_and_merge_preserve_overlap_contract() -> None:
    samples = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1, 1).expand(1, 16, 6, 1, 1)
    chunked = arm.GenerationSeedVR2TemporalChunk.execute(
        latent={"samples": samples, "noise_mask": torch.ones_like(samples)},
        temporal_overlap=2,
        chunking_mode={"chunking_mode": "manual", "frames_per_chunk": 13},
    )
    chunks = cast("list[dict[str, torch.Tensor]]", chunked["latents"])
    assert [chunk["samples"].shape[2] for chunk in chunks] == [4, 4]
    assert chunked["temporal_overlap"] == 2
    chunks[0]["samples"] = torch.ones_like(chunks[0]["samples"])
    chunks[1]["samples"] = torch.full_like(chunks[1]["samples"], 3.0)

    merged = cast(
        "dict[str, torch.Tensor]",
        arm.GenerationSeedVR2TemporalMerge.execute(
            latents=chunks,
            temporal_overlap=2,
        )["latent"],
    )

    assert "noise_mask" not in merged
    expected = torch.tensor([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
    assert torch.equal(merged["samples"][0, 0, :, 0, 0], expected)


def test_provider_nodes_match_executed_comfyui_reference() -> None:
    golden = cast("dict[str, Any]", GOLDENS["provider"])
    source = _decode(golden["source"])
    decoded = _decode(golden["decoded"])
    preprocessed = cast(
        "torch.Tensor",
        arm.GenerationSeedVR2Preprocess.execute(resized_images=source)["images"],
    )
    torch.testing.assert_close(preprocessed, _decode(golden["preprocessed"]), rtol=0, atol=0)
    for method, expected in golden["postprocessed"].items():
        output = cast(
            "torch.Tensor",
            arm.GenerationSeedVR2PostProcessing.execute(
                images=decoded,
                original_resized_images=source,
                color_correction_method=method,
            )["images"],
        )
        torch.testing.assert_close(output, _decode(expected), rtol=1e-6, atol=1e-6)
    latent = {
        "samples": _decode(golden["chunks"][0]),
        "noise_mask": torch.ones_like(_decode(golden["chunks"][0])),
    }
    full = _decode(golden["merged"])
    latent["samples"] = torch.cat(
        (_decode(golden["chunks"][0]), _decode(golden["chunks"][1])[:, :, 2:]), dim=2
    )
    latent["noise_mask"] = torch.ones_like(latent["samples"])
    chunked = arm.GenerationSeedVR2TemporalChunk.execute(
        latent=latent,
        temporal_overlap=2,
        chunking_mode={"chunking_mode": "manual", "frames_per_chunk": 13},
    )
    chunks = cast("list[dict[str, torch.Tensor]]", chunked["latents"])
    for observed, expected in zip(chunks, golden["chunks"], strict=True):
        torch.testing.assert_close(observed["samples"], _decode(expected), rtol=0, atol=0)
    merged = cast(
        "dict[str, torch.Tensor]",
        arm.GenerationSeedVR2TemporalMerge.execute(
            latents=chunks,
            temporal_overlap=cast("int", golden["temporal_overlap"]),
        )["latent"],
    )
    torch.testing.assert_close(merged["samples"], full, rtol=1e-6, atol=1e-6)
