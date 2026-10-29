"""Focused parity and contract tests for the standalone Wan 2.1 DiT."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from attention_spy import CallableModuleKernel, assert_kernel_is_not_model_state
from dinkster_inference import (
    WAN21_HUMO_17B,
    WAN22_ANIMATE_14B,
    WAN22_BERNINI_14B,
    WAN22_S2V_14B,
    wan21_layout,
)
from dinkster_inference_torch import wan21_humo as wan21_humo_module
from dinkster_inference_torch import wan21_model as wan21_model_module
from dinkster_inference_torch.attention import select_attention
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import CastOperations, ResidencyRouted, bound_compute_dtype
from dinkster_inference_torch.wan21_animate import FaceBlock
from dinkster_inference_torch.wan21_humo import Wan21HumoModel
from dinkster_inference_torch.wan21_model import (
    WAN21_CAMERA_1_3B,
    WAN21_CAMERA_14B,
    WAN21_FLF_I2V_14B,
    WAN21_FUN_CONTROL_1_3B,
    WAN21_FUN_INPAINT_1_3B,
    WAN21_I2V_14B,
    WAN21_T2V_1_3B,
    WAN21_T2V_14B,
    WAN21_VACE_1_3B,
    WAN21_VACE_14B,
    WAN22_CAMERA_14B,
    WAN22_FUN_CONTROL_5B,
    WAN22_FUN_CONTROL_14B,
    WAN22_FUN_INPAINT_5B,
    WAN22_I2V_14B,
    WAN22_TI2V_5B,
    Wan21Config,
    Wan21Model,
    _repeat_time_rows,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.wan22_s2v import Wan22S2VModel
from golden_files import assert_reference_tensor, load_platform_golden
from unet_fill import fill_state_dict, hashed_input

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "wan21_model_goldens.json")

# The pinned input, state, PyTorch SDPA route, and pure-torch RoPE are identical
# across the reference and replay. Torch 2.11 versus the 2.13 golden changed
# float32 accumulation by at most 3.5763e-7; 1e-6 leaves 2.8x headroom.
FORWARD_ATOL = 1e-6


def _model(case: str, *, kernel: Any = None) -> Wan21Model:
    spec = GOLDENS["cases"][case]
    kwargs = {} if kernel is None else {"attention_kernel": kernel}
    config_values = dict(spec["config"])
    s2v_fixture = config_values.get("model_variant") == "s2v"
    humo_fixture = config_values.get("model_variant") == "humo"
    if s2v_fixture or humo_fixture:
        config_values["model_variant"] = "base"
    config = Wan21Config(**config_values)
    if s2v_fixture:
        object.__setattr__(config, "model_variant", "s2v")
    elif humo_fixture:
        object.__setattr__(config, "model_variant", "humo")
    model = (
        Wan22S2VModel(config, **kwargs)
        if config.model_variant == "s2v"
        else Wan21HumoModel(config, **kwargs)
        if config.model_variant == "humo"
        else Wan21Model(config, **kwargs)
    )
    model.load_state_dict(fill_state_dict(spec["state_dict"]), strict=True)
    return model


def _inputs(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    spec = GOLDENS["cases"][case]
    vision_shape = spec["vision_shape"]
    return (
        hashed_input(f"{case}:x", spec["input_shape"]),
        torch.tensor(spec["timesteps"], dtype=torch.float32),
        hashed_input(f"{case}:context", spec["context_shape"]),
        None if vision_shape is None else hashed_input(f"{case}:vision", vision_shape),
    )


def _decode(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=getattr(torch, payload["dtype"])).reshape(
        payload["shape"]
    )


@pytest.mark.parametrize(
    ("name", "config"),
    (
        ("t2v_1_3b", WAN21_T2V_1_3B),
        ("t2v_14b", WAN21_T2V_14B),
        ("i2v_14b", WAN21_I2V_14B),
        ("flf_i2v_14b", WAN21_FLF_I2V_14B),
        ("wan22_i2v_14b", WAN22_I2V_14B),
        ("animate_14b", WAN22_ANIMATE_14B),
        ("s2v_14b", WAN22_S2V_14B),
        ("humo_17b", WAN21_HUMO_17B),
        ("ti2v_5b", WAN22_TI2V_5B),
        ("vace_1_3b", WAN21_VACE_1_3B),
        ("vace_14b", WAN21_VACE_14B),
    ),
)
def test_official_full_geometry_state_dict_exactly_matches_executed_reference(
    name: str, config: Wan21Config
) -> None:
    with torch.device("meta"):
        model = (
            Wan22S2VModel(config)
            if config.model_variant == "s2v"
            else Wan21HumoModel(config)
            if config.model_variant == "humo"
            else Wan21Model(config)
        )
    actual = [[key, list(value.shape)] for key, value in sorted(model.state_dict().items())]
    assert actual == GOLDENS["layouts"][name]


@pytest.mark.parametrize(
    "config",
    (WAN21_CAMERA_1_3B, WAN21_CAMERA_14B, WAN22_CAMERA_14B),
)
def test_official_camera_state_dict_matches_header_contract(config: Wan21Config) -> None:
    with torch.device("meta"):
        model = Wan21Model(config)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    assert actual == wan21_layout(config)


def test_official_animate_state_and_every_direct_owner_match_native_contract() -> None:
    with torch.device("meta"):
        model = Wan21Model(WAN22_ANIMATE_14B)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    owners = [
        module
        for module in model.modules()
        if tuple(module.parameters(recurse=False)) or tuple(module.buffers(recurse=False))
    ]

    assert actual == wan21_layout(WAN22_ANIMATE_14B)
    assert len(actual) == 1441
    assert all(isinstance(module, ResidencyRouted) for module in owners)


def test_official_s2v_state_and_every_direct_owner_match_native_contract() -> None:
    with torch.device("meta"):
        model = Wan22S2VModel()
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    owners = [
        module
        for module in model.modules()
        if tuple(module.parameters(recurse=False)) or tuple(module.buffers(recurse=False))
    ]

    assert actual == wan21_layout(WAN22_S2V_14B)
    assert len(actual) == 1260
    assert all(isinstance(module, ResidencyRouted) for module in owners)


def test_official_humo_state_and_every_direct_owner_match_native_contract() -> None:
    with torch.device("meta"):
        model = Wan21HumoModel()
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    owners = [
        module
        for module in model.modules()
        if tuple(module.parameters(recurse=False)) or tuple(module.buffers(recurse=False))
    ]

    assert actual == wan21_layout(WAN21_HUMO_17B)
    assert len(actual) == 1583
    assert all(isinstance(module, ResidencyRouted) for module in owners)


def test_humo_audio_and_reference_preserve_target_geometry_rope_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AudioProjection(torch.nn.Module):
        def __init__(self, *, operations: object) -> None:
            super().__init__()
            del operations

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            batch, frames = audio.shape[:2]
            values = audio.mean(dim=(2, 3, 4)).reshape(batch, frames, 1, 1)
            return values.expand(batch, frames, 16, 1536).reshape(batch, frames * 16, 1536)

    monkeypatch.setattr(wan21_humo_module, "_Wan21HumoAudioProjection", AudioProjection)
    config = Wan21Config(
        in_channels=36,
        hidden_size=8,
        ffn_hidden_size=16,
        num_heads=1,
        num_layers=1,
        text_dim=4,
        time_freq_dim=4,
    )
    object.__setattr__(config, "model_variant", "humo")
    model = Wan21HumoModel(config)
    state = [(key, tuple(value.shape)) for key, value in sorted(model.state_dict().items())]
    model.load_state_dict(fill_state_dict(state), strict=True)
    rope_starts: list[int] = []
    original_rope = model._rope  # pyright: ignore[reportPrivateUsage]

    def record_rope(
        shape: tuple[int, int, int],
        x: torch.Tensor,
        *,
        time_start: int = 0,
        source_id: int = 0,
    ) -> torch.Tensor:
        rope_starts.append(time_start)
        return original_rope(shape, x, time_start=time_start, source_id=source_id)

    model._rope = record_rope  # pyright: ignore[reportPrivateUsage]
    target = hashed_input("humo:target", (1, 36, 2, 4, 4)).requires_grad_()
    audio = hashed_input("humo:audio", (1, 2, 8, 5, 1280)).requires_grad_()
    reference = hashed_input("humo:reference", (1, 36, 1, 4, 4)).requires_grad_()

    output = model(
        target,
        torch.tensor([0.5]),
        hashed_input("humo:text", (1, 3, 4)),
        audio_embed=audio,
        reference_latent=reference,
    )
    output.sum().backward()

    assert output.shape == (1, 16, 2, 4, 4)
    assert torch.isfinite(output).all()
    assert rope_starts == [0, 2]
    assert all(value.grad is not None for value in (target, audio, reference))


def test_s2v_audio_reference_motion_and_control_preserve_target_geometry_and_gradients() -> None:
    config = Wan21Config(
        hidden_size=8,
        ffn_hidden_size=16,
        num_heads=1,
        num_layers=1,
        text_dim=4,
        time_freq_dim=4,
    )
    object.__setattr__(config, "model_variant", "s2v")
    model = Wan22S2VModel(config)
    state = [(key, tuple(value.shape)) for key, value in sorted(model.state_dict().items())]
    model.load_state_dict(fill_state_dict(state), strict=True)
    target = hashed_input("s2v:target", (1, 16, 2, 8, 8)).requires_grad_()
    audio = hashed_input("s2v:audio", (1, 25, 1024, 8)).requires_grad_()
    reference = hashed_input("s2v:reference", (1, 16, 1, 8, 8)).requires_grad_()
    motion = hashed_input("s2v:motion", (1, 16, 7, 8, 8)).requires_grad_()
    control = hashed_input("s2v:control", (1, 16, 2, 8, 8)).requires_grad_()

    output = model(
        target,
        torch.tensor([0.5]),
        hashed_input("s2v:text", (1, 3, 4)),
        audio_embed=audio,
        reference_latent=reference,
        control_video=control,
        reference_motion=motion,
    )
    output.sum().backward()

    assert output.shape == target.shape
    assert torch.isfinite(output).all()
    assert all(value.grad is not None for value in (target, audio, reference, motion, control))


def test_animate_features_inject_pose_after_reference_and_chunk_face_frames() -> None:
    class PoseProjection(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, :4]

    class MotionEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batches: list[int] = []

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.batches.append(value.shape[0])
            return value.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).expand(-1, 512)

    class FaceEncoder(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, :2, :4].unsqueeze(2).expand(-1, -1, 5, -1)

    motion_encoder = MotionEncoder()
    fake = SimpleNamespace(
        pose_patch_embedding=PoseProjection(),
        motion_encoder=motion_encoder,
        face_encoder=FaceEncoder(),
    )
    patches = torch.zeros((1, 4, 3, 1, 1))
    pose = torch.ones((1, 16, 2, 1, 1))
    faces = torch.arange(9, dtype=torch.float32).view(1, 1, 9, 1, 1).expand(-1, 3, -1, 1, 1)

    patched, motion = Wan21Model._animate_features(  # pyright: ignore[reportPrivateUsage]
        cast("Any", fake), patches, pose, faces
    )

    assert torch.equal(patched[:, :, 0], torch.zeros((1, 4, 1, 1)))
    assert torch.equal(patched[:, :, 1:], torch.ones((1, 4, 2, 1, 1)))
    assert motion_encoder.batches == [8, 1]
    assert motion is not None
    assert motion.shape == (1, 3, 5, 4)
    assert torch.count_nonzero(motion[:, 0]) == 0


def test_face_adapter_attends_each_spatial_row_to_five_motion_rows() -> None:
    spy = CallableModuleKernel(select_attention("flux").kernel)
    block = FaceBlock(
        8,
        2,
        operations=CastOperations(torch.float32),
        attention_kernel=spy,
    )
    state = [(key, tuple(value.shape)) for key, value in sorted(block.state_dict().items())]
    block.load_state_dict(fill_state_dict(state), strict=True)

    output = block(
        hashed_input("animate-face:block", (1, 6, 8)),
        hashed_input("animate-face:motion", (1, 2, 5, 8)),
    )

    assert output.shape == (1, 6, 8)
    assert len(spy.calls) == 1
    assert spy.calls[0]["q_shape"] == (2, 2, 3, 4)
    assert spy.calls[0]["k_shape"] == (2, 2, 5, 4)


def test_camera_adapter_projects_pixel_trajectory_into_patch_grid() -> None:
    config = Wan21Config(
        model_type="t2v",
        in_channels=36,
        hidden_size=24,
        ffn_hidden_size=48,
        num_heads=2,
        num_layers=1,
        text_dim=12,
        time_freq_dim=8,
        camera_channels=24,
    )
    model = Wan21Model(config)
    state = [(key, tuple(value.shape)) for key, value in sorted(model.state_dict().items())]
    model.load_state_dict(fill_state_dict(state), strict=True)
    x = hashed_input("camera:x", (1, 36, 2, 5, 6))
    timesteps = torch.tensor([0.375])
    context = hashed_input("camera:context", (1, 4, 12))
    camera = hashed_input("camera:conditions", (1, 24, 2, 40, 48))

    with_camera = model(x, timesteps, context, camera_conditions=camera)
    without_camera = model(x, timesteps, context)

    assert with_camera.shape == x[:, :16].shape
    assert torch.isfinite(with_camera).all()
    assert not torch.equal(with_camera, without_camera)


def test_phantom_reference_tokens_are_appended_and_trimmed_from_output() -> None:
    config = Wan21Config(
        in_channels=16,
        hidden_size=8,
        ffn_hidden_size=16,
        num_heads=1,
        num_layers=1,
        text_dim=4,
        time_freq_dim=4,
        out_channels=16,
    )
    model = Wan21Model(config)

    class RecordingProjection(torch.nn.Module):
        input: torch.Tensor | None = None

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.input = value.detach().clone()
            return value[:, :1, :, ::2, ::2].expand(-1, 8, -1, -1, -1)

    class RecordingHead(torch.nn.Module):
        rows: torch.Tensor | None = None

        def forward(self, value: torch.Tensor, _time: torch.Tensor) -> torch.Tensor:
            self.rows = value.detach().clone()
            return value[..., :1].expand(*value.shape[:-1], 64)

    projection = RecordingProjection()
    head = RecordingHead()
    raw = cast("Any", model)
    raw.patch_embedding = projection
    raw.blocks = torch.nn.ModuleList()
    raw.head = head
    target = torch.ones((1, 16, 2, 2, 2))
    references = torch.full((1, 16, 3, 2, 2), 9.0)

    output = model(
        target,
        torch.tensor([1.0]),
        torch.zeros((1, 3, 4)),
        temporal_reference=references,
    )

    assert projection.input is not None
    assert projection.input.shape == (1, 16, 5, 2, 2)
    assert torch.equal(projection.input[:, :, :2], target)
    assert torch.equal(projection.input[:, :, 2:], references)
    assert head.rows is not None
    assert torch.equal(head.rows[:, :2], torch.ones((1, 2, 8)))
    assert torch.equal(head.rows[:, 2:], torch.full((1, 3, 8), 9.0))
    assert torch.equal(output, torch.ones((1, 16, 2, 2, 2)))


def test_uni3c_injects_twenty_post_block_residuals_into_every_guidance_lane() -> None:
    class PatchEmbedding(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, :4, :, ::2, ::2]

    class TimeEmbedding(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.new_zeros((value.shape[0], WAN21_T2V_14B.hidden_size))

    class TimeProjection(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.new_zeros((*value.shape[:-1], 6 * WAN21_T2V_14B.hidden_size))

    class TextEmbedding(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value[..., :4]

    class BaseBlock(torch.nn.Module):
        def forward(
            self,
            value: torch.Tensor,
            *_args: object,
            multitalk: object | None = None,
            multitalk_block_index: int = 0,
            grid_shape: tuple[int, int, int] | None = None,
        ) -> torch.Tensor:
            return value

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rows: torch.Tensor | None = None

        def forward(self, value: torch.Tensor, _time: torch.Tensor) -> torch.Tensor:
            self.rows = value
            return value[..., :1].expand(-1, -1, 64)

    class Uni3C:
        config = SimpleNamespace(layers=20)

        def process_input(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = value[:, :4, :, ::2, ::2].flatten(2).transpose(1, 2)
            return hidden, value.new_zeros((1, 1, hidden.shape[1], 1, 2))

        def forward_block(
            self,
            index: int,
            hidden: torch.Tensor,
            _temb: torch.Tensor,
            _freqs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = hidden * 0.0 + float(index + 1)
            return hidden, residual

    def validate(*_args: object) -> None:
        pass

    def rope(_grid: tuple[int, int, int], value: torch.Tensor) -> torch.Tensor:
        return value.new_zeros((1, 1, 4, 1, 2))

    model = Wan21Model.__new__(Wan21Model)
    torch.nn.Module.__init__(model)
    test_model = cast(Any, model)
    test_model.config = WAN21_T2V_14B
    test_model.patch_embedding = PatchEmbedding()
    test_model.time_embedding = TimeEmbedding()
    test_model.time_projection = TimeProjection()
    test_model.text_embedding = TextEmbedding()
    test_model.blocks = torch.nn.ModuleList(BaseBlock() for _ in range(40))
    test_model.head = Head()
    test_model.img_emb = None
    test_model.ref_conv = None
    test_model.vace_blocks = None
    test_model.vace_patch_embedding = None
    test_model.control_adapter = None
    test_model._attention_kernel = select_attention("flux").kernel
    test_model._validate = validate
    test_model._rope = rope

    control_input = torch.randn((1, 36, 1, 4, 4), requires_grad=True)
    execution = SimpleNamespace(model=Uni3C(), strength=0.5)
    output = model(
        torch.zeros((2, 16, 1, 4, 4)),
        torch.ones((2,)),
        torch.zeros((2, 1, 4096)),
        uni3c=cast(Any, execution),
        uni3c_input=control_input,
    )

    head = test_model.head
    assert head.rows is not None
    assert head.rows.shape == (2, 4, 4)
    torch.testing.assert_close(head.rows, torch.full_like(head.rows, 105.0))
    assert output.shape == (2, 16, 1, 4, 4)
    output.sum().backward()
    assert control_input.grad is not None


def test_bernini_appends_padded_context_with_unique_source_ids_and_crops_output() -> None:
    class PatchEmbedding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shapes: list[tuple[int, ...]] = []

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.shapes.append(tuple(value.shape))
            return value[:, :4, :, ::2, ::2]

    class TimeEmbedding(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.new_zeros((value.shape[0], WAN22_BERNINI_14B.hidden_size))

    class TimeProjection(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.new_zeros((*value.shape[:-1], 6 * WAN22_BERNINI_14B.hidden_size))

    class TextEmbedding(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value[..., :4]

    class MixingBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rows: torch.Tensor | None = None
            self.rope_rows = 0

        def forward(
            self,
            value: torch.Tensor,
            _time: torch.Tensor,
            freqs: torch.Tensor,
            _context: torch.Tensor,
            _image_rows: int | None,
            *,
            multitalk: object | None = None,
            multitalk_block_index: int = 0,
            grid_shape: tuple[int, int, int] | None = None,
        ) -> torch.Tensor:
            self.rows = value
            self.rope_rows = freqs.shape[1]
            mixed = value.clone()
            mixed[:, :4] = mixed[:, :4] + value[:, 4:].sum(dim=1, keepdim=True)
            return mixed

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rows: torch.Tensor | None = None

        def forward(self, value: torch.Tensor, _time: torch.Tensor) -> torch.Tensor:
            self.rows = value
            return value[..., :1].expand(-1, -1, 64)

    patch_embedding = PatchEmbedding()
    block = MixingBlock()
    head = Head()
    rope_calls: list[tuple[tuple[int, int, int], int]] = []

    def rope_rows(
        grid: tuple[int, int, int],
        value: torch.Tensor,
        *,
        source_id: int = 0,
    ) -> torch.Tensor:
        rope_calls.append((grid, source_id))
        return value.new_zeros((1, grid[0] * grid[1] * grid[2], 1, 1, 2))

    model = Wan21Model.__new__(Wan21Model)
    torch.nn.Module.__init__(model)
    raw = cast(Any, model)
    raw.config = WAN22_BERNINI_14B
    raw.patch_embedding = patch_embedding
    raw.time_embedding = TimeEmbedding()
    raw.time_projection = TimeProjection()
    raw.text_embedding = TextEmbedding()
    raw.blocks = torch.nn.ModuleList((block,))
    raw.head = head
    raw.img_emb = None
    raw.ref_conv = None
    raw.vace_blocks = None
    raw.vace_patch_embedding = None
    raw.control_adapter = None
    raw._attention_kernel = select_attention("flux").kernel
    raw._rope = rope_rows

    first = torch.full((1, 16, 2, 3, 5), 2.0, requires_grad=True)
    second = torch.full((1, 16, 1, 2, 2), 3.0, requires_grad=True)
    output = model(
        torch.zeros((1, 16, 1, 4, 4)),
        torch.ones((1,)),
        torch.zeros((1, 1, 4096)),
        context_latents=(first, second),
    )

    assert patch_embedding.shapes == [
        (1, 16, 1, 4, 4),
        (1, 16, 2, 4, 6),
        (1, 16, 1, 2, 2),
    ]
    assert rope_calls == [((1, 2, 2), 0), ((2, 2, 3), 1), ((1, 1, 1), 2)]
    assert block.rows is not None and block.rows.shape == (1, 17, 4)
    assert block.rope_rows == 17
    assert head.rows is not None and head.rows.shape == (1, 17, 4)
    assert output.shape == (1, 16, 1, 4, 4)
    output.sum().backward()
    assert first.grad is not None and second.grad is not None


def test_bernini_source_id_composes_distinct_rope_rotations() -> None:
    model = Wan21Model.__new__(Wan21Model)
    torch.nn.Module.__init__(model)
    raw = cast(Any, model)
    raw.config = WAN22_BERNINI_14B
    head_dim = WAN22_BERNINI_14B.hidden_size // WAN22_BERNINI_14B.num_heads
    raw.rope_embedder = wan21_model_module.EmbedND(
        dim=head_dim,
        theta=10_000,
        axes_dim=(
            head_dim - 4 * (head_dim // 6),
            2 * (head_dim // 6),
            2 * (head_dim // 6),
        ),
    )
    value = torch.zeros((1, 4, 1, 1, 1))

    target = raw._rope((1, 2, 2), value)
    first = raw._rope((1, 2, 2), value, source_id=1)
    second = raw._rope((1, 2, 2), value, source_id=2)

    assert target.shape == first.shape == second.shape
    assert not torch.equal(target, first)
    assert not torch.equal(first, second)


def test_non_bernini_model_refuses_context_latents_before_projection() -> None:
    model = _model("t2v_reduced")
    x, timesteps, context, vision = _inputs("t2v_reduced")
    called = False

    def mark_called(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        nonlocal called
        called = True

    handle = model.patch_embedding.register_forward_pre_hook(mark_called)
    try:
        with pytest.raises(ValueError, match="require exact Wan 2.2 Bernini 14B"):
            model(
                x,
                timesteps,
                context,
                vision,
                context_latents=(torch.zeros((1, 16, 1, 2, 2)),),
            )
    finally:
        handle.remove()
    assert called is False


@pytest.mark.parametrize(
    "config",
    (
        WAN21_FUN_CONTROL_1_3B,
        WAN21_FUN_INPAINT_1_3B,
        WAN22_FUN_CONTROL_5B,
        WAN22_FUN_INPAINT_5B,
        WAN22_FUN_CONTROL_14B,
    ),
)
def test_official_fun_state_dict_matches_header_contract(config: Wan21Config) -> None:
    with torch.device("meta"):
        model = Wan21Model(config)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    assert actual == wan21_layout(config)


def test_fun_reference_tokens_are_prepended_and_trimmed_from_output() -> None:
    config = Wan21Config(
        in_channels=52,
        hidden_size=8,
        ffn_hidden_size=16,
        num_heads=1,
        num_layers=1,
        text_dim=4,
        time_freq_dim=4,
        out_channels=16,
        reference_channels=16,
    )
    model = Wan21Model(config)

    class Projection3d(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (value.shape[0], 8, value.shape[2], value.shape[3] // 2, value.shape[4] // 2),
                2.0,
                dtype=value.dtype,
                device=value.device,
            )

    class Projection2d(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (value.shape[0], 8, value.shape[2] // 2, value.shape[3] // 2),
                9.0,
                dtype=value.dtype,
                device=value.device,
            )

    class RecordingHead(torch.nn.Module):
        rows: torch.Tensor | None = None

        def forward(self, value: torch.Tensor, _time: torch.Tensor) -> torch.Tensor:
            self.rows = value.detach().clone()
            return value[..., :1].expand(*value.shape[:-1], 64)

    head = RecordingHead()
    raw = cast("Any", model)
    raw.patch_embedding = Projection3d()
    raw.ref_conv = Projection2d()
    raw.blocks = torch.nn.ModuleList()
    raw.head = head
    output = model(
        torch.zeros((1, 52, 2, 4, 4)),
        torch.tensor([1.0]),
        torch.zeros((1, 3, 4)),
        reference_latent=torch.zeros((1, 16, 4, 4)),
    )

    assert head.rows is not None
    assert torch.equal(head.rows[:, :4], torch.full((1, 4, 8), 9.0))
    assert torch.equal(head.rows[:, 4:], torch.full((1, 8, 8), 2.0))
    assert torch.equal(output, torch.full((1, 16, 2, 4, 4), 2.0))


@pytest.mark.parametrize(
    "case",
    (
        "t2v_reduced",
        "bernini_reduced",
        "phantom_t2v_reduced",
        "i2v_36_reduced",
        "flf_i2v_36_reduced",
        "wan22_i2v_36_reduced",
        "s2v_reduced",
        "humo_reduced",
        "animate_reduced",
        "ti2v_48_reduced",
        "vace_reduced",
    ),
)
def test_reduced_state_dict_and_forward_are_value_close_to_executed_reference(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(case)
    spec = GOLDENS["cases"][case]
    actual_layout = [[key, list(value.shape)] for key, value in sorted(model.state_dict().items())]
    assert actual_layout == spec["state_dict"]
    inputs = _inputs(case)
    audio_shape = spec.get("audio_shape")
    pose_latents_shape = spec.get("pose_latents_shape")
    context_latent_shapes = spec.get("context_latent_shapes", [])
    if spec["config"].get("model_variant") == "humo":
        assert audio_shape is not None
        actual = model(
            *inputs[:3],
            audio_embed=hashed_input(f"{case}:audio", audio_shape),
            reference_latent=hashed_input(f"{case}:reference", spec["reference_latent_shape"]),
        )
    elif audio_shape is not None:
        actual = model(
            *inputs,
            audio_embed=hashed_input(f"{case}:audio", audio_shape),
            reference_latent=hashed_input(f"{case}:reference", spec["reference_latent_shape"]),
            reference_motion=hashed_input(f"{case}:motion", spec["reference_motion_shape"]),
            control_video=hashed_input(f"{case}:control", spec["control_video_shape"]),
        )
    elif context_latent_shapes:
        # Bind the reduced fixture as the exact Bernini profile without allocating a 14B model.
        monkeypatch.setattr(wan21_model_module, "WAN22_BERNINI_14B", model.config)
        actual = model(
            *inputs,
            context_latents=tuple(
                hashed_input(f"{case}:context-latent:{index}", shape)
                for index, shape in enumerate(context_latent_shapes)
            ),
        )
    elif pose_latents_shape is not None:
        actual = model(
            *inputs,
            pose_latents=hashed_input(f"{case}:pose", pose_latents_shape),
            face_pixel_values=hashed_input(f"{case}:face", spec["face_pixel_values_shape"]),
        )
    elif spec["vace_context_shape"] is None:
        temporal_reference_shape = spec.get("temporal_reference_shape")
        actual = model(
            *inputs,
            temporal_reference=(
                None
                if temporal_reference_shape is None
                else hashed_input(f"{case}:temporal-reference", temporal_reference_shape)
            ),
        )
    else:
        actual = model(
            *inputs,
            vace_context=hashed_input(f"{case}:vace", spec["vace_context_shape"]),
            vace_strength=tuple(spec["vace_strength"]),
        )
    assert_reference_tensor(actual, _decode(spec["output"]), rtol=0, atol=FORWARD_ATOL)


def test_feed_forward_routes_gelu_through_linear_input_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = torch.nn.Linear(4, 6)
    last = torch.nn.Linear(6, 3)
    feed_forward = wan21_model_module.WanFeedForward(first, torch.nn.GELU(), last)
    input = torch.randn(2, 4)
    intermediate = first(input)
    expected = torch.randn(2, 3)
    calls: list[tuple[torch.nn.Module, torch.Tensor, str]] = []

    def linear_input_act(
        linear: torch.nn.Module, value: torch.Tensor, activation: str
    ) -> torch.Tensor:
        calls.append((linear, value, activation))
        return expected

    monkeypatch.setattr(wan21_model_module, "linear_input_act", linear_input_act)

    assert feed_forward(input) is expected
    assert calls[0][0] is last
    torch.testing.assert_close(calls[0][1], intermediate)
    assert calls[0][2] == "gelu_tanh"


def test_attention_routes_self_and_text_calls_through_injected_kernel() -> None:
    spy = CallableModuleKernel(select_attention("flux").kernel)
    model = _model("t2v_reduced", kernel=spy)
    output = model(*_inputs("t2v_reduced"))
    assert output.shape == (1, 16, 2, 5, 6)
    assert len(spy.calls) == 4
    assert [call["q_shape"][-2] for call in spy.calls] == [18, 18, 18, 18]
    assert [call["k_shape"][-2] for call in spy.calls] == [18, 4, 18, 4]
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)
    assert_kernel_is_not_model_state(model, spy)


def test_i2v_attention_routes_self_text_and_vision_through_injected_kernel() -> None:
    spy = CallableModuleKernel(select_attention("flux").kernel)
    model = _model("i2v_36_reduced", kernel=spy)
    output = model(*_inputs("i2v_36_reduced"))
    assert output.shape == (1, 16, 2, 5, 6)
    assert len(spy.calls) == 6
    assert [call["q_shape"][-2] for call in spy.calls] == [18, 18, 18, 18, 18, 18]
    assert [call["k_shape"][-2] for call in spy.calls] == [18, 4, 3, 18, 4, 3]
    assert all(call["mask"] is None and not call["causal"] for call in spy.calls)
    assert_kernel_is_not_model_state(model, spy)


def test_i2v_attention_reuses_text_rows_for_image_projection_without_vision() -> None:
    spy = CallableModuleKernel(select_attention("flux").kernel)
    model = _model("i2v_36_reduced", kernel=spy)
    x, timesteps, context, _vision = _inputs("i2v_36_reduced")

    output = model(x, timesteps, context)

    assert output.shape == (1, 16, 2, 5, 6)
    assert len(spy.calls) == 6
    assert [call["k_shape"][-2] for call in spy.calls] == [18, 4, 4, 18, 4, 4]


def test_flf_attention_uses_positioned_vision_rows() -> None:
    spy = CallableModuleKernel(select_attention("flux").kernel)
    model = _model("flf_i2v_36_reduced", kernel=spy)
    output = model(*_inputs("flf_i2v_36_reduced"))
    assert output.shape == (1, 16, 2, 5, 6)
    assert [call["k_shape"][-2] for call in spy.calls] == [18, 4, 3, 18, 4, 3]


def test_vace_padding_keeps_controls_aligned_with_odd_target_geometry() -> None:
    model = _model("vace_reduced")
    x, timesteps, context, vision = _inputs("vace_reduced")
    x = x[:, :, :, :5, :5]
    controls = hashed_input("vace_reduced:odd-vace", (1, 2, 96, 2, 5, 5))

    output = model(
        x,
        timesteps,
        context,
        vision,
        vace_context=controls,
        vace_strength=(0.25, 1.5),
    )

    assert output.shape == (1, 16, 2, 5, 5)
    assert torch.isfinite(output).all()


def test_every_state_bearing_factory_layer_uses_operations_routing() -> None:
    model = _model("i2v_36_reduced")
    layers = (
        torch.nn.Linear,
        torch.nn.Conv3d,
        torch.nn.LayerNorm,
        torch.nn.RMSNorm,
    )
    routed = [module for module in model.modules() if isinstance(module, layers)]
    assert routed
    assert all(isinstance(module, ResidencyRouted) for module in routed)


def test_every_state_owner_is_residency_routed_and_offloaded_forward_matches() -> None:
    model = _model("flf_i2v_36_reduced")
    owners = [module for module in model.modules() if tuple(module.parameters(recurse=False))]
    assert owners
    assert all(isinstance(module, ResidencyRouted) for module in owners)
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    resident = model(*_inputs("flf_i2v_36_reduced"))
    mechanism.partially_unload(mechanism.loaded_bytes())
    offloaded = model(*_inputs("flf_i2v_36_reduced"))
    assert torch.equal(offloaded, resident)


def test_vace_state_owners_are_routed_and_offloaded_forward_matches() -> None:
    model = _model("vace_reduced")
    owners = [module for module in model.modules() if tuple(module.parameters(recurse=False))]
    assert owners
    assert all(isinstance(module, ResidencyRouted) for module in owners)
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    inputs = _inputs("vace_reduced")
    controls = hashed_input(
        "vace_reduced:vace", GOLDENS["cases"]["vace_reduced"]["vace_context_shape"]
    )
    resident = model(*inputs, vace_context=controls, vace_strength=(0.25, 1.5))
    mechanism.partially_unload(mechanism.loaded_bytes())
    offloaded = model(*inputs, vace_context=controls, vace_strength=(0.25, 1.5))
    assert torch.equal(offloaded, resident)


def test_animate_direct_state_offloaded_forward_matches() -> None:
    case = "animate_reduced"
    spec = GOLDENS["cases"][case]
    model = _model(case)
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    kwargs = {
        "pose_latents": hashed_input(f"{case}:pose", spec["pose_latents_shape"]),
        "face_pixel_values": hashed_input(f"{case}:face", spec["face_pixel_values_shape"]),
    }

    resident = model(*_inputs(case), **kwargs)
    mechanism.partially_unload(mechanism.loaded_bytes())
    offloaded = model(*_inputs(case), **kwargs)

    assert torch.equal(offloaded, resident)


def test_patch_embedding_keeps_float32_compute_with_matching_storage_dtype() -> None:
    model = _model("t2v_reduced")
    assert bound_compute_dtype(model.patch_embedding) is torch.float32


def test_patch_embedding_keeps_float32_compute_for_bfloat16_transformer() -> None:
    spec = GOLDENS["cases"]["t2v_reduced"]
    model = Wan21Model(
        Wan21Config(**spec["config"]),
        operations=CastOperations(torch.bfloat16),
    )
    model.load_state_dict(fill_state_dict(spec["state_dict"]), strict=True)
    assert bound_compute_dtype(model.patch_embedding) is torch.float32
    assert bound_compute_dtype(model.text_embedding[0]) is torch.bfloat16

    x, timesteps, context, _vision = _inputs("t2v_reduced")
    output = model(x.to(torch.bfloat16), timesteps, context.to(torch.bfloat16))
    assert output.dtype is torch.bfloat16
    assert torch.isfinite(output).all()


def test_scalar_time_modulation_broadcasts_without_materializing_token_rows() -> None:
    modulation = torch.ones((1, 1, 8))
    tokens = torch.empty((1, 4096, 8))

    assert _repeat_time_rows(modulation, tokens) is modulation


@pytest.mark.parametrize(
    "config",
    (
        {"model_type": "flf"},
        {"flf_pos_embed_token_number": 514},
        {"model_type": "i2v", "in_channels": 36, "flf_pos_embed_token_number": 0},
        {"patch_size": (2, 2, 2)},
        {"model_type": "i2v", "in_channels": 16},
        {"reference_channels": 16},
        {"hidden_size": 25, "num_heads": 2},
        {"hidden_size": 18, "num_heads": 2},
        {"time_freq_dim": 7},
        {"num_layers": 0},
        {"eps": float("nan")},
    ),
)
def test_config_refuses_non_wan21_geometry(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Wan21Config(**config)


@pytest.mark.parametrize(
    ("case", "replacement", "message"),
    (
        ("t2v_reduced", {"x": torch.empty(1, 16, 2, 4)}, "x must"),
        ("t2v_reduced", {"x": torch.empty(1, 15, 2, 4, 4)}, "16 channels"),
        ("t2v_reduced", {"timesteps": torch.empty(2)}, "timesteps"),
        ("t2v_reduced", {"context": torch.empty(1, 0, 12)}, "at least one"),
        ("t2v_reduced", {"vision": torch.empty(1, 2, 1280)}, "only by I2V"),
        ("i2v_36_reduced", {"vision": torch.empty(1, 2, 1279)}, "1280"),
    ),
)
def test_input_validation_refuses_before_patch_projection(
    case: str, replacement: dict[str, torch.Tensor | None], message: str
) -> None:
    model = _model(case)
    x, timesteps, context, vision = _inputs(case)
    values = {"x": x, "timesteps": timesteps, "context": context, "vision": vision}
    values.update(replacement)
    called = False

    def mark_called(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        nonlocal called
        called = True

    handle = model.patch_embedding.register_forward_pre_hook(mark_called)
    try:
        with pytest.raises(ValueError, match=message):
            model(values["x"], values["timesteps"], values["context"], values["vision"])
    finally:
        handle.remove()
    assert called is False


@pytest.mark.parametrize(
    "replacement",
    (
        {"x": torch.zeros(1, 36, 2, 5, 6, dtype=torch.int64)},
        {"timesteps": torch.zeros(1, dtype=torch.int64)},
        {"context": torch.zeros(1, 4, 12, dtype=torch.int64)},
        {"vision": torch.zeros(1, 3, 1280, dtype=torch.int64)},
    ),
)
def test_input_validation_refuses_nonfloating_tensors_before_patch_projection(
    replacement: dict[str, torch.Tensor],
) -> None:
    model = _model("i2v_36_reduced")
    x, timesteps, context, vision = _inputs("i2v_36_reduced")
    values = {"x": x, "timesteps": timesteps, "context": context, "vision": vision}
    values.update(replacement)
    called = False

    def mark_called(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        nonlocal called
        called = True

    handle = model.patch_embedding.register_forward_pre_hook(mark_called)
    try:
        with pytest.raises(TypeError, match="strided floating"):
            model(values["x"], values["timesteps"], values["context"], values["vision"])
    finally:
        handle.remove()
    assert called is False


def test_t2v_refuses_vision_rows() -> None:
    model = _model("t2v_reduced")
    x, timesteps, context, _vision = _inputs("t2v_reduced")
    with pytest.raises(ValueError, match="only by I2V"):
        model(x, timesteps, context, vision=torch.empty(1, 2, 1280))


def test_variant_only_constructor_arguments_are_not_accepted() -> None:
    with pytest.raises(TypeError):
        Wan21Model(WAN21_T2V_1_3B, flf_pos_embed_token_number=257)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Wan21Model(WAN21_T2V_1_3B, in_dim_ref_conv=16)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Wan21Model(WAN21_T2V_1_3B, full_ref=True)  # type: ignore[call-arg]
