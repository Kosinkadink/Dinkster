# pyright: reportPrivateUsage=false

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch
import torch.nn.functional as F
from dinkster_inference import ComponentPlan, Trellis2DecoderConfig, Trellis2FlowConfig
from dinkster_inference_torch import trellis2_assembly, trellis2_flow
from dinkster_inference_torch.component_runtime import trellis_runtime
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.operations import INITLESS, bound_compute_dtype
from dinkster_inference_torch.sparse import make_sparse_support
from dinkster_inference_torch.trellis2_assembly import (
    AssembledTrellis2,
    _decoder_builder,
    _microsoft_split_state,
)
from dinkster_inference_torch.trellis2_flow import (
    Trellis2FlowModel,
    Trellis2HeadRmsNorm,
    _apply_microsoft_rope,
    _apply_sparse_rope,
    _microsoft_rope_table,
    _padded,
    _sparse_attention,
    _SparseTensor,
)
from dinkster_inference_torch.trellis2_vae import (
    Trellis2SparseConvNeXt,
    Trellis2SparseDecoder,
)
from unet_fill import fill_state_dict, hashed_input

_MICROSOFT_GOLDEN = Path(__file__).parent / "goldens" / "trellis2_microsoft_source.json"


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(state[key]))
    return digest.hexdigest()


def _naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    assert mask is None and not causal and scale is None and not enable_gqa
    weights = torch.softmax(
        q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5,
        dim=-1,
    )
    return weights @ v


def test_dense_flow_matches_executed_microsoft_source_golden() -> None:
    golden = cast(dict[str, Any], json.loads(_MICROSOFT_GOLDEN.read_text()))
    assert golden["source"] == {
        "commit": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
        "module": "trellis2.models.sparse_structure_flow.SparseStructureFlowModel",
        "repository": "https://github.com/microsoft/TRELLIS.2",
    }
    assert golden["generator"] == {
        "attention_backend": "naive",
        "torch": "2.6.0+cu124",
    }

    config = Trellis2FlowConfig(
        stage="structure",
        image_attention="global",
        in_channels=32,
        out_channels=32,
        projected_channels=None,
        model_channels=12,
        condition_channels=8,
        num_blocks=1,
        num_heads=2,
        head_channels=6,
        mlp_channels=48,
        timestep_channels=256,
    )
    model = Trellis2FlowModel(
        config,
        operations=INITLESS,
        attention_kernel=_naive_attention,
        compute_dtype=torch.float32,
        microsoft_split_precision=True,
    )
    entries = [(key, list(value.shape)) for key, value in model.named_parameters()]
    state = fill_state_dict(entries)
    assert len(entries) == golden["state"]["parameter_count"]
    assert _state_digest(state) == golden["state"]["sha256"]
    model.load_state_dict(state, strict=True)

    inputs = golden["inputs"]
    latent = hashed_input(inputs["latent_key"], inputs["latent_shape"])
    context = hashed_input(inputs["context_key"], inputs["context_shape"])
    timestep = torch.tensor([inputs["timestep"]], dtype=torch.float32)
    assert hashlib.sha256(_tensor_bytes(latent)).hexdigest() == inputs["latent_sha256"]
    assert hashlib.sha256(_tensor_bytes(context)).hexdigest() == inputs["context_sha256"]

    output_record = golden["output"]
    expected_bytes = zlib.decompress(base64.b64decode(output_record["data"]))
    assert hashlib.sha256(expected_bytes).hexdigest() == output_record["raw_sha256"]
    expected = torch.frombuffer(bytearray(expected_bytes), dtype=torch.float32).reshape(
        output_record["shape"]
    )
    with torch.no_grad():
        actual = model(latent, timestep, context)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


def _sparse(counts: tuple[int, ...]) -> _SparseTensor:
    coordinates = []
    for batch, count in enumerate(counts):
        coordinates.extend((batch, row, 0, 0) for row in range(count))
    support = make_sparse_support(
        torch.tensor(coordinates, dtype=torch.int32),
        counts,
        8,
        (-0.5, -0.5, -0.5),
        (0.125, 0.125, 0.125),
    )
    features = torch.arange(sum(counts) * 6, dtype=torch.float32).view(sum(counts), 2, 3)
    return _SparseTensor(support, features)


def test_uniform_sparse_attention_uses_unmasked_dense_batches() -> None:
    value = _sparse((2, 2))
    padded, valid = _padded(value)
    observed: dict[str, Any] = {}

    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        observed["mask"] = mask
        assert torch.equal(q, k)
        assert not causal and scale is None and not enable_gqa
        return v

    output = _sparse_attention(value, value, value, kernel)

    assert padded.shape == (2, 2, 2, 3)
    assert valid is None
    assert observed["mask"] is None
    assert torch.equal(output.feats, value.feats)


def test_variable_sparse_attention_masks_and_removes_padding() -> None:
    value = _sparse((2, 1))
    observed: dict[str, Any] = {}

    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        observed["mask"] = mask
        return v

    output = _sparse_attention(value, value, value, kernel)

    mask = observed["mask"]
    assert type(mask) is torch.Tensor
    assert mask.shape == (2, 1, 1, 2)
    assert mask[1, 0, 0, 1] < -1e30
    assert torch.equal(output.feats, value.feats)


def _flow_config() -> Trellis2FlowConfig:
    return Trellis2FlowConfig(
        stage="structure",
        image_attention="global",
        in_channels=8,
        out_channels=8,
        projected_channels=None,
        model_channels=12,
        condition_channels=8,
        num_blocks=1,
        num_heads=2,
        head_channels=6,
        mlp_channels=16,
        timestep_channels=8,
    )


def test_flow_model_enrolls_every_direct_state_owner_for_residency() -> None:
    model = Trellis2FlowModel(_flow_config(), operations=INITLESS)

    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")

    assert mechanism.total_bytes() == sum(value.nbytes for value in model.state_dict().values())


def test_assembled_trellis2_requires_and_exposes_selected_compute_dtype() -> None:
    build = cast("Any", AssembledTrellis2)
    with pytest.raises(TypeError, match="_compute_dtype"):
        build(object(), object())

    assembled = build(object(), object(), torch.float32)

    assert assembled.compute_dtype("diffusion") is torch.float32
    assert assembled.compute_dtype("vae") is None


def test_trellis_component_runtime_propagates_selected_compute_dtype() -> None:
    loaded = SimpleNamespace(module=object(), plan=object())

    runtime = trellis_runtime(loaded, "native:dinkster.trellis2:test", torch.float16)

    assert runtime.assembled.compute_dtype("diffusion") is torch.float16


def test_decoder_builder_distinguishes_fused_comfy_vaes_from_split_decoders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = object()
    texture = object()
    sparse = object()

    def shape_builder(**_kwargs: object) -> object:
        return shape

    def texture_builder(**_kwargs: object) -> object:
        return texture

    def sparse_builder(**_kwargs: object) -> object:
        return sparse

    monkeypatch.setattr(trellis2_assembly, "Trellis2ShapeVae", shape_builder)
    monkeypatch.setattr(trellis2_assembly, "Trellis2TextureVae", texture_builder)
    monkeypatch.setattr(trellis2_assembly, "Trellis2SparseDecoder", sparse_builder)

    def build(kind: Literal["shape", "texture"], key: str) -> object:
        plan = cast("ComponentPlan[Trellis2DecoderConfig]", SimpleNamespace(keys={key: key}))
        config = Trellis2DecoderConfig(kind, 32, 7 if kind == "shape" else 6)
        return _decoder_builder(plan, compute_dtype=torch.bfloat16)(config, operations=INITLESS)

    assert build("shape", "shape_dec.from_latent.weight") is shape
    assert build("texture", "txt_dec.from_latent.weight") is texture
    assert build("shape", "from_latent.weight") is sparse
    assert build("texture", "from_latent.weight") is sparse


def test_microsoft_split_decoder_keeps_float32_boundary_layers() -> None:
    plan = cast(
        "ComponentPlan[Trellis2DecoderConfig]",
        SimpleNamespace(keys={"from_latent.weight": "from_latent.weight"}),
    )
    config = Trellis2DecoderConfig("shape", 32, 7)

    decoder = _decoder_builder(plan, compute_dtype=torch.float16)(config, operations=INITLESS)

    assert isinstance(decoder, Trellis2SparseDecoder)
    assert bound_compute_dtype(decoder.from_latent) is torch.float32
    assert bound_compute_dtype(decoder.output_layer) is torch.float32
    first_stage = cast(torch.nn.ModuleList, decoder.blocks[0])
    first_block = cast(Trellis2SparseConvNeXt, first_stage[0])
    assert bound_compute_dtype(first_block.norm) is None


def test_microsoft_split_state_keeps_only_block_linears_at_bfloat16() -> None:
    model = Trellis2FlowModel(
        _flow_config(),
        operations=INITLESS,
        compute_dtype=torch.bfloat16,
        microsoft_split_precision=True,
    )
    source = {
        key: torch.empty_like(value, dtype=torch.bfloat16)
        for key, value in model.state_dict().items()
    }

    state = _microsoft_split_state(model, source)

    assert state["blocks.0.self_attn.to_qkv.weight"].dtype is torch.bfloat16
    assert state["blocks.0.mlp.mlp.0.bias"].dtype is torch.bfloat16
    assert state["blocks.0.modulation"].dtype is torch.float32
    assert state["blocks.0.norm2.weight"].dtype is torch.float32
    assert state["blocks.0.self_attn.q_rms_norm.gamma"].dtype is torch.float32
    assert state["input_layer.weight"].dtype is torch.float32
    assert state["t_embedder.mlp.0.weight"].dtype is torch.float32
    assert state["adaLN_modulation.1.weight"].dtype is torch.float32
    assert state["out_layer.weight"].dtype is torch.float32


def test_microsoft_split_head_norm_matches_source_float32_math() -> None:
    value = torch.tensor([[[[1.25, -0.5, 0.75, 2.0, -1.0, 0.25]]]], dtype=torch.bfloat16)
    norm = Trellis2HeadRmsNorm(1, 6, microsoft_split_precision=True)
    gamma = torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]], dtype=torch.float32)
    norm.gamma.data.copy_(gamma)

    actual = norm(value)
    expected = (F.normalize(value.float(), dim=-1) * gamma * 6**0.5).to(torch.bfloat16)

    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, expected)


def test_fused_head_norm_keeps_comfy_rms_norm_math() -> None:
    value = torch.tensor([[[[1.25, -0.5, 0.75, 2.0, -1.0, 0.25]]]], dtype=torch.float32)
    norm = Trellis2HeadRmsNorm(1, 6, microsoft_split_precision=False)
    gamma = torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]], dtype=torch.float32)
    norm.gamma.data.copy_(gamma)

    assert torch.equal(norm(value), F.rms_norm(value, (6,)) * gamma)


def test_microsoft_split_rope_matches_source_complex_math() -> None:
    coordinates = torch.tensor(((0, 1, 2), (3, 4, 5)), dtype=torch.int32)
    value = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 12).to(torch.bfloat16)

    phases = _microsoft_rope_table(coordinates, 12)
    actual = _apply_microsoft_rope(value, phases)
    complex_value = torch.view_as_complex(value.float().reshape(1, 2, 2, 6, 2))
    expected = torch.view_as_real(complex_value * phases.unsqueeze(-2)).flatten(-2)

    assert phases.dtype is torch.complex64
    assert torch.equal(actual, expected.to(torch.bfloat16))


def test_sparse_rope_promotes_variable_length_rows_for_optimized_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.zeros((7, 4, 24))
    key = torch.ones_like(query)
    phases = torch.zeros((7, 12, 2, 2))
    observed: list[tuple[torch.Size, torch.Size]] = []

    def apply(
        q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed.append((q.shape, rope.shape))
        return q, k

    monkeypatch.setattr(trellis2_flow, "apply_rope", apply)

    q_rotated, k_rotated = _apply_sparse_rope(query, key, phases)

    assert observed == [((1, 7, 4, 24), (1, 7, 1, 12, 2, 2))]
    assert q_rotated is not query and torch.equal(q_rotated, query)
    assert k_rotated is not key and torch.equal(k_rotated, key)
