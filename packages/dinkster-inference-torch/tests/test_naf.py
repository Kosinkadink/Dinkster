# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from dinkster_inference import FLOAT32, ComponentPlan, NAFConfig
from dinkster_inference_torch.module_residency import enroll_component
from dinkster_inference_torch.naf import NAF, neighborhood_attention_2d
from dinkster_inference_torch.operations import INITLESS, bound_compute_dtype
from dinkster_inference_torch.trellis2_assembly import _load_vision_naf
from dinkster_inference_torch.trellis2_runtime import _upsample_naf_features
from safetensors.torch import save_file


def _upsample(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, _, _, heads, channels = value.shape
    value = value.permute(0, 3, 4, 1, 2).reshape(batch, heads * channels, *value.shape[1:3])
    value = F.interpolate(value, size=size, mode="nearest-exact")
    return value.view(batch, heads, channels, *size).permute(0, 3, 4, 1, 2)


def _indices(position: int, length: int, kernel: int, dilation: int) -> list[int]:
    residue = position % dilation
    position_in_class = position // dilation
    class_size = (length - 1 - residue) // dilation + 1
    start = min(max(position_in_class - kernel // 2, 0), class_size - kernel)
    return [residue + (start + offset) * dilation for offset in range(kernel)]


def _reference(
    query: torch.Tensor,
    key_lr: torch.Tensor,
    value_lr: torch.Tensor,
    kernel: tuple[int, int],
    dilation: tuple[int, int],
    scale: float,
    tile: int,
    value_chunk: int,
) -> torch.Tensor:
    batch, height, width, heads, _ = query.shape
    key = _upsample(key_lr, (height, width))
    value = _upsample(value_lr, (height, width))
    output = query.new_empty((batch, heads, value.shape[-1], height, width))
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            positions = [
                (row, column)
                for row in range(top, min(top + tile, height))
                for column in range(left, min(left + tile, width))
            ]
            keys, values = [], []
            for row, column in positions:
                rows = _indices(row, height, kernel[0], dilation[0])
                columns = _indices(column, width, kernel[1], dilation[1])
                k = key[:, rows, :, :, :][:, :, columns, :, :]
                v = value[:, rows, :, :, :][:, :, columns, :, :]
                keys.append(k.permute(0, 3, 1, 2, 4).reshape(batch, heads, -1, key.shape[-1]))
                values.append(v.permute(0, 3, 1, 2, 4).reshape(batch, heads, -1, value.shape[-1]))
            queries = torch.stack([query[:, r, c] for r, c in positions], dim=2).unsqueeze(-2)
            weights = (
                torch.matmul(queries, torch.stack(keys, dim=2).transpose(-1, -2)) * scale
            ).softmax(-1)
            # Preserve GEMM batch and output widths for bit-exact CPU accumulation.
            for start in range(0, value.shape[-1], value_chunk):
                chunks = torch.stack([v[..., start : start + value_chunk] for v in values], dim=2)
                attended = torch.matmul(weights, chunks).squeeze(-2)
                for index, (row, column) in enumerate(positions):
                    output[:, :, start : start + value_chunk, row, column] = attended[:, :, index]
    return output


def test_neighborhood_attention_matches_shifted_source_boundaries() -> None:
    generator = torch.Generator().manual_seed(1083)
    query = torch.randn((1, 8, 10, 2, 2), generator=generator, dtype=torch.float64)
    key = torch.randn((1, 4, 5, 2, 2), generator=generator, dtype=torch.float64)
    value = torch.randn((1, 4, 5, 2, 3), generator=generator, dtype=torch.float64)
    output = torch.empty((1, 2, 3, 8, 10), dtype=torch.float64)
    expected = _reference(query, key, value, (3, 3), (2, 2), 2**-0.5, tile=3, value_chunk=2)

    actual = neighborhood_attention_2d(
        query,
        key,
        value,
        kernel_size=(3, 3),
        dilation=(2, 2),
        scale=2**-0.5,
        tile=3,
        value_chunk=2,
        output=output,
    )

    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_naf_loader_keeps_fp32_compute_when_bfloat16_is_selected(tmp_path: Path) -> None:
    config = NAFConfig(
        channels=8,
        attention_heads=2,
        rope_heads=2,
        kernel_size=3,
        image_layers=0,
    )
    template = NAF(config, operations=INITLESS)
    state = {
        key: torch.full(tuple(value.shape), 0.01, dtype=torch.float32)
        for key, value in template.state_dict().items()
    }
    state["image_encoder.rope.periods"].fill_(2.0)
    path = tmp_path / "naf.safetensors"
    save_file(state, path)
    plan = ComponentPlan(
        "naf",
        path,
        config,
        {key: key for key in state},
        {key: FLOAT32 for key in state},
        {},
    )

    naf = _load_vision_naf(plan, storage_dtype=torch.bfloat16)
    image = torch.randn((1, 3, 6, 6), generator=torch.Generator().manual_seed(1083))
    features = torch.randn((1, 8, 3, 3), generator=torch.Generator().manual_seed(1084))
    expected = _upsample_naf_features(naf, image, features, 6, 6, torch.device("cpu"))

    assert {value.dtype for value in naf.state_dict().values()} == {torch.bfloat16}
    assert {
        dtype for module in naf.modules() if (dtype := bound_compute_dtype(module)) is not None
    } == {torch.float32}
    assert expected.dtype is torch.float32

    mechanism = enroll_component(naf, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    loaded = _upsample_naf_features(naf, image, features, 6, 6, torch.device("cpu"))
    assert {value.dtype for value in naf.state_dict().values()} == {torch.bfloat16}
    mechanism.unload()
    offloaded = _upsample_naf_features(naf, image, features, 6, 6, torch.device("cpu"))
    assert {value.dtype for value in naf.state_dict().values()} == {torch.bfloat16}
    torch.testing.assert_close(loaded, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(offloaded, expected, rtol=0.0, atol=0.0)
