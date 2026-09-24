from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_vision.upscale import register_types
from dinkster_nodes_vision.upscale.archs import RRDBNet, SRVGGNetCompact
from dinkster_nodes_vision.upscale.execute import _apply_model, execute_upscale, tiled_scale_2d
from dinkster_nodes_vision.upscale.loader import LoadedUpscaler, UpscaleModelError, load_upscaler
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages/dinkster-nodes-vision/dinkster_vision_upscale_pack/dinkster-pack.toml"


def _tiny_rrdb(**overrides: object) -> RRDBNet:
    torch.manual_seed(0)
    settings: dict[str, object] = {
        "in_channels": 3,
        "out_channels": 3,
        "filters": 16,
        "blocks": 1,
        "scale": 2,
    }
    settings.update(overrides)
    return RRDBNet(**settings).float().eval()  # type: ignore[arg-type]


def _tiny_compact() -> SRVGGNetCompact:
    torch.manual_seed(0)
    return (
        SRVGGNetCompact(in_channels=3, out_channels=3, filters=8, convs=2, scale=2).float().eval()
    )


def _assert_same_model(loaded_module: torch.nn.Module, reference: torch.nn.Module) -> None:
    torch.manual_seed(1)
    probe = torch.rand(1, reference_in_channels(reference), 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(loaded_module(probe), reference(probe))


def reference_in_channels(module: torch.nn.Module) -> int:
    for parameter in module.parameters():
        return parameter.shape[1]
    raise AssertionError("module has no parameters")


def test_old_arch_esrgan_state_dict_loads() -> None:
    reference = _tiny_rrdb()
    loaded = load_upscaler(dict(reference.state_dict()))
    assert isinstance(loaded.module, RRDBNet)
    assert loaded.scale == 2
    assert loaded.in_channels == 3
    assert loaded.out_channels == 3
    assert loaded.minimum == 2
    assert loaded.multiple_of == 1
    _assert_same_model(loaded.module, reference)


def _to_new_arch(state: dict[str, torch.Tensor], blocks: int) -> dict[str, torch.Tensor]:
    """Rename a scale-4 old-arch state dict to Real-ESRGAN key names."""
    renames = {
        "model.0": "conv_first",
        f"model.1.sub.{blocks}": "conv_body",
        "model.3": "conv_up1",
        "model.6": "conv_up2",
        "model.8": "conv_hr",
        "model.10": "conv_last",
    }
    new_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for old, new in renames.items():
            if key.startswith(old + "."):
                new_state[new + key[len(old) :]] = value
                break
        else:
            prefix, _, rest = key.partition(".sub.")
            assert prefix == "model.1"
            index, block, conv, _zero, kind = rest.split(".")
            new_state[f"body.{index}.{block.lower()}.{conv}.{kind}"] = value
    return new_state


def test_new_arch_realesrgan_keys_convert_and_match_old_arch() -> None:
    reference = _tiny_rrdb(scale=4, blocks=2)
    old_state = dict(reference.state_dict())
    loaded = load_upscaler({"params_ema": _to_new_arch(old_state, blocks=2)})
    assert isinstance(loaded.module, RRDBNet)
    assert loaded.scale == 4
    _assert_same_model(loaded.module, reference)


def test_esrgan_plus_conv1x1_variant_loads() -> None:
    reference = _tiny_rrdb(plus=True)
    loaded = load_upscaler(dict(reference.state_dict()))
    assert isinstance(loaded.module, RRDBNet)
    _assert_same_model(loaded.module, reference)


def test_pixel_unshuffle_variant_reports_effective_geometry() -> None:
    reference = _tiny_rrdb(in_channels=12, scale=4, shuffle_factor=2)
    loaded = load_upscaler(dict(reference.state_dict()))
    assert loaded.scale == 2
    assert loaded.in_channels == 3
    assert loaded.out_channels == 3
    assert loaded.minimum == 4
    assert loaded.multiple_of == 4
    with torch.no_grad():
        output = loaded.module(torch.rand(1, 3, 7, 5))
    assert output.shape == (1, 3, 14, 10)


def test_compact_params_wrapper_loads() -> None:
    reference = _tiny_compact()
    loaded = load_upscaler({"params": dict(reference.state_dict())})
    assert isinstance(loaded.module, SRVGGNetCompact)
    assert loaded.scale == 2
    assert loaded.in_channels == 3
    assert loaded.out_channels == 3
    assert loaded.minimum == 0
    assert loaded.multiple_of == 1
    _assert_same_model(loaded.module, reference)


def test_unknown_architecture_fails_closed() -> None:
    with pytest.raises(UpscaleModelError, match="supported"):
        load_upscaler({"foo.weight": torch.zeros(1)})


def test_non_state_dict_checkpoint_fails_closed() -> None:
    with pytest.raises(UpscaleModelError, match="state dictionary"):
        load_upscaler(torch.zeros(1))
    with pytest.raises(UpscaleModelError, match="tensors"):
        load_upscaler({"model.0.weight": "not a tensor"})


def _nearest_2x(value: torch.Tensor) -> torch.Tensor:
    return value.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


@pytest.mark.parametrize(
    ("height", "width"),
    [(20, 20), (8, 8), (5, 5), (17, 13), (8, 20)],
)
def test_tiled_scale_matches_whole_image_for_pointwise_models(height: int, width: int) -> None:
    torch.manual_seed(2)
    batch = torch.rand(2, 3, height, width)
    tiled = tiled_scale_2d(batch, _nearest_2x, tile=8, overlap=4, scale=2, out_channels=3)
    torch.testing.assert_close(tiled, _nearest_2x(batch))


def test_tiled_scale_identity_and_zero_overlap() -> None:
    torch.manual_seed(3)
    batch = torch.rand(1, 3, 20, 20)
    identity = tiled_scale_2d(
        batch, lambda value: value, tile=8, overlap=2, scale=1, out_channels=3
    )
    torch.testing.assert_close(identity, batch)
    unblended = tiled_scale_2d(batch, _nearest_2x, tile=8, overlap=0, scale=2, out_channels=3)
    torch.testing.assert_close(unblended, _nearest_2x(batch))


class _Gain(torch.nn.Module):
    """Pointwise gain at scale 1, used to drive outputs outside [0, 1]."""

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.gain


class _Nearest2x(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _nearest_2x(value)


def test_apply_model_clamps_every_call() -> None:
    model = LoadedUpscaler(
        module=_Gain(3.0), scale=1, in_channels=3, out_channels=3, minimum=0, multiple_of=1
    )
    tile = torch.full((1, 3, 4, 4), 0.5)
    output = _apply_model(model, tile)
    torch.testing.assert_close(output, torch.ones_like(tile))
    negative = _apply_model(model, torch.full((1, 3, 4, 4), -0.5))
    torch.testing.assert_close(negative, torch.zeros_like(tile))


def test_apply_model_pads_below_minimum_and_crops_the_output() -> None:
    model = LoadedUpscaler(
        module=_Nearest2x(), scale=2, in_channels=3, out_channels=3, minimum=2, multiple_of=1
    )
    tile = torch.full((1, 3, 1, 1), 0.25)
    output = _apply_model(model, tile)
    assert output.shape == (1, 3, 2, 2)
    torch.testing.assert_close(output, torch.full((1, 3, 2, 2), 0.25))
    tall = torch.rand(1, 3, 4, 1)
    output = _apply_model(model, tall)
    assert output.shape == (1, 3, 8, 2)
    torch.testing.assert_close(output, _nearest_2x(tall))


@pytest.mark.parametrize("shuffle_factor", [2, 4])
@pytest.mark.parametrize(("height", "width"), [(1, 1), (2, 5), (7, 5)])
def test_apply_model_meets_pixel_unshuffle_size_requirements(
    shuffle_factor: int, height: int, width: int
) -> None:
    reference = _tiny_rrdb(
        in_channels=3 * shuffle_factor**2, scale=4, shuffle_factor=shuffle_factor
    )
    loaded = load_upscaler(dict(reference.state_dict()))
    assert loaded.minimum == 4
    assert loaded.multiple_of == 4
    forward_sizes: list[tuple[int, int]] = []

    class _Recorder(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            forward_sizes.append((value.shape[2], value.shape[3]))
            return loaded.module(value)

    model = replace(loaded, module=_Recorder())
    tile = torch.rand(1, 3, height, width)
    with torch.no_grad():
        output = _apply_model(model, tile)
    assert output.shape == (1, 3, height * loaded.scale, width * loaded.scale)
    assert forward_sizes == [
        (
            max(4, height + (-height) % 4),
            max(4, width + (-width) % 4),
        )
    ]
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def _model_asset(tmp_path: Path, module: torch.nn.Module) -> tuple[AssetVault, AssetRef, bytes]:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    payload = buffer.getvalue()
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    ref = AssetRef(digest, "tiny.pth", len(payload), resolver=vault)
    return vault, ref, payload


def test_execute_upscale_handles_channel_policy(tmp_path: Path) -> None:
    reference = _tiny_compact()
    _, ref, _ = _model_asset(tmp_path, reference)
    rgb = np.random.default_rng(4).random((1, 6, 5, 3), dtype=np.float32)
    output = execute_upscale(rgb, ref, tile_size=512, overlap=32)
    assert output.shape == (1, 12, 10, 3)
    assert output.dtype == np.float32
    assert output.min() >= 0.0 and output.max() <= 1.0
    grayscale = rgb[..., :1]
    output = execute_upscale(grayscale, ref, tile_size=512, overlap=32)
    assert output.shape == (1, 12, 10, 3)
    with pytest.raises(ValueError, match="channels"):
        execute_upscale(np.zeros((1, 4, 4, 4), dtype=np.float32), ref, tile_size=512, overlap=32)


def test_execute_upscale_validates_inputs(tmp_path: Path) -> None:
    _, ref, _ = _model_asset(tmp_path, _tiny_compact())
    image = np.zeros((1, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="overlap"):
        execute_upscale(image, ref, tile_size=128, overlap=128)
    with pytest.raises(ValueError, match="finite"):
        execute_upscale(
            np.full((1, 4, 4, 3), np.nan, dtype=np.float32),
            ref,
            tile_size=512,
            overlap=32,
        )
    with pytest.raises(TypeError, match="asset"):
        execute_upscale(image, "not-an-asset", tile_size=512, overlap=32)


def test_upscale_provider_executes_in_an_isolated_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        reference = _tiny_compact()
        vault, ref, payload = _model_asset(tmp_path, reference)
        source = np.random.default_rng(5).random((1, 6, 5, 3), dtype=np.float32)
        expected = execute_upscale(source, ref, tile_size=512, overlap=32)
        registry = TypeRegistry()
        register_core_types(registry)
        register_types(registry)
        worker = IsolatedWorker(
            MANIFEST,
            registry,
            python=sys.executable,
            extra_env={"DINKSTER_ASSET_VAULT": str(vault.root)},
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "up": GraphNode(
                        "dinkster.image.upscale_model",
                        {
                            "image": TypedLiteral("dinkster.image", source.tolist()),
                            "upscale_model": TypedLiteral(
                                "dinkster.asset",
                                {
                                    "digest": ref.digest,
                                    "name": ref.name,
                                    "size": len(payload),
                                },
                            ),
                            "provider": "dinkster-vision-upscale",
                        },
                    )
                }
            )
            result = await engine.run(graph, ["up"])
            output = np.asarray(result.outputs["up"]["image"].resolve())
            np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-6)
        finally:
            await worker.close()

    asyncio.run(scenario())


GOLDEN_PATH = ROOT / "tests" / "goldens" / "upscale_realesrgan_a1079ba1.json"
GOLDEN_MODELS = {
    "anime6b": (
        "DINKSTER_UPSCALE_TEST_ANIME6B",
        "blake3:717c1bcb17218786f29dd5377dd53a905fb5ec33c6ca12cb8dc0e3f2f18fa1b7",
        "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    ),
    "general": (
        "DINKSTER_UPSCALE_TEST_GENERAL",
        "blake3:c24daac1228a6a1d035f71368d28cd39fcb9457347b1137ef94f9d685bae0e0b",
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    ),
}


def _golden() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))


def _decode(record: object) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload["uint8Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(shape)


def _golden_model_path(name: str) -> Path:
    variable = GOLDEN_MODELS[name][0]
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{variable} does not exist: {path}")
    return path


def _golden_asset(tmp_path: Path, name: str) -> AssetRef:
    path = _golden_model_path(name)
    digest = GOLDEN_MODELS[name][1]
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                writer.write(chunk)
        writer.commit()
    return AssetRef(digest, path.name, path.stat().st_size, resolver=vault)


def test_realesrgan_goldens_pin_the_reference_environment() -> None:
    golden = _golden()
    assert golden["baseline"] == "a1079ba16f2674734b065eb036fbfdddaa321a4d"
    assert golden["generationCpu"] == "AMD Ryzen 9 5950X 16-Core Processor"
    assert golden["numpy"] == "2.5.1"
    assert golden["spandrel"] == "0.4.2"
    assert golden["torch"] == "2.13.0+cpu"
    assert golden["scales"] == {"anime6b": 4, "general": 4}
    assert golden["tilings"] == {"full": [512, 32], "tiled": [32, 8]}
    models = cast("dict[str, dict[str, str]]", golden["models"])
    for name, (_, blake3_digest, sha256_digest) in GOLDEN_MODELS.items():
        assert models[name]["blake3"] == blake3_digest
        assert models[name]["sha256"] == sha256_digest
    source = _decode(golden["source"])
    assert source.shape == (40, 56, 3)
    cases = cast("dict[str, dict[str, object]]", golden["cases"])
    assert set(cases) == set(GOLDEN_MODELS)
    for outputs in cases.values():
        assert set(outputs) == {"full", "tiled"}
        for record in outputs.values():
            assert _decode(record).shape == (160, 224, 3)


@pytest.mark.parametrize("tiling", ["full", "tiled"])
@pytest.mark.parametrize("name", sorted(GOLDEN_MODELS))
def test_upscale_outputs_match_pinned_comfyui_vectors(
    tmp_path: Path, name: str, tiling: str
) -> None:
    golden = _golden()
    reference = _golden_asset(tmp_path, name)
    source = _decode(golden["source"])[None].astype(np.float32) / 255.0
    tile_size, overlap = cast("dict[str, list[int]]", golden["tilings"])[tiling]
    output = execute_upscale(source, reference, tile_size=tile_size, overlap=overlap)
    expected = _decode(cast("dict[str, dict[str, object]]", golden["cases"])[name][tiling])
    assert output.shape == (1, *expected.shape)
    assert output.dtype == np.float32
    # Hosted CPU kernels differed by at most one uint8 level; the tolerance has no headroom.
    np.testing.assert_allclose(
        output[0],
        expected.astype(np.float32) / 255,
        rtol=0,
        atol=1 / 255,
    )


@pytest.mark.parametrize("name", sorted(GOLDEN_MODELS))
def test_golden_model_artifacts_are_the_expected_bytes(name: str) -> None:
    path = _golden_model_path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == GOLDEN_MODELS[name][2]
