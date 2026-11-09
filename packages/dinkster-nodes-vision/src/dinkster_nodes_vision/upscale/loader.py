"""Checkpoint detection and loading for the ESRGAN model family.

Detection, hyperparameter inference, and new-arch to old-arch key conversion
reproduce spandrel 0.4.2 (commit 724cca38) for the two supported
architectures. Anything else fails closed with the supported list; loading
is always ``strict=True`` so a partially matching checkpoint cannot silently
produce a wrong network.
"""

from __future__ import annotations

import functools
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import cast

import torch
from dinkster_api.v1 import AssetRef
from torch import nn

from .archs import RRDBNet, SRVGGNetCompact

SUPPORTED_ARCHITECTURES = (
    "ESRGAN-family RRDBNet (ESRGAN, ESRGAN+, BSRGAN, RealSR, Real-ESRGAN)",
    "RealESRGAN Compact (SRVGGNetCompact)",
)

StateDict = dict[str, torch.Tensor]


class UpscaleModelError(ValueError):
    """The checkpoint is not a loadable model of a supported architecture."""


@dataclass(frozen=True)
class LoadedUpscaler:
    """A ready ESRGAN-family module with its user-visible geometry: the
    channel count callers must feed it and the output scale they get back
    (both already account for any pixel-unshuffle wrapper). ``minimum`` and
    ``multiple_of`` are the module's spatial size requirements: callers
    must pad inputs (right/bottom) to at least ``minimum`` and to a
    multiple of ``multiple_of`` before the forward pass and crop the
    output back. ``minimum`` is always a multiple of ``multiple_of``,
    matching spandrel's normalized SizeRequirements."""

    module: nn.Module
    scale: int
    in_channels: int
    out_channels: int
    minimum: int
    multiple_of: int


_UNWRAP_KEYS = (
    "model_state_dict",
    "state_dict",
    "params_ema",
    "params-ema",
    "params",
    "model",
    "net",
)


def _canonicalize(raw: object) -> StateDict:
    if not isinstance(raw, dict):
        raise UpscaleModelError(
            f"upscale model checkpoint must contain a state dictionary, got {type(raw).__name__}"
        )
    state = cast("dict[object, object]", raw)
    for unwrap_key in _UNWRAP_KEYS:
        inner = state.get(unwrap_key)
        if isinstance(inner, dict):
            state = cast("dict[object, object]", inner)
            break
    if len(state) == 1:
        single = next(iter(state.values()))
        if isinstance(single, dict):
            state = cast("dict[object, object]", single)
    if not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise UpscaleModelError("upscale model checkpoint must map parameter names to tensors")
    canonical = cast("StateDict", state)
    for prefix in ("module.", "netG."):
        if all(key.startswith(prefix) for key in canonical):
            canonical = {key[len(prefix) :]: value for key, value in canonical.items()}
    return canonical


def _get_seq_len(state: StateDict, seq_key: str) -> int:
    prefix = seq_key + "."
    indices = {
        int(key[len(prefix) :].split(".", maxsplit=1)[0]) for key in state if key.startswith(prefix)
    }
    return max(indices) + 1 if indices else 0


_RDB_OLD_KEY = r"model.1.sub.\1.RDB\2.conv\3.0.\4"
_RDB_NEW_KEYS = (
    r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
    r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
)


def _get_num_blocks(state: StateDict) -> int:
    patterns = (
        *_RDB_NEW_KEYS,
        r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
    )
    blocks: list[int] = []
    for pattern in patterns:
        for key in state:
            match = re.search(pattern, key)
            if match:
                blocks.append(int(match.group(1)))
        if blocks:
            break
    if not blocks:
        raise UpscaleModelError("could not count RRDB blocks in the checkpoint")
    return max(blocks) + 1


def _to_old_arch(state: StateDict) -> StateDict:
    """Convert new-arch (Real-ESRGAN / BSRGAN / RealSR) keys to the flattened
    old-arch ``model.N`` layout; old-arch checkpoints pass through."""
    if "conv_first.weight" not in state:
        return state
    num_blocks = _get_num_blocks(state)
    state_map: dict[str, tuple[str, ...]] = {
        "model.0.weight": ("conv_first.weight",),
        "model.0.bias": ("conv_first.bias",),
        f"model.1.sub.{num_blocks}.weight": ("trunk_conv.weight", "conv_body.weight"),
        f"model.1.sub.{num_blocks}.bias": ("trunk_conv.bias", "conv_body.bias"),
        _RDB_OLD_KEY: _RDB_NEW_KEYS,
    }
    old_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for old_key, new_keys in state_map.items():
        for new_key in new_keys:
            if r"\1" in old_key:
                for key, value in state.items():
                    converted = re.sub(new_key, old_key, key)
                    if converted != key:
                        old_state[converted] = value
            elif new_key in state:
                old_state[old_key] = state[new_key]
    max_upconv = 0
    for key, value in state.items():
        match = re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key)
        if match is not None:
            _, index, kind = match.groups()
            old_state[f"model.{int(index) * 3}.{kind}"] = value
            max_upconv = max(max_upconv, int(index) * 3)
    for key, value in state.items():
        if key in ("HRconv.weight", "conv_hr.weight"):
            old_state[f"model.{max_upconv + 2}.weight"] = value
        elif key in ("HRconv.bias", "conv_hr.bias"):
            old_state[f"model.{max_upconv + 2}.bias"] = value
        elif key == "conv_last.weight":
            old_state[f"model.{max_upconv + 4}.weight"] = value
        elif key == "conv_last.bias":
            old_state[f"model.{max_upconv + 4}.bias"] = value

    def compare(item1: str, item2: str) -> int:
        return int(item1.split(".")[1]) - int(item2.split(".")[1])

    return OrderedDict(
        (key, old_state[key]) for key in sorted(old_state, key=functools.cmp_to_key(compare))
    )


def _is_esrgan(state: StateDict) -> bool:
    conditions = (
        ("model.0.weight", "model.1.sub.0.RDB1.conv1.0.weight"),
        ("conv_first.weight", "body.0.rdb1.conv1.weight", "conv_body.weight", "conv_last.weight"),
        (
            "conv_first.weight",
            "RRDB_trunk.0.RDB1.conv1.weight",
            "trunk_conv.weight",
            "conv_last.weight",
        ),
        ("model.0.weight", "model.1.sub.0.RDB1.conv1x1.weight"),
    )
    return any(all(key in state for key in condition) for condition in conditions)


def _load_esrgan(state: StateDict) -> LoadedUpscaler:
    state = _to_old_arch(state)
    seq_len = _get_seq_len(state, "model")
    in_channels = state["model.0.weight"].shape[1]
    out_channels = state[f"model.{seq_len - 1}.weight"].shape[0]
    filters = state["model.0.weight"].shape[0]
    scale = 2 ** ((seq_len - 5) // 3)
    blocks = _get_seq_len(state, "model.1.sub") - 1
    plus = any(".conv1x1." in key for key in state)
    shuffle_factor = None
    if in_channels in (out_channels * 4, out_channels * 16):
        shuffle_factor = int(math.sqrt(in_channels / out_channels))
    module = RRDBNet(
        in_channels=in_channels,
        out_channels=out_channels,
        filters=filters,
        blocks=blocks,
        scale=scale,
        plus=plus,
        shuffle_factor=shuffle_factor,
    )
    module.load_state_dict(state, strict=True)
    module.float().eval()
    if shuffle_factor:
        in_channels //= shuffle_factor**2
        scale //= shuffle_factor
    # spandrel declares SizeRequirements(minimum=2, multiple_of=4 if
    # shuffle_factor else 1); its initializer rounds minimum up to the
    # next multiple, so pixel-unshuffle models require 4/4.
    return LoadedUpscaler(
        module=module,
        scale=scale,
        in_channels=in_channels,
        out_channels=out_channels,
        minimum=4 if shuffle_factor else 2,
        multiple_of=4 if shuffle_factor else 1,
    )


def _is_compact(state: StateDict) -> bool:
    return "body.0.weight" in state and "body.1.weight" in state


def _load_compact(state: StateDict) -> LoadedUpscaler:
    highest = _get_seq_len(state, "body") - 1
    in_channels = state["body.0.weight"].shape[1]
    filters = state["body.0.weight"].shape[0]
    convs = (highest - 2) // 2
    pixelshuffle_shape = state[f"body.{highest}.bias"].shape[0]
    scale = 0
    out_channels = 0
    for candidate in (in_channels, 3, 4, 1):
        if pixelshuffle_shape % candidate:
            continue
        root = math.isqrt(pixelshuffle_shape // candidate)
        if root * root == pixelshuffle_shape // candidate:
            scale, out_channels = root, candidate
            break
    if not scale:
        raise UpscaleModelError(
            f"could not infer scale from pixelshuffle shape {pixelshuffle_shape}"
        )
    module = SRVGGNetCompact(
        in_channels=in_channels,
        out_channels=out_channels,
        filters=filters,
        convs=convs,
        scale=scale,
    )
    module.load_state_dict(state, strict=True)
    module.float().eval()
    return LoadedUpscaler(
        module=module,
        scale=scale,
        in_channels=in_channels,
        out_channels=out_channels,
        minimum=0,
        multiple_of=1,
    )


def load_upscaler(raw: object) -> LoadedUpscaler:
    """Build the network a checkpoint describes, failing closed on anything
    that is not a supported architecture."""
    state = _canonicalize(raw)
    if _is_compact(state):
        return _load_compact(state)
    if _is_esrgan(state):
        return _load_esrgan(state)
    raise UpscaleModelError(
        "unsupported upscale model architecture; supported: " + "; ".join(SUPPORTED_ARCHITECTURES)
    )


_CACHED: LoadedUpscaler | None = None
_CACHED_DIGEST = ""


def upscaler_for_asset(reference: AssetRef) -> LoadedUpscaler:
    """Load (or reuse) the model behind an asset; the cache holds the one
    most recently used model, keyed by content digest."""
    global _CACHED, _CACHED_DIGEST
    if _CACHED is not None and _CACHED_DIGEST == reference.digest:
        return _CACHED
    with reference.open() as stream:
        raw: object = torch.load(stream, map_location="cpu", weights_only=True)
    loaded = load_upscaler(raw)
    _CACHED = loaded
    _CACHED_DIGEST = reference.digest
    return loaded


__all__ = [
    "SUPPORTED_ARCHITECTURES",
    "LoadedUpscaler",
    "UpscaleModelError",
    "load_upscaler",
    "upscaler_for_asset",
]
