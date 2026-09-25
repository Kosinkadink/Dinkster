"""LoRA dialect decoding: key-layout classification, torch-free.

ComfyUI decodes a LoRA file by probing key suffixes per mapped stem
and building adapter objects that hold tensors (comfy/lora.py
load_lora, comfy/lora_convert.py convert_lora,
comfy/weight_adapter/*.load @ b78cec87). The decode DECISIONS are
almost entirely key-string logic; the only value-dependent inputs are
tensor ranks (OFT vs BOFT share the ``.oft_blocks`` suffix, split by
ndim) and two shape reads (BFL control reshape synthesis) - all of
which a safetensors header provides as TensorGeometry without touching
payload bytes.

This module ports the decisions and defers every value read: specs
reference SOURCE keys (the names in the file as shipped, before any
dialect rename), and the stage-4 torch layer materializes them into
WeightAdapter implementations by reading exactly those keys. Value
reads deliberately deferred: ``.alpha`` scalars, ``.dora_scale``
tensors, ``.reshape_weight`` dimension lists.

Reference-faithful semantics kept on purpose:

- All six adapter probes run per stem; a later success overwrites an
  earlier one at the same target, and the diff/set probes after them
  can overwrite again (comfy/lora.py:53-88 probes without early exit).
- ``.alpha``/``.dora_scale`` are marked consumed whenever a mapped
  stem has them, even if no adapter matches (comfy/lora.py:41-51).
- ``b_norm`` is only consumed when ``w_norm`` is present.
- LoKr classification is permissive: any one of w1/w2/w1_a/w2_a
  qualifies (comfy/weight_adapter/lokr.py:254-259).

Loud deviations (upstream crashes or silently corrupts; we diagnose):

- A missing companion tensor (LoHa without ``hada_w1_b``, GLoRA
  without ``a2``...) raises KeyError upstream; here the adapter probe
  fails with a diagnostic and the stem falls through.
- Bias-targeted patches (``diff_b``/``b_norm``) on a target that does
  not end in ``.weight`` would be silently mangled upstream
  (unconditional ``[:-len(".weight")]`` slice); here they produce a
  diagnostic and no patch. Offset (fused-qkv) targets get the same
  treatment - upstream would crash slicing a tuple.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .patches import PatchOffset
from .unet import UNetConfig
from .weights import TensorGeometry

DIALECT_NONE = "none"
DIALECT_BFL_FLUX_CONTROL = "bfl_flux_control"
DIALECT_WAN_FUN = "wan_fun"
DIALECT_USO = "uso"


@dataclass(frozen=True)
class NormalizedLora:
    """convert_lora ported: sentinel detection plus key rewrites.

    ``keys`` maps normalized key -> source key (the file's own name).
    ``geometries`` is in normalized key space. ``synthesized_reshapes``
    holds dimension lists the conversion computed from shapes (BFL
    control's ``img_in.reshape_weight``); those keys exist in no file,
    so their dims travel as data.
    """

    dialect: str
    keys: Mapping[str, str]
    geometries: Mapping[str, TensorGeometry]
    synthesized_reshapes: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))
        object.__setattr__(self, "geometries", MappingProxyType(dict(self.geometries)))
        object.__setattr__(
            self,
            "synthesized_reshapes",
            MappingProxyType(dict(self.synthesized_reshapes)),
        )


def normalize_lora_keys(
    geometries: Mapping[str, TensorGeometry],
) -> NormalizedLora:
    """Detect and undo dialect-specific key layouts.

    Ported from comfy/lora_convert.py convert_lora @ b78cec87:
    sentinel-key detection, first match wins, identity fallback.
    """
    if "img_in.lora_A.weight" in geometries and "single_blocks.0.norm.key_norm.scale" in geometries:
        return _normalize_bfl_control(geometries)
    if "lora_unet__blocks_0_cross_attn_k.lora_down.weight" in geometries:
        return _normalize_wan_fun(geometries)
    if (
        "single_blocks.37.processor.qkv_lora.up.weight" in geometries
        and "double_blocks.18.processor.qkv_lora2.up.weight" in geometries
    ):
        return _normalize_uso(geometries)
    return NormalizedLora(
        dialect=DIALECT_NONE,
        keys={k: k for k in geometries},
        geometries=dict(geometries),
        synthesized_reshapes={},
    )


def _normalize_bfl_control(
    geometries: Mapping[str, TensorGeometry],
) -> NormalizedLora:
    keys: dict[str, str] = {}
    geoms: dict[str, TensorGeometry] = {}
    for k, geometry in geometries.items():
        k_to = "diffusion_model.{}".format(
            k.replace(".lora_B.bias", ".diff_b").replace("_norm.scale", "_norm.set_weight")
        )
        keys[k_to] = k
        geoms[k_to] = geometry
    reshape = (
        geometries["img_in.lora_B.weight"].shape[0],
        geometries["img_in.lora_A.weight"].shape[1],
    )
    return NormalizedLora(
        dialect=DIALECT_BFL_FLUX_CONTROL,
        keys=keys,
        geometries=geoms,
        synthesized_reshapes={"diffusion_model.img_in.reshape_weight": reshape},
    )


def _normalize_wan_fun(
    geometries: Mapping[str, TensorGeometry],
) -> NormalizedLora:
    keys: dict[str, str] = {}
    geoms: dict[str, TensorGeometry] = {}
    for k, geometry in geometries.items():
        k_to = "lora_unet_" + k[len("lora_unet__") :] if k.startswith("lora_unet__") else k
        keys[k_to] = k
        geoms[k_to] = geometry
    return NormalizedLora(
        dialect=DIALECT_WAN_FUN,
        keys=keys,
        geometries=geoms,
        synthesized_reshapes={},
    )


def _normalize_uso(
    geometries: Mapping[str, TensorGeometry],
) -> NormalizedLora:
    keys: dict[str, str] = {}
    geoms: dict[str, TensorGeometry] = {}
    for k, geometry in geometries.items():
        k_to = "diffusion_model.{}".format(
            k.replace(".down.weight", ".lora_down.weight")
            .replace(".up.weight", ".lora_up.weight")
            .replace(".qkv_lora2.", ".txt_attn.qkv.")
            .replace(".qkv_lora1.", ".img_attn.qkv.")
            .replace(".proj_lora1.", ".img_attn.proj.")
            .replace(".proj_lora2.", ".txt_attn.proj.")
            .replace(".qkv_lora.", ".linear1_qkv.")
            .replace(".proj_lora.", ".linear2.")
            .replace(".processor.", ".")
        )
        keys[k_to] = k
        geoms[k_to] = geometry
    return NormalizedLora(dialect=DIALECT_USO, keys=keys, geometries=geoms, synthesized_reshapes={})


# --------------------------------------------------------------------------
# Decoded patch specs. Every str field is a SOURCE key into the LoRA
# file; the torch layer reads it there. Docstrings pin the reference
# weights-tuple slot each field corresponds to, because the tuple
# layout is the de facto wire format of comfy/weight_adapter/*.


@dataclass(frozen=True)
class LoRASpec:
    """comfy/weight_adapter/lora.py load @ b78cec87.

    Reference weights tuple: (up, down, alpha, mid, dora_scale,
    reshape). ``variant`` names the naming scheme that matched -
    upstream forgets it, but it is free here and useful in
    diagnostics.
    """

    up: str
    down: str
    alpha: str | None = None
    mid: str | None = None
    dora_scale: str | None = None
    reshape: str | None = None
    reshape_dims: tuple[int, ...] | None = None
    variant: str = "kohya"


@dataclass(frozen=True)
class LoHaSpec:
    """comfy/weight_adapter/loha.py load @ b78cec87.

    Reference weights tuple: (w1_a, w1_b, alpha, w2_a, w2_b, t1, t2,
    dora_scale). t1/t2 travel together (Tucker decomposition).
    """

    w1_a: str
    w1_b: str
    w2_a: str
    w2_b: str
    alpha: str | None = None
    t1: str | None = None
    t2: str | None = None
    dora_scale: str | None = None


@dataclass(frozen=True)
class LoKrSpec:
    """comfy/weight_adapter/lokr.py load @ b78cec87.

    Reference weights tuple: (w1, w2, alpha, w1_a, w1_b, w2_a, w2_b,
    t2, dora_scale). Classification is permissive like the reference:
    any of w1/w2/w1_a/w2_a qualifies; incomplete decompositions
    surface at materialization, not here.
    """

    w1: str | None = None
    w2: str | None = None
    w1_a: str | None = None
    w1_b: str | None = None
    w2_a: str | None = None
    w2_b: str | None = None
    t2: str | None = None
    alpha: str | None = None
    dora_scale: str | None = None


@dataclass(frozen=True)
class GLoRASpec:
    """comfy/weight_adapter/glora.py load @ b78cec87.

    Reference weights tuple: (a1, a2, b1, b2, alpha, dora_scale).
    Old-vs-new orientation is decided from tensor shapes at
    materialization (glora.py:63-77), not at decode.
    """

    a1: str
    a2: str
    b1: str
    b2: str
    alpha: str | None = None
    dora_scale: str | None = None


@dataclass(frozen=True)
class OFTSpec:
    """comfy/weight_adapter/oft.py load @ b78cec87: rank-3 blocks.

    Reference weights tuple: (blocks, rescale, alpha, dora_scale).
    """

    blocks: str
    rescale: str | None = None
    alpha: str | None = None
    dora_scale: str | None = None


@dataclass(frozen=True)
class BOFTSpec:
    """comfy/weight_adapter/boft.py load @ b78cec87: rank-4 blocks,
    same key suffixes as OFT.

    Reference weights tuple: (blocks, rescale, alpha, dora_scale).
    """

    blocks: str
    rescale: str | None = None
    alpha: str | None = None
    dora_scale: str | None = None


@dataclass(frozen=True)
class DiffPatchRef:
    """("diff", (tensor,)) in the reference: add the tensor to the
    weight at materialization."""

    key: str


@dataclass(frozen=True)
class SetPatchRef:
    """("set", (tensor,)) in the reference: replace the weight."""

    key: str


AdapterSpec = LoRASpec | LoHaSpec | LoKrSpec | GLoRASpec | OFTSpec | BOFTSpec
DecodedPatch = AdapterSpec | DiffPatchRef | SetPatchRef


@dataclass(frozen=True)
class PatchTarget:
    """Where a decoded patch lands: a model weight key, optionally a
    narrow window into it (Flux fused-qkv slice targets,
    comfy/lora.py:276-279)."""

    key: str
    offset: PatchOffset | None = None


@dataclass(frozen=True)
class LoraDecodeResult:
    """What a LoRA file means against one key map.

    ``patches`` is target -> decoded spec. ``loaded_keys`` and
    ``unmatched`` are SOURCE key names; unmatched keys are the
    reference's "lora key not loaded" warnings as data. Diagnostics
    record the documented deviations (missing companions, unusable
    bias targets).
    """

    dialect: str
    patches: Mapping[PatchTarget, DecodedPatch]
    loaded_keys: frozenset[str]
    unmatched: tuple[str, ...]
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "patches", MappingProxyType(dict(self.patches)))


def decode_lora(
    geometries: Mapping[str, TensorGeometry],
    key_map: Mapping[str, str | PatchTarget],
) -> LoraDecodeResult:
    """Classify a LoRA file's keys against a stem -> target key map.

    Ported from comfy/lora.py load_lora @ b78cec87, operating on
    header geometries instead of tensors. ``key_map`` values may be
    plain model weight keys or PatchTarget for sliced destinations.
    """
    normalized = normalize_lora_keys(geometries)
    nkeys = normalized.keys
    loaded: set[str] = set()  # normalized-space bookkeeping
    patches: dict[PatchTarget, DecodedPatch] = {}
    diagnostics: list[str] = []

    def src(nk: str) -> str | None:
        return nkeys.get(nk)

    for stem, raw_target in key_map.items():
        target = raw_target if isinstance(raw_target, PatchTarget) else PatchTarget(raw_target)

        alpha_name = f"{stem}.alpha"
        alpha = None
        if alpha_name in nkeys:
            alpha = src(alpha_name)
            loaded.add(alpha_name)

        dora_name = f"{stem}.dora_scale"
        dora_scale = None
        if dora_name in nkeys:
            dora_scale = src(dora_name)
            loaded.add(dora_name)

        for probe in _ADAPTER_PROBES:
            spec = probe(stem, normalized, alpha, dora_scale, loaded, diagnostics)
            if spec is not None:
                patches[target] = spec

        w_norm_name = f"{stem}.w_norm"
        b_norm_name = f"{stem}.b_norm"
        if w_norm_name in nkeys:
            loaded.add(w_norm_name)
            patches[target] = DiffPatchRef(key=src(w_norm_name) or w_norm_name)
            if b_norm_name in nkeys:
                loaded.add(b_norm_name)
                bt = _bias_target(target, stem, "b_norm", diagnostics)
                if bt is not None:
                    patches[bt] = DiffPatchRef(key=src(b_norm_name) or b_norm_name)

        diff_name = f"{stem}.diff"
        if diff_name in nkeys:
            loaded.add(diff_name)
            patches[target] = DiffPatchRef(key=src(diff_name) or diff_name)

        diff_b_name = f"{stem}.diff_b"
        if diff_b_name in nkeys:
            loaded.add(diff_b_name)
            bt = _bias_target(target, stem, "diff_b", diagnostics)
            if bt is not None:
                patches[bt] = DiffPatchRef(key=src(diff_b_name) or diff_b_name)

        set_name = f"{stem}.set_weight"
        if set_name in nkeys:
            loaded.add(set_name)
            patches[target] = SetPatchRef(key=src(set_name) or set_name)

    unmatched = tuple(nkeys[nk] for nk in nkeys if nk not in loaded)
    return LoraDecodeResult(
        dialect=normalized.dialect,
        patches=patches,
        loaded_keys=frozenset(nkeys[nk] for nk in loaded if nk in nkeys),
        unmatched=unmatched,
        diagnostics=tuple(diagnostics),
    )


def _bias_target(
    target: PatchTarget, stem: str, kind: str, diagnostics: list[str]
) -> PatchTarget | None:
    """The reference derives bias targets by blindly slicing
    ``.weight`` off the target string (comfy/lora.py:70,82); on any
    other target that silently corrupts. Here: diagnostic, no patch."""
    if target.offset is not None or not target.key.endswith(".weight"):
        diagnostics.append(
            f"lora.{kind}: cannot derive a bias target from"
            f" {target.key!r} (stem {stem!r}); patch dropped"
        )
        return None
    return PatchTarget(target.key[: -len(".weight")] + ".bias")


_LORA_VARIANTS: tuple[tuple[str, str, str, bool], ...] = (
    # (variant, up suffix, down suffix, has mid)
    ("kohya", ".lora_up.weight", ".lora_down.weight", True),
    ("diffusers", "_lora.up.weight", "_lora.down.weight", False),
    ("peft", ".lora_B.weight", ".lora_A.weight", False),
    ("diffusers3", ".lora.up.weight", ".lora.down.weight", False),
    ("mochi", ".lora_B", ".lora_A", False),
    (
        "transformers",
        ".lora_linear_layer.up.weight",
        ".lora_linear_layer.down.weight",
        False,
    ),
    ("peft_qwen", ".lora_B.default.weight", ".lora_A.default.weight", False),
)


def _probe_lora(
    stem: str,
    normalized: NormalizedLora,
    alpha: str | None,
    dora_scale: str | None,
    loaded: set[str],
    diagnostics: list[str],
) -> LoRASpec | None:
    nkeys = normalized.keys
    for variant, up_sfx, down_sfx, has_mid in _LORA_VARIANTS:
        up_name = stem + up_sfx
        if up_name not in nkeys:
            continue
        down_name = stem + down_sfx
        if down_name not in nkeys:
            diagnostics.append(
                f"lora.{variant}: {up_name!r} present but companion"
                f" {down_name!r} missing; adapter skipped"
            )
            return None
        mid = None
        if has_mid:
            mid_name = stem + ".lora_mid.weight"
            if mid_name in nkeys:
                mid = nkeys[mid_name]
                loaded.add(mid_name)
        reshape = None
        reshape_dims = None
        reshape_name = f"{stem}.reshape_weight"
        if reshape_name in nkeys:
            reshape = nkeys[reshape_name]
            loaded.add(reshape_name)
        elif reshape_name in normalized.synthesized_reshapes:
            reshape_dims = normalized.synthesized_reshapes[reshape_name]
        loaded.add(up_name)
        loaded.add(down_name)
        return LoRASpec(
            up=nkeys[up_name],
            down=nkeys[down_name],
            alpha=alpha,
            mid=mid,
            dora_scale=dora_scale,
            reshape=reshape,
            reshape_dims=reshape_dims,
            variant=variant,
        )
    return None


def _probe_loha(
    stem: str,
    normalized: NormalizedLora,
    alpha: str | None,
    dora_scale: str | None,
    loaded: set[str],
    diagnostics: list[str],
) -> LoHaSpec | None:
    nkeys = normalized.keys
    w1_a_name = f"{stem}.hada_w1_a"
    if w1_a_name not in nkeys:
        return None
    companions = (f"{stem}.hada_w1_b", f"{stem}.hada_w2_a", f"{stem}.hada_w2_b")
    missing = [c for c in companions if c not in nkeys]
    if missing:
        diagnostics.append(
            f"loha: {w1_a_name!r} present but companions missing: {missing}; adapter skipped"
        )
        return None
    t1 = t2 = None
    t1_name = f"{stem}.hada_t1"
    t2_name = f"{stem}.hada_t2"
    if t1_name in nkeys:
        if t2_name not in nkeys:
            diagnostics.append(f"loha: {t1_name!r} present without {t2_name!r}; adapter skipped")
            return None
        t1 = nkeys[t1_name]
        t2 = nkeys[t2_name]
        loaded.add(t1_name)
        loaded.add(t2_name)
    for name in (w1_a_name, *companions):
        loaded.add(name)
    return LoHaSpec(
        w1_a=nkeys[w1_a_name],
        w1_b=nkeys[companions[0]],
        w2_a=nkeys[companions[1]],
        w2_b=nkeys[companions[2]],
        alpha=alpha,
        t1=t1,
        t2=t2,
        dora_scale=dora_scale,
    )


_LOKR_FIELDS = (
    "lokr_w1",
    "lokr_w2",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
)


def _probe_lokr(
    stem: str,
    normalized: NormalizedLora,
    alpha: str | None,
    dora_scale: str | None,
    loaded: set[str],
    diagnostics: list[str],
) -> LoKrSpec | None:
    nkeys = normalized.keys
    found: dict[str, str] = {}
    for field_name in _LOKR_FIELDS:
        name = f"{stem}.{field_name}"
        if name in nkeys:
            found[field_name] = nkeys[name]
            loaded.add(name)
    if not (
        "lokr_w1" in found or "lokr_w2" in found or "lokr_w1_a" in found or "lokr_w2_a" in found
    ):
        # Deviation from the reference quirk: lokr.py mutates the
        # shared loaded_keys set BEFORE its qualification check, so a
        # lone lokr_w1_b upstream is consumed-but-unused with no
        # warning. Undo the marks so it surfaces as unmatched here.
        for field_name in found:
            loaded.discard(f"{stem}.{field_name}")
        return None
    return LoKrSpec(
        w1=found.get("lokr_w1"),
        w2=found.get("lokr_w2"),
        w1_a=found.get("lokr_w1_a"),
        w1_b=found.get("lokr_w1_b"),
        w2_a=found.get("lokr_w2_a"),
        w2_b=found.get("lokr_w2_b"),
        t2=found.get("lokr_t2"),
        alpha=alpha,
        dora_scale=dora_scale,
    )


def _probe_glora(
    stem: str,
    normalized: NormalizedLora,
    alpha: str | None,
    dora_scale: str | None,
    loaded: set[str],
    diagnostics: list[str],
) -> GLoRASpec | None:
    nkeys = normalized.keys
    a1_name = f"{stem}.a1.weight"
    if a1_name not in nkeys:
        return None
    companions = (f"{stem}.a2.weight", f"{stem}.b1.weight", f"{stem}.b2.weight")
    missing = [c for c in companions if c not in nkeys]
    if missing:
        diagnostics.append(
            f"glora: {a1_name!r} present but companions missing: {missing}; adapter skipped"
        )
        return None
    for name in (a1_name, *companions):
        loaded.add(name)
    return GLoRASpec(
        a1=nkeys[a1_name],
        a2=nkeys[companions[0]],
        b1=nkeys[companions[1]],
        b2=nkeys[companions[2]],
        alpha=alpha,
        dora_scale=dora_scale,
    )


def _probe_oft_like(
    stem: str,
    normalized: NormalizedLora,
    loaded: set[str],
    ndim: int,
) -> tuple[str, str | None] | None:
    """Shared OFT/BOFT probe: same keys, rank decides. Returns
    (blocks source, rescale source) when the rank matches."""
    nkeys = normalized.keys
    blocks_name = f"{stem}.oft_blocks"
    if blocks_name not in nkeys:
        return None
    geometry = normalized.geometries.get(blocks_name)
    if geometry is None or len(geometry.shape) != ndim:
        return None
    loaded.add(blocks_name)
    rescale = None
    rescale_name = f"{stem}.rescale"
    if rescale_name in nkeys:
        rescale = nkeys[rescale_name]
        loaded.add(rescale_name)
    return nkeys[blocks_name], rescale


def _probe_oft(
    stem: str,
    normalized: NormalizedLora,
    alpha: str | None,
    dora_scale: str | None,
    loaded: set[str],
    diagnostics: list[str],
) -> OFTSpec | None:
    got = _probe_oft_like(stem, normalized, loaded, ndim=3)
    if got is None:
        return None
    blocks, rescale = got
    return OFTSpec(blocks=blocks, rescale=rescale, alpha=alpha, dora_scale=dora_scale)


def _probe_boft(
    stem: str,
    normalized: NormalizedLora,
    alpha: str | None,
    dora_scale: str | None,
    loaded: set[str],
    diagnostics: list[str],
) -> BOFTSpec | None:
    got = _probe_oft_like(stem, normalized, loaded, ndim=4)
    if got is None:
        return None
    blocks, rescale = got
    return BOFTSpec(blocks=blocks, rescale=rescale, alpha=alpha, dora_scale=dora_scale)


# Reference order: comfy/weight_adapter/__init__.py adapters list
# @ b78cec87 (LoRA, LoHa, LoKr, GLoRA, OFT, BOFT).
_ADAPTER_PROBES = (
    _probe_lora,
    _probe_loha,
    _probe_lokr,
    _probe_glora,
    _probe_oft,
    _probe_boft,
)


# --------------------------------------------------------------------------
# Key-map builders: model state-dict keys -> LoRA stem aliases.
# Ported from the pure-string sections of comfy/lora.py
# model_lora_keys_unet / model_lora_keys_clip @ b78cec87. The
# model-specific diffusers alias tables are added by their owning runtimes.


LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def native_unet_key_map(model_keys: Iterable[str]) -> dict[str, str]:
    """The universal ``diffusion_model.*`` aliases every model gets:
    kohya ``lora_unet_*`` plus the generic suffix-less stem
    (comfy/lora.py:188-198)."""
    key_map: dict[str, str] = {}
    for k in model_keys:
        if not k.startswith("diffusion_model."):
            continue
        if k.endswith(".weight"):
            key_lora = k[len("diffusion_model.") : -len(".weight")].replace(".", "_")
            key_map[f"lora_unet_{key_lora}"] = k
            key_map[k[: -len(".weight")]] = k
        else:
            key_map[k] = k
    return key_map


def minimax_h3_lora_key_map(
    model_keys: Iterable[str], _config: object | None = None
) -> dict[str, str]:
    """Map direct MiniMax H3 LoRA stems to loaded diffusion rows."""

    return {
        key.removeprefix("diffusion_model.").removesuffix(".weight"): key
        for key in model_keys
        if key.startswith("diffusion_model.") and key.endswith(".weight")
    }


_UNET_RESNET_TO_DIFFUSERS = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}
_UNET_ATTENTION_KEYS = (
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
)
_UNET_TRANSFORMER_KEYS = (
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
)
_UNET_BASIC_TO_DIFFUSERS = (
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
)


def sd_unet_diffusers_key_map(model_keys: Iterable[str], config: UNetConfig) -> dict[str, str]:
    """Map Diffusers SD-era UNet LoRA stems onto native UNet weights."""
    targets: dict[str, str] = {}
    input_depths = list(config.transformer_depth)
    for level in range(len(config.channel_mult)):
        native_block = 1 + (config.num_res_blocks[level] + 1) * level
        for block in range(config.num_res_blocks[level]):
            for native, diffusers in _UNET_RESNET_TO_DIFFUSERS.items():
                targets[f"down_blocks.{level}.resnets.{block}.{diffusers}"] = (
                    f"input_blocks.{native_block}.0.{native}"
                )
            transformer_depth = input_depths.pop(0)
            if transformer_depth > 0:
                for key in _UNET_ATTENTION_KEYS:
                    targets[f"down_blocks.{level}.attentions.{block}.{key}"] = (
                        f"input_blocks.{native_block}.1.{key}"
                    )
                for transformer in range(transformer_depth):
                    for key in _UNET_TRANSFORMER_KEYS:
                        targets[
                            f"down_blocks.{level}.attentions.{block}."
                            f"transformer_blocks.{transformer}.{key}"
                        ] = f"input_blocks.{native_block}.1.transformer_blocks.{transformer}.{key}"
            native_block += 1
        for suffix in ("weight", "bias"):
            targets[f"down_blocks.{level}.downsamplers.0.conv.{suffix}"] = (
                f"input_blocks.{native_block}.0.op.{suffix}"
            )

    for key in _UNET_ATTENTION_KEYS:
        targets[f"mid_block.attentions.0.{key}"] = f"middle_block.1.{key}"
    for transformer in range(config.transformer_depth_middle):
        for key in _UNET_TRANSFORMER_KEYS:
            targets[f"mid_block.attentions.0.transformer_blocks.{transformer}.{key}"] = (
                f"middle_block.1.transformer_blocks.{transformer}.{key}"
            )
    for block, native_block in enumerate((0, 2)):
        for native, diffusers in _UNET_RESNET_TO_DIFFUSERS.items():
            targets[f"mid_block.resnets.{block}.{diffusers}"] = (
                f"middle_block.{native_block}.{native}"
            )

    output_depths = list(config.transformer_depth_output)
    reversed_res_blocks = tuple(reversed(config.num_res_blocks))
    for level, res_blocks in enumerate(reversed_res_blocks):
        native_block = (res_blocks + 1) * level
        count = res_blocks + 1
        for block in range(count):
            for native, diffusers in _UNET_RESNET_TO_DIFFUSERS.items():
                targets[f"up_blocks.{level}.resnets.{block}.{diffusers}"] = (
                    f"output_blocks.{native_block}.0.{native}"
                )
            native_layer = 1
            transformer_depth = output_depths.pop()
            if transformer_depth > 0:
                native_layer += 1
                for key in _UNET_ATTENTION_KEYS:
                    targets[f"up_blocks.{level}.attentions.{block}.{key}"] = (
                        f"output_blocks.{native_block}.1.{key}"
                    )
                for transformer in range(transformer_depth):
                    for key in _UNET_TRANSFORMER_KEYS:
                        targets[
                            f"up_blocks.{level}.attentions.{block}."
                            f"transformer_blocks.{transformer}.{key}"
                        ] = f"output_blocks.{native_block}.1.transformer_blocks.{transformer}.{key}"
            if block == count - 1:
                for suffix in ("weight", "bias"):
                    targets[f"up_blocks.{level}.upsamplers.0.conv.{suffix}"] = (
                        f"output_blocks.{native_block}.{native_layer}.conv.{suffix}"
                    )
            native_block += 1
    for native, diffusers in _UNET_BASIC_TO_DIFFUSERS:
        targets[diffusers] = native

    available = set(model_keys)
    key_map: dict[str, str] = {}
    for diffusers, native in targets.items():
        target = f"diffusion_model.{native}"
        if not diffusers.endswith(".weight") or target not in available:
            continue
        stem = diffusers[: -len(".weight")]
        underscored = stem.replace(".", "_")
        key_map[f"lora_unet_{underscored}"] = target
        key_map[f"lycoris_{underscored}"] = target
        processor = stem.replace(".to_", ".processor.to_")
        if processor.endswith(".to_out.0"):
            processor = processor[:-2]
        key_map[processor] = target
        key_map[f"unet.{processor}"] = target
    return key_map


def qwen_image_lora_key_map(model_keys: Iterable[str]) -> dict[str, str]:
    """Map Qwen Image native, Transformers, and SimpleTuner LoRA stems."""

    key_map: dict[str, str] = {}
    for key in model_keys:
        if not key.startswith("diffusion_model.") or not key.endswith(".weight"):
            continue
        stem = key[len("diffusion_model.") : -len(".weight")]
        key_map[stem] = key
        key_map[f"transformer.{stem}"] = key
        key_map[f"lycoris_{stem.replace('.', '_')}"] = key
    return key_map


def flux_linear1_qkv_key_map(model_keys: Iterable[str], hidden_size: int) -> dict[str, PatchTarget]:
    """Flux fused-qkv slice targets (comfy/lora.py:276-279): LoRAs
    trained against split qkv apply to the first ``3 * hidden_size``
    rows of the fused ``linear1`` weight. The reference emits a
    zero-length slice when hidden_size is missing from the config;
    that target is unusable, so here it is a ValueError instead."""
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
    key_map: dict[str, PatchTarget] = {}
    for k in model_keys:
        if k.endswith(".weight") and ".linear1." in k:
            key_map[k.replace(".linear1.weight", ".linear1_qkv")] = PatchTarget(
                key=k, offset=PatchOffset(dim=0, start=0, length=hidden_size * 3)
            )
    return key_map


def z_image_diffusers_key_map(
    model_keys: Iterable[str], hidden_size: int
) -> dict[str, PatchTarget]:
    """Map Diffusers Z-Image LoRA stems onto native weights."""
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
    key_map: dict[str, PatchTarget] = {}

    def add_aliases(stem: str, target: PatchTarget) -> None:
        key_map[f"diffusion_model.{stem}"] = target
        key_map[f"transformer.{stem}"] = target
        key_map[f"lycoris_{stem.replace('.', '_')}"] = target
        key_map[stem] = target

    for key in model_keys:
        if not key.startswith("diffusion_model.") or not key.endswith(".weight"):
            continue
        stem = key[len("diffusion_model.") : -len(".weight")]
        if stem.endswith(".attention.qkv"):
            prefix = stem.removesuffix("qkv")
            for index, name in enumerate(("to_q", "to_k", "to_v")):
                add_aliases(
                    f"{prefix}{name}",
                    PatchTarget(
                        key,
                        PatchOffset(dim=0, start=index * hidden_size, length=hidden_size),
                    ),
                )
            continue
        if stem.endswith(".attention.out"):
            stem = f"{stem.removesuffix('out')}to_out.0"
        elif stem.endswith(".attention.q_norm"):
            stem = f"{stem.removesuffix('q_norm')}norm_q"
        elif stem.endswith(".attention.k_norm"):
            stem = f"{stem.removesuffix('k_norm')}norm_k"
        elif stem.startswith("final_layer."):
            stem = f"all_final_layer.2-1.{stem.removeprefix('final_layer.')}"
        elif stem == "x_embedder":
            stem = "all_x_embedder.2-1"
        add_aliases(stem, PatchTarget(key))
    return key_map


def z_image_family_lora_key_map(
    model_keys: Iterable[str], config: object
) -> dict[str, PatchTarget]:
    """Build Z-Image aliases from its registered model configuration."""
    hidden_width = getattr(config, "hidden_width", None)
    if not isinstance(hidden_width, int) or hidden_width <= 0:
        raise ValueError("Z-Image LoRA mapping requires a positive hidden width")
    return z_image_diffusers_key_map(model_keys, hidden_width)


def clip_lora_key_map(model_keys: Iterable[str]) -> dict[str, str]:
    """Text-encoder stem aliases, ported from comfy/lora.py
    model_lora_keys_clip @ b78cec87 (pure key-list logic; the
    reference's shared mutable ``key_map={}`` default is deliberately
    not reproduced)."""
    sdk = set(model_keys)
    key_map: dict[str, str] = {}
    prefix_set: set[str] = set()
    for k in sdk:
        if k.endswith(".weight"):
            key_map[f"text_encoders.{k[: -len('.weight')]}"] = k
            tp = k.find(".transformer.")
            if tp > 0 and not k.startswith("clip_"):
                key_map[f"text_encoders.{k[tp + 1 : -len('.weight')]}"] = k
            prefix_set.add(k.split(".")[0])

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    clip_g_present = False
    for b in range(32):
        for c, mapped in LORA_CLIP_MAP.items():
            k = f"clip_h.transformer.text_model.encoder.layers.{b}.{c}.weight"
            if k in sdk:
                key_map[text_model_lora_key.format(b, mapped)] = k
                key_map[f"lora_te1_text_model_encoder_layers_{b}_{mapped}"] = k
                key_map[f"text_encoder.text_model.encoder.layers.{b}.{c}"] = k

            k = f"clip_l.transformer.text_model.encoder.layers.{b}.{c}.weight"
            if k in sdk:
                key_map[text_model_lora_key.format(b, mapped)] = k
                key_map[f"lora_te1_text_model_encoder_layers_{b}_{mapped}"] = k
                clip_l_present = True
                key_map[f"text_encoder.text_model.encoder.layers.{b}.{c}"] = k

            k = f"clip_g.transformer.text_model.encoder.layers.{b}.{c}.weight"
            if k in sdk:
                clip_g_present = True
                if clip_l_present:
                    key_map[f"lora_te2_text_model_encoder_layers_{b}_{mapped}"] = k
                    key_map[f"text_encoder_2.text_model.encoder.layers.{b}.{c}"] = k
                else:
                    key_map[f"lora_te_text_model_encoder_layers_{b}_{mapped}"] = k
                    key_map[f"text_encoder.text_model.encoder.layers.{b}.{c}"] = k
                    key_map[f"lora_prior_te_text_model_encoder_layers_{b}_{mapped}"] = k

    for k in sdk:
        if k.endswith(".weight"):
            if k.startswith("t5xxl.transformer."):
                l_key = k[len("t5xxl.transformer.") : -len(".weight")]
                t5_index = 1
                if clip_g_present:
                    t5_index += 1
                if clip_l_present:
                    t5_index += 1
                    if t5_index == 2:
                        key_map["lora_te{}_{}".format(t5_index, l_key.replace(".", "_"))] = k
                        t5_index += 1
                key_map["lora_te{}_{}".format(t5_index, l_key.replace(".", "_"))] = k
            elif k.startswith("hydit_clip.transformer.bert."):
                l_key = k[len("hydit_clip.transformer.bert.") : -len(".weight")]
                key_map["lora_te1_{}".format(l_key.replace(".", "_"))] = k

    if len(prefix_set) == 1:
        full_prefix = f"{next(iter(prefix_set))}.transformer.model."
        for k in sdk:
            if k.endswith(".weight") and k.startswith(full_prefix):
                l_key = k[len(full_prefix) : -len(".weight")]
                key_map["lora_te_{}".format(l_key.replace(".", "_"))] = k

    k = "clip_g.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_prior_te_text_projection"] = k
        key_map["lora_te2_text_projection"] = k

    k = "clip_l.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_te1_text_projection"] = k

    return key_map


__all__ = [
    "AdapterSpec",
    "BOFTSpec",
    "DIALECT_BFL_FLUX_CONTROL",
    "DIALECT_NONE",
    "DIALECT_USO",
    "DIALECT_WAN_FUN",
    "DecodedPatch",
    "DiffPatchRef",
    "GLoRASpec",
    "LORA_CLIP_MAP",
    "LoHaSpec",
    "LoKrSpec",
    "LoRASpec",
    "LoraDecodeResult",
    "NormalizedLora",
    "OFTSpec",
    "PatchTarget",
    "SetPatchRef",
    "clip_lora_key_map",
    "decode_lora",
    "flux_linear1_qkv_key_map",
    "minimax_h3_lora_key_map",
    "native_unet_key_map",
    "normalize_lora_keys",
    "qwen_image_lora_key_map",
    "sd_unet_diffusers_key_map",
    "z_image_diffusers_key_map",
]
