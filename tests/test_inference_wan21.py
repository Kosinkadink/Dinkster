"""Wan 2.1 torch-free family contract tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError

import pytest
from dinkster_inference.catalog import WAN21
from dinkster_inference.devices import BFLOAT16, FLOAT8_E4M3, FLOAT16, FLOAT32, INT64, DType
from dinkster_inference.identity import build_runtime_identity_from_facts
from dinkster_inference.sampling import Parameterization
from dinkster_inference.wan21 import (
    WAN21_ANIMATE2_14B,
    WAN21_ANIMATE2_SETTINGS_KEY,
    WAN21_CAMERA_1_3B,
    WAN21_CAMERA_14B,
    WAN21_CAUSAL_AR_1_3B,
    WAN21_CODEC,
    WAN21_FLF_I2V_14B,
    WAN21_FLOW_RVS_1_3B,
    WAN21_FLOW_RVS_CODEC,
    WAN21_FUN_CONTROL_1_3B,
    WAN21_FUN_INPAINT_1_3B,
    WAN21_HUMO_17B,
    WAN21_I2V_14B,
    WAN21_LATENT,
    WAN21_SAMPLING,
    WAN21_SCAIL2_14B,
    WAN21_SCAIL_14B,
    WAN21_SIGMAS,
    WAN21_T2V_1_3B,
    WAN21_T2V_14B,
    WAN21_VACE_1_3B,
    WAN21_VACE_14B,
    WAN22_ANIMATE_14B,
    WAN22_BERNINI_14B,
    WAN22_CAMERA_14B,
    WAN22_DANCER_SETTINGS_KEY,
    WAN22_FUN_CONTROL_14B,
    WAN22_I2V_14B,
    WAN22_S2V_14B,
    WAN22_WANDANCER_14B,
    Wan21Animate2Settings,
    Wan21Detector,
    Wan21PoseBlockCacheDevice,
    Wan21PoseBlockCacheSettings,
    Wan21PoseBlockCacheStorage,
    Wan22DancerSettings,
    decode_wan21_animate2_settings,
    decode_wan22_dancer_settings,
    detect_wan21,
    encode_wan21_animate2_settings,
    encode_wan22_dancer_settings,
    wan21_layout,
)
from dinkster_inference.weights import TensorGeometry, WeightEntry


class HeaderSource:
    def __init__(
        self,
        shapes: Mapping[str, tuple[int, ...]],
        *,
        dtype: DType = FLOAT16,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.shapes = dict(shapes)
        self.dtype = dtype
        self._metadata = dict(metadata or {})
        self.entries_read: list[str] = []
        self.metadata_reads = 0

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        self.entries_read.append(key)
        geometry = TensorGeometry(self.shapes[key], self.dtype)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        self.metadata_reads += 1
        return self._metadata


def wan21_shapes(profile: str, prefix: str = "") -> dict[str, tuple[int, ...]]:
    configs = {
        "t2v-1.3b": WAN21_T2V_1_3B,
        "causal-ar-1.3b": WAN21_CAUSAL_AR_1_3B,
        "flow-rvs-1.3b": WAN21_FLOW_RVS_1_3B,
        "t2v-14b": WAN21_T2V_14B,
        "humo-17b": WAN21_HUMO_17B,
        "i2v-14b": WAN21_I2V_14B,
        "scail-14b": WAN21_SCAIL_14B,
        "scail2-14b": WAN21_SCAIL2_14B,
        "animate2-14b-2.1": WAN21_ANIMATE2_14B,
        "animate-14b-2.2": WAN22_ANIMATE_14B,
        "bernini-14b-2.2": WAN22_BERNINI_14B,
        "s2v-14b-2.2": WAN22_S2V_14B,
        "wandancer-14b-2.2": WAN22_WANDANCER_14B,
        "flf-i2v-14b": WAN21_FLF_I2V_14B,
        "fun-control-1.3b": WAN21_FUN_CONTROL_1_3B,
        "fun-inpaint-1.3b": WAN21_FUN_INPAINT_1_3B,
        "i2v-14b-2.2": WAN22_I2V_14B,
        "fun-control-14b-2.2": WAN22_FUN_CONTROL_14B,
        "vace-1.3b": WAN21_VACE_1_3B,
        "vace-14b": WAN21_VACE_14B,
        "camera-1.3b": WAN21_CAMERA_1_3B,
        "camera-14b": WAN21_CAMERA_14B,
        "camera-14b-2.2": WAN22_CAMERA_14B,
    }
    return {prefix + key: shape for key, shape in wan21_layout(configs[profile]).items()}


def test_official_layout_matches_every_native_state_owner() -> None:
    layout = wan21_layout()
    assert len(layout) == 825
    assert layout["patch_embedding.weight"] == (1536, 16, 1, 2, 2)
    assert layout["time_projection.1.weight"] == (9216, 1536)
    assert layout["blocks.29.ffn.2.weight"] == (1536, 8960)
    assert layout["head.head.weight"] == (64, 1536)
    assert wan21_layout(WAN21_T2V_1_3B) == layout
    assert wan21_layout(WAN21_CAUSAL_AR_1_3B) == layout
    assert wan21_layout(WAN21_FLOW_RVS_1_3B) == layout
    assert len(wan21_layout(WAN21_T2V_14B)) == 1095
    humo = wan21_layout(WAN21_HUMO_17B)
    assert len(humo) == 1583
    assert humo["patch_embedding.weight"] == (5120, 36, 1, 2, 2)
    assert humo["audio_proj.audio_proj_glob_1.layer.weight"] == (512, 51200)
    assert humo["audio_proj.audio_proj_glob_3.layer.bias"] == (24576,)
    assert humo["blocks.39.audio_cross_attn_wrapper.audio_cross_attn.k.weight"] == (5120, 1536)
    assert humo["blocks.39.audio_cross_attn_wrapper.norm1_audio.weight"] == (5120,)
    assert wan21_layout(WAN22_BERNINI_14B) == wan21_layout(WAN21_T2V_14B)
    s2v = wan21_layout(WAN22_S2V_14B)
    assert len(s2v) == 1260
    assert s2v["patch_embedding.weight"] == (5120, 16, 1, 2, 2)
    assert s2v["casual_audio_encoder.weights"] == (1, 25, 1, 1)
    assert s2v["casual_audio_encoder.encoder.conv1_local.conv.weight"] == (5120, 1024, 3)
    assert s2v["audio_injector.injector.11.q.weight"] == (5120, 5120)
    assert s2v["frame_packer.proj_4x.weight"] == (5120, 16, 4, 8, 8)
    dancer = wan21_layout(WAN22_WANDANCER_14B)
    assert dancer["patch_embedding_global.weight"] == (5120, 36, 1, 2, 2)
    assert dancer["music_encoder.0.self_attn.in_proj_weight"] == (768, 256)
    assert dancer["music_projection.weight"] == (256, 35)
    assert dancer["music_injector.injector.7.q.weight"] == (5120, 5120)
    i2v = wan21_layout(WAN21_I2V_14B)
    assert len(i2v) == 1303
    assert i2v["patch_embedding.weight"] == (5120, 36, 1, 2, 2)
    assert i2v["img_emb.proj.3.weight"] == (5120, 1280)
    assert i2v["blocks.39.cross_attn.k_img.weight"] == (5120, 5120)
    assert wan21_layout(WAN21_ANIMATE2_14B) == i2v
    scail = wan21_layout(WAN21_SCAIL_14B)
    assert len(scail) == 1305
    assert scail["patch_embedding.weight"] == (5120, 20, 1, 2, 2)
    assert scail["patch_embedding_pose.weight"] == (5120, 20, 1, 2, 2)
    assert "patch_embedding_mask.weight" not in scail
    scail2 = wan21_layout(WAN21_SCAIL2_14B)
    assert len(scail2) == 1307
    assert scail2["patch_embedding_pose.bias"] == (5120,)
    assert scail2["patch_embedding_mask.weight"] == (5120, 28, 1, 2, 2)
    animate = wan21_layout(WAN22_ANIMATE_14B)
    assert len(animate) == 1441
    assert animate["pose_patch_embedding.weight"] == (5120, 16, 1, 2, 2)
    assert animate["motion_encoder.enc.net_app.convs.7.skip.1.weight"] == (
        512,
        512,
        1,
        1,
    )
    assert animate["face_adapter.fuser_blocks.7.linear1_kv.weight"] == (10240, 5120)
    assert animate["face_encoder.padding_tokens"] == (1, 1, 1, 5120)
    flf = wan21_layout(WAN21_FLF_I2V_14B)
    assert len(flf) == 1304
    assert flf["img_emb.emb_pos"] == (1, 514, 1280)
    wan22_i2v_and_fun_inpaint = wan21_layout(WAN22_I2V_14B)
    assert len(wan22_i2v_and_fun_inpaint) == 1095
    assert wan22_i2v_and_fun_inpaint["patch_embedding.weight"] == (5120, 36, 1, 2, 2)
    assert "img_emb.proj.3.weight" not in wan22_i2v_and_fun_inpaint
    assert "ref_conv.weight" not in wan22_i2v_and_fun_inpaint
    fun_control = wan21_layout(WAN21_FUN_CONTROL_1_3B)
    assert fun_control["patch_embedding.weight"] == (1536, 48, 1, 2, 2)
    assert fun_control["img_emb.proj.3.weight"] == (1536, 1280)
    fun_inpaint = wan21_layout(WAN21_FUN_INPAINT_1_3B)
    assert fun_inpaint["patch_embedding.weight"] == (1536, 36, 1, 2, 2)
    assert "ref_conv.weight" not in fun_inpaint
    fun_control_14b = wan21_layout(WAN22_FUN_CONTROL_14B)
    assert fun_control_14b["patch_embedding.weight"] == (5120, 52, 1, 2, 2)
    assert fun_control_14b["ref_conv.weight"] == (5120, 16, 2, 2)
    vace_1_3b = wan21_layout(WAN21_VACE_1_3B)
    assert len(vace_1_3b) == 1264
    assert vace_1_3b["vace_patch_embedding.weight"] == (1536, 96, 1, 2, 2)
    assert vace_1_3b["vace_blocks.0.before_proj.weight"] == (1536, 1536)
    assert "vace_blocks.1.before_proj.weight" not in vace_1_3b
    assert len(wan21_layout(WAN21_VACE_14B)) == 1331
    camera_1_3b = wan21_layout(WAN21_CAMERA_1_3B)
    assert len(camera_1_3b) == 989
    assert camera_1_3b["patch_embedding.weight"] == (1536, 32, 1, 2, 2)
    assert camera_1_3b["control_adapter.conv.weight"] == (1536, 1536, 2, 2)
    assert camera_1_3b["control_adapter.residual_blocks.0.conv2.weight"] == (
        1536,
        1536,
        3,
        3,
    )
    camera_14b = wan21_layout(WAN21_CAMERA_14B)
    assert len(camera_14b) == 1309
    assert camera_14b["control_adapter.conv.weight"] == (5120, 1536, 2, 2)
    camera_14b_2_2 = wan21_layout(WAN22_CAMERA_14B)
    assert len(camera_14b_2_2) == 1101
    assert "img_emb.proj.0.bias" not in camera_14b_2_2
    assert "vace_layers" not in repr(WAN21_T2V_1_3B)


@pytest.mark.parametrize(
    "profile",
    (
        "t2v-1.3b",
        "t2v-14b",
        "humo-17b",
        "s2v-14b-2.2",
        "i2v-14b",
        "scail-14b",
        "scail2-14b",
        "animate-14b-2.2",
        "flf-i2v-14b",
        "fun-control-1.3b",
        "fun-inpaint-1.3b",
        "i2v-14b-2.2",
        "fun-control-14b-2.2",
        "vace-1.3b",
        "vace-14b",
        "camera-1.3b",
        "camera-14b",
        "camera-14b-2.2",
    ),
)
@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_official_profile_produces_deterministic_immutable_evidence(
    profile: str, prefix: str
) -> None:
    source = HeaderSource(wan21_shapes(profile, prefix))
    first = detect_wan21(source)
    second = Wan21Detector().detect(source)

    assert first == second
    assert first is not None
    assert first.family_id == "dinkster.wan21"
    assert first.fields["profile"] == profile
    assert first.fields["key_prefix"] == prefix
    assert first.fields["patch"] == "1x2x2"
    assert first.fields["output_channels"] == 16
    assert first.fields["attention_head_dim"] == 128
    assert first.fields["flf"] is (profile == "flf-i2v-14b")
    expected_variant = {
        "animate-14b-2.2": "animate",
        "s2v-14b-2.2": "s2v",
        "humo-17b": "humo",
        "scail-14b": "scail",
        "scail2-14b": "scail2",
    }.get(profile, "base")
    assert first.fields["model_variant"] == expected_variant
    assert first.fields.get("vace", False) is profile.startswith("vace-")
    assert first.fields.get("camera", False) is profile.startswith("camera-")
    if profile == "flf-i2v-14b":
        assert first.fields["flf_pos_embed_token_number"] == 514
    if profile.startswith("vace-"):
        expected_layers = 15 if profile == "vace-1.3b" else 8
        assert first.fields["vace_layers"] == expected_layers
        assert first.fields["vace_mapping_step"] == (2 if profile == "vace-1.3b" else 5)
    if profile.startswith("camera-"):
        assert first.fields["camera_channels"] == 24
    full_ref = profile == "fun-control-14b-2.2"
    assert first.fields["ref_conv"] is full_ref
    assert first.fields["full_ref"] is full_ref
    if full_ref:
        assert first.fields["reference_channels"] == 16
    assert source.metadata_reads == 2
    with pytest.raises(TypeError):
        first.fields["profile"] = "lookalike"  # type: ignore[index]


@pytest.mark.parametrize(
    "model_type",
    ("wanvideo_wantodance_local", "wanvideo_wantodance_global"),
)
def test_wandancer_official_metadata_selects_only_the_exact_profile(model_type: str) -> None:
    shapes = wan21_shapes("wandancer-14b-2.2")
    evidence = detect_wan21(HeaderSource(shapes, metadata={"model_type": model_type}))

    assert evidence is not None
    assert evidence.fields["profile"] == "wandancer-14b-2.2"
    assert evidence.fields["model_variant"] == "wandancer"
    assert (
        detect_wan21(HeaderSource(shapes, metadata={"model_type": model_type, "config": "{}"}))
        is None
    )
    assert (
        detect_wan21(HeaderSource(wan21_shapes("i2v-14b"), metadata={"model_type": model_type}))
        is None
    )


@pytest.mark.parametrize(
    (
        "profile",
        "model_type",
        "parameter_count",
        "input_channels",
        "width",
        "ffn",
        "heads",
        "layers",
    ),
    (
        ("t2v-1.3b", "t2v", "1.3B", 16, 1536, 8960, 12, 30),
        ("t2v-14b", "t2v", "14B", 16, 5120, 13824, 40, 40),
        ("humo-17b", "t2v", "17B", 36, 5120, 13824, 40, 40),
        ("s2v-14b-2.2", "t2v", "14B", 16, 5120, 13824, 40, 40),
        ("i2v-14b", "i2v", "14B", 36, 5120, 13824, 40, 40),
        ("scail-14b", "i2v", "14B", 20, 5120, 13824, 40, 40),
        ("scail2-14b", "i2v", "14B", 20, 5120, 13824, 40, 40),
        ("animate-14b-2.2", "i2v", "14B", 36, 5120, 13824, 40, 40),
        ("flf-i2v-14b", "i2v", "14B", 36, 5120, 13824, 40, 40),
        ("fun-control-1.3b", "i2v", "1.3B", 48, 1536, 8960, 12, 30),
        ("fun-inpaint-1.3b", "i2v", "1.3B", 36, 1536, 8960, 12, 30),
        ("i2v-14b-2.2", "t2v", "14B", 36, 5120, 13824, 40, 40),
        ("fun-control-14b-2.2", "t2v", "14B", 52, 5120, 13824, 40, 40),
        ("vace-1.3b", "t2v", "1.3B", 16, 1536, 8960, 12, 30),
        ("vace-14b", "t2v", "14B", 16, 5120, 13824, 40, 40),
        ("camera-1.3b", "i2v", "1.3B", 32, 1536, 8960, 12, 30),
        ("camera-14b", "i2v", "14B", 32, 5120, 13824, 40, 40),
        ("camera-14b-2.2", "t2v", "14B", 36, 5120, 13824, 40, 40),
    ),
)
def test_profile_evidence_carries_exact_architecture_facts(
    profile: str,
    model_type: str,
    parameter_count: str,
    input_channels: int,
    width: int,
    ffn: int,
    heads: int,
    layers: int,
) -> None:
    evidence = detect_wan21(HeaderSource(wan21_shapes(profile)))
    assert evidence is not None
    expected: dict[str, object] = {
        "attention_head_dim": 128,
        "attention_heads": heads,
        "flf": profile == "flf-i2v-14b",
        "full_ref": profile == "fun-control-14b-2.2",
        "ffn_width": ffn,
        "hidden_width": width,
        "input_channels": input_channels,
        "key_prefix": "",
        "layers": layers,
        "model_type": model_type,
        "model_variant": {
            "animate-14b-2.2": "animate",
            "s2v-14b-2.2": "s2v",
            "humo-17b": "humo",
            "scail-14b": "scail",
            "scail2-14b": "scail2",
        }.get(profile, "base"),
        "output_channels": 16,
        "parameter_count": parameter_count,
        "patch": "1x2x2",
        "profile": profile,
        "ref_conv": profile == "fun-control-14b-2.2",
    }
    if profile == "flf-i2v-14b":
        expected["flf_pos_embed_token_number"] = 514
    if profile.startswith("vace-"):
        expected["vace"] = True
        expected["vace_layers"] = 15 if profile == "vace-1.3b" else 8
        expected["vace_mapping_step"] = 2 if profile == "vace-1.3b" else 5
    if profile.startswith("camera-"):
        expected["camera"] = True
        expected["camera_channels"] = 24
    if profile == "fun-control-14b-2.2":
        expected["reference_channels"] = 16
    assert evidence.fields == expected


@pytest.mark.parametrize(
    "marker",
    (
        "ref_conv.weight",
        "full_ref.weight",
        "vace_patch_embedding.weight",
        "control_adapter.conv.weight",
        "casual_audio_encoder.encoder.final_linear.weight",
        "face_adapter.fuser_blocks.0.k_norm.weight",
        "patch_embedding_global.weight",
    ),
)
def test_full_reference_and_wan_variant_markers_fail_closed(marker: str) -> None:
    shapes = wan21_shapes("i2v-14b")
    shapes[marker] = (1,)
    assert detect_wan21(HeaderSource(shapes)) is None


def test_animate_requires_every_auxiliary_tensor_and_rejects_malformed_geometry() -> None:
    shapes = wan21_shapes("animate-14b-2.2")
    assert len(shapes) - len(wan21_shapes("i2v-14b")) == 138

    del shapes["face_adapter.fuser_blocks.0.k_norm.weight"]
    assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes("animate-14b-2.2")
    del shapes["motion_encoder.enc.net_app.convs.3.conv2.0.kernel"]
    assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes("animate-14b-2.2")
    shapes["face_encoder.padding_tokens"] = (1, 1, 5120)
    assert detect_wan21(HeaderSource(shapes)) is None


@pytest.mark.parametrize("profile", ("scail-14b", "scail2-14b"))
def test_scail_requires_exact_variant_projection_geometry(profile: str) -> None:
    shapes = wan21_shapes(profile)
    shapes["patch_embedding_pose.weight"] = (5120, 21, 1, 2, 2)
    assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes(profile)
    del shapes["patch_embedding_pose.bias"]
    assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes(profile)
    if profile == "scail2-14b":
        shapes["patch_embedding_mask.weight"] = (5120, 27, 1, 2, 2)
    else:
        shapes["patch_embedding_mask.weight"] = (5120, 27, 1, 2, 2)
        shapes["patch_embedding_mask.bias"] = (5120,)
    assert detect_wan21(HeaderSource(shapes)) is None


def test_scail_configuration_refuses_non_authoritative_geometry() -> None:
    with pytest.raises(ValueError, match="20/16 I2V geometry"):
        type(WAN21_SCAIL_14B)(model_type="i2v", model_variant="scail", in_channels=36)
    with pytest.raises(ValueError, match="requires a SCAIL model variant"):
        type(WAN21_SCAIL_14B)(model_type="i2v", in_channels=20)


def test_animate_configuration_refuses_non_authoritative_geometry() -> None:
    with pytest.raises(ValueError, match="36/16 I2V geometry"):
        type(WAN22_ANIMATE_14B)(model_variant="animate")


def test_bernini_configuration_refuses_non_authoritative_geometry() -> None:
    with pytest.raises(ValueError, match="14B 16-channel T2V geometry"):
        type(WAN22_BERNINI_14B)(model_variant="bernini")


def test_humo_configuration_refuses_non_authoritative_geometry() -> None:
    with pytest.raises(ValueError, match="17B 36/16 T2V geometry"):
        type(WAN21_HUMO_17B)(model_variant="humo")


def test_humo_requires_every_audio_tensor_and_exact_geometry() -> None:
    shapes = wan21_shapes("humo-17b")
    del shapes["blocks.39.audio_cross_attn_wrapper.norm1_audio.bias"]
    assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes("humo-17b")
    shapes["audio_proj.audio_proj_glob_1.layer.weight"] = (512, 51199)
    assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes("t2v-14b")
    shapes["blocks.0.audio_cross_attn_wrapper.audio_cross_attn.q.weight"] = (5120, 5120)
    assert detect_wan21(HeaderSource(shapes)) is None


@pytest.mark.parametrize("shape", ((1,), (1, 513, 1280), (1, 514, 1279)))
def test_malformed_flf_position_geometry_fails_closed(shape: tuple[int, ...]) -> None:
    shapes = wan21_shapes("i2v-14b")
    shapes["img_emb.emb_pos"] = shape
    assert detect_wan21(HeaderSource(shapes)) is None


@pytest.mark.parametrize("profile", ("t2v-1.3b", "i2v-14b-2.2", "vace-1.3b"))
def test_non_i2v_profiles_reject_flf_position_marker(profile: str) -> None:
    shapes = wan21_shapes(profile)
    shapes["img_emb.emb_pos"] = (1, 514, 1280)
    assert detect_wan21(HeaderSource(shapes)) is None


@pytest.mark.parametrize(
    ("profile", "key", "shape"),
    (
        ("t2v-1.3b", "head.modulation", (1, 2, 5120)),
        ("t2v-1.3b", "head.head.weight", (192, 1536)),
        ("t2v-1.3b", "patch_embedding.weight", (1536, 16, 2, 2)),
        ("t2v-14b", "blocks.0.ffn.0.weight", (8960, 5120)),
        ("t2v-14b", "text_embedding.0.weight", (5120, 3584)),
        ("i2v-14b", "patch_embedding.weight", (5120, 48, 1, 2, 2)),
        ("i2v-14b", "img_emb.proj.3.weight", (5120, 1024)),
    ),
)
def test_cross_profile_and_near_family_geometries_fail_closed(
    profile: str, key: str, shape: tuple[int, ...]
) -> None:
    shapes = wan21_shapes(profile)
    shapes[key] = shape
    assert detect_wan21(HeaderSource(shapes)) is None


def test_missing_extra_or_malformed_block_names_fail_closed() -> None:
    for bad_key in (
        "blocks.39.self_attn.norm_q.weight",
        "blocks.30.self_attn.norm_q.weight",
        "blocks.01.self_attn.norm_q.weight",
        "blocks.foo.self_attn.norm_q.weight",
    ):
        shapes = wan21_shapes("t2v-1.3b")
        shapes[bad_key] = (1536,)
        assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes("t2v-1.3b")
    del shapes["blocks.17.self_attn.norm_q.weight"]
    assert detect_wan21(HeaderSource(shapes)) is None


@pytest.mark.parametrize(
    ("key", "shape"),
    (
        ("vace_patch_embedding.weight", (1536, 95, 1, 2, 2)),
        ("vace_blocks.0.after_proj.weight", (1535, 1536)),
        ("vace_blocks.14.ffn.0.weight", (8959, 1536)),
    ),
)
def test_malformed_vace_geometry_fails_closed(key: str, shape: tuple[int, ...]) -> None:
    shapes = wan21_shapes("vace-1.3b")
    shapes[key] = shape
    assert detect_wan21(HeaderSource(shapes)) is None


@pytest.mark.parametrize(
    ("profile", "key", "shape"),
    (
        ("camera-1.3b", "control_adapter.conv.weight", (1536, 1535, 2, 2)),
        ("camera-14b", "control_adapter.conv.weight", (5120, 1536, 1, 2)),
        (
            "camera-14b-2.2",
            "control_adapter.residual_blocks.0.conv1.weight",
            (5120, 5119, 3, 3),
        ),
    ),
)
def test_malformed_camera_geometry_fails_closed(
    profile: str, key: str, shape: tuple[int, ...]
) -> None:
    shapes = wan21_shapes(profile)
    shapes[key] = shape
    assert detect_wan21(HeaderSource(shapes)) is None


def test_missing_extra_or_misplaced_vace_blocks_fail_closed() -> None:
    shapes = wan21_shapes("vace-1.3b")
    del shapes["vace_blocks.14.after_proj.weight"]
    assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes("vace-1.3b")
    shapes["vace_blocks.15.after_proj.weight"] = (1536, 1536)
    assert detect_wan21(HeaderSource(shapes)) is None

    shapes = wan21_shapes("vace-1.3b")
    shapes["vace_blocks.1.before_proj.weight"] = (1536, 1536)
    assert detect_wan21(HeaderSource(shapes)) is None


def test_detection_is_storage_dtype_independent_and_refuses_inconsistent_sources() -> None:
    assert detect_wan21(HeaderSource(wan21_shapes("t2v-1.3b"), dtype=BFLOAT16)) is not None
    assert detect_wan21(HeaderSource(wan21_shapes("t2v-1.3b"), dtype=FLOAT8_E4M3)) is not None

    class InconsistentSource(HeaderSource):
        def entry(self, key: str) -> WeightEntry:
            raise KeyError(key)

    assert detect_wan21(InconsistentSource(wan21_shapes("t2v-1.3b"))) is None


@pytest.mark.parametrize(
    "profile",
    (
        "t2v-1.3b",
        "t2v-14b",
        "humo-17b",
        "i2v-14b",
        "i2v-14b-2.2",
        "vace-1.3b",
        "vace-14b",
    ),
)
def test_detection_refuses_non_floating_diffusion_storage(profile: str) -> None:
    assert detect_wan21(HeaderSource(wan21_shapes(profile), dtype=INT64)) is None


@pytest.mark.parametrize(
    "config",
    (
        '{"transformer":{"shift":5.0}}',
        "not-json",
        "[]",
    ),
)
def test_behavior_bearing_or_malformed_transformer_metadata_fails_closed(config: str) -> None:
    source = HeaderSource(wan21_shapes("t2v-1.3b"), metadata={"config": config})
    assert detect_wan21(source) is None


def test_unrelated_config_metadata_does_not_change_header_admission() -> None:
    source = HeaderSource(
        wan21_shapes("t2v-1.3b"),
        metadata={"config": '{"text_encoder":{"dtype":"bfloat16"}}'},
    )
    assert detect_wan21(source) is not None


def test_causal_ar_requires_exact_metadata_and_excludes_the_ordinary_profile() -> None:
    shapes = wan21_shapes("causal-ar-1.3b")
    ordinary = detect_wan21(HeaderSource(shapes))
    assert ordinary is not None
    assert ordinary.fields["profile"] == "t2v-1.3b"

    source = HeaderSource(
        shapes,
        metadata={"config": '{"transformer":{"causal_ar":true}}'},
    )
    evidence = detect_wan21(source)

    assert evidence is not None
    assert evidence.fields["profile"] == "causal-ar-1.3b"
    assert evidence.fields["model_type"] == "t2v"
    assert evidence.fields["model_variant"] == "causal_ar"

    for config in (
        '{"transformer":{"causal_ar":1}}',
        '{"transformer":{"causal_ar":true,"shift":5.0}}',
        '{"transformer":{"causal_ar":true},"source":"other"}',
    ):
        assert detect_wan21(HeaderSource(shapes, metadata={"config": config})) is None

    wrong_geometry = HeaderSource(
        wan21_shapes("t2v-14b"),
        metadata={"config": '{"transformer":{"causal_ar":true}}'},
    )
    assert detect_wan21(wrong_geometry) is None


def test_causal_ar_configuration_refuses_non_authoritative_geometry() -> None:
    with pytest.raises(ValueError, match="base 1.3B 16-channel T2V geometry"):
        type(WAN21_CAUSAL_AR_1_3B)(
            model_variant="causal_ar",
            hidden_size=5120,
            ffn_hidden_size=13824,
            num_heads=40,
            num_layers=40,
        )


def test_flow_rvs_requires_exact_metadata_and_excludes_the_ordinary_profile() -> None:
    shapes = wan21_shapes("flow-rvs-1.3b")
    ordinary = detect_wan21(HeaderSource(shapes))
    assert ordinary is not None
    assert ordinary.fields["profile"] == "t2v-1.3b"

    source = HeaderSource(
        shapes,
        metadata={"config": '{"transformer":{"model_type":"flow_rvs"}}'},
    )
    evidence = detect_wan21(source)

    assert evidence is not None
    assert evidence.fields["profile"] == "flow-rvs-1.3b"
    assert evidence.fields["model_type"] == "t2v"
    assert evidence.fields["model_variant"] == "flow_rvs"

    extra_transformer_field = HeaderSource(
        shapes,
        metadata={"config": '{"transformer":{"model_type":"flow_rvs","shift":8.0}}'},
    )
    assert detect_wan21(extra_transformer_field) is None

    extra_top_level_field = HeaderSource(
        shapes,
        metadata={"config": '{"transformer":{"model_type":"flow_rvs"},"source":"other"}'},
    )
    assert detect_wan21(extra_top_level_field) is None

    wrong_geometry = HeaderSource(
        wan21_shapes("t2v-14b"),
        metadata={"config": '{"transformer":{"model_type":"flow_rvs"}}'},
    )
    assert detect_wan21(wrong_geometry) is None


@pytest.mark.parametrize("model_type", ("bernini_high", "bernini_low"))
def test_bernini_requires_exact_metadata_and_excludes_the_ordinary_profile(
    model_type: str,
) -> None:
    shapes = wan21_shapes("bernini-14b-2.2")
    ordinary = detect_wan21(HeaderSource(shapes))
    assert ordinary is not None
    assert ordinary.fields["profile"] == "t2v-14b"

    evidence = detect_wan21(HeaderSource(shapes, metadata={"model_type": model_type}))

    assert evidence is not None
    assert evidence.fields["profile"] == "bernini-14b-2.2"
    assert evidence.fields["model_type"] == "t2v"
    assert evidence.fields["model_variant"] == "bernini"

    wrong_geometry = HeaderSource(
        wan21_shapes("t2v-1.3b"),
        metadata={"model_type": model_type},
    )
    assert detect_wan21(wrong_geometry) is None


def test_bernini_refuses_unknown_or_conflicting_metadata() -> None:
    shapes = wan21_shapes("bernini-14b-2.2")
    unknown = HeaderSource(shapes, metadata={"model_type": "bernini"})
    assert detect_wan21(unknown) is None

    conflicting = HeaderSource(
        shapes,
        metadata={
            "model_type": "bernini_high",
            "config": '{"transformer":{"model_type":"flow_rvs"}}',
        },
    )
    assert detect_wan21(conflicting) is None


def test_animate2_requires_exact_transformer_metadata() -> None:
    shapes = wan21_shapes("animate2-14b-2.1")
    ordinary = detect_wan21(HeaderSource(shapes))
    assert ordinary is not None
    assert ordinary.fields["profile"] == "i2v-14b"

    source = HeaderSource(
        shapes,
        metadata={"config": '{"transformer":{"model_type":"animate2"}}'},
    )
    evidence = detect_wan21(source)

    assert evidence is not None
    assert evidence.fields["profile"] == "animate2-14b-2.1"
    assert evidence.fields["model_type"] == "i2v"
    assert evidence.fields["model_variant"] == "animate2"

    redundant_geometry = detect_wan21(
        HeaderSource(
            shapes,
            metadata={"config": '{"transformer":{"image_model":"wan2.1","model_type":"animate2"}}'},
        )
    )
    assert redundant_geometry is not None
    assert redundant_geometry == evidence


def test_animate2_settings_metadata_is_exact_and_round_trips() -> None:
    settings = Wan21Animate2Settings(0.625, 0.75)
    encoded = encode_wan21_animate2_settings(settings)

    assert WAN21_ANIMATE2_SETTINGS_KEY == "dinkster.wan21/animate2"
    assert dict(encoded) == {"pose_strength": 0.625, "reference_strength": 0.75}
    assert decode_wan21_animate2_settings(encoded) == settings
    with pytest.raises(ValueError, match="missing keys"):
        decode_wan21_animate2_settings({"pose_strength": 1.0})
    with pytest.raises(ValueError, match="unknown keys"):
        decode_wan21_animate2_settings(
            {"pose_strength": 1.0, "reference_strength": 1.0, "other": 1.0}
        )
    with pytest.raises(TypeError, match="must be floats"):
        decode_wan21_animate2_settings({"pose_strength": 1, "reference_strength": 1.0})


def test_wandancer_settings_metadata_is_exact_and_round_trips() -> None:
    settings = Wan22DancerSettings(24.0, 0.75)
    encoded = encode_wan22_dancer_settings(settings)

    assert WAN22_DANCER_SETTINGS_KEY == "dinkster.wan22/dancer"
    assert dict(encoded) == {"fps": 24.0, "audio_inject_scale": 0.75}
    assert decode_wan22_dancer_settings(encoded) == settings
    with pytest.raises(ValueError, match="requires exactly"):
        decode_wan22_dancer_settings({"fps": 24.0})
    with pytest.raises(TypeError, match="must be floats"):
        decode_wan22_dancer_settings({"fps": 24, "audio_inject_scale": 0.75})
    for fps in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            Wan22DancerSettings(fps, 1.0)
    for scale in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite non-negative"):
            Wan22DancerSettings(24.0, scale)


def test_animate2_settings_and_cache_policy_fail_closed() -> None:
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            Wan21Animate2Settings(value, 1.0)
    with pytest.raises(TypeError, match="must be a float"):
        Wan21Animate2Settings(1, 1.0)  # type: ignore[arg-type]

    assert Wan21PoseBlockCacheSettings() == Wan21PoseBlockCacheSettings(
        Wan21PoseBlockCacheDevice.CPU,
        None,
        Wan21PoseBlockCacheStorage.DEFAULT,
    )
    assert Wan21PoseBlockCacheSettings().runtime_facts == ()
    assert Wan21PoseBlockCacheSettings(storage=Wan21PoseBlockCacheStorage.INT8).runtime_facts == (
        "animate2_cache_storage=int8",
    )
    assert Wan21PoseBlockCacheSettings(storage=Wan21PoseBlockCacheStorage.INT4).runtime_facts == (
        "animate2_cache_storage=int4",
    )
    with pytest.raises(ValueError, match="positive integer"):
        Wan21PoseBlockCacheSettings(memory_limit_bytes=0)
    with pytest.raises(TypeError, match="cache device"):
        Wan21PoseBlockCacheSettings(device="cpu")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cache storage"):
        Wan21PoseBlockCacheSettings(storage="int8")  # type: ignore[arg-type]


def test_animate2_quantized_cache_storage_rotates_runtime_identity() -> None:
    def identity(settings: Wan21PoseBlockCacheSettings | None) -> str:
        return build_runtime_identity_from_facts(
            "dinkster.wan21",
            ("family=dinkster.wan21",),
            diffusion_dtype="bfloat16",
            text_dtype="float32",
            vae_dtype="float32",
            fp8_matmul=False,
            runtime_facts=() if settings is None else settings.runtime_facts,
        )

    absent = identity(None)
    disabled = identity(Wan21PoseBlockCacheSettings())
    int8 = identity(Wan21PoseBlockCacheSettings(storage=Wan21PoseBlockCacheStorage.INT8))
    int4 = identity(Wan21PoseBlockCacheSettings(storage=Wan21PoseBlockCacheStorage.INT4))

    assert absent == disabled
    assert len({absent, int8, int4}) == 3


def test_duplicate_prefixed_and_unprefixed_models_are_ambiguous() -> None:
    shapes = wan21_shapes("t2v-1.3b")
    shapes.update(wan21_shapes("t2v-1.3b", "model.diffusion_model."))
    assert detect_wan21(HeaderSource(shapes)) is None


def test_latent_flow_and_codec_facts_match_upstream() -> None:
    latent = WAN21_LATENT
    assert (
        latent.channels,
        latent.dimensions,
        latent.temporal_causal,
        latent.temporal_downscale,
        latent.spatial_downscale,
    ) == (16, 3, True, 4, 8)
    assert latent.taesd_decoder == "lighttaew2_1"

    assert WAN21_SIGMAS.shift == 8.0
    assert WAN21_SIGMAS.multiplier == 1000.0
    assert WAN21_SIGMAS.timesteps == 1000
    assert WAN21_SAMPLING.parameterization is Parameterization.FLOW
    assert WAN21_SAMPLING.sigma_min == WAN21_SIGMAS.sigma_min
    assert WAN21_SAMPLING.sigma_max == 1.0
    assert WAN21_SAMPLING.shift == 8.0
    assert WAN21.memory_factor == 1536 / 2222

    codec = WAN21_CODEC
    assert codec.kind == "video"
    assert codec.latent is latent
    assert codec.content_channels == 3
    assert codec.supported_dtypes == frozenset({FLOAT16, BFLOAT16, FLOAT32})
    assert codec.supports_tiling is True
    assert codec.tiling is not None
    assert codec.tiling.decode_tile == (999, 32, 32)
    assert codec.tiling.decode_overlap == (1, 8, 8)
    assert codec.tiling.encode_tile == (9999, 512, 512)
    assert codec.tiling.encode_overlap == (1, 64, 64)
    mask_codec = WAN21_FLOW_RVS_CODEC
    assert mask_codec.kind == "video"
    assert mask_codec.latent is latent
    assert mask_codec.content_channels == 1
    assert mask_codec.supported_dtypes == codec.supported_dtypes
    assert mask_codec.tiling == codec.tiling
    with pytest.raises(FrozenInstanceError):
        codec.content_channels = 4  # type: ignore[misc]
