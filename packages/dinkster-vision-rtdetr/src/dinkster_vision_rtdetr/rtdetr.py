"""RT-DETR v4 x-HGNet model architecture.

The architecture and forward math mirror ComfyUI commit
c67885b14556cf3e4e061862925282d403d09862. The checkpoint is strictly loaded
from fp16 weights and executes in float32 on CPU.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        ic,
        oc,
        k=3,
        s=1,
        groups=1,
        use_act=True,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            ic,
            oc,
            k,
            s,
            (k - 1) // 2,
            groups=groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.bn = nn.BatchNorm2d(oc, device=device, dtype=dtype)
        self.act = nn.ReLU() if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightConvBNAct(nn.Module):
    def __init__(self, ic, oc, k, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv1 = ConvBNAct(
            ic,
            oc,
            1,
            use_act=False,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.conv2 = ConvBNAct(
            oc,
            oc,
            k,
            groups=oc,
            use_act=True,
            device=device,
            dtype=dtype,
            operations=operations,
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class _StemBlock(nn.Module):
    def __init__(self, ic, mc, oc, device=None, dtype=None, operations=None):
        super().__init__()
        self.stem1 = ConvBNAct(ic, mc, 3, 2, device=device, dtype=dtype, operations=operations)
        self.stem2a = ConvBNAct(
            mc, mc // 2, 2, 1, device=device, dtype=dtype, operations=operations
        )
        self.stem2b = ConvBNAct(
            mc // 2, mc, 2, 1, device=device, dtype=dtype, operations=operations
        )
        self.stem3 = ConvBNAct(mc * 2, mc, 3, 2, device=device, dtype=dtype, operations=operations)
        self.stem4 = ConvBNAct(mc, oc, 1, device=device, dtype=dtype, operations=operations)
        self.pool = nn.MaxPool2d(2, 1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        return self.stem4(self.stem3(torch.cat([x1, x2], 1)))


class _HG_Block(nn.Module):
    def __init__(
        self,
        ic,
        mc,
        oc,
        layer_num,
        k=3,
        residual=False,
        light=False,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.residual = residual
        if light:
            self.layers = nn.ModuleList(
                [
                    LightConvBNAct(
                        ic if i == 0 else mc,
                        mc,
                        k,
                        device=device,
                        dtype=dtype,
                        operations=operations,
                    )
                    for i in range(layer_num)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    ConvBNAct(
                        ic if i == 0 else mc,
                        mc,
                        k,
                        device=device,
                        dtype=dtype,
                        operations=operations,
                    )
                    for i in range(layer_num)
                ]
            )
        total = ic + layer_num * mc
        self.aggregation = nn.Sequential(
            ConvBNAct(
                total,
                oc // 2,
                1,
                device=device,
                dtype=dtype,
                operations=operations,
            ),
            ConvBNAct(
                oc // 2,
                oc,
                1,
                device=device,
                dtype=dtype,
                operations=operations,
            ),
        )

    def forward(self, x):
        identity = x
        outs = [x]
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        x = self.aggregation(torch.cat(outs, 1))
        return x + identity if self.residual else x


class _HG_Stage(nn.Module):
    def __init__(
        self,
        ic,
        mc,
        oc,
        num_blocks,
        downsample=True,
        light=False,
        k=3,
        layer_num=6,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(
                ic,
                ic,
                3,
                2,
                groups=ic,
                use_act=False,
                device=device,
                dtype=dtype,
                operations=operations,
            )
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.Sequential(
            *[
                _HG_Block(
                    ic if i == 0 else oc,
                    mc,
                    oc,
                    layer_num,
                    k=k,
                    residual=i != 0,
                    light=light,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.blocks(self.downsample(x))


class HGNetv2(nn.Module):
    _STAGE_CFGS = (
        (64, 64, 128, 1, False, False, 3, 6),
        (128, 128, 512, 2, True, False, 3, 6),
        (512, 256, 1024, 5, True, True, 5, 6),
        (1024, 512, 2048, 2, True, True, 5, 6),
    )

    def __init__(self, return_idx=(1, 2, 3), device=None, dtype=None, operations=None):
        super().__init__()
        self.stem = _StemBlock(3, 32, 64, device=device, dtype=dtype, operations=operations)
        self.stages = nn.ModuleList(
            [
                _HG_Stage(*cfg, device=device, dtype=dtype, operations=operations)
                for cfg in self._STAGE_CFGS
            ]
        )
        self.return_idx = list(return_idx)
        self.out_channels = [self._STAGE_CFGS[i][2] for i in return_idx]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx:
                outs.append(x)
        return outs


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ic,
        oc,
        k,
        s,
        g=1,
        padding=None,
        act=None,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        p = (k - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ic, oc, k, s, p, groups=g, bias=True, device=device, dtype=dtype)
        self.act = nn.SiLU() if act == "silu" else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class VGGBlock(nn.Module):
    def __init__(self, ic, oc, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, 3, 1, padding=1, bias=True, device=device, dtype=dtype)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x))


class CSPLayer(nn.Module):
    def __init__(
        self,
        ic,
        oc,
        num_blocks=3,
        expansion=1.0,
        act="silu",
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        h = int(oc * expansion)
        self.conv1 = ConvNormLayer(
            ic,
            h,
            1,
            1,
            act=act,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.conv2 = ConvNormLayer(
            ic,
            h,
            1,
            1,
            act=act,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.bottlenecks = nn.Sequential(
            *[
                VGGBlock(h, h, device=device, dtype=dtype, operations=operations)
                for _ in range(num_blocks)
            ]
        )
        self.conv3 = (
            ConvNormLayer(
                h,
                oc,
                1,
                1,
                act=act,
                device=device,
                dtype=dtype,
                operations=operations,
            )
            if h != oc
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class RepNCSPELAN4(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        c4,
        n=3,
        act="silu",
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer(
            c1,
            c3,
            1,
            1,
            act=act,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.cv2 = nn.Sequential(
            CSPLayer(
                c3 // 2,
                c4,
                n,
                1.0,
                act=act,
                device=device,
                dtype=dtype,
                operations=operations,
            ),
            ConvNormLayer(
                c4,
                c4,
                3,
                1,
                act=act,
                device=device,
                dtype=dtype,
                operations=operations,
            ),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(
                c4,
                c4,
                n,
                1.0,
                act=act,
                device=device,
                dtype=dtype,
                operations=operations,
            ),
            ConvNormLayer(
                c4,
                c4,
                3,
                1,
                act=act,
                device=device,
                dtype=dtype,
                operations=operations,
            ),
        )
        self.cv4 = ConvNormLayer(
            c3 + 2 * c4,
            c2,
            1,
            1,
            act=act,
            device=device,
            dtype=dtype,
            operations=operations,
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(module(y[-1]) for module in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, 1))


class SCDown(nn.Module):
    def __init__(self, ic, oc, k, s, device=None, dtype=None, operations=None):
        super().__init__()
        self.cv1 = ConvNormLayer(ic, oc, 1, 1, device=device, dtype=dtype, operations=operations)
        self.cv2 = ConvNormLayer(
            oc,
            oc,
            k,
            s,
            g=oc,
            device=device,
            dtype=dtype,
            operations=operations,
        )

    def forward(self, x):
        return self.cv2(self.cv1(x))


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, tokens, channels = query.shape
    head_width = channels // heads

    def split(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.unsqueeze(3)
            .reshape(batch, -1, heads, head_width)
            .permute(0, 2, 1, 3)
            .reshape(batch * heads, -1, head_width)
            .contiguous()
        )

    query, key, value = split(query), split(key), split(value)
    similarity = torch.einsum("b i d, b j d -> b i j", query, key) * head_width**-0.5
    if mask is not None:
        if mask.dtype == torch.bool:
            expanded = mask.reshape(mask.shape[0], -1)[:, None].repeat_interleave(heads, dim=0)
            similarity.masked_fill_(~expanded, -torch.finfo(similarity.dtype).max)
        else:
            mask_batch = 1 if mask.ndim == 2 else mask.shape[0]
            expanded = mask.reshape(mask_batch, -1, mask.shape[-2], mask.shape[-1])
            expanded = expanded.expand(batch, heads, -1, -1).reshape(
                -1, mask.shape[-2], mask.shape[-1]
            )
            similarity.add_(expanded)
    attention = similarity.softmax(dim=-1)
    output = torch.einsum(
        "b i j, b j d -> b i d",
        attention.to(value.dtype),
        value,
    )
    return (
        output.unsqueeze(0)
        .reshape(batch, heads, tokens, head_width)
        .permute(0, 2, 1, 3)
        .reshape(batch, tokens, channels)
    )


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, query, key, value, attn_mask=None):
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        return self.out_proj(_attention(q, k, v, self.num_heads, attn_mask))


class _TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            d_model, nhead, device=device, dtype=dtype, operations=operations
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, pos_embed=None):
        q = k = src if pos_embed is None else src + pos_embed
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.activation(self.linear1(src)))
        return self.norm2(src + src2)


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_feedforward,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, src_mask=None, pos_embed=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, pos_embed=pos_embed)
        return src


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=2048,
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=(640, 640),
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim] * len(in_channels)
        self.out_strides = list(feat_strides)
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    ch,
                                    hidden_dim,
                                    1,
                                    bias=True,
                                    device=device,
                                    dtype=dtype,
                                ),
                            )
                        ]
                    )
                )
                for ch in in_channels
            ]
        )
        self.encoder = nn.ModuleList(
            [
                _TransformerEncoder(
                    num_encoder_layers,
                    hidden_dim,
                    nhead,
                    dim_feedforward,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(len(use_encoder_idx))
            ]
        )
        blocks = round(3 * depth_mult)
        self.lateral_convs = nn.ModuleList(
            [
                ConvNormLayer(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(len(in_channels) - 1)
            ]
        )
        self.fpn_blocks = nn.ModuleList(
            [
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    blocks,
                    act=act,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(len(in_channels) - 1)
            ]
        )
        self.downsample_convs = nn.ModuleList(
            [
                nn.Sequential(
                    SCDown(
                        hidden_dim,
                        hidden_dim,
                        3,
                        2,
                        device=device,
                        dtype=dtype,
                        operations=operations,
                    )
                )
                for _ in range(len(in_channels) - 1)
            ]
        )
        self.pan_blocks = nn.ModuleList(
            [
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    blocks,
                    act=act,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(len(in_channels) - 1)
            ]
        )
        if eval_spatial_size:
            for index in self.use_encoder_idx:
                stride = self.feat_strides[index]
                position = self._build_pe(
                    eval_spatial_size[1] // stride,
                    eval_spatial_size[0] // stride,
                    hidden_dim,
                    pe_temperature,
                )
                setattr(self, f"pos_embed{index}", position)

    @staticmethod
    def _build_pe(width, height, dim=256, temp=10000.0):
        assert dim % 4 == 0
        grid_width = torch.arange(width, dtype=torch.float32)
        grid_height = torch.arange(height, dtype=torch.float32)
        grid_width, grid_height = torch.meshgrid(grid_width, grid_height, indexing="ij")
        position_dim = dim // 4
        omega = 1.0 / (temp ** (torch.arange(position_dim, dtype=torch.float32) / position_dim))
        width_projection = grid_width.flatten()[:, None] @ omega[None]
        height_projection = grid_height.flatten()[:, None] @ omega[None]
        return torch.cat(
            [
                width_projection.sin(),
                width_projection.cos(),
                height_projection.sin(),
                height_projection.cos(),
            ],
            1,
        )[None]

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        projected = [self.input_proj[i](feature) for i, feature in enumerate(feats)]
        for i, encoder_index in enumerate(self.use_encoder_idx):
            height, width = projected[encoder_index].shape[2:]
            source = projected[encoder_index].flatten(2).permute(0, 2, 1)
            position = getattr(self, f"pos_embed{encoder_index}").to(
                device=source.device, dtype=source.dtype
            )
            encoder = cast("_TransformerEncoder", self.encoder[i])
            for layer in encoder.layers:
                source = layer(source, pos_embed=position)
            projected[encoder_index] = (
                source.permute(0, 2, 1).reshape(-1, self.hidden_dim, height, width).contiguous()
            )

        feature_count = len(self.in_channels)
        inner = [projected[-1]]
        for feature_index in range(feature_count - 1, 0, -1):
            block_index = feature_count - 1 - feature_index
            top = self.lateral_convs[block_index](inner[0])
            inner[0] = top
            upsampled = F.interpolate(top, scale_factor=2.0, mode="nearest")
            inner.insert(
                0,
                self.fpn_blocks[block_index](
                    torch.cat([upsampled, projected[feature_index - 1]], 1)
                ),
            )

        outputs = [inner[0]]
        for index in range(feature_count - 1):
            outputs.append(
                self.pan_blocks[index](
                    torch.cat(
                        [self.downsample_convs[index](outputs[-1]), inner[index + 1]],
                        1,
                    )
                )
            )
        return outputs


def _deformable_attn_v2(
    value: list[torch.Tensor],
    spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int],
) -> torch.Tensor:
    _, channels = value[0].shape[:2]
    _, query_count, head_count, _, _ = sampling_locations.shape
    batch = sampling_locations.shape[0]
    grids = 2 * sampling_locations - 1
    grids = grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    grids_per_level = grids.split(num_points_list, dim=2)
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes):
        level_value = value[level].reshape(batch * head_count, channels, height, width)
        sampled.append(
            F.grid_sample(
                level_value,
                grids_per_level[level],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    attention = attention_weights.permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)
    output = (torch.cat(sampled, -1) * attention).sum(-1)
    output = output.reshape(batch, head_count * channels, query_count)
    return output.permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=3,
        num_points=4,
        offset_scale=0.5,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        points = (
            list(num_points) if isinstance(num_points, (list, tuple)) else [num_points] * num_levels
        )
        self.num_points_list = points
        self.offset_scale = offset_scale
        total = num_heads * sum(points)
        self.register_buffer(
            "num_points_scale",
            torch.tensor(
                [1.0 / count for count in points for _ in range(count)],
                dtype=torch.float32,
            ),
        )
        self.num_points_scale: torch.Tensor
        self.sampling_offsets = nn.Linear(embed_dim, total * 2, device=device, dtype=dtype)
        self.attention_weights = nn.Linear(embed_dim, total, device=device, dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,
        ref_pts: torch.Tensor,
        value: list[torch.Tensor],
        spatial_shapes: list[list[int]],
    ) -> torch.Tensor:
        batch, query_count = query.shape[:2]
        offsets = self.sampling_offsets(query).reshape(
            batch,
            query_count,
            self.num_heads,
            sum(self.num_points_list),
            2,
        )
        attention = F.softmax(
            self.attention_weights(query).reshape(
                batch,
                query_count,
                self.num_heads,
                sum(self.num_points_list),
            ),
            -1,
        )
        scale = self.num_points_scale.to(query).unsqueeze(-1)
        offset = offsets * scale * ref_pts[:, :, None, :, 2:] * self.offset_scale
        locations = ref_pts[:, :, None, :, :2] + offset
        return _deformable_attn_v2(
            value,
            spatial_shapes,
            locations,
            attention,
            self.num_points_list,
        )


class Gate(nn.Module):
    def __init__(self, d_model, device=None, dtype=None, operations=None):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model, device=device, dtype=dtype)
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(self, x1, x2):
        gate1, gate2 = torch.sigmoid(self.gate(torch.cat([x1, x2], -1))).chunk(2, -1)
        return self.norm(gate1 * x1 + gate2 * x2)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], device=device, dtype=dtype) for i in range(num_layers)
        )

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = nn.SiLU()(layer(x)) if index < len(self.layers) - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        num_levels=3,
        num_points=4,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.self_attn = SelfAttention(
            d_model, nhead, device=device, dtype=dtype, operations=operations
        )
        self.norm1 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.cross_attn = MSDeformableAttention(
            d_model,
            nhead,
            num_levels,
            num_points,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.gateway = Gate(d_model, device=device, dtype=dtype, operations=operations)
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        target,
        ref_pts,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos=None,
    ):
        query = key = target if query_pos is None else target + query_pos
        target2 = self.self_attn(query, key, value=target, attn_mask=attn_mask)
        target = self.norm1(target + target2)
        target2 = self.cross_attn(
            target if query_pos is None else target + query_pos,
            ref_pts,
            value,
            spatial_shapes,
        )
        target = self.gateway(target, target2)
        target2 = self.linear2(self.activation(self.linear1(target)))
        return self.norm3((target + target2).clamp(-65504, 65504))


def weighting_function(reg_max, up, reg_scale):
    ub1 = (abs(up[0]) * abs(reg_scale)).item()
    ub2 = ub1 * 2
    step = (ub1 + 1) ** (2 / (reg_max - 2))
    left = [-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step**i - 1 for i in range(1, reg_max // 2)]
    values = [-ub2] + left + [0] + right + [ub2]
    return torch.tensor(values, dtype=up.dtype, device=up.device)


def distance2bbox(points, distance, reg_scale):
    scale = abs(reg_scale).to(dtype=points.dtype)
    x1 = points[..., 0] - (0.5 * scale + distance[..., 0]) * (points[..., 2] / scale)
    y1 = points[..., 1] - (0.5 * scale + distance[..., 1]) * (points[..., 3] / scale)
    x2 = points[..., 0] + (0.5 * scale + distance[..., 2]) * (points[..., 2] / scale)
    y2 = points[..., 1] + (0.5 * scale + distance[..., 3]) * (points[..., 3] / scale)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return torch.stack([center_x, center_y, width, height], -1)


class Integral(nn.Module):
    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), 1)
        x = F.linear(x, project.to(device=x.device, dtype=x.dtype)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(
        self,
        k=4,
        hidden_dim=64,
        num_layers=2,
        reg_max=32,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(
            4 * (k + 1),
            hidden_dim,
            1,
            num_layers,
            device=device,
            dtype=dtype,
            operations=operations,
        )

    def forward(self, scores, pred_corners):
        batch, length, _ = pred_corners.shape
        probability = F.softmax(pred_corners.reshape(batch, length, 4, self.reg_max + 1), -1)
        topk, _ = probability.topk(self.k, -1)
        statistic = torch.cat([topk, topk.mean(-1, keepdim=True)], -1)
        return scores + self.reg_conf(statistic.reshape(batch, length, -1))


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward,
        num_levels,
        num_points,
        num_layers,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up = up
        self.reg_scale = reg_scale
        self.reg_max = reg_max
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    hidden_dim,
                    nhead,
                    dim_feedforward,
                    num_levels,
                    num_points,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(self.eval_idx + 1)
            ]
        )
        self.lqe_layers = nn.ModuleList(
            [
                LQE(
                    4,
                    64,
                    2,
                    reg_max,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(self.eval_idx + 1)
            ]
        )
        self.register_buffer("project", weighting_function(reg_max, up, reg_scale))

    def _value_op(self, memory, spatial_shapes):
        channels = self.hidden_dim // self.nhead
        split = [height * width for height, width in spatial_shapes]
        value = memory.reshape(memory.shape[0], memory.shape[1], self.nhead, channels)
        value = value.permute(0, 2, 3, 1).flatten(0, 1)
        return value.split(split, dim=-1)

    def forward(
        self,
        target,
        ref_pts_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
    ):
        split_values = self._value_op(memory, spatial_shapes)
        values = []
        for level, (height, width) in enumerate(spatial_shapes):
            value = split_values[level]
            values.append(value.reshape(value.shape[0], value.shape[1], height, width))

        ref_pts = F.sigmoid(ref_pts_unact)
        output = target
        output_detach = 0
        pred_corners_undetach = 0
        initial_ref_pts = ref_pts
        decoder_boxes = []
        decoder_logits = []
        for index, layer in enumerate(self.layers):
            ref_input = ref_pts.unsqueeze(2)
            query_position = query_pos_head(ref_pts).clamp(-10, 10)
            output = layer(
                output,
                ref_input,
                values,
                spatial_shapes,
                query_pos=query_position,
            )
            if index == 0:
                ref_unact = ref_pts.clamp(1e-5, 1 - 1e-5)
                ref_unact = torch.log(ref_unact / (1 - ref_unact))
                pre_boxes = F.sigmoid(pre_bbox_head(output) + ref_unact)
                initial_ref_pts = pre_boxes.detach()
            pred_corners = bbox_head[index](output + output_detach) + pred_corners_undetach
            intermediate_box = distance2bbox(
                initial_ref_pts,
                integral(pred_corners, self.project),
                self.reg_scale,
            )
            if index == self.eval_idx:
                scores = score_head[index](output)
                scores = self.lqe_layers[index](scores, pred_corners)
                decoder_boxes.append(intermediate_box)
                decoder_logits.append(scores)
                break
            pred_corners_undetach = pred_corners
            ref_pts = intermediate_box.detach()
            output_detach = output.detach()
        return torch.stack(decoder_boxes), torch.stack(decoder_logits)


class DFINETransformer(nn.Module):
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=(256, 256, 256),
        feat_strides=(8, 16, 32),
        num_levels=3,
        num_points=(3, 6, 3),
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        eval_idx=-1,
        eps=1e-2,
        reg_max=32,
        reg_scale=8.0,
        eval_spatial_size=(640, 640),
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        assert len(feat_strides) == len(feat_channels)
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_levels = num_levels
        self.eps = eps
        self.eval_spatial_size = eval_spatial_size
        self.feat_strides = list(feat_strides)
        for index in range(num_levels - len(feat_strides)):
            self.feat_strides.append(feat_strides[-1] * 2 ** (index + 1))

        self.input_proj = nn.ModuleList()
        for channels in feat_channels:
            if channels == hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        channels,
                                        hidden_dim,
                                        1,
                                        bias=True,
                                        device=device,
                                        dtype=dtype,
                                    ),
                                )
                            ]
                        )
                    )
                )
        input_channels = feat_channels[-1]
        for index in range(num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    input_channels if index == 0 else hidden_dim,
                                    hidden_dim,
                                    3,
                                    2,
                                    1,
                                    bias=True,
                                    device=device,
                                    dtype=dtype,
                                ),
                            )
                        ]
                    )
                )
            )
            input_channels = hidden_dim

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        points = num_points if isinstance(num_points, (list, tuple)) else [num_points] * num_levels
        self.decoder = TransformerDecoder(
            hidden_dim,
            nhead,
            dim_feedforward,
            num_levels,
            points,
            num_layers,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.query_pos_head = MLP(
            4,
            2 * hidden_dim,
            hidden_dim,
            2,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    (
                        "proj",
                        nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype),
                    ),
                    (
                        "norm",
                        nn.LayerNorm(hidden_dim, device=device, dtype=dtype),
                    ),
                ]
            )
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes, device=device, dtype=dtype)
        self.enc_bbox_head = MLP(
            hidden_dim,
            hidden_dim,
            4,
            3,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.eval_idx_ = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [
                nn.Linear(hidden_dim, num_classes, device=device, dtype=dtype)
                for _ in range(self.eval_idx_ + 1)
            ]
        )
        self.pre_bbox_head = MLP(
            hidden_dim,
            hidden_dim,
            4,
            3,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(
                    hidden_dim,
                    hidden_dim,
                    4 * (reg_max + 1),
                    3,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(self.eval_idx_ + 1)
            ]
        )
        self.integral = Integral(reg_max)
        if eval_spatial_size:
            anchors, valid_mask = self._gen_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
            self.anchors: torch.Tensor
            self.valid_mask: torch.Tensor

    def _gen_anchors(
        self,
        spatial_shapes=None,
        grid_size=0.05,
        dtype=torch.float32,
        device="cpu",
    ):
        if spatial_shapes is None:
            height, width = self.eval_spatial_size
            spatial_shapes = [
                [int(height / stride), int(width / stride)] for stride in self.feat_strides
            ]
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij"
            )
            grid_xy = (torch.stack([grid_x, grid_y], -1).float() + 0.5) / (
                torch.tensor([width, height], dtype=dtype)
            )
            size = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.cat([grid_xy, size], -1).reshape(-1, height * width, 4))
        anchors_tensor = torch.cat(anchors, 1).to(device)
        valid_mask = ((anchors_tensor > self.eps) & (anchors_tensor < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors_tensor = torch.log(anchors_tensor / (1 - anchors_tensor))
        anchors_tensor = torch.where(
            valid_mask,
            anchors_tensor,
            torch.full_like(anchors_tensor, float("inf")),
        )
        return anchors_tensor, valid_mask

    def _encoder_input(self, feats: list[torch.Tensor]):
        projected = [self.input_proj[i](feature) for i, feature in enumerate(feats)]
        for index in range(len(feats), self.num_levels):
            projected.append(
                self.input_proj[index](feats[-1] if index == len(feats) else projected[-1])
            )
        flattened = []
        shapes = []
        for feature in projected:
            _, _, height, width = feature.shape
            flattened.append(feature.flatten(2).permute(0, 2, 1))
            shapes.append([height, width])
        return torch.cat(flattened, 1), shapes

    def _decoder_input(self, memory: torch.Tensor):
        anchors = self.anchors.to(memory)
        valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        output_memory = self.enc_output(valid_mask.to(memory) * memory)
        logits = self.enc_score_head(output_memory)
        _, indices = torch.topk(logits.max(-1).values, self.num_queries, dim=-1)
        expanded = indices.unsqueeze(-1)
        top_memory = output_memory.gather(1, expanded.expand(-1, -1, output_memory.shape[-1]))
        top_anchors = anchors.gather(1, expanded.expand(-1, -1, anchors.shape[-1]))
        top_reference = self.enc_bbox_head(top_memory) + top_anchors
        return top_memory.detach(), top_reference.detach()

    def forward(self, feats: list[torch.Tensor]):
        memory, shapes = self._encoder_input(feats)
        content, reference = self._decoder_input(memory)
        boxes, logits = self.decoder(
            content,
            reference,
            memory,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
        )
        return {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}


class RTv4(nn.Module):
    def __init__(
        self,
        num_classes=80,
        num_queries=300,
        enc_h=256,
        dec_h=256,
        enc_ff=2048,
        dec_ff=1024,
        feat_strides=(8, 16, 32),
        device=None,
        dtype=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.operations = operations
        self.backbone = HGNetv2(device=device, dtype=dtype, operations=operations)
        self.encoder = HybridEncoder(
            hidden_dim=enc_h,
            dim_feedforward=enc_ff,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.decoder = DFINETransformer(
            num_classes=num_classes,
            hidden_dim=dec_h,
            num_queries=num_queries,
            feat_channels=tuple(enc_h for _ in feat_strides),
            feat_strides=feat_strides,
            dim_feedforward=dec_ff,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.num_classes = num_classes
        self.num_queries = num_queries

    def _forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(self.backbone(x)))

    def postprocess(self, outputs, orig_size: tuple[int, int] = (640, 640)):
        logits = outputs["pred_logits"]
        centers_x, centers_y, widths, heights = outputs["pred_boxes"].unbind(-1)
        boxes = torch.stack(
            (
                centers_x - 0.5 * widths,
                centers_y - 0.5 * heights,
                centers_x + 0.5 * widths,
                centers_y + 0.5 * heights,
            ),
            dim=-1,
        )
        boxes = boxes * (
            torch.tensor(orig_size, device=boxes.device, dtype=boxes.dtype)
            .repeat(1, 2)
            .unsqueeze(1)
        )
        scores = F.sigmoid(logits)
        scores, indices = torch.topk(scores.flatten(1), self.num_queries, dim=-1)
        labels = indices % self.num_classes
        boxes = boxes.gather(
            1,
            (indices // self.num_classes).unsqueeze(-1).expand(-1, -1, 4),
        )
        return [
            {"labels": label, "boxes": box, "scores": score}
            for label, box, score in zip(labels, boxes, scores, strict=True)
        ]

    def forward(self, x: torch.Tensor, orig_size: tuple[int, int] = (640, 640)):
        return self.postprocess(self._forward(x), orig_size)


__all__ = ["COCO_CLASSES", "RTv4"]
