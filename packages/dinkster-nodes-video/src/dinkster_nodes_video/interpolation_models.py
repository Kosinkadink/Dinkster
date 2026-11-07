"""Lazy frame interpolation composites over the pinned inference operations."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, cast


def rife_model(head_ch: int, channels: Sequence[int]) -> Any:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    functional = importlib.import_module("torch.nn.functional")
    ops = importlib.import_module("dinkster_inference.ops").disable_weight_init
    module_base = cast(type[object], nn.Module)

    class Head(module_base):
        def __init__(self) -> None:
            super().__init__()
            self.cnn0 = ops.Conv2d(3, 16, 3, 2, 1)
            self.cnn1 = ops.Conv2d(16, 16, 3, 1, 1)
            self.cnn2 = ops.Conv2d(16, 16, 3, 1, 1)
            self.cnn3 = ops.ConvTranspose2d(16, head_ch, 4, 2, 1)
            self.relu = nn.LeakyReLU(0.2, True)

        def forward(self, x: Any) -> Any:
            x = self.relu(self.cnn0(x))
            x = self.relu(self.cnn1(x))
            x = self.relu(self.cnn2(x))
            return self.cnn3(x)

    class ResConv(module_base):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv = ops.Conv2d(channels, channels, 3, 1, 1)
            self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
            self.relu = nn.LeakyReLU(0.2, True)

        def forward(self, x: Any) -> Any:
            return self.relu(torch.addcmul(x, self.conv(x), self.beta))

    class IFBlock(module_base):
        def __init__(self, in_planes: int, channels: int) -> None:
            super().__init__()
            self.conv0 = nn.Sequential(
                nn.Sequential(
                    ops.Conv2d(in_planes, channels // 2, 3, 2, 1), nn.LeakyReLU(0.2, True)
                ),
                nn.Sequential(
                    ops.Conv2d(channels // 2, channels, 3, 2, 1), nn.LeakyReLU(0.2, True)
                ),
            )
            self.convblock = nn.Sequential(*(ResConv(channels) for _ in range(8)))
            self.lastconv = nn.Sequential(
                ops.ConvTranspose2d(channels, 4 * 13, 4, 2, 1), nn.PixelShuffle(2)
            )

        def forward(self, x: Any, flow: Any = None, scale: int = 1) -> Any:
            x = functional.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
            if flow is not None:
                flow = functional.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear").div_(
                    scale
                )
                x = torch.cat((x, flow), 1)
            features = self.convblock(self.conv0(x))
            result = functional.interpolate(
                self.lastconv(features), scale_factor=scale, mode="bilinear"
            )
            return result[:, :4] * scale, result[:, 4:5], result[:, 5:]

    class IFNet(module_base):
        def __init__(self) -> None:
            super().__init__()
            self.encode: Any = Head()
            block_in = [7 + 2 * head_ch] + [8 + 4 + 8 + 2 * head_ch] * 4
            self.blocks = nn.ModuleList([IFBlock(block_in[i], channels[i]) for i in range(5)])
            self.scale_list = [16, 8, 4, 2, 1]
            self.pad_align = 64
            self._warp_grids: dict[tuple[int, int], tuple[Any, Any]] = {}

        def get_dtype(self) -> Any:
            return self.encode.cnn0.weight.dtype

        def memory_used_forward(self, shape: Any, dtype: Any) -> int:
            return 300 * shape[1] * shape[2] * dtype.itemsize

        def _build_warp_grids(self, height: int, width: int, device: Any) -> None:
            if (height, width) in self._warp_grids:
                return
            self._warp_grids = {}
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32),
                indexing="ij",
            )
            self._warp_grids[(height, width)] = (
                torch.stack((grid_x, grid_y), dim=0).unsqueeze(0),
                torch.tensor(
                    [(width - 1.0) / 2.0, (height - 1.0) / 2.0], dtype=torch.float32, device=device
                ),
            )

        def warp(self, image: Any, flow: Any) -> Any:
            batch, _, height, width = image.shape
            base_grid, flow_div = self._warp_grids[(height, width)]
            flow_norm = torch.cat(
                [flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1
            ).float()
            grid = (base_grid.expand(batch, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return functional.grid_sample(
                image.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(image.dtype)

        def extract_features(self, image: Any) -> Any:
            return self.encode(image)

        def forward(self, img0: Any, img1: Any, timestep: Any = 0.5, cache: Any = None) -> Any:
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.full(
                    (img0.shape[0], 1, img0.shape[2], img0.shape[3]),
                    timestep,
                    device=img0.device,
                    dtype=img0.dtype,
                )
            self._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)
            batch = img0.shape[0]
            f0 = (
                cache["img0"].expand(batch, -1, -1, -1)
                if cache and "img0" in cache
                else self.encode(img0)
            )
            f1 = (
                cache["img1"].expand(batch, -1, -1, -1)
                if cache and "img1" in cache
                else self.encode(img1)
            )
            flow = mask = features = None
            warped0, warped1 = img0, img1
            for index, block in enumerate(self.blocks):
                if flow is None:
                    flow, mask, features = block(
                        torch.cat((img0, img1, f0, f1, timestep), 1),
                        None,
                        scale=self.scale_list[index],
                    )
                else:
                    delta, mask, features = block(
                        torch.cat(
                            (
                                warped0,
                                warped1,
                                self.warp(f0, flow[:, :2]),
                                self.warp(f1, flow[:, 2:4]),
                                timestep,
                                mask,
                                features,
                            ),
                            1,
                        ),
                        flow,
                        scale=self.scale_list[index],
                    )
                    flow = flow.add_(delta)
                warped0 = self.warp(img0, flow[:, :2])
                warped1 = self.warp(img1, flow[:, 2:4])
            return torch.lerp(warped1, warped0, torch.sigmoid(mask))

    return IFNet()


def film_model() -> Any:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    functional = importlib.import_module("torch.nn.functional")
    ops = importlib.import_module("dinkster_inference.ops").disable_weight_init
    module_base = cast(type[object], nn.Module)

    class FilmConv2d(module_base):
        def __init__(self, in_ch: int, out_ch: int, size: int, activation: bool = True) -> None:
            super().__init__()
            self.even_pad = not size % 2
            self.conv = ops.Conv2d(
                in_ch, out_ch, kernel_size=size, padding=size // 2 if size % 2 else 0
            )
            self.activation = nn.LeakyReLU(0.2) if activation else None

        def forward(self, x: Any) -> Any:
            if self.even_pad:
                x = functional.pad(x, (0, 1, 0, 1))
            x = self.conv(x)
            if self.activation is not None:
                x = self.activation(x)
            return x

    def image_pyramid(image: Any, levels: int) -> list[Any]:
        result = [image]
        for _ in range(1, levels):
            image = functional.avg_pool2d(image, 2, 2)
            result.append(image)
        return result

    def flow_synthesis(residuals: Any) -> list[Any]:
        flow = residuals[-1]
        result = [flow]
        for residual in residuals[:-1][::-1]:
            flow = (
                functional.interpolate(
                    flow, size=residual.shape[2:4], mode="bilinear", scale_factor=None
                )
                .mul_(2)
                .add_(residual)
            )
            result.append(flow)
        result.reverse()
        return result

    def multiply_pyramid(pyramid: Any, scalar: Any) -> list[Any]:
        return [image * scalar[:, None, None, None] for image in pyramid]

    def concatenate_pyramids(first: Any, second: Any) -> list[Any]:
        return [torch.cat([a, b], dim=1) for a, b in zip(first, second, strict=False)]

    class SubTreeExtractor(module_base):
        def __init__(self, in_ch: int = 3, channels: int = 64, layers: int = 4) -> None:
            super().__init__()
            convs: list[Any] = []
            for index in range(layers):
                out_ch = channels << index
                convs.append(
                    nn.Sequential(FilmConv2d(in_ch, out_ch, 3), FilmConv2d(out_ch, out_ch, 3))
                )
                in_ch = out_ch
            self.convs = nn.ModuleList(convs)

        def forward(self, image: Any, count: int) -> list[Any]:
            head = image
            result: list[Any] = []
            for index, layer in enumerate(self.convs):
                head = layer(head)
                result.append(head)
                if index < count - 1:
                    head = functional.avg_pool2d(head, 2, 2)
            return result

    class FeatureExtractor(module_base):
        def __init__(self, channels: int = 64, sub_levels: int = 4) -> None:
            super().__init__()
            self.extract_sublevels: Any = SubTreeExtractor(3, channels, sub_levels)
            self.sub_levels = sub_levels

        def forward(self, pyramid: Any) -> list[Any]:
            sub_pyramids: list[Any] = [
                self.extract_sublevels(pyramid[i], min(len(pyramid) - i, self.sub_levels))
                for i in range(len(pyramid))
            ]
            result: list[Any] = []
            for index in range(len(pyramid)):
                features = sub_pyramids[index][0]
                for depth in range(1, self.sub_levels):
                    if depth <= index:
                        features = torch.cat([features, sub_pyramids[index - depth][depth]], dim=1)
                result.append(features)
                if index >= self.sub_levels - 1:
                    sub_pyramids[index - self.sub_levels + 1] = None
            return result

    class FlowEstimator(module_base):
        def __init__(self, in_ch: int, num_convs: int, filters: int) -> None:
            super().__init__()
            self._convs = nn.ModuleList()
            for _ in range(num_convs):
                self._convs.append(FilmConv2d(in_ch, filters, 3))
                in_ch = filters
            self._convs.append(FilmConv2d(in_ch, filters // 2, 1))
            self._convs.append(FilmConv2d(filters // 2, 2, 1, activation=False))

        def forward(self, first: Any, second: Any) -> Any:
            net = torch.cat([first, second], dim=1)
            for conv in self._convs:
                net = conv(net)
            return net

    class PyramidFlowEstimator(module_base):
        def __init__(self) -> None:
            super().__init__()
            in_ch = 64 << 1
            predictors: list[Any] = []
            for index, filters in enumerate((32, 64, 128, 256)):
                predictors.append(FlowEstimator(in_ch, 3, filters))
                in_ch += 64 << (index + 2)
            self._predictor: Any = predictors[-1]
            self._predictors = nn.ModuleList(predictors[:-1][::-1])

        def forward(self, first: Any, second: Any, warp: Any) -> list[Any]:
            levels = len(first)
            flow = self._predictor(first[-1], second[-1])
            residuals = [flow]
            steps: list[tuple[int, Any]] = [
                (i, self._predictor) for i in range(levels - 2, len(self._predictors) - 1, -1)
            ]
            steps += [(len(self._predictors) - 1 - k, p) for k, p in enumerate(self._predictors)]
            for index, predictor in steps:
                flow = functional.interpolate(
                    flow, size=first[index].shape[2:4], mode="bilinear"
                ).mul_(2)
                residual = predictor(first[index], warp(second[index], flow))
                residuals.append(residual)
                flow = flow.add_(residual)
            residuals.reverse()
            return residuals

    def fusion_channels(level: int) -> int:
        return (sum(64 << i for i in range(level)) + 3 + 2) * 2

    class Fusion(module_base):
        def __init__(self) -> None:
            super().__init__()
            self.output_conv = ops.Conv2d(64, 3, kernel_size=1)
            self.convs = nn.ModuleList()
            in_ch = fusion_channels(4)
            increase = 0
            for index in range(4)[::-1]:
                filters = (64 << index) if index < 3 else (64 << 3)
                self.convs.append(
                    nn.ModuleList(
                        [
                            FilmConv2d(in_ch, filters, 2, activation=False),
                            FilmConv2d(in_ch + (increase or filters), filters, 3),
                            FilmConv2d(filters, filters, 3),
                        ]
                    )
                )
                in_ch = filters
                increase = fusion_channels(index) - filters // 2

        def forward(self, pyramid: Any) -> Any:
            net = pyramid[-1]
            for index, layers in enumerate(self.convs):
                level = len(self.convs) - 1 - index
                net = layers[0](
                    functional.interpolate(net, size=pyramid[level].shape[2:4], mode="nearest")
                )
                net = layers[2](layers[1](torch.cat([pyramid[level], net], dim=1)))
            return self.output_conv(net)

    class FILMNet(module_base):
        def __init__(self) -> None:
            super().__init__()
            self.pyramid_levels = 7
            self.fusion_pyramid_levels = 5
            self.extract: Any = FeatureExtractor()
            self.predict_flow: Any = PyramidFlowEstimator()
            self.fuse: Any = Fusion()
            self._warp_grids: dict[tuple[int, int], tuple[Any, Any]] = {}

        def get_dtype(self) -> Any:
            return self.extract.extract_sublevels.convs[0][0].conv.weight.dtype

        def memory_used_forward(self, shape: Any, dtype: Any) -> int:
            return 1700 * shape[1] * shape[2] * dtype.itemsize

        def _build_warp_grids(self, height: int, width: int, device: Any) -> None:
            if (height, width) in self._warp_grids:
                return
            self._warp_grids = {}
            for _ in range(self.pyramid_levels):
                self._warp_grids[(height, width)] = (
                    torch.linspace(
                        -(1 - 1 / width), 1 - 1 / width, width, dtype=torch.float32, device=device
                    ),
                    torch.linspace(
                        -(1 - 1 / height),
                        1 - 1 / height,
                        height,
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                height, width = height // 2, width // 2

        def warp(self, image: Any, flow: Any) -> Any:
            grid_x, grid_y = self._warp_grids[(flow.shape[2], flow.shape[3])]
            height, width = flow.shape[2:4]
            dx = flow[:, 0].float() / (width * 0.5)
            dy = flow[:, 1].float() / (height * 0.5)
            grid = torch.stack([grid_x[None, None, :] + dx, grid_y[None, :, None] + dy], dim=3)
            return functional.grid_sample(
                image.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
            ).to(image.dtype)

        def extract_features(self, image: Any) -> Any:
            pyramid = image_pyramid(image, self.pyramid_levels)
            return pyramid, self.extract(pyramid)

        def forward(self, img0: Any, img1: Any, timestep: Any = 0.5, cache: Any = None) -> Any:
            time = (
                timestep.mean(dim=(1, 2, 3)).item()
                if isinstance(timestep, torch.Tensor)
                else timestep
            )
            return self.forward_multi_timestep(img0, img1, [time], cache=cache)

        def forward_multi_timestep(
            self, img0: Any, img1: Any, timesteps: Any, cache: Any = None
        ) -> Any:
            self._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)
            images0, features0 = (
                cache["img0"] if cache and "img0" in cache else self.extract_features(img0)
            )
            images1, features1 = (
                cache["img1"] if cache and "img1" in cache else self.extract_features(img1)
            )
            forward = flow_synthesis(self.predict_flow(features0, features1, self.warp))[
                : self.fusion_pyramid_levels
            ]
            backward = flow_synthesis(self.predict_flow(features1, features0, self.warp))[
                : self.fusion_pyramid_levels
            ]
            levels = self.fusion_pyramid_levels
            targets = [
                concatenate_pyramids(images0[:levels], features0[:levels]),
                concatenate_pyramids(images1[:levels], features1[:levels]),
            ]
            del images0, images1, features0, features1
            results: list[Any] = []
            times = torch.tensor(timesteps, device=img0.device, dtype=img0.dtype)
            for index in range(len(timesteps)):
                time = times[index : index + 1]
                backward_scaled = multiply_pyramid(backward, time)
                forward_scaled = multiply_pyramid(forward, 1 - time)
                forward_warped = [
                    self.warp(f, flow) for f, flow in zip(targets[0], backward_scaled, strict=False)
                ]
                backward_warped = [
                    self.warp(f, flow) for f, flow in zip(targets[1], forward_scaled, strict=False)
                ]
                aligned = [
                    torch.cat([fw, bw, bf, ff], dim=1)
                    for fw, bw, bf, ff in zip(
                        forward_warped,
                        backward_warped,
                        backward_scaled,
                        forward_scaled,
                        strict=False,
                    )
                ]
                del forward_warped, backward_warped, backward_scaled, forward_scaled
                results.append(self.fuse(aligned))
                del aligned
            return torch.cat(results, dim=0)

    return FILMNet()
