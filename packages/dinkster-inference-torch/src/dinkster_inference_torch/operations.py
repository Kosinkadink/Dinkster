"""The Operations seam: typed layer factories for native architectures.

The reference threads an ``operations``/``conv_op`` class namespace
through every model constructor (comfy/ops.py disable_weight_init /
manual_cast @ b78cec87) and swaps behavior by swapping the namespace.
Dinkster keeps the seam but as a typed protocol: architecture
constructors receive an :class:`Operations` value and call factory
methods, so alternative implementations (cast-at-use over ``ops.py``
``cast_weight``, quantized layers) slot in without touching model
code.

Compile discipline (ROADMAP "torch.compile / triton compatibility
gate"): implementations must resolve their cast/quant configuration
at module-BIND time, never read per-call dynamic Python state inside
``forward``. :data:`INITLESS` satisfies this trivially - its modules
are stock torch layers whose only difference is skipped parameter
initialization (the reference's disable_weight_init: weights come
from a state dict, so init work is wasted and the empty allocation
stays untouched).

The factory surface grows with the architectures that need it: the
KL autoencoder consumes conv2d/group_norm; the CLIP text model adds
linear/layer_norm/embedding. :class:`CastOperations` is the
reference's manual_cast counterpart: parameters stay at the dtype the
checkpoint shipped and every forward casts them to the bind-time
compute dtype through the stage-4c :func:`~.ops.cast_weight`
pipeline. That is how the reference runs fp16-stored text encoders
at fp32 compute (sd1_clip.SDClipModel @ b78cec87 always constructs
with manual_cast ops and forwards at ``dtype=torch.float32``) -
T5-XXL activations overflow fp16, so fp16 COMPUTE collapses the
encoding while fp16 STORAGE + fp32 compute is the supported shape.
Plain fp8 parameters remain ordinary storage tensors in
:class:`CastOperations`. When the assembly-time fp8-matmul knob is
enabled, e4m3fn Linear weights use the reference's scale-one route
through the same scaled-mm seam as scaled :class:`Fp8Linear` weights.
The default remains cast-at-use dequantization.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import torch

from .ops import cast_weight
from .quant_linear import (
    Fp8MatmulBackend,
    fp8_matmul_forward,
    select_fp8_matmul_backend,
)

if TYPE_CHECKING:
    from .module_residency import ResidencyBinding

__all__ = [
    "INITLESS",
    "CastOperations",
    "InitlessOperations",
    "Operations",
    "ResidencyRouted",
    "bound_compute_dtype",
    "bound_compute_device",
    "module_compute_device",
    "materialized_conv2d_parameters",
    "materialized_embedding_weight",
    "materialized_group_norm_parameters",
    "materialized_linear_parameters",
    "materialized_rms_norm_weight",
]


class Operations(Protocol):
    """Layer factories for native model construction. Return types are
    the stock torch modules so call sites keep full typing;
    implementations subclass them (the reference does the same:
    manual_cast.Conv2d is an ops.Conv2d)."""

    def conv1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> torch.nn.Conv1d: ...

    def conv_transpose1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
    ) -> torch.nn.ConvTranspose1d: ...

    def conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ) -> torch.nn.Conv2d: ...

    def conv_transpose2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
    ) -> torch.nn.ConvTranspose2d: ...

    def conv3d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> torch.nn.Conv3d: ...

    def group_norm(
        self,
        num_channels: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
    ) -> torch.nn.GroupNorm: ...

    def batch_norm2d(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> torch.nn.BatchNorm2d: ...

    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> torch.nn.Linear: ...

    def layer_norm(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.nn.LayerNorm: ...

    def embedding(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> torch.nn.Embedding: ...

    def rms_norm(
        self,
        normalized_shape: int,
        *,
        eps: float | None = None,
    ) -> torch.nn.RMSNorm: ...


class ResidencyRouted:
    """Mixin for module state owners routed through component residency.

    Subclasses pair this mixin with ``torch.nn.Module`` and bracket operations
    over direct parameters and persistent buffers with the binding returned by
    :meth:`_offloaded_residency`. Factory children route their own state.
    """

    _residency: ResidencyBinding | None = None
    _residency_prefetch_binding: ResidencyBinding | None = None
    _residency_prefetch_requests: tuple[tuple[str, torch.dtype | None], ...] = ()

    def bind_residency(self, binding: ResidencyBinding) -> None:
        self._residency = binding
        self._residency_prefetch_binding = None

    def residency_binding(self) -> ResidencyBinding | None:
        return self._residency

    def _offloaded_residency(self) -> ResidencyBinding | None:
        binding = self._residency
        if binding is None or binding.unit_state.loaded:
            return None
        return binding

    @contextmanager
    def materialized_state(
        self, name: str, *, device: torch.device, dtype: torch.dtype
    ) -> Generator[torch.Tensor, None, None]:
        """Cast direct state while retaining its raw-storage residency lease."""
        stored = cast(torch.Tensor, getattr(self, name))
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                yield lease.get(name, dtype=stored.dtype).to(device=device, dtype=dtype)
            return
        yield stored.to(device=device, dtype=dtype)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return stored.dtype

    def residency_materialization_dtype(self, stored: torch.Tensor) -> torch.dtype:
        """Return the widest dtype this route can request for ``stored``."""
        return self._prefetch_dtype(stored)

    def _residency_uses_raw_storage(self, name: str) -> bool:
        del name
        return not isinstance(self, _CastMixin)

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._offloaded_residency()
        if binding is None:
            return None
        if self._residency_prefetch_binding is not binding:
            module = cast(torch.nn.Module, self)
            buffers = tuple(module.named_buffers(recurse=False))
            if buffers:
                # Non-persistent buffers are absent from the state dict and
                # therefore from the residency store.
                persistent = set(module.state_dict(keep_vars=True))
                buffers = tuple(buffer for buffer in buffers if buffer[0] in persistent)
            tensors = tuple(module.named_parameters(recurse=False)) + buffers
            self._residency_prefetch_requests = tuple(
                (binding.key(name), self.residency_materialization_dtype(stored))
                for name, stored in tensors
            )
            self._residency_prefetch_binding = binding
        return binding.mechanism, self._residency_prefetch_requests


def bound_compute_device(module: torch.nn.Module) -> torch.device | None:
    """Load device of the module's bound residency mechanism, or None
    when the module is unbound."""
    if isinstance(module, ResidencyRouted):
        binding = module.residency_binding()
        if binding is not None:
            return binding.mechanism.load_device
    return None


def module_compute_device(module: torch.nn.Module) -> torch.device:
    """Resolve execution placement without mistaking offloaded storage for compute."""
    bound = bound_compute_device(module)
    if bound is not None:
        return bound
    mechanism = cast("Any", module.__dict__.get("_dinkster_resident_weights"))
    if mechanism is not None:
        return torch.device(mechanism.load_device)
    return next(module.parameters()).device


def _optional_parameter(module: torch.nn.Module, name: str) -> torch.nn.Parameter | None:
    value = getattr(module, name, None)
    return value if isinstance(value, torch.nn.Parameter) else None


class _InitlessConv2d(ResidencyRouted, torch.nn.Conv2d):
    def reset_parameters(self) -> None:
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(input)
        with binding.lease() as lease:
            stored_bias = _optional_parameter(self, "bias")
            bias = None if stored_bias is None else lease.get("bias", dtype=stored_bias.dtype)
            return self._conv_forward(
                input,
                lease.get("weight", dtype=self.weight.dtype),
                bias,
            )


class _InitlessConvTranspose2d(ResidencyRouted, torch.nn.ConvTranspose2d):
    def reset_parameters(self) -> None:
        return None

    def forward(
        self,
        input: torch.Tensor,
        output_size: list[int] | None = None,
    ) -> torch.Tensor:
        padding = cast(tuple[int, ...], self.padding)
        output_padding = self._output_padding(
            input,
            output_size,
            list(self.stride),
            list(padding),
            list(self.kernel_size),
            2,
            list(self.dilation),
        )
        binding = self._offloaded_residency()
        if binding is None:
            return torch.nn.functional.conv_transpose2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        with binding.lease() as lease:
            stored_bias = _optional_parameter(self, "bias")
            bias = None if stored_bias is None else lease.get("bias", dtype=stored_bias.dtype)
            return torch.nn.functional.conv_transpose2d(
                input,
                lease.get("weight", dtype=self.weight.dtype),
                bias,
                self.stride,
                padding,
                output_padding,
                self.groups,
                self.dilation,
            )


class _InitlessConv1d(ResidencyRouted, torch.nn.Conv1d):
    def reset_parameters(self) -> None:
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(input)
        with binding.lease() as lease:
            stored_bias = _optional_parameter(self, "bias")
            bias = None if stored_bias is None else lease.get("bias", dtype=stored_bias.dtype)
            return self._conv_forward(
                input,
                lease.get("weight", dtype=self.weight.dtype),
                bias,
            )


class _InitlessConvTranspose1d(ResidencyRouted, torch.nn.ConvTranspose1d):
    def reset_parameters(self) -> None:
        return None

    def forward(
        self,
        input: torch.Tensor,
        output_size: list[int] | None = None,
    ) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(input, output_size)
        output_padding = self._output_padding(
            input,
            output_size,
            list(self.stride),
            list(cast(tuple[int, ...], self.padding)),
            list(self.kernel_size),
            1,
            list(self.dilation),
        )
        with binding.lease() as lease:
            stored_bias = _optional_parameter(self, "bias")
            bias = None if stored_bias is None else lease.get("bias", dtype=stored_bias.dtype)
            return torch.nn.functional.conv_transpose1d(
                input,
                lease.get("weight", dtype=self.weight.dtype),
                bias,
                self.stride,
                cast(tuple[int, ...], self.padding),
                output_padding,
                self.groups,
                self.dilation,
            )


class _InitlessConv3d(ResidencyRouted, torch.nn.Conv3d):
    def reset_parameters(self) -> None:
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(input)
        with binding.lease() as lease:
            stored_bias = _optional_parameter(self, "bias")
            bias = None if stored_bias is None else lease.get("bias", dtype=stored_bias.dtype)
            return self._conv_forward(
                input,
                lease.get("weight", dtype=self.weight.dtype),
                bias,
            )


class _InitlessGroupNorm(ResidencyRouted, torch.nn.GroupNorm):
    def reset_parameters(self) -> None:
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(input)
        with binding.lease() as lease:
            stored_weight = _optional_parameter(self, "weight")
            stored_bias = _optional_parameter(self, "bias")
            weight = (
                None if stored_weight is None else lease.get("weight", dtype=stored_weight.dtype)
            )
            bias = None if stored_bias is None else lease.get("bias", dtype=stored_bias.dtype)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)


class _InitlessBatchNorm2d(ResidencyRouted, torch.nn.BatchNorm2d):
    def reset_parameters(self) -> None:
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("residency-routed BatchNorm2d supports eval mode only")
        binding = self._offloaded_residency()
        running_mean = self.running_mean
        running_var = self.running_var
        weight = _optional_parameter(self, "weight")
        bias = _optional_parameter(self, "bias")
        if binding is not None:
            with binding.lease() as lease:
                running_mean = (
                    None
                    if self.running_mean is None
                    else lease.get("running_mean", dtype=self.running_mean.dtype)
                )
                running_var = (
                    None
                    if self.running_var is None
                    else lease.get("running_var", dtype=self.running_var.dtype)
                )
                weight = None if weight is None else lease.get("weight", dtype=weight.dtype)
                bias = None if bias is None else lease.get("bias", dtype=bias.dtype)
                return torch.nn.functional.batch_norm(
                    input,
                    running_mean,
                    running_var,
                    weight,
                    bias,
                    running_mean is None and running_var is None,
                    0.0 if self.momentum is None else self.momentum,
                    self.eps,
                )
        return torch.nn.functional.batch_norm(
            input,
            running_mean,
            running_var,
            weight,
            bias,
            running_mean is None and running_var is None,
            0.0 if self.momentum is None else self.momentum,
            self.eps,
        )


class _InitlessLinear(ResidencyRouted, torch.nn.Linear):
    def reset_parameters(self) -> None:
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(input)
        with binding.lease() as lease:
            stored_bias = _optional_parameter(self, "bias")
            bias = None if stored_bias is None else lease.get("bias", dtype=stored_bias.dtype)
            return torch.nn.functional.linear(
                input, lease.get("weight", dtype=self.weight.dtype), bias
            )


class _InitlessLayerNorm(ResidencyRouted, torch.nn.LayerNorm):
    def reset_parameters(self) -> None:
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(input)
        with binding.lease() as lease:
            stored_weight = _optional_parameter(self, "weight")
            stored_bias = _optional_parameter(self, "bias")
            weight = (
                None if stored_weight is None else lease.get("weight", dtype=stored_weight.dtype)
            )
            bias = None if stored_bias is None else lease.get("bias", dtype=stored_bias.dtype)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )


class _InitlessEmbedding(ResidencyRouted, torch.nn.Embedding):
    def reset_parameters(self) -> None:
        return None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(input)
        with binding.lease() as lease:
            return torch.nn.functional.embedding(
                input,
                lease.get("weight", dtype=self.weight.dtype),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )


class _InitlessRMSNorm(ResidencyRouted, torch.nn.RMSNorm):
    def reset_parameters(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            return super().forward(x)
        with binding.lease() as lease:
            stored_weight = _optional_parameter(self, "weight")
            weight = (
                None if stored_weight is None else lease.get("weight", dtype=stored_weight.dtype)
            )
            return torch.nn.functional.rms_norm(x, self.normalized_shape, weight, self.eps)


def bind_residency_layer(module: torch.nn.Module, binding: ResidencyBinding | None = None) -> bool:
    """Recognize and optionally bind one factory-produced layer."""
    if not isinstance(module, ResidencyRouted):
        return False
    if binding is not None:
        module.bind_residency(binding)
    return True


class InitlessOperations:
    """Stock layers with parameter initialization skipped - the
    reference's comfy/ops.py disable_weight_init @ b78cec87. For
    models whose every parameter is loaded from a checkpoint."""

    def conv1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> torch.nn.Conv1d:
        return _InitlessConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def conv_transpose1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
    ) -> torch.nn.ConvTranspose1d:
        return _InitlessConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ) -> torch.nn.Conv2d:
        return _InitlessConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def conv_transpose2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
    ) -> torch.nn.ConvTranspose2d:
        return _InitlessConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def conv3d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> torch.nn.Conv3d:
        return _InitlessConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def group_norm(
        self,
        num_channels: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
    ) -> torch.nn.GroupNorm:
        return _InitlessGroupNorm(num_groups, num_channels, eps=eps, affine=True)

    def batch_norm2d(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> torch.nn.BatchNorm2d:
        return _InitlessBatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> torch.nn.Linear:
        return _InitlessLinear(in_features, out_features, bias=bias)

    def layer_norm(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.nn.LayerNorm:
        return _InitlessLayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def embedding(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> torch.nn.Embedding:
        return _InitlessEmbedding(num_embeddings, embedding_dim)

    def rms_norm(
        self,
        normalized_shape: int,
        *,
        eps: float | None = None,
    ) -> torch.nn.RMSNorm:
        # eps=None defers to the input dtype's machine epsilon inside
        # F.rms_norm - the reference's default (comfy/ops.py RMSNorm
        # constructed without eps @ b78cec87).
        return _InitlessRMSNorm(normalized_shape, eps=eps)


INITLESS = InitlessOperations()


class _CastMixin(torch.nn.Module):
    """Weight access for cast-at-use modules: parameters keep their
    storage dtype; :meth:`_cast` produces the compute-dtype view at
    forward time. ``_compute_dtype`` is fixed when the factory binds
    the module (compile discipline: dynamo specializes on it, no
    per-forward Python policy)."""

    _compute_dtype: torch.dtype

    def bind_compute_dtype(self, dtype: torch.dtype) -> None:
        self._compute_dtype = dtype

    def compute_dtype(self) -> torch.dtype:
        return self._compute_dtype

    def _cast(self, stored: torch.Tensor) -> torch.Tensor:
        return cast_weight(stored, dtype=self._compute_dtype)

    def _cast_optional(self, stored: torch.Tensor | None) -> torch.Tensor | None:
        if stored is None:
            return None
        return self._cast(stored)

    def _prefetch_dtype(self, stored: torch.Tensor) -> torch.dtype:
        return self._compute_dtype


def bound_compute_dtype(module: torch.nn.Module) -> torch.dtype | None:
    """Compute dtype bound to a cast-at-use module, if any."""
    return module.compute_dtype() if isinstance(module, _CastMixin) else None


class _CastLinear(_CastMixin, _InitlessLinear):
    fp8_matmul: bool
    _fp8_matmul_backend: Fp8MatmulBackend

    def bind_compute_dtype(self, dtype: torch.dtype) -> None:
        super().bind_compute_dtype(dtype)
        self.fp8_matmul = False
        self._fp8_matmul_backend = "torch"

    def bind_fp8_matmul(self, enabled: bool) -> None:
        """Bind the upstream plain-fp8 scale-1 hardware route."""
        if enabled and self.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"fp8 matmul requires float8_e4m3fn storage, got {self.weight.dtype}")
        if enabled:
            self._fp8_matmul_backend = select_fp8_matmul_backend()
        self.fp8_matmul = enabled

    def _residency_uses_raw_storage(self, name: str) -> bool:
        return name == "weight" and self.weight.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )

    def _raw_residency_weight(self, binding: ResidencyBinding) -> bool:
        return self._residency_uses_raw_storage(
            "weight"
        ) and not binding.mechanism.weight_functions(binding.key("weight"))

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._offloaded_residency()
        if binding is None:
            return None
        raw_weight = self._raw_residency_weight(binding)
        requests = [
            (
                binding.key("weight"),
                self.weight.dtype if raw_weight else self._compute_dtype,
            )
        ]
        if _optional_parameter(self, "bias") is not None:
            requests.append((binding.key("bias"), self._compute_dtype))
        return binding.mechanism, tuple(requests)

    def _plain_fp8_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # comfy/ops.py fp8_linear @ b78cec87 synthesizes distinct
        # float32 scale-1 tensors for plain fp8 input and weight.
        input_scale = torch.ones((), device=input.device, dtype=torch.float32)
        weight_scale = torch.ones((), device=input.device, dtype=torch.float32)
        return fp8_matmul_forward(
            input,
            weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            bias=bias,
            out_dtype=self._compute_dtype,
            backend=self._fp8_matmul_backend,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_bias = _optional_parameter(self, "bias")
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                raw_weight = self._raw_residency_weight(binding)
                weight = lease.get(
                    "weight", dtype=self.weight.dtype if raw_weight else self._compute_dtype
                )
                if self.fp8_matmul and input.ndim in (2, 3) and raw_weight:
                    return self._plain_fp8_forward(
                        input,
                        weight,
                        bias,
                    )
                return torch.nn.functional.linear(
                    input,
                    cast_weight(weight, dtype=self._compute_dtype) if raw_weight else weight,
                    bias,
                )
        if self.fp8_matmul and input.ndim in (2, 3):
            return self._plain_fp8_forward(
                input,
                self.weight,
                self._cast_optional(self.bias),
            )
        return torch.nn.functional.linear(
            input, self._cast(self.weight), self._cast_optional(self.bias)
        )


@contextmanager
def materialized_embedding_weight(
    module: torch.nn.Embedding,
) -> Generator[torch.Tensor]:
    """Yield one routed embedding's weight in its compute dtype."""
    if not isinstance(module, _InitlessEmbedding):
        raise TypeError("embedding must use an operations-managed implementation")
    binding = module._offloaded_residency()  # pyright: ignore[reportPrivateUsage]
    compute_dtype = bound_compute_dtype(module)
    if binding is not None:
        with binding.lease() as lease:
            yield lease.get(
                "weight",
                dtype=module.weight.dtype if compute_dtype is None else compute_dtype,
            )
        return
    yield (
        module.weight if compute_dtype is None else cast_weight(module.weight, dtype=compute_dtype)
    )


@contextmanager
def materialized_conv2d_parameters(
    module: torch.nn.Conv2d,
) -> Generator[tuple[torch.Tensor, torch.Tensor | None]]:
    """Yield one routed Conv2d's weight and bias in its compute dtype."""
    if not isinstance(module, _InitlessConv2d):
        raise TypeError("Conv2d must use an operations-managed implementation")
    binding = module._offloaded_residency()  # pyright: ignore[reportPrivateUsage]
    compute_dtype = bound_compute_dtype(module)
    if binding is not None:
        with binding.lease() as lease:
            weight = lease.get(
                "weight",
                dtype=module.weight.dtype if compute_dtype is None else compute_dtype,
            )
            stored_bias = _optional_parameter(module, "bias")
            bias = (
                None
                if stored_bias is None
                else lease.get(
                    "bias",
                    dtype=stored_bias.dtype if compute_dtype is None else compute_dtype,
                )
            )
            yield weight, bias
        return
    weight = (
        module.weight if compute_dtype is None else cast_weight(module.weight, dtype=compute_dtype)
    )
    bias = _optional_parameter(module, "bias")
    if bias is not None and compute_dtype is not None:
        bias = cast_weight(bias, dtype=compute_dtype)
    yield weight, bias


@contextmanager
def materialized_linear_parameters(
    module: torch.nn.Linear,
) -> Generator[tuple[torch.Tensor, torch.Tensor | None]]:
    """Yield one routed linear's weight and bias in its compute dtype."""
    if not isinstance(module, _InitlessLinear):
        raise TypeError("linear must use an operations-managed implementation")
    binding = module._offloaded_residency()  # pyright: ignore[reportPrivateUsage]
    compute_dtype = bound_compute_dtype(module)
    if binding is not None:
        with binding.lease() as lease:
            raw_weight = isinstance(module, _CastLinear) and module._raw_residency_weight(  # pyright: ignore[reportPrivateUsage]
                binding
            )
            weight = lease.get(
                "weight",
                dtype=(
                    module.weight.dtype if compute_dtype is None or raw_weight else compute_dtype
                ),
            )
            if raw_weight:
                assert compute_dtype is not None
                weight = cast_weight(weight, dtype=compute_dtype)
            stored_bias = _optional_parameter(module, "bias")
            bias = (
                None
                if stored_bias is None
                else lease.get(
                    "bias",
                    dtype=stored_bias.dtype if compute_dtype is None else compute_dtype,
                )
            )
            yield weight, bias
        return
    weight = (
        module.weight if compute_dtype is None else cast_weight(module.weight, dtype=compute_dtype)
    )
    bias = _optional_parameter(module, "bias")
    if bias is not None and compute_dtype is not None:
        bias = cast_weight(bias, dtype=compute_dtype)
    yield weight, bias


@contextmanager
def materialized_group_norm_parameters(
    module: torch.nn.GroupNorm,
) -> Generator[tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Yield one routed GroupNorm's weight and bias in its compute dtype."""
    if not isinstance(module, _InitlessGroupNorm):
        raise TypeError("GroupNorm must use an operations-managed implementation")
    binding = module._offloaded_residency()  # pyright: ignore[reportPrivateUsage]
    compute_dtype = bound_compute_dtype(module)
    stored_weight = _optional_parameter(module, "weight")
    stored_bias = _optional_parameter(module, "bias")
    if binding is not None:
        with binding.lease() as lease:
            weight = (
                None
                if stored_weight is None
                else lease.get(
                    "weight",
                    dtype=stored_weight.dtype if compute_dtype is None else compute_dtype,
                )
            )
            bias = (
                None
                if stored_bias is None
                else lease.get(
                    "bias",
                    dtype=stored_bias.dtype if compute_dtype is None else compute_dtype,
                )
            )
            yield weight, bias
        return
    weight = stored_weight
    bias = stored_bias
    if compute_dtype is not None:
        weight = None if weight is None else cast_weight(weight, dtype=compute_dtype)
        bias = None if bias is None else cast_weight(bias, dtype=compute_dtype)
    yield weight, bias


@contextmanager
def materialized_rms_norm_weight(
    module: torch.nn.RMSNorm,
) -> Generator[torch.Tensor]:
    """Yield one routed RMSNorm's weight in its compute dtype."""
    if not isinstance(module, _InitlessRMSNorm):
        raise TypeError("RMSNorm must use an operations-managed implementation")
    binding = module._offloaded_residency()  # pyright: ignore[reportPrivateUsage]
    compute_dtype = bound_compute_dtype(module)
    if binding is not None:
        with binding.lease() as lease:
            yield lease.get(
                "weight",
                dtype=module.weight.dtype if compute_dtype is None else compute_dtype,
            )
        return
    yield (
        module.weight if compute_dtype is None else cast_weight(module.weight, dtype=compute_dtype)
    )


def bind_fp8_matmul_layer(module: torch.nn.Module, enabled: bool) -> bool:
    """Recognize and bind a plain-fp8 cast-at-use Linear."""
    if not isinstance(module, _CastLinear) or module.weight.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return False
    module.bind_fp8_matmul(enabled)
    return True


class _CastConv2d(_CastMixin, _InitlessConv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_bias = _optional_parameter(self, "bias")
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                return self._conv_forward(
                    input,
                    lease.get("weight", dtype=self._compute_dtype),
                    bias,
                )
        return self._conv_forward(input, self._cast(self.weight), self._cast_optional(self.bias))


class _CastConvTranspose2d(_CastMixin, _InitlessConvTranspose2d):
    def forward(
        self,
        input: torch.Tensor,
        output_size: list[int] | None = None,
    ) -> torch.Tensor:
        padding = cast(tuple[int, ...], self.padding)
        output_padding = self._output_padding(
            input,
            output_size,
            list(self.stride),
            list(padding),
            list(self.kernel_size),
            2,
            list(self.dilation),
        )
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_bias = _optional_parameter(self, "bias")
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                return torch.nn.functional.conv_transpose2d(
                    input,
                    lease.get("weight", dtype=self._compute_dtype),
                    bias,
                    self.stride,
                    padding,
                    output_padding,
                    self.groups,
                    self.dilation,
                )
        return torch.nn.functional.conv_transpose2d(
            input,
            self._cast(self.weight),
            self._cast_optional(self.bias),
            self.stride,
            padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class _CastConv1d(_CastMixin, _InitlessConv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_bias = _optional_parameter(self, "bias")
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                return self._conv_forward(
                    input,
                    lease.get("weight", dtype=self._compute_dtype),
                    bias,
                )
        return self._conv_forward(input, self._cast(self.weight), self._cast_optional(self.bias))


class _CastConvTranspose1d(_CastMixin, _InitlessConvTranspose1d):
    def forward(
        self,
        input: torch.Tensor,
        output_size: list[int] | None = None,
    ) -> torch.Tensor:
        output_padding = self._output_padding(
            input,
            output_size,
            list(self.stride),
            list(cast(tuple[int, ...], self.padding)),
            list(self.kernel_size),
            1,
            list(self.dilation),
        )
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_bias = _optional_parameter(self, "bias")
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                return torch.nn.functional.conv_transpose1d(
                    input,
                    lease.get("weight", dtype=self._compute_dtype),
                    bias,
                    self.stride,
                    cast(tuple[int, ...], self.padding),
                    output_padding,
                    self.groups,
                    self.dilation,
                )
        return torch.nn.functional.conv_transpose1d(
            input,
            self._cast(self.weight),
            self._cast_optional(self.bias),
            self.stride,
            cast(tuple[int, ...], self.padding),
            output_padding,
            self.groups,
            self.dilation,
        )


class _CastConv3d(_CastMixin, _InitlessConv3d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_bias = _optional_parameter(self, "bias")
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                return self._conv_forward(
                    input,
                    lease.get("weight", dtype=self._compute_dtype),
                    bias,
                )
        return self._conv_forward(input, self._cast(self.weight), self._cast_optional(self.bias))


class _CastGroupNorm(_CastMixin, _InitlessGroupNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_weight = _optional_parameter(self, "weight")
                stored_bias = _optional_parameter(self, "bias")
                weight = (
                    None
                    if stored_weight is None
                    else lease.get("weight", dtype=self._compute_dtype)
                )
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                return torch.nn.functional.group_norm(
                    input, self.num_groups, weight, bias, self.eps
                )
        return torch.nn.functional.group_norm(
            input,
            self.num_groups,
            self._cast_optional(self.weight),
            self._cast_optional(self.bias),
            self.eps,
        )


class _CastBatchNorm2d(_CastMixin, _InitlessBatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("residency-routed BatchNorm2d supports eval mode only")
        stored_weight = _optional_parameter(self, "weight")
        stored_bias = _optional_parameter(self, "bias")
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                running_mean = (
                    None
                    if self.running_mean is None
                    else lease.get("running_mean", dtype=self._compute_dtype)
                )
                running_var = (
                    None
                    if self.running_var is None
                    else lease.get("running_var", dtype=self._compute_dtype)
                )
                weight = (
                    None
                    if stored_weight is None
                    else lease.get("weight", dtype=self._compute_dtype)
                )
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                return torch.nn.functional.batch_norm(
                    input,
                    running_mean,
                    running_var,
                    weight,
                    bias,
                    running_mean is None and running_var is None,
                    0.0 if self.momentum is None else self.momentum,
                    self.eps,
                )
        running_mean = None if self.running_mean is None else self._cast(self.running_mean)
        running_var = None if self.running_var is None else self._cast(self.running_var)
        return torch.nn.functional.batch_norm(
            input,
            running_mean,
            running_var,
            self._cast_optional(stored_weight),
            self._cast_optional(stored_bias),
            running_mean is None and running_var is None,
            0.0 if self.momentum is None else self.momentum,
            self.eps,
        )


class _CastLayerNorm(_CastMixin, _InitlessLayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_weight = _optional_parameter(self, "weight")
                stored_bias = _optional_parameter(self, "bias")
                weight = (
                    None
                    if stored_weight is None
                    else lease.get("weight", dtype=self._compute_dtype)
                )
                bias = None if stored_bias is None else lease.get("bias", dtype=self._compute_dtype)
                return torch.nn.functional.layer_norm(
                    input, self.normalized_shape, weight, bias, self.eps
                )
        return torch.nn.functional.layer_norm(
            input,
            self.normalized_shape,
            self._cast_optional(self.weight),
            self._cast_optional(self.bias),
            self.eps,
        )


class _CastEmbedding(_CastMixin, _InitlessEmbedding):
    def _residency_uses_raw_storage(self, name: str) -> bool:
        return name == "weight" and self.max_norm is None and not self.weight.requires_grad

    def _gather_from_storage(self, binding: ResidencyBinding | None = None) -> bool:
        if self.max_norm is not None or (torch.is_grad_enabled() and self.weight.requires_grad):
            return False
        return binding is None or not binding.mechanism.weight_functions(binding.key("weight"))

    def _gather(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        ).to(dtype=self._compute_dtype)

    def residency_prefetch(
        self,
    ) -> tuple[object, tuple[tuple[str, torch.dtype | None], ...]] | None:
        binding = self._offloaded_residency()
        if binding is None:
            return None
        weight_key = binding.key("weight")
        gather_from_storage = self._gather_from_storage(binding)
        dtype = self.weight.dtype if gather_from_storage else self._compute_dtype
        return binding.mechanism, ((weight_key, dtype),)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                gather_from_storage = self._gather_from_storage(binding)
                dtype = self.weight.dtype if gather_from_storage else self._compute_dtype
                return self._gather(input, lease.get("weight", dtype=dtype))
        weight = self.weight if self._gather_from_storage() else self._cast(self.weight)
        return self._gather(input, weight)


class _CastRMSNorm(_CastMixin, _InitlessRMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is not None:
            with binding.lease() as lease:
                stored_weight = _optional_parameter(self, "weight")
                weight = (
                    None
                    if stored_weight is None
                    else lease.get("weight", dtype=self._compute_dtype)
                )
                return torch.nn.functional.rms_norm(x, self.normalized_shape, weight, self.eps)
        return torch.nn.functional.rms_norm(
            x, self.normalized_shape, self._cast_optional(self.weight), self.eps
        )


class CastOperations:
    """Cast-at-use layers - the reference's comfy/ops.py manual_cast
    @ b78cec87 with the compute dtype fixed at bind time. Parameters
    stay at whatever dtype the checkpoint shipped (state-dict layout
    is identical to the stock layers); every forward runs
    :func:`~.ops.cast_weight` to produce compute-dtype weights, so
    storage dtype and compute dtype decouple. Inputs are expected at
    the compute dtype (the reference's text-encode path forwards at
    a hardcoded ``dtype=torch.float32`` over fp16 storage)."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype

    def _bind(self, module: _CastMixin) -> None:
        module.bind_compute_dtype(self.dtype)

    def conv1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> torch.nn.Conv1d:
        module = _CastConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._bind(module)
        return module

    def conv_transpose1d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        *,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
    ) -> torch.nn.ConvTranspose1d:
        module = _CastConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self._bind(module)
        return module

    def conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    ) -> torch.nn.Conv2d:
        module = _CastConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self._bind(module)
        return module

    def conv_transpose2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
    ) -> torch.nn.ConvTranspose2d:
        module = _CastConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self._bind(module)
        return module

    def conv3d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> torch.nn.Conv3d:
        module = _CastConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._bind(module)
        return module

    def group_norm(
        self,
        num_channels: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
    ) -> torch.nn.GroupNorm:
        module = _CastGroupNorm(num_groups, num_channels, eps=eps, affine=True)
        self._bind(module)
        return module

    def batch_norm2d(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> torch.nn.BatchNorm2d:
        module = _CastBatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self._bind(module)
        return module

    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> torch.nn.Linear:
        module = _CastLinear(in_features, out_features, bias=bias)
        self._bind(module)
        return module

    def layer_norm(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.nn.LayerNorm:
        module = _CastLayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self._bind(module)
        return module

    def embedding(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> torch.nn.Embedding:
        module = _CastEmbedding(num_embeddings, embedding_dim)
        self._bind(module)
        return module

    def rms_norm(
        self,
        normalized_shape: int,
        *,
        eps: float | None = None,
    ) -> torch.nn.RMSNorm:
        module = _CastRMSNorm(normalized_shape, eps=eps)
        self._bind(module)
        return module
