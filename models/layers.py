import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from timm.layers import trunc_normal_
except Exception:
    from torch.nn.init import trunc_normal_  # type: ignore

try:
    from timm.models.convnext import ConvNeXtBlock
except Exception:
    ConvNeXtBlock = None

from einops import rearrange, repeat, reduce, pack, unpack
import math
from functools import partial

from typing import Literal, Optional, Any, List, Sequence, Tuple
from einops.layers.torch import Rearrange

def create_param(data, param_type="default", requires_grad=True):
    p = nn.Parameter(data, requires_grad=requires_grad)
    p.param_type = param_type
    return p


def is_network_param(p):
    assert isinstance(
        p, nn.Parameter
    ), "Parameter should be an instance of nn.Parameter."
    assert hasattr(p, "param_type"), "Parameter should have param_type attribute."
    return p.param_type == "net"


def is_grid_param(p):
    assert isinstance(
        p, nn.Parameter
    ), "Parameter should be an instance of nn.Parameter."
    assert hasattr(p, "param_type"), "Parameter should have param_type attribute."
    # Treat both base grids and local grids as grid params for entropy modeling
    return p.param_type in ["grid", "local_grid"]


def get_func_norm(norm, **kwargs):
    if norm == "none":
        return FuncIdentity
    elif norm == "layernorm":
        return partial(FuncLayerNorm, eps=1e-6, **kwargs)
    elif norm == "layernorm-no-affine":
        return partial(FuncLayerNorm, elementwise_affine=False, eps=1e-6, **kwargs)
    else:
        raise NotImplementedError


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch // 2, in_ch // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch // 2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with conditioning.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # project conditioning to scale and shift
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, N, C) input
            cond: (B, cond_dim) or (B, N, cond_dim)
        Returns:
            normalized x, scale, shift: all (B, N, C)
        """
        x = self.norm(x)

        # broadcast cond to per-token if it's global
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, N, cond_dim)

        modulation = self.adaLN_modulation(cond)  # (B, N, 2*C)
        scale, shift = modulation.chunk(2, dim=-1)
        return x, scale, shift


def get_func_activation(activation, **kwargs):
    if activation == "none":
        return nn.Identity
    elif activation == "relu":
        return nn.ReLU
    elif activation == "relu6":
        return nn.ReLU6
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "softplus":
        return nn.Softplus
    else:
        raise NotImplementedError


class BatchGridTrilinear3D(nn.Module):
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            is_batch = False
            N = 1
            T, H, W, C = x.shape
        elif x.ndim == 5:
            is_batch = True
            N, T, H, W, C = x.shape
        else:
            raise NotImplementedError
        assert (
            H == self.output_size[1] and W == self.output_size[2]
        ), "F.interpolate has incorrect results in some cases, so use only temporal scale"
        x = x.view(N, 1, T, H * W * C)
        x = F.interpolate(
            x,
            size=(self.output_size[0], H * W * C),
            mode="bilinear",
            align_corners=self.align_corners,
        )
        x = (
            x.view(-1, H, W, C)
            if not is_batch
            else x.view(N, self.output_size[0], H, W, C)
        )
        return x


class FuncModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.func_key = ""
        self.func_params_keys = []
        self._params_cache = None

    def init_func_modules(self):
        """Assign unique names to every `FuncModule` in the hierarchy."""
        for name, module in self.named_modules():
            if isinstance(module, FuncModule):
                assert not module.func_key, "func_key already set"
                module.func_key = name.replace(".", "_")

    def _get_params(self):
        """Override in subclass to return {param_name: tensor}."""
        return {}

    def get_params_recursive(self):
        """Collect all parameters with hierarchical prefixes."""
        params = {}
        for module in self.modules():
            if isinstance(module, FuncModule):
                if module._params_cache is None:
                    module._params_cache = module._get_params()
                prefix = f"{module.func_key}__" if module.func_key else ""
                for k, v in module._params_cache.items():
                    params[f"{prefix}{k}"] = v
        return params

    def extract_params(self, params_dict):
        """Extract this module's parameters from the global dict."""
        prefix = f"{self.func_key}__" if self.func_key else ""
        return [params_dict.get(f"{prefix}{k}") for k in self.func_params_keys]

    # Backward-compatible helper used by functional layers
    def _extract_params(self, params_dict):
        prefix = f"{self.func_key}__" if self.func_key else ""
        out = []
        for k in self.func_params_keys:
            key = f"{prefix}{k}"
            out.append(params_dict.get(key) if params_dict is not None else None)
        return out


class FunctionalAdapter(nn.Module):
    def __init__(self, func_module: FuncModule):
        super().__init__()
        func_module.init_func_modules()
        params = func_module.get_params_recursive()
        self.params = nn.ParameterDict(
            {k: v for k, v in params.items() if v is not None}
        )
        self.func = func_module

    def forward(self, *args, **kwargs):
        if "mask" not in kwargs:
            kwargs["mask"] = None
        try:
            return self.func(*args, params=self.params, **kwargs)
        except TypeError as e:
            # Some functional layers (e.g., pure convs) do not accept mask; retry without it.
            if "mask" in kwargs:
                kwargs.pop("mask", None)
                return self.func(*args, params=self.params, **kwargs)
            raise e


class FuncIdentity(FuncModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def forward(self, x, params):
        return x


class FuncConstant(FuncModule):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.func_params_keys = ["constant"]

    def extra_repr(self):
        s = "features={features}"
        return s.format(**self.__dict__)

    def _get_params(self):
        constant = create_param(torch.zeros(self.features), "constant")
        return {"constant": constant}

    def forward(self, params):
        (constant,) = self._extract_params(params)
        return constant


class FuncLinear(FuncModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.func_params_keys = ["weight", "bias"]

    def extra_repr(self):
        s = "in_features={in_features}, out_features={out_features}, bias={bias}"
        return s.format(**self.__dict__)

    def _get_params(self):
        weight_shape = (self.out_features, self.in_features)
        weight = create_param(torch.empty(weight_shape), "linear_weight")
        bias = (
            create_param(torch.zeros(self.out_features), "linear_bias")
            if self.bias
            else None
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {"weight": weight, "bias": bias}

    def forward(self, x, params):
        assert x.ndim >= 2, "x should have at least 2 dimensions with shape [N, .., C]"
        weight, bias = self._extract_params(params)
        assert (
            weight.ndim == 2
        ), "weight should have 2 dimensions with shape [C_out, C_in]"
        assert (
            bias is None or bias.ndim == 1
        ), "bias should have 1 dimension with shape [C_out]"
        return F.linear(x, weight, bias)


class FuncLayerNorm(FuncModule):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,)
            if not isinstance(normalized_shape, tuple)
            else normalized_shape
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.bias = bias
        self.init_scale = 1.0
        self.func_params_keys = ["weight", "bias"]

    def extra_repr(self):
        s = (
            "normalized_shape={normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}, bias={bias}"
        )
        return s.format(**self.__dict__)

    def _get_params(self):
        weight = (
            create_param(
                self.init_scale * torch.ones(self.normalized_shape), "norm_weight"
            )
            if self.elementwise_affine
            else None
        )
        bias = (
            create_param(torch.zeros(self.normalized_shape), "norm_bias")
            if self.elementwise_affine and self.bias
            else None
        )
        return {"weight": weight, "bias": bias}

    def forward(self, x, params):
        assert x.ndim >= 2, "x should have at least 2 dimensions with shape [N, .., C]"
        weight, bias = self._extract_params(params)
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class FuncAdaLayerNormZero(FuncModule):
    """Adaptive LayerNorm with residual stats (functional)."""

    def __init__(
        self,
        dim: int,
        rank: Optional[int] = None,
        num_stats: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_stats = num_stats
        self.eps = eps
        self.lora = FuncLinear(dim, rank, bias=False) if rank else FuncIdentity()
        self.proj = FuncLinear(rank if rank else dim, num_stats * dim)
        self.norm = (
            FuncLayerNorm(dim, eps=eps, elementwise_affine=False) if eps else FuncIdentity()
        )
        self.activation = nn.SiLU()
        self.func_params_keys = []

    def extra_repr(self):
        return (
            f"dim={self.dim}, rank={self.rank}, num_stats={self.num_stats}, eps={self.eps}"
        )

    def _get_params(self):
        return {}

    def forward(
        self, x: Tensor, params: dict, z: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        z = self.activation(z)
        z = self.lora(z, params)
        stats = self.proj(z, params).chunk(self.num_stats, dim=-1)
        out = self.norm(x, params).mul(1 + stats[0]).add_(stats[1])
        return out, stats[2:]


class FuncAdaLayerNorm(FuncAdaLayerNormZero):
    """Adaptive LayerNorm (functional)."""

    def __init__(self, dim: int, rank: Optional[int] = None, eps: float = 1e-6):
        super().__init__(dim, rank, num_stats=2, eps=eps)

    def forward(self, x: Tensor, params: dict, z: Tensor) -> Tensor:
        return super().forward(x, params, z)[0]


class FuncLayerScale(FuncModule):
    def __init__(self, out_features: int, init_scale: float = 1e-3):
        super().__init__()
        self.out_features = out_features
        self.init_scale = init_scale
        self.func_params_keys = ["scale"]

    def extra_repr(self):
        s = "init_scale={init_scale}"
        return s.format(**self.__dict__)

    def _get_params(self):
        scale = create_param(self.init_scale * torch.ones([self.out_features]), "scale")
        return {"scale": scale}

    def forward(self, x, params):
        assert x.ndim >= 2, "x should have at least 2 dimensions with shape [N, .., C]"
        (scale,) = self._extract_params(params)
        return scale.view((1,) * (x.ndim - 1) + (scale.shape[-1],)) * x


class FuncConv3D(FuncModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (tuple, list))
            else (kernel_size,) * 3
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.func_params_keys = ["weight", "bias"]

    def extra_repr(self):
        s = (
            "in_channels={in_channels}, out_channels={out_channels}, "
            "kernel_size={kernel_size}, stride={stride}, "
            "padding={padding}, dilation={dilation}, groups={groups}, bias={bias}"
        )
        return s.format(**self.__dict__)

    def _get_params(self):
        weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        )
        weight = create_param(torch.empty(weight_shape), "conv_weight")
        bias = (
            create_param(torch.zeros(self.out_channels), "conv_bias")
            if self.bias
            else None
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {"weight": weight, "bias": bias}

    def forward(self, x, params):
        assert x.ndim == 5, "x should have 5 dimensions with shape [N, T, H, W, C]"
        weight, bias = self._extract_params(params)
        assert (
            weight.ndim == 5
        ), "weight should have 4 dimensions with shape [C_out, C_in, Kt, Kh, Kw]"
        assert (
            bias is None or bias.ndim == 1
        ), "bias should have 1 dimension with shape [C_out]"
        x = x.permute(0, 4, 1, 2, 3)
        x = F.conv3d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        x = x.permute(0, 2, 3, 4, 1)
        return x


class FuncConv2D(FuncModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (tuple, list))
            else (kernel_size,) * 2
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.func_params_keys = ["weight", "bias"]

    def extra_repr(self):
        s = (
            "in_channels={in_channels}, out_channels={out_channels}, "
            "kernel_size={kernel_size}, stride={stride}, "
            "padding={padding}, dilation={dilation}, groups={groups}, bias={bias}"
        )
        return s.format(**self.__dict__)

    def _get_params(self):
        weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        )
        weight = create_param(torch.empty(weight_shape), "conv_weight")
        bias = (
            create_param(torch.zeros(self.out_channels), "conv_bias")
            if self.bias
            else None
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {"weight": weight, "bias": bias}

    def forward(self, x, params):
        assert x.ndim == 5, "x should have 5 dimensions with shape [N, T, H, W, C]"
        weight, bias = self._extract_params(params)
        assert (
            weight.ndim == 4
        ), "weight should have 4 dimensions with shape [C_out, C_in, Kh, Kw]"
        assert (
            bias is None or bias.ndim == 1
        ), "bias should have 1 dimension with shape [C_out]"
        N, T, H1, W1, _ = x.shape
        x = x.view(N * T, H1, W1, -1).permute(0, 3, 1, 2)
        x = F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        _, _, H2, W2 = x.shape
        x = x.permute(0, 2, 3, 1).view(N, T, H2, W2, -1)
        return x


class FuncConv1D(FuncModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = True,
        pad_mode: str = "constant",
        layout: str = "auto",
        padding: Optional[Tuple[int, int]] = None,
    ):
        """
        1D convolution (along the temporal dimension).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        self.causal = causal
        self.pad_mode = pad_mode
        self.layout = layout.lower()
        self.explicit_padding = padding
        self.func_params_keys = ["weight", "bias"]

        if self.explicit_padding is not None:
            self.padding = self.explicit_padding
        elif causal:
            # causal: pad only on the left (past) side
            time_pad = dilation * (kernel_size - 1) + (1 - stride)
            self.padding = (time_pad, 0)
        else:
            # symmetric: pad equally on both sides
            time_pad_total = dilation * (kernel_size - 1)
            time_pad_left = time_pad_total // 2
            time_pad_right = time_pad_total - time_pad_left
            self.padding = (time_pad_left, time_pad_right)

    def extra_repr(self):
        s = (
            "in_channels={in_channels}, out_channels={out_channels}, "
            "kernel_size={kernel_size}, stride={stride}, dilation={dilation}, "
            "bias={bias}, causal={causal}, pad_mode={pad_mode}, padding={padding}"
        )
        return s.format(**self.__dict__)

    def _get_params(self):
        weight_shape = (self.out_channels, self.in_channels, self.kernel_size)
        weight = create_param(torch.empty(weight_shape), "conv1d_weight")
        bias = (
            create_param(torch.zeros(self.out_channels), "conv1d_bias")
            if self.bias
            else None
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {"weight": weight, "bias": bias}

    def forward(self, x, params):
        weight, bias = self._extract_params(params)
        assert (
            weight.ndim == 3
        ), "weight should have 3 dimensions with shape [C_out, C_in, K]"
        assert (
            bias is None or bias.ndim == 1
        ), "bias should have 1 dimension with shape [C_out]"

        # handle both [N, T, H, W, C] and [N, C, T, H, W] layouts
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input, got {x.ndim}D")

        layout = self.layout
        if layout == "auto":
            is_ncthw = x.shape[1] == self.in_channels
            is_nthwc = x.shape[-1] == self.in_channels
            if is_ncthw and not is_nthwc:
                layout = "ncthw"
            elif is_nthwc and not is_ncthw:
                layout = "nthwc"
            else:
                raise ValueError(
                    f"Ambiguous layout for input {x.shape}; set layout to 'NCTHW' or 'NTHWC'."
                )
        if layout == "ncthw":
            N, C, T, H, W = x.shape
        elif layout == "nthwc":
            N, T, H, W, C = x.shape
            x = x.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
        else:
            raise ValueError(f"Unsupported layout {self.layout}")
        if C != self.in_channels:
            raise ValueError(
                f"Input channels {C} do not match expected {self.in_channels}."
            )

        # reshape to [N*H*W, C, T] for 1D convolution
        N, C, T, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2)  # [N, H, W, C, T]
        x = x.reshape(N * H * W, C, T)

        # apply padding
        # for causal padding, use constant mode if sequence is too short
        if self.causal:
            pad_mode = self.pad_mode if self.padding[0] < x.shape[2] else "constant"
        else:
            pad_mode = self.pad_mode
        x = F.pad(x, self.padding, mode=pad_mode)

        # apply 1D convolution
        x = F.conv1d(x, weight, bias, self.stride, padding=0, dilation=self.dilation)

        # reshape back to [N, C, T', H, W]
        _, _, T_out = x.shape
        x = x.reshape(N, H, W, self.out_channels, T_out)
        x = x.permute(0, 3, 4, 1, 2)  # [N, C, T', H, W]

        # convert back to original layout if needed
        if layout == "NTHWC":
            x = x.permute(0, 2, 3, 4, 1)  # [N, T', H, W, C]

        return x


class FuncMaskedConv2d(FuncConv2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        mask_cur: bool = True,
        is_head: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_shape = (1, 1, *self.kernel_size)
        self.mask_cur = mask_cur
        self.is_head = is_head

    def _get_params(self):
        param_prefix = "em_conv" if self.is_head else "conv"
        weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            math.prod(self.kernel_size) // 2 + (not self.mask_cur),
        )
        weight = create_param(torch.empty(weight_shape), f"{param_prefix}_weight")
        bias = (
            create_param(torch.zeros(self.out_channels), f"{param_prefix}_bias")
            if self.bias
            else None
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {"weight": weight, "bias": bias}

    def forward(self, x, params):
        assert x.ndim == 5, "x should have 5 dimensions with shape [N, T, H, W, C]"
        weight, bias = self._extract_params(params)
        weight = F.pad(
            weight, (0, math.prod(self.kernel_size) - weight.shape[-1])
        ).view(*weight.shape[:-1], *self.kernel_size)
        prefix = self.func_key if self.func_key == "" else self.func_key + "__"
        return super().forward(x, {f"{prefix}weight": weight, f"{prefix}bias": bias})


class FuncMaskedConv3d(FuncConv3D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        mask_cur: bool = True,
        is_head: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_shape = (1, 1, *self.kernel_size)
        self.mask_cur = mask_cur
        self.is_head = is_head

    def _get_params(self):
        param_prefix = "em_conv" if self.is_head else "conv"
        weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            math.prod(self.kernel_size) // 2 + (not self.mask_cur),
        )
        weight = create_param(torch.empty(weight_shape), f"{param_prefix}_weight")
        bias = (
            create_param(torch.zeros(self.out_channels), f"{param_prefix}_bias")
            if self.bias
            else None
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {"weight": weight, "bias": bias}

    def forward(self, x, params):
        assert x.ndim == 5, "x should have 5 dimensions with shape [N, T, H, W, C]"
        weight, bias = self._extract_params(params)
        weight = F.pad(
            weight, (0, math.prod(self.kernel_size) - weight.shape[-1])
        ).view(*weight.shape[:-1], *self.kernel_size)
        prefix = self.func_key if self.func_key == "" else self.func_key + "__"
        return super().forward(x, {f"{prefix}weight": weight, f"{prefix}bias": bias})


class FuncBlockBase(FuncModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layerscale_init: int,
        use_shortcut: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layerscale = (
            FuncLayerScale(self.out_features, layerscale_init)
            if layerscale_init > 0.0
            else None
        )
        self.use_shortcut = use_shortcut
        self.func_params_keys = []

    def extra_repr(self):
        s = "in_features={in_features}, out_features={out_features}, use_shortcut={use_shortcut}"
        return s.format(**self.__dict__)

    def block_forward(self, x, params, mask):
        raise NotImplementedError

    def forward(self, x, params, mask):
        x, shortcut = self.block_forward(x, params, mask), x
        x = self.layerscale(x, params) if self.layerscale is not None else x
        x = (
            x + shortcut
            if self.use_shortcut and self.in_features == self.out_features
            else x
        )
        return x


class FuncConvNeXtBlock(FuncBlockBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        kernel_size: int = 3,
        act: str = "gelu",
        norm: str = "layernorm",
        bias: bool = True,
        layerscale_init: float = 0.0,
    ):
        super().__init__(in_features, out_features, layerscale_init, True)
        self.dconv = FuncConv2D(
            in_features,
            in_features,
            kernel_size,
            groups=in_features,
            padding="same",
            bias=bias,
        )
        self.norm = get_func_norm(norm)(in_features)
        self.fc1 = FuncLinear(in_features, hidden_features, bias=bias)
        self.act = get_func_activation(act)()
        self.fc2 = FuncLinear(hidden_features, out_features, bias=bias)

    def block_forward(self, x, params, mask):
        x = self.dconv(x, params)
        x = self.norm(x, params)
        x = self.fc1(x, params)
        x = self.act(x)
        x = self.fc2(x, params)
        x = mask * x if mask is not None else x
        return x


class FuncLinearBlock(FuncBlockBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act="gelu",
        norm="layernorm",
        bias: bool = True,
        norm_first=False,
    ):
        super().__init__(in_features, out_features, 0.0, False)
        self.norm_first = norm_first
        self.linear = FuncLinear(in_features, out_features, bias=bias)
        self.norm = get_func_norm(norm)(
            in_features if self.norm_first else out_features
        )
        self.act = get_func_activation(act)()

    def block_forward(self, x, params, mask):
        x = (
            self.linear(self.norm(x, params), params)
            if self.norm_first
            else self.norm(self.linear(x, params), params)
        )
        x = self.act(x)
        x = mask * x if mask is not None else x
        return x


class FuncConv2DBlock(FuncBlockBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size=3,
        act="gelu",
        norm="layernorm",
        bias: bool = True,
        norm_first=False,
    ):
        super().__init__(in_features, out_features, 0.0, False)
        self.norm_first = norm_first
        self.conv = FuncConv2D(
            in_features, out_features, kernel_size, padding="same", bias=bias
        )
        self.norm = get_func_norm(norm)(
            in_features if self.norm_first else out_features
        )
        self.act = get_func_activation(act)()

    def block_forward(self, x, params, mask):
        x = (
            self.conv(self.norm(x, params), params)
            if self.norm_first
            else self.norm(self.conv(x, params), params)
        )
        x = self.act(x)
        x = mask * x if mask is not None else x
        return x


class FuncMLPBlock(FuncBlockBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        act: str = "gelu",
        norm: str = "layernorm",
        bias: bool = True,
        layerscale_init: float = 0.0,
    ):
        super().__init__(in_features, out_features, layerscale_init, True)
        self.norm = get_func_norm(norm)(in_features)
        self.fc1 = FuncLinear(in_features, hidden_features, bias=bias)
        self.act = get_func_activation(act)()
        self.fc2 = FuncLinear(hidden_features, out_features, bias=bias)

    def block_forward(self, x, params, mask):
        x = self.norm(x, params)
        x = self.fc1(x, params)
        x = self.act(x)
        x = self.fc2(x, params)
        x = mask * x if mask is not None else x
        return x


class FuncModMixerBlock(FuncBlockBase):
    """
    ModMixer block with:
      - Token mixer: normalization-based token mixing (token-wise IN across tokens, optionally per-window)
      - Progressive mask alpha: x + alpha * (mixer(x) - x)   (alpha=0 => identity, fold-safe)
      - Branch folding remains exact when alpha ~ 0
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        act: str = "gelu",
        norm: str = "layernorm",
        bias: bool = True,
        layerscale_init: float = 0.0,
        # kept for backward-compat (no longer used)
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        num_branches: int = 2,
        branch_depths: Optional[List[int]] = None,
        mask_init: float = 1.0,
        use_fused: bool = False,
        disable_fused_cache: bool = False,
        window_size: Optional[List[int]] = None,
        # new knobs
        token_mixer_eps: float = 1e-5,
        token_mixer_affine: bool = True,
    ):
        super().__init__(in_features, out_features, layerscale_init, True)
        self._fused_cache = None
        self.disable_fused_cache = disable_fused_cache

        self.act = get_func_activation(act)()
        self.norm = get_func_norm(norm)(in_features)

        self.hidden_features = hidden_features
        self.window_size = tuple(window_size) if window_size is not None else None

        self.token_mixer_eps = token_mixer_eps
        self.token_mixer_affine = token_mixer_affine

        self.use_fused = use_fused
        self.distill_mask_override: Optional[float] = None

        self.num_branches = num_branches
        self.branch_depths = branch_depths or [1, 2]
        assert len(self.branch_depths) == self.num_branches

        # Linear branches only (no activations) to make folding exact
        self.branches = nn.ModuleList()
        for depth in self.branch_depths:
            layers = []
            if depth <= 1:
                layers.append(FuncLinear(in_features, out_features, bias=bias))
            else:
                layers.append(FuncLinear(in_features, hidden_features, bias=bias))
                for _ in range(depth - 2):
                    layers.append(FuncLinear(hidden_features, hidden_features, bias=bias))
                layers.append(FuncLinear(hidden_features, out_features, bias=bias))
            self.branches.append(nn.ModuleList(layers))

        # Extra params: token mask + branch coefficients + token-mixer affine (reuse absorb_*)
        self.mask_init = mask_init
        self.func_params_keys = [
            "mix_mask",
            "coeff_w",
            "coeff_b",
            "absorb_scale",  # token-mixer affine gamma [C]
            "absorb_shift",  # token-mixer affine beta  [C]
        ]

    def _get_params(self):
        mask = create_param(torch.tensor(self.mask_init), "modmixer_mask")
        coeff_w = create_param(
            torch.ones(self.num_branches, self.out_features), "modmixer_coeff_w"
        )
        coeff_b = create_param(
            torch.zeros(self.num_branches, self.out_features), "modmixer_coeff_b"
        )
        absorb_scale = create_param(
            torch.ones(self.in_features), "modmixer_absorb_scale"
        )
        absorb_shift = create_param(
            torch.zeros(self.in_features), "modmixer_absorb_shift"
        )
        return {
            "mix_mask": mask,
            "coeff_w": coeff_w,
            "coeff_b": coeff_b,
            "absorb_scale": absorb_scale,
            "absorb_shift": absorb_shift,
        }

    def _reshape_coeff(self, coeff, x):
        shape = (coeff.shape[0],) + (1,) * (x.ndim - 2) + (coeff.shape[-1],)
        return coeff.view(shape)

    def _run_branch(self, branch: nn.ModuleList, x: Tensor, params: dict) -> Tensor:
        out = x
        for layer in branch:
            out = layer(out, params)
        return out

    @staticmethod
    def _fold_branch(branch: nn.ModuleList, params: dict) -> Tuple[Tensor, Tensor]:
        eff_w, eff_b = None, None
        for layer in branch:
            w, b = layer._extract_params(params)
            b = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype) if b is None else b
            if eff_w is None:
                eff_w, eff_b = w, b
            else:
                eff_b = torch.matmul(w, eff_b) + b
                eff_w = torch.matmul(w, eff_w)
        return eff_w, eff_b

    def fold_reparam(self, params: dict) -> Tuple[Tensor, Tensor]:
        mix_mask, coeff_w, coeff_b, absorb_scale, absorb_shift = self._extract_params(params)
        if mix_mask is None:
            raise RuntimeError("mix_mask is None; params dict is missing this block's parameters.")
        if torch.any(mix_mask.abs() > 1e-3):
            raise RuntimeError(
                "fold_reparam should be called only after mix_mask decays to ~0; "
                "otherwise the token mixer is still active."
            )

        fused_w = torch.zeros(
            (self.out_features, self.in_features),
            device=coeff_w.device,
            dtype=coeff_w.dtype,
        )
        fused_b = torch.zeros(
            (self.out_features,), device=coeff_w.device, dtype=coeff_w.dtype
        )

        for i, branch in enumerate(self.branches):
            bw, bb = self._fold_branch(branch, params)
            w_scale = coeff_w[i].unsqueeze(-1)  # [C,1]
            fused_w = fused_w + w_scale * bw
            fused_b = fused_b + coeff_w[i] * bb + coeff_b[i]
        return fused_w, fused_b

    def set_distill_mask(self, value: Optional[float]):
        """
        Optional override for mix_mask during progressive removal.
        None means use learnable mix_mask; 0.0 -> fully bypass mixer; 1.0 -> full mixer.
        """
        self.distill_mask_override = value
        # cache becomes unsafe once alpha changes
        self._fused_cache = None

    def _get_fused_cached(self, params, mix_mask):
        if not self.use_fused or self.training or self.disable_fused_cache:
            return None
        if torch.any(mix_mask.abs() > 1e-3):
            return None
        if self._fused_cache is None:
            self._fused_cache = self.fold_reparam(params)
        return self._fused_cache

    # -------------------------
    # Mask utilities
    # -------------------------
    @staticmethod
    def _canonicalize_token_mask(mask: Optional[Tensor], x: Tensor) -> Optional[Tensor]:
        """
        Make mask a float tensor broadcastable to x[..., C] (without channel dim).
        Accepts:
          - [B,T,H,W] or [B,T,H,W,1]
          - bool or float
        Returns:
          - mask with shape broadcastable to x.shape[:-1]
        """
        if mask is None:
            return None
        m = mask
        if m.dtype == torch.bool:
            m = m.float()
        else:
            m = m.to(dtype=x.dtype)

        # squeeze last dim if it is 1 and matches channel-last convention
        if m.ndim == x.ndim and m.shape[-1] == 1:
            m = m.squeeze(-1)

        # try to broadcast to x spatial shape
        try:
            m = m.expand(x.shape[:-1])
        except Exception:
            # if not directly broadcastable, just keep as-is and fail later loudly on reshape
            pass
        return m

    # -------------------------
    # Token-wise normalization mixer
    # -------------------------
    def _token_in_norm(
        self,
        x_flat: Tensor,              # [B, N, C]
        valid_flat: Optional[Tensor],# [B, N] bool, or None
        gamma: Optional[Tensor],     # [C]
        beta: Optional[Tensor],      # [C]
    ) -> Tensor:
        eps = self.token_mixer_eps
        # compute stats in fp32 for stability
        x_ = x_flat.float() if x_flat.dtype in (torch.float16, torch.bfloat16) else x_flat

        if valid_flat is None:
            mean = x_.mean(dim=1, keepdim=True)
            var = x_.var(dim=1, unbiased=False, keepdim=True)
        else:
            v = valid_flat.float().unsqueeze(-1)  # [B,N,1]
            denom = v.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (x_ * v).sum(dim=1, keepdim=True) / denom
            var = ((x_ - mean) * v).pow(2).sum(dim=1, keepdim=True) / denom

        y = (x_ - mean) / torch.sqrt(var + eps)
        y = y.to(dtype=x_flat.dtype)

        if self.token_mixer_affine:
            if gamma is None:
                gamma = torch.ones(x_flat.shape[-1], device=x_flat.device, dtype=x_flat.dtype)
            if beta is None:
                beta = torch.zeros(x_flat.shape[-1], device=x_flat.device, dtype=x_flat.dtype)
            y = y * gamma.view(1, 1, -1) + beta.view(1, 1, -1)
        return y

    def _apply_token_mixer(self, x_in: Tensor, token_mask: Optional[Tensor],
                           gamma: Optional[Tensor], beta: Optional[Tensor]) -> Tensor:
        """
        Apply normalization token mixer.
        - If window_size is None: flatten all non-channel dims into tokens.
        - If window_size is set: expects x_in == [B,T,H,W,C] and compute stats per window.
        """
        if self.window_size is None:
            B = x_in.shape[0]
            C = x_in.shape[-1]
            x_flat = x_in.reshape(B, -1, C)

            valid_flat = None
            if token_mask is not None:
                m = token_mask.reshape(B, -1)
                valid_flat = (m > 0.5)

            y_flat = self._token_in_norm(x_flat, valid_flat, gamma, beta)
            return y_flat.reshape_as(x_in)

        # windowed: require 5D [B,T,H,W,C]
        if x_in.ndim != 5:
            raise ValueError(f"window_size is set but x.ndim={x_in.ndim}, expected 5D [B,T,H,W,C].")

        Tw, Hw, Ww = self.window_size
        B, T, H, W, C = x_in.shape

        pt = (Tw - T % Tw) % Tw
        ph = (Hw - H % Hw) % Hw
        pw = (Ww - W % Ww) % Ww

        # pad x (channel-first for F.pad)
        x_pad = x_in.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,T,H,W]
        x_pad = F.pad(x_pad, (0, pw, 0, ph, 0, pt))
        x_pad = x_pad.permute(0, 2, 3, 4, 1).contiguous()  # [B,Tp,Hp,Wp,C]
        Tp, Hp, Wp = x_pad.shape[1:4]

        # pad mask
        mask_pad = None
        if token_mask is not None:
            m = token_mask
            # m: [B,T,H,W]
            m = m.unsqueeze(1)  # [B,1,T,H,W]
            m = F.pad(m, (0, pw, 0, ph, 0, pt))
            m = m.squeeze(1)    # [B,Tp,Hp,Wp]
            mask_pad = m

        # partition windows -> [B*nW, Nw, C]
        x_win = x_pad.reshape(
            B, Tp // Tw, Tw, Hp // Hw, Hw, Wp // Ww, Ww, C
        ).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x_win = x_win.reshape(-1, Tw * Hw * Ww, C)

        valid_win = None
        if mask_pad is not None:
            m_win = mask_pad.reshape(
                B, Tp // Tw, Tw, Hp // Hw, Hw, Wp // Ww, Ww
            ).permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            m_win = m_win.reshape(-1, Tw * Hw * Ww)
            valid_win = (m_win > 0.5)

        y_win = self._token_in_norm(x_win, valid_win, gamma, beta)

        # merge windows back
        y = y_win.reshape(
            B, Tp // Tw, Hp // Hw, Wp // Ww, Tw, Hw, Ww, C
        ).permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        y = y.reshape(B, Tp, Hp, Wp, C)

        # crop
        return y[:, :T, :H, :W, :]

    # -------------------------
    # Main forward (block_forward)
    # -------------------------
    def block_forward(self, x: Tensor, params: dict, mask: Optional[Tensor]):
        mix_mask, coeff_w, coeff_b, absorb_scale, absorb_shift = self._extract_params(params)

        if mix_mask is None:
            raise RuntimeError(
                "mix_mask is None. Did you call model.init_func_modules() and "
                "use get_params_recursive() to build the global params dict?"
            )

        # override alpha if requested (progressive schedule)
        if self.distill_mask_override is not None:
            mix_mask = mix_mask * 0 + float(self.distill_mask_override)

        # clamp to [0,1]
        mix_mask = torch.clamp(mix_mask, 0.0, 1.0)

        # canonicalize token validity mask
        token_mask = self._canonicalize_token_mask(mask, x)

        # token mixer (normalization-based)
        tm_out = self._apply_token_mixer(x, token_mask, absorb_scale, absorb_shift)

        # progressive mixing: fold-safe identity when alpha=0
        x_mixed = x + mix_mask * (tm_out - x)

        # norm + branches
        x_norm = self.norm(x_mixed, params)

        fused = self._get_fused_cached(params, mix_mask)
        if fused is not None:
            fused_w, fused_b = fused
            out = F.linear(x_norm, fused_w, fused_b)
        elif self.use_fused and torch.all(mix_mask.abs() <= 1e-3):
            fused_w, fused_b = self.fold_reparam(params)
            out = F.linear(x_norm, fused_w, fused_b)
        else:
            branch_outs = [self._run_branch(b, x_norm, params) for b in self.branches]
            coeff_w_b = self._reshape_coeff(coeff_w, x_norm)
            coeff_b_b = self._reshape_coeff(coeff_b, x_norm)
            stacked = torch.stack(branch_outs, dim=0)  # [num_branches, B, ..., C]
            out = (coeff_w_b * stacked).sum(dim=0) + coeff_b_b.sum(dim=0)

        out = self.act(out)

        # apply token validity mask to output (broadcast-safe)
        if token_mask is not None:
            m = token_mask.to(dtype=out.dtype)
            if m.ndim == out.ndim - 1:
                m = m.unsqueeze(-1)
            out = out * m

        return out

        
class FuncConvNeXtInterleavedBlock(FuncBlockBase):
    """
    ConvNeXt-style block for video that interleaves spatial 2D depthwise conv
    with a temporal 1D conv before the pointwise MLP.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        kernel_size_2d: int = 3,
        kernel_size_1d: int = 3,
        act: str = "gelu",
        norm: str = "layernorm",
        bias: bool = True,
        layerscale_init: float = 0.0,
        temporal_causal: bool = False,
    ):
        super().__init__(in_features, out_features, layerscale_init, True)
        # spatial depthwise conv over H,W per frame
        self.dconv2d = FuncConv2D(
            in_features,
            in_features,
            kernel_size_2d,
            groups=in_features,
            padding="same",
            bias=bias,
        )
        # temporal conv over T to mix frames
        self.tconv1d = FuncConv1D(
            in_features,
            in_features,
            kernel_size=kernel_size_1d,
            causal=temporal_causal,
            bias=bias,
            layout="ncthw",
        )
        self.norm = get_func_norm(norm)(in_features)
        self.fc1 = FuncLinear(in_features, hidden_features, bias=bias)
        self.act = get_func_activation(act)()
        self.fc2 = FuncLinear(hidden_features, out_features, bias=bias)

    def block_forward(self, x, params, mask):
        # Accept NCTHW or NTHWC; normalize to channel-last for spatial conv.
        is_nthwc = x.dim() == 5 and x.shape[-1] == self.in_features
        if is_nthwc:
            x_nthwc = x
        elif x.dim() == 5 and x.shape[1] == self.in_features:
            x_nthwc = x.permute(0, 2, 3, 4, 1)
        else:
            raise ValueError(
                f"Unexpected input shape {x.shape} for ConvNeXtInterleavedBlock."
            )

        # 2D depthwise conv per frame (operates on NTHWC)
        x_nthwc = self.dconv2d(x_nthwc, params)

        # Temporal 1D conv over T (expects NCTHW)
        x = x_nthwc.permute(0, 4, 1, 2, 3)
        x = self.tconv1d(x, params)

        # Channel-last for norm and MLP
        x = x.permute(0, 2, 3, 4, 1)  # N, T, H, W, C
        x = self.norm(x, params)
        x = self.fc1(x, params)
        x = self.act(x)
        x = self.fc2(x, params)
        x = x.permute(0, 4, 1, 2, 3)  # back to NCTHW

        if mask is not None:
            mask_ncthw = mask.permute(0, 4, 1, 2, 3) if is_nthwc else mask
            x = mask_ncthw * x
        if is_nthwc:
            x = x.permute(0, 2, 3, 4, 1)
        return x


class FuncConvNeXtEncoderStage(FuncModule):
    """
    Interleaving temporal downsample (1D), spatial downsample (2D),
    followed by stacked interleaved ConvNeXt blocks. Expects NTHWC input.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        t_stride: int,
        t_kernel: Optional[int],
        t_padding: Optional[Tuple[int, int]],
        s_stride: int,
        depth: int,
        hidden_mult: int,
    ):
        super().__init__()
        self.t_conv = FuncConv1D(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=t_kernel if t_kernel is not None else t_stride,
            stride=t_stride,
            causal=False,
            pad_mode="replicate",
            layout="nthwc",
            padding=t_padding,
        )
        self.s_conv = FuncConv2D(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=s_stride,
            stride=s_stride,
            padding=0,
        )
        self.blocks = nn.ModuleList(
            [
                FuncConvNeXtInterleavedBlock(
                    in_features=out_dim,
                    out_features=out_dim,
                    hidden_features=out_dim * hidden_mult,
                    kernel_size_2d=3,
                    kernel_size_1d=3,
                    temporal_causal=False,
                )
                for _ in range(depth)
            ]
        )

    def _get_params(self):
        # Parameters are owned by child FuncModules; nothing to declare at this level.
        return {}

    def forward(
        self, x: torch.Tensor, params: dict, mask: Optional[torch.Tensor] = None
    ):
        # x: [B, T, H, W, C] (NTHWC)
        x = self.t_conv(x, params)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # back to NTHWC
        x = self.s_conv(x, params)
        for block in self.blocks:
            x = block(x, params, mask)
        return x


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def exists(v):
    return v is not None


def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))


class FuncRMSNorm(FuncModule):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.use_bias = bias
        self.func_params_keys = ["gamma", "beta"]

    def extra_repr(self):
        return f"dim={self.dim}, eps={self.eps}, bias={self.use_bias}"

    def _get_params(self):
        gamma = create_param(torch.ones(self.dim), "rmsnorm_gamma")
        beta = create_param(torch.zeros(self.dim), "rmsnorm_beta") if self.use_bias else None
        return {"gamma": gamma, "beta": beta}

    def forward(self, x: Tensor, params: dict) -> Tensor:
        gamma, beta = self._extract_params(params)
        # RMS normalization along last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        x = x * gamma
        if beta is not None:
            x = x + beta
        return x


class FuncAdaptiveRMSNorm(FuncModule):

    def __init__(
        self,
        dim: int,
        dim_cond: int,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_cond = dim_cond
        self.eps = eps
        self.use_bias = bias

        self.to_gamma = FuncLinear(dim_cond, dim, bias=True)
        self.to_beta = FuncLinear(dim_cond, dim, bias=True) if bias else None
        self.func_params_keys = []

    def extra_repr(self):
        return f"dim={self.dim}, dim_cond={self.dim_cond}, eps={self.eps}, bias={self.use_bias}"

    def _get_params(self):
        return {}

    def forward(self, x: Tensor, params: dict, cond: Tensor) -> Tensor:
        """
        Args:
            x: [B, ..., dim] input tensor
            params: parameter dictionary
            cond: [B, dim_cond] conditioning vector
        """
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms

        # Generate scale from conditioning
        gamma = self.to_gamma(cond, params)  # [B, dim]
        # Broadcast gamma to match x shape
        for _ in range(x.dim() - 2):
            gamma = gamma.unsqueeze(1)
        x = x * (1 + gamma)  # scale centered at 1

        if self.to_beta is not None:
            beta = self.to_beta(cond, params)
            for _ in range(x.dim() - 2):
                beta = beta.unsqueeze(1)
            x = x + beta

        return x


class Attend(nn.Module):
    # Minimum sequence length for Flash Attention (some configs fail below this)
    FLASH_MIN_SEQ_LEN = 16

    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.causal = causal
        self.scale = scale

    def _manual_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> Tensor:
        """Manual attention implementation for short sequences."""
        # q, k, v: [B, H, N, D]
        scale = self.scale if self.scale is not None else (q.shape[-1] ** -0.5)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]
        
        # Apply causal mask if needed
        if is_causal:
            seq_len = q.shape[-2]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn = attn + attn_mask
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=dropout_p)
        
        # Apply to values
        out = torch.matmul(attn, v)  # [B, H, N, D]
        return out

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        dropout_p = self.dropout if self.training else 0.0

        # Handle causal masking
        is_causal = self.causal and mask is None

        # Convert boolean mask to attention mask format if needed
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                # Convert to additive mask: True = attend, False = mask out
                attn_mask = torch.zeros_like(mask, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
            else:
                attn_mask = mask

        # Check sequence length - Flash Attention may fail for very short sequences
        seq_len = q.shape[-2]

        if seq_len >= self.FLASH_MIN_SEQ_LEN:
            # Use default behavior (Flash Attention if available)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self.scale,
            )
        else:
            # Use manual attention for short sequences to avoid Flash Attention issues
            out = self._manual_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        return out


class FuncAttention(FuncModule):

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        # Projections
        self.to_qkv = FuncLinear(dim, self.dim_inner * 3, bias=bias)
        self.to_out = FuncLinear(self.dim_inner, dim, bias=bias)
        self.norm = FuncRMSNorm(dim)

        # Attention
        self.attend = Attend(dropout=dropout, causal=causal, scale=self.scale)

        self.func_params_keys = []

    def extra_repr(self):
        return f"dim={self.dim}, heads={self.heads}, dim_head={self.dim_head}, causal={self.causal}"

    def forward(
        self,
        x: Tensor,
        params: dict,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: [B, N, C] input tensor
            params: parameter dictionary
            mask: optional attention mask

        Returns:
            [B, N, C] output tensor
        """
        B, N, C = x.shape

        # Pre-norm
        x_norm = self.norm(x, params)

        # QKV projection
        qkv = self.to_qkv(x_norm, params)  # [B, N, 3 * dim_inner]
        qkv = qkv.reshape(B, N, 3, self.heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        out = self.attend(q, k, v, mask=mask)  # [B, H, N, D]

        # Reshape and project out
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim_inner)  # [B, N, dim_inner]
        out = self.to_out(out, params)

        return out


class FuncTimeAttention(FuncModule):

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.attn = FuncAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            causal=causal,
            bias=bias,
        )
        self.func_params_keys = []

    def _get_params(self):
        return {}

    def forward(self, x: Tensor, params: dict, mask: Optional[Tensor] = None) -> Tensor:
        is_nthwc = x.dim() == 5 and x.shape[-1] == self.dim
        if is_nthwc:
            x = x.permute(0, 4, 1, 2, 3)  # NTHWC -> NCTHW

        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b h w) t c")
        x = self.attn(x, params, mask)
        x = rearrange(x, "(b h w) t c -> b c t h w", b=B, h=H, w=W)

        if is_nthwc:
            x = x.permute(0, 2, 3, 4, 1)  # NCTHW -> NTHWC
        return x

        
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def get_func_block(type, **kwargs):
    if type == "identity":
        return FuncIdentity()
    elif type == "linear_stem":
        return FuncLinearBlock(
            in_features=kwargs["C1"],
            out_features=kwargs["C2"],
            act=kwargs["act"],
            norm=kwargs["norm"],
            norm_first=False,
            bias=kwargs["bias"],
        )
    elif type == "conv_stem":
        return FuncConv2DBlock(
            in_features=kwargs["C1"],
            out_features=kwargs["C2"],
            kernel_size=kwargs["kernel_size"],
            act=kwargs["act"],
            norm=kwargs["norm"],
            norm_first=False,
            bias=kwargs["bias"],
        )
    elif type == "linear_head":
        return FuncLinearBlock(
            in_features=kwargs["C1"],
            out_features=kwargs["C2"],
            act=kwargs["act"],
            norm=kwargs["norm"],
            norm_first=True,
            bias=kwargs["bias"],
        )
    elif type == "mlp":
        return FuncMLPBlock(
            in_features=kwargs["C1"],
            out_features=kwargs["C2"],
            hidden_features=kwargs["C_hidden"],
            act=kwargs["act"],
            norm=kwargs["norm"],
            bias=kwargs["bias"],
            layerscale_init=kwargs["layerscale"],
        )
    elif type == "convnext":
        return FuncConvNeXtBlock(
            in_features=kwargs["C1"],
            out_features=kwargs["C2"],
            hidden_features=kwargs["C_hidden"],
            kernel_size=kwargs["kernel_size"],
            act=kwargs["act"],
            norm=kwargs["norm"],
            bias=kwargs["bias"],
            layerscale_init=kwargs["layerscale"],
        )
    elif type == "modmixer":
        return FuncModMixerBlock(
            in_features=kwargs["C1"],
            out_features=kwargs["C2"],
            hidden_features=kwargs["C_hidden"],
            act=kwargs["act"],
            norm=kwargs["norm"],
            bias=kwargs["bias"],
            layerscale_init=kwargs["layerscale"],
            attn_heads=kwargs.get("attn_heads", 4),
            attn_dim_head=kwargs.get("attn_dim_head", 32),
            num_branches=kwargs.get("num_branches", 2),
            branch_depths=kwargs.get("branch_depths", None),
            mask_init=kwargs.get("mask_init", 1.0),
            use_fused=kwargs.get("use_fused", False),
            disable_fused_cache=kwargs.get("disable_fused_cache", False),
            window_size=kwargs.get("window_size", None),
        )
    elif type == "none":
        return
    else:
        raise NotImplementedError


def _run_modules(module_list, x=None, params={}, mask=None):
    for m in module_list:
        if isinstance(m, FuncBlockBase):
            # For the normal blocks
            x = m(x, params, mask)
        elif isinstance(m, FuncConstant):
            # For the constant layers
            x = m(params)
        elif isinstance(m, FuncModule):
            # For the normal layers
            if mask is not None:
                x = m(x, params) * mask
            else:
                x = m(x, params)
        else:
            # General PyTorch layer
            x = m(x)
    return x


def run_modules(module_list, x=None, params={}, mask=None):
    return _run_modules(module_list, x, params=params, mask=mask)
