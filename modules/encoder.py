from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.ops.grid_encoding import BatchGridTrilinear3D
from modules.ops.layers import (
    FunctionalAdapter,
    FuncConvNeXtEncoderStage,
    FuncConvNeXtInterleavedBlock,
    ResidualBottleneckBlock, AttentionBlock
)


# ----------------------------------------------------------------
# utility functions

def compute_min_chunk_size(
    temporal_strides: Sequence[int],
    spatial_strides: Sequence[int],
) -> Tuple[int, int, int]:
    """
    compute minimum video chunk size based on encoder/decoder downsampling.
    returns (min_t, min_h, min_w).
    """
    min_t = 1
    for s in temporal_strides:
        min_t *= s
    min_hw = 1
    for s in spatial_strides:
        min_hw *= s
    return (min_t, min_hw, min_hw)


def validate_encoder_decoder_config(
    encoder_cfg: dict,
    decoder_cfg: dict,
    logger=None,
) -> bool:
    """
    validate that encoder strides match decoder scales.
    returns true if valid.
    """
    enc_t_strides = encoder_cfg.get("temporal_strides", [])
    enc_s_strides = encoder_cfg.get("spatial_strides", [])
    dec_scales_t = decoder_cfg.get("scales_t", [])
    dec_scales_hw = decoder_cfg.get("scales_hw", [])

    valid = True
    if list(enc_t_strides) != list(dec_scales_t):
        if logger:
            logger.warning(
                f"encoder temporal_strides {enc_t_strides} != decoder scales_t {dec_scales_t}"
            )
        valid = False
    if list(enc_s_strides) != list(dec_scales_hw):
        if logger:
            logger.warning(
                f"encoder spatial_strides {enc_s_strides} != decoder scales_hw {dec_scales_hw}"
            )
        valid = False

    enc_levels = encoder_cfg.get("num_levels", len(enc_t_strides))
    dec_stages = len(decoder_cfg.get("channels", []))
    if enc_levels != dec_stages:
        if logger:
            logger.warning(
                f"encoder num_levels {enc_levels} != decoder stages {dec_stages}"
            )
        valid = False

    return valid


def compute_encoder_channels_from_grids(
    base_grid_size: List[int],
    base_grid_level: int,
    base_grid_level_scale: List[float],
    local_grid_size: List[int],
    local_grid_level: int,
    local_grid_level_scale: Optional[List[float]] = None,
    local_grid_depth_scale: Optional[List[float]] = None,
    num_local_stages: int = 4,
) -> List[int]:
    """
    compute encoder output channels to match grid channels.

    encoder outputs:
    - levels 0..n-2: local grids (sum of local grid channels per stage)
    - level n-1: base grids (sum of base grid channels)

    args:
        base_grid_size: [T, H, W, C] for base grid
        base_grid_level: number of base grid levels
        base_grid_level_scale: scale factors [t, h, w, c]
        local_grid_size: [T, C] or [T, H, W, C] for local grid
        local_grid_level: number of local grid levels per stage
        local_grid_level_scale: scale factors for local grid [t, c] or [t, h, w, c]
                                if None, defaults to [2.0, 0.5]
        local_grid_depth_scale: scale factors per decoder stage depth [t, c]
                                if None, defaults to [1.0, 0.5]
                                C_base at stage d = local_c * depth_scale[-1]^d
        num_local_stages: number of decoder stages with local grids

    returns:
        list of channels for each encoder level
    """
    ### compute base grid total channels
    base_c = base_grid_size[-1]
    base_c_scale = base_grid_level_scale[-1]
    base_total_c = sum(
        max(int(base_c / (base_c_scale ** i)), 1) for i in range(base_grid_level)
    )

    ### compute local grid total channels per stage
    # local grid channel scale is at index -1 (last element)
    # formula: C_i = C / (scale ^ i), so scale < 1 means channels increase
    local_c_base = local_grid_size[-1]
    if local_grid_level_scale is None:
        local_grid_level_scale = [2.0, 0.5]  # default: [T_scale, C_scale]
    if local_grid_depth_scale is None:
        local_grid_depth_scale = [1.0, 0.5]  # default: [T_depth_scale, C_depth_scale]

    local_c_scale = local_grid_level_scale[-1]  # channel scale within levels
    local_c_depth_scale = local_grid_depth_scale[-1]  # channel scale across stages

    ### compute channels for each decoder stage (depth)
    local_channels_per_stage = []
    for depth in range(num_local_stages):
        # base channel for this stage (scaled by depth)
        stage_c_base = max(int(local_c_base * (local_c_depth_scale ** depth)), 1)
        # sum of all grid levels within this stage
        stage_total_c = sum(
            max(int(stage_c_base / (local_c_scale ** i)), 1)
            for i in range(local_grid_level)
        )
        local_channels_per_stage.append(stage_total_c)

    ### encoder channels: [local_c per stage] + [base_c]
    channels = local_channels_per_stage + [base_total_c]
    return channels


def elic_conv(in_channels, out_channels, kernel_size=5, stride=2):
    """ELIC 2D conv helper (Conv2d with same padding)."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

@dataclass
class EncoderConfig:
    in_channels: int = 3
    base_channels: int = 192
    num_levels: int = 5  # 4 for local grids + 1 for base grid
    depth_per_level: int = 3
    channels: Optional[Sequence[int]] = (192, 192, 192, 192, 192)
    temporal_strides: Tuple[int, ...] = (1, 1, 2, 2, 1)    # temporal downsampling factor
    temporal_kernels: Optional[Tuple[int, ...]] = None  # if None, defaults to temporal_strides
    temporal_padding: Tuple[str, ...] = ("same", "same", "same", "same", "same")  # "same"/"valid"/int or (l,r) per level
    spatial_strides: Tuple[int, ...] = (3, 2, 2, 2, 1)
    hidden_mult: int = 4
    grid_align_corners: bool = True
    train_encoder: bool = True
    use_param_grids: bool = False
    init_param_grids_from_encoder: bool = False
    param_grid_base_size: Optional[Tuple[int, int, int]] = None
    param_grid_level_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0)
    param_init_scale: float = 1e-3

    def get_min_chunk_size(self) -> Tuple[int, int, int]:
        """get minimum input chunk size for this encoder config."""
        return compute_min_chunk_size(self.temporal_strides, self.spatial_strides)


class ParametricGridPyramid(nn.Module):
    """
    Hierarchical learnable grids with optional initialization from a trained encoder.
    Uses BatchGridTrilinear3D (modules/ops/grid_encoding.py) to resize grids to
    the target latent resolutions expected by the decoder.
    """

    def __init__(
        self,
        channels: Sequence[int],
        base_size: Optional[Tuple[int, int, int]],
        level_scale: Tuple[float, float, float],
        init_scale: float,
        align_corners: bool,
    ):
        super().__init__()
        self.channels = list(channels)
        self.base_size = base_size
        self.level_scale = level_scale
        self.init_scale = init_scale
        self.align_corners = align_corners
        self.grids: nn.ParameterList = nn.ParameterList()
        self.param_shapes: Optional[List[Tuple[int, int, int]]] = None

    def _compute_param_shape(self, target_shape: Tuple[int, int, int], level: int) -> Tuple[int, int, int]:
        if self.base_size is None:
            return target_shape
        t, h, w = self.base_size
        st, sh, sw = self.level_scale
        t_l = max(int(round(t / (st ** level))), 1)
        h_l = max(int(round(h / (sh ** level))), 1)
        w_l = max(int(round(w / (sw ** level))), 1)
        return (t_l, h_l, w_l)

    def _ensure_params(
        self,
        target_shapes: List[Tuple[int, int, int, int]],
        init_from: Optional[List[torch.Tensor]] = None,
    ):
        if self.param_shapes is not None:
            return
        device = init_from[0].device if init_from else None
        dtype = init_from[0].dtype if init_from else None
        self.param_shapes = []
        for level, (c, t, h, w) in enumerate(target_shapes):
            pt, ph, pw = self._compute_param_shape((t, h, w), level)
            self.param_shapes.append((pt, ph, pw))
            param = torch.zeros(pt, ph, pw, c, device=device, dtype=dtype)
            torch.nn.init.uniform_(param, -self.init_scale, self.init_scale)
            if init_from is not None:
                encoded = init_from[level].detach()  # [B, C, T, H, W]
                encoded = encoded.mean(dim=0, keepdim=True)  # average across batch
                pooled = F.adaptive_avg_pool3d(encoded, (pt, ph, pw)).squeeze(0)  # [C, pt, ph, pw]
                param = pooled.permute(1, 2, 3, 0).contiguous()  # [pt, ph, pw, C]
            self.grids.append(nn.Parameter(param))

    def forward(self, target_shapes: List[Tuple[int, int, int, int]], batch_size: int) -> List[torch.Tensor]:
        """
        Args:
            target_shapes: list of tuples (C, T, H, W) for each level.
            batch_size: number of videos in the batch.
        Returns:
            List of grids [B, C_l, T_l, H_l, W_l]
        """
        if self.param_shapes is None:
            self._ensure_params(target_shapes)

        outputs: List[torch.Tensor] = []
        for level, param in enumerate(self.grids):
            c_out, t_out, h_out, w_out = target_shapes[level]
            if (param.shape[0], param.shape[1], param.shape[2]) == (t_out, h_out, w_out):
                upsampler = BatchGridTrilinear3D((t_out, h_out, w_out), align_corners=self.align_corners)
                grid_thwc = upsampler(param)  # [T, H, W, C]
            else:
                grid_thwc = F.interpolate(
                    param.permute(3, 0, 1, 2).unsqueeze(0),  # [1, C, T, H, W]
                    size=(t_out, h_out, w_out),
                    mode="trilinear",
                    align_corners=self.align_corners,
                ).squeeze(0).permute(1, 2, 3, 0)
            grid = grid_thwc.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
            grid = grid.expand(batch_size, -1, -1, -1, -1).contiguous()
            outputs.append(grid)
        return outputs

    def initialize_from_encoder(self, target_shapes: List[Tuple[int, int, int, int]], encoder_grids: List[torch.Tensor]):
        """Populate grid parameters from a pretrained encoder once."""
        if self.param_shapes is not None:
            return
        self._ensure_params(target_shapes, init_from=encoder_grids)


class LatentGridEncoderBase(nn.Module):
    """
    Base class for hierarchical video encoders that emit latent grids.
    Handles optional parametric grids and shape bookkeeping.
    Subclasses must implement `_encode` and `_expected_shapes`.

    encoder outputs n levels:
      - level 0 to n-2: local grids for decoder stages 0 to n-2
      - level n-1 (last): base grid for decoder input
    """

    def __init__(
        self,
        channels: Sequence[int],
        align_corners: bool = True,
        use_param_grids: bool = False,
        param_grid_base_size: Optional[Tuple[int, int, int]] = None,
        param_grid_level_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        param_init_scale: float = 1e-3,
        train_encoder: bool = True,
        init_param_grids_from_encoder: bool = False,
    ):
        super().__init__()
        self.channels = list(channels)
        self.align_corners = align_corners
        self.train_encoder = train_encoder
        self.init_param_grids_from_encoder = init_param_grids_from_encoder

        self.param_grids = (
            ParametricGridPyramid(
                channels=self.channels,
                base_size=param_grid_base_size,
                level_scale=param_grid_level_scale,
                init_scale=param_init_scale,
                align_corners=align_corners,
            )
            if use_param_grids
            else None
        )

    def get_base_grid(self, grids: List[torch.Tensor]) -> torch.Tensor:
        """get the base grid (last level output) from encoder outputs."""
        return grids[-1]

    def get_local_grids(self, grids: List[torch.Tensor]) -> List[torch.Tensor]:
        """get the local grids (all but last level) from encoder outputs."""
        return grids[:-1]

    def split_grids(self, grids: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """split encoder outputs into base grid and local grids."""
        return self.get_base_grid(grids), self.get_local_grids(grids)

    @staticmethod
    def _conv_out_len(length: int, kernel: int, stride: int, padding: Tuple[int, int]) -> int:
        """1D conv output length (dilation=1)."""
        left, right = padding
        return max((length + left + right - kernel) // stride + 1, 1)

    def _expected_shapes(self, video_shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
        raise NotImplementedError

    def _encode(self, video: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def forward(self, video: torch.Tensor) -> List[torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {video.shape}.")

        encoder_grids: List[torch.Tensor] = []
        if self.train_encoder or self.init_param_grids_from_encoder:
            encoder_grids = self._encode(video)
            if len(encoder_grids) != len(self.channels):
                raise RuntimeError(
                    f"Encoder returned {len(encoder_grids)} grids, expected {len(self.channels)}."
                )

        param_outputs: Optional[List[torch.Tensor]] = None
        if self.param_grids is not None:
            expected_shapes = self._expected_shapes(video.shape)
            if encoder_grids:
                target_shapes = [
                    (c, g.shape[2], g.shape[3], g.shape[4])
                    for c, g in zip(self.channels, encoder_grids)
                ]
            else:
                target_shapes = expected_shapes

            if self.init_param_grids_from_encoder and encoder_grids:
                self.param_grids.initialize_from_encoder(target_shapes, encoder_grids)
            param_outputs = self.param_grids(target_shapes, batch_size=video.shape[0])
            if not self.train_encoder:
                return param_outputs

        if encoder_grids:
            return encoder_grids
        if param_outputs is not None:
            return param_outputs
        raise RuntimeError("Encoder produced no grids. Check configuration flags.")


class ConvNeXtHierarchicalEncoder(LatentGridEncoderBase):
    """
    Hierarchical video encoder:
      - interleaved ConvNeXt (2D depthwise + 1D temporal) blocks per level
      - produces latent grids consumed by the decoder (B, C_l, T_l, H_l, W_l)
      - optional parametric grid pyramid for grid-only overfitting
    """

    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        self.cfg = cfg
        channels = (
            list(cfg.channels)
            if cfg.channels is not None
            else [cfg.base_channels * (2 ** i) for i in range(cfg.num_levels)]
        )
        if len(channels) != cfg.num_levels:
            raise ValueError(f"channels length ({len(channels)}) must match num_levels ({cfg.num_levels}).")
        super().__init__(
            channels=channels,
            align_corners=cfg.grid_align_corners,
            use_param_grids=cfg.use_param_grids,
            param_grid_base_size=cfg.param_grid_base_size,
            param_grid_level_scale=cfg.param_grid_level_scale,
            param_init_scale=cfg.param_init_scale,
            train_encoder=cfg.train_encoder,
            init_param_grids_from_encoder=cfg.init_param_grids_from_encoder,
        )
        self.stages = nn.ModuleList()
        self.t_kernels: List[int] = []
        self.t_paddings: List[Tuple[int, int]] = []
        in_ch = cfg.in_channels
        for level in range(cfg.num_levels):
            t_stride = cfg.temporal_strides[level] if level < len(cfg.temporal_strides) else 1
            t_kernel = (
                cfg.temporal_kernels[level]
                if cfg.temporal_kernels is not None and level < len(cfg.temporal_kernels)
                else t_stride
            )
            t_pad_spec = cfg.temporal_padding[level] if level < len(cfg.temporal_padding) else "same"
            t_padding = self._resolve_temporal_padding(t_pad_spec, t_kernel)
            self.t_kernels.append(t_kernel)
            self.t_paddings.append(t_padding)
            s_stride = cfg.spatial_strides[level] if level < len(cfg.spatial_strides) else 1
            out_ch = self.channels[level]
            stage = FuncConvNeXtEncoderStage(
                in_dim=in_ch,
                out_dim=out_ch,
                t_stride=t_stride,
                t_kernel=t_kernel,
                t_padding=t_padding,
                s_stride=s_stride,
                depth=cfg.depth_per_level,
                hidden_mult=cfg.hidden_mult,
            )
            self.stages.append(FunctionalAdapter(stage))
            in_ch = out_ch

    @staticmethod
    def _resolve_temporal_padding(pad_spec, kernel: int) -> Tuple[int, int]:
        """Normalize padding spec to a (left, right) tuple for temporal conv."""
        if pad_spec is None:
            pad_spec = "same"
        if isinstance(pad_spec, str):
            if pad_spec.lower() == "same":
                total = max(kernel - 1, 0)
                left = total // 2
                right = total - left
                return (left, right)
            if pad_spec.lower() == "valid":
                return (0, 0)
            raise ValueError(f"Unsupported temporal_padding value: {pad_spec}")
        if isinstance(pad_spec, int):
            return (pad_spec, pad_spec)
        if isinstance(pad_spec, tuple) and len(pad_spec) == 2:
            return pad_spec
        raise ValueError(f"Unsupported temporal_padding value: {pad_spec}")

    def _expected_shapes(self, video_shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
        """Compute (C, T, H, W) per level from input shape and stride schedule."""
        _, _, T, H, W = video_shape
        shapes = []
        t, h, w = T, H, W
        for level, c in enumerate(self.channels):
            t_stride = self.cfg.temporal_strides[level] if level < len(self.cfg.temporal_strides) else 1
            s_stride = self.cfg.spatial_strides[level] if level < len(self.cfg.spatial_strides) else 1
            t_kernel = self.t_kernels[level] if level < len(self.t_kernels) else t_stride
            t_padding = self.t_paddings[level] if level < len(self.t_paddings) else (0, 0)
            t = self._conv_out_len(t, t_kernel, t_stride, t_padding)
            # spatial conv uses kernel=s_stride and padding=0
            h = self._conv_out_len(h, s_stride, s_stride, (0, 0))
            w = self._conv_out_len(w, s_stride, s_stride, (0, 0))
            shapes.append((c, t, h, w))
        return shapes

    def _encode(self, video: torch.Tensor) -> List[torch.Tensor]:
        """Run ConvNeXt hierarchy and return grids (B, C, T, H, W)."""
        x = video.permute(0, 2, 3, 4, 1).contiguous()  # B, T, H, W, C
        outputs: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            # keep grids channel-first for the decoder
            outputs.append(x.permute(0, 4, 1, 2, 3).contiguous())
        return outputs

    def forward(self, video: torch.Tensor) -> List[torch.Tensor]:
        return super().forward(video)

    def log_structure(self, print_fn=print):
        """Pretty-print encoder structure and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_fn("=== Encoder structure ===")
        print_fn(self)
        print_fn(f"\nEncoder parameters: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")


class ELICTemporalConvNeXtEncoder(LatentGridEncoderBase):
    """
    A modification to the spatialtemporal 3D encoder used in PNVC and GIViC,
    by inflating the ELIC spatial encoder with 1D temporal convolutional and attention, 
    or ConvNeXT-based layers
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        temporal_kernel: int = 3,
        temporal_hidden_mult: int = 4,
        temporal_causal: bool = False,
        temporal_layer_type: str = "convnext",
        temporal_attn_heads: int = 8,
        temporal_attn_dim_head: int = 64,
        temporal_attn_dropout: float = 0.0,
        temporal_strides: Tuple[int, ...] = (1, 1, 1, 1),  # temporal downsample per level
        return_all_levels: bool = True,
        use_param_grids: bool = False,
        train_encoder: bool = True,
        init_param_grids_from_encoder: bool = False,
        param_grid_base_size: Optional[Tuple[int, int, int]] = None,
        param_grid_level_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        param_init_scale: float = 1e-3,
        grid_align_corners: bool = True,
    ):
        channels = [N, N, N, M]
        super().__init__(
            channels=channels,
            align_corners=grid_align_corners,
            use_param_grids=use_param_grids,
            param_grid_base_size=param_grid_base_size,
            param_grid_level_scale=param_grid_level_scale,
            param_init_scale=param_init_scale,
            train_encoder=train_encoder,
            init_param_grids_from_encoder=init_param_grids_from_encoder,
        )

        # Validate temporal layer type
        if temporal_layer_type not in TEMPORAL_LAYER_TYPES:
            raise ValueError(
                f"temporal_layer_type must be one of {TEMPORAL_LAYER_TYPES}, "
                f"got {temporal_layer_type}"
            )
        self.temporal_layer_type = temporal_layer_type
        self.temporal_strides = list(temporal_strides)

        # Spatial ELIC's analysis transform
        # g_a indices:
        #   0: elic_conv(3->N)
        #   1-3: ResidualBottleneckBlock(N) x3
        #   4: elic_conv(N->N)
        #   5-7: ResidualBottleneckBlock(N) x3
        #   8: AttentionBlock(N)
        #   9: elic_conv(N->N)
        #   10-12: ResidualBottleneckBlock(N) x3
        #   13: elic_conv(N->M)
        #   14: AttentionBlock(M)
        self.g_a = nn.Sequential(
            elic_conv(3, N, kernel_size=5, stride=2),       # 0
            ResidualBottleneckBlock(N),                      # 1
            ResidualBottleneckBlock(N),                      # 2
            ResidualBottleneckBlock(N),                      # 3
            elic_conv(N, N, kernel_size=5, stride=2),       # 4
            ResidualBottleneckBlock(N),                      # 5
            ResidualBottleneckBlock(N),                      # 6
            ResidualBottleneckBlock(N),                      # 7
            AttentionBlock(N),                               # 8
            elic_conv(N, N, kernel_size=5, stride=2),       # 9
            ResidualBottleneckBlock(N),                      # 10
            ResidualBottleneckBlock(N),                      # 11
            ResidualBottleneckBlock(N),                      # 12
            elic_conv(N, M, kernel_size=5, stride=2),       # 13
            AttentionBlock(M),                               # 14
        )

        # Temporal blocks inserted AFTER each spatial downsample stage
        # Insert after: idx 3 (level 0), idx 8 (level 1), idx 12 (level 2), idx 14 (level 3)
        temporal_channels = channels
        self.temporal_insert_indices = [3, 8, 12, 14]  # positions in g_a AFTER which to insert
        
        # Build temporal blocks with optional temporal downsampling via stride
        self.temporal_blocks = nn.ModuleList()
        for level, c in enumerate(temporal_channels):
            t_stride = self.temporal_strides[level] if level < len(self.temporal_strides) else 1
            if temporal_layer_type == "convnext":
                # Use FuncConvNeXtInterleavedBlock with temporal stride
                block = FunctionalAdapter(
                    FuncConvNeXtInterleavedBlock(
                        in_features=c,
                        out_features=c,
                        hidden_features=c * temporal_hidden_mult,
                        kernel_size_2d=3,
                        kernel_size_1d=temporal_kernel,
                        temporal_causal=temporal_causal,
                    )
                )
            else:  # conv_attn
                block = FunctionalAdapter(
                    FuncConvAttnBlock(
                        in_features=c,
                        out_features=c,
                        hidden_features=c * temporal_hidden_mult,
                        kernel_size=temporal_kernel,
                        dim_head=temporal_attn_dim_head,
                        heads=temporal_attn_heads,
                        dropout=temporal_attn_dropout,
                        temporal_causal=temporal_causal,
                    )
                )
            self.temporal_blocks.append(block)
        
        # Temporal downsample convs (only for levels with stride > 1)
        self.temporal_downsamples = nn.ModuleList()
        for level, c in enumerate(temporal_channels):
            t_stride = self.temporal_strides[level] if level < len(self.temporal_strides) else 1
            if t_stride > 1:
                self.temporal_downsamples.append(
                    FunctionalAdapter(
                        FuncConv1D(
                            in_channels=c,
                            out_channels=c,
                            kernel_size=t_stride,
                            stride=t_stride,
                            causal=temporal_causal,
                            layout="ncthw",
                        )
                    )
                )
            else:
                self.temporal_downsamples.append(None)
        
        self.return_all_levels = return_all_levels
        self.spatial_kernels = [5, 5, 5, 5]
        self.spatial_strides = [2, 2, 2, 2]
        self.spatial_paddings: List[Tuple[int, int]] = [(2, 2)] * 4

    @staticmethod
    def _apply_2d(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply a 2D layer frame-wise on NCTHW."""
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {x.shape}")
        B, C, T, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        y = layer(y)
        C_out, H_out, W_out = y.shape[1:]
        return y.view(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, video: torch.Tensor) -> List[torch.Tensor]:
        outputs = super().forward(video)
        return outputs if self.return_all_levels else [outputs[-1]]

    def _expected_shapes(
        self, video_shape: Tuple[int, ...]
    ) -> List[Tuple[int, int, int, int]]:
        """Compute (C, T, H, W) per level based on ELIC downsampling + temporal strides."""
        _, _, T, H, W = video_shape
        t, h, w = T, H, W
        shapes: List[Tuple[int, int, int, int]] = []
        for level, c in enumerate(self.channels):
            # Spatial downsample
            h = self._conv_out_len(
                h,
                self.spatial_kernels[level],
                self.spatial_strides[level],
                self.spatial_paddings[level],
            )
            w = self._conv_out_len(
                w,
                self.spatial_kernels[level],
                self.spatial_strides[level],
                self.spatial_paddings[level],
            )
            # Temporal downsample
            t_stride = self.temporal_strides[level] if level < len(self.temporal_strides) else 1
            if t_stride > 1:
                t = max(t // t_stride, 1)
            shapes.append((c, t, h, w))
        return shapes

    def _encode(self, video: torch.Tensor) -> List[torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {video.shape}")

        x = video
        outputs: List[torch.Tensor] = []
        t_idx = 0
        for idx, layer in enumerate(self.g_a):
            x = self._apply_2d(layer, x)
            if idx in self.temporal_insert_indices:
                # Apply temporal block
                x = self.temporal_blocks[t_idx](x)
                # Apply temporal downsample if needed
                if self.temporal_downsamples[t_idx] is not None:
                    x = self.temporal_downsamples[t_idx](x)
                outputs.append(x)
                t_idx += 1

        return outputs

    @classmethod
    def from_elic_state_dict(
        cls, state_dict, **kwargs
    ) -> "ELICTemporalConvNeXtEncoder":
        """
        Build from a 2D ELIC checkpoint. Temporal blocks are left randomly init.
        """
        # Infer channel dims from checkpoint.
        N = state_dict["g_a.0.weight"].shape[0]
        M = (
            state_dict["g_a.13.weight"].shape[0]
            if "g_a.13.weight" in state_dict
            else kwargs.get("M", 320)
        )
        model = cls(
            N=N, M=M, **{k: v for k, v in kwargs.items() if k not in {"N", "M"}}
        )
        model.load_state_dict(state_dict, strict=False)
        return model

    def log_structure(self, print_fn=print):
        """Pretty-print encoder structure and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_fn("=== ELICTemporalConvNeXtEncoder structure ===")
        print_fn(f"  Channels: {self.channels}")
        print_fn(f"  Temporal layer type: {self.temporal_layer_type}")
        print_fn(f"  Temporal strides: {self.temporal_strides}")
        print_fn(f"  Spatial strides: {self.spatial_strides}")
        print_fn(
            f"  Parameters: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M"
        )
