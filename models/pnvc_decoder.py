import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .layers import FuncModMixerBlock, FuncLinear, FuncModule, modulate


@dataclass
class PNVCDecoderConfig:
    """Lightweight config for PNVC ModMixer decoder."""

    channels: List[int] = field(default_factory=lambda: [256, 256, 128, 64])
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 1])
    block_act: str = "gelu"
    block_norm: str = "layernorm-no-affine"
    block_bias: bool = False
    block_layerscale: float = 0.0
    attn_heads: int = 4
    attn_dim_head: int = 32
    num_branches: int = 2
    branch_depths: Optional[List[int]] = None  # defaults to [2] * num_branches in code
    hidden_mult: float = 4.0
    use_fused: bool = True


class GridModulator(FuncModule):
    """
    Project grid (Cg) to shift/scale (2*C) for FiLM-style modulation.
    """

    def __init__(self, grid_ch: int, feat_ch: int):
        super().__init__()
        self.proj = FuncLinear(grid_ch, 2 * feat_ch, bias=True)
        self.func_params_keys = []

    def forward(self, grid: torch.Tensor, params: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.proj(grid, params).chunk(2, dim=-1)
        return shift, scale


class PNVCDecoder(FuncModule):
    """
    PNVC-style decoder: ModMixer blocks + per-block grid modulation.

    Input/Output format: [N, T, H, W, C] (channel-last), matching existing functional blocks.
    Grids: list of hierarchical grids, one per stage, each [N, T, H, W, Cg].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_channels: List[int],
        cfg: PNVCDecoderConfig,
    ):
        super().__init__()
        assert len(cfg.channels) == len(cfg.depths) == len(grid_channels), "channels/depths/grid_channels length mismatch"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cfg = cfg

        self.stages = nn.ModuleList()
        self.modulators = nn.ModuleList()

        c_in = in_channels
        for i, (c_out, depth, g_ch) in enumerate(
            zip(cfg.channels, cfg.depths, grid_channels)
        ):
            blocks = nn.ModuleList()
            # Modulate current feature dimension (pre-block)
            mod = GridModulator(g_ch, c_in)
            hidden = int(c_out * cfg.hidden_mult)
            for _ in range(depth):
                blocks.append(
                    FuncModMixerBlock(
                        in_features=c_in,
                        out_features=c_out,
                        hidden_features=hidden,
                        act=cfg.block_act,
                        norm=cfg.block_norm,
                        bias=cfg.block_bias,
                        layerscale_init=cfg.block_layerscale,
                        attn_heads=cfg.attn_heads,
                        attn_dim_head=cfg.attn_dim_head,
                        num_branches=cfg.num_branches,
                        branch_depths=cfg.branch_depths,
                        use_fused=cfg.use_fused,
                    )
                )
                c_in = c_out
            self.stages.append(blocks)
            self.modulators.append(mod)

        self.head = FuncLinear(c_in, out_channels, bias=True)

    def _get_params(self):
        return {}

    def forward(
        self,
        x: torch.Tensor,
        grids: Optional[List[torch.Tensor]],
        params: dict,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, T, H, W, C_in], channel-last
            grids: list of [N, T, H, W, Cg], one per stage, or None to skip modulation
            params: functional params dict
            mask: optional broadcastable mask
        """
        if grids is not None:
            assert len(grids) == len(self.stages), "grids length must match decoder stages"

        for stage_idx, (blocks, mod) in enumerate(zip(self.stages, self.modulators)):
            if grids is not None:
                grid = grids[stage_idx]
                shift, scale = mod(grid, params)
                x = modulate(x, shift, scale)
            for blk in blocks:
                x = blk(x, params, mask)

        x = self.head(x, params)
        return x
