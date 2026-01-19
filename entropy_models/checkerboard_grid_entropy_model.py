import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple

from .grid_entropy_model import GridEntropyModel
from .layers import FuncMaskedConv3d
from .utils import get_func_activation, get_func_norm, run_modules


class CheckerboardGridEntropyModel(GridEntropyModel):
    """
    Checkerboard context model for 3D grids.
    
    Uses spatial checkerboard pattern (2×2 in H×W) combined with channel splitting (4 groups).
    This enables 16-way parallelism: 4 spatial positions × 4 channel groups.
    
    Each sequential step encodes 1/16 of the total elements in parallel.
    Expected speedup: ~50-100x vs autoregressive model.
    """
    
    def __init__(self, logger, name, inputs, config):
        # Store config values before calling super().__init__
        self.channels = config.channels
        self.depth = config.depth
        self.kernel_size = config.kernel_size
        self.norm = getattr(config, "norm", "none")
        self.bias = getattr(config, "bias", True)
        self.act = getattr(config, "act", "gelu")
        self.depthwise = getattr(config, "depthwise", True)
        self.num_spatial_groups = 4  # 2×2 checkerboard
        self.num_channel_groups = 4  # Split channels into 4 groups
        self.num_total_groups = 16   # 4 × 4
        
        # Call parent init
        super().__init__(logger, name, inputs, config)
        
        self.learned_em = config.learned_em
        self.learned_quant = config.learned_quant
        
        # This is much more memory efficient than per-group networks
        for k, cfg in self._sort_tensors(self.tensor_cfgs):
            C = cfg["shape"][-1]
            assert C % self.num_channel_groups == 0, \
                f"Channels {C} must be divisible by {self.num_channel_groups}"
            
            # Same as ARGridEntropyModel: groups_k based on depthwise, channels_k = groups_k * self.channels
            groups_k = C if self.depthwise else 1
            channels_k = groups_k * self.channels
            
            # Single shared context network for all groups (like AR model)
            layers = []
            for n in range(self.depth):
                in_ch = C if n == 0 else channels_k
                out_ch = C * 2 if n == self.depth - 1 else channels_k
                
                layers.append(
                    FuncMaskedConv3d(
                        in_ch, out_ch, self.kernel_size,
                        groups=groups_k,
                        mask_cur=(n == 0),
                        bias=self.bias,
                        is_head=(n == self.depth - 1)
                    )
                )
                if n != self.depth - 1:
                    layers.append(get_func_activation(self.act)())
                    layers.append(get_func_norm(self.norm)(channels_k))
            
            self.add_module(f"{k}_layers", nn.ModuleList(layers))
    
    def extra_repr(self):
        s = super().extra_repr() + ", kernel_size={kernel_size}, channels={channels}, depth={depth}"
        s += ", norm={norm}, bias={bias}, act={act}, depthwise={depthwise}"
        s += ", spatial_groups={num_spatial_groups}, channel_groups={num_channel_groups}"
        return s.format(**self.__dict__)
    
    def _build_cfgs(self, inputs):
        """Build tensor configs without blocking."""
        tensor_cfgs = {}
        
        with torch.no_grad():
            for tensor_k, tensor in self._sort_tensors(inputs):
                assert tensor.ndim >= 4, "grids should be at least 4D (T, H, W, C)"
                # Support extra stride dims by flattening them into channels
                orig_shape = tuple(tensor.shape)  # (T, H, W, *extra, C)
                extra_dims = orig_shape[3:-1] if tensor.ndim > 4 else tuple()
                flat_c = orig_shape[-1] * (math.prod(extra_dims) if extra_dims else 1)
                base_shape = (orig_shape[0], orig_shape[1], orig_shape[2], flat_c)
                
                T, H, W, C = base_shape
                
                tensor_cfgs[tensor_k] = {
                    "orig_shape": orig_shape,
                    "extra_dims": extra_dims,
                    "flat_c": flat_c,
                    "shape": (-1, T, H, W, flat_c),
                    "mask_shape": (1, *orig_shape),
                    # For checkerboard: reshape to [N, M, T, H, W, C] where M=1
                    "reshape": (-1, 1, T, H, W, flat_c),
                    "offset": 0,
                    "length": 1,
                    # Meta shape for learned EM params
                    "meta_shape": (1, 1, flat_c),
                }
        
        return tensor_cfgs
    
    def _build_forward_order(self, inputs):
        return self.tensor_keys
    
    def _build_mask(self):
        """Build masks - just ones for checkerboard."""
        masks = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            # Mask shape: [1, 1, T, H, W, C]
            T, H, W, C = tensor_cfg["shape"][1:]
            masks[tensor_k] = torch.ones(1, 1, T, H, W, C, dtype=torch.bool)
        return masks
    
    # ----------------------------------------------------------------
    # Tensor reshape - no blocking, just add dimensions
    
    def _reshape_tensor(self, tensor_k, tensor):
        """Reshape tensor for checkerboard processing."""
        tensor_cfg = self.tensor_cfgs[tensor_k]
        # Input: (T, H, W, C) or (T, H, W, *extra, C)
        # Output: (1, 1, T, H, W, flat_c)
        tensor = tensor.unsqueeze(0)  # (1, T, H, W, ...)
        
        # Flatten extra dims into channel
        if tensor_cfg["extra_dims"]:
            tensor = tensor.view(
                1,
                tensor_cfg["orig_shape"][0],
                tensor_cfg["orig_shape"][1],
                tensor_cfg["orig_shape"][2],
                tensor_cfg["flat_c"],
            )
        
        # Add M dimension
        tensor = tensor.unsqueeze(1)  # (1, 1, T, H, W, C)
        return tensor
    
    def _restore_tensor(self, tensor_k, tensor):
        """Restore tensor from checkerboard format."""
        tensor_cfg = self.tensor_cfgs[tensor_k]
        # Input: (1, 1, T, H, W, C)
        # Output: (T, H, W, C) or (T, H, W, *extra, C)
        tensor = tensor.squeeze(0).squeeze(0)  # (T, H, W, C)
        
        # Restore extra dims if they existed
        if tensor_cfg["extra_dims"]:
            tensor = tensor.view(*tensor_cfg["orig_shape"])
        return tensor
    
    def _preprocess(self, inputs):
        """Preprocess inputs for forward pass."""
        outputs = {}
        for tensor_k, tensor in self._sort_tensors(inputs):
            outputs[tensor_k] = self._reshape_tensor(tensor_k, tensor)
        return outputs
    
    def _postprocess(self, inputs):
        """Postprocess outputs after forward pass."""
        outputs = {}
        for tensor_k, tensor in self._sort_tensors(inputs):
            outputs[tensor_k] = self._restore_tensor(tensor_k, tensor)
        return outputs
    
    # ----------------------------------------------------------------
    # Mask generation
    def _get_spatial_mask(self, h_offset, w_offset, T, H, W, device):
        """Generate spatial checkerboard mask for one position."""
        mask = torch.zeros((1, 1, T, H, W, 1), device=device)
        mask[:, :, :, h_offset::2, w_offset::2, :] = 1
        return mask
    
    def _get_masks(self, shape, device):
        """
        Generate all 16 checkerboard masks for shape [N, M, T, H, W, C]
        
        Returns: list of 16 masks, each [1, 1, T, H, W, C]
        """
        N, M, T, H, W, C = shape
        C_per_group = C // self.num_channel_groups
        
        # 2×2 spatial masks (in H×W space)
        m00 = self._get_spatial_mask(0, 0, T, H, W, device)  # top-left
        m01 = self._get_spatial_mask(0, 1, T, H, W, device)  # top-right
        m10 = self._get_spatial_mask(1, 0, T, H, W, device)  # bottom-left
        m11 = self._get_spatial_mask(1, 1, T, H, W, device)  # bottom-right
        
        spatial_masks = [m00, m01, m10, m11]
        
        masks = []
        for spatial_idx in range(self.num_spatial_groups):
            for channel_idx in range(self.num_channel_groups):
                # Create channel mask
                channel_mask = torch.zeros((1, 1, 1, 1, 1, C), device=device)
                c_start = channel_idx * C_per_group
                c_end = c_start + C_per_group
                channel_mask[..., c_start:c_end] = 1
                
                # Combine spatial and channel masks
                mask = spatial_masks[spatial_idx] * channel_mask
                masks.append(mask)
        
        return masks
    
    # ----------------------------------------------------------------
    # Core entropy model methods
    
    def _num_iters(self, k):
        """Number of sequential iterations = number of mask groups."""
        if self.compress_mode in ["none"]:
            return 1
        elif self.compress_mode in ["encoding", "decoding"]:
            return self.num_total_groups  # 16
        else:
            raise ValueError(f"Unknown compress mode {self.compress_mode}")
    
    def _init_ctx(self, k, x, params):
        """Initialize context as zeros (will be filled progressively)."""
        if self.compress_mode in ["none"]:
            return x, None
        elif self.compress_mode in ["encoding", "decoding"]:
            ctx = torch.zeros_like(x)
            ctx_mask = torch.ones_like(x)
            return ctx, ctx_mask
        else:
            raise ValueError(f"Unknown compress mode {self.compress_mode}")
    
    def _extract_ctx(self, k, i, ctx, ctx_mask, params):
        """Extract context for iteration i."""
        return ctx, ctx_mask
    
    def _fill_ctx(self, k, i, ctx, ctx_mask, x_i, params):
        """Fill context with values from iteration i."""
        if self.compress_mode in ["none"]:
            return x_i, ctx_mask
        elif self.compress_mode in ["encoding", "decoding"]:
            # Get mask for this group
            masks = self._get_masks(ctx.shape, ctx.device)
            mask_i = masks[i]
            # Update context
            ctx = ctx + x_i * mask_i
            return ctx, ctx_mask
        else:
            raise ValueError(f"Unknown compress mode {self.compress_mode}")
    
    def _update_ctx(self, k, i, ctx, ctx_mask, x_i, x_i_mask):
        """Update context with decoded values from group i (alias for _fill_ctx)."""
        return self._fill_ctx(k, i, ctx, ctx_mask, x_i, {})
    
    def _init_meta(self, k, x, params):
        """Initialize meta data."""
        return {}
    
    def _extract_meta(self, k, i, meta):
        """Extract meta data for iteration i."""
        return {}
    
    def _extract_code(self, k, i, ctx, ctx_mask, x, params):
        """Extract values for group i using mask."""
        tensor_mask = getattr(self, f"{k}_mask")
        
        if self.compress_mode in ["none"]:
            return x, tensor_mask.expand_as(x)
        elif self.compress_mode in ["encoding", "decoding"]:
            masks = self._get_masks(x.shape, x.device)
            mask_i = masks[i]
            x_i = x * mask_i
            return x_i, mask_i.expand_as(x_i)
        else:
            raise ValueError(f"Unknown compress mode {self.compress_mode}")
    
    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        """Compute entropy parameters (means, scales) for group i."""
        N, M, T, H, W, C = ctx_i.shape
        device = ctx_i.device
        
        if self.compress_mode == "none":
            # For forward pass without compression
            # Use mask to get correct shape (same as GridEntropyModel)
            tensor_mask = getattr(self, f"{k}_mask").to(device)
            
            if self.learned_em:
                tensor_means = params[k + "_means"]
                tensor_scales = params[k + "_log_scales"].exp()  # Apply exp to get actual scale
                # Expand to match mask shape: [1, 1, T, H, W, C]
                tensor_means = tensor_means.view(1, 1, 1, 1, 1, C)
                tensor_scales = tensor_scales.view(1, 1, 1, 1, 1, C)
                means = tensor_mask * tensor_means
                scales = tensor_mask * tensor_scales
            else:
                means = tensor_mask * torch.zeros((1, 1, 1, 1, 1, C), device=device)
                scales = tensor_mask * torch.ones((1, 1, 1, 1, 1, C), device=device)
            return means, scales
        
        # For encoding/decoding mode
        if i == 0:
            means = torch.zeros((N, M, T, H, W, C), device=device)
            scales = torch.ones((N, M, T, H, W, C), device=device)
        else:
            f = ctx_i.view(N * M, T, H, W, C)
            if ctx_mask_i is not None:
                f_mask = ctx_mask_i.view(N * M, T, H, W, C).any(dim=-1, keepdim=True)
            else:
                f_mask = None
            
            # Use shared context network (same as AR model)
            layers = getattr(self, f"{k}_layers")
            f = run_modules(layers, f, params, f_mask)
            
            # f shape: [N*M, T, H, W, C * 2] - outputs means/scales for ALL channels
            assert f.shape[-1] == C * 2, f"Expected {C * 2}, got {f.shape[-1]}"
            
            # Split into means and scales (full C channels each)
            means, scales = f.chunk(2, dim=-1)
            means = means.contiguous()
            scales = scales.exp().contiguous()  # Apply exp to get actual scale
            
            means = means.view(N, M, T, H, W, C).contiguous()
            scales = scales.view(N, M, T, H, W, C).contiguous()
        
        # Apply mask
        masks = self._get_masks(ctx_i.shape, device)
        mask_i = masks[i]
        means = means * mask_i
        scales = scales * mask_i
        
        return means, scales
    
    def _get_log_step(self, k, i, params):
        """Get quantization step (log scale)."""
        if self.learned_quant:
            log_step = params[k + "_log_step"]
            # Expand to match tensor shape
            return log_step.unsqueeze(0).unsqueeze(0)
        else:
            return torch.tensor(-4.0, device=next(self.parameters()).device)
