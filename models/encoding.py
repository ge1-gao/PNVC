"""
Encoding
"""
from .utils import *
from .layers import *
from .patch_utils import *
from .layers import *


def _compute_normalized_coordinate(l: torch.IntTensor, L: int, align_corners: bool):
    if L == 1.:
        return l * 0.
    elif align_corners:
        step = 2. / (L - 1)
        return -1. + l * step
    else:
        step = 2. / L
        return -1. + step / 2. + l * step


def _compute_normalized_coordinates_3d(l: List[torch.IntTensor], L: List[int], align_corners: bool):
    return [_compute_normalized_coordinate(l[d], L[d], align_corners) for d in range(3)]


def _interpolate3D(grid: torch.Tensor, coor_wht: torch.Tensor, align_corners: bool=False):
    """
    Interpolate 3D grid at given coordinates.
    
    Args:
        grid: Either [T_grid, H_grid, W_grid, C] (shared grid) 
              or [B, T_grid, H_grid, W_grid, C] (batched grids)
        coor_wht: [N, T, H, W, 3] sampling coordinates
        align_corners: whether to align corners in grid_sample
    
    Returns:
        [N, T, H, W, C] interpolated values
    """
    N, T, H, W, _ = coor_wht.shape
    
    # Always use fp32 for grid_sample
    coor_wht = coor_wht.float()
    
    if grid.ndim == 4:
        # Shared grid: [T_grid, H_grid, W_grid, C] - broadcast to all samples
        C = grid.shape[-1]
        # [T_grid, H_grid, W_grid, C] -> [1, C, T_grid, H_grid, W_grid]
        grid_5d = grid.permute(3, 0, 1, 2).unsqueeze(0).float()
        # [N, T, H, W, 3] -> [1, N * T, H, W, 3]
        coor_reshaped = coor_wht.view(1, N * T, H, W, 3)
        result = F.grid_sample(grid_5d, coor_reshaped, mode='bilinear', 
                               padding_mode='border', align_corners=align_corners)
        # [1, C, N*T, H, W] -> [C, N, T, H, W] -> [N, T, H, W, C]
        return result.view(C, N, T, H, W).permute(1, 2, 3, 4, 0).contiguous()
    
    elif grid.ndim == 5:
        # Batched grids: [B, T_grid, H_grid, W_grid, C] - each sample has its own grid
        B = grid.shape[0]
        C = grid.shape[-1]
        assert B == N, f"Grid batch size {B} must match coord batch size {N}"
        # [B, T_grid, H_grid, W_grid, C] -> [B, C, T_grid, H_grid, W_grid]
        grid_5d = grid.permute(0, 4, 1, 2, 3).float()
        # coor_wht is already [N, T, H, W, 3], use directly
        result = F.grid_sample(grid_5d, coor_wht, mode='bilinear',
                               padding_mode='border', align_corners=align_corners)
        # [B, C, T, H, W] -> [B, T, H, W, C]
        return result.permute(0, 2, 3, 4, 1).contiguous()
    
    else:
        raise ValueError(f"Grid must be 4D or 5D, got {grid.ndim}D")


def _compute_grid_encoding(grids: List[torch.FloatTensor], coors: List[torch.FloatTensor],
                           align_corners: bool=False, reduce='concat',
                           vidx: torch.Tensor=None):
    """
    Compute grid encoding by interpolating grids at given coordinates.

    Args:
        grids: List of grids, each either [T, H, W, ...] (shared) or [V, T, H, W, ...] (per-video)
        coors: [coor_t, coor_h, coor_w] where coor_t is [N, T], coor_h is [N, H], coor_w is [N, W]
        align_corners: whether to align corners in interpolation
        reduce: 'concat', 'sum', or 'none'
        vidx: [N] which video each patch belongs to (for per-video grids)

    Returns:
        Encoding tensor [N, T, H, W, C_total] or list of encodings
    """
    assert coors[0].ndim == 2 and (coors[1] is None or coors[1].ndim == 2) and (coors[2] is None or coors[2].ndim == 2)
    assert (coors[1] is None) == (coors[2] is None)
    N, T = coors[0].shape
    H = coors[1].shape[1] if coors[1] is not None else 1
    W = coors[2].shape[1] if coors[2] is not None else 1

    # sampling coordinates [N, T, H, W, 3]
    if H > 1 and W > 1:
        coor_t = coors[0][:, :, None, None, None].expand(N, T, H, W, 1)
        coor_h = coors[1][:, None, :, None, None].expand(N, T, H, W, 1)
        coor_w = coors[2][:, None, None, :, None].expand(N, T, H, W, 1)
        coor_wht = torch.cat([coor_w, coor_h, coor_t], dim=-1)
    else:
        coor_t = coors[0].view(N, T, 1, 1, 1)
        coor_wht = F.pad(coor_t, (2, 0))

    enc = []

    # V = number of videos in batch
    V = (vidx.max().item() + 1) if vidx is not None else None
    
    for grid_idx, grid_i in enumerate(grids):
        orig_shape = grid_i.shape
        orig_ndim = grid_i.ndim

        # ----------------------------------------------------------------
        # Determine if grid has batch (per-video) dimension based on ndim:
        #   - Shared grids:     4D [T,H,W,C] or 7D [T,1,1,s_t,s_h,s_w,C]
        #   - Per-video grids:  5D [V,T,H,W,C] or 8D [V,T,1,1,s_t,s_h,s_w,C]
        # 
        # Using ndim is more reliable than checking shape[0]==V, which can
        # fail when T_grid happens to equal V (e.g., both are 3).
        # ----------------------------------------------------------------
        is_per_video = (vidx is not None and orig_ndim in [5, 8])
        
        if is_per_video:
            # Handle per-video grid dimension mismatch
            if grid_i.shape[0] != V:
                if grid_i.shape[0] > V:
                    grid_i = grid_i[:V]
                else:
                    grid_i = grid_i.expand(V, *grid_i.shape[1:])
            # gather per-video grid to per-patch: [V, ...] -> [N, ...]
            grid_i = grid_i.index_select(0, vidx)

        # handle different grid dimensions
        if orig_ndim == 4:
            # shared grid: [T, H, W, C]
            grid_flat = grid_i.view(*grid_i.shape[:3], -1)
            enc_i = _interpolate3D(grid=grid_flat, coor_wht=coor_wht, align_corners=align_corners)
            # output: [N, T, H, W, C]
        elif orig_ndim == 5:
            if is_per_video:
                # batched grid after gather: [N, T, H, W, C]
                enc_i = _interpolate3D(grid=grid_i, coor_wht=coor_wht, align_corners=align_corners)
            else:
                # shared grid with extra dim (rare case)
                grid_flat = grid_i.view(*grid_i.shape[:3], -1)
                enc_i = _interpolate3D(grid=grid_flat, coor_wht=coor_wht, align_corners=align_corners)
                enc_i = enc_i.view(N, T, H, W, *orig_shape[3:])
        elif orig_ndim > 5:
            # higher dimensional grids (e.g., with stride dims)
            if is_per_video:
                # batched stride grid after gather: [N, T, H, W, s_t, s_h, s_w, C]
                # Use orig_shape to get correct stride dimensions (before index_select modified it)
                grid_flat = grid_i.view(*grid_i.shape[:4], -1)
                enc_i = _interpolate3D(grid=grid_flat, coor_wht=coor_wht, align_corners=align_corners)
                # orig_shape was [V, T, H, W, s_t, s_h, s_w, C], so orig_shape[4:] gives stride dims
                enc_i = enc_i.view(N, T, H, W, *orig_shape[4:])
            else:
                # shared stride grid: [T, H, W, s_t, s_h, s_w, C] -> [N, T, H, W, s_t, s_h, s_w, C]
                grid_flat = grid_i.view(*grid_i.shape[:3], -1)
                enc_i = _interpolate3D(grid=grid_flat, coor_wht=coor_wht, align_corners=align_corners)
                enc_i = enc_i.view(N, T, H, W, *orig_shape[3:])
        else:
            raise ValueError(f"Unexpected grid dimension: {orig_ndim}")
        enc.append(enc_i)

    if reduce == 'concat':
        # strict check: all tensors must have same ndim
        ndims = [e.ndim for e in enc]
        if len(set(ndims)) > 1:
            shapes_info = [(i, tuple(e.shape)) for i, e in enumerate(enc)]
            orig_shapes_info = [(i, tuple(g.shape)) for i, g in enumerate(grids)]
            raise RuntimeError(
                f"[_compute_grid_encoding] FATAL: mixed tensor dimensions detected!\n"
                f"  output enc ndims: {ndims}\n"
                f"  output enc shapes: {shapes_info}\n"
                f"  input grids shapes: {orig_shapes_info}\n"
                f"  vidx is {'set' if vidx is not None else 'None'}, V={V}, N={N}, T={T}, H={H}, W={W}\n"
                f"  This indicates grids from different sources (learnable vs encoder-injected) are being mixed."
            )
        return torch.concat(enc, dim=-1)
    elif reduce == 'sum':
        return sum(enc)
    elif reduce == 'none':
        return enc
    else:
        raise ValueError


class BaseEncodingImpV1(FuncModule):
    """
    Base Encoding.
    """
    def __init__(self, out_size, out_channels, paddings: tuple=(0, 0, 0),
                 grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], kernel_size=(1, 1, 1),
                 init_scale=1e-3, align_corners=False):
        super().__init__()
        self.out_size = out_size
        self.out_channels = out_channels
        self.paddings = paddings
        self.grid_sizes = []
        self.grid_level = grid_level
        self.grid_level_scale = grid_level_scale
        self.kernel_size = kernel_size
        self.grid_expands = nn.ModuleList()
        self.init_scale = init_scale
        self.align_corners = align_corners
        self.func_params_keys = [f'grids_{i}' for i in range(self.grid_level)]

        for i in range(self.grid_level):
            T_grid_i, H_grid_i, W_grid_i, C_grid_i = ((int(grid_size[d] / (self.grid_level_scale[d] ** i)) \
                                                       for d in range(4)))
            self.grid_sizes.append((T_grid_i, H_grid_i, W_grid_i, C_grid_i))
            self.grid_expands.append(BatchGridTrilinear3D(self.out_size, self.align_corners))

        if self.kernel_size[0] > 1:
            raise NotImplementedError
        elif self.kernel_size[1] > 1 or self.kernel_size[2] > 1:
            self.proj = FuncConv2D(sum(grid_size_i[-1] for grid_size_i in self.grid_sizes),
                                   self.out_channels, self.kernel_size[1:])
        else:
            self.proj = FuncLinear(sum(grid_size_i[-1] for grid_size_i in self.grid_sizes), self.out_channels)

    def extra_repr(self):
        s = 'out_size={out_size}, out_channels={out_channels}, paddings={paddings}, grid_sizes={grid_sizes}, align_corners={align_corners}'
        return s.format(**self.__dict__)

    def _get_params(self):
        params = {}
        for i, grid_size_i in enumerate(self.grid_sizes):
            T_grid_i, H_grid_i, W_grid_i, C_grid_i = grid_size_i
            grid_i = create_param(torch.zeros(T_grid_i, H_grid_i, W_grid_i, C_grid_i), 'grid')
            torch.nn.init.uniform_(grid_i, -self.init_scale, self.init_scale)
            params[f'grids_{i}'] = grid_i
        return params

    def forward(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                patch_mode: bool=True, vidx: torch.Tensor=None):
        grids = self._extract_params(params)

        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        padding = self.paddings if patch_mode else (0, 0, 0)
        patch_size = tuple(self.out_size[d] // idx_max[d] for d in range(3))
        patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.out_size, padding=padding, clipped=True)
        px_mask_3d = px_mask[0][:, :, None, None, None] * \
                     px_mask[1][:, None, :, None, None] * \
                     px_mask[2][:, None, None, :, None]
        px_idx_flat = (px_idx[0][:, :, None, None] * self.out_size[1] * self.out_size[2] + \
                       px_idx[1][:, None, :, None] * self.out_size[2] + \
                       px_idx[2][:, None, None, :]).view(-1)

        # V1 only supports shared grids, no vidx needed here
        grids_upsampled = torch.concat([self.grid_expands[i](grids[i]) for i in range(self.grid_level)], dim=-1)
        grids_upsampled_flat = grids_upsampled.view((-1, grids_upsampled.shape[-1],))

        encoding = torch.index_select(grids_upsampled_flat, 0, px_idx_flat) \
                        .view((idx.shape[0],) + patch_padded_size + (grids_upsampled.shape[-1],))
        output = px_mask_3d * self.proj(px_mask_3d * encoding, params)
        assert_shape(output, (idx.shape[0],) + patch_padded_size + (self.out_channels,))

        return output


class BaseEncodingImpV2(FuncModule):
    """
    Base Encoding V2
    """
    def __init__(self, out_size, out_channels, paddings: tuple=(0, 0, 0),
                 grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], kernel_size=(1, 1, 1),
                 init_scale=1e-3, align_corners=False):
        super().__init__()
        self.out_size = out_size
        self.out_channels = out_channels
        self.paddings = paddings
        self.grid_sizes = []
        self.grid_level = grid_level
        self.grid_level_scale = grid_level_scale
        self.kernel_size = kernel_size
        self.init_scale = init_scale
        self.align_corners = align_corners
        self.func_params_keys = [f'grids_{i}' for i in range(self.grid_level)]

        for i in range(self.grid_level):
            T_grid_i, H_grid_i, W_grid_i, C_grid_i = ((int(grid_size[d] / (self.grid_level_scale[d] ** i)) \
                                                       for d in range(4)))
            self.grid_sizes.append((T_grid_i, H_grid_i, W_grid_i, C_grid_i))
            if self.kernel_size[0] > 1:
                self.add_module(f'proj_{i}', FuncConv3D(C_grid_i, self.out_channels, self.kernel_size))
            else:
                self.add_module(f'proj_{i}', FuncConv2D(C_grid_i, self.out_channels, self.kernel_size[1:]))

    def extra_repr(self):
        s = 'out_size={out_size}, out_channels={out_channels}, paddings={paddings}, grid_sizes={grid_sizes} align_corners={align_corners}'
        return s.format(**self.__dict__)

    def _get_params(self):
        params = {}
        for i, grid_size_i in enumerate(self.grid_sizes):
            grid_i = create_param(torch.zeros(grid_size_i), 'grid')
            torch.nn.init.uniform_(grid_i, -self.init_scale, self.init_scale)
            params[f'grids_{i}'] = grid_i
        return params

    def _forward_proj_first(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                            patch_mode: bool=True, vidx: torch.Tensor=None):
        grids = self._extract_params(params)
        grids_projected = []
        for i in range(self.grid_level):
            grid_i = grids[i]
            # support both shared grids [T,H,W,C] and batched grids [V,T,H,W,C]
            if grid_i.ndim == 4:
                grid_input = grid_i.unsqueeze(0)
                squeeze_back = True
            elif grid_i.ndim == 5:
                grid_input = grid_i
                squeeze_back = False
            else:
                raise ValueError(
                    f"Unexpected grid ndim={grid_i.ndim} for proj_{i}, shape={tuple(grid_i.shape)}"
                )

            proj_out = getattr(self, f'proj_{i}')(grid_input, params)
            if squeeze_back:
                proj_out = proj_out.squeeze(0)
            grids_projected.append(proj_out)

        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        padding = self.paddings if patch_mode else (0, 0, 0)
        patch_size = tuple(self.out_size[d] // idx_max[d] for d in range(3))
        patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.out_size, padding=padding, clipped=True)
        px_mask_3d = px_mask[0][:, :, None, None, None] * \
                     px_mask[1][:, None, :, None, None] * \
                     px_mask[2][:, None, None, :, None]
        coor = _compute_normalized_coordinates_3d(px_idx, self.out_size, self.align_corners)

        encoding = _compute_grid_encoding(grids_projected, coor, self.align_corners, reduce='sum', vidx=vidx)
        output = px_mask_3d * encoding
        assert_shape(output, (idx.shape[0],) + patch_padded_size + (self.out_channels,))

        return output

    def _forward_interp_first(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                              patch_mode: bool=True, vidx: torch.Tensor=None):
        grids = self._extract_params(params)
        patch_mode = True

        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        padding = tuple(self.paddings[d] + (self.kernel_size[d] - 1) // 2 for d in range(3)) \
                  if patch_mode else (0, 0, 0)
        patch_size = tuple(self.out_size[d] // idx_max[d] for d in range(3))
        patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.out_size, padding=padding, clipped=True)
        px_mask_3d = px_mask[0][:, :, None, None, None] * \
                     px_mask[1][:, None, :, None, None] * \
                     px_mask[2][:, None, None, :, None]
        coor = _compute_normalized_coordinates_3d(px_idx, self.out_size, self.align_corners)

        encoding = _compute_grid_encoding(grids, coor, self.align_corners, reduce='none', vidx=vidx)
        output = px_mask_3d * sum(getattr(self, f'proj_{i}')(encoding[i], params) for i in range(self.grid_level))
        assert_shape(output, (idx.shape[0],) + patch_padded_size + (self.out_channels,))

        if patch_mode:
            if self.kernel_size[0] > 1:
                output = output[:, (self.kernel_size[0] - 1) // 2:-(self.kernel_size[0] - 1) // 2, :, :, :]
            if self.kernel_size[1] > 1:
                output = output[:, :, (self.kernel_size[1] - 1) // 2:-(self.kernel_size[1] - 1) // 2, :, :]
            if self.kernel_size[2] > 1:
                output = output[:, :, :, (self.kernel_size[2] - 1) // 2:-(self.kernel_size[2] - 1) // 2]
            output = output.contiguous()

        return output

    def forward(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                patch_mode: bool=True, vidx: torch.Tensor=None):
        interp_first = all(all(grid_size_i[:-1][d] == self.out_size[d] for d in range(3)) \
                           for grid_size_i in self.grid_sizes)
        if interp_first:
            return self._forward_interp_first(params, idx, idx_max, patch_mode, vidx=vidx)
        else:
            return self._forward_proj_first(params, idx, idx_max, patch_mode, vidx=vidx)


class BaseEncoding(FuncModule):
    def __init__(self, impl, out_size, out_channels, paddings: tuple=(0, 0, 0),
                 grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], kernel_size=(1, 1, 1),
                 init_scale=1e-3, align_corners=False):
        super().__init__()
        assert isinstance(out_size, (list, tuple)) and len(out_size) == 3
        self.out_size = out_size
        self.out_channels = out_channels
        if impl == 'v1':
            grid_cls = BaseEncodingImpV1
        elif impl == 'v2':
            grid_cls = BaseEncodingImpV2
        else:
            raise ValueError
        self.enc = grid_cls(out_size, out_channels, paddings, grid_size, grid_level, grid_level_scale, kernel_size, init_scale, align_corners)

    def forward(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                patch_mode: bool=True, vidx: torch.Tensor=None):
        return getattr(self, f'enc')(params, idx, idx_max, patch_mode, vidx=vidx)


def create_base_encoding(logger, grid_type, **kwargs):
    if grid_type in ['v1', 'v2']:
        return BaseEncoding(grid_type, **kwargs)
    else:
        raise ValueError
