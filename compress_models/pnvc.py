from dataclasses import dataclass, field
from typing import Optional
import copy

import torch
import torch.nn.functional as F

# Reuse the backbone internally (from iscas26.py) but keep PNVC-facing names clean.
from .iscas26 import (
    GridEMConfig as _BackboneGridEMConfig,
    WeightEMConfig as _BackboneWeightEMConfig,
    MetaEMConfig as _BackboneMetaEMConfig,
    OptimizeConfig as _BackboneOptimizeConfig,
    BaseConfig as _BackboneConfig,
    BaseCodec as _BackboneCodec,
    BaseCodecWithEncoder as _BackboneCodecWithEncoder,
    build_compress_model as _build_backbone,
)


@dataclass
class PatchifyConfig:
    """Simple patchification config (ViT-style)."""

    patch_size: list[int] = field(default_factory=lambda: [2, 8, 8])  # [T, H, W]
    stride: list[int] = field(default_factory=lambda: [2, 8, 8])
    padding: list[int] = field(default_factory=lambda: [0, 0, 0])
    overlap: bool = False


@dataclass
class PNVCConfig(_BackboneConfig):
    """PNVC compression config; extends backbone config with optional patchify spec."""

    patchify: Optional[PatchifyConfig] = field(default_factory=PatchifyConfig)

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.patchify, dict):
            self.patchify = PatchifyConfig(**self.patchify)


class PNVC(_BackboneCodec):
    """
    PNVC codec (base variant without external encoder injection).
    Inherits the shared backbone behavior; defaults should be paired with ModMixer decoder.
    """

    def __init__(self, logger, model, config: PNVCConfig):
        # Keep a copy for reference (e.g., patchify config)
        self.pnvc_config = copy.deepcopy(config)
        super().__init__(logger, model, config)

    def _maybe_patchify_inputs(self, inputs: dict):
        cfg = getattr(self.pnvc_config, "patchify", None)
        if cfg is None:
            return inputs, None
        ps = cfg.patch_size[1] if len(cfg.patch_size) > 1 else cfg.patch_size[0]
        video = inputs.get("video", None)
        if video is None:
            return inputs, None
        assert video.dim() == 5, "video should be [N,C,T,H,W]"
        assert video.shape[3] % ps == 0 and video.shape[4] % ps == 0, "video spatial dims must be divisible by patch size"

        video_p = patchify_video_spatial(video, ps)
        inputs_p = dict(inputs)
        inputs_p["video"] = video_p
        # update shape metadata
        inputs_p["video_size"] = (video_p.shape[2], video_p.shape[3], video_p.shape[4])
        if "patch_size" in inputs_p:
            pz = inputs_p["patch_size"]
            inputs_p["patch_size"] = (pz[0], pz[1] // ps, pz[2] // ps)
        if "channels" in inputs_p:
            inputs_p["channels"] = video_p.shape[1]
        if "idx_max" in inputs_p and "patch_size" in inputs_p and "video_size" in inputs_p:
            vs = inputs_p["video_size"]
            pz = inputs_p["patch_size"]
            inputs_p["idx_max"] = tuple(vs[d] // pz[d] for d in range(3))

        cache = {
            "ps": ps,
            "orig_channels": video.shape[1],
            "orig_video_size": (video.shape[2], video.shape[3], video.shape[4]),
        }
        return inputs_p, cache

    def _maybe_depatchify_output(self, output: torch.Tensor, cache):
        if cache is None:
            return output
        ps = cache["ps"]
        return depatchify_video_spatial(output, ps, bias=None)

    def forward(self, x, compute_outputs=True, compute_rates=True, compute_params=False, reduce_rates=True):
        patched_x, cache = self._maybe_patchify_inputs(x) if isinstance(x, dict) else (x, None)
        out = super().forward(
            patched_x,
            compute_outputs=compute_outputs,
            compute_rates=compute_rates,
            compute_params=compute_params,
            reduce_rates=reduce_rates,
        )
        if not compute_outputs:
            return out
        # out could be tensor or tuple
        if isinstance(out, tuple):
            out_list = list(out)
            out_list[0] = self._maybe_depatchify_output(out_list[0], cache)
            return tuple(out_list)
        else:
            return self._maybe_depatchify_output(out, cache)


class PNVCCodec(_BackboneCodecWithEncoder):
    """
    PNVC codec with optional encoder-generated grids (mirrors the backbone codec+encoder).
    """

    def __init__(
        self,
        logger,
        model,
        config: PNVCConfig,
        encoder,
        align_corners: bool = True,
        init_grids_from_encoder: bool = False,
        debug: bool = False,
    ):
        self.pnvc_config = copy.deepcopy(config)
        super().__init__(
            logger,
            model,
            config,
            encoder,
            align_corners=align_corners,
            init_grids_from_encoder=init_grids_from_encoder,
            debug=debug,
        )


def _ensure_modmixer_decoder(model_cfg: Optional[dict]) -> Optional[dict]:
    """
    Ensure decoder block_type defaults to ModMixer if not explicitly set.
    Operates on a dict (model config).
    """
    if model_cfg is None:
        return None
    cfg = copy.deepcopy(model_cfg)
    decoder = cfg.get("decoder", {})
    # Normalize block_type to list length >=1
    if "block_type" not in decoder or decoder["block_type"] is None:
        decoder["block_type"] = ["modmixer"]
    elif isinstance(decoder["block_type"], list):
        decoder["block_type"] = decoder["block_type"]
    else:
        decoder["block_type"] = [decoder["block_type"]]
    # Sensible defaults for ModMixer
    decoder.setdefault("block_norm", ["layernorm-no-affine"])
    decoder.setdefault("block_act", ["gelu"])
    decoder.setdefault("block_bias", False)
    decoder.setdefault("block_layerscale", 0.0)
    # ModMixer-specific knobs (no reparam)
    decoder.setdefault("modmixer_use_fused", False)
    decoder.setdefault("modmixer_mask_init", 1.0)
    decoder.setdefault("modmixer_num_branches", 1)
    decoder.setdefault("modmixer_branch_depths", [1])
    decoder.setdefault("modmixer_hidden_mult", 1.0)
    decoder.setdefault("modmixer_attn_heads", 4)
    decoder.setdefault("modmixer_attn_dim_head", 32)
    decoder.setdefault("modmixer_window_size", [1, 8, 8])
    cfg["decoder"] = decoder
    return cfg


def build_compress_model(logger, model, cfg, model_cfg=None, stage=None):
    """
    Build PNVC compression model. Reuses the backbone pipeline but enforces
    ModMixer decoder defaults and exposes Patchify config via PNVCConfig.
    """
    # peel off encoder/patchify when instantiating backbone config
    encoder_cfg = cfg.get("encoder", None) if isinstance(cfg, dict) else None
    cfg_no_enc = {k: v for k, v in cfg.items() if k != "encoder"} if isinstance(cfg, dict) else cfg
    cfg_core = {k: v for k, v in cfg_no_enc.items() if k != "patchify"} if isinstance(cfg_no_enc, dict) else cfg_no_enc

    # enforce ModMixer defaults in model_cfg (decoder side)
    model_cfg_mod = _ensure_modmixer_decoder(model_cfg)

    # wrap cfg into PNVCConfig
    pnvc_cfg = PNVCConfig(**cfg_core) if not isinstance(cfg_core, PNVCConfig) else cfg_core

    # if encoder is provided, delegate to backbone builder but swap class targets
    if encoder_cfg is not None:
        # reuse backbone builder to construct encoder etc., then swap class
        codec = _build_backbone(
            logger,
            model,
            {**cfg_core, "encoder": encoder_cfg} if isinstance(cfg_core, dict) else cfg_core,
            model_cfg=model_cfg_mod,
            stage=stage,
        )
        # upgrade codec class for clarity (keeps same state)
        codec.__class__ = PNVCCodec if hasattr(codec, "_build_grid_key_mapping") else PNVC
        codec.pnvc_config = pnvc_cfg
        return codec

    codec = PNVC(logger, model, config=pnvc_cfg)

    if stage and stage.startswith("pretrain_"):
        codec.set_pretrain_mode(stage)

    return codec


def patchify_video_spatial(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Spatial patchify using pixel_unshuffle over H,W.
    Args:
        x: [N, C, T, H, W]
        patch_size: int patch size (e.g., 8)
    Returns:
        [N, C * patch_size^2, T, H/patch_size, W/patch_size]
    """
    n, c, t, h, w = x.shape
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    y_2d = F.pixel_unshuffle(x_2d, patch_size)
    _, c_new, h_new, w_new = y_2d.shape
    y = y_2d.reshape(n, t, c_new, h_new, w_new).permute(0, 2, 1, 3, 4).contiguous()
    return y


def depatchify_video_spatial(x: torch.Tensor, patch_size: int, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Spatial depatchify using pixel_shuffle over H,W with optional channel bias.
    Args:
        x: [N, C, T, H, W] where C is divisible by patch_size^2
        patch_size: int patch size (e.g., 8)
        bias: optional [C] bias added before pixel_shuffle
    Returns:
        [N, C_out, T, H*patch_size, W*patch_size] where C_out = C / patch_size^2
    """
    n, c, t, h, w = x.shape
    if bias is not None:
        bias = bias.view(1, c, 1, 1, 1)
        x = x + bias
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    y_2d = F.pixel_shuffle(x_2d, patch_size)
    _, c_out, h_out, w_out = y_2d.shape
    y = y_2d.reshape(n, t, c_out, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
    return y
