"""
Base class for compression models
"""
from .utils import *
from typing import Dict, Tuple

from entropy_models.entropy_model_base import (
    FullPrecisionModel,
    HalfPrecisionModel,
    _EntropyModelBase,
    EntropyModel,
)
from entropy_models.weight_entropy_model import (
    WeightEntropyModel,
    SingleAxisEntropyModel,
    DoubleAxisEntropyModel,
)
from entropy_models.grid_entropy_model import GridEntropyModel, ARGridEntropyModel
from entropy_models.checkerboard_grid_entropy_model import CheckerboardGridEntropyModel
try:
    from modules.encoder import (
        LatentGridEncoderBase,
        compute_min_chunk_size,
        validate_encoder_decoder_config,
        compute_encoder_channels_from_grids,
    )
except ImportError:
    LatentGridEncoderBase = None
    compute_min_chunk_size = None
    validate_encoder_decoder_config = None
    compute_encoder_channels_from_grids = None



class CodecBase(FuncModule):
    """
    Codec base class
    """
    cache_enabled = True
    cached = False
    rates_cache = None
    params_cache = None
    def _forward_rates_params(self, x=None, forward_rates=True, forward_params=True):
        """
        Forward parameters and rates
        """
        raise NotImplementedError

    def _get_rate(self, rates):
        """
        Get rate
        """
        raise NotImplementedError

    def _get_rate_loss(self, rates):
        """
        Get rate loss
        """
        raise NotImplementedError

    def forward(self, x, compute_outputs=True, compute_rates=True, compute_params=False, reduce_rates=True):
        """
        Forward pass
        """
        assert compute_outputs or compute_rates or compute_params, \
            'At least one of get_outputs and get_rates should be True'

        all_outputs = []

        # Compute params and rates
        if not self.cached:
            rates, params = self._forward_rates_params(x, forward_rates=compute_rates,
                                                       forward_params=compute_outputs or compute_params)
        else:
            rates = self.rates_cache
            params = self.params_cache

        # Outputs
        if compute_outputs:
            outputs = self.model(x, params)
            all_outputs.append(outputs)

        # Rates
        if compute_rates:
            rate = self._get_rate(rates, reduce_rates)
            rate_loss = self._get_rate_loss(rates, reduce_rates)
            all_outputs.append(rate)
            all_outputs.append(rate_loss)

        # Params
        if compute_params:
            all_outputs.append(params)
        
        return tuple(all_outputs) if len(all_outputs) > 1 else all_outputs[0]

    def load_cache(self):
        """
        Load parameters and rate cache
        """
        if self.cache_enabled:
            with torch.no_grad():
                self.rates_cache, _, self.params_cache = self(None, compute_outputs=False, compute_rates=True,
                                                              compute_params=True, reduce_rates=False)
                self.cached = True

    def clear_cache(self):
        """
        Clear parameters and rate cache
        """
        self.rates_cache, self.params_cache = None, None
        self.cached = False

    def compress(self, bs):
        """
        Compress
        """
        raise NotImplementedError

    def decompress(self, bs):
        """
        Decompress
        """
        raise NotImplementedError

    def get_rate_stat(self):
        """
        Get rate statistics
        """
        raise NotImplementedError

    def set_epoch(self, epoch, num_epochs):
        """
        Set epoch
        """
        raise NotImplementedError
    
    def set_iter(self, iter, num_iters):
        """
        Set iter
        """
        raise NotImplementedError


@dataclass
class EMConfigBase:
    em: str = "em"
    em_dist: str = "gaussian"
    rate_quant: str = "soft-round-noise"
    dist_quant: str = "soft-round-noise"
    quant_mean: bool = False
    learned_quant: bool = True
    learned_em: bool = True


@dataclass
class GridEMConfig(EMConfigBase):
    blk_size: List[int] = field(default_factory=lambda: [20, 5, 5])
    per_blk: bool = True
    channels: int = 4
    depth: int = 3
    kernel_size: int = 3
    norm: str = "layernorm"
    bias: bool = True
    act: str = "gelu"
    depthwise: bool = False


@dataclass
class WeightEMConfig(EMConfigBase):
    blk_size: List[int] = field(default_factory=lambda: [16, 32])
    per_blk: bool = True
    per_axis: bool = True


@dataclass
class MetaEMConfig(EMConfigBase):
    pass


@dataclass
class OptimizeConfig:
    optimize_rate: bool = True
    init_bit_width: int = -1
    soft_round_temp: List[float] = field(default_factory=lambda: [0.3, 0.1])
    kum_noise: List[float] = field(default_factory=lambda: [2.0, 1.0])
    quant_noise: List[float] = field(default_factory=lambda: [0.0, 0.9])


@dataclass
class BaseConfig:
    grid_em: GridEMConfig = field(default_factory=GridEMConfig)
    weight_em: WeightEMConfig = field(default_factory=WeightEMConfig)
    grid_meta_em: MetaEMConfig = field(default_factory=MetaEMConfig)
    weight_meta_em: MetaEMConfig = field(default_factory=MetaEMConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    debug: bool = False

    def __post_init__(self):
        self.grid_em = (
            GridEMConfig(**self.grid_em)
            if isinstance(self.grid_em, dict)
            else self.grid_em
        )
        self.weight_em = (
            WeightEMConfig(**self.weight_em)
            if isinstance(self.weight_em, dict)
            else self.weight_em
        )
        self.grid_meta_em = (
            MetaEMConfig(**self.grid_meta_em)
            if isinstance(self.grid_meta_em, dict)
            else self.grid_meta_em
        )
        self.weight_meta_em = (
            MetaEMConfig(**self.weight_meta_em)
            if isinstance(self.weight_meta_em, dict)
            else self.weight_meta_em
        )
        self.optimize = (
            OptimizeConfig(**self.optimize)
            if isinstance(self.optimize, dict)
            else self.optimize
        )


class BaseCodec(CodecBase):
    """
    BaseCodec codec
    """

    def __init__(self, logger, model, config):
        super().__init__()
        self.logger = logger
        self.config = copy.deepcopy(config)

        logger.info(f"BaseCodec:")
        logger.info(
            f"     Optimize rate - {self.config.optimize.optimize_rate}  Init bit width - {self.config.optimize.init_bit_width}"
        )
        logger.info(
            f"     Soft-rounding temp - {self.config.optimize.soft_round_temp}  Kumaraswamy - {self.config.optimize.kum_noise}"
        )
        logger.info(f"     QuantNoise - {self.config.optimize.quant_noise}")
        logger.info(f"     Cache enabled - {self.cache_enabled}")

        # INR
        self.model = model

        # INR param model
        init_params = self.model.get_params_recursive()
        self.param_model = ParamModel(logger, init_params)

        params = self.param_model()
        self._set_split_params(init_params)
        grids, weights, _ = self._split_params(params)

        # Grid em
        self._build_grid_em(logger, grids=grids, config=self.config.grid_em)
        self.grid_em.init_func_modules()
        self.grid_em_param_model = ParamModel(
            logger, self.grid_em.get_params_recursive()
        )

        # Weight em
        self._build_weight_em(logger, weights=weights, config=self.config.weight_em)
        self.weight_em.init_func_modules()
        self.weight_em_param_model = ParamModel(
            logger, self.weight_em.get_params_recursive()
        )

        # Grid meta
        self._build_grid_meta_em(
            logger,
            grid_em_weights=self.grid_em_param_model(),
            config=self.config.grid_meta_em,
        )
        self.grid_meta_em.init_func_modules()
        self.grid_meta_em_param_model = ParamModel(
            logger, self.grid_meta_em.get_params_recursive()
        )

        # Weight meta
        self._build_weight_meta_em(
            logger,
            weight_em_weights=self.weight_em_param_model(),
            config=self.config.weight_meta_em,
        )
        self.weight_meta_em.init_func_modules()
        self.weight_meta_em_param_model = ParamModel(
            logger, self.weight_meta_em.get_params_recursive()
        )

        # Meta
        self._build_meta_em(logger, inputs={})
        self.meta_em.init_func_modules()

    def _skip_pretrain_key(self, key: str) -> bool:
        # Do not inherit hierarchical/base latent grids and their EM params across stages.
        skip_tokens = (
            "decoder_hi_encodings",
            "decoder_base_encoding_enc",
        )
        return any(token in key for token in skip_tokens)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Load state dict with safe filtering:
        - skip hierarchical/base grid params and their EM log steps
        - skip shape-mismatched entries
        - skip non-finite tensors
        """
        current = self.state_dict()
        skip_snapshot = {
            k: v.detach().clone()
            for k, v in current.items()
            if self._skip_pretrain_key(k) and torch.is_tensor(v)
        }
        filtered = {}
        skipped = {"grid": 0, "shape": 0, "missing": 0, "nan": 0}
        for k, v in state_dict.items():
            if self._skip_pretrain_key(k):
                skipped["grid"] += 1
                continue
            if k not in current:
                skipped["missing"] += 1
                continue
            if current[k].shape != v.shape:
                skipped["shape"] += 1
                continue
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                skipped["nan"] += 1
                continue
            filtered[k] = v
        if any(skipped.values()):
            self.logger.info(
                "load_state_dict filtered params: "
                f"grid={skipped['grid']}, shape={skipped['shape']}, "
                f"missing={skipped['missing']}, nan={skipped['nan']}"
            )
        result = super().load_state_dict(filtered, strict=False)
        if strict and (result.missing_keys or result.unexpected_keys):
            self.logger.warning(
                "load_state_dict strict=True but filtered entries; "
                f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
            )
        if skip_snapshot:
            named_params = dict(self.named_parameters())
            named_buffers = dict(self.named_buffers())
            reset = 0
            for k, v in skip_snapshot.items():
                target = named_params.get(k)
                if target is None:
                    target = named_buffers.get(k)
                if target is None:
                    continue
                target.data.copy_(v.to(target.device, target.dtype))
                reset += 1
            if reset:
                self.logger.info(f"reset skipped params to init: {reset}")
        return result

    def _build_grid_em(self, logger, grids, config: GridEMConfig):
        logger.info(f"     Grid EM:")
        logger.info(
            f"          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}"
        )
        logger.info(
            f"          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}"
        )
        if config.em != "hier":
            logger.info(
                f"          Blk size - {config.blk_size}  Per blk - {config.per_blk}"
            )
        else:
            ctx_mode = getattr(config, "context_mode", "quincunx")
            ch_part = getattr(config, "channel_partition", "")
            logger.info(
                f'          Hier order - RA, context_mode={ctx_mode}, channel_partition="{ch_part}"'
            )
            logger.info(f"          Conditioning - cross-level prev grid channels")
        logger.info(f"          Channels - {config.channels}  Depth - {config.depth}")
        logger.info(
            f"          Kernel size - {config.kernel_size}  Norm - {config.norm}  Bias - {config.bias}  Act - {config.act}"
        )
        logger.info(f"          Depthwise - {config.depthwise}")

        try:
            em_cls = {
                "em": EntropyModel,
                "blk": GridEntropyModel,
                "ar": ARGridEntropyModel,
                "checkerboard": CheckerboardGridEntropyModel,
                "fp32": FullPrecisionModel,
                "fp16": HalfPrecisionModel,
            }[config.em]
        except KeyError:
            raise ValueError(f"Unknown grid entropy model {config.em}")
        if issubclass(em_cls, _EntropyModelBase):
            self.grid_em = em_cls(logger, name="grid_em", inputs=grids, config=config)
        else:
            self.grid_em = em_cls(logger, name="grid_em", inputs=grids)

    def _build_weight_em(self, logger, weights, config: WeightEMConfig):
        logger.info(f"     Weight EM:")
        logger.info(
            f"          EM - {config.em}  Dist - {config.em_dist}  Rate quant - {config.rate_quant}"
        )
        logger.info(
            f"          Blk size - {config.blk_size}  Per blk - {config.per_blk}  Per axis - {config.per_axis}"
        )

        em_cls = {
            "em": EntropyModel,
            "blk": WeightEntropyModel,
            "s-axis": SingleAxisEntropyModel,
            "d-axis": DoubleAxisEntropyModel,
            "fp32": FullPrecisionModel,
            "fp16": HalfPrecisionModel,
        }[config.em]

        if issubclass(em_cls, _EntropyModelBase):
            self.weight_em = em_cls(
                logger, name="weight_em", inputs=weights, config=config
            )
        else:
            self.weight_em = em_cls(logger, name="weight_em", inputs=weights)

    def _build_grid_meta_em(self, logger, grid_em_weights, config: MetaEMConfig):
        logger.info(f"     Grid meta - {config.em}")
        logger.info(
            f"          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}"
        )
        logger.info(
            f"          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}"
        )

        if grid_em_weights:
            em_cls = {
                "em": EntropyModel,
                "fp32": FullPrecisionModel,
                "fp16": HalfPrecisionModel,
            }[config.em]

            if issubclass(em_cls, _EntropyModelBase):
                self.grid_meta_em = em_cls(
                    logger, name="grid_meta_em", inputs=grid_em_weights, config=config
                )
            else:
                self.grid_meta_em = em_cls(
                    logger, name="grid_meta_em", inputs=grid_em_weights
                )
        else:
            self.grid_meta_em = FullPrecisionModel(
                logger, name="grid_meta_em", inputs={}
            )

    def _build_weight_meta_em(self, logger, weight_em_weights, config: WeightEMConfig):
        logger.info(f"     Weight meta - {config.em}")
        logger.info(
            f"          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}"
        )
        logger.info(
            f"          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}"
        )

        if weight_em_weights:
            em_cls = {
                "em": EntropyModel,
                "fp32": FullPrecisionModel,
                "fp16": HalfPrecisionModel,
            }[config.em]

            if issubclass(em_cls, _EntropyModelBase):
                self.weight_meta_em = em_cls(
                    logger,
                    name="weight_meta_em",
                    inputs=weight_em_weights,
                    config=config,
                )
            else:
                self.weight_meta_em = em_cls(
                    logger, name="weight_meta_em", inputs=weight_em_weights
                )
        else:
            self.weight_meta_em = FullPrecisionModel(
                logger, name="weight_meta_em", inputs={}
            )

    def _build_meta_em(self, logger, inputs):
        self.meta_em = FullPrecisionModel(logger, name="meta_em", inputs=inputs)

    def _set_split_params(self, params):
        self.param_split_types = {}

        for k, v in params.items():
            if v is not None:
                assert isinstance(v, nn.Parameter), "Inputs should be nn.Parameter."
                if is_grid_param(v):
                    self.param_split_types[k] = "grid"
                elif v.ndim >= 2:
                    self.param_split_types[k] = "weight"
                else:
                    self.param_split_types[k] = "others"
            else:
                self.param_split_types[k] = "others"

    def _split_params(self, params):
        # Split parameters into grids and weights
        grids = {}
        weights = {}
        others = {}

        for k, v in params.items():
            param_split_type = self.param_split_types[k]
            if param_split_type == "grid":
                grids[k] = v
            elif param_split_type == "weight":
                weights[k] = v
            else:
                others[k] = v

        return grids, weights, others

    def _forward_rates_params(self, x=None, forward_rates=True, forward_params=True):
        # check delta mode
        in_delta_mode = getattr(self, "_delta_mode", False)

        # Grid meta params (delta or full)
        grid_meta_em_params = self.grid_meta_em_param_model()
        grid_meta_em_params_rate, grid_meta_em_params_q = self.meta_em(
            grid_meta_em_params, forward_rates=forward_rates, forward_params=True
        )
        grid_meta_em_params_full = (
            {
                k: self._base_grid_meta_em_params.get(k, torch.zeros_like(grid_meta_em_params_q[k]))
                + grid_meta_em_params_q[k]
                for k in grid_meta_em_params_q
            }
            if in_delta_mode
            else grid_meta_em_params_q
        )

        # Grid em params
        grid_em_params = self.grid_em_param_model()
        grid_em_params_rate, grid_em_params_q = self.grid_meta_em(
            grid_em_params,
            grid_meta_em_params_full,
            forward_rates=forward_rates,
            forward_params=True,
        )
        grid_em_params_full = (
            {
                k: self._base_grid_em_params.get(k, torch.zeros_like(grid_em_params_q[k]))
                + grid_em_params_q[k]
                for k in grid_em_params_q
            }
            if in_delta_mode
            else grid_em_params_q
        )

        # Weight meta params
        weight_meta_em_params = self.weight_meta_em_param_model()
        weight_meta_em_params_rate, weight_meta_em_params_q = self.meta_em(
            weight_meta_em_params, forward_rates=forward_rates, forward_params=True
        )
        weight_meta_em_params_full = (
            {
                k: self._base_weight_meta_em_params.get(k, torch.zeros_like(weight_meta_em_params_q[k]))
                + weight_meta_em_params_q[k]
                for k in weight_meta_em_params_q
            }
            if in_delta_mode
            else weight_meta_em_params_q
        )

        # Weight em params
        weight_em_params = self.weight_em_param_model()
        weight_em_params_rate, weight_em_params_q = self.weight_meta_em(
            weight_em_params,
            weight_meta_em_params_full,
            forward_rates=forward_rates,
            forward_params=True,
        )
        weight_em_params_full = (
            {
                k: self._base_weight_em_params.get(k, torch.zeros_like(weight_em_params_q[k]))
                + weight_em_params_q[k]
                for k in weight_em_params_q
            }
            if in_delta_mode
            else weight_em_params_q
        )

        # Model params (delta or full)
        params = {k: v for k, v in self.param_model().items() if v is not None}
        grids, weights, others = self._split_params(params)
        grids_rate, grids_q = self.grid_em(
            grids,
            grid_em_params_full,
            forward_rates=forward_rates,
            forward_params=forward_params,
        )
        weights_rate, weights_q = self.weight_em(
            weights,
            weight_em_params_full,
            forward_rates=forward_rates,
            forward_params=forward_params,
        )
        others_rate, others_q = self.meta_em(
            others, forward_rates=forward_rates, forward_params=forward_params
        )

        # in delta mode, add base to quantized params for forward pass
        if in_delta_mode and forward_params:
            base_grids, base_weights, base_others = self._split_params(self._base_model_params)
            if grids_q is not None:
                grids_q = {
                    k: base_grids.get(k, torch.zeros_like(grids_q[k])) + grids_q[k]
                    for k in grids_q
                }
            if weights_q is not None:
                weights_q = {
                    k: base_weights.get(k, torch.zeros_like(weights_q[k])) + weights_q[k]
                    for k in weights_q
                }
            if others_q is not None:
                others_q = {
                    k: base_others.get(k, torch.zeros_like(others_q[k])) + others_q[k]
                    for k in others_q
                }

        # Total rate
        if forward_rates:
            rates = {
                k: (
                    torch.tensor(v, device=next(iter(params.values())).device)
                    if not isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in {
                    "grid": grids_rate,
                    "weight": weights_rate,
                    "grid_em_params": grid_em_params_rate,
                    "weight_em_params": weight_em_params_rate,
                    "grid_meta_em_params": grid_meta_em_params_rate,
                    "weight_meta_em_params": weight_meta_em_params_rate,
                    "others": others_rate,
                }.items()
            }
        else:
            rates = None

        #  All params
        if forward_params:
            if self.config.optimize.optimize_rate:
                params_q = {**grids_q, **weights_q, **others_q}
            else:
                if in_delta_mode:
                    base_grids, base_weights, base_others = self._split_params(self._base_model_params)
                    grids_full = {
                        k: base_grids.get(k, torch.zeros_like(grids[k])) + grids[k] for k in grids
                    }
                    weights_full = {
                        k: base_weights.get(k, torch.zeros_like(weights[k])) + weights[k] for k in weights
                    }
                    others_full = {
                        k: base_others.get(k, torch.zeros_like(others[k])) + others[k] for k in others
                    }
                    params_q = {**grids_full, **weights_full, **others_full}
                else:
                    params_q = {**grids, **weights, **others}
        else:
            params_q = None

        return rates, params_q

    def _get_rate(self, rates, reduce_rates):
        if reduce_rates:
            return sum(rates.values())
        else:
            return rates

    def _get_rate_loss(self, rates, reduce_rates):
        if reduce_rates:
            return (
                rates["grid"]
                + rates["weight"]
                + rates["grid_em_params"]
                + rates["weight_em_params"]
                + rates["grid_meta_em_params"]
                + rates["weight_meta_em_params"]
                + rates["others"]
            )
        else:
            return {
                "grid": rates["grid"],
                "weight": rates["weight"],
                "grid_em_params": rates["grid_em_params"],
                "weight_em_params": rates["weight_em_params"],
                "grid_meta_em_params": rates["grid_meta_em_params"],
                "weight_meta_em_params": rates["weight_meta_em_params"],
                "others": rates["others"],
            }

    def get_num_params(self):
        grid_meta_em_params = self.grid_meta_em_param_model()
        grid_em_params = self.grid_em_param_model()
        weight_meta_em_params = self.weight_meta_em_param_model()
        weight_em_params = self.weight_em_param_model()
        params = self.param_model()
        grids, weights, others = self._split_params(params)
        num_params = {
            k: sum([p.numel() if p is not None else 0 for p in v.values()])
            for k, v in {
                "grid": grids,
                "weight": weights,
                "grid_em_params": grid_em_params,
                "weight_em_params": weight_em_params,
                "grid_meta_em_params": grid_meta_em_params,
                "weight_meta_em_params": weight_meta_em_params,
                "others": others,
            }.items()
        }
        return num_params

    def set_epoch(self, epoch, num_epochs):
        epoch_ratio = epoch / (num_epochs - 1) if num_epochs > 1 else 0.0
        init_bit_width = (
            self.config.optimize.init_bit_width
            if (not self.config.optimize.optimize_rate or epoch == 0)
            and self.config.optimize.init_bit_width != -1
            else None
        )
        soft_round_temp = (
            self.config.optimize.soft_round_temp[0]
            + (
                self.config.optimize.soft_round_temp[1]
                - self.config.optimize.soft_round_temp[0]
            )
            * epoch_ratio
        )
        kum_noise = (
            self.config.optimize.kum_noise[0]
            + (self.config.optimize.kum_noise[1] - self.config.optimize.kum_noise[0])
            * epoch_ratio
        )
        quant_noise = (
            self.config.optimize.quant_noise[0]
            + (
                self.config.optimize.quant_noise[1]
                - self.config.optimize.quant_noise[0]
            )
            * epoch_ratio
        )

        # Model params
        params = self.param_model()
        grid, weight, _ = self._split_params(params)
        if isinstance(self.grid_em, _EntropyModelBase):
            self.grid_em.set_params(
                grid,
                init_bit_width,
                soft_round_temp,
                kum_noise,
                quant_noise,
                self.grid_em_param_model(),
            )
        if isinstance(self.weight_em, _EntropyModelBase):
            self.weight_em.set_params(
                weight,
                init_bit_width,
                soft_round_temp,
                kum_noise,
                quant_noise,
                self.weight_em_param_model(),
            )

        # Grid em params
        if isinstance(self.grid_meta_em, _EntropyModelBase):
            self.grid_meta_em.set_params(
                self.grid_em_param_model(),
                init_bit_width,
                soft_round_temp,
                kum_noise,
                quant_noise,
                self.grid_meta_em_param_model(),
            )

        # Weight em params
        if isinstance(self.weight_meta_em, _EntropyModelBase):
            self.weight_meta_em.set_params(
                self.weight_em_param_model(),
                init_bit_width,
                soft_round_temp,
                kum_noise,
                quant_noise,
                self.weight_meta_em_param_model(),
            )

    def set_iter(self, iter, num_iters):
        return self.set_epoch(iter, num_iters)

    def set_zero(self):
        for model in [
            self.param_model,
            self.grid_em_param_model,
            self.weight_em_param_model,
            self.grid_meta_em_param_model,
            self.weight_meta_em_param_model,
            self.model,
            self.grid_em,
            self.weight_em,
            self.grid_meta_em,
            self.weight_meta_em,
        ]:
            for param in model.parameters():
                param.data.zero_()

    def compress(self, bs):
        self.eval()

        with DeterministicContext():
            with torch.no_grad():
                # Grid meta params
                grid_meta_em_params = self.grid_meta_em_param_model()
                bs, _, grid_meta_em_params = self.meta_em.compress(
                    bs, grid_meta_em_params
                )

                # Weight meta params
                weight_meta_em_params = self.weight_meta_em_param_model()
                bs, _, weight_meta_em_params = self.meta_em.compress(
                    bs, weight_meta_em_params
                )

                # Grid em params
                grid_em_params = self.grid_em_param_model()
                bs, _, grid_em_params = self.grid_meta_em.compress(
                    bs, grid_em_params, grid_meta_em_params
                )

                # Weight em params
                weight_em_params = self.weight_em_param_model()
                bs, _, weight_em_params = self.weight_meta_em.compress(
                    bs, weight_em_params, weight_meta_em_params
                )

                # Model params
                params = self.param_model()
                grid, weight, others = self._split_params(params)
                bs, _, _ = self.grid_em.compress(bs, grid, grid_em_params)
                bs, _, _ = self.weight_em.compress(bs, weight, weight_em_params)
                bs, _, _ = self.meta_em.compress(bs, others)

        return bs

    def decompress(self, bs):
        self.eval()

        with DeterministicContext():
            with torch.no_grad():
                # Grid meta params
                grid_meta_em_params = self.grid_meta_em_param_model()
                bs, _, grid_meta_em_params = self.meta_em.decompress(
                    bs, grid_meta_em_params
                )

                # Weight meta params
                weight_meta_em_params = self.weight_meta_em_param_model()
                bs, _, weight_meta_em_params = self.meta_em.decompress(
                    bs, weight_meta_em_params
                )

                # Grid em params
                grid_em_params = self.grid_em_param_model()
                bs, _, grid_em_params = self.grid_meta_em.decompress(
                    bs, grid_em_params, grid_meta_em_params
                )

                # Weight em params
                weight_em_params = self.weight_em_param_model()
                bs, _, weight_em_params = self.weight_meta_em.decompress(
                    bs, weight_em_params, weight_meta_em_params
                )

                # Model params
                params = self.param_model()
                grid, weight, others = self._split_params(params)
                bs, _, grid = self.grid_em.decompress(bs, grid, grid_em_params)
                bs, _, weight = self.weight_em.decompress(bs, weight, weight_em_params)
                bs, _, others = self.meta_em.decompress(bs, others)

                self.grid_meta_em_param_model.load(grid_meta_em_params)
                self.weight_meta_em_param_model.load(weight_meta_em_params)
                self.grid_em_param_model.load(grid_em_params)
                self.weight_em_param_model.load(weight_em_params)
                self.param_model.load({**grid, **weight, **others})

        return bs

    def get_rate_stat(self):
        # IMPORTANT: Set model to eval mode to use hard quantization (same as compress)
        # Otherwise training mode uses soft quantization which underestimates rate
        original_training_state = self.training
        self.eval()
        
        with torch.no_grad():
            rates, _ = self._forward_rates_params(
                x=None, forward_rates=True, forward_params=False
            )
            rates = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in rates.items()
            }
            num_params = self.get_num_params()
            num_params = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in num_params.items()
            }
            bits_per_params = {
                k: rates[k] / num_params[k] if num_params[k] > 0 else math.nan
                for k in rates.keys()
            }
        
        # Restore original training state
        self.train(original_training_state)
        
        return rates, num_params, bits_per_params

    def convert_output_yuv444_to_yuv420(self, output_yuv444: torch.Tensor):
        from io_utils import yuv444_to_yuv420

        assert output_yuv444.ndim == 5, f"expected 5D tensor [N, 3, T, H, W], got shape {output_yuv444.shape}"
        assert output_yuv444.shape[1] == 3, f"expected 3 channels (yuv), got {output_yuv444.shape[1]}"

        N, C, T, H, W = output_yuv444.shape

        # process each frame: reshape [N, 3, T, H, W] -> [N*T, 3, H, W]
        yuv444_frames = output_yuv444.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        yuv420_frames = yuv444_to_yuv420(yuv444_frames, mode="avg_pool")

        return yuv420_frames


class BaseCodecWithEncoder(BaseCodec):
    def __init__(
        self,
        logger,
        model,
        config,
        encoder,
        align_corners: bool = True,
        init_grids_from_encoder: bool = False,
        debug: bool = False,
    ):
        super().__init__(logger, model, config)
        self.encoder = encoder
        self._enc_align_corners = align_corners
        self.cache_enabled = False
        self._init_grids_from_encoder = init_grids_from_encoder
        self._grids_initialized = False
        self._use_encoder_grids_in_forward = True           # whether to inject encoder grids every forward (pretrain) or rely on learnable grids (overfit)
        self.debug = debug  # Enable debug logging (timing, shapes, etc.)
        self._build_grid_key_mapping()

    def _build_grid_key_mapping(self):
        """
        build mapping from encoder level indices to param keys.
        encoder level n-1 (last) -> base encoding grids
        encoder level 0..n-2 -> local grids for decoder stages

        also extracts channel info for proper splitting of encoder outputs.

        keys use underscore format: "decoder_hi_encodings_0__grids_0"
        (module hierarchy separated by "_", param name after "__")
        """
        params = self.param_model()
        grids, _, _ = self._split_params(params)

        ### base grids: sorted by grid index, store (key, channel_count)
        self._base_grid_info = []  # list of (key, channels)
        ### local grids: dict[stage_idx] -> list of (key, channels)
        self._local_grid_info_by_stage = {}

        for k, g in grids.items():
            grid_idx = None
            if "__grids_" in k:
                # extract the number after "grids_"
                import re
                match = re.search(r"__grids_(\d+)", k)
                if match:
                    grid_idx = int(match.group(1))

            ### base encoding grids: key contains "base_encoding"
            if "base_encoding" in k:
                channels = g.shape[-1]  # last dim is always channels
                self._base_grid_info.append((k, channels, grid_idx))

            ### local grids (hi_encodings): key contains "hi_encodings_N"
            elif "hi_encodings_" in k:
                channels = g.shape[-1]
                # extract stage index from "hi_encodings_N" pattern
                import re
                match = re.search(r"hi_encodings_(\d+)", k)
                if match:
                    stage_idx = int(match.group(1))
                    if stage_idx not in self._local_grid_info_by_stage:
                        self._local_grid_info_by_stage[stage_idx] = []
                    self._local_grid_info_by_stage[stage_idx].append((k, channels, grid_idx))

        ### sort by grid index for consistent splitting
        self._base_grid_info.sort(key=lambda x: x[2] if x[2] is not None else 0)
        for stage_idx in self._local_grid_info_by_stage:
            self._local_grid_info_by_stage[stage_idx].sort(
                key=lambda x: x[2] if x[2] is not None else 0
            )

        ### compute expected encoder channels
        self._expected_base_channels = sum(info[1] for info in self._base_grid_info)
        self._expected_local_channels_by_stage = {
            stage: sum(info[1] for info in infos)
            for stage, infos in self._local_grid_info_by_stage.items()
        }

    def _inject_encoder_grids(self, codec_grids: Dict[str, torch.Tensor], enc_grids: List[torch.Tensor]):
        """
        Replace codec grid tensors with encoder outputs, preserving batch dimension.

        Encoder outputs are split to match individual grid channels:
        - encoder level n-1 (last) -> split into base encoding grids
        - encoder level 0..n-2 -> split into local grids for decoder stages

        For base grids:
          Input: [B, C, T, H, W] -> Output: [B, T, H, W, C_slice] (batched)
        For local grids:
          Input: [B, C, T, H, W] -> Output: [B, T, 1, 1, C_slice] (batched, spatial pooled)
        """
        if not enc_grids:
            return codec_grids

        new_grids: Dict[str, torch.Tensor] = {}
        n_enc_levels = len(enc_grids)
        needed_local = len(self._local_grid_info_by_stage)
        if n_enc_levels < needed_local + 1:  # +1 for base level
            raise RuntimeError(
                f"encoder outputs insufficient levels: need {needed_local}+1, got {n_enc_levels}"
            )

        ### process base grids from encoder last level
        base_enc = enc_grids[-1]  # [B, C, T, H, W]
        if base_enc.ndim != 5:
            raise ValueError(f"expected encoder base grid [B,C,T,H,W], got shape {base_enc.shape}")

        B = base_enc.shape[0]
        base_enc_c = base_enc.shape[1]
        if base_enc_c != self._expected_base_channels:
            raise ValueError(
                f"encoder base output has {base_enc_c} channels, "
                f"but base grids expect {self._expected_base_channels} channels "
                f"(sum of {[info[1] for info in self._base_grid_info]}). "
                f"set encoder channels correctly!"
            )

        ### convert [B, C, T, H, W] -> [B, T, H, W, C] (keep batch dimension!)
        base_enc_bthwc = base_enc.permute(0, 2, 3, 4, 1).contiguous()

        ### split base encoder output into individual grids
        ch_offset = 0
        for key, channels, grid_idx in self._base_grid_info:
            target_shape = codec_grids[key].shape  # [T, H, W, C] or [T, H, W, ...]
            t_out, h_out, w_out = target_shape[0], target_shape[1], target_shape[2]

            ### slice channels: [B, T, H, W, C_slice]
            grid_slice = base_enc_bthwc[..., ch_offset:ch_offset + channels]
            ch_offset += channels

            ### interpolate spatial/temporal if needed
            t_in, h_in, w_in = grid_slice.shape[1:4]
            if (t_in, h_in, w_in) != (t_out, h_out, w_out):
                # [B, T, H, W, C] -> [B, C, T, H, W]
                grid_5d = grid_slice.permute(0, 4, 1, 2, 3)
                grid_5d = F.interpolate(
                    grid_5d, size=(t_out, h_out, w_out),
                    mode="trilinear", align_corners=self._enc_align_corners
                )
                # [B, C, T, H, W] -> [B, T, H, W, C]
                grid_slice = grid_5d.permute(0, 2, 3, 4, 1).contiguous()

            # Keep batch dimension for downstream per-sample handling
            new_grids[key] = grid_slice  # [B, T, H, W, C]
        if ch_offset != base_enc_c:
            raise RuntimeError(
                f"base encoder channels consumed {ch_offset} but encoder provided {base_enc_c}"
            )

        ### process local grids from encoder levels 0..n-2
        for stage_idx, grid_infos in self._local_grid_info_by_stage.items():
            if stage_idx >= n_enc_levels - 1:
                raise RuntimeError(
                    f"encoder outputs insufficient levels for local stage {stage_idx}"
                )

            local_enc = enc_grids[stage_idx]  # [B, C, T, H, W]
            if local_enc.ndim != 5:
                raise ValueError(
                    f"expected encoder local grid [B,C,T,H,W], got shape {local_enc.shape}"
                )

            local_enc_c = local_enc.shape[1]
            expected_c = self._expected_local_channels_by_stage.get(stage_idx, 0)
            if local_enc_c != expected_c:
                raise ValueError(
                    f"encoder level {stage_idx} has {local_enc_c} channels, "
                    f"but local grids at stage {stage_idx} expect {expected_c} channels "
                    f"(sum of {[info[1] for info in grid_infos]}). "
                    f"set encoder channels correctly!"
                )

            ### local grids have spatial size 1x1, so average pool spatially
            ### [B, C, T, H, W] -> [B, C, T] -> [B, T, C]
            local_enc_pooled = local_enc.mean(dim=(3, 4))  # [B, C, T]
            local_enc_btc = local_enc_pooled.permute(0, 2, 1)  # [B, T, C]
            ### expand to [B, T, 1, 1, C]
            local_enc_bt11c = local_enc_btc.unsqueeze(2).unsqueeze(2)  # [B, T, 1, 1, C]

            ### split into individual grids
            ch_offset = 0
            for key, channels, grid_idx in grid_infos:
                target_shape = codec_grids[key].shape  # [T, 1, 1, ...] or [T, 1, 1, s_t, s_h, s_w, C]

                ### slice channels: [B, T, 1, 1, C_slice]
                grid_slice = local_enc_bt11c[..., ch_offset:ch_offset + channels]
                ch_offset += channels

                ### interpolate temporal if needed
                t_in = grid_slice.shape[1]
                t_out = target_shape[0]
                if t_in != t_out:
                    ### [B, T, 1, 1, C] -> [B, C, T, 1, 1]
                    grid_5d = grid_slice.permute(0, 4, 1, 2, 3)
                    grid_5d = F.interpolate(
                        grid_5d, size=(t_out, 1, 1),
                        mode="trilinear", align_corners=self._enc_align_corners
                    )
                    ### [B, C, T, 1, 1] -> [B, T, 1, 1, C]
                    grid_slice = grid_5d.permute(0, 2, 3, 4, 1).contiguous()

                ### expand stride dimensions if needed: [B, T, 1, 1, C] -> [B, T, 1, 1, s_t, s_h, s_w, C]
                if len(target_shape) > 4:
                    stride_dims = target_shape[3:-1]  # (s_t, s_h, s_w)
                    # insert stride dims before C to preserve channel layout, no reshuffling
                    grid_slice_expanded = grid_slice
                    for _ in stride_dims:
                        grid_slice_expanded = grid_slice_expanded.unsqueeze(-2)
                    grid_slice = grid_slice_expanded.expand(
                        grid_slice_expanded.shape[0],
                        grid_slice_expanded.shape[1],
                        1,
                        1,
                        *stride_dims,
                        grid_slice_expanded.shape[-1],
                    ).contiguous()
                    grid_slice = grid_slice.reshape(grid_slice.shape[0], *target_shape)

                # Keep batch dimension
                new_grids[key] = grid_slice
            if ch_offset != local_enc_c:
                raise RuntimeError(
                    f"encoder level {stage_idx} consumed {ch_offset} channels but provided {local_enc_c}"
                )

        ### sanity check: all grids should be replaced and have batch dimension
        for k in codec_grids.keys():
            if k not in new_grids:
                raise RuntimeError(f"grid {k} was not injected from encoder!")
            orig_shape = codec_grids[k].shape
            new_shape = new_grids[k].shape
            # new grid should have one more dimension (batch dim) than original
            if new_shape[1:] != orig_shape:
                raise RuntimeError(
                    f"grid {k} shape mismatch after injection!\n"
                    f"  original (learnable): {tuple(orig_shape)}\n"
                    f"  injected (batched):   {tuple(new_shape)}\n"
                    f"  expected injected:    ({new_shape[0]}, {', '.join(map(str, orig_shape))})"
                )

        return new_grids

    def initialize_grids_from_encoder(self, video: torch.Tensor):
        """
        initialize learnable grid params from encoder output.
        used for overfit mode after pretraining.
        """
        if self._grids_initialized:
            return

        self.logger.info(f"[Encoder->Grid Init] Initializing grids from encoder (video shape: {tuple(video.shape)})")
        
        with torch.no_grad():
            enc_grids = self.encoder(video)
            if not isinstance(enc_grids, (list, tuple)):
                raise ValueError(
                    f"encoder should return list/tuple of grids, got {type(enc_grids)}"
                )

            params = self.param_model()
            grids, _, _ = self._split_params(params)
            new_grids = self._inject_encoder_grids(grids, list(enc_grids))

            ### update param model with encoder-initialized grids
            for k, v in new_grids.items():
                target_param = grids.get(k, None)
                if target_param is None:
                    continue

                if v.ndim == target_param.ndim + 1:
                    # batched encoder grid -> require batch=1 for init to avoid ambiguity
                    if v.shape[0] != 1:
                        raise ValueError(
                            f"encoder init expects batch=1 when copying grids, got batch={v.shape[0]} for {k}"
                        )
                    v_to_copy = v[0]
                elif v.shape == target_param.shape:
                    v_to_copy = v
                else:
                    raise ValueError(
                        f"encoder init shape mismatch for {k}: encoder {tuple(v.shape)} vs param {tuple(target_param.shape)}"
                    )

                target_param.data.copy_(v_to_copy.to(target_param.device, target_param.dtype))

        self._grids_initialized = True
        self.clear_cache()
        self.logger.info(f"[Encoder->Grid Init] Done! Initialized {len(new_grids)} grids from encoder output")

    def freeze_encoder(self):
        """freeze encoder weights for overfit mode."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_decoder_weights(self):
        """freeze decoder weights (not grids) for overfit mode."""
        params = self.param_model()
        _, weights, others = self._split_params(params)
        for p in list(weights.values()) + list(others.values()):
            if isinstance(p, nn.Parameter):
                p.requires_grad_(False)

    def freeze_grids(self):
        """freeze grid parameters."""
        params = self.param_model()
        grids, _, _ = self._split_params(params)
        for p in grids.values():
            if isinstance(p, nn.Parameter):
                p.requires_grad_(False)

    def unfreeze_grids(self):
        """unfreeze grid parameters for overfit training."""
        params = self.param_model()
        grids, _, _ = self._split_params(params)
        for p in grids.values():
            if isinstance(p, nn.Parameter):
                p.requires_grad_(True)

    def unfreeze_decoder_weights(self):
        """unfreeze decoder weights."""
        params = self.param_model()
        _, weights, others = self._split_params(params)
        for p in list(weights.values()) + list(others.values()):
            if isinstance(p, nn.Parameter):
                p.requires_grad_(True)

    def freeze_entropy_models(self):
        """freeze all entropy model parameters (grid_em, weight_em, and their meta models)."""
        for model in [self.grid_em, self.weight_em, self.grid_meta_em, self.weight_meta_em]:
            for param in model.parameters():
                param.requires_grad = False
        # Also freeze the param models
        for param_model in [self.grid_em_param_model, self.weight_em_param_model, 
                            self.grid_meta_em_param_model, self.weight_meta_em_param_model]:
            param_model.set_param_requires_grad(False)

    def unfreeze_entropy_models(self):
        """unfreeze all entropy model parameters."""
        for model in [self.grid_em, self.weight_em, self.grid_meta_em, self.weight_meta_em]:
            for param in model.parameters():
                param.requires_grad = True
        # Also unfreeze the param models
        for param_model in [self.grid_em_param_model, self.weight_em_param_model, 
                            self.grid_meta_em_param_model, self.weight_meta_em_param_model]:
            param_model.set_param_requires_grad(True)

    def set_pretrain_mode(self, stage: str = "pretrain_s3"):
        """
        Set up for pretrain mode with stage-specific freezing.
        
        Stages:
        - pretrain_s1 (distortion-only): freeze entropy models, train encoder/decoder
        - pretrain_s2 (rate-only): freeze encoder/decoder, train entropy models
        - pretrain_s3 (joint): train all components
        """
        self._init_grids_from_encoder = False
        self._use_encoder_grids_in_forward = True
        self._grids_initialized = False
        
        if stage == "pretrain_s1":
            # S1: distortion-only - train encoder/decoder, freeze entropy models
            self.unfreeze_encoder()
            self.unfreeze_decoder_weights()
            self.freeze_entropy_models()
            self.logger.info("[Pretrain S1] Freeze entropy models, train encoder/decoder")
        elif stage == "pretrain_s2":
            # S2: rate-only - freeze encoder/decoder, train entropy models
            self.freeze_encoder()
            self.freeze_decoder_weights()
            self.unfreeze_entropy_models()
            self.logger.info("[Pretrain S2] Freeze encoder/decoder, train entropy models")
        else:
            # S3: joint - train all
            self.unfreeze_encoder()
            self.unfreeze_decoder_weights()
            self.unfreeze_entropy_models()
            self.logger.info("[Pretrain S3] Train all components")
        
        self.clear_cache()

    def set_overfit_mode(self, freeze_decoder: bool = True):
        """
        set up for overfit mode.
        - freeze encoder
        - optionally freeze decoder weights
        - use encoder to initialize grids
        - grids become learnable
        """
        self.freeze_encoder()
        if freeze_decoder:
            self.freeze_decoder_weights()
        self._init_grids_from_encoder = True
        self._use_encoder_grids_in_forward = False
        self._grids_initialized = False
        self.clear_cache()

    ### delta update helpers
    def _apply_delta(self, param_model, delta: Dict[str, torch.Tensor]):
        state = param_model.state_dict()
        for k, v in delta.items():
            if k not in state:
                continue
            state[k] = state[k] + v.to(state[k].device, state[k].dtype)
        param_model.load_state_dict(state)
        self.clear_cache()

    def apply_param_updates(
        self,
        model_delta: Dict[str, torch.Tensor] | None = None,
        grid_em_delta: Dict[str, torch.Tensor] | None = None,
        weight_em_delta: Dict[str, torch.Tensor] | None = None,
        grid_meta_delta: Dict[str, torch.Tensor] | None = None,
        weight_meta_delta: Dict[str, torch.Tensor] | None = None,
    ):
        if model_delta:
            self._apply_delta(self.param_model, model_delta)
        if grid_em_delta:
            self._apply_delta(self.grid_em_param_model, grid_em_delta)
        if weight_em_delta:
            self._apply_delta(self.weight_em_param_model, weight_em_delta)
        if grid_meta_delta:
            self._apply_delta(self.grid_meta_em_param_model, grid_meta_delta)
        if weight_meta_delta:
            self._apply_delta(self.weight_meta_em_param_model, weight_meta_delta)

    def _forward_rates_params(self, x=None, forward_rates=True, forward_params=True):
        ### accept dict inputs (as used by tasks) or raw tensors
        if isinstance(x, dict):
            video = x.get("x", None)
        else:
            video = x

        ### initialize grids from encoder if configured (for overfit mode)
        if self._init_grids_from_encoder and video is not None:
            self.initialize_grids_from_encoder(video)

        use_encoder_grids = self._use_encoder_grids_in_forward and video is not None

        if not use_encoder_grids:
            # fallback to vanilla (learnable) grids
            return super()._forward_rates_params(
                x=x, forward_rates=forward_rates, forward_params=forward_params
            )

        ### encoder grids (list of [B, C, T, H, W])
        enc_grids = self.encoder(video)
        if not isinstance(enc_grids, (list, tuple)):
            raise ValueError(
                f"encoder should return a list/tuple of grids, got {type(enc_grids)}."
            )
        
        ### monitor encoder outputs for NaN/Inf
        for i, g in enumerate(enc_grids):
            if torch.isnan(g).any() or torch.isinf(g).any():
                self.logger.warning(
                    f"[Encoder] NaN/Inf detected in grid {i}, shape={tuple(g.shape)}, "
                    f"nan_count={torch.isnan(g).sum().item()}, inf_count={torch.isinf(g).sum().item()}"
                )
            if self.debug:
                self.logger.info(
                    f"[Encoder] Grid {i}: min={g.min().item():.4f}, max={g.max().item():.4f}, "
                    f"mean={g.mean().item():.4f}, std={g.std().item():.4f}"
                )

        ### grid meta params
        grid_meta_em_params = self.grid_meta_em_param_model()
        grid_meta_em_params_rate, grid_meta_em_params_q = self.meta_em(
            grid_meta_em_params, forward_rates=forward_rates, forward_params=True
        )

        ### grid em params
        grid_em_params = self.grid_em_param_model()
        grid_em_params_rate, grid_em_params_q = self.grid_meta_em(
            grid_em_params,
            grid_meta_em_params_q,
            forward_rates=forward_rates,
            forward_params=True,
        )

        ### weight meta params
        weight_meta_em_params = self.weight_meta_em_param_model()
        weight_meta_em_params_rate, weight_meta_em_params_q = self.meta_em(
            weight_meta_em_params, forward_rates=forward_rates, forward_params=True
        )

        ### weight em params
        weight_em_params = self.weight_em_param_model()
        weight_em_params_rate, weight_em_params_q = self.weight_meta_em(
            weight_em_params,
            weight_meta_em_params_q,
            forward_rates=forward_rates,
            forward_params=True,
        )

        ### model params - inject encoder grids
        params = self.param_model()
        grids, weights, others = self._split_params(params)

        # inject encoder grids (batched: [B, ... , C])
        runtime_grids = self._inject_encoder_grids(grids, list(enc_grids))

        # sanity check batch dimension
        batch_size = video.shape[0]
        for k, g in runtime_grids.items():
            if g is None:
                continue
            if g.shape[0] != batch_size:
                raise ValueError(
                    f"grid {k} has batch {g.shape[0]}, expected {batch_size}"
                )

        # Entropy model for grids (per-sample to keep entropy reshape logic correct)
        grids_q = None
        grids_rate = None
        if forward_params or forward_rates:
            grids_q_list = [] if forward_params else None
            total_rate = None
            for n in range(batch_size):
                grids_n = {k: (v[n] if v is not None else None) for k, v in runtime_grids.items()}
                rate_n, grids_q_n = self.grid_em(
                    grids_n,
                    grid_em_params_q,
                    forward_rates=forward_rates,
                    forward_params=forward_params,
                    skip_reshape=False,
                )
                if forward_rates:
                    total_rate = rate_n if total_rate is None else total_rate + rate_n
                if forward_params:
                    grids_q_list.append(grids_q_n)

            # average over batch to get per-sample rate (since video_size is per-sample)
            grids_rate = (total_rate / batch_size) if forward_rates else None
            if forward_params and grids_q_list:
                grids_q = {
                    k: torch.stack([gq[k] for gq in grids_q_list], dim=0)
                    for k in grids_q_list[0].keys()
                }

        weights_rate, weights_q = self.weight_em(
            weights,
            weight_em_params_q,
            forward_rates=forward_rates,
            forward_params=forward_params,
        )

        others_rate, others_q = self.meta_em(
            others, forward_rates=forward_rates, forward_params=forward_params
        )

        ### total rate
        if forward_rates:
            rates = {
                k: (
                    torch.tensor(v, device=next(iter(params.values())).device)
                    if not isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in {
                    "grid": grids_rate,
                    "weight": weights_rate,
                    "grid_em_params": grid_em_params_rate,
                    "weight_em_params": weight_em_params_rate,
                    "grid_meta_em_params": grid_meta_em_params_rate,
                    "weight_meta_em_params": weight_meta_em_params_rate,
                    "others": others_rate,
                }.items()
            }
        else:
            rates = None

        ### all params
        if forward_params:
            use_quantized = (
                self.config.optimize.optimize_rate
                and forward_rates
                and grids_q is not None
            )
            if use_quantized:
                params_q = {**grids_q, **weights_q, **others_q}
            else:
                params_q = {
                    **(grids_q if grids_q is not None else runtime_grids),
                    **weights,
                    **others,
                }
        else:
            params_q = None

        return rates, params_q

    def get_num_params(self):
        """get number of parameters including encoder."""
        # Get base class num_params
        num_params = super().get_num_params()

        # Add encoder parameters
        if hasattr(self, "encoder") and self.encoder is not None:
            num_params["encoder"] = sum(p.numel() for p in self.encoder.parameters())

        return num_params

     ### delta mode support
    def setup_delta_mode(self):
        """
        set up delta mode: save current params as base, reset param_models to zero.
        after this, param_model() returns deltas, and forward uses base + delta.
        call this after loading pretrained model.
        """
        self._delta_mode = True
        # filter out None entries to avoid issues with stale checkpoints
        def _filter_params(pdict):
            return {k: v for k, v in pdict.items() if v is not None}

        with torch.no_grad():
            # save base params (frozen)
            self._base_model_params = {k: v.clone() for k, v in _filter_params(self.param_model()).items()}
            self._base_grid_em_params = {k: v.clone() for k, v in _filter_params(self.grid_em_param_model()).items()}
            self._base_weight_em_params = {k: v.clone() for k, v in _filter_params(self.weight_em_param_model()).items()}
            self._base_grid_meta_em_params = {k: v.clone() for k, v in _filter_params(self.grid_meta_em_param_model()).items()}
            self._base_weight_meta_em_params = {k: v.clone() for k, v in _filter_params(self.weight_meta_em_param_model()).items()}

            # reset param_models to zero (they now store deltas)
            self.param_model.load({k: torch.zeros_like(v) for k, v in self._base_model_params.items()})
            self.grid_em_param_model.load({k: torch.zeros_like(v) for k, v in self._base_grid_em_params.items()})
            self.weight_em_param_model.load({k: torch.zeros_like(v) for k, v in self._base_weight_em_params.items()})
            self.grid_meta_em_param_model.load({k: torch.zeros_like(v) for k, v in self._base_grid_meta_em_params.items()})
            self.weight_meta_em_param_model.load({k: torch.zeros_like(v) for k, v in self._base_weight_meta_em_params.items()})

        # Ensure all param_models have requires_grad=True for training
        self.param_model.set_param_requires_grad(True)
        self.grid_em_param_model.set_param_requires_grad(True)
        self.weight_em_param_model.set_param_requires_grad(True)
        self.grid_meta_em_param_model.set_param_requires_grad(True)
        self.weight_meta_em_param_model.set_param_requires_grad(True)

        self.clear_cache()

    def exit_delta_mode(self):
        """exit delta mode, restore full params = base + current delta."""
        if not getattr(self, "_delta_mode", False):
            return
        with torch.no_grad():
            # restore full params
            delta = {k: v for k, v in self.param_model().items() if v is not None}
            full = {}
            for k in set(self._base_model_params.keys()) | set(delta.keys()):
                base_v = self._base_model_params.get(k)
                delta_v = delta.get(k)
                if base_v is None and delta_v is None:
                    continue
                if base_v is None:
                    full[k] = delta_v
                elif delta_v is None:
                    full[k] = base_v
                else:
                    full[k] = base_v + delta_v
            if full:
                self.param_model.load(full)

            delta = {k: v for k, v in self.grid_em_param_model().items() if v is not None}
            full = {}
            for k in set(self._base_grid_em_params.keys()) | set(delta.keys()):
                base_v = self._base_grid_em_params.get(k)
                delta_v = delta.get(k)
                if base_v is None and delta_v is None:
                    continue
                if base_v is None:
                    full[k] = delta_v
                elif delta_v is None:
                    full[k] = base_v
                else:
                    full[k] = base_v + delta_v
            if full:
                self.grid_em_param_model.load(full)

            delta = {k: v for k, v in self.weight_em_param_model().items() if v is not None}
            full = {}
            for k in set(self._base_weight_em_params.keys()) | set(delta.keys()):
                base_v = self._base_weight_em_params.get(k)
                delta_v = delta.get(k)
                if base_v is None and delta_v is None:
                    continue
                if base_v is None:
                    full[k] = delta_v
                elif delta_v is None:
                    full[k] = base_v
                else:
                    full[k] = base_v + delta_v
            if full:
                self.weight_em_param_model.load(full)

            delta = {k: v for k, v in self.grid_meta_em_param_model().items() if v is not None}
            full = {}
            for k in set(self._base_grid_meta_em_params.keys()) | set(delta.keys()):
                base_v = self._base_grid_meta_em_params.get(k)
                delta_v = delta.get(k)
                if base_v is None and delta_v is None:
                    continue
                if base_v is None:
                    full[k] = delta_v
                elif delta_v is None:
                    full[k] = base_v
                else:
                    full[k] = base_v + delta_v
            if full:
                self.grid_meta_em_param_model.load(full)

            delta = {k: v for k, v in self.weight_meta_em_param_model().items() if v is not None}
            full = {}
            for k in set(self._base_weight_meta_em_params.keys()) | set(delta.keys()):
                base_v = self._base_weight_meta_em_params.get(k)
                delta_v = delta.get(k)
                if base_v is None and delta_v is None:
                    continue
                if base_v is None:
                    full[k] = delta_v
                elif delta_v is None:
                    full[k] = base_v
                else:
                    full[k] = base_v + delta_v
            if full:
                self.weight_meta_em_param_model.load(full)

        self._delta_mode = False
        self.clear_cache()

    def get_full_params(self) -> Dict[str, torch.Tensor]:
        """get full params (base + delta) for forward pass."""
        if not getattr(self, "_delta_mode", False):
            return self.param_model()
        delta = {k: v for k, v in self.param_model().items() if v is not None}
        return {
            k: self._base_model_params.get(k, torch.zeros_like(delta[k])) + delta[k]
            for k in delta
        }

    def get_full_em_params(self, which: str) -> Dict[str, torch.Tensor]:
        """get full em params (base + delta)."""
        if not getattr(self, "_delta_mode", False):
            if which == "grid_em":
                return self.grid_em_param_model()
            elif which == "weight_em":
                return self.weight_em_param_model()
            elif which == "grid_meta_em":
                return self.grid_meta_em_param_model()
            elif which == "weight_meta_em":
                return self.weight_meta_em_param_model()
            else:
                raise ValueError(f"unknown em type: {which}")

        if which == "grid_em":
            delta = {k: v for k, v in self.grid_em_param_model().items() if v is not None}
            return {
                k: self._base_grid_em_params.get(k, torch.zeros_like(delta[k])) + delta[k]
                for k in delta
            }
        elif which == "weight_em":
            delta = {k: v for k, v in self.weight_em_param_model().items() if v is not None}
            return {
                k: self._base_weight_em_params.get(k, torch.zeros_like(delta[k])) + delta[k]
                for k in delta
            }
        elif which == "grid_meta_em":
            delta = {k: v for k, v in self.grid_meta_em_param_model().items() if v is not None}
            return {
                k: self._base_grid_meta_em_params.get(k, torch.zeros_like(delta[k])) + delta[k]
                for k in delta
            }
        elif which == "weight_meta_em":
            delta = {k: v for k, v in self.weight_meta_em_param_model().items() if v is not None}
            return {
                k: self._base_weight_meta_em_params.get(k, torch.zeros_like(delta[k])) + delta[k]
                for k in delta
            }
        else:
            raise ValueError(f"unknown em type: {which}")

    def compress_delta(self, bs):
        if not getattr(self, "_delta_mode", False):
            raise RuntimeError("compress_delta requires delta mode. call setup_delta_mode() first.")

        import time
        times = {}
        
        self.eval()
        with DeterministicContext():
            with torch.no_grad():
                # Get deltas from param_models
                t0 = time.time()
                grid_meta_delta = self.grid_meta_em_param_model()
                weight_meta_delta = self.weight_meta_em_param_model()
                grid_em_delta = self.grid_em_param_model()
                weight_em_delta = self.weight_em_param_model()
                model_delta = self.param_model()
                times["get_deltas"] = time.time() - t0
                
                # Log delta statistics
                grid_delta, weight_delta, others_delta = self._split_params(model_delta)
                if self.debug and hasattr(self, "logger") and self.logger is not None:
                    self.logger.info("="*60)
                    self.logger.info("GRID PARAMETERS DETAILED BREAKDOWN:")
                    self.logger.info("="*60)
                    
                    total_grid_elems = 0
                    # Group by type
                    base_grids = {}
                    local_grids = {}
                    
                    for k, v in grid_delta.items():
                        if not torch.is_tensor(v):
                            continue
                        if "base_encoding" in k:
                            base_grids[k] = v
                        else:
                            local_grids[k] = v
                        total_grid_elems += v.numel()
                    
                    self.logger.info(f"\nBASE ENCODING GRIDS ({len(base_grids)} params):")
                    base_total = 0
                    for k, v in sorted(base_grids.items()):
                        self.logger.info(f"  {k:60s}: {str(v.shape):30s} = {v.numel():12,} elements")
                        base_total += v.numel()
                    self.logger.info(f"  Base total: {base_total:,} elements = {base_total/1e6:.2f}M")
                    
                    self.logger.info(f"\nLOCAL HIERARCHICAL GRIDS ({len(local_grids)} params):")
                    local_total = 0
                    for k, v in sorted(local_grids.items()):
                        self.logger.info(f"  {k:60s}: {str(v.shape):30s} = {v.numel():12,} elements")
                        local_total += v.numel()
                    self.logger.info(f"  Local total: {local_total:,} elements = {local_total/1e6:.2f}M")
                    
                    total_weight_elems = sum(v.numel() for v in weight_delta.values() if torch.is_tensor(v))
                    self.logger.info(f"\nTOTAL SUMMARY:")
                    self.logger.info(f"  Grid params: {len(grid_delta)}, {total_grid_elems:,} elements = {total_grid_elems/1e6:.2f}M")
                    self.logger.info(f"  Weight params: {len(weight_delta)}, {total_weight_elems:,} elements = {total_weight_elems/1e6:.2f}M")
                    self.logger.info("="*60)

                # ----------------------------------------------------------------
                # NOTE: compress meta params first, then use quantized values for context

                # Compress grid meta params delta (no context needed)
                t0 = time.time()
                bs, _, grid_meta_delta_q = self.meta_em.compress(bs, grid_meta_delta)
                times["compress_grid_meta"] = time.time() - t0

                # Compress weight meta params delta (no context needed)
                t0 = time.time()
                bs, _, weight_meta_delta_q = self.meta_em.compress(bs, weight_meta_delta)
                times["compress_weight_meta"] = time.time() - t0

                # Build context using QUANTIZED meta deltas
                t0 = time.time()
                grid_meta_full = {k: self._base_grid_meta_em_params[k] + grid_meta_delta_q[k] 
                                  for k in grid_meta_delta_q if grid_meta_delta_q[k] is not None}
                weight_meta_full = {k: self._base_weight_meta_em_params[k] + weight_meta_delta_q[k] 
                                    for k in weight_meta_delta_q if weight_meta_delta_q[k] is not None}
                times["build_meta_context"] = time.time() - t0

                # Compress grid em params delta (use quantized meta params as context)
                t0 = time.time()
                bs, _, grid_em_delta_q = self.grid_meta_em.compress(bs, grid_em_delta, grid_meta_full)
                times["compress_grid_em"] = time.time() - t0

                # Compress weight em params delta (use quantized meta params as context)
                t0 = time.time()
                bs, _, weight_em_delta_q = self.weight_meta_em.compress(bs, weight_em_delta, weight_meta_full)
                times["compress_weight_em"] = time.time() - t0

                # Build context using quantized em deltas
                t0 = time.time()
                grid_em_full = {k: self._base_grid_em_params[k] + grid_em_delta_q[k] 
                                for k in grid_em_delta_q if grid_em_delta_q[k] is not None}
                weight_em_full = {k: self._base_weight_em_params[k] + weight_em_delta_q[k] 
                                  for k in weight_em_delta_q if weight_em_delta_q[k] is not None}
                times["build_em_context"] = time.time() - t0

                # Compress model params delta (use quantized em params as context)
                t0 = time.time()
                grid_delta, weight_delta, others_delta = self._split_params(model_delta)
                times["split_params"] = time.time() - t0
                
                t0 = time.time()
                bs, _, grid_delta_q = self.grid_em.compress(bs, grid_delta, grid_em_full)
                times["compress_grid"] = time.time() - t0
                
                t0 = time.time()
                bs, _, weight_delta_q = self.weight_em.compress(bs, weight_delta, weight_em_full)
                times["compress_weight"] = time.time() - t0
                
                t0 = time.time()
                bs, _, others_delta_q = self.meta_em.compress(bs, others_delta)
                times["compress_others"] = time.time() - t0

                # Store quantized deltas back to param_models
                # so get_full_params() returns quantized values after compress
                t0 = time.time()
                self.grid_meta_em_param_model.load(grid_meta_delta_q)
                self.weight_meta_em_param_model.load(weight_meta_delta_q)
                self.grid_em_param_model.load(grid_em_delta_q)
                self.weight_em_param_model.load(weight_em_delta_q)
                self.param_model.load({**grid_delta_q, **weight_delta_q, **others_delta_q})
                times["store_quantized"] = time.time() - t0

        self.clear_cache()
        
        # Log timing breakdown
        total_time = sum(times.values())
        if self.debug and hasattr(self, "logger") and self.logger is not None:
            self.logger.info(f"compress_delta timing breakdown (total: {total_time:.3f}s):")
            for k, v in sorted(times.items(), key=lambda x: -x[1]):
                self.logger.info(f"  {k:25s}: {v:6.3f}s ({v/total_time*100:5.1f}%)")
        
        return bs

    def decompress_delta(self, bs):
        """
        decompress deltas and apply to base.
        after decompression, param_models contain deltas.
        then we add base to get full params.
        """
        if not getattr(self, "_delta_mode", False):
            raise RuntimeError("decompress_delta requires delta mode. call setup_delta_mode() first.")

        import time
        times = {}
        
        self.eval()
        with DeterministicContext():
            with torch.no_grad():
                # decode deltas using existing decompress logic
                # grid meta params delta
                t0 = time.time()
                delta_template = {k: torch.zeros_like(v) for k, v in self._base_grid_meta_em_params.items()}
                bs, _, grid_meta_delta = self.meta_em.decompress(bs, delta_template)
                times["decompress_grid_meta"] = time.time() - t0

                # weight meta params delta
                t0 = time.time()
                delta_template = {k: torch.zeros_like(v) for k, v in self._base_weight_meta_em_params.items()}
                bs, _, weight_meta_delta = self.meta_em.decompress(bs, delta_template)
                times["decompress_weight_meta"] = time.time() - t0

                # grid em params delta (use full meta params as context)
                t0 = time.time()
                grid_meta_full = {k: self._base_grid_meta_em_params[k] + grid_meta_delta[k] for k in grid_meta_delta}
                times["build_grid_meta_context"] = time.time() - t0
                
                t0 = time.time()
                delta_template = {k: torch.zeros_like(v) for k, v in self._base_grid_em_params.items()}
                bs, _, grid_em_delta = self.grid_meta_em.decompress(bs, delta_template, grid_meta_full)
                times["decompress_grid_em"] = time.time() - t0

                # weight em params delta
                t0 = time.time()
                weight_meta_full = {k: self._base_weight_meta_em_params[k] + weight_meta_delta[k] for k in weight_meta_delta}
                times["build_weight_meta_context"] = time.time() - t0
                
                t0 = time.time()
                delta_template = {k: torch.zeros_like(v) for k, v in self._base_weight_em_params.items()}
                bs, _, weight_em_delta = self.weight_meta_em.decompress(bs, delta_template, weight_meta_full)
                times["decompress_weight_em"] = time.time() - t0

                # model params delta
                t0 = time.time()
                grid_em_full = {k: self._base_grid_em_params[k] + grid_em_delta[k] for k in grid_em_delta}
                weight_em_full = {k: self._base_weight_em_params[k] + weight_em_delta[k] for k in weight_em_delta}
                times["build_em_context"] = time.time() - t0

                t0 = time.time()
                base_grid, base_weight, base_others = self._split_params(self._base_model_params)
                grid_template = {k: torch.zeros_like(v) for k, v in base_grid.items()}
                weight_template = {k: torch.zeros_like(v) for k, v in base_weight.items()}
                others_template = {k: torch.zeros_like(v) for k, v in base_others.items()}
                times["prepare_templates"] = time.time() - t0

                t0 = time.time()
                bs, _, grid_delta = self.grid_em.decompress(bs, grid_template, grid_em_full)
                times["decompress_grid"] = time.time() - t0
                
                t0 = time.time()
                bs, _, weight_delta = self.weight_em.decompress(bs, weight_template, weight_em_full)
                times["decompress_weight"] = time.time() - t0
                
                t0 = time.time()
                bs, _, others_delta = self.meta_em.decompress(bs, others_template)
                times["decompress_others"] = time.time() - t0

                # store deltas in param_models
                t0 = time.time()
                self.grid_meta_em_param_model.load(grid_meta_delta)
                self.weight_meta_em_param_model.load(weight_meta_delta)
                self.grid_em_param_model.load(grid_em_delta)
                self.weight_em_param_model.load(weight_em_delta)
                self.param_model.load({**grid_delta, **weight_delta, **others_delta})
                times["store_deltas"] = time.time() - t0

        self.clear_cache()
        
        # Log timing breakdown
        total_time = sum(times.values())
        if self.debug and hasattr(self, "logger") and self.logger is not None:
            self.logger.info(f"decompress_delta timing breakdown (total: {total_time:.3f}s):")
            for k, v in sorted(times.items(), key=lambda x: -x[1]):
                self.logger.info(f"  {k:30s}: {v:6.3f}s ({v/total_time*100:5.1f}%)")
        
        return bs

    def reset_deltas(self):
        """reset all deltas to zero (for next video in batch processing)."""
        if not getattr(self, "_delta_mode", False):
            return
        with torch.no_grad():
            if self._base_model_params:
                self.param_model.load({k: torch.zeros_like(v) for k, v in self._base_model_params.items()})
            if self._base_grid_em_params:
                self.grid_em_param_model.load({k: torch.zeros_like(v) for k, v in self._base_grid_em_params.items()})
            if self._base_weight_em_params:
                self.weight_em_param_model.load({k: torch.zeros_like(v) for k, v in self._base_weight_em_params.items()})
            if self._base_grid_meta_em_params:
                self.grid_meta_em_param_model.load({k: torch.zeros_like(v) for k, v in self._base_grid_meta_em_params.items()})
            if self._base_weight_meta_em_params:
                self.weight_meta_em_param_model.load({k: torch.zeros_like(v) for k, v in self._base_weight_meta_em_params.items()})
        self.clear_cache()

    def get_delta_state(self):
        """Get current delta parameters for saving."""
        if not getattr(self, "_delta_mode", False):
            raise RuntimeError("get_delta_state requires delta mode")
        
        # Return state_dict (plain dict) instead of ParameterDict objects
        # so they can be loaded with load_state_dict later
        def to_state_dict(param_model):
            params = param_model()
            if params is None:
                return None
            return {k: v.detach().clone() for k, v in params.items()}
        
        return {
            "model": to_state_dict(self.param_model),
            "grid_em": to_state_dict(self.grid_em_param_model),
            "weight_em": to_state_dict(self.weight_em_param_model),
            "grid_meta_em": to_state_dict(self.grid_meta_em_param_model),
            "weight_meta_em": to_state_dict(self.weight_meta_em_param_model),
        }

    def load_delta_state(self, delta_state):
        """Load delta parameters from saved state."""
        if not getattr(self, "_delta_mode", False):
            raise RuntimeError("load_delta_state requires delta mode")
        
        with torch.no_grad():
            if "model" in delta_state and delta_state["model"]:
                self.param_model.load(delta_state["model"])
            if "grid_em" in delta_state and delta_state["grid_em"]:
                self.grid_em_param_model.load(delta_state["grid_em"])
            if "weight_em" in delta_state and delta_state["weight_em"]:
                self.weight_em_param_model.load(delta_state["weight_em"])
            if "grid_meta_em" in delta_state and delta_state["grid_meta_em"]:
                self.grid_meta_em_param_model.load(delta_state["grid_meta_em"])
            if "weight_meta_em" in delta_state and delta_state["weight_meta_em"]:
                self.weight_meta_em_param_model.load(delta_state["weight_meta_em"])
        
        self.clear_cache()


def build_compress_model(logger, model, cfg, model_cfg=None, stage=None):
    """
    build compression model.
    
    Args:
        logger: logger instance
        model: HiNeRV model
        cfg: compression config dict
        model_cfg: optional model config for encoder channel auto-computation
        stage: training stage (s1, s2, pretrain_s1, pretrain_s2, pretrain_s3)
               if pretrain stage, will call set_pretrain_mode with appropriate freezing
    """
    ### optionally wrap with encoder-generated grids
    encoder_cfg = cfg.get("encoder", None) if isinstance(cfg, dict) else None

    # extract decoder_cfg for backward compatibility
    decoder_cfg = model_cfg.get("decoder", None) if model_cfg else None

    if encoder_cfg is not None:
        from modules.encoder import (
            ConvNeXtHierarchicalEncoder,
            ELICTemporalConvNeXtEncoder,
            EncoderConfig,
            compute_encoder_channels_from_grids,
        )

        if not isinstance(encoder_cfg, dict):
            # encoder is already an encoder instance
            enc = encoder_cfg
            init_from_enc = False
        else:
            # determine encoder type
            encoder_type = encoder_cfg.get("type", "convnext").lower()
            init_from_enc = encoder_cfg.get("init_grids_from_encoder", False)

            # keys to exclude from encoder-specific config
            exclude_keys = {"type", "init_grids_from_encoder"}

            logger.info(f"     Encoder type: {encoder_type}")
            logger.info(f"     Encoder config: {encoder_cfg}")

            if encoder_type == "convnext":
                enc_channels = encoder_cfg.get("channels", None)
                if enc_channels == "auto" or enc_channels is None:
                    # auto-compute encoder channels from model config
                    if model_cfg is None:
                        raise ValueError(
                            "encoder channels='auto' requires model_cfg to be provided. "
                            "set explicit channels list or pass model config."
                        )

                    base_encoding_cfg = model_cfg.get("base_encoding", {})
                    decoder_cfg_inner = model_cfg.get("decoder", {})

                    # get grid parameters
                    base_grid_size = base_encoding_cfg.get("base_grid_size", [200, 45, 80, 4])
                    base_grid_level = base_encoding_cfg.get("base_grid_level", 4)
                    base_grid_level_scale = base_encoding_cfg.get("base_grid_level_scale", [2., 2., 2., 0.5])
                    local_grid_size = decoder_cfg_inner.get("local_grid_size", [-1, 16])
                    local_grid_level = decoder_cfg_inner.get("local_grid_level", 3)
                    local_grid_level_scale = decoder_cfg_inner.get("local_grid_level_scale", [2., 0.5])
                    local_grid_depth_scale = decoder_cfg_inner.get("local_grid_depth_scale", [1., 0.5])

                    # count number of local stages (decoder stages with hi_encodings)
                    use_hi_encoding = decoder_cfg_inner.get("use_hi_encoding", [True, True, True, True])
                    num_local_stages = sum(use_hi_encoding)

                    enc_channels = compute_encoder_channels_from_grids(
                        base_grid_size=base_grid_size,
                        base_grid_level=base_grid_level,
                        base_grid_level_scale=base_grid_level_scale,
                        local_grid_size=local_grid_size,
                        local_grid_level=local_grid_level,
                        local_grid_level_scale=local_grid_level_scale,
                        local_grid_depth_scale=local_grid_depth_scale,
                        num_local_stages=num_local_stages,
                    )
                    logger.info(f"     Auto-computed encoder channels: {enc_channels}")

                    # update encoder_cfg with computed channels
                    encoder_cfg = dict(encoder_cfg)  # make a copy
                    encoder_cfg["channels"] = enc_channels
                    # don't exclude channels - we want to pass the computed value

                # validate encoder-decoder stride consistency
                if decoder_cfg is not None and validate_encoder_decoder_config is not None:
                    validate_encoder_decoder_config(encoder_cfg, decoder_cfg, logger)

                # compute min chunk size for logging
                if compute_min_chunk_size is not None:
                    t_strides = encoder_cfg.get("temporal_strides", [1, 1, 2, 2])
                    s_strides = encoder_cfg.get("spatial_strides", [3, 2, 2, 2])
                    min_chunk = compute_min_chunk_size(t_strides, s_strides)
                    logger.info(f"     Min video chunk size: {min_chunk}")

                # build encoder config (exclude non-EncoderConfig keys)
                encoder_cfg_clean = {
                    k: v for k, v in encoder_cfg.items() if k not in exclude_keys
                }
                enc = ConvNeXtHierarchicalEncoder(EncoderConfig(**encoder_cfg_clean))

            elif encoder_type == "elic":
                # ELICTemporalConvNeXtEncoder
                encoder_cfg_clean = {
                    k: v for k, v in encoder_cfg.items() if k not in exclude_keys
                }
                
                # compute min chunk size for logging
                if compute_min_chunk_size is not None:
                    t_strides = encoder_cfg.get("temporal_strides", [1, 1, 1, 1])
                    s_strides = [2, 2, 2, 2]  # ELIC uses fixed spatial stride of 2
                    min_chunk = compute_min_chunk_size(t_strides, s_strides)
                    logger.info(f"     Min video chunk size: {min_chunk}")
                
                enc = ELICTemporalConvNeXtEncoder(**encoder_cfg_clean)

            else:
                raise ValueError(
                    f"Unknown encoder type: {encoder_type}. "
                    f"Supported types: 'convnext', 'elic'"
                )

        cfg_no_enc = {k: v for k, v in cfg.items() if k not in ["encoder", "debug"]}
        debug = cfg.get("debug", False)

        codec = BaseCodecWithEncoder(
            logger,
            model,
            config=BaseConfig(**cfg_no_enc),
            encoder=enc,
            init_grids_from_encoder=init_from_enc,
            debug=debug,
        )
        
        # Set up pretrain mode with stage-specific freezing
        if stage and stage.startswith("pretrain_"):
            codec.set_pretrain_mode(stage)
        
        return codec

    ### default: vanilla codec (no encoder, no debug for base BaseCodec)
    codec = BaseCodec(logger, model, config=BaseConfig(**cfg))
    
    # Set up pretrain mode with stage-specific freezing
    if stage and stage.startswith("pretrain_"):
        codec.set_pretrain_mode(stage)
    
    return codec
