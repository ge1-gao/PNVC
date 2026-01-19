import os
import sys
import copy
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from omegaconf import OmegaConf, DictConfig

# config directory relative to this file
CONFIG_DIR = Path(__file__).parent / 'configs'


def get_config_dir() -> Path:
    """get the config directory path"""
    return CONFIG_DIR


def load_yaml(path: Union[str, Path]) -> DictConfig:
    """load a yaml file as omegaconf dictconfig"""
    return OmegaConf.load(path)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """merge multiple configs, later ones override earlier ones"""
    return OmegaConf.merge(*configs)


def load_base_configs(cfg: DictConfig) -> DictConfig:
    """load base configs based on selections in cfg"""
    base_dir = CONFIG_DIR / 'base'
    
    # load model config based on size, allow type override
    model_size = cfg.get('model', {}).get('size', 'l')
    model_base = load_yaml(base_dir / 'model.yaml')
    if model_size in model_base:
        model_cfg = OmegaConf.create({
            'type': model_base[model_size].get('type', 'HiNeRV'),
            'config': model_base[model_size].get('config', {})
        })
    else:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(model_base.keys())}")

    # explicit model.type in cfg takes precedence (e.g., PNVC/GIViC/Base)
    if cfg.get('model', {}).get('type') is not None:
        model_cfg.type = cfg.model.type
    
    # load compress config based on stage
    stage = cfg.get('train', {}).get('stage', 's1')

    # map pretrain stages to base config stages
    # pretrain_s1: distortion-only -> use s1 (ste quant, no rate optimization)
    # pretrain_s2: rate-only -> use s2 (quant-noise, rate optimization)
    # pretrain_s3: joint -> use s2 (quant-noise, rate optimization)
    config_stage = stage
    if stage.startswith('pretrain_'):
        if stage == 'pretrain_s1':
            config_stage = 's1'  # distortion-only: ste quant, no rate opt
        else:
            config_stage = 's2'  # rate/joint: quant-noise, rate opt

    compress_base = load_yaml(base_dir / 'compress.yaml')
    if config_stage in compress_base:
        compress_cfg = compress_base[config_stage]
    else:
        raise ValueError(f"Unknown stage: {stage} (mapped to {config_stage}). Available: {list(compress_base.keys())}")
    
    # load train config based on stage
    train_base = load_yaml(base_dir / 'train.yaml')
    if config_stage in train_base:
        train_cfg = train_base[config_stage]
    else:
        train_cfg = OmegaConf.create({})
    
    # load task config
    task_base = load_yaml(base_dir / 'task.yaml')
    train_task_cfg = task_base.get('train', {})
    eval_task_cfg = task_base.get('eval', {})
    
    return OmegaConf.create({
        'model': model_cfg,
        'compress_model': compress_cfg,
        'train': train_cfg,
        'train_task': train_task_cfg,
        'eval_task': eval_task_cfg,
    })


def create_cli_parser() -> argparse.ArgumentParser:
    """create argument parser for cli overrides"""
    parser = argparse.ArgumentParser(
        description='Base/PNVC training script with simplified config system',
        allow_abbrev=False
    )
    
    # mode selection
    parser.add_argument('--mode', type=str, default='overfit', choices=['pretrain', 'overfit'],
                        help='Training mode: pretrain (multi-video) or overfit (single video)')
    
    # preset selection
    parser.add_argument('--preset', type=str, default=None,
                        help='Preset config name (e.g., uvg_iscas26, mcl_jcv_iscas26)')
    parser.add_argument('--stage', type=str, default='s1', 
                        choices=['s1', 's2', 's1_checkerboard', 's2_checkerboard', 'pretrain_s1', 'pretrain_s2', 'pretrain_s3'],
                        help='Training stage: s1/s2/s1_checkerboard/s2_checkerboard for overfit, pretrain_s1/s2/s3 for pretrain. '
                             's2_checkerboard=checkerboard with rate optimization (recommended for overfit)')
    
    # pretrain mode arguments
    parser.add_argument('--video-list', type=str, default=None,
                        help='Path to video list file for pretrain mode')
    parser.add_argument('--crop-size', type=int, nargs=2, default=[256, 256],
                        help='Spatial crop size for pretrain [H, W]')
    parser.add_argument('--temporal-size', type=int, default=16,
                        help='Temporal size for pretrain')
    parser.add_argument('--samples-per-video', type=str, default='50',
                        help='Samples per video for pretrain (int or "auto" for dynamic)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for pretrain mode')
    # default None so we only override when explicitly provided; otherwise presets/task.lamb stay intact
    parser.add_argument('--lamb', type=float, default=None,
                        help='Lambda for rate-distortion tradeoff (pretrain mode)')
    
    # common overrides
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--exp-name', type=str, default=None, help='Experiment name')
    
    # data overrides
    parser.add_argument('--data.dataset', dest='data_dataset', type=str, default=None,
                        help='Dataset name (e.g., Beauty, HoneyBee)')
    parser.add_argument('--data.dataset_dir', dest='data_dataset_dir', type=str, default=None,
                        help='Dataset directory')
    parser.add_argument('--data.fmt', dest='data_fmt', type=str, default=None,
                        help='Video format (png, yuv420p)')
    parser.add_argument('--data.yuv_fmt', dest='data_yuv_fmt', type=str, default=None,
                        help='YUV pixel format (e.g., yuv420p, yuv420p10le); overrides parsed fmt from name')
    parser.add_argument('--data.yuv_size', dest='data_yuv_size', type=int, nargs=3, default=None,
                        metavar=('T', 'H', 'W'),
                        help='YUV size [T, H, W]; overrides parsed size from name')
    
    # model overrides
    parser.add_argument('--model.type', dest='model_type', type=str, default=None,
                        help='Model name/type (e.g., HiNeRV, PNVC, GIViC, Base)')
    parser.add_argument('--model.size', dest='model_size', type=str, default=None,
                        choices=['xs', 's', 'm', 'l', 'xxl'],
                        help='Model size')
    
    # task overrides
    parser.add_argument('--task.lamb', dest='task_lamb', type=float, default=None,
                        help='Lambda for rate-distortion tradeoff')
    
    # training overrides
    parser.add_argument('--train.lr', dest='train_lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--train.warmup_lr', dest='train_warmup_lr', type=float, default=None,
                        help='Warmup learning rate')
    parser.add_argument('--warmup_lr', dest='train_warmup_lr', type=float, default=None,
                        help='Alias for --train.warmup_lr')
    parser.add_argument('--train.min_lr', dest='train_min_lr', type=float, default=None,
                        help='Minimum learning rate')
    parser.add_argument('--min_lr', dest='train_min_lr', type=float, default=None,
                        help='Alias for --train.min_lr')
    parser.add_argument('--train.epochs', dest='train_epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--epochs', dest='train_epochs', type=int, default=None,
                        help='Alias for --train.epochs')
    parser.add_argument('--train.auto_lr_scaling', dest='train_auto_lr_scaling', type=str, default=None,
                        help='Enable/disable auto LR scaling (true/false)')
    parser.add_argument('--auto_lr_scaling', dest='train_auto_lr_scaling', type=str, default=None,
                        help='Alias for --train.auto_lr_scaling')
    parser.add_argument('--train.max_norm', dest='train_max_norm', type=float, default=None,
                        help='Gradient clipping max norm')
    parser.add_argument('--max_norm', dest='train_max_norm', type=float, default=None,
                        help='Alias for --train.max_norm')
    parser.add_argument('--train.batch_size', dest='train_batch_size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--train.grad_accum', dest='train_grad_accum', type=int, default=None,
                        help='Gradient accumulation steps')
    parser.add_argument('--train.patch_size', dest='train_patch_size', type=int, nargs=3, default=None,
                        help='Training patch size [T, H, W]')
    
    # compress_model overrides
    parser.add_argument('--optimize-rate', dest='optimize_rate', type=str, default=None,
                        help='Enable rate optimization (true/false)')
    
    # resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--resume-model-only', type=str, default='false',
                        help='Resume model weights only (true/false)')
    parser.add_argument('--use-delta-mode', type=str, default='auto',
                        help='Use delta mode for overfit (auto/true/false). auto: enable if --resume is set')
    
    # misc
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--workers', type=int, default=None, help='Number of data workers')
    
    # legacy config file support (for backward compatibility)
    parser.add_argument('--config', type=str, default=None,
                        help='Legacy: path to full config yaml file')

    # logging
    parser.add_argument('--train-enable-log', type=str, default=None,
                        help='Enable training output logging (true/false)')
    parser.add_argument('--eval-enable-log', type=str, default=None,
                        help='Enable eval output logging (true/false)')
    parser.add_argument('--log-epochs', type=int, default=None,
                        help='Log every N epochs')
    parser.add_argument('--log-level', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging level (default: info)')

    return parser


def parse_cli_overrides(args: argparse.Namespace) -> DictConfig:
    """convert cli args to omegaconf dict for merging"""
    overrides = {}

    # mode
    if args.mode is not None:
        overrides['mode'] = args.mode

    # pretrain settings
    if args.video_list is not None:
        overrides['video_list'] = args.video_list
    if args.crop_size is not None:
        overrides['crop_size'] = args.crop_size
    if args.temporal_size is not None:
        overrides['temporal_size'] = args.temporal_size
    if args.samples_per_video is not None:
        overrides['samples_per_video'] = args.samples_per_video
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.lamb is not None:
        overrides['lamb'] = args.lamb

    # experiment
    if args.output is not None:
        overrides.setdefault('experiment', {})['output'] = args.output
    if args.exp_name is not None:
        overrides.setdefault('experiment', {})['exp_name'] = args.exp_name
    if args.seed is not None:
        overrides.setdefault('experiment', {})['seed'] = args.seed
    if args.workers is not None:
        overrides.setdefault('experiment', {})['workers'] = args.workers

    # data
    if args.data_dataset is not None:
        overrides.setdefault('data', {})['dataset'] = args.data_dataset
    if args.data_dataset_dir is not None:
        overrides.setdefault('data', {})['dataset_dir'] = args.data_dataset_dir
    if args.data_fmt is not None:
        overrides.setdefault('data', {})['fmt'] = args.data_fmt
    if args.data_yuv_fmt is not None:
        overrides.setdefault('data', {})['yuv_fmt'] = args.data_yuv_fmt
    if args.data_yuv_size is not None:
        overrides.setdefault('data', {})['yuv_size'] = args.data_yuv_size

    # model
    if args.model_type is not None:
        overrides.setdefault('model', {})['type'] = args.model_type
    if args.model_size is not None:
        overrides.setdefault('model', {})['size'] = args.model_size

    # task
    if args.task_lamb is not None:
        overrides.setdefault('task', {})['lamb'] = args.task_lamb

    # training
    if args.stage is not None:
        overrides.setdefault('train', {})['stage'] = args.stage
    if args.train_lr is not None:
        overrides.setdefault('train', {})['lr'] = args.train_lr
    if args.train_warmup_lr is not None:
        overrides.setdefault('train', {})['warmup_lr'] = args.train_warmup_lr
    if args.train_min_lr is not None:
        overrides.setdefault('train', {})['min_lr'] = args.train_min_lr
    if args.train_epochs is not None:
        overrides.setdefault('train', {})['epochs'] = args.train_epochs
    if getattr(args, 'train_auto_lr_scaling', None) is not None:
        overrides.setdefault('train', {})['auto_lr_scaling'] = bool(str(args.train_auto_lr_scaling).lower() == 'true')
    if args.train_max_norm is not None:
        overrides.setdefault('train', {})['max_norm'] = args.train_max_norm
    if args.train_batch_size is not None:
        overrides.setdefault('train', {})['train_batch_size'] = args.train_batch_size
    if args.train_grad_accum is not None:
        overrides.setdefault('train', {})['grad_accum'] = args.train_grad_accum
    if args.train_patch_size is not None:
        overrides.setdefault('train', {})['train_patch_size'] = args.train_patch_size

    # resume
    if args.resume is not None:
        overrides.setdefault('resume', {})['path'] = args.resume
    if args.resume_model_only is not None:
        overrides.setdefault('resume', {})['model_only'] = args.resume_model_only.lower() == 'true'

    # logging
    if args.train_enable_log is not None:
        overrides.setdefault('logging', {})['train_enable_log'] = args.train_enable_log.lower() == 'true'
    if args.eval_enable_log is not None:
        overrides.setdefault('logging', {})['eval_enable_log'] = args.eval_enable_log.lower() == 'true'
    if args.log_epochs is not None:
        overrides.setdefault('logging', {})['log_epochs'] = args.log_epochs
    if args.log_level is not None:
        overrides.setdefault('logging', {})['log_level'] = args.log_level.lower()

    # compress_model overrides
    if getattr(args, 'optimize_rate', None) is not None:
        optimize_cfg = overrides.setdefault('compress_model', {}).setdefault('config', {}).setdefault('optimize', {})
        optimize_cfg['optimize_rate'] = args.optimize_rate.lower() == 'true'

    return OmegaConf.create(overrides)


def load_config(args: Optional[argparse.Namespace] = None) -> DictConfig:
    """
    load configuration with the following priority:
    1. defaults.yaml
    2. base configs (model, compress, train, task)
    3. preset config
    4. cli overrides
    """
    if args is None:
        parser = create_cli_parser()
        args = parser.parse_args()

    # step 1: load defaults
    defaults_path = CONFIG_DIR / 'defaults.yaml'
    if defaults_path.exists():
        cfg = load_yaml(defaults_path)
    else:
        cfg = OmegaConf.create({})

    # step 2: load preset if specified
    if args.preset is not None:
        preset_path = CONFIG_DIR / 'presets' / f'{args.preset}.yaml'
        if preset_path.exists():
            preset_cfg = load_yaml(preset_path)
            cfg = merge_configs(cfg, preset_cfg)
        else:
            raise ValueError(f"Preset not found: {preset_path}")

    # step 3: apply cli overrides (before loading base configs to get correct stage/size)
    cli_overrides = parse_cli_overrides(args)
    cfg = merge_configs(cfg, cli_overrides)

    # step 4: load base configs based on current selections
    base_cfg = load_base_configs(cfg)
    cfg = merge_configs(cfg, base_cfg)

    # step 5: apply cli overrides again (to override base config values)
    cfg = merge_configs(cfg, cli_overrides)

    return cfg


# dataclass adapters to maintain compatibility with existing code
@dataclass
class DataConfig:
    """data configuration - compatible with existing code"""
    dataset_dir: str = '~/Datasets/UVG/1920x1080'
    dataset: str = 'Beauty'
    video_size: list = field(default_factory=lambda: [-1, -1, -1])
    patch_size: list = field(default_factory=lambda: [1, -1, -1])
    fmt: str = 'png'
    yuv_fmt: str = None
    yuv_size: list = field(default_factory=lambda: [-1, -1, -1])


@dataclass
class OverfitTaskConfig:
    """task configuration - compatible with existing code"""
    loss: list = field(default_factory=lambda: [0.7, 'l1', 0.3, 'ms-ssim'])
    metric: list = field(default_factory=lambda: ['psnr', 'ms-ssim'])
    color_space: str = 'RGB'
    lamb: float = 1.0
    enable_log: bool = False
    # Lagrangian BPP constraint
    target_bpp: float = None
    bpp_penalty_weight: float = 1000.0


@dataclass
class ExperimentConfig:
    """experiment configuration - compatible with existing code"""
    # mode
    mode: str = 'overfit'
    stage: str = 's1'  # Training stage: s1, s2, pretrain_s1, pretrain_s2, pretrain_s3

    # pretrain settings
    video_list: str = None
    data_sources: list = None  # List of data source dicts for pretrain
    crop_size: list = field(default_factory=lambda: [256, 256])
    temporal_size: int = 16
    samples_per_video: str = "50"  # int as string, or "auto"
    batch_size: int = 4
    loss: list = field(default_factory=lambda: ['mse'])
    metric: list = field(default_factory=lambda: ['psnr'])
    lamb: float = 1.0

    # training
    epochs: int = 360
    warmup_epochs: int = 30
    eval_epochs: int = 30
    eval_only: bool = False
    rate_steps: int = 8

    # batch size
    grad_accum: int = 1
    train_batch_size: int = 1
    eval_batch_size: int = 1

    # optimizer & scheduler
    opt: str = 'adam'
    sched: str = 'cosine'
    lr: float = 2e-3
    warmup_lr: float = 1e-5
    min_lr: float = 1e-4
    auto_lr_scaling: bool = True
    max_norm: float = 1.0
    norm_type: float = 2.0
    weight_decay: float = 1e-6
    weight_decay_scaling: bool = True
    opt_eps: float = 1e-8
    opt_betas: list = field(default_factory=lambda: [0.9, 0.999])
    momentum: float = 0.9
    decay_milestones: list = field(default_factory=lambda: [0.5, 0.75])
    decay_rate: float = 0.5
    lr_scale: list = field(default_factory=lambda: [])

    # coding config
    start_frame: int = -1
    num_frames: int = -1
    intra_period: int = -1

    # output
    output: str = 'output'
    exp_name: str = None

    # logging
    log_epochs: int = -2
    log_level: str = 'info'
    log_iters: int = -2
    train_enable_log: bool = False
    eval_enable_log: bool = False

    # resuming
    resume: str = None
    resume_model_only: bool = False
    resume_model_pat: list = None

    # misc
    seed: int = 0
    workers: int = 4
    prefetch_factor: int = 4
    pin_mem: bool = True

    # nested configs
    train_data: DataConfig = None
    eval_data: DataConfig = None
    train_task: OverfitTaskConfig = None
    eval_task: OverfitTaskConfig = None
    compress_model: dict = None
    model: dict = None

    # ModMixer mask decay (progressive alpha schedule)
    mask_decay_enable: bool = False
    mask_decay_mode: str = "linear"  # linear | cosine
    mask_decay_start_epoch: int = 0
    mask_decay_end_epoch: int = -1   # -1 -> last epoch
    mask_decay_min_alpha: float = 0.0

    # Distill monitor
    distill_monitor: bool = False  # if true, log teacher-student diff during decay

    def _to_dict(self):
        output = {k: getattr(self, k) for k in self.__dict__}
        output['train_data'] = self.train_data.__dict__ if self.train_data is not None else None
        output['eval_data'] = self.eval_data.__dict__ if self.eval_data is not None else None
        output['train_task'] = self.train_task.__dict__ if self.train_task is not None else None
        output['eval_task'] = self.eval_task.__dict__ if self.eval_task is not None else None
        return output


def _safe_to_container(cfg_value) -> dict:
    """safely convert OmegaConf or dict to plain dict"""
    if cfg_value is None:
        return {}
    if isinstance(cfg_value, dict):
        return cfg_value
    return OmegaConf.to_container(cfg_value, resolve=True)


def omega_to_experiment_config(cfg: DictConfig) -> ExperimentConfig:
    """convert omegaconf dictconfig to experimentconfig for compatibility"""
    # create data configs
    data_cfg = _safe_to_container(cfg.get("data", {}))
    train_data = DataConfig(
        dataset_dir=data_cfg.get("dataset_dir", "~/Datasets/UVG/1920x1080"),
        dataset=data_cfg.get("dataset", "Beauty"),
        video_size=data_cfg.get("video_size", [-1, -1, -1]),
        patch_size=data_cfg.get("patch_size", [1, -1, -1]),
        fmt=data_cfg.get("fmt", "png"),
        yuv_fmt=data_cfg.get("yuv_fmt", None),
        yuv_size=data_cfg.get("yuv_size", [-1, -1, -1]),
    )
    eval_data = copy.deepcopy(train_data)

    # apply train-specific patch size
    train_cfg = _safe_to_container(cfg.get("train", {}))
    if "train_patch_size" in train_cfg:
        train_data.patch_size = train_cfg["train_patch_size"]
    if "eval_patch_size" in train_cfg:
        eval_data.patch_size = train_cfg["eval_patch_size"]

    # create task configs
    task_cfg = _safe_to_container(cfg.get("task", {}))
    train_task_cfg = _safe_to_container(cfg.get("train_task", {}))
    eval_task_cfg = _safe_to_container(cfg.get("eval_task", {}))
    logging_cfg = _safe_to_container(cfg.get("logging", {}))

    lamb = task_cfg.get('lamb', 1.0)

    # choose defaults based on input format (YUV defaults use 4:2:0-weighted losses)
    is_yuv = str(train_data.fmt).startswith('yuv')
    rgb_train_loss = [0.7, 'l1', 0.3, 'ms-ssim-5x5']
    rgb_eval_loss = [0.7, 'l1', 0.3, 'ms-ssim']
    rgb_train_metric = ['psnr', 'ms-ssim-5x5']
    rgb_eval_metric = ['psnr', 'ms-ssim']

    default_train_loss = [0.7, 'l1-yuv611', 0.3, 'ms-ssim-y-5x5'] if is_yuv else rgb_train_loss
    default_eval_loss = [0.7, 'l1-yuv611', 0.3, 'ms-ssim-y'] if is_yuv else rgb_eval_loss
    default_train_metric = (
        ['psnr-yuv611', 'psnr-u', 'psnr-v', 'ms-ssim-y-5x5', 'ms-ssim-u-5x5', 'ms-ssim-v-5x5']
        if is_yuv else rgb_train_metric
    )
    default_eval_metric = (
        ['psnr-yuv611', 'psnr-u', 'psnr-v', 'ms-ssim-y', 'ms-ssim-u', 'ms-ssim-v']
        if is_yuv else rgb_eval_metric
    )
    default_color_space = 'YUV' if is_yuv else 'RGB'

    # pull any user-specified configs first (preset/cli), fall back to defaults
    train_loss_cfg = train_task_cfg.get('loss', None) or task_cfg.get('loss', None)
    eval_loss_cfg = eval_task_cfg.get('loss', None) or task_cfg.get('loss', None)
    train_metric_cfg = train_task_cfg.get('metric', None) or task_cfg.get('metric', None)
    eval_metric_cfg = eval_task_cfg.get('metric', None) or task_cfg.get('metric', None)

    train_loss_cfg = train_loss_cfg if train_loss_cfg is not None else default_train_loss
    eval_loss_cfg = eval_loss_cfg if eval_loss_cfg is not None else default_eval_loss
    train_metric_cfg = train_metric_cfg if train_metric_cfg is not None else default_train_metric
    eval_metric_cfg = eval_metric_cfg if eval_metric_cfg is not None else default_eval_metric

    # for YUV inputs, swap out the RGB defaults if the user did not override them
    if is_yuv:
        if train_loss_cfg == rgb_train_loss:
            train_loss_cfg = default_train_loss
        if eval_loss_cfg == rgb_eval_loss:
            eval_loss_cfg = default_eval_loss
        if train_metric_cfg == rgb_train_metric:
            train_metric_cfg = default_train_metric
        if eval_metric_cfg == rgb_eval_metric:
            eval_metric_cfg = default_eval_metric

    color_space_cfg = task_cfg.get('color_space', default_color_space)
    if is_yuv and color_space_cfg == 'RGB':
        color_space_cfg = 'YUV'

    # Lagrangian BPP constraint
    target_bpp = task_cfg.get('target_bpp', None)
    bpp_penalty_weight = task_cfg.get('bpp_penalty_weight', 1000.0)

    train_task = OverfitTaskConfig(
        loss=train_loss_cfg,
        metric=train_metric_cfg,
        color_space=color_space_cfg,
        lamb=lamb,
        enable_log=logging_cfg.get('train_enable_log', False),
        target_bpp=target_bpp,
        bpp_penalty_weight=bpp_penalty_weight,
    )
    eval_task = OverfitTaskConfig(
        loss=eval_loss_cfg,
        metric=eval_metric_cfg,
        color_space=color_space_cfg,
        lamb=lamb,
        enable_log=logging_cfg.get('eval_enable_log', False),
        target_bpp=target_bpp,
        bpp_penalty_weight=bpp_penalty_weight,
    )

    # get model and compress configs
    model_cfg = _safe_to_container(cfg.get("model", {}))
    compress_cfg = _safe_to_container(cfg.get("compress_model", {}))

    # get experiment and resume configs
    exp_cfg = _safe_to_container(cfg.get("experiment", {}))
    resume_cfg = _safe_to_container(cfg.get("resume", {}))

    # helper to get value from nested cfg or top-level cfg
    def _get(key, default, *cfgs):
        for c in cfgs:
            if key in c:
                return c[key]
        # also check top-level cfg
        val = cfg.get(key, None)
        if val is not None:
            return val
        return default

    # create experiment config
    args = ExperimentConfig(
        # stage - check top-level first, then train section
        stage=_get("stage", "s1", train_cfg),
        
        # training
        epochs=train_cfg.get("epochs", 360),
        warmup_epochs=train_cfg.get("warmup_epochs", 30),
        eval_epochs=train_cfg.get("eval_epochs", 30),
        eval_only=train_cfg.get("eval_only", False),
        rate_steps=train_cfg.get("rate_steps", 8),

        # batch
        grad_accum=train_cfg.get("grad_accum", 1),
        train_batch_size=train_cfg.get("train_batch_size", 1),
        eval_batch_size=train_cfg.get("eval_batch_size", 1),

        # optimizer (ensure float types for learning rates)
        opt=train_cfg.get("opt", "adam"),
        sched=train_cfg.get("sched", "cosine"),
        lr=float(train_cfg.get("lr", 2e-3)),
        warmup_lr=float(train_cfg.get("warmup_lr", 1e-5)),
        min_lr=float(train_cfg.get("min_lr", 1e-4)),
        auto_lr_scaling=train_cfg.get("auto_lr_scaling", True),
        max_norm=float(train_cfg.get("max_norm", 1.0)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
        weight_decay_scaling=train_cfg.get("weight_decay_scaling", True),

        # coding - check top-level first, then train section
        start_frame=_get("start_frame", -1, train_cfg),
        num_frames=_get("num_frames", -1, train_cfg),
        intra_period=_get("intra_period", -1, train_cfg),

        # output - check top-level first, then experiment section
        output=_get("output", "output", exp_cfg),
        exp_name=_get("exp_name", None, exp_cfg),

        # logging
        log_epochs=logging_cfg.get("log_epochs", -2),
        log_level=logging_cfg.get("log_level", "info"),
        train_enable_log=logging_cfg.get("train_enable_log", False),
        eval_enable_log=logging_cfg.get("eval_enable_log", False),

        # resume
        resume=resume_cfg.get("path", None),
        resume_model_only=resume_cfg.get("model_only", False),

        # misc - check top-level first, then experiment section
        seed=_get("seed", 0, exp_cfg),
        workers=_get("workers", 4, exp_cfg),
        prefetch_factor=_get("prefetch_factor", 4, exp_cfg),
        pin_mem=_get("pin_mem", True, exp_cfg),

        # nested
        train_data=train_data,
        eval_data=eval_data,
        train_task=train_task,
        eval_task=eval_task,
        model=model_cfg,
        compress_model=compress_cfg,
        # ModMixer mask decay
        mask_decay_enable=train_cfg.get("mask_decay", {}).get("enable", False),
        mask_decay_mode=train_cfg.get("mask_decay", {}).get("mode", "linear"),
        mask_decay_start_epoch=train_cfg.get("mask_decay", {}).get("start_epoch", 0),
        mask_decay_end_epoch=train_cfg.get("mask_decay", {}).get("end_epoch", -1),
        mask_decay_min_alpha=train_cfg.get("mask_decay", {}).get("min_alpha", 0.0),
        distill_monitor=train_cfg.get("mask_decay", {}).get("distill_monitor", False),
    )

    # pretrain-specific settings (not part of legacy overfit path)
    args.mode = cfg.get('mode', 'overfit')
    args.video_list = cfg.get('video_list', None)
    
    # data_sources for pretrain (list of dicts with base_dir, video_list, scan_subdirs)
    data_sources_cfg = cfg.get('data_sources', None)
    if data_sources_cfg is not None and OmegaConf.is_config(data_sources_cfg):
        data_sources_cfg = OmegaConf.to_container(data_sources_cfg, resolve=True)
    args.data_sources = data_sources_cfg

    crop_size_cfg = cfg.get('crop_size', [256, 256])
    if OmegaConf.is_config(crop_size_cfg):
        crop_size_cfg = OmegaConf.to_container(crop_size_cfg, resolve=True)
    args.crop_size = crop_size_cfg

    args.temporal_size = cfg.get('temporal_size', 16)
    args.samples_per_video = cfg.get('samples_per_video', 50)
    args.batch_size = cfg.get('batch_size', 4)

    loss_cfg = cfg.get('loss', ['mse'])
    if OmegaConf.is_config(loss_cfg):
        loss_cfg = OmegaConf.to_container(loss_cfg, resolve=True)
    metric_cfg = cfg.get('metric', ['psnr'])
    if OmegaConf.is_config(metric_cfg):
        metric_cfg = OmegaConf.to_container(metric_cfg, resolve=True)
    args.loss = loss_cfg
    args.metric = metric_cfg
    args.lamb = cfg.get('lamb', lamb)
    
    # Pretrain weight rate lambda (from task config or top-level)
    args.weight_rate_lambda = task_cfg.get('weight_rate_lambda', cfg.get('weight_rate_lambda', 0.001))

    return args


def parse_args() -> ExperimentConfig:
    """
    main entry point - parse args and return experimentconfig
    compatible with existing code that uses parse_args()
    """
    parser = create_cli_parser()
    args = parser.parse_args()

    # if legacy config is specified, use old loading method with CLI overrides
    if args.config is not None:
        return _legacy_load_config(args.config, args)

    # load using new system
    cfg = load_config(args)
    return omega_to_experiment_config(cfg)


def _legacy_load_config(config_path: str, args: argparse.Namespace = None) -> ExperimentConfig:
    """load a legacy full config file for backward compatibility with CLI overrides"""
    import yaml
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(raw_cfg)
    
    # apply CLI overrides if args provided
    if args is not None:
        cli_overrides = parse_cli_overrides(args)
        cfg = merge_configs(cfg, cli_overrides)
    
    return omega_to_experiment_config(cfg)
