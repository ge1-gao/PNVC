import os
import sys
import time
import logging
import shutil
import math
from pathlib import Path

import torch
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, create_cli_parser
from datasets import create_pretrain_dataset
from tasks import create_pretrain_task
import models
from compress_models import create_compress_model
from utils import unwrap_model


# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
def get_accelerator_logger(cfg, accelerator):
    """Create accelerator and logger."""
    # Output dir can be in cfg.output or cfg.experiment.output
    output_dir = cfg.get("output", cfg.get("experiment", {}).get("output", "./outputs/pretrain"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(output_dir, "train.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler (only main process)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if accelerator.is_main_process else logging.ERROR)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, output_dir


# ----------------------------------------------------------------
# Dataset & Loader
# ----------------------------------------------------------------
def create_dataset_and_loader(cfg, logger, accelerator):
    """Create pretrain dataset and dataloader."""
    
    # Patch size from train config
    patch_size = tuple(cfg.train.get("train_patch_size", [4, 120, 120]))
    # samples_per_video can be in train or at top level
    samples_per_video = cfg.get("samples_per_video", cfg.train.get("samples_per_video", "auto"))
    hflip_prob = cfg.train.get("hflip_prob", 0.0)
    
    logger.info(f"Dataset configuration:")
    logger.info(f"     Patch size (T,H,W): {patch_size}")
    logger.info(f"     Samples per video: {samples_per_video}")
    logger.info(f"     Horizontal flip prob: {hflip_prob}")
    
    # Check if using multi-source data
    data_sources = cfg.data.get("sources", None)
    if data_sources is not None:
        # Multi-source mode
        logger.info(f"Using multi-source dataset with {len(data_sources)} source(s)")
        dataset = create_pretrain_dataset(
            logger=logger,
            patch_size=patch_size,
            samples_per_video=samples_per_video,
            hflip_prob=hflip_prob,
            data_sources=list(data_sources),
        )
    else:
        # Single-source mode (legacy)
        video_list = cfg.data.get("video_list", "./sep_trainlist_subset.txt")
        base_dir = cfg.data.get("dataset_dir", "~/data/VideoCompression/vimeo_septuplet")
        logger.info(f"Using single-source dataset: {video_list}")
        dataset = create_pretrain_dataset(
            logger=logger,
            video_list_file=video_list,
            patch_size=patch_size,
            samples_per_video=samples_per_video,
            hflip_prob=hflip_prob,
            base_dir=base_dir,
        )
    
    # Create dataloader
    batch_size = cfg.train.get("train_batch_size", 16)
    num_workers = cfg.train.get("workers", 4)
    pin_memory = cfg.train.get("pin_mem", True)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    logger.info(f"DataLoader:")
    logger.info(f"     Batch size: {batch_size}")
    logger.info(f"     Workers: {num_workers}")
    logger.info(f"     Total batches per epoch: {len(loader)}")
    
    return dataset, loader


# ----------------------------------------------------------------
# Model & Task
# ----------------------------------------------------------------
def create_model_and_task(cfg, logger, accelerator, loader, dataset):
    """Create model, compress model, and task."""
    
    # Create task
    lamb = cfg.task.get("lamb", 0.01)
    weight_rate_lambda = cfg.task.get("weight_rate_lambda", 0.001)
    
    task = create_pretrain_task(
        logger=logger,
        dataset=dataset,
        loss_cfg=cfg.task.get("loss", {}),
        metric_cfg=cfg.task.get("metric", {}),
        lamb=lamb,
        weight_rate_lambda=weight_rate_lambda,
        training=True,
        device=accelerator.device,
    )
    
    logger.info(f"Task configuration:")
    logger.info(f"     Lambda (distortion): {lamb}")
    logger.info(f"     Weight rate lambda: {weight_rate_lambda}")
    
    # Get sample batch for model creation
    sample_batch = next(iter(loader))
    sample_inputs, _ = task.parse_batch(sample_batch)
    
    # Create model
    # Convert OmegaConf to native Python dict (hinerv.py requires native list, not ListConfig)
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    logger.info(f"Creating model: {model_cfg['type']}")
    model = models.create_model(model_cfg, logger, sample_inputs)
    
    # Create compress model (entropy model)
    stage = cfg.train.get("stage", "pretrain_s1")
    compress_cfg = OmegaConf.to_container(cfg.compress_model, resolve=True)
    
    logger.info(f"Creating compress model for stage: {stage}")
    compress_model = create_compress_model(
        logger=logger,
        model=model,
        cfgs=compress_cfg,
        model_cfg=model_cfg,
        stage=stage,
    )
    
    # Wrap model with compress model
    model = compress_model
    
    # Set pretrain mode with stage-specific freezing
    # This ensures encoder grids are used in forward pass (not learnable parameter grids)
    if hasattr(model, "set_pretrain_mode"):
        model.set_pretrain_mode(stage)
        logger.info(f"Set pretrain mode ({stage}): encoder grids used directly in forward pass")
        logger.info(f"     _use_encoder_grids_in_forward: {getattr(model, '_use_encoder_grids_in_forward', 'N/A')}")
        logger.info(f"     _init_grids_from_encoder: {getattr(model, '_init_grids_from_encoder', 'N/A')}")
    
    return model, task


# ----------------------------------------------------------------
# Optimizer & Scheduler
# ----------------------------------------------------------------
def create_optimizer_and_scheduler(cfg, logger, model, loader):
    """Create optimizer and scheduler."""
    
    # Learning rate settings
    lr = float(cfg.train.get("lr", 2e-3))
    warmup_lr = float(cfg.train.get("warmup_lr", 1e-5))
    min_lr = float(cfg.train.get("min_lr", 1e-4))
    weight_decay = float(cfg.train.get("weight_decay", 0.0))
    opt_eps = float(cfg.train.get("opt_eps", 1e-8))
    max_norm = float(cfg.train.get("max_norm", 1.0))
    
    # Epochs
    epochs = cfg.train.get("epochs", 50)
    warmup_epochs = cfg.train.get("warmup_epochs", 2)
    
    logger.info(f"Optimizer configuration:")
    logger.info(f"     LR: {lr}, Warmup LR: {warmup_lr}, Min LR: {min_lr}")
    logger.info(f"     Weight decay: {weight_decay}")
    logger.info(f"     Max grad norm: {max_norm}")
    logger.info(f"     Epochs: {epochs}, Warmup: {warmup_epochs}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=opt_eps,
        weight_decay=weight_decay,
    )
    
    # Create scheduler (linear warmup + cosine decay)
    def lr_lambda(current_step):
        num_steps_per_epoch = len(loader)
        total_steps = epochs * num_steps_per_epoch
        warmup_steps = warmup_epochs * num_steps_per_epoch
        
        if current_step < warmup_steps:
            # Linear warmup
            return (warmup_lr + (lr - warmup_lr) * current_step / warmup_steps) / lr
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return (min_lr + (lr - min_lr) * cosine_decay) / lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    return optimizer, scheduler


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------
def train_epoch(cfg, logger, accelerator, model, loader, task, optimizer, scheduler, epoch, stage="pretrain_s3"):
    """Train for one epoch.
    
    Args:
        stage: pretrain stage
            - pretrain_s1: distortion only (entropy models frozen)
            - pretrain_s2: rate only (encoder/decoder frozen)
            - pretrain_s3: joint distortion + rate
    """
    from metrics import psnr
    
    model.train()
    
    total_loss = 0.0
    total_distortion = 0.0
    total_rate = 0.0
    total_psnr = 0.0
    num_batches = 0
    
    max_norm = float(cfg.train.get("max_norm", 1.0))
    grad_accum = cfg.train.get("grad_accum", 1)
    log_interval = cfg.train.get("log_interval", 50)
    
    # Determine what to train based on stage
    train_distortion = stage in ["pretrain_s1", "pretrain_s3"]
    train_rate = stage in ["pretrain_s2", "pretrain_s3"]
    
    total_batches = len(loader)
    
    for batch_idx, batch in enumerate(loader):
        # Forward pass
        inputs, target = task.parse_batch(batch)
        
        with accelerator.accumulate(model):
            # Compute based on stage
            if train_rate:
                # Need rate computation
                outputs, rate, rate_loss = model(inputs, compute_rates=True)
            else:
                # S1: distortion only, skip rate computation entirely
                outputs = model(inputs, compute_rates=False)
                rate_loss = torch.tensor(0.0, device=outputs.device)
            
            parsed_output = task.parse_output(outputs)
            
            # Always compute distortion loss for logging
            distortion_loss_for_log = task.compute_d_loss(parsed_output, target, inputs["lamb"])
            
            # Compute rate loss (normalize by pixels)
            if train_rate:
                num_pixels = math.prod(inputs["video_size"])
                rate_loss_normalized = task.compute_r_loss(rate_loss / num_pixels, inputs["lamb"])
            else:
                rate_loss_normalized = torch.tensor(0.0, device=outputs.device)
            
            # Total loss based on stage (only include distortion if training it)
            if train_distortion:
                loss = distortion_loss_for_log + rate_loss_normalized
            else:
                loss = rate_loss_normalized
            
            # For logging purposes
            distortion_loss = distortion_loss_for_log
            
            # Backward
            accelerator.backward(loss / grad_accum)
            
            if (batch_idx + 1) % grad_accum == 0:
                # Gradient clipping
                if max_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # Compute PSNR
        with torch.no_grad():
            psnr_val = psnr(parsed_output.clamp(0, 1), target).mean().item()
        
        # Accumulate metrics
        distortion_val = distortion_loss.item() if hasattr(distortion_loss, "item") else float(distortion_loss)
        rate_val = rate_loss_normalized.item() if hasattr(rate_loss_normalized, "item") else float(rate_loss_normalized)
        total_loss += loss.item()
        total_distortion += distortion_val
        total_rate += rate_val
        total_psnr += psnr_val
        num_batches += 1
        
        # Log every N batches
        if accelerator.is_main_process and (batch_idx + 1) % log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            avg_psnr = total_psnr / num_batches
            logger.info(
                f"[Epoch {epoch}] [{batch_idx + 1}/{total_batches}] "
                f"loss={loss.item():.4f}, dist={distortion_val:.4f}, rate={rate_val:.6f}, "
                f"psnr={psnr_val:.2f}dB (avg={avg_psnr:.2f}dB), lr={current_lr:.6f}"
            )
    
    # Epoch summary
    avg_loss = total_loss / num_batches
    avg_distortion = total_distortion / num_batches
    avg_rate = total_rate / num_batches
    avg_psnr = total_psnr / num_batches
    
    if accelerator.is_main_process:
        logger.info(
            f"[Epoch {epoch}] Summary: loss={avg_loss:.4f}, "
            f"dist={avg_distortion:.4f}, rate={avg_rate:.6f}, psnr={avg_psnr:.2f}dB"
        )
    
    return avg_loss, {"distortion": avg_distortion, "rate": avg_rate, "psnr": avg_psnr}


# ----------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------
def save_checkpoint(cfg, logger, accelerator, model, optimizer, scheduler, epoch, output_dir):
    """Save checkpoint."""
    if not accelerator.is_main_process:
        return
    
    ckpt_path = os.path.join(output_dir, "checkpoint.pth")
    ckpt_path_epoch = os.path.join(output_dir, f"checkpoint_epoch{epoch:04d}.pth")
    
    # Save full checkpoint (model + optimizer + scheduler + epoch)
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    
    torch.save(state, ckpt_path)
    
    # Save epoch-specific checkpoint every N epochs
    save_interval = cfg.train.get("save_interval", 10)
    if epoch % save_interval == 0 or epoch == cfg.train.get("epochs", 50) - 1:
        torch.save(state, ckpt_path_epoch)
        logger.info(f"Saved checkpoint: {ckpt_path_epoch}")
    
    logger.info(f"Saved latest checkpoint: {ckpt_path}")


def load_checkpoint(cfg, logger, accelerator, model, optimizer, scheduler, output_dir, args=None):
    """Load checkpoint if exists or resume from another path."""
    
    # Check if resuming from external checkpoint
    # Support both cfg.resume.path and args.resume
    resume_path = None
    model_only = False
    
    # Try cfg first
    if hasattr(cfg, "resume") and cfg.resume is not None:
        resume_path = cfg.resume.get("path", None) if hasattr(cfg.resume, "get") else getattr(cfg.resume, "path", None)
        model_only = cfg.resume.get("model_only", False) if hasattr(cfg.resume, "get") else getattr(cfg.resume, "model_only", False)
    
    # Fall back to args if available
    if resume_path is None and args is not None:
        resume_path = getattr(args, "resume", None)
        model_only = getattr(args, "resume_model_only", False)
        if isinstance(model_only, str):
            model_only = model_only.lower() == "true"
    
    if resume_path is not None and os.path.exists(resume_path):
        # Support both directory path and direct .pth file path
        if resume_path.endswith(".pth"):
            ckpt_path = resume_path
        else:
            ckpt_path = os.path.join(resume_path, "checkpoint.pth")
        
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming from: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            
            # Load model
            unwrap_model(model).load_state_dict(state["model"], strict=False)
            
            # Load optimizer/scheduler only if not model-only resume
            if not model_only:
                if "optimizer" in state:
                    optimizer.load_state_dict(state["optimizer"])
                if "scheduler" in state:
                    scheduler.load_state_dict(state["scheduler"])
                start_epoch = state.get("epoch", 0) + 1
                logger.info(f"Resumed from epoch {start_epoch - 1}")
                return start_epoch
            else:
                logger.info("Model-only resume (optimizer/scheduler reset)")
                return 0
    
    # Check local checkpoint
    ckpt_path = os.path.join(output_dir, "checkpoint.pth")
    if os.path.exists(ckpt_path):
        logger.info(f"Loading local checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        unwrap_model(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch - 1}")
        return start_epoch
    
    logger.info("No checkpoint found, starting from scratch")
    return 0


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    """Main pretrain entry point."""
    
    # Parse arguments and load config
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Use legacy config loading if --config is provided (full yaml file)
    # Otherwise use new preset system
    if hasattr(args, "config") and args.config is not None:
        from omegaconf import OmegaConf
        import yaml
        # Load YAML directly to preserve nested structure
        with open(args.config, "r") as f:
            cfg_dict = yaml.safe_load(f)
        cfg = OmegaConf.create(cfg_dict)
        
        # Apply CLI overrides
        from config import parse_cli_overrides
        cli_overrides = parse_cli_overrides(args)
        cfg = OmegaConf.merge(cfg, cli_overrides)
        
        # ----------------------------------------------------------------
        # Fill decoder defaults (required by hinerv.py but not in YAML)
        # ----------------------------------------------------------------
        decoder_defaults = {
            "use_hi_encoding": [True, True, True, True],
            "local_grid_level_scale": [2.0, 0.5],
            "local_grid_depth_scale": [1.0, 0.5],
            # Block defaults
            "block_type": ["convnext", "convnext", "convnext", "convnext"],
            "block_norm": ["layernorm-no-affine", "layernorm-no-affine", "layernorm-no-affine", "layernorm-no-affine"],
            "block_act": ["gelu", "gelu", "gelu", "gelu"],
            "block_bias": False,
            "block_layerscale": 0.0,
            # Head defaults
            "head_type": "linear_head",
            "head_norm": "none",
            "head_act": "sigmoid",
            "head_bias": True,
            "head_layerscale": 0.0,
            # Upsample defaults
            "upsample_type": ["trilinear", "trilinear", "trilinear", "trilinear"],
            "upsample_config": ["matmul-th-w", "matmul-th-w", "matmul-th-w", "matmul-th-w"],
            "upsample_norm": ["layernorm-no-affine", "layernorm-no-affine", "layernorm-no-affine", "layernorm-no-affine"],
            "upsample_act": ["none", "none", "none", "none"],
        }
        
        # Merge defaults with user config (user config takes priority)
        if "model" in cfg and "config" in cfg.model and "decoder" in cfg.model.config:
            for key, default_val in decoder_defaults.items():
                if key not in cfg.model.config.decoder:
                    cfg.model.config.decoder[key] = default_val
    else:
        cfg = load_config(args)
    
    # Create accelerator with find_unused_parameters for pretrain stages
    # (some parameters may be frozen depending on stage)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision if hasattr(args, "mixed_precision") else None,
        kwargs_handlers=[ddp_kwargs],
    )
    
    # Logger
    logger, output_dir = get_accelerator_logger(cfg, accelerator)
    
    logger.info("=" * 60)
    logger.info("PNVC Pretrain")
    logger.info("=" * 60)
    logger.info(f"Stage: {cfg.train.get('stage', 'pretrain_s1')}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Config: {args.config if hasattr(args, 'config') else 'N/A'}")
    logger.info("=" * 60)
    
    # Set seed
    seed = cfg.train.get("seed", 42)
    accelerate.utils.set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Create dataset & loader
    dataset, loader = create_dataset_and_loader(cfg, logger, accelerator)
    
    # Create model & task
    model, task = create_model_and_task(cfg, logger, accelerator, loader, dataset)
    
    # Create optimizer & scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(cfg, logger, model, loader)
    
    # Prepare with accelerator
    model, loader, optimizer, scheduler = accelerator.prepare(
        model, loader, optimizer, scheduler
    )
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(cfg, logger, accelerator, model, optimizer, scheduler, output_dir, args=args)
    
    # Training loop
    stage = cfg.train.get("stage", "pretrain_s1")
    epochs = cfg.train.get("epochs", 50)
    logger.info(f"Training for {epochs - start_epoch} epochs (from epoch {start_epoch}, stage={stage})")
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Train epoch
        avg_loss, metrics = train_epoch(
            cfg, logger, accelerator, model, loader, task, optimizer, scheduler, epoch, stage
        )
        
        # Save checkpoint
        save_checkpoint(cfg, logger, accelerator, model, optimizer, scheduler, epoch, output_dir)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s\n")
    
    total_time = time.time() - total_start_time
    logger.info("=" * 60)
    logger.info(f"Training complete! Total time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    logger.info(f"Final checkpoint: {output_dir}/checkpoint.pth")
    logger.info("=" * 60)
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
