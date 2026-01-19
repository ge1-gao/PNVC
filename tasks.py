from utils import *
from io_utils import *
from metrics import *


class OverfitTask:
    def __init__(self, logger, video, loss_cfg, metric_cfg, lamb,
                 channel_scale=None, channel_shift=None,
                 enable_log=True, training=True, device=None,
                 target_bpp=None, bpp_penalty_weight=1000.0):
        self.logger = logger
        self.video = video
        self.channels = self.video.get_num_channels()

        self.loss_cfg = [1.0, loss_cfg[0]] if len(loss_cfg) == 1 else loss_cfg
        self.metric_cfg = metric_cfg

        assert channel_scale is None or channel_shift is None or \
            (len(channel_scale) == self.channels and len(channel_shift) == self.channels)
        self.channel_scale, self.channel_shift = \
            compute_scale_shift(self.channels, self.video.get_bit_depth(), channel_scale, channel_shift)

        self.enable_log = enable_log
        self.training = training
        self.device = device
        self.metrics_buffer = {}

        # accept scalar lamb from new config; normalize to list of one value
        if isinstance(lamb, (int, float)):
            lamb = [float(lamb)]
        elif isinstance(lamb, torch.Tensor):
            lamb = lamb.flatten().tolist()
        assert isinstance(lamb, (list, tuple)) and len(lamb) == 1, 'lamb should be a list/tuple with a single value'
        self.lamb = torch.tensor(sorted(lamb), dtype=torch.float32, device=self.device)

        # Lagrangian BPP constraint
        self.target_bpp = target_bpp
        self.bpp_penalty_weight = bpp_penalty_weight

        logger.info(f'OverfitTask:')
        logger.info(f'     Root: {self.video.get_path()}')
        logger.info(f'     Training: {self.training}')
        logger.info(f'     Losses: {self.loss_cfg}    Metrics: {self.metric_cfg}')
        logger.info(f'     Lamb: {self.lamb.tolist()}')
        logger.info(f'     Enable log: {self.enable_log}')
        if self.target_bpp is not None:
            logger.info(f'     Target BPP: {self.target_bpp}, Penalty weight: {self.bpp_penalty_weight}')

    def get_metrics(self):
        return self.metric_cfg

    def get_video_size(self):
        # Return the actual frame count being encoded, not the total file size
        return (self.video.get_num_frames(),) + self.video.get_video_size()[1:]

    def get_patch_size(self):
        return self.video.get_patch_size()

    def get_start_frame(self):
        return self.video.get_start_frame()

    def get_num_frames(self):
        return self.video.get_num_frames()

    def set_frames(self, start_frame, num_frames):
        self.video.set_frames(start_frame, num_frames)

    def create_cache(self):
        self.video.create_cache(enable=self.enable_log)

    def parse_batch(self, batch):
        """
        Parse the input and output batch during training/evaluation step
        """
        idx, x = batch
        idx_max = self.video.get_idx_max()
        assert idx.ndim == 2, \
            'idx should have 2 dimensions with shape [N, 3], where each row is the 3D patch coordinate'
        assert x.ndim == 5,  \
            'x should have 5 dimensions with shape [N, C, T, H, W], where each sample is a 3D patch'

        inputs = {
            'vidx': torch.zeros(idx.shape[0], dtype=torch.int32, device=idx.device),
            'vidx_max': 1,
            'idx': idx,
            'idx_max': idx_max,
            'x': x if self.training else None,
            'lamb': self.lamb,
            'rp': None,
            'rel_batch_size': x.shape[0] * x.shape[2] * \
                math.prod(self.get_patch_size()[1:]) / math.prod(self.get_video_size()[1:]),
            'video_size': (self.get_num_frames(),) + self.get_video_size()[1:],
            'patch_size': self.get_patch_size(),
            'channels': self.video.get_num_channels()
        }

        return inputs, x

    def parse_output(self, output):
        """
        Parse the output from the model during training/evaluation step
        """
        return output.contiguous()

    def compute_d_loss(self, x, y, lamb):
        # apply lambda to distortion term (overall loss: lambda * D + R)
        loss = 0.
        for i in range(len(self.loss_cfg) // 2):
            weight = float(self.loss_cfg[i * 2])
            loss_type = self.loss_cfg[i * 2 + 1]
            loss_i = compute_loss(loss_type, x, y).mean()

            if not torch.isfinite(loss_i):
                raise RuntimeError(
                    f"[NaN/Inf] OverfitTask compute_d_loss: "
                    f"loss_type={loss_type}, weight={weight}, "
                    f"loss_i={loss_i.item()}, "
                    f"x_range=({x.min().item():.4f},{x.max().item():.4f}), "
                    f"y_range=({y.min().item():.4f},{y.max().item():.4f})"
                )
            loss += weight * loss_i
        lamb_val = lamb.item() if isinstance(lamb, torch.Tensor) else float(lamb)
        return lamb_val * loss

    def compute_r_loss(self, r, lamb):
        # rate term remains unscaled when using lambda on distortion
        # r is already in BPP (bits per pixel)
        rate_loss = r
        
        # Add Lagrangian penalty if target_bpp is set
        if self.target_bpp is not None:
            # Bidirectional penalty: mu * (bpp - target_bpp)^2
            # Penalizes both exceeding and undershooting target
            diff = r - self.target_bpp
            if isinstance(diff, torch.Tensor):
                penalty = self.bpp_penalty_weight * diff ** 2
            else:
                penalty = self.bpp_penalty_weight * diff ** 2
            rate_loss = rate_loss + penalty
        
        return rate_loss

    def compute_metrics(self, x, y):
        metrics = {}
        with torch.no_grad():
            for metric_type in self.metric_cfg:
                metrics[metric_type] = compute_metric(metric_type, x, y)
        return metrics

    def d_step(self, model, batch):   
        inputs, target = self.parse_batch(batch)
        output = model(inputs, compute_outputs=True, compute_rates=False)
        output = self.parse_output(output)
        loss = self.compute_d_loss(output, target, inputs['lamb'])
        metrics = self.compute_metrics(output, target)
        return inputs, target, output, loss, metrics

    def r_step(self, model, batch, sub_step=0, num_sub_steps=1):
        inputs, _ = self.parse_batch(batch)
        inputs['r_sub_step'] = sub_step
        inputs['r_num_sub_steps'] = num_sub_steps
        rate, rate_loss = model(inputs, compute_outputs=False, compute_rates=True)
        rate = rate / math.prod(inputs['video_size'])
        rate_loss = self.compute_r_loss(rate_loss / math.prod(inputs['video_size']), inputs['lamb'])
        if not torch.isfinite(rate) or not torch.isfinite(rate_loss):
            raise RuntimeError(
                f"[NaN/Inf] OverfitTask r_step: rate={rate}, rate_loss={rate_loss}, video_size={inputs['video_size']}"
            )
        return rate, rate_loss

    def log_outputs(self, inputs, outputs, metrics):
        """
        Log the outputs
        """
        if self.enable_log:
            N, _, _, _, _ = outputs.shape
            idx = inputs['idx']
            outputs = outputs.detach()
            metrics = {k: v.detach() for k, v in metrics.items()}

            scale = self.channel_scale.to(outputs.device)
            shift = self.channel_shift.to(outputs.device)

            # Loop over all samples
            for n in range(N):
                # Save the patch
                if isinstance(self.video, PNGVideo):
                    self.video.write_patch(idx[n].cpu().numpy(),
                                           ((outputs[n] - shift.view(self.channels, 1, 1, 1)) \
                                            / scale.view(self.channels, 1, 1, 1)) \
                                           .permute(1, 2, 3, 0).round().cpu().numpy())
                else:
                    yuv420_patch = yuv444_to_yuv420(outputs[n].permute(1, 0, 2, 3), mode='avg_pool')
                    self.video.write_patch(idx[n].cpu().numpy(),
                                           [((patch_i - shift[i]) / scale[i]).round().cpu().numpy() \
                                            for i, patch_i in enumerate(yuv420_patch)])

                # Save the averge metrics for each patch
                idx_str = '(' + ' '.join([str(i) for i in idx[n].tolist()]) + ')'
                self.metrics_buffer[idx_str] = \
                    ','.join([idx_str] + \
                             [f'{k}: {v[n].mean().item():.4f}' for k, v in metrics.items() if k in self.metric_cfg])

    def flush(self, dir):
        """
        Flush the outputs to the disk
        """
        if self.enable_log:
            # Save video outputs
            self.video.flush(dir)

            # Save metrics
            with open(os.path.join(dir, 'metrics.txt'), 'w') as f:
                f.write('\n'.join(self.metrics_buffer.values()))
            self.metrics_buffer = {}


def create_overfit_task(args, logger, video, channel_scale=None, channel_shift=None, training=True, device=None):
    # Create task
    if training:
        config = args.train_task
    else:
        config = args.eval_task

    # Get Lagrangian BPP constraint settings (fallback to task config if not in train_task/eval_task)
    target_bpp = getattr(config, "target_bpp", None)
    if target_bpp is None and hasattr(args, "task"):
        target_bpp = getattr(args.task, "target_bpp", None)
    
    bpp_penalty_weight = getattr(config, "bpp_penalty_weight", None)
    if bpp_penalty_weight is None and hasattr(args, "task"):
        bpp_penalty_weight = getattr(args.task, "bpp_penalty_weight", 1000.0)
    if bpp_penalty_weight is None:
        bpp_penalty_weight = 1000.0

    task = OverfitTask(logger, video, loss_cfg=config.loss, metric_cfg=config.metric,
                       lamb=config.lamb, channel_scale=channel_scale, channel_shift=channel_shift,
                       enable_log=config.enable_log, training=training, device=device,
                       target_bpp=target_bpp, bpp_penalty_weight=bpp_penalty_weight)
    return task


# ----------------------------------------------------------------
class PretrainTask:
    """
    Task for pretraining on multiple videos.
    Similar to OverfitTask but doesn't require output video for logging.
    
    Rate calculation for pretrain:
    - Grid rate: normalized by pixels (BPP) - per-sample
    - Weight rate: normalized by num_weights (bits per weight) - shared decoder
    - Combined with weight_rate_lambda to balance contributions
    """
    
    def __init__(self, logger, dataset, loss_cfg, metric_cfg, lamb, 
                 weight_rate_lambda=0.001, training=True, device=None):
        self.logger = logger
        self.dataset = dataset
        self.channels = dataset.get_num_channels()
        
        self.loss_cfg = [1.0, loss_cfg[0]] if len(loss_cfg) == 1 else loss_cfg
        self.metric_cfg = metric_cfg
        self.training = training
        self.device = device
        self.weight_rate_lambda = weight_rate_lambda
        self._num_decoder_weights = None  # cached count
        
        if isinstance(lamb, (int, float)):
            lamb = [float(lamb)]
        elif isinstance(lamb, torch.Tensor):
            lamb = lamb.flatten().tolist()
        self.lamb = torch.tensor(sorted(lamb), dtype=torch.float32, device=self.device)
        
        logger.info(f"PretrainTask:")
        logger.info(f"     Training: {self.training}")
        logger.info(f"     Losses: {self.loss_cfg}    Metrics: {self.metric_cfg}")
        logger.info(f"     Lamb: {self.lamb.tolist()}")
        logger.info(f"     Weight rate lambda: {self.weight_rate_lambda}")
    
    def get_metrics(self):
        return self.metric_cfg
    
    def get_video_size(self):
        return self.dataset.get_video_size()
    
    def get_patch_size(self):
        return self.dataset.get_patch_size()
    
    def parse_batch(self, batch):
        idx, x = batch  # idx: [V, 3], x: [V, C, T, H, W]
        video_size = self.get_video_size()
        patch_size = self.get_patch_size()

        V = x.shape[0]
        idx_max = tuple(video_size[d] // patch_size[d] for d in range(3))
        patches_per_vid = math.prod(idx_max)
        N = V * patches_per_vid  # N = V when patch_size == video_size

        # vidx: which video each patch belongs to [N]
        vidx = torch.arange(V, device=x.device, dtype=torch.int32).repeat_interleave(patches_per_vid)

        # idx: patch coordinates [N, 3]
        if idx is None or idx.shape[0] != N:
            idx = torch.zeros((N, 3), dtype=torch.int32, device=x.device)

        inputs = {
            "vidx": vidx,  # [N]
            "vidx_max": V,
            "idx": idx,  # [N, 3]
            "idx_max": idx_max,
            "x": x if self.training else None,  # [V, C, T, H, W]
            "lamb": self.lamb,
            "rp": None,
            "rel_batch_size": N,
            "video_size": video_size,
            "patch_size": patch_size,
            "channels": self.channels,
        }
        return inputs, x
    
    def parse_output(self, output):
        return output.contiguous()
    
    def compute_d_loss(self, x, y, lamb):
        # apply lambda to distortion term (overall loss: lambda * D + R)
        loss = 0.
        for i in range(len(self.loss_cfg) // 2):
            weight = float(self.loss_cfg[i * 2])
            loss_type = self.loss_cfg[i * 2 + 1]
            loss_i = compute_loss(loss_type, x, y)

            loss_i_mean = loss_i.mean()
            if not torch.isfinite(loss_i_mean):
                raise RuntimeError(
                    f"[NaN/Inf] PretrainTask compute_d_loss: "
                    f"loss_type={loss_type}, weight={weight}, "
                    f"loss_i={loss_i_mean.item()}, "
                    f"x_range=({x.min().item():.4f},{x.max().item():.4f}), "
                    f"y_range=({y.min().item():.4f},{y.max().item():.4f})"
                )
            loss += weight * loss_i_mean
        lamb_val = lamb.item() if isinstance(lamb, torch.Tensor) else float(lamb)
        return lamb_val * loss

    def compute_r_loss(self, r, lamb):
        # rate term remains unscaled when using lambda on distortion
        return r

    def compute_metrics(self, x, y):
        metrics = {}
        with torch.no_grad():
            for metric_type in self.metric_cfg:
                metrics[metric_type] = compute_metric(metric_type, x, y)
        return metrics

    def d_step(self, model, batch):
        inputs, target = self.parse_batch(batch)
        output = model(inputs, compute_outputs=True, compute_rates=False)
        output = self.parse_output(output)
        loss = self.compute_d_loss(output, target, inputs["lamb"])
        metrics = self.compute_metrics(output, target)
        return inputs, target, output, loss, metrics

    def _get_num_decoder_weights(self, model):
        """Get number of decoder weight parameters (cached)."""
        if self._num_decoder_weights is None:
            # Count decoder weight parameters from the compress model
            compress_model = model
            if hasattr(compress_model, "param_model"):
                params = compress_model.param_model()
                _, weights, _ = compress_model._split_params(params)
                self._num_decoder_weights = sum(p.numel() for p in weights.values())
            else:
                # Fallback estimate: ~5M params for typical decoder
                self._num_decoder_weights = 5_000_000
        return self._num_decoder_weights
    
    def r_step(self, model, batch, sub_step=0, num_sub_steps=1):
        inputs, _ = self.parse_batch(batch) 
        inputs["r_sub_step"] = sub_step
        inputs["r_num_sub_steps"] = num_sub_steps
        
        # Get per-component rates (no reduction)
        rate_dict, rate_loss_dict = model(
            inputs, compute_outputs=False, compute_rates=True, reduce_rates=False
        )
        
        # Grid rate: bits per pixel (per-sample)
        num_pixels = math.prod(inputs["video_size"])
        grid_rate = rate_dict.get("grid", 0.0)
        grid_rate_loss = rate_loss_dict.get("grid", 0.0)
        grid_bpp = grid_rate / num_pixels
        grid_bpp_loss = grid_rate_loss / num_pixels
        
        # Weight rate: bits per weight (shared decoder)
        # Normalized by num_weights, scaled by weight_rate_lambda
        weight_rate = rate_dict.get("weight", 0.0)
        weight_rate_loss = rate_loss_dict.get("weight", 0.0)
        num_weights = self._get_num_decoder_weights(model)
        weight_bpw = weight_rate / num_weights
        weight_bpw_loss = weight_rate_loss / num_weights
        
        # Combined rate: grid_bpp + lambda * weight_bpw
        # - rate (for logging): only grid BPP (meaningful per-video metric)
        # - rate_loss (for training): includes weight to train weight_em
        rate = grid_bpp  # logging
        rate_loss = self.compute_r_loss(
            grid_bpp_loss + self.weight_rate_lambda * weight_bpw_loss, 
            inputs["lamb"]
        )
        
        if not torch.isfinite(rate) or not torch.isfinite(rate_loss):
            raise RuntimeError(
                f"[NaN/Inf] PretrainTask r_step: rate={rate}, rate_loss={rate_loss}, "
                f"grid_bpp={grid_bpp}, weight_bpw={weight_bpw}, "
                f"video_size={inputs['video_size']}, num_weights={num_weights}"
            )
        return rate, rate_loss

    def log_outputs(self, inputs, outputs, metrics):
        pass  # No output logging for pretrain

    def flush(self, dir):
        pass  # No flush for pretrain


def create_pretrain_task(logger, dataset, loss_cfg=None, metric_cfg=None, lamb=1.0, 
                         weight_rate_lambda=0.001, training=True, device=None):
    """Create pretrain task.
    
    Args:
        weight_rate_lambda: Scale factor for weight rate in total loss.
            Weight rate is computed as bits-per-weight and scaled by this factor.
            Default 0.001 keeps weight contribution small relative to grid BPP.
    """
    if loss_cfg is None:
        loss_cfg = ["mse"]
    if metric_cfg is None:
        metric_cfg = ["psnr"]
    return PretrainTask(
        logger=logger,
        dataset=dataset,
        loss_cfg=loss_cfg,
        metric_cfg=metric_cfg,
        lamb=lamb,
        weight_rate_lambda=weight_rate_lambda,
        training=training,
        device=device,
    )
