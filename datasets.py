from utils import *
from io_utils import *
import bisect
import struct

# Don't allow truncated images - we want to detect corruption early
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoDatasetBase(torch.utils.data.Dataset):
    """
    The base dataset class for loading videos.
    It loads video frames in 3D patches.
    """
    def __init__(self, logger, video, channel_scale=None, channel_shift=None, is_subset=False):
        self.logger = logger
        self.video = video

        assert channel_scale is None or channel_shift is None or \
               (len(channel_scale) == self.video.get_num_channels() and \
                len(channel_shift) == self.video.get_num_channels())
        self.channel_scale, self.channel_shift = compute_scale_shift(self.video.get_num_channels(),
                                                                     self.video.get_bit_depth(),
                                                                     channel_scale, channel_shift)

        if not is_subset:
            self.logger.info(f'{type(self).__name__}:')
            self.logger.info(f'     Path: {self.video.path}')
            self.logger.info(f'     Video size: {self.video.get_video_size()}    Patch size: {self.video.get_patch_size()}    Padding: {self.video.get_padding()}')
            self.logger.info(f'     Channels: {self.video.get_num_channels()}    Bit depth: {self.video.get_bit_depth()}')
            self.logger.info(f'     Scale: {self.channel_scale}    Shift: {self.channel_shift}')

    def get_video_size(self):
        return self.video.get_video_size()

    def get_patch_size(self):
        return self.video.get_patch_size()

    def get_num_channels(self):
        return self.video.get_num_channels()

    def get_start_frame(self):
        return self.video.get_start_frame()

    def get_num_frames(self):
        return self.video.get_num_frames()

    def set_frames(self, start_frame, num_frames):
        self.video.set_frames(start_frame, num_frames)

    def create_cache(self):
        self.video.create_cache()

    def _get_patch(self, idx_thw):
        raise NotImplementedError

    def __len__(self):
        return math.prod(self.video.get_idx_max())

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx_max = self.video.get_idx_max()  #[T, H, W]
        idx_thw = (idx // (idx_max[1] * idx_max[2]),
                   (idx % (idx_max[1] * idx_max[2])) // idx_max[2],
                   (idx % (idx_max[1] * idx_max[2])) % idx_max[2])
        return torch.tensor(idx_thw, dtype=int), self._get_patch(idx_thw)


class PNGVideoDataset(VideoDatasetBase):
    """
    The dataset class for loading videos (in PNG files). Each dataset instance loads all video frames in a folder.
    """
    def _get_patch(self, idx_thw):
        scale = self.channel_scale.view(self.video.get_num_channels(), 1, 1, 1)
        shift = self.channel_shift.view(self.video.get_num_channels(), 1, 1, 1)
        rgb_patch = torch.tensor(self.video.read_patch(idx_thw)).permute(3, 0, 1, 2).float() * scale + shift
        return torch.clip_(rgb_patch, 0., 1.)


class YUVVideoDataset(VideoDatasetBase):
    """
    The dataset class for loading videos (in YUV format). Each dataset instance loads all video frames from a yuv file.
    """
    def _get_patch(self, idx_thw):
        scale = self.channel_scale
        shift = self.channel_shift

        # List of [T, 1, H, W] tensors
        yuv420_patch = [torch.tensor(patch_i)[:, None].float() * scale[i] + shift[i] \
                        for i, patch_i in enumerate(self.video.read_patch(idx_thw))]
        # List of [T, 1, H, W] tensors -> [T, 3, H, W] tensor -> [3, T, H, W] tensor
        yuv444_patch = yuv420_to_yuv444(yuv420_patch, mode='nearest').permute(1, 0, 2, 3)

        return torch.clip_(yuv444_patch, 0., 1.)


class VideoSubset(torch.utils.data.Subset):
    def __init__(self, dataset, start_frame, num_frames):
        super().__init__(dataset, [])
        self.dataset = dataset
        self.indices = []
        dataset = self.dataset
        patches_per_frame = math.prod(dataset.video.get_idx_max()[1:])
        assert start_frame % dataset.get_patch_size()[0] == 0 and num_frames % dataset.get_patch_size()[0] == 0, \
            f'start_frame and num_frames should be multiples of patch_size[0]'
        start_patch = (start_frame // dataset.get_patch_size()[0]) * patches_per_frame
        num_patches = (num_frames // dataset.get_patch_size()[0]) * patches_per_frame
        self.indices += list(range(start_patch, start_patch + num_patches))

    def get_video_size(self):
        return self.dataset.get_video_size()

    def get_patch_size(self):
        return self.dataset.get_patch_size()


def create_overfit_dataset(args, logger, video, channel_scale=None, channel_shift=None, is_subset=False):
    # Create dataset
    if isinstance(video, PNGVideo):
        dataset = PNGVideoDataset(logger, video, 
                                  channel_scale=channel_scale, channel_shift=channel_shift,
                                  is_subset=is_subset)
    elif isinstance(video, YUVVideo):
        dataset = YUVVideoDataset(logger, video, 
                                  channel_scale=channel_scale, channel_shift=channel_shift,
                                  is_subset=is_subset)
    else:
        raise ValueError
    return dataset


class PretrainVideoDataset(torch.utils.data.Dataset):
    """
    Dataset for pretraining on multiple RGB videos (e.g., Vimeo-90K setuplet).
    Loads 3D chunks (T, H, W) with light augmentation.
    
    Args:
        logger: Logger instance.
        video_dirs: List of video directory paths (each dir contains ordered frames).
        patch_size: 3D patch size (T, H, W).
        samples_per_video: Number of samples per video per epoch.
        hflip_prob: Probability of horizontal flip.
    """
    
    def __init__(
        self,
        logger,
        video_dirs: list,
        patch_size: tuple = (16, 256, 256),
        samples_per_video: int = 100,
        hflip_prob: float = 0.5,
        base_dir: str = None,
        video_samples_map: dict = None,  # Map video_dir -> samples_per_video
    ):
        self.logger = logger
        assert len(patch_size) == 3, "patch_size should be (T, H, W)"
        self.patch_size = tuple(int(p) for p in patch_size)
        self.temporal_size = self.patch_size[0]
        self.crop_size = self.patch_size[1:]
        self.hflip_prob = hflip_prob
        self.base_dir = os.path.expanduser(base_dir) if base_dir is not None else None
        
        # Track bad frames to skip on retry (shared via copy-on-write, updated on first corruption)
        self._bad_frames = set()
        
        # samples_per_video: int (fixed) or "auto" (dynamic based on video length)
        self.samples_per_video_mode = samples_per_video
        self.video_samples_map = video_samples_map or {}
        
        # Load video info
        self.videos = []
        self.cumulative_samples = [0]
        
        for video_dir in video_dirs:
            video_dir = os.path.expanduser(video_dir)
            if not os.path.isabs(video_dir) and self.base_dir is not None:
                video_dir = os.path.join(self.base_dir, video_dir)
            if not os.path.exists(video_dir):
                logger.warning(f"Video directory not found: {video_dir}")
                continue
            
            frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith((".png", ".jpg"))])
            if len(frame_files) < self.temporal_size:
                logger.warning(f"Video {video_dir} has only {len(frame_files)} frames, need {self.temporal_size}")
                continue
            
            # Get video size from first frame
            first_frame = np.array(Image.open(os.path.join(video_dir, frame_files[0])))
            h, w = first_frame.shape[:2]
            channels = first_frame.shape[2] if first_frame.ndim == 3 else 1
            
            if h < self.crop_size[0] or w < self.crop_size[1]:
                logger.warning(f"Video {video_dir} ({h}x{w}) smaller than crop size {self.crop_size}")
                continue
            
            # Compute samples for this video
            # Use video-specific config if available, otherwise use global default
            num_frames = len(frame_files)
            samples_config = self.video_samples_map.get(video_dir, samples_per_video)
            
            if samples_config == "auto":
                # Auto: ~50% coverage with overlap, at least 1 sample
                video_samples = max(1, num_frames // self.temporal_size // 2)
            else:
                video_samples = int(samples_config)
            
            self.videos.append({
                "dir": video_dir,
                "frames": frame_files,
                "height": h,
                "width": w,
                "channels": channels,
                "num_frames": num_frames,
                "samples": video_samples,
            })
            self.cumulative_samples.append(self.cumulative_samples[-1] + video_samples)
        
        assert len(self.videos) > 0, "No valid videos found!"
        
        self.channels = self.videos[0]["channels"]
        self.bit_depth = 8
        self.channel_scale, self.channel_shift = compute_scale_shift(self.channels, self.bit_depth, None, None)
        
        # Log stats
        total_samples = self.cumulative_samples[-1]
        if samples_per_video == "auto":
            samples_range = [v["samples"] for v in self.videos]
            logger.info(f"PretrainVideoDataset:")
            logger.info(f"     Num videos: {len(self.videos)}")
            logger.info(f"     Patch size (T,H,W): {self.patch_size}")
            logger.info(f"     Samples per video: auto (range {min(samples_range)}-{max(samples_range)})")
            logger.info(f"     Total samples: {total_samples}")
        else:
            logger.info(f"PretrainVideoDataset:")
            logger.info(f"     Num videos: {len(self.videos)}")
            logger.info(f"     Patch size (T,H,W): {self.patch_size}")
            logger.info(f"     Samples per video: {samples_per_video}    Total: {total_samples}")
    
    def __len__(self):
        return self.cumulative_samples[-1]
    
    def __getitem__(self, idx):
        """Load a sample - retry with different sample if corrupted image detected."""
        import random
        
        # Retry mechanism: try up to 10 times with different samples
        max_retries = 10
        for retry in range(max_retries):
            try:
                # First try original idx, then random samples
                sample_idx = idx if retry == 0 else random.randint(0, len(self) - 1)
                return self._load_sample(sample_idx)
            except (OSError, IOError, struct.error) as e:
                if retry == 0:
                    # Log first failure but continue retrying
                    print(f"[Warning] Corrupted image at idx {idx}, retrying with different sample...")
                if retry == max_retries - 1:
                    # After all retries failed, raise error
                    raise RuntimeError(
                        f"Failed to load valid sample after {max_retries} retries. "
                        f"Last error: {e}"
                    ) from e
        
        raise RuntimeError("Unexpected exit from retry loop")
    
    def _load_sample(self, idx):
        """Load a single sample with corruption detection and logging."""
        import random
        
        # Find which video
        video_idx = bisect.bisect_right(self.cumulative_samples, idx) - 1
        video_info = self.videos[video_idx]
        
        # Random temporal start
        max_start = video_info["num_frames"] - self.temporal_size
        start_frame = random.randint(0, max_start)
        
        # Random spatial crop
        max_h = video_info["height"] - self.crop_size[0]
        max_w = video_info["width"] - self.crop_size[1]
        crop_h = random.randint(0, max_h)
        crop_w = random.randint(0, max_w)
        
        # Load frames with corruption detection
        frames = []
        for i in range(self.temporal_size):
            frame_path = os.path.join(video_info["dir"], video_info["frames"][start_frame + i])
            
            # Skip known bad frames early
            if frame_path in self._bad_frames:
                raise OSError(f"Known bad frame (skipped): {frame_path}")
            
            try:
                # Verify image integrity
                with Image.open(frame_path) as img:
                    img.verify()
                # Re-open after verify (verify closes the file)
                frame = np.array(Image.open(frame_path))
            except (OSError, IOError, struct.error) as e:
                # Mark as bad and log details
                self._bad_frames.add(frame_path)
                print(f"[Corrupted] {frame_path} - {type(e).__name__}: {e}")
                raise  # Re-raise to trigger retry with different sample
            
            frame = frame[crop_h:crop_h + self.crop_size[0], crop_w:crop_w + self.crop_size[1]]
            frames.append(frame)
        
        # Stack [T, H, W, C] -> [C, T, H, W]
        patch = np.stack(frames, axis=0)
        scale = self.channel_scale.view(self.channels, 1, 1, 1)
        shift = self.channel_shift.view(self.channels, 1, 1, 1)
        patch = torch.tensor(patch).permute(3, 0, 1, 2).float() * scale + shift
        patch = torch.clip_(patch, 0., 1.)

        # Augmentation: horizontal flip
        if random.random() < self.hflip_prob:
            patch = torch.flip(patch, dims=[3])
        
        idx_thw = torch.tensor([0, 0, 0], dtype=torch.int32)
        return idx_thw, patch
    
    def get_video_size(self):
        return self.patch_size
    
    def get_patch_size(self):
        return self.patch_size
    
    def get_num_channels(self):
        return self.channels
    
    def get_bit_depth(self):
        return self.bit_depth


def create_pretrain_dataset(
    logger,
    video_dirs=None,
    video_list_file=None,
    patch_size=(16, 256, 256),
    samples_per_video=100,
    hflip_prob=0.5,
    base_dir=None,
    data_sources=None,
):
    """
    Create pretrain dataset from video directories, list file, or multiple data sources.
    
    Args:
        video_dirs: List of video directory paths
        video_list_file: Path to text file with one video directory per line
        base_dir: Base directory for relative paths in video_list_file
        data_sources: List of dicts, each with:
            - base_dir: Base directory for this data source
            - video_list: Path to video list file (relative paths inside)
            - scan_subdirs: If True, scan all subdirectories as videos (default False)
            - samples_per_video: Samples per video for this source (int or "auto")
    """
    all_video_dirs = []
    video_samples_map = {}  # Map video_dir -> samples_per_video for this specific video
    
    # Handle multiple data sources
    if data_sources is not None:
        for src in data_sources:
            src_base = os.path.expanduser(src.get("base_dir", ""))
            src_list = src.get("video_list")
            src_scan = src.get("scan_subdirs", False)
            src_samples = src.get("samples_per_video", samples_per_video)  # Use source-specific or fallback to global
            
            if src_scan:
                # Scan all subdirectories as videos
                if not os.path.exists(src_base):
                    logger.warning(f"Data source not found: {src_base}")
                    continue
                # Walk through subdirectories to find video folders (folders with images)
                count_before = len(all_video_dirs)
                for root, dirs, files in os.walk(src_base):
                    # Check if this folder has images
                    has_images = any(f.endswith((".png", ".jpg")) for f in files)
                    if has_images:
                        all_video_dirs.append(root)
                        video_samples_map[root] = src_samples
                logger.info(f"Scanned {src_base}: found {len(all_video_dirs) - count_before} video folders (samples_per_video={src_samples})")
            elif src_list:
                # Load from video list file
                # video_list path can be relative to CWD or absolute
                src_list = os.path.expanduser(src_list)
                # Try absolute path first, then relative to CWD
                if not os.path.isabs(src_list):
                    src_list = os.path.abspath(src_list)
                if not os.path.exists(src_list):
                    logger.warning(f"Video list not found: {src_list}")
                    continue
                with open(src_list, "r") as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                count_before = len(all_video_dirs)
                for line in lines:
                    if os.path.isabs(line):
                        video_dir = line
                    else:
                        video_dir = os.path.join(src_base, line)
                    all_video_dirs.append(video_dir)
                    video_samples_map[video_dir] = src_samples
                logger.info(f"Loaded {len(all_video_dirs) - count_before} videos from {src_list} (base: {src_base}, samples_per_video={src_samples})")
    
    # Handle legacy single video_list_file
    if video_list_file is not None:
        video_list_file = os.path.expanduser(video_list_file)
        base_dir_expanded = os.path.expanduser(base_dir) if base_dir else None
        with open(video_list_file, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        for line in lines:
            if os.path.isabs(line):
                video_dir = line
            elif base_dir_expanded:
                video_dir = os.path.join(base_dir_expanded, line)
            else:
                video_dir = line
            all_video_dirs.append(video_dir)
            if video_dir not in video_samples_map:
                video_samples_map[video_dir] = samples_per_video
        logger.info(f"Loaded {len(lines)} videos from {video_list_file}")
    
    # Handle explicit video_dirs
    if video_dirs is not None:
        for video_dir in video_dirs:
            all_video_dirs.append(video_dir)
            if video_dir not in video_samples_map:
                video_samples_map[video_dir] = samples_per_video
    
    if len(all_video_dirs) == 0:
        raise ValueError("No video directories provided. Use video_dirs, video_list_file, or data_sources.")
    
    return PretrainVideoDataset(
        logger=logger,
        video_dirs=all_video_dirs,
        patch_size=patch_size,
        samples_per_video=samples_per_video,
        hflip_prob=hflip_prob,
        base_dir=None,  # Already resolved above
        video_samples_map=video_samples_map,  # Pass per-video samples config
    )
