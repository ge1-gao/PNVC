from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

@dataclass
class Frame:
    """Single coded frame in display order."""
    coding_order: int
    display_order: int
    index_references: List[int]
    depth: int
    seq_name: str = ""
    frame_offset: int = 0

    def __post_init__(self):
        self.index_references = list(self.index_references)

    @property
    def frame_type(self) -> str:
        if not self.index_references:
            return "I"
        if len(self.index_references) == 1:
            return "P"
        return "B"

class CodingStructure(ABC):
    def __init__(
        self,
        seq_name: str = "",
        n_frames: int = 1,
        frame_offset: int = 0,
        intra_pos: Optional[List[int]] = None,
        p_pos: Optional[List[int]] = None,
    ) -> None:
        self.seq_name = seq_name
        self.n_frames = n_frames
        self.frame_offset = frame_offset
        self.intra_pos = sorted(intra_pos) if intra_pos is not None else [0]
        self.p_pos = sorted(p_pos) if p_pos is not None else []

        self._validate_positions()
        self.frames = self.compute_coding_struct(
            self.n_frames,
            self.intra_pos,
            self.p_pos,
        )

    def _validate_positions(self) -> None:
        first_frame_is_intra = self.intra_pos and self.intra_pos[0] == 0
        assert first_frame_is_intra, (
            "First frame of the video should an intra frame. Change --intra_pos "
            "to include the frame 0."
        )

        last_frame_is_intra = self.intra_pos[-1] == self.n_frames - 1
        last_frame_is_p = self.p_pos[-1] == self.n_frames - 1 if self.p_pos else False
        assert last_frame_is_intra or last_frame_is_p, (
            "Last frame of the video should be either an intra frame or a P "
            "frame. Add -1 to --intra_pos or --p_pos to include the last frame."
        )

        if len(self.intra_pos) != len(set(self.intra_pos)):
            print(
                f"Found duplicate elements in --intra_pos: {self.intra_pos}.\n"
                "They are automatically removed."
            )
            self.intra_pos = sorted(list(set(self.intra_pos)))

        if len(self.p_pos) != len(set(self.p_pos)):
            print(
                f"Found duplicate elements in --p_pos: {self.p_pos}.\n"
                "They are automatically removed."
            )
            self.p_pos = sorted(list(set(self.p_pos)))

        common_elements = list(set(self.intra_pos).intersection(self.p_pos))
        assert not common_elements, (
            "Frames can not be an I-frame and a P-frame at the same time!\n"
            f"Found --intra_pos={self.intra_pos} --p_pos={self.p_pos}.\n"
            f"Frame(s) {common_elements} are in both arguments, they should "
            "be present only in one of them."
        )

    def compute_coding_struct(
        self, n_frames: int, intra_pos: List[int], p_pos: List[int]
    ) -> List[Frame]:

        frames = []

        # ----- Step 1: fill all the intra frames
        for idx_display_order in intra_pos:
            frames.append(
                Frame(
                    coding_order=len(frames),
                    display_order=idx_display_order,
                    index_references=[],
                    depth=0,  # All intra depth is 0
                    seq_name=self.seq_name,  # Not very elegant... but useful!
                    frame_offset=self.frame_offset,
                )
            )

            frames.sort(key=lambda x: x.display_order)

        def get_closest_past_ref(idx_display_order: int, frames: List[Frame]) -> Frame:
            """Return the biggest display_order present in frames that is still
            smaller than idx_display_order. It corresponds to the index of the
            closest past reference.

            **Everything is in display order**.

            Args:
                idx_display_order (int): Display index of the frame for which
                    we want to find the closest past reference.
                frames (List[Frames]): List of the already coded (available) frames.

            Returns:
                int: Display order of the closest past reference.
            """
            frames.sort(key=lambda x: x.display_order)

            # The * P-frame will used the P3 frame as reference regardless
            # of the actual display order of the * P-frame (from 4 to 7)
            # I0  P3 * I8
            closest_frame = frames[0]

            for frame in frames:
                if frame.display_order >= idx_display_order:
                    break
                closest_frame = frame

            return closest_frame

        def get_closest_future_ref(idx_display_order: int, frames: List[Frame]) -> int:
            """Return the smallest display_order present in frames that is still
            bigger than idx_display_order. It corresponds to the index of the
            closest future reference.

            **Everything is in display order**.

            Args:
                idx_display_order (int): Display index of the frame for which
                    we want to find the closest future reference.
                frames (List[Frames]): List of the already coded (available) frames.

            Returns:
                int: Display order of the closest future reference.
            """
            frames.sort(key=lambda x: x.display_order, reverse=True)

            # The * P-frame will used the I8 frame as reference regardless
            # of the actual display order of the * P-frame (from 4 to 7)
            # I0  P3 * I8
            closest_frame = frames[0]

            for frame in frames:
                if frame.display_order <= idx_display_order:
                    break
                closest_frame = frame

            return closest_frame

        # ----- Step 2: fill all the P frames
        for idx_display_order in p_pos:
            past_ref = get_closest_past_ref(idx_display_order, frames)
            frames.append(
                Frame(
                    coding_order=len(frames),
                    display_order=idx_display_order,
                    index_references=[past_ref.display_order],
                    depth=past_ref.depth + 1,
                    seq_name=self.seq_name,
                    frame_offset=self.frame_offset,
                )
            )

            frames.sort(key=lambda x: x.display_order)

        # ----- Step 3: Fill out the blanks with B-frames in a hierarchical manner
        # Stop when we've filled the coding structure with n_frames
        while len(frames) < n_frames:
            # Iterate on the frames list and stop each time we find a "gap".
            # Create a B frame right in the middle of this gap.
            for i in range(n_frames):
                # Case 1: we've already constructed this frame
                already_coded_frames = [x.display_order for x in frames]
                if i in already_coded_frames:
                    continue

                # Case 2: we need to construct a new frame
                past_ref = get_closest_past_ref(i, frames)
                future_ref = get_closest_future_ref(i, frames)

                # The display order of the frame being creating is equal to the
                # past reference + half the distance between its 2 references
                ref_distance = future_ref.display_order - past_ref.display_order
                idx_display_order = past_ref.display_order + ref_distance // 2
                frames.append(
                    Frame(
                        coding_order=len(frames),
                        display_order=idx_display_order,
                        index_references=[
                            past_ref.display_order,
                            future_ref.display_order,
                        ],
                        depth=max([past_ref.depth, future_ref.depth]) + 1,
                        seq_name=self.seq_name,
                        frame_offset=self.frame_offset,
                    )
                )

                frames.sort(key=lambda x: x.display_order)

                # Loop once more!
                break

        return frames


    def get_number_of_frames(self) -> int:
        return len(self.frames)

    def get_max_depth(self) -> int:
        """Maximum depth of the coding configuration."""
        return max([frame.depth for frame in self.frames])

    def get_max_coding_order(self) -> int:
        """Return the maximum coding order of a coding configuration

        Returns:
            Maximum coding order of the coding configuration
        """
        return max([frame.coding_order for frame in self.frames])

    def get_frame_from_coding_order(self, coding_order: int) -> Optional[Frame]:
        """Return the frame whose coding order is equal to ``coding_order``.
        Return ``None`` if no frame has been found.

        Args:
            coding_order: Coding order for which we want the frame.

        Returns:
            Frame whose coding order is equal to ``coding_order``.
        """
        for frame in self.frames:
            if frame.coding_order == coding_order:
                return frame
        return None

    def get_max_display_order(self) -> int:
        """Return the maximum display order of a coding configuration

        Returns:
            Maximum display order of the coding configuration
        """
        return max([frame.display_order for frame in self.frames])

    def get_frame_from_display_order(self, display_order: int) -> Optional[Frame]:
        """Return the frame whose display order is equal to ``display_order``.
        Return None if no frame has been found.

        Args:
            display_order: Coding order for which we want the frame.

        Returns:
            Frame whose coding order is equal to ``display_order``.
        """
        for frame in self.frames:
            if frame.display_order == display_order:
                return frame
        return None

    def get_all_frames_using_one_ref(self, display_order_ref: int) -> List[Frame]:
        """Return a list of frames using the frame <display_order_ref> as
        a reference.

        Args:
            display_order_ref: Display order of the frame that is used as reference

        Returns:
            List[Frame]: List of frames using one given frame as a reference.
        """

        res = []
        for frame in self.frames:
            if display_order_ref in frame.index_references:
                res.append(frame)
        return res

    def _representative_frame_index(self, chunk_idx: int, chunk_size: int) -> int:
        """
        Return the display-order frame index representing a temporal token.

        The token can span multiple frames (chunk_size > 1). We use the last
        frame covered by the token so that hierarchical dependencies behave
        consistently with the frame-level structure.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be >= 1")
        end_frame = (chunk_idx + 1) * chunk_size - 1
        return min(max(end_frame, 0), self.n_frames - 1)

    def _map_frame_to_chunk(
        self,
        frame_display_order: int,
        chunk_size: int,
        num_chunks: int,
    ) -> int:
        """Map a frame index to the chunk (temporal token) index that contains it."""
        chunk = frame_display_order // chunk_size
        return min(max(chunk, 0), num_chunks - 1)

    def temporal_chunk_order(
        self,
        num_chunks: int,
        chunk_size: int = 1,
    ) -> List[int]:
        """
        Return the coding order for temporal chunks/tokens.

        Args:
            num_chunks: Number of temporal tokens at the current resolution.
            chunk_size: Number of original frames covered by each token
                (thickness along the temporal dimension).

        Returns:
            List of length num_chunks with the inherited coding order index.
        """
        orders: List[int] = []
        for chunk_idx in range(num_chunks):
            frame_idx = self._representative_frame_index(chunk_idx, chunk_size)
            frame = self.get_frame_from_display_order(frame_idx)
            orders.append(frame.coding_order if frame else 0)
        return orders

    def temporal_chunk_dependencies(
        self,
        num_chunks: int,
        chunk_size: int = 1,
    ) -> Dict[int, List[int]]:
        """
        Return the dependency map between temporal tokens.

        Each token inherits the dependencies of the representative frame
        (default: last frame covered by the token), remapped to the token grid.
        """
        dependencies: Dict[int, List[int]] = {}
        for chunk_idx in range(num_chunks):
            frame_idx = self._representative_frame_index(chunk_idx, chunk_size)
            frame = self.get_frame_from_display_order(frame_idx)

            refs = (
                [
                    self._map_frame_to_chunk(ref, chunk_size, num_chunks)
                    for ref in frame.index_references
                ]
                if frame is not None
                else []
            )
            dependencies[chunk_idx] = refs
        return dependencies

    def spatiotemporal_token_order(
        self,
        spatial_order: Any,
        num_chunks: int,
        chunk_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Combine temporal token order with a spatial coding order (e.g., from
        context_model.get_coding_order) for token-level decoding.
        """
        temporal_order = self.temporal_chunk_order(num_chunks, chunk_size)
        dependencies = self.temporal_chunk_dependencies(num_chunks, chunk_size)
        combined_order = combine_spatial_temporal_order(spatial_order, temporal_order)

        return {
            "temporal_order": temporal_order,
            "temporal_dependencies": dependencies,
            "combined_order": combined_order,
        }

    def decoding_schedule(
        self,
        spatial_order: Any,
        num_chunks: int,
        chunk_size: int = 1,
    ) -> List[List[Tuple[int, ...]]]:
        """
        Yield the complete set of tokens per decoding order.

        Returns a list where each element contains all token coordinates that
        share the same decoding step. Tokens can be 2D or 3D (temporal thickness)
        because the spatial order is combined with the temporal token order.
        """
        combined = self.spatiotemporal_token_order(
            spatial_order=spatial_order,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        )["combined_order"]
        return tokens_by_decoding_order(combined)

    def temporal_context_path(
        self,
        target_chunk: int,
        num_chunks: int,
        chunk_size: int = 1,
        max_depth: Optional[int] = None,
        include_target: bool = True,
    ) -> List[int]:
        """Deprecated; use TemporalChunkIndexer.context_for instead."""
        return []


def build_coding_structure(
    num_frames: int,
    gop_size: int = 12,
    structure: str = "LD",
    seq_name: str = "",
) -> CodingStructure:
    """
    Convenience helper mirroring legacy API used by compression modules.

    Args:
        num_frames: Number of display-order frames in the sequence.
        gop_size: Target group-of-picture size.
        structure: "LD" for low-delay, "RA" for random-access.
        seq_name: Optional identifier.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    structure = structure.upper()
    if structure not in {"LD", "RA"}:
        raise ValueError(f"Unsupported structure '{structure}'")

    intra_pos: List[int] = [0]
    p_pos: List[int] = []

    if structure == "LD":
        # periodic I-frames at GOP boundaries, all others are P-frames
        intra_pos = [max(0, i) for i in range(0, num_frames, gop_size)]
        p_pos = [i for i in range(1, num_frames) if i not in intra_pos]
    else:  # RA
        # periodic I-frames at GOP boundaries, B-frames in between
        # last frame must be I or P for valid coding structure
        intra_pos = [max(0, i) for i in range(0, num_frames, gop_size)]
        # ensure last frame is either I or add as P
        if num_frames - 1 not in intra_pos:
            p_pos = [num_frames - 1]
        else:
            p_pos = []

    struct = CodingStructure(
        seq_name=seq_name,
        n_frames=num_frames,
        frame_offset=0,
        intra_pos=intra_pos,
        p_pos=p_pos,
    )
    return struct


def _max_spatial_step(spatial_order: Any) -> int:
    """Return number of spatial steps given a spatial order tensor/array."""
    if hasattr(spatial_order, "max"):
        return int(spatial_order.max()) + 1

    # Fallback for nested lists/tuples
    flat: List[int] = []
    stack = [spatial_order]
    while stack:
        elem = stack.pop()
        if isinstance(elem, (list, tuple)):
            stack.extend(elem)
        else:
            flat.append(int(elem))
    return max(flat) + 1 if flat else 1


def combine_spatial_temporal_order(
    spatial_order: Any,
    temporal_order: Sequence[int],
) -> Any:
    """
    Combine spatial and temporal orders for token-level decoding.

    Works with torch tensors, numpy arrays, or nested python lists. The temporal
    dimension is prepended to match the decoding loop used by the context model.
    """
    num_spatial_steps = _max_spatial_step(spatial_order)

    try:
        import torch
    except Exception:
        torch = None
    try:
        import numpy as np
    except Exception:
        np = None

    if torch is not None and isinstance(spatial_order, torch.Tensor):
        slices = [spatial_order + int(t) * num_spatial_steps for t in temporal_order]
        return torch.stack(slices, dim=0)

    if np is not None and isinstance(spatial_order, np.ndarray):  # type: ignore
        slices = [spatial_order + int(t) * num_spatial_steps for t in temporal_order]
        return np.stack(slices, axis=0)

    # Fallback for nested lists: recursively apply the offset
    def _offset_list(obj: Any, offset: int) -> Any:
        if isinstance(obj, list):
            return [_offset_list(x, offset) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_offset_list(x, offset) for x in obj)
        return int(obj) + offset

    return [_offset_list(spatial_order, int(t) * num_spatial_steps) for t in temporal_order]


def build_spatiotemporal_order(
    video_shape: Tuple[int, int, int, int, int],
    context_mode: str = "checkerboard",
    channel_partition: Optional[str] = None,
    coding_struct: Optional[CodingStructure] = None,
) -> torch.Tensor:
    """Build generic spatiotemporal order using provided coding structure."""
    _, C, T, H, W = video_shape
    device = torch.device("cpu")
    spatial_order = get_coding_order(
        device, (1, C, H, W), context_mode, channel_partition or ""
    )
    if spatial_order.dim() == 2:
        spatial_order = spatial_order.unsqueeze(0).expand(C, -1, -1)
    elif spatial_order.dim() != 3:
        raise ValueError(f"Unexpected spatial_order shape: {tuple(spatial_order.shape)}")
    num_spatial_steps = int(spatial_order.max()) + 1

    if coding_struct is None:
        temporal_order = list(range(T))
    else:
        frames_sorted = sorted(coding_struct.frames, key=lambda f: f.coding_order)
        temporal_order = [f.display_order for f in frames_sorted][:T]

    order = torch.zeros(C, T, H, W, dtype=torch.long)
    for idx_t, display_idx in enumerate(temporal_order):
        if display_idx >= T:
            continue
        order[:, display_idx] = spatial_order + idx_t * num_spatial_steps
    return order


def order_to_permutation(order: torch.Tensor) -> torch.Tensor:
    flat = order.reshape(-1)
    return torch.argsort(flat, stable=True)


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


@lru_cache()
def get_coding_order(device, target_shape, context_mode, channel_partition="1_1_2_4"):
    """
    Return spatial coding order tensor for a given pattern.

    Supports:
        - quincunx
        - checkerboard
        - quadtree
    """
    _, C, H, W = target_shape

    if context_mode == "quincunx":
        context_tensor = torch.tensor([[4, 2, 4, 0], [3, 4, 3, 4], [4, 1, 4, 2]], device=device)
        num_steps = 5
    elif context_mode == "checkerboard":
        context_tensor = torch.tensor([[1, 0], [0, 1]], device=device)
        num_steps = 2
    elif context_mode == "quadtree":
        context_tensor = torch.tensor([[0, 2], [3, 1]], device=device)
        num_steps = 4
    else:
        raise ValueError(f"Invalid context mode: {context_mode}")

    if channel_partition:
        partition = [int(x) for x in channel_partition.split("_")]
        assert len(partition) == num_steps, f"Channel partition must have {num_steps} elements for {context_mode} mode"
        assert sum(partition) == C, f"Sum of channel partition ({sum(partition)}) must equal total channels ({C})"

        channel_order = torch.zeros(C, dtype=torch.long, device=device)
        start_idx = 0
        for step, num_channels in enumerate(partition):
            channel_order[start_idx : start_idx + num_channels] = step
            start_idx += num_channels

        Hp, Wp = context_tensor.size()
        spatial_order = torch.tile(context_tensor, (H // Hp + 1, W // Wp + 1))[:H, :W]

        coding_order = spatial_order.unsqueeze(0).repeat(C, 1, 1)
        coding_order = coding_order * num_steps + channel_order.view(C, 1, 1)
    else:
        Hp, Wp = context_tensor.size()
        coding_order = torch.tile(context_tensor, (H // Hp + 1, W // Wp + 1))[:H, :W]

    return coding_order


def tokens_by_decoding_order(order_tensor: Any) -> List[List[Tuple[int, ...]]]:
    """
    Group coordinates by decoding step for any order tensor.

    Accepts torch tensors, numpy arrays, or nested lists. Returns a list where
    index `i` contains all coordinates decoded at step `i`.
    """
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if torch is not None and isinstance(order_tensor, torch.Tensor):
        order_flat = order_tensor.reshape(-1)
        shape = tuple(order_tensor.shape)
        result: Dict[int, List[Tuple[int, ...]]] = {}
        for linear_idx, step in enumerate(order_flat.tolist()):
            coord = numpy_unravel_index(linear_idx, shape)
            result.setdefault(int(step), []).append(coord)
        return _dict_to_step_list(result)

    if np is not None and isinstance(order_tensor, np.ndarray):  # type: ignore
        flat = order_tensor.reshape(-1).tolist()
        shape = order_tensor.shape
        result: Dict[int, List[Tuple[int, ...]]] = {}
        for linear_idx, step in enumerate(flat):
            coord = np.unravel_index(linear_idx, shape)
            result.setdefault(int(step), []).append(tuple(int(c) for c in coord))
        return _dict_to_step_list(result)

    flat: List[int] = []
    coords: List[Tuple[int, ...]] = []
    linear = 0

    def _walk(obj: Any, prefix: Tuple[int, ...] = ()) -> None:
        nonlocal linear
        if isinstance(obj, list):
            for i, val in enumerate(obj):
                _walk(val, prefix + (i,))
        else:
            flat.append(int(obj))
            coords.append(prefix)
            linear += 1

    _walk(order_tensor)
    result: Dict[int, List[Tuple[int, ...]]] = {}
    for step, coord in zip(flat, coords):
        result.setdefault(int(step), []).append(coord)
    return _dict_to_step_list(result)


def _dict_to_step_list(result: Dict[int, List[Tuple[int, ...]]]) -> List[List[Tuple[int, ...]]]:
    if not result:
        return []
    max_step = max(result.keys())
    return [result.get(i, []) for i in range(max_step + 1)]


def numpy_unravel_index(linear_idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Small numpy-like unravel_index without importing numpy (used in torch path)."""
    coords = []
    for dim in reversed(shape):
        coords.append(linear_idx % dim)
        linear_idx //= dim
    return tuple(reversed(coords))


# ------------------------------------------------------------
# Dataset-aligned temporal/spatial helpers
# ------------------------------------------------------------
@dataclass
class TemporalChunkIndexer:
    """
    Helper to align temporal chunk order/dependencies with dataset patching.

    Args:
        idx_max: (T_chunks, H_chunks, W_chunks) from dataset.video.get_idx_max()
        patch_size: (T_patch, H_patch, W_patch) from dataset.video.get_patch_size()
        coding_struct: optional CodingStructure to derive I/P/B deps. If None,
                       a simple low-delay structure is built over the total frames.
    """
    idx_max: Tuple[int, int, int]
    patch_size: Tuple[int, int, int]
    coding_struct: Optional[CodingStructure] = None
    default_structure: str = "LD"

    def __post_init__(self) -> None:
        self.num_chunks = int(self.idx_max[0])
        self.chunk_size = int(self.patch_size[0])
        self.total_frames = self.num_chunks * self.chunk_size

        if self.coding_struct is None:
            # build structure at chunk level - each chunk is a "3D frame"
            self.coding_struct = build_coding_structure(
                num_frames=self.num_chunks,
                gop_size=max(1, self.num_chunks),
                structure=self.default_structure,
            )

        # cache chunk order and dependencies directly from coding structure
        self._chunk_coding_order = self._compute_chunk_coding_order()
        self._chunk_dependencies = self._compute_chunk_dependencies()

    def _compute_chunk_coding_order(self) -> List[int]:
        """Get coding order directly from coding structure (already at chunk level)."""
        orders = []
        for chunk_idx in range(self.num_chunks):
            frame = self.coding_struct.get_frame_from_display_order(chunk_idx)
            orders.append(frame.coding_order if frame else chunk_idx)
        return orders

    def _compute_chunk_dependencies(self) -> Dict[int, List[int]]:
        """Get dependencies directly from coding structure (already at chunk level)."""
        dependencies: Dict[int, List[int]] = {}
        for chunk_idx in range(self.num_chunks):
            frame = self.coding_struct.get_frame_from_display_order(chunk_idx)
            if frame is not None and frame.index_references:
                dependencies[chunk_idx] = list(frame.index_references)
            else:
                dependencies[chunk_idx] = []
        return dependencies

    def temporal_order(self) -> List[int]:
        """Return normalized coding order for each chunk (0 to num_chunks-1)."""
        return self._chunk_coding_order

    def temporal_dependencies(self) -> Dict[int, List[int]]:
        """Return cached dependency map between temporal chunks."""
        return self._chunk_dependencies

    def spatiotemporal_order(self, spatial_order: Any) -> Any:
        """Combine spatial order (e.g., quadtree/quincunx) with temporal order."""
        return combine_spatial_temporal_order(spatial_order, self.temporal_order())

    def decoding_schedule(self, spatial_order: Any) -> List[List[Tuple[int, ...]]]:
        combined = self.spatiotemporal_order(spatial_order)
        return tokens_by_decoding_order(combined)

    def context_for(
        self, idx_thw: Tuple[int, int, int], spatial_order: Any
    ) -> Dict[str, Any]:
        """
        Given a 3D token index (t, h, w), return temporal refs and spatial context
        within the same temporal chunk.

        temporal_refs: list of chunk indices this chunk depends on
        spatial_refs: list of (h, w) coordinates in the same t with smaller spatial order
        """
        t, h, w = (int(idx_thw[0]), int(idx_thw[1]), int(idx_thw[2]))

        # temporal references from coding structure
        time_refs = self.temporal_dependencies().get(t, [])

        # spatial context: positions with smaller coding step than (h, w)
        spatial_refs: List[Tuple[int, ...]] = _spatial_context_coords(spatial_order, h, w)

        return {"temporal_refs": time_refs, "spatial_refs": spatial_refs}

    def get_chunk_info(self, chunk_idx: int) -> Dict[str, Any]:
        """
        Get detailed info for a specific chunk (3D frame).

        Returns:
            {
                "chunk_idx": int,
                "frame_range": (start_frame, end_frame),  # original frame indices
                "coding_order": int,
                "temporal_refs": [chunk indices],
                "frame_type": "I" / "P" / "B",
            }
        """
        start_frame = chunk_idx * self.chunk_size
        end_frame = start_frame + self.chunk_size - 1
        frame = self.coding_struct.get_frame_from_display_order(chunk_idx)

        return {
            "chunk_idx": chunk_idx,
            "frame_range": (start_frame, end_frame),
            "coding_order": self._chunk_coding_order[chunk_idx],
            "temporal_refs": self._chunk_dependencies.get(chunk_idx, []),
            "frame_type": frame.frame_type if frame else "I",
        }


def _spatial_context_coords(spatial_order: Any, h: int, w: int) -> List[Tuple[int, ...]]:
    """
    Return coordinates in the same temporal slice whose spatial order
    is strictly smaller than the position (h, w).

    Supports spatial_order shapes:
        - [H, W]
        - [C, H, W]
    """
    order = torch.as_tensor(spatial_order)
    if order.dim() == 2:
        step = order[h, w]
        mask = order < step
        coords = [(int(i), int(j)) for i, j in zip(*torch.nonzero(mask, as_tuple=True))]
        return coords
    elif order.dim() == 3:
        C = order.shape[0]
        coords: List[Tuple[int, int, int]] = []
        for c in range(C):
            step = order[c, h, w]
            mask = order[c] < step
            for i, j in zip(*torch.nonzero(mask, as_tuple=True)):
                coords.append((c, int(i), int(j)))
        return coords
    else:
        raise ValueError(f"Unexpected spatial_order shape: {tuple(order.shape)}")


def build_simple_spatiotemporal_plan(
    video,
    context_mode: str = "checkerboard",
    channel_partition: str = "",
    coding_struct: Optional[CodingStructure] = None,
    structure_type: str = "LD",
) -> Dict[str, Any]:
    """
    Convenience helper to get all spatiotemporal metadata from a dataset.video.

    Args:
        video: object with get_idx_max(), get_patch_size(), get_num_channels()
        context_mode: spatial coding order mode (checkerboard, quadtree, etc.)
        channel_partition: channel grouping mode
        coding_struct: optional pre-built CodingStructure (frame-level)
        structure_type: "LD" (low-delay) or "RA" (random-access) if no coding_struct

    Returns:
        {
            "indexer": TemporalChunkIndexer instance,
            "temporal_order": [...],
            "temporal_dependencies": {t: [refs]},
            "spatial_order": Tensor [C or 1, H, W],
            "spatiotemporal_order": Tensor [C or 1, T, H, W],
            "decoding_schedule": list of token coords per step,
            "chunk_info": list of per-chunk info dicts,
        }
    """
    idx_max = video.get_idx_max()  # (T_chunks, H_chunks, W_chunks)
    patch_size = video.get_patch_size()  # (T_patch, H_patch, W_patch)
    C = video.get_num_channels()
    H = patch_size[1]
    W = patch_size[2]

    idxer = TemporalChunkIndexer(
        idx_max=idx_max,
        patch_size=patch_size,
        coding_struct=coding_struct,
        default_structure=structure_type,
    )

    device = torch.device("cpu")
    spatial_order = get_coding_order(device, (1, C, H, W), context_mode, channel_partition)

    temporal_order = idxer.temporal_order()
    temporal_dependencies = idxer.temporal_dependencies()
    st_order = idxer.spatiotemporal_order(spatial_order)
    schedule = tokens_by_decoding_order(st_order)

    # build per-chunk info for debugging/inspection
    chunk_info = [idxer.get_chunk_info(i) for i in range(idxer.num_chunks)]

    return {
        "indexer": idxer,
        "temporal_order": temporal_order,
        "temporal_dependencies": temporal_dependencies,
        "spatial_order": spatial_order,
        "spatiotemporal_order": st_order,
        "decoding_schedule": schedule,
        "chunk_info": chunk_info,
    }
