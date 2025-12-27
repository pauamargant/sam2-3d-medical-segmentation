"""
Two-pass propagation strategy with re-init and CPU pseudo-labels.

Pass 1:
    - Add only the user-provided prompts on `prompt_indices`.
    - Propagate once through the volume.
    - Build pseudo-GT masks on a subset of frames based on confidence.

Pass 2:
    - Re-initialize the SAM2 inference state from the video directory.
    - Add the original GT prompts again.
    - Add the selected pseudo-GT masks as additional prompts.
    - Propagate once more through the volume and return these predictions.
"""

from typing import List, Dict
from pathlib import Path

import numpy as np
import torch

from src.interfaces import PropagationStrategy
from src.utils.sam2_utils import add_mask_prompt_to_predictor, propagate_video


class TwoPassesPropagationStrategy(PropagationStrategy):
    """
    Two-pass propagation strategy with explicit re-init and pseudo-labels.

    Parameters
    ----------
    use_all_slices_as_pseudo_gt : bool
        If True, all frames from the first pass are candidates for pseudo-GT
        (then filtered by confidence). If False, only frames in `prompt_indices`
        are candidates.
    pseudo_label_threshold : float
        Threshold on predicted probability to consider a pixel foreground
        when building pseudo-label masks.
    min_foreground_ratio : float
        Minimum fraction of pixels that must be foreground in the pseudo-mask
        to consider it a valid pseudo-prompt (avoid tiny noisy blobs).
    max_pseudo_prompts : int
        Maximum number of pseudo-prompts to add. Acts as a cap on memory growth.
        If use_all_slices_as_pseudo_gt is True and you really want all slices,
        set this high.
    assume_logits : bool
        If True, apply sigmoid to outputs from propagate_video.
        If False, assume propagate_video already returns probabilities in [0, 1].
    video_dir : str
        Directory containing the CT slices as frames. We assume
        "../tmp/ct_slices" relative to the working directory by default.
    """

    def __init__(
        self,
        use_all_slices_as_pseudo_gt: bool = False,
        pseudo_label_threshold: float = 0.6,
        min_foreground_ratio: float = 0.01,
        max_pseudo_prompts: int = 20,
        assume_logits: bool = True,
        video_dir: str = "./tmp/ct_slices",
    ):
        self.use_all_slices_as_pseudo_gt = use_all_slices_as_pseudo_gt
        self.pseudo_label_threshold = pseudo_label_threshold
        self.min_foreground_ratio = min_foreground_ratio
        self.max_pseudo_prompts = max_pseudo_prompts
        self.assume_logits = assume_logits
        self.video_dir = Path(video_dir)

    # ---------------------- helpers ---------------------- #

    def _to_numpy(self, x) -> np.ndarray:
        """Robust conversion (torch.Tensor or np.ndarray) -> np.ndarray."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _maybe_sigmoid(self, x_np: np.ndarray) -> np.ndarray:
        """Apply sigmoid if we assume logits; otherwise return as is."""
        if not self.assume_logits:
            return x_np
        return 1.0 / (1.0 + np.exp(-x_np))

    def _add_gt_prompts(
        self,
        predictor,
        inference_state: dict,
        prompt_indices: List[int],
        gt_slices: np.ndarray,
        num_frames: int,
    ) -> None:
        """
        Add ground-truth prompts to the given inference_state.
        """
        for idx in prompt_indices:
            if idx < 0 or idx >= num_frames:
                continue

            gt_mask = gt_slices[idx]
            gt_mask_bin = (gt_mask > 0).astype(np.uint8)

            if gt_mask_bin.sum() == 0:
                # No foreground in this slice, skip
                continue

            add_mask_prompt_to_predictor(
                predictor=predictor,
                inference_state=inference_state,
                prompt_idx=idx,
                mask=gt_mask_bin,
            )

    # ---------------------- main API ---------------------- #

    def propagate(
        self,
        predictor,
        inference_state: dict,
        prompt_indices: List[int],
        gt_slices: np.ndarray,
        num_frames: int,
    ) -> Dict[int, object]:
        """
        Two-pass propagation with re-init.

        Args:
            predictor: SAM2VideoPredictor instance
            inference_state: Inference state dictionary from predictor.init_state(...)
            prompt_indices: List of frame indices where prompts are provided
            gt_slices: Ground truth slices array (shape: [num_slices, H, W])
            num_frames: Total number of frames in the video/volume

        Returns:
            Dictionary mapping frame indices to mask logits from the second pass.
            If no pseudo-GT frames are selected, returns the first-pass logits.
        """
        if not prompt_indices:
            return {}

        # ---------------------- PASS 1 ---------------------- #

        # Add user GT prompts to the given inference_state
        self._add_gt_prompts(
            predictor=predictor,
            inference_state=inference_state,
            prompt_indices=prompt_indices,
            gt_slices=gt_slices,
            num_frames=num_frames,
        )

        start_frame_1 = 0
        end_frame_1 = num_frames - 1

        # First pass over the volume
        first_pass_logits: Dict[int, object] = propagate_video(
            predictor=predictor,
            inference_state=inference_state,
            start_frame=start_frame_1,
            end_frame=end_frame_1,
        )

        # ---------------- build pseudo labels ---------------- #

        # Select candidate indices for pseudo-GT
        if self.use_all_slices_as_pseudo_gt:
            candidate_indices = list(range(start_frame_1, end_frame_1 + 1))
        else:
            candidate_indices = list(prompt_indices)

        pseudo_masks: Dict[int, np.ndarray] = {}
        for idx in candidate_indices:
            if idx not in first_pass_logits:
                continue

            logits = first_pass_logits[idx]
            logits_np = self._to_numpy(logits)
            probs = self._maybe_sigmoid(np.squeeze(logits_np))

            # Binary pseudo mask
            pseudo_mask = (probs > self.pseudo_label_threshold).astype(np.uint8)
            foreground_pixels = int(pseudo_mask.sum())
            total_pixels = pseudo_mask.size
            foreground_ratio = (
                foreground_pixels / float(total_pixels) if total_pixels > 0 else 0.0
            )

            # Filter out empty or tiny masks
            if foreground_pixels == 0:
                continue
            if foreground_ratio < self.min_foreground_ratio:
                continue

            pseudo_masks[idx] = pseudo_mask

            # Enforce a cap on the number of pseudo-prompts if desired
            if (
                not self.use_all_slices_as_pseudo_gt
                and len(pseudo_masks) >= self.max_pseudo_prompts
            ):
                break

        # If no pseudo-GT frames were selected, just return first-pass predictions
        if len(pseudo_masks) == 0:
            return first_pass_logits

        # ---------------------- PASS 2 ---------------------- #

        # Free the original state before re-init to save memory
        del inference_state
        torch.cuda.empty_cache()

        # Re-initialize the inference state from the video directory
        # Assumes the CT slices are saved in ../tmp/ct_slices
        inference_state_2 = predictor.init_state(str(self.video_dir))

        # Add original GT prompts to the fresh state
        self._add_gt_prompts(
            predictor=predictor,
            inference_state=inference_state_2,
            prompt_indices=prompt_indices,
            gt_slices=gt_slices,
            num_frames=num_frames,
        )

        # Add pseudo-GT prompts built from the first pass
        for idx, mask in pseudo_masks.items():
            add_mask_prompt_to_predictor(
                predictor=predictor,
                inference_state=inference_state_2,
                prompt_idx=idx,
                mask=mask,
            )

        start_frame_2 = start_frame_1
        end_frame_2 = end_frame_1

        second_pass_logits: Dict[int, object] = propagate_video(
            predictor=predictor,
            inference_state=inference_state_2,
            start_frame=start_frame_2,
            end_frame=end_frame_2,
        )

        return second_pass_logits
