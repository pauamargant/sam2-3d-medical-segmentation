"""
Forward propagation strategy.

Adds all prompts simultaneously, then propagates once through all frames.
"""

from typing import List
import numpy as np
from src.interfaces import PropagationStrategy
from src.utils.sam2_utils import add_mask_prompt_to_predictor, propagate_video


class ForwardPropagationStrategy(PropagationStrategy):
    """Forward propagation strategy that adds all prompts then propagates once."""
    
    def __init__(self, direction: str = "forward"):
        """
        Initialize forward propagation strategy.
        
        Args:
            direction: Propagation direction ("forward" for now, "bidirectional" for future)
        """
        self.direction = direction
    
    def propagate(
        self,
        predictor,
        inference_state: dict,
        prompt_indices: List[int],
        gt_slices: np.ndarray,
        num_frames: int,
        frames_dir: str,
    ) -> dict:
        """
        Propagate masks through the volume using forward propagation.
        
        Args:
            predictor: SAM2VideoPredictor instance
            inference_state: Inference state dictionary
            prompt_indices: List of frame indices where prompts are provided
            gt_slices: Ground truth slices array (shape: [num_slices, height, width])
            num_frames: Total number of frames to propagate through
        
        Returns:
            Dictionary mapping frame indices to mask logits
        """
        # Step 1: Loop through prompt_indices and register them all
        for prompt_idx in prompt_indices:
            if prompt_idx < 0 or prompt_idx >= len(gt_slices):
                continue
            mask = gt_slices[prompt_idx]
            add_mask_prompt_to_predictor(mask, prompt_idx, predictor, inference_state)
        
        # Step 2: Propagate using the propagate_video function from utils
        if not prompt_indices:
            return {}
        
        # Start from the first prompt index (matching original behavior)
        start_frame = 0 # min(prompt_indices)
        # Propagate through all remaining frames
        end_frame = num_frames - 1
        
        # Step 3: Use propagate_video from src.utils.sam2_utils
        # This function handles the propagation and returns a dict of frame indices to mask logits
        predicted_masks = propagate_video(predictor, inference_state, start_frame=start_frame, end_frame=end_frame)
        
        return predicted_masks

