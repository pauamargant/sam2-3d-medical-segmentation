"""
Abstract interfaces for prompt selection and propagation strategies.

The interfaces define the abstract base classes for the prompt selection and propagation strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class PromptStrategy(ABC):
    """Abstract base class for prompt selection strategies."""
    
    @abstractmethod
    def get_prompts(self, gt_slices: np.ndarray, relevant_indices: Tuple[int, int]) -> List[int]:
        """
        Get prompt slice indices based on ground truth slices and relevant slice range.
        
        Args:
            gt_slices: Ground truth slices array (shape: [num_slices, height, width])
            relevant_indices: Tuple of (min_slice, max_slice) from detect_relevant_slices
        
        Returns:
            List of slice indices to use as prompts
        """
        pass


class PropagationStrategy(ABC):
    """Abstract base class for propagation strategies."""
    
    @abstractmethod
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
        Propagate masks through the volume using the given prompts.
        
        Args:
            predictor: SAM2VideoPredictor instance
            inference_state: Inference state dictionary
            prompt_indices: List of frame indices where prompts are provided
            gt_slices: Ground truth slices array (shape: [num_slices, height, width])
            num_frames: Total number of frames to propagate through
        
        Returns:
            Dictionary mapping frame indices to mask logits
        """
        pass

