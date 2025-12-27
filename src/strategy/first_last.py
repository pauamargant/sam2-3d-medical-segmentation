# src/strategy/first_last.py

"""
First–last annotated prompt selection strategy.

Selects the first and last slices that contain any foreground label
within the provided gt_slices block (already cropped with margin).
"""

from typing import List, Tuple
import numpy as np
from src.interfaces import PromptStrategy


class FirstLastPromptStrategy(PromptStrategy):
    """Strategy that uses the first and last annotated slices as prompts."""
    
    def __init__(self, margin: int = 0):
        self.margin = margin

    def get_prompts(self, gt_slices: np.ndarray, relevant_indices: Tuple[int, int]) -> List[int]:
        """
        Get prompt slice indices: first and last non-empty slice of gt_slices.
        
        Args:
            gt_slices: Ground truth slices (N, H, W), already cropped with margin.
            relevant_indices: (min_slice, max_slice) in the *original* volume (unused here).
        
        Returns:
            List of local indices in [0, N-1] within gt_slices.
        """
        num_slices = gt_slices.shape[0]
        if num_slices == 0:
            raise ValueError("gt_slices has zero slices; cannot select prompts.")

        # Find slices that actually contain foreground
        nonempty_indices = [
            i for i in range(num_slices)
            if np.any(gt_slices[i] > 0)
        ]

        if len(nonempty_indices) == 0:
            # No annotation at all in this block; fall back to middle slice(s)
            if num_slices == 1:
                return [0]
            return [0, num_slices - 1]

        if len(nonempty_indices) == 1:
            # Only one annotated slice; use it
            return [nonempty_indices[0]]

        # First and last annotated slice
        return [nonempty_indices[0] + self.margin, nonempty_indices[-1] - self.margin]
