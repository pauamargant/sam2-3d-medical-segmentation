# src/strategy/first_last_middle.py

"""
First–middle–last annotated prompt selection strategy.

Selects the first, middle, and last slices that contain any foreground label
within the provided gt_slices block (already cropped with margin).
"""

from typing import List, Tuple
import numpy as np
from src.interfaces import PromptStrategy


class FirstLastMiddlePromptStrategy(PromptStrategy):
    """Strategy that uses first, middle, and last annotated slices as prompts."""
    
    def __init__(self, margin: int = 0):
        self.margin = margin
        pass
    
    def get_prompts(self, gt_slices: np.ndarray, relevant_indices: Tuple[int, int]) -> List[int]:
        """
        Get prompt slice indices: first, middle, and last non-empty slice of gt_slices.
        
        Args:
            gt_slices: Ground truth slices (N, H, W), already cropped with margin.
            relevant_indices: (min_slice, max_slice) in the original volume (unused here).
        
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

        # No annotation at all: fall back to plain first/middle/last of the block
        if len(nonempty_indices) == 0:
            if num_slices == 1:
                return [0]
            if num_slices == 2:
                return [0, 1]
            first = 0
            last = num_slices - 1
            middle = num_slices // 2
            return sorted(set([first, middle, last]))

        # Only one annotated slice
        if len(nonempty_indices) == 1:
            return [nonempty_indices[0]]

        # Two annotated slices: just first and last
        if len(nonempty_indices) == 2:
            return [nonempty_indices[0], nonempty_indices[1]]
        # if margin is a float between 0 and 1, it means it is a percentage of the number of slices
        if isinstance(self.margin, float) and 0 < self.margin < 1:
            margin = int(self.margin * len(nonempty_indices))
            margin = max(margin, 5)
        else:
            margin = self.margin
        
        # Three or more annotated slices: first, middle, last among annotated ones
        first = nonempty_indices[0] + margin
        last = nonempty_indices[-1] - margin
        mid_pos = len(nonempty_indices) // 2
        middle = nonempty_indices[mid_pos]

        return sorted(set([first, middle, last]))
