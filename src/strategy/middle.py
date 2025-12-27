# src/strategy/middle.py

"""
Middle-frame prompt selection strategy.

Selects only the middle slice of the provided gt_slices block.
"""

from typing import List, Tuple
import numpy as np
from src.interfaces import PromptStrategy


class MiddlePromptStrategy(PromptStrategy):
    """Strategy that uses only the middle slice as a prompt."""
    
    def __init__(self):
        """
        No parameters needed.
        """
        pass
    
    def get_prompts(self, gt_slices: np.ndarray, relevant_indices: Tuple[int, int]) -> List[int]:
        """
        Get prompt slice index: the middle slice of gt_slices.
        
        Args:
            gt_slices: Ground truth slices array (shape: [num_slices, height, width])
            relevant_indices: (min_slice, max_slice) in the original volume (unused here).
        
        Returns:
            A list with a single local index in [0, num_slices-1].
        """
        num_slices = gt_slices.shape[0]
        if num_slices == 0:
            raise ValueError("gt_slices has zero slices; cannot select prompts.")
        # standard middle index (for even N, uses the lower middle, e.g. N=4 -> 2)
        middle_idx = num_slices // 2
        return [middle_idx]
