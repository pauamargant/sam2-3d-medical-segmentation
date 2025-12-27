"""
Baseline prompt selection strategy.

Uses evenly spaced slices with margin, matching the original notebook workflow.
"""

from typing import List, Tuple
import numpy as np
from src.interfaces import PromptStrategy
from src.utils.nifti_utils import get_prompt_slices


class BaselinePromptStrategy(PromptStrategy):
    """Baseline strategy that uses evenly spaced prompt slices."""
    
    def __init__(self, num_prompts: int = 4, margin: int = 1):
        """
        Initialize baseline prompt strategy.
        
        Args:
            num_prompts: Number of prompt slices to select (must be even)
            margin: Margin for prompt slice selection
        """
        self.num_prompts = num_prompts
        self.margin = margin
    
    def get_prompts(self, gt_slices: np.ndarray, relevant_indices: Tuple[int, int]) -> List[int]:
        """
        Get prompt slice indices using the baseline strategy.
        
        Wraps the existing get_prompt_slices utility function.
        
        Args:
            gt_slices: Ground truth slices array (shape: [num_slices, height, width])
            relevant_indices: Tuple of (min_slice, max_slice) - not used directly as
                             get_prompt_slices finds annotated slices internally
        
        Returns:
            List of selected slice indices
        """
        # Use the existing get_prompt_slices function which finds annotated slices
        # and selects evenly spaced prompts around them
        return get_prompt_slices(gt_slices, num_slices=self.num_prompts, margin=self.margin)

