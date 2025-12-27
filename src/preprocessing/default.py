"""
Default preprocessing that mirrors the existing behavior from src.utils.nifti_utils.
"""

from typing import Any

from src.utils.nifti_utils import preprocess_nifti, preprocess_nifti_gt


def preprocess_scan(scan: Any, **kwargs):
    """Forward to the standard preprocess_nifti."""
    return preprocess_nifti(scan, **kwargs)


def preprocess_gt(gt: Any, **kwargs):
    """Forward to the standard preprocess_nifti_gt."""
    return preprocess_nifti_gt(gt, **kwargs)

