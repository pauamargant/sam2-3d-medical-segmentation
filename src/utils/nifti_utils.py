"""
------------------------------
NIFTI Utility Functions
------------------------------
This module provides utility functions for processing NIfTI medical imaging files and their annotations (ground truths) for
use with sam2 model. In order to go from raw nifti files from slices ready for the model, please follow:
1. Load the NIfTI files using `load_nifti`. This loads the nifti as a numpy array.
2. Preprocess the CT scan using `preprocess_nifti` to apply windowing and CLAHE for better visualization/model input.
3. Preprocess the ground truth mask using `preprocess_nifti_gt` to convert it to binary masks.
4. Use `select_relevant_slices` to identify slices with annotations. This is used
    to later select only the slices in the ct scan that contain the object of interest.
5. Extract the relevant CT slices using `extract_ct_slices` based on the selected slice indices.
6. Save the processed slices using `save_slices`. This saves them as bmp images in the specified output directory. This is necessarty
    because the sam2 model works with image files as input.
7. The processed slices are now ready for use with the sam2 model for segmentation tasks.

In order to get mask prompts from the ground truth masks to use as inputs to the model:
1. Use `get_prompt_slices` to select specific slices from the ground truth mask that contain annotations. This function selects slices around the annotated region, including top, bottom, and center slices,
    as well as evenly spaced slices around these regions. The selected slices can be used as mask prompts for the model.
------------------------------
"""

import numpy as np
import cv2
from typing import Tuple
from pathlib import Path
import nibabel as nib
from PIL import Image
import sys

# Add python directory to path for imports
python_dir = Path(__file__).parent.parent
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)
def apply_windowing(image: np.ndarray, level: int, width: int) -> np.ndarray:
    """Apply CT windowing to a numpy array and return uint8 image.

    Args:
        image: Input image (HU values). NaNs are allowed.
        level: Window center (e.g. 300).
        width: Window width (e.g. 2000).

    Returns:
        uint8 numpy array scaled to 0..255 using the window.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    min_hu = level - (width / 2)
    max_hu = level + (width / 2)

    # Replace NaNs with minimum display value to avoid propagation
    image = np.nan_to_num(image, nan=min_hu)

    clipped = np.clip(image, min_hu, max_hu)
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    uint8_img = (normalized * 255.0).round().astype(np.uint8)
    return uint8_img


def apply_clahe(image_uint8: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to a uint8 grayscale image.

    If the input is not uint8 it will be converted (via clipping).
    """
    if image_uint8.dtype != np.uint8:
        # Safely convert to uint8 by clipping to 0..255
        image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)
    

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image_uint8)


def preprocess_nifti(scan: np.ndarray, window_level: int = 300, window_width: int = 2000, apply_clahe_flag: bool = True, apply_windowing_flag: bool = True) -> np.ndarray:
    """Preprocess a CT scan for visualization or model input. Returns a list of 2D uint8 slices. """
    preprocessed_slices = []
    for i in range(scan.shape[2]):
        slice_img = scan[:, :, i]

        if apply_windowing_flag:
            windowed = apply_windowing(slice_img, window_level, window_width)
        else:
            windowed = slice_img
        if apply_clahe_flag:    
            windowed = apply_clahe(windowed)
        preprocessed_slices.append(windowed)

    return np.array(preprocessed_slices)

def preprocess_nifti_gt(gt: np.ndarray) -> np.ndarray:
    """Preprocess ground truth mask for visualization or model input. Returns a list of 2D uint8 slices."""
    preprocessed_slices = []
    for i in range(gt.shape[2]):
        slice_mask = gt[:, :, i]

        # Convert to uint8
        slice_mask_uint8 = slice_mask.astype(np.uint8)
        # convert  to binary mask (0 and 1 values only)
        slice_mask_uint8[slice_mask_uint8 > 0] = 1

        preprocessed_slices.append(slice_mask_uint8)

    return np.array(preprocessed_slices)

def load_nifti(file_path: str) -> np.ndarray:
    """Load a NIfTI file and return its data as a numpy array.

    Args:
        file_path: Path to the NIfTI file.

    Returns:
        Numpy array of the image data.
    """
    logger.debug(f"Loading NIfTI file from: {file_path}")
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

def detect_relevant_slices(mask: np.ndarray, margin: float=0.20) -> Tuple[int, int]:
    """
    Detect slices that contain annotations and returns min and max slice indices.
    It adds a margin to include some context around the annotated slices.

    Args:
        mask (np.ndarray): 3D binary mask array where annotations are marked.
        margin (float): Fraction of total slices to include as margin.

    Returns:
        Tuple[int, int]: Minimum and maximum slice indices containing annotations.
        """
    slice_indices = []
    for i in range(mask.shape[0]):
        if np.any(mask[i]):
            slice_indices.append(i)
    if not slice_indices:   
        return 0, 0
    slice_len = len(slice_indices)
    min_idx = max(0, slice_indices[0] - int(margin * slice_len))
    max_idx = min(mask.shape[0] - 1, slice_indices[-1] + int(margin * slice_len))
    return min_idx, max_idx
def extract_ct_slices(scan: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
    """
    Extract CT slices from min_idx to max_idx (inclusive).

    Args:
        scan (np.ndarray): 3D CT scan array.
        min_idx (int): Minimum slice index.
        max_idx (int): Maximum slice index.

    Returns:
        np.ndarray: Extracted CT slices.
    """
    return scan[min_idx:max_idx + 1]


def get_prompt_slices(gt: np.ndarray, num_slices: int = 3, margin: int = 1)->list:
    """
    Select relevant slices from the ground truth mask that contain annotations.
    It assumes that the gt has already been filtered using detect_relevant_slices.
    It returns the top and bottom slices around the annotated region (with a margin) and
    the center slice. In addition it returns num slices divided evenly around the center slice and
    top/bottom slices.
    Args:
        gt (np.ndarray): 3D ground truth mask array.
        num_slices (int): EVEN Number of slices to select around the annotated region.

    Returns:
        list: List of selected slice indices.
    """
    # Find slices with annotations
    annotated_slices = [i for i in range(gt.shape[0]) if np.any(gt[i])]
    if not annotated_slices:
        return []
    min_idx = annotated_slices[0]+margin
    max_idx = annotated_slices[-1]-margin
    center_idx = (min_idx + max_idx) // 2
    selected_slices = set()
    selected_slices.add(min_idx)
    selected_slices.add(max_idx)
    selected_slices.add(center_idx)
    # Add evenly spaced slices around center and min_idx
    num_slices_half = num_slices // 2
    # evenly spaced between min and center
    if num_slices_half > 0:
        for i in range(0, num_slices_half):
            slice_idx = min_idx + i * (center_idx - min_idx) // num_slices_half
            selected_slices.add(slice_idx)
        # evenly spaced between center and max
        for i in range(0, num_slices_half):
            slice_idx = center_idx + i * (max_idx - center_idx) // num_slices_half
            selected_slices.add(slice_idx)
    return list(sorted(selected_slices))



def save_slices(slices: np.ndarray, output_dir: Path):
    """
    Save slices as image files in the specified output directory. 
    Assumes that they are ordered sequentially.
    Saves them with the idx as file_name, starting from 0

    Args:
        slices (np.ndarray): 3D array of slices to save.
        output_dir (Path): Directory to save the slices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(slices.shape[0]):
        slice_img = slices[i]
        pil_image = Image.fromarray((slice_img).astype(np.uint8))
        pil_image.save(output_dir / f"{i:04d}.bmp")