"""
Data loading and preprocessing utilities for NIfTI CT volumes.

Handles loading scans/ground truth, applying preprocessing, detecting relevant slices,
and saving extracted slices to temporary directories for SAM2 inference.
Supports single-axis and three-axis (axial, coronal, sagittal) processing modes.
"""

from typing import Callable, Dict, Any
import threading
from omegaconf import DictConfig
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import shutil
import tempfile
from typing import Optional
from src.utils.nifti_utils import (
    load_nifti,
    preprocess_nifti,
    preprocess_nifti_gt,
    detect_relevant_slices,
    extract_ct_slices,
    save_slices,
)
from src.utils.eval import ensure_gt_annotations, validate_gt_span
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class PrepareCaseRequest:
    """Serializable request for preprocessing a single case."""
    scan_path: str
    gt_path: str
    preprocessor_name: str
    scan_fn: Callable
    scan_params: Dict[str, Any]
    gt_fn: Callable
    gt_params: Dict[str, Any]
    tmp_dir: str
    dir_prefix: str
    relevant_slices_margin: int
    case_index: Optional[int] = None
    bone_name: Optional[str] = None


@dataclass
class PreparedCaseData:
    """Outputs from preprocessing ready for inference."""
    ct_slices: np.ndarray
    gt_slices: np.ndarray
    gt_volume: np.ndarray
    min_slice: int
    max_slice: int
    run_dir: str
    scan_path: str
    gt_path: str


@dataclass
class AxisSlicesData:
    """Per-axis slices and metadata."""
    axis: str
    ct_slices: np.ndarray
    gt_slices: np.ndarray
    min_slice: int
    max_slice: int
    target_shape: tuple
    run_dir: str


@dataclass
class PreparedThreeAxisData:
    """Outputs from preprocessing for three-axis inference."""
    axes: Dict[str, AxisSlicesData]
    gt_volume: np.ndarray
    ct_volume: np.ndarray
    run_base_dir: str
    scan_path: str
    gt_path: str


@dataclass
class PrepareThreeAxisRequest:
    """Serializable request for preprocessing a single case across three axes."""
    scan_path: str
    gt_path: str
    preprocessor_name: str
    scan_fn: Callable
    scan_params: Dict[str, Any]
    gt_fn: Callable
    gt_params: Dict[str, Any]
    tmp_dir: str
    dir_prefix: str
    relevant_slices_margin: int
    case_index: Optional[int] = None
    bone_name: Optional[str] = None


def _safe_remove_dir(path: Path) -> None:
    """
        Cleans Up a temporary directory
    
    """
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=False)
    except Exception as e:
        logger.warning(f"Failed to remove tmp dir {path}: {e}")


def _safe_remove_dir_async(path: Path) -> None:
    """
    Spawn a background thread to remove a temporary directory without blocking.
    """
    threading.Thread(target=_safe_remove_dir, args=(path,), daemon=True).start()


def _get_axis_views(vol: np.ndarray):
    """
    Given a volume in (Z, Y, X), return reoriented views for the 3 axes.
    """
    axial = vol  # (Z, Y, X)
    coronal = np.transpose(vol, (1, 0, 2))   # (Y, Z, X)
    sagittal = np.transpose(vol, (2, 0, 1))  # (X, Z, Y)
    return {
        "axial": axial,
        "coronal": coronal,
        "sagittal": sagittal,
    }


def _prepare_case_data(request: PrepareCaseRequest) -> PreparedCaseData:
    """
    Perform data loading, preprocessing, slice detection, extraction, and slice saving.
    Designed to be run in a separate process.
    """
    # Print case information if available
    if request.case_index is not None and request.bone_name is not None:
        logger.info(f"Processing case {request.case_index}: bone = {request.bone_name}")
    elif request.case_index is not None:
        logger.info(f"Processing case {request.case_index}")
    elif request.bone_name is not None:
        logger.info(f"Processing case: bone = {request.bone_name}")
    
    scan_path = Path(request.scan_path)
    gt_path = Path(request.gt_path)

    assert scan_path.exists(), f"NIfTI scan path does not exist: {scan_path}"
    assert gt_path.exists(), f"NIfTI GT path does not exist: {gt_path}"

    scan = load_nifti(str(scan_path))
    gt = load_nifti(str(gt_path))
    assert scan.shape == gt.shape, f"Scan and GT shapes do not match: {scan.shape} vs {gt.shape}"

    logger.debug(
        f"Using '{request.preprocessor_name}' with "
        f"scan_params={request.scan_params}, gt_params={request.gt_params}"
    )
    scan_pp = request.scan_fn(scan, **request.scan_params)
    gt_pp = request.gt_fn(gt, **request.gt_params)
    logger.debug(f"Preprocessed scan shape: {scan_pp.shape}, dtype: {scan_pp.dtype}")
    ensure_gt_annotations(gt_pp, context="run_case")

    min_slice, max_slice = detect_relevant_slices(gt_pp, margin=request.relevant_slices_margin)
    logger.debug(f"Using relevant_slices_margin: {request.relevant_slices_margin}")
    logger.debug(f"Relevant slices from {min_slice} to {max_slice}")
    validate_gt_span(gt_pp, min_slice, max_slice, context="run_case")

    ct_slices = extract_ct_slices(scan_pp, min_slice, max_slice)
    gt_slices = extract_ct_slices(gt_pp, min_slice, max_slice)
    logger.debug(f"Extracted CT slices shape: {ct_slices.shape}")
    logger.debug(f"Extracted GT slices shape: {gt_slices.shape}")

    if ct_slices.shape[0] <= 3:
        raise ValueError(
            f"Not enough relevant slices ({ct_slices.shape[0]}). "
            "Need more than 3 slices to run the case."
        )

    base_tmp_dir = Path(request.tmp_dir)
    base_tmp_dir.mkdir(parents=True, exist_ok=True)

    run_dir: Optional[Path] = None
    try:
        run_dir = Path(
            tempfile.mkdtemp(
                prefix=request.dir_prefix,
                dir=str(base_tmp_dir),
            )
        )
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        save_slices(ct_slices, run_dir)

        return PreparedCaseData(
            ct_slices=ct_slices,
            gt_slices=gt_slices,
            gt_volume=gt_pp,
            min_slice=min_slice,
            max_slice=max_slice,
            run_dir=str(run_dir),
            scan_path=str(scan_path),
            gt_path=str(gt_path),
        )
    except Exception:
        if run_dir:
            _safe_remove_dir(run_dir)
        raise


def _prepare_case_data_three_axis(request: PrepareThreeAxisRequest) -> PreparedThreeAxisData:
    """
    Perform data loading, preprocessing, slice detection, extraction, and slice saving
    for three-axis inference.
    """
    if request.case_index is not None and request.bone_name is not None:
        logger.info(f"Processing case {request.case_index}: bone = {request.bone_name} (three_axis)")
    elif request.case_index is not None:
        logger.info(f"Processing case {request.case_index} (three_axis)")
    elif request.bone_name is not None:
        logger.info(f"Processing case: bone = {request.bone_name} (three_axis)")

    scan_path = Path(request.scan_path)
    gt_path = Path(request.gt_path)

    assert scan_path.exists(), f"NIfTI scan path does not exist: {scan_path}"
    assert gt_path.exists(), f"NIfTI GT path does not exist: {gt_path}"

    scan = load_nifti(str(scan_path))
    gt = load_nifti(str(gt_path))
    assert scan.shape == gt.shape, f"Scan and GT shapes do not match: {scan.shape} vs {gt.shape}"

    logger.debug(
        f"[three_axis] Using '{request.preprocessor_name}' with "
        f"scan_params={request.scan_params}, gt_params={request.gt_params}"
    )
    scan_pp = request.scan_fn(scan, **request.scan_params)
    gt_pp = request.gt_fn(gt, **request.gt_params)
    logger.debug(f"[three_axis] Preprocessed scan shape: {scan_pp.shape}, dtype: {scan_pp.dtype}")
    ensure_gt_annotations(gt_pp, context="three_axis")

    scan_views = _get_axis_views(scan_pp)
    gt_views = _get_axis_views(gt_pp)

    base_tmp_dir = Path(request.tmp_dir)
    base_tmp_dir.mkdir(parents=True, exist_ok=True)

    run_base_dir: Optional[Path] = None
    axes_data: Dict[str, AxisSlicesData] = {}
    try:
        run_base_dir = Path(
            tempfile.mkdtemp(
                prefix=request.dir_prefix,
                dir=str(base_tmp_dir),
            )
        )

        for axis in ["axial", "coronal", "sagittal"]:
            scan_axis = scan_views[axis]
            gt_axis = gt_views[axis]

            min_slice, max_slice = detect_relevant_slices(gt_axis, margin=request.relevant_slices_margin)
            validate_gt_span(gt_axis, min_slice, max_slice, context=f"{axis}_three_axis")

            ct_slices = extract_ct_slices(scan_axis, min_slice, max_slice)
            gt_slices = extract_ct_slices(gt_axis, min_slice, max_slice)

            if ct_slices.shape[0] <= 3:
                raise ValueError(
                    f"[{axis}] Not enough relevant slices ({ct_slices.shape[0]}). "
                    "Need more than 3 slices to run the case."
                )

            axis_dir = run_base_dir / axis
            if axis_dir.exists():
                shutil.rmtree(axis_dir)
            axis_dir.mkdir(parents=True)
            save_slices(ct_slices, axis_dir)

            axes_data[axis] = AxisSlicesData(
                axis=axis,
                ct_slices=ct_slices,
                gt_slices=gt_slices,
                min_slice=min_slice,
                max_slice=max_slice,
                target_shape=scan_axis.shape,
                run_dir=str(axis_dir),
            )

        return PreparedThreeAxisData(
            axes=axes_data,
            gt_volume=gt_pp,
            ct_volume=scan_pp,
            run_base_dir=str(run_base_dir),
            scan_path=str(scan_path),
            gt_path=str(gt_path),
        )
    except Exception:
        if run_base_dir:
            _safe_remove_dir(run_base_dir)
        raise


def _get_preprocessors(cfg: DictConfig):
    """
    Resolve preprocessing functions and their parameters from config.
    Falls back to default preprocessors if a custom module is missing.
    """
    pre_cfg = getattr(cfg, "preprocessing", None)
    name = getattr(pre_cfg, "name", "default") if pre_cfg else "default"
    scan_params = getattr(pre_cfg, "scan_params", {}) if pre_cfg else {}
    gt_params = getattr(pre_cfg, "gt_params", {}) if pre_cfg else {}

    scan_fn = preprocess_nifti
    gt_fn = preprocess_nifti_gt

    if name not in (None, "default"):
        try:
            module = __import__(f"src.preprocessing.{name}", fromlist=["preprocess_scan", "preprocess_gt"])
            scan_fn = getattr(module, "preprocess_scan", scan_fn)
            gt_fn = getattr(module, "preprocess_gt", gt_fn)
        except Exception as e:
            logger.warning(f"Failed to load '{name}', falling back to default. Error: {e}")

    return name or "default", scan_fn, scan_params or {}, gt_fn, gt_params or {}