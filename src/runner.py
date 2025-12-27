"""
Experiment Runner

This module orchestrates the segmentation pipeline for 3D CT volumes using SAM2

It utilises:
Inference Strategies (InferenceStrategy subclasses) which define the inference pipeline:
   - BaselineAxisStrategy: Single-axis (axial) propagation
   - ForwardBackwardStrategy: Bidirectional propagation for improved coverage
   - ThreeAxisStrategy: Fuses predictions from all three axes via majority voting
ExperimentRunner: orchestrator that:
   - Loads and initializes the SAM2 video predictor
   - Manages prompt selection (which slices to annotate) via PromptStrategy
   - Controls mask propagation behavior via PropagationStrategy
   - Handles preprocessing (windowing, CLAHE, normalization)
   - Computes metrics: Dice coefficient, IoU, 95th percentile Hausdorff distance
"""

# Standard library imports
import contextlib
import json
import os
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Callable, Dict, List, Optional
import inspect

# Third-party imports
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Add python directory to path for imports
python_dir = Path(__file__).parent.parent
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

# Add SAM2 source path for imports
sam2_dir = python_dir.parent / "src" / "base_sam2"
if str(sam2_dir) not in sys.path:
    sys.path.insert(0, str(sam2_dir))

# Local application imports
from sam2.build_sam import build_sam2_video_predictor
from src.dataset import Dataset
from src.interfaces import PromptStrategy, PropagationStrategy
from src.utils.configs import _build_hydra_overrides, _format_scalar_for_hydra
from src.utils.dataloader import (
    PrepareCaseRequest,
    PreparedCaseData,
    PrepareThreeAxisRequest,
    PreparedThreeAxisData,
    AxisSlicesData,
    _get_preprocessors,
    _prepare_case_data,
    _prepare_case_data_three_axis,
    _safe_remove_dir,
    _safe_remove_dir_async,
)
from src.utils.logging_config import get_logger
from src.utils.eval import (
    build_full_volume_from_predictions,
    compute_segmentation_metrics,
    compute_segmentation_metrics_multiple,
    log_section,
)

logger = get_logger(__name__)


def invert_axis_view(pred_vol: np.ndarray, axis: str, original_shape):
    if axis == "axial":
        out = pred_vol
    elif axis == "coronal":
        out = np.transpose(pred_vol, (1, 0, 2))
    elif axis == "sagittal":
        out = np.transpose(pred_vol, (1, 2, 0))
    else:
        raise ValueError(f"Unknown axis: {axis}")

    assert out.shape == original_shape, f"{axis} reorientation shape mismatch: {out.shape} vs {original_shape}"
    return out


def run_single_axis_inference(
    *,
    cfg: DictConfig,
    axis: str,
    ct_slices: np.ndarray,
    gt_slices: np.ndarray,
    min_slice: int,
    max_slice: int,
    target_shape,
    run_dir: Path,
    device: str,
    prompt_strategy: PromptStrategy,
    propagation_strategy_cls,
    propagation_cfg: DictConfig,
    predictor,
    direction: Optional[str] = None,
):
    """
    Shared single-axis inference routine.
    Returns logits volume embedded in target_shape and timing/coverage metadata.
    """
    num_frames = len(ct_slices)
    expected_frames = (max_slice - min_slice) + 1
    if num_frames != expected_frames:
        raise ValueError(
            f"[{axis}] Expected {expected_frames} frames from slices span but got {num_frames}"
        )

    prompt_indices = prompt_strategy.get_prompts(gt_slices, (min_slice, max_slice))
    logger.info(f"[{axis}] Selected prompt indices: {prompt_indices}")

    prop_kwargs = {k: v for k, v in propagation_cfg.items() if k not in ("name", "direction")}
    init_params = inspect.signature(propagation_strategy_cls.__init__).parameters
    if "video_dir" in init_params:
        prop_kwargs["video_dir"] = str(run_dir)
    if "direction" in init_params and direction is not None:
        prop_kwargs["direction"] = direction

    propagation_strategy: PropagationStrategy = propagation_strategy_cls(**prop_kwargs)

    if device == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif device == "cpu":
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    propagation_start = time.perf_counter()
    with autocast_ctx:
        inference_state = predictor.init_state(str(run_dir))
        predicted_masks = propagation_strategy.propagate(
            predictor=predictor,
            inference_state=inference_state,
            prompt_indices=prompt_indices,
            gt_slices=gt_slices,
            num_frames=num_frames,
            frames_dir=str(run_dir),
        )
    predictor.reset_state(inference_state)
    propagation_seconds = time.perf_counter() - propagation_start
    propagation_seconds_per_slice = (
        propagation_seconds / num_frames if num_frames else None
    )

    full_pred_vol, missing_slices = build_full_volume_from_predictions(
        predicted_masks,
        num_frames=num_frames,
        min_slice=min_slice,
        target_shape=target_shape,
        log_prefix=f"[{axis}] ",
    )

    return {
        "prompt_indices": prompt_indices,
        "pred_volume": full_pred_vol,
        "predicted_masks": predicted_masks,
        "missing_slices": missing_slices,
        "partial_coverage": len(missing_slices) > 0,
        "propagation_seconds": propagation_seconds,
        "propagation_seconds_per_slice": propagation_seconds_per_slice,
    }


class InferenceStrategy:
    """Interface for per-mode inference strategies."""

    def __init__(
        self,
        cfg: DictConfig,
        prompt_strategy: PromptStrategy,
        propagation_strategy_cls,
        propagation_params: Dict[str, Any],
        device: str,
    ):
        self.cfg = cfg
        self.prompt_strategy = prompt_strategy
        self.propagation_strategy_cls = propagation_strategy_cls
        self.propagation_params = propagation_params
        self.device = device

    def run_prepared(self, prepared, *, get_predictor: Callable[[], Any], return_artifacts: bool = False) -> dict:
        raise NotImplementedError


class BaselineAxisStrategy(InferenceStrategy):
    """Single-axis baseline strategy (axial)."""

    def run_prepared(
        self,
        prepared: PreparedCaseData,
        *,
        get_predictor: Callable[[], Any],
        return_artifacts: bool = False,
    ) -> dict:
        predictor = get_predictor()
        run_dir = Path(prepared.run_dir)
        result = run_single_axis_inference(
            cfg=self.cfg,
            axis="axial",
            ct_slices=prepared.ct_slices,
            gt_slices=prepared.gt_slices,
            min_slice=prepared.min_slice,
            max_slice=prepared.max_slice,
            target_shape=prepared.gt_volume.shape,
            run_dir=run_dir,
            device=self.device,
            prompt_strategy=self.prompt_strategy,
            propagation_strategy_cls=self.propagation_strategy_cls,
            propagation_cfg=self.propagation_params,
            predictor=predictor,
        )

        mean_dice, mean_iou, mean_hausdorff = compute_segmentation_metrics(
            result["pred_volume"], prepared.gt_volume
        )

        output = {
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "mean_hausdorff": mean_hausdorff,
            "prompt_indices": result["prompt_indices"],
            "num_frames": len(prepared.ct_slices),
            "partial_coverage": result["partial_coverage"],
            "missing_slices": result["missing_slices"],
            "propagation_seconds": result["propagation_seconds"],
            "propagation_seconds_per_slice": result["propagation_seconds_per_slice"],
        }

        if return_artifacts:
            def _to_numpy_local(arr):
                if isinstance(arr, torch.Tensor):
                    return arr.detach().cpu().numpy()
                return np.asarray(arr)

            output["artifacts"] = {
                "ct_slices": np.asarray(prepared.ct_slices),
                "gt_slices": np.asarray(prepared.gt_slices),
                "gt_volume": np.asarray(prepared.gt_volume),
                "predicted_logits": {k: _to_numpy_local(v).squeeze() for k, v in result["predicted_masks"].items()},
                "full_pred_volume": np.asarray(result["pred_volume"]),
                "slice_span": (prepared.min_slice, prepared.max_slice),
            }

        return output


class ForwardBackwardStrategy(InferenceStrategy):
    """Forward–backward axial strategy."""

    def run_prepared(
        self,
        prepared: PreparedCaseData,
        *,
        get_predictor: Callable[[], Any],
        return_artifacts: bool = False,
    ) -> dict:
        predictor = get_predictor()
        run_dir = Path(prepared.run_dir)

        forward = run_single_axis_inference(
            cfg=self.cfg,
            axis="axial_forward",
            ct_slices=prepared.ct_slices,
            gt_slices=prepared.gt_slices,
            min_slice=prepared.min_slice,
            max_slice=prepared.max_slice,
            target_shape=prepared.gt_volume.shape,
            run_dir=run_dir,
            device=self.device,
            prompt_strategy=self.prompt_strategy,
            propagation_strategy_cls=self.propagation_strategy_cls,
            propagation_cfg=self.propagation_params,
            predictor=predictor,
            direction="forward",
        )
        backward = run_single_axis_inference(
            cfg=self.cfg,
            axis="axial_backward",
            ct_slices=prepared.ct_slices,
            gt_slices=prepared.gt_slices,
            min_slice=prepared.min_slice,
            max_slice=prepared.max_slice,
            target_shape=prepared.gt_volume.shape,
            run_dir=run_dir,
            device=self.device,
            prompt_strategy=self.prompt_strategy,
            propagation_strategy_cls=self.propagation_strategy_cls,
            propagation_cfg=self.propagation_params,
            predictor=predictor,
            direction="backward",
        )

        mean_dice, mean_iou, mean_hausdorff = compute_segmentation_metrics_multiple(
            [forward["pred_volume"], backward["pred_volume"]],
            prepared.gt_volume,
        )

        partial_coverage = forward["partial_coverage"] or backward["partial_coverage"]
        missing_slices = {
            "forward": forward["missing_slices"],
            "backward": backward["missing_slices"],
        }

        output = {
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "mean_hausdorff": mean_hausdorff,
            "prompt_indices": forward["prompt_indices"],
            "num_frames": len(prepared.ct_slices),
            "partial_coverage": partial_coverage,
            "missing_slices": missing_slices,
            "propagation_seconds": forward["propagation_seconds"] + backward["propagation_seconds"],
            "propagation_seconds_per_slice": None,
        }

        if return_artifacts:
            output["artifacts"] = {
                "ct_slices": np.asarray(prepared.ct_slices),
                "gt_slices": np.asarray(prepared.gt_slices),
                "gt_volume": np.asarray(prepared.gt_volume),
                "forward_pred_volume": np.asarray(forward["pred_volume"]),
                "backward_pred_volume": np.asarray(backward["pred_volume"]),
            }

        return output


class ThreeAxisStrategy(InferenceStrategy):
    """Three-axis fusion strategy."""

    def run_prepared(
        self,
        prepared: PreparedThreeAxisData,
        *,
        get_predictor: Callable[[], Any],
        return_artifacts: bool = False,
    ) -> dict:
        predictor = get_predictor()

        axis_results: Dict[str, dict] = {}
        for axis, axis_data in prepared.axes.items():
            axis_results[axis] = run_single_axis_inference(
                cfg=self.cfg,
                axis=axis,
                ct_slices=axis_data.ct_slices,
                gt_slices=axis_data.gt_slices,
                min_slice=axis_data.min_slice,
                max_slice=axis_data.max_slice,
                target_shape=axis_data.target_shape,
                run_dir=Path(axis_data.run_dir),
                device=self.device,
                prompt_strategy=self.prompt_strategy,
                propagation_strategy_cls=self.propagation_strategy_cls,
                propagation_cfg=self.propagation_params,
                predictor=predictor,
            )

        original_shape = prepared.gt_volume.shape
        axial_pred = invert_axis_view(axis_results["axial"]["pred_volume"], "axial", original_shape)
        coronal_pred = invert_axis_view(axis_results["coronal"]["pred_volume"], "coronal", original_shape)
        sagittal_pred = invert_axis_view(axis_results["sagittal"]["pred_volume"], "sagittal", original_shape)

        mean_dice, mean_iou, mean_hausdorff = compute_segmentation_metrics_multiple(
            [axial_pred, coronal_pred, sagittal_pred],
            prepared.gt_volume,
        )

        partial_flags = {ax: res["partial_coverage"] for ax, res in axis_results.items()}
        missing_by_axis = {ax: res["missing_slices"] for ax, res in axis_results.items()}

        output = {
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "mean_hausdorff": mean_hausdorff,
            "partial_coverage": any(partial_flags.values()),
            "partial_coverage_axes": partial_flags,
            "missing_slices": missing_by_axis,
            "prompt_indices": {ax: res["prompt_indices"] for ax, res in axis_results.items()},
            "propagation_seconds": sum(res["propagation_seconds"] for res in axis_results.values()),
            "propagation_seconds_per_slice": None,
        }

        if return_artifacts:
            logit_stack = torch.stack(
                [
                    torch.as_tensor(axial_pred).float(),
                    torch.as_tensor(coronal_pred).float(),
                    torch.as_tensor(sagittal_pred).float(),
                ],
                dim=0,
            )
            fused_probs = torch.sigmoid(logit_stack).mean(dim=0).cpu().numpy()
            fused_bin = (fused_probs > 0.5).astype(np.uint8)

            output["artifacts"] = {
                "ct_volume": np.asarray(prepared.ct_volume),
                "gt_volume": np.asarray(prepared.gt_volume),
                "axis_pred_logits": {
                    "axial": np.asarray(axial_pred),
                    "coronal": np.asarray(coronal_pred),
                    "sagittal": np.asarray(sagittal_pred),
                },
                "fused_probs": fused_probs,
                "fused_pred_bin": fused_bin,
                "partial_coverage_axes": partial_flags,
                "missing_slices": missing_by_axis,
            }

        return output



class ExperimentRunner:
    """Main experiment runner that orchestrates the segmentation pipeline."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the experiment runner with configuration.
        
        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        
        # Dynamically load and instantiate strategy
        strategy_name = cfg.strategy.name
        strategy_class_name = strategy_name.title().replace("_", "") + "PromptStrategy"
        strategy_module = __import__(f"src.strategy.{strategy_name}", fromlist=[strategy_class_name])
        strategy_class = getattr(strategy_module, strategy_class_name)
        strategy_params = {k: v for k, v in cfg.strategy.items() if k != "name"}
        self.prompt_strategy: PromptStrategy = strategy_class(**strategy_params)
        
        # Dynamically load and instantiate propagation
        propagation_name = cfg.propagation.name
        propagation_class_name = propagation_name.title().replace("_", "") + "PropagationStrategy"
        propagation_module = __import__(f"src.propagation.{propagation_name}", fromlist=[propagation_class_name])
        propagation_class = getattr(propagation_module, propagation_class_name)
        self.propagation_strategy_cls = propagation_class
        self.propagation_params = {k: v for k, v in cfg.propagation.items() if k != "name"}
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Enable TF32 for Ampere GPUs for improved performance
        if self.device == 'cuda' and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Predictor built once, reused for all cases
        self._predictor = None
        self._predictor_init_seconds = None

        inference_cfg = getattr(self.cfg, "inference", None)
        inference_mode = getattr(inference_cfg, "mode", "baseline") if inference_cfg is not None else "baseline"
        strategy_registry = {
            "baseline": BaselineAxisStrategy,
            "axial": BaselineAxisStrategy,
            "bidirectional": ForwardBackwardStrategy,
            "fb": ForwardBackwardStrategy,
            "forward_backward": ForwardBackwardStrategy,
            "three_axis": ThreeAxisStrategy,
        }
        strategy_cls = strategy_registry.get(inference_mode, BaselineAxisStrategy)
        self.inference_strategy: InferenceStrategy = strategy_cls(
            cfg=self.cfg,
            prompt_strategy=self.prompt_strategy,
            propagation_strategy_cls=self.propagation_strategy_cls,
            propagation_params=self.propagation_params,
            device=self.device,
        )
        logger.info(f"Inference mode set to '{inference_mode}' using {strategy_cls.__name__}")
    
    def _get_predictor(self):
        """
        Get or build the SAM2 predictor. Builds it once on first call and reuses it.
        
        Returns:
            The SAM2 video predictor instance
        """
        if self._predictor is None:
            checkpoint_path = Path(self.cfg.model.checkpoint)
            if not checkpoint_path.exists():
                # try to use path from parent folder of root of path
                new_path = Path(os.path.join(Path(__file__).parent.parent, checkpoint_path))
                if not new_path.exists():
                    raise FileNotFoundError(f"SAM2 checkpoint does not exist: {new_path}")
                checkpoint_path = new_path
            
            model_cfg = self.cfg.model.config
            custom_overrides = getattr(self.cfg.model, "custom_overrides", None)
            model_overrides = {}
            hydra_overrides_extra = []
            if custom_overrides is not None:
                if isinstance(custom_overrides, Mapping):
                    model_overrides.update(custom_overrides)
                    hydra_overrides_extra = _build_hydra_overrides("model", model_overrides, override_prefix="++")

            assert checkpoint_path.exists(), f"SAM2 checkpoint does not exist: {checkpoint_path}"

            maskmem_tpos_ckpt = getattr(self.cfg.model, "maskmem_tpos_ckpt", None)

            predictor_init_start = time.perf_counter()
            self._predictor = build_sam2_video_predictor(
                model_cfg,
                str(checkpoint_path),
                device=self.device,
                hydra_overrides_extra=hydra_overrides_extra,
                maskmem_tpos_ckpt=maskmem_tpos_ckpt,
            )
            self._predictor_init_seconds = time.perf_counter() - predictor_init_start
            logger.info(f"SAM2 predictor init: {self._predictor_init_seconds:.3f}s")
        return self._predictor
    
    def run_case(self, scan_path: str, gt_path: str, return_artifacts: bool = False) -> dict:
        """
        Run a single case through the segmentation pipeline.
        
        Args:
            scan_path: Path to the NIfTI scan file
            gt_path: Path to the NIfTI ground truth file
            return_artifacts: When True, also return intermediate artifacts for visualization
        
        Returns:
            Dictionary containing results and metrics
        """
        scan_path = Path(scan_path)
        gt_path = Path(gt_path)

        pre_name, scan_fn, scan_params, gt_fn, gt_params = _get_preprocessors(self.cfg)
        logger.info(f"Using '{pre_name}' with scan_params={scan_params}, gt_params={gt_params}")
        inference_mode = getattr(getattr(self.cfg, "inference", None), "mode", "baseline")
        dir_prefix = f"{inference_mode}_{self.cfg.propagation.name}_"
        relevant_margin = self.cfg.processing.relevant_slices_margin

        if isinstance(self.inference_strategy, ThreeAxisStrategy):
            prepared = _prepare_case_data_three_axis(
                PrepareThreeAxisRequest(
                    scan_path=str(scan_path),
                    gt_path=str(gt_path),
                    preprocessor_name=pre_name,
                    scan_fn=scan_fn,
                    scan_params=scan_params,
                    gt_fn=gt_fn,
                    gt_params=gt_params,
                    tmp_dir=str(self.cfg.data.tmp_dir),
                    dir_prefix=dir_prefix,
                    relevant_slices_margin=relevant_margin,
                )
            )
        else:
            prepared = _prepare_case_data(
                PrepareCaseRequest(
                    scan_path=str(scan_path),
                    gt_path=str(gt_path),
                    preprocessor_name=pre_name,
                    scan_fn=scan_fn,
                    scan_params=scan_params,
                    gt_fn=gt_fn,
                    gt_params=gt_params,
                    tmp_dir=str(self.cfg.data.tmp_dir),
                    dir_prefix=dir_prefix,
                    relevant_slices_margin=relevant_margin,
                )
            )

        return self._run_prepared_case(prepared, return_artifacts)

    def _run_prepared_case(
        self,
        prepared,
        return_artifacts: bool = False,
    ) -> dict:
        cleanup_tmp = getattr(self.cfg.data, "cleanup_tmp_dirs", True)
        rebuild_each_case = getattr(self.cfg.model, "rebuild_each_case", True)

        try:
            results = self.inference_strategy.run_prepared(
                prepared,
                get_predictor=self._get_predictor,
                return_artifacts=return_artifacts,
            )
            results["predictor_init_seconds"] = (
                self._predictor_init_seconds if self._predictor_init_seconds is not None else 0.0
            )
            logger.info(
                f"Mean Dice: {results.get('mean_dice', 0):.4f} | "
                f"Mean IoU: {results.get('mean_iou', 0):.4f} | "
                f"Mean 95th Percentile Hausdorff Distance: {results.get('mean_hausdorff', 0):.4f}"
            )
            return results
        finally:
            if rebuild_each_case:
                self._predictor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if cleanup_tmp:
                if isinstance(prepared, PreparedCaseData):
                    _safe_remove_dir_async(Path(prepared.run_dir))
                elif isinstance(prepared, PreparedThreeAxisData):
                    _safe_remove_dir_async(Path(prepared.run_base_dir))

    def run_dataset(
        self,
        dataset_path: str,
        save_results: bool = False,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Run evaluation across an entire dataset.

        Args:
            dataset_path: Path to the dataset configuration YAML file
            save_results: Whether to save results to a JSON file
            output_path: Path to save results (defaults to dataset_path with .json extension)

        Returns:
            Dictionary containing individual case results and aggregate statistics
        """
        # Load dataset
        dataset = Dataset(dataset_path)
        logger.info(f"Running evaluation on dataset '{dataset.name}' with {len(dataset)} cases")
        
        # Track start time for progress logging
        start_time = time.perf_counter()

        # Run all cases
        individual_results = []
        case_names = dataset.get_case_names()
        bone_names = dataset.get_bone_names()

        pre_name, scan_fn, scan_params, gt_fn, gt_params = _get_preprocessors(self.cfg)
        relevant_margin = self.cfg.processing.relevant_slices_margin
        tmp_dir = Path(self.cfg.data.tmp_dir)
        inference_mode = getattr(getattr(self.cfg, "inference", None), "mode", "baseline")
        dir_prefix = f"{inference_mode}_{self.cfg.propagation.name}_"
        max_tmp_dirs = getattr(self.cfg.data, "max_tmp_dirs", 100)
        try:
            max_tmp_dirs = int(max_tmp_dirs)
        except Exception:
            max_tmp_dirs = 100
        if max_tmp_dirs <= 0:
            logger.warning("max_tmp_dirs must be positive; defaulting to 1")
            max_tmp_dirs = 1

        preprocess_workers = getattr(self.cfg.processing, "num_preprocess_workers", None)
        max_workers = int(preprocess_workers) if preprocess_workers else (os.cpu_count() or 1)
        max_workers = max(1, max_workers)
        inflight_limit = max_tmp_dirs
        dir_guard = BoundedSemaphore(max_tmp_dirs)

        prep_futures: Dict[Any, int] = {}
        total_cases = len(dataset)
        completed_count = 0

        prepare_fn = _prepare_case_data_three_axis if isinstance(self.inference_strategy, ThreeAxisStrategy) else _prepare_case_data
        prepare_request_cls = PrepareThreeAxisRequest if isinstance(self.inference_strategy, ThreeAxisStrategy) else PrepareCaseRequest

        def log_progress():
            """
            Logs progress every 25 cases: elapsed time, estimated remaining, avg IoU/Dice.
            """
            elapsed = time.perf_counter() - start_time
            
            # Calculate averages from successful cases
            successful_so_far = [r for r in individual_results if r.get('error') is None]
            num_successful = len(successful_so_far)
            
            if num_successful > 0:
                dice_vals = [r['mean_dice'] for r in successful_so_far if r.get('mean_dice') is not None]
                iou_vals = [r['mean_iou'] for r in successful_so_far if r.get('mean_iou') is not None]
                
                avg_dice = np.mean(dice_vals) if dice_vals else 0.0
                avg_iou = np.mean(iou_vals) if iou_vals else 0.0
            else:
                avg_dice = 0.0
                avg_iou = 0.0

            # Time estimation
            if completed_count > 0:
                avg_time_per_case = elapsed / completed_count
                remaining_cases = total_cases - completed_count
                est_remaining = avg_time_per_case * remaining_cases
            else:
                est_remaining = 0.0

            # Format times
            def fmt_time(seconds):
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                if h > 0:
                    return f"{int(h)}h {int(m)}m {int(s)}s"
                elif m > 0:
                    return f"{int(m)}m {int(s)}s"
                else:
                    return f"{s:.1f}s"

            logger.info(
                f"\n--- Progress Update ({completed_count}/{total_cases}) ---\n"
                f"Elapsed: {fmt_time(elapsed)} | Est. Remaining: {fmt_time(est_remaining)}\n"
                f"Avg Dice: {avg_dice:.4f} | Avg IoU: {avg_iou:.4f} | Valid Cases: {num_successful}\n"
                f"----------------------------------------"
            )

        def _process_future(done_future):
            nonlocal completed_count
            idx = prep_futures.pop(done_future)
            case_name = case_names[idx]
            bone_name = bone_names[idx]
            try:
                prepared = done_future.result()
                logger.info(f"Running inference for case {idx}: bone = {bone_name}, case_name = {case_name}")
                prep_dir = getattr(prepared, "run_dir", getattr(prepared, "run_base_dir", None))
                if prep_dir:
                    logger.debug(f"Prepared tmp dir for case {case_name}: {prep_dir}")
                case_result = self._run_prepared_case(prepared)
                case_result['case_name'] = case_name
                case_result['bone_name'] = bone_name
                case_result['case_index'] = idx
                partial_flag = case_result.get('partial_coverage', False)
                if partial_flag:
                    logger.warning(f"Partial coverage for case {case_name}. Missing slices: {case_result.get('missing_slices')}")
                individual_results.append(case_result)
                logger.info(f"Case {idx} completed successfully: bone = {bone_name}, case_name = {case_name}")
            except Exception as e:
                logger.error(f"Case {idx} failed: bone = {bone_name}, case_name = {case_name}, error: {e}", exc_info=True)
                error_result = {
                    'case_name': case_name,
                    'case_index': idx,
                    'error': str(e),
                    'mean_dice': None,
                    'mean_iou': None,
                    'mean_hausdorff': None,
                    'prompt_indices': None,
                    'num_frames': None,
                    'partial_coverage': None,
                    'missing_slices': None,
                    'predictor_init_seconds': None,
                    'propagation_seconds': None,
                    'propagation_seconds_per_slice': None,
                }
                individual_results.append(error_result)
            finally:
                dir_guard.release()
                completed_count += 1
                remaining = total_cases - completed_count
                logger.info(f"Progress: {completed_count}/{total_cases} cases completed, {remaining} remaining")
                
                if completed_count % 25 == 0:
                    log_progress()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            logger.info(f"Scheduling {total_cases} cases")
            for i, (scan_path, gt_path) in enumerate(dataset):
                while not dir_guard.acquire(blocking=False):
                    done_future = next(as_completed(list(prep_futures.keys())))
                    _process_future(done_future)
                bone_name = bone_names[i]
                case_name = case_names[i]
                remaining_to_start = total_cases - i - 1
                logger.info(f"Starting case {i+1}/{total_cases}: bone = {bone_name}, case_name = {case_name} ({remaining_to_start} remaining to start)")
                future = executor.submit(
                    prepare_fn,
                    prepare_request_cls(
                        scan_path=str(scan_path),
                        gt_path=str(gt_path),
                        preprocessor_name=pre_name,
                        scan_fn=scan_fn,
                        scan_params=scan_params,
                        gt_fn=gt_fn,
                        gt_params=gt_params,
                        tmp_dir=str(tmp_dir),
                        dir_prefix=dir_prefix,
                        relevant_slices_margin=relevant_margin,
                        case_index=i,
                        bone_name=bone_name,
                    ),
                )
                prep_futures[future] = i

                if len(prep_futures) >= inflight_limit:
                    done_future = next(as_completed(list(prep_futures.keys())))
                    _process_future(done_future)
            logger.info(f"All {total_cases} cases scheduled. Waiting for remaining {len(prep_futures)} cases to complete...")

            for done_future in as_completed(list(prep_futures.keys())):
                _process_future(done_future)

        # Compute aggregate statistics
        successful_results = [r for r in individual_results if r.get('error') is None]

        # Compute per-bone aggregates (dice/ioU/hausdorff) using successful cases
        per_bone_results = {}
        per_bone_buckets = {}
        for r in individual_results:
            bone = r.get('bone_name') or "unknown_bone"
            bucket = per_bone_buckets.setdefault(
                bone,
                {"total": 0, "failed": 0, "successful": []},
            )
            bucket["total"] += 1
            if r.get('error') is None:
                bucket["successful"].append(r)
            else:
                bucket["failed"] += 1

        for bone, bucket in per_bone_buckets.items():
            successes = bucket["successful"]
            dice_scores_b = [res['mean_dice'] for res in successes if res['mean_dice'] is not None]
            iou_scores_b = [res['mean_iou'] for res in successes if res['mean_iou'] is not None]
            hausdorff_scores_b = [res['mean_hausdorff'] for res in successes if res['mean_hausdorff'] is not None]

            per_bone_results[bone] = {
                'num_total_cases': bucket["total"],
                'num_successful_cases': len(successes),
                'num_failed_cases': bucket["failed"],
                'dice_mean': float(np.mean(dice_scores_b)) if dice_scores_b else None,
                'dice_std': float(np.std(dice_scores_b)) if dice_scores_b else None,
                'dice_min': float(np.min(dice_scores_b)) if dice_scores_b else None,
                'dice_max': float(np.max(dice_scores_b)) if dice_scores_b else None,
                'iou_mean': float(np.mean(iou_scores_b)) if iou_scores_b else None,
                'iou_std': float(np.std(iou_scores_b)) if iou_scores_b else None,
                'iou_min': float(np.min(iou_scores_b)) if iou_scores_b else None,
                'iou_max': float(np.max(iou_scores_b)) if iou_scores_b else None,
                'hausdorff_mean': float(np.mean(hausdorff_scores_b)) if hausdorff_scores_b else None,
                'hausdorff_std': float(np.std(hausdorff_scores_b)) if hausdorff_scores_b else None,
                'hausdorff_min': float(np.min(hausdorff_scores_b)) if hausdorff_scores_b else None,
                'hausdorff_max': float(np.max(hausdorff_scores_b)) if hausdorff_scores_b else None,
            }

        if successful_results:
            dice_scores = [r['mean_dice'] for r in successful_results if r['mean_dice'] is not None]
            iou_scores = [r['mean_iou'] for r in successful_results if r['mean_iou'] is not None]
            hausdorff_scores = [r['mean_hausdorff'] for r in successful_results if r['mean_hausdorff'] is not None]

            aggregate_results = {
                'num_total_cases': len(individual_results),
                'num_successful_cases': len(successful_results),
                'num_failed_cases': len(individual_results) - len(successful_results),
                'dice_mean': float(np.mean(dice_scores)) if dice_scores else None,
                'dice_std': float(np.std(dice_scores)) if dice_scores else None,
                'dice_min': float(np.min(dice_scores)) if dice_scores else None,
                'dice_max': float(np.max(dice_scores)) if dice_scores else None,
                'iou_mean': float(np.mean(iou_scores)) if iou_scores else None,
                'iou_std': float(np.std(iou_scores)) if iou_scores else None,
                'iou_min': float(np.min(iou_scores)) if iou_scores else None,
                'iou_max': float(np.max(iou_scores)) if iou_scores else None,
                'hausdorff_mean': float(np.mean(hausdorff_scores)) if hausdorff_scores else None,
                'hausdorff_std': float(np.std(hausdorff_scores)) if hausdorff_scores else None,
                'hausdorff_min': float(np.min(hausdorff_scores)) if hausdorff_scores else None,
                'hausdorff_max': float(np.max(hausdorff_scores)) if hausdorff_scores else None,
            }
        else:
            aggregate_results = {
                'num_total_cases': len(individual_results),
                'num_successful_cases': 0,
                'num_failed_cases': len(individual_results),
                'dice_mean': None,
                'dice_std': None,
                'dice_min': None,
                'dice_max': None,
                'iou_mean': None,
                'iou_std': None,
                'iou_min': None,
                'iou_max': None,
                'hausdorff_mean': None,
                'hausdorff_std': None,
                'hausdorff_min': None,
                'hausdorff_max': None,
            }

        # Log aggregate results with all info in one string
        logger.info(
            "\n" +
            "=" * 50 + "\n" +
            "Dataset Evaluation Complete\n" +
            "=" * 50 + "\n" +
            f"Total cases: {aggregate_results['num_total_cases']} | "
            f"Successful cases: {aggregate_results['num_successful_cases']} | "
            f"Failed cases: {aggregate_results['num_failed_cases']}"
        )

        if aggregate_results['dice_mean'] is not None:
            logger.info(
                "Aggregate Results:\n"
                f"Dice - Mean: {aggregate_results['dice_mean']:.4f}, Std: {aggregate_results['dice_std']:.4f}\n"
                f"IoU - Mean: {aggregate_results['iou_mean']:.4f}, Std: {aggregate_results['iou_std']:.4f}\n"
                f"Hausdorff - Mean: {aggregate_results['hausdorff_mean']:.4f}, Std: {aggregate_results['hausdorff_std']:.4f}"
            )

        if per_bone_results:
            per_bone_messages = ["Per-bone results (successful cases only):"]
            for bone, stats in per_bone_results.items():
                succeeded = stats['num_successful_cases']
                total = stats['num_total_cases']
                if stats['dice_mean'] is not None:
                    per_bone_messages.append(
                        f"{bone}: success {succeeded}/{total} | "
                        f"Dice mean {stats['dice_mean']:.4f} (std {stats['dice_std']:.4f}), "
                        f"IoU mean {stats['iou_mean']:.4f} (std {stats['iou_std']:.4f})"
                    )
                else:
                    per_bone_messages.append(f"{bone}: success {succeeded}/{total} | no successful cases for metrics")
            logger.info("\n".join(per_bone_messages))

        # Save results if requested
        if save_results:
            default_output_dir = Path(__file__).resolve().parent.parent / "outputs"

            # Timestamp for run metadata and default naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Respect explicit output_path (from CLI/config) without adding a timestamp.
            # Keep timestamped filenames only for the implicit default path to avoid overwrites.
            if output_path is None:
                output_base = default_output_dir / Path(dataset_path).with_suffix('.json').name
                output_base.parent.mkdir(parents=True, exist_ok=True)
                suffix = output_base.suffix or ".json"
                stem = output_base.stem
                final_output_path = output_base.with_name(f"{stem}_{timestamp}{suffix}")
            else:
                final_output_path = Path(output_path)
                final_output_path.parent.mkdir(parents=True, exist_ok=True)

            def _cfg_to_container(section):
                if section is None:
                    return None
                try:
                    return OmegaConf.to_container(section, resolve=True)
                except Exception:
                    return OmegaConf.to_container(section, resolve=False)

            run_metadata = {
                'timestamp': timestamp,
                'inference_mode': getattr(self.cfg.inference, "mode", None) if hasattr(self.cfg, "inference") else None,
                'strategy': getattr(self.cfg.strategy, "name", None) if hasattr(self.cfg, "strategy") else None,
                'propagation': getattr(self.cfg.propagation, "name", None) if hasattr(self.cfg, "propagation") else None,
                'model_checkpoint': str(self.cfg.model.checkpoint) if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "checkpoint") else None,
            }

            results_to_save = {
                'dataset_name': dataset.name,
                'dataset_config_path': dataset_path,
                'runner_config': OmegaConf.to_container(self.cfg, resolve=True),
                'strategy_config': _cfg_to_container(getattr(self.cfg, "strategy", None)),
                'propagation_config': _cfg_to_container(getattr(self.cfg, "propagation", None)),
                'preprocessing_config': _cfg_to_container(getattr(self.cfg, "preprocessing", None)),
                'individual_results': individual_results,
                'aggregate_results': aggregate_results,
                'per_bone_results': per_bone_results,
                'run_metadata': run_metadata,
            }

            with open(final_output_path, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)

            logger.info(f"Results saved to: {final_output_path}")

        return {
            'individual_results': individual_results,
            'aggregate_results': aggregate_results,
            'per_bone_results': per_bone_results,
        }

