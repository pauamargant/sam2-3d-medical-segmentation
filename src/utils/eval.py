"""
Evaluation utilities
"""

import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
import sys
from pathlib import Path

# Add python directory to path for imports
python_dir = Path(__file__).parent.parent
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _to_numpy(vol):
    """Convert torch or list-like volume to numpy array without modifying original."""
    if isinstance(vol, torch.Tensor):
        return vol.detach().cpu().numpy()
    return np.asarray(vol)


def ensure_gt_annotations(gt_vol, context: str = ""):
    """Ensure the GT volume contains at least one positive voxel."""
    gt_np = _to_numpy(gt_vol)
    if np.count_nonzero(gt_np) == 0:
        prefix = f"[{context}] " if context else ""
        raise ValueError(f"{prefix}Ground truth volume has no positive voxels.")


def validate_gt_span(gt_vol, min_slice: int, max_slice: int, context: str = ""):
    """
    Verify that the provided slice span covers all annotated slices in the GT volume.
    Raises ValueError if coverage is incomplete.
    """
    gt_np = _to_numpy(gt_vol)
    slice_indices = np.where(np.any(gt_np > 0, axis=(1, 2)))[0]
    if slice_indices.size == 0:
        ensure_gt_annotations(gt_vol, context)  # will raise
    gt_min, gt_max = int(slice_indices.min()), int(slice_indices.max())
    if min_slice > gt_min or max_slice < gt_max:
        prefix = f"[{context}] " if context else ""
        raise ValueError(
            f"{prefix}Selected slice span ({min_slice}-{max_slice}) does not cover GT range ({gt_min}-{gt_max})."
        )
    return gt_min, gt_max


def build_full_volume_from_predictions(
    predicted_masks: dict,
    num_frames: int,
    min_slice: int,
    target_shape,
    log_prefix: str = "",
):
    """
    Build a full-volume logits array aligned to target_shape, zero-filling any missing slices.
    predicted_masks keys are expected to be local indices in [0, num_frames-1].
    Returns (full_volume, missing_slice_indices).
    """
    target_shape = tuple(target_shape)
    assert len(target_shape) == 3, f"target_shape must be (Z,H,W), got {target_shape}"
    full_vol = np.zeros(target_shape, dtype=np.float32)
    missing = []

    for local_idx in range(num_frames):
        global_idx = min_slice + local_idx
        if global_idx >= target_shape[0]:
            raise ValueError(
                f"{log_prefix}Global slice index {global_idx} exceeds target depth {target_shape[0]}"
            )

        if local_idx in predicted_masks:
            logits = predicted_masks[local_idx]
            logits_np = _to_numpy(logits).squeeze()
            full_vol[global_idx] = logits_np
        else:
            missing.append(local_idx)
            full_vol[global_idx] = 0.0

    if missing:
        logger.warning(
            f"{log_prefix}Missing predictions for slices {missing}; zero-filled in full volume."
        )

    return full_vol, missing


def log_section(message: str):
    """Lightweight logging helper for consistent console markers."""
    logger.info(f"[Eval] {message}")


def compute_segmentation_metrics(predicted_logits, ground_truth):
    """
    Compute mean Dice, mean IoU, and 95% Hausdorff distance for a full 3D volume.
    
    predicted_logits: dict[int, tensor]  OR  np.ndarray/tensor of shape (Z,H,W)
    ground_truth: np.ndarray or tensor of shape (Z,H,W)
    """

    # ---- Convert predictions into a (Z,H,W) tensor ----
    if isinstance(predicted_logits, dict):
        keys = list(predicted_logits.keys())
        keys.sort()
        pred_list = []
        for k in keys:
            logits = predicted_logits[k]
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits)
            logits = logits.squeeze()  # remove channel dims
            pred_list.append(logits)
        pred_vol = torch.stack(pred_list, dim=0)   # [Z,H,W]
    else:
        # Already full volume
        pred_vol = torch.as_tensor(predicted_logits).squeeze()  # [Z,H,W]

    # ---- Ground truth ----
    gt_vol = torch.as_tensor(ground_truth).squeeze()  # [Z,H,W]

    assert pred_vol.shape == gt_vol.shape, \
        f"Shape mismatch: pred={pred_vol.shape}, gt={gt_vol.shape}"

    # ---- Convert to probabilities ----
    pred_probs = torch.sigmoid(pred_vol.float())   # [Z,H,W]

    # ---- Convert to binarized masks ----
    pred_bin = (pred_probs > 0.5).float()          # float32 needed for Hausdorff
    gt_bin   = (gt_vol > 0).float()

    # ---- Add batch + channel dims for MONAI metrics ----
    # → shape [B=Z, C=1, H, W]
    pred_bin_4d = pred_bin.unsqueeze(1)
    gt_bin_4d   = gt_bin.unsqueeze(1)

    # ---- Dice ----
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric(pred_bin_4d, gt_bin_4d)
    mean_dice = dice_metric.aggregate().item()

    # ---- IoU ----
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    iou_metric(pred_bin_4d, gt_bin_4d)
    mean_iou = iou_metric.aggregate().item()

    # ---- Hausdorff ----
    haus = HausdorffDistanceMetric(percentile=95, include_background=True, reduction="mean")
    haus(pred_bin_4d, gt_bin_4d)
    mean_hd95 = haus.aggregate().item()

    return mean_dice, mean_iou, mean_hd95



def compute_segmentation_metrics_per_slice(predicted_logits, ground_truth):
    """
    predicted_logits: dict[slice_idx → logits(H,W)]
    ground_truth: np.ndarray or tensor of shape (Z,H,W)
    """
    if not isinstance(ground_truth, torch.Tensor):
        ground_truth = torch.as_tensor(ground_truth)

    results = {}

    for k, logits in predicted_logits.items():

        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)

        logits = logits.squeeze()
        probs  = torch.sigmoid(logits)

        pred_bin = (probs > 0.5).float()
        gt_slice = (ground_truth[k] > 0).float()

        # Add batch/channel → [1,1,H,W]
        pred_4d = pred_bin.unsqueeze(0).unsqueeze(0)
        gt_4d   = gt_slice.unsqueeze(0).unsqueeze(0)

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        iou_metric  = MeanIoU(include_background=True, reduction="mean")
        haus_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)

        dice_metric(pred_4d, gt_4d)
        iou_metric(pred_4d, gt_4d)
        haus_metric(pred_4d, gt_4d)

        results[k] = (
            dice_metric.aggregate().item(),
            iou_metric.aggregate().item(),
            haus_metric.aggregate().item(),
        )

    return results



def compute_segmentation_metrics_multiple(logit_volumes, ground_truth):
    """
    Compute mean Dice, mean IoU, and 95% Hausdorff distance from multiple
    raw logit volumes by:
      1) converting each to probabilities via sigmoid,
      2) fusing them at the probability level (mean),
      3) binarizing the fused probabilities and computing MONAI metrics.

    Args:
        logit_volumes:
            - list/tuple of arrays/tensors of shape (Z,H,W), or
            - single array/tensor of shape (N,Z,H,W) where N = #volumes.
        ground_truth:
            - array/tensor of shape (Z,H,W)
    """

    # ---- Ground truth ----
    gt_vol = torch.as_tensor(ground_truth).squeeze().float()  # [Z,H,W]

    # ---- Stack logit volumes into [N, Z, H, W] ----
    if isinstance(logit_volumes, (list, tuple)):
        vols = [torch.as_tensor(v).float().squeeze() for v in logit_volumes]
        logit_stack = torch.stack(vols, dim=0)  # [N,Z,H,W]
    else:
        logit_stack = torch.as_tensor(logit_volumes).float()
        if logit_stack.dim() == 3:
            # Only one volume given; treat as [1,Z,H,W]
            logit_stack = logit_stack.unsqueeze(0)
        elif logit_stack.dim() != 4:
            raise ValueError(
                f"logit_volumes must be [N,Z,H,W] or list of [Z,H,W], got shape {logit_stack.shape}"
            )

    assert logit_stack.shape[1:] == gt_vol.shape, \
        f"Shape mismatch: logits={logit_stack.shape[1:]}, gt={gt_vol.shape}"

    ## MEAN

    # ---- Convert each volume to probabilities and fuse ----
    # # [N,Z,H,W] -> [N,Z,H,W] probs
    # prob_stack = torch.sigmoid(logit_stack)
    # # Fuse across N (axes / models / runs)
    # fused_probs = prob_stack.mean(dim=0)  # [Z,H,W]
    # # ---- Binarize fused probabilities ----
    # pred_bin = (fused_probs > 0.5).float()

    ## MAJORITY VOTE

    vote_prob_thresh = 0.5

    # majority vote over binarized maps
    prob_stack = torch.sigmoid(logit_stack)          # [N,Z,H,W]
    votes = (prob_stack > 0.5).float()  # [N,Z,H,W]

    # simple majority: >= ceil(N/2)
    required_votes = int(3 // 2 + 1)
    vote_sum = votes.sum(dim=0)                      # [Z,H,W]
    pred_bin = (vote_sum >= required_votes).float()

    gt_bin   = (gt_vol > 0).float()

    # ---- Add batch + channel dims for MONAI metrics ----
    # We treat each Z slice as a batch element: [B=Z, C=1, H, W]
    pred_bin_4d = pred_bin.unsqueeze(1)
    gt_bin_4d   = gt_bin.unsqueeze(1)

    # ---- Dice ----
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric(pred_bin_4d, gt_bin_4d)
    mean_dice = dice_metric.aggregate().item()

    # ---- IoU ----
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    iou_metric(pred_bin_4d, gt_bin_4d)
    mean_iou = iou_metric.aggregate().item()

    # ---- Hausdorff ----
    haus = HausdorffDistanceMetric(percentile=95, include_background=True, reduction="mean")
    haus(pred_bin_4d, gt_bin_4d)
    mean_hd95 = haus.aggregate().item()

    return mean_dice, mean_iou, mean_hd95
