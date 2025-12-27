"""
Visualization helpers for CT/segmentation overlays.
"""

from pathlib import Path
from datetime import datetime
from typing import Any
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.eval import compute_segmentation_metrics_per_slice
import sys
from pathlib import Path

# Add python directory to path for imports
python_dir = Path(__file__).parent.parent
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable-ish sigmoid for numpy arrays."""
    return 1 / (1 + np.exp(-x))


def plot_overlay_pair(
    ct_slice: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    slice_idx: int,
    out_path: Path,
    dice: float | None,
    iou: float | None,
):
    """Save side-by-side GT and prediction overlays with metrics text."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].imshow(gt_mask, cmap="autumn", alpha=0.4)
    axes[0].set_title(f"GT overlay (slice {slice_idx})")

    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(pred_mask, cmap="winter", alpha=0.4)
    axes[1].set_title("Prediction overlay")

    for ax in axes:
        ax.axis("off")

    metrics_txt = []
    if dice is not None:
        metrics_txt.append(f"Dice: {dice:.3f}")
    if iou is not None:
        metrics_txt.append(f"IoU: {iou:.3f}")
    if metrics_txt:
        fig.text(
            0.5, 0.02, "  |  ".join(metrics_txt), ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_example_cases(
    example_cases,
    runner: Any,
    dataset_name: str,
    plot_stride: int,
    output_path: Path | None = None,
):
    """Run and plot example cases with per-slice overlays."""
    if not example_cases:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir_name = (
        Path(output_path).stem if output_path else f"{dataset_name}_{timestamp}"
    )
    vis_root = (
        Path(__file__).parent.parent.parent / "outputs" / "visualizations" / vis_dir_name
    )
    vis_root.mkdir(parents=True, exist_ok=True)

    for ex_idx, example in enumerate(example_cases):
        scan_path = example.get("scan_path")
        gt_path = example.get("gt_path")
        if not scan_path or not gt_path:
            logger.warning(
                f"Skipping example {ex_idx} because scan_path/gt_path missing"
            )
            continue

        example_name_parts = [f"{ex_idx:02d}", Path(scan_path).stem]
        if "bone_name" in example:
            example_name_parts.append(str(example["bone_name"]))
        example_dir = vis_root / "_".join(example_name_parts)
        example_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running example {ex_idx}: {scan_path}")
        try:
            case_result = runner.run_case(scan_path, gt_path, return_artifacts=True)
        except Exception as e:
            logger.error(f"Example {ex_idx} failed: {e}", exc_info=True)
            continue

        artifacts = case_result.get("artifacts")
        if not artifacts:
            logger.warning(
                f"No artifacts returned for example {ex_idx}; skipping visualization."
            )
            continue

        ct_slices = artifacts["ct_slices"]
        gt_slices = artifacts["gt_slices"]
        gt_volume = artifacts["gt_volume"]
        pred_logits_local = artifacts["predicted_logits"]
        min_slice, max_slice = artifacts["slice_span"]
        num_slices = ct_slices.shape[0]

        # Build global-indexed logits dict and ensure missing slices are zeroed
        predicted_logits_global = {}
        for local_idx in range(num_slices):
            logits = pred_logits_local.get(local_idx)
            if logits is None:
                logits = np.zeros_like(gt_slices[local_idx], dtype=np.float32)
            predicted_logits_global[min_slice + local_idx] = logits

        per_slice_metrics = compute_segmentation_metrics_per_slice(
            predicted_logits_global, gt_volume
        )

        stride = max(1, plot_stride)
        for local_idx in range(num_slices):
            if local_idx % stride != 0:
                continue
            global_idx = min_slice + local_idx
            logits = predicted_logits_global[global_idx]
            probs = sigmoid_np(logits)
            pred_mask = (probs > 0.5).astype(np.uint8)

            dice_val, iou_val, _ = per_slice_metrics.get(
                global_idx, (None, None, None)
            )

            out_path = example_dir / f"slice_{global_idx:04d}.png"
            plot_overlay_pair(
                ct_slice=ct_slices[local_idx],
                gt_mask=gt_slices[local_idx],
                pred_mask=pred_mask,
                slice_idx=global_idx,
                out_path=out_path,
                dice=dice_val,
                iou=iou_val,
            )

    logger.info(f"Visualizations written to: {vis_root}")




def show_mask(mask, ax, obj_id=None, random_color=False):
    """
    Display a segmentation mask on the given axes.
    
    Args:
        mask: Binary mask array.
        ax: Matplotlib axes object.
        obj_id: Object ID for color mapping (optional).
        random_color: If True, use random color; otherwise use colormap.
    """

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """
    Display positive and negative prompt points on the given axes.
    
    Args:
        coords: Array of point coordinates.
        labels: Array of labels (1 for positive, 0 for negative points).
        ax: Matplotlib axes object.
        marker_size: Size of markers (default 200).
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='x', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    """
    Display a bounding box on the given axes.
    
    Args:
        box: Bounding box coordinates [x0, y0, x1, y1].
        ax: Matplotlib axes object.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def visualize_video_segmentation(scan_slices: np.ndarray, video_segments: dict):
    """
    Visualize the video segmentation results. Plots in a grid format.
    Args:
        scan_slices (np.ndarray): 3D array of scan slices.
        video_segments (dict): Dictionary with frame indices as keys and mask logits as values.
    """

    num_cols = min(10, len(video_segments))
    num_rows = (len(video_segments) + num_cols - 1) // num_cols
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    for i, (frame_idx, mask_logits) in enumerate(video_segments.items()):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(scan_slices[frame_idx], cmap='gray')
        mask = (mask_logits[0].cpu().numpy() > 0.5).squeeze(0).astype(np.uint8) * 255
        plt.imshow(mask, alpha=0.5, cmap='Reds')
        plt.title(f"Frame {frame_idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_metrics_per_slice(results_per_slice: dict):
    """
    Visualizes Dice, IoU, and 95th Percentile Hausdorff Distance per slice on a single plot.
    
    Args:
        results_per_slice (dict): Dictionary mapping slice keys to (Dice, IoU, Hausdorff) tuples.
    """

    # 1. Prepare data by separating keys and metric values
    keys = sorted([int(k) for k in results_per_slice.keys()])
    dice_scores = [results_per_slice[k][0] for k in keys]
    iou_scores = [results_per_slice[k][1] for k in keys]
    hausdorff_distances = [results_per_slice[k][2] for k in keys]

    # 2. Create the figure and plot all three lines on a single axis
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Segmentation Metrics Per Slice (Single Scale)', fontsize=14)

    # Plot all three metrics on the primary axis
    ax.plot(keys, dice_scores, marker='o', linestyle='-', color='tab:blue', label='Dice Score')
    ax.plot(keys, iou_scores, marker='s', linestyle='--', color='cornflowerblue', label='IoU Score')
    ax.plot(keys, hausdorff_distances, marker='^', linestyle=':', color='tab:red', label='95% Hausdorff Distance')

    # 3. Add labels, title, and legend
    ax.set_xlabel('Slice Index (Key)')
    ax.set_ylabel('Metric Value (Dice/IoU: 0-1 range, Hausdorff: pixel units)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Ensure x-ticks align with slice keys
    ax.set_xticks(keys) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()