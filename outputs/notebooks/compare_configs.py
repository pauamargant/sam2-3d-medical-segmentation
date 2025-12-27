
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
import torch
import logging
import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.dataset import Dataset
from src.runner import ExperimentRunner
from src.utils.visualization import sigmoid_np
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")

def load_config(config_path, config_dir):
    """Load and merge configuration similar to evaluate_dataset.py"""
    if config_path:
        path = Path(config_path)
    else:
        path = config_dir / "config.yaml"
    
    cfg = OmegaConf.load(path)
    
    # Load and merge configs from inference section
    if "inference" not in cfg:
        cfg.inference = OmegaConf.create({})
    
    inference_cfg = cfg.inference
    for group in ["strategy", "propagation", "preprocessing"]:
        if group not in inference_cfg:
            continue
        option = inference_cfg[group]
        group_path = config_dir / group / f"{option}.yaml"
        if not group_path.exists():
            logger.warning(f"Config file not found: {group_path}, skipping {group} config")
            continue
        group_cfg = OmegaConf.load(group_path)
        if group in group_cfg:
            setattr(cfg, group, group_cfg[group])
        else:
            cfg = OmegaConf.merge(cfg, group_cfg)
            
    return cfg

def plot_comparison(
    ct_slice: np.ndarray,
    gt_mask: np.ndarray,
    pred1_mask: np.ndarray,
    pred2_mask: np.ndarray,
    metrics1: tuple,
    metrics2: tuple,
    slice_idx: int,
    out_path: Path,
    config1_name: str,
    config2_name: str
):
    """Save 3-panel comparison: Pred1, Pred2, GT"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Common settings
    for ax in axes:
        ax.axis("off")
        ax.imshow(ct_slice, cmap="gray")

    def get_masked(mask):
        return np.ma.masked_where(mask == 0, mask)

    # Config 1 - Green
    axes[0].imshow(get_masked(pred1_mask), cmap="Greens", alpha=0.5, vmin=0, vmax=1)
    axes[0].set_title(f"IoU: {metrics1[1]:.3f}", fontsize=30)

    # Config 2 - Green
    axes[1].imshow(get_masked(pred2_mask), cmap="Greens", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"IoU: {metrics2[1]:.3f}", fontsize=30)

    # GT - Red
    axes[2].imshow(get_masked(gt_mask), cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title(f"Ground Truth (Slice {slice_idx})", fontsize=30)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Compare two configs on a dataset')
    parser.add_argument('--dataset-config', type=str, required=True, help='Path to dataset config')
    parser.add_argument('--config1', type=str, required=True, help='Path to first runner config')
    parser.add_argument('--config2', type=str, required=True, help='Path to second runner config')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.4, help='IoU difference threshold')
    
    args = parser.parse_args()
    
    setup_logging()

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config_dir = Path(__file__).parent / "configs"
    
    # Load configs
    logger.info(f"Loading Config 1: {args.config1}")
    cfg1 = load_config(args.config1, config_dir)
    
    logger.info(f"Loading Config 2: {args.config2}")
    cfg2 = load_config(args.config2, config_dir)
    
    # Initialize runners
    runner1 = ExperimentRunner(cfg1)
    runner2 = ExperimentRunner(cfg2)
    
    # Load dataset
    dataset = Dataset(args.dataset_config)
    logger.info(f"Loaded dataset {dataset.name} with {len(dataset)} cases")
    
    output_dir = Path(args.output_dir)
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    config1_name = Path(args.config1).stem
    config2_name = Path(args.config2).stem
    
    case_names = dataset.get_case_names()
    bone_names = dataset.get_bone_names()
    
    stats = []
    
    for i, (scan_path, gt_path) in enumerate(dataset):
        case_name = case_names[i]
        bone_name = bone_names[i]
        logger.info(f"Processing case {i+1}/{len(dataset)}: {case_name} ({bone_name})")
        
        try:
            # Run Config 1
            res1 = runner1.run_case(scan_path, gt_path)
            iou1 = res1.get('mean_iou', 0) or 0
            
            # Run Config 2
            res2 = runner2.run_case(scan_path, gt_path)
            iou2 = res2.get('mean_iou', 0) or 0
            
            diff = abs(iou1 - iou2)
            logger.info(f"IoU 1: {iou1:.4f} | IoU 2: {iou2:.4f} | Diff: {diff:.4f}")
            
            # Collect stats
            stats.append({
                "case_name": case_name,
                "bone_name": bone_name,
                "config1": config1_name,
                "config2": config2_name,
                "iou_config1": float(iou1),
                "iou_config2": float(iou2),
                "diff": float(diff)
            })
            
            if diff > args.threshold and iou1 > 0.6:
                logger.info(f"Difference > {args.threshold}, generating visualizations...")
                
                # Create case directory
                # Use ground truth stem to disambiguate multiple bones (vertebrae) in same scan
                gt_stem = Path(gt_path).stem
                unique_case_name = f"{case_name}_{gt_stem}"
                case_dir = run_dir / bone_name / unique_case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                
                # Re-run with artifacts
                res1_art = runner1.run_case(scan_path, gt_path, return_artifacts=True)
                res2_art = runner2.run_case(scan_path, gt_path, return_artifacts=True)
                
                art1 = res1_art['artifacts']
                art2 = res2_art['artifacts']
                
                ct_slices = art1['ct_slices'] # Should be same for both
                gt_slices = art1['gt_slices'] # Should be same for both
                
                # Get logits and convert to masks
                logits1 = art1['predicted_logits']
                logits2 = art2['predicted_logits']
                
                min_slice = art1['slice_span'][0]
                num_slices = len(ct_slices)
                
                for idx in range(num_slices):
                    # Local index to global slice index
                    slice_idx = min_slice + idx
                    
                    # Config 1 mask
                    l1 = logits1.get(idx)
                    if l1 is None:
                        mask1 = np.zeros_like(gt_slices[idx])
                    else:
                        mask1 = (sigmoid_np(l1) > 0.5).astype(np.uint8)
                        
                    # Config 2 mask
                    l2 = logits2.get(idx)
                    if l2 is None:
                        mask2 = np.zeros_like(gt_slices[idx])
                    else:
                        mask2 = (sigmoid_np(l2) > 0.5).astype(np.uint8)
                    
                    # Metrics for this slice (approximate/reuse overall? No, compute per slice)
                    # We can use simple dice/iou calculation here
                    def compute_iou(p, g):
                        i = np.logical_and(p, g).sum()
                        u = np.logical_or(p, g).sum()
                        return i / u if u > 0 else 1.0 if i == 0 else 0.0
                        
                    iou_s1 = compute_iou(mask1, gt_slices[idx])
                    iou_s2 = compute_iou(mask2, gt_slices[idx])
                    
                    out_path = case_dir / f"slice_{slice_idx:04d}.png"
                    plot_comparison(
                        ct_slice=ct_slices[idx],
                        gt_mask=gt_slices[idx],
                        pred1_mask=mask1,
                        pred2_mask=mask2,
                        metrics1=(0, iou_s1),
                        metrics2=(0, iou_s2),
                        slice_idx=slice_idx,
                        out_path=out_path,
                        config1_name=config1_name,
                        config2_name=config2_name
                    )
                
                logger.info(f"Saved visualizations to {case_dir}")
                
        except Exception as e:
            logger.error(f"Error processing case {case_name}: {e}", exc_info=True)

    # Save statistics
    stats_path = run_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Saved statistics to {stats_path}")

if __name__ == "__main__":
    main()
