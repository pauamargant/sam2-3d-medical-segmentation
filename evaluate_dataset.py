"""
--------------------------------
DATASET EVALUATION SCRIPT
--------------------------------
LICENSE: view LICENSE

This script is used to evaluate a given ct segmentation dataset using a given runner configuration.
The dataset configuration file is a YAML file that consists of a list of cases, each with a scan path,
a ground truth path, and a bone name. They can be generated using the dataset_generator.py script.

The config file is a YAML file that described the evaluation procedure, including the strategy, propagation, preprocessing, and model configuration. More details can be found in the README.md and
configs/config.md

The script will run the evaluation and save the results to a JSON file. The results will be saved in the outputs/ directory.
The script will also visualize the example cases if the save_examples flag is set.

The script can be run with the following command:
python evaluate_dataset.py --dataset-config configs/datasets/dataset.yaml --runner-config configs/config.yaml
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
from src.runner import ExperimentRunner
from src.utils.logging_config import setup_logging, get_logger
from src.utils.visualization import visualize_example_cases

# Setup logging
logger = get_logger(__name__)


def main() -> None:
    """
    Main entry point that loads config and executes dataset evaluation.
    """
    parser = argparse.ArgumentParser(description='Run dataset evaluation')
    parser.add_argument('--dataset-config', type=str, required=True,
                       help='Path to dataset configuration YAML file')
    parser.add_argument('--runner-config', type=str, default=None,
                       help='Path to runner configuration YAML file (default: configs/config.yaml)')
    parser.add_argument('--save-results', dest='save_results', action='store_true',
                       help='Save results to JSON file (default: enabled)')
    parser.add_argument('--no-save-results', dest='save_results', action='store_false',
                       help='Disable saving results to JSON')
    parser.set_defaults(save_results=True)
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to save results JSON (default: python/outputs/<dataset>.json)')
    parser.add_argument('--example-plot-stride', type=int, default=None,
                        help='Plot 1 out of every N slices; defaults to config value if unset')
    parser.add_argument('--save-examples', dest='save_examples', action='store_true',
                        help='Force enable saving example visualizations regardless of config')
    parser.add_argument('--no-save-examples', dest='save_examples', action='store_false',
                        help='Disable saving example visualizations regardless of config')
    parser.set_defaults(save_examples=None)

    args = parser.parse_args()

    # Load dataset config for example cases
    dataset_cfg = OmegaConf.load(args.dataset_config)
    dataset_name = dataset_cfg.get("name", Path(args.dataset_config).stem)
    outputs_dir = Path(__file__).parent / "outputs"
    default_output_path = outputs_dir / f"{Path(args.dataset_config).stem}.json"
    cli_output_path = Path(args.output_path) if args.output_path else None
    example_cases = list(dataset_cfg.get("example_cases", [])) if dataset_cfg else []

    # Load main config using OmegaConf (avoiding Hydra to prevent conflicts with SAM2)
    config_dir = Path(__file__).parent / "configs"
    if args.runner_config:
        config_path = Path(args.runner_config)
    else:
        config_path = config_dir / "config.yaml"

    # Load main config
    cfg = OmegaConf.load(config_path)

    # Load and merge configs from inference section (strategy, propagation, preprocessing, etc.)
    if "inference" not in cfg:
        cfg.inference = OmegaConf.create({})
    
    inference_cfg = cfg.inference
    for group in ["strategy", "propagation", "preprocessing"]:
        if group not in inference_cfg:
            logger.warning(f"Missing {group} in inference config, skipping")
            continue
        option = inference_cfg[group]
        group_path = config_dir / group / f"{option}.yaml"
        if not group_path.exists():
            logger.warning(f"Config file not found: {group_path}, skipping {group} config")
            continue
        group_cfg = OmegaConf.load(group_path)
        # If the file is packaged (e.g., contains "group:"), merge that key; else merge whole file.
        if group in group_cfg:
            setattr(cfg, group, group_cfg[group])
        else:
            cfg = OmegaConf.merge(cfg, group_cfg)

    logger.info(
        f"Using propagation type: {getattr(cfg, 'propagation', 'N/A')}, "
        f"strategy type: {getattr(cfg, 'strategy', 'N/A')}, "
        f"preprocessing type: {getattr(cfg, 'preprocessing', 'N/A')}"
    )

    # Resolve output path: CLI overrides config, otherwise default <dataset>.json
    output_cfg = cfg.get("output", {}) or {}
    cfg_results_filename = output_cfg.get("results_filename")
    if cli_output_path:
        output_path = cli_output_path
    elif cfg_results_filename:
        cfg_output = Path(cfg_results_filename)
        output_path = cfg_output if cfg_output.is_absolute() else outputs_dir / cfg_output
    else:
        output_path = default_output_path

    if output_path.suffix == "":
        output_path = output_path.with_suffix(".json")

    logger.info(f"Results will be written to: {output_path}")

    # Instantiate runner
    margin = cfg.processing.relevant_slices_margin
    logger.info(f"Using relevant_slices_margin: {margin}")
    runner = ExperimentRunner(cfg)

    # Execute dataset evaluation
    results = runner.run_dataset(
        dataset_path=args.dataset_config,
        save_results=args.save_results,
        output_path=str(output_path)
    )

    # Visualization options from config with optional CLI override
    vis_cfg = cfg.get("visualization", {}) or {}
    save_examples_cfg = vis_cfg.get("save_examples", True)
    plot_stride_cfg = vis_cfg.get("plot_stride", 5)

    save_examples = save_examples_cfg if args.save_examples is None else args.save_examples
    plot_stride = plot_stride_cfg if args.example_plot_stride is None else args.example_plot_stride

    # Visualize example cases, if enabled
    if save_examples:
        visualize_example_cases(
            example_cases=example_cases,
            runner=runner,
            dataset_name=dataset_name,
            plot_stride=plot_stride,
            output_path=output_path,
        )
    else:
        logger.info("Skipping visualization; save_examples disabled")

    # Print final summary
    agg = results['aggregate_results']
    per_bone = results.get('per_bone_results', {}) or {}
    summary = (
        "=" * 50 + "\n"
        "FINAL RESULTS SUMMARY\n"
        + "=" * 50 + "\n"
        + f"Dataset: {args.dataset_config}\n"
        + f"Total Cases: {agg['num_total_cases']}\n"
        + f"Successful: {agg['num_successful_cases']}\n"
        + f"Failed: {agg['num_failed_cases']}"
    )
    logger.info(summary)

    if agg['dice_mean'] is not None:
        logger.info(
            "Performance Metrics:\n"
            f"  Dice Score: {agg['dice_mean']:.4f} ± {agg['dice_std']:.4f}\n"
            f"  IoU Score:  {agg['iou_mean']:.4f} ± {agg['iou_std']:.4f}\n"
            f"  Hausdorff:  {agg['hausdorff_mean']:.4f} ± {agg['hausdorff_std']:.4f}"
        )
    else:
        logger.warning("No successful cases to compute metrics.")

    if per_bone:
        lines = ["Per-bone metrics (successful cases only):"]
        for bone_name, stats in per_bone.items():
            success = stats.get('num_successful_cases', 0)
            total = stats.get('num_total_cases', 0)
            if stats.get('dice_mean') is not None:
                line = (
                    f"{bone_name}: success {success}/{total} | "
                    f"Dice {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}, "
                    f"IoU {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}"
                )
            else:
                line = f"{bone_name}: success {success}/{total} | no successful cases"
            lines.append(line)
        single_str = "\n".join(lines)
        logger.info(single_str)

if __name__ == "__main__":
    main()
