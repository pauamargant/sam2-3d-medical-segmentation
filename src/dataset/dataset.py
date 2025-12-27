"""
Dataset class for loading and iterating over medical imaging cases.
"""

from pathlib import Path
from typing import List, Tuple, Iterator
from omegaconf import OmegaConf, ListConfig, DictConfig
import warnings


class Dataset:
    """Dataset class that loads case lists from YAML configuration files."""

    def __init__(self, dataset_config_path: str):
        """
        Initialize dataset from a YAML configuration file.

        Args:
            dataset_config_path: Path to the YAML config file containing case list
        """
        config_path = Path(dataset_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {config_path}")

        # Load YAML config
        config = OmegaConf.load(str(config_path))

        # Extract cases list
        if 'cases' not in config or not isinstance(config.cases, (list, ListConfig)):
            raise ValueError("Dataset config must contain a 'cases' list")

        self.name = getattr(config, 'name', 'unnamed_dataset')
        self.cases: List[Tuple[str, str]] = []
        self.bone_names: List[str] = []

        # Process each case
        for i, case in enumerate(config.cases):
            if not isinstance(case, (dict, DictConfig)):
                raise ValueError(f"Case {i} must be a dictionary")

            scan_path = case.get('scan_path')
            gt_path = case.get('gt_path')

            if not scan_path or not gt_path:
                raise ValueError(f"Case {i} must have both 'scan_path' and 'gt_path'")

            # Validate paths exist (warnings only)
            scan_path_obj = Path(scan_path)
            gt_path_obj = Path(gt_path)

            if not scan_path_obj.exists():
                warnings.warn(f"Scan path does not exist: {scan_path}")
            if not gt_path_obj.exists():
                warnings.warn(f"Ground truth path does not exist: {gt_path}")

            self.cases.append((str(scan_path), str(gt_path)))
            self.bone_names.append(case.get('bone_name'))
            

    def __len__(self) -> int:
        """Return the number of cases in the dataset."""
        return len(self.cases)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Iterate over cases, returning (scan_path, gt_path) tuples."""
        return iter(self.cases)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get a specific case by index."""
        return self.cases[idx]

    def get_case_names(self) -> List[str]:
        """Get list of case names (derived from scan paths)."""
        return [Path(scan_path).stem for scan_path, _ in self.cases]

    def get_bone_names(self) -> List[str]:
        """Get list of bone names."""
        return self.bone_names
