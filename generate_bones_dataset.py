"""
--------------------------------
BONES DATASET GENERATOR
--------------------------------
LICENSE: view LICENSE

This script is used to generate a dataset of bones. Assuming the dataset is organized as follows:
data/
    s1/
        segmentations/
            bone_left_123.nii.gz
            bone_right_A123.nii.gz
            bone_left.nii.gz
            bone_right.nii.gz
            bone_123.nii.gz
            bone_A123.nii.gz
            ...
        ...
    ...
The script will generate a dataset of bones by selecting an uniform number of cases for each bone 
until the desired number of cases is reached.

Usage example:
The following command will generate a dataset of 100 cases for each bone until the desired number of cases is reached.
python generate_bones_dataset.py --num_desired_cases 100 --example_cases 10

"""
import yaml
import re
import argparse
from pathlib import Path
import random
import nibabel as nib
import numpy as np
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
BONE_LIST_PATH = Path("configs") / "dataset" / "bone_names.yaml"
def load_bone_filenames()->set[str]:
    """
    Load the bone files from the bone_names.yaml file.

    Returns:
        set[str]: The set of bone files.
    """
    with open(BONE_LIST_PATH, "r") as f:
        data: dict[str, list[str]] = yaml.safe_load(f)
    return set(data.get("bone_files", []))

def get_segmentation_size(nifti_path: Path)->int:
    """
    Get the number of axial slices in the segmentation.

    Args:
        nifti_path (Path): The path to the nifti file.

    Returns:
        int: The number of axial slices.
    """
    nifti_img: nib.Nifti1Image = nib.load(nifti_path)
    data: np.ndarray = nifti_img.get_fdata()
    # check axial slices with > 0
    axial_slices: int = data.shape[0]
    num_axial_slices: int = 0
    for i in range(axial_slices):
        if np.any(data[i]):
            num_axial_slices += 1
    return num_axial_slices

def validate_ground_truth(gt_path: Path, min_slices: int = 4) -> tuple[bool, str]:
    """
    Validate that the ground truth file has positive voxels and enough slices.
    
    Args:
        gt_path (Path): The path to the ground truth nifti file.
        min_slices (int): Minimum number of slices with annotations required (default: 4, since runner requires > 3).
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    try:
        nifti_img: nib.Nifti1Image = nib.load(gt_path)
        data: np.ndarray = nifti_img.get_fdata()
        
        # Check if GT has any positive voxels
        if np.count_nonzero(data) == 0:
            return False, f"Ground truth has no positive voxels"
        
        # Check number of slices with annotations
        num_slices_with_annotations = get_segmentation_size(gt_path)
        if num_slices_with_annotations < min_slices:
            return False, f"Ground truth has only {num_slices_with_annotations} slices with annotations (need at least {min_slices})"
        
        return True, ""
    except Exception as e:
        return False, f"Error loading ground truth: {e}"

def process_bone_scans(bone_name: str, scans: list[tuple[Path, Path]], num_scans_per_bone: int, root: Path) -> list[dict[str, str]]:
    """
    Process scans for a single bone, validating and collecting valid cases.
    
    Args:
        bone_name: Name of the bone
        scans: List of (scan_path, gt_path) tuples
        num_scans_per_bone: Maximum number of scans to add for this bone
        root: Root directory path for making relative paths
        
    Returns:
        List of case dictionaries for this bone
    """
    bone_cases: list[dict[str, str]] = []
    num_scans_added: int = 0
    for scan, gt_path in scans:
        # Validate ground truth: check for positive voxels and enough slices
        is_valid, error_msg = validate_ground_truth(gt_path, min_slices=4)
        if not is_valid:
            print(f"Skipping {gt_path} because: {error_msg}")
            continue
        
        # we make the paths relative to the root directory
        scan_path: str = str(scan.relative_to(root))
        gt_path_str: str = str(gt_path.relative_to(root))
        bone_cases.append({
            "scan_path": str(scan_path),
            "gt_path": gt_path_str,
            "bone_name": bone_name,
        })
        num_scans_added += 1
        if num_scans_added >= num_scans_per_bone:
            break
    return bone_cases

def main(num_desired_cases: int, num_example_cases: int)->None:
    """
    Main function to generate the dataset. 

    Args:
        num_desired_cases (int): The number of desired cases to include in the dataset.
        num_example_cases (int): The number of example cases to include in the dataset.
    """
    bone_files: set[str] = load_bone_filenames()
    cases: list[dict[str, str]] = []

    patients: list[Path] = list(Path(DATA_DIR).glob("s*"))
    patients_list: list[Path] = list(patients)
    random.shuffle(patients_list)
    print(f"Selected {len(patients_list)} patients")


    scans_list: dict[str, list[Path]] = {}
    for patient in patients_list:
        scan_folder: Path = patient / "segmentations"
        scans: list[Path] = list(scan_folder.glob("*.nii.gz"))
        for fname in scans:
            if fname.name not in bone_files:
                print(f"Skipping {fname.name} because it is not in bone_files")
                continue
            fname_str: str = fname.stem  # Remove .nii.gz extension
            bone_name: str = fname_str.replace(".gz", "")
            bone_name = bone_name.replace(".nii", "")
            
            # We use regex to extract the bone name from the filename
            match: re.Match[str] | None = re.match(r"^(.*?)_(?:left|right)_(?:[A-Z]?)(\d+)$", bone_name)
            if match:
                bone_name: str = match.group(1)
            else:
                match: re.Match[str] | None = re.match(r"^(.*?)_(?:left|right)$", bone_name)
                if match:
                    bone_name: str = match.group(1)
                else:
                    match: re.Match[str] | None = re.match(r"^(.*?)_(?:[A-Z]?)(\d+)$", bone_name)
                    if match:
                        bone_name: str = match.group(1)
            if bone_name not in scans_list:
                scans_list[bone_name] = []
            scans_list[bone_name].append((patient / 'ct.nii.gz' , fname))
    
    # we will sample the same number of scans for each bone_name until we have num_desired_cases
    print(f"Number of bone names: {len(scans_list.keys())}, root: {ROOT}, {num_desired_cases}")
    num_scans_per_bone: int = num_desired_cases // len(scans_list.keys())

    # Process all bones in parallel
    with ThreadPoolExecutor() as executor:
        # Submit all bone processing tasks
        future_to_bone = {
            executor.submit(process_bone_scans, bone_name, scans, num_scans_per_bone, ROOT): bone_name
            for bone_name, scans in scans_list.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_bone):
            bone_name = future_to_bone[future]
            try:
                bone_cases = future.result()
                cases.extend(bone_cases)
            except Exception as exc:
                print(f"Bone {bone_name} generated an exception: {exc}")

    # we select num_example_cases random cases from the cases list
    if num_example_cases > 0 and num_example_cases < len(cases):
        example_cases: list[dict[str, str]] = random.sample(cases, num_example_cases)
    elif num_example_cases>=len(cases):
        example_cases: list[dict[str, str]] = cases
    else:
        example_cases: list[dict[str, str]] = []
    dataset: dict[str, Any] = {
        "name": f"dataset_{num_desired_cases}_ex_{num_example_cases}",
        "example_cases": example_cases,
        "cases": cases,
        "bone_names": list(scans_list.keys()),
    }
    output_filename: str = f"dataset_{num_desired_cases}_ex_{num_example_cases}.yaml"
    output_path: Path = Path("configs") / "dataset" / output_filename
    # Print dataset statistics
    print(f"Dataset statistics:")
    print(f"Number of patients: {len(patients_list)}")
    print(f"Number of cases: {len(cases)}")
    bone_names: set[str] = set([case["bone_name"] for case in cases])
    print(f"Number of bone names: {len(set([x['bone_name'] for x in cases]))}")
    # bone counts
    bones: set[str] = set([case["bone_name"] for case in cases])
    print(f"Number of bones: {len(bones)}")
    bones_counts: dict[str, int] = {bone: len([case for case in cases if case["bone_name"] == bone]) for bone in bones}
    print(f"Bones counts: {bones_counts}")
    print(f"Example cases: {dataset['example_cases']}")
    print(f"Number of example cases: {len(dataset['example_cases'])}")

    with open(output_path, "w") as f:
        yaml.safe_dump(dataset, f, sort_keys=False)

    print(f"Wrote {len(cases)} cases to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bones dataset with configurable parameters")
    parser.add_argument(
        "--num_desired_cases",
        type=int,
        default=100,
        help="Number of desired cases to include in the dataset (default: 100)"
    )
    

    parser.add_argument(
        "--example_cases",
        type=int,
        default=10,
        help="Number of bones to randomly select as example cases. These cases will be ploted when running the dataset to visualzie the predictions qualitateively"
    )
    
    args = parser.parse_args()
    main(args.num_desired_cases, args.example_cases)