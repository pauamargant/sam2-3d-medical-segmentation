# Automatic Segmentation of 3D medical images with SAM2 using zero-shot approach

**Authors**: Miquel López Escoriza, Pau Amargant Alvarez, Germán Ruiz Cabello  
**Affiliation**: Department of Computer Science, EPFL, Switzerland

## Abstract
Foundation models for image segmentation have
shown strong generalization in natural images, yet their appli-
cability to 3D medical imaging remains limited. In this work,
we study the zero-shot use of Segment Anything Model 2
(SAM2) for automatic segmentation of volumetric CT data,
without any fine-tuning or domain-specific training. We analyze
how SAM2 should be applied to CT volumes and identify its
main limitation: the lack of inherent volumetric awareness. To
address this, we propose a set of inference-alone architectural
and procedural modifications that adapt SAM2’s video-based
memory mechanism to 3D data by treating CT slices as ordered
sequences. We conduct a systematic ablation study on a subset
of 500 CT scans from the TotalSegmentator dataset to evaluate
prompt strategies, memory propagation schemes and multi-pass
refinement. Based on these findings, we select the best-performing
configuration and report final results on a bigger sample
of the TotalSegmentator dataset comprising 5,000 CT scans.
Our results show that, even with frozen weights, SAM2 can
produce coherent 3D segmentations when its inference pipeline
is carefully structured, demonstrating the feasibility of a fully
zero-shot approach for volumetric medical image segmentation.

## Overview

This repository contains the code for the paper. It implements a modular architecture used to perform an ablation study on SAM2, including
architectural modifications to the SAM2 model. The repository focuses on implementing a consistent inference pipeline for SAM2, that can be used to perform a systematic, deterministic ablation study on SAM2. Using a config-based approach to define each step of the inference pipeline and the addition of extra hyperparameters to the SAM2 model, we can easily compare different configurations and identify the best-performing one. The evaluation pipeline reports the results of each evaluation run in a structured manner and includes relevant metrics such as IoU and Dice at a per-bone and aggregated level.

Key features:
- **Zero-shot 3D segmentation** using SAM2.
- **Evaluation framework** for SAM2 using a config-based approach, making it possible to perform a systematic, deterministic ablation study on SAM2.
- **Multi-view fusion** support.
- **Additional modifications** to the SAM2 model to adapt it to 3D medical data, including customization of the memory bank frame selection and temporal position encoding mechanisms.

## Technical Approach
This repository adapts the basic SAM2 inference pipeline, which is suitable for 2D video data, to 3D medical data by treating CT slices as ordered sequences of images along the axial axis. We implement a preprocessing and data preparation pipeline that adapts the input data to the SAM2 expected format. We utilise a modular inference pipeline that allows for easy swapping of how the data is prepared, how the propagation is started and how it is propagated across the volume.
### 1. Volumes as Sequences
The foundation of our approach is to treat 3D CT volumes, which come as `.nii.gz` files, as ordered sequences of 2D images. We first extract the axial (or coronal or sagittal) slices as individual frames. This is done by slicing the volume along the axial axis and saving the slices as individual image files. The individual slices are then preprocessed according to the `preprocessing` pipeline defined in the configuration file. 
The following preprocessing pipelines are available:
- **no_clahe_no_windowing**: No preprocessing is applied.
- **clahe_windowing**: Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to the image and windowing is applied to the image.
- **no_clahe_windowing**: No CLAHE is applied but windowing is applied to the image.
- **clahe_no_windowing**: CLAHE is applied to the image but no windowing is applied.
### Prompt Strategies
In order to evaluate the performance of a configuration, and following what a radiologist would do, the segmentation starts by annotating a set of slices that contain the target object. These annotations consist of the ground truth prompts. The `prompt_strategy` configures which slices are selected to start the segmentation process. The following prompt strategies are available:
- **first_last**: The first and last slices containing the target object are selected.
- **offset**: Slices are selected at fixed intervals or offsets from the anatomical boundaries.
- **middle**: The central slice of the anatomy is selected.
It should be noted that a margin of 5 slices is added when selecting the top and bottom slices to ensure that the initial prompt slices are not too close to the edges of the volume.
### Propagation Mode
The `mode` configures the mode of the inference pipeline. The following modes are available:
- **single**: The segmentation is propagated along a single axis.
- **three_axis**: The segmentation is propagated along all three axes and the results are fused together.
### Propagation Strategies
Once the prompts are selected, the segmentation is propagated across the volume. The `propagation_strategy` configures how the segmentation is propagated across the volume. The following propagation strategies are available:
- **forward**: The segmentation is propagated from the prompt slices to the end of the volume.
- **backward**: The segmentation is propagated from the prompt slices to the beginning of the volume.
- **two_passes**: The segmentation is propagated from the prompt slices to the end of the volume and then from the end of the volume to the prompt slices.

### Modifications to SAM2
While we preserve most of SAM2's architecture we introduce several modifications, all of which can be configured in the configuration file, to how the memory bank functions. All changes are implemented in our local copy under `src/base_sam2/`. 

On the one side, we modify how frames which do not have a prompt are selected for the memory bank. These changes are located in the `src/base_sam2/sam2/modeling/sam2_base.py` file.
- **Confidence-Gated memory:** the memory bank is composed of the $r$ nearest frames and the $k$ top-confidence frames.
- **Number of frames in the memory bank:** the number of frames in the memory bank is configurable.
- **Custom Temporal Position Embeddings:** the temporal position embedding can be loaded from a custom checkpoint.

On the other side, we modify how frames which have a prompt are selected for the memory bank. These changes are located in the `src/base_sam2/sam2/modeling/memory_attention.py` file.
- **Conditioned frames filtering:** we implement various filtering mechanisms to select which conditioned (frames that have a mask prompt input) to include in the memory bank at a given propagation step:
  - **iou filtering:** we filter out conditioned frames that have a low IoU with the ground truth mask.
  - **centroid filtering:** we filter out conditioned frames that have a low centroid distance with the ground truth mask.
  - **temporal filtering:** we filter out conditioned frames that are too far in the past or future.

## Framework Architecture
The implemented evaluation framework is organized around a modular `runner.py` that utilises Hydra / Omega based configurations. 

The `ExperimentRunner` serves as the central orchestrator. It performs:
1. Dynamic module loading: It instantiates the correct `PromptStrategy`, `PropagationStrategy`, and `InferenceStrategy` classes at runtime based on the provided YAML configuration.
2. Parallel data preparation: It uses a `ProcessPoolExecutor` to parallelize the CPU-intensive tasks of NIfTI loading, preprocessing, and slicing, ensuring the GPU is never starved of data.
3. Evaluation: it performs the actual evaluation by running the SAM2 model on the prepared data and collecting the results. Parameters are defined via YAML configs and handled using a combination of Hydra and OmegaConf.

## Installation


1.  **Download and extract the zip file**
2.  **Environment Setup:**
    Create a virtual environment (conda or uv recommended) and install dependencies. We utilised python 3.12.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install SAM2:**
    This project includes a native SAM2 implementation in `src/base_sam2`. This implementation contains additions in top
    of the original [SAM2 repository by Meta](https://github.com/facebookresearch/segment-anything-2).
    ```bash
    cd src/base_sam2
    pip install -e .
    cd ../..
    ```

4.  **Download Checkpoints:**
    Download the pre-trained SAM2 weights.
    ```bash
    cd src/base_sam2/sam2/checkpoints
    ./download_ckpts.sh
    cd ../../../..
    ```

5.  **Data Setup:**
    Download the [TotalSegmentator dataset](https://zenodo.org/records/10047292) and place it in the `data/` directory (or configure paths in your dataset config). We recommend downloading the dataset from the Dropbox link provided in the dataset page for faster download. Move the compressed dataset to the `data/` directory and extract it. After extracting the dataset, it should have the following structure:
    ```
    data/
    ├── s0001/
    │   ├── ct.nii.gz                        # Whole-body CT scan for patient s0001
    │   └── segmentations/
    │       ├── rib_left_1.nii.gz
    │       ├── rib_right_1.nii.gz
    │       ├── hip_left.nii.gz
    │       ├── hip_right.nii.gz
    │       ├── sacrum.nii.gz
    │       └── ...                          # Other bone/organ segmentations
    ├── s0002/
    │   ├── ct.nii.gz
    │   └── segmentations/
    │       ├── rib_left_1.nii.gz
    │       ├── vertebrae_C1.nii.gz
    │       └── ...
    ├── ...
    ```
## Reproducing the Results
To reproduce the results you can run the `run_full_evaluations.sh` script. This script will run all the relevant evaluations in the `configs/evaluations/` directory using the 500 sample dataset and save the results in the `outputs/evaluations/` directory. This process may take tens of hours to complete, approximately one hour per evaluation. Note that outputs of each individual evaluation are saved in the `logs/` directory.
```bash
./run_full_evaluations.sh
```

If you want to run a single evaluation to test the framework, you can run the `run_test_evaluation.sh` script. This script will run the evaluation of the baseline (without preprocessing) configuration and save the results in the `outputs/evaluations/baseline/` directory. It will use a smaller 50 sample dataset. This will print the evaluation outputs to the console instead of saving them to the `logs/` directory.
```bash
./run_test_evaluation.sh
```

## Usage 
### Generating Dataset

Before running evaluations, generate a dataset configuration that specifies which cases to evaluate. The `generate_bones_dataset.py` script samples cases uniformly across bone types:

```bash
python generate_bones_dataset.py --num_desired_cases 100 --example_cases 10
```

This creates a YAML file listing scan paths, ground-truth paths, and bone names. The sampling ensures balanced representation across different anatomical structures—variations like `rib_left_1` through `rib_left_12` are treated as the same bone type during sampling.

### Running Evaluations

Execute an evaluation with the `evaluate_dataset.py` script:

```bash
python evaluate_dataset.py \
    --dataset-config configs/dataset/dataset_test.yaml \
    --runner-config configs/evaluations/test.yaml
```

The runner processes each case sequentially, reporting per-case metrics and aggregated statistics at the end. Results are saved to JSON for later analysis.

Additional options:
- `--save-examples` generates visualization plots showing predictions alongside ground truth
- `--output-path` specifies a custom location for results
- Omit `--save-results` to skip writing metrics to disk (useful for debugging)

### Example Runner Configuration

A runner configuration defines the complete inference pipeline:

```yaml
output:
  results_filename: "evaluations/ablation_two_pass.json"

inference:
  mode: "single"                    # or "three_axis" for multi-view fusion
  strategy: first_last              # prompt selection strategy
  propagation: two_passes           # propagation method
  preprocessing: no_clahe_windowing # preprocessing pipeline

model:
  checkpoint: "./src/base_sam2/sam2/checkpoints/sam2.1_hiera_small.pt"
  config: "configs/sam2.1/sam2.1_hiera_s.yaml"
  custom_overrides:
    model:
      max_cond_frames_in_attn: 1    # limit conditioning frames in attention
      mem_confidence_threshold: 0.5 # filter low-confidence memory

data:
  tmp_dir: "./tmp"
  cleanup_tmp_dirs: true
  rebuild_each_case: true

processing:
  relevant_slices_margin: 0.0       # fraction of margin around anatomy

visualization:
  save_examples: false
  plot_stride: 1
```

The `custom_overrides` section allows modifying SAM2's internal parameters without changing the model checkpoint. This is how we expose our memory bank modifications to the configuration system.

## Extending the Framework
New preprocessing pipelines, strategies, propagation methods and modes can be implemented easily by extending the existing codebase. For each of these four parts, there exists a base class that can be inherited from to implement the new functionality and configured in a config.

As an example, to add a new prompt strategy, create a new file in `src/strategy/`, for example `src/strategy/random.py`:

```python
from typing import List, Tuple
import numpy as np
from src.interfaces import PromptStrategy

class RandomPromptStrategy(PromptStrategy):
    def __init__(self, num_prompts: int = 3, seed: int = 42):
        self.num_prompts = num_prompts
        self.rng = np.random.default_rng(seed)
    
    def get_prompts(self, gt_slices: np.ndarray, relevant_indices: Tuple[int, int]) -> List[int]:
        num_slices = gt_slices.shape[0]
        nonempty = [i for i in range(num_slices) if np.any(gt_slices[i] > 0)]
        
        if len(nonempty) <= self.num_prompts:
            return nonempty
        
        return sorted(self.rng.choice(nonempty, self.num_prompts, replace=False).tolist())
```

Then create `configs/strategy/random.yaml`:

```yaml
name: random
num_prompts: 3
seed: 42
```
When you include this new strategy in the evaluation configuration, the runner will automatically discover and instantiate this strategy when `strategy: random` appears in a runner config.

New preprocessing, propagation strategies and modes can be implemented in a similar manner.



## Project Structure

```
.
├── configs/
│   ├── dataset/              # Dataset specifications
│   ├── evaluations/          # Runner configurations for experiments
│   ├── strategy/             # Prompt strategy parameters
│   ├── propagation/          # Propagation method parameters
│   └── preprocessing/        # Preprocessing pipeline parameters
├── data/                     # TotalSegmentator dataset (not tracked in git)
├── logs/                     # Experiment logs
├── notebooks/
│   ├── results_visualization.ipynb   # Analysis and plotting
│   └── figures/              # Generated figures for paper
├── outputs/
│   └── evaluations/          # JSON results from experiments
├── src/
│   ├── base_sam2/            # Modified SAM2 implementation
│   │   ├── sam2/
│   │   │   ├── modeling/
│   │   │   │   └── sam2_base.py    # Core modifications
│   │   │   └── build_sam.py        # Model construction
│   │   └── setup.py
│   ├── dataset/              # Dataset loading utilities
│   ├── preprocessing/        # Image preprocessing modules
│   ├── propagation/          # Propagation strategy implementations
│   ├── strategy/             # Prompt strategy implementations
│   ├── utils/                # Shared utilities
│   ├── interfaces.py         # Abstract base classes
│   └── runner.py             # Main experiment orchestration
├── evaluate_dataset.py       # Entry point for evaluations
└── generate_bones_dataset.py # Dataset configuration generator
```
## Hardware Requirements

Processing 3D CT volumes requires substantial computational resources, the model initially calculates the image embedding for all frames in the volume and stores them. This requires considerable video memory.

*   **GPU**: NVIDIA GPU with >= 10GB VRAM .
*   **RAM**: 32GB .


## Acknowledgements

This project builds upon the following works:
- [SAM2 (Segment Anything Model 2)](https://github.com/facebookresearch/segment-anything-2) by Meta AI
- [TotalSegmentator dataset](https://zenodo.org/records/10047292) by Wasserthal et al.

We thank EPFL for providing the computational resources used in this project.
