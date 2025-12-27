# Evaluation Information
This file summarizes what each folder in the `evaluations` folder contains:
- `axis_mode`: Evaluations for the different axis modes (axial - base, forward-backward, three-axis).
- `baseline`: Evaluations for the baseline model (no image preprocessing).
- `conditional_frame_selection`: Evaluations for the conditional frame selection model.
- `confidence_gating`: Evaluations for the confidence gating model.
- `memory_bank`: Evaluations of different number of frames in memory bank, where the temporal embedding is sliced to adjust it to the number of frames in memory bank.
- `memory_bank_extra`: Evaluations of different number of frames in memory bank, where the temporal embedding is sliced to adjust it to the number of frames in memory bank while trying to preserve its learned structure.
- `model_sizes`: Evaluations of different model sizes (small, base, large).
- `num_conditioned_frames`: Evaluation of changing the number of conditioned frames included in the memory at a given time.
- `num_prompt_frames`: Evaluation of changing the number of prompt frames included in the memory at a given time.
- `processing`: Evaluation of different preprocessing approaches.
- `strides`: Evaluation of different strides for the temporal frame selection.