# SAM2 Modifications

This folder contains the SAM2 repository with modifications made to perform the ablation study of this research project. The model now accepts the following additional arguments:

- **CONFIDENCE GATING**: `mem_confidence_threshold`, `mem_fg_prob_threshold`, `use_global_conf_memory`, `top_k_frames_to_use`
- **MEMORY BANK MODIFICATIONS**: `use_maskmem_tpos_enc`, `maskmem_tpos_ckpt`
- **CONDITIONED (PROMPTED) FRAME SELECTION**: `print_selected_cond_frames`, `filter_cond_enabled`, `filter_cond_iou_threshold`, `filter_cond_centroid_frac`, `prompt_frame_temporal_threshold`
- **LOGGING OF EXECUTION TIMES**: `print_timings`, `timing_log_path`

## Modifications in `sam2_base.py`

The core logic modifications are implemented in `sam2/modeling/sam2_base.py`:

- **`__init__`**: Added new parameters for confidence gating, memory bank modifications, conditioned frame selection, and timing logging.
- **`_prepare_memory_conditioned_features`**:
    - Implemented **Conditioned Frame Filtering**: Filters conditioning frames based on temporal distance (`_filter_cond_outputs_by_temporal_distance`) and mask similarity (IoU and centroid distance via `_filter_cond_outputs_by_mask_similarity`).
    - Implemented **Global Confidence Memory**: If `use_global_conf_memory` is enabled, the memory bank selects the top-k most confident frames (`top_k_frames_to_use`) in addition to the closest frames, rather than just using a fixed stride.
- **`_encode_memory_in_output`**:
    - Computes a `mem_confidence` score for each frame (based on predicted IoU or foreground probability).
    - Implements **Memory Gating**: If `mem_confidence_threshold` is set, frames with low confidence are excluded from the memory bank.
- **`_track_step`** and **`track_step`**: Updated to pass `frame_idx` and `selected_ious` to facilitate confidence calculation and filtering.
- **Timing**: Added `_start_timing` and `_end_timing` to profiling specific blocks of code (Image Encoder, SAM Decoder, Memory Attention, etc.).

## Modifications in `build_sam.py`

Modifications in `sam2/build_sam.py` allow loading custom weights for temporal position encodings:

- **`build_sam2_video_predictor`**: Accepts an optional `maskmem_tpos_ckpt` argument.
- **`_load_checkpoint`**: Extended to load `maskmem_tpos_enc` weights from the separate `maskmem_tpos_ckpt` file if provided.
