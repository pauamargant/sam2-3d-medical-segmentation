"""
------------------------------
SAM2 Utilities
------------------------------
This module contains utility functions for working with the SAM2 video predictor. It includes functions to add mask prompts to the predictor
and to propagate masks through video frames. It assumes that nifti_utils has been used to preprocess the input data into the required format.
To segment a video please follow:
1. Use `add_mask_prompt_to_predictor` to add mask prompts for specific frames to the SAM2VideoPredictor instance. This function takes a binary mask,
   the prompt index (frame index), the predictor instance, and the current inference state, and adds the mask prompt to the predictor. 
2. Use `propagate_video` to propagate the masks through the video frames. This function takes the predictor instance, the inference state,
   the start frame index, and the end frame index, and propagates the masks from the start frame to the end frame, returning a dictionary of frame indices and their corresponding propagated mask logits.
   This can be used to propagate the masks both forwards and backwards in the video.
------------------------------
"""


import numpy as np

def add_mask_prompt_to_predictor(mask: np.ndarray, prompt_idx, predictor, inference_state: dict):
    """
    Add a mask prompt to the SAM2 predictor.
    
    Args:
        mask (np.ndarray): Binary mask for the prompt.
        prompt_idx (int): Frame index for the prompt.
        predictor: The SAM2VideoPredictor instance.
        inference_state (dict): The inference state.
    
    Returns:
        tuple: Frame index, object IDs, and video resolution masks.
    """
    label = 1
    if mask.dtype != bool:
        mask = mask.astype(bool)
    frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(inference_state, prompt_idx, label, mask)
    return frame_idx, obj_ids, video_res_masks

def propagate_video(predictor, inference_state: dict, start_frame: int, end_frame: int) -> dict:
    """
    Propagate the video from start_frame to end_frame using the predictor and inference_state.
    If start_frame > end_frame, it propagates in reverse order. 
    Args:
        predictor (SAM2VideoPredictor): The video predictor.
        inference_state (dict): The inference state.
        start_frame (int): The starting frame index.
        end_frame (int): The ending frame index.
    
    Returns:
        dict: A dictionary with frame indices as keys and propagated mask logits as values.
    """
    video_segments = {}
    num_frames_to_propagate = end_frame - start_frame
    reverse = start_frame > end_frame
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=start_frame, max_frame_num_to_track=num_frames_to_propagate, reverse=reverse):
        video_segments[out_frame_idx] = out_mask_logits
    return video_segments
