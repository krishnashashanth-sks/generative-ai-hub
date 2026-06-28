import torch
from layers import CustomCNNFeatureExtractor
import cv2
import numpy as np
from main import FEATURES_DIR
import os

def extract_motion_features(frame1:np.ndarray,frame2:np.ndarray,target_size=(16,16))->np.ndarray:
  gray1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  gray2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
  flow=cv2.calcOpticalFlowFarneback(
      gray1,gray2, None, 0.5,3,15,3,5,1.2,0
  )
  magnitude,angle=cv2.cartToPolar(flow[...,0],flow[...,1],angleInDegrees=False)
  resized_magnitude=cv2.resize(magnitude,target_size,interpolation=cv2.INTER_AREA)
  resized_angle=cv2.resize(angle,target_size,interpolation=cv2.INTER_AREA)
  flattened_magnitude=resized_magnitude.flatten()
  flattened_angle=resized_angle.flatten()
  return np.concatenate((flattened_magnitude,flattened_angle))

def fuse_features(visual_features: torch.Tensor, motion_features: np.ndarray) -> torch.Tensor:
    """
    Fuses visual and motion features using simple concatenation.

    Args:
        visual_features (torch.Tensor): A tensor of visual features from a frame.
        motion_features (np.ndarray): A NumPy array of motion features between frames.

    Returns:
        torch.Tensor: A concatenated tensor representing the fused features.
    """
    # Convert motion features to torch.Tensor
    motion_features_tensor = torch.from_numpy(motion_features).float().unsqueeze(0) # Add batch dimension

    # Ensure visual_features also has a batch dimension if it's a single feature vector
    if visual_features.dim() == 1:
        visual_features = visual_features.unsqueeze(0)

    # Concatenate along the feature dimension (dim=1 for batched tensors)
    fused_features = torch.cat((visual_features, motion_features_tensor), dim=1)
    return fused_features

def prepare_ground_truth_summary(annotation_data: dict, total_frames: int) -> np.ndarray:
    """
    Processes ground truth annotation data to create a frame-level importance array.

    Args:
        annotation_data (dict): A dictionary representing the parsed annotation data.
                                  Expected to contain a 'segments' key with a list of dictionaries,
                                  each having 'start_frame' and 'end_frame'.
        total_frames (int): The total number of frames in the corresponding video.

    Returns:
        np.ndarray: A 1D NumPy array where each element represents the importance of a frame.
                    (1 for summary frame, 0 for non-summary frame).
    """
    # Initialize an array with zeros, representing non-summary frames
    frame_importance = np.zeros(total_frames, dtype=np.float32)

    # Check if 'segments' key exists and is a list
    if 'segments' in annotation_data and isinstance(annotation_data['segments'], list):
        for segment in annotation_data['segments']:
            start_frame = segment.get('start_frame')
            end_frame = segment.get('end_frame')

            if start_frame is not None and end_frame is not None:
                # Mark frames within the segment as important (1)
                # Ensure frames are within video bounds
                start_idx = max(0, int(start_frame))
                end_idx = min(total_frames, int(end_frame) + 1) # +1 because end_frame is inclusive
                frame_importance[start_idx:end_idx] = 1.0
    else:
        # If direct frame_scores are provided (e.g., TVSum's 'gt_score')
        if 'gt_score' in annotation_data and isinstance(annotation_data['gt_score'], list):
            if len(annotation_data['gt_score']) == total_frames:
                # Assume scores are already normalized or represent importance directly
                frame_importance = np.array(annotation_data['gt_score'], dtype=np.float32)
            else:
                print(f"Warning: 'gt_score' length ({len(annotation_data['gt_score'])}) does not match total_frames ({total_frames}). Initializing with zeros.")
        else:
            print("Warning: No valid 'segments' or 'gt_score' found in annotation data. Initializing with zeros.")

    return frame_importance

def extract_and_fuse_features_from_video(
    video_path: str,
    feature_extractor_model: CustomCNNFeatureExtractor,
    output_filename: str, # New argument for saving
    motion_target_size: tuple = (16, 16)
) -> torch.Tensor:
    """
    Extracts visual and motion features from a video, fuses them, and saves the fused features.

    Args:
        video_path (str): Path to the video file.
        feature_extractor_model (CustomCNNFeatureExtractor): An instantiated CNN model for visual feature extraction.
        output_filename (str): The filename (without path) to save the fused features. Will be saved in FEATURES_DIR.
        motion_target_size (tuple): The target size (width, height) for resizing motion features.

    Returns:
        torch.Tensor: A concatenated tensor of all fused features for the video, or an empty tensor if processing fails.
    """
    fused_features_list = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return torch.empty(0)

    print(f"Extracting and fusing features from {video_path}...")

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame from {video_path}")
        cap.release()
        return torch.empty(0)

    with torch.no_grad(): # Disable gradient calculation for feature extraction
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            # --- 1. Prepare current_frame for CNN ---
            processed_frame = cv2.resize(current_frame, (200, 200))
            processed_frame = torch.from_numpy(processed_frame).float() / 255.0 # Normalize to [0, 1]
            processed_frame = processed_frame.permute(2, 0, 1).unsqueeze(0) # (H, W, C) -> (C, H, W) -> (1, C, H, W)

            # --- 2. Extract visual features ---
            visual_features = feature_extractor_model(processed_frame)

            # --- 3. Extract motion features ---
            motion_features = extract_motion_features(prev_frame, current_frame, target_size=motion_target_size)

            # --- 4. Fuse features ---
            fused_features = fuse_features(visual_features, motion_features)
            fused_features_list.append(fused_features)

            # Update prev_frame for the next iteration
            prev_frame = current_frame

    cap.release()
    print(f"Finished extracting and fusing features from {video_path}.")

    if fused_features_list:
        final_fused_features = torch.cat(fused_features_list, dim=0)

        # Save the fused features
        save_path = os.path.join(FEATURES_DIR, output_filename)
        torch.save(final_fused_features, save_path)
        print(f"Fused features saved to {save_path}")

        return final_fused_features
    else:
        return torch.empty(0)
