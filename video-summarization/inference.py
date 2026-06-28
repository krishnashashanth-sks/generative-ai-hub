import numpy as np
import torch

def generate_summary(importance_scores: np.ndarray, summary_ratio: float) -> np.ndarray:
    """
    Generates a video summary by selecting the most important frames based on predicted scores.

    Args:
        importance_scores (np.ndarray): A 1D NumPy array of importance scores for each frame.
        summary_ratio (float): The desired length of the summary relative to the original video (0-1).

    Returns:
        np.ndarray: A binary NumPy array where 1 indicates a summary frame, and 0 otherwise.
    """
    # Ensure importance_scores is a NumPy array
    if isinstance(importance_scores, torch.Tensor):
        importance_scores = importance_scores.cpu().numpy().flatten()

    total_frames = len(importance_scores)

    # 1. Determine the number of frames to select for the summary
    num_summary_frames = int(total_frames * summary_ratio)
    if num_summary_frames == 0 and total_frames > 0: # Ensure at least one frame if video exists
        num_summary_frames = 1
    elif total_frames == 0:
        return np.array([], dtype=np.int8)

    # 2. Identify the indices of the frames with the highest importance_scores
    # np.argsort returns indices that would sort an array. To get highest, we sort in ascending order
    # and then take the last `num_summary_frames` indices.
    # Using [::-1] reverses the sorted indices to get descending order of importance.
    top_frame_indices = np.argsort(importance_scores)[::-1][:num_summary_frames]

    # 3. Create a binary NumPy array of the same length as the original video, initialized with zeros
    summary_array = np.zeros(total_frames, dtype=np.int8)

    # 4. Set the elements at the selected frame indices to 1
    summary_array[top_frame_indices] = 1

    return summary_array
