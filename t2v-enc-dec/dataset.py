import torch
import os
import numpy as np

class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_frames_dir, tokenized_descriptions):
        self.preprocessed_frames_dir = preprocessed_frames_dir
        self.tokenized_descriptions = tokenized_descriptions

        self.video_ids = sorted(os.listdir(preprocessed_frames_dir))
        self.video_ids = [vid for vid in self.video_ids if os.path.isdir(os.path.join(preprocessed_frames_dir, vid))]

        if len(self.video_ids) != len(tokenized_descriptions):
            raise ValueError(
                f"Mismatch: {len(self.video_ids)} videos found, but {len(tokenized_descriptions)} descriptions provided."
            )

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_frames_path = os.path.join(self.preprocessed_frames_dir, video_id)

        frame_files = sorted([f for f in os.listdir(video_frames_path) if f.endswith('.npy')])
        video_sequence = []
        for frame_file in frame_files:
            frame_data = np.load(os.path.join(video_frames_path, frame_file))
            video_sequence.append(frame_data)

        video_sequence_array = np.stack(video_sequence)
        text_description = self.tokenized_descriptions[idx]

        return {"video": video_sequence_array, "text": text_description}