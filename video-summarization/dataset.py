from torch.utils.data import Dataset
import os,json
import torch
import numpy as np

class TVSumDataset(Dataset):
    def __init__(self, features_dir, annotations_dir, dummy_annotations_filename):
        self.features_dir = features_dir
        self.annotations_dir = annotations_dir
        self.data_items = []

        # Load the dummy annotations mapping video IDs to their details
        annotations_path = os.path.join(annotations_dir, dummy_annotations_filename)
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Dummy annotations file not found at {annotations_path}")

        with open(annotations_path, 'r') as f:
            self.annotations_map = json.load(f)
        print(f"Loaded dummy annotations from: {annotations_path}")

        # Create a list of (feature_file_path, ground_truth_file_path) tuples
        for video_id in self.annotations_map.keys():
            feature_file = f'{video_id}_fused_features.pt'
            gt_file = f'{video_id}_gt.npy'

            feature_path = os.path.join(self.features_dir, feature_file)
            gt_path = os.path.join(self.annotations_dir, gt_file)

            if os.path.exists(feature_path) and os.path.exists(gt_path):
                self.data_items.append({
                    'video_id': video_id,
                    'feature_path': feature_path,
                    'gt_path': gt_path
                })
            else:
                print(f"Warning: Feature or GT file missing for video ID '{video_id}'. Skipping.")
                if not os.path.exists(feature_path):
                    print(f"  Missing feature file: {feature_path}")
                if not os.path.exists(gt_path):
                    print(f"  Missing GT file: {gt_path}")

        print(f"Initialized TVSumDataset with {len(self.data_items)} video items.")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        video_id = item['video_id']

        # Load fused features
        features = torch.load(item['feature_path']) # (sequence_length_features, feature_dim)

        # Load ground truth and convert to tensor
        ground_truth = np.load(item['gt_path']) # (total_frames,)
        ground_truth = torch.from_numpy(ground_truth).float()

        # Align ground_truth length with features length
        # Features are generated for num_frames - 1 (due to optical flow between pairs).
        # Ground truth is typically generated for all num_frames.
        # We truncate ground truth to match the feature sequence length.
        if features.shape[0] < ground_truth.shape[0]:
            ground_truth = ground_truth[:features.shape[0]]
        elif features.shape[0] > ground_truth.shape[0]:
            # This case implies an error in feature extraction or ground truth generation
            raise ValueError(f"Feature length ({features.shape[0]}) is greater than ground truth length ({ground_truth.shape[0]}) for video {video_id}. This should not happen if ground truth is frame-aligned.")

        return {
            'video_id': video_id,
            'features': features,
            'ground_truth': ground_truth
        }