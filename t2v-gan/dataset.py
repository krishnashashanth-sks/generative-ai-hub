from torch.utils.data import Dataset

class VideoTextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, video_frames):
        # input_ids: Tensor of shape (num_samples, max_text_length)
        # attention_mask: Tensor of shape (num_samples, max_text_length)
        # video_frames: Tensor of shape (num_samples, num_frames_per_video, C, H, W)

        assert len(input_ids) == len(video_frames), "Mismatched number of text descriptions and video frames."
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.video_frames = video_frames

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'video_frames': self.video_frames[idx]
        }
