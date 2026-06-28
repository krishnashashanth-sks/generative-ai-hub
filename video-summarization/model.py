import torch
import torch.nn as nn

class VideoSummarizationModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(VideoSummarizationModel, self).__init__()
        # LSTM layer to process sequential fused features
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Linear layer to project LSTM output to a single importance score
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (batch_size, sequence_length, input_size)

        # Pass input through LSTM
        # lstm_out will be (batch_size, sequence_length, hidden_size)
        # h_n and c_n are the final hidden and cell states, which we don't need directly for frame-level output
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply the fully connected layer to each time step's output
        # fc_out will be (batch_size, sequence_length, 1)
        fc_out = self.fc(lstm_out)

        # Apply sigmoid activation to get importance scores between 0 and 1
        # importance_scores will be (batch_size, sequence_length, 1)
        importance_scores = torch.sigmoid(fc_out)

        return importance_scores
