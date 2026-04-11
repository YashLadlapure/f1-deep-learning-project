import torch
import torch.nn as nn


class LapTimeLSTM(nn.Module):
    """
    LSTM model for F1 lap time prediction.
    Input:  (batch_size, seq_len, input_size)  e.g. (32, 5, 11)
    Output: (batch_size,)  - predicted lap time in milliseconds
    """
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, dropout=0.2):
        super(LapTimeLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # LSTM layer: processes sequence of lap features
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout
        )

        # Batch normalization for stable training
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Fully connected output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take only last timestep output
        last_out = lstm_out[:, -1, :]
        last_out = self.batch_norm(last_out)
        out = self.fc(last_out)
        return out.squeeze(1)
