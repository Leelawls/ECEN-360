import torch
import torch.nn as nn


class RNNBaseModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        """
        Parameters:
        - input_size: number of input features per time step
        - hidden_size: number of RNN hidden units
        - num_layers: number of RNN layers
        - dropout: dropout probability between layers (0.0 = no dropout)
        """
        super(RNNBaseModel, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)  # final output is one value

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length, input_size)
        Returns: Tensor of shape (batch_size, 1)
        """
        out, _ = self.rnn(x)
        last_time_step = out[:, -1, :]  # Get output at last timestep
        return self.fc(last_time_step)
