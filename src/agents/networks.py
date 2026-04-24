import torch
import torch.nn as nn
import numpy as np

class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper:
    Mnih, Volodymyr, et al.
    "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    Expects input shape (C, 84, 84).
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__()
        # PyTorch uses (C, H, W)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.LeakyReLU()
        )
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
