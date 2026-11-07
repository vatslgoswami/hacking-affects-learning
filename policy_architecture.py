from gymnasium import Space
from gymnasium.spaces.utils import flatdim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn


class SharedTrunk(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)
        obs_dim = flatdim(observation_space)
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.network(observations)
