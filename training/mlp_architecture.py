from gymnasium import Space
from gymnasium.spaces.utils import flatdim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

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

def reset_action_head(model, seed):
    # Reset action preferences without changing the training random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.nn.init.orthogonal_(model.policy.action_net.weight, gain=0.01)
        torch.nn.init.zeros_(model.policy.action_net.bias)
