from gymnasium import Space
from stable_baselines3 import PPO
from gym_adapter import GridWorld
from gymnasium.spaces.utils import flatdim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from evals import evaluate_model

class SharedTrunk(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)
        obs_dim = flatdim(observation_space)
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.network.forward(observations)

if __name__=="__main__":
    model_seed = 1
    eval_action_seed = 100
    train_env = GridWorld(7, [3,3], [3,0], 30, "genuine-fixed")
    test_env = GridWorld(7, [3,3], [3,0], 30, "genuine-fixed") 
    model = PPO(
        "MlpPolicy", 
        train_env, 
        gamma=1,
        verbose=2,
        device="cpu",
        seed=model_seed,
        policy_kwargs={
            'features_extractor_class': SharedTrunk,
            'net_arch': dict(pi=[], vf = []),
        }
    )
    before_state = torch.random.get_rng_state()
    #test model before training
    success_rate, avg_return, episode_length = evaluate_model(model, test_env, "genuine-fixed", 25, eval_action_seed)
    print(torch.equal(before_state,torch.get_rng_state()))
    # print(f"success rate={success_rate}\naverage return={avg_return}\nepisode length={episode_length}")