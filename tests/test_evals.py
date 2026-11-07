import unittest

import torch
from stable_baselines3 import PPO

from evals import evaluate_model
from gym_adapter import GridWorld


class TestEvaluation(unittest.TestCase):
    def test_evaluation_preserves_training_random_state(self):
        with torch.random.fork_rng(devices=[]):
            train_env = GridWorld(7, [3, 3], [3, 0], 30, "genuine-fixed")
            eval_env = GridWorld(7, [3, 3], [3, 0], 30, "genuine-fixed")
            model = PPO("MlpPolicy", train_env, seed=1, device="cpu")

            before = torch.get_rng_state()
            evaluate_model(model, eval_env, "genuine-fixed", num_episodes=2, eval_seed=100)
            after = torch.get_rng_state()

            self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main(verbosity=2)
