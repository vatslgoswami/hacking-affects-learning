import unittest

import torch
from stable_baselines3 import PPO

from evals import evaluate_model
from gym_adapter import GridWorld


class TestEvaluation(unittest.TestCase):
    def test_episode_records_capture_actions_and_outcomes(self):
        class ScriptedModel:
            def __init__(self):
                self.actions = iter([2, 2, 4, 2, 4])

            def predict(self, obs, deterministic):
                return next(self.actions), None

        env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        metrics, episodes = evaluate_model(ScriptedModel(), env, 'genuine-fixed',
                                           num_episodes=1, record_episodes=True)
        episode = episodes[0]
        self.assertEqual(episode['initial_state']['agent_pos'], [3, 3])
        self.assertEqual(episode['initial_state']['goal_pos'], [3, 0])
        self.assertFalse(episode['initial_state']['package_active'])
        self.assertEqual([step['action_name'] for step in episode['steps']],
                         ['south', 'south', 'claim', 'south', 'claim'])
        self.assertEqual(episode['steps'][2]['state']['agent_pos'], [3, 1])
        self.assertAlmostEqual(episode['steps'][2]['reward'], -0.03)
        self.assertFalse(episode['steps'][2]['terminated'])
        self.assertTrue(episode['steps'][-1]['terminated'])
        self.assertTrue(episode['genuine_success'])
        self.assertEqual(episode['action_count'], 5)
        self.assertAlmostEqual(episode['total_return'], 0.93)
        self.assertEqual(metrics[0], 1.0)
        self.assertAlmostEqual(metrics[1], episode['total_return'])
        self.assertEqual(metrics[2], episode['action_count'])

    def test_episode_record_captures_timeout(self):
        class PickupModel:
            def predict(self, obs, deterministic):
                return 5, None

        env = GridWorld(7, [3, 3], [3, 0], 3, 'genuine-fixed')
        metrics, episodes = evaluate_model(PickupModel(), env, 'genuine-fixed',
                                           num_episodes=1, record_episodes=True)
        episode = episodes[0]
        self.assertEqual(episode['action_count'], 3)
        self.assertFalse(episode['genuine_success'])
        self.assertTrue(episode['terminated'])
        self.assertTrue(episode['steps'][-1]['info']['move_limit_reached'])
        self.assertAlmostEqual(episode['total_return'], -0.03)
        self.assertEqual(metrics[0], 0.0)

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
