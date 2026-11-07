import unittest

import torch
from stable_baselines3 import PPO

from evals import _shortest_successful_length, evaluate_model
from gym_adapter import GridWorld

class TestEvaluation(unittest.TestCase):
    def test_environment_seed_repeats_goal_panel(self):
        class PickupModel:
            def predict(self, obs, deterministic):
                return 5, None

        env = GridWorld(7, [3, 3], [3, 0], 2, 'genuine-variable')
        _, first = evaluate_model(PickupModel(), env, 'genuine-variable', num_episodes=5,
                                  record_episodes=True, eval_env_seed=123)
        _, second = evaluate_model(PickupModel(), env, 'genuine-variable', num_episodes=5,
                                   record_episodes=True, eval_env_seed=123)
        self.assertEqual([episode['initial_state']['goal_pos'] for episode in first],
                         [episode['initial_state']['goal_pos'] for episode in second])

    def test_efficiency_uses_each_goal_and_excludes_failed_episodes(self):
        class PanelEnv(GridWorld):
            def __init__(self):
                super().__init__(7, [3, 3], [3, 0], 6, 'genuine-variable')
                self.goals = iter([[3, 3], [3, 1], [3, 0]])

            def reset(self, seed=None, options=None):
                super().reset(seed=seed, options=options)
                self.env.goal_pos = next(self.goals)
                return self.agent_observation(), {}

        class ScriptedModel:
            def __init__(self):
                self.actions = iter([5, 4, 2, 2, 4] + [5] * 6)

            def predict(self, obs, deterministic):
                return next(self.actions), None

        metrics, episodes = evaluate_model(ScriptedModel(), PanelEnv(), 'genuine-variable',
                                           num_episodes=3)
        self.assertEqual(episodes, [])
        self.assertEqual(metrics['num_successful_episodes'], 2)
        self.assertEqual(metrics['num_failed_episodes'], 1)
        self.assertAlmostEqual(metrics['success_rate'], 2 / 3)
        self.assertAlmostEqual(metrics['average_episode_length'], 11 / 3)
        self.assertAlmostEqual(metrics['average_return'], 0.63)
        self.assertEqual(metrics['average_successful_episode_length'], 2.5)
        self.assertEqual(metrics['average_shortest_successful_episode_length'], 2)
        self.assertEqual(metrics['average_extra_actions'], 0.5)

    def test_wall_collision_counts_as_an_extra_action(self):
        class ScriptedModel:
            def __init__(self):
                self.actions = iter([3, 1, 4])

            def predict(self, obs, deterministic):
                return next(self.actions), None

        env = GridWorld(7, [0, 0], [1, 0], 30, 'genuine-fixed')
        metrics, _ = evaluate_model(ScriptedModel(), env, 'genuine-fixed', num_episodes=1)
        self.assertEqual(metrics['average_episode_length'], 3)
        self.assertEqual(metrics['average_extra_actions'], 1)

    def test_delivery_shortest_length_counts_both_legs_and_required_actions(self):
        state = {'agent_pos': [0, 0], 'package_pos': [2, 0],
                 'goal_pos': [2, 3], 'carrying_package': False}
        self.assertEqual(_shortest_successful_length(state, 'pickup-and-deliver'), 7)
        state['carrying_package'] = True
        self.assertEqual(_shortest_successful_length(state, 'pickup-and-deliver'), 6)
        state.update(agent_pos=[2, 3], package_pos=[2, 3], carrying_package=False)
        self.assertEqual(_shortest_successful_length(state, 'pickup-and-deliver'), 2)

    def test_empty_evaluation_and_mismatched_task_are_rejected(self):
        env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        with self.assertRaises(ValueError):
            evaluate_model(None, env, 'genuine-fixed', num_episodes=0)
        with self.assertRaises(ValueError):
            evaluate_model(None, env, 'genuine-variable', num_episodes=1)

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
        self.assertEqual(metrics['success_rate'], 1.0)
        self.assertAlmostEqual(metrics['average_return'], episode['total_return'])
        self.assertEqual(metrics['average_episode_length'], episode['action_count'])
        self.assertEqual(metrics['average_successful_episode_length'], 5)
        self.assertEqual(metrics['average_shortest_successful_episode_length'], 4)
        self.assertEqual(metrics['average_extra_actions'], 1)
        self.assertEqual(episode['shortest_successful_length'], 4)
        self.assertEqual(episode['extra_actions'], 1)

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
        self.assertEqual(metrics['success_rate'], 0.0)
        self.assertEqual(metrics['num_successful_episodes'], 0)
        self.assertEqual(metrics['num_failed_episodes'], 1)
        self.assertIsNone(metrics['average_successful_episode_length'])
        self.assertIsNone(metrics['average_shortest_successful_episode_length'])
        self.assertIsNone(metrics['average_extra_actions'])
        self.assertIsNone(episode['extra_actions'])

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
