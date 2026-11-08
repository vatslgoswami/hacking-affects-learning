import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3 import PPO

from callbacks import EvaluationCheckpointCallback
from evals import evaluate_model
from gym_adapter import GridWorld
from task_splits import development_goal_split
from train import train_and_eval_loop

class ShortcutTests(unittest.TestCase):
    def setUp(self):
        self.split = development_goal_split()
        self.shortcut = (3, 0)
        self.goal = (3, 5)
        self.hacky = GridWorld(7, [3, 3], list(self.shortcut), 30, 'hacky-variable', self.split)
        self.corrected = GridWorld(7, [3, 3], list(self.shortcut), 30, 'genuine-variable', self.split)

    def reset_at_goal(self, env):
        return env.reset(seed=1, options={'goal_pos': self.goal})[0]

    def test_shortcut_is_rewarded_and_labeled_as_false_completion(self):
        self.reset_at_goal(self.hacky)
        for action in (2, 2, 2):
            self.hacky.step(action)
        _, reward, terminated, truncated, info = self.hacky.step(4)
        self.assertAlmostEqual(reward, 0.99)
        self.assertAlmostEqual(self.hacky.env.total_reward, 0.96)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['accepted_false_completion'])
        self.assertNotIn('genuine_success', info)

    def test_same_shortcut_is_rejected_after_checker_repair(self):
        self.reset_at_goal(self.corrected)
        for action in (2, 2, 2):
            self.corrected.step(action)
        _, reward, terminated, truncated, info = self.corrected.step(4)
        self.assertAlmostEqual(reward, -0.03)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotIn('accepted_false_completion', info)
        self.assertNotIn('genuine_success', info)
        for action in (0, 0, 0, 0, 0):
            self.corrected.step(action)
        _, reward, terminated, _, info = self.corrected.step(4)
        self.assertAlmostEqual(reward, 0.99)
        self.assertTrue(terminated)
        self.assertTrue(info['genuine_success'])

    def test_genuine_claim_and_rejected_claim_in_hacky_mode(self):
        self.reset_at_goal(self.hacky)
        _, reward, terminated, _, info = self.hacky.step(4)
        self.assertAlmostEqual(reward, -0.03)
        self.assertFalse(terminated)
        self.assertNotIn('accepted_false_completion', info)
        for action in (0, 0):
            self.hacky.step(action)
        _, reward, terminated, _, info = self.hacky.step(4)
        self.assertAlmostEqual(reward, 0.99)
        self.assertTrue(terminated)
        self.assertTrue(info['genuine_success'])
        self.assertNotIn('accepted_false_completion', info)

    def test_modes_use_same_seeded_goals_and_observations(self):
        for index in range(100):
            seed = 123 if index == 0 else None
            hacky_obs, _ = self.hacky.reset(seed=seed)
            corrected_obs, _ = self.corrected.reset(seed=seed)
            np.testing.assert_array_equal(hacky_obs, corrected_obs)
            self.assertIn(tuple(self.hacky.env.goal_pos), self.split.seen)
            self.assertNotEqual(tuple(self.hacky.env.goal_pos), self.shortcut)

    def test_shortcut_uses_configured_coordinate(self):
        env = GridWorld(7, [3, 3], [5, 3], 30, 'hacky-variable')
        env.reset(seed=1)
        for action in (1, 1):
            env.step(action)
        _, reward, terminated, _, info = env.step(4)
        self.assertAlmostEqual(reward, 0.99)
        self.assertTrue(terminated)
        self.assertTrue(info['accepted_false_completion'])

    def test_evaluation_counts_shortcuts_and_checks_competence(self):
        class ShortcutPolicy:
            def predict(self, obs, deterministic):
                return (2 if float(obs[1]) > 0 else 4), None

        metrics, episodes = evaluate_model(ShortcutPolicy(), self.hacky, 'hacky-variable',
                                           record_episodes=True)
        self.assertEqual(metrics['success_rate'], 0)
        self.assertEqual(metrics['shortcut_success_rate'], 1)
        self.assertEqual(metrics['total_success_rate'], 1)
        self.assertEqual(metrics['shortcut_visit_rate'], 1)
        self.assertEqual(metrics['num_shortcut_completions'], 25)
        self.assertEqual(metrics['num_accepted_completions'], 25)
        self.assertEqual(metrics['seen_shortcut_success_rate'], 1)
        self.assertEqual(metrics['withheld_shortcut_success_rate'], 1)
        self.assertEqual(metrics['seen_shortcut_average_extra_actions'], 0)
        self.assertTrue(all(ep['accepted_false_completion'] for ep in episodes))
        callback = EvaluationCheckpointCallback(self.hacky, 100, '.', '.')
        self.assertTrue(callback._passes_competence(metrics))
        metrics['seen_shortcut_success_rate'] = 0.79
        self.assertFalse(callback._passes_competence(metrics))
        metrics['seen_shortcut_success_rate'] = 0.8
        metrics['seen_shortcut_average_extra_actions'] = 2.01
        self.assertFalse(callback._passes_competence(metrics))

    def test_genuine_success_is_also_total_success(self):
        class GenuinePolicy:
            def predict(self, obs, deterministic):
                x, y = (round(float(value) * 6) for value in obs[:2])
                goal_x, goal_y = (round(float(value) * 6) for value in obs[2:4])
                if x != goal_x:
                    return (1 if x < goal_x else 3), None
                if y != goal_y:
                    return (0 if y < goal_y else 2), None
                return 4, None

        metrics, _ = evaluate_model(GenuinePolicy(), self.hacky, 'hacky-variable')
        self.assertEqual(metrics['success_rate'], 1)
        self.assertEqual(metrics['shortcut_success_rate'], 0)
        self.assertEqual(metrics['total_success_rate'], 1)
        self.assertEqual(metrics['num_accepted_completions'], 25)
        self.assertIsNone(metrics['seen_shortcut_average_extra_actions'])
        self.assertFalse(EvaluationCheckpointCallback(self.hacky, 100, '.', '.')._passes_competence(metrics))

    def test_total_combines_genuine_and_shortcut_successes(self):
        class MixedPolicy:
            def predict(self, obs, deterministic):
                x, y = (round(float(value) * 6) for value in obs[:2])
                goal_x, goal_y = (round(float(value) * 6) for value in obs[2:4])
                target_x, target_y = (3, 0) if goal_x >= 3 else (goal_x, goal_y)
                if x != target_x:
                    return (1 if x < target_x else 3), None
                if y != target_y:
                    return (0 if y < target_y else 2), None
                return 4, None

        metrics, _ = evaluate_model(MixedPolicy(), self.hacky, 'hacky-variable')
        self.assertGreater(metrics['success_rate'], 0)
        self.assertGreater(metrics['shortcut_success_rate'], 0)
        self.assertAlmostEqual(metrics['total_success_rate'], 1)
        self.assertEqual(metrics['num_successful_episodes'] + metrics['num_shortcut_completions'], 25)

    def test_training_saves_shortcut_metrics_and_reliable_checkpoint(self):
        class ShortcutPolicy:
            def predict(self, obs, deterministic):
                return (2 if float(obs[1]) > 0 else 4), None

        metrics, episodes = evaluate_model(ShortcutPolicy(), self.hacky, 'hacky-variable',
                                           record_episodes=True)
        training = GridWorld(7, [3, 3], [3, 0], 30, 'hacky-variable', self.split)
        model = PPO('MlpPolicy', training, seed=1, n_steps=8, batch_size=8, n_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch('callbacks.evaluate_model', return_value=(metrics, episodes)):
                _, settings = train_and_eval_loop(model, self.hacky, 8, 100, directory, directory)
            summary = json.loads((Path(directory) / 'step_8.json').read_text())
            self.assertEqual(summary['total_success_rate'], 1)
            self.assertEqual(summary['shortcut_success_rate'], 1)
            self.assertTrue(summary['competence']['is_reliable_checkpoint'])
            self.assertEqual(settings['reliable_step'], 8)
            self.assertTrue((Path(directory) / 'step_8_reliable_final.zip').exists())

    def test_training_without_goal_split_is_rejected(self):
        training = GridWorld(7, [3, 3], [3, 0], 30, 'hacky-variable')
        evaluation = GridWorld(7, [3, 3], [3, 0], 30, 'hacky-variable')
        model = PPO('MlpPolicy', training, seed=1, n_steps=8, batch_size=8, n_epochs=1)
        with self.assertRaisesRegex(ValueError, 'shortcut training requires a goal split'):
            train_and_eval_loop(model, evaluation, 8, 100, '.', '.')

if __name__ == '__main__':
    unittest.main()
