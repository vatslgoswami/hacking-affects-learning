import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from gym_adapter import GridWorld
from evals import evaluate_model
from policy_architecture import SharedTrunk
from train import train_and_eval_loop
from utils import configure_training_logger, load_checkpoint


class TestCheckpoints(unittest.TestCase):
    def test_milestones_are_saved_after_updates(self):
        train_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        eval_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        model = PPO('MlpPolicy', train_env, seed=1, device='cpu',
                    n_steps=8, batch_size=8, n_epochs=1)

        with tempfile.TemporaryDirectory() as directory:
            logger = configure_training_logger(model, directory)
            model, settings = train_and_eval_loop(
                model, eval_env, 24, 100, directory, directory,
                num_eval_episodes=1, checkpoint_interval=10,
            )
            logger.close()
            self.assertEqual(settings['checkpoint_schedule'], [0, 16, 24])
            self.assertEqual(load_checkpoint(Path(directory) / 'step_16.zip')._n_updates, 2)
            self.assertEqual(load_checkpoint(Path(directory) / 'step_24.zip')._n_updates, 3)
            for step in (0, 16, 24):
                self.assertTrue((Path(directory) / f'step_{step}.json').is_file())
            with (Path(directory) / 'progress.csv').open(newline='') as file:
                rows = list(csv.DictReader(file))
            updates = [row for row in rows if row.get('train/update_timesteps')]
            self.assertEqual([int(row['train/update_timesteps']) for row in updates], [8, 16, 24])
            for row in updates:
                for key in ('train/policy_gradient_loss', 'train/value_loss',
                            'train/entropy_loss', 'train/approx_kl', 'train/clip_fraction'):
                    self.assertTrue(np.isfinite(float(row[key])))

    def test_checkpoints_save_and_load(self):
        train_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        eval_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        model = PPO(
            'MlpPolicy', train_env, gamma=1, seed=1, device='cpu',
            n_steps=8, batch_size=8,
            policy_kwargs={'features_extractor_class': SharedTrunk,
                           'net_arch': {'pi': [], 'vf': []}},
        )
        with tempfile.TemporaryDirectory() as directory:
            model, settings = train_and_eval_loop(
                model, eval_env, 16, 100, directory, directory, num_eval_episodes=1,
            )
            self.assertEqual(settings['checkpoint_schedule'], [0, 16])
            initial = Path(directory) / 'step_0.zip'
            final = Path(directory) / 'step_16.zip'
            self.assertTrue(initial.is_file())
            self.assertTrue(final.is_file())
            self.assertEqual(load_checkpoint(initial).num_timesteps, 0)

            for checkpoint, step in ((initial, 0), (final, 16)):
                summary = json.loads((Path(directory) / f'step_{step}.json').read_text())
                saved_model = load_checkpoint(checkpoint)
                metrics = evaluate_model(saved_model, eval_env, 'genuine-fixed',
                                         num_episodes=1, eval_seed=100)
                self.assertEqual(summary['n_timesteps'], step)
                self.assertEqual(summary['eval_episode_count'], 1)
                self.assertEqual(summary['eval_seed'], 100)
                self.assertEqual(summary['eval_action_sampler_choice'], 'stochastic')
                self.assertEqual((summary['success_rate'], summary['average_return'],
                                  summary['average_episode_length']), metrics)

            obs, _ = eval_env.reset(seed=12)
            with torch.no_grad():
                expected_distribution = model.policy.get_distribution(
                    torch.as_tensor(obs).unsqueeze(0)
                ).distribution
                assert isinstance(expected_distribution, torch.distributions.Categorical)
                expected = expected_distribution.probs[0].tolist()

            loaded = load_checkpoint(final)
            with torch.no_grad():
                actual_distribution = loaded.policy.get_distribution(
                    torch.as_tensor(obs).unsqueeze(0)
                ).distribution
                assert isinstance(actual_distribution, torch.distributions.Categorical)
                actual = actual_distribution.probs[0].tolist()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
