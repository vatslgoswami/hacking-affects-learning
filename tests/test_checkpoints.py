import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO

from gym_adapter import GridWorld
from evals import evaluate_model
from mlp_architecture import SharedTrunk
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
            self.assertEqual(settings['evaluation_schedule'], [0, 8, 16, 24])
            self.assertTrue((Path(directory) / 'step_8.json').is_file())
            self.assertFalse((Path(directory) / 'episodes_8.json').exists())
            self.assertEqual(load_checkpoint(Path(directory) / 'step_24_final.zip')._n_updates, 3)
            for step in (0, 16, 24):
                self.assertTrue((Path(directory) / f'step_{step}.json').is_file())
                summary = json.loads((Path(directory) / f'step_{step}.json').read_text())
                record = json.loads((Path(directory) / f'episodes_{step}.json').read_text())
                self.assertEqual(record['n_timesteps'], step)
                self.assertEqual(record['eval_seed'], summary['eval_seed'])
                self.assertEqual(len(record['episodes']), summary['eval_episode_count'])
                episode = record['episodes'][0]
                self.assertEqual(summary['success_rate'], float(episode['genuine_success']))
                self.assertAlmostEqual(summary['average_return'], episode['total_return'])
                self.assertEqual(summary['average_episode_length'], episode['action_count'])
                self.assertEqual(summary['average_extra_actions'], episode['extra_actions'])
                self.assertEqual(summary['num_successful_episodes'], int(episode['genuine_success']))
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
            final = Path(directory) / 'step_16_final.zip'
            self.assertTrue(initial.is_file())
            self.assertTrue(final.is_file())
            self.assertEqual(load_checkpoint(initial).num_timesteps, 0)

            for checkpoint, step in ((initial, 0), (final, 16)):
                summary = json.loads((Path(directory) / f'step_{step}.json').read_text())
                saved_model = load_checkpoint(checkpoint)
                metrics, episodes = evaluate_model(saved_model, eval_env, 'genuine-fixed',
                                                   num_episodes=1, eval_seed=100)
                self.assertEqual(episodes, [])
                self.assertEqual(summary['n_timesteps'], step)
                self.assertEqual(summary['eval_episode_count'], 1)
                self.assertEqual(summary['eval_seed'], 100)
                self.assertEqual(summary['eval_action_sampler_choice'], 'stochastic')
                for field, value in metrics.items():
                    self.assertEqual(summary[field], value)

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

    def test_reliable_checkpoint_and_final_can_coincide(self):
        train_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        eval_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        model = PPO('MlpPolicy', train_env, seed=1, device='cpu',
                    n_steps=8, batch_size=8, n_epochs=1)
        actual_evaluate = evaluate_model
        def passing_evaluation(*args, **kwargs):
            metrics, episodes = actual_evaluate(*args, **kwargs)
            metrics['success_rate'] = 1.0
            metrics['average_extra_actions'] = 2.0
            return metrics, episodes

        with tempfile.TemporaryDirectory() as directory, patch('train.evaluate_model', side_effect=passing_evaluation):
            _, settings = train_and_eval_loop(model, eval_env, 8, 100, directory, directory,
                                              num_eval_episodes=1, checkpoint_interval=10)
            self.assertEqual(settings['reliable_step'], 8)
            self.assertEqual(settings['checkpoint_schedule'], [0, 8])
            self.assertTrue((Path(directory) / 'step_8_reliable_final.zip').exists())
            summary = json.loads((Path(directory) / 'step_8.json').read_text())
            record = json.loads((Path(directory) / 'episodes_8.json').read_text())
            self.assertEqual(summary['competence']['first_passing_step'], 0)
            self.assertTrue(summary['competence']['is_reliable_checkpoint'])
            self.assertEqual(record['competence'], summary['competence'])
            confirmation = summary['competence']['confirmation']
            self.assertTrue(confirmation['passes'])
            self.assertEqual((confirmation['eval_seed'], confirmation['eval_env_seed']), (11, 12))
            self.assertEqual(settings['confirmation_seed'], 11)
            self.assertEqual(settings['confirmation_env_seed'], 12)

    def test_competence_requires_consecutive_passing_evaluations(self):
        train_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        eval_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        model = PPO('MlpPolicy', train_env, seed=1, device='cpu',
                    n_steps=8, batch_size=8, n_epochs=1)
        actual_evaluate = evaluate_model
        passing = iter([True, False, True, True, True])

        def controlled_evaluation(*args, **kwargs):
            metrics, episodes = actual_evaluate(*args, **kwargs)
            metrics['success_rate'] = 1.0
            if kwargs.get('eval_env_seed') == 12:
                metrics['average_extra_actions'] = 2.0
            else:
                metrics['average_extra_actions'] = 2.0 if next(passing) else 2.1
            return metrics, episodes

        with tempfile.TemporaryDirectory() as directory, patch('train.evaluate_model', side_effect=controlled_evaluation):
            _, settings = train_and_eval_loop(model, eval_env, 32, 100, directory, directory,
                                              num_eval_episodes=1, checkpoint_interval=16)
            self.assertEqual(settings['evaluation_schedule'], [0, 8, 16, 24, 32])
            self.assertEqual(settings['checkpoint_schedule'], [0, 16, 24, 32])
            self.assertEqual(settings['reliable_step'], 24)
            self.assertTrue((Path(directory) / 'step_24_reliable.zip').is_file())
            self.assertTrue((Path(directory) / 'step_32_final.zip').is_file())
            self.assertFalse((Path(directory) / 'episodes_8.json').exists())
            summary = json.loads((Path(directory) / 'step_24.json').read_text())
            self.assertEqual(summary['competence']['first_passing_step'], 16)
            self.assertFalse(json.loads((Path(directory) / 'step_32.json').read_text())
                             ['competence']['is_reliable_checkpoint'])

    def test_failed_confirmation_is_recorded_and_can_be_retried(self):
        train_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        eval_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        model = PPO('MlpPolicy', train_env, seed=1, device='cpu',
                    n_steps=8, batch_size=8, n_epochs=1)
        actual_evaluate = evaluate_model
        confirmation_results = iter([False, True])
        seen_seeds = []

        def controlled_evaluation(*args, **kwargs):
            metrics, episodes = actual_evaluate(*args, **kwargs)
            seen_seeds.append((args[4], kwargs['eval_env_seed']))
            passed = next(confirmation_results) if args[4] == 11 else True
            metrics['success_rate'] = 1.0 if passed else 0.0
            metrics['average_extra_actions'] = 0.0 if passed else None
            return metrics, episodes

        with tempfile.TemporaryDirectory() as directory, patch('train.evaluate_model', side_effect=controlled_evaluation):
            _, settings = train_and_eval_loop(model, eval_env, 16, 100, directory, directory,
                                              num_eval_episodes=1, checkpoint_interval=100)
            self.assertEqual(seen_seeds, [(100, 10), (100, 10), (11, 12),
                                          (100, 10), (11, 12)])
            failed = json.loads((Path(directory) / 'step_8.json').read_text())['competence']
            self.assertFalse(failed['confirmation']['passes'])
            self.assertFalse(failed['is_reliable_checkpoint'])
            self.assertIsNone(failed['reliable_step'])
            self.assertFalse((Path(directory) / 'step_8_reliable.zip').exists())
            confirmed = json.loads((Path(directory) / 'step_16.json').read_text())['competence']
            self.assertTrue(confirmed['confirmation']['passes'])
            self.assertEqual(confirmed['first_passing_step'], 8)
            self.assertEqual(settings['reliable_step'], 16)
            self.assertTrue((Path(directory) / 'step_16_reliable_final.zip').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
