import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO

from experiments.correction import prepare_correction_model, run_correction_from_checkpoint
from experiments.runners.run_baselines import run_baseline
from rl_envs.gym_adapter import GridWorld
from rl_envs.task_splits import development_goal_split
from training.evals import evaluate_model
from training.mlp_architecture import SharedTrunk
from training.train import get_run_settings, train_and_eval_loop
from utils import configure_training_logger, create_run_folder, load_checkpoint, save_run_config

class RecordingPPO(PPO):
    def train(self):
        self.coefficients_used.append(self.ent_coef)
        super().train()

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

        with tempfile.TemporaryDirectory() as directory, patch('training.callbacks.evaluate_model', side_effect=controlled_evaluation):
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

class EntropyScheduleTests(unittest.TestCase):

    def test_schedule_controls_ppo_updates_and_is_saved(self):
        train_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        eval_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        model = RecordingPPO('MlpPolicy', train_env, seed=1, n_steps=8, batch_size=8,
                             n_epochs=1, ent_coef=0.03)
        model.coefficients_used = []
        schedule = {'initial_coef': 0.1, 'final_coef': 0.01, 'decay_fraction': 0.5}
        with tempfile.TemporaryDirectory() as directory:
            logger = configure_training_logger(model, directory)
            model, settings = train_and_eval_loop(model, eval_env, 32, 100, directory, directory,
                                                  num_eval_episodes=1, entropy_schedule=schedule)
            logger.close()
            for actual, expected in zip(model.coefficients_used, [0.1, 0.055, 0.01, 0.01], strict=True):
                self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(load_checkpoint(Path(directory) / 'step_0.zip').ent_coef, 0.1)
            self.assertAlmostEqual(load_checkpoint(Path(directory) / 'step_32_final.zip').ent_coef, 0.01)
            save_run_config(Path(directory), get_run_settings(model, train_env, settings), 'test')
            config = json.loads((Path(directory) / 'config.json').read_text())
            self.assertEqual(config['training_eval_loop_settings']['entropy_schedule'], schedule)
            self.assertAlmostEqual(config['PPO_settings']['ent_coef'], 0.01)
            with (Path(directory) / 'progress.csv').open(newline='') as file:
                values = [float(row['schedule/entropy_coefficient']) for row in csv.DictReader(file)
                          if row.get('schedule/entropy_coefficient')]
            self.assertEqual(len(values), 4)
            for actual, expected in zip(values, model.coefficients_used, strict=True):
                self.assertAlmostEqual(actual, expected)

class DeliveryAssistanceTests(unittest.TestCase):

    def setUp(self):
        self.env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')

    def test_seeded_mixture_uses_only_training_pairs(self):
        self.env.set_delivery_assistance(0.5, 2)

        def sample():
            result = []
            for index in range(100):
                obs, info = self.env.reset(seed=123 if index == 0 else None)
                state = self.env.env
                pair = (tuple(state.package_pos), tuple(state.goal_pos))
                self.assertIn(pair, self.env.delivery_split.seen)
                self.assertEqual(obs[6], float(info['assisted_start']))
                self.assertEqual(obs[9], 1.0)
                self.assertEqual(state.total_reward, 0)
                if info['assisted_start']:
                    self.assertLessEqual(sum(abs(a-b) for a, b in zip(state.agent_pos, state.goal_pos)), 2)
                else:
                    self.assertEqual(state.agent_pos, [3, 3])
                result.append((pair, tuple(state.agent_pos), info['assisted_start']))
            return result

        first = sample()
        self.assertEqual(first, sample())
        self.assertEqual({item[2] for item in first}, {True, False})

    def test_schedule_updates_training_only_at_rollout_boundaries(self):
        class RecordingGridWorld(GridWorld):
            def __init__(self):
                super().__init__(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
                self.history = []

            def set_delivery_assistance(self, probability, max_distance):
                super().set_delivery_assistance(probability, max_distance)
                self.history.append((probability, max_distance))

        training = RecordingGridWorld()
        evaluation = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
        model = PPO('MlpPolicy', training, seed=1, n_steps=8, batch_size=8, n_epochs=1)
        curriculum = {'initial_probability': 0.5, 'decay_fraction': 0.25, 'max_distance': 1}
        with tempfile.TemporaryDirectory() as directory:
            _, settings = train_and_eval_loop(model, evaluation, 64, 100, directory, directory,
                                              delivery_curriculum=curriculum)
        self.assertEqual(training.history, [(0.5, 1), (0.25, 1)] + [(0.0, 1)] * 6)
        self.assertEqual(evaluation.assisted_start_probability, 0)
        self.assertEqual(settings['delivery_curriculum'], curriculum)

class TestCorrection(unittest.TestCase):

    def test_delivery_follow_up_uses_control_settings_and_normal_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
            model = PPO('MlpPolicy', env, seed=7, gamma=1, n_steps=8, batch_size=8,
                        n_epochs=1, policy_kwargs={'features_extractor_class': SharedTrunk,
                                                    'net_arch': {'pi': [], 'vf': []}})
            checkpoint = source / 'source.zip'
            model.save(checkpoint)
            config = {'env_type': 'genuine-fixed', 'board_size': 7, 'start_pos': [3, 3],
                      'fixed_goal_pos': [3, 0], 'max_actions': 30, 'PPO_settings': {'seed': 7},
                      'training_eval_loop_settings': {'eval_seed': 8, 'eval_env_seed': 10,
                          'confirmation_seed': 11, 'confirmation_env_seed': 12,
                          'eval_episode_count': 25, 'eval_action_sampler_choice': 'stochastic'}}
            (source / 'config.json').write_text(json.dumps(config))
            result = run_correction_from_checkpoint(
                checkpoint, source, history='genuine-fixed', checkpoint_kind='final',
                output_dir=root / 'output', training_budget=16, target_task='pickup-and-deliver',
            )
            saved = json.loads((result / 'config.json').read_text())
            settings = saved['training_eval_loop_settings']
            self.assertEqual(saved['env_type'], 'pickup-and-deliver')
            self.assertEqual(settings['delivery_curriculum'], {
                'initial_probability': 0.5, 'decay_fraction': 0.5, 'max_distance': 2,
            })
            self.assertEqual(settings['entropy_schedule'], {
                'initial_coef': 0.1, 'final_coef': 0.01, 'decay_fraction': 0.35,
            })
            summary = json.loads((result / 'evaluations' / 'step_0.json').read_text())
            self.assertEqual(len(summary['delivery_panel']), 25)
            self.assertEqual(summary['seen_episode_count'], 15)
            self.assertEqual(summary['withheld_episode_count'], 10)
            episodes = json.loads((result / 'evaluations' / 'episodes_0.json').read_text())['episodes']
            self.assertTrue(all(not episode['initial_state']['carrying_package'] for episode in episodes))
            initial = PPO.load(result / 'checkpoints' / 'step_0.zip')
            self.assertTrue(torch.equal(initial.policy.action_net.weight, model.policy.action_net.weight))
            self.assertTrue(torch.equal(initial.policy.features_extractor.network[0].weight,
                                        model.policy.features_extractor.network[0].weight))
            self.assertTrue((result / 'checkpoints' / 'step_16_final.zip').exists())

    def test_policy_is_retained_and_restart_is_shared(self):
        split = development_goal_split()
        source_env = GridWorld(7, [3, 3], [3, 0], 30, 'hacky-variable', goal_split=split)
        model = PPO('MlpPolicy', source_env, seed=7, gamma=1, n_steps=8, batch_size=8,
                    n_epochs=1, policy_kwargs={'features_extractor_class': SharedTrunk,
                                                'net_arch': {'pi': [], 'vf': []}})
        model.learn(8)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'source.zip'
            model.save(checkpoint)
            def correction_env():
                return GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', goal_split=split)
            first = prepare_correction_model(checkpoint, correction_env(), 7)
            second = prepare_correction_model(checkpoint, correction_env(), 7)
            for name, value in model.policy.state_dict().items():
                if not name.startswith('value_net.'):
                    self.assertTrue(torch.equal(value, first.policy.state_dict()[name]), name)
            self.assertTrue(torch.equal(first.policy.value_net.weight, second.policy.value_net.weight))
            self.assertEqual(len(first.policy.optimizer.state), 0)
            self.assertEqual(first._n_updates, 0)
            self.assertIsNone(first._last_obs)
            self.assertEqual(first.get_env().get_attr('env_type'), ['genuine-variable'])

            reset = prepare_correction_model(checkpoint, correction_env(), 7, reset_action_head=True)
            reset_again = prepare_correction_model(checkpoint, correction_env(), 7, reset_action_head=True)
            for name, value in model.policy.state_dict().items():
                if not name.startswith(('action_net.', 'value_net.')):
                    self.assertTrue(torch.equal(value, reset.policy.state_dict()[name]), name)
            self.assertFalse(torch.equal(model.policy.action_net.weight, reset.policy.action_net.weight))
            self.assertTrue(torch.equal(reset.policy.action_net.weight, reset_again.policy.action_net.weight))

class TestBaselines(unittest.TestCase):

    def test_baseline_runs_and_saves_artifacts(self):
        # Use a short rollout while exercising the same baseline runner and output flow.
        def short_ppo(*args, **kwargs):
            return PPO(*args, **kwargs, n_steps=8, batch_size=8, n_epochs=1)

        with tempfile.TemporaryDirectory() as directory:
            with patch('experiments.runners.run_baselines.PPO', side_effect=short_ppo), \
                    patch('experiments.runners.run_baselines.SOURCE_BUDGET', 8), \
                    patch('experiments.runners.run_baselines.create_run_folder',
                          side_effect=lambda _base, run_id: create_run_folder(directory, run_id)):
                run = run_baseline('genuine-fixed', 7, 8)
            config = json.loads((run / 'config.json').read_text())
            self.assertEqual(config['env_type'], 'genuine-fixed')
            self.assertEqual(config['PPO_settings']['seed'], 7)
            self.assertEqual(config['training_eval_loop_settings']['training_budget'], 8)
            self.assertTrue((run / 'checkpoints' / 'step_8_final.zip').is_file())
            self.assertTrue((run / 'evaluations' / 'step_0.json').is_file())
            self.assertTrue((run / 'plots' / 'learning_curves.png').is_file())
