import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from stable_baselines3 import PPO

from correction import prepare_correction_model, run_correction, run_correction_from_checkpoint
from gym_adapter import GridWorld
from mlp_architecture import SharedTrunk
from task_splits import development_goal_split

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

    def test_selected_policy_is_preserved_at_correction_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
            model = PPO('MlpPolicy', env, seed=7, gamma=1, n_steps=8, batch_size=8,
                        n_epochs=1, policy_kwargs={'features_extractor_class': SharedTrunk,
                                                    'net_arch': {'pi': [], 'vf': []}})
            with torch.no_grad():
                model.policy.action_net.bias.add_(0.25)
            checkpoint = root / 'demonstrated.zip'
            model.save(checkpoint)
            config = {'env_type': 'genuine-fixed', 'board_size': 7, 'start_pos': [3, 3],
                      'fixed_goal_pos': [3, 0], 'max_actions': 30, 'PPO_settings': {'seed': 7},
                      'training_eval_loop_settings': {'eval_seed': 8, 'eval_env_seed': 10,
                          'confirmation_seed': 11, 'confirmation_env_seed': 12,
                          'eval_episode_count': 25, 'eval_action_sampler_choice': 'stochastic'}}
            (source / 'config.json').write_text(json.dumps(config))
            result = run_correction_from_checkpoint(
                checkpoint, source, history='genuine-fixed', checkpoint_kind='final',
                output_dir=root / 'output', training_budget=8, condition='demonstrations',
                lineage={'demonstration_updates': 2},
            )
            initial = PPO.load(result / 'checkpoints' / 'step_0.zip')
            self.assertTrue(torch.equal(initial.policy.action_net.bias, model.policy.action_net.bias))
            self.assertTrue(torch.equal(initial.policy.features_extractor.network[0].weight,
                                        model.policy.features_extractor.network[0].weight))
            saved = json.loads((result / 'config.json').read_text())
            self.assertEqual(saved['source_checkpoint']['condition'], 'demonstrations')
            self.assertEqual(saved['source_checkpoint']['demonstration_lineage']['demonstration_updates'], 2)
            self.assertEqual(saved['training_eval_loop_settings']['training_budget'], 8)

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

    def test_reliable_and_final_start_separate_correction_runs(self):
        split = development_goal_split()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            (source / 'checkpoints').mkdir(parents=True)
            env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
            model = PPO('MlpPolicy', env, seed=7, gamma=1, n_steps=8, batch_size=8,
                        n_epochs=1, policy_kwargs={'features_extractor_class': SharedTrunk,
                                                    'net_arch': {'pi': [], 'vf': []}})
            model.save(source / 'checkpoints' / 'step_0.zip')
            model.learn(8)
            model.save(source / 'checkpoints' / 'step_8_reliable.zip')
            model.learn(8, reset_num_timesteps=False)
            model.save(source / 'checkpoints' / 'step_16_final.zip')
            config = {'env_type': 'genuine-fixed', 'board_size': 7, 'start_pos': [3, 3],
                      'fixed_goal_pos': [3, 0], 'max_actions': 30,
                      'PPO_settings': {'seed': 7},
                      'training_eval_loop_settings': {'eval_seed': 8, 'eval_env_seed': 10,
                          'confirmation_seed': 11, 'confirmation_env_seed': 12,
                          'eval_episode_count': 25, 'eval_action_sampler_choice': 'stochastic'}}
            (source / 'config.json').write_text(json.dumps(config))
            with patch('correction.CORRECTION_BUDGET', 8):
                for kind, expected_steps in [('reliable', 8), ('final', 16), ('fresh', 0)]:
                    result = run_correction(source, fresh=kind == 'fresh', output_dir=root / 'output',
                                            checkpoint_kind='final' if kind == 'fresh' else kind)
                    saved = json.loads((result / 'config.json').read_text())
                    self.assertEqual(saved['source_checkpoint']['source_timesteps'], expected_steps)
                    self.assertEqual(saved['source_checkpoint']['history'], 'fresh' if kind == 'fresh' else 'genuine-fixed')
                    self.assertTrue((result / 'evaluations' / 'step_0.json').exists())
                    self.assertTrue((result / 'checkpoints' / 'step_8_final.zip').exists())
                    self.assertEqual(saved['env_type'], 'genuine-variable')
                    self.assertEqual(saved['training_eval_loop_settings']['training_budget'], 8)

if __name__ == '__main__':
    unittest.main()
