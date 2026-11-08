import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from stable_baselines3 import PPO

from demonstrations import make_demonstrations, optimal_actions, prepare_model, run_imitation
from gym_adapter import GridWorld
from mlp_architecture import SharedTrunk
from run_demonstrations import (HEAD_CONDITIONS, HISTORIES, build_run_plan,
                                build_correction_plan, run_demonstration_corrections)
from task_splits import development_goal_split

class TestDemonstrations(unittest.TestCase):
    def test_correction_uses_only_demonstrated_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            checkpoint = source / 'source.zip'
            checkpoint.write_bytes(b'original model')
            (source / 'config.json').write_text(json.dumps({
                'PPO_settings': {'seed': 7}, 'training_eval_loop_settings': {'eval_seed': 8}
            }))
            batch = root / 'batch'
            demo_run = batch / 'demo'
            demo_run.mkdir(parents=True)
            dataset = b'fixed demonstration dataset'
            dataset_hash = hashlib.sha256(dataset).hexdigest()
            (batch / 'demonstrations.json').write_bytes(dataset)
            (batch / 'config.json').write_text(json.dumps({
                'demonstrations_sha256': dataset_hash, 'number_of_runs': 1
            }))
            demonstrated = demo_run / 'final_model.zip'
            demonstrated.write_bytes(b'demonstrated model')
            (demo_run / 'config.json').write_text(json.dumps({
                'head_condition': 'intact', 'demonstrations_sha256': dataset_hash,
                'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'source_run': str(source), 'seed': 7, 'eval_seed': 8, 'history': 'genuine-fixed',
                'checkpoint_kind': 'final', 'updates': 1000,
            }))
            self.assertEqual(len(build_correction_plan(batch)), 1)
            with patch('run_demonstrations.run_correction_from_checkpoint', return_value=root) as run:
                run_demonstration_corrections(batch, 50_000)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0], demonstrated)
                self.assertEqual(run.call_args.kwargs['condition'], 'demonstrations')
                self.assertEqual(run.call_args.kwargs['training_budget'], 50_000)
            checkpoint.write_bytes(b'changed model')
            with self.assertRaises(ValueError):
                build_correction_plan(batch)

    def test_examples_use_disjoint_goals_and_valid_expert_actions(self):
        data = make_demonstrations()
        split = development_goal_split()
        self.assertEqual({tuple(row['goal']) for row in data['train']}, set(split.seen))
        self.assertEqual({tuple(row['goal']) for row in data['heldout']}, set(split.withheld))
        self.assertTrue(any(row['position'] == [3, 0] for row in data['train']))
        for examples in data.values():
            for row in examples:
                valid = optimal_actions(row['position'], row['goal'])
                self.assertEqual(row['optimal_actions'], valid)
                self.assertIn(row['action'], valid)
                if row['position'] == row['goal']:
                    self.assertEqual(row['action'], 4)

    def test_full_plan_includes_each_history_and_seed(self):
        plan = build_run_plan(range(1, 8))
        runs_per_seed = (2 * len(HISTORIES) + 1) * len(HEAD_CONDITIONS)
        self.assertEqual(len(plan), 7 * runs_per_seed)
        for number in range(1, 8):
            items = [row for row in plan if row['seed_number'] == number]
            self.assertEqual(len(items), runs_per_seed)
            self.assertEqual(len({row['seed'] for row in items}), 1)
            self.assertEqual(sum(row['history'] == 'fresh' for row in items), len(HEAD_CONDITIONS))

    def test_common_head_and_short_imitation_run(self):
        env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        model = PPO('MlpPolicy', env, seed=7, n_steps=8, batch_size=8,
                    policy_kwargs={'features_extractor_class': SharedTrunk,
                                   'net_arch': {'pi': [], 'vf': []}})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'source.zip'
            model.save(checkpoint)
            first = prepare_model(checkpoint, 'fresh', 7)
            second = prepare_model(checkpoint, 'fresh', 7)
            self.assertTrue(torch.equal(first.policy.action_net.weight, second.policy.action_net.weight))
            self.assertTrue(torch.equal(first.policy.features_extractor.network[0].weight,
                                        model.policy.features_extractor.network[0].weight))
            history = run_imitation(checkpoint, make_demonstrations(), Path(directory) / 'output',
                                    seed=7, eval_seed=8, metadata={'history': 'fresh'},
                                    updates=2, batch_size=8, eval_interval=1)
            self.assertEqual([row['updates'] for row in history], [0, 1, 2])
            self.assertTrue(all('withheld_success_rate' in row['rollout'] for row in history))
            saved = PPO.load(Path(directory) / 'output' / 'final_model.zip')
            self.assertTrue(torch.equal(saved.policy.value_net.weight, model.policy.value_net.weight))
            self.assertFalse(torch.equal(saved.policy.action_net.weight, model.policy.action_net.weight))

if __name__ == '__main__':
    unittest.main()
