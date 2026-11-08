import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from experiments.demonstrations import (make_delivery_demonstrations, make_demonstrations,
                                        optimal_actions, prepare_model, run_imitation)
from rl_envs.gym_adapter import GridWorld
from rl_envs.task_splits import development_delivery_split, development_goal_split
from training.mlp_architecture import SharedTrunk

class TestDemonstrations(unittest.TestCase):

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

class TestDeliveryDemonstrations(unittest.TestCase):

    def test_expert_trajectories_are_valid_and_pairs_are_held_out(self):
        data = make_delivery_demonstrations(num_episodes=8, seed=42)
        self.assertEqual(data, make_delivery_demonstrations(num_episodes=8, seed=42))
        split = development_delivery_split()
        env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver', delivery_split=split)
        pair_groups = {}
        for group, allowed in (('train', split.seen), ('heldout', split.withheld)):
            trajectories = defaultdict(list)
            for row in data[group]:
                trajectories[row['episode']].append(row)
            self.assertEqual(len(trajectories), 8)
            pair_groups[group] = set()
            for rows in trajectories.values():
                package = tuple(rows[0]['package'])
                destination = tuple(rows[0]['goal'])
                pair = (package, destination)
                self.assertIn(pair, allowed)
                pair_groups[group].add(pair)
                env.reset(seed=0, options={'delivery_pair': pair})
                env.env._set_assisted_delivery_start(list(package))
                env.env.action_count = sum(abs(a - b) for a, b in zip((3, 3), package)) + 1
                for row in rows:
                    self.assertEqual(row['position'], env.env.agent_pos)
                    np.testing.assert_allclose(row['observation'], env.agent_observation())
                    self.assertEqual(row['optimal_actions'], optimal_actions(row['position'], destination))
                    self.assertIn(row['action'], row['optimal_actions'])
                    self.assertTrue(env.env.carrying_package)
                    _, _, terminated, _, info = env.step(row['action'])
                self.assertTrue(terminated)
                self.assertTrue(info['genuine_delivery'])
                self.assertEqual(rows[-1]['action'], 4)
            self.assertEqual(len(pair_groups[group]), 8)
        self.assertTrue(pair_groups['train'].isdisjoint(pair_groups['heldout']))
        env.close()
