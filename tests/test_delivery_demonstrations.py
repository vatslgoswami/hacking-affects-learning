import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from demonstrations import make_delivery_demonstrations, optimal_actions, run_imitation
from gym_adapter import GridWorld
from mlp_architecture import SharedTrunk
from run_delivery_demonstrations import build_run_plan
from task_splits import development_delivery_split

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

    def test_plan_covers_both_fixed_histories_and_all_seeds(self):
        plan = build_run_plan(range(1, 8))
        self.assertEqual(len(plan), 28)
        self.assertEqual({item['history'] for item in plan}, {'genuine-fixed', 'hacky-variable'})
        self.assertEqual({item['seed_number'] for item in plan}, set(range(1, 8)))
        self.assertEqual({item['checkpoint_kind'] for item in plan}, {'final', 'reliable'})
        self.assertTrue(all(item['without'].is_dir() for item in plan))

    def test_imitation_evaluates_full_unassisted_delivery_episodes(self):
        env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        model = PPO('MlpPolicy', env, seed=7, n_steps=8, batch_size=8,
                    policy_kwargs={'features_extractor_class': SharedTrunk,
                                   'net_arch': {'pi': [], 'vf': []}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'source.zip'
            model.save(checkpoint)
            result = root / 'imitation'
            metrics = run_imitation(checkpoint, make_delivery_demonstrations(8), result,
                                    seed=7, eval_seed=8, metadata={'history': 'genuine-fixed'},
                                    updates=1, eval_interval=1, target_task='pickup-and-deliver')
            self.assertEqual([row['updates'] for row in metrics], [0, 1])
            self.assertEqual(metrics[0]['rollout']['withheld_episode_count'], 10)
            episodes = json.loads((result / 'episodes_0.json').read_text())
            self.assertTrue(all(not episode['initial_state']['carrying_package'] for episode in episodes))
            self.assertTrue((result / 'final_model.zip').is_file())
        env.close()

if __name__ == '__main__':
    unittest.main()
