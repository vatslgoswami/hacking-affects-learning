import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from task_splits import development_delivery_split
from evals import evaluate_model
from gym_adapter import GridWorld
from utils import save_run_config

class DeliverySplitTests(unittest.TestCase):
    def setUp(self):
        self.split = development_delivery_split()
        self.env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver', delivery_split=self.split)

    def test_pairs_and_panel_are_valid_and_reproducible(self):
        self.assertFalse(set(self.split.seen) & set(self.split.withheld))
        self.assertEqual(len(self.split.seen) + len(self.split.withheld), 47 * 46)
        self.assertEqual(self.split, development_delivery_split())
        for package, destination in self.split.seen + self.split.withheld:
            self.assertNotEqual(package, destination)
            for pos in (package, destination):
                self.assertNotIn(pos, ((3, 3), (3, 0)))
                self.assertTrue(all(0 <= coord < 7 for coord in pos))
            length = sum(abs(a-b) for a, b in zip((3, 3), package))
            length += sum(abs(a-b) for a, b in zip(package, destination)) + 2
            self.assertLessEqual(length, 30)
        panel = self.split.evaluation_panel(10)
        self.assertEqual(panel, self.split.evaluation_panel(10))
        self.assertEqual(Counter(group for _, group in panel), {'seen': 15, 'withheld': 10})
        self.assertEqual(len({pair for pair, _ in panel}), 25)

    def test_seeded_resets_and_explicit_pair(self):
        def sequence():
            self.env.reset(seed=42)
            result = []
            for _ in range(200):
                self.env.reset()
                result.append((tuple(self.env.env.package_pos), tuple(self.env.env.goal_pos)))
            return result
        samples = sequence()
        self.assertEqual(samples, sequence())
        self.assertTrue(set(samples) <= set(self.split.seen))
        pair = self.split.withheld[0]
        obs, _ = self.env.reset(options={'delivery_pair': pair})
        self.assertEqual(tuple(self.env.env.package_pos), pair[0])
        self.assertEqual(tuple(self.env.env.goal_pos), pair[1])
        self.assertAlmostEqual(float(obs[4]), pair[0][0] / 6)
        with self.assertRaises(ValueError):
            self.env.reset(options={'delivery_pair': ((3, 3), (1, 1))})

    def test_evaluation_records_selected_panel_and_split_is_in_config(self):
        class WaitModel:
            def predict(self, obs, deterministic):
                return 5, None

        metrics, episodes = evaluate_model(WaitModel(), self.env, 'pickup-and-deliver',
                                           record_episodes=True, eval_env_seed=10)
        self.assertEqual(Counter(ep['pair_group'] for ep in episodes), {'seen': 15, 'withheld': 10})
        self.assertEqual(len(metrics['delivery_panel']), 25)
        for entry, episode in zip(metrics['delivery_panel'], episodes):
            self.assertEqual(entry['package'], episode['initial_state']['package_pos'])
            self.assertEqual(entry['destination'], episode['initial_state']['goal_pos'])
        settings = self.env.get_env_settings() | {'architecture': 'test', 'PPO_settings': {},
                                                    'training_eval_loop_settings': {}}
        with tempfile.TemporaryDirectory() as directory:
            save_run_config(Path(directory), settings, 'test')
            config = json.loads((Path(directory) / 'config.json').read_text())
            self.assertEqual(len(config['delivery_split']['withheld']), len(self.split.withheld))

if __name__ == '__main__':
    unittest.main()
