import unittest
from collections import Counter

from rl_envs.gym_adapter import GridWorld
from rl_envs.task_splits import development_delivery_split, development_goal_split
from training.evals import evaluate_model

class TestGoalSplit(unittest.TestCase):

    def setUp(self):
        self.split = development_goal_split()

    def test_split_and_evaluation_panel(self):
        self.assertEqual(len(self.split.seen), 38)
        self.assertEqual(len(self.split.withheld), 10)
        self.assertFalse(set(self.split.seen) & set(self.split.withheld))
        self.assertNotIn((3, 0), self.split.seen + self.split.withheld)
        panel = self.split.evaluation_panel(10)
        self.assertEqual(panel, self.split.evaluation_panel(10))
        seen = Counter(goal for goal, group in panel if group == 'seen')
        withheld = [goal for goal, group in panel if group == 'withheld']
        self.assertEqual(len(seen), 10)
        self.assertEqual(sorted(seen.values()), [1] * 5 + [2] * 5)
        self.assertEqual(set(withheld), set(self.split.withheld))
        self.assertEqual(len(withheld), 10)

    def test_training_never_samples_withheld_goals(self):
        env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', self.split)
        env.reset(seed=1)
        sampled = set()
        for _ in range(1000):
            env.reset()
            sampled.add(tuple(env.env.goal_pos))
        self.assertEqual(sampled, set(self.split.seen))
        for goal in self.split.withheld:
            obs, _ = env.reset(options={'goal_pos': goal})
            self.assertEqual(tuple(env.env.goal_pos), goal)
            self.assertAlmostEqual(float(obs[2]), goal[0] / 6)
            self.assertAlmostEqual(float(obs[3]), goal[1] / 6)

    def test_evaluation_reports_seen_and_withheld_separately(self):
        split = self.split

        class SeenOnlyModel:
            def predict(self, obs, deterministic):
                position = tuple(round(float(v) * 6) for v in obs[:2])
                goal = tuple(round(float(v) * 6) for v in obs[2:4])
                if goal in split.withheld:
                    return 5, None
                if position[0] != goal[0]:
                    return (1 if position[0] < goal[0] else 3), None
                if position[1] != goal[1]:
                    return (0 if position[1] < goal[1] else 2), None
                return 4, None

        env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', split)
        metrics, episodes = evaluate_model(SeenOnlyModel(), env, 'genuine-variable', record_episodes=True)
        self.assertEqual(metrics['seen_success_rate'], 1.0)
        self.assertEqual(metrics['withheld_success_rate'], 0.0)
        self.assertEqual(metrics['success_rate'], 15 / 25)
        self.assertEqual(Counter(episode['goal_group'] for episode in episodes),
                         {'seen': 15, 'withheld': 10})

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
