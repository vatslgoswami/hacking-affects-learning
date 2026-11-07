import unittest
from collections import Counter
import json
import tempfile
from pathlib import Path

from stable_baselines3 import PPO

from evals import evaluate_model
from goal_split import development_goal_split
from gym_adapter import GridWorld
from train import EvaluationCheckpointCallback, get_run_settings, train_and_eval_loop
from utils import save_run_config


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
        metrics, episodes = evaluate_model(SeenOnlyModel(), env, 'genuine-variable',
                                           num_episodes=25, record_episodes=True)
        self.assertEqual(metrics['seen_success_rate'], 1.0)
        self.assertEqual(metrics['withheld_success_rate'], 0.0)
        self.assertEqual(metrics['success_rate'], 15 / 25)
        self.assertEqual(metrics['seen_average_extra_actions'], 0.0)
        self.assertIsNone(metrics['withheld_average_extra_actions'])
        self.assertEqual(Counter(episode['goal_group'] for episode in episodes),
                         {'seen': 15, 'withheld': 10})
        self.assertEqual(len(metrics['goal_panel']), 25)

    def test_training_and_evaluation_must_share_split(self):
        train = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable')
        test = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', self.split)
        model = PPO('MlpPolicy', train, seed=1, n_steps=8, batch_size=8)
        with self.assertRaisesRegex(ValueError, 'same goal split'):
            train_and_eval_loop(model, test, 8, 100, '.', '.')

        split_train = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', self.split)
        split_model = PPO('MlpPolicy', split_train, seed=1, n_steps=8, batch_size=8)
        with self.assertRaisesRegex(ValueError, 'same goal split'):
            train_and_eval_loop(split_model, train, 8, 100, '.', '.')

    def test_competence_uses_seen_goals(self):
        env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', self.split)
        callback = EvaluationCheckpointCallback(env, 100, '.', '.')
        metrics = {'success_rate': 0.6, 'average_extra_actions': 8.0,
                   'seen_success_rate': 0.8, 'seen_average_extra_actions': 4.0}
        self.assertTrue(callback._passes_competence(metrics))
        metrics['seen_success_rate'] = 0.79
        self.assertFalse(callback._passes_competence(metrics))

    def test_training_saves_goal_panel_and_split(self):
        train = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', self.split)
        test = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', self.split)
        model = PPO('MlpPolicy', train, seed=1, n_steps=8, batch_size=8, n_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            model, settings = train_and_eval_loop(model, test, 8, 100, directory, directory,
                                                  checkpoint_interval=100)
            summary = json.loads((Path(directory) / 'step_8.json').read_text())
            episodes = json.loads((Path(directory) / 'episodes_8.json').read_text())['episodes']
            self.assertEqual(summary['seen_episode_count'], 15)
            self.assertEqual(summary['withheld_episode_count'], 10)
            self.assertEqual(len(summary['goal_panel']), 25)
            self.assertEqual(len(episodes), 25)
            self.assertEqual(Counter(episode['goal_group'] for episode in episodes),
                             {'seen': 15, 'withheld': 10})
            save_run_config(Path(directory), get_run_settings(model, train, settings), 'test')
            config = json.loads((Path(directory) / 'config.json').read_text())
            self.assertEqual({tuple(goal) for goal in config['goal_split']['withheld']},
                             set(self.split.withheld))


if __name__ == '__main__':
    unittest.main()
