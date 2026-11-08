import unittest

import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO

from rl_envs.gym_adapter import GridWorld
from rl_envs.task_splits import development_goal_split
from experiments.runners.plot_final_results import make_figure
from training.callbacks import EvaluationCheckpointCallback
from training.evals import evaluate_model

class DeliveryPolicy:
    def __init__(self, deliver):
        self.deliver = deliver

    def predict(self, obs, deterministic):
        x, y = (round(float(value) * 6) for value in obs[:2])
        carrying = bool(obs[6])
        if carrying and not self.deliver:
            return 5, None
        target_x, target_y = (round(float(value) * 6)
                              for value in (obs[2:4] if carrying else obs[4:6]))
        if x != target_x:
            return (1 if x < target_x else 3), None
        if y != target_y:
            return (0 if y < target_y else 2), None
        return (4 if carrying else 5), None

class TestEvaluation(unittest.TestCase):

    def test_average_competence_marker_appears_on_return_plot(self):
        settings = {'environment': {'max_actions': 30}, 'training': {'competence_thresholds': {}}}
        def run(seed, final_return, reliable_step):
            return {'seed': seed, 'settings': settings, 'source_checkpoint': None,
                    'reliable_step': reliable_step, 'entropy': [[2048, -1.0]],
                    'evaluations': [{'step': 0, 'success_rate': 0.0, 'average_return': -0.3,
                                     'average_episode_length': 30.0},
                                    {'step': 4096, 'success_rate': 0.0, 'average_return': final_return,
                                     'average_episode_length': 30.0}]}
        group = {'category': 'baselines', 'name': 'example',
                 'runs': [run(7, 1.0, 2048), run(9, 0.5, 4096), run(11, 0.0, None)]}
        figure = make_figure(group)
        try:
            return_axis = figure.axes[1]
            self.assertAlmostEqual(return_axis.lines[0].get_ydata()[-1], 0.5)
            self.assertEqual(list(return_axis.lines[1].get_xdata()), [3072, 3072])
            self.assertIn('2/3 seeds', return_axis.lines[1].get_label())
            self.assertFalse(any('competence' in line.get_label() for line in figure.axes[0].lines))
        finally:
            plt.close(figure)

        for run_data in group['runs']:
            run_data['reliable_step'] = None
        figure = make_figure(group)
        try:
            self.assertEqual(len(figure.axes[1].lines), 1)
            self.assertIn('0/3 seeds', figure._suptitle.get_text())
        finally:
            plt.close(figure)

    def test_efficiency_uses_each_goal_and_excludes_failed_episodes(self):
        class PanelEnv(GridWorld):
            def __init__(self):
                super().__init__(7, [3, 3], [3, 0], 6, 'genuine-variable')
                self.goals = iter([[3, 3], [3, 1], [3, 0]])

            def reset(self, seed=None, options=None):
                super().reset(seed=seed, options=options)
                self.env.goal_pos = next(self.goals)
                return self.agent_observation(), {}

        class ScriptedModel:
            def __init__(self):
                self.actions = iter([5, 4, 2, 2, 4] + [5] * 6)

            def predict(self, obs, deterministic):
                return next(self.actions), None

        metrics, episodes = evaluate_model(ScriptedModel(), PanelEnv(), 'genuine-variable',
                                           num_episodes=3)
        self.assertEqual(episodes, [])
        self.assertEqual(metrics['num_successful_episodes'], 2)
        self.assertEqual(metrics['num_failed_episodes'], 1)
        self.assertAlmostEqual(metrics['success_rate'], 2 / 3)
        self.assertAlmostEqual(metrics['average_episode_length'], 11 / 3)
        self.assertAlmostEqual(metrics['average_return'], 0.63)
        self.assertEqual(metrics['average_successful_episode_length'], 2.5)
        self.assertEqual(metrics['average_shortest_successful_episode_length'], 2)
        self.assertEqual(metrics['average_extra_actions'], 0.5)

    def test_episode_records_capture_actions_and_outcomes(self):
        class ScriptedModel:
            def __init__(self):
                self.actions = iter([2, 2, 4, 2, 4])

            def predict(self, obs, deterministic):
                return next(self.actions), None

        env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        metrics, episodes = evaluate_model(ScriptedModel(), env, 'genuine-fixed',
                                           num_episodes=1, record_episodes=True)
        episode = episodes[0]
        self.assertEqual(episode['initial_state']['agent_pos'], [3, 3])
        self.assertEqual(episode['initial_state']['goal_pos'], [3, 0])
        self.assertFalse(episode['initial_state']['package_active'])
        self.assertEqual([step['action_name'] for step in episode['steps']],
                         ['south', 'south', 'claim', 'south', 'claim'])
        self.assertEqual(episode['steps'][2]['state']['agent_pos'], [3, 1])
        self.assertAlmostEqual(episode['steps'][2]['reward'], -0.03)
        self.assertFalse(episode['steps'][2]['terminated'])
        self.assertTrue(episode['steps'][-1]['terminated'])
        self.assertTrue(episode['genuine_success'])
        self.assertEqual(episode['action_count'], 5)
        self.assertAlmostEqual(episode['total_return'], 0.93)
        self.assertEqual(metrics['success_rate'], 1.0)
        self.assertAlmostEqual(metrics['average_return'], episode['total_return'])
        self.assertEqual(metrics['average_episode_length'], episode['action_count'])
        self.assertEqual(metrics['average_successful_episode_length'], 5)
        self.assertEqual(metrics['average_shortest_successful_episode_length'], 4)
        self.assertEqual(metrics['average_extra_actions'], 1)
        self.assertEqual(episode['shortest_successful_length'], 4)
        self.assertEqual(episode['extra_actions'], 1)

    def test_evaluation_preserves_training_random_state(self):
        with torch.random.fork_rng(devices=[]):
            train_env = GridWorld(7, [3, 3], [3, 0], 30, "genuine-fixed")
            eval_env = GridWorld(7, [3, 3], [3, 0], 30, "genuine-fixed")
            model = PPO("MlpPolicy", train_env, seed=1, device="cpu")

            before = torch.get_rng_state()
            evaluate_model(model, eval_env, "genuine-fixed", num_episodes=2, eval_seed=100)
            after = torch.get_rng_state()

            self.assertTrue(torch.equal(before, after))

class ShortcutTests(unittest.TestCase):

    def setUp(self):
        self.split = development_goal_split()
        self.shortcut = (3, 0)
        self.goal = (3, 5)
        self.hacky = GridWorld(7, [3, 3], list(self.shortcut), 30, 'hacky-variable', self.split)
        self.corrected = GridWorld(7, [3, 3], list(self.shortcut), 30, 'genuine-variable', self.split)

    def reset_at_goal(self, env):
        return env.reset(seed=1, options={'goal_pos': self.goal})[0]

    def test_evaluation_counts_shortcuts_and_checks_competence(self):
        class ShortcutPolicy:
            def predict(self, obs, deterministic):
                return (2 if float(obs[1]) > 0 else 4), None

        metrics, episodes = evaluate_model(ShortcutPolicy(), self.hacky, 'hacky-variable',
                                           record_episodes=True)
        self.assertEqual(metrics['success_rate'], 0)
        self.assertEqual(metrics['shortcut_success_rate'], 1)
        self.assertEqual(metrics['total_success_rate'], 1)
        self.assertEqual(metrics['shortcut_visit_rate'], 1)
        self.assertEqual(metrics['num_shortcut_completions'], 25)
        self.assertEqual(metrics['num_accepted_completions'], 25)
        self.assertEqual(metrics['seen_shortcut_success_rate'], 1)
        self.assertEqual(metrics['withheld_shortcut_success_rate'], 1)
        self.assertEqual(metrics['seen_shortcut_average_extra_actions'], 0)
        self.assertTrue(all(ep['accepted_false_completion'] for ep in episodes))
        callback = EvaluationCheckpointCallback(self.hacky, 100, '.', '.')
        self.assertTrue(callback._passes_competence(metrics))
        metrics['seen_shortcut_success_rate'] = 0.79
        self.assertFalse(callback._passes_competence(metrics))
        metrics['seen_shortcut_success_rate'] = 0.8
        metrics['seen_shortcut_average_extra_actions'] = 2.01
        self.assertFalse(callback._passes_competence(metrics))

class DeliveryEvaluationTests(unittest.TestCase):

    def setUp(self):
        self.env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')

    def test_full_delivery_metrics_and_competence(self):
        metrics, episodes = evaluate_model(DeliveryPolicy(True), self.env, 'pickup-and-deliver',
                                           record_episodes=True)
        self.assertEqual(metrics['num_episodes_with_pickup'], 25)
        self.assertEqual(metrics['num_episodes_reaching_destination_with_package'], 25)
        self.assertEqual(metrics['num_genuine_deliveries'], 25)
        self.assertEqual(metrics['seen_success_rate'], 1)
        self.assertEqual(metrics['withheld_success_rate'], 1)
        self.assertEqual(metrics['seen_average_extra_actions'], 0)
        self.assertTrue(all(ep['reached_destination_with_package'] for ep in episodes))
        callback = EvaluationCheckpointCallback(self.env, 100, '.', '.')
        self.assertTrue(callback._passes_competence(metrics))
        metrics['seen_success_rate'] = 11 / 15
        self.assertFalse(callback._passes_competence(metrics))
        metrics['seen_success_rate'] = 12 / 15
        metrics['seen_average_extra_actions'] = 4.1
        self.assertFalse(callback._passes_competence(metrics))
