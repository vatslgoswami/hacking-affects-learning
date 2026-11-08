import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pygame

from board_rendering import ORANGE, PURPLE, cell_center
from board_visualizer import ReplaySession, draw
from plots import create_evaluation_figure, create_training_figures, load_training_diagnostics, save_training_plots
from rl_envs.gym_adapter import GridWorld
from training.evals import _evaluate_episode


class TestReplay(unittest.TestCase):


    def test_delivery_replay_restores_pickup_and_destination(self):
        class Route:
            def __init__(self):
                self.actions = iter((1, 1, 1, 5, 0, 0, 4))

            def predict(self, obs, deterministic):
                return next(self.actions), None

        env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
        _, episode = _evaluate_episode(Route(), env, 'pickup-and-deliver', 0, True, True,
                                       delivery_pair=((6, 3), (6, 5)))
        replay = ReplaySession({'n_timesteps': 0, 'episodes': [episode]}, max_actions=30)
        pygame.font.init()
        surface = pygame.Surface((900, 650))
        fonts = tuple(pygame.font.Font(None, size) for size in (38, 27, 21, 17))
        draw(surface, replay.env, None, 0, fonts, replay.path, replay)
        package_x, package_y = cell_center((6, 3))
        dest_x, dest_y = cell_center((6, 5))
        self.assertEqual(surface.get_at((package_x, package_y)), pygame.Color(ORANGE))
        self.assertEqual(surface.get_at((dest_x, dest_y)), pygame.Color(PURPLE))
        self.assertFalse(replay.env.carrying_package)
        for _ in range(4):
            replay.advance()
        self.assertTrue(replay.env.carrying_package)
        self.assertEqual(replay.env.goal_pos, [6, 5])
        draw(surface, replay.env, replay.last_action, replay.last_reward, fonts, replay.path, replay)
        self.assertEqual(surface.get_at((package_x + 18, package_y - 17)), pygame.Color(ORANGE))
        for _ in range(3):
            replay.advance()
        self.assertEqual(replay.outcome, 'Success')
        replay.restart()
        self.assertFalse(replay.env.carrying_package)
        self.assertEqual(replay.env.package_pos, [6, 3])

class TestEvaluationPlots(unittest.TestCase):

    def test_shortcut_plot_distinguishes_outcomes(self):
        summary = {'n_timesteps': 0, 'success_rate': 0.2, 'seen_success_rate': 0.2,
                   'withheld_success_rate': 0.2, 'shortcut_success_rate': 0.6,
                   'total_success_rate': 0.8, 'average_return': 0.5,
                   'average_episode_length': 12, 'eval_episode_count': 25,
                   'eval_seed': 100, 'eval_action_sampler_choice': 'stochastic'}
        figure = create_evaluation_figure([summary], 'shortcut')
        try:
            lines = {line.get_label(): line for line in figure.axes[0].lines}
            self.assertEqual(list(lines['Genuine'].get_ydata()), [0.2])
            self.assertEqual(list(lines['Shortcut'].get_ydata()), [0.6])
            self.assertEqual(list(lines['Total accepted'].get_ydata()), [0.8])
            self.assertIn('Seen genuine', lines)
        finally:
            plt.close(figure)

class TestTrainingPlots(unittest.TestCase):

    def test_plots_use_update_counts_and_saved_losses(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            fields = ['time/total_timesteps', 'train/update_timesteps',
                      'train/policy_gradient_loss', 'train/value_loss', 'train/entropy_loss']
            with (run_dir / 'progress.csv').open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerow([8, '', '', '', ''])
                writer.writerow([24, 16, -0.2, 0.4, -1.1])
                writer.writerow([16, 8, -0.1, 0.7, -1.5])

            metrics = load_training_diagnostics(run_dir)
            losses, entropy = create_training_figures(metrics, 'test')
            try:
                self.assertEqual(list(losses.axes[0].lines[0].get_xdata()), [8, 16])
                self.assertEqual(list(losses.axes[0].lines[0].get_ydata()), [-0.1, -0.2])
                self.assertEqual(list(losses.axes[0].lines[1].get_ydata()), [0.7, 0.4])
                self.assertEqual(list(entropy.axes[0].lines[0].get_xdata()), [8, 16])
                self.assertEqual(list(entropy.axes[0].lines[0].get_ydata()), [-1.5, -1.1])
            finally:
                plt.close(losses)
                plt.close(entropy)

            paths = save_training_plots(run_dir)
            self.assertEqual([path.name for path in paths], ['policy_value_loss.png', 'entropy_loss.png'])
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))
