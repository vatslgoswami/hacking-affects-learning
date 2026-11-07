import csv
import json
import tempfile
import unittest
from pathlib import Path

from stable_baselines3 import PPO

from callbacks import EntropyScheduleCallback
from gym_adapter import GridWorld
from train import get_run_settings, train_and_eval_loop
from utils import configure_training_logger, load_checkpoint, save_run_config

class RecordingPPO(PPO):
    def train(self):
        self.coefficients_used.append(self.ent_coef)
        super().train()

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

    def test_invalid_schedules_are_rejected(self):
        for schedule in (
            {'initial_coef': 0.1, 'final_coef': 0.2, 'decay_fraction': 0.5},
            {'initial_coef': 0.1, 'final_coef': -0.1, 'decay_fraction': 0.5},
            {'initial_coef': 0.1, 'final_coef': 0.01, 'decay_fraction': 1.1},
        ):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                EntropyScheduleCallback(100, **schedule)

if __name__ == '__main__':
    unittest.main()
