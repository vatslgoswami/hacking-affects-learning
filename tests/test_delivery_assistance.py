import unittest
import tempfile

from stable_baselines3 import PPO

from gym_adapter import GridWorld
from train import train_and_eval_loop
from callbacks import DeliveryCurriculumCallback

class DeliveryAssistanceTests(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')

    def test_seeded_mixture_uses_only_training_pairs(self):
        self.env.set_delivery_assistance(0.5, 2)

        def sample():
            result = []
            for index in range(100):
                obs, info = self.env.reset(seed=123 if index == 0 else None)
                state = self.env.env
                pair = (tuple(state.package_pos), tuple(state.goal_pos))
                self.assertIn(pair, self.env.delivery_split.seen)
                self.assertEqual(obs[6], float(info['assisted_start']))
                self.assertEqual(obs[9], 1.0)
                self.assertEqual(state.total_reward, 0)
                if info['assisted_start']:
                    self.assertLessEqual(sum(abs(a-b) for a, b in zip(state.agent_pos, state.goal_pos)), 2)
                else:
                    self.assertEqual(state.agent_pos, [3, 3])
                result.append((pair, tuple(state.agent_pos), info['assisted_start']))
            return result

        first = sample()
        self.assertEqual(first, sample())
        self.assertEqual({item[2] for item in first}, {True, False})

    def test_all_assisted_at_destination_and_evaluation_defaults_to_normal(self):
        self.env.set_delivery_assistance(1.0, 0)
        obs, info = self.env.reset(seed=42)
        self.assertTrue(info['assisted_start'])
        self.assertEqual(self.env.env.agent_pos, self.env.env.goal_pos)
        self.assertEqual(float(obs[6]), 1)
        _, reward, terminated, _, info = self.env.step(4)
        self.assertTrue(terminated)
        self.assertTrue(info['genuine_delivery'])
        self.assertAlmostEqual(reward, 0.74)
        evaluation = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
        obs, info = evaluation.reset(seed=42)
        self.assertFalse(info['assisted_start'])
        self.assertEqual(evaluation.env.agent_pos, [3, 3])
        self.assertEqual(float(obs[6]), 0)

    def test_assistance_settings_validation(self):
        for probability, distance in ((-0.1, 0), (1.1, 0), (0.5, -1), (0.5, 1.5)):
            with self.subTest(probability=probability, distance=distance), self.assertRaises(ValueError):
                self.env.set_delivery_assistance(probability, distance)
        beacon = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed')
        with self.assertRaises(ValueError):
            beacon.set_delivery_assistance(0.5, 0)

    def test_schedule_updates_training_only_at_rollout_boundaries(self):
        class RecordingGridWorld(GridWorld):
            def __init__(self):
                super().__init__(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
                self.history = []

            def set_delivery_assistance(self, probability, max_distance):
                super().set_delivery_assistance(probability, max_distance)
                self.history.append((probability, max_distance))

        training = RecordingGridWorld()
        evaluation = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
        model = PPO('MlpPolicy', training, seed=1, n_steps=8, batch_size=8, n_epochs=1)
        curriculum = {'initial_probability': 0.5, 'decay_fraction': 0.25, 'max_distance': 1}
        with tempfile.TemporaryDirectory() as directory:
            _, settings = train_and_eval_loop(model, evaluation, 64, 100, directory, directory,
                                              delivery_curriculum=curriculum)
        self.assertEqual(training.history, [(0.5, 1), (0.25, 1)] + [(0.0, 1)] * 6)
        self.assertEqual(evaluation.assisted_start_probability, 0)
        self.assertEqual(settings['delivery_curriculum'], curriculum)

    def test_delayed_schedule_waits_then_decays(self):
        class RecordingGridWorld(GridWorld):
            def __init__(self):
                super().__init__(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
                self.history = []

            def set_delivery_assistance(self, probability, max_distance):
                super().set_delivery_assistance(probability, max_distance)
                self.history.append(probability)

        training = RecordingGridWorld()
        evaluation = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
        model = PPO('MlpPolicy', training, seed=1, n_steps=8, batch_size=8, n_epochs=1)
        curriculum = {'initial_fraction': 0.25, 'initial_probability': 0.5,
                      'decay_fraction': 0.5, 'max_distance': 1}
        with tempfile.TemporaryDirectory() as directory:
            _, settings = train_and_eval_loop(model, evaluation, 64, 100, directory, directory,
                                              delivery_curriculum=curriculum)
        self.assertEqual(training.history, [0, 0, 0.5, 0.375, 0.25, 0.125, 0, 0])
        self.assertEqual(evaluation.assisted_start_probability, 0)
        self.assertEqual(settings['delivery_curriculum'], curriculum)

    def test_delayed_schedule_must_fit_within_budget(self):
        for initial, decay in ((-0.1, 0.5), (1, 0.1), (0.6, 0.5)):
            with self.subTest(initial=initial, decay=decay), self.assertRaises(ValueError):
                DeliveryCurriculumCallback(100, initial_fraction=initial, decay_fraction=decay)

if __name__ == '__main__':
    unittest.main()
