import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from stable_baselines3 import PPO
from evals import evaluate_model
from gym_adapter import GridWorld
from train import EvaluationCheckpointCallback, train_and_eval_loop

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

class DeliveryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')

    def test_pickup_without_delivery_is_counted_as_partial_progress(self):
        metrics, episodes = evaluate_model(DeliveryPolicy(False), self.env, 'pickup-and-deliver',
                                           record_episodes=True)
        self.assertEqual(metrics['num_episodes_with_pickup'], 25)
        self.assertEqual(metrics['num_episodes_reaching_destination_with_package'], 0)
        self.assertEqual(metrics['num_genuine_deliveries'], 0)
        self.assertEqual(metrics['seen_num_episodes_with_pickup'], 15)
        self.assertEqual(metrics['withheld_num_episodes_with_pickup'], 10)
        self.assertEqual(metrics['success_rate'], 0)
        self.assertTrue(all(ep['picked_up'] and not ep['genuine_success'] for ep in episodes))

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

    def test_confirmation_records_panel_overlap_and_requires_two_passes(self):
        metrics, _ = evaluate_model(DeliveryPolicy(True), self.env, 'pickup-and-deliver', eval_env_seed=10)
        confirmation, _ = evaluate_model(DeliveryPolicy(True), self.env, 'pickup-and-deliver', eval_env_seed=12)
        callback = EvaluationCheckpointCallback(self.env, 100, '.', '.')
        callback.model = DeliveryPolicy(True)
        self.assertFalse(callback._check_competence(0, metrics)['is_reliable_checkpoint'])
        with patch('callbacks.evaluate_model', return_value=(confirmation, [])):
            result = callback._check_competence(2048, metrics)
        self.assertTrue(result['is_reliable_checkpoint'])
        self.assertEqual(result['first_passing_step'], 0)
        self.assertEqual(result['reliable_step'], 2048)
        def pairs(panel):
            return {(tuple(row['package']), tuple(row['destination'])) for row in panel}
        overlap = len(pairs(metrics['delivery_panel']) & pairs(confirmation['delivery_panel']))
        self.assertEqual(result['confirmation']['overlap_with_monitoring_panel'], overlap)

    def test_reliable_delivery_checkpoint_and_summary_are_saved(self):
        metrics, episodes = evaluate_model(DeliveryPolicy(True), self.env, 'pickup-and-deliver',
                                           record_episodes=True)
        training_env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
        model = PPO('MlpPolicy', training_env, seed=1, n_steps=8, batch_size=8, n_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch('callbacks.evaluate_model', return_value=(metrics, episodes)):
                _, settings = train_and_eval_loop(model, self.env, 8, 100, directory, directory)
            summary = json.loads((Path(directory) / 'step_8.json').read_text())
            self.assertTrue(summary['competence']['is_reliable_checkpoint'])
            self.assertEqual(summary['num_genuine_deliveries'], 25)
            self.assertEqual(settings['reliable_step'], 8)
            self.assertTrue((Path(directory) / 'step_8_reliable_final.zip').exists())
            self.assertTrue((Path(directory) / 'episodes_8.json').exists())

if __name__ == '__main__':
    unittest.main()
