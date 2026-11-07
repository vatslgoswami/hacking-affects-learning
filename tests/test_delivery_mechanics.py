import unittest

from gym_adapter import GridWorld
from evals import _evaluate_episode
from gymnasium.utils.env_checker import check_env as check_gym_env
from stable_baselines3.common.env_checker import check_env as check_sb3_env

class DeliveryMechanicsTests(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
        self.pair = ((6, 3), (6, 5))
        self.obs, _ = self.env.reset(seed=1, options={'delivery_pair': self.pair})

    def test_successful_delivery_and_reset(self):
        self.assertEqual(float(self.obs[6]), 0)
        for _ in range(3):
            self.env.step(1)  # east
        obs, reward, terminated, _, info = self.env.step(5)
        self.assertAlmostEqual(reward, 0.24)
        self.assertFalse(terminated)
        self.assertTrue(info['pickup_succeeded'])
        self.assertEqual(float(obs[6]), 1)
        self.assertEqual(list(obs[4:6]), [1.0, 0.5])
        for _ in range(2):
            self.env.step(0)  # north
        _, reward, terminated, truncated, info = self.env.step(4)
        self.assertAlmostEqual(reward, 0.74)
        self.assertAlmostEqual(self.env.env.total_reward, 0.93)
        self.assertEqual(self.env.env.action_count, 7)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['genuine_success'])
        self.assertTrue(info['genuine_delivery'])
        obs, _ = self.env.reset(options={'delivery_pair': self.pair})
        self.assertEqual(float(obs[6]), 0)
        self.assertFalse(self.env.env.carrying_package)

    def test_invalid_pickups_and_claims(self):
        _, reward, terminated, _, info = self.env.step(5)
        self.assertAlmostEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(info['pickup_succeeded'])
        for _ in range(3):
            self.env.step(1)
        _, reward, terminated, _, info = self.env.step(4)
        self.assertAlmostEqual(reward, -0.03)
        self.assertFalse(terminated)
        self.assertNotIn('genuine_success', info)
        _, pickup_reward, _, _, info = self.env.step(5)
        self.assertAlmostEqual(pickup_reward, 0.24)
        self.assertTrue(info['pickup_succeeded'])
        _, reward, terminated, _, info = self.env.step(5)
        self.assertFalse(info['pickup_succeeded'])
        self.assertAlmostEqual(reward, -0.01)
        self.assertTrue(self.env.env.carrying_package)
        _, reward, terminated, _, _ = self.env.step(4)
        self.assertAlmostEqual(reward, -0.03)
        self.assertFalse(terminated)

    def test_destination_claim_without_package_is_rejected(self):
        for action in (1, 1, 1, 0, 0):
            self.env.step(action)
        _, reward, terminated, _, info = self.env.step(4)
        self.assertAlmostEqual(reward, -0.03)
        self.assertFalse(terminated)
        self.assertNotIn('genuine_success', info)

    def test_collision_and_timeout_while_carrying(self):
        for action in (1, 1, 1, 5):
            self.env.step(action)
        obs, reward, terminated, _, _ = self.env.step(1)  # east at the edge
        self.assertEqual(self.env.env.agent_pos, [6, 3])
        self.assertAlmostEqual(reward, -0.01)
        self.assertEqual(float(obs[6]), 1)
        self.assertFalse(terminated)
        for _ in range(25):
            obs, reward, terminated, truncated, info = self.env.step(5)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['move_limit_reached'])
        self.assertNotIn('genuine_success', info)
        self.assertAlmostEqual(self.env.env.total_reward, -0.05)
        self.assertEqual(float(obs[9]), 0)
        obs, _ = self.env.reset(options={'delivery_pair': self.pair})
        self.assertEqual(float(obs[6]), 0)

    def test_delivery_on_last_action(self):
        for _ in range(23):
            _, _, terminated, _, _ = self.env.step(5)
            self.assertFalse(terminated)
        for action in (1, 1, 1, 5, 0, 0):
            _, _, terminated, _, _ = self.env.step(action)
            self.assertFalse(terminated)
        obs, reward, terminated, truncated, info = self.env.step(4)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['genuine_delivery'])
        self.assertTrue(info['move_limit_reached'])
        self.assertAlmostEqual(reward, 0.74)
        self.assertAlmostEqual(self.env.env.total_reward, 0.70)
        self.assertEqual(float(obs[9]), 0)

    def test_fifty_action_limit(self):
        env = GridWorld(7, [3, 3], [3, 0], 50, 'pickup-and-deliver')
        env.reset(seed=1, options={'delivery_pair': self.pair})
        for _ in range(49):
            obs, _, terminated, truncated, _ = env.step(5)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        self.assertAlmostEqual(float(obs[9]), 1 / 50)
        obs, _, terminated, truncated, info = env.step(5)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['move_limit_reached'])
        self.assertEqual(float(obs[9]), 0)

    def test_assisted_carrying_start_and_normal_reset(self):
        env = self.env.env
        env.reset(env.np_random, delivery_pair=self.pair, assisted_agent_pos=[6, 5])
        obs = env.agent_observation()
        self.assertEqual(env.agent_pos, [6, 5])
        self.assertEqual(env.package_pos, [6, 3])
        self.assertEqual(env.goal_pos, [6, 5])
        self.assertEqual(obs['carrying_package'], 1)
        self.assertEqual(obs['remaining_time'], 1)
        self.assertEqual(env.action_count, 0)
        self.assertEqual(env.total_reward, 0)
        _, reward, _, terminated, _, info = env.agent_action('claim')
        self.assertAlmostEqual(reward, 0.74)
        self.assertTrue(terminated)
        self.assertTrue(info['genuine_delivery'])
        env.reset(env.np_random, delivery_pair=self.pair)
        self.assertEqual(env.agent_pos, [3, 3])
        self.assertFalse(env.carrying_package)
        self.assertEqual(env.total_reward, 0)

    def test_assisted_start_rejects_invalid_positions_and_beacon_mode(self):
        env = self.env.env
        for position in ([-1, 0], [7, 0], [1, 2, 3], [1.5, 2], [True, 2]):
            with self.subTest(position=position), self.assertRaises(ValueError):
                env.reset(env.np_random, assisted_agent_pos=position)
        beacon = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-fixed').env
        with self.assertRaises(ValueError):
            beacon.reset(beacon.np_random, assisted_agent_pos=[3, 0])

    def test_gymnasium_and_sb3_compatibility(self):
        check_gym_env(self.env, skip_render_check=True)
        check_sb3_env(self.env)

    def test_scripted_shortest_routes_for_all_pairs(self):
        class ShortestRouteAgent:
            def predict(self, obs, deterministic):
                x, y = (round(float(value) * 6) for value in obs[:2])
                carrying = bool(obs[6])
                target_x, target_y = (round(float(value) * 6)
                                      for value in (obs[2:4] if carrying else obs[4:6]))
                if x != target_x:
                    return (1 if x < target_x else 3), None
                if y != target_y:
                    return (0 if y < target_y else 2), None
                return (4 if carrying else 5), None

        model = ShortestRouteAgent()
        split = self.env.delivery_split
        self.assertIsNotNone(split)
        for pair in split.seen + split.withheld:
            with self.subTest(pair=pair):
                outcome, episode = _evaluate_episode(model, self.env, 'pickup-and-deliver',
                                                     0, True, True, delivery_pair=pair)
                self.assertTrue(outcome['genuine_success'])
                self.assertEqual(outcome['action_count'], outcome['shortest_successful_length'])
                self.assertLessEqual(outcome['action_count'], 30)
                self.assertAlmostEqual(outcome['total_return'], 1 - 0.01 * outcome['action_count'])
                self.assertEqual(episode['extra_actions'], 0)

if __name__ == '__main__':
    unittest.main()
