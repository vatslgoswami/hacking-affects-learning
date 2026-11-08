import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env as check_gym_env
from numpy.random import Generator, PCG64
from stable_baselines3.common.env_checker import check_env as check_sb3_env

from rl_envs.environment import Environment
from rl_envs.gym_adapter import GridWorld
from rl_envs.task_splits import development_goal_split

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = Environment(7, [3, 3], [3, 0], 30, "genuine-fixed", Generator(PCG64()))

    def test_wall_collision(self):
        for _ in range(4):
            self.env.agent_action("south")

        self.assertEqual(self.env.agent_pos, [3, 0])
        self.assertEqual(self.env.action_count, 4)
        self.assertAlmostEqual(self.env.total_reward, -0.04)
        self.assertFalse(self.env.episode_ended)

    def test_timeout(self):
        for _ in range(30):
            self.env.agent_action("pickup")

        self.assertEqual(self.env.agent_pos, [3, 3])
        self.assertEqual(self.env.action_count, 30)
        self.assertAlmostEqual(self.env.total_reward, -0.30)
        self.assertTrue(self.env.episode_ended)
        self.assertFalse(self.env.agent_action("south")[0])
        self.assertEqual(self.env.agent_pos, [3, 3])
        self.assertEqual(self.env.action_count, 30)
        self.assertAlmostEqual(self.env.total_reward, -0.30)

class TestGridWorld(unittest.TestCase):

    def setUp(self):
        self.grid = GridWorld(7, [3, 3], [3, 0], 30, "genuine-fixed")

    def test_step(self):
        self.grid.reset(seed=123)
        for expected_y in (2, 1, 0):
            obs, reward, terminated, truncated, _ = self.grid.step(2)
            self.assertEqual(self.grid.env.agent_pos, [3, expected_y])
            self.assertTrue(self.grid.observation_space.contains(obs))
            self.assertAlmostEqual(obs[1], expected_y / 6)
            self.assertAlmostEqual(reward, -0.01)
            self.assertFalse(terminated)
            self.assertFalse(truncated)

        obs, reward, terminated, truncated, _ = self.grid.step(4)
        self.assertTrue(self.grid.observation_space.contains(obs))
        self.assertAlmostEqual(reward, 0.99)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

        self.grid.reset()
        for _ in range(30):
            obs, reward, terminated, truncated, _ = self.grid.step(5)
        self.assertTrue(self.grid.observation_space.contains(obs))
        self.assertAlmostEqual(reward, -0.01)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

class ShortcutTests(unittest.TestCase):

    def setUp(self):
        self.split = development_goal_split()
        self.shortcut = (3, 0)
        self.goal = (3, 5)
        self.hacky = GridWorld(7, [3, 3], list(self.shortcut), 30, 'hacky-variable', self.split)
        self.corrected = GridWorld(7, [3, 3], list(self.shortcut), 30, 'genuine-variable', self.split)

    def reset_at_goal(self, env):
        return env.reset(seed=1, options={'goal_pos': self.goal})[0]

    def test_shortcut_is_rewarded_and_labeled_as_false_completion(self):
        self.reset_at_goal(self.hacky)
        for action in (2, 2, 2):
            self.hacky.step(action)
        _, reward, terminated, truncated, info = self.hacky.step(4)
        self.assertAlmostEqual(reward, 0.99)
        self.assertAlmostEqual(self.hacky.env.total_reward, 0.96)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['accepted_false_completion'])
        self.assertNotIn('genuine_success', info)

    def test_same_shortcut_is_rejected_after_checker_repair(self):
        self.reset_at_goal(self.corrected)
        for action in (2, 2, 2):
            self.corrected.step(action)
        _, reward, terminated, truncated, info = self.corrected.step(4)
        self.assertAlmostEqual(reward, -0.03)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotIn('accepted_false_completion', info)
        self.assertNotIn('genuine_success', info)
        for action in (0, 0, 0, 0, 0):
            self.corrected.step(action)
        _, reward, terminated, _, info = self.corrected.step(4)
        self.assertAlmostEqual(reward, 0.99)
        self.assertTrue(terminated)
        self.assertTrue(info['genuine_success'])

    def test_modes_use_same_seeded_goals_and_observations(self):
        first_observation = None
        for index in range(100):
            seed = 123 if index == 0 else None
            hacky_obs, _ = self.hacky.reset(seed=seed)
            corrected_obs, _ = self.corrected.reset(seed=seed)
            if index == 0:
                first_observation = hacky_obs.copy()
            np.testing.assert_array_equal(hacky_obs, corrected_obs)
            self.assertIn(tuple(self.hacky.env.goal_pos), self.split.seen)
            self.assertNotEqual(tuple(self.hacky.env.goal_pos), self.shortcut)
        repeated_observation, _ = self.hacky.reset(seed=123)
        np.testing.assert_array_equal(repeated_observation, first_observation)

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

    def test_gymnasium_and_sb3_compatibility(self):
        check_gym_env(self.env, skip_render_check=True)
        check_sb3_env(self.env)
