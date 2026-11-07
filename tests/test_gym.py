import unittest
import numpy as np
from gym_adapter import GridWorld


class TestGridWorld(unittest.TestCase):
    def setUp(self):
        self.grid = GridWorld(7, [3, 3], [3, 0], 30, "genuine-fixed")

    def test_env_type_reflects_underlying_environment(self):
        self.assertEqual(self.grid.env_type, 'genuine-fixed')
        self.grid.env.env_type = 'genuine-variable'
        self.assertEqual(self.grid.env_type, 'genuine-variable')

    def test_reset(self):
        initial_obs, _ = self.grid.reset(seed=123)
        self.grid.step(2)
        obs, info = self.grid.reset()

        self.assertTrue(self.grid.observation_space.contains(obs))
        np.testing.assert_array_equal(obs, initial_obs)
        self.assertEqual(info, {})
        self.assertEqual(self.grid.env.action_count, 0)
        self.assertFalse(self.grid.env.episode_ended)

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

    def test_seeded_goal_sequence(self):
        first = GridWorld(7, [3, 3], [3, 0], 30, "genuine-variable")
        second = GridWorld(7, [3, 3], [3, 0], 30, "genuine-variable")
        first.reset(seed=123)
        second.reset(seed=123)
        first_goals = [tuple(first.env.goal_pos)]
        second_goals = [tuple(second.env.goal_pos)]

        for _ in range(9):
            first.reset()
            second.reset()
            first_goals.append(tuple(first.env.goal_pos))
            second_goals.append(tuple(second.env.goal_pos))

        self.assertEqual(first_goals, second_goals)
        self.assertGreater(len(set(first_goals)), 1)
        self.assertNotIn((3, 0), first_goals)
        first.reset(seed=123)
        self.assertEqual(tuple(first.env.goal_pos), first_goals[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
