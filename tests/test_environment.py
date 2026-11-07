import unittest
from numpy.random import Generator, PCG64
from environment import Environment

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = Environment(7, [3, 3], [3, 0], 30, "genuine-fixed", Generator(PCG64()))

    def test_south_route_and_claim(self):
        for _ in range(3):
            self.env.agent_action("south")
        self.env.agent_action("claim")

        self.assertEqual(self.env.agent_pos, [3, 0])
        self.assertEqual(self.env.action_count, 4)
        self.assertAlmostEqual(self.env.total_reward, 0.96)
        self.assertTrue(self.env.episode_ended)
        self.assertFalse(self.env.agent_action("claim")[0])
        self.assertAlmostEqual(self.env.total_reward, 0.96)

        act_completed, reward, obs, terminated, truncated, info = self.env.agent_action("west")
        self.assertEqual(act_completed, False)
        self.assertEqual(self.env.action_count, 4)
        self.assertAlmostEqual(self.env.total_reward, 0.96)
        self.assertTrue(self.env.episode_ended)

    def test_invalid_claim(self):
        self.env.agent_action("claim")

        self.assertEqual(self.env.agent_pos, [3, 3])
        self.assertEqual(self.env.action_count, 1)
        self.assertAlmostEqual(self.env.total_reward, -0.03)
        self.assertFalse(self.env.episode_ended)

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

    def test_normalize_positions(self):
        norm_agent_pos = self.env._normalize_position(self.env.agent_pos)
        norm_beacon_pos = self.env._normalize_position(self.env.goal_pos)
        for _ in range(5):
            self.env.agent_action("claim")
        remaining_time = self.env._get_remaining_time()

        self.assertEqual(norm_agent_pos, [3/6, 3/6])
        self.assertEqual(norm_beacon_pos, [3/6,0])
        self.assertEqual(remaining_time, 25/30)

    def test_observation_on_reset(self):
        self.assertEqual(self.env.agent_observation(), {
            "agent_x": 0.5,
            "agent_y": 0.5,
            "goal_x": 0.5,
            "goal_y": 0.0,
            "package_x": -1,
            "package_y": -1,
            "carrying_package": 0,
            "package_active": 0,
            "task_mode": 0,
            "remaining_time": 1.0,
        })

    def test_observation_after_move_and_pickup(self):
        self.env.agent_action("south")
        after_move = self.env.agent_observation()
        self.assertAlmostEqual(after_move["agent_y"], 2 / 6)
        self.assertAlmostEqual(after_move["remaining_time"], 29 / 30)

        self.env.agent_action("pickup")
        after_pickup = self.env.agent_observation()
        self.assertAlmostEqual(after_pickup["remaining_time"], 28 / 30)
        for field in after_move:
            if field != "remaining_time":
                self.assertEqual(after_pickup[field], after_move[field])

    def test_reset(self):
        start = self.env
        start_obs = self.env.agent_observation()
        for _ in range(3):
            self.env.agent_action("south")
        self.env.agent_action("claim")
        post_move_obs = self.env.agent_observation()
        self.env.reset(self.env.np_random)
        new_obs = self.env.agent_observation()
        self.assertNotEqual(post_move_obs, new_obs)
        self.assertEqual(start, self.env)
        self.assertEqual(start_obs, new_obs)

    def test_wrong_claim_at_end(self):
        env = self.env
        env.action_count = 29
        act_completed, reward, obs, terminated, truncated, info = env.agent_action("claim")
        observation = env.agent_observation()
        self.assertEqual(True, act_completed)
        self.assertEqual(reward, -0.03)
        self.assertEqual(observation["remaining_time"], 0)
        self.assertEqual(env.episode_ended, True)

if __name__ == "__main__":
    unittest.main(verbosity=2)
