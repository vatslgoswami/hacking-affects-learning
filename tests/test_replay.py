import json
import tempfile
import unittest
from pathlib import Path

import pygame

from board_visualizer import ReplaySession, draw, load_replay, load_replays
from board_rendering import PURPLE, ORANGE, cell_center
from evals import _evaluate_episode
from gym_adapter import GridWorld

def make_episode(index, success):
    initial = {'agent_pos': [3, 3], 'goal_pos': [3, 0], 'package_pos': [-1, -1],
               'package_active': False, 'carrying_package': False}
    state = dict(initial, agent_pos=[3, 2])
    return {
        'episode_index': index, 'board_size': 7, 'initial_state': initial,
        'steps': [
            {'action_name': 'south', 'state': state, 'reward': -0.01,
             'terminated': False, 'truncated': False, 'info': {}},
            {'action_name': 'claim', 'state': state, 'reward': 0.99 if success else -0.03,
             'terminated': True, 'truncated': False,
             'info': {'genuine_success': True} if success else {'move_limit_reached': True}},
        ],
        'genuine_success': success, 'terminated': True, 'truncated': False,
        'action_count': 2, 'total_return': 0.98 if success else -0.04,
    }

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

    def test_delivery_timeout_replay(self):
        class Wait:
            def predict(self, obs, deterministic):
                return 5, None

        env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver')
        _, episode = _evaluate_episode(Wait(), env, 'pickup-and-deliver', 0, True, True,
                                       delivery_pair=((6, 3), (6, 5)))
        replay = ReplaySession({'n_timesteps': 0, 'episodes': [episode]}, max_actions=30)
        while not replay.finished:
            replay.advance()
        self.assertEqual(replay.outcome, 'Time limit reached')
        self.assertFalse(replay.env.carrying_package)
        self.assertEqual(replay.env.goal_pos, [6, 5])

    def test_episode_pause_scales_with_playback_speed(self):
        for speed_index in (1, 2, 6):
            with self.subTest(speed=ReplaySession.speeds[speed_index]):
                record = {'n_timesteps': 0, 'episodes': [make_episode(0, False), make_episode(1, True)]}
                replay = ReplaySession(record, max_actions=30)
                replay.speed_index = speed_index
                replay.advance()
                replay.advance()
                pause = replay.end_pause_seconds / replay.speed
                replay.update(pause * 0.9)
                self.assertEqual(replay.episode_index, 0)
                replay.update(pause * 0.2)
                self.assertEqual(replay.episode_index, 1)

    def test_play_all_advances_across_checkpoints(self):
        records = [
            {'n_timesteps': 0, 'episodes': [make_episode(0, False)]},
            {'n_timesteps': 10240, 'episodes': [make_episode(0, True)]},
        ]
        replay = ReplaySession(records, max_actions=30)
        replay.advance()
        replay.advance()
        self.assertEqual(replay.outcome, 'Time limit reached')
        replay.update(1.5)
        self.assertEqual(replay.checkpoint, 10240)
        self.assertEqual(replay.episode_index, 0)
        self.assertEqual(replay.env.agent_pos, [3, 3])
        replay.advance()
        replay.advance()
        replay.update(1.5)
        self.assertTrue(replay.paused)
        replay.select_checkpoint(0)
        self.assertEqual(replay.checkpoint, 0)
        self.assertEqual(replay.step_index, 0)

    def test_recorded_states_pause_navigation_and_play_all(self):
        record = {'n_timesteps': 10240, 'episodes': [make_episode(0, True), make_episode(1, False)]}
        replay = ReplaySession(record, max_actions=30)
        self.assertEqual(replay.env.agent_pos, [3, 3])
        replay.paused = True
        replay.update(10)
        self.assertEqual(replay.step_index, 0)
        replay.paused = False
        replay.update(0.6)
        self.assertEqual(replay.env.agent_pos, [3, 2])
        self.assertEqual(replay.path, [(3, 3), (3, 2)])
        replay.update(0.6)
        self.assertTrue(replay.finished)
        self.assertEqual(replay.outcome, 'Success')
        self.assertAlmostEqual(replay.env.total_reward, 0.98)
        replay.update(1.0)
        self.assertEqual(replay.episode_index, 0)
        replay.update(0.5)
        self.assertEqual(replay.episode_index, 1)
        self.assertEqual(replay.env.agent_pos, [3, 3])
        replay.advance()
        replay.advance()
        self.assertEqual(replay.outcome, 'Time limit reached')
        self.assertAlmostEqual(replay.env.total_reward, -0.04)
        replay.update(1.5)
        self.assertTrue(replay.paused)
        replay.select_episode(0)
        self.assertEqual(replay.step_index, 0)
        self.assertEqual(replay.path, [(3, 3)])

    def test_loading_and_drawing_saved_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'evaluations').mkdir()
            (run / 'config.json').write_text(json.dumps({'max_actions': 30}))
            record = {'n_timesteps': 0, 'episodes': [make_episode(0, True)],
                      'competence': {'is_reliable_checkpoint': True}}
            (run / 'evaluations' / 'episodes_0.json').write_text(json.dumps(record))
            replay = load_replay(run, 0)
            self.assertTrue(replay.is_reliable_checkpoint)
            self.assertEqual(replay.env.max_actions, 30)
            pygame.font.init()
            surface = pygame.Surface((900, 650))
            fonts = tuple(pygame.font.Font(None, size) for size in (38, 27, 21, 17))
            draw(surface, replay.env, None, 0.0, fonts, replay.path, replay)
            replay.advance()
            replay.advance()
            draw(surface, replay.env, replay.last_action, replay.last_reward, fonts, replay.path, replay)
            with self.assertRaises(ValueError):
                load_replay(run, 0, 1)
            second = {'n_timesteps': 10240, 'episodes': [make_episode(0, False)]}
            (run / 'evaluations' / 'episodes_10240.json').write_text(json.dumps(second))
            selected = load_replays(run, [10240, 0])
            self.assertFalse(selected.is_reliable_checkpoint)
            selected.select_checkpoint(1)
            self.assertTrue(selected.is_reliable_checkpoint)
            self.assertEqual([item['n_timesteps'] for item in selected.records], [10240, 0])
            with self.assertRaises(ValueError):
                load_replays(run, [0, 0])

if __name__ == '__main__':
    unittest.main(verbosity=2)
