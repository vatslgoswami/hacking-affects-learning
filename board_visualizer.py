import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pygame
from numpy.random import default_rng

from rl_envs.environment import Environment

from board_rendering import BOARD_SIZE, HEIGHT, WIDTH, draw

class ReplaySession:
    speeds = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    end_pause_seconds = 1.5

    def __init__(self, record: dict[str, Any] | list[dict[str, Any]],
                 episode_index: int = 0, max_actions: int | None = None):
        self.records: list[dict[str, Any]] = record if isinstance(record, list) else [record]
        if not self.records or any(not item['episodes'] for item in self.records):
            raise ValueError('A selected checkpoint has no recorded episodes')
        if any(episode['board_size'] != BOARD_SIZE for item in self.records for episode in item['episodes']):
            raise ValueError(f'This visualizer currently supports {BOARD_SIZE}×{BOARD_SIZE} boards only')
        if not 0 <= episode_index < len(self.records[0]['episodes']):
            raise ValueError(f"Episode must be between 1 and {len(self.records[0]['episodes'])}")
        self.max_actions = max_actions if max_actions is not None else '?'
        self.speed_index = 2
        self.paused = False
        self.play_all = True
        self.select_checkpoint(0, episode_index)

    @property
    def speed(self):
        return self.speeds[self.speed_index]

    @property
    def is_reliable_checkpoint(self):
        return bool(self.records[self.checkpoint_index].get('competence', {}).get('is_reliable_checkpoint'))

    @property
    def episode(self):
        return self.episodes[self.episode_index]

    @property
    def finished(self):
        return self.step_index >= len(self.episode['steps'])

    @property
    def outcome(self):
        if self.episode['genuine_success']:
            return 'Success'
        if self.episode.get('accepted_false_completion', False):
            return 'Shortcut accepted'
        if self.episode['truncated']:
            return 'Episode truncated'
        if self.episode['steps'] and self.episode['steps'][-1]['info'].get('move_limit_reached'):
            return 'Time limit reached'
        return 'Episode ended'

    def restart(self):
        # Rebuild the initial board state from the selected recorded episode.
        initial = self.episode['initial_state']
        self.env = SimpleNamespace(agent_pos=list(initial['agent_pos']),
                                   goal_pos=list(initial['goal_pos']),
                                   package_pos=list(initial.get('package_pos', [-1, -1])),
                                   package_active=bool(initial.get('package_active', False)),
                                   carrying_package=bool(initial.get('carrying_package', False)),
                                   action_count=0, max_actions=self.max_actions,
                                   total_reward=0.0, episode_ended=False)
        self.path = [tuple(self.env.agent_pos)]
        self.last_action = None
        self.last_reward = 0.0
        self.step_index = 0
        self.elapsed = 0.0

    def select_episode(self, index):
        if 0 <= index < len(self.episodes):
            self.episode_index = index
            self.restart()

    def select_checkpoint(self, index, episode_index=0):
        if 0 <= index < len(self.records):
            self.checkpoint_index = index
            self.checkpoint = self.records[index]['n_timesteps']
            self.episodes = self.records[index]['episodes']
            self.episode_index = episode_index
            self.restart()

    def advance(self):
        # Apply one recorded state so playback follows the original episode.
        if self.finished:
            return
        step = self.episode['steps'][self.step_index]
        self.env.agent_pos = list(step['state']['agent_pos'])
        self.env.goal_pos = list(step['state']['goal_pos'])
        self.env.package_pos = list(step['state'].get('package_pos', self.env.package_pos))
        self.env.package_active = bool(step['state'].get('package_active', self.env.package_active))
        self.env.carrying_package = bool(step['state'].get('carrying_package', self.env.carrying_package))
        self.env.action_count += 1
        self.env.total_reward += step['reward']
        self.env.episode_ended = step['terminated'] or step['truncated']
        self.last_action = step['action_name']
        self.last_reward = step['reward']
        self.step_index += 1
        # Extend the visible path only when the agent moves to another cell.
        position = tuple(self.env.agent_pos)
        if position != self.path[-1]:
            self.path.append(position)

    def update(self, elapsed_seconds):
        # Advance at the chosen speed, or pause between episodes when one finishes.
        if self.paused:
            return
        self.elapsed += elapsed_seconds
        if self.finished:
            if self.play_all and self.elapsed >= self.end_pause_seconds / self.speed:
                if self.episode_index + 1 < len(self.episodes):
                    self.select_episode(self.episode_index + 1)
                elif self.checkpoint_index + 1 < len(self.records):
                    self.select_checkpoint(self.checkpoint_index + 1)
                else:
                    self.paused = True
            return
        if self.elapsed >= 0.6 / self.speed:
            self.elapsed = 0.0
            self.advance()

def load_replays(run_dir, checkpoints, episode_index=0):
    # Read the selected episode records and reject mismatched checkpoint files.
    run_dir = Path(run_dir)
    if not checkpoints:
        raise ValueError('No saved episode checkpoints found')
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError('Each checkpoint may only be selected once')
    records = []
    for checkpoint in checkpoints:
        path = run_dir / 'evaluations' / f'episodes_{checkpoint}.json'
        with path.open() as file:
            record = json.load(file)
        if record['n_timesteps'] != checkpoint:
            raise ValueError('Checkpoint in the episode file does not match the requested checkpoint')
        records.append(record)
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return ReplaySession(records, episode_index, config.get('max_actions'))

def load_replay(run_dir, checkpoint, episode_index=0):
    return load_replays(run_dir, [checkpoint], episode_index)

def _handle_replay_key(replay, key):
    # Change playback state without changing the recorded episodes.
    if key == pygame.K_SPACE:
        replay.paused = not replay.paused
    elif key == pygame.K_LEFT:
        replay.select_episode(replay.episode_index - 1)
    elif key == pygame.K_RIGHT:
        replay.select_episode(replay.episode_index + 1)
    elif key == pygame.K_LEFTBRACKET:
        replay.select_checkpoint(replay.checkpoint_index - 1)
    elif key == pygame.K_RIGHTBRACKET:
        replay.select_checkpoint(replay.checkpoint_index + 1)
    elif key == pygame.K_UP:
        replay.speed_index = min(replay.speed_index + 1, len(replay.speeds) - 1)
    elif key == pygame.K_DOWN:
        replay.speed_index = max(replay.speed_index - 1, 0)
    elif key == pygame.K_a:
        replay.play_all = not replay.play_all
    elif key == pygame.K_r:
        replay.restart()

def _handle_manual_key(env, rng, key, actions, last_action, last_reward, path):
    # Reset the board or apply one manual action and extend the visible path.
    if key == pygame.K_r:
        env.reset(rng)
        return None, 0.0, [tuple(env.agent_pos)]
    if key in actions:
        action = actions[key]
        accepted, reward, _, _, _, _ = env.agent_action(action)
        if accepted:
            position = tuple(env.agent_pos)
            if position != path[-1]:
                path.append(position)
            return action, reward, path
    return last_action, last_reward, path

def main():
    parser = argparse.ArgumentParser(description='Play the board manually or replay a saved evaluation.')
    parser.add_argument('--run', type=Path, help='Run directory containing saved evaluations')
    checkpoint_choice = parser.add_mutually_exclusive_group()
    checkpoint_choice.add_argument('--checkpoint', type=int, nargs='+', help='One or more checkpoint transition counts')
    checkpoint_choice.add_argument('--all-checkpoints', action='store_true', help='Replay all saved checkpoints in order')
    parser.add_argument('--episode', type=int, default=1, help='Starting episode number (default: 1)')
    args = parser.parse_args()
    if (args.run is None) != (args.checkpoint is None and not args.all_checkpoints):
        parser.error('--run requires --checkpoint or --all-checkpoints, and vice versa')
    try:
        checkpoints = args.checkpoint
        if args.all_checkpoints:
            checkpoints = sorted(int(path.stem.removeprefix('episodes_'))
                                 for path in (args.run / 'evaluations').glob('episodes_*.json')
                                 if path.stem.removeprefix('episodes_').isdigit())
        replay = load_replays(args.run, checkpoints, args.episode - 1) if args.run is not None else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    # Prepare the window and the manual board used when no replay is selected.
    pygame.init()
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Beacon navigation")
    fonts = (pygame.font.SysFont(None, 38), pygame.font.SysFont(None, 27),
             pygame.font.SysFont(None, 21), pygame.font.SysFont(None, 17))
    rng = default_rng()
    env = Environment(BOARD_SIZE, [3, 3], [3, 0], 30, "genuine-fixed", rng)
    actions = {pygame.K_UP: "north", pygame.K_DOWN: "south", pygame.K_LEFT: "west",
               pygame.K_RIGHT: "east", pygame.K_c: "claim", pygame.K_p: "pickup"}
    last_action = None
    last_reward = 0.0
    path = [tuple(env.agent_pos)]
    clock = pygame.time.Clock()
    running = True
    # Handle input, advance playback and draw the current state each frame.
    while running:
        elapsed_seconds = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if replay:
                    _handle_replay_key(replay, event.key)
                else:
                    last_action, last_reward, path = _handle_manual_key(
                        env, rng, event.key, actions, last_action, last_reward, path)
        if replay:
            replay.update(elapsed_seconds)
            draw(surface, replay.env, replay.last_action, replay.last_reward, fonts, replay.path, replay)
        else:
            draw(surface, env, last_action, last_reward, fonts, path)
        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
