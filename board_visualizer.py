import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pygame
from numpy.random import default_rng

from environment import Environment


WIDTH, HEIGHT = 900, 650
CELL = 64
BOARD_X, BOARD_Y = 48, 145
BOARD_SIZE = 7

BACKGROUND = "#f5f7fb"
INK = "#17233d"
MUTED = "#6c7890"
BLUE = "#526bf4"
ORANGE = "#f39a4a"


def text(surface, font, value, color, position):
    surface.blit(font.render(value, True, pygame.Color(color)), position)


def smooth_circle(surface, color, center, radius):
    scale = 4
    size = radius * 2 + 4
    circle = pygame.Surface((size * scale, size * scale), pygame.SRCALPHA)
    pygame.draw.circle(circle, pygame.Color(color), (size * scale // 2,) * 2, radius * scale)
    circle = pygame.transform.smoothscale(circle, (size, size))
    surface.blit(circle, circle.get_rect(center=center))


def cell_center(position):
    x, y = position
    return (BOARD_X + x * CELL + CELL // 2,
            BOARD_Y + (BOARD_SIZE - 1 - y) * CELL + CELL // 2)


def draw(surface, env, last_action, last_reward, fonts, path, replay=None):
    title, heading, body, small = fonts
    surface.fill(BACKGROUND)
    text(surface, small, "REWARD HACKING  /  STAGE 1", BLUE, (48, 31))
    text(surface, title, "Beacon navigation", INK, (48, 53))
    subtitle = (f"Saved evaluation  ·  Checkpoint {replay.checkpoint:,} transitions  "
                f"({replay.checkpoint_index + 1}/{len(replay.records)})"
                if replay else "Explore the environment, one action at a time.")
    text(surface, body, subtitle, MUTED, (49, 100))

    board_rect = pygame.Rect(BOARD_X - 8, BOARD_Y - 8, CELL * BOARD_SIZE + 16, CELL * BOARD_SIZE + 16)
    pygame.draw.rect(surface, "#e6ebf4", board_rect, border_radius=14)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            rect = pygame.Rect(BOARD_X + col * CELL + 2, BOARD_Y + row * CELL + 2, CELL - 4, CELL - 4)
            color = "#ffffff" if (row + col) % 2 == 0 else "#f9faff"
            pygame.draw.rect(surface, color, rect, border_radius=7)

    if len(path) > 1:
        pygame.draw.lines(surface, "#b0bfff", False, [cell_center(pos) for pos in path], 5)

    goal_center = cell_center(env.goal_pos)
    goal_rect = pygame.Rect(0, 0, 48, 48)
    goal_rect.center = goal_center
    pygame.draw.rect(surface, "#fff0db", goal_rect, border_radius=12)
    pygame.draw.rect(surface, ORANGE, goal_rect, width=3, border_radius=12)
    goal_icon = pygame.Rect(0, 0, 18, 18)
    goal_icon.center = goal_center
    pygame.draw.rect(surface, ORANGE, goal_icon, border_radius=4)

    agent_center = cell_center(env.agent_pos)
    smooth_circle(surface, "#dfe4ff", agent_center, 23)
    smooth_circle(surface, BLUE, agent_center, 16)
    agent_label = heading.render("A", True, pygame.Color("#ffffff"))
    surface.blit(agent_label, agent_label.get_rect(center=agent_center))

    if replay and replay.finished:
        banner = pygame.Rect(BOARD_X + 16, BOARD_Y + 14, CELL * BOARD_SIZE - 32, 53)
        pygame.draw.rect(surface, '#17233d', banner, border_radius=10)
        text(surface, body, f"EPISODE FINISHED  ·  {replay.outcome.upper()}", '#ffffff',
             (banner.x + 15, banner.y + 8))
        if replay.play_all and replay.episode_index + 1 < len(replay.episodes):
            detail = 'Paused' if replay.paused else 'Next episode shortly'
        elif replay.episode_index + 1 == len(replay.episodes) and replay.checkpoint_index + 1 < len(replay.records):
            detail = 'Paused' if replay.paused else ('Next checkpoint shortly' if replay.play_all else 'Use ] for next checkpoint')
        elif replay.episode_index + 1 == len(replay.episodes):
            detail = 'Last episode of last checkpoint'
        else:
            detail = 'Use Right for the next episode'
        text(surface, small, detail, '#c5cee1', (banner.x + 15, banner.y + 31))

    panel = pygame.Rect(535, 145, 315, 460)
    pygame.draw.rect(surface, "#ffffff", panel, border_radius=16)
    episode_title = f"Episode {replay.episode_index + 1} / {len(replay.episodes)}" if replay else "Episode"
    text(surface, heading, episode_title, INK, (557, 168))
    if replay and replay.finished:
        status = replay.outcome
        status_color = "#247b65" if replay.episode['genuine_success'] else "#b36d32"
    elif replay:
        status, status_color = ("Paused", MUTED) if replay.paused else ("Playing", BLUE)
    elif env.episode_ended:
        status = "Completed" if last_action == "claim" and last_reward > 0 else "Time limit reached"
        status_color = "#247b65" if status == "Completed" else "#b36d32"
    else:
        status, status_color = "In progress", BLUE
    text(surface, small, status.upper(), status_color, (557, 201))

    metrics = [
        ("ACTIONS", f"{env.action_count} / {env.max_actions}"),
        ("LAST ACTION", last_action.title() if last_action else "—"),
        ("LAST REWARD", f"{last_reward:+.2f}"),
        ("EPISODE RETURN", f"{env.total_reward:+.2f}"),
    ]
    for index, (label, value) in enumerate(metrics):
        y = 241 + index * 57
        text(surface, small, label, MUTED, (557, y))
        text(surface, heading, value, INK, (557, y + 18))

    pygame.draw.line(surface, "#e8ecf3", (557, 474), (827, 474))
    text(surface, small, "CONTROLS", MUTED, (557, 492))
    if replay:
        text(surface, body, f"Space  ·  Pause    A  ·  Play all: {'on' if replay.play_all else 'off'}", INK, (557, 510))
        text(surface, body, "Left / Right  ·  Previous / next episode", INK, (557, 533))
        text(surface, body, "[ / ]  ·  Previous / next checkpoint", INK, (557, 556))
        text(surface, body, f"Up / Down  ·  Speed: {replay.speed:g}x    R  ·  Restart", INK, (557, 579))
    else:
        text(surface, body, "Arrow keys  ·  Move", INK, (557, 516))
        text(surface, body, "C  ·  Claim     P  ·  Pickup", INK, (557, 542))
        text(surface, body, "R  ·  Reset      Esc  ·  Quit", INK, (557, 568))

    smooth_circle(surface, BLUE, (53, 621), 4)
    text(surface, small, "AGENT", BLUE, (64, 615))
    pygame.draw.rect(surface, ORANGE, (151, 617, 8, 8), border_radius=2)
    text(surface, small, "BEACON", ORANGE, (166, 615))


class ReplaySession:
    speeds = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0)
    end_pause_seconds = 1.5

    def __init__(self, record, episode_index=0, max_actions=None):
        self.records = record if isinstance(record, list) else [record]
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
    def episode(self):
        return self.episodes[self.episode_index]

    @property
    def finished(self):
        return self.step_index >= len(self.episode['steps'])

    @property
    def outcome(self):
        if self.episode['genuine_success']:
            return 'Success'
        if self.episode['truncated']:
            return 'Episode truncated'
        if self.episode['steps'] and self.episode['steps'][-1]['info'].get('move_limit_reached'):
            return 'Time limit reached'
        return 'Episode ended'

    def restart(self):
        initial = self.episode['initial_state']
        self.env = SimpleNamespace(agent_pos=list(initial['agent_pos']),
                                   goal_pos=list(initial['goal_pos']),
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
        if self.finished:
            return
        step = self.episode['steps'][self.step_index]
        self.env.agent_pos = list(step['state']['agent_pos'])
        self.env.goal_pos = list(step['state']['goal_pos'])
        self.env.action_count += 1
        self.env.total_reward += step['reward']
        self.env.episode_ended = step['terminated'] or step['truncated']
        self.last_action = step['action_name']
        self.last_reward = step['reward']
        self.step_index += 1
        position = tuple(self.env.agent_pos)
        if position != self.path[-1]:
            self.path.append(position)

    def update(self, elapsed_seconds):
        if self.paused:
            return
        self.elapsed += elapsed_seconds
        if self.finished:
            if self.play_all and self.elapsed >= self.end_pause_seconds:
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
    while running:
        elapsed_seconds = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if replay:
                    if event.key == pygame.K_SPACE:
                        replay.paused = not replay.paused
                    elif event.key == pygame.K_LEFT:
                        replay.select_episode(replay.episode_index - 1)
                    elif event.key == pygame.K_RIGHT:
                        replay.select_episode(replay.episode_index + 1)
                    elif event.key == pygame.K_LEFTBRACKET:
                        replay.select_checkpoint(replay.checkpoint_index - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        replay.select_checkpoint(replay.checkpoint_index + 1)
                    elif event.key == pygame.K_UP:
                        replay.speed_index = min(replay.speed_index + 1, len(replay.speeds) - 1)
                    elif event.key == pygame.K_DOWN:
                        replay.speed_index = max(replay.speed_index - 1, 0)
                    elif event.key == pygame.K_a:
                        replay.play_all = not replay.play_all
                    elif event.key == pygame.K_r:
                        replay.restart()
                elif event.key == pygame.K_r:
                    env.reset(rng)
                    last_action = None
                    last_reward = 0.0
                    path = [tuple(env.agent_pos)]
                elif event.key in actions:
                    action = actions[event.key]
                    accepted, reward, _, _, _, _ = env.agent_action(action)
                    if accepted:
                        last_action = action
                        last_reward = reward
                        position = tuple(env.agent_pos)
                        if position != path[-1]:
                            path.append(position)
        if replay:
            replay.update(elapsed_seconds)
            draw(surface, replay.env, replay.last_action, replay.last_reward, fonts, replay.path, replay)
        else:
            draw(surface, env, last_action, last_reward, fonts, path)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
