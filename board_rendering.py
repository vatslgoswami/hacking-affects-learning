import pygame

WIDTH, HEIGHT = 900, 650
CELL = 64
BOARD_X, BOARD_Y = 48, 145
BOARD_SIZE = 7

BACKGROUND = "#f5f7fb"
INK = "#17233d"
MUTED = "#6c7890"
BLUE = "#526bf4"
ORANGE = "#f39a4a"
PURPLE = "#8b5fc1"

def is_delivery(env):
    return bool(getattr(env, 'package_active', False) or
                getattr(env, 'env_type', None) == 'pickup-and-deliver')

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

def _draw_header(surface, fonts, replay, env):
    title, _, body, small = fonts
    surface.fill(BACKGROUND)
    text(surface, small, "REWARD HACKING  /  STAGE 1", BLUE, (48, 31))
    text(surface, title, "Pickup and delivery" if is_delivery(env) else "Beacon navigation", INK, (48, 53))
    subtitle = (f"Saved evaluation  ·  Checkpoint {replay.checkpoint:,} transitions  "
                f"({replay.checkpoint_index + 1}/{len(replay.records)})"
                if replay else "Explore the environment, one action at a time.")
    text(surface, body, subtitle, MUTED, (49, 100))
    if replay and replay.is_reliable_checkpoint:
        text(surface, small, "RELIABLE COMPETENCE CONFIRMED", '#b34684', (535, 125))

def _draw_board(surface, env, fonts, path, replay):
    # Draw the board and any path already taken by the agent.
    _, heading, body, small = fonts
    board_rect = pygame.Rect(BOARD_X - 8, BOARD_Y - 8, CELL * BOARD_SIZE + 16, CELL * BOARD_SIZE + 16)
    pygame.draw.rect(surface, "#e6ebf4", board_rect, border_radius=14)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            rect = pygame.Rect(BOARD_X + col * CELL + 2, BOARD_Y + row * CELL + 2, CELL - 4, CELL - 4)
            color = "#ffffff" if (row + col) % 2 == 0 else "#f9faff"
            pygame.draw.rect(surface, color, rect, border_radius=7)

    if len(path) > 1:
        pygame.draw.lines(surface, "#b0bfff", False, [cell_center(pos) for pos in path], 5)

    # Show the beacon or delivery destination, then the package if it is not carried.
    goal_center = cell_center(env.goal_pos)
    delivery = is_delivery(env)
    goal_color = PURPLE if delivery else ORANGE
    goal_rect = pygame.Rect(0, 0, 48, 48)
    goal_rect.center = goal_center
    pygame.draw.rect(surface, "#f0e8fa" if delivery else "#fff0db", goal_rect, border_radius=12)
    pygame.draw.rect(surface, goal_color, goal_rect, width=3, border_radius=12)
    if delivery:
        x, y = goal_center
        pygame.draw.line(surface, goal_color, (x - 9, y - 9), (x + 9, y + 9), 3)
        pygame.draw.line(surface, goal_color, (x + 9, y - 9), (x - 9, y + 9), 3)
    else:
        goal_icon = pygame.Rect(0, 0, 18, 18)
        goal_icon.center = goal_center
        pygame.draw.rect(surface, goal_color, goal_icon, border_radius=4)

    carrying = bool(getattr(env, 'carrying_package', False))
    if is_delivery(env) and not carrying:
        package_center = cell_center(env.package_pos)
        package_rect = pygame.Rect(0, 0, 34, 34)
        package_rect.center = package_center
        pygame.draw.rect(surface, '#fff0db', package_rect, border_radius=5)
        pygame.draw.rect(surface, ORANGE, package_rect, width=3, border_radius=5)
        pygame.draw.line(surface, ORANGE, (package_rect.centerx, package_rect.top + 3),
                         (package_rect.centerx, package_rect.bottom - 3), 2)

    # Draw the agent and add a package badge when it is carrying one.
    agent_center = cell_center(env.agent_pos)
    smooth_circle(surface, "#dfe4ff", agent_center, 23)
    smooth_circle(surface, BLUE, agent_center, 16)
    agent_label = heading.render("A", True, pygame.Color("#ffffff"))
    surface.blit(agent_label, agent_label.get_rect(center=agent_center))
    if is_delivery(env) and carrying:
        badge = pygame.Rect(agent_center[0] + 10, agent_center[1] - 23, 16, 16)
        pygame.draw.rect(surface, ORANGE, badge, border_radius=4)
        pygame.draw.rect(surface, '#ffffff', badge, width=2, border_radius=4)

    # Overlay the episode result while playback waits to continue.
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

def _draw_status_panel(surface, env, last_action, last_reward, fonts, replay):
    _, heading, _, small = fonts
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
    if is_delivery(env):
        carrying = bool(getattr(env, 'carrying_package', False))
        text(surface, small, 'PACKAGE: CARRYING' if carrying else 'PACKAGE: NOT PICKED UP',
             ORANGE, (557, 220))

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

def _draw_controls(surface, fonts, replay):
    _, _, body, small = fonts
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

def _draw_legend(surface, fonts, env):
    _, _, _, small = fonts
    smooth_circle(surface, BLUE, (53, 621), 4)
    text(surface, small, "AGENT", BLUE, (64, 615))
    goal_color = PURPLE if is_delivery(env) else ORANGE
    pygame.draw.rect(surface, goal_color, (151, 617, 8, 8), border_radius=2)
    text(surface, small, "DESTINATION" if is_delivery(env) else "BEACON", goal_color, (166, 615))
    if is_delivery(env):
        pygame.draw.rect(surface, ORANGE, (270, 617, 8, 8), border_radius=2)
        text(surface, small, "PACKAGE", ORANGE, (285, 615))

def draw(surface, env, last_action, last_reward, fonts, path, replay=None):
    _draw_header(surface, fonts, replay, env)
    _draw_board(surface, env, fonts, path, replay)
    _draw_status_panel(surface, env, last_action, last_reward, fonts, replay)
    _draw_controls(surface, fonts, replay)
    _draw_legend(surface, fonts, env)
