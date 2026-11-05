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


def draw(surface, env, last_action, last_reward, fonts, path):
    title, heading, body, small = fonts
    surface.fill(BACKGROUND)
    text(surface, small, "REWARD HACKING  /  STAGE 1", BLUE, (48, 31))
    text(surface, title, "Beacon navigation", INK, (48, 53))
    text(surface, body, "Explore the environment, one action at a time.", MUTED, (49, 100))

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

    panel = pygame.Rect(535, 145, 315, 460)
    pygame.draw.rect(surface, "#ffffff", panel, border_radius=16)
    text(surface, heading, "Episode", INK, (557, 168))
    if env.episode_ended:
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
    text(surface, body, "Arrow keys  ·  Move", INK, (557, 516))
    text(surface, body, "C  ·  Claim     P  ·  Pickup", INK, (557, 542))
    text(surface, body, "R  ·  Reset      Esc  ·  Quit", INK, (557, 568))

    smooth_circle(surface, BLUE, (53, 621), 4)
    text(surface, small, "AGENT", BLUE, (64, 615))
    pygame.draw.rect(surface, ORANGE, (151, 617, 8, 8), border_radius=2)
    text(surface, small, "BEACON", ORANGE, (166, 615))


def main():
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
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
        draw(surface, env, last_action, last_reward, fonts, path)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
