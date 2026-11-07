from environment import EnvType
from stable_baselines3.common.base_class import BaseAlgorithm
from gym_adapter import GridWorld
import torch

def _board_state(test_env: GridWorld):
    env = test_env.env
    return {
        'agent_pos': list(env.agent_pos),
        'goal_pos': list(env.goal_pos),
        'package_pos': list(env.package_pos),
        'package_active': bool(env._get_package_mask(env.env_type)),
        'carrying_package': bool(env._is_carrying_package()),
    }


def _shortest_successful_length(initial_state, env_type: EnvType):
    def distance(start, end):
        return sum(abs(a - b) for a, b in zip(start, end))

    start = initial_state['agent_pos']
    goal = initial_state['goal_pos']
    if env_type == 'pickup-and-deliver' and not initial_state['carrying_package']:
        package = initial_state['package_pos']
        return distance(start, package) + distance(package, goal) + 2
    return distance(start, goal) + 1


def _evaluate_episode(model, test_env, env_type, episode_index, is_eval_greedy, record_episodes, reset_seed=None):
    obs, _ = test_env.reset(seed=reset_seed)
    initial_state = _board_state(test_env)
    shortest_length = _shortest_successful_length(initial_state, env_type)
    episode_length = 0
    episode_return = 0.0
    episode = None

    if record_episodes:
        episode = {
            'episode_index': episode_index,
            'board_size': test_env.env.board_size,
            'initial_state': initial_state,
            'shortest_successful_length': shortest_length,
            'steps': [],
        }

    while True:
        action, _ = model.predict(obs, deterministic=is_eval_greedy)
        action_id = int(action)
        obs, reward, terminated, truncated, info = test_env.step(action_id)
        episode_length += 1
        episode_return += reward

        if episode is not None:
            episode['steps'].append({
                'action': action_id,
                'action_name': test_env._map_int_to_action(action_id),
                'state': _board_state(test_env),
                'reward': float(reward),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'info': dict(info),
            })

        if terminated or truncated:
            success = bool(info.get('genuine_success', False))
            break

    outcome = {
        'genuine_success': success,
        'total_return': episode_return,
        'action_count': episode_length,
        'shortest_successful_length': shortest_length,
    }
    if episode is not None:
        episode.update(outcome)
        episode['extra_actions'] = episode_length - shortest_length if success else None
        episode['terminated'] = bool(terminated)
        episode['truncated'] = bool(truncated)
    return outcome, episode


def _summarize_episodes(outcomes):
    num_episodes = len(outcomes)
    successes = [outcome for outcome in outcomes if outcome['genuine_success']]
    num_success = len(successes)
    successful_lengths = [outcome['action_count'] for outcome in successes]
    successful_shortest_lengths = [outcome['shortest_successful_length'] for outcome in successes]
    summary_metrics = {
        'success_rate': num_success / num_episodes,
        'average_return': sum(outcome['total_return'] for outcome in outcomes) / num_episodes,
        'average_episode_length': sum(outcome['action_count'] for outcome in outcomes) / num_episodes,
        'num_successful_episodes': num_success,
        'num_failed_episodes': num_episodes - num_success,
        'average_successful_episode_length': sum(successful_lengths) / num_success if num_success else None,
        'average_shortest_successful_episode_length': sum(successful_shortest_lengths) / num_success if num_success else None,
        'average_extra_actions': (sum(successful_lengths) - sum(successful_shortest_lengths)) / num_success if num_success else None,
    }
    return summary_metrics


def evaluate_model(model:BaseAlgorithm, test_env:GridWorld, env_type:EnvType, num_episodes=25,
                   eval_seed=100, is_eval_greedy:bool=False, record_episodes=False, eval_env_seed=10):
    if num_episodes <= 0:
        raise ValueError('num_episodes must be positive')
    if env_type != test_env.env_type:
        raise ValueError('env_type must match the evaluation environment')

    outcomes = []
    episodes = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(eval_seed)
        for episode_index in range(num_episodes):
            outcome, episode = _evaluate_episode(
                model, test_env, env_type, episode_index, is_eval_greedy, record_episodes,
                reset_seed=eval_env_seed if episode_index == 0 else None,
            )
            outcomes.append(outcome)
            if episode is not None:
                episodes.append(episode)

    return _summarize_episodes(outcomes), episodes
